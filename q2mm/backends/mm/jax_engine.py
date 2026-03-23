"""JAX-based differentiable MM backend for analytical gradients.

Provides a pure-JAX molecular mechanics engine using harmonic bond/angle
and 12-6 Lennard-Jones energy functions.  Torsion energy functions are
implemented but not yet wired for ``Q2MMMolecule`` (which lacks torsion
detection); they will activate once torsion matching is added.
All energy functions are differentiable via ``jax.grad``, enabling analytical
gradient computation for force field parameter optimization.

ForceField stores parameters in canonical units (kcal/mol/Å² for bond_k,
kcal/mol/rad² for angle_k) with energy convention ``E = k·(x − x₀)²``.
The JAX energy functions use the same convention, so no unit conversion is
needed at the engine boundary.

For MM3-specific JAX forms, see issue #91.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.constants import (
    AMU_TO_KG,
    BOHR_TO_ANG,
    KCAL_TO_KJ,
    KJMOLA2_TO_HESSIAN_AU,
    MASSES,
    SPEED_OF_LIGHT_MS,
)
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

try:
    import jax
    import jax.numpy as jnp

    # JAX defaults to float32 which is insufficient for MM parameter
    # optimization (energy differences ~1e-6 kcal/mol matter).  This is
    # standard practice in JAX-based chemistry packages.  Guard: only set
    # if not already configured (respects JAX_ENABLE_X64 env var).
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)
    _HAS_JAX = True
except ImportError:  # pragma: no cover
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


def _ensure_jax():
    if not _HAS_JAX:
        raise ImportError("JAX is required for JaxEngine. Install with: pip install jax jaxlib")


# ---------------------------------------------------------------------------
# Hessian unit conversion: kcal/mol/Å² → Hartree/Bohr²
# ---------------------------------------------------------------------------
_KCALMOLA2_TO_HESSIAN_AU = KCAL_TO_KJ * KJMOLA2_TO_HESSIAN_AU

# ---------------------------------------------------------------------------
# Force field param vector unit conversions
# ---------------------------------------------------------------------------
# ForceField stores bond_k in kcal/(mol·Å²) and angle_k in kcal/(mol·rad²)
# (canonical units).  The JAX energy functions use the same units.
_BOND_K_CONV = 1.0
_ANGLE_K_CONV = 1.0


# ---------------------------------------------------------------------------
# Numerically stable geometry primitives
# ---------------------------------------------------------------------------


def _safe_norm(x, axis=-1):
    """Euclidean norm with gradient-safe floor to avoid NaN at zero."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=axis), 1e-30))


def _safe_arccos(x):
    """arccos clipped to (-1, 1) to avoid NaN gradients at boundaries."""
    return jnp.arccos(jnp.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


def _normalize(x, axis=-1):
    """Unit-normalize vectors along axis."""
    return x / _safe_norm_keepdims(x, axis=axis) if x.ndim > 1 else x / _safe_norm(x)


def _safe_norm_keepdims(x, axis=-1):
    """Norm with keepdims for broadcasting."""
    return jnp.sqrt(jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), 1e-30))


# ---------------------------------------------------------------------------
# OPLSAA-style energy functions (pure JAX, differentiable)
# ---------------------------------------------------------------------------


def _harmonic_bond_energy(k, r0, coords, bond_indices):
    """Harmonic bond stretch: E = sum_i k_i * (r_i - r0_i)^2.

    Parameters
    ----------
    k : jnp.ndarray, shape (n_bonds,)
        Force constants in kcal/mol/Å².
    r0 : jnp.ndarray, shape (n_bonds,)
        Equilibrium distances in Å.
    coords : jnp.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Å.
    bond_indices : jnp.ndarray, shape (n_bonds, 2)
        Atom index pairs for each bond.
    """
    dr = coords[bond_indices[:, 0]] - coords[bond_indices[:, 1]]
    r = _safe_norm(dr, axis=-1)
    return jnp.sum(k * (r - r0) ** 2)


def _harmonic_angle_energy(k, theta0, coords, angle_indices):
    """Harmonic angle bend: E = sum_i k_i * (theta_i - theta0_i)^2.

    Parameters
    ----------
    k : jnp.ndarray, shape (n_angles,)
        Force constants in kcal/mol/rad².
    theta0 : jnp.ndarray, shape (n_angles,)
        Equilibrium angles in radians.
    coords : jnp.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Å.
    angle_indices : jnp.ndarray, shape (n_angles, 3)
        Atom index triples (i, j, k) where j is the central atom.
    """
    rij = coords[angle_indices[:, 0]] - coords[angle_indices[:, 1]]
    rkj = coords[angle_indices[:, 2]] - coords[angle_indices[:, 1]]
    rij_norm = rij / _safe_norm_keepdims(rij, axis=-1)
    rkj_norm = rkj / _safe_norm_keepdims(rkj, axis=-1)
    cos_theta = jnp.sum(rij_norm * rkj_norm, axis=-1)
    theta = _safe_arccos(cos_theta)
    return jnp.sum(k * (theta - theta0) ** 2)


def _fourier_torsion_energy(k, n, gamma, coords, torsion_indices):
    """Fourier torsion: E = sum_i k_i * (1 + cos(n_i * phi_i - gamma_i)).

    Parameters
    ----------
    k : jnp.ndarray, shape (n_torsions,)
        Barrier heights in kcal/mol.
    n : jnp.ndarray, shape (n_torsions,)
        Periodicities.
    gamma : jnp.ndarray, shape (n_torsions,)
        Phase angles in radians.
    coords : jnp.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Å.
    torsion_indices : jnp.ndarray, shape (n_torsions, 4)
        Atom index quadruples.
    """
    p0 = coords[torsion_indices[:, 0]]
    p1 = coords[torsion_indices[:, 1]]
    p2 = coords[torsion_indices[:, 2]]
    p3 = coords[torsion_indices[:, 3]]

    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    n1 = jnp.cross(b0, b1)
    n2 = jnp.cross(b1, b2)
    n1 = n1 / _safe_norm_keepdims(n1, axis=-1)
    n2 = n2 / _safe_norm_keepdims(n2, axis=-1)

    cos_phi = jnp.sum(n1 * n2, axis=-1)
    phi = _safe_arccos(cos_phi)

    return jnp.sum(k * (1.0 + jnp.cos(n * phi - gamma)))


def _lj_12_6_energy(per_atom_sigma, per_atom_epsilon, coords, pair_indices):
    """Lennard-Jones 12-6: E = sum_{i<j} 4*eps_ij * [(sig_ij/r)^12 - (sig_ij/r)^6].

    Uses geometric combining rules: sig_ij = sqrt(sig_i * sig_j),
    eps_ij = sqrt(eps_i * eps_j).

    Parameters
    ----------
    per_atom_sigma : jnp.ndarray, shape (n_atoms,)
        Per-atom sigma in Å.
    per_atom_epsilon : jnp.ndarray, shape (n_atoms,)
        Per-atom epsilon in kcal/mol.
    coords : jnp.ndarray, shape (n_atoms, 3)
        Cartesian coordinates in Å.
    pair_indices : jnp.ndarray, shape (n_pairs, 2)
        Non-excluded atom pairs (i < j).
    """
    if pair_indices.shape[0] == 0:
        return jnp.float64(0.0)

    sig_i = per_atom_sigma[pair_indices[:, 0]]
    sig_j = per_atom_sigma[pair_indices[:, 1]]
    eps_i = per_atom_epsilon[pair_indices[:, 0]]
    eps_j = per_atom_epsilon[pair_indices[:, 1]]

    sig_ij = jnp.sqrt(sig_i * sig_j)
    eps_ij = jnp.sqrt(eps_i * eps_j)

    dr = coords[pair_indices[:, 0]] - coords[pair_indices[:, 1]]
    r = _safe_norm(dr, axis=-1)

    sr6 = (sig_ij / r) ** 6
    return jnp.sum(4.0 * eps_ij * (sr6**2 - sr6))


# ---------------------------------------------------------------------------
# Topology / handle data structures
# ---------------------------------------------------------------------------


def _build_vdw_pairs(n_atoms: int, bond_pairs: list[tuple[int, int]]) -> np.ndarray:
    """Build non-excluded vdW pair list (1-2 and 1-3 exclusions)."""
    excluded: set[tuple[int, int]] = set()

    # 1-2 exclusions
    adj: dict[int, set[int]] = {i: set() for i in range(n_atoms)}
    for i, j in bond_pairs:
        pair = (min(i, j), max(i, j))
        excluded.add(pair)
        adj[i].add(j)
        adj[j].add(i)

    # 1-3 exclusions
    for atom in range(n_atoms):
        neighbors = list(adj[atom])
        for ni in range(len(neighbors)):
            for nj in range(ni + 1, len(neighbors)):
                pair = (min(neighbors[ni], neighbors[nj]), max(neighbors[ni], neighbors[nj]))
                excluded.add(pair)

    pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms) if (i, j) not in excluded]
    return np.array(pairs, dtype=np.int32) if pairs else np.empty((0, 2), dtype=np.int32)


@dataclass
class JaxHandle:
    """Cached topology and parameter mapping for a molecule.

    Created once per molecule, reused across parameter updates.
    """

    molecule: Q2MMMolecule
    # Topology arrays (static, used inside JIT closures)
    bond_indices: np.ndarray  # (n_matched_bonds, 2) atom indices
    angle_indices: np.ndarray  # (n_matched_angles, 3) atom indices
    torsion_indices: np.ndarray  # (n_matched_torsions, 4) atom indices
    vdw_pair_indices: np.ndarray  # (n_vdw_pairs, 2) non-excluded pairs
    # Mappings: detected term index → ForceField param type index
    bond_param_map: np.ndarray  # (n_matched_bonds,) → index into ff.bonds
    angle_param_map: np.ndarray  # (n_matched_angles,) → index into ff.angles
    torsion_param_map: np.ndarray  # (n_matched_torsions,) → index into ff.torsions
    atom_vdw_map: np.ndarray  # (n_atoms,) → index into ff.vdws
    # Param vector layout
    n_bond_types: int
    n_angle_types: int
    n_torsion_types: int
    n_vdw_types: int
    # Compiled energy function (captures topology, JIT-compiled)
    _energy_fn: Callable | None = field(default=None, repr=False)
    _grad_fn: Callable | None = field(default=None, repr=False)
    _coord_hess_fn: Callable | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Matching helpers (reuse OpenMM logic patterns)
# ---------------------------------------------------------------------------


def _match_bond(forcefield, elements, env_id="", ff_row=None):
    matched = forcefield.match_bond(elements, env_id=env_id, ff_row=ff_row)
    if matched is not None:
        return forcefield.bonds.index(matched), matched
    return None, None


def _match_angle(forcefield, elements, env_id="", ff_row=None):
    matched = forcefield.match_angle(elements, env_id=env_id, ff_row=ff_row)
    if matched is not None:
        return forcefield.angles.index(matched), matched
    return None, None


def _match_vdw(forcefield, atom_type="", element="", ff_row=None):
    matched = forcefield.match_vdw(atom_type=atom_type, element=element, ff_row=ff_row)
    if matched is not None:
        return forcefield.vdws.index(matched), matched
    return None, None


# ---------------------------------------------------------------------------
# JaxEngine
# ---------------------------------------------------------------------------


class JaxEngine(MMEngine):
    """Differentiable MM backend using JAX with OPLSAA-style energy functions.

    Provides analytical gradients of the energy with respect to force field
    parameters via ``jax.grad``, eliminating the need for finite differences
    in parameter optimization.

    The energy functions use standard harmonic/LJ forms (not MM3). Near
    equilibrium, results are similar to MM3 but not identical. For exact
    MM3 parity, use ``OpenMMEngine``.

    Examples
    --------
    >>> engine = JaxEngine()
    >>> energy = engine.energy(molecule, forcefield)
    >>> energy, grad = engine.energy_and_param_grad(molecule, forcefield)
    """

    def __init__(self):
        _ensure_jax()

    @property
    def name(self) -> str:
        return "JAX (harmonic)"

    def supported_functional_forms(self) -> frozenset[str]:
        """JAX currently supports harmonic forms only (see issue #91 for MM3)."""
        return frozenset({"harmonic"})

    def is_available(self) -> bool:
        return _HAS_JAX

    def supports_runtime_params(self) -> bool:
        return True

    def supports_analytical_gradients(self) -> bool:
        return True

    def create_context(self, structure, forcefield: ForceField | None = None) -> JaxHandle:
        """Build topology and compile energy function for a molecule."""
        if forcefield is not None:
            self._validate_forcefield(forcefield)
        molecule = _as_molecule(structure)
        if forcefield is None:
            forcefield = ForceField.create_for_molecule(molecule)

        # Match bonds
        bond_atom_indices = []
        bond_param_map = []
        for bond in molecule.bonds:
            idx, param = _match_bond(forcefield, bond.elements, env_id=bond.env_id, ff_row=bond.ff_row)
            if param is not None:
                bond_atom_indices.append((bond.atom_i, bond.atom_j))
                bond_param_map.append(idx)

        # Match angles
        angle_atom_indices = []
        angle_param_map = []
        for angle in molecule.angles:
            idx, param = _match_angle(forcefield, angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if param is not None:
                angle_atom_indices.append((angle.atom_i, angle.atom_j, angle.atom_k))
                angle_param_map.append(idx)

        # Match torsions (NOTE: Q2MMMolecule does not yet detect torsions,
        # so this loop will be empty until torsion matching is added.)
        torsion_atom_indices = []
        torsion_param_map = []
        for i_tor, torsion in enumerate(molecule.torsions if hasattr(molecule, "torsions") else []):
            for j_ff, ff_tor in enumerate(forcefield.torsions):
                if ff_tor.ff_row is not None and hasattr(torsion, "ff_row") and torsion.ff_row == ff_tor.ff_row:
                    torsion_atom_indices.append((torsion.atom_i, torsion.atom_j, torsion.atom_k, torsion.atom_l))
                    torsion_param_map.append(j_ff)
                    break

        # Match vdW
        atom_vdw_map = []
        for atom_index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=False)):
            idx, param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
            if param is not None:
                atom_vdw_map.append(idx)
            else:
                atom_vdw_map.append(-1)

        # Validate vdW mapping: if the force field defines vdW parameters,
        # disallow any unmatched atoms.  Using -1 as an index would silently
        # select the last vdW entry via JAX negative indexing, corrupting
        # the energy and gradients.  When the force field defines no vdW
        # terms at all, vdW energy is effectively disabled and unmatched
        # atoms are safe.
        unmatched = [i for i, idx in enumerate(atom_vdw_map) if idx == -1]
        if getattr(forcefield, "vdws", None) and unmatched:
            raise ValueError(
                f"Unmatched vdW parameters for atoms at indices {unmatched}. "
                "Ensure all atom types/elements have corresponding vdW "
                "parameters in the force field, or remove vdW terms from "
                "the force field if vdW interactions are not intended."
            )

        # Build vdW pair list with 1-2 and 1-3 exclusions
        vdw_pairs = _build_vdw_pairs(
            len(molecule.symbols),
            [(b.atom_i, b.atom_j) for b in molecule.bonds],
        )

        bond_indices_arr = (
            np.array(bond_atom_indices, dtype=np.int32) if bond_atom_indices else np.empty((0, 2), dtype=np.int32)
        )
        angle_indices_arr = (
            np.array(angle_atom_indices, dtype=np.int32) if angle_atom_indices else np.empty((0, 3), dtype=np.int32)
        )
        torsion_indices_arr = (
            np.array(torsion_atom_indices, dtype=np.int32) if torsion_atom_indices else np.empty((0, 4), dtype=np.int32)
        )

        handle = JaxHandle(
            molecule=copy.deepcopy(molecule),
            bond_indices=bond_indices_arr,
            angle_indices=angle_indices_arr,
            torsion_indices=torsion_indices_arr,
            vdw_pair_indices=vdw_pairs,
            bond_param_map=np.array(bond_param_map, dtype=np.int32),
            angle_param_map=np.array(angle_param_map, dtype=np.int32),
            torsion_param_map=np.array(torsion_param_map, dtype=np.int32),
            atom_vdw_map=np.array(atom_vdw_map, dtype=np.int32),
            n_bond_types=len(forcefield.bonds),
            n_angle_types=len(forcefield.angles),
            n_torsion_types=len(forcefield.torsions),
            n_vdw_types=len(forcefield.vdws),
        )

        # Compile energy function
        handle._energy_fn = _compile_energy_fn(handle)
        return handle

    def _get_handle(self, structure, forcefield):
        """Get or create a JaxHandle."""
        if isinstance(structure, JaxHandle):
            return structure
        molecule = _as_molecule(structure)
        return self.create_context(molecule, forcefield)

    def _params_and_coords(self, handle, forcefield):
        """Extract JAX arrays from forcefield and molecule."""
        params = jnp.array(forcefield.get_param_vector(), dtype=jnp.float64)
        coords = jnp.array(handle.molecule.geometry, dtype=jnp.float64)
        return params, coords

    def energy(self, structure, forcefield) -> float:
        """Calculate energy in kcal/mol."""
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        return float(handle._energy_fn(params, coords))

    def energy_and_param_grad(self, structure, forcefield) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. FF parameters.

        Returns
        -------
        energy : float
            Energy in kcal/mol.
        grad : np.ndarray
            Gradient dE/d(param_vector), same shape as ``forcefield.get_param_vector()``.
        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._grad_fn is None:
            handle._grad_fn = jax.jit(jax.value_and_grad(handle._energy_fn, argnums=0))

        val, grad = handle._grad_fn(params, coords)
        return float(val), np.asarray(grad)

    def hessian(self, structure, forcefield) -> np.ndarray:
        """Compute Hessian via jax.hessian (d²E/d_coords²) in Hartree/Bohr²."""
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._coord_hess_fn is None:

            def _energy_of_flat_coords(flat_coords, params_):
                return handle._energy_fn(params_, flat_coords.reshape(-1, 3))

            handle._coord_hess_fn = jax.jit(jax.hessian(_energy_of_flat_coords, argnums=0))

        flat_coords = coords.flatten()
        hess_kcal_a2 = handle._coord_hess_fn(flat_coords, params)
        return np.asarray(hess_kcal_a2) * _KCALMOLA2_TO_HESSIAN_AU

    def frequencies(self, structure, forcefield) -> list[float]:
        """Compute vibrational frequencies in cm⁻¹ from the Hessian."""
        handle = self._get_handle(structure, forcefield)
        hess_au = self.hessian(handle, forcefield)
        n_atoms = len(handle.molecule.symbols)

        masses = np.array([MASSES[s] for s in handle.molecule.symbols])
        mass_weights = np.repeat(masses, 3)
        sqrt_inv_mass = 1.0 / np.sqrt(mass_weights)
        mw_hess = hess_au * np.outer(sqrt_inv_mass, sqrt_inv_mass)

        eigenvalues = np.linalg.eigvalsh(mw_hess)

        # Convert eigenvalues (Hartree / (amu * Bohr²)) → cm⁻¹
        hartree_to_j = 4.359744650e-18
        bohr_to_m = BOHR_TO_ANG * 1e-10
        factor = hartree_to_j / (AMU_TO_KG * bohr_to_m**2)

        freqs = []
        for ev in eigenvalues:
            val = ev * factor
            if val < 0:
                freq_hz = -math.sqrt(-val)
            else:
                freq_hz = math.sqrt(val)
            freq_cm = freq_hz / (2.0 * math.pi * SPEED_OF_LIGHT_MS * 100.0)
            freqs.append(freq_cm)

        return sorted(freqs)

    def minimize(self, structure, forcefield, max_iterations=200):
        """Minimize energy w.r.t. coordinates using analytical JAX gradients."""
        from scipy.optimize import minimize as scipy_minimize

        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        energy_fn = handle._energy_fn
        coord_grad_fn = jax.jit(jax.grad(lambda c, p: energy_fn(p, c.reshape(-1, 3)), argnums=0))

        x0 = np.asarray(coords.flatten())

        def objective(x):
            return float(energy_fn(params, jnp.array(x).reshape(-1, 3)))

        def gradient(x):
            return np.asarray(coord_grad_fn(jnp.array(x), params))

        result = scipy_minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            options={"maxiter": max_iterations},
        )

        opt_coords = result.x.reshape(-1, 3)
        opt_energy = float(result.fun)
        return opt_energy, list(handle.molecule.symbols), opt_coords


# ---------------------------------------------------------------------------
# Energy function compiler
# ---------------------------------------------------------------------------


def _compile_energy_fn(handle: JaxHandle) -> Callable:
    """Create a JIT-compiled energy function for a specific topology.

    The returned function has signature: ``(params, coords) -> energy``
    where ``params`` is the flat parameter vector from ``ForceField.get_param_vector()``
    and ``coords`` is shape ``(n_atoms, 3)`` in Å. Energy returned in kcal/mol.
    """
    # Capture topology as JAX arrays in the closure
    has_bonds = handle.n_bond_types > 0 and len(handle.bond_indices) > 0
    has_angles = handle.n_angle_types > 0 and len(handle.angle_indices) > 0
    has_torsions = handle.n_torsion_types > 0 and len(handle.torsion_indices) > 0
    has_vdw = handle.n_vdw_types > 0 and len(handle.vdw_pair_indices) > 0

    _bond_indices = jnp.array(handle.bond_indices) if has_bonds else None
    _bond_map = jnp.array(handle.bond_param_map) if has_bonds else None
    _angle_indices = jnp.array(handle.angle_indices) if has_angles else None
    _angle_map = jnp.array(handle.angle_param_map) if has_angles else None
    _torsion_indices = jnp.array(handle.torsion_indices) if has_torsions else None
    _torsion_map = jnp.array(handle.torsion_param_map) if has_torsions else None
    _vdw_pairs = jnp.array(handle.vdw_pair_indices) if has_vdw else None
    _atom_vdw_map = jnp.array(handle.atom_vdw_map) if has_vdw else None

    n_bt = handle.n_bond_types
    n_at = handle.n_angle_types
    n_tt = handle.n_torsion_types
    n_vt = handle.n_vdw_types

    # Param vector offsets
    bond_offset = 0
    angle_offset = 2 * n_bt
    torsion_offset = angle_offset + 2 * n_at
    vdw_offset = torsion_offset + n_tt

    @jax.jit
    def energy_fn(params, coords):
        E = jnp.float64(0.0)

        if has_bonds:
            bond_params = params[bond_offset : bond_offset + 2 * n_bt].reshape(n_bt, 2)
            # bond_k already in kcal/mol/Å² (canonical units)
            k = bond_params[_bond_map, 0] * _BOND_K_CONV
            r0 = bond_params[_bond_map, 1]
            E = E + _harmonic_bond_energy(k, r0, coords, _bond_indices)

        if has_angles:
            angle_params = params[angle_offset : angle_offset + 2 * n_at].reshape(n_at, 2)
            # angle_k already in kcal/mol/rad² (canonical units)
            k = angle_params[_angle_map, 0] * _ANGLE_K_CONV
            theta0_deg = angle_params[_angle_map, 1]
            theta0 = theta0_deg * (jnp.pi / 180.0)
            E = E + _harmonic_angle_energy(k, theta0, coords, _angle_indices)

        if has_torsions:
            # Torsion terms are not yet correctly supported in the JAX backend:
            # - Periodicity and phase from the ForceField are not plumbed through.
            # - The dihedral computation uses arccos which loses the sign of phi.
            # Disable explicitly until a full atan2-based implementation is added.
            raise NotImplementedError(
                "Torsion energies are not yet supported in the JAX backend. "
                "Q2MMMolecule does not detect torsions, so this block is normally "
                "unreachable. If you reach this, implement proper periodicity/phase "
                "handling and a signed dihedral (atan2) formulation first."
            )

        if has_vdw:
            vdw_params = params[vdw_offset : vdw_offset + 2 * n_vt].reshape(n_vt, 2)
            # Map per-type → per-atom (radius and epsilon)
            per_atom_radius = vdw_params[_atom_vdw_map, 0]
            per_atom_epsilon = vdw_params[_atom_vdw_map, 1]
            # Convert radius to sigma for 12-6 LJ: sigma = 2*radius / 2^(1/6)
            # (MM3/OPLS use radius as Rmin/2, sigma = Rmin / 2^(1/6))
            per_atom_sigma = per_atom_radius * 2.0 / (2.0 ** (1.0 / 6.0))
            E = E + _lj_12_6_energy(per_atom_sigma, per_atom_epsilon, coords, _vdw_pairs)

        return E

    return energy_fn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_molecule(structure) -> Q2MMMolecule:
    """Coerce input to a Q2MMMolecule."""
    if isinstance(structure, JaxHandle):
        return structure.molecule
    if isinstance(structure, Q2MMMolecule):
        return structure
    raise TypeError(f"JaxEngine expects a Q2MMMolecule or JaxHandle, got {type(structure).__name__}.")
