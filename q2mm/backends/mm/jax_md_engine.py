"""JAX-MD differentiable MM backend for periodic systems.

Wraps `jax-md <https://github.com/jax-md/jax-md>`_'s ``mm_forcefields.oplsaa``
module to provide a production-quality OPLSAA engine with:

- Harmonic bonds, angles, and proper torsions (torsion math implemented;
  requires molecule-level torsion detection — see `#127
  <https://github.com/ericchansen/q2mm/issues/127>`_)
- Lennard-Jones 12-6 with geometric combining rules
- Electrostatics (Cutoff, Ewald, or PME Coulomb)
- Periodic boundary conditions with neighbor lists
- Analytical parameter gradients via ``jax.grad``
- Analytical coordinate Hessians via ``jax.hessian``

**Units:** jax-md's OPLSAA module uses the same canonical units as Q2MM
(kcal/mol/Å² for bonds, kcal/mol/rad² for angles, kcal/mol for LJ epsilon).
Angle equilibria are converted from degrees (ForceField) to radians (jax-md)
at the boundary. vdW radii are converted from Rmin/2 (ForceField) to
LJ sigma (jax-md).

**Relationship to JaxEngine:** JaxEngine is a lightweight gas-phase backend
with hand-rolled energy functions. JaxMDEngine wraps jax-md's full
OPLSAA implementation including torsions, electrostatics, and PBC support.
Use JaxEngine for simple gas-phase work, JaxMDEngine when you need the
full feature set of jax-md.

.. note::
   Importing this module enables 64-bit precision globally via
   ``jax.config.update("jax_enable_x64", True)``, which is required for
   numerical accuracy in force field calculations. This affects all JAX
   operations in the same Python process.

.. note::
   Electrostatic charges are not yet optimizable. Coulomb energy is computed
   with zero charges unless a future extension adds charge support.
   The Coulomb handler is still invoked (returning zero) so the API is
   forward-compatible.

.. note::
   Improper torsions are not yet supported. The topology arrays are
   allocated empty. Support will be added when the Q2MM data model
   includes improper parameters.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

from q2mm.backends.base import MMEngine, coerce_molecule
from q2mm.backends.registry import register_mm
from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

from q2mm.backends.mm._jax_common import (
    compute_param_offsets,
    ensure_jax,
    jax,
    jnp,
    match_angle as _match_angle,
    match_bond as _match_bond,
    match_vdw as _match_vdw,
    params_and_coords as _params_and_coords_impl,
)

try:
    from jax_md.mm_forcefields.base import (
        NonbondedOptions,
    )
    from jax_md.mm_forcefields.nonbonded.electrostatics import (
        CoulombHandler,
        CutoffCoulomb,
    )
    from jax_md.mm_forcefields.oplsaa import energy as oplsaa_energy
    from jax_md.mm_forcefields.oplsaa.params import create_parameters
    from jax_md.mm_forcefields.oplsaa.topology import create_topology

    _HAS_JAX_MD = True
except ImportError:  # pragma: no cover
    _HAS_JAX_MD = False

# Hessian unit conversion imported from q2mm.models.units


def _ensure_jax_md() -> None:
    """Raise ``ImportError`` if jax-md is not installed."""
    ensure_jax("JaxMDEngine")
    if not _HAS_JAX_MD:
        raise ImportError("jax-md is required for JaxMDEngine. Install with: pip install jax-md")


# ---------------------------------------------------------------------------
# Handle (cached topology + compiled functions)
# ---------------------------------------------------------------------------


@dataclass
class JaxMDHandle:
    """Cached jax-md topology, parameters, and compiled functions.

    Created once per (molecule, box, coulomb) configuration. The compiled
    energy function captures the topology; only parameters change between
    calls.

    Attributes:
        molecule: Deep copy of the input molecule.
        box: Simulation box dimensions, shape ``(3,)``.
        bond_indices: Matched bond atom pairs, shape ``(n_matched_bonds, 2)``.
        angle_indices: Matched angle atom triples, shape ``(n_matched_angles, 3)``.
        torsion_indices: Matched torsion atom quads, shape ``(n_matched_torsions, 4)``.
        bond_param_map: Maps each matched bond to a ForceField bond index.
        angle_param_map: Maps each matched angle to a ForceField angle index.
        torsion_param_map: Maps each matched torsion to a ForceField torsion index.
        atom_vdw_map: Maps each atom to a ForceField vdW index.
        n_bond_types: Number of unique bond parameter types.
        n_angle_types: Number of unique angle parameter types.
        n_torsion_types: Number of unique torsion parameter types.
        n_vdw_types: Number of unique vdW parameter types.
        n_atoms: Number of atoms.

    """

    molecule: Q2MMMolecule
    box: np.ndarray
    # Matched atom indices (parallel with param maps)
    bond_indices: np.ndarray  # (n_matched_bonds, 2) atom indices
    angle_indices: np.ndarray  # (n_matched_angles, 3) atom indices
    torsion_indices: np.ndarray  # (n_matched_torsions, 4) atom indices
    # Mappings: matched term index → ForceField param type index
    bond_param_map: np.ndarray
    angle_param_map: np.ndarray
    torsion_param_map: np.ndarray
    atom_vdw_map: np.ndarray
    # Param vector layout
    n_bond_types: int
    n_angle_types: int
    n_torsion_types: int
    n_vdw_types: int
    n_atoms: int
    # Whether charges are present
    has_charges: bool = False
    # Compiled functions (lazy, JIT-compiled)
    _energy_fn: Callable | None = field(default=None, repr=False)
    _scalar_energy_fn: Callable | None = field(default=None, repr=False)
    _grad_fn: Callable | None = field(default=None, repr=False)
    _coord_hess_fn: Callable | None = field(default=None, repr=False)
    # jax-md neighbor list function
    _neighbor_fn: object | None = field(default=None, repr=False)
    _nlist: object | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Parameter builder (differentiable: ForceField param vector → jax-md params)
# ---------------------------------------------------------------------------


def _build_jaxmd_params_fn(
    handle: JaxMDHandle,
) -> Callable[
    [jnp.ndarray],
    tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ],
]:
    """Create a function that maps a flat param vector to jax-md Parameters.

    The returned function is pure JAX and differentiable, enabling
    ``jax.grad`` to flow through the parameter mapping.

    Args:
        handle: A :class:`JaxMDHandle` with topology mappings populated.

    Returns:
        Callable: ``build_params(param_vector) -> (bonded_tuple, nonbonded_tuple)``

    """
    n_bt = handle.n_bond_types
    n_at = handle.n_angle_types
    n_tt = handle.n_torsion_types
    n_vt = handle.n_vdw_types
    n_atoms = handle.n_atoms

    bond_map = jnp.array(handle.bond_param_map, dtype=jnp.int32)
    angle_map = jnp.array(handle.angle_param_map, dtype=jnp.int32)
    torsion_map = jnp.array(handle.torsion_param_map, dtype=jnp.int32)
    atom_vdw_map = jnp.array(handle.atom_vdw_map, dtype=jnp.int32)

    # Param vector offsets (same layout as ForceField.get_param_vector)
    _offsets = compute_param_offsets(n_bt, n_at, n_tt)
    bond_offset = _offsets["bond"]
    angle_offset = _offsets["angle"]
    torsion_offset = _offsets["torsion"]
    vdw_offset = _offsets["vdw"]

    n_bonds = len(handle.bond_param_map)
    n_angles = len(handle.angle_param_map)
    n_torsions = len(handle.torsion_param_map)

    def build_params(
        param_vector: jnp.ndarray,
    ) -> tuple[
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    ]:
        """Unpack a flat parameter vector into per-term JAX-MD arrays.

        Args:
            param_vector: Flat 1-D array of all force-field parameters.

        Returns:
            A ``(bond_tuple, angle_tuple)`` pair of per-interaction arrays
            ready for the energy function.

        """
        # Bonds: extract k and r0 per topology bond
        if n_bt > 0 and n_bonds > 0:
            bond_params = param_vector[bond_offset : bond_offset + 2 * n_bt].reshape(n_bt, 2)
            bond_k = bond_params[bond_map, 0]
            bond_r0 = bond_params[bond_map, 1]
        else:
            bond_k = jnp.zeros(max(n_bonds, 1))
            bond_r0 = jnp.zeros(max(n_bonds, 1))

        # Angles: extract k and theta0 (convert deg → rad)
        if n_at > 0 and n_angles > 0:
            angle_params = param_vector[angle_offset : angle_offset + 2 * n_at].reshape(n_at, 2)
            angle_k = angle_params[angle_map, 0]
            angle_theta0 = angle_params[angle_map, 1] * (jnp.pi / 180.0)
        else:
            angle_k = jnp.zeros(max(n_angles, 1))
            angle_theta0 = jnp.zeros(max(n_angles, 1))

        # Torsions: extract k (periodicity and phase are static, from ForceField)
        if n_tt > 0 and n_torsions > 0:
            torsion_k = param_vector[torsion_offset : torsion_offset + n_tt][torsion_map]
        else:
            torsion_k = jnp.zeros(max(n_torsions, 1))

        # vdW: extract radius and epsilon per type, map to per-atom sigma
        if n_vt > 0:
            vdw_params = param_vector[vdw_offset : vdw_offset + 2 * n_vt].reshape(n_vt, 2)
            per_atom_radius = vdw_params[atom_vdw_map, 0]
            per_atom_epsilon = vdw_params[atom_vdw_map, 1]
            # Convert Rmin/2 → LJ sigma
            per_atom_sigma = per_atom_radius * 2.0 / (2.0 ** (1.0 / 6.0))
        else:
            per_atom_sigma = jnp.zeros(n_atoms)
            per_atom_epsilon = jnp.zeros(n_atoms)

        # Charges (not optimized — zero for now, or from molecule)
        charges = jnp.zeros(n_atoms)

        return (bond_k, bond_r0, angle_k, angle_theta0, torsion_k), (
            charges,
            per_atom_sigma,
            per_atom_epsilon,
        )

    return build_params


# ---------------------------------------------------------------------------
# Energy function compiler
# ---------------------------------------------------------------------------


def _compile_energy_fn(
    handle: JaxMDHandle, forcefield: ForceField, coulomb: object, nb_options: object
) -> tuple[Callable, Callable, object, object]:
    """Compile the jax-md energy function for a specific topology.

    Args:
        handle: Handle with topology and mappings.
        forcefield: ForceField (used for torsion periodicity/phase).
        coulomb: CoulombHandler instance.
        nb_options: NonbondedOptions instance.

    Returns:
        tuple: (energy_fn, scalar_energy_fn, neighbor_fn, nlist) where
            energy_fn takes (param_vector, coords) and returns a dict,
            scalar_energy_fn returns just the total scalar.

    """
    # Build jax-md topology arrays
    molecule = handle.molecule
    n_atoms = handle.n_atoms

    # Use pre-matched atom indices from the handle (parallel with param maps)
    bond_arr = handle.bond_indices
    angle_arr = handle.angle_indices
    torsion_arr = handle.torsion_indices

    # Improper torsions — placeholder, not yet supported
    improper_arr = np.empty((0, 4), dtype=np.int32)

    # Torsion periodicity and phase (static, from ForceField)
    if len(handle.torsion_param_map) > 0:
        torsion_n_static = jnp.array(
            [
                forcefield.torsions[handle.torsion_param_map[i]].periodicity
                for i in range(len(handle.torsion_param_map))
            ],
            dtype=jnp.float64,
        )
        torsion_gamma_static = jnp.array(
            [
                forcefield.torsions[handle.torsion_param_map[i]].phase * (jnp.pi / 180.0)
                for i in range(len(handle.torsion_param_map))
            ],
            dtype=jnp.float64,
        )
    else:
        torsion_n_static = jnp.zeros(1)
        torsion_gamma_static = jnp.zeros(1)

    # Improper statics (empty for now)
    improper_k_static = jnp.zeros(1)
    improper_n_static = jnp.zeros(1)
    improper_gamma_static = jnp.zeros(1)

    # Build jax-md Topology
    box_jnp = jnp.array(handle.box, dtype=jnp.float64)
    topology = create_topology(
        n_atoms=n_atoms,
        bonds=jnp.array(bond_arr),
        angles=jnp.array(angle_arr),
        torsions=jnp.array(torsion_arr) if torsion_arr.shape[0] > 0 else jnp.empty((0, 4), dtype=jnp.int32),
        impropers=jnp.array(improper_arr) if improper_arr.shape[0] > 0 else jnp.empty((0, 4), dtype=jnp.int32),
    )

    # Build the param-mapping function
    build_params = _build_jaxmd_params_fn(handle)

    # Build jax-md energy function
    # This returns (energy_fn, neighbor_fn, displacement_fn)
    _unused_energy_fn, neighbor_fn, displacement_fn = oplsaa_energy(
        topology,
        create_parameters(
            bond_k=jnp.ones(max(bond_arr.shape[0], 1)),
            bond_r0=jnp.ones(max(bond_arr.shape[0], 1)),
            angle_k=jnp.ones(max(angle_arr.shape[0], 1)),
            angle_theta0=jnp.ones(max(angle_arr.shape[0], 1)),
            torsion_k=jnp.ones(max(torsion_arr.shape[0], 1)),
            torsion_n=torsion_n_static if torsion_arr.shape[0] > 0 else jnp.ones(1),
            torsion_gamma=torsion_gamma_static if torsion_arr.shape[0] > 0 else jnp.zeros(1),
            improper_k=improper_k_static,
            improper_n=improper_n_static,
            improper_gamma=improper_gamma_static,
            charges=jnp.zeros(n_atoms),
            sigma=jnp.ones(n_atoms),
            epsilon=jnp.zeros(n_atoms),
        ),
        box_jnp,
        coulomb,
        nb_options,
    )

    # Initial neighbor list
    coords_init = jnp.array(molecule.geometry, dtype=jnp.float64)
    nlist = neighbor_fn.allocate(coords_init)

    # Now build our wrapper that takes (param_vector, coords) and
    # re-parameterizes the jax-md energy function on the fly.
    # Since jax-md's energy functions close over params at build time,
    # we need to rebuild the bonded energy terms with dynamic params.
    # The approach: compute bonded energies directly using the topology
    # arrays and jax-md's primitives, while using jax-md's LJ and
    # Coulomb for nonbonded.

    # Actually, looking at jax-md's energy.py, the bonded energy functions
    # close over `bonded = params.bonded` at build time. To make params
    # dynamic, we need to rebuild the inner functions or compute directly.
    #
    # Simplest correct approach: reimplement the energy computation using
    # jax-md's topology arrays and displacement_fn, with dynamic params
    # from our param vector. This gives us full differentiability.

    from jax_md.util import safe_norm, safe_arccos, normalize
    from jax import vmap

    _bond_indices = jnp.array(bond_arr)
    _angle_indices = jnp.array(angle_arr)
    _torsion_indices = jnp.array(torsion_arr) if torsion_arr.shape[0] > 0 else None
    has_bonds = bond_arr.shape[0] > 0
    has_angles = angle_arr.shape[0] > 0
    has_torsions = torsion_arr.shape[0] > 0
    has_vdw = handle.n_vdw_types > 0

    def energy_fn(param_vector: jnp.ndarray, coords: jnp.ndarray, nlist_: object) -> dict[str, jnp.ndarray]:
        """Compute total energy given param vector and coordinates.

        Returns dict with per-term breakdown.
        """
        bonded_params, nb_params = build_params(param_vector)
        bond_k, bond_r0, angle_k, angle_theta0, torsion_k_dyn = bonded_params
        charges, sigma, epsilon = nb_params

        E_bond = jnp.float64(0.0)
        E_angle = jnp.float64(0.0)
        E_torsion = jnp.float64(0.0)
        E_lj = jnp.float64(0.0)
        E_coulomb = jnp.float64(0.0)

        # Bond energy: E = k * (r - r0)²
        if has_bonds:
            i, j = _bond_indices[:, 0], _bond_indices[:, 1]
            disp = vmap(displacement_fn)(coords[i], coords[j])
            r = safe_norm(disp)
            E_bond = jnp.sum(bond_k * (r - bond_r0) ** 2)

        # Angle energy: E = k * (theta - theta0)²
        if has_angles:
            i, j, k = _angle_indices[:, 0], _angle_indices[:, 1], _angle_indices[:, 2]
            rij = vmap(displacement_fn)(coords[i], coords[j])
            rkj = vmap(displacement_fn)(coords[k], coords[j])
            rij_norm = normalize(rij)
            rkj_norm = normalize(rkj)
            cos_theta = jnp.sum(rij_norm * rkj_norm, axis=-1)
            theta = safe_arccos(cos_theta)
            E_angle = jnp.sum(angle_k * (theta - angle_theta0) ** 2)

        # Torsion energy: E = k * (1 + cos(n*phi - gamma))
        if has_torsions:
            idx = _torsion_indices

            def compute_dihedral(p0: jnp.ndarray, p1: jnp.ndarray, p2: jnp.ndarray, p3: jnp.ndarray) -> jnp.ndarray:
                """Compute signed dihedral angle (radians) for four points."""
                b0 = displacement_fn(p1, p0)
                b1 = displacement_fn(p2, p1)
                b2 = displacement_fn(p3, p2)
                n1 = jnp.cross(b0, b1)
                n2 = jnp.cross(b1, b2)
                # atan2-based signed dihedral (preserves sign for γ ≠ 0)
                b1_norm = safe_norm(b1)
                m1 = jnp.cross(n1, b1 / jnp.maximum(b1_norm, 1e-10))
                x = jnp.sum(n1 * n2)
                y = jnp.sum(m1 * n2)
                phi = jnp.arctan2(y, x)
                return phi

            phi = vmap(compute_dihedral)(
                coords[idx[:, 0]],
                coords[idx[:, 1]],
                coords[idx[:, 2]],
                coords[idx[:, 3]],
            )
            E_torsion = jnp.sum(torsion_k_dyn * (1 + jnp.cos(torsion_n_static * phi - torsion_gamma_static)))

        # Nonbonded: use jax-md's LJ and Coulomb via the topology
        # For now, compute LJ inline with the topology exclusion/1-4 masks
        if has_vdw:
            n = coords.shape[0]
            max_neighbors = nlist_.idx.shape[1]
            idx_i = jnp.repeat(jnp.arange(n)[:, None], max_neighbors, axis=1)
            idx_j = nlist_.idx
            valid = (idx_j >= 0) & (idx_j < n)
            idx_j_safe = jnp.where(valid, idx_j, 0)
            idx_i_safe = jnp.where(valid, idx_i, 0)
            ri = coords[idx_i_safe]
            rj = coords[idx_j_safe]
            batched_disp = vmap(vmap(displacement_fn, in_axes=(0, 0)), in_axes=(0, 0))
            disp_nb = batched_disp(ri, rj)
            r_sq = jnp.sum(disp_nb**2, axis=-1)
            r_sq_safe = jnp.maximum(r_sq, 1e-4)
            r_nb = jnp.sqrt(r_sq_safe)

            sigma_i = sigma[idx_i_safe]
            sigma_j = sigma[idx_j_safe]
            epsilon_i = epsilon[idx_i_safe]
            epsilon_j = epsilon[idx_j_safe]
            sigma_ij = jnp.sqrt(sigma_i * sigma_j)
            epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)

            sr = sigma_ij / jnp.sqrt(r_sq_safe)
            sr6 = sr**6
            lj_val = 4.0 * epsilon_ij * (sr6**2 - sr6)

            same = idx_i_safe == idx_j_safe
            excluded = topology.exclusion_mask[idx_i_safe, idx_j_safe]
            is_14 = topology.pair_14_mask[idx_i_safe, idx_j_safe]
            include = valid & (~same) & (~excluded) & (r_nb < nb_options.r_cut)
            scale = jnp.where(is_14, nb_options.scale_14_lj, 1.0)
            E_lj = 0.5 * jnp.sum(jnp.where(include, scale * lj_val, 0.0))

        # Coulomb
        E_coulomb = coulomb.energy(
            coords,
            charges,
            box_jnp,
            topology.exclusion_mask,
            topology.pair_14_mask,
            nlist_,
            nb_options.scale_14_coul,
        )

        E_total = E_bond + E_angle + E_torsion + E_lj + E_coulomb
        return {
            "bond": E_bond,
            "angle": E_angle,
            "torsion": E_torsion,
            "lj": E_lj,
            "coulomb": E_coulomb,
            "total": E_total,
        }

    def scalar_energy_fn(param_vector: jnp.ndarray, coords: jnp.ndarray, nlist_: object) -> jnp.ndarray:
        """Scalar total energy for gradient/hessian computation."""
        return energy_fn(param_vector, coords, nlist_)["total"]

    # JIT-compile both energy functions for performance
    energy_fn = jax.jit(energy_fn)
    scalar_energy_fn = jax.jit(scalar_energy_fn)

    return energy_fn, scalar_energy_fn, neighbor_fn, nlist


# ---------------------------------------------------------------------------
# JaxMDEngine
# ---------------------------------------------------------------------------


@register_mm("jax-md")
class JaxMDEngine(MMEngine):
    """Differentiable MM backend using jax-md for periodic systems.

    Wraps jax-md's OPLSAA energy functions for molecular mechanics with
    analytical parameter gradients via ``jax.grad``.

    Args:
        box: Simulation box dimensions as ``(Lx, Ly, Lz)`` in Å.
            Defaults to ``(100.0, 100.0, 100.0)``.
        coulomb: Electrostatics handler. Defaults to
            :class:`CutoffCoulomb(r_cut=12.0)`.
        nb_options: Nonbonded interaction options. Defaults to
            :class:`NonbondedOptions(r_cut=12.0)`.

    Example:
        >>> from q2mm.backends.mm.jax_md_engine import JaxMDEngine
        >>> engine = JaxMDEngine(box=(50.0, 50.0, 50.0))
        >>> energy = engine.energy(molecule, forcefield)
        >>> energy, grad = engine.energy_and_param_grad(molecule, forcefield)

    """

    def __init__(
        self,
        box: tuple[float, float, float] = (100.0, 100.0, 100.0),
        coulomb: CoulombHandler | None = None,
        nb_options: NonbondedOptions | None = None,
    ) -> None:
        _ensure_jax_md()
        self._box = np.array(box, dtype=np.float64)
        self._coulomb = coulomb if coulomb is not None else CutoffCoulomb(r_cut=12.0)
        self._nb_options = nb_options if nb_options is not None else NonbondedOptions(r_cut=12.0)
        devices = jax.devices()
        logger.info("JAX-MD devices: %s", [str(d) for d in devices])

    @property
    def name(self) -> str:
        """Return the engine display name including device type."""
        backend = jax.default_backend()
        return f"JAX-MD (OPLSAA, {backend})"

    def supported_functional_forms(self) -> frozenset[str]:
        """OPLSAA uses harmonic bonds/angles + proper torsions + LJ 12-6."""
        return frozenset({"harmonic"})

    def is_available(self) -> bool:
        """Return whether JAX-MD dependencies are installed."""
        return _HAS_JAX_MD

    def supports_runtime_params(self) -> bool:
        """Return True — JAX-MD supports runtime parameter updates."""
        return True

    def supports_analytical_gradients(self) -> bool:
        """Return True — JAX-MD supports analytical gradients via autodiff."""
        return True

    def create_context(
        self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField | None = None
    ) -> JaxMDHandle:
        """Build jax-md topology and compile energy function.

        Args:
            structure: A :class:`Q2MMMolecule` or :class:`JaxMDHandle`.
            forcefield: Force field parameters.

        Returns:
            JaxMDHandle: Compiled handle for energy evaluation.

        """
        if forcefield is not None:
            self._validate_forcefield(forcefield)
        molecule = coerce_molecule(structure, engine_name="JaxMDEngine")
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

        # Match torsions — each detected torsion may match multiple FF
        # entries (one per periodicity component)
        torsion_atom_indices: list[tuple[int, int, int, int]] = []
        torsion_param_map: list[int] = []
        torsion_param_index = {id(p): i for i, p in enumerate(forcefield.torsions)}
        for torsion in molecule.torsions:
            matches = forcefield.match_torsion(
                torsion.element_quad, env_id=torsion.env_id, ff_row=torsion.ff_row, is_improper=False
            )
            for param in matches:
                j_ff = torsion_param_index[id(param)]
                torsion_atom_indices.append((torsion.atom_i, torsion.atom_j, torsion.atom_k, torsion.atom_l))
                torsion_param_map.append(j_ff)

        # Match vdW
        atom_vdw_map = []
        for symbol, atom_type in zip(molecule.symbols, molecule.atom_types, strict=False):
            idx, param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
            atom_vdw_map.append(idx if idx is not None else -1)

        # Validate vdW
        unmatched = [i for i, idx in enumerate(atom_vdw_map) if idx == -1]
        if getattr(forcefield, "vdws", None) and unmatched:
            raise ValueError(
                f"Unmatched vdW parameters for atoms at indices {unmatched}. "
                "Ensure all atom types/elements have corresponding vdW "
                "parameters in the force field."
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

        handle = JaxMDHandle(
            molecule=copy.deepcopy(molecule),
            box=self._box.copy(),
            bond_indices=bond_indices_arr,
            angle_indices=angle_indices_arr,
            torsion_indices=torsion_indices_arr,
            bond_param_map=np.array(bond_param_map, dtype=np.int32),
            angle_param_map=np.array(angle_param_map, dtype=np.int32),
            torsion_param_map=np.array(torsion_param_map, dtype=np.int32),
            atom_vdw_map=np.array(atom_vdw_map, dtype=np.int32),
            n_bond_types=len(forcefield.bonds),
            n_angle_types=len(forcefield.angles),
            n_torsion_types=len(forcefield.torsions),
            n_vdw_types=len(forcefield.vdws),
            n_atoms=len(molecule.symbols),
        )

        # Compile energy function
        energy_fn, scalar_energy_fn, neighbor_fn, nlist = _compile_energy_fn(
            handle,
            forcefield,
            self._coulomb,
            self._nb_options,
        )
        handle._energy_fn = energy_fn
        handle._scalar_energy_fn = scalar_energy_fn
        handle._neighbor_fn = neighbor_fn
        handle._nlist = nlist

        return handle

    def _get_handle(self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField) -> JaxMDHandle:
        if isinstance(structure, JaxMDHandle):
            return structure
        return self.create_context(coerce_molecule(structure, engine_name="JaxMDEngine"), forcefield)

    def _params_and_coords(self, handle: JaxMDHandle, forcefield: ForceField) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract JAX arrays from force field and molecule."""
        return _params_and_coords_impl(handle.molecule.geometry, forcefield)

    def energy(self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField) -> float:
        """Calculate energy in kcal/mol.

        Args:
            structure: Molecule or handle.
            forcefield: Force field parameters.

        Returns:
            float: Total energy in kcal/mol.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        result = handle._energy_fn(params, coords, handle._nlist)
        return float(result["total"])

    def energy_breakdown(self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField) -> dict[str, float]:
        """Calculate energy with per-term breakdown.

        Args:
            structure: Molecule or handle.
            forcefield: Force field parameters.

        Returns:
            dict: Energy components (bond, angle, torsion, lj, coulomb, total)
                in kcal/mol.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        result = handle._energy_fn(params, coords, handle._nlist)
        return {k: float(v) for k, v in result.items()}

    def energy_and_param_grad(
        self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField
    ) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. FF parameters.

        Args:
            structure: Molecule or handle.
            forcefield: Force field parameters.

        Returns:
            tuple: ``(energy, grad)`` where energy is kcal/mol and grad
                has same shape as ``forcefield.get_param_vector()``.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._grad_fn is None:
            handle._grad_fn = jax.jit(jax.value_and_grad(handle._scalar_energy_fn, argnums=0))

        val, grad = handle._grad_fn(params, coords, handle._nlist)
        return float(val), np.asarray(grad)

    def supports_batched_energy(self) -> bool:  # noqa: D102
        """Return True; JaxMDEngine supports batched energy evaluation via jax.vmap."""
        return True

    def batched_energy(
        self,
        structure: Q2MMMolecule | JaxMDHandle,
        forcefield: ForceField,
        param_matrix: np.ndarray,
    ) -> np.ndarray:
        """Evaluate energy for a batch of parameter vectors via ``jax.vmap``.

        Args:
            structure: Molecule or cached :class:`JaxMDHandle`.
            forcefield: Base force field (topology only).
            param_matrix: Shape ``(batch, n_params)`` parameter vectors.

        Returns:
            np.ndarray: Shape ``(batch,)`` energies in kcal/mol.

        """
        handle = self._get_handle(structure, forcefield)
        coords = jnp.array(handle.molecule.geometry, dtype=jnp.float64)
        batch_params = jnp.array(param_matrix, dtype=jnp.float64)
        nlist = handle._nlist

        if not hasattr(handle, "_batched_energy_fn") or handle._batched_energy_fn is None:
            handle._batched_energy_fn = jax.jit(jax.vmap(handle._scalar_energy_fn, in_axes=(0, None, None)))

        return np.asarray(handle._batched_energy_fn(batch_params, coords, nlist))

    def hessian(self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField) -> np.ndarray:
        """Compute Hessian (d²E/dcoords²) in Hartree/Bohr².

        Args:
            structure: Molecule or handle.
            forcefield: Force field parameters.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian.

        """
        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)

        if handle._coord_hess_fn is None:
            nlist = handle._nlist

            def _energy_of_flat_coords(flat_coords: jnp.ndarray, params_: jnp.ndarray) -> jnp.ndarray:
                return handle._scalar_energy_fn(params_, flat_coords.reshape(-1, 3), nlist)

            handle._coord_hess_fn = jax.jit(jax.hessian(_energy_of_flat_coords, argnums=0))

        flat_coords = coords.flatten()
        hess_kcal_a2 = handle._coord_hess_fn(flat_coords, params)
        return np.asarray(hess_kcal_a2) * KCALMOLA2_TO_HESSIAN_AU

    def minimize(
        self, structure: Q2MMMolecule | JaxMDHandle, forcefield: ForceField, max_iterations: int = 200
    ) -> tuple:
        """Minimize energy w.r.t. coordinates.

        Args:
            structure: Molecule or handle.
            forcefield: Force field parameters.
            max_iterations: Maximum optimizer iterations.

        Returns:
            tuple: ``(energy, atoms, coords)``.

        """
        from scipy.optimize import minimize as scipy_minimize

        handle = self._get_handle(structure, forcefield)
        params, coords = self._params_and_coords(handle, forcefield)
        neighbor_fn = handle._neighbor_fn

        scalar_fn = handle._scalar_energy_fn

        # Mutable container for neighbor list — updated when atoms move
        nlist_ref = [handle._nlist]

        def _update_nlist(new_coords: jnp.ndarray) -> None:
            """Re-allocate neighbor list for new coordinates."""
            nlist_ref[0] = neighbor_fn.allocate(new_coords)

        def objective(x: np.ndarray) -> float:
            """Evaluate energy at flat coordinate vector *x*."""
            c = jnp.array(x).reshape(-1, 3)
            _update_nlist(c)
            return float(scalar_fn(params, c, nlist_ref[0]))

        coord_grad_fn = jax.jit(jax.grad(lambda c, p, nl: scalar_fn(p, c.reshape(-1, 3), nl), argnums=0))

        def gradient(x: np.ndarray) -> np.ndarray:
            """Return energy gradient w.r.t. flat coordinate vector *x*."""
            c = jnp.array(x)
            _update_nlist(c.reshape(-1, 3))
            return np.asarray(coord_grad_fn(c, params, nlist_ref[0]))

        x0 = np.asarray(coords.flatten())

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
# Helpers
# ---------------------------------------------------------------------------
