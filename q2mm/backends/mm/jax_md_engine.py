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

**Relationship to JaxBackend:** JaxBackend is a lightweight gas-phase backend
with hand-rolled energy functions. JaxMdBackend wraps jax-md's full
OPLSAA implementation including torsions, electrostatics, and PBC support.
Use JaxBackend for simple gas-phase work, JaxMdBackend when you need the
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

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendUnavailableError,
    BatchedEnergyRequest,
    BatchedEnergyResult,
    Capability,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
    FrequencyRequest,
    FrequencyResult,
    FrequencyUnit,
    GeometryResult,
    HessianRequest,
    HessianResult,
    HessianUnit,
    LengthUnit,
    MinimizationRequest,
    ParameterGradientRequest,
    ParameterGradientResult,
    PreparationError,
    PreparationRequest,
    readonly_array,
)
from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Molecule

from q2mm.backends.mm._jax_common import (
    ensure_jax as _ensure_jax_common,
    layout_block_offsets,
    match_angle as _match_angle,
    match_bond as _match_bond,
    match_vdw as _match_vdw,
    params_and_coords as _params_and_coords_impl,
)

# Lazy placeholders — populated by _ensure_jax_md().
jax = None
jnp = None

# Cheap availability check — does NOT import jax-md or initialize CUDA.
_HAS_JAX_MD: bool = importlib.util.find_spec("jax_md") is not None

# Lazy-loaded jax-md symbols (populated by _ensure_jax_md).
NonbondedOptions = None
CoulombHandler = None
CutoffCoulomb = None
oplsaa_energy = None
create_parameters = None
create_topology = None

# Hessian unit conversion imported from q2mm.models.units


def _ensure_jax_md() -> None:
    """Import JAX and jax-md lazily, raising ``ImportError`` if missing."""
    global jax, jnp  # noqa: PLW0603
    _ensure_jax_common("JaxMdBackend")
    import q2mm.backends.mm._jax_common as _jc

    jax = _jc.jax
    jnp = _jc.jnp

    if not _HAS_JAX_MD:
        raise ImportError("jax-md is required for JaxMdBackend. Install with: pip install jax-md")

    global NonbondedOptions, CoulombHandler, CutoffCoulomb  # noqa: PLW0603
    global oplsaa_energy, create_parameters, create_topology  # noqa: PLW0603

    if NonbondedOptions is None:
        from jax_md.mm_forcefields.base import NonbondedOptions as _NBO
        from jax_md.mm_forcefields.nonbonded.electrostatics import (
            CoulombHandler as _CH,
            CutoffCoulomb as _CC,
        )
        from jax_md.mm_forcefields.oplsaa import energy as _energy
        from jax_md.mm_forcefields.oplsaa.params import create_parameters as _cp
        from jax_md.mm_forcefields.oplsaa.topology import create_topology as _ct

        NonbondedOptions = _NBO
        CoulombHandler = _CH
        CutoffCoulomb = _CC
        oplsaa_energy = _energy
        create_parameters = _cp
        create_topology = _ct


# ---------------------------------------------------------------------------
# Handle (cached topology + compiled functions)
# ---------------------------------------------------------------------------


@dataclass
class _JaxMdState:
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

    molecule: Molecule
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
    n_sb_types: int = 0
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
    state: _JaxMdState,
    forcefield: ForceField,
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
        state: A :class:`_JaxMdState` with topology mappings populated.
        forcefield: The force field *state* was built from — used only
            to derive the parameter-vector block offsets via its
            :class:`~q2mm.models.parameters.ParameterLayout`.

    Returns:
        Callable: ``build_params(param_vector) -> (bonded_tuple, nonbonded_tuple)``

    """
    n_bt = state.n_bond_types
    n_at = state.n_angle_types
    n_tt = state.n_torsion_types
    n_vt = state.n_vdw_types
    n_atoms = state.n_atoms

    bond_map = jnp.array(state.bond_param_map, dtype=jnp.int32)
    angle_map = jnp.array(state.angle_param_map, dtype=jnp.int32)
    torsion_map = jnp.array(state.torsion_param_map, dtype=jnp.int32)
    atom_vdw_map = jnp.array(state.atom_vdw_map, dtype=jnp.int32)

    # Param vector offsets, derived from the force field's ParameterLayout
    # (the one source of truth for vector order — see q2mm.models.parameters).
    from q2mm.models.parameters import ParameterLayout

    _offsets = layout_block_offsets(ParameterLayout.from_force_field(forcefield))
    bond_offset = _offsets["bond"]
    angle_offset = _offsets["angle"]
    torsion_offset = _offsets["torsion"]
    vdw_offset = _offsets["vdw"]

    n_bonds = len(state.bond_param_map)
    n_angles = len(state.angle_param_map)
    n_torsions = len(state.torsion_param_map)

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
            vdw_mask = atom_vdw_map >= 0
            safe_vdw_map = jnp.maximum(atom_vdw_map, 0)
            per_atom_radius = vdw_params[safe_vdw_map, 0] * vdw_mask
            per_atom_epsilon = vdw_params[safe_vdw_map, 1] * vdw_mask
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
    state: _JaxMdState, forcefield: ForceField, coulomb: object, nb_options: object
) -> tuple[Callable, Callable, object, object]:
    """Compile the jax-md energy function for a specific topology.

    Args:
        state: Handle with topology and mappings.
        forcefield: ForceField (used for torsion periodicity/phase).
        coulomb: CoulombHandler instance.
        nb_options: NonbondedOptions instance.

    Returns:
        tuple: (energy_fn, scalar_energy_fn, neighbor_fn, nlist) where
            energy_fn takes (param_vector, coords) and returns a dict,
            scalar_energy_fn returns just the total scalar.

    """
    # Build jax-md topology arrays
    molecule = state.molecule
    n_atoms = state.n_atoms

    # Use pre-matched atom indices from the state (parallel with param maps)
    bond_arr = state.bond_indices
    angle_arr = state.angle_indices
    torsion_arr = state.torsion_indices

    # Improper torsions — placeholder, not yet supported
    improper_arr = np.empty((0, 4), dtype=np.int32)

    # Torsion periodicity and phase (static, from ForceField)
    if len(state.torsion_param_map) > 0:
        torsion_n_static = jnp.array(
            [forcefield.torsions[state.torsion_param_map[i]].periodicity for i in range(len(state.torsion_param_map))],
            dtype=jnp.float64,
        )
        torsion_gamma_static = jnp.array(
            [
                forcefield.torsions[state.torsion_param_map[i]].phase * (jnp.pi / 180.0)
                for i in range(len(state.torsion_param_map))
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
    box_jnp = jnp.array(state.box, dtype=jnp.float64)
    topology = create_topology(
        n_atoms=n_atoms,
        bonds=jnp.array(bond_arr),
        angles=jnp.array(angle_arr),
        torsions=jnp.array(torsion_arr) if torsion_arr.shape[0] > 0 else jnp.empty((0, 4), dtype=jnp.int32),
        impropers=jnp.array(improper_arr) if improper_arr.shape[0] > 0 else jnp.empty((0, 4), dtype=jnp.int32),
    )

    # Build the param-mapping function
    build_params = _build_jaxmd_params_fn(state, forcefield)

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
    has_vdw = state.n_vdw_types > 0

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
# JaxMdBackend
# ---------------------------------------------------------------------------


class JaxMdBackend:
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

    """

    def __init__(
        self,
        box: tuple[float, float, float] = (100.0, 100.0, 100.0),
        coulomb: CoulombHandler | None = None,
        nb_options: NonbondedOptions | None = None,
    ) -> None:
        if not _HAS_JAX_MD:
            raise BackendUnavailableError("jax-md is not installed. Install with `pip install jax-md`.")
        _ensure_jax_md()
        self._box = np.array(box, dtype=np.float64)
        self._coulomb = coulomb if coulomb is not None else CutoffCoulomb(r_cut=12.0)
        self._nb_options = nb_options if nb_options is not None else NonbondedOptions(r_cut=12.0)

    @property
    def info(self) -> BackendInfo:
        """Immutable capability declaration for this backend."""
        device = jax.default_backend()
        provenance = BackendProvenance(
            backend="jax-md",
            role=BackendRole.MM,
            version=getattr(jax, "__version__", ""),
            details={
                "implementation": {"name": "JAX-MD", "jax_version": getattr(jax, "__version__", "")},
                "model": {"identity": "OPLSAA"},
                "platform": {"backend": device},
            },
        )
        return BackendInfo(
            name=f"JAX-MD (OPLSAA, {device})",
            role=BackendRole.MM,
            capabilities=frozenset(
                {
                    Capability.ENERGY,
                    Capability.MINIMIZE,
                    Capability.HESSIAN,
                    Capability.FREQUENCIES,
                    Capability.PARAMETER_GRADIENT,
                    Capability.BATCHED_ENERGY,
                    Capability.REUSABLE_STATE,
                }
            ),
            functional_forms=frozenset({"harmonic"}),
            provenance=provenance,
        )

    def prepare(self, request: PreparationRequest) -> PreparedJaxMd:
        """Build a prepared session for one training case.

        Args:
            request: Preparation request carrying the molecule and base
                force field.

        Returns:
            PreparedJaxMd: A per-case session owning a compiled
                :class:`_JaxMdState`.

        Raises:
            PreparationError: If no force field is supplied or its functional
                form is unsupported.

        """
        from q2mm.models.parameters import ParameterLayout

        if request.force_field is None:
            raise PreparationError("JAX-MD requires a base ForceField in the PreparationRequest.")
        info = self.info
        form = request.force_field.functional_form.value
        if not info.supports_form(form):
            raise PreparationError(
                f"JAX-MD does not support functional form {form!r}. Supported: {sorted(info.functional_forms)}"
            )
        layout = ParameterLayout.from_force_field(request.force_field)
        try:
            state = self._build_state(request.molecule, request.force_field)
        except PreparationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise PreparationError(f"JAX-MD failed to prepare case {request.case_id!r}: {exc}") from exc
        return PreparedJaxMd(
            backend=self,
            info=info,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=layout,
            state=state,
        )

    def _build_state(self, molecule: Molecule, forcefield: ForceField) -> _JaxMdState:
        """Build jax-md topology and compile energy function.

        Requires an explicit :class:`ForceField`; no auto-generation and no
        state pass-through union.

        Args:
            molecule: The molecule to build a native state for.
            forcefield: Force field parameters.

        Returns:
            _JaxMdState: Compiled private native state for energy evaluation.

        """
        if forcefield.stretch_bends:
            raise PreparationError(
                "JAX-MD does not support stretch-bend cross terms. "
                "Use the JAX backend for force fields with stretch-bend parameters."
            )

        # Match bonds
        bond_atom_indices = []
        bond_param_map = []
        for bond in molecule.bonds:
            idx, param = _match_bond(
                forcefield,
                bond.elements,
                env_id=bond.env_id,
                ff_row=bond.ff_row,
                bond_order=getattr(bond, "bond_order", ""),
                bond_length=bond.length,
            )
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
        excluded_types = {value.casefold() for value in forcefield.nonbonded_excluded_atom_types}
        for symbol, atom_type in zip(molecule.symbols, molecule.atom_types, strict=False):
            if symbol.casefold() in excluded_types or atom_type.casefold() in excluded_types:
                atom_vdw_map.append(-1)
                continue
            idx, param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
            atom_vdw_map.append(idx if idx is not None else -1)

        # Validate vdW
        explicitly_excluded = {
            index
            for index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=True))
            if symbol.casefold() in excluded_types or atom_type.casefold() in excluded_types
        }
        unmatched = [i for i, idx in enumerate(atom_vdw_map) if idx == -1 and i not in explicitly_excluded]
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

        state = _JaxMdState(
            molecule=molecule,
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
            n_sb_types=len(forcefield.stretch_bends),
            n_atoms=len(molecule.symbols),
        )

        # Compile energy function
        energy_fn, scalar_energy_fn, neighbor_fn, nlist = _compile_energy_fn(
            state,
            forcefield,
            self._coulomb,
            self._nb_options,
        )
        state._energy_fn = energy_fn
        state._scalar_energy_fn = scalar_energy_fn
        state._neighbor_fn = neighbor_fn
        state._nlist = nlist

        return state

    def _params_and_coords(self, state: _JaxMdState, forcefield: ForceField) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Extract JAX arrays from force field and molecule."""
        return _params_and_coords_impl(state.molecule.geometry, forcefield)

    def _evaluate_energy(self, state: _JaxMdState, forcefield: ForceField) -> float:
        """Total energy in kcal/mol for a prepared native state."""
        params, coords = self._params_and_coords(state, forcefield)
        result = state._energy_fn(params, coords, state._nlist)
        return float(result["total"])

    def _evaluate_energy_breakdown(self, state: _JaxMdState, forcefield: ForceField) -> dict[str, float]:
        """Per-term energy breakdown (kcal/mol) for a prepared native state."""
        params, coords = self._params_and_coords(state, forcefield)
        result = state._energy_fn(params, coords, state._nlist)
        return {k: float(v) for k, v in result.items()}

    def _evaluate_param_grad(self, state: _JaxMdState, forcefield: ForceField) -> tuple[float, np.ndarray]:
        """Energy and analytical gradient w.r.t. FF parameters (kcal/mol)."""
        params, coords = self._params_and_coords(state, forcefield)
        if state._grad_fn is None:
            state._grad_fn = jax.jit(jax.value_and_grad(state._scalar_energy_fn, argnums=0))
        val, grad = state._grad_fn(params, coords, state._nlist)
        return float(val), np.asarray(grad)

    def _evaluate_batched_energy(
        self,
        state: _JaxMdState,
        forcefield: ForceField,
        param_matrix: np.ndarray,
    ) -> np.ndarray:
        """Evaluate a batch of parameter vectors via ``jax.vmap`` (kcal/mol)."""
        coords = jnp.array(state.molecule.geometry, dtype=jnp.float64)
        batch_params = jnp.array(param_matrix, dtype=jnp.float64)
        nlist = state._nlist

        if not hasattr(state, "_batched_energy_fn") or state._batched_energy_fn is None:
            state._batched_energy_fn = jax.jit(jax.vmap(state._scalar_energy_fn, in_axes=(0, None, None)))

        return np.asarray(state._batched_energy_fn(batch_params, coords, nlist))

    def _evaluate_hessian(self, state: _JaxMdState, forcefield: ForceField) -> np.ndarray:
        """Coordinate Hessian (d2E/dcoords2) in Hartree/Bohr2."""
        params, coords = self._params_and_coords(state, forcefield)

        if state._coord_hess_fn is None:
            nlist = state._nlist

            def _energy_of_flat_coords(flat_coords: jnp.ndarray, params_: jnp.ndarray) -> jnp.ndarray:
                return state._scalar_energy_fn(params_, flat_coords.reshape(-1, 3), nlist)

            state._coord_hess_fn = jax.jit(jax.hessian(_energy_of_flat_coords, argnums=0))

        flat_coords = coords.flatten()
        hess_kcal_a2 = state._coord_hess_fn(flat_coords, params)
        return np.asarray(hess_kcal_a2) * KCALMOLA2_TO_HESSIAN_AU

    def _evaluate_minimize(self, state: _JaxMdState, forcefield: ForceField, *, max_iterations: int = 200) -> tuple:
        """Minimize coordinates with analytical JAX gradients (L-BFGS-B)."""
        from scipy.optimize import minimize as scipy_minimize

        params, coords = self._params_and_coords(state, forcefield)
        neighbor_fn = state._neighbor_fn

        scalar_fn = state._scalar_energy_fn

        # Mutable container for neighbor list — updated when atoms move
        nlist_ref = [state._nlist]

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
        return opt_energy, list(state.molecule.symbols), opt_coords


class PreparedJaxMd(AbstractPreparedBackend):
    """Prepared JAX-MD session for a single training case.

    Owns the molecule, base force field, parameter layout, and one compiled
    private :class:`_JaxMdState` reused across parameter vectors.
    """

    def __init__(
        self,
        *,
        backend: JaxMdBackend,
        info: BackendInfo,
        case_id: str,
        molecule: Molecule,
        force_field: ForceField,
        layout: object,
        state: _JaxMdState,
    ) -> None:
        super().__init__(
            info=info,
            case_id=case_id,
            molecule=molecule,
            force_field=force_field,
            layout=layout,  # type: ignore[arg-type]
        )
        self._backend = backend
        self._state = state

    def _ff_for(self, parameters: np.ndarray) -> ForceField:
        vec = self._validate_vector(parameters)
        return self.layout.replace(self.force_field, vec)

    def _energy(self, request: EnergyRequest) -> EnergyResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        try:
            value = self._backend._evaluate_energy(self._state, ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD energy evaluation failed: {exc}") from exc
        return EnergyResult(energy=float(value), unit=EnergyUnit.KCAL_PER_MOL, provenance=self._info.provenance)

    def _minimize(self, request: MinimizationRequest) -> GeometryResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        max_iterations = request.max_iterations if request.max_iterations is not None else 200
        try:
            energy, atoms, coords = self._backend._evaluate_minimize(self._state, ff, max_iterations=max_iterations)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD minimization failed: {exc}") from exc
        return GeometryResult(
            energy=float(energy),
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=tuple(atoms),
            coordinates=readonly_array(coords),
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=self._info.provenance,
        )

    def _hessian(self, request: HessianRequest) -> HessianResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        try:
            hess = self._backend._evaluate_hessian(self._state, ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD Hessian evaluation failed: {exc}") from exc
        return HessianResult(
            hessian=readonly_array(hess), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=self._info.provenance
        )

    def _frequencies(self, request: FrequencyRequest) -> FrequencyResult:  # type: ignore[override]
        from q2mm.models.hessian import hessian_to_frequencies

        ff = self._ff_for(request.parameters)
        try:
            hess_au = self._backend._evaluate_hessian(self._state, ff)
            freqs = hessian_to_frequencies(hess_au, list(self.molecule.symbols), on_error=request.on_error)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD frequency evaluation failed: {exc}") from exc
        return FrequencyResult(
            frequencies=readonly_array(freqs), unit=FrequencyUnit.INVERSE_CM, provenance=self._info.provenance
        )

    def _parameter_gradient(self, request: ParameterGradientRequest) -> ParameterGradientResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        try:
            energy, grad = self._backend._evaluate_param_grad(self._state, ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD parameter-gradient evaluation failed: {exc}") from exc
        return ParameterGradientResult(
            energy=float(energy),
            gradient=readonly_array(grad),
            unit=EnergyUnit.KCAL_PER_MOL,
            provenance=self._info.provenance,
        )

    def _batched_energy(self, request: BatchedEnergyRequest) -> BatchedEnergyResult:  # type: ignore[override]
        mat = self._validate_matrix(request.parameter_matrix)
        try:
            energies = self._backend._evaluate_batched_energy(self._state, self.force_field, mat)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"JAX-MD batched-energy evaluation failed: {exc}") from exc
        return BatchedEnergyResult(
            energies=readonly_array(energies), unit=EnergyUnit.KCAL_PER_MOL, provenance=self._info.provenance
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
