"""Batched molecule evaluation using JAX vmap.

Groups molecules by topology compatibility and uses ``jax.vmap`` to
compute Hessians for multiple geometries in a single vectorized call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover
    _HAS_JAX = False

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU

if _HAS_JAX:
    from q2mm.backends.mm.jax_engine import JaxEngine, JaxHandle


@dataclass
class TopologyGroup:
    """A group of molecules sharing the same topology (handle).

    All molecules in a group have the same atom count, bond connectivity,
    angle terms, etc.  Only their coordinates differ.
    """

    handle: object  # JaxHandle
    mol_indices: list[int] = field(default_factory=list)
    geometries: list[np.ndarray] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------


def _topology_signature(handle: JaxHandle) -> str:
    """Create a hashable signature for a handle's topology.

    Two handles with the same signature are guaranteed to share identical
    connectivity and parameter mapping, so their energy functions are
    interchangeable up to coordinate differences.
    """
    parts: list[str] = [
        str(handle.n_bond_types),
        str(handle.n_angle_types),
        str(handle.n_torsion_types),
        str(handle.n_vdw_types),
        str(len(handle.bond_indices)),
        str(len(handle.angle_indices)),
        str(len(handle.torsion_indices)),
        str(len(handle.vdw_pair_indices)),
        str(handle.bond_param_map.tobytes()),
        str(handle.angle_param_map.tobytes()),
        str(handle.torsion_param_map.tobytes()),
        str(handle.atom_vdw_map.tobytes()),
        str(handle.bond_indices.tobytes()),
        str(handle.angle_indices.tobytes()),
    ]
    return "|".join(parts)


def group_by_topology(
    molecules: list[Q2MMMolecule],
    forcefield: ForceField,
    engine: JaxEngine,
    handles: dict[int, JaxHandle] | None = None,
) -> list[TopologyGroup]:
    """Group molecules that share compatible topologies.

    Two molecules are compatible if they have the same bond connectivity,
    angle connectivity, torsion connectivity, vdW pair list, and parameter
    mappings.  In practice this catches the common case of multiple
    conformations (GS, TS) of the same molecule.

    Args:
        molecules: List of molecules to group.
        forcefield: The force field (used to build handles if needed).
        engine: The JaxEngine instance.
        handles: Optional pre-built handle cache (mol_idx → JaxHandle).

    Returns:
        List of :class:`TopologyGroup` instances.

    """
    groups: dict[str, TopologyGroup] = {}

    for i, mol in enumerate(molecules):
        if handles is not None and i in handles:
            handle = handles[i]
        else:
            handle = engine.create_context(mol, forcefield)

        sig = _topology_signature(handle)

        if sig in groups:
            groups[sig].mol_indices.append(i)
            groups[sig].geometries.append(np.asarray(mol.geometry, dtype=np.float64))
        else:
            groups[sig] = TopologyGroup(
                handle=handle,
                mol_indices=[i],
                geometries=[np.asarray(mol.geometry, dtype=np.float64)],
            )

    return list(groups.values())


# ---------------------------------------------------------------------------
# Batched Hessian computation
# ---------------------------------------------------------------------------


def batched_hessians(
    group: TopologyGroup,
    forcefield: ForceField,
) -> list[np.ndarray]:
    """Compute Hessians for all molecules in a topology group using vmap.

    For groups with a single molecule, falls back to standard (non-vmap)
    evaluation.  For groups with multiple molecules, uses ``jax.vmap``
    over flattened coordinates so that all Hessians are computed in a
    single vectorised call.

    Args:
        group: A :class:`TopologyGroup` with compatible molecules.
        forcefield: The force field.

    Returns:
        List of ``(3N, 3N)`` Hessian arrays in Hartree/Bohr², one per
        molecule in the order of ``group.mol_indices``.

    """
    handle: JaxHandle = group.handle  # type: ignore[assignment]
    params = jnp.array(forcefield.get_param_vector(), dtype=jnp.float64)

    # Ensure the single-molecule coord Hessian function is compiled
    if handle._coord_hess_fn is None:

        def _energy_of_flat_coords(
            flat_coords: jnp.ndarray,
            params_: jnp.ndarray,
        ) -> jnp.ndarray:
            return handle._energy_fn(params_, flat_coords.reshape(-1, 3))

        handle._coord_hess_fn = jax.jit(
            jax.hessian(_energy_of_flat_coords, argnums=0),
        )

    if len(group.geometries) == 1:
        flat_coords = jnp.array(group.geometries[0], dtype=jnp.float64).flatten()
        hess = handle._coord_hess_fn(flat_coords, params)
        return [np.asarray(hess) * KCALMOLA2_TO_HESSIAN_AU]

    # --- Multiple molecules: vmap path ---
    if handle._batched_coord_hess_fn is None:
        handle._batched_coord_hess_fn = jax.jit(
            jax.vmap(handle._coord_hess_fn, in_axes=(0, None)),
        )

    batch_coords = jnp.stack(
        [jnp.array(g, dtype=jnp.float64).flatten() for g in group.geometries],
    )  # (n_mols, 3*n_atoms)

    batch_hess = handle._batched_coord_hess_fn(batch_coords, params)
    # batch_hess shape: (n_mols, 3N, 3N)

    return [np.asarray(batch_hess[i]) * KCALMOLA2_TO_HESSIAN_AU for i in range(len(group.geometries))]


def batched_frequencies(
    group: TopologyGroup,
    forcefield: ForceField,
    symbols_per_mol: list[list[str]],
) -> list[list[float]]:
    """Compute frequencies for all molecules in a group.

    Uses :func:`batched_hessians` internally, then converts each
    Hessian to vibrational frequencies.

    Args:
        group: A :class:`TopologyGroup` with compatible molecules.
        forcefield: The force field.
        symbols_per_mol: Atomic symbols for each molecule (same order
            as ``group.mol_indices``).

    Returns:
        List of frequency lists (cm⁻¹), one per molecule.

    """
    from q2mm.models.hessian import hessian_to_frequencies

    hessians = batched_hessians(group, forcefield)
    return [hessian_to_frequencies(hess, symbols_per_mol[i]) for i, hess in enumerate(hessians)]
