"""Shared utilities for JAX-based MM backends.

Contains the JAX import guard, float64 configuration, parameter-vector
offset calculations, and ForceField matching helpers used by both
:mod:`jax_engine` and :mod:`jax_md_engine`.
"""

from __future__ import annotations

from collections.abc import Sequence

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam

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


def ensure_jax(engine_name: str = "JaxEngine") -> None:
    """Raise ``ImportError`` if JAX is not installed.

    Args:
        engine_name: Name of the engine requesting JAX, used in the
            error message.

    Raises:
        ImportError: If the ``jax`` package cannot be imported.

    """
    if not _HAS_JAX:
        raise ImportError(f"JAX is required for {engine_name}. Install with: pip install jax jaxlib")


def compute_param_offsets(
    n_bond_types: int,
    n_angle_types: int,
    n_torsion_types: int,
) -> dict[str, int]:
    """Compute parameter vector offsets for bond/angle/torsion/vdw blocks.

    The parameter vector layout is:
      ``[bond_k, bond_r0, ..., angle_k, angle_theta0, ..., torsion_k, ...,
      vdw_radius, vdw_eps, ...]``

    Each bond type contributes 2 values (k, r0), each angle type 2
    (k, theta0), each torsion type 1 (k), and each vdW type 2
    (radius, epsilon).

    Args:
        n_bond_types: Number of unique bond parameter types.
        n_angle_types: Number of unique angle parameter types.
        n_torsion_types: Number of unique torsion parameter types.

    Returns:
        dict with keys ``"bond"``, ``"angle"``, ``"torsion"``, ``"vdw"``
        mapping to the starting index of each block in the flat parameter
        vector.

    """
    bond_offset = 0
    angle_offset = 2 * n_bond_types
    torsion_offset = angle_offset + 2 * n_angle_types
    vdw_offset = torsion_offset + n_torsion_types
    return {
        "bond": bond_offset,
        "angle": angle_offset,
        "torsion": torsion_offset,
        "vdw": vdw_offset,
    }


# ---------------------------------------------------------------------------
# ForceField matching helpers
# ---------------------------------------------------------------------------


def match_bond(
    forcefield: ForceField,
    elements: Sequence[str],
    env_id: str = "",
    ff_row: int | None = None,
) -> tuple[int | None, BondParam | None]:
    """Match a bond to its ForceField index.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the two bonded atoms.
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.

    Returns:
        tuple[int | None, BondParam | None]: ``(index, param)`` or
            ``(None, None)`` if unmatched.

    """
    matched = forcefield.match_bond(elements, env_id=env_id, ff_row=ff_row)
    if matched is not None:
        return forcefield.bonds.index(matched), matched
    return None, None


def match_angle(
    forcefield: ForceField,
    elements: Sequence[str],
    env_id: str = "",
    ff_row: int | None = None,
) -> tuple[int | None, AngleParam | None]:
    """Match an angle to its ForceField index.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the three atoms.
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.

    Returns:
        tuple[int | None, AngleParam | None]: ``(index, param)`` or
            ``(None, None)`` if unmatched.

    """
    matched = forcefield.match_angle(elements, env_id=env_id, ff_row=ff_row)
    if matched is not None:
        return forcefield.angles.index(matched), matched
    return None, None


def match_vdw(
    forcefield: ForceField,
    atom_type: str = "",
    element: str = "",
    ff_row: int | None = None,
) -> tuple[int | None, VdwParam | None]:
    """Match a vdW parameter to its ForceField index.

    Args:
        forcefield: Force field to search.
        atom_type: Atom type label for matching.
        element: Element symbol for fallback matching.
        ff_row: Optional row index hint for matching.

    Returns:
        tuple[int | None, VdwParam | None]: ``(index, param)`` or
            ``(None, None)`` if unmatched.

    """
    matched = forcefield.match_vdw(atom_type=atom_type, element=element, ff_row=ff_row)
    if matched is not None:
        return forcefield.vdws.index(matched), matched
    return None, None
