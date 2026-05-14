"""Shared utilities for JAX-based MM backends.

Contains the JAX import guard, float64 configuration, parameter-vector
offset calculations, and ForceField matching helpers used by both
:mod:`jax_engine` and :mod:`jax_md_engine`.

JAX is imported lazily — :func:`ensure_jax` performs the actual import
and CUDA initialization on first use, so merely importing this module
does not allocate GPU memory.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Sequence
from types import ModuleType

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam

# Cheap availability check — does NOT import JAX or initialize CUDA.
_HAS_JAX: bool = importlib.util.find_spec("jax") is not None
_HAS_JAXOPT: bool = importlib.util.find_spec("jaxopt") is not None

# These are populated lazily by ensure_jax() / ensure_jaxopt().
jax: ModuleType | None = None
jnp: ModuleType | None = None
jaxopt: ModuleType | None = None
_jax_initialized: bool = False


def ensure_jax(engine_name: str = "JaxEngine") -> None:
    """Import JAX and configure float64 on first call.

    Subsequent calls are no-ops.  This is the single entry point that
    triggers ``import jax`` and any associated XLA/CUDA initialization.

    Args:
        engine_name: Name of the engine requesting JAX, used in the
            error message.

    Raises:
        ImportError: If the ``jax`` package cannot be imported.

    """
    global jax, jnp, _jax_initialized  # noqa: PLW0603

    if _jax_initialized:
        return
    if not _HAS_JAX:
        raise ImportError(f"JAX is required for {engine_name}. Install with: pip install jax jaxlib")

    import jax as _jax
    import jax.numpy as _jnp

    # JAX defaults to float32.  For MM parameter optimization float64 is the
    # safe default (energy differences ~1e-6 kcal/mol matter).
    #
    # Honour the standard JAX_ENABLE_X64 env-var: when the user has set it
    # explicitly, we do NOT override JAX's own interpretation.  Otherwise we
    # enable float64 (standard practice in JAX-based chemistry packages).
    _user_set_jax_enable_x64 = "JAX_ENABLE_X64" in os.environ
    if not _jax.config.jax_enable_x64 and not _user_set_jax_enable_x64:
        _jax.config.update("jax_enable_x64", True)

    jax = _jax
    jnp = _jnp
    _jax_initialized = True


def ensure_jaxopt() -> None:
    """Import jaxopt, ensuring JAX float64 is configured first.

    Subsequent calls are no-ops.

    Raises:
        ImportError: If the ``jaxopt`` package cannot be imported.

    """
    global jaxopt  # noqa: PLW0603
    if jaxopt is not None:
        return
    if not _HAS_JAXOPT:
        raise ImportError("jaxopt is required. Install with: pip install q2mm[jax]")
    ensure_jax(engine_name="jaxopt")

    import jaxopt as _jaxopt

    jaxopt = _jaxopt


def compute_param_offsets(
    n_bond_types: int,
    n_angle_types: int,
    n_torsion_types: int,
    n_vdw_types: int = 0,
    n_sb_types: int = 0,
) -> dict[str, int]:
    """Compute parameter vector offsets for bond/angle/torsion/sb/vdw/ub blocks.

    The parameter vector layout is:
      ``[bond_k, bond_r0, ..., angle_k, angle_theta0, ..., torsion_k, ...,
      sb_k, ..., vdw_radius, vdw_eps, ..., ub_k, ub_eq, ...]``

    Each bond type contributes 2 values (k, r0), each angle type 2
    (k, theta0), each torsion type 1 (k), each stretch-bend type 1 (k),
    each vdW type 2 (radius, epsilon), and each UB type 2 (k, eq).

    Args:
        n_bond_types: Number of unique bond parameter types.
        n_angle_types: Number of unique angle parameter types.
        n_torsion_types: Number of unique torsion parameter types.
        n_vdw_types: Number of unique vdW parameter types.
        n_sb_types: Number of unique stretch-bend parameter types.

    Returns:
        dict with keys ``"bond"``, ``"angle"``, ``"torsion"``, ``"sb"``,
        ``"vdw"``, ``"ub"`` mapping to the starting index of each block
        in the flat parameter vector.

    """
    bond_offset = 0
    angle_offset = 2 * n_bond_types
    torsion_offset = angle_offset + 2 * n_angle_types
    sb_offset = torsion_offset + n_torsion_types
    vdw_offset = sb_offset + n_sb_types
    ub_offset = vdw_offset + 2 * n_vdw_types
    return {
        "bond": bond_offset,
        "angle": angle_offset,
        "torsion": torsion_offset,
        "sb": sb_offset,
        "vdw": vdw_offset,
        "ub": ub_offset,
    }


# ---------------------------------------------------------------------------
# ForceField matching helpers
# ---------------------------------------------------------------------------


def match_bond(
    forcefield: ForceField,
    elements: Sequence[str],
    env_id: str = "",
    ff_row: int | None = None,
    *,
    bond_order: str = "",
    bond_length: float | None = None,
) -> tuple[int | None, BondParam | None]:
    """Match a bond to its ForceField index.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the two bonded atoms.
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.
        bond_order: Bond order symbol (``"-"``, ``"="``, ``"*"``, ``"%"``).
        bond_length: Measured bond length in Å for closest-r₀ matching.

    Returns:
        tuple[int | None, BondParam | None]: ``(index, param)`` or
            ``(None, None)`` if unmatched.

    """
    matched = forcefield.match_bond(
        elements,
        env_id=env_id,
        ff_row=ff_row,
        bond_order=bond_order,
        bond_length=bond_length,
    )
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


def params_and_coords(
    molecule_geometry: object,
    forcefield: ForceField,
) -> tuple:
    """Extract parameter and coordinate JAX arrays.

    Shared helper for :class:`~jax_engine.JaxEngine` and
    :class:`~jax_md_engine.JaxMDEngine`.

    Args:
        molecule_geometry: ``handle.molecule.geometry`` array-like.
        forcefield: Force field whose parameter vector to extract.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(params, coords)`` as JAX
            float64 arrays.

    """
    ensure_jax()
    params = jnp.array(forcefield.get_param_vector(), dtype=jnp.float64)  # type: ignore[union-attr]
    coords = jnp.array(molecule_geometry, dtype=jnp.float64)  # type: ignore[union-attr]
    return params, coords
