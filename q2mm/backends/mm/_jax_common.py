"""Shared utilities for JAX-based MM backends.

Contains parameter-vector offset calculations and ForceField matching
helpers used by both :mod:`jax_engine` and :mod:`jax_md_engine`, plus
this backend layer's ``jax``/``jnp``/``jaxopt`` module globals and
``ensure_jax``/``ensure_jaxopt`` entry points.

The actual JAX-import-guard and float64-configuration logic is *not*
implemented here — it lives in the dependency-free, foundational
:mod:`q2mm._jax_support` (shared with :mod:`q2mm.models.hessian`, which
cannot import anything under ``q2mm.backends``). ``ensure_jax`` below
is a thin backend-local wrapper: it delegates to
:func:`q2mm._jax_support.load_jax` and rebinds this module's own
``jax``/``jnp`` globals, so existing callers that do
``from q2mm.backends.mm._jax_common import jax, jnp`` (or call
``ensure_jax(...)``) keep working unchanged. ``ensure_jaxopt`` and the
``jaxopt`` global are backend-specific (only the MM backends' optimizer
integrations use jaxopt) and have no counterpart in
``q2mm._jax_support``.

JAX is imported lazily — :func:`ensure_jax` performs the actual import
and CUDA initialization on first use, so merely importing this module
does not allocate GPU memory.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

from q2mm._jax_support import has_jax, load_jax
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam

if TYPE_CHECKING:
    from q2mm.models.parameters import ParameterLayout

# Cheap availability check — does NOT import JAX or initialize CUDA.
_HAS_JAX: bool = has_jax()
_HAS_JAXOPT: bool = importlib.util.find_spec("jaxopt") is not None

# These are populated lazily by ensure_jax() / ensure_jaxopt().
jax: ModuleType | None = None
jnp: ModuleType | None = None
jaxopt: ModuleType | None = None
_jax_initialized: bool = False


def ensure_jax(engine_name: str = "JaxEngine") -> None:
    """Import JAX and configure float64 on first call.

    Subsequent calls are no-ops.  Thin wrapper over the shared
    :func:`q2mm._jax_support.load_jax`: delegates the actual
    import-guard/float64-configuration logic there, then rebinds this
    module's own ``jax``/``jnp`` globals so existing
    ``from q2mm.backends.mm._jax_common import jax, jnp``-style callers
    keep working unchanged.

    Args:
        engine_name: Name of the engine requesting JAX, used in the
            error message.

    Raises:
        ImportError: If the ``jax`` package cannot be imported.

    """
    global jax, jnp, _jax_initialized  # noqa: PLW0603

    if _jax_initialized:
        return

    jax, jnp = load_jax(caller_name=engine_name)
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


def layout_block_offsets(layout: ParameterLayout) -> dict[str, int]:
    """Derive parameter-vector block-start offsets from a ParameterLayout.

    Rather than independently re-deriving the canonical
    bond/angle/torsion/sb/vdw/ub block order and per-type slot counts,
    this reads the actual slot indices straight from *layout* — the one
    source of truth for full-vector order (see
    :mod:`q2mm.models.parameters`).

    Each bond contributes 2 values (k, r0) at consecutive indices, each
    angle 2 (k, theta0), each torsion 1 (k), each stretch-bend 1 (k),
    each vdW 2 (radius, epsilon), and each Urey-Bradley angle 2 (k, eq)
    — the block for a kind with zero slots collapses to ``len(layout)``
    (an unused, out-of-range sentinel; callers only read a block's
    offset when their own ``has_<kind>`` flag is ``True``).

    Args:
        layout: The force field's :class:`~q2mm.models.parameters.ParameterLayout`.

    Returns:
        dict with keys ``"bond"``, ``"angle"``, ``"torsion"``, ``"sb"``,
        ``"vdw"``, ``"ub"`` mapping to the starting index of each block
        in the flat parameter vector.

    """
    from q2mm.models.parameters import ParameterKind

    total = len(layout)
    by_kind = layout.indices_by_kind

    def _block_start(kind: ParameterKind) -> int:
        indices = by_kind.get(kind)
        return indices[0] if indices else total

    return {
        "bond": _block_start(ParameterKind.BOND_FORCE_CONSTANT),
        "angle": _block_start(ParameterKind.ANGLE_FORCE_CONSTANT),
        "torsion": _block_start(ParameterKind.TORSION_FORCE_CONSTANT),
        "sb": _block_start(ParameterKind.STRETCH_BEND_FORCE_CONSTANT),
        "vdw": _block_start(ParameterKind.VDW_RADIUS),
        "ub": _block_start(ParameterKind.UREY_BRADLEY_FORCE_CONSTANT),
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
    from q2mm.models.parameters import ParameterLayout

    params = jnp.array(ParameterLayout.from_force_field(forcefield).vector(forcefield), dtype=jnp.float64)  # type: ignore[union-attr]
    coords = jnp.array(molecule_geometry, dtype=jnp.float64)  # type: ignore[union-attr]
    return params, coords
