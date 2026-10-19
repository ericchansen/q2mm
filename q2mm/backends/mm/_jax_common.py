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
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm._jax_support import has_jax, load_jax
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BatchedEnergyRequest,
    BatchedEnergyResult,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
    FrequencyRequest,
    FrequencyResult,
    HessianRequest,
    HessianResult,
    HessianUnit,
    ParameterGradientRequest,
    ParameterGradientResult,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceHessianRequest,
    readonly_array,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam
from q2mm.models.molecule import Angle, Bond, Molecule, Torsion

if TYPE_CHECKING:
    from q2mm.models.parameters import ParameterLayout

# Cheap availability check — does NOT import JAX or initialize CUDA.
_HAS_JAX: bool = has_jax()
_HAS_JAXOPT: bool = importlib.util.find_spec("jaxopt") is not None

# These are populated lazily by ensure_jax() / ensure_jaxopt().
jax: Any = None
jnp: Any = None
jaxopt: ModuleType | None = None
_jax_initialized: bool = False


def ensure_jax(engine_name: str = "JaxBackend") -> None:
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
        (elements[0], elements[1]),
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
    matched = forcefield.match_angle((elements[0], elements[1], elements[2]), env_id=env_id, ff_row=ff_row)
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

    Shared helper for :class:`~jax_engine.JaxBackend` and
    :class:`~jax_md_engine.JaxMdBackend`.

    Args:
        molecule_geometry: ``handle.molecule.geometry`` array-like.
        forcefield: Force field whose parameter vector to extract.

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: ``(params, coords)`` as JAX
            float64 arrays.

    """
    ensure_jax()
    from q2mm.models.parameters import ParameterLayout

    params = jnp.array(ParameterLayout.from_force_field(forcefield).vector(forcefield), dtype=jnp.float64)
    coords = jnp.array(molecule_geometry, dtype=jnp.float64)
    return params, coords


@dataclass(frozen=True)
class MatchedTopology:
    """Matched topology indices and ForceField parameter maps for JAX backends."""

    matched_bonds: tuple[tuple[Bond, int, BondParam], ...]
    matched_angles: tuple[tuple[Angle, int, AngleParam], ...]
    bond_indices: np.ndarray
    bond_param_map: np.ndarray
    angle_indices: np.ndarray
    angle_param_map: np.ndarray
    torsion_indices: np.ndarray
    torsion_param_map: np.ndarray
    atom_vdw_map: np.ndarray
    explicitly_excluded_atoms: frozenset[int]


def match_mm_topology(
    molecule: Molecule,
    forcefield: ForceField,
    *,
    include_impropers: bool,
    mask_excluded_vdw: bool,
) -> MatchedTopology:
    """Match bonds, angles, torsions, and vdW parameters for a JAX MM state."""
    matched_bonds: list[tuple[Bond, int, BondParam]] = []
    for bond in molecule.bonds or ():
        bond_idx, bond_param = match_bond(
            forcefield,
            bond.elements,
            env_id=bond.env_id,
            ff_row=bond.ff_row,
            bond_order=getattr(bond, "bond_order", ""),
            bond_length=bond.length,
        )
        if bond_idx is not None and bond_param is not None:
            matched_bonds.append((bond, bond_idx, bond_param))

    matched_angles: list[tuple[Angle, int, AngleParam]] = []
    for angle in molecule.angles or ():
        angle_idx, angle_param = match_angle(forcefield, angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
        if angle_idx is not None and angle_param is not None:
            matched_angles.append((angle, angle_idx, angle_param))

    torsion_atom_indices: list[tuple[int, int, int, int]] = []
    torsion_param_map_list: list[int] = []
    torsion_param_index = {id(p): i for i, p in enumerate(forcefield.torsions)}
    sources: list[tuple[Sequence[Torsion], bool]] = [(molecule.torsions or (), False)]
    if include_impropers:
        sources.append((molecule.improper_torsions, True))
    for torsion_list, is_improper in sources:
        for torsion in torsion_list:
            matches = forcefield.match_torsion(
                torsion.element_quad, env_id=torsion.env_id, ff_row=torsion.ff_row, is_improper=is_improper
            )
            for torsion_param in matches:
                torsion_atom_indices.append((torsion.atom_i, torsion.atom_j, torsion.atom_k, torsion.atom_l))
                torsion_param_map_list.append(torsion_param_index[id(torsion_param)])

    atom_types = molecule.atom_types if molecule.atom_types is not None else molecule.symbols
    excluded_types = {value.casefold() for value in forcefield.nonbonded_excluded_atom_types}
    explicitly_excluded = frozenset(
        index
        for index, (symbol, atom_type) in enumerate(zip(molecule.symbols, atom_types, strict=True))
        if symbol.casefold() in excluded_types or atom_type.casefold() in excluded_types
    )
    atom_vdw_map_list: list[int] = []
    for index, (symbol, atom_type) in enumerate(zip(molecule.symbols, atom_types, strict=False)):
        if mask_excluded_vdw and index in explicitly_excluded:
            atom_vdw_map_list.append(-1)
            continue
        vdw_idx, vdw_param = match_vdw(forcefield, atom_type=atom_type, element=symbol)
        atom_vdw_map_list.append(vdw_idx if vdw_idx is not None and vdw_param is not None else -1)

    unmatched = [i for i, idx in enumerate(atom_vdw_map_list) if idx == -1 and i not in explicitly_excluded]
    if getattr(forcefield, "vdws", None) and unmatched:
        raise ValueError(
            f"Unmatched vdW parameters for atoms at indices {unmatched}. "
            "Ensure all atom types/elements have corresponding vdW "
            "parameters in the force field, or remove vdW terms from "
            "the force field if vdW interactions are not intended."
        )

    bond_indices = (
        np.array([(b.atom_i, b.atom_j) for b, _, _ in matched_bonds], dtype=np.int32)
        if matched_bonds
        else np.empty((0, 2), dtype=np.int32)
    )
    angle_indices = (
        np.array([(a.atom_i, a.atom_j, a.atom_k) for a, _, _ in matched_angles], dtype=np.int32)
        if matched_angles
        else np.empty((0, 3), dtype=np.int32)
    )
    torsion_indices = (
        np.array(torsion_atom_indices, dtype=np.int32) if torsion_atom_indices else np.empty((0, 4), dtype=np.int32)
    )

    return MatchedTopology(
        matched_bonds=tuple(matched_bonds),
        matched_angles=tuple(matched_angles),
        bond_indices=bond_indices,
        bond_param_map=np.array([idx for _, idx, _ in matched_bonds], dtype=np.int32),
        angle_indices=angle_indices,
        angle_param_map=np.array([idx for _, idx, _ in matched_angles], dtype=np.int32),
        torsion_indices=torsion_indices,
        torsion_param_map=np.array(torsion_param_map_list, dtype=np.int32),
        atom_vdw_map=np.array(atom_vdw_map_list, dtype=np.int32),
        explicitly_excluded_atoms=explicitly_excluded,
    )


class _PreparedJaxBase(AbstractPreparedBackend):
    """Shared prepared-session base for JAX and JAX-MD backends."""

    _backend_label: str
    _backend: Any
    _state: Any

    def __init__(
        self,
        *,
        backend: Any,
        info: BackendInfo,
        case_id: str,
        molecule: Molecule,
        force_field: ForceField,
        layout: ParameterLayout,
        state: Any,
    ) -> None:
        super().__init__(
            info=info,
            case_id=case_id,
            molecule=molecule,
            force_field=force_field,
            layout=layout,
        )
        self._backend = backend
        self._state = state

    def _energy(self, request: EnergyRequest | ReferenceEnergyRequest) -> EnergyResult:
        assert isinstance(request, EnergyRequest)
        ff = self._ff_for(request.parameters)
        try:
            value = self._backend._evaluate_energy(self._state, ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"{self._backend_label} energy evaluation failed: {exc}") from exc
        return EnergyResult(energy=float(value), unit=EnergyUnit.KCAL_PER_MOL, provenance=self._provenance)

    def _raw_hessian(self, ff: ForceField) -> np.ndarray:
        hess: np.ndarray = self._backend._evaluate_hessian(self._state, ff)
        return hess

    def _hessian(self, request: HessianRequest | ReferenceHessianRequest) -> HessianResult:
        assert isinstance(request, HessianRequest)
        ff = self._ff_for(request.parameters)
        try:
            hess = self._raw_hessian(ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"{self._backend_label} Hessian evaluation failed: {exc}") from exc
        return HessianResult(
            hessian=readonly_array(hess), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=self._provenance
        )

    def _frequencies(self, request: FrequencyRequest | ReferenceFrequencyRequest) -> FrequencyResult:
        assert isinstance(request, FrequencyRequest)
        return self._mm_frequencies(request, backend_label=self._backend_label)

    def _parameter_gradient(self, request: ParameterGradientRequest) -> ParameterGradientResult:
        ff = self._ff_for(request.parameters)
        try:
            energy, grad = self._backend._evaluate_param_grad(self._state, ff)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"{self._backend_label} parameter-gradient evaluation failed: {exc}") from exc
        return ParameterGradientResult(
            energy=float(energy),
            gradient=readonly_array(grad),
            unit=EnergyUnit.KCAL_PER_MOL,
            provenance=self._provenance,
        )

    def _batched_energy(self, request: BatchedEnergyRequest) -> BatchedEnergyResult:
        mat = self._validate_matrix(request.parameter_matrix)
        try:
            energies = self._backend._evaluate_batched_energy(self._state, self.force_field, mat)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"{self._backend_label} batched-energy evaluation failed: {exc}") from exc
        return BatchedEnergyResult(
            energies=readonly_array(energies), unit=EnergyUnit.KCAL_PER_MOL, provenance=self._provenance
        )
