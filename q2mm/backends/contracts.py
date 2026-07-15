"""Backend capability contracts and prepared-session vocabulary.

This module is the single source of truth for how Q2MM talks to a computational
backend (MM or QM).  It defines:

* :class:`BackendRole` / :class:`Capability` / :class:`BackendInfo` /
  :class:`BackendProvenance` — the vocabulary a backend uses to *declare* what
  it can do.  Capabilities and functional forms both default to **empty**;
  every backend must explicitly enumerate every operation and functional form
  it supports.
* Immutable, typed *preparation* and *evaluation* requests, and typed,
  canonical-unit *results*.  Requests defensively copy their arrays to
  read-only in ``__post_init__``; results defensively copy and shape-validate
  their arrays in ``__post_init__``, so direct construction is safe regardless
  of the producing backend.  Every result carries an explicit unit enum and a
  :class:`BackendProvenance`.
* Typed errors: :class:`BackendUnavailableError`,
  :class:`BackendConfigurationError`, :class:`PreparationError`,
  :class:`UnsupportedCapabilityError`, :class:`EvaluationError` — there is no
  broad silent fallback.
* The :class:`Backend` / :class:`PreparedBackend` lifecycle protocols and the
  :class:`AbstractPreparedBackend` base that enforces capability checks,
  request-family/role validation, and full-vector dimension validation.  A
  concrete backend exposes only ``info`` and ``prepare`` (plus clearly
  backend-specific serialization/config); the prepared session is the only
  evaluation surface.
* Side-effect-free registry :class:`BackendDescriptor` (which carries a static
  :class:`BackendInfo` and a descriptor-API version) / :class:`DependencyProbe`
  plumbing used by :mod:`q2mm.backends.registry`.

**Canonical unit contracts** (results always carry these units):

* MM energy: **kcal/mol** (:attr:`EnergyUnit.KCAL_PER_MOL`); QM energy:
  **Hartree** (:attr:`EnergyUnit.HARTREE`) — must match :class:`BackendRole`.
* Geometry: **Å** (:attr:`LengthUnit.ANGSTROM`).
* Hessian: **Hartree/Bohr²** (:attr:`HessianUnit.HARTREE_PER_BOHR2`).
* Frequency: **cm⁻¹** (:attr:`FrequencyUnit.INVERSE_CM`).
* Parameter gradients have length exactly ``len(ParameterLayout)``.

.. warning::

   This is an internal, unstable API.  It is *not* covered by semantic
   versioning and may change without notice between Q2MM releases.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import shutil
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Molecule
    from q2mm.models.parameters import ParameterLayout


#: Descriptor-API version.  Bumped when the descriptor/contract shape changes
#: in a way out-of-tree plugins must adapt to.
DESCRIPTOR_API_VERSION = 1


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class BackendRole(str, enum.Enum):
    """Whether a backend computes molecular-mechanics or quantum energies."""

    MM = "mm"
    QM = "qm"


class Capability(str, enum.Enum):
    """A discrete operation a backend may declare that it supports.

    A backend that lists a capability in its :class:`BackendInfo` **must**
    implement the corresponding prepared-session method; a backend that does
    not list it must raise :class:`UnsupportedCapabilityError` when the method
    is called.
    """

    ENERGY = "energy"
    MINIMIZE = "minimize"
    HESSIAN = "hessian"
    FREQUENCIES = "frequencies"
    PARAMETER_GRADIENT = "parameter_gradient"
    HESSIAN_PARAMETER_JACOBIAN = "hessian_parameter_jacobian"
    BATCHED_ENERGY = "batched_energy"
    BATCHED_HESSIAN = "batched_hessian"
    GEOMETRY_OPTIMIZATION = "geometry_optimization"
    REUSABLE_STATE = "reusable_state"


class EnergyUnit(str, enum.Enum):
    """Explicit canonical energy unit for a result."""

    KCAL_PER_MOL = "kcal/mol"
    HARTREE = "hartree"


class LengthUnit(str, enum.Enum):
    """Explicit canonical length unit for coordinates."""

    ANGSTROM = "angstrom"


class HessianUnit(str, enum.Enum):
    """Explicit canonical Hessian unit (atomic units)."""

    HARTREE_PER_BOHR2 = "hartree/bohr^2"


class FrequencyUnit(str, enum.Enum):
    """Explicit canonical vibrational-frequency unit."""

    INVERSE_CM = "cm^-1"


#: Canonical energy unit implied by a backend role.
_ROLE_ENERGY_UNIT = {
    BackendRole.MM: EnergyUnit.KCAL_PER_MOL,
    BackendRole.QM: EnergyUnit.HARTREE,
}


@dataclass(frozen=True)
class BackendProvenance:
    """Immutable record of which backend produced a result and how.

    Args:
        backend: Registry key of the backend (e.g. ``"openmm"``, ``"jax"``).
        role: Whether the producing backend is MM or QM.
        version: Backend library version string if known (else ``""``).
        detail: Free-form detail — QM method/basis, OpenMM platform/device, etc.

    """

    backend: str
    role: BackendRole
    version: str = ""
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or not self.backend:
            raise ValueError("BackendProvenance.backend must be a non-empty string.")
        if not isinstance(self.role, BackendRole):
            raise ValueError(f"BackendProvenance.role must be a BackendRole; got {self.role!r}.")
        if not isinstance(self.version, str):
            raise ValueError(f"BackendProvenance.version must be a string; got {self.version!r}.")
        if not isinstance(self.detail, str):
            raise ValueError(f"BackendProvenance.detail must be a string; got {self.detail!r}.")


@dataclass(frozen=True)
class BackendInfo:
    """Immutable capability declaration for a backend.

    Both :attr:`capabilities` and :attr:`functional_forms` default to the
    empty set: a backend must *explicitly* declare every operation and every
    functional form it supports.  Nothing is inferred.

    Args:
        name: Human-readable backend name (e.g. ``"OpenMM"``).
        role: MM or QM.
        capabilities: Operations the backend supports.
        functional_forms: :class:`~q2mm.models.forcefield.FunctionalForm`
            values (as strings) the backend can evaluate.  Empty for QM
            backends, which do not consume force fields.
        provenance: Canonical provenance stamped onto every result.

    """

    name: str
    role: BackendRole
    capabilities: frozenset[Capability] = field(default_factory=frozenset)
    functional_forms: frozenset[str] = field(default_factory=frozenset)
    provenance: BackendProvenance | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("BackendInfo.name must be a non-empty string.")
        if not isinstance(self.role, BackendRole):
            raise ValueError(f"BackendInfo.role must be a BackendRole; got {self.role!r}.")
        # Normalize capabilities/forms to frozensets of the right element types.
        caps = frozenset(self.capabilities)
        if not all(isinstance(c, Capability) for c in caps):
            raise ValueError("BackendInfo.capabilities must all be Capability members.")
        forms = frozenset(self.functional_forms)
        if not all(isinstance(f, str) for f in forms):
            raise ValueError("BackendInfo.functional_forms must all be strings.")
        object.__setattr__(self, "capabilities", caps)
        object.__setattr__(self, "functional_forms", forms)
        if self.role is BackendRole.QM and forms:
            raise ValueError("BackendInfo: QM backends must declare no functional_forms.")
        if self.provenance is not None:
            if not isinstance(self.provenance, BackendProvenance):
                raise ValueError("BackendInfo.provenance must be a BackendProvenance or None.")
            if self.provenance.role is not self.role:
                raise ValueError(
                    f"BackendInfo.provenance.role ({self.provenance.role.value}) must agree with "
                    f"info role ({self.role.value})."
                )

    def supports(self, capability: Capability) -> bool:
        """Return ``True`` if *capability* is declared by this backend."""
        return capability in self.capabilities

    def supports_form(self, form: str) -> bool:
        """Return ``True`` if functional-form string *form* is supported."""
        return form in self.functional_forms

    def matches(self, other: BackendInfo) -> bool:
        """Return ``True`` if role, capabilities, and functional forms agree.

        Compares :attr:`role`, :attr:`capabilities`, and
        :attr:`functional_forms` only; the human-readable :attr:`name` and the
        :attr:`provenance` are intentionally ignored.  Descriptor loading checks
        the runtime provenance's registry key/role separately (see
        :meth:`BackendDescriptor.load`).
        """
        return (
            self.role is other.role
            and self.capabilities == other.capabilities
            and self.functional_forms == other.functional_forms
        )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BackendError(RuntimeError):
    """Base class for all typed backend errors."""


class BackendUnavailableError(BackendError):
    """A backend's native dependencies are not installed/importable."""


class BackendConfigurationError(BackendError):
    """A backend is installed but mis-configured (bad option, missing path)."""


class PreparationError(BackendError):
    """Building a prepared session for a training case failed."""


class UnsupportedCapabilityError(BackendError):
    """A prepared session was asked for an operation it does not declare."""

    def __init__(self, backend: str, capability: Capability) -> None:
        self.backend = backend
        self.capability = capability
        super().__init__(f"Backend {backend!r} does not support capability {capability.value!r}.")


class EvaluationError(BackendError):
    """A prepared-session evaluation failed at runtime."""


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------


def readonly_array(values: object, *, dtype: npt.DTypeLike = float) -> np.ndarray:
    """Return a contiguous, read-only copy of *values*.

    Args:
        values: Anything array-like.
        dtype: Target dtype (default ``float``).

    Returns:
        np.ndarray: A read-only defensive copy.

    """
    arr: np.ndarray = np.array(values, dtype=dtype, copy=True)
    arr.setflags(write=False)
    return arr


def _readonly_vector(values: object, *, name: str) -> np.ndarray:
    arr = readonly_array(values)
    if arr.ndim != 1:
        raise EvaluationError(f"{name} must be 1-D; got shape {arr.shape}.")
    return arr


def _readonly_matrix(values: object, *, name: str, ncols: int | None = None) -> np.ndarray:
    arr = readonly_array(values)
    if arr.ndim != 2:
        raise EvaluationError(f"{name} must be 2-D; got shape {arr.shape}.")
    if ncols is not None and arr.shape[1] != ncols:
        raise EvaluationError(f"{name} must have {ncols} columns; got shape {arr.shape}.")
    return arr


def _readonly_square(values: object, *, name: str) -> np.ndarray:
    arr = readonly_array(values)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise EvaluationError(f"{name} must be a square 2-D matrix; got shape {arr.shape}.")
    return arr


def _require_finite(arr: np.ndarray, *, name: str) -> np.ndarray:
    """Raise :class:`EvaluationError` if *arr* has any non-finite entry."""
    if not np.all(np.isfinite(arr)):
        raise EvaluationError(f"{name} contains non-finite values.")
    return arr


def _require_finite_scalar(value: float, *, name: str) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise EvaluationError(f"{name} must be finite; got {v!r}.")
    return v


def _validate_result_provenance(provenance: object, *, cls_name: str) -> BackendProvenance:
    """Validate a result's provenance is a non-empty, role-typed record."""
    if not isinstance(provenance, BackendProvenance):
        raise EvaluationError(f"{cls_name}.provenance must be a BackendProvenance.")
    if not provenance.backend:
        raise EvaluationError(f"{cls_name}.provenance.backend must be non-empty.")
    if not isinstance(provenance.role, BackendRole):
        raise EvaluationError(f"{cls_name}.provenance.role must be a BackendRole.")
    return provenance


def _check_unit_type(unit: object, expected_enum: type, *, cls_name: str) -> None:
    if not isinstance(unit, expected_enum):
        raise EvaluationError(f"{cls_name}.unit must be a {expected_enum.__name__}; got {unit!r}.")


def _check_energy_unit_matches_role(unit: EnergyUnit, provenance: BackendProvenance, *, cls_name: str) -> None:
    expected = _ROLE_ENERGY_UNIT[provenance.role]
    if unit is not expected:
        raise EvaluationError(
            f"{cls_name}.unit {unit.value!r} does not match provenance role "
            f"{provenance.role.value} (expected {expected.value!r})."
        )


# ---------------------------------------------------------------------------
# Preparation request
# ---------------------------------------------------------------------------


def _deep_freeze(value: object) -> object:
    """Return a recursively immutable copy of *value*.

    Mappings become read-only :class:`~types.MappingProxyType` with **keys
    preserved verbatim** and values recursively frozen; lists/tuples become
    tuples of frozen values; sets become :class:`frozenset`; ndarrays become
    read-only copies.  Source mutation at any depth cannot leak into the
    returned structure.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return readonly_array(value, dtype=value.dtype)
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(v) for v in value)
    return value


def _freeze_options(options: Mapping[str, object]) -> Mapping[str, object]:
    """Return a deeply-immutable mapping proxy of *options*.

    Option **keys** must be strings and are preserved verbatim (not
    transformed); option **values** are recursively frozen and copied (nested
    mappings, sequences, sets, and arrays), so mutating the caller's original
    structure at any depth after construction cannot affect the stored options.

    Raises:
        PreparationError: If *options* is not a mapping or any key is not a
            string.

    """
    if not isinstance(options, Mapping):
        raise PreparationError("PreparationRequest.options must be a mapping.")
    frozen: dict[str, object] = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise PreparationError(f"PreparationRequest.options keys must be strings; got {key!r}.")
        frozen[key] = _deep_freeze(value)
    return MappingProxyType(frozen)


@dataclass(frozen=True, eq=False)
class PreparationRequest:
    """Immutable request to build a prepared session for one training case.

    Args:
        case_id: Stable, non-empty identifier for the training case.  Exactly
            one prepared session is built per ``case_id``.
        molecule: The molecule (with reference geometry) to prepare.
        force_field: Base force field (MM backends only; ``None`` for QM).
            The prepared session derives its :class:`ParameterLayout` from
            this force field and owns both.
        options: Backend-specific preparation options (string keys).  Copied to
            an immutable mapping proxy (keys preserved, values deep-frozen) so
            caller mutation after construction has no effect.

    """

    case_id: str
    molecule: Molecule
    force_field: ForceField | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id:
            raise PreparationError("PreparationRequest.case_id must be a non-empty string.")
        object.__setattr__(self, "options", _freeze_options(self.options))


# ---------------------------------------------------------------------------
# MM evaluation requests (carry validated full parameter vectors/matrices)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class EnergyRequest:
    """Single-point energy for a full parameter vector."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


@dataclass(frozen=True, eq=False)
class MinimizationRequest:
    """Energy-minimize (relax) the geometry for a full parameter vector.

    Args:
        parameters: Full parameter vector.
        max_iterations: Maximum minimizer iterations, or ``None`` to use the
            backend's native default (preserves per-backend defaults).
        tolerance: Convergence tolerance in the backend's native units, or
            ``None`` to use the backend's native default.

    """

    parameters: np.ndarray
    max_iterations: int | None = None
    tolerance: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise EvaluationError("MinimizationRequest.max_iterations must be positive.")


@dataclass(frozen=True, eq=False)
class HessianRequest:
    """Cartesian Hessian for a full parameter vector."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


@dataclass(frozen=True, eq=False)
class FrequencyRequest:
    """Vibrational frequencies for a full parameter vector.

    Args:
        parameters: Full parameter vector.
        on_error: Forwarded to
            :func:`~q2mm.models.hessian.hessian_to_frequencies`.

    """

    parameters: np.ndarray
    on_error: str = "raise"

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


@dataclass(frozen=True, eq=False)
class ParameterGradientRequest:
    """Energy plus analytical ``dE/dp`` for a full parameter vector."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


@dataclass(frozen=True, eq=False)
class HessianJacobianRequest:
    """Hessian plus its analytical ``dH/dp`` Jacobian for a full vector."""

    parameters: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


@dataclass(frozen=True, eq=False)
class BatchedEnergyRequest:
    """Energies for a batch of full parameter vectors.

    Args:
        parameter_matrix: Shape ``(batch, len(layout))`` parameter vectors.

    """

    parameter_matrix: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameter_matrix", _readonly_matrix(self.parameter_matrix, name="parameter_matrix"))


@dataclass(frozen=True, eq=False)
class BatchedHessianRequest:
    """Cartesian Hessians for a batch of topology-compatible prepared cases.

    Carries one full parameter vector applied to every case in the batch; the
    batch object owns the compatible cases and their coordinates.  No force
    field crosses this boundary.

    Args:
        parameters: Full parameter vector (length ``len(layout)``).

    """

    parameters: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly_vector(self.parameters, name="parameters"))


#: MM evaluation request types (used for request-family validation).
_MM_REQUEST_TYPES = (
    EnergyRequest,
    MinimizationRequest,
    HessianRequest,
    FrequencyRequest,
    ParameterGradientRequest,
    HessianJacobianRequest,
    BatchedEnergyRequest,
    BatchedHessianRequest,
)


# ---------------------------------------------------------------------------
# QM evaluation requests (no parameter vectors — method/basis fixed at prepare)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QMEnergyRequest:
    """Single-point QM energy request."""


@dataclass(frozen=True)
class QMHessianRequest:
    """QM Hessian request."""


@dataclass(frozen=True)
class QMFrequencyRequest:
    """QM vibrational-frequency request."""


@dataclass(frozen=True)
class QMGeometryOptimizationRequest:
    """QM geometry optimization request.

    Args:
        opt_type: ``"min"`` for a minimum, ``"ts"`` for a transition state.

    """

    opt_type: str = "min"


#: QM evaluation request types (used for request-family validation).
_QM_REQUEST_TYPES = (
    QMEnergyRequest,
    QMHessianRequest,
    QMFrequencyRequest,
    QMGeometryOptimizationRequest,
)


# ---------------------------------------------------------------------------
# Results (canonical units, read-only arrays, provenance-stamped)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnergyResult:
    """Single-point energy in a canonical unit.

    Args:
        energy: Energy value.
        unit: Explicit canonical unit (kcal/mol for MM, Hartree for QM).
        provenance: Producing backend.

    """

    energy: float
    unit: EnergyUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        prov = _validate_result_provenance(self.provenance, cls_name="EnergyResult")
        _check_unit_type(self.unit, EnergyUnit, cls_name="EnergyResult")
        _check_energy_unit_matches_role(self.unit, prov, cls_name="EnergyResult")
        object.__setattr__(self, "energy", _require_finite_scalar(self.energy, name="EnergyResult.energy"))


@dataclass(frozen=True, eq=False)
class GeometryResult:
    """Optimized geometry and its energy.

    Args:
        energy: Energy at the optimized geometry.
        energy_unit: Canonical energy unit.
        symbols: Element symbols, length ``N``.
        coordinates: ``(N, 3)`` coordinates (read-only, defensive copy).
        coordinate_unit: Canonical length unit (Å).
        provenance: Producing backend.

    """

    energy: float
    energy_unit: EnergyUnit
    symbols: tuple[str, ...]
    coordinates: np.ndarray
    coordinate_unit: LengthUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        prov = _validate_result_provenance(self.provenance, cls_name="GeometryResult")
        _check_unit_type(self.energy_unit, EnergyUnit, cls_name="GeometryResult")
        _check_energy_unit_matches_role(self.energy_unit, prov, cls_name="GeometryResult")
        _check_unit_type(self.coordinate_unit, LengthUnit, cls_name="GeometryResult")
        object.__setattr__(self, "energy", _require_finite_scalar(self.energy, name="GeometryResult.energy"))
        object.__setattr__(self, "symbols", tuple(self.symbols))
        coords = _readonly_matrix(self.coordinates, name="coordinates", ncols=3)
        _require_finite(coords, name="GeometryResult.coordinates")
        if coords.shape[0] != len(self.symbols):
            raise EvaluationError(f"coordinates rows ({coords.shape[0]}) must match symbols ({len(self.symbols)}).")
        object.__setattr__(self, "coordinates", coords)


@dataclass(frozen=True, eq=False)
class HessianResult:
    """Cartesian Hessian in atomic units.

    Args:
        hessian: ``(3N, 3N)`` Hessian (read-only, defensive copy).
        unit: Canonical Hessian unit.
        provenance: Producing backend.

    """

    hessian: np.ndarray
    unit: HessianUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        _validate_result_provenance(self.provenance, cls_name="HessianResult")
        _check_unit_type(self.unit, HessianUnit, cls_name="HessianResult")
        hess = _readonly_square(self.hessian, name="hessian")
        _require_finite(hess, name="HessianResult.hessian")
        object.__setattr__(self, "hessian", hess)


@dataclass(frozen=True, eq=False)
class FrequencyResult:
    """Vibrational frequencies.

    Args:
        frequencies: Array of frequencies (read-only, defensive copy).  Values
            are finite; a fully-penalized region uses the finite
            :data:`~q2mm.models.hessian.PENALTY_FREQUENCY` sentinel (still
            finite), never NaN/Inf.
        unit: Canonical frequency unit.
        provenance: Producing backend.

    """

    frequencies: np.ndarray
    unit: FrequencyUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        _validate_result_provenance(self.provenance, cls_name="FrequencyResult")
        _check_unit_type(self.unit, FrequencyUnit, cls_name="FrequencyResult")
        freqs = _readonly_vector(self.frequencies, name="frequencies")
        _require_finite(freqs, name="FrequencyResult.frequencies")
        object.__setattr__(self, "frequencies", freqs)


@dataclass(frozen=True, eq=False)
class ParameterGradientResult:
    """Energy plus analytical parameter gradient.

    Args:
        energy: Energy value.
        gradient: ``dE/dp`` of length exactly ``len(layout)`` (read-only copy).
        unit: Canonical energy unit borne by ``energy`` and ``gradient``.
        provenance: Producing backend.

    """

    energy: float
    gradient: np.ndarray
    unit: EnergyUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        prov = _validate_result_provenance(self.provenance, cls_name="ParameterGradientResult")
        _check_unit_type(self.unit, EnergyUnit, cls_name="ParameterGradientResult")
        _check_energy_unit_matches_role(self.unit, prov, cls_name="ParameterGradientResult")
        object.__setattr__(self, "energy", _require_finite_scalar(self.energy, name="ParameterGradientResult.energy"))
        grad = _readonly_vector(self.gradient, name="gradient")
        _require_finite(grad, name="ParameterGradientResult.gradient")
        object.__setattr__(self, "gradient", grad)


@dataclass(frozen=True, eq=False)
class HessianJacobianResult:
    """Hessian plus its analytical parameter Jacobian.

    Args:
        hessian: ``(3N, 3N)`` Hessian (read-only copy).
        jacobian: ``(3N, 3N, len(layout))`` Jacobian (read-only copy).
        unit: Canonical Hessian unit borne by both.
        provenance: Producing backend.

    """

    hessian: np.ndarray
    jacobian: np.ndarray
    unit: HessianUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        _validate_result_provenance(self.provenance, cls_name="HessianJacobianResult")
        _check_unit_type(self.unit, HessianUnit, cls_name="HessianJacobianResult")
        hess = _readonly_square(self.hessian, name="hessian")
        jac = readonly_array(self.jacobian)
        if jac.ndim != 3 or jac.shape[0] != hess.shape[0] or jac.shape[1] != hess.shape[1]:
            raise EvaluationError(
                f"jacobian must have shape (3N, 3N, n_params) matching hessian {hess.shape}; got {jac.shape}."
            )
        _require_finite(hess, name="HessianJacobianResult.hessian")
        _require_finite(jac, name="HessianJacobianResult.jacobian")
        object.__setattr__(self, "hessian", hess)
        object.__setattr__(self, "jacobian", jac)


@dataclass(frozen=True, eq=False)
class BatchedEnergyResult:
    """Energies for a batch of parameter vectors.

    Args:
        energies: ``(batch,)`` energies (read-only copy).
        unit: Canonical energy unit.
        provenance: Producing backend.

    """

    energies: np.ndarray
    unit: EnergyUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        prov = _validate_result_provenance(self.provenance, cls_name="BatchedEnergyResult")
        _check_unit_type(self.unit, EnergyUnit, cls_name="BatchedEnergyResult")
        _check_energy_unit_matches_role(self.unit, prov, cls_name="BatchedEnergyResult")
        energies = _readonly_vector(self.energies, name="energies")
        _require_finite(energies, name="BatchedEnergyResult.energies")
        object.__setattr__(self, "energies", energies)


@dataclass(frozen=True, eq=False)
class BatchedHessianResult:
    """Cartesian Hessians for a batch of topology-compatible cases.

    Produced by a typed batch object (e.g. ``PreparedJaxBatch``) evaluated for
    one full parameter vector.  Each row corresponds to one prepared case, in
    the same order as :attr:`case_ids`.

    Args:
        case_ids: Stable case IDs, one per batched case (order matches rows).
        hessians: ``(n_cases, 3N, 3N)`` Hessians (read-only copy).
        unit: Canonical Hessian unit.
        provenance: Producing backend.

    """

    case_ids: tuple[str, ...]
    hessians: np.ndarray
    unit: HessianUnit
    provenance: BackendProvenance

    def __post_init__(self) -> None:
        _validate_result_provenance(self.provenance, cls_name="BatchedHessianResult")
        _check_unit_type(self.unit, HessianUnit, cls_name="BatchedHessianResult")
        case_ids = tuple(self.case_ids)
        if not all(isinstance(c, str) and c for c in case_ids):
            raise EvaluationError("BatchedHessianResult.case_ids must all be non-empty strings.")
        arr = readonly_array(self.hessians)
        if arr.ndim != 3 or arr.shape[1] != arr.shape[2]:
            raise EvaluationError(f"hessians must have shape (n, 3N, 3N); got {arr.shape}.")
        if arr.shape[0] != len(case_ids):
            raise EvaluationError(f"hessians batch dim ({arr.shape[0]}) must match case_ids ({len(case_ids)}).")
        _require_finite(arr, name="BatchedHessianResult.hessians")
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "hessians", arr)


# ---------------------------------------------------------------------------
# Lifecycle protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class PreparedBackend(Protocol):
    """One prepared session for one stable training case.

    A prepared session owns its molecule, base force field, parameter layout,
    and any reusable native state.  Evaluation requests carry validated full
    parameter vectors/matrices; the session never accepts a ForceField or a
    native handle across the boundary.  Any operation not declared in
    :attr:`info` raises :class:`UnsupportedCapabilityError`.
    """

    @property
    def info(self) -> BackendInfo:
        """Backend capability declaration."""
        ...

    @property
    def case_id(self) -> str:
        """Stable training-case identifier."""
        ...

    @property
    def molecule(self) -> Molecule:
        """The molecule owned by this session."""
        ...

    def energy(self, request: EnergyRequest | QMEnergyRequest) -> EnergyResult:
        """Single-point energy."""
        ...

    def minimize(self, request: MinimizationRequest) -> GeometryResult:
        """Energy-minimize (MM)."""
        ...

    def optimize_geometry(self, request: QMGeometryOptimizationRequest) -> GeometryResult:
        """Geometry-optimize (QM)."""
        ...

    def hessian(self, request: HessianRequest | QMHessianRequest) -> HessianResult:
        """Cartesian Hessian."""
        ...

    def frequencies(self, request: FrequencyRequest | QMFrequencyRequest) -> FrequencyResult:
        """Vibrational frequencies."""
        ...

    def parameter_gradient(self, request: ParameterGradientRequest) -> ParameterGradientResult:
        """Energy plus parameter gradient (MM)."""
        ...

    def hessian_parameter_jacobian(self, request: HessianJacobianRequest) -> HessianJacobianResult:
        """Hessian plus its parameter Jacobian (MM)."""
        ...

    def batched_energy(self, request: BatchedEnergyRequest) -> BatchedEnergyResult:
        """Energies for a batch of vectors (MM)."""
        ...


@runtime_checkable
class Backend(Protocol):
    """A backend factory that prepares per-case sessions.

    A concrete backend exposes only :attr:`info` and :meth:`prepare` as its
    generic surface (plus clearly backend-specific serialization/config where
    unavoidable).  All evaluation happens through the returned
    :class:`PreparedBackend`.
    """

    @property
    def info(self) -> BackendInfo:
        """Backend capability declaration."""
        ...

    def prepare(self, request: PreparationRequest) -> PreparedBackend:
        """Build a prepared session for one training case."""
        ...


# ---------------------------------------------------------------------------
# Batched-Hessian protocols and capability-first helper
# ---------------------------------------------------------------------------


@runtime_checkable
class PreparedHessianBatch(Protocol):
    """A typed batch of topology-compatible prepared cases (Hessian batching).

    The batch shares one compiled/native evaluation kernel internally while
    each member case keeps its own coordinates/native state.  Its only
    evaluation surface is :meth:`hessians`, which takes a typed
    :class:`BatchedHessianRequest` (one full parameter vector applied to every
    member) and returns a typed :class:`BatchedHessianResult`.
    """

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Stable case IDs of the batched members, in result-row order."""
        ...

    def hessians(self, request: BatchedHessianRequest) -> BatchedHessianResult:
        """Per-case Cartesian Hessians for one full parameter vector."""
        ...


@runtime_checkable
class HessianBatchPreparer(Protocol):
    """Optional backend surface that groups prepared sessions into batches.

    A backend declaring :attr:`Capability.BATCHED_HESSIAN` **must** implement
    this protocol.  It groups topology-compatible prepared sessions into typed
    :class:`PreparedHessianBatch` objects; the concrete grouping/compilation is
    backend-specific and never crosses this boundary.
    """

    def prepare_hessian_batches(self, sessions: Sequence[PreparedBackend]) -> list[PreparedHessianBatch]:
        """Group *sessions* into topology-compatible Hessian batches."""
        ...


def prepare_hessian_batches(
    backend: Backend,
    sessions: Sequence[PreparedBackend],
) -> list[PreparedHessianBatch]:
    """Capability-first, backend-neutral entry to batched-Hessian preparation.

    This is the only surface callers (e.g. the objective function) should use
    to batch Hessians.  It is fully backend-agnostic: it checks the declared
    capability and the batch-preparer protocol, delegates grouping to the
    backend, and validates the returned batch objects.

    Args:
        backend: The backend to batch with.
        sessions: Prepared sessions to group (must be topology-compatible
            subsets as the backend defines).

    Returns:
        list[PreparedHessianBatch]: Validated typed batch objects.

    Raises:
        UnsupportedCapabilityError: If the backend does not declare
            :attr:`Capability.BATCHED_HESSIAN`.
        BackendConfigurationError: If the backend declares the capability but
            does not implement :class:`HessianBatchPreparer`, or returns
            objects that are not valid :class:`PreparedHessianBatch` instances.

    """
    if not backend.info.supports(Capability.BATCHED_HESSIAN):
        raise UnsupportedCapabilityError(backend.info.name, Capability.BATCHED_HESSIAN)
    if not isinstance(backend, HessianBatchPreparer):
        raise BackendConfigurationError(
            f"Backend {backend.info.name!r} declares BATCHED_HESSIAN but does not implement "
            "the HessianBatchPreparer protocol (prepare_hessian_batches)."
        )
    batches = backend.prepare_hessian_batches(sessions)
    if not isinstance(batches, list):
        raise BackendConfigurationError(
            f"Backend {backend.info.name!r} prepare_hessian_batches must return a list; got {type(batches).__name__}."
        )
    for batch in batches:
        if not isinstance(batch, PreparedHessianBatch):
            raise BackendConfigurationError(
                f"Backend {backend.info.name!r} prepare_hessian_batches returned "
                f"{type(batch).__name__}, which is not a valid PreparedHessianBatch."
            )
    return batches


# ---------------------------------------------------------------------------
# Abstract prepared-session base
# ---------------------------------------------------------------------------


class AbstractPreparedBackend(ABC):
    """Base that enforces capability, request-family, and vector validation.

    Concrete prepared sessions override the ``_energy`` / ``_minimize`` / …
    hooks for the capabilities they declare.  The public methods verify the
    capability is declared, that the request family matches the backend role,
    that request parameter vectors have exactly ``len(layout)`` finite entries,
    and that returned energy units match the role.
    """

    def __init__(
        self,
        *,
        info: BackendInfo,
        case_id: str,
        molecule: Molecule,
        force_field: ForceField | None,
        layout: ParameterLayout | None,
    ) -> None:
        if not isinstance(case_id, str) or not case_id:
            raise PreparationError("Prepared session case_id must be a non-empty string.")
        self._info = info
        self._case_id = case_id
        self._molecule = molecule
        self._force_field = force_field
        self._layout = layout

    # -- read-only accessors ------------------------------------------------

    @property
    def info(self) -> BackendInfo:
        """Immutable capability declaration for the owning backend."""
        return self._info

    @property
    def case_id(self) -> str:
        """Stable training-case identifier this session was prepared for."""
        return self._case_id

    @property
    def molecule(self) -> Molecule:
        """The molecule owned by this prepared session."""
        return self._molecule

    @property
    def force_field(self) -> ForceField:
        """The base force field owned by this prepared session."""
        if self._force_field is None:
            raise UnsupportedCapabilityError(self._info.name, Capability.ENERGY)
        return self._force_field

    @property
    def layout(self) -> ParameterLayout:
        """The parameter layout derived from the base force field."""
        if self._layout is None:
            raise UnsupportedCapabilityError(self._info.name, Capability.ENERGY)
        return self._layout

    # -- validation helpers -------------------------------------------------

    def _require(self, capability: Capability) -> None:
        if capability not in self._info.capabilities:
            raise UnsupportedCapabilityError(self._info.name, capability)

    def _require_exact_request(
        self,
        request: object,
        *,
        mm_type: type | None,
        qm_type: type | None,
        operation: str,
    ) -> None:
        """Validate *request* is exactly the type this role expects for *operation*.

        Wrong role (operation not defined for the role) or wrong request type
        raises :class:`EvaluationError` **before** dispatch.
        """
        expected = mm_type if self._info.role is BackendRole.MM else qm_type
        if expected is None:
            raise EvaluationError(
                f"{self._info.name} (role={self._info.role.value}) does not support the "
                f"{operation!r} operation for its role."
            )
        if type(request) is not expected:
            raise EvaluationError(
                f"{self._info.name} (role={self._info.role.value}) {operation!r} expected a "
                f"{expected.__name__}; got {type(request).__name__}."
            )

    def _n_atoms(self) -> int:
        return len(self._molecule.symbols)

    def _check_result_provenance(self, provenance: object, *, op: str) -> None:
        if not isinstance(provenance, BackendProvenance):
            raise EvaluationError(f"{self._info.name}: {op} result provenance must be a BackendProvenance.")
        if not provenance.backend:
            raise EvaluationError(f"{self._info.name}: {op} result provenance.backend must be non-empty.")
        if provenance.role is not self._info.role:
            raise EvaluationError(
                f"{self._info.name}: {op} result provenance role {provenance.role.value} does not match "
                f"session role {self._info.role.value}."
            )
        info_prov = self._info.provenance
        if info_prov is not None and provenance.backend != info_prov.backend:
            raise EvaluationError(
                f"{self._info.name}: {op} result provenance backend {provenance.backend!r} does not match "
                f"session backend {info_prov.backend!r}."
            )

    def _validate_vector(self, parameters: np.ndarray) -> np.ndarray:
        """Return a defensive copy of *parameters*, checked against layout."""
        vec = np.array(parameters, dtype=float, copy=True)
        if vec.ndim != 1:
            raise EvaluationError(f"Parameter vector must be 1-D; got shape {vec.shape}.")
        expected = len(self.layout)
        if vec.shape[0] != expected:
            raise EvaluationError(
                f"{self._info.name}: parameter vector has {vec.shape[0]} entries, expected {expected} (len(layout))."
            )
        if not np.all(np.isfinite(vec)):
            raise EvaluationError(f"{self._info.name}: parameter vector contains non-finite values.")
        return vec

    def _validate_matrix(self, parameter_matrix: np.ndarray) -> np.ndarray:
        mat = np.array(parameter_matrix, dtype=float, copy=True)
        if mat.ndim != 2:
            raise EvaluationError(f"Parameter matrix must be 2-D; got shape {mat.shape}.")
        expected = len(self.layout)
        if mat.shape[1] != expected:
            raise EvaluationError(
                f"{self._info.name}: parameter matrix has {mat.shape[1]} columns, expected {expected} (len(layout))."
            )
        if not np.all(np.isfinite(mat)):
            raise EvaluationError(f"{self._info.name}: parameter matrix contains non-finite values.")
        return mat

    def _expected_energy_unit(self) -> EnergyUnit:
        return _ROLE_ENERGY_UNIT[self._info.role]

    def _check_energy_unit(self, unit: EnergyUnit) -> None:
        expected = self._expected_energy_unit()
        if unit is not expected:
            raise EvaluationError(
                f"{self._info.name}: energy unit {unit.value!r} does not match role "
                f"{self._info.role.value} (expected {expected.value!r})."
            )

    # -- central result validation ------------------------------------------

    def _validate_energy_result(self, result: EnergyResult) -> EnergyResult:
        if not isinstance(result, EnergyResult):
            raise EvaluationError(f"{self._info.name}: energy hook must return an EnergyResult.")
        self._check_result_provenance(result.provenance, op="energy")
        self._check_energy_unit(result.unit)
        return result

    def _validate_geometry_result(self, result: GeometryResult, *, op: str) -> GeometryResult:
        if not isinstance(result, GeometryResult):
            raise EvaluationError(f"{self._info.name}: {op} hook must return a GeometryResult.")
        self._check_result_provenance(result.provenance, op=op)
        self._check_energy_unit(result.energy_unit)
        if result.coordinate_unit is not LengthUnit.ANGSTROM:
            raise EvaluationError(f"{self._info.name}: {op} coordinate unit must be ANGSTROM.")
        n = self._n_atoms()
        if len(result.symbols) != n:
            raise EvaluationError(f"{self._info.name}: {op} returned {len(result.symbols)} symbols, expected {n}.")
        if tuple(result.symbols) != tuple(self._molecule.symbols):
            raise EvaluationError(f"{self._info.name}: {op} symbols do not match the prepared molecule.")
        if result.coordinates.shape != (n, 3):
            raise EvaluationError(f"{self._info.name}: {op} coordinates shape {result.coordinates.shape} != ({n}, 3).")
        return result

    def _validate_hessian_result(self, result: HessianResult) -> HessianResult:
        if not isinstance(result, HessianResult):
            raise EvaluationError(f"{self._info.name}: hessian hook must return a HessianResult.")
        self._check_result_provenance(result.provenance, op="hessian")
        if result.unit is not HessianUnit.HARTREE_PER_BOHR2:
            raise EvaluationError(f"{self._info.name}: hessian unit must be HARTREE_PER_BOHR2.")
        n3 = 3 * self._n_atoms()
        if result.hessian.shape != (n3, n3):
            raise EvaluationError(f"{self._info.name}: hessian shape {result.hessian.shape} != ({n3}, {n3}).")
        return result

    def _validate_frequency_result(self, result: FrequencyResult) -> FrequencyResult:
        if not isinstance(result, FrequencyResult):
            raise EvaluationError(f"{self._info.name}: frequencies hook must return a FrequencyResult.")
        self._check_result_provenance(result.provenance, op="frequencies")
        if result.unit is not FrequencyUnit.INVERSE_CM:
            raise EvaluationError(f"{self._info.name}: frequency unit must be INVERSE_CM.")
        return result

    def _validate_param_grad_result(self, result: ParameterGradientResult) -> ParameterGradientResult:
        if not isinstance(result, ParameterGradientResult):
            raise EvaluationError(f"{self._info.name}: parameter_gradient hook must return a ParameterGradientResult.")
        self._check_result_provenance(result.provenance, op="parameter_gradient")
        self._check_energy_unit(result.unit)
        expected = len(self.layout)
        if result.gradient.shape != (expected,):
            raise EvaluationError(
                f"{self._info.name}: gradient shape {result.gradient.shape} != ({expected},) (len(layout))."
            )
        return result

    def _validate_hess_jac_result(self, result: HessianJacobianResult) -> HessianJacobianResult:
        if not isinstance(result, HessianJacobianResult):
            raise EvaluationError(
                f"{self._info.name}: hessian_parameter_jacobian hook must return a HessianJacobianResult."
            )
        self._check_result_provenance(result.provenance, op="hessian_parameter_jacobian")
        if result.unit is not HessianUnit.HARTREE_PER_BOHR2:
            raise EvaluationError(f"{self._info.name}: hessian-jacobian unit must be HARTREE_PER_BOHR2.")
        n3 = 3 * self._n_atoms()
        p = len(self.layout)
        if result.hessian.shape != (n3, n3):
            raise EvaluationError(f"{self._info.name}: hessian shape {result.hessian.shape} != ({n3}, {n3}).")
        if result.jacobian.shape != (n3, n3, p):
            raise EvaluationError(f"{self._info.name}: jacobian shape {result.jacobian.shape} != ({n3}, {n3}, {p}).")
        return result

    def _validate_batched_energy_result(self, result: BatchedEnergyResult, *, n_rows: int) -> BatchedEnergyResult:
        if not isinstance(result, BatchedEnergyResult):
            raise EvaluationError(f"{self._info.name}: batched_energy hook must return a BatchedEnergyResult.")
        self._check_result_provenance(result.provenance, op="batched_energy")
        self._check_energy_unit(result.unit)
        if result.energies.shape != (n_rows,):
            raise EvaluationError(f"{self._info.name}: batched energies length {result.energies.shape} != ({n_rows},).")
        return result

    # -- public API (capability- and role-gated) ----------------------------

    def energy(self, request: EnergyRequest | QMEnergyRequest) -> EnergyResult:
        """Single-point energy in the backend's canonical unit."""
        self._require(Capability.ENERGY)
        self._require_exact_request(request, mm_type=EnergyRequest, qm_type=QMEnergyRequest, operation="energy")
        return self._validate_energy_result(self._energy(request))

    def minimize(self, request: MinimizationRequest) -> GeometryResult:
        """Energy-minimize (relax) the geometry (MM)."""
        self._require(Capability.MINIMIZE)
        self._require_exact_request(request, mm_type=MinimizationRequest, qm_type=None, operation="minimize")
        return self._validate_geometry_result(self._minimize(request), op="minimize")

    def optimize_geometry(self, request: QMGeometryOptimizationRequest) -> GeometryResult:
        """Geometry-optimize the structure (QM)."""
        self._require(Capability.GEOMETRY_OPTIMIZATION)
        self._require_exact_request(
            request, mm_type=None, qm_type=QMGeometryOptimizationRequest, operation="optimize_geometry"
        )
        return self._validate_geometry_result(self._optimize_geometry(request), op="optimize_geometry")

    def hessian(self, request: HessianRequest | QMHessianRequest) -> HessianResult:
        """Cartesian Hessian in Hartree/Bohr²."""
        self._require(Capability.HESSIAN)
        self._require_exact_request(request, mm_type=HessianRequest, qm_type=QMHessianRequest, operation="hessian")
        return self._validate_hessian_result(self._hessian(request))

    def frequencies(self, request: FrequencyRequest | QMFrequencyRequest) -> FrequencyResult:
        """Vibrational frequencies in cm⁻¹."""
        self._require(Capability.FREQUENCIES)
        self._require_exact_request(
            request, mm_type=FrequencyRequest, qm_type=QMFrequencyRequest, operation="frequencies"
        )
        return self._validate_frequency_result(self._frequencies(request))

    def parameter_gradient(self, request: ParameterGradientRequest) -> ParameterGradientResult:
        """Energy plus analytical parameter gradient (MM)."""
        self._require(Capability.PARAMETER_GRADIENT)
        self._require_exact_request(
            request, mm_type=ParameterGradientRequest, qm_type=None, operation="parameter_gradient"
        )
        return self._validate_param_grad_result(self._parameter_gradient(request))

    def hessian_parameter_jacobian(self, request: HessianJacobianRequest) -> HessianJacobianResult:
        """Hessian plus its analytical parameter Jacobian (MM)."""
        self._require(Capability.HESSIAN_PARAMETER_JACOBIAN)
        self._require_exact_request(
            request, mm_type=HessianJacobianRequest, qm_type=None, operation="hessian_parameter_jacobian"
        )
        return self._validate_hess_jac_result(self._hessian_parameter_jacobian(request))

    def batched_energy(self, request: BatchedEnergyRequest) -> BatchedEnergyResult:
        """Energies for a batch of full parameter vectors (MM)."""
        self._require(Capability.BATCHED_ENERGY)
        self._require_exact_request(request, mm_type=BatchedEnergyRequest, qm_type=None, operation="batched_energy")
        n_rows = int(np.asarray(request.parameter_matrix).shape[0])
        return self._validate_batched_energy_result(self._batched_energy(request), n_rows=n_rows)

    # -- hooks (override in concrete sessions) ------------------------------

    def _energy(self, request: EnergyRequest | QMEnergyRequest) -> EnergyResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.ENERGY)

    def _minimize(self, request: MinimizationRequest) -> GeometryResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.MINIMIZE)

    def _optimize_geometry(self, request: QMGeometryOptimizationRequest) -> GeometryResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.GEOMETRY_OPTIMIZATION)

    def _hessian(self, request: HessianRequest | QMHessianRequest) -> HessianResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.HESSIAN)

    def _frequencies(self, request: FrequencyRequest | QMFrequencyRequest) -> FrequencyResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.FREQUENCIES)

    def _parameter_gradient(self, request: ParameterGradientRequest) -> ParameterGradientResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.PARAMETER_GRADIENT)

    def _hessian_parameter_jacobian(self, request: HessianJacobianRequest) -> HessianJacobianResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)

    def _batched_energy(self, request: BatchedEnergyRequest) -> BatchedEnergyResult:
        raise UnsupportedCapabilityError(self._info.name, Capability.BATCHED_ENERGY)


# ---------------------------------------------------------------------------
# Registry descriptors (cheap, side-effect-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DependencyProbe:
    """Cheap, side-effect-free availability probe for a backend.

    Only ``importlib.util.find_spec`` (for Python modules) and
    ``shutil.which`` (for executables) are used.  No backend is constructed,
    no device is enumerated, and no CUDA/XLA/OpenMM platform is initialized.

    Args:
        modules: Importable module names that must resolve.
        executables: Executable names that must be found on ``PATH``.

    """

    modules: tuple[str, ...] = ()
    executables: tuple[str, ...] = ()

    def check(self) -> tuple[bool, str]:
        """Return ``(healthy, reason)`` without importing or constructing."""
        missing_modules = [m for m in self.modules if importlib.util.find_spec(m) is None]
        if missing_modules:
            return False, f"missing Python module(s): {', '.join(missing_modules)}"
        missing_exes = [e for e in self.executables if shutil.which(e) is None]
        if missing_exes:
            return False, f"missing executable(s): {', '.join(missing_exes)}"
        return True, ""


@dataclass(frozen=True)
class BackendDescriptor:
    """Validated, lazily-loadable description of a backend.

    The descriptor carries a **static** :class:`BackendInfo` (generic
    provenance) so the catalog can report a backend's exact capabilities and
    functional forms without importing or constructing it.  On explicit load,
    the constructed backend's runtime info is validated against this static
    declaration.

    Args:
        name: Registry key (e.g. ``"openmm"``, ``"jax-md"``).
        info: Static capability declaration (role/capabilities/forms).
        factory: Import string ``"pkg.module:Attribute"`` naming a zero-arg
            callable (typically the backend class) that returns a
            :class:`Backend`.
        probe: Cheap dependency probe used for **listing only**.
        api_version: Descriptor-API version this descriptor targets.

    """

    name: str
    info: BackendInfo
    factory: str
    probe: DependencyProbe = field(default_factory=DependencyProbe)
    api_version: int = DESCRIPTOR_API_VERSION

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("BackendDescriptor.name must be non-empty.")
        if ":" not in self.factory:
            raise ValueError(f"BackendDescriptor.factory must be 'module:attr'; got {self.factory!r}.")
        if not isinstance(self.info, BackendInfo):
            raise ValueError("BackendDescriptor.info must be a BackendInfo.")
        if self.info.provenance is None or self.info.provenance.backend != self.name:
            raise ValueError(f"BackendDescriptor.info.provenance.backend must equal name {self.name!r}.")
        if self.api_version != DESCRIPTOR_API_VERSION:
            raise ValueError(
                f"BackendDescriptor {self.name!r} targets api_version {self.api_version}, "
                f"this runtime is {DESCRIPTOR_API_VERSION}."
            )

    @property
    def role(self) -> BackendRole:
        """Role from the static info."""
        return self.info.role

    def is_available(self) -> tuple[bool, str]:
        """Return ``(healthy, reason)`` via the cheap probe only (catalog use)."""
        return self.probe.check()

    def load(self, **kwargs: object) -> Backend:
        """Import the factory and construct the backend.

        This is the only place that triggers a real import of the backend
        module.  The probe is **not** consulted here — explicit configuration
        (e.g. an explicit Tinker directory) must be honoured even when a
        generic PATH probe is unhealthy.  The constructor is responsible for
        raising typed :class:`BackendUnavailableError` /
        :class:`BackendConfigurationError` when the backend truly cannot run.

        Args:
            **kwargs: Forwarded to the factory callable.

        Returns:
            Backend: The constructed, validated backend.

        Raises:
            BackendUnavailableError: If the backend module cannot be imported
                or the backend reports itself unavailable.
            BackendConfigurationError: If the factory attribute is missing,
                construction fails, or the runtime info disagrees with the
                static descriptor info.

        """
        module_path, _, attr = self.factory.partition(":")
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise BackendUnavailableError(f"Backend {self.name!r} could not be imported: {exc}") from exc
        try:
            factory = getattr(module, attr)
        except AttributeError as exc:
            raise BackendConfigurationError(
                f"Backend {self.name!r} factory {attr!r} not found in {module_path!r}."
            ) from exc
        try:
            backend: Backend = factory(**kwargs)
        except (BackendUnavailableError, BackendConfigurationError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed config error
            raise BackendConfigurationError(f"Backend {self.name!r} failed to construct: {exc}") from exc

        # Structural protocol validation: the factory must return a Backend
        # (an object exposing ``info`` and ``prepare``).
        if not isinstance(backend, Backend):
            raise BackendConfigurationError(
                f"Backend {self.name!r} factory returned {type(backend).__name__}, "
                "which does not satisfy the Backend protocol (needs 'info' and 'prepare')."
            )
        runtime_info = backend.info
        if not isinstance(runtime_info, BackendInfo):
            raise BackendConfigurationError(
                f"Backend {self.name!r} .info is {type(runtime_info).__name__}, expected BackendInfo."
            )
        if not runtime_info.matches(self.info):
            raise BackendConfigurationError(
                f"Backend {self.name!r} runtime info (role={runtime_info.role.value}, "
                f"capabilities={sorted(c.value for c in runtime_info.capabilities)}, "
                f"forms={sorted(runtime_info.functional_forms)}) disagrees with descriptor "
                f"(role={self.info.role.value}, "
                f"capabilities={sorted(c.value for c in self.info.capabilities)}, "
                f"forms={sorted(self.info.functional_forms)})."
            )
        # Runtime provenance must identify the same backend key and role
        # (the human display name may differ, but the registry key must match).
        prov = runtime_info.provenance
        if prov is None:
            raise BackendConfigurationError(f"Backend {self.name!r} runtime info carries no provenance.")
        if prov.backend != self.name:
            raise BackendConfigurationError(
                f"Backend {self.name!r} runtime provenance.backend {prov.backend!r} does not match "
                f"descriptor name {self.name!r}."
            )
        if prov.role is not self.info.role:
            raise BackendConfigurationError(
                f"Backend {self.name!r} runtime provenance role {prov.role.value} does not match "
                f"descriptor role {self.info.role.value}."
            )
        return backend


@dataclass(frozen=True)
class BackendStatus:
    """Explicit health report for one descriptor in the catalog.

    Args:
        descriptor: The described backend.
        healthy: Whether the cheap probe passed.
        reason: Human-readable reason when ``healthy`` is ``False``.

    """

    descriptor: BackendDescriptor
    healthy: bool
    reason: str

    @property
    def name(self) -> str:
        """Registry key of the described backend."""
        return self.descriptor.name

    @property
    def role(self) -> BackendRole:
        """Role of the described backend."""
        return self.descriptor.role

    @property
    def info(self) -> BackendInfo:
        """Static capability declaration of the described backend."""
        return self.descriptor.info
