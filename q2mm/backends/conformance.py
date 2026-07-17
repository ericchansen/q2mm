"""Dependency-light public conformance checks for backend API version 1.

Backend authors provide an immutable typed case containing the loaded backend,
its validated descriptor, and canonical Q2MM inputs.  The runners execute only
the selected declared capabilities, validate canonical result contracts, and
prove that every undeclared public operation is capability-gated.
"""

from __future__ import annotations

import dataclasses
import enum
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from q2mm.backends.contracts import (
    Backend,
    BackendDescriptor,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BatchedEnergyRequest,
    BatchedEnergyResult,
    BatchedHessianRequest,
    BatchedHessianResult,
    Capability,
    CoordinateGradientResult,
    CoordinateGradientUnit,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    FrequencyRequest,
    FrequencyResult,
    FrequencyUnit,
    GeometryResult,
    HessianJacobianRequest,
    HessianJacobianResult,
    HessianRequest,
    HessianResult,
    HessianUnit,
    LengthUnit,
    MinimizationRequest,
    ParameterGradientRequest,
    ParameterGradientResult,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
    UnsupportedCapabilityError,
    prepare_hessian_batches,
)
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout

__all__ = [
    "ConformanceError",
    "ConformanceOutcome",
    "MMConformanceCase",
    "ReferenceConformanceCase",
    "run_mm_conformance",
    "run_reference_conformance",
]


class ConformanceError(AssertionError):
    """Raised deterministically when a backend violates API-v1 conformance."""


@dataclass(frozen=True)
class ConformanceOutcome:
    """Deterministic summary of one conformance run."""

    backend: str
    role: BackendRole
    executed: tuple[Capability, ...]
    unsupported_verified: tuple[Capability, ...]


@dataclass(frozen=True)
class MMConformanceCase:
    """Typed inputs and bounded capability selection for an MM backend."""

    descriptor: BackendDescriptor
    backend: Backend
    molecule: Molecule
    force_field: ForceField
    capabilities: frozenset[Capability] = field(default_factory=lambda: frozenset({Capability.ENERGY}))
    case_id: str = "conformance-mm"

    def __post_init__(self) -> None:
        _validate_case(self.descriptor, self.backend, self.molecule, self.capabilities, self.case_id, BackendRole.MM)
        if not isinstance(self.force_field, ForceField):
            raise TypeError("MMConformanceCase.force_field must be a ForceField.")
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


@dataclass(frozen=True)
class ReferenceConformanceCase:
    """Typed inputs and bounded capability selection for a reference backend."""

    descriptor: BackendDescriptor
    backend: Backend
    molecule: Molecule
    capabilities: frozenset[Capability] = field(default_factory=lambda: frozenset({Capability.ENERGY}))
    case_id: str = "conformance-reference"

    def __post_init__(self) -> None:
        _validate_case(
            self.descriptor,
            self.backend,
            self.molecule,
            self.capabilities,
            self.case_id,
            BackendRole.REFERENCE,
        )
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))


def _validate_case(
    descriptor: BackendDescriptor,
    backend: Backend,
    molecule: Molecule,
    capabilities: frozenset[Capability],
    case_id: str,
    role: BackendRole,
) -> None:
    if not isinstance(descriptor, BackendDescriptor):
        raise TypeError("conformance descriptor must be a BackendDescriptor.")
    if not isinstance(backend, Backend):
        raise TypeError("conformance backend must satisfy the Backend protocol.")
    if not isinstance(molecule, Molecule):
        raise TypeError("conformance molecule must be a Molecule.")
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("conformance case_id must be a non-empty string.")
    selected = frozenset(capabilities)
    if not all(isinstance(capability, Capability) for capability in selected):
        raise TypeError("conformance capabilities must contain only Capability members.")
    if descriptor.role is not role:
        raise ValueError(f"{descriptor.name}: {role.value} conformance requires a {role.value} descriptor.")


def _validate_runtime(descriptor: BackendDescriptor, backend: Backend) -> BackendInfo:
    info = backend.info
    if not isinstance(info, BackendInfo):
        raise ConformanceError(f"{descriptor.name}: runtime info is not BackendInfo.")
    if info.role is not descriptor.role:
        raise ConformanceError(
            f"{descriptor.name}: runtime role {info.role.value!r} does not equal descriptor role "
            f"{descriptor.role.value!r}."
        )
    capability_overclaims = info.capabilities - descriptor.capability_ceiling
    if capability_overclaims:
        values = sorted(capability.value for capability in capability_overclaims)
        raise ConformanceError(f"{descriptor.name}: runtime capabilities exceed the static ceiling: {values}.")
    form_overclaims = info.functional_forms - descriptor.functional_form_ceiling
    if form_overclaims:
        raise ConformanceError(
            f"{descriptor.name}: runtime functional forms exceed the static ceiling: {sorted(form_overclaims)}."
        )
    provenance = info.provenance
    if provenance is None:
        raise ConformanceError(f"{descriptor.name}: runtime info has no provenance.")
    _validate_provenance(descriptor, provenance)
    return info


def _validate_selection(info: BackendInfo, selected: frozenset[Capability]) -> None:
    undeclared = selected - info.capabilities
    if undeclared:
        values = sorted(capability.value for capability in undeclared)
        raise ConformanceError(f"{info.name}: selected capabilities are not declared at runtime: {values}.")
    if Capability.ENERGY in info.capabilities and Capability.ENERGY not in selected:
        raise ConformanceError(f"{info.name}: ENERGY is declared and must be selected for conformance.")


def _validate_provenance(descriptor: BackendDescriptor, provenance: BackendProvenance) -> None:
    if provenance.backend != descriptor.name or provenance.role is not descriptor.role:
        raise ConformanceError(
            f"{descriptor.name}: result provenance must identify backend {descriptor.name!r} "
            f"with role {descriptor.role.value!r}."
        )
    try:
        json.dumps(
            {
                "backend": provenance.backend,
                "role": provenance.role.value,
                "version": provenance.version,
                "details": provenance.details,
            },
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ConformanceError(f"{descriptor.name}: result provenance is not JSON-serializable.") from exc
    if not _is_deeply_immutable_json(provenance.details):
        raise ConformanceError(f"{descriptor.name}: result provenance details are not deeply immutable.")


def _is_deeply_immutable_json(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, tuple):
        return all(_is_deeply_immutable_json(item) for item in value)
    if isinstance(value, Mapping):
        if type(value) is dict:
            return False
        return all(isinstance(key, str) and _is_deeply_immutable_json(item) for key, item in value.items())
    return False


def _snapshot(value: object) -> object:
    """Create a comparable immutable snapshot of canonical conformance inputs."""
    if isinstance(value, np.ndarray):
        return ("array", value.dtype.str, value.shape, value.tobytes())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (
            type(value).__qualname__,
            tuple((item.name, _snapshot(getattr(value, item.name))) for item in dataclasses.fields(value)),
        )
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _snapshot(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_snapshot(item) for item in value), key=repr))
    if isinstance(value, (enum.Enum, Path)):
        return (type(value).__qualname__, str(value))
    return value


def _assert_unmutated(name: str, value: object, before: object) -> None:
    if _snapshot(value) != before:
        raise ConformanceError(f"{name}: backend mutated a conformance input.")


def _call_declared(
    descriptor: BackendDescriptor,
    capability: Capability,
    method: Callable[[object], object],
    request: object,
) -> object:
    request_before = _snapshot(request)
    try:
        result = method(request)
    except UnsupportedCapabilityError as exc:
        raise ConformanceError(
            f"{descriptor.name}: declared capability {capability.value} raised UnsupportedCapabilityError."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - normalized into deterministic public outcome
        raise ConformanceError(f"{descriptor.name}: declared capability {capability.value} failed: {exc!r}.") from exc
    if _snapshot(request) != request_before:
        raise ConformanceError(f"{descriptor.name}: {capability.value} mutated its typed request.")
    return result


def _prove_unsupported(
    descriptor: BackendDescriptor,
    capability: Capability,
    method: Callable[[object], object],
    request: object,
) -> None:
    request_before = _snapshot(request)
    try:
        method(request)
    except UnsupportedCapabilityError as exc:
        if _snapshot(request) != request_before:
            raise ConformanceError(
                f"{descriptor.name}: undeclared {capability.value} mutated its typed request."
            ) from exc
        if exc.capability is not capability:
            raise ConformanceError(
                f"{descriptor.name}: undeclared {capability.value} reported {exc.capability.value}."
            ) from exc
    except Exception as exc:  # noqa: BLE001 - wrong public error type
        raise ConformanceError(
            f"{descriptor.name}: undeclared capability {capability.value} raised "
            f"{type(exc).__name__}, expected UnsupportedCapabilityError."
        ) from exc
    else:
        raise ConformanceError(
            f"{descriptor.name}: undeclared capability {capability.value} did not raise UnsupportedCapabilityError."
        )


def _validate_result(
    descriptor: BackendDescriptor,
    capability: Capability,
    result: object,
    expected_type: type,
    *,
    n_atoms: int,
    n_params: int,
    batch_size: int = 1,
) -> None:
    if type(result) is not expected_type:
        raise ConformanceError(
            f"{descriptor.name}: {capability.value} returned {type(result).__name__}; "
            f"expected exact {expected_type.__name__}."
        )
    provenance = getattr(result, "provenance", None)
    if not isinstance(provenance, BackendProvenance):
        raise ConformanceError(f"{descriptor.name}: {capability.value} result has no BackendProvenance.")
    _validate_provenance(descriptor, provenance)
    expected_energy_unit = EnergyUnit.KCAL_PER_MOL if descriptor.role is BackendRole.MM else EnergyUnit.HARTREE
    if isinstance(result, EnergyResult) and result.unit is not expected_energy_unit:
        raise ConformanceError(f"{descriptor.name}: energy result has a non-canonical unit.")
    if isinstance(result, GeometryResult):
        if result.energy_unit is not expected_energy_unit or result.coordinate_unit is not LengthUnit.ANGSTROM:
            raise ConformanceError(f"{descriptor.name}: geometry result has non-canonical units.")
        if result.coordinates.shape != (n_atoms, 3):
            raise ConformanceError(f"{descriptor.name}: geometry result has shape {result.coordinates.shape}.")
    if isinstance(result, HessianResult):
        if result.unit is not HessianUnit.HARTREE_PER_BOHR2 or result.hessian.shape != (3 * n_atoms, 3 * n_atoms):
            raise ConformanceError(f"{descriptor.name}: hessian result has non-canonical units or shape.")
    if isinstance(result, FrequencyResult) and result.unit is not FrequencyUnit.INVERSE_CM:
        raise ConformanceError(f"{descriptor.name}: frequency result has a non-canonical unit.")
    if isinstance(result, ParameterGradientResult):
        if result.unit is not expected_energy_unit or result.gradient.shape != (n_params,):
            raise ConformanceError(f"{descriptor.name}: parameter-gradient result has non-canonical units or shape.")
    if isinstance(result, CoordinateGradientResult):
        if result.unit is not CoordinateGradientUnit.HARTREE_PER_BOHR or result.gradient.shape != (n_atoms, 3):
            raise ConformanceError(f"{descriptor.name}: coordinate-gradient result has non-canonical units or shape.")
    if isinstance(result, HessianJacobianResult):
        expected = (3 * n_atoms, 3 * n_atoms)
        if (
            result.unit is not HessianUnit.HARTREE_PER_BOHR2
            or result.hessian.shape != expected
            or result.jacobian.shape != (*expected, n_params)
        ):
            raise ConformanceError(f"{descriptor.name}: Hessian-Jacobian result has non-canonical units or shape.")
    if isinstance(result, BatchedEnergyResult):
        if result.unit is not expected_energy_unit or result.energies.shape != (batch_size,):
            raise ConformanceError(f"{descriptor.name}: batched-energy result has non-canonical units or shape.")
    if isinstance(result, BatchedHessianResult):
        expected_batch = (batch_size, 3 * n_atoms, 3 * n_atoms)
        if result.unit is not HessianUnit.HARTREE_PER_BOHR2 or result.hessians.shape != expected_batch:
            raise ConformanceError(f"{descriptor.name}: batched-Hessian result has non-canonical units or shape.")


_MM_DRIVERS: dict[Capability, tuple[str, Callable[[np.ndarray], object], type]] = {
    Capability.ENERGY: ("energy", lambda vector: EnergyRequest(parameters=vector), EnergyResult),
    Capability.MINIMIZE: ("minimize", lambda vector: MinimizationRequest(parameters=vector), GeometryResult),
    Capability.HESSIAN: ("hessian", lambda vector: HessianRequest(parameters=vector), HessianResult),
    Capability.FREQUENCIES: ("frequencies", lambda vector: FrequencyRequest(parameters=vector), FrequencyResult),
    Capability.GEOMETRY_OPTIMIZATION: (
        "optimize_geometry",
        lambda _vector: ReferenceGeometryOptimizationRequest(),
        GeometryResult,
    ),
    Capability.PARAMETER_GRADIENT: (
        "parameter_gradient",
        lambda vector: ParameterGradientRequest(parameters=vector),
        ParameterGradientResult,
    ),
    Capability.COORDINATE_GRADIENT: (
        "coordinate_gradient",
        lambda _vector: ReferenceCoordinateGradientRequest(),
        CoordinateGradientResult,
    ),
    Capability.HESSIAN_PARAMETER_JACOBIAN: (
        "hessian_parameter_jacobian",
        lambda vector: HessianJacobianRequest(parameters=vector),
        HessianJacobianResult,
    ),
    Capability.BATCHED_ENERGY: (
        "batched_energy",
        lambda vector: BatchedEnergyRequest(parameter_matrix=vector.reshape(1, -1)),
        BatchedEnergyResult,
    ),
}

_REFERENCE_DRIVERS: dict[Capability, tuple[str, Callable[[], object], type]] = {
    Capability.ENERGY: ("energy", ReferenceEnergyRequest, EnergyResult),
    Capability.MINIMIZE: (
        "minimize",
        lambda: MinimizationRequest(parameters=np.empty(0)),
        GeometryResult,
    ),
    Capability.COORDINATE_GRADIENT: (
        "coordinate_gradient",
        ReferenceCoordinateGradientRequest,
        CoordinateGradientResult,
    ),
    Capability.HESSIAN: ("hessian", ReferenceHessianRequest, HessianResult),
    Capability.FREQUENCIES: ("frequencies", ReferenceFrequencyRequest, FrequencyResult),
    Capability.GEOMETRY_OPTIMIZATION: (
        "optimize_geometry",
        ReferenceGeometryOptimizationRequest,
        GeometryResult,
    ),
    Capability.PARAMETER_GRADIENT: (
        "parameter_gradient",
        lambda: ParameterGradientRequest(parameters=np.empty(0)),
        ParameterGradientResult,
    ),
    Capability.HESSIAN_PARAMETER_JACOBIAN: (
        "hessian_parameter_jacobian",
        lambda: HessianJacobianRequest(parameters=np.empty(0)),
        HessianJacobianResult,
    ),
    Capability.BATCHED_ENERGY: (
        "batched_energy",
        lambda: BatchedEnergyRequest(parameter_matrix=np.empty((1, 0))),
        BatchedEnergyResult,
    ),
}


def run_mm_conformance(case: MMConformanceCase) -> ConformanceOutcome:
    """Run the selected API-v1 MM checks and all undeclared-operation gates."""
    if not isinstance(case, MMConformanceCase):
        raise TypeError("run_mm_conformance requires an MMConformanceCase.")
    info = _validate_runtime(case.descriptor, case.backend)
    _validate_selection(info, case.capabilities)
    molecule_before = _snapshot(case.molecule)
    force_field_before = _snapshot(case.force_field)
    layout = ParameterLayout.from_force_field(case.force_field)
    vector = layout.vector(case.force_field)
    vector_before = vector.copy()
    prepared = case.backend.prepare(
        PreparationRequest(
            case_id=case.case_id,
            molecule=case.molecule,
            force_field=case.force_field,
        )
    )
    executed: list[Capability] = []
    unsupported: list[Capability] = []
    reusable: tuple[Callable[[object], object], object, type, Capability] | None = None

    for capability, (method_name, build_request, result_type) in _MM_DRIVERS.items():
        method = getattr(prepared, method_name)
        request = build_request(vector)
        if capability in info.capabilities:
            if capability not in case.capabilities:
                continue
            result = _call_declared(case.descriptor, capability, method, request)
            _validate_result(
                case.descriptor,
                capability,
                result,
                result_type,
                n_atoms=case.molecule.n_atoms,
                n_params=len(vector),
            )
            executed.append(capability)
            if reusable is None or capability is Capability.ENERGY:
                reusable = (method, request, result_type, capability)
        else:
            _prove_unsupported(case.descriptor, capability, method, request)
            unsupported.append(capability)

    if Capability.BATCHED_HESSIAN in info.capabilities:
        if Capability.BATCHED_HESSIAN in case.capabilities:
            try:
                batches = prepare_hessian_batches(case.backend, [prepared])
                if not batches:
                    raise ConformanceError(f"{case.descriptor.name}: BATCHED_HESSIAN prepared no batches.")
                for batch in batches:
                    result = batch.hessians(BatchedHessianRequest(parameters=vector))
                    _validate_result(
                        case.descriptor,
                        Capability.BATCHED_HESSIAN,
                        result,
                        BatchedHessianResult,
                        n_atoms=case.molecule.n_atoms,
                        n_params=len(vector),
                        batch_size=len(batch.case_ids),
                    )
            except ConformanceError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise ConformanceError(
                    f"{case.descriptor.name}: declared capability batched_hessian failed: {exc!r}."
                ) from exc
            executed.append(Capability.BATCHED_HESSIAN)
    else:
        try:
            prepare_hessian_batches(case.backend, [prepared])
        except UnsupportedCapabilityError:
            unsupported.append(Capability.BATCHED_HESSIAN)
        except Exception as exc:  # noqa: BLE001
            raise ConformanceError(
                f"{case.descriptor.name}: undeclared capability batched_hessian raised "
                f"{type(exc).__name__}, expected UnsupportedCapabilityError."
            ) from exc
        else:
            raise ConformanceError(
                f"{case.descriptor.name}: undeclared capability batched_hessian did not raise "
                "UnsupportedCapabilityError."
            )

    if Capability.REUSABLE_STATE in info.capabilities:
        if reusable is None:
            raise ConformanceError(
                f"{case.descriptor.name}: reusable_state was selected but no selected prepared-session "
                "capability can prove reuse."
            )
        method, request, result_type, capability = reusable
        first = _call_declared(case.descriptor, capability, method, request)
        second = _call_declared(case.descriptor, capability, method, request)
        for result in (first, second):
            _validate_result(
                case.descriptor,
                capability,
                result,
                result_type,
                n_atoms=case.molecule.n_atoms,
                n_params=len(vector),
            )
        executed.append(Capability.REUSABLE_STATE)

    _assert_unmutated(case.descriptor.name, case.molecule, molecule_before)
    _assert_unmutated(case.descriptor.name, case.force_field, force_field_before)
    if not np.array_equal(vector, vector_before):
        raise ConformanceError(f"{case.descriptor.name}: backend mutated the conformance parameter vector.")
    return ConformanceOutcome(
        backend=case.descriptor.name,
        role=case.descriptor.role,
        executed=tuple(executed),
        unsupported_verified=tuple(unsupported),
    )


def run_reference_conformance(case: ReferenceConformanceCase) -> ConformanceOutcome:
    """Run the selected API-v1 reference checks and undeclared-operation gates."""
    if not isinstance(case, ReferenceConformanceCase):
        raise TypeError("run_reference_conformance requires a ReferenceConformanceCase.")
    info = _validate_runtime(case.descriptor, case.backend)
    _validate_selection(info, case.capabilities)
    molecule_before = _snapshot(case.molecule)
    prepared = case.backend.prepare(PreparationRequest(case_id=case.case_id, molecule=case.molecule))
    executed: list[Capability] = []
    unsupported: list[Capability] = []
    reusable: tuple[Callable[[object], object], object, type, Capability] | None = None
    for capability, (method_name, build_request, result_type) in _REFERENCE_DRIVERS.items():
        method = getattr(prepared, method_name)
        request = build_request()
        if capability in info.capabilities:
            if capability not in case.capabilities:
                continue
            result = _call_declared(case.descriptor, capability, method, request)
            _validate_result(
                case.descriptor,
                capability,
                result,
                result_type,
                n_atoms=case.molecule.n_atoms,
                n_params=0,
            )
            executed.append(capability)
            if reusable is None or capability is Capability.ENERGY:
                reusable = (method, request, result_type, capability)
        else:
            _prove_unsupported(case.descriptor, capability, method, request)
            unsupported.append(capability)

    try:
        prepare_hessian_batches(case.backend, [prepared])
    except UnsupportedCapabilityError:
        unsupported.append(Capability.BATCHED_HESSIAN)
    except Exception as exc:  # noqa: BLE001
        raise ConformanceError(
            f"{case.descriptor.name}: undeclared capability batched_hessian raised "
            f"{type(exc).__name__}, expected UnsupportedCapabilityError."
        ) from exc
    else:
        raise ConformanceError(
            f"{case.descriptor.name}: undeclared capability batched_hessian did not raise UnsupportedCapabilityError."
        )

    if Capability.REUSABLE_STATE in info.capabilities:
        if reusable is None:
            raise ConformanceError(
                f"{case.descriptor.name}: reusable_state was selected but no selected reference capability can prove reuse."
            )
        method, request, result_type, capability = reusable
        for result in (
            _call_declared(case.descriptor, capability, method, request),
            _call_declared(case.descriptor, capability, method, request),
        ):
            _validate_result(
                case.descriptor,
                capability,
                result,
                result_type,
                n_atoms=case.molecule.n_atoms,
                n_params=0,
            )
        executed.append(Capability.REUSABLE_STATE)

    _assert_unmutated(case.descriptor.name, case.molecule, molecule_before)
    return ConformanceOutcome(
        backend=case.descriptor.name,
        role=case.descriptor.role,
        executed=tuple(executed),
        unsupported_verified=tuple(unsupported),
    )
