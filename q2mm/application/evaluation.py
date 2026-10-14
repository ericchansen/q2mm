"""Explicit evaluator construction, numerical evaluation, and reference service."""

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np

from q2mm.backends.contracts import (
    Backend,
    BackendError,
    BackendRole,
    Capability,
    CoordinateGradientResult,
    EnergyResult,
    FrequencyResult,
    GeometryResult,
    HessianResult,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
)
from q2mm.models.molecule import Molecule
from q2mm.objectives.protocols import Evaluation, ObjectiveEvaluator

from .models import ApplicationConfigurationError, ApplicationEvaluationError

Property = Literal["energy", "coordinate_gradient", "hessian", "frequencies", "geometry_optimization"]
ReferenceResult = EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult

_PROPERTIES = {
    "energy": (Capability.ENERGY, ReferenceEnergyRequest, EnergyResult, "energy"),
    "coordinate_gradient": (
        Capability.COORDINATE_GRADIENT,
        ReferenceCoordinateGradientRequest,
        CoordinateGradientResult,
        "coordinate_gradient",
    ),
    "hessian": (Capability.HESSIAN, ReferenceHessianRequest, HessianResult, "hessian"),
    "frequencies": (Capability.FREQUENCIES, ReferenceFrequencyRequest, FrequencyResult, "frequencies"),
    "geometry_optimization": (
        Capability.GEOMETRY_OPTIMIZATION,
        ReferenceGeometryOptimizationRequest,
        GeometryResult,
        "optimize_geometry",
    ),
}


def evaluate(evaluator: ObjectiveEvaluator, parameters: np.ndarray) -> Evaluation:
    """Evaluate the supplied full vector, preserving executor accounting and errors."""
    try:
        result = evaluator.evaluate(parameters)
    except (BackendError, ValueError, TypeError) as exc:
        raise ApplicationEvaluationError(f"Objective evaluation failed: {exc}") from exc
    if not isinstance(result, Evaluation):
        raise ApplicationEvaluationError(f"Executor returned {type(result).__name__}; expected Evaluation.")
    return result


def evaluate_reference(
    molecule: Molecule,
    *,
    backend: Backend,
    property: Property,
    preparation_options: Mapping[str, object] | None = None,
) -> ReferenceResult:
    """Request exactly one property from an already-loaded reference backend."""
    if not isinstance(molecule, Molecule):
        raise ApplicationConfigurationError("evaluate_reference requires a Molecule.")
    if backend.info.role is not BackendRole.REFERENCE:
        raise ApplicationEvaluationError(
            f"Reference property evaluation requires a REFERENCE backend; {backend.info.name!r} "
            f"has role {backend.info.role.value!r}."
        )
    if property not in _PROPERTIES:
        raise ApplicationConfigurationError(f"Unknown property {property!r}.")
    capability, request_type, result_type, operation = _PROPERTIES[property]
    if capability not in backend.info.capabilities:
        raise ApplicationEvaluationError(
            f"Reference backend {backend.info.name!r} does not declare capability {capability.value!r}."
        )
    try:
        prepared = backend.prepare(
            PreparationRequest(
                case_id=molecule.name or "application-property",
                molecule=molecule,
                options={} if preparation_options is None else preparation_options,
            )
        )
        result: Any = getattr(prepared, operation)(request_type())
    except (BackendError, ValueError, TypeError) as exc:
        raise ApplicationEvaluationError(f"Property {property!r} evaluation failed: {exc}") from exc
    if not isinstance(result, result_type):
        raise ApplicationEvaluationError(
            f"Backend returned {type(result).__name__} for property {property!r}; expected {result_type.__name__}."
        )
    return cast(ReferenceResult, result)


__all__ = ["Property", "evaluate", "evaluate_reference"]
