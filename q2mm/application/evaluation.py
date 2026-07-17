"""Typed generic evaluation services for canonical application inputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

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
from q2mm.models.problem import OptimizationProblem
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import Evaluation, GradientMode, ObjectiveEvaluator
from q2mm.objectives.python import PythonObjectiveExecutor

from .models import ApplicationConfigurationError, ApplicationEvaluationError

Property = Literal[
    "energy",
    "coordinate_gradient",
    "hessian",
    "frequencies",
    "geometry_optimization",
]
ExecutorKind = Literal["auto", "python", "jax"]

_PROPERTY_CAPABILITIES: dict[Property, Capability] = {
    "energy": Capability.ENERGY,
    "coordinate_gradient": Capability.COORDINATE_GRADIENT,
    "hessian": Capability.HESSIAN,
    "frequencies": Capability.FREQUENCIES,
    "geometry_optimization": Capability.GEOMETRY_OPTIMIZATION,
}


def _backend_capabilities(backend: Backend) -> frozenset[Capability]:
    return frozenset(backend.info.capabilities)


def _backend_role(backend: Backend) -> str:
    return backend.info.role.value


def _backend_name(backend: Backend) -> str:
    provenance = backend.info.provenance
    return provenance.backend if provenance is not None else backend.info.name


def _is_jax_backend(backend: Backend) -> bool:
    return _backend_name(backend).strip().lower() == "jax"


def evaluate_problem(
    problem: OptimizationProblem,
    backend: Backend | str,
    *,
    executor: ExecutorKind = "auto",
    gradient_mode: GradientMode | str | None = None,
    fd_step: float = 1e-4,
    regularization: float = 0.0,
    backend_options: Mapping[str, object] | None = None,
) -> Evaluation:
    """Evaluate a canonical optimization problem without changing its inputs.

    ``auto`` chooses the analytical per-case JAX executor only for a configured
    JAX backend. All other backends use the Python executor. The selected
    gradient policy is explicit and never falls back.
    """
    if not isinstance(problem, OptimizationProblem):
        raise ApplicationConfigurationError("evaluate_problem requires an OptimizationProblem.")
    if isinstance(backend, str):
        from q2mm.backends.registry import load_backend

        try:
            backend = load_backend(backend, **dict(backend_options or {}))
        except Exception as exc:
            raise ApplicationConfigurationError(f"Could not load backend for problem evaluation: {exc}") from exc
    elif backend_options:
        raise ApplicationConfigurationError("backend_options are valid only when backend is a registered string.")
    if _backend_role(backend) != "mm":
        raise ApplicationEvaluationError(
            f"OptimizationProblem evaluation requires an MM backend; {_backend_name(backend)!r} has role "
            f"{_backend_role(backend)!r}."
        )
    if executor not in {"auto", "python", "jax"}:
        raise ApplicationConfigurationError(f"Unknown executor {executor!r}; expected 'auto', 'python', or 'jax'.")
    selected = "jax" if executor == "auto" and _is_jax_backend(backend) else executor
    if selected == "auto":
        selected = "python"
    plan = ObjectivePlan.from_problem(problem, regularization=regularization)
    params = problem.layout.vector(problem.starting_force_field)
    try:
        if selected == "jax":
            if not _is_jax_backend(backend):
                raise ApplicationEvaluationError(
                    f"JAX executor requires the registered JAX backend, not {_backend_name(backend)!r}."
                )
            mode = GradientMode.ANALYTICAL if gradient_mode is None else GradientMode(gradient_mode)
            if mode is not GradientMode.ANALYTICAL:
                raise ApplicationConfigurationError(
                    "The JAX executor provides analytical gradients only; choose executor='python' "
                    "for finite-difference or scalar evaluation."
                )
            evaluator: ObjectiveEvaluator = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
        else:
            mode = GradientMode.NONE if gradient_mode is None else GradientMode(gradient_mode)
            if mode is GradientMode.ANALYTICAL:
                raise ApplicationConfigurationError(
                    "The Python executor does not provide analytical gradients; choose executor='jax'."
                )
            evaluator = PythonObjectiveExecutor(
                plan,
                backend,
                problem.starting_force_field,
                gradient_mode=mode,
                fd_step=fd_step,
            )
        return evaluator.evaluate(params)
    except (ApplicationConfigurationError, ApplicationEvaluationError):
        raise
    except (BackendError, ValueError, TypeError) as exc:
        raise ApplicationEvaluationError(f"Problem evaluation failed: {exc}") from exc


def evaluate_property(
    molecule: Molecule,
    backend: Backend,
    *,
    property: Property | None = None,
    capability: Capability | str | None = None,
    preparation_options: dict[str, object] | None = None,
) -> EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult:
    """Evaluate exactly one property with an already-loaded reference backend."""
    if not isinstance(molecule, Molecule):
        raise ApplicationConfigurationError("evaluate_property requires a Molecule.")
    if backend.info.role is not BackendRole.REFERENCE:
        raise ApplicationEvaluationError(
            f"Bare-molecule property evaluation requires a REFERENCE backend; {_backend_name(backend)!r} has "
            f"role {_backend_role(backend)!r}."
        )
    property_from_capability: Property | None = None
    if capability is not None:
        try:
            requested_capability = capability if isinstance(capability, Capability) else Capability(capability)
        except ValueError as exc:
            raise ApplicationConfigurationError(f"Unknown capability {capability!r}.") from exc
        matches = [name for name, value in _PROPERTY_CAPABILITIES.items() if value is requested_capability]
        if not matches:
            raise ApplicationConfigurationError(
                f"Capability {requested_capability.value!r} is not a supported bare-molecule property."
            )
        property_from_capability = matches[0]
    if property is None and property_from_capability is None:
        raise ApplicationConfigurationError("Specify exactly one property or capability.")
    if property is not None and property not in _PROPERTY_CAPABILITIES:
        raise ApplicationConfigurationError(f"Unknown property {property!r}.")
    if property is not None and property_from_capability is not None:
        raise ApplicationConfigurationError("Specify property or capability, not both.")
    selected = property if property is not None else property_from_capability
    assert selected is not None
    required = _PROPERTY_CAPABILITIES[selected]
    if required not in _backend_capabilities(backend):
        raise ApplicationEvaluationError(
            f"Reference backend {_backend_name(backend)!r} does not declare capability {required.value!r}."
        )
    request: Any
    if selected == "energy":
        request = ReferenceEnergyRequest()
        result_type: type[Any] = EnergyResult
        operation = "energy"
    elif selected == "coordinate_gradient":
        request = ReferenceCoordinateGradientRequest()
        result_type = CoordinateGradientResult
        operation = "coordinate_gradient"
    elif selected == "hessian":
        request = ReferenceHessianRequest()
        result_type = HessianResult
        operation = "hessian"
    elif selected == "frequencies":
        request = ReferenceFrequencyRequest()
        result_type = FrequencyResult
        operation = "frequencies"
    else:
        request = ReferenceGeometryOptimizationRequest()
        result_type = GeometryResult
        operation = "optimize_geometry"
    try:
        prepared = backend.prepare(
            PreparationRequest(
                case_id=molecule.name or "application-property",
                molecule=molecule,
                options={} if preparation_options is None else preparation_options,
            )
        )
        result = getattr(prepared, operation)(request)
    except (BackendError, ValueError, TypeError) as exc:
        raise ApplicationEvaluationError(f"Property {selected!r} evaluation failed: {exc}") from exc
    if not isinstance(result, result_type):
        raise ApplicationEvaluationError(
            f"Backend returned {type(result).__name__} for property {selected!r}; expected {result_type.__name__}."
        )
    return cast(EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult, result)


def evaluate(
    target: OptimizationProblem | Molecule,
    backend: Backend | str,
    *,
    property: Property | None = None,
    capability: Capability | str | None = None,
    executor: ExecutorKind = "auto",
    gradient_mode: GradientMode | str | None = None,
    fd_step: float = 1e-4,
    regularization: float = 0.0,
    preparation_options: dict[str, object] | None = None,
    backend_options: Mapping[str, object] | None = None,
) -> Evaluation | EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult:
    """Dispatch to canonical problem or bare-molecule evaluation."""
    if isinstance(target, OptimizationProblem):
        if property is not None or capability is not None or preparation_options is not None:
            raise ApplicationConfigurationError(
                "Property, capability, and preparation_options apply only to bare Molecule evaluation."
            )
        return evaluate_problem(
            target,
            backend,
            executor=executor,
            gradient_mode=gradient_mode,
            fd_step=fd_step,
            regularization=regularization,
            backend_options=backend_options,
        )
    if isinstance(target, Molecule):
        if isinstance(backend, str):
            raise ApplicationConfigurationError(
                "Bare-molecule property evaluation requires an already-loaded reference backend."
            )
        if backend_options:
            raise ApplicationConfigurationError("backend_options apply only to registered problem backends.")
        if executor != "auto" or gradient_mode is not None or fd_step != 1e-4 or regularization != 0:
            raise ApplicationConfigurationError(
                "Executor, gradient, finite-difference, regularization, and evaluation-count arguments apply "
                "only to OptimizationProblem evaluation."
            )
        return evaluate_property(
            target,
            backend,
            property=property,
            capability=capability,
            preparation_options=preparation_options,
        )
    raise ApplicationConfigurationError("evaluate requires an OptimizationProblem or Molecule.")


__all__ = ["ExecutorKind", "Property", "evaluate", "evaluate_problem", "evaluate_property"]
