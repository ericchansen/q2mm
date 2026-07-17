"""Generic optimization application service."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

import numpy as np

from q2mm._canonical import canonical_fingerprint
from q2mm.backends.contracts import Backend, BackendRole
from q2mm.models.forcefield import ForceField
from q2mm.models.problem import OptimizationProblem, StationaryPointKind
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import GradientMode
from q2mm.optimizers.catalog import (
    OptimizerSpec,
    expected_result_gradient,
    optimizer_spec,
    resolve_optimizer,
)
from q2mm.optimizers.protocols import _Optimizer
from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow, Workflow, make_evaluator_factory

from .models import (
    ApplicationConfigurationError,
    ApplicationOptimizationError,
    OptimizationRun,
    ResolvedBackendConfiguration,
    ResolvedExecutionConfiguration,
    ResolvedExecutorConfiguration,
    ResolvedOptimizerConfiguration,
    ResolvedWorkflowConfiguration,
    problem_fingerprint,
    problem_input_fingerprints,
)

Recipe = Literal["recommended", "explicit"]
Executor = Literal["auto", "python", "jax"]

_RECOMMENDED_OPTIMIZER = "scipy-lbfgsb-jax"


def _load_backend(value: Backend | str, options: Mapping[str, object] | None) -> tuple[Backend, str]:
    if isinstance(value, str):
        from q2mm.backends.registry import load_backend

        try:
            return load_backend(value, **dict(options or {})), value
        except Exception as exc:
            raise ApplicationConfigurationError(f"Could not load backend {value!r}: {exc}") from exc
    if options:
        raise ApplicationConfigurationError("backend_options are valid only when backend is a registered string.")
    if not hasattr(value, "info") or not hasattr(value, "prepare"):
        raise ApplicationConfigurationError("backend must be a registered key or Backend object.")
    provenance = value.info.provenance
    key = provenance.backend if provenance is not None else value.info.name
    return value, key


def _backend_configuration(
    backend: Backend, key: str, options: Mapping[str, object] | None
) -> ResolvedBackendConfiguration:
    info = backend.info
    provenance = info.provenance
    details: dict[str, object] = {}
    if provenance is not None:
        details.update(provenance.details)
    if options:
        details["factory_options"] = dict(options)
    return ResolvedBackendConfiguration(
        key=key,
        name=info.name,
        role=info.role.value,
        version="" if provenance is None else provenance.version,
        capabilities=tuple(capability.value for capability in info.capabilities),
        functional_forms=tuple(info.functional_forms),
        details=details,
    )


def _problem_kind(problem: OptimizationProblem) -> StationaryPointKind | None:
    kinds = {case.stationary_point for case in problem.cases}
    return next(iter(kinds)) if len(kinds) == 1 else None


def _recommended_defaults(kind: StationaryPointKind) -> tuple[str, dict[str, object], float | None]:
    if kind is StationaryPointKind.GROUND_STATE:
        return "recommended-jax-gs-v1", {"ftol": 1e-8}, 10.0
    return (
        "recommended-jax-ts-v1",
        {"ftol": 1e-12, "fc_fraction": 0.20, "eq_fraction": 0.05},
        None,
    )


def _custom_component_settings(value: object) -> dict[str, str]:
    cls = type(value)
    return {"class": cls.__qualname__, "module": cls.__module__}


def _resolve_workflow(
    value: str | Workflow,
    options: Mapping[str, Any] | None,
) -> tuple[Workflow, ResolvedWorkflowConfiguration]:
    supplied = dict(options or {})
    if not isinstance(value, str):
        if supplied:
            raise ApplicationConfigurationError("workflow_options cannot be applied to a workflow object.")
        if not isinstance(value, Workflow):
            raise ApplicationConfigurationError("workflow object does not implement the Workflow protocol.")
        key = str(value.name)
        return value, ResolvedWorkflowConfiguration(key=key, settings=_custom_component_settings(value))
    if value == "single-stage":
        if supplied:
            raise ApplicationConfigurationError(
                f"SingleStageWorkflow takes no workflow_options; got {sorted(supplied)}."
            )
        return SingleStageWorkflow(), ResolvedWorkflowConfiguration(
            key="single-stage", settings={"name": "single-stage"}
        )
    if value == "method-e2":
        allowed = {
            "negative_fc_threshold",
            "replace_with_round2",
            "allow_negative",
            "near_zero_replace_with",
        }
        unknown = set(supplied) - allowed
        if unknown:
            raise ApplicationConfigurationError(f"Unknown method-e2 workflow options: {sorted(unknown)}.")
        try:
            workflow = MethodE2Workflow(**supplied)
        except (TypeError, ValueError) as exc:
            raise ApplicationConfigurationError(f"Invalid method-e2 workflow options: {exc}") from exc
        settings = {
            "name": "method-e2",
            "negative_fc_threshold": workflow.negative_fc_threshold,
            "replace_with_round2": workflow.replace_with_round2,
            "allow_negative": workflow.allow_negative,
            "near_zero_replace_with": dict(workflow.near_zero_replace_with),
        }
        return workflow, ResolvedWorkflowConfiguration(key="method-e2", settings=settings)
    raise ApplicationConfigurationError("Unknown workflow; expected 'single-stage', 'method-e2', or a Workflow object.")


def _resolve_optimizer(
    value: str | OptimizerSpec | _Optimizer,
    options: Mapping[str, Any] | None,
    *,
    executor: Executor,
    requested_gradient_mode: GradientMode | str | None,
    requested_fd_step: float,
) -> tuple[_Optimizer, ResolvedOptimizerConfiguration, Literal["python", "jax"], str, float]:
    try:
        requested_mode = None if requested_gradient_mode is None else GradientMode(requested_gradient_mode)
    except ValueError as exc:
        raise ApplicationConfigurationError(f"Unknown gradient_mode {requested_gradient_mode!r}.") from exc
    if isinstance(value, (str, OptimizerSpec)):
        try:
            spec = optimizer_spec(value)
            optimizer, settings = resolve_optimizer(spec, options)
        except (TypeError, ValueError) as exc:
            raise ApplicationConfigurationError(f"Invalid optimizer configuration: {exc}") from exc
        if executor != "auto" and executor != spec.evaluator:
            raise ApplicationConfigurationError(
                f"executor={executor!r} conflicts with optimizer {spec.key!r}, which requires {spec.evaluator!r}."
            )
        if requested_mode is not None and requested_mode.value != spec.gradient_mode:
            raise ApplicationConfigurationError(
                f"gradient_mode={requested_mode.value!r} conflicts with optimizer "
                f"{spec.key!r}, which requires {spec.gradient_mode!r}."
            )
        if requested_fd_step != 1e-4 and spec.gradient_mode != "finite_difference":
            raise ApplicationConfigurationError("fd_step applies only to finite-difference executor configurations.")
        return (
            optimizer,
            ResolvedOptimizerConfiguration(
                key=spec.key,
                label=spec.label,
                method=spec.method,
                settings=settings,
                expected_result_gradient_mode=expected_result_gradient(spec),
            ),
            cast(Literal["python", "jax"], spec.evaluator),
            spec.gradient_mode,
            requested_fd_step if spec.gradient_mode == "finite_difference" else spec.fd_step,
        )
    if options:
        raise ApplicationConfigurationError("optimizer_options cannot be applied to an optimizer object.")
    if executor == "auto":
        raise ApplicationConfigurationError("A custom optimizer object requires an explicit executor.")
    if not isinstance(value, _Optimizer):
        raise ApplicationConfigurationError("optimizer object does not implement the optimizer protocol.")
    gradient = (
        "analytical"
        if executor == "jax"
        else GradientMode.NONE.value
        if requested_mode is None
        else requested_mode.value
    )
    if executor == "jax" and requested_mode not in (None, GradientMode.ANALYTICAL):
        raise ApplicationConfigurationError("A custom JAX optimizer requires analytical gradient mode.")
    if requested_fd_step != 1e-4 and gradient != "finite_difference":
        raise ApplicationConfigurationError("fd_step applies only to finite-difference executor configurations.")
    settings = _custom_component_settings(value)
    return (
        value,
        ResolvedOptimizerConfiguration(
            key=f"custom:{type(value).__module__}.{type(value).__qualname__}",
            label=type(value).__qualname__,
            method=type(value).__qualname__,
            settings=settings,
            expected_result_gradient_mode=gradient,
        ),
        executor,
        gradient,
        requested_fd_step,
    )


def execute_optimization(
    problem: OptimizationProblem,
    backend: Backend,
    optimizer: _Optimizer,
    workflow: Workflow,
    *,
    executor: Literal["python", "jax"],
    gradient_mode: GradientMode = GradientMode.NONE,
    fd_step: float = 1e-4,
    regularization: float = 0.0,
    n_evals: int = 1,
) -> tuple[OptimizationResult, ForceField]:
    """Execute already-resolved components and materialize the final force field."""
    if executor == "jax":
        if gradient_mode is not GradientMode.ANALYTICAL:
            raise ApplicationConfigurationError("JAX execution requires gradient_mode=ANALYTICAL.")
        make_evaluator = make_evaluator_factory(
            backend,
            problem.starting_force_field,
            executor="jax",
        )
    else:
        if gradient_mode is GradientMode.ANALYTICAL:
            raise ApplicationConfigurationError("Python execution cannot provide analytical gradients.")
        make_evaluator = make_evaluator_factory(
            backend,
            problem.starting_force_field,
            executor="python",
            gradient_mode=gradient_mode,
            fd_step=fd_step,
        )
    result = workflow.run(
        problem,
        make_evaluator,
        optimizer,
        n_evals=n_evals,
        regularization=regularization,
    )
    if not isinstance(result, OptimizationResult):
        raise ApplicationOptimizationError(f"Workflow returned {type(result).__name__}; expected OptimizationResult.")
    if result.layout_fingerprint != problem.layout.fingerprint or result.n_params != len(problem.layout):
        raise ApplicationOptimizationError("Optimization result does not match the problem parameter layout.")
    inactive = np.setdiff1d(np.arange(problem.active_space.n_full), problem.active_space.active_indices)
    baseline = np.asarray(problem.active_space.baseline)
    if not np.array_equal(result.initial_params, baseline):
        raise ApplicationOptimizationError("Optimization result initial vector does not match the problem baseline.")
    if not np.array_equal(result.final_params[inactive], baseline[inactive]):
        raise ApplicationOptimizationError("Optimization result changed frozen parameter slots.")
    return result, problem.layout.replace(problem.starting_force_field, result.final_params)


def optimize(
    problem: OptimizationProblem,
    backend: Backend | str,
    *,
    recipe: Recipe = "recommended",
    optimizer: str | OptimizerSpec | _Optimizer | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
    workflow: str | Workflow | None = None,
    workflow_options: Mapping[str, Any] | None = None,
    executor: Executor = "auto",
    gradient_mode: GradientMode | str | None = None,
    fd_step: float = 1e-4,
    backend_options: Mapping[str, object] | None = None,
    regularization: float | None = None,
    n_evals: int = 1,
) -> OptimizationRun:
    """Resolve and execute one canonical optimization problem.

    The recommended recipe is deliberately narrow: it exists only for the
    built-in JAX backend and unambiguously all-ground-state or all-transition-
    state problems. Passing explicit optimizer/workflow components records and
    applies those overrides.
    """
    if not isinstance(problem, OptimizationProblem):
        raise ApplicationConfigurationError("optimize requires an OptimizationProblem.")
    if recipe not in {"recommended", "explicit"}:
        raise ApplicationConfigurationError("recipe must be 'recommended' or 'explicit'.")
    if executor not in {"auto", "python", "jax"}:
        raise ApplicationConfigurationError("executor must be 'auto', 'python', or 'jax'.")
    loaded_backend, backend_key = _load_backend(backend, backend_options)
    if loaded_backend.info.role is not BackendRole.MM:
        raise ApplicationConfigurationError("Optimization requires an MM backend.")
    if not loaded_backend.info.supports_form(problem.starting_force_field.functional_form.value):
        raise ApplicationConfigurationError(
            f"Backend {backend_key!r} does not support functional form "
            f"{problem.starting_force_field.functional_form.value!r}."
        )

    explicit_optimizer = optimizer is not None
    explicit_workflow = workflow is not None
    overrides: list[str] = []
    if explicit_optimizer:
        overrides.append("optimizer")
    if explicit_workflow:
        overrides.append("workflow")
    if executor != "auto":
        overrides.append("executor")
    if gradient_mode is not None:
        overrides.append("gradient_mode")
    if fd_step != 1e-4:
        overrides.append("fd_step")
    if regularization is not None:
        overrides.append("regularization")

    problem_kind = _problem_kind(problem)
    recipe_id = "explicit-v1"
    ratio_tol: float | None = None
    recipe_optimizer_options: dict[str, object] = {}
    if recipe == "recommended":
        fully_explicit = explicit_optimizer and explicit_workflow
        if not fully_explicit:
            if backend_key != "jax":
                raise ApplicationConfigurationError(
                    "recipe='recommended' is measured only for the built-in JAX backend; "
                    "provide both optimizer and workflow explicitly for other backends."
                )
            if problem_kind is None:
                raise ApplicationConfigurationError(
                    "recipe='recommended' is ambiguous for mixed stationary-point problems; "
                    "provide both optimizer and workflow explicitly."
                )
            recipe_id, recipe_optimizer_options, ratio_tol = _recommended_defaults(problem_kind)
        else:
            recipe_id = "explicit-v1"
    elif optimizer is None or workflow is None:
        raise ApplicationConfigurationError("recipe='explicit' requires both optimizer and workflow.")

    selected_optimizer = optimizer if optimizer is not None else _RECOMMENDED_OPTIMIZER
    selected_workflow = workflow if workflow is not None else "single-stage"
    merged_optimizer_options = dict(recipe_optimizer_options) if optimizer is None else {}
    if optimizer_options:
        merged_optimizer_options.update(optimizer_options)
    optimizer_object, optimizer_config, executor_kind, gradient_mode, fd_step = _resolve_optimizer(
        selected_optimizer,
        merged_optimizer_options,
        executor=executor,
        requested_gradient_mode=gradient_mode,
        requested_fd_step=fd_step,
    )
    workflow_object, workflow_config = _resolve_workflow(selected_workflow, workflow_options)

    if executor_kind == "jax" and backend_key != "jax":
        raise ApplicationConfigurationError(
            f"Analytical per-case JAX execution requires the built-in JAX backend, not {backend_key!r}."
        )
    if executor_kind == "jax":
        executor_config = ResolvedExecutorConfiguration(kind="jax", gradient_mode="analytical")
    else:
        mode = GradientMode(gradient_mode)
        executor_config = ResolvedExecutorConfiguration(
            kind="python",
            gradient_mode=mode.value,
            fd_step=fd_step if mode is GradientMode.FINITE_DIFFERENCE else None,
        )

    effective_regularization = (
        optimizer_spec(selected_optimizer).regularization
        if regularization is None and isinstance(selected_optimizer, (str, OptimizerSpec))
        else 0.0
        if regularization is None
        else float(regularization)
    )
    configuration = ResolvedExecutionConfiguration(
        recipe_id=recipe_id,
        backend=_backend_configuration(loaded_backend, backend_key, backend_options),
        optimizer=optimizer_config,
        workflow=workflow_config,
        executor=executor_config,
        overrides=tuple(overrides),
        regularization=effective_regularization,
        n_evals=n_evals,
        ratio_tol=ratio_tol,
    )
    try:
        result, final_force_field = execute_optimization(
            problem,
            loaded_backend,
            optimizer_object,
            workflow_object,
            executor=executor_kind,
            gradient_mode=GradientMode.ANALYTICAL if executor_kind == "jax" else GradientMode(gradient_mode),
            fd_step=fd_step,
            n_evals=n_evals,
            regularization=effective_regularization,
        )
    except Exception as exc:
        raise ApplicationOptimizationError(f"Optimization failed: {exc}") from exc
    expected_gradient = optimizer_config.expected_result_gradient_mode
    if result.gradient_mode != expected_gradient:
        raise ApplicationOptimizationError(
            f"Optimization result gradient_mode={result.gradient_mode!r}; expected {expected_gradient!r}."
        )
    run_provenance: dict[str, object] = {
        "case_ids": list(problem.case_ids),
        "stationary_points": [case.stationary_point.value for case in problem.cases],
        "n_cases": len(problem.cases),
        "n_observations": len(problem.observations.values),
        "n_active": problem.active_space.n_active,
        "n_parameters": len(problem.layout),
    }
    if problem.preparation_provenance is not None:
        preparation = problem.preparation_provenance
        preparation_payload = {
            "schema_version": preparation.schema_version,
            "profile": preparation.profile,
            "initialize_source": preparation.initialize_source,
            "functional_form": preparation.functional_form,
            "qfuerza_settings": dict(preparation.qfuerza_settings),
            "pre_qfuerza_vector_fingerprint": preparation.pre_qfuerza_vector_fingerprint,
            "parameter_counts": dict(preparation.parameter_counts),
            "stationary_points": list(preparation.stationary_points),
            "case_ids": list(preparation.case_ids),
            "input_fingerprints": dict(preparation.input_fingerprints),
            "observation_recipe": dict(preparation.observation_recipe),
        }
        run_provenance["preparation"] = preparation_payload
        run_provenance["preparation_fingerprint"] = canonical_fingerprint(
            preparation_payload,
            screen_secrets=True,
        )
    if problem.publication_metadata is not None:
        publication_payload = problem.publication_metadata.to_dict()
        run_provenance["publication_metadata"] = publication_payload
        run_provenance["publication_metadata_fingerprint"] = problem.publication_metadata.fingerprint
        run_provenance["objective_profile"] = problem.publication_metadata.objective_profile.identifier
        run_provenance["reproduction_status"] = problem.publication_metadata.status.value
    return OptimizationRun(
        result=result,
        final_force_field=final_force_field,
        configuration=configuration,
        problem_fingerprint=problem_fingerprint(problem),
        layout_fingerprint=problem.layout.fingerprint,
        input_fingerprints=problem_input_fingerprints(problem),
        active_indices=tuple(int(index) for index in problem.active_space.active_indices),
        baseline=problem.active_space.baseline,
        provenance=run_provenance,
    )


__all__ = ["Executor", "Recipe", "execute_optimization", "optimize"]
