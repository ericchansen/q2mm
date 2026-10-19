"""The canonical numerical optimization boundary, shared by SDK and benchmarks."""

from dataclasses import replace

import numpy as np

from q2mm.models.problem import OptimizationProblem
from q2mm.models.results import OptimizationResult, _validate_executor_gradient
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.protocols import _Optimizer
from q2mm.settings import (
    BasinHoppingSettings,
    CyclingSettings,
    MultiStartSettings,
    ScipySettings,
)
from q2mm.workflows.base import EvaluatorFactory, Workflow

from .models import (
    ApplicationConfigurationError,
    ApplicationOptimizationError,
    _validate_execution_controls,
    _validate_result_vectors,
)


def _result_gradient_mode(target: object, executor_gradient: str) -> str:
    """Resolve declared typed algorithm semantics, including configured subclasses."""
    settings = getattr(target, "settings", target)
    if isinstance(settings, MultiStartSettings):
        return _result_gradient_mode(getattr(target, "optimizer", settings.optimizer), executor_gradient)
    if isinstance(settings, ScipySettings):
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if settings.method in ScipyOptimizer.DERIVATIVE_FREE_METHODS:
            return "none"
        if settings.method == "least_squares" or executor_gradient == "none":
            return "finite_difference"
        return executor_gradient
    if isinstance(settings, CyclingSettings):
        return "none" if settings.max_cycles == 0 else _result_gradient_mode(settings.full_optimizer, executor_gradient)
    if isinstance(settings, BasinHoppingSettings):
        return "finite_difference" if executor_gradient == "none" else executor_gradient
    return executor_gradient


def optimize(
    problem: OptimizationProblem,
    *,
    optimizer: _Optimizer,
    workflow: Workflow,
    make_evaluator: EvaluatorFactory,
    regularization: float,
    n_evals: int,
) -> OptimizationResult:
    """Execute configured components, validating the full-layout result.

    This function does not select policies, merge overrides, construct named
    solvers, assemble run records, or persist. Interrupts propagate unchanged.
    """
    if not isinstance(problem, OptimizationProblem):
        raise ApplicationConfigurationError("optimize requires an OptimizationProblem.")
    _validate_execution_controls(regularization, n_evals)
    executor_controls: tuple[str, float | None] | None = None

    def checked_factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
        nonlocal executor_controls
        if plan.regularization != regularization:
            raise ApplicationOptimizationError("Workflow objective plan regularization does not match execution.")
        evaluator = make_evaluator(plan)
        if evaluator.plan.regularization != regularization:
            raise ApplicationOptimizationError("Evaluator objective plan regularization does not match execution.")
        controls = (evaluator.gradient_mode.value, evaluator.finite_difference_step)
        _validate_executor_gradient(*controls)
        if executor_controls is not None and controls != executor_controls:
            raise ApplicationOptimizationError("Workflow evaluators disagree on executor gradient mode or fd_step.")
        executor_controls = controls
        return evaluator

    try:
        result = workflow.run(
            problem,
            checked_factory,
            optimizer,
            n_evals=n_evals,
            regularization=regularization,
        )
    except Exception as exc:
        raise ApplicationOptimizationError(f"Optimization failed: {exc}") from exc
    if not isinstance(result, OptimizationResult):
        raise ApplicationOptimizationError(f"Workflow returned {type(result).__name__}; expected OptimizationResult.")
    if result.layout_fingerprint != problem.layout.fingerprint or result.n_params != len(problem.layout):
        raise ApplicationOptimizationError("Optimization result does not match the problem parameter layout.")
    _validate_result_vectors(result, np.asarray(problem.active_space.baseline), problem.active_space.active_indices)
    if executor_controls is not None:
        expected_gradient = _result_gradient_mode(optimizer, executor_controls[0])
        if result.gradient_mode != expected_gradient:
            raise ApplicationOptimizationError(
                f"Optimization result gradient_mode={result.gradient_mode!r}; expected {expected_gradient!r}."
            )
    from q2mm.optimizers.scipy_opt import ScipyOptimizer, _is_feasible
    from q2mm.workflows.single_stage import SingleStageWorkflow

    if isinstance(optimizer, ScipyOptimizer) and isinstance(workflow, SingleStageWorkflow):
        space = problem.active_space
        bounds = optimizer._resolve_bounds(space, space.pack(space.baseline))
        if not _is_feasible(space.pack(result.final_params), bounds):
            raise ApplicationOptimizationError("Optimization result violates the effective SciPy active bounds.")
    if len(result.initial_samples) != n_evals or len(result.final_samples) != n_evals:
        raise ApplicationOptimizationError("Optimization result endpoint sample counts do not match execution n_evals.")
    if executor_controls is None:
        raise ApplicationOptimizationError("Workflow did not create an evaluator to verify executor controls.")
    return replace(
        result,
        regularization=float(regularization),
        n_evals=n_evals,
        executor_gradient_mode=executor_controls[0],
        executor_fd_step=executor_controls[1],
    )


__all__ = ["optimize"]
