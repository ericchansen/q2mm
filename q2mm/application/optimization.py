"""The canonical numerical optimization boundary, shared by SDK and benchmarks."""

from dataclasses import replace

import numpy as np

from q2mm.models.problem import OptimizationProblem
from q2mm.models.results import OptimizationResult
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.protocols import _Optimizer
from q2mm.settings import ScipySettings, MultiStartSettings
from q2mm.workflows.base import EvaluatorFactory, Workflow

from .models import ApplicationConfigurationError, ApplicationOptimizationError


def _result_gradient_mode(optimizer: _Optimizer, executor_gradient: str) -> str:
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    def scipy_mode(settings: ScipySettings) -> str:
        if settings.method in ScipyOptimizer.DERIVATIVE_FREE_METHODS:
            return "none"
        if settings.method == "least_squares" or executor_gradient == "none":
            return "finite_difference"
        return executor_gradient

    if type(optimizer) is ScipyOptimizer:
        return scipy_mode(optimizer.settings)
    from q2mm.optimizers.cycling import CyclingOptimizer
    from q2mm.optimizers.multistart import MultiStartOptimizer
    from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

    if type(optimizer) is MultiStartOptimizer:
        return _result_gradient_mode(optimizer.optimizer, executor_gradient)
    if type(optimizer) is CyclingOptimizer:
        if optimizer.settings.max_cycles == 0:
            return "none"
        phase = optimizer.settings.full_optimizer
        if isinstance(phase, MultiStartSettings):
            phase = phase.optimizer
        if isinstance(phase, ScipySettings):
            return scipy_mode(phase)
        return "finite_difference" if executor_gradient == "none" else executor_gradient
    if type(optimizer) is BasinHoppingOptimizer:
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
    if isinstance(regularization, bool) or not np.isfinite(regularization) or regularization < 0:
        raise ApplicationConfigurationError("regularization must be non-negative and finite.")
    if isinstance(n_evals, bool) or not isinstance(n_evals, int) or n_evals < 0:
        raise ApplicationConfigurationError("n_evals must be a non-negative integer.")
    gradient_modes: list[str] = []

    def checked_factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
        if plan.regularization != regularization:
            raise ApplicationOptimizationError("Workflow objective plan regularization does not match execution.")
        evaluator = make_evaluator(plan)
        if evaluator.plan.regularization != regularization:
            raise ApplicationOptimizationError("Evaluator objective plan regularization does not match execution.")
        gradient_modes.append(_result_gradient_mode(optimizer, evaluator.gradient_mode.value))
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
    inactive = np.setdiff1d(np.arange(problem.active_space.n_full), problem.active_space.active_indices)
    baseline = np.asarray(problem.active_space.baseline)
    if not np.array_equal(result.initial_params, baseline):
        raise ApplicationOptimizationError("Optimization result initial vector does not match the problem baseline.")
    if not np.array_equal(result.final_params[inactive], baseline[inactive]):
        raise ApplicationOptimizationError("Optimization result changed frozen parameter slots.")
    if gradient_modes and result.gradient_mode != gradient_modes[-1]:
        raise ApplicationOptimizationError(
            f"Optimization result gradient_mode={result.gradient_mode!r}; expected {gradient_modes[-1]!r}."
        )
    from q2mm.optimizers.scipy_opt import ScipyOptimizer, _is_feasible
    from q2mm.workflows.single_stage import SingleStageWorkflow

    if isinstance(optimizer, ScipyOptimizer) and isinstance(workflow, SingleStageWorkflow):
        space = problem.active_space
        bounds = optimizer._resolve_bounds(space, space.pack(baseline))
        if not _is_feasible(space.pack(result.final_params), bounds):
            raise ApplicationOptimizationError("Optimization result violates the effective SciPy active bounds.")
    if len(result.initial_samples) != n_evals or len(result.final_samples) != n_evals:
        raise ApplicationOptimizationError("Optimization result endpoint sample counts do not match execution n_evals.")
    return replace(result, regularization=float(regularization), n_evals=n_evals)


__all__ = ["optimize"]
