"""The standard single-pass Q2MM workflow.

Compiles an :class:`~q2mm.objectives.plan.ObjectivePlan` from the problem,
builds an executor via the injected factory, runs one optimizer pass, and
samples the real objective at the endpoints for noise quantification —
returning the one canonical :class:`~q2mm.models.results.OptimizationResult`
with a single :class:`~q2mm.models.results.StageRecord`.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from q2mm.models.results import StageRecord
from q2mm.objectives.metrics import category_metrics, evaluate_samples
from q2mm.objectives.plan import ObjectivePlan

if TYPE_CHECKING:
    from collections.abc import Callable

    from q2mm.models.problem import OptimizationProblem
    from q2mm.models.results import OptimizationResult
    from q2mm.objectives.protocols import ObjectiveEvaluator
    from q2mm.optimizers.protocols import _Optimizer


class SingleStageWorkflow:
    """One optimization pass against the problem's objective."""

    name: str = "single-stage"

    def run(
        self,
        problem: OptimizationProblem,
        make_evaluator: Callable[[ObjectivePlan], ObjectiveEvaluator],
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> OptimizationResult:
        """Execute one optimizer pass; return the canonical result."""
        plan = ObjectivePlan.from_problem(problem)
        evaluator = make_evaluator(plan)

        t0 = time.perf_counter()
        opt_result = optimizer.optimize(evaluator, problem.active_space)
        elapsed = time.perf_counter() - t0

        initial_samples = evaluate_samples(evaluator, opt_result.initial_params, n_evals)
        final_samples = evaluate_samples(evaluator, opt_result.final_params, n_evals)
        categories = category_metrics(plan, evaluator.evaluate(opt_result.final_params))

        stage = StageRecord(
            name="optimize",
            n_params=problem.active_space.n_full,
            layout_fingerprint=problem.active_space.layout.fingerprint,
            initial_score=float(opt_result.initial_score),
            final_score=float(opt_result.final_score),
            n_iterations=int(opt_result.n_iterations),
            n_evaluations=int(opt_result.n_evaluations),
            converged=bool(opt_result.success),
            message=str(opt_result.message),
            gradient_mode=opt_result.gradient_mode,
            fd_step=opt_result.fd_step,
            elapsed_s=elapsed,
        )

        return replace(
            opt_result,
            method=self.name,
            stages=(stage,),
            initial_samples=tuple(initial_samples),
            final_samples=tuple(final_samples),
            category_metrics=categories,
        )
