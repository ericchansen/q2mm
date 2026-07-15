"""The standard single-pass Q2MM workflow.

Wraps the canonical pattern used throughout the codebase — build an
``ObjectiveFunction`` from a loaded ``OptimizationProblem``, run one
optimizer pass, sample the real objective at the endpoints for noise
quantification — in the :class:`~q2mm.workflows.base.Workflow` Protocol
so it composes with multi-stage protocols (Method E2) and can be
swapped without changing the call site.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.workflows.base import StageResult, WorkflowResult

if TYPE_CHECKING:
    from q2mm.models.problem import OptimizationProblem
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.workflows.base import _Optimizer


def _r2(ref: np.ndarray, calc: np.ndarray) -> float:
    """Q2MM coefficient of determination; NaN when undefined.

    Mirrors :func:`q2mm.benchmark_runner._r2` so this workflow produces
    the same numbers as the canonical convergence-runner module.
    """
    if ref.size < 2:
        return float("nan")
    ss_res = float(np.sum((ref - calc) ** 2))
    mean = float(np.mean(ref))
    ss_tot = float(np.sum((ref - mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _category_stats(ref_values: np.ndarray, calc_values: np.ndarray) -> dict[str, float]:
    """Per-category n / R² / RMSD / MAE."""
    n = int(ref_values.size)
    if n == 0:
        return {"n_refs": 0, "r2": float("nan"), "rmsd": float("nan"), "mae": float("nan")}
    residuals = ref_values - calc_values
    return {
        "n_refs": n,
        "r2": _r2(ref_values, calc_values),
        "rmsd": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def _per_category_metrics(obj: ObjectiveFunction, ff: Any) -> dict[str, dict[str, float]]:
    """Bucket residuals by reference kind, undo weights, compute stats.

    ``ObjectiveFunction._compute_residuals(ff)`` returns
    ``w_i * (ref - calc)`` for every reference value in order.  We
    invert the weight to recover ``calc``, group by ``ref.kind``,
    and apply :func:`_category_stats` per bucket.  References with
    ``weight == 0.0`` are skipped (e.g. the imaginary mode in TS
    eigenmatrix fits).
    """
    assert obj.layout is not None  # diagnostics helper is only called with a layout-backed objective
    residuals = obj._compute_residuals(obj.layout.vector(ff))  # noqa: SLF001 — diagnostics-only API
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ref, weighted in zip(obj.reference.values, residuals, strict=True):
        if ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        buckets[ref.kind].append((float(ref.value), calc_value))
    return {
        kind: _category_stats(
            np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]),
        )
        for kind, pairs in buckets.items()
    }


def _evaluate_samples(obj: ObjectiveFunction, params: np.ndarray, n_evals: int) -> list[float]:
    """Sample the real objective ``n_evals`` times at the given parameters.

    Quantifies per-call engine non-determinism (geometry-relaxation
    seeding, JIT compile order, etc.).  ``ObjectiveFunction.__call__``
    appends to ``obj.history`` and increments ``obj.n_eval``; this
    helper restores both after each call so that the optimizer's
    bookkeeping is not polluted by post-hoc sampling.  Truncates
    ``obj.history`` rather than copying it — O(1) vs O(len(history))
    per sample, which matters when the optimizer has accumulated
    thousands of evaluations.  Mirrors
    :func:`q2mm.benchmark_runner._evaluate_objective_samples`.
    """
    n_eval_before = obj.n_eval
    history_len_before = len(obj.history)
    scores: list[float] = []
    for _ in range(n_evals):
        try:
            score = float(obj(params))
        finally:
            obj.n_eval = n_eval_before
            del obj.history[history_len_before:]
        scores.append(score)
    return scores


class SingleStageWorkflow:
    """One optimization pass against the problem's ``ObjectiveFunction``.

    Equivalent to::

        obj = ObjectiveFunction(problem.starting_force_field, backend,
                                list(problem.molecules), problem.observations,
                                case_ids=list(problem.case_ids), layout=problem.layout)
        result = optimizer.optimize(obj, problem.active_space)
        final_ff = problem.layout.replace(problem.starting_force_field, result.final_params)
        # ... noise-floor sampling, per-category metrics ...

    Use this as the default workflow.  For multi-stage TSFF
    parameterization with negative-FC handling, use
    ``MethodE2Workflow``.
    """

    name: str = "single-stage"

    def run(
        self,
        problem: OptimizationProblem,
        backend: Any,  # noqa: ANN401
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> WorkflowResult:
        """Execute one optimizer pass; return a fully-populated WorkflowResult.

        Args:
            problem: Loaded optimization problem.  ``problem.starting_force_field``
                is the starting point and — being immutable — is never
                mutated; it is also returned as ``WorkflowResult.initial_ff``.
            backend: MM backend.
            optimizer: Pre-configured optimizer (e.g. ``ScipyOptimizer``).
            n_evals: Real-objective samples at initial and final params.
                ``0`` skips sampling.

        Returns:
            :class:`WorkflowResult` with one :class:`StageResult` named
            ``"optimize"`` and per-category metrics on the optimized FF.

        """
        from q2mm.optimizers.objective import ObjectiveFunction

        initial_ff = problem.starting_force_field
        obj = ObjectiveFunction(
            initial_ff,
            backend,
            list(problem.molecules),
            problem.observations,
            case_ids=list(problem.case_ids),
            layout=problem.layout,
        )

        t0 = time.perf_counter()
        opt_result = optimizer.optimize(obj, problem.active_space)
        elapsed = time.perf_counter() - t0

        final_ff = problem.layout.replace(initial_ff, opt_result.final_params)

        initial_samples = _evaluate_samples(obj, opt_result.initial_params, n_evals)
        final_samples = _evaluate_samples(obj, opt_result.final_params, n_evals)
        optimized_categories = _per_category_metrics(obj, final_ff)

        stage = StageResult(
            name="optimize",
            initial_score=float(opt_result.initial_score),
            final_score=float(opt_result.final_score),
            n_iterations=int(opt_result.n_iterations),
            n_evaluations=int(opt_result.n_evaluations),
            converged=bool(opt_result.success),
            message=str(opt_result.message),
            jac_mode=str(opt_result.jac_mode) if opt_result.jac_mode is not None else "unknown",
            elapsed_s=elapsed,
        )

        return WorkflowResult(
            workflow_name=self.name,
            final_ff=final_ff,
            initial_ff=initial_ff,
            stages=[stage],
            initial_obj_samples=initial_samples,
            final_obj_samples=final_samples,
            optimized_categories=optimized_categories,
        )
