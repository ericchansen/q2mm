"""Multi-start optimizer backed by jaxopt, one replica at a time.

Unlike the previous implementation, replicas are **not** fused into a
single ``jax.vmap`` over an all-molecule loss kernel.  Each deterministic
candidate is dispatched independently through the
:class:`~q2mm.objectives.jax.JaxObjectiveExecutor` (per-case JIT + Python
aggregation), preserving the per-case split and recording every candidate
(success or failure) as a :class:`~q2mm.models.results.CandidateRecord`.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np

from q2mm.models.results import CandidateRecord, OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.jaxopt_opt import _METHOD_REGISTRY, JaxOptOptimizer, _require_jax_executor

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)


class JaxMultiStartOptimizer:
    """Multi-start optimizer that dispatches jaxopt replicas independently."""

    def __init__(
        self,
        *,
        method: str = "lbfgs",
        n_starts: int = 10,
        maxiter: int = 200,
        tol: float = 1e-6,
        perturbation_pct: float = 0.1,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        if method not in _METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(sorted(_METHOD_REGISTRY))}")
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")
        if perturbation_pct < 0:
            raise ValueError("perturbation_pct must be >= 0")
        self.method = method
        self.n_starts = n_starts
        self.maxiter = maxiter
        self.tol = tol
        self.perturbation_pct = perturbation_pct
        self.seed = seed
        self.verbose = verbose

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run ``n_starts`` jaxopt replicas independently; keep the best."""
        _require_jax_executor(evaluator)

        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        initial_full = np.array(space.baseline, dtype=float)
        x0 = space.pack(initial_full)
        starts = self._generate_starts(x0, space.bounds)

        inner = JaxOptOptimizer(method=self.method, maxiter=self.maxiter, tol=self.tol, verbose=False)
        n_eval_before = evaluator.n_evaluations
        true_initial_score = float(evaluator.value(initial_full))

        best_converged: OptimizationResult | None = None
        best_any: OptimizationResult | None = None
        candidates: list[CandidateRecord] = []
        n_converged = 0
        n_failed = 0
        method_str = f"jaxopt-multi:{self.method}"

        if self.verbose:
            logger.info(
                "Starting %s: n_starts=%d, maxiter=%d, initial score %.6f",
                method_str,
                self.n_starts,
                self.maxiter,
                true_initial_score,
            )

        for i, x_active in enumerate(starts):
            full_start = space.expand(x_active, base=initial_full)
            start_space = space.with_baseline(full_start)
            try:
                result = inner.optimize(evaluator, start_space)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                logger.warning("  Replica %d/%d failed: %s", i + 1, self.n_starts, exc)
                candidates.append(
                    CandidateRecord(
                        index=i,
                        status="failure",
                        n_params=n_params,
                        layout_fingerprint=fingerprint,
                        initial_params=full_start,
                        final_params=full_start,
                        final_score=float("inf"),
                        message=f"{type(exc).__name__}: {exc}",
                        seed=self.seed,
                    )
                )
                continue
            converged = bool(result.success)
            if converged:
                n_converged += 1
            else:
                n_failed += 1
            candidates.append(
                CandidateRecord(
                    index=i,
                    status="success" if converged else "failure",
                    n_params=n_params,
                    layout_fingerprint=fingerprint,
                    initial_params=full_start,
                    final_params=result.final_params,
                    initial_score=result.initial_score,
                    final_score=result.final_score,
                    message=result.message,
                    seed=self.seed,
                )
            )
            if best_any is None or result.final_score < best_any.final_score:
                best_any = result
            if converged and (best_converged is None or result.final_score < best_converged.final_score):
                best_converged = result

        candidates_t = tuple(candidates)
        selected = best_converged if best_converged is not None else best_any
        if selected is None:
            return OptimizationResult(
                success=False,
                message=f"jaxopt-multi: all {self.n_starts} replicas failed",
                initial_score=true_initial_score,
                final_score=float("inf"),
                n_iterations=0,
                n_evaluations=evaluator.n_evaluations - n_eval_before,
                n_params=n_params,
                layout_fingerprint=fingerprint,
                initial_params=initial_full,
                final_params=initial_full,
                history=(true_initial_score,),
                method=method_str,
                gradient_mode="analytical",
                candidates=candidates_t,
            )

        overall_success = best_converged is not None
        if overall_success:
            message = f"jaxopt-multi best of {n_converged}/{self.n_starts} converged: {selected.message}"
        else:
            message = (
                f"jaxopt-multi: no converged replica ({n_failed}/{self.n_starts} failed); "
                f"best nonconverged score {selected.final_score:.6g}"
            )
        if self.verbose:
            logger.info("%s best: %.6f (%d converged)", method_str, selected.final_score, n_converged)

        return replace(
            selected,
            success=overall_success,
            message=message,
            initial_score=true_initial_score,
            initial_params=initial_full,
            n_evaluations=evaluator.n_evaluations - n_eval_before,
            method=method_str,
            candidates=candidates_t,
        )

    def _generate_starts(self, x0: np.ndarray, bounds: np.ndarray | None) -> list[np.ndarray]:
        rng = np.random.default_rng(self.seed)
        starts = [x0.copy()]
        bounds_arr = None if bounds is None else np.asarray(bounds, dtype=float)
        for _ in range(self.n_starts - 1):
            scale = np.maximum(np.abs(x0) * self.perturbation_pct, 1e-6)
            x_new = x0 + rng.uniform(-scale, scale)
            if bounds_arr is not None and bounds_arr.size > 0:
                x_new = np.clip(x_new, bounds_arr[:, 0], bounds_arr[:, 1])
            starts.append(x_new)
        return starts
