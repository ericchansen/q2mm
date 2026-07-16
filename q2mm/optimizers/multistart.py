"""Multi-start meta-optimizer for force field parameterization.

Runs an inner optimizer from multiple deterministically-perturbed starting
points and keeps the best result.  Every candidate — successful or failed —
is recorded as a :class:`~q2mm.models.results.CandidateRecord` carrying its
own full-length generated start and final vectors, so no start is silently
dropped and no records are lost when a start fails.  If **every** candidate
fails, a canonical ``OptimizationResult(success=False, ...)`` is returned
(never an exception), preserving all candidate records.

The inner optimizer is driven with the same
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator`; each start is a
distinct rebased :class:`~q2mm.models.parameters.ActiveParameterSpace`.  No
force field is mutated — the evaluator operates on full vectors only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from q2mm.models.results import CandidateRecord, OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.protocols import _Optimizer

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)


class MultiStartOptimizer:
    """Meta-optimizer: run from N deterministic starts, keep the best."""

    def __init__(
        self,
        optimizer: _Optimizer,
        n_starts: int = 5,
        perturbation_pct: float = 0.1,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")
        if perturbation_pct < 0:
            raise ValueError("perturbation_pct must be >= 0")
        self.optimizer = optimizer
        self.n_starts = n_starts
        self.perturbation_pct = perturbation_pct
        self.seed = seed
        self.verbose = verbose

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the inner optimizer from multiple starts; keep the best."""
        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        baseline = np.array(space.baseline, dtype=float)
        x0_active = space.pack(baseline)
        rng = np.random.default_rng(self.seed)
        starts = self._generate_starts(x0_active, space.bounds, rng)

        if self.verbose:
            logger.info("Multi-start: %d starts, perturbation ±%.0f%%", self.n_starts, self.perturbation_pct * 100)

        n_eval_before = evaluator.n_evaluations
        true_initial_score = float(evaluator.value(baseline))

        best_converged: OptimizationResult | None = None
        best_any: OptimizationResult | None = None
        candidates: list[CandidateRecord] = []
        n_converged = 0
        n_failed = 0

        for i, x_active in enumerate(starts):
            full_start = space.expand(x_active, base=baseline)
            start_space = space.with_baseline(full_start)
            try:
                result = self.optimizer.optimize(evaluator, start_space)
            except Exception as exc:  # noqa: BLE001
                n_failed += 1
                logger.warning("  Start %d/%d failed: %s", i + 1, self.n_starts, exc)
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

            # A start that ran but did not converge is a failed candidate; it
            # keeps its finite score and stays eligible as a fallback best only
            # when no converged run exists (final success then reflects that).
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
            if self.verbose:
                logger.info(
                    "  Start %d/%d: %.6f → %.6f (%s)",
                    i + 1,
                    self.n_starts,
                    result.initial_score,
                    result.final_score,
                    "converged" if converged else "nonconverged",
                )
            if best_any is None or result.final_score < best_any.final_score:
                best_any = result
            if converged and (best_converged is None or result.final_score < best_converged.final_score):
                best_converged = result

        total_evals = evaluator.n_evaluations - n_eval_before
        candidates_t = tuple(candidates)
        selected = best_converged if best_converged is not None else best_any

        if selected is None:
            # Every start raised: return a canonical failed result that
            # preserves every candidate record rather than raising.
            return OptimizationResult(
                success=False,
                message=f"multi-start: all {self.n_starts} runs failed",
                initial_score=true_initial_score,
                final_score=float("inf"),
                n_iterations=0,
                n_evaluations=total_evals,
                n_params=n_params,
                layout_fingerprint=fingerprint,
                initial_params=baseline,
                final_params=baseline,
                history=(true_initial_score,),
                method="multi-start",
                gradient_mode="none",
                candidates=candidates_t,
            )

        overall_success = best_converged is not None
        if overall_success:
            message = f"multi-start best of {n_converged}/{self.n_starts} converged: {selected.message}"
        else:
            message = (
                f"multi-start: no converged run ({n_failed}/{self.n_starts} failed); "
                f"best nonconverged score {selected.final_score:.6g}"
            )
        if self.verbose:
            logger.info(
                "Multi-start best: %.6f (%d converged, %d failed)",
                selected.final_score,
                n_converged,
                n_failed,
            )

        return OptimizationResult(
            success=overall_success,
            message=message,
            initial_score=true_initial_score,
            final_score=selected.final_score,
            n_iterations=selected.n_iterations,
            n_evaluations=total_evals,
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=baseline,
            final_params=selected.final_params,
            history=selected.history,
            method=f"multi-start({selected.method})",
            gradient_mode=selected.gradient_mode,
            fd_step=selected.fd_step,
            candidates=candidates_t,
        )

    def _generate_starts(
        self,
        x0: np.ndarray,
        bounds: np.ndarray | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        starts = [x0.copy()]
        bounds_arr = None if bounds is None else np.asarray(bounds, dtype=float)
        for _ in range(self.n_starts - 1):
            scale = np.maximum(np.abs(x0) * self.perturbation_pct, 1e-6)
            perturbation = rng.uniform(-scale, scale)
            x_new = x0 + perturbation
            if bounds_arr is not None and bounds_arr.size > 0:
                x_new = np.clip(x_new, bounds_arr[:, 0], bounds_arr[:, 1])
            starts.append(x_new)
        return starts
