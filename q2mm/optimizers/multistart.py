"""Multi-start meta-optimizer for force field parameterization.

Runs an inner optimizer from multiple perturbed starting points and
keeps the best result.  This is the simplest global strategy — it
doesn't require stochastic hops or temperature schedules, just brute
force from diverse initial conditions.

Any optimizer with an ``optimize(objective) -> OptimizationResult``
interface can be wrapped: :class:`ScipyOptimizer`,
:class:`OptaxOptimizer`, or :class:`BasinHoppingOptimizer`.
"""

from __future__ import annotations

import logging

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.protocols import _Optimizer
from q2mm.optimizers.scipy_opt import OptimizationResult

logger = logging.getLogger(__name__)


class MultiStartOptimizer:
    """Meta-optimizer: run from N perturbed starts, keep the best.

    Args:
        optimizer: Inner optimizer instance (ScipyOptimizer,
            OptaxOptimizer, BasinHoppingOptimizer, etc.).
        n_starts: Number of starting points to try.
        perturbation_pct: Maximum perturbation as a fraction of each
            parameter's value.  E.g. ``0.1`` perturbs by ±10%.
        seed: Random seed for reproducible perturbations.
        verbose: Log progress during optimization.

    """

    def __init__(
        self,
        optimizer: _Optimizer,
        n_starts: int = 5,
        perturbation_pct: float = 0.1,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.optimizer = optimizer
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")
        if perturbation_pct < 0:
            raise ValueError("perturbation_pct must be >= 0")
        self.n_starts = n_starts
        self.perturbation_pct = perturbation_pct
        self.seed = seed
        self.verbose = verbose

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the inner optimizer from multiple starting points.

        The first start uses the original parameters.  Subsequent starts
        perturb by up to ``±perturbation_pct`` of each parameter value,
        clipped to bounds if available.

        Args:
            objective: Configured objective function.

        Returns:
            OptimizationResult from the best (lowest final score) run.

        """
        x0_original = objective.forcefield.get_param_vector().copy()
        bounds = objective.forcefield.get_bounds()
        rng = np.random.default_rng(self.seed)

        starts = self._generate_starts(x0_original, bounds, rng)

        if self.verbose:
            logger.info(
                "Multi-start: %d starts, perturbation ±%.0f%%",
                self.n_starts,
                self.perturbation_pct * 100,
            )

        n_eval_before = objective.n_eval
        best_result: OptimizationResult | None = None
        best_history: list[float] = []
        all_scores: list[float] = []
        n_failed = 0
        # Evaluate once at the original (unperturbed) parameters so
        # initial_score always corresponds to initial_params=x0_original.
        objective.forcefield.set_param_vector(x0_original)
        true_initial_score = objective(x0_original)

        for i, x0 in enumerate(starts):
            # Reset starting point for each run
            objective.forcefield.set_param_vector(x0)

            try:
                result = self.optimizer.optimize(objective)
            except Exception:
                n_failed += 1
                logger.warning("  Start %d/%d failed, skipping", i + 1, self.n_starts)
                continue

            all_scores.append(result.final_score)

            if self.verbose:
                logger.info(
                    "  Start %d/%d: %.6f → %.6f",
                    i + 1,
                    self.n_starts,
                    result.initial_score,
                    result.final_score,
                )

            if best_result is None or result.final_score < best_result.final_score:
                best_result = result
                best_history = list(result.history)

        if best_result is None:
            raise RuntimeError(f"All {self.n_starts} multi-start runs failed")

        # Apply best parameters
        objective.forcefield.set_param_vector(best_result.final_params)

        total_evals = objective.n_eval - n_eval_before

        if self.verbose:
            logger.info(
                "Multi-start best: %.6f (scores: %s, %d failed)",
                best_result.final_score,
                ", ".join(f"{s:.1f}" for s in all_scores),
                n_failed,
            )

        # Wrap the result with multi-start metadata
        return OptimizationResult(
            success=best_result.success,
            message=f"multi-start best of {len(all_scores)}/{self.n_starts}: {best_result.message}",
            initial_score=true_initial_score,
            final_score=best_result.final_score,
            n_iterations=best_result.n_iterations,
            n_evaluations=total_evals,
            initial_params=x0_original,
            final_params=best_result.final_params.copy(),
            history=best_history,
            method=f"multi-start({best_result.method})",
            jac_mode=best_result.jac_mode,
            eps=best_result.eps,
        )

    def _generate_starts(
        self,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Generate starting points: original + (n_starts-1) perturbed."""
        starts = [x0.copy()]
        for _ in range(self.n_starts - 1):
            # Perturbation scale based on absolute parameter value
            scale = np.maximum(np.abs(x0) * self.perturbation_pct, 1e-6)
            perturbation = rng.uniform(-scale, scale)
            x_new = x0 + perturbation
            if bounds is not None:
                lower = np.array([b[0] for b in bounds])
                upper = np.array([b[1] for b in bounds])
                x_new = np.clip(x_new, lower, upper)
            starts.append(x_new)
        return starts
