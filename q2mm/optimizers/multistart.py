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
from typing import TYPE_CHECKING

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.protocols import _Optimizer
from q2mm.optimizers.scipy_opt import OptimizationResult

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

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

    def optimize(self, objective: ObjectiveFunction, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the inner optimizer from multiple starting points.

        The first start uses the original active parameters. Subsequent
        starts perturb by up to ``±perturbation_pct`` of each active
        parameter value, clipped to bounds if available.  Frozen
        parameters are never perturbed. ``objective.forcefield`` is
        restored to its original value before returning — this method
        has no net side effect on it; materialize the winning force
        field explicitly via
        ``objective.layout.replace(objective.forcefield, result.final_params)``.

        Args:
            objective: Configured objective function.
            space: The active/frozen projection over ``objective.layout``,
                used both to generate perturbed starts and forwarded to
                the inner optimizer for each run.

        Returns:
            OptimizationResult from the best (lowest final score) run,
            with full-vector (length ``space.n_full``) parameters.

        """
        layout = objective.layout
        original_forcefield = objective.forcefield
        if layout is None or original_forcefield is None:
            raise ValueError("MultiStartOptimizer.optimize() requires objective.forcefield and objective.layout.")
        initial_full = layout.vector(original_forcefield)
        x0_active_original = space.pack(initial_full)
        bounds = space.bounds
        rng = np.random.default_rng(self.seed)

        starts = self._generate_starts(x0_active_original, bounds, rng)

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
        # initial_score always corresponds to initial_params=initial_full.
        true_initial_score = objective(initial_full)

        try:
            for i, x_active in enumerate(starts):
                # Reset starting point for each run — reassigns the
                # ObjectiveFunction's forcefield *reference* (a new
                # immutable value), never mutates a ForceField in place.
                full_start = space.expand(x_active, base=initial_full)
                objective.forcefield = layout.replace(original_forcefield, full_start)
                start_space = space.with_baseline(full_start)

                try:
                    result = self.optimizer.optimize(objective, start_space)
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
        finally:
            objective.forcefield = original_forcefield

        if best_result is None:
            raise RuntimeError(f"All {self.n_starts} multi-start runs failed")

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
            initial_params=initial_full,
            final_params=best_result.final_params.copy(),
            history=best_history,
            method=f"multi-start({best_result.method})",
            jac_mode=best_result.jac_mode,
            eps=best_result.eps,
        )

    def _generate_starts(
        self,
        x0: np.ndarray,
        bounds: np.ndarray | list[tuple[float, float]] | None,
        rng: np.random.Generator,
    ) -> list[np.ndarray]:
        """Generate starting points: original + (n_starts-1) perturbed."""
        starts = [x0.copy()]
        bounds_arr = None if bounds is None else np.asarray(bounds, dtype=float)
        for _ in range(self.n_starts - 1):
            # Perturbation scale based on absolute parameter value
            scale = np.maximum(np.abs(x0) * self.perturbation_pct, 1e-6)
            perturbation = rng.uniform(-scale, scale)
            x_new = x0 + perturbation
            if bounds_arr is not None and bounds_arr.size > 0:
                lower = bounds_arr[:, 0]
                upper = bounds_arr[:, 1]
                x_new = np.clip(x_new, lower, upper)
            starts.append(x_new)
        return starts
