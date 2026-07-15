"""Basin-hopping global optimizer for force field parameterization.

Wraps :func:`scipy.optimize.basinhopping` with a local L-BFGS-B minimizer
(using analytical gradients when available) and bounded perturbation steps.
Basin-hopping combines stochastic global search with gradient-based local
exploitation — it "hops" between basins by randomly perturbing parameters,
then locally optimizes each landing point.

This is particularly effective for the non-convex MM3 landscapes where
L-BFGS-B alone gets trapped in local minima.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import OptimizationResult

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)


class _BoundedStep:
    """Random perturbation step that respects parameter bounds.

    Each hop perturbs the current parameter vector by
    ``uniform(-stepsize, +stepsize)`` and clips to bounds.
    """

    def __init__(
        self,
        stepsize: float,
        bounds: list[tuple[float, float]] | None,
        rng: np.random.Generator,
    ) -> None:
        self.stepsize = stepsize
        self.bounds = bounds
        self.rng = rng

    def __call__(self, x: np.ndarray) -> np.ndarray:
        perturbation = self.rng.uniform(-self.stepsize, self.stepsize, size=x.shape)
        x_new = x + perturbation
        if self.bounds is not None:
            lower = np.array([b[0] for b in self.bounds])
            upper = np.array([b[1] for b in self.bounds])
            x_new = np.clip(x_new, lower, upper)
        return x_new


class BasinHoppingOptimizer:
    """Global optimizer using basin-hopping with L-BFGS-B local steps.

    Args:
        niter: Number of basin-hopping iterations (hops).
        T: Temperature parameter for the Metropolis acceptance criterion.
            Higher values accept more uphill moves.
        stepsize: Magnitude of random perturbation at each hop.
        local_method: Local minimizer method (default ``'L-BFGS-B'``).
        local_maxiter: Maximum iterations per local minimization.
        jac: Jacobian strategy for local minimizer.  ``'auto'`` probes
            the engine for analytical gradient support.
        seed: Random seed for reproducible basin-hopping runs.
        verbose: Log progress during optimization.

    """

    def __init__(
        self,
        niter: int = 50,
        T: float = 1.0,
        stepsize: float = 0.5,
        local_method: str = "L-BFGS-B",
        local_maxiter: int = 200,
        jac: str | None = "auto",
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.local_method = local_method
        self.local_maxiter = local_maxiter
        self.jac = jac
        self.seed = seed
        self.verbose = verbose

    def optimize(self, objective: ObjectiveFunction, space: ActiveParameterSpace) -> OptimizationResult:
        """Run basin-hopping optimization.

        Args:
            objective: Configured objective with forcefield, backend,
                molecules, and reference data.
            space: The active/frozen projection over ``objective.layout``.
                Only active parameters are perturbed/optimized;
                ``objective.forcefield`` is never mutated — materialize
                the optimized force field explicitly via
                ``objective.layout.replace(objective.forcefield, result.final_params)``.

        Returns:
            OptimizationResult with the globally best full-vector
            parameters found (length ``space.n_full``).

        """
        from scipy.optimize import basinhopping

        if objective.forcefield is None or objective.layout is None:
            raise ValueError("BasinHoppingOptimizer.optimize() requires objective.forcefield and objective.layout.")

        objective.history.clear()
        n_eval_before = objective.n_eval

        initial_full = objective.layout.vector(objective.forcefield)
        x0 = space.pack(initial_full)
        initial_score = objective(initial_full)

        def wrapped_objective(x_active: np.ndarray) -> float:
            return objective(space.expand(x_active))

        bounds = space.bounds.tolist()
        rng = np.random.default_rng(self.seed)

        # Resolve Jacobian for the local minimizer
        jac_fn = self._resolve_jac(objective, space)

        # Local minimizer kwargs
        options: dict[str, Any] = {"maxiter": self.local_maxiter}
        if jac_fn is None:
            # Match ScipyOptimizer: SciPy default FD step is too small for
            # force-field parameters.
            options["eps"] = 1e-3

        minimizer_kwargs: dict[str, Any] = {
            "method": self.local_method,
            "jac": jac_fn,
            "options": options,
        }
        if bounds and self.local_method in ("L-BFGS-B", "trust-constr", "SLSQP"):
            minimizer_kwargs["bounds"] = bounds

        take_step = _BoundedStep(self.stepsize, bounds, rng)

        if self.verbose:
            logger.info(
                "Basin-hopping: %d hops, T=%.2f, stepsize=%.2f, local=%s (maxiter=%d)",
                self.niter,
                self.T,
                self.stepsize,
                self.local_method,
                self.local_maxiter,
            )

        if x0.size == 0:
            return OptimizationResult(
                success=True,
                message="No active parameters to optimize",
                initial_score=initial_score,
                final_score=initial_score,
                n_iterations=0,
                n_evaluations=objective.n_eval - n_eval_before,
                initial_params=initial_full,
                final_params=initial_full.copy(),
                history=list(objective.history),
                method=f"basinhopping({self.local_method})",
                jac_mode=self.jac,
                eps=None,
            )

        result = basinhopping(
            wrapped_objective,
            x0,
            niter=self.niter,
            T=self.T,
            take_step=take_step,
            minimizer_kwargs=minimizer_kwargs,
            seed=int(rng.integers(2**31)),
        )

        final_active = result.x.copy()
        final_params = space.expand(final_active)
        final_score = float(result.fun)

        if self.verbose:
            logger.info(
                "Basin-hopping %s: score %.6f → %.6f (%d evals, %d hops)",
                "converged" if result.lowest_optimization_result.success else "finished",
                initial_score,
                final_score,
                objective.n_eval - n_eval_before,
                self.niter,
            )

        return OptimizationResult(
            success=bool(result.lowest_optimization_result.success),
            message=str(result.message[0] if isinstance(result.message, list) else result.message),
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=self.niter,
            n_evaluations=objective.n_eval - n_eval_before,
            initial_params=initial_full,
            final_params=final_params,
            history=list(objective.history),
            method=f"basinhopping({self.local_method})",
            jac_mode=self.jac,
            eps=None if jac_fn is not None else 1e-3,
        )

    def _resolve_jac(self, objective: ObjectiveFunction, space: ActiveParameterSpace) -> Any:
        """Resolve the Jacobian function for the local minimizer."""
        if self.jac == "analytical":
            return lambda x_active: space.pack(objective.gradient(space.expand(x_active)))
        if self.jac == "auto":
            if hasattr(objective, "backend"):
                from q2mm.backends.contracts import Capability

                if objective.backend.info.supports(Capability.PARAMETER_GRADIENT):
                    if self.verbose:
                        logger.info("  Basin-hopping: using analytical gradients")
                    return lambda x_active: space.pack(objective.gradient(space.expand(x_active)))
        return None
