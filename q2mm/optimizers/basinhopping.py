"""Basin-hopping global optimizer for force field parameterization.

Wraps :func:`scipy.optimize.basinhopping` with a local L-BFGS-B minimizer
and bounded perturbation steps.  Consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` and returns the
canonical :class:`~q2mm.models.results.OptimizationResult`.

Active/full projection uses ``space.pack``/``space.expand`` directly.  The
local minimizer uses the evaluator's analytical/FD gradients whenever the
evaluator declares a non-``NONE`` gradient mode; otherwise SciPy computes
its own finite differences.  The gradient behaviour is chosen by the
caller's executor selection — never probed here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)


class _BoundedStep:
    """Random perturbation step that respects parameter bounds."""

    def __init__(self, stepsize: float, bounds: list[tuple[float, float]] | None, rng: np.random.Generator) -> None:
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
    """Global optimizer using basin-hopping with L-BFGS-B local steps."""

    def __init__(
        self,
        niter: int = 50,
        T: float = 1.0,
        stepsize: float = 0.5,
        local_method: str = "L-BFGS-B",
        local_maxiter: int = 200,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.niter = niter
        self.T = T
        self.stepsize = stepsize
        self.local_method = local_method
        self.local_maxiter = local_maxiter
        self.seed = seed
        self.verbose = verbose

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run basin-hopping and return the canonical result."""
        from scipy.optimize import basinhopping

        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        baseline = np.array(space.baseline, dtype=float)
        x0 = space.pack(baseline)

        n_eval_before = evaluator.n_evaluations
        hist_before = len(evaluator.history)
        initial_score = evaluator.value(space.expand(x0, base=baseline))

        use_evaluator_gradient = evaluator.gradient_mode is not GradientMode.NONE
        if use_evaluator_gradient:
            gradient_mode = evaluator.gradient_mode.value
            fd_step = evaluator.finite_difference_step
        else:
            gradient_mode = "finite_difference"
            fd_step = 1e-3

        bounds = space.bounds.tolist()
        rng = np.random.default_rng(self.seed)

        options: dict[str, Any] = {"maxiter": self.local_maxiter}
        jac_fn = None
        if use_evaluator_gradient:

            def jac_fn(x_active: np.ndarray) -> np.ndarray:  # noqa: F811
                return space.pack(evaluator.value_and_gradient(space.expand(x_active, base=baseline))[1])
        else:
            options["eps"] = 1e-3

        def value_only(x_active: np.ndarray) -> float:
            return evaluator.value(space.expand(x_active, base=baseline))

        minimizer_kwargs: dict[str, Any] = {"method": self.local_method, "jac": jac_fn, "options": options}
        if bounds and self.local_method in ("L-BFGS-B", "trust-constr", "SLSQP"):
            minimizer_kwargs["bounds"] = bounds

        take_step = _BoundedStep(self.stepsize, bounds, rng)

        if self.verbose:
            logger.info(
                "Basin-hopping: %d hops, T=%.2f, stepsize=%.2f, local=%s",
                self.niter,
                self.T,
                self.stepsize,
                self.local_method,
            )

        if x0.size == 0:
            return OptimizationResult(
                success=True,
                message="No active parameters to optimize",
                initial_score=initial_score,
                final_score=initial_score,
                n_iterations=0,
                n_evaluations=evaluator.n_evaluations - n_eval_before,
                n_params=n_params,
                layout_fingerprint=fingerprint,
                initial_params=baseline,
                final_params=baseline.copy(),
                history=evaluator.history[hist_before:] or (initial_score,),
                method=f"basinhopping({self.local_method})",
                gradient_mode=gradient_mode,
                fd_step=fd_step,
            )

        result = basinhopping(
            value_only,
            x0,
            niter=self.niter,
            T=self.T,
            take_step=take_step,
            minimizer_kwargs=minimizer_kwargs,
            seed=int(rng.integers(2**31)),
        )

        final_params = space.expand(result.x.copy(), base=baseline)
        final_score = float(result.fun)

        if self.verbose:
            logger.info(
                "Basin-hopping %s: score %.6f → %.6f (%d hops)",
                "converged" if result.lowest_optimization_result.success else "finished",
                initial_score,
                final_score,
                self.niter,
            )

        return OptimizationResult(
            success=bool(result.lowest_optimization_result.success),
            message=str(result.message[0] if isinstance(result.message, list) else result.message),
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=self.niter,
            n_evaluations=evaluator.n_evaluations - n_eval_before,
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=baseline,
            final_params=final_params,
            history=evaluator.history[hist_before:] or (initial_score, final_score),
            method=f"basinhopping({self.local_method})",
            gradient_mode=gradient_mode,
            fd_step=fd_step,
        )
