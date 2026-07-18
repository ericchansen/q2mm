"""SciPy-based optimizer for Q2MM force field parameterization.

Wraps :func:`scipy.optimize.minimize` and :func:`scipy.optimize.least_squares`.
The optimizer consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` plus an
:class:`~q2mm.models.parameters.ActiveParameterSpace` and returns the one
canonical :class:`~q2mm.models.results.OptimizationResult`.

Active/full projection is done directly through the sole projection object
(``space.pack`` / ``space.expand``) — there is no stateful wrapper.  The
canonical result is constructed exactly once at the end with full-length
vectors bound to the space's layout identity.

Gradient behaviour is explicit and driven by the executor, never probed:

- The evaluator declares ``analytical`` or ``finite_difference`` gradients:
  SciPy is driven with ``jac=True`` and the reported ``gradient_mode`` is
  the executor's declared mode.
- The evaluator declares ``none``: for a gradient-based method SciPy
  computes its own finite differences (step ``eps``) and the reported
  ``gradient_mode`` is ``"finite_difference"``; for a derivative-free
  method it is ``"none"``.
- ``least_squares`` uses SciPy's own finite-difference Jacobian and reports
  ``"finite_difference"``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)


class ScipyOptimizer:
    """Force field optimizer using :mod:`scipy.optimize`.

    Args:
        method: SciPy minimization method (``'L-BFGS-B'`` default,
            ``'Nelder-Mead'``, ``'Powell'``, ``'trust-constr'``,
            ``'least_squares'``).
        maxiter: Maximum iterations.
        ftol: Function tolerance for convergence.
        gtol: Projected-gradient tolerance for L-BFGS-B.
        maxls: Maximum L-BFGS-B line-search steps per iteration.
        eps: Finite-difference step for SciPy's internal FD.
        use_bounds: Whether to use bounds from ``space.bounds``.
        verbose: Log progress.
        divergence_factor: Early-stop threshold multiple of the initial
            score; ``None`` disables.
        divergence_patience: Consecutive divergent callbacks before stopping.
        fc_fraction: Fractional bounds for force-constant params.
        eq_fraction: Fractional bounds for equilibrium params.

    """

    DERIVATIVE_FREE_METHODS = {"Nelder-Mead", "Powell"}
    BOUNDED_METHODS = {"L-BFGS-B", "trust-constr", "least_squares"}

    def __init__(
        self,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        ftol: float = 1e-8,
        gtol: float = 1e-5,
        maxls: int = 100,
        eps: float = 1e-3,
        use_bounds: bool = True,
        verbose: bool = True,
        divergence_factor: float | None = 3.0,
        divergence_patience: int = 5,
        fc_fraction: float | None = None,
        eq_fraction: float | None = None,
    ) -> None:
        if not np.isfinite(gtol) or gtol <= 0:
            raise ValueError("gtol must be positive and finite.")
        if not isinstance(maxls, int) or maxls < 1:
            raise ValueError("maxls must be a positive int.")
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol
        self.gtol = gtol
        self.maxls = maxls
        self.eps = eps
        self.use_bounds = use_bounds
        self.verbose = verbose
        self.divergence_factor = divergence_factor
        self.divergence_patience = divergence_patience
        self.fc_fraction = fc_fraction
        self.eq_fraction = eq_fraction

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the optimization and return the canonical result."""
        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        baseline = np.array(space.baseline, dtype=float)
        x0 = space.pack(baseline)

        derivative_free = self.method in self.DERIVATIVE_FREE_METHODS
        is_least_squares = self.method == "least_squares"
        # "Evaluator gradient" means the executor supplies its own gradient —
        # analytical *or* its own finite differences — via value_and_gradient.
        use_evaluator_gradient = (
            (not derivative_free) and (not is_least_squares) and (evaluator.gradient_mode is not GradientMode.NONE)
        )

        bounds = self._resolve_bounds(space, x0)
        n_eval_before = evaluator.n_evaluations
        hist_before = len(evaluator.history)

        # Reported gradient provenance.
        if derivative_free:
            gradient_mode, fd_step = "none", None
        elif is_least_squares:
            # SciPy differences the residual vector internally (diff_step).
            gradient_mode, fd_step = "finite_difference", self.eps
        elif use_evaluator_gradient:
            # The executor supplies the gradient: analytical (step None) or its
            # own explicit finite differences (its configured FD step).
            gradient_mode = evaluator.gradient_mode.value
            fd_step = evaluator.finite_difference_step
        else:  # scipy internal finite differences (evaluator declares NONE)
            gradient_mode, fd_step = "finite_difference", self.eps

        initial_score = evaluator.value(space.expand(x0, base=baseline))

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d active params (%d total), initial score %.6f",
                self.method,
                space.n_active,
                n_params,
                initial_score,
            )

        if x0.size == 0:
            final_x, final_score, nit, success, message = x0, initial_score, 0, True, "No active parameters to optimize"
        elif is_least_squares:
            final_x, final_score, nit, success, message = self._run_least_squares(
                evaluator, space, baseline, x0, bounds
            )
        else:
            final_x, final_score, nit, success, message = self._run_minimize(
                evaluator, space, baseline, x0, bounds, initial_score, use_evaluator_gradient
            )

        final_full = space.expand(final_x, base=baseline)
        run_history = evaluator.history[hist_before:]
        # Every path records per-call scalars into the evaluator now (minimize
        # via value/value_and_gradient, least_squares via record_evaluation),
        # so history and the evaluation delta track the true call count.
        history = run_history if run_history else (initial_score, final_score)
        n_evaluations = evaluator.n_evaluations - n_eval_before

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d evals)",
                "succeeded" if success else "failed",
                initial_score,
                final_score,
                n_evaluations,
            )

        return OptimizationResult(
            success=bool(success),
            message=str(message),
            initial_score=float(initial_score),
            final_score=float(final_score),
            n_iterations=int(nit),
            n_evaluations=int(max(n_evaluations, 0)),
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=baseline,
            final_params=final_full,
            history=history,
            method=self.method if not is_least_squares else "least_squares",
            gradient_mode=gradient_mode,
            fd_step=fd_step,
        )

    def _resolve_bounds(self, space: ActiveParameterSpace, x0: np.ndarray) -> list[tuple[float, float]] | None:
        if not self.use_bounds:
            return None
        use_fractional = self.fc_fraction is not None or self.eq_fraction is not None
        if use_fractional:
            from q2mm.models.parameters import fractional_bounds

            return fractional_bounds(
                space.kinds, space.bounds, x0, fc_fraction=self.fc_fraction, eq_fraction=self.eq_fraction
            ).tolist()
        return space.bounds.tolist()

    def _run_minimize(
        self,
        evaluator: ObjectiveEvaluator,
        space: ActiveParameterSpace,
        baseline: np.ndarray,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
        initial_score: float,
        use_evaluator_gradient: bool,
    ) -> tuple[np.ndarray, float, int, bool, str]:
        from scipy import optimize

        options: dict = {"maxiter": self.maxiter}
        if self.method == "Nelder-Mead":
            options["fatol"] = self.ftol
            options["xatol"] = self.ftol
        elif self.method == "Powell":
            options["ftol"] = self.ftol
            options["xtol"] = self.ftol
        elif self.method == "trust-constr":
            options["gtol"] = self.ftol
        else:
            options["ftol"] = self.ftol
            if self.method == "L-BFGS-B":
                options["gtol"] = self.gtol
                options["maxls"] = self.maxls

        effective_bounds = bounds if (bounds and self.method in self.BOUNDED_METHODS) else None
        callback = self._make_callback(evaluator, initial_score)
        use_bound_scaling = (
            use_evaluator_gradient
            and self.method == "L-BFGS-B"
            and effective_bounds is not None
            and all(np.isfinite(bound).all() and bound[1] > bound[0] for bound in effective_bounds)
        )
        if use_bound_scaling:
            physical_bounds = np.asarray(effective_bounds, dtype=float)
            centers = np.mean(physical_bounds, axis=1)
            half_widths = (physical_bounds[:, 1] - physical_bounds[:, 0]) / 2.0
            solver_x0 = (x0 - centers) / half_widths
            solver_bounds: list[tuple[float, float]] | None = [(-1.0, 1.0)] * len(x0)
        else:
            centers = np.zeros_like(x0)
            half_widths = np.ones_like(x0)
            solver_x0 = x0
            solver_bounds = effective_bounds

        def to_physical(x_solver: np.ndarray) -> np.ndarray:
            return centers + half_widths * x_solver if use_bound_scaling else np.asarray(x_solver, dtype=float)

        best_x = x0.copy()
        best_score = float(initial_score)

        def remember(x_physical: np.ndarray, value: float) -> None:
            nonlocal best_x, best_score
            if np.isfinite(value) and value < best_score:
                best_x = np.asarray(x_physical, dtype=float).copy()
                best_score = float(value)

        def value_only(x_solver: np.ndarray) -> float:
            x_active = to_physical(x_solver)
            value = evaluator.value(space.expand(x_active, base=baseline))
            remember(x_active, value)
            return value

        def value_and_grad(x_solver: np.ndarray) -> tuple[float, np.ndarray]:
            x_active = to_physical(x_solver)
            val, full_grad = evaluator.value_and_gradient(space.expand(x_active, base=baseline))
            remember(x_active, val)
            active_grad = space.pack(full_grad)
            return val, active_grad * half_widths if use_bound_scaling else active_grad

        if use_evaluator_gradient:
            scipy_result = optimize.minimize(
                value_and_grad,
                solver_x0,
                method=self.method,
                jac=True,
                bounds=solver_bounds,
                options=options,
                callback=callback,
            )
        else:
            if self.method not in self.DERIVATIVE_FREE_METHODS:
                options["eps"] = self.eps
            scipy_result = optimize.minimize(
                value_only,
                solver_x0,
                method=self.method,
                jac=None,
                bounds=solver_bounds,
                options=options,
                callback=callback,
            )

        final_x = to_physical(np.asarray(scipy_result.x, dtype=float))
        final_score = float(scipy_result.fun)
        if bounds and self.method not in self.BOUNDED_METHODS:
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            clipped = np.clip(final_x, lower, upper)
            if not np.array_equal(clipped, final_x):
                final_score = float(value_only(clipped))
            final_x = clipped

        recovery_tolerance = 1e-8 * max(1.0, abs(best_score), abs(final_score)) if np.isfinite(final_score) else 0.0
        recovered_best = not np.isfinite(final_score) or best_score < final_score - recovery_tolerance
        if recovered_best:
            logger.warning(
                "Optimizer terminated at score %.6g after evaluating a better score %.6g; "
                "returning the best evaluated parameters.",
                final_score,
                best_score,
            )
            final_x = best_x
            final_score = best_score

        abandoned = getattr(callback, "state", {}).get("abandoned", False)
        message = "Abandoned: sustained divergence from initial score" if abandoned else str(scipy_result.message)
        if recovered_best:
            message = f"Recovered best evaluated point after terminal result: {message}"
        nit = int(scipy_result.get("nit", 0))

        if initial_score > 0 and nit <= 2 and abs(final_score - initial_score) / initial_score < 0.01:
            logger.warning(
                "%s exited after %d iteration(s) with negligible change (initial=%.4g, final=%.4g, "
                "|Δ|/init=%.2e). The optimizer likely did NOT optimize. Common causes: ftol too loose, "
                "or bounds clamp the starting point. Last scipy message: %r",
                self.method,
                nit,
                initial_score,
                final_score,
                abs(final_score - initial_score) / initial_score,
                message,
            )
        return final_x, final_score, nit, bool(scipy_result.success and not recovered_best), message

    def _run_least_squares(
        self,
        evaluator: ObjectiveEvaluator,
        space: ActiveParameterSpace,
        baseline: np.ndarray,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
    ) -> tuple[np.ndarray, float, int, bool, str]:
        from scipy import optimize

        if bounds:
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            scipy_bounds: tuple[Any, Any] = (lower, upper)
            ls_method = "trf"
        else:
            scipy_bounds = (-np.inf, np.inf)
            ls_method = "lm"

        def residuals(x_active: np.ndarray) -> np.ndarray:
            r = evaluator.least_squares_residuals(space.expand(x_active, base=baseline))
            # Record each residual evaluation as one objective evaluation so
            # n_evaluations/history track the true nfev with no duplicate
            # backend work (sum of squares == data + L2 == total).
            evaluator.record_evaluation(float(np.sum(np.asarray(r, dtype=float) ** 2)))
            return r

        scipy_result = optimize.least_squares(
            residuals,
            x0,
            method=ls_method,
            bounds=scipy_bounds,
            max_nfev=self.maxiter,
            ftol=self.ftol,
            diff_step=self.eps,
        )
        final_score = float(scipy_result.cost * 2.0)
        nfev = int(getattr(scipy_result, "nfev", 0))
        return scipy_result.x.copy(), final_score, nfev, bool(scipy_result.success), str(scipy_result.message)

    def _make_callback(self, evaluator: ObjectiveEvaluator, initial_score: float) -> Callable:
        diverge_count = 0
        factor = self.divergence_factor
        patience = self.divergence_patience
        verbose = self.verbose
        state = {"abandoned": False}

        def callback(_xk: Any, *args: Any, **kwargs: Any) -> bool:
            nonlocal diverge_count
            history = evaluator.history
            score = history[-1] if history else float("nan")
            n = evaluator.n_evaluations
            if verbose and n % 10 == 0:
                logger.info("  eval %4d  score %.6f", n, score)
            if factor is not None and initial_score > 0:
                threshold = initial_score * factor
                if score > threshold:
                    diverge_count += 1
                    if diverge_count >= patience:
                        logger.warning(
                            "Early stop: score %.1f > %.1f (%.0f× initial) for %d consecutive iterations",
                            score,
                            threshold,
                            factor,
                            patience,
                        )
                        state["abandoned"] = True
                        return True
                else:
                    diverge_count = 0
            return False

        callback.state = state  # type: ignore[attr-defined]
        return callback
