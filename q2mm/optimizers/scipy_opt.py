"""Scipy-based optimizer for Q2MM force field parameterization.

Wraps :func:`scipy.optimize.minimize` and :func:`scipy.optimize.least_squares`
with sensible defaults for force field optimization, bounds from the
:class:`~q2mm.models.forcefield.ForceField` model, and convergence tracking.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import optimize

from q2mm.models.forcefield import ForceField
from q2mm.optimizers.objective import ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a force field optimization."""

    success: bool
    message: str
    initial_score: float
    final_score: float
    n_iterations: int
    n_evaluations: int
    initial_params: np.ndarray
    final_params: np.ndarray
    history: list[float]
    method: str

    @property
    def improvement(self) -> float:
        """Fractional improvement (0 = no change, 1 = perfect)."""
        if self.initial_score == 0.0:
            return 0.0
        return (self.initial_score - self.final_score) / self.initial_score

    def summary(self) -> str:
        """Human-readable summary."""
        return (
            f"Method: {self.method}\n"
            f"Success: {self.success} — {self.message}\n"
            f"Score: {self.initial_score:.6f} → {self.final_score:.6f} "
            f"({self.improvement:.1%} improvement)\n"
            f"Iterations: {self.n_iterations}, Evaluations: {self.n_evaluations}"
        )


class ScipyOptimizer:
    """Force field optimizer using scipy.optimize.

    Parameters
    ----------
    method : str
        Scipy minimization method. Supported:
        - ``'L-BFGS-B'``: Bounded quasi-Newton (default, good for smooth problems)
        - ``'Nelder-Mead'``: Simplex (derivative-free, robust)
        - ``'trust-constr'``: Trust-region constrained
        - ``'Powell'``: Direction-set (derivative-free)
        - ``'least_squares'``: Levenberg-Marquardt (uses residual vector)
    maxiter : int
        Maximum number of iterations.
    ftol : float
        Function tolerance for convergence.
    eps : float
        Finite-difference step size for gradient-based methods.
        Force field parameters have magnitudes ~0.5–10, so the default
        scipy step (~1e-8) is too small; 1e-3 works well.
    use_bounds : bool
        Whether to use parameter bounds from ForceField.get_bounds().
    verbose : bool
        Log progress during optimization.
    """

    BOUNDED_METHODS = {"L-BFGS-B", "trust-constr", "least_squares"}

    def __init__(
        self,
        method: str = "L-BFGS-B",
        maxiter: int = 500,
        ftol: float = 1e-8,
        eps: float = 1e-3,
        use_bounds: bool = True,
        verbose: bool = True,
    ):
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol
        self.eps = eps
        self.use_bounds = use_bounds
        self.verbose = verbose

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the optimization.

        Parameters
        ----------
        objective : ObjectiveFunction
            Configured objective with forcefield, engine, molecules, and reference data.

        Returns
        -------
        OptimizationResult
            Optimization outcome with final parameters and convergence history.
        """
        objective.reset()
        x0 = objective.forcefield.get_param_vector().copy()
        initial_score = objective(x0)

        bounds = objective.forcefield.get_bounds() if self.use_bounds else None

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d params, initial score %.6f",
                self.method,
                len(x0),
                initial_score,
            )

        if self.method == "least_squares":
            result = self._run_least_squares(objective, x0, bounds)
        else:
            result = self._run_minimize(objective, x0, bounds)

        # Apply final parameters to the forcefield
        objective.forcefield.set_param_vector(result.final_params)

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d evals)",
                "succeeded" if result.success else "failed",
                result.initial_score,
                result.final_score,
                result.n_evaluations,
            )

        return result

    def _run_minimize(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizationResult:
        """Run scipy.optimize.minimize."""
        options: dict = {"maxiter": self.maxiter}

        # Method-specific convergence tolerance and step size
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

        # Finite-difference step for gradient-based methods
        if self.method not in ("Nelder-Mead", "Powell"):
            options["eps"] = self.eps

        # Only pass bounds for methods that support them
        effective_bounds = bounds if (bounds and self.method in self.BOUNDED_METHODS) else None

        callback = self._make_callback(objective) if self.verbose else None

        scipy_result = optimize.minimize(
            objective,
            x0,
            method=self.method,
            bounds=effective_bounds,
            options=options,
            callback=callback,
        )

        return OptimizationResult(
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            initial_score=objective.history[0] if objective.history else 0.0,
            final_score=float(scipy_result.fun),
            n_iterations=int(scipy_result.get("nit", 0)),
            n_evaluations=objective.n_eval,
            initial_params=x0,
            final_params=scipy_result.x.copy(),
            history=list(objective.history),
            method=self.method,
        )

    def _run_least_squares(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
    ) -> OptimizationResult:
        """Run scipy.optimize.least_squares (Levenberg-Marquardt or trf)."""
        if bounds:
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            scipy_bounds = (lower, upper)
            ls_method = "trf"
        else:
            scipy_bounds = (-np.inf, np.inf)
            ls_method = "lm"

        scipy_result = optimize.least_squares(
            objective.residuals,
            x0,
            method=ls_method,
            bounds=scipy_bounds,
            max_nfev=self.maxiter,
            ftol=self.ftol,
            diff_step=self.eps,
        )

        final_score = float(scipy_result.cost * 2.0)  # cost = 0.5 * sum(r^2)

        return OptimizationResult(
            success=bool(scipy_result.success),
            message=str(scipy_result.message),
            initial_score=objective.history[0] if objective.history else 0.0,
            final_score=final_score,
            n_iterations=int(getattr(scipy_result, "njev", 0)),
            n_evaluations=int(scipy_result.nfev),
            initial_params=x0,
            final_params=scipy_result.x.copy(),
            history=list(objective.history),
            method=f"least_squares({ls_method})",
        )

    @staticmethod
    def _make_callback(objective: ObjectiveFunction):
        """Create a logging callback for minimize."""

        def callback(xk):
            n = objective.n_eval
            score = objective.history[-1] if objective.history else float("nan")
            if n % 10 == 0:
                logger.info("  eval %4d  score %.6f", n, score)

        return callback
