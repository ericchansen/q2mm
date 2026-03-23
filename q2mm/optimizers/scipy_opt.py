"""Scipy-based optimizer for Q2MM force field parameterization.

Wraps :func:`scipy.optimize.minimize` and :func:`scipy.optimize.least_squares`
with sensible defaults for force field optimization, bounds from the
:class:`~q2mm.models.forcefield.ForceField` model, and convergence tracking.

Migration note — upstream optimization methods
-----------------------------------------------
The upstream Q2MM code provided five gradient-based methods
(``gradient.py``):

- **central_diff** — central finite-difference gradient.  Equivalent to
  scipy L-BFGS-B / trust-constr with ``eps`` finite-difference step.
- **forward_diff** — forward finite-difference gradient.  Approximated by
  scipy when using ``'2-point'`` in ``jac_options``.
- **lstsq** — NumPy least-squares solve (``np.linalg.lstsq``).  Use
  ``scipy.optimize.least_squares(method='lm')`` for the same capability
  with better convergence control.
- **lagrange** — Lagrange multiplier constrained optimization.  Use
  ``scipy.optimize.minimize(method='trust-constr', constraints=...)``
  for constrained problems.
- **svd** — SVD-based parameter update.  Handled internally by scipy's
  trust-region and Levenberg-Marquardt solvers.

These are *not* ported as standalone functions because scipy provides
equivalent or superior implementations with better numerical stability,
convergence diagnostics, and bounds support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from q2mm.models.forcefield import ForceField
from q2mm.optimizers.objective import ObjectiveFunction

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of a force field optimization.

    Attributes:
        success (bool): Whether the optimizer converged.
        message (str): Human-readable convergence message.
        initial_score (float): Objective value before optimization.
        final_score (float): Objective value after optimization.
        n_iterations (int): Number of optimizer iterations.
        n_evaluations (int): Number of objective function evaluations.
        initial_params (np.ndarray): Parameter vector before optimization.
        final_params (np.ndarray): Parameter vector after optimization.
        history (list[float]): Objective value at each evaluation.
        method (str): Scipy method used for optimization.
    """

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
        """Fractional improvement (0 = no change, 1 = perfect).

        Returns:
            float: ``(initial_score - final_score) / initial_score``,
                or 0.0 if ``initial_score`` is zero.
        """
        if self.initial_score == 0.0:
            return 0.0
        return (self.initial_score - self.final_score) / self.initial_score

    def summary(self) -> str:
        """Human-readable summary.

        Returns:
            str: Multi-line summary of the optimization result.
        """
        return (
            f"Method: {self.method}\n"
            f"Success: {self.success} — {self.message}\n"
            f"Score: {self.initial_score:.6f} → {self.final_score:.6f} "
            f"({self.improvement:.1%} improvement)\n"
            f"Iterations: {self.n_iterations}, Evaluations: {self.n_evaluations}"
        )


class ScipyOptimizer:
    """Force field optimizer using scipy.optimize.

    Args:
        method (str): Scipy minimization method. Supported:
            ``'L-BFGS-B'`` (bounded quasi-Newton, default),
            ``'Nelder-Mead'`` (simplex, derivative-free),
            ``'trust-constr'`` (trust-region constrained),
            ``'Powell'`` (direction-set, derivative-free),
            ``'least_squares'`` (Levenberg-Marquardt, uses residual
            vector).
        maxiter (int): Maximum number of iterations.
        ftol (float): Function tolerance for convergence.
        eps (float): Finite-difference step size for gradient-based
            methods. Force field parameters have magnitudes ~0.5–10,
            so the default scipy step (~1e-8) is too small; 1e-3 works
            well.
        use_bounds (bool): Whether to use parameter bounds from
            :meth:`ForceField.get_bounds`.
        verbose (bool): Log progress during optimization.
        jac (str | None): Jacobian computation strategy.
            ``None`` (default) uses scipy's built-in finite differences.
            ``'analytical'`` uses :meth:`ObjectiveFunction.gradient` for
            exact analytical gradients via JAX autodiff. Only applies to
            ``scipy.optimize.minimize`` paths; not supported for
            ``method='least_squares'``.
        divergence_factor (float | None): Early stopping threshold. If
            the objective score exceeds ``divergence_factor *
            initial_score`` for ``divergence_patience`` consecutive
            callbacks, the optimizer is halted. Set to ``None`` to
            disable.
        divergence_patience (int): Number of consecutive divergent
            callbacks required before stopping.
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
        jac: str | None = None,
        divergence_factor: float | None = 3.0,
        divergence_patience: int = 5,
    ):
        """Initialize the optimizer.

        Args:
            method (str): Scipy minimization method.
            maxiter (int): Maximum number of iterations.
            ftol (float): Function tolerance for convergence.
            eps (float): Finite-difference step size.
            use_bounds (bool): Whether to use parameter bounds.
            verbose (bool): Log progress during optimization.
            jac (str | None): Jacobian computation strategy.
            divergence_factor (float | None): Early stopping threshold.
            divergence_patience (int): Consecutive divergent callbacks
                before stopping.
        """
        self.method = method
        self.maxiter = maxiter
        self.ftol = ftol
        self.eps = eps
        self.use_bounds = use_bounds
        self.verbose = verbose
        self.jac = jac
        self.divergence_factor = divergence_factor
        self.divergence_patience = divergence_patience

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the optimization.

        Args:
            objective (ObjectiveFunction): Configured objective with
                forcefield, engine, molecules, and reference data.

        Returns:
            OptimizationResult: Optimization outcome with final parameters
                and convergence history.
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
            if self.jac == "analytical":
                raise ValueError(
                    "jac='analytical' is not supported with method='least_squares'. "
                    "Use a minimize-based method (e.g. 'L-BFGS-B') for analytical gradients, "
                    "or set jac=None for least_squares."
                )
            result = self._run_least_squares(objective, x0, bounds)
        else:
            result = self._run_minimize(objective, x0, bounds, initial_score)

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
        initial_score: float,
    ) -> OptimizationResult:
        """Run scipy.optimize.minimize.

        Args:
            objective (ObjectiveFunction): The objective function.
            x0 (np.ndarray): Initial parameter vector.
            bounds (list[tuple[float, float]] | None): Parameter bounds.
            initial_score (float): Objective value at ``x0``.

        Returns:
            OptimizationResult: Result of the minimization.
        """
        from scipy import optimize

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
        if self.method not in ("Nelder-Mead", "Powell") and self.jac != "analytical":
            options["eps"] = self.eps

        # Only pass bounds for methods that support them
        effective_bounds = bounds if (bounds and self.method in self.BOUNDED_METHODS) else None

        callback = self._make_callback(objective, initial_score)

        # Analytical Jacobian via JAX
        jac = None
        if self.jac == "analytical":
            jac = objective.gradient
            if self.verbose:
                logger.info("  Using analytical JAX gradients (jac='analytical')")

        scipy_result = optimize.minimize(
            objective,
            x0,
            method=self.method,
            jac=jac,
            bounds=effective_bounds,
            options=options,
            callback=callback,
        )

        # Detect callback-triggered early stop
        abandoned = getattr(callback, "state", {}).get("abandoned", False)
        if abandoned:
            message = "Abandoned: sustained divergence from initial score"
        else:
            message = str(scipy_result.message)

        return OptimizationResult(
            success=bool(scipy_result.success),
            message=message,
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
        """Run scipy.optimize.least_squares (Levenberg-Marquardt or trf).

        Args:
            objective (ObjectiveFunction): The objective function.
            x0 (np.ndarray): Initial parameter vector.
            bounds (list[tuple[float, float]] | None): Parameter bounds.

        Returns:
            OptimizationResult: Result of the least-squares optimization.
        """
        from scipy import optimize

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

    def _make_callback(self, objective: ObjectiveFunction, initial_score: float):
        """Create a callback for minimize with optional early stopping.

        Scipy calls this after each iteration.  If the callback returns
        ``True``, scipy stops the optimization.  We use this to detect
        sustained divergence: if the score exceeds ``divergence_factor``
        times the initial score for ``divergence_patience`` consecutive
        callbacks, we bail out early rather than grinding for minutes on
        a lost cause.

        Args:
            objective (ObjectiveFunction): The objective function (used
                to read evaluation history).
            initial_score (float): Objective value before optimization.

        Returns:
            Callable: Callback function for :func:`scipy.optimize.minimize`.
        """
        diverge_count = 0
        factor = self.divergence_factor
        patience = self.divergence_patience
        verbose = self.verbose
        # Mutable flag so _run_minimize can detect callback-triggered stops
        state = {"abandoned": False}

        def callback(xk, *args, **kwargs):
            nonlocal diverge_count
            score = objective.history[-1] if objective.history else float("nan")

            if verbose:
                n = objective.n_eval
                if n % 10 == 0:
                    logger.info("  eval %4d  score %.6f", n, score)

            # Early stopping on sustained divergence
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

        callback.state = state
        return callback
