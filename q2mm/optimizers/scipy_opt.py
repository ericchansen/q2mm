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
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from q2mm.optimizers._metrics import fractional_improvement
from q2mm.optimizers.objective import ObjectiveFunction

logger = logging.getLogger(__name__)


class _ActiveObjectiveWrapper:
    """Objective adapter exposing only the active force-field parameters."""

    def __init__(self, objective: ObjectiveFunction, mask: np.ndarray, frozen_full: np.ndarray) -> None:
        self._objective = objective
        self._mask = np.asarray(mask, dtype=bool)
        self._active_indices = np.flatnonzero(self._mask)
        self._frozen_full = np.asarray(frozen_full, dtype=float).copy()

    @property
    def history(self) -> list[float]:
        return self._objective.history

    @property
    def n_eval(self) -> int:
        return self._objective.n_eval

    @property
    def engine(self) -> Any:
        return self._objective.engine

    @property
    def forcefield(self) -> Any:
        return self._objective.forcefield

    def expand(self, x_active: np.ndarray) -> np.ndarray:
        full = self._frozen_full.copy()
        full[self._active_indices] = np.asarray(x_active, dtype=float)
        return full

    def __call__(self, x_active: np.ndarray) -> float:
        return self._objective(self.expand(x_active))

    def gradient(self, x_active: np.ndarray) -> np.ndarray:
        return self._objective.gradient(self.expand(x_active))[self._mask]

    def residuals(self, x_active: np.ndarray) -> np.ndarray:
        return self._objective.residuals(self.expand(x_active))


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
        jac_mode (str | None): Requested Jacobian strategy (``"auto"``,
            ``"analytical"``, or ``None``).
        eps (float | None): Finite-difference step size used by SciPy,
            or ``None`` when the method is derivative-free or uses
            analytical gradients.

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
    jac_mode: str | None = None
    eps: float | None = 1e-3

    @property
    def improvement(self) -> float:
        """Fractional improvement (0 = no change, 1 = perfect).

        Returns:
            float: ``(initial_score - final_score) / initial_score``,
                or 0.0 if ``initial_score`` is zero.

        """
        return fractional_improvement(self.initial_score, self.final_score)

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
            ``'auto'`` probes the engine: if it supports analytical
            gradients and the method is gradient-based, uses
            :meth:`ObjectiveFunction.gradient` for a hybrid
            analytical+FD Jacobian; otherwise falls back to scipy FD.
            ``'analytical'`` forces :meth:`ObjectiveFunction.gradient`
            regardless of engine support. Only applies to
            ``scipy.optimize.minimize`` paths; not supported for
            ``method='least_squares'``.
        divergence_factor (float | None): Early stopping threshold. If
            the objective score exceeds ``divergence_factor *
            initial_score`` for ``divergence_patience`` consecutive
            callbacks, the optimizer is halted. Set to ``None`` to
            disable.
        divergence_patience (int): Number of consecutive divergent
            callbacks required before stopping.
        ratio_tol (float | None): Tolerance for the JaxLoss vs
            ObjectiveFunction ratio check when ``jac='auto'``.
            Default 0.15 requires ratio within [0.85, 1.15].
            Set to ``None`` to skip the check entirely.

    """

    # Derivative-free methods — never use a Jacobian
    DERIVATIVE_FREE_METHODS = {"Nelder-Mead", "Powell"}

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
        ratio_tol: float | None = 0.15,
        fc_fraction: float | None = None,
        eq_fraction: float | None = None,
    ) -> None:
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
            ratio_tol (float | None): Tolerance for the JaxLoss vs
                ObjectiveFunction ratio check when ``jac='auto'``.
                Set to ``None`` to skip the check and always use
                JaxLoss analytical gradients.
            fc_fraction (float | None): Fractional bounds for
                force-constant parameters. When set, bounds are
                ``(val ± fc_fraction * |val|)`` instead of sanity bounds.
                Use this for **from-poor-start** runs (e.g. starting
                from QFUERZA) to prevent the optimizer from escaping the
                starting basin. ``None`` means use sanity bounds.
            eq_fraction (float | None): Same as ``fc_fraction`` but
                applied to equilibrium parameters.

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
        self.ratio_tol = ratio_tol
        self.fc_fraction = fc_fraction
        self.eq_fraction = eq_fraction

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the optimization.

        Args:
            objective (ObjectiveFunction): Configured objective with
                forcefield, engine, molecules, and reference data.

        Returns:
            OptimizationResult: Optimization outcome with final parameters
                and convergence history.

        """
        # Clear history for a fresh divergence-callback baseline but do NOT
        # reset n_eval or cached engine handles.  Resetting n_eval broke
        # cumulative eval tracking in OptimizationLoop, and clearing handles
        # forced expensive re-JIT compilation every cycle.
        objective.history.clear()
        n_eval_before = objective.n_eval

        ff = objective.forcefield
        initial_full = ff.get_param_vector().copy()
        has_frozen = ff.n_active_params < ff.n_params
        wrapped_objective: ObjectiveFunction | _ActiveObjectiveWrapper = objective

        use_fractional = self.fc_fraction is not None or self.eq_fraction is not None
        if use_fractional and not self.use_bounds:
            logger.warning("fc_fraction/eq_fraction set but use_bounds=False — ignoring fractional bounds.")

        if has_frozen:
            wrapped_objective = _ActiveObjectiveWrapper(objective, ff.active_mask, initial_full)
            x0 = ff.get_active_param_vector().copy()
            if self.use_bounds:
                if use_fractional:
                    full_bounds = np.asarray(
                        ff.get_fractional_bounds(self.fc_fraction, self.eq_fraction),
                        dtype=float,
                    )
                    bounds = full_bounds[ff.active_mask].tolist()
                else:
                    bounds = ff.get_active_bounds().tolist()
            else:
                bounds = None
            expand = wrapped_objective.expand
        else:
            x0 = initial_full.copy()
            if self.use_bounds:
                if use_fractional:
                    bounds = ff.get_fractional_bounds(self.fc_fraction, self.eq_fraction)
                else:
                    bounds = ff.get_bounds()
            else:
                bounds = None

            def expand(x: np.ndarray) -> np.ndarray:
                return np.asarray(x, dtype=float).copy()

        initial_score = wrapped_objective(x0)

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d active params (%d total), initial score %.6f",
                self.method,
                ff.n_active_params,
                ff.n_params,
                initial_score,
            )

        if x0.size == 0:
            result = OptimizationResult(
                success=True,
                message="No active parameters to optimize",
                initial_score=initial_score,
                final_score=initial_score,
                n_iterations=0,
                n_evaluations=objective.n_eval - n_eval_before,
                initial_params=initial_full,
                final_params=initial_full.copy(),
                history=list(objective.history),
                method=self.method,
                jac_mode=self.jac,
                eps=None,
            )
        elif self.method == "least_squares":
            if self.jac in ("analytical", "auto"):
                raise ValueError(
                    f"jac='{self.jac}' is not supported with method='least_squares'. "
                    "Use a minimize-based method (e.g. 'L-BFGS-B') for analytical gradients, "
                    "or set jac=None for least_squares."
                )
            result = self._run_least_squares(wrapped_objective, x0, bounds, n_eval_before)
        else:
            result = self._run_minimize(wrapped_objective, x0, bounds, initial_score, n_eval_before)

        final_full = expand(result.final_params)
        if has_frozen:
            result = replace(
                result,
                initial_params=initial_full,
                final_params=final_full,
            )

        # Apply final parameters to the forcefield
        objective.forcefield.set_param_vector(final_full)

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
        n_eval_before: int,
    ) -> OptimizationResult:
        """Run scipy.optimize.minimize.

        Args:
            objective (ObjectiveFunction): The objective function.
            x0 (np.ndarray): Initial parameter vector.
            bounds (list[tuple[float, float]] | None): Parameter bounds.
            initial_score (float): Objective value at ``x0``.
            n_eval_before (int): ``objective.n_eval`` before this run,
                used to compute the evaluation count delta.

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

        # Only pass bounds for methods that support them
        effective_bounds = bounds if (bounds and self.method in self.BOUNDED_METHODS) else None

        # Telemetry for the JaxLoss path: scipy is driven by
        # ``use_jax_loss_fun`` which never calls ``objective.__call__``, so
        # ``objective.n_eval`` / ``objective.history`` stay frozen. Track the
        # surrogate call count and latest surrogate score here so the
        # divergence callback and reported n_evaluations reflect real work.
        jax_telemetry: dict[str, Any] = {"n_eval": 0, "last_score": None, "initial": None}

        callback = self._make_callback(objective, initial_score, jax_telemetry)

        # Resolve Jacobian strategy:
        #   - jac="analytical" → always use objective.gradient
        #   - jac="auto" + gradient-based method + JaxEngine → JaxLoss analytical gradients
        #   - jac="auto" + gradient-based method + engine supports it → auto-enable
        #   - jac=None → scipy's own finite differences (default, safest)
        jac = None
        uses_scipy_fd = False
        use_jax_loss_fun = None  # set to a (loss, grad) function if JaxLoss path
        jac_mode_str = self.jac
        if self.method in self.DERIVATIVE_FREE_METHODS:
            pass  # no gradients needed
        elif self.jac == "analytical":
            jac = objective.gradient
            if self.verbose:
                logger.info("  Using analytical gradients (jac='analytical')")
        elif self.jac == "auto" and self.method not in self.DERIVATIVE_FREE_METHODS:
            # Try JaxLoss path first — per-molecule JIT dispatch,
            # analytical gradients for all ref types including geometry.
            try:
                from q2mm.backends.mm.jax_engine import JaxEngine

                if isinstance(objective.engine, JaxEngine):
                    from q2mm.optimizers.jaxloss import JaxLoss

                    # Unwrap _ActiveObjectiveWrapper to get raw ObjectiveFunction
                    raw_obj = getattr(objective, "_objective", objective)
                    spec = raw_obj.to_jax_spec()
                    jax_loss = JaxLoss(spec, raw_obj.engine, raw_obj.molecules, raw_obj.forcefield)

                    ff = raw_obj.forcefield
                    active_idx = np.flatnonzero(ff.active_mask)
                    frozen_full = np.array(ff.get_param_vector(), dtype=float)

                    def _jax_loss_fun(x_active: np.ndarray) -> tuple[float, np.ndarray]:
                        full = frozen_full.copy()
                        full[active_idx] = np.asarray(x_active, dtype=float)
                        loss, grad_full = jax_loss.loss_and_grad(full)
                        jax_telemetry["n_eval"] += 1
                        jax_telemetry["last_score"] = float(loss)
                        return loss, grad_full[active_idx]

                    # Validate JaxLoss/ObjectiveFunction agreement at x0.
                    # If the ratio deviates significantly, JaxLoss is an
                    # unreliable surrogate — fall back to FD gradients.
                    # Set ratio_tol=None to skip this check entirely.
                    jl_val, _ = _jax_loss_fun(x0)
                    jax_telemetry["initial"] = float(jl_val)
                    ratio = jl_val / initial_score if initial_score > 0 else 1.0
                    tol = self.ratio_tol
                    if tol is None:
                        # Bypass ratio check — always use JaxLoss
                        use_jax_loss_fun = _jax_loss_fun
                        jac_mode_str = "jax_loss"
                        if self.verbose:
                            logger.info(
                                "  Using JaxLoss analytical gradients (ratio check disabled, ratio=%.3f)",
                                ratio,
                            )
                    elif not math.isfinite(ratio) or not (1 - tol <= ratio <= 1 + tol):
                        logger.warning(
                            "JaxLoss/ObjectiveFunction ratio %.3f outside [%.2f, %.2f] — "
                            "geometry relaxation methods disagree. "
                            "Falling back to finite-difference gradients.",
                            ratio,
                            1 - tol,
                            1 + tol,
                        )
                    else:
                        use_jax_loss_fun = _jax_loss_fun
                        jac_mode_str = "jax_loss"
                        if self.verbose:
                            logger.info("  Using JaxLoss analytical gradients (per-molecule JIT dispatch)")
            except (ImportError, AttributeError) as exc:
                logger.debug("JaxLoss auto-path unavailable (%s: %s)", type(exc).__name__, exc)

            if use_jax_loss_fun is None and objective.engine.supports_analytical_gradients():
                jac = objective.gradient
                if self.verbose:
                    logger.info(
                        "  Auto-detected analytical gradient support from %s — using analytical+FD hybrid Jacobian",
                        type(objective.engine).__name__,
                    )
        if jac is None and use_jax_loss_fun is None and self.method not in self.DERIVATIVE_FREE_METHODS:
            uses_scipy_fd = True

        # Finite-difference step: only needed when scipy computes its own FD gradient
        if uses_scipy_fd:
            options["eps"] = self.eps

        if use_jax_loss_fun is not None:
            # JaxLoss path: fun returns (loss, grad), jac=True tells scipy
            scipy_result = optimize.minimize(
                use_jax_loss_fun,
                x0,
                method=self.method,
                jac=True,
                bounds=effective_bounds,
                options=options,
                callback=callback,
            )
        else:
            scipy_result = optimize.minimize(
                objective,
                x0,
                method=self.method,
                jac=jac,
                bounds=effective_bounds,
                options=options,
                callback=callback,
            )

        # For methods without native bounds support (Powell, Nelder-Mead),
        # project final params into the feasible region.  We don't wrap the
        # objective during optimization because clamping distorts the simplex
        # geometry and can cause divergence.
        final_x = scipy_result.x.copy()
        final_score = float(scipy_result.fun)
        if bounds and self.method not in self.BOUNDED_METHODS:
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            clipped = np.clip(final_x, lower, upper)
            if not np.array_equal(clipped, final_x):
                final_score = float(objective(clipped))
            final_x = clipped

        # When using the JaxLoss path, scipy_result.fun is the JaxLoss
        # value (a differentiable surrogate).  Re-evaluate with the
        # ObjectiveFunction so initial_score and final_score are in the
        # same units — both from ObjectiveFunction.
        if use_jax_loss_fun is not None:
            final_score = float(objective(final_x))
            # Safety guard: if the JaxLoss-guided step worsened the
            # ObjectiveFunction, revert to initial parameters.  Note that
            # for systems with appreciable engine non-determinism (q2mm#284)
            # this comparison can be dominated by per-call noise; downstream
            # consumers should treat single-call apparent improvements/worsenings
            # smaller than the per-system noise floor as inconclusive.
            if final_score > initial_score:
                logger.warning(
                    "JaxLoss-guided step worsened ObjectiveFunction: %.0f -> %.0f (%.1f%% worse). "
                    "Reverting to initial parameters.",
                    initial_score,
                    final_score,
                    (final_score / initial_score - 1) * 100,
                )
                final_x = x0.copy()
                final_score = initial_score

        # Detect callback-triggered early stop
        abandoned = getattr(callback, "state", {}).get("abandoned", False)
        if abandoned:
            message = "Abandoned: sustained divergence from initial score"
        else:
            message = str(scipy_result.message)

        # Diagnostic: silent-failure detection.  L-BFGS-B can return
        # "convergence" after 0-2 iterations when the line search can't
        # find a descent direction (e.g. JaxLoss-vs-OF mismatch, or
        # ftol is too loose).  Surface this so the next run isn't
        # mistaken for "the optimizer did its best".
        nit = int(scipy_result.get("nit", 0))
        if initial_score > 0 and nit <= 2 and abs(final_score - initial_score) / initial_score < 0.01:
            logger.warning(
                "%s exited after %d iteration(s) with negligible change "
                "(initial=%.4g, final=%.4g, |Δ|/init=%.2e). The optimizer "
                "likely did NOT optimize. Common causes: (a) ftol too loose "
                "for this scale of objective, (b) JaxLoss/ObjectiveFunction "
                "ratio is far from 1 (surrogate gradients unreliable), "
                "(c) bounds clamp the starting point. Last scipy message: %r",
                self.method,
                nit,
                initial_score,
                final_score,
                abs(final_score - initial_score) / initial_score,
                message,
            )

        return OptimizationResult(
            success=bool(scipy_result.success),
            message=message,
            initial_score=objective.history[0] if objective.history else 0.0,
            final_score=final_score,
            n_iterations=nit,
            n_evaluations=(
                jax_telemetry["n_eval"] if use_jax_loss_fun is not None else objective.n_eval - n_eval_before
            ),
            initial_params=x0,
            final_params=final_x,
            history=list(objective.history),
            method=self.method,
            jac_mode=jac_mode_str,
            eps=self.eps if uses_scipy_fd else None,
        )

    def _run_least_squares(
        self,
        objective: ObjectiveFunction,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
        n_eval_before: int,
    ) -> OptimizationResult:
        """Run scipy.optimize.least_squares (Levenberg-Marquardt or trf).

        Args:
            objective (ObjectiveFunction): The objective function.
            x0 (np.ndarray): Initial parameter vector.
            bounds (list[tuple[float, float]] | None): Parameter bounds.
            n_eval_before (int): ``objective.n_eval`` before this run,
                used to compute the evaluation count delta.

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
            n_evaluations=objective.n_eval - n_eval_before,
            initial_params=x0,
            final_params=scipy_result.x.copy(),
            history=list(objective.history),
            method=f"least_squares({ls_method})",
            jac_mode=self.jac,
            eps=self.eps,
        )

    def _make_callback(
        self,
        objective: ObjectiveFunction,
        initial_score: float,
        telemetry: dict[str, Any] | None = None,
    ) -> Callable:
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
            telemetry (dict | None): Live JaxLoss telemetry
                (``n_eval`` / ``last_score`` / ``initial``).  When the
                optimizer is driven by the JaxLoss surrogate,
                ``objective.history`` stays frozen, so the callback reads
                the surrogate score and its own baseline from here instead
                — keeping the score and the divergence threshold in the
                same (JaxLoss) units.

        Returns:
            Callable: Callback function for :func:`scipy.optimize.minimize`.

        """
        diverge_count = 0
        factor = self.divergence_factor
        patience = self.divergence_patience
        verbose = self.verbose
        # Mutable flag so _run_minimize can detect callback-triggered stops
        state = {"abandoned": False}

        def callback(_xk: Any, *args: Any, **kwargs: Any) -> bool:
            """Log progress and trigger early stopping on sustained divergence."""
            nonlocal diverge_count
            if telemetry is not None and telemetry.get("last_score") is not None:
                score = telemetry["last_score"]
                n = telemetry["n_eval"]
                baseline = telemetry["initial"] if telemetry.get("initial") is not None else initial_score
            else:
                score = objective.history[-1] if objective.history else float("nan")
                n = objective.n_eval
                baseline = initial_score

            if verbose:
                if n % 10 == 0:
                    logger.info("  eval %4d  score %.6f", n, score)

            # Early stopping on sustained divergence
            if factor is not None and baseline > 0:
                threshold = baseline * factor
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
