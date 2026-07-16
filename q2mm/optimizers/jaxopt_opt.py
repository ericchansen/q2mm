"""JAXopt-based optimizer for JAX-native force field optimization.

Runs a jaxopt solver (L-BFGS by default) driven by a
:class:`~q2mm.objectives.jax.JaxObjectiveExecutor`'s JAX-native
``value_and_grad_jax`` (per-case JIT + Python aggregation).  ``jit=False``
makes ``solver.run`` execute a Python while-loop that dispatches the
per-molecule compiled functions each step, so no single XLA program ever
contains all molecules.
"""

from __future__ import annotations

import logging
import math
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)

_HAS_JAXOPT = find_spec("jaxopt") is not None
_METHOD_REGISTRY = frozenset({"lbfgs", "lbfgsb", "gradient_descent"})


def _ensure_jaxopt() -> None:
    """Lazily import jaxopt, configuring JAX float64 first."""
    from q2mm.backends.mm._jax_common import ensure_jaxopt

    ensure_jaxopt()


def _require_jax_executor(evaluator: ObjectiveEvaluator) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor

    if not isinstance(evaluator, JaxObjectiveExecutor):
        raise TypeError(
            f"{type(evaluator).__name__} is not a JaxObjectiveExecutor; jaxopt-based optimizers require the "
            "JAX executor for JAX-native value_and_grad."
        )


class JaxOptOptimizer:
    """JIT-dispatched force field optimizer using jaxopt."""

    def __init__(self, method: str = "lbfgs", maxiter: int = 200, tol: float = 1e-6, verbose: bool = True) -> None:
        if method not in _METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(sorted(_METHOD_REGISTRY))}")
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the jaxopt optimization and return the canonical result."""
        _require_jax_executor(evaluator)
        _ensure_jaxopt()

        import jaxopt
        from q2mm.backends.mm._jax_common import jax, jnp

        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        n_eval_before = evaluator.n_evaluations
        hist_before = len(evaluator.history)

        initial_full = np.array(space.baseline, dtype=float)
        x0 = space.pack(initial_full)
        active_bounds = space.bounds
        baseline_jax = jnp.array(space.baseline, dtype=jnp.float64)
        active_indices_jax = jnp.array(space.active_indices, dtype=jnp.int32)

        def expand_jax(x_active: Any):  # noqa: ANN202
            return baseline_jax.at[active_indices_jax].set(x_active)

        def vag_fn(x_active: Any):  # noqa: ANN202
            loss, full_grad = evaluator.value_and_grad_jax(expand_jax(x_active))
            return loss, full_grad[active_indices_jax]

        method_str = f"jaxopt:{self.method}"
        initial_score = float(evaluator.value(initial_full))
        params = jnp.array(x0, dtype=jnp.float64)

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d active params (%d total), initial score %.6f, maxiter=%d",
                method_str,
                space.n_active,
                space.n_full,
                initial_score,
                self.maxiter,
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
                initial_params=initial_full,
                final_params=initial_full.copy(),
                history=evaluator.history[hist_before:] or (initial_score,),
                method=method_str,
                gradient_mode="analytical",
                fd_step=None,
            )

        solver = self._build_solver(jaxopt, vag_fn)
        if self.method == "lbfgsb":
            backend = jax.default_backend()
            if backend != "cpu":
                raise RuntimeError(
                    f"jaxopt LBFGSB is not supported on the {backend!r} backend — use method='lbfgs' on GPU."
                )
            lower = jnp.array(active_bounds[:, 0], dtype=jnp.float64)
            upper = jnp.array(active_bounds[:, 1], dtype=jnp.float64)
            result = solver.run(params, bounds=(lower, upper))
        else:
            result = solver.run(params)
        final_params_jax, state = result

        final_active = np.asarray(final_params_jax, dtype=float)
        final_params = space.expand(final_active, base=initial_full)
        final_score_surrogate = float(vag_fn(final_params_jax)[0])

        error_val = float(getattr(state, "error", float("inf")))
        n_iter = int(getattr(state, "iter_num", self.maxiter))
        diverged = not math.isfinite(error_val) or not math.isfinite(final_score_surrogate)

        if diverged:
            converged = False
            message = f"Optimizer diverged after {n_iter} iterations (gradient or loss became NaN)."
        elif error_val < self.tol:
            converged = True
            message = f"Converged: error {error_val:.2e} < tol {self.tol:.2e}"
        else:
            converged = False
            message = f"Max iterations ({self.maxiter}) reached (error={error_val:.2e})"

        report_final = float(evaluator.value(final_params))
        if diverged or report_final > initial_score:
            logger.warning(
                "JaxOpt final score (%.4f) worse than initial (%.4f); reverting to initial parameters.",
                report_final,
                initial_score,
            )
            final_params = initial_full.copy()
            report_final = initial_score
            converged = False
            message = f"Reverted to initial params — final score was worse than initial ({initial_score:.4f})"

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d iterations)",
                "converged" if converged else "stopped",
                initial_score,
                report_final,
                n_iter,
            )

        return OptimizationResult(
            success=converged,
            message=message,
            initial_score=initial_score,
            final_score=report_final,
            n_iterations=n_iter,
            n_evaluations=evaluator.n_evaluations - n_eval_before,
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=initial_full,
            final_params=final_params,
            history=evaluator.history[hist_before:] or (initial_score, report_final),
            method=method_str,
            gradient_mode="analytical",
            fd_step=None,
        )

    def _build_solver(self, jaxopt_mod: object, loss_fn: object) -> object:
        if self.method == "lbfgs":
            return jaxopt_mod.LBFGS(fun=loss_fn, value_and_grad=True, jit=False, maxiter=self.maxiter, tol=self.tol)
        if self.method == "lbfgsb":
            return jaxopt_mod.LBFGSB(fun=loss_fn, value_and_grad=True, jit=False, maxiter=self.maxiter, tol=self.tol)
        if self.method == "gradient_descent":
            return jaxopt_mod.GradientDescent(
                fun=loss_fn, value_and_grad=True, jit=False, maxiter=self.maxiter, tol=self.tol
            )
        raise ValueError(f"Unhandled method: {self.method}")  # pragma: no cover
