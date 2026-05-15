"""JAXopt-based optimizer for JAX-native force field optimization.

Provides a JIT-compiled L-BFGS optimizer that consumes a
:class:`~q2mm.optimizers.jaxloss.JaxLoss` directly, running the entire
optimization loop inside JAX.  This avoids Python-loop overhead and
enables GPU-accelerated parameter optimization.

Unlike :class:`~q2mm.optimizers.optax.OptaxOptimizer` which uses
``ObjectiveFunction.gradient()`` in a Python loop, this optimizer
operates entirely within JAX's XLA backend — parameters, loss, and
gradients never leave the JIT boundary during optimization.

Usage::

    from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

    optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200)
    result = optimizer.optimize(objective_function)

"""

from __future__ import annotations

import logging
import math
from importlib.util import find_spec
from typing import Any

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import OptimizationResult

logger = logging.getLogger(__name__)

_HAS_JAXOPT = find_spec("jaxopt") is not None


def _ensure_jaxopt() -> None:
    """Lazily import jaxopt, configuring JAX float64 first."""
    from q2mm.backends.mm._jax_common import ensure_jaxopt

    ensure_jaxopt()


# Supported method names
_METHOD_REGISTRY = frozenset({"lbfgs", "lbfgsb", "gradient_descent"})


class JaxOptOptimizer:
    """JIT-compiled force field optimizer using jaxopt.

    Compiles the objective function into a single JAX loss via
    :class:`~q2mm.optimizers.jaxloss.JaxLoss` and runs a JAX-native
    optimizer (L-BFGS by default) entirely inside XLA.

    This optimizer requires:

    1. A **JaxEngine** backend (other engines are not supported).
    2. Reference types supported by the JIT loss: energy, frequency,
       hessian-element, eigenmatrix, and geometry (bond_length,
       bond_angle, torsion_angle).  Geometry references are handled via
       implicit differentiation through an inner ``jaxopt.LBFGS``
       geometry minimization.

    Args:
        method: Optimization method.  One of ``'lbfgs'`` (default),
            ``'lbfgsb'`` (L-BFGS with box constraints), or
            ``'gradient_descent'``.
        maxiter: Maximum number of optimizer iterations.
        tol: Convergence tolerance on the gradient norm.
        verbose: Log progress during optimization.

    """

    def __init__(
        self,
        method: str = "lbfgs",
        maxiter: int = 200,
        tol: float = 1e-6,
        verbose: bool = True,
    ) -> None:
        if method not in _METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(sorted(_METHOD_REGISTRY))}")
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        self.verbose = verbose

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the JIT-compiled optimization.

        Builds a :class:`~q2mm.optimizers.jaxloss.JaxLoss` from the
        objective's spec and runs the JAX-native optimizer.

        Args:
            objective: Configured objective with forcefield, engine,
                molecules, and reference data.  Engine must be a
                JaxEngine.

        Returns:
            Optimization result with final parameters and history.

        Raises:
            TypeError: If the engine is not a JaxEngine.
            ImportError: If jaxopt is not installed.

        """
        _ensure_jaxopt()

        from q2mm.backends.mm._jax_common import jax, jnp
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.jaxloss import JaxLoss

        if not isinstance(objective.engine, JaxEngine):
            raise TypeError(
                f"JaxOptOptimizer requires a JaxEngine, got {type(objective.engine).__name__}. "
                "Use ScipyOptimizer or OptaxOptimizer for other backends."
            )

        import jaxopt

        # Build the JIT loss
        spec = objective.to_jax_spec()
        jax_loss = JaxLoss(spec, objective.engine, objective.molecules, objective.forcefield)

        ff = objective.forcefield
        initial_full = np.array(ff.get_param_vector(), dtype=float)
        has_frozen = ff.n_active_params < ff.n_params
        active_indices = np.flatnonzero(ff.active_mask) if has_frozen else None

        if has_frozen:
            x0 = np.array(ff.get_active_param_vector(), dtype=float)
            active_bounds = np.asarray(ff.get_active_bounds(), dtype=float)
            frozen_template = jnp.array(initial_full, dtype=jnp.float64)
            active_indices_jax = jnp.array(active_indices, dtype=jnp.int32)

            def expand_jax(x_active: Any):  # noqa: ANN202
                return frozen_template.at[active_indices_jax].set(x_active)

            def expand_np(x_active: np.ndarray) -> np.ndarray:
                full = initial_full.copy()
                full[active_indices] = np.asarray(x_active, dtype=float)
                return full

            def vag_fn(x_active: Any):  # noqa: ANN202
                loss, full_grad = jax_loss.value_and_grad_jax(expand_jax(x_active))
                return loss, full_grad[active_indices_jax]
        else:
            x0 = initial_full.copy()
            active_bounds = np.asarray(spec.lower_bounds, dtype=float)

            def expand_jax(x_active: Any):  # noqa: ANN202
                return x_active

            def expand_np(x_active: np.ndarray) -> np.ndarray:
                return np.asarray(x_active, dtype=float).copy()

            vag_fn = jax_loss.value_and_grad_jax

        # Use Python dispatch (value_and_grad_jax) to avoid compiling
        # all molecules into one XLA program.  jit=False makes
        # solver.run() use a Python while-loop; value_and_grad=True
        # tells jaxopt not to wrap vag_fn with jax.value_and_grad.
        initial_score = float(vag_fn(jnp.array(x0, dtype=jnp.float64))[0])
        params = jnp.array(x0, dtype=jnp.float64)

        # Build the jaxopt solver
        method_str = f"jaxopt:{self.method}"
        solver = self._build_solver(jaxopt, vag_fn)

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d active params (%d total), initial score %.6f, maxiter=%d",
                method_str,
                ff.n_active_params,
                ff.n_params,
                initial_score,
                self.maxiter,
            )

        if x0.size == 0:
            final_params = initial_full.copy()
            objective.forcefield.set_param_vector(final_params)
            return OptimizationResult(
                success=True,
                message="No active parameters to optimize",
                initial_score=initial_score,
                final_score=initial_score,
                n_iterations=0,
                n_evaluations=0,
                initial_params=initial_full,
                final_params=final_params,
                history=[initial_score],
                method=method_str,
                jac_mode="analytical",
                eps=None,
            )

        # Run the solver (LBFGSB needs bounds passed to run())
        if self.method == "lbfgsb":
            # jaxopt's LBFGSB uses XLA argsort/scatter primitives that
            # currently fail on GPU backends with dtype-mismatch errors.
            # Bail out early with a clear message rather than let the user
            # hit a cryptic XLA trace mid-compile.
            backend = jax.default_backend()
            if backend != "cpu":
                raise RuntimeError(
                    "jaxopt LBFGSB is not supported on the "
                    f"{backend!r} backend — it triggers an XLA "
                    "argsort/scatter dtype error. Use method='lbfgs' "
                    "on GPU, or force CPU with "
                    "`jax.config.update('jax_platform_name', 'cpu')`."
                )
            if has_frozen:
                lower = jnp.array(active_bounds[:, 0], dtype=jnp.float64)
                upper = jnp.array(active_bounds[:, 1], dtype=jnp.float64)
            else:
                lower = jnp.array(spec.lower_bounds, dtype=jnp.float64)
                upper = jnp.array(spec.upper_bounds, dtype=jnp.float64)
            result = solver.run(params, bounds=(lower, upper))
        else:
            result = solver.run(params)
        final_params_jax, state = result

        final_active = np.asarray(final_params_jax, dtype=float)
        final_params = expand_np(final_active)
        final_score = float(vag_fn(final_params_jax)[0])

        # Determine convergence — detect NaN divergence explicitly
        error_val = float(getattr(state, "error", float("inf")))
        n_iter = int(getattr(state, "iter_num", self.maxiter))
        diverged = not math.isfinite(error_val) or not math.isfinite(final_score)

        if diverged:
            converged = False
            message = (
                f"Optimizer diverged after {n_iter} iterations "
                "(gradient or loss became NaN). "
                "Try reducing --max-iter or tightening parameter bounds."
            )
        elif error_val < self.tol:
            converged = True
            message = f"Converged: error {error_val:.2e} < tol {self.tol:.2e}"
        else:
            converged = False
            message = f"Max iterations ({self.maxiter}) reached (error={error_val:.2e})"

        # Revert to initial parameters if optimizer diverged or made things
        # worse.  NaN > x is always False, so check diverged flag explicitly.
        if diverged or final_score > initial_score:
            diverged_score = final_score
            logger.warning(
                "JaxOpt final score (%.4f) is worse than initial (%.4f); reverting to initial parameters.",
                diverged_score,
                initial_score,
            )
            final_params = initial_full.copy()
            final_score = initial_score
            converged = False
            if not math.isfinite(diverged_score):
                message = f"Reverted to initial params — final score was NaN/Inf (initial={initial_score:.4f})"
            else:
                message = (
                    f"Reverted to initial params — final score ({diverged_score:.4f}) "
                    f"was worse than initial ({initial_score:.4f})"
                )

        # Apply final parameters to the forcefield
        objective.forcefield.set_param_vector(final_params)

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d iterations)",
                "converged" if converged else "stopped",
                initial_score,
                final_score,
                n_iter,
            )

        return OptimizationResult(
            success=converged,
            message=message,
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=n_iter,
            n_evaluations=n_iter,
            initial_params=initial_full if has_frozen else x0,
            final_params=final_params,
            history=[initial_score, final_score],
            method=method_str,
            jac_mode="analytical",
            eps=None,
        )

    def _build_solver(
        self,
        jaxopt_mod: object,
        loss_fn: object,
    ) -> object:
        """Construct the jaxopt solver.

        Uses ``jit=False`` so ``solver.run()`` executes a Python
        while-loop instead of ``lax.while_loop``.  This prevents the
        outer JIT from tracing through the per-molecule dispatch loop
        in ``JaxLoss``, which would re-inline all molecules into one
        XLA program and OOM.

        Uses ``value_and_grad=True`` so jaxopt does not wrap
        *loss_fn* with ``jax.value_and_grad`` (the function already
        returns ``(value, grad)``).

        Args:
            jaxopt_mod: The imported jaxopt module.
            loss_fn: Function returning ``(value, grad)``.

        Returns:
            A jaxopt solver instance.

        """
        if self.method == "lbfgs":
            return jaxopt_mod.LBFGS(
                fun=loss_fn,
                value_and_grad=True,
                jit=False,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        if self.method == "lbfgsb":
            return jaxopt_mod.LBFGSB(
                fun=loss_fn,
                value_and_grad=True,
                jit=False,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        if self.method == "gradient_descent":
            return jaxopt_mod.GradientDescent(
                fun=loss_fn,
                value_and_grad=True,
                jit=False,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        raise ValueError(f"Unhandled method: {self.method}")  # pragma: no cover
