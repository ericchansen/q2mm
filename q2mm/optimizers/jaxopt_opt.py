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
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import OptimizationResult

if TYPE_CHECKING:
    from q2mm.optimizers.jaxloss import JaxLoss

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
    2. Only reference types supported by the JIT loss: energy,
       frequency, hessian-element, eigenmatrix.  Geometry references
       are silently excluded.

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

        x0 = np.array(objective.forcefield.get_param_vector(), dtype=float)
        initial_score = float(jax_loss(x0))
        params = jnp.array(x0, dtype=jnp.float64)

        # Build the jaxopt solver
        method_str = f"jaxopt:{self.method}"
        solver = self._build_solver(jaxopt, jax_loss)

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d params, initial score %.6f, maxiter=%d",
                method_str,
                len(x0),
                initial_score,
                self.maxiter,
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
            lower = jnp.array(spec.lower_bounds, dtype=jnp.float64)
            upper = jnp.array(spec.upper_bounds, dtype=jnp.float64)
            result = solver.run(params, bounds=(lower, upper))
        else:
            result = solver.run(params)
        final_params_jax, state = result

        final_params = np.asarray(final_params_jax, dtype=float)
        final_score = float(jax_loss(final_params))

        # Determine convergence
        converged = bool(getattr(state, "error", float("inf")) < self.tol)
        n_iter = int(getattr(state, "iter_num", self.maxiter))

        if converged:
            message = f"Converged: gradient error {getattr(state, 'error', 'N/A'):.2e} < {self.tol:.2e}"
        else:
            message = f"Max iterations ({self.maxiter}) reached (error={getattr(state, 'error', 'N/A')})"

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
            initial_params=x0,
            final_params=final_params,
            history=[initial_score, final_score],
            method=method_str,
            jac_mode="analytical",
            eps=None,
        )

    def _build_solver(
        self,
        jaxopt_mod: object,
        jax_loss: JaxLoss,
    ) -> object:
        """Construct the jaxopt solver.

        Args:
            jaxopt_mod: The imported jaxopt module.
            jax_loss: Compiled loss function.

        Returns:
            A jaxopt solver instance.

        """
        loss_fn = jax_loss._loss_fn

        if self.method == "lbfgs":
            return jaxopt_mod.LBFGS(
                fun=loss_fn,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        if self.method == "lbfgsb":
            return jaxopt_mod.LBFGSB(
                fun=loss_fn,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        if self.method == "gradient_descent":
            return jaxopt_mod.GradientDescent(
                fun=loss_fn,
                maxiter=self.maxiter,
                tol=self.tol,
            )
        raise ValueError(f"Unhandled method: {self.method}")  # pragma: no cover
