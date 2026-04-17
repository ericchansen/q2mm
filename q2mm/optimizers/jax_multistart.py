"""Multi-start optimizer that fuses N replicas into one XLA kernel.

Rather than running an inner optimizer N times in a Python loop (see
:class:`~q2mm.optimizers.multistart.MultiStartOptimizer`), this
optimizer ``jax.vmap``-s a single jaxopt solver ``run`` call over a
batch of initial parameter vectors.  The entire search — loss,
gradients, L-BFGS line search, and the best-of-N selection — happens
inside one JIT-compiled graph.

Expected wins on GPU scale with ``n_starts``: each replica shares the
same compiled ``JaxLoss._loss_fn``, and XLA can overlap the independent
replicas on the device.

Only applies to :class:`~q2mm.optimizers.jaxopt_opt.JaxOptOptimizer`'s
supported methods.  ``lbfgsb`` inherits the CPU-only limitation
(upstream jaxopt argsort/scatter dtype issue on GPU).

Usage::

    from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

    opt = JaxMultiStartOptimizer(
        method="lbfgs",
        n_starts=100,
        maxiter=500,
        perturbation_pct=0.1,
        seed=0,
    )
    result = opt.optimize(objective_function)

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from q2mm.optimizers.jaxopt_opt import _METHOD_REGISTRY, _ensure_jaxopt
from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import OptimizationResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class JaxMultiStartOptimizer:
    """Vmap-fused multi-start optimizer backed by jaxopt.

    Runs ``n_starts`` replicas of a jaxopt solver in parallel via
    ``jax.vmap``, all inside one XLA kernel.  The first replica uses
    the unperturbed initial parameters; the remaining ``n_starts - 1``
    are uniformly perturbed by ``±perturbation_pct`` of each parameter
    value.

    Args:
        method: jaxopt method name.  One of ``'lbfgs'`` (default),
            ``'lbfgsb'`` (CPU-only), or ``'gradient_descent'``.
        n_starts: Number of parallel replicas.  Must be ``>= 1``.
        maxiter: Maximum optimizer iterations per replica.
        tol: Convergence tolerance passed to each replica's solver;
            individual replicas may stop early when this tolerance is
            met.
        perturbation_pct: Max perturbation as a fraction of each
            parameter's value.  ``0.1`` means ±10%.  Perturbations are
            clipped to the force field's parameter bounds.
        seed: RNG seed for reproducible perturbations.
        verbose: Log progress.

    Raises:
        ValueError: On invalid method, n_starts, or perturbation_pct.

    """

    def __init__(
        self,
        *,
        method: str = "lbfgs",
        n_starts: int = 10,
        maxiter: int = 200,
        tol: float = 1e-6,
        perturbation_pct: float = 0.1,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        if method not in _METHOD_REGISTRY:
            raise ValueError(f"Unknown method '{method}'. Choose from: {', '.join(sorted(_METHOD_REGISTRY))}")
        if n_starts < 1:
            raise ValueError("n_starts must be >= 1")
        if perturbation_pct < 0:
            raise ValueError("perturbation_pct must be >= 0")
        self.method = method
        self.n_starts = n_starts
        self.maxiter = maxiter
        self.tol = tol
        self.perturbation_pct = perturbation_pct
        self.seed = seed
        self.verbose = verbose

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run ``n_starts`` replicas in parallel, return the best.

        Args:
            objective: Configured objective with a JaxEngine backend.

        Returns:
            OptimizationResult for the replica with the lowest final
            loss.  ``n_iterations`` and ``success`` are taken from the
            winning replica's jaxopt state.

        Raises:
            TypeError: If the engine is not a JaxEngine.
            ImportError: If jaxopt is not installed.
            RuntimeError: If ``method='lbfgsb'`` on a non-CPU backend.

        """
        _ensure_jaxopt()

        from q2mm.backends.mm._jax_common import jax, jnp
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.jaxloss import JaxLoss

        if not isinstance(objective.engine, JaxEngine):
            raise TypeError(f"JaxMultiStartOptimizer requires a JaxEngine, got {type(objective.engine).__name__}.")

        if self.method == "lbfgsb":
            backend = jax.default_backend()
            if backend != "cpu":
                raise RuntimeError(
                    f"jaxopt LBFGSB is not supported on the {backend!r} backend — use method='lbfgs' on GPU."
                )

        import jaxopt

        spec = objective.to_jax_spec()
        jax_loss = JaxLoss(spec, objective.engine, objective.molecules, objective.forcefield)

        x0 = np.array(objective.forcefield.get_param_vector(), dtype=float)
        bounds = objective.forcefield.get_bounds()

        # Evaluate the unperturbed initial score for reporting.
        initial_score = float(jax_loss(x0))

        # Generate n_starts initial parameter vectors.
        starts_np = self._generate_starts(x0, bounds)
        starts = jnp.asarray(starts_np, dtype=jnp.float64)

        method_str = f"jaxopt-multi:{self.method}"
        loss_fn = jax_loss._loss_fn

        if self.verbose:
            logger.info(
                "Starting %s: n_starts=%d, maxiter=%d, initial score %.6f",
                method_str,
                self.n_starts,
                self.maxiter,
                initial_score,
            )

        # Build the solver once and vmap .run over the batch of inits.
        if self.method == "lbfgs":
            solver = jaxopt.LBFGS(fun=loss_fn, maxiter=self.maxiter, tol=self.tol)
            run_one = lambda p: solver.run(p)  # noqa: E731
            final_params_batch, state_batch = jax.vmap(run_one)(starts)
        elif self.method == "lbfgsb":
            solver = jaxopt.LBFGSB(fun=loss_fn, maxiter=self.maxiter, tol=self.tol)
            lower = jnp.array(spec.lower_bounds, dtype=jnp.float64)
            upper = jnp.array(spec.upper_bounds, dtype=jnp.float64)
            run_one = lambda p: solver.run(p, bounds=(lower, upper))  # noqa: E731
            final_params_batch, state_batch = jax.vmap(run_one)(starts)
        elif self.method == "gradient_descent":
            solver = jaxopt.GradientDescent(fun=loss_fn, maxiter=self.maxiter, tol=self.tol)
            run_one = lambda p: solver.run(p)  # noqa: E731
            final_params_batch, state_batch = jax.vmap(run_one)(starts)
        else:  # pragma: no cover - guarded by __init__ validation
            raise ValueError(f"Unhandled method: {self.method}")

        # Score each replica's final params on-device and pick argmin.
        # Only the best replica's params/state are transferred to host;
        # the full (n_starts, n_params) batch stays on-device.
        final_scores = jax.vmap(loss_fn)(final_params_batch)
        best_idx = int(jax.device_get(jnp.argmin(final_scores)))
        best_params = np.asarray(jax.device_get(final_params_batch[best_idx]), dtype=float)
        best_score = float(jax.device_get(final_scores[best_idx]))

        # Pull best replica's solver state (per-replica metadata lives
        # on-device until we index into it here).
        best_error = (
            float(jax.device_get(state_batch.error[best_idx])) if hasattr(state_batch, "error") else float("inf")
        )
        best_iter = (
            int(jax.device_get(state_batch.iter_num[best_idx])) if hasattr(state_batch, "iter_num") else self.maxiter
        )
        converged = best_error < self.tol

        # Apply best params to the forcefield.
        objective.forcefield.set_param_vector(best_params)

        if self.verbose:
            min_score = float(jax.device_get(jnp.min(final_scores)))
            median_score = float(jax.device_get(jnp.median(final_scores)))
            max_score = float(jax.device_get(jnp.max(final_scores)))
            logger.info(
                "%s best: %.6f (replica %d/%d; scores min=%.6f, median=%.6f, max=%.6f)",
                method_str,
                best_score,
                best_idx,
                self.n_starts,
                min_score,
                median_score,
                max_score,
            )

        if converged:
            message = (
                f"jaxopt-multi best of {self.n_starts}: replica {best_idx}, final {best_score:.6g} "
                f"(converged: error {best_error:.2e} < {self.tol:.2e})"
            )
        else:
            message = (
                f"jaxopt-multi best of {self.n_starts}: replica {best_idx}, final {best_score:.6g} "
                f"(maxiter reached, error={best_error:.2e})"
            )

        return OptimizationResult(
            success=converged,
            message=message,
            initial_score=initial_score,
            final_score=best_score,
            n_iterations=best_iter,
            n_evaluations=best_iter * self.n_starts,
            initial_params=x0,
            final_params=best_params,
            history=[initial_score, best_score],
            method=method_str,
            jac_mode="analytical",
            eps=None,
        )

    def _generate_starts(
        self,
        x0: np.ndarray,
        bounds: list[tuple[float, float]] | None,
    ) -> np.ndarray:
        """Generate ``(n_starts, n_params)`` initial parameter batch.

        The first row is ``x0`` unchanged; subsequent rows are
        ``x0 + U(-scale, +scale)`` where
        ``scale = max(|x0| * perturbation_pct, 1e-6)``.
        """
        rng = np.random.default_rng(self.seed)
        n = self.n_starts
        starts = np.tile(x0, (n, 1))
        if n > 1 and self.perturbation_pct > 0:
            scale = np.maximum(np.abs(x0) * self.perturbation_pct, 1e-6)
            perturbations = rng.uniform(-scale, scale, size=(n - 1, len(x0)))
            starts[1:] = x0 + perturbations
            if bounds is not None:
                lower = np.array([b[0] for b in bounds])
                upper = np.array([b[1] for b in bounds])
                starts[1:] = np.clip(starts[1:], lower, upper)
        return starts
