"""Optax-based optimizer for Q2MM force field parameterization.

Provides iterative gradient-based optimizers (Adam, AdaGrad, SGD) from the
`optax <https://optax.readthedocs.io/>`_ library with learning rate schedules,
bounds enforcement, and convergence detection.

Unlike :class:`~q2mm.optimizers.scipy_opt.ScipyOptimizer` which wraps
``scipy.optimize.minimize``, this module implements an explicit training loop:
each step calls :meth:`ObjectiveFunction.gradient` for the gradient and
:func:`optax.apply_updates` for the parameter update.  This works with any
engine — JAX engines use analytical gradients while others fall back to
finite differences transparently.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import Any

import numpy as np

from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import OptimizationResult

logger = logging.getLogger(__name__)

_HAS_OPTAX = find_spec("optax") is not None

# Populated by ensure_optax()
optax = None
jnp = None


def ensure_optax() -> None:
    """Lazily import optax and jax.numpy, with float64 enabled."""
    global optax, jnp  # noqa: PLW0603
    if optax is not None:
        return
    if not _HAS_OPTAX:
        raise ImportError("optax is required for OptaxOptimizer. Install it with: pip install q2mm[jax]")
    # Route through ensure_jax to get float64 configured before any jnp use
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax(engine_name="OptaxOptimizer")

    import optax as _optax
    from q2mm.backends.mm._jax_common import jnp as _jnp

    optax = _optax
    jnp = _jnp


# Supported optimizer names → optax constructor
_OPTIMIZER_REGISTRY: dict[str, str] = {
    "adam": "adam",
    "adamw": "adamw",
    "adagrad": "adagrad",
    "sgd": "sgd",
}


class OptaxOptimizer:
    """Force field optimizer using optax gradient transformations.

    Note:
        Bounds are enforced via clamping after each update step.
        Momentum-based optimizers (Adam, SGD) may accumulate state that
        pushes into bound walls, which can slow convergence near bounds.
        For tight bound constraints, consider
        :class:`~q2mm.optimizers.scipy_opt.ScipyOptimizer` with
        ``method='L-BFGS-B'`` which handles bounds natively.

    Args:
        optimizer: Optimizer name.  One of ``'adam'``, ``'adamw'``,
            ``'adagrad'``, ``'sgd'``.
        learning_rate: Base learning rate.  Overridden by *schedule* if
            provided.
        max_steps: Maximum number of gradient steps.
        ftol: Convergence tolerance on relative score change.  The
            optimizer stops when the score changes by less than
            ``ftol * score`` over *patience* consecutive steps.
        grad_norm_tol: Stop when the L2 gradient norm falls below this.
        patience: Number of consecutive steps below *ftol* before
            declaring convergence.
        momentum: Momentum coefficient for SGD (ignored for Adam/AdaGrad).
        b1: Exponential decay rate for the first moment (Adam/AdamW).
        b2: Exponential decay rate for the second moment (Adam/AdamW).
        schedule: Learning rate schedule.  ``'cosine'`` uses cosine
            annealing over *max_steps*; ``'exponential'`` uses exponential
            decay with ``decay_rate``.  ``None`` uses a constant LR.
        decay_rate: Decay rate for exponential schedule.
        use_bounds: Whether to clamp parameters to force field bounds
            after each step.
        divergence_factor: Early stopping threshold.  If the score
            exceeds ``divergence_factor * initial_score`` for
            *divergence_patience* consecutive steps, the optimizer halts.
        divergence_patience: Consecutive divergent steps before stopping.
        verbose: Log progress during optimization.
        log_interval: Log every N steps (when *verbose* is ``True``).

    """

    def __init__(
        self,
        optimizer: str = "adam",
        learning_rate: float = 1e-3,
        max_steps: int = 1000,
        ftol: float = 1e-8,
        grad_norm_tol: float = 1e-6,
        patience: int = 20,
        momentum: float = 0.9,
        b1: float = 0.9,
        b2: float = 0.999,
        schedule: str | None = None,
        decay_rate: float = 0.99,
        use_bounds: bool = True,
        divergence_factor: float | None = 3.0,
        divergence_patience: int = 10,
        verbose: bool = True,
        log_interval: int = 50,
    ) -> None:
        if optimizer not in _OPTIMIZER_REGISTRY:
            raise ValueError(f"Unknown optimizer '{optimizer}'. Choose from: {', '.join(sorted(_OPTIMIZER_REGISTRY))}")
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.ftol = ftol
        self.grad_norm_tol = grad_norm_tol
        self.patience = patience
        self.momentum = momentum
        self.b1 = b1
        self.b2 = b2
        self.schedule = schedule
        self.decay_rate = decay_rate
        self.use_bounds = use_bounds
        self.divergence_factor = divergence_factor
        self.divergence_patience = divergence_patience
        self.verbose = verbose
        self.log_interval = log_interval

    def _build_optimizer(self) -> object:
        """Construct the optax optimizer with optional LR schedule.

        Returns:
            An optax ``GradientTransformation``.

        """
        ensure_optax()

        lr = self._build_schedule()

        name = _OPTIMIZER_REGISTRY[self.optimizer_name]
        if name == "adam":
            return optax.adam(learning_rate=lr, b1=self.b1, b2=self.b2)
        if name == "adamw":
            return optax.adamw(learning_rate=lr, b1=self.b1, b2=self.b2)
        if name == "adagrad":
            return optax.adagrad(learning_rate=lr)
        if name == "sgd":
            return optax.sgd(learning_rate=lr, momentum=self.momentum)
        raise ValueError(f"Unhandled optimizer: {name}")  # pragma: no cover

    def _build_schedule(self) -> float | object:
        """Build a learning rate schedule or return constant LR.

        Returns:
            A float (constant LR) or an optax schedule callable.

        """
        ensure_optax()

        if self.schedule is None:
            return self.learning_rate
        if self.schedule == "cosine":
            return optax.cosine_decay_schedule(
                init_value=self.learning_rate,
                decay_steps=self.max_steps,
            )
        if self.schedule == "exponential":
            return optax.exponential_decay(
                init_value=self.learning_rate,
                transition_steps=self.max_steps,
                decay_rate=self.decay_rate,
            )
        raise ValueError(f"Unknown schedule '{self.schedule}'. Choose from: 'cosine', 'exponential', or None.")

    def optimize(self, objective: ObjectiveFunction) -> OptimizationResult:
        """Run the optimization.

        Args:
            objective: Configured objective with forcefield, engine,
                molecules, and reference data.

        Returns:
            Optimization outcome with final parameters and convergence
            history.

        """
        ensure_optax()

        objective.history.clear()
        n_eval_before = objective.n_eval

        ff = objective.forcefield
        initial_full = ff.get_param_vector().copy()
        has_frozen = ff.n_active_params < ff.n_params
        active_indices = np.flatnonzero(ff.active_mask) if has_frozen else None

        if has_frozen:
            x0 = ff.get_active_param_vector().copy()
            bounds = ff.get_active_bounds() if self.use_bounds else None
        else:
            x0 = initial_full.copy()
            bounds = ff.get_bounds() if self.use_bounds else None

        bounds_arr = None if bounds is None else np.asarray(bounds, dtype=np.float64)
        if bounds_arr is not None and bounds_arr.size > 0:
            lower = bounds_arr[:, 0]
            upper = bounds_arr[:, 1]

        def expand_np(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            if not has_frozen:
                return x.copy()
            full = initial_full.copy()
            full[active_indices] = x
            return full

        initial_score = objective(expand_np(x0))

        # Check gradient support — warn if FD fallback will be used
        jac_mode = "analytical"
        try:
            grad_support = objective.per_evaluator_gradient_support()
            has_fd = any(not v for v in grad_support.values())
            if has_fd:
                jac_mode = "auto"
                fd_cats = [k for k, v in grad_support.items() if not v]
                logger.warning(
                    "OptaxOptimizer: FD fallback for categories %s — "
                    "each step will be slow. Consider using ScipyOptimizer "
                    "for non-JAX engines.",
                    fd_cats,
                )
        except Exception:
            jac_mode = "auto"

        opt = self._build_optimizer()

        # When using JaxEngine, route gradients through JaxLoss to avoid
        # materializing the (3N, 3N, n_params) Hessian-parameter Jacobian
        # that causes GPU OOM.  See issue analysis in AGENTS.md §9.
        use_jax_loss = False
        jax_loss = None
        try:
            from q2mm.backends.mm.jax_engine import JaxEngine

            if isinstance(objective.engine, JaxEngine):
                from q2mm.optimizers.jaxloss import JaxLoss

                spec = objective.to_jax_spec()
                jax_loss = JaxLoss(spec, objective.engine, objective.molecules, objective.forcefield)
                use_jax_loss = True
                jac_mode = "jax_loss"
                logger.info("OptaxOptimizer: using JaxLoss gradient path (memory-efficient)")
        except (ImportError, AttributeError):
            pass  # JaxEngine not available or objective lacks .engine

        params = jnp.array(x0, dtype=jnp.float64)
        opt_state = opt.init(params)

        if has_frozen:
            frozen_template = jnp.array(initial_full, dtype=jnp.float64)
            active_indices_jax = jnp.array(active_indices, dtype=jnp.int32)

            def expand_jax(x_active: Any):  # noqa: ANN202
                return frozen_template.at[active_indices_jax].set(x_active)

        else:

            def expand_jax(x_active: Any):  # noqa: ANN202
                return x_active

        method_str = f"optax:{self.optimizer_name}"
        if self.schedule:
            method_str += f"+{self.schedule}"

        if use_jax_loss:
            # Use Python-dispatch value_and_grad to avoid compiling
            # all molecules into one XLA program.  Each per-molecule
            # function is compiled independently.
            if has_frozen:

                def jax_loss_vag(x_active: Any):  # noqa: ANN202
                    loss, full_grad = jax_loss.value_and_grad_jax(expand_jax(x_active))
                    return loss, full_grad[active_indices_jax]

                def jax_loss_eval(x_active: Any):  # noqa: ANN202
                    return float(jax_loss.value_and_grad_jax(expand_jax(x_active))[0])

            else:

                def jax_loss_vag(x_active: Any):  # noqa: ANN202
                    return jax_loss.value_and_grad_jax(x_active)

                def jax_loss_eval(x_active: Any):  # noqa: ANN202
                    return float(jax_loss.value_and_grad_jax(x_active)[0])

        else:
            jax_loss_vag = None
            jax_loss_eval = None

        if self.verbose:
            logger.info(
                "Starting %s optimization: %d active params (%d total), initial score %.6f, lr=%.1e, max_steps=%d",
                method_str,
                ff.n_active_params,
                ff.n_params,
                initial_score,
                self.learning_rate,
                self.max_steps,
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
                n_evaluations=objective.n_eval - n_eval_before,
                initial_params=initial_full,
                final_params=final_params,
                history=list(objective.history),
                method=method_str,
                jac_mode=jac_mode,
                eps=None,
            )

        best_score = initial_score
        best_params = x0.copy()
        converged = False
        message = f"Max steps ({self.max_steps}) reached"
        stall_count = 0
        diverge_count = 0
        prev_score = initial_score

        for step in range(self.max_steps):
            params_np = np.asarray(params, dtype=np.float64)

            # Compute gradient
            if use_jax_loss:
                # Per-molecule Python-dispatch path — each compiled
                # value_and_grad is dispatched independently.
                _pre_loss, grad_jax = jax_loss_vag(params)
                grad_np = np.asarray(grad_jax, dtype=np.float64)
            else:
                grad_np = objective.gradient(expand_np(params_np))
                if has_frozen:
                    grad_np = grad_np[active_indices]
            grad = jnp.array(grad_np, dtype=jnp.float64)

            # Optax update
            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Enforce bounds via clamping
            if bounds_arr is not None and bounds_arr.size > 0:
                params = jnp.clip(params, lower, upper)

            # Evaluate post-update score
            params_np = np.asarray(params, dtype=np.float64)
            if use_jax_loss:
                score = jax_loss_eval(params)
                # Manually track evaluation for JaxLoss path
                objective.n_eval += 1
                objective.history.append(score)
            else:
                score = objective(expand_np(params_np))

            # Track best
            if score < best_score:
                best_score = score
                best_params = params_np.copy()

            # Logging
            if self.verbose and (step + 1) % self.log_interval == 0:
                grad_norm = float(np.linalg.norm(grad_np))
                logger.info(
                    "  step %4d  score %.6f  grad_norm %.2e  best %.6f",
                    step + 1,
                    score,
                    grad_norm,
                    best_score,
                )

            # Convergence: gradient norm
            grad_norm = float(np.linalg.norm(grad_np))
            if grad_norm < self.grad_norm_tol:
                converged = True
                message = f"Converged: gradient norm {grad_norm:.2e} < {self.grad_norm_tol:.2e}"
                break

            # Convergence: score plateau
            if prev_score > 0:
                rel_change = abs(prev_score - score) / prev_score
                if rel_change < self.ftol:
                    stall_count += 1
                    if stall_count >= self.patience:
                        converged = True
                        message = f"Converged: score plateau for {self.patience} steps (rel_change < {self.ftol:.1e})"
                        break
                else:
                    stall_count = 0

            # Divergence detection
            if self.divergence_factor is not None and initial_score > 0:
                threshold = initial_score * self.divergence_factor
                if score > threshold:
                    diverge_count += 1
                    if diverge_count >= self.divergence_patience:
                        message = (
                            f"Abandoned: score {score:.1f} > "
                            f"{threshold:.1f} ({self.divergence_factor:.0f}× initial) "
                            f"for {self.divergence_patience} consecutive steps"
                        )
                        if self.verbose:
                            logger.warning(message)
                        break
                else:
                    diverge_count = 0

            prev_score = score

        # Use best params found during the run
        final_active = best_params
        final_score = best_score
        final_params = expand_np(final_active)

        # Apply final parameters to the forcefield
        objective.forcefield.set_param_vector(final_params)

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d steps, %d evals)",
                "converged" if converged else "stopped",
                initial_score,
                final_score,
                step + 1 if self.max_steps > 0 else 0,
                objective.n_eval - n_eval_before,
            )

        return OptimizationResult(
            success=converged,
            message=message,
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=step + 1 if self.max_steps > 0 else 0,
            n_evaluations=objective.n_eval - n_eval_before,
            initial_params=initial_full if has_frozen else x0,
            final_params=final_params,
            history=list(objective.history),
            method=method_str,
            jac_mode=jac_mode,
            eps=None,
        )
