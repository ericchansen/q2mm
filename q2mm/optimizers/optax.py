"""Optax-based optimizer for Q2MM force field parameterization.

Iterative gradient-based optimizers (Adam, AdaGrad, SGD) from optax.  The
explicit training loop drives the
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` via its
``value_and_gradient`` (full-vector value + gradient, packed to the active
subset) — analytical for a
:class:`~q2mm.objectives.jax.JaxObjectiveExecutor` or an analytical
:class:`~q2mm.objectives.python.PythonObjectiveExecutor`, or explicit
finite differences for a finite-difference executor.  The evaluator must
declare a non-``NONE`` gradient mode.
"""

from __future__ import annotations

import logging
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator, ObjectiveGradientError

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace

logger = logging.getLogger(__name__)

_HAS_OPTAX = find_spec("optax") is not None

optax = None
jnp = None


def ensure_optax() -> None:
    """Lazily import optax and jax.numpy, with float64 enabled."""
    global optax, jnp  # noqa: PLW0603
    if optax is not None:
        return
    if not _HAS_OPTAX:
        raise ImportError("optax is required for OptaxOptimizer. Install it with: pip install q2mm[jax]")
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax(engine_name="OptaxOptimizer")
    import optax as _optax
    from q2mm.backends.mm._jax_common import jnp as _jnp

    optax = _optax
    jnp = _jnp


_OPTIMIZER_REGISTRY: dict[str, str] = {"adam": "adam", "adamw": "adamw", "adagrad": "adagrad", "sgd": "sgd"}


class OptaxOptimizer:
    """Force field optimizer using optax gradient transformations."""

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
        ensure_optax()
        if self.schedule is None:
            return self.learning_rate
        if self.schedule == "cosine":
            return optax.cosine_decay_schedule(init_value=self.learning_rate, decay_steps=self.max_steps)
        if self.schedule == "exponential":
            return optax.exponential_decay(
                init_value=self.learning_rate, transition_steps=self.max_steps, decay_rate=self.decay_rate
            )
        raise ValueError(f"Unknown schedule '{self.schedule}'. Choose from: 'cosine', 'exponential', or None.")

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the optimization and return the canonical result."""
        ensure_optax()
        if evaluator.gradient_mode is GradientMode.NONE:
            raise ObjectiveGradientError(
                f"OptaxOptimizer requires an evaluator with gradients, but {type(evaluator).__name__} "
                "declares gradient_mode=none. Select an analytical/FD executor."
            )
        gradient_mode_str = evaluator.gradient_mode.value
        # Explicit FD provenance: the evaluator's own FD step in FD mode, else None.
        fd_step = evaluator.finite_difference_step

        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        baseline = np.array(space.baseline, dtype=float)
        n_eval_before = evaluator.n_evaluations
        hist_before = len(evaluator.history)
        x0 = space.pack(baseline)

        def value_only(x_active: np.ndarray) -> float:
            return evaluator.value(space.expand(x_active, base=baseline))

        def value_and_grad(x_active: np.ndarray) -> tuple[float, np.ndarray]:
            val, full_grad = evaluator.value_and_gradient(space.expand(x_active, base=baseline))
            return val, space.pack(full_grad)

        initial_score = value_only(x0)

        bounds_arr = space.bounds if self.use_bounds else None
        lower = upper = None
        if bounds_arr is not None and bounds_arr.size > 0:
            lower = bounds_arr[:, 0]
            upper = bounds_arr[:, 1]

        opt = self._build_optimizer()
        params = jnp.array(x0, dtype=jnp.float64)
        opt_state = opt.init(params)

        method_str = f"optax:{self.optimizer_name}"
        if self.schedule:
            method_str += f"+{self.schedule}"

        if self.verbose:
            logger.info(
                "Starting %s: %d active params (%d total), initial score %.6f, lr=%.1e, max_steps=%d",
                method_str,
                space.n_active,
                space.n_full,
                initial_score,
                self.learning_rate,
                self.max_steps,
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
                method=method_str,
                gradient_mode=gradient_mode_str,
                fd_step=fd_step,
            )

        best_score = initial_score
        best_params = x0.copy()
        converged = False
        message = f"Max steps ({self.max_steps}) reached"
        stall_count = 0
        diverge_count = 0
        prev_score = initial_score
        step = 0

        for step in range(self.max_steps):
            params_np = np.asarray(params, dtype=np.float64)
            score, grad_active = value_and_grad(params_np)
            grad_np = np.asarray(grad_active, dtype=np.float64)
            grad = jnp.array(grad_np, dtype=jnp.float64)

            updates, opt_state = opt.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            if lower is not None:
                params = jnp.clip(params, lower, upper)

            if score < best_score:
                best_score = score
                # ``score``/``grad`` were evaluated at the PRE-update iterate
                # (``params_np``); store that same vector so best_score and
                # best_params identify the identical evaluated point (not the
                # post-update, possibly-overshot iterate).
                best_params = params_np.copy()

            grad_norm = float(np.linalg.norm(grad_np))
            if self.verbose and (step + 1) % self.log_interval == 0:
                logger.info("  step %4d  score %.6f  grad_norm %.2e  best %.6f", step + 1, score, grad_norm, best_score)

            if grad_norm < self.grad_norm_tol:
                converged = True
                message = f"Converged: gradient norm {grad_norm:.2e} < {self.grad_norm_tol:.2e}"
                break

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

            if self.divergence_factor is not None and initial_score > 0:
                threshold = initial_score * self.divergence_factor
                if score > threshold:
                    diverge_count += 1
                    if diverge_count >= self.divergence_patience:
                        message = (
                            f"Abandoned: score {score:.1f} > {threshold:.1f} "
                            f"({self.divergence_factor:.0f}× initial) for {self.divergence_patience} steps"
                        )
                        if self.verbose:
                            logger.warning(message)
                        break
                else:
                    diverge_count = 0

            prev_score = score

        final_params = space.expand(best_params, base=baseline)
        final_score = float(value_only(best_params))

        if self.verbose:
            logger.info(
                "Optimization %s: score %.6f → %.6f (%d steps)",
                "converged" if converged else "stopped",
                initial_score,
                final_score,
                step + 1,
            )

        return OptimizationResult(
            success=converged,
            message=message,
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=step + 1 if self.max_steps > 0 else 0,
            n_evaluations=evaluator.n_evaluations - n_eval_before,
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=baseline,
            final_params=final_params,
            history=evaluator.history[hist_before:] or (initial_score, final_score),
            method=method_str,
            gradient_mode=gradient_mode_str,
            fd_step=fd_step,
        )
