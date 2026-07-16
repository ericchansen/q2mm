"""Parameter cycling and sensitivity-based selection for Q2MM optimization.

Implements the upstream grad-simp loop: alternate a full-space
gradient/optimizer pass with a subspace Nelder-Mead simplex pass over the
few most-sensitive active parameters.  Consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` +
:class:`~q2mm.models.parameters.ActiveParameterSpace` and returns the one
canonical :class:`~q2mm.models.results.OptimizationResult` (per-cycle
diagnostics are attached as :class:`~q2mm.models.results.StageRecord`
entries).

References:
    Norrby, P.-O.; Liljefors, T. *J. Comput. Chem.* **1998**, 19, 1146.
    Quinn, T.R. et al. *PLOS ONE* **2022**, 17, e0264960.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult, StageRecord
from q2mm.objectives.protocols import ObjectiveEvaluator

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class SensitivityResult:
    """Result of parameter sensitivity analysis via central differentiation."""

    d1: np.ndarray
    d2: np.ndarray
    simp_var: np.ndarray
    ranking: np.ndarray
    metric: str
    n_evals: int


def compute_sensitivity(
    evaluator: ObjectiveEvaluator,
    base_vector: np.ndarray,
    step_sizes: np.ndarray | None = None,
    metric: Literal["simp_var", "abs_d1"] = "simp_var",
    bounds: list[tuple[float, float]] | None = None,
) -> SensitivityResult:
    """Central differentiation to rank parameter sensitivity.

    Evaluates ``evaluator.value`` at ``base_vector`` and at ``base ± h_i``
    for each parameter with a nonzero (bound-shrunk) step.  Parameters with
    zero step are skipped entirely.
    """
    layout = evaluator.plan.layout
    x0 = np.asarray(base_vector, dtype=float)
    n = len(x0)

    if step_sizes is None:
        step_sizes = layout.steps
    step_sizes = np.asarray(step_sizes, dtype=float)
    if len(step_sizes) != n:
        raise ValueError(f"step_sizes length {len(step_sizes)} != param vector length {n}")

    eff_steps = step_sizes.copy()
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds length {len(bounds)} != param vector length {n}")
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        if np.any(lower > upper):
            raise ValueError("bounds must satisfy lower <= upper for every parameter")
        max_symmetric = np.minimum(upper - x0, x0 - lower)
        eff_steps = np.where(max_symmetric > 0, np.minimum(np.abs(eff_steps), max_symmetric), 0.0)

    f0 = float(evaluator.value(x0))
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    active_mask = eff_steps != 0
    n_evals = 1
    for i in range(n):
        if not active_mask[i]:
            continue
        x_fwd = x0.copy()
        x_bwd = x0.copy()
        x_fwd[i] += eff_steps[i]
        x_bwd[i] -= eff_steps[i]
        f_fwd = float(evaluator.value(x_fwd))
        f_bwd = float(evaluator.value(x_bwd))
        n_evals += 2
        d1[i] = (f_fwd - f_bwd) * 0.5
        d2[i] = f_fwd + f_bwd - 2.0 * f0

    with np.errstate(divide="ignore", invalid="ignore"):
        simp_var = np.where(d1 != 0, d2 / (d1**2), np.inf)

    if metric == "simp_var":
        ranking = np.argsort(simp_var)
    elif metric == "abs_d1":
        normalised_d1 = np.where(eff_steps != 0, d1 / eff_steps, 0.0)
        ranking = np.argsort(-np.abs(normalised_d1))
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return SensitivityResult(d1=d1, d2=d2, simp_var=simp_var, ranking=ranking, metric=metric, n_evals=n_evals)


class OptimizationLoop:
    """grad-simp cycling loop returning the canonical OptimizationResult."""

    def __init__(
        self,
        evaluator: ObjectiveEvaluator,
        space: ActiveParameterSpace,
        *,
        max_params: int = 3,
        convergence: float = 0.01,
        max_cycles: int = 10,
        full_method: str = "L-BFGS-B",
        simp_method: str = "Nelder-Mead",
        full_maxiter: int = 200,
        simp_maxiter: int = 200,
        sensitivity_metric: Literal["simp_var", "abs_d1"] = "simp_var",
        eps: float = 1e-3,
        verbose: bool = True,
    ) -> None:
        self.evaluator = evaluator
        self.space = space
        self.max_params = max_params
        self.convergence = convergence
        self.max_cycles = max_cycles
        self.full_method = full_method
        self.simp_method = simp_method
        self.full_maxiter = full_maxiter
        self.simp_maxiter = simp_maxiter
        self.sensitivity_metric = sensitivity_metric
        self.eps = eps
        self.verbose = verbose

    def run(self) -> OptimizationResult:
        """Execute the grad-simp cycling loop."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        evaluator = self.evaluator
        layout = evaluator.plan.layout
        space = self.space
        current_full = np.array(space.baseline, dtype=float)
        initial_params = current_full.copy()

        prev_on_error = getattr(evaluator, "on_error", None)
        if prev_on_error is not None:
            evaluator.on_error = "penalty"  # type: ignore[attr-defined]

        initial_score = float(evaluator.value(current_full))
        n_eval_start = evaluator.n_evaluations
        n_params = space.n_full
        fingerprint = space.layout.fingerprint
        last_grad_mode = "none"
        last_fd_step: float | None = None
        cycle_scores: list[float] = [initial_score]
        stages: list[StageRecord] = []
        converged = False

        use_optax = self.full_method.startswith("optax:")
        use_jaxopt = self.full_method.startswith("jaxopt:")
        use_basinhopping = self.full_method.startswith("basinhopping")
        use_multistart = self.full_method.startswith("multi:")

        candidate_step_sizes = layout.steps.copy()
        frozen_mask = np.ones(len(layout), dtype=bool)
        frozen_mask[space.active_indices] = False
        candidate_step_sizes[frozen_mask] = 0.0
        full_bounds = layout.bounds
        param_labels = [kind.value for kind in layout.kinds]

        try:
            if self.verbose:
                logger.info("OptimizationLoop: initial score = %.6f, max_params = %d", initial_score, self.max_params)

            for cycle in range(1, self.max_cycles + 1):
                score_before = cycle_scores[-1]
                cycle_eval_start = evaluator.n_evaluations
                cycle_space = space.with_baseline(current_full)

                full_opt = self._build_full_optimizer(
                    use_optax, use_jaxopt, use_basinhopping, use_multistart, ScipyOptimizer
                )
                full_result = full_opt.optimize(evaluator, cycle_space)
                score_after_grad = full_result.final_score
                current_full = np.asarray(full_result.final_params, dtype=float)
                last_grad_mode = full_result.gradient_mode
                last_fd_step = full_result.fd_step

                sens = compute_sensitivity(
                    evaluator,
                    current_full,
                    step_sizes=candidate_step_sizes,
                    metric=self.sensitivity_metric,
                    bounds=[tuple(b) for b in full_bounds],
                )

                n_selected = min(self.max_params, space.n_active)
                if n_selected < 1:
                    raise ValueError(
                        f"OptimizationLoop requires at least one active parameter, "
                        f"but max_params={self.max_params} and space.n_active={space.n_active}."
                    )
                allowed = frozenset(int(i) for i in space.active_indices)
                selected = [int(i) for i in sens.ranking if int(i) in allowed][:n_selected]
                if len(selected) != n_selected:
                    raise RuntimeError(
                        f"Sensitivity ranking produced {len(selected)} selectable parameters; expected {n_selected}."
                    )

                simp_space = space.with_baseline(current_full).with_active_indices(selected)
                simp_result = ScipyOptimizer(
                    method=self.simp_method,
                    maxiter=self.simp_maxiter,
                    eps=self.eps,
                    verbose=False,
                ).optimize(
                    evaluator,
                    simp_space,
                )
                score_after_simp = simp_result.final_score
                if score_after_simp < score_after_grad:
                    current_full = np.asarray(simp_result.final_params, dtype=float)
                else:
                    score_after_simp = score_after_grad

                cycle_scores.append(score_after_simp)
                stages.append(
                    StageRecord(
                        name=f"cycle-{cycle}",
                        n_params=n_params,
                        layout_fingerprint=fingerprint,
                        initial_score=score_before,
                        final_score=score_after_simp,
                        n_iterations=full_result.n_iterations + simp_result.n_iterations,
                        n_evaluations=evaluator.n_evaluations - cycle_eval_start,
                        converged=full_result.success and simp_result.success,
                        message=(
                            f"full={full_result.message}; subspace={simp_result.message}; grad={score_after_grad:.6g}"
                        ),
                        gradient_mode=last_grad_mode,
                        fd_step=last_fd_step,
                        notes={
                            "selected_indices": selected,
                            "selected_labels": [f"{param_labels[i]}[{i}]" for i in selected],
                            "sensitivity_ranking": sens.ranking.tolist(),
                            "score_after_grad": score_after_grad,
                            "score_after_subspace": simp_result.final_score,
                        },
                    )
                )

                change = (score_before - score_after_simp) / score_before if score_before > 0 else 0.0
                if self.verbose:
                    logger.info("  Cycle %d: %.2f%% improvement", cycle, change * 100)
                if 0 <= change < self.convergence:
                    converged = True
                    break
        finally:
            if prev_on_error is not None:
                evaluator.on_error = prev_on_error  # type: ignore[attr-defined]

        final_score = cycle_scores[-1]
        n_cycles = len(cycle_scores) - 1
        return OptimizationResult(
            success=converged,
            message="converged" if converged else f"max cycles ({self.max_cycles}) reached",
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=n_cycles,
            n_evaluations=evaluator.n_evaluations - n_eval_start,
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=initial_params,
            final_params=current_full,
            history=tuple(cycle_scores),
            method=f"grad-simp({self.full_method})",
            gradient_mode=last_grad_mode,
            fd_step=last_fd_step,
            stages=tuple(stages),
        )

    def _build_full_optimizer(
        self,
        use_optax: bool,
        use_jaxopt: bool,
        use_basinhopping: bool,
        use_multistart: bool,
        scipy_cls: type,
    ) -> object:
        if use_optax:
            from q2mm.optimizers.optax import OptaxOptimizer

            optax_spec = self.full_method.split(":", 1)[1]
            if "+" in optax_spec:
                optax_name, schedule = optax_spec.split("+", 1)
            else:
                optax_name, schedule = optax_spec, None
            return OptaxOptimizer(optimizer=optax_name, max_steps=self.full_maxiter, schedule=schedule, verbose=False)
        if use_jaxopt:
            from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

            return JaxOptOptimizer(method=self.full_method.split(":", 1)[1], maxiter=self.full_maxiter, verbose=False)
        if use_basinhopping:
            from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

            bh_spec = self.full_method.split(":", 1)[1].strip() or "L-BFGS-B" if ":" in self.full_method else "L-BFGS-B"
            return BasinHoppingOptimizer(local_method=bh_spec, local_maxiter=self.full_maxiter, verbose=False)
        if use_multistart:
            from q2mm.optimizers.multistart import MultiStartOptimizer

            ms_spec = self.full_method.split(":", 1)[1]
            inner_opt = scipy_cls(method=ms_spec, maxiter=self.full_maxiter, eps=self.eps, verbose=False)
            return MultiStartOptimizer(optimizer=inner_opt, n_starts=5, verbose=False)
        return scipy_cls(method=self.full_method, maxiter=self.full_maxiter, eps=self.eps, verbose=False)
