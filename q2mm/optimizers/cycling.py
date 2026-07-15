"""Parameter cycling and sensitivity-based selection for Q2MM optimization.

Implements the upstream Q2MM grad-simp optimization loop, adapted for our
scipy-based optimizer architecture.  The key insight from Norrby & Liljefors
(1998) and Quinn et al. (2022) is that the Nelder-Mead simplex converges
well on ≤40 parameters but fails on larger sets; the upstream code therefore
selected only the 2-4 *most sensitive* parameters for each simplex pass.

Sensitivity is measured by central differentiation of all parameters:

- d1 = (f(x+h) - f(x-h)) / 2      (1st derivative, unnormalised)
- d2 = f(x+h) + f(x-h) - 2·f(x)   (2nd derivative, unnormalised)
- simp_var = d2 / d1²               (upstream selection metric)

Low ``simp_var`` identifies parameters where the objective is steep but
shallow-bottomed — gradient methods struggle here, but simplex can make
progress.  The upstream authors noted this criterion is imperfect
(``simplex.py:414``: "Sorting based upon the 2nd derivative isn't such
a good criterion"); we therefore also support ``|d1|`` (absolute first
derivative) as an alternative.

References
----------
- Norrby, P.-O.; Liljefors, T. *J. Comput. Chem.* **1998**, 19, 1146-1166.
- Hansen, E.C. "Development and Applications of Q2MM" (PhD dissertation,
  University of Notre Dame, 2016).
- Quinn, T.R. et al. *PLOS ONE* **2022**, 17, e0264960.
- Upstream code: github.com/nsf-c-cas/q2mm-2 (``simplex.py``, ``opt.py``).

"""

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.optimizers._metrics import fractional_improvement
from q2mm.optimizers.objective import ObjectiveFunction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class SensitivityResult:
    """Result of parameter sensitivity analysis via central differentiation.

    Attributes:
        d1 (np.ndarray): First derivative (unnormalised) for each parameter.
        d2 (np.ndarray): Second derivative (unnormalised) for each parameter.
        simp_var (np.ndarray): Upstream "simplex variable":
            ``d2 / d1**2`` for each parameter.
        ranking (np.ndarray): Parameter indices sorted by sensitivity
            (most sensitive first).
        metric (str): Which metric was used for ranking.
        n_evals (int): Number of objective function evaluations performed.

    """

    d1: np.ndarray
    d2: np.ndarray
    simp_var: np.ndarray
    ranking: np.ndarray
    metric: str
    n_evals: int


@dataclass
class LoopResult:
    """Result of an :class:`OptimizationLoop` run.

    Attributes:
        success (bool): ``True`` if converged before hitting *max_cycles*.
        initial_score (float): Objective value before any optimisation.
        final_score (float): Objective value after the last cycle.
        n_cycles (int): Number of grad-simp cycles completed.
        initial_params (np.ndarray): Full-length parameter vector before
            any optimisation.
        final_params (np.ndarray): Full-length parameter vector after the
            last cycle.  ``ObjectiveFunction.forcefield`` is never
            mutated in place — materialize the optimised force field via
            ``objective.layout.replace(objective.forcefield, result.final_params)``.
        n_eval (int): Total objective function evaluations across all
            cycles (grad + sensitivity + simp).
        cycle_scores (list[float]): Objective value at the end of each
            cycle.
        selected_indices (list[list[int]]): Parameter indices selected
            for each simplex pass.
        sensitivity_results (list[SensitivityResult]): Full sensitivity
            analysis for each cycle.
        message (str): Human-readable summary.

    """

    success: bool
    initial_score: float
    final_score: float
    n_cycles: int
    initial_params: np.ndarray
    final_params: np.ndarray
    n_eval: int = 0
    cycle_scores: list[float] = field(default_factory=list)
    selected_indices: list[list[int]] = field(default_factory=list)
    sensitivity_results: list[SensitivityResult] = field(default_factory=list)
    message: str = ""

    @property
    def improvement(self) -> float:
        """Fractional improvement: ``(initial - final) / initial``.

        Returns:
            float: Fractional improvement, or 0.0 if ``initial_score``
                is zero.

        """
        return fractional_improvement(self.initial_score, self.final_score)

    def summary(self) -> str:
        """Human-readable summary string.

        Returns:
            str: Multi-line summary of the loop result.

        """
        lines = [
            f"OptimizationLoop: {'converged' if self.success else 'max cycles reached'}",
            f"  Cycles:      {self.n_cycles}",
            f"  Evaluations: {self.n_eval}",
            f"  Score:       {self.initial_score:.6f} → {self.final_score:.6f}",
            f"  Improvement: {self.improvement:.2%}",
        ]
        if self.message:
            lines.append(f"  Message:     {self.message}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# SubspaceObjective — project onto a parameter subset
# ---------------------------------------------------------------------------


class SubspaceObjective:
    """Wraps an :class:`ObjectiveFunction` to optimise a parameter subset.

    The wrapper accepts a *sub-vector* of length ``len(active_indices)``
    and maps it into the full parameter vector before delegating to the
    underlying objective.  This lets :func:`scipy.optimize.minimize` run
    Nelder-Mead (or any other method) on just the selected parameters
    while the rest stay fixed.

    Args:
        objective (ObjectiveFunction): The full objective function.
        active_indices (list[int] | np.ndarray): Indices into the full
            parameter vector that are active.
        full_vector (np.ndarray): The current full parameter vector
            (inactive params are taken from this snapshot).

    """

    def __init__(
        self,
        objective: ObjectiveFunction,
        active_indices: list[int] | np.ndarray,
        full_vector: np.ndarray,
    ) -> None:
        """Initialize the subspace objective wrapper.

        Args:
            objective (ObjectiveFunction): The full objective function.
            active_indices (list[int] | np.ndarray): Indices into the
                full parameter vector that are active.
            full_vector (np.ndarray): The current full parameter vector.

        Raises:
            ValueError: If ``active_indices`` is empty.

        """
        self.objective = objective
        self.active_indices = np.asarray(active_indices, dtype=int)
        self._base_vector = full_vector.copy()
        if len(self.active_indices) == 0:
            raise ValueError("active_indices must not be empty")

    def build_full_vector(self, sub_vector: np.ndarray) -> np.ndarray:
        """Map a sub-vector back into the full parameter vector.

        Args:
            sub_vector (np.ndarray): Values for the active parameters.

        Returns:
            np.ndarray: Full parameter vector with active slots replaced.

        """
        full = self._base_vector.copy()
        full[self.active_indices] = sub_vector
        return full

    def __call__(self, sub_vector: np.ndarray) -> float:
        """Evaluate the objective on *sub_vector*.

        Args:
            sub_vector (np.ndarray): Values for the active parameters.

        Returns:
            float: Objective function score.

        """
        return float(self.objective(self.build_full_vector(sub_vector)))

    def residuals(self, sub_vector: np.ndarray) -> np.ndarray:
        """Return the residual vector (for ``least_squares``).

        Args:
            sub_vector (np.ndarray): Values for the active parameters.

        Returns:
            np.ndarray: Weighted residual vector.

        """
        return self.objective.residuals(self.build_full_vector(sub_vector))

    def get_initial_vector(self) -> np.ndarray:
        """Return the sub-vector corresponding to current active parameters.

        Returns:
            np.ndarray: Copy of active parameter values from the base
                vector.

        """
        return self._base_vector[self.active_indices].copy()

    def get_bounds(self) -> list[tuple[float, float]]:
        """Bounds for the active parameters only.

        Returns:
            list[tuple[float, float]]: Lower/upper bound pairs for each
                active parameter.

        """
        all_bounds = self.objective.layout.bounds
        return [tuple(all_bounds[i]) for i in self.active_indices]


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def compute_sensitivity(
    objective: ObjectiveFunction,
    step_sizes: np.ndarray | None = None,
    metric: Literal["simp_var", "abs_d1"] = "simp_var",
    bounds: list[tuple[float, float]] | None = None,
) -> SensitivityResult:
    """Central differentiation to rank parameter sensitivity.

    For each parameter *i*, evaluates ``f(x + h_i e_i)`` and
    ``f(x - h_i e_i)`` and computes:

    - ``d1_i = (f_fwd - f_bwd) / 2``
    - ``d2_i = f_fwd + f_bwd - 2·f_0``
    - ``simp_var_i = d2_i / d1_i²``   (upstream criterion)

    The ranking is by *ascending* ``simp_var`` (lowest = most suitable
    for simplex) or *descending* ``|d1|`` (largest gradient = most
    sensitive), depending on *metric*.

    When *bounds* are provided, step sizes are shrunk so that
    ``x0 ± h_eff`` stays within ``[lower, upper]`` for each parameter.
    Parameters where no symmetric step is possible (i.e. ``x0`` is at a
    bound) are skipped and assigned ``inf`` simp_var / zero d1.

    When the backend declares batched-energy support and all
    references are energy-only, all required ``2K + 1`` evaluations
    (for the ``K`` parameters with nonzero step sizes, ``K ≤ N``) are
    vectorised into a single call where possible (e.g. ``jax.vmap`` on GPU).

    Cost: ``2K + 1`` objective evaluations (1 baseline + 2 per *active*
    parameter for central differentiation; in the worst case ``K = N``,
    giving ``2N + 1``).

    Args:
        objective (ObjectiveFunction): Must already be evaluable (engine
            and molecules configured).
        step_sizes (np.ndarray | None): Per-parameter step sizes.
            Defaults to :attr:`ParameterLayout.steps` (from
            ``objective.layout``) if not provided.  Callers wanting to
            restrict candidates to a subset (e.g. only active
            parameters) should zero out the step for any index to
            exclude — a zero step is treated as "not a candidate" and
            skipped entirely (no evaluation cost, ``simp_var = inf``).
        metric (str): Ranking criterion — ``"simp_var"`` or
            ``"abs_d1"``.
        bounds (list[tuple[float, float]] | None): Per-parameter
            ``(lower, upper)`` bounds.  When provided, step sizes are
            shrunk to the largest symmetric step that stays in bounds.

    Returns:
        SensitivityResult: Derivatives, rankings, and evaluation count.

    Raises:
        ValueError: If ``step_sizes`` length does not match the parameter
            vector, or if *metric* is unknown.

    """
    layout = objective.layout
    x0 = layout.vector(objective.forcefield)
    n = len(x0)

    if step_sizes is None:
        step_sizes = layout.steps
    step_sizes = np.asarray(step_sizes, dtype=float)
    if len(step_sizes) != n:
        raise ValueError(f"step_sizes length {len(step_sizes)} != param vector length {n}")

    # Compute effective step sizes: shrink to stay within bounds
    eff_steps = step_sizes.copy()
    if bounds is not None:
        if len(bounds) != n:
            raise ValueError(f"bounds length {len(bounds)} != param vector length {n}")
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        if np.any(lower > upper):
            raise ValueError("bounds must satisfy lower <= upper for every parameter")
        # Largest symmetric step that keeps x0 ± h within [lower, upper]
        max_up = upper - x0
        max_down = x0 - lower
        max_symmetric = np.minimum(max_up, max_down)
        # Shrink step to fit; zero out if no room
        eff_steps = np.where(
            max_symmetric > 0,
            np.minimum(np.abs(eff_steps), max_symmetric),
            0.0,
        )
        n_shrunk = int(np.sum((eff_steps < np.abs(step_sizes)) & (eff_steps > 0)))
        n_skipped = int(np.sum((step_sizes != 0) & (eff_steps == 0)))
        if n_shrunk > 0 or n_skipped > 0:
            logger.info(
                "compute_sensitivity: %d params shrunk to bounds, %d skipped (at bounds)",
                n_shrunk,
                n_skipped,
            )

    # Build the full (2K+1) parameter matrix: [x0, x0+h0, x0-h0, x0+h1, x0-h1, ...]
    param_rows = [x0]
    active_mask = eff_steps != 0
    for i in range(n):
        if not active_mask[i]:
            continue
        x_fwd = x0.copy()
        x_bwd = x0.copy()
        x_fwd[i] += eff_steps[i]
        x_bwd[i] -= eff_steps[i]
        param_rows.append(x_fwd)
        param_rows.append(x_bwd)

    param_matrix = np.array(param_rows)  # (2K+1, n) where K = active params
    n_evals = len(param_matrix)

    # Evaluate all parameter vectors — batched when possible
    all_scores = objective.batched_scores(param_matrix)

    # Unpack: first row is baseline, then pairs of (fwd, bwd)
    f0 = all_scores[0]
    d1 = np.zeros(n)
    d2 = np.zeros(n)
    pair_idx = 1
    for i in range(n):
        if not active_mask[i]:
            continue
        f_fwd = all_scores[pair_idx]
        f_bwd = all_scores[pair_idx + 1]
        pair_idx += 2
        d1[i] = (f_fwd - f_bwd) * 0.5
        d2[i] = f_fwd + f_bwd - 2.0 * f0

    # Compute simp_var, guarding against zero d1
    with np.errstate(divide="ignore", invalid="ignore"):
        simp_var = np.where(d1 != 0, d2 / (d1**2), np.inf)

    # Ranking
    if metric == "simp_var":
        ranking = np.argsort(simp_var)  # ascending: lowest = most suitable
    elif metric == "abs_d1":
        # Normalise by effective step size so ranking reflects true gradient
        # magnitude rather than being biased by per-type step size differences.
        normalised_d1 = np.where(eff_steps != 0, d1 / eff_steps, 0.0)
        ranking = np.argsort(-np.abs(normalised_d1))  # descending: largest = most sensitive
    else:
        raise ValueError(f"Unknown metric: {metric!r}")

    return SensitivityResult(
        d1=d1,
        d2=d2,
        simp_var=simp_var,
        ranking=ranking,
        metric=metric,
        n_evals=n_evals,
    )


# ---------------------------------------------------------------------------
# OptimizationLoop — grad-simp cycling
# ---------------------------------------------------------------------------


class OptimizationLoop:
    """grad-simp cycling loop inspired by the upstream Q2MM workflow.

    Each cycle:
      1. **Full-space pass** — run ``full_method`` (default L-BFGS-B) on
         all *active* parameters (respecting ``space``).
      2. **Sensitivity analysis** — central differentiation to rank the
         active parameters (frozen parameters are never candidates).
      3. **Subspace simplex** — run ``simp_method`` (default Nelder-Mead)
         on only the top ``max_params`` most sensitive active parameters
         — a temporary subspace derived from ``space``.
      4. **Convergence check** — stop if the fractional improvement in the
         objective falls below ``convergence``.

    Args:
        objective (ObjectiveFunction): The objective function to minimise.
        space (ActiveParameterSpace): The active/frozen projection over
            ``objective.layout``.  Only active parameters are ever
            selected as full-space or subspace candidates;
            ``objective.forcefield`` is never mutated — read the
            optimised parameters from :attr:`LoopResult.final_params`.
        max_params (int): Number of parameters per simplex pass (upstream
            default: 3).
        convergence (float): Stop when ``(score_before - score_after) /
            score_before < convergence``.
        max_cycles (int): Maximum number of grad-simp cycles.
        full_method (str): Scipy method for the full-space pass.
        simp_method (str): Scipy method for the subspace pass.
        full_maxiter (int): Max iterations for the full-space pass.
        simp_maxiter (int): Max iterations for the subspace pass.
        sensitivity_metric (str): How to rank parameters for selection
            — ``"simp_var"`` or ``"abs_d1"``.
        eps (float): Finite-difference step size for the full-space
            optimizer.
        verbose (bool): Whether to log progress.

    References:
        Upstream ``loop.py:Loop.opt_loop()`` and
        ``simplex.py:Simplex.run()``.

    """

    def __init__(
        self,
        objective: ObjectiveFunction,
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
        full_jac: str | None = None,
        verbose: bool = True,
    ) -> None:
        """Initialize the optimization loop.

        Args:
            objective (ObjectiveFunction): The objective function to
                minimise.
            space (ActiveParameterSpace): The active/frozen projection
                over ``objective.layout``.
            max_params (int): Number of parameters per simplex pass.
            convergence (float): Fractional improvement threshold.
            max_cycles (int): Maximum number of grad-simp cycles.
            full_method (str): Scipy method for the full-space pass.
            simp_method (str): Scipy method for the subspace pass.
            full_maxiter (int): Max iterations for the full-space pass.
            simp_maxiter (int): Max iterations for the subspace pass.
            sensitivity_metric (str): Parameter ranking criterion.
            eps (float): Finite-difference step size.
            full_jac (str | None): Jacobian strategy for the full-space
                pass.  ``None`` uses scipy finite differences;
                ``'analytical'`` uses :meth:`ObjectiveFunction.gradient`.
            verbose (bool): Whether to log progress.

        """
        self.objective = objective
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
        self.full_jac = full_jac
        self.verbose = verbose

    def run(self) -> LoopResult:
        """Execute the grad-simp cycling loop.

        Returns:
            LoopResult: Contains convergence status, per-cycle scores,
                and selected parameter indices.

        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        layout = self.objective.layout
        space = self.space
        original_forcefield = self.objective.forcefield
        current_full = layout.vector(original_forcefield)
        initial_params = current_full.copy()

        # Enable penalty fallback for eigendecomposition failures during
        # optimisation.  This lets the optimizer retreat from pathological
        # parameter regions instead of crashing the entire run.
        prev_on_error = self.objective.on_error
        self.objective.on_error = "penalty"

        initial_score = float(self.objective(current_full))

        n_eval_start = self.objective.n_eval  # track cumulative evals
        cycle_scores: list[float] = [initial_score]
        selected_indices: list[list[int]] = []
        sensitivity_results: list[SensitivityResult] = []
        converged = False

        # Detect optax: prefix for full-space optimizer
        use_optax = self.full_method.startswith("optax:")
        use_jaxopt = self.full_method.startswith("jaxopt:")
        use_basinhopping = self.full_method.startswith("basinhopping")
        use_multistart = self.full_method.startswith("multi:")

        # Frozen parameters are never sensitivity/simplex candidates:
        # zero out their step size so compute_sensitivity() skips them
        # entirely (no evaluation cost, simp_var=inf, sorted last).
        candidate_step_sizes = layout.steps.copy()
        frozen_mask = np.ones(len(layout), dtype=bool)
        frozen_mask[space.active_indices] = False
        candidate_step_sizes[frozen_mask] = 0.0
        full_bounds = layout.bounds
        param_labels = [kind.value for kind in layout.kinds]

        try:
            if self.verbose:
                logger.info(
                    "OptimizationLoop: initial score = %.6f, max_params = %d",
                    initial_score,
                    self.max_params,
                )

            for cycle in range(1, self.max_cycles + 1):
                score_before = cycle_scores[-1]

                # Sync objective.forcefield and derive this cycle's active
                # space from the current full vector before the
                # full-space pass reads them.
                self.objective.forcefield = layout.replace(original_forcefield, current_full)
                cycle_space = space.with_baseline(current_full)

                # --- Step 1: Full-space optimisation (active params only) ---
                if use_optax:
                    from q2mm.optimizers.optax import OptaxOptimizer

                    optax_spec = self.full_method.split(":", 1)[1]
                    if "+" in optax_spec:
                        optax_name, schedule = optax_spec.split("+", 1)
                    else:
                        optax_name = optax_spec
                        schedule = None
                    full_opt = OptaxOptimizer(
                        optimizer=optax_name,
                        max_steps=self.full_maxiter,
                        schedule=schedule,
                        verbose=False,
                    )
                    full_result = full_opt.optimize(self.objective, cycle_space)
                elif use_jaxopt:
                    from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

                    jaxopt_method = self.full_method.split(":", 1)[1]
                    full_opt = JaxOptOptimizer(
                        method=jaxopt_method,
                        maxiter=self.full_maxiter,
                        verbose=False,
                    )
                    full_result = full_opt.optimize(self.objective, cycle_space)
                elif use_basinhopping:
                    from q2mm.optimizers.basinhopping import BasinHoppingOptimizer

                    if ":" in self.full_method:
                        bh_spec = self.full_method.split(":", 1)[1].strip() or "L-BFGS-B"
                    else:
                        bh_spec = "L-BFGS-B"
                    full_opt = BasinHoppingOptimizer(
                        local_method=bh_spec,
                        local_maxiter=self.full_maxiter,
                        jac=self.full_jac,
                        verbose=False,
                    )
                    full_result = full_opt.optimize(self.objective, cycle_space)
                elif use_multistart:
                    from q2mm.optimizers.multistart import MultiStartOptimizer

                    ms_spec = self.full_method.split(":", 1)[1]
                    inner_opt = ScipyOptimizer(
                        method=ms_spec,
                        maxiter=self.full_maxiter,
                        eps=self.eps,
                        jac=self.full_jac,
                        verbose=False,
                    )
                    full_opt = MultiStartOptimizer(
                        optimizer=inner_opt,
                        n_starts=5,
                        verbose=False,
                    )
                    full_result = full_opt.optimize(self.objective, cycle_space)
                else:
                    full_opt = ScipyOptimizer(
                        method=self.full_method,
                        maxiter=self.full_maxiter,
                        eps=self.eps,
                        jac=self.full_jac,
                        verbose=False,
                    )
                    full_result = full_opt.optimize(self.objective, cycle_space)
                score_after_grad = full_result.final_score
                current_full = full_result.final_params

                if self.verbose:
                    logger.info(
                        "  Cycle %d GRAD (%s): %.6f → %.6f",
                        cycle,
                        self.full_method,
                        score_before,
                        score_after_grad,
                    )

                # Sync objective.forcefield to the post-grad state before
                # sensitivity analysis reads it.
                self.objective.forcefield = layout.replace(original_forcefield, current_full)

                # --- Step 2: Sensitivity analysis (active params only) ---
                sens = compute_sensitivity(
                    self.objective,
                    step_sizes=candidate_step_sizes,
                    metric=self.sensitivity_metric,
                    bounds=full_bounds,
                )
                sensitivity_results.append(sens)

                # Select top max_params; ensure we have at least one active parameter
                n_selected = min(self.max_params, space.n_active)
                if n_selected < 1:
                    raise ValueError(
                        f"OptimizationLoop requires at least one active parameter, "
                        f"but max_params={self.max_params} and space.n_active={space.n_active}."
                    )
                active = sens.ranking[:n_selected].tolist()
                selected_indices.append(active)

                if self.verbose:
                    selected_labels = [f"{param_labels[i]}[{i}]" for i in active]
                    logger.info(
                        "  Cycle %d sensitivity (%s): selected %s",
                        cycle,
                        self.sensitivity_metric,
                        ", ".join(selected_labels),
                    )

                # --- Step 3: Subspace simplex ---
                sub_obj = SubspaceObjective(self.objective, active, current_full)

                from scipy import optimize as sp_opt

                sub_x0 = sub_obj.get_initial_vector()
                sub_bounds = sub_obj.get_bounds()

                scipy_options: dict = {"maxiter": self.simp_maxiter}
                if self.simp_method == "Nelder-Mead":
                    scipy_options["xatol"] = 1e-6
                    scipy_options["fatol"] = 1e-8

                # Pass bounds when the method supports them
                bounded_methods = {"L-BFGS-B", "trust-constr", "SLSQP"}
                use_bounds = self.simp_method in bounded_methods

                scipy_result = sp_opt.minimize(
                    sub_obj,
                    sub_x0,
                    method=self.simp_method,
                    bounds=sub_bounds if use_bounds else None,
                    options=scipy_options,
                )

                # Only accept the simplex result if it actually improved the score
                best_sub = scipy_result.x
                best_full = sub_obj.build_full_vector(best_sub)
                score_after_simp = float(self.objective(best_full))

                if score_after_simp < score_after_grad:
                    current_full = best_full
                else:
                    # Simplex didn't improve — keep the post-gradient vector
                    # (current_full already holds it; no restore needed
                    # since objective evaluations are non-mutating).
                    score_after_simp = score_after_grad
                    if self.verbose:
                        logger.info(
                            "  Cycle %d SIMP: no improvement (%.6f ≥ %.6f), keeping GRAD result",
                            cycle,
                            score_after_simp,
                            score_after_grad,
                        )

                if self.verbose:
                    logger.info(
                        "  Cycle %d SIMP (%s, %d params): %.6f → %.6f",
                        cycle,
                        self.simp_method,
                        n_selected,
                        score_after_grad,
                        score_after_simp,
                    )

                cycle_scores.append(score_after_simp)

                # --- Step 4: Convergence check ---
                if score_before > 0:
                    change = (score_before - score_after_simp) / score_before
                else:
                    change = 0.0

                if self.verbose:
                    logger.info(
                        "  Cycle %d: %.2f%% improvement (threshold: %.2f%%)",
                        cycle,
                        change * 100,
                        self.convergence * 100,
                    )

                if 0 <= change < self.convergence:
                    converged = True
                    if self.verbose:
                        logger.info("  Converged after %d cycles.", cycle)
                    break

        finally:
            self.objective.on_error = prev_on_error
            self.objective.forcefield = original_forcefield

        final_score = cycle_scores[-1]
        n_cycles = len(cycle_scores) - 1  # exclude initial
        total_evals = self.objective.n_eval - n_eval_start

        return LoopResult(
            success=converged,
            initial_score=initial_score,
            final_score=final_score,
            n_cycles=n_cycles,
            initial_params=initial_params,
            final_params=current_full,
            n_eval=total_evals,
            cycle_scores=cycle_scores,
            selected_indices=selected_indices,
            sensitivity_results=sensitivity_results,
            message="converged" if converged else f"max cycles ({self.max_cycles}) reached",
        )
