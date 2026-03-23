"""Parameter cycling and sensitivity-based selection for Q2MM optimization.

Implements the upstream Q2MM GRAD→SIMP optimization loop, adapted for our
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

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from q2mm.models.forcefield import ForceField
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
        n_cycles (int): Number of GRAD→SIMP cycles completed.
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
        if self.initial_score == 0:
            return 0.0
        return (self.initial_score - self.final_score) / self.initial_score

    def summary(self) -> str:
        """Human-readable summary string.

        Returns:
            str: Multi-line summary of the loop result.
        """
        lines = [
            f"OptimizationLoop: {'converged' if self.success else 'max cycles reached'}",
            f"  Cycles:      {self.n_cycles}",
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
    ):
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
        """The sub-vector corresponding to current active parameters.

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
        all_bounds = self.objective.forcefield.get_bounds()
        return [all_bounds[i] for i in self.active_indices]


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------


def compute_sensitivity(
    objective: ObjectiveFunction,
    step_sizes: np.ndarray | None = None,
    metric: Literal["simp_var", "abs_d1"] = "simp_var",
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

    Cost: ``2N + 1`` objective evaluations (1 baseline + 2 per parameter
    for central differentiation).

    Args:
        objective (ObjectiveFunction): Must already be evaluable (engine
            and molecules configured).
        step_sizes (np.ndarray | None): Per-parameter step sizes.
            Defaults to :meth:`ForceField.get_step_sizes` if not
            provided.
        metric (str): Ranking criterion — ``"simp_var"`` or
            ``"abs_d1"``.

    Returns:
        SensitivityResult: Derivatives, rankings, and evaluation count.

    Raises:
        ValueError: If ``step_sizes`` length does not match the parameter
            vector, or if *metric* is unknown.
    """
    ff = objective.forcefield
    x0 = ff.get_param_vector().copy()
    n = len(x0)

    if step_sizes is None:
        step_sizes = ff.get_step_sizes()
    step_sizes = np.asarray(step_sizes)
    if len(step_sizes) != n:
        raise ValueError(f"step_sizes length {len(step_sizes)} != param vector length {n}")

    d1 = np.zeros(n)
    d2 = np.zeros(n)
    n_evals = 0

    try:
        # Baseline evaluation
        f0 = float(objective(x0))
        n_evals = 1

        for i in range(n):
            h = step_sizes[i]
            if h == 0:
                continue

            x_fwd = x0.copy()
            x_bwd = x0.copy()
            x_fwd[i] += h
            x_bwd[i] -= h

            f_fwd = float(objective(x_fwd))
            f_bwd = float(objective(x_bwd))
            n_evals += 2

            # Unnormalised derivatives (upstream convention)
            d1[i] = (f_fwd - f_bwd) * 0.5
            d2[i] = f_fwd + f_bwd - 2.0 * f0
    finally:
        # Restore the forcefield to x0 so subsequent steps start from the
        # correct parameters.  ObjectiveFunction.__call__ mutates the FF
        # via set_param_vector(), leaving it at the last perturbed state.
        objective(x0)
        n_evals += 1

    # Compute simp_var, guarding against zero d1
    with np.errstate(divide="ignore", invalid="ignore"):
        simp_var = np.where(d1 != 0, d2 / (d1**2), np.inf)

    # Ranking
    if metric == "simp_var":
        ranking = np.argsort(simp_var)  # ascending: lowest = most suitable
    elif metric == "abs_d1":
        # Normalise by step size so ranking reflects true gradient magnitude
        # rather than being biased by per-type step size differences.
        normalised_d1 = np.where(step_sizes != 0, d1 / step_sizes, 0.0)
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
# OptimizationLoop — GRAD→SIMP cycling
# ---------------------------------------------------------------------------


class OptimizationLoop:
    """GRAD→SIMP cycling loop inspired by the upstream Q2MM workflow.

    Each cycle:
      1. **Full-space pass** — run ``full_method`` (default L-BFGS-B) on
         all parameters.
      2. **Sensitivity analysis** — central differentiation to rank params.
      3. **Subspace simplex** — run ``simp_method`` (default Nelder-Mead)
         on only the top ``max_params`` most sensitive parameters.
      4. **Convergence check** — stop if the fractional improvement in the
         objective falls below ``convergence``.

    Args:
        objective (ObjectiveFunction): The objective function to minimise.
        max_params (int): Number of parameters per simplex pass (upstream
            default: 3).
        convergence (float): Stop when ``(score_before - score_after) /
            score_before < convergence``.
        max_cycles (int): Maximum number of GRAD→SIMP cycles.
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
    ):
        """Initialize the optimization loop.

        Args:
            objective (ObjectiveFunction): The objective function to
                minimise.
            max_params (int): Number of parameters per simplex pass.
            convergence (float): Fractional improvement threshold.
            max_cycles (int): Maximum number of GRAD→SIMP cycles.
            full_method (str): Scipy method for the full-space pass.
            simp_method (str): Scipy method for the subspace pass.
            full_maxiter (int): Max iterations for the full-space pass.
            simp_maxiter (int): Max iterations for the subspace pass.
            sensitivity_metric (str): Parameter ranking criterion.
            eps (float): Finite-difference step size.
            verbose (bool): Whether to log progress.
        """
        self.objective = objective
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

    def run(self) -> LoopResult:
        """Execute the GRAD→SIMP cycling loop.

        Returns:
            LoopResult: Contains convergence status, per-cycle scores,
                and selected parameter indices.
        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        ff = self.objective.forcefield
        x0 = ff.get_param_vector().copy()
        initial_score = float(self.objective(x0))

        cycle_scores: list[float] = [initial_score]
        selected_indices: list[list[int]] = []
        sensitivity_results: list[SensitivityResult] = []
        converged = False

        if self.verbose:
            logger.info(
                "OptimizationLoop: initial score = %.6f, max_params = %d",
                initial_score,
                self.max_params,
            )

        for cycle in range(1, self.max_cycles + 1):
            score_before = cycle_scores[-1]

            # --- Step 1: Full-space optimisation ---
            full_opt = ScipyOptimizer(
                method=self.full_method,
                maxiter=self.full_maxiter,
                eps=self.eps,
                verbose=False,
            )
            full_result = full_opt.optimize(self.objective)
            score_after_grad = full_result.final_score

            if self.verbose:
                logger.info(
                    "  Cycle %d GRAD (%s): %.6f → %.6f",
                    cycle,
                    self.full_method,
                    score_before,
                    score_after_grad,
                )

            # --- Step 2: Sensitivity analysis ---
            step_sizes = ff.get_step_sizes()
            sens = compute_sensitivity(
                self.objective,
                step_sizes=step_sizes,
                metric=self.sensitivity_metric,
            )
            sensitivity_results.append(sens)

            # Select top max_params; ensure we have at least one active parameter
            n_active = min(self.max_params, ff.n_params)
            if n_active < 1:
                raise ValueError(
                    f"OptimizationLoop requires at least one active parameter, "
                    f"but max_params={self.max_params} and ff.n_params={ff.n_params}."
                )
            active = sens.ranking[:n_active].tolist()
            selected_indices.append(active)

            if self.verbose:
                labels = ff.get_param_type_labels()
                selected_labels = [f"{labels[i]}[{i}]" for i in active]
                logger.info(
                    "  Cycle %d sensitivity (%s): selected %s",
                    cycle,
                    self.sensitivity_metric,
                    ", ".join(selected_labels),
                )

            # --- Step 3: Subspace simplex ---
            current_full = ff.get_param_vector().copy()
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
                ff.set_param_vector(best_full)
            else:
                # Revert to post-gradient parameters
                score_after_simp = score_after_grad
                ff.set_param_vector(current_full)
                self.objective(current_full)  # restore objective state
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
                    n_active,
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

        final_score = cycle_scores[-1]
        n_cycles = len(cycle_scores) - 1  # exclude initial

        return LoopResult(
            success=converged,
            initial_score=initial_score,
            final_score=final_score,
            n_cycles=n_cycles,
            cycle_scores=cycle_scores,
            selected_indices=selected_indices,
            sensitivity_results=sensitivity_results,
            message="converged" if converged else f"max cycles ({self.max_cycles}) reached",
        )
