"""JaxOpt construction adapter for the shared sequential multi-start optimizer.

The supported JAX constructor retains its defaults and result labels while
:class:`~q2mm.optimizers.multistart.MultiStartOptimizer` owns start generation,
execution, failure records, winner selection, and result assembly. Inner
solves retain per-case JIT and Python aggregation; replicas are not fused.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer, _require_jax_executor
from q2mm.optimizers.multistart import MultiStartOptimizer

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace


class JaxMultiStartOptimizer(MultiStartOptimizer):
    """Configure shared multi-start execution with a JaxOpt inner optimizer."""

    _name = "jaxopt-multi"
    _candidate_name = "replica"
    _failure_gradient_mode = "analytical"

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
        self.method = method
        self.maxiter = maxiter
        self.tol = tol
        super().__init__(
            optimizer=self._make_inner_optimizer(),
            n_starts=n_starts,
            perturbation_pct=perturbation_pct,
            seed=seed,
            verbose=verbose,
        )

    def _make_inner_optimizer(self) -> JaxOptOptimizer:
        return JaxOptOptimizer(method=self.method, maxiter=self.maxiter, tol=self.tol, verbose=False)

    def _result_method(self, selected: OptimizationResult | None) -> str:
        return f"jaxopt-multi:{self.method}"

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Validate the JAX executor and delegate all multi-start execution."""
        _require_jax_executor(evaluator)
        self.optimizer = self._make_inner_optimizer()
        return super().optimize(evaluator, space)
