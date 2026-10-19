"""JaxOpt construction adapter for the shared sequential multi-start optimizer.

The supported JAX constructor retains its defaults and result labels while
:class:`~q2mm.optimizers.multistart.MultiStartOptimizer` owns start generation,
execution, failure records, winner selection, and result assembly. Inner
solves retain per-case JIT and Python aggregation; replicas are not fused.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer, _require_jax_executor
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.settings import JaxMultiStartSettings, MultiStartSettings

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace


class JaxMultiStartOptimizer(MultiStartOptimizer):
    """Configure shared multi-start execution with a JaxOpt inner optimizer."""

    _name = "jaxopt-multi"
    _candidate_name = "replica"
    _failure_gradient_mode = "analytical"
    settings: JaxMultiStartSettings

    def __init__(
        self,
        settings: JaxMultiStartSettings = JaxMultiStartSettings(),
    ) -> None:
        super().__init__(
            MultiStartSettings(
                optimizer=settings.optimizer,
                n_starts=settings.n_starts,
                perturbation_pct=settings.perturbation_pct,
                seed=settings.seed,
                verbose=settings.verbose,
            ),
            optimizer=JaxOptOptimizer(settings.optimizer),
        )
        self.settings = settings

    def configuration_settings(self) -> Mapping[str, Any]:
        """Capture this adapter's settings and its actual constructed JaxOpt child."""
        return super().configuration_settings()

    def _result_method(self, selected: OptimizationResult | None) -> str:
        return f"jaxopt-multi:{self.settings.optimizer.method}"

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Validate the JAX executor and delegate all multi-start execution."""
        _require_jax_executor(evaluator)
        return super().optimize(evaluator, space)
