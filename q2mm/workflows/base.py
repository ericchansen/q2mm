"""Workflow protocol + evaluator-factory plumbing.

A workflow is a multi-stage orchestration of optimizer passes.  Workflows
consume:

- an immutable :class:`~q2mm.models.problem.OptimizationProblem`,
- an **evaluator factory** — a ``Callable[[ObjectivePlan], ObjectiveEvaluator]``
  that builds a concrete executor (Python or JAX) for a compiled plan, and
- a pre-configured optimizer.

They compile :class:`~q2mm.objectives.plan.ObjectivePlan` objects, drive the
optimizer, and aggregate per-stage diagnostics into the one canonical
:class:`~q2mm.models.results.OptimizationResult`.  Workflows never
construct backends or objectives directly, never mutate a force field, and
never store a force-field snapshot in the result — callers materialize a
force field explicitly via ``layout.replace(base_ff, result.final_params)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from collections.abc import Callable

from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.protocols import _Optimizer
from q2mm.settings import ObjectiveSettings, WorkflowSettings, SingleStageSettings, MethodE2Settings

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend
    from q2mm.models.forcefield import ForceField
    from q2mm.models.problem import OptimizationProblem
    from q2mm.models.results import OptimizationResult

#: A callable that builds a concrete objective executor for a compiled plan.
type EvaluatorFactory = Callable[[ObjectivePlan], ObjectiveEvaluator]


def make_evaluator_factory(
    backend: Backend,
    base_force_field: ForceField,
    *,
    settings: ObjectiveSettings,
) -> EvaluatorFactory:
    """Close over explicit construction choices; each stage gets its own executor."""
    from q2mm.objectives import create_evaluator

    def factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
        return create_evaluator(plan, backend=backend, force_field=base_force_field, settings=settings)

    return factory


def build_workflow(settings: WorkflowSettings) -> Workflow:
    """Construct exactly the declared workflow, without policy resolution."""
    if isinstance(settings, SingleStageSettings):
        from q2mm.workflows.single_stage import SingleStageWorkflow

        return SingleStageWorkflow(settings)
    if isinstance(settings, MethodE2Settings):
        from q2mm.workflows.method_e2 import MethodE2Workflow

        return MethodE2Workflow(settings)
    raise TypeError(f"Unsupported workflow settings {type(settings).__name__}.")


@runtime_checkable
class Workflow(Protocol):
    """Multi-stage force field parameterization workflow."""

    @property
    def name(self) -> str:
        """The configured workflow's public identifier."""
        ...

    def run(
        self,
        problem: OptimizationProblem,
        make_evaluator: Callable[[ObjectivePlan], ObjectiveEvaluator],
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
        regularization: float = 0.0,
    ) -> OptimizationResult:
        """Execute the workflow and return the canonical result.

        Args:
            problem: The immutable optimization problem.
            make_evaluator: Factory that builds an executor from a plan.
            optimizer: Pre-configured optimizer.
            n_evals: Configured-executor samples at the initial and final
                parameters (``0`` skips). Multi-stage workflows may sample
                different objective plans at the two endpoints.
            regularization: Non-negative L2 penalty threaded into every
                :class:`~q2mm.objectives.plan.ObjectivePlan` the workflow
                compiles, so optimization and reporting share one objective.

        """
        ...
