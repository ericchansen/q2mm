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
from typing import TypeAlias

from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator
from q2mm.optimizers.protocols import _Optimizer

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend
    from q2mm.models.forcefield import ForceField
    from q2mm.models.problem import OptimizationProblem
    from q2mm.models.results import OptimizationResult

#: A callable that builds a concrete objective executor for a compiled plan.
EvaluatorFactory: TypeAlias = Callable[[ObjectivePlan], ObjectiveEvaluator]

#: The executor selection strings accepted by :func:`make_evaluator_factory`.
_EXECUTORS = frozenset({"python", "jax"})


def make_evaluator_factory(
    backend: Backend,
    base_force_field: ForceField,
    *,
    executor: str = "python",
    gradient_mode: GradientMode | None = None,
    fd_step: float = 1e-4,
) -> EvaluatorFactory:
    """Return an evaluator factory closing over the backend and base FF.

    Args:
        backend: The MM backend the executors evaluate against.
        base_force_field: Structure/topology source for prepared sessions.
        executor: ``"python"`` (default) selects
            :class:`~q2mm.objectives.python.PythonObjectiveExecutor`;
            ``"jax"`` selects
            :class:`~q2mm.objectives.jax.JaxObjectiveExecutor`.
        gradient_mode: Gradient mode for the Python executor.  Defaults to
            :attr:`~q2mm.objectives.protocols.GradientMode.NONE`.  The JAX
            executor is always analytical, so passing a ``gradient_mode``
            with ``executor="jax"`` is an error rather than being silently
            ignored.
        fd_step: Central finite-difference step for the Python executor's
            :attr:`~q2mm.objectives.protocols.GradientMode.FINITE_DIFFERENCE`
            gradients (must be positive).  Ignored by the JAX executor.

    Raises:
        ValueError: If *executor* is not ``"python"``/``"jax"``, or a
            ``gradient_mode`` is passed with ``executor="jax"``.

    """
    if executor not in _EXECUTORS:
        raise ValueError(f"executor must be one of {sorted(_EXECUTORS)}, got {executor!r}.")
    if executor == "jax" and gradient_mode is not None:
        raise ValueError("The JAX executor is always analytical; do not pass gradient_mode with executor='jax'.")
    py_gradient_mode = GradientMode.NONE if gradient_mode is None else gradient_mode

    def factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
        if executor == "jax":
            from q2mm.objectives.jax import JaxObjectiveExecutor

            return JaxObjectiveExecutor(plan, backend, base_force_field)
        from q2mm.objectives.python import PythonObjectiveExecutor

        return PythonObjectiveExecutor(plan, backend, base_force_field, gradient_mode=py_gradient_mode, fd_step=fd_step)

    return factory


@runtime_checkable
class Workflow(Protocol):
    """Multi-stage force field parameterization workflow."""

    name: str

    def run(
        self,
        problem: OptimizationProblem,
        make_evaluator: Callable[[ObjectivePlan], ObjectiveEvaluator],
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> OptimizationResult:
        """Execute the workflow and return the canonical result.

        Args:
            problem: The immutable optimization problem.
            make_evaluator: Factory that builds an executor from a plan.
            optimizer: Pre-configured optimizer.
            n_evals: Real-objective samples at the initial and final
                parameters for noise-floor quantification (``0`` skips).

        """
        ...
