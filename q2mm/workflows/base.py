"""Workflow Protocol + result dataclasses.

A workflow is a multi-stage orchestration of preliminary parameter
estimation, optimization, and post-processing.  See the
:mod:`q2mm.workflows` module docstring for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from q2mm.diagnostics.systems import SystemData
    from q2mm.models.forcefield import ForceField


@dataclass
class StageResult:
    """Result of a single optimization stage within a workflow.

    Most workflows have one stage (``SingleStageWorkflow``); the
    planned two-stage protocols (Method E2) report two
    :class:`StageResult` instances back-to-back.

    Attributes:
        name: Human-readable stage label (e.g. ``"optimize"``,
            ``"round-1-method-d"``, ``"round-2-method-c"``).
        initial_score: Optimizer-reported objective value at the start
            of this stage.  May be the surrogate (JaxLoss) when
            ``jac_mode="auto"`` resolves to analytical gradients —
            see ``OptimizationResult.final_score`` for the same
            caveat.
        final_score: Optimizer-reported objective value at the end of
            this stage.  Same surrogate caveat as ``initial_score``.
        n_iterations: Optimizer iteration count.
        n_evaluations: Objective/gradient call count.
        converged: Optimizer's convergence flag.
        message: Optimizer's convergence message.
        jac_mode: Resolved Jacobian strategy for this stage
            (``"analytical"``, ``"finite-difference"``, or
            ``"derivative-free"``).  May be ``"unknown"`` if the
            optimizer didn't report.
        elapsed_s: Wall time for the optimizer call.
        locked_param_indices: Full-vector indices of parameters
            frozen at the start of this stage by the workflow itself
            (distinct from parameters frozen on the FF before
            workflow invocation).  Empty for ``SingleStageWorkflow``;
            populated by Method E2's Round 2.
        notes: Free-form per-stage metadata (e.g. "method": "D",
            "replace_with": 0.03) for downstream reporting.

    """

    name: str
    initial_score: float
    final_score: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    message: str
    jac_mode: str
    elapsed_s: float
    locked_param_indices: list[int] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Complete output of a :meth:`Workflow.run` call.

    Attributes:
        workflow_name: Identifier of the workflow that produced this
            result (e.g. ``"single-stage"``).
        final_ff: Force field at the end of the workflow.
        initial_ff: Snapshot of the force field as it was when the
            workflow started.  Workflow implementations are
            responsible for materializing this snapshot via
            ``forcefield.with_params(initial_vector)`` before invoking
            the optimizer — the current ``ScipyOptimizer`` mutates the
            FF inside the ObjectiveFunction during the search, so a
            bare reference to ``system.forcefield`` would point at the
            *final* state.
        stages: Per-stage diagnostics, ordered chronologically.  Length
            1 for ``SingleStageWorkflow``; 2 for the planned
            ``MethodE2Workflow``.
        initial_obj_samples: ``n_evals`` samples of the *real*
            :class:`~q2mm.optimizers.objective.ObjectiveFunction`
            evaluated at the initial parameters.  Engine non-
            determinism (geometry-relaxation seeding, etc.) means a
            single eval is noisy; ``n_evals > 1`` quantifies that
            noise.  Empty list is valid (``n_evals=0`` skip).
        final_obj_samples: Same as ``initial_obj_samples`` but at the
            final parameters.
        optimized_categories: Per-reference-category fit metrics
            (R², RMSD, MAE, n) for the final FF.  Keys depend on what
            reference types are populated (bond_length, bond_angle,
            eig_diagonal, ...).

    """

    workflow_name: str
    final_ff: ForceField
    initial_ff: ForceField
    stages: list[StageResult]
    initial_obj_samples: list[float] = field(default_factory=list)
    final_obj_samples: list[float] = field(default_factory=list)
    optimized_categories: dict[str, dict[str, float]] = field(default_factory=dict)


@runtime_checkable
class _Optimizer(Protocol):
    """Minimal optimizer interface shared by ScipyOptimizer / multistart / etc.

    Mirrors the existing ``_Optimizer`` protocol in
    :mod:`q2mm.optimizers.multistart`.  Defined here so workflows
    don't pull a hard dependency on a specific concrete optimizer.
    """

    def optimize(self, objective: Any) -> Any:  # noqa: ANN401
        """Run the optimization; return an ``OptimizationResult``-shaped object."""
        ...


@runtime_checkable
class Workflow(Protocol):
    """Multi-stage force field parameterization workflow.

    Implementations build the :class:`ObjectiveFunction` internally
    from ``system``, call ``optimizer.optimize(...)`` one or more
    times, and aggregate per-stage diagnostics into a
    :class:`WorkflowResult`.
    """

    name: str

    def run(
        self,
        system: SystemData,
        engine: Any,  # noqa: ANN401
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> WorkflowResult:
        """Execute the workflow.

        .. warning::

           Implementations using the current :class:`ScipyOptimizer`
           will see ``system.forcefield`` mutated in place during the
           search (``ObjectiveFunction.forcefield.set_param_vector``
           runs on every step).  Use ``WorkflowResult.initial_ff`` and
           ``WorkflowResult.final_ff`` as the stable before/after
           snapshots; do not rely on ``system.forcefield`` being
           unchanged after the call.

        Args:
            system: Loaded benchmark system; supplies the force field,
                molecules, and reference data.  ``system.forcefield``
                may be mutated by the inner optimizer — see warning.
            engine: MM backend used to evaluate the objective.
            optimizer: Pre-configured search algorithm.  Constructor
                kwargs (method, bounds strategy, ftol, etc.) belong
                here, not on the workflow.
            n_evals: Number of *real* objective samples to take at
                the initial and final parameter vectors for noise-
                floor quantification.  ``0`` skips post-hoc sampling
                (final FF is still computed).

        Returns:
            :class:`WorkflowResult` with per-stage diagnostics, final
            FF, and per-category fit metrics.

        """
        ...
