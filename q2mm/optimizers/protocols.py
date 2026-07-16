"""Shared structural protocol for the optimizer layer.

Defines the minimal :class:`_Optimizer` interface implemented by every
concrete optimizer.  Every optimizer consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` plus an
:class:`~q2mm.models.parameters.ActiveParameterSpace` and returns the one
canonical :class:`~q2mm.models.results.OptimizationResult`.

This module has no runtime Q2MM imports (concrete types are pulled only
under ``TYPE_CHECKING``), so importing it can never introduce an import
cycle regardless of import order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.models.results import OptimizationResult
    from q2mm.objectives.protocols import ObjectiveEvaluator


@runtime_checkable
class _Optimizer(Protocol):
    """Minimal structural interface for any wrappable optimizer.

    Any object exposing an ``optimize(evaluator, space)`` method that
    returns an :class:`~q2mm.models.results.OptimizationResult` satisfies
    this protocol.  It is ``@runtime_checkable`` so ``isinstance`` checks
    succeed for conforming optimizers.
    """

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the optimization and return an ``OptimizationResult``."""
        ...
