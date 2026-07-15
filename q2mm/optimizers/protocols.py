"""Shared structural protocols for the optimizer layer.

Defines the minimal :class:`_Optimizer` interface implemented by every
concrete optimizer (:class:`~q2mm.optimizers.scipy_opt.ScipyOptimizer`,
:class:`~q2mm.optimizers.optax.OptaxOptimizer`,
:class:`~q2mm.optimizers.basinhopping.BasinHoppingOptimizer`,
:class:`~q2mm.optimizers.multistart.MultiStartOptimizer`, ...).

Declared once here so :mod:`q2mm.optimizers.multistart` and
:mod:`q2mm.workflows.base` can share a single protocol instead of each
re-declaring their own copy.  This module intentionally has **no
runtime Q2MM imports** (the concrete reference/return types are pulled
only under ``TYPE_CHECKING``), so importing it can never introduce an
import cycle regardless of import order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.scipy_opt import OptimizationResult


@runtime_checkable
class _Optimizer(Protocol):
    """Minimal structural interface for any wrappable optimizer.

    Any object exposing an ``optimize(objective, space)`` method that
    returns an :class:`~q2mm.optimizers.scipy_opt.OptimizationResult`
    satisfies this protocol.  It is ``@runtime_checkable`` so
    ``isinstance`` checks succeed for conforming optimizers (see
    ``test/test_workflows.py``).
    """

    def optimize(self, objective: ObjectiveFunction, space: ActiveParameterSpace) -> OptimizationResult:
        """Run the optimization and return an ``OptimizationResult``."""
        ...
