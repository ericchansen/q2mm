"""Shared structural protocol for the optimizer layer.

Defines the minimal :class:`_Optimizer` interface implemented by every
concrete optimizer.  Every optimizer consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` plus an
:class:`~q2mm.models.parameters.ActiveParameterSpace` and returns the one
canonical :class:`~q2mm.models.results.OptimizationResult`.

Importing this module has no runtime Q2MM imports, so it cannot introduce
an import cycle. The shared projection helper imports its optional
selection interface only when called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np

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


def _active_value_and_gradient(
    evaluator: ObjectiveEvaluator, space: ActiveParameterSpace, full_vector: np.ndarray
) -> tuple[float, np.ndarray]:
    """Request current active FD coordinates, or retain the full-gradient path."""
    import numpy as np

    from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError, SelectedGradientEvaluator

    if evaluator.gradient_mode is GradientMode.FINITE_DIFFERENCE and isinstance(evaluator, SelectedGradientEvaluator):
        value, gradient = evaluator.value_and_gradient_selected(full_vector, space.active_indices)
        active = np.array(gradient, dtype=float, copy=True)
        if active.shape != (space.n_active,):
            raise ObjectiveGradientError(f"Selected gradient must have shape ({space.n_active},), got {active.shape}.")
        return value, active
    value, gradient = evaluator.value_and_gradient(full_vector)
    return value, space.pack(gradient)
