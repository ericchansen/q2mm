"""The one objective-evaluator protocol consumed by optimizers/workflows.

Every optimizer and workflow drives an :class:`ObjectiveEvaluator`.  The
protocol exposes explicit value / evaluation / residual / category /
gradient interfaces and the exact :class:`~q2mm.objectives.plan.ObjectivePlan`
(hence layout / active-space / full-vector) identity.  It performs **no**
backend probing: the concrete backend is bound inside a concrete executor
(:class:`~q2mm.objectives.python.PythonObjectiveExecutor` or
:class:`~q2mm.objectives.jax.JaxObjectiveExecutor`), and the executor's
:attr:`gradient_mode` declares — explicitly, up front — whether analytical
gradients, finite-difference gradients, or no gradients are available.

Gradient behaviour is explicit: an optimizer that needs gradients requests
them, and an executor whose :attr:`gradient_mode` is
:attr:`GradientMode.NONE` raises :class:`ObjectiveGradientError` from
:meth:`ObjectiveEvaluator.value_and_gradient` rather than silently falling
back.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from q2mm.objectives.plan import ObjectivePlan

__all__ = [
    "GradientMode",
    "ObjectiveError",
    "ObjectiveGradientError",
    "Evaluation",
    "ObjectiveEvaluator",
]


class GradientMode(str, Enum):
    """How an executor produces gradients — declared, never probed.

    Attributes:
        ANALYTICAL: Exact analytical gradients (backend parameter
            gradient / Hessian-parameter Jacobian, or JAX autodiff).
        FINITE_DIFFERENCE: Explicit central finite-difference gradients
            computed by the executor (its extra evaluations are counted).
        NONE: No executor-provided gradient.  A gradient-based optimizer
            must supply its own (e.g. SciPy's internal finite
            differences), or request analytical/FD and get an error.

    """

    ANALYTICAL = "analytical"
    FINITE_DIFFERENCE = "finite_difference"
    NONE = "none"


class ObjectiveError(RuntimeError):
    """Base class for objective-evaluation errors."""


class ObjectiveGradientError(ObjectiveError):
    """Raised when an analytical gradient is requested but unavailable.

    There is no silent fallback to finite differences — the caller must
    either select an executor with the required gradient mode or request
    finite differences explicitly.
    """


@dataclass(frozen=True, eq=False)
class Evaluation:
    """Immutable record of one objective evaluation.

    All per-observation arrays are read-only defensive copies aligned with
    ``plan.observations.values`` order; ``category_scores`` is a frozen
    mapping.

    Attributes:
        total: Total objective value (data term + regularization).
        data_value: Sum of squared weighted residuals (no regularization).
        regularization: L2 penalty contribution.
        calculated: Per-observation calculated values.
        raw_residuals: Per-observation ``ref - calc`` (torsion-wrapped).
        weighted_residuals: Per-observation ``weight * (ref - calc)``.
        category_scores: Sum of squared weighted residuals per evaluator
            category.

    """

    total: float
    data_value: float
    regularization: float
    calculated: np.ndarray
    raw_residuals: np.ndarray
    weighted_residuals: np.ndarray
    category_scores: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in ("calculated", "raw_residuals", "weighted_residuals"):
            arr = np.array(getattr(self, name), dtype=float, copy=True)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)
        object.__setattr__(
            self, "category_scores", MappingProxyType({str(k): float(v) for k, v in self.category_scores.items()})
        )


@runtime_checkable
class ObjectiveEvaluator(Protocol):
    """The one interface every optimizer/workflow consumes.

    All parameter vectors crossing this boundary are **full-length**
    (``len(plan.layout)``).  Active/full projection is the sole job of
    :class:`~q2mm.models.parameters.ActiveParameterSpace`, applied by the
    optimizer — never by the evaluator.
    """

    @property
    def plan(self) -> ObjectivePlan:
        """The immutable objective plan (layout, active space, cases)."""
        ...

    @property
    def gradient_mode(self) -> GradientMode:
        """Declared gradient capability of this executor."""
        ...

    @property
    def finite_difference_step(self) -> float | None:
        """Central finite-difference step this executor uses, or ``None``.

        ``None`` when the executor provides analytical gradients or no
        gradient at all; a positive float when :attr:`gradient_mode` is
        :attr:`GradientMode.FINITE_DIFFERENCE`.
        """
        ...

    @property
    def n_evaluations(self) -> int:
        """Number of scalar objective evaluations counted via :meth:`value`."""
        ...

    @property
    def n_gradient_evaluations(self) -> int:
        """Extra evaluations spent inside finite-difference gradients."""
        ...

    @property
    def history(self) -> tuple[float, ...]:
        """Immutable objective value history since the last reset."""
        ...

    def value(self, full_vector: np.ndarray) -> float:
        """Total objective value at *full_vector* (counts one evaluation)."""
        ...

    def record_evaluation(self, score: float) -> None:
        """Record one already-computed objective *score* as an evaluation.

        Appends *score* to :attr:`history` and increments
        :attr:`n_evaluations` **without** re-running the backend.  Used by
        optimizers (e.g. SciPy ``least_squares``) whose native callback
        already computed the residuals, so the evaluation count and history
        track the true ``nfev`` with no duplicate backend work.
        """
        ...

    def sample(self, full_vector: np.ndarray) -> float:
        """Total objective value without recording an evaluation.

        Re-evaluates the objective (capturing any per-call engine
        non-determinism) but leaves the evaluation counter and history
        untouched — used for post-hoc noise-floor sampling.
        """
        ...

    def residuals(self, full_vector: np.ndarray) -> np.ndarray:
        """Weighted data residuals ``weight * (ref - calc)`` (no regularization)."""
        ...

    def least_squares_residuals(self, full_vector: np.ndarray) -> np.ndarray:
        """Weighted residuals with ``sqrt(λ) (p - p_ref)`` L2 terms appended."""
        ...

    def value_and_gradient(self, full_vector: np.ndarray) -> tuple[float, np.ndarray]:
        """Total value and full-length gradient.

        Raises:
            ObjectiveGradientError: If :attr:`gradient_mode` is
                :attr:`GradientMode.NONE`.

        """
        ...

    def gradient(self, full_vector: np.ndarray) -> np.ndarray:
        """Full-length gradient (see :meth:`value_and_gradient`)."""
        ...

    def evaluate(self, full_vector: np.ndarray) -> Evaluation:
        """Rich, side-effect-free evaluation (does not count an evaluation)."""
        ...

    def reset(self) -> None:
        """Clear the evaluation counter and history."""
        ...
