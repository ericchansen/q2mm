"""Backend-neutral objective plans and pluggable objective executors.

The objective layer separates *what* to fit (:class:`ObjectivePlan`,
backend-neutral) from *how* to evaluate it (an
:class:`ObjectiveEvaluator` — either the general-purpose
:class:`PythonObjectiveExecutor` over the typed backend contract, or the
:class:`JaxObjectiveExecutor` with per-case JIT + analytical gradients).

Optimizers and workflows consume an :class:`ObjectiveEvaluator`; callers
select the concrete executor explicitly.
"""

from __future__ import annotations

from typing import Any

from q2mm.objectives.plan import KIND_TO_CATEGORY, ObjectivePlan
from q2mm.objectives.protocols import (
    Evaluation,
    GradientMode,
    ObjectiveError,
    ObjectiveEvaluator,
    ObjectiveGradientError,
    UnsupportedObservationError,
)
from q2mm.objectives.python import PythonObjectiveExecutor

__all__ = [
    "ObjectivePlan",
    "KIND_TO_CATEGORY",
    "GradientMode",
    "Evaluation",
    "ObjectiveError",
    "ObjectiveGradientError",
    "UnsupportedObservationError",
    "ObjectiveEvaluator",
    "PythonObjectiveExecutor",
    "JaxObjectiveExecutor",
]


def __getattr__(name: str) -> Any:
    """Lazily import the optional JAX executor.

    Keeps ``import q2mm.objectives`` working without JAX installed, while
    still exposing :class:`JaxObjectiveExecutor` on first access.
    """
    if name == "JaxObjectiveExecutor":
        from q2mm.objectives.jax import JaxObjectiveExecutor

        return JaxObjectiveExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
