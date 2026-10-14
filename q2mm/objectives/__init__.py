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
    ObjectiveConvergenceError,
    ObjectiveError,
    ObjectiveEvaluator,
    ObjectiveGradientError,
    SelectedGradientEvaluator,
    UnsupportedObservationError,
)
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.backends.contracts import Backend, BackendRole
from q2mm.models.forcefield import ForceField
from q2mm.settings import ObjectiveSettings

__all__ = [
    "ObjectivePlan",
    "KIND_TO_CATEGORY",
    "GradientMode",
    "Evaluation",
    "ObjectiveConvergenceError",
    "ObjectiveError",
    "ObjectiveGradientError",
    "UnsupportedObservationError",
    "ObjectiveEvaluator",
    "SelectedGradientEvaluator",
    "PythonObjectiveExecutor",
    "JaxObjectiveExecutor",
    "create_evaluator",
]


def create_evaluator(
    plan: ObjectivePlan,
    *,
    backend: Backend,
    force_field: ForceField,
    settings: ObjectiveSettings,
) -> ObjectiveEvaluator:
    """Construct the declared executor without loading engines or selecting policy."""
    if plan.regularization != settings.regularization:
        raise ObjectiveError("Objective plan regularization does not match the declared objective settings.")
    if backend.info.role is not BackendRole.MM:
        raise ObjectiveError("Objective evaluation requires an MM backend.")
    if not backend.info.supports_form(force_field.functional_form.value):
        raise ObjectiveError(
            f"Backend {backend.info.name!r} does not support form {force_field.functional_form.value!r}."
        )
    if settings.executor == "jax":
        provenance = backend.info.provenance
        key = provenance.backend if provenance is not None else backend.info.name
        if key != "jax":
            raise ObjectiveError(f"JAX executor requires the registered JAX backend, not {key!r}.")
        from q2mm.objectives.jax import JaxObjectiveExecutor

        return JaxObjectiveExecutor(plan, backend, force_field)
    fd_options = {} if settings.fd_step is None else {"fd_step": settings.fd_step}
    return PythonObjectiveExecutor(
        plan,
        backend,
        force_field,
        gradient_mode=GradientMode(settings.gradient_mode),
        **fd_options,
    )


def __getattr__(name: str) -> Any:
    """Lazily import the optional JAX executor.

    Keeps ``import q2mm.objectives`` working without JAX installed, while
    still exposing :class:`JaxObjectiveExecutor` on first access.
    """
    if name == "JaxObjectiveExecutor":
        from q2mm.objectives.jax import JaxObjectiveExecutor

        return JaxObjectiveExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
