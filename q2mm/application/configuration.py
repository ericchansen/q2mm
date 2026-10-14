"""Capture actual component settings before execution, without reconstruction."""

from collections.abc import Mapping
from typing import Any

from q2mm.backends.contracts import Backend
from q2mm.configuration import ConfigurationCaptureError, ConfigurationProvider, capture_component_settings
from q2mm.optimizers.protocols import _Optimizer
from q2mm.settings import OptimizationSettings
from q2mm.workflows.base import Workflow

from .models import (
    ApplicationConfigurationError,
    ResolvedBackendConfiguration,
    ResolvedExecutionConfiguration,
    ResolvedExecutorConfiguration,
    ResolvedOptimizerConfiguration,
    ResolvedWorkflowConfiguration,
)


def _component_settings(value: object) -> Mapping[str, Any]:
    """Translate neutral snapshot failures into the application error contract."""
    try:
        return capture_component_settings(value)
    except ConfigurationCaptureError as exc:
        if isinstance(exc.__cause__, ApplicationConfigurationError):
            raise exc.__cause__
        raise ApplicationConfigurationError(str(exc)) from (exc.__cause__ or exc)


def capture_execution_configuration(
    settings: OptimizationSettings,
    *,
    backend: Backend,
    optimizer: _Optimizer,
    workflow: Workflow,
) -> ResolvedExecutionConfiguration:
    """Record verified runtime facts and actual providers before numerical execution."""
    from .optimization import _result_gradient_mode
    from q2mm.optimizers.catalog import optimizer_method

    info = backend.info
    provenance = info.provenance
    details: dict[str, object] = {} if provenance is None else dict(provenance.details)
    if settings.backend.options:
        details["factory_options"] = dict(settings.backend.options)
    optimizer_configuration = _component_settings(optimizer)
    workflow_configuration = _component_settings(workflow)
    known_settings = getattr(optimizer, "settings", None)
    method = (
        optimizer_method(known_settings)
        if (known_settings is not None and "version" in optimizer_configuration)
        else type(optimizer).__qualname__
    )
    return ResolvedExecutionConfiguration(
        recipe_id=settings.recipe_id,
        backend=ResolvedBackendConfiguration(
            key=info.name if provenance is None else provenance.backend,
            name=info.name,
            role=info.role.value,
            version="" if provenance is None else provenance.version,
            capabilities=tuple(capability.value for capability in info.capabilities),
            functional_forms=tuple(info.functional_forms),
            details=details,
        ),
        optimizer=ResolvedOptimizerConfiguration(
            key=f"{type(optimizer).__module__}.{type(optimizer).__qualname__}",
            label=type(optimizer).__qualname__,
            method=method,
            settings=optimizer_configuration,
            expected_result_gradient_mode=_result_gradient_mode(optimizer, settings.objective.gradient_mode),
        ),
        workflow=ResolvedWorkflowConfiguration(key=workflow.name, settings=workflow_configuration),
        executor=ResolvedExecutorConfiguration(
            kind=settings.objective.executor,
            gradient_mode=settings.objective.gradient_mode,
            fd_step=settings.objective.fd_step if settings.objective.gradient_mode == "finite_difference" else None,
        ),
        overrides=settings.requested_fields,
        regularization=settings.objective.regularization,
        n_evals=settings.n_evals,
        ratio_tol=settings.ratio_tol,
    )


__all__ = ["ConfigurationProvider", "capture_execution_configuration"]
