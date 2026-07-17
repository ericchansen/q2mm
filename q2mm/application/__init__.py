"""Stable application services with lazy execution-module imports."""

from __future__ import annotations

from typing import Any

from .models import (
    ApplicationConfigurationError,
    ApplicationError,
    ApplicationEvaluationError,
    ApplicationOptimizationError,
    OptimizationRun,
    OutputExistsError,
    OutputFormatError,
    PersistenceError,
    ResolvedBackendConfiguration,
    ResolvedExecutionConfiguration,
    ResolvedExecutorConfiguration,
    ResolvedOptimizerConfiguration,
    ResolvedWorkflowConfiguration,
    SavedOutput,
    problem_fingerprint,
    problem_input_fingerprints,
)

_LAZY_EXPORTS = {
    "evaluate": ("q2mm.application.evaluation", "evaluate"),
    "evaluate_problem": ("q2mm.application.evaluation", "evaluate_problem"),
    "evaluate_property": ("q2mm.application.evaluation", "evaluate_property"),
    "optimize": ("q2mm.application.optimization", "optimize"),
    "MANIFEST_SUFFIX": ("q2mm.application.persistence", "MANIFEST_SUFFIX"),
    "save": ("q2mm.application.persistence", "save"),
}


def __getattr__(name: str) -> Any:
    """Import execution and persistence services only on first access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    from importlib import import_module

    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


__all__ = [
    "MANIFEST_SUFFIX",
    "ApplicationConfigurationError",
    "ApplicationError",
    "ApplicationEvaluationError",
    "ApplicationOptimizationError",
    "OptimizationRun",
    "OutputExistsError",
    "OutputFormatError",
    "PersistenceError",
    "ResolvedBackendConfiguration",
    "ResolvedExecutionConfiguration",
    "ResolvedExecutorConfiguration",
    "ResolvedOptimizerConfiguration",
    "ResolvedWorkflowConfiguration",
    "SavedOutput",
    "evaluate",
    "evaluate_problem",
    "evaluate_property",
    "optimize",
    "problem_fingerprint",
    "problem_input_fingerprints",
    "save",
]
