"""Lazy public exports of canonical Q2MM operations and scientific values."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("q2mm")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

_LAZY = {
    "Molecule": ("q2mm.models.molecule", "Molecule"),
    "ForceField": ("q2mm.models.forcefield", "ForceField"),
    "OptimizationProblem": ("q2mm.models.problem", "OptimizationProblem"),
    "OptimizationResult": ("q2mm.models.results", "OptimizationResult"),
    "OptimizationRun": ("q2mm.application.models", "OptimizationRun"),
    "Evaluation": ("q2mm.objectives.protocols", "Evaluation"),
    "ExternalDataRoots": ("q2mm.benchmarks.systems._paths", "ExternalDataRoots"),
    "prepare": ("q2mm.preparation", "prepare"),
    "match_frequencies": ("q2mm.preparation", "match_frequencies"),
    "evaluate": ("q2mm.application.evaluation", "evaluate"),
    "evaluate_reference": ("q2mm.application.evaluation", "evaluate_reference"),
    "create_evaluator": ("q2mm.objectives", "create_evaluator"),
    "make_evaluator_factory": ("q2mm.workflows.base", "make_evaluator_factory"),
    "capture_execution_configuration": ("q2mm.application.configuration", "capture_execution_configuration"),
    "build_optimizer": ("q2mm.optimizers.catalog", "build_optimizer"),
    "build_workflow": ("q2mm.workflows.base", "build_workflow"),
    "optimize": ("q2mm.application.optimization", "optimize"),
    "save": ("q2mm.application.persistence", "save"),
}


def __getattr__(name: str) -> Any:
    """Resolve to the canonical object, not a forwarding implementation."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _LAZY[name]
    value = getattr(import_module(module), attribute)
    globals()[name] = value
    return value


__all__ = ["__version__", *_LAZY]
