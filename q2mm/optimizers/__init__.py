"""Lazy concrete algorithm exports and the single declarative preset catalog."""

from importlib import import_module
from typing import Any

from q2mm.optimizers.catalog import (
    OPTIMIZER_CATALOG,
    OptimizerSpec,
    build_optimizer,
    expected_result_gradient,
    optimizer_settings,
    optimizer_spec,
)

_LAZY = {
    "ScipyOptimizer": ("scipy_opt", "optimize"),
    "CyclingOptimizer": ("cycling", "optimize"),
    "SensitivityResult": ("cycling", "optimize"),
    "compute_sensitivity": ("cycling", "optimize"),
    "OptaxOptimizer": ("optax", "jax"),
    "JaxOptOptimizer": ("jaxopt_opt", "jax"),
    "JaxMultiStartOptimizer": ("jax_multistart", "jax"),
    "BasinHoppingOptimizer": ("basinhopping", "optimize"),
    "MultiStartOptimizer": ("multistart", "optimize"),
}


def __getattr__(name: str) -> Any:
    """Load one requested class; failures retain their actionable original cause."""
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, extra = _LAZY[name]
    try:
        value = getattr(import_module(f"q2mm.optimizers.{module}"), name)
    except ImportError as exc:
        raise ImportError(f"{name} requires optional dependencies; install q2mm[{extra}]: {exc}") from exc
    globals()[name] = value
    return value


__all__ = [
    "OPTIMIZER_CATALOG",
    "OptimizerSpec",
    "build_optimizer",
    "expected_result_gradient",
    "optimizer_settings",
    "optimizer_spec",
    *_LAZY,
]
