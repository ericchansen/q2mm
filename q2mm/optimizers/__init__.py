"""Force field optimizers for Q2MM.

Every optimizer consumes an
:class:`~q2mm.objectives.protocols.ObjectiveEvaluator` plus an
:class:`~q2mm.models.parameters.ActiveParameterSpace` and returns the one
canonical :class:`~q2mm.models.results.OptimizationResult`.

Optional optimizers
-------------------

Several optimizers depend on optional packages declared via
``[project.optional-dependencies]`` in ``pyproject.toml``:

================================ ==============================
Optimizer                        Install hint
================================ ==============================
``ScipyOptimizer``               ``pip install 'q2mm[optimize]'``
``BasinHoppingOptimizer``        ``pip install 'q2mm[optimize]'``
``MultiStartOptimizer``          ``pip install 'q2mm[optimize]'``
``OptimizationLoop`` (cycling)   ``pip install 'q2mm[optimize]'``
``OptaxOptimizer``               ``pip install 'q2mm[jax]'``
``JaxOptOptimizer``              ``pip install 'q2mm[jax]'``
``JaxMultiStartOptimizer``       ``pip install 'q2mm[jax]'``
================================ ==============================
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from q2mm.models.results import CandidateRecord, OptimizationResult, StageRecord

logger = logging.getLogger(__name__)

__all__ = [
    "OptimizationResult",
    "CandidateRecord",
    "StageRecord",
    "available_optimizers",
    "get_optimizer",
]

# (public attribute name, defining module, install hint).
_OPTIONAL_OPTIMIZERS: tuple[tuple[str, str, str], ...] = (
    ("ScipyOptimizer", "q2mm.optimizers.scipy_opt", "pip install 'q2mm[optimize]'"),
    ("OptimizationLoop", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("SensitivityResult", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("compute_sensitivity", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("OptaxOptimizer", "q2mm.optimizers.optax", "pip install 'q2mm[jax]'"),
    ("JaxOptOptimizer", "q2mm.optimizers.jaxopt_opt", "pip install 'q2mm[jax]'"),
    ("JaxMultiStartOptimizer", "q2mm.optimizers.jax_multistart", "pip install 'q2mm[jax]'"),
    ("BasinHoppingOptimizer", "q2mm.optimizers.basinhopping", "pip install 'q2mm[optimize]'"),
    ("MultiStartOptimizer", "q2mm.optimizers.multistart", "pip install 'q2mm[optimize]'"),
)

_FAILED: dict[str, tuple[str, ImportError]] = {}


def _discover() -> None:
    for attr, module_path, hint in _OPTIONAL_OPTIMIZERS:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            _FAILED[attr] = (hint, exc)
            logger.debug("Optimizer %s unavailable: %s (%s)", attr, exc, hint)
            continue
        except Exception as exc:
            logger.warning("Unexpected error importing optimizer module %s: %s", module_path, exc, exc_info=True)
            _FAILED[attr] = (hint, ImportError(str(exc)))
            continue
        if hasattr(module, attr):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
        else:
            logger.warning("Optimizer module %s loaded but exports no %s", module_path, attr)


_discover()


def available_optimizers() -> list[str]:
    """Return names of optimizer attributes that imported successfully."""
    return [attr for attr, *_ in _OPTIONAL_OPTIMIZERS if attr in globals()]


def get_optimizer(name: str) -> Any:
    """Look up an optimizer by name with descriptive error handling."""
    if name in globals():
        return globals()[name]
    if name in _FAILED:
        hint, exc = _FAILED[name]
        raise ImportError(
            f"Optimizer {name!r} requires an optional dependency that is not installed: {exc}. Install with: {hint}"
        ) from exc
    known = ", ".join(sorted(attr for attr, *_ in _OPTIONAL_OPTIMIZERS))
    raise KeyError(f"Unknown optimizer {name!r}. Known optimizers: {known}")


def __getattr__(name: str) -> Any:
    if name in _FAILED:
        hint, exc = _FAILED[name]
        raise ImportError(
            f"{name!r} requires an optional dependency that is not installed: {exc}. Install with: {hint}"
        ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
