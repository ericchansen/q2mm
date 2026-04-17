"""Force field optimizers for Q2MM.

Provides a clean, composable optimization framework built on
:mod:`scipy.optimize`, :mod:`optax`, and the Q2MM clean model layer.

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

If a missing-dep optimizer is requested, you get a descriptive
``ImportError`` naming the original failure and the install hint —
not a silent ``AttributeError`` deep inside a call stack.

Use :func:`available_optimizers` to introspect what's installed and
:func:`get_optimizer` to look one up by name with explicit error handling.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

logger = logging.getLogger(__name__)

__all__ = [
    "ObjectiveFunction",
    "ReferenceData",
    "available_optimizers",
    "get_optimizer",
]

# (public attribute name, defining module, install hint).
# Order is the order they are tried (and the order returned by
# ``available_optimizers``); install hints match ``pyproject.toml`` extras.
_OPTIONAL_OPTIMIZERS: tuple[tuple[str, str, str], ...] = (
    ("ScipyOptimizer", "q2mm.optimizers.scipy_opt", "pip install 'q2mm[optimize]'"),
    ("OptimizationResult", "q2mm.optimizers.scipy_opt", "pip install 'q2mm[optimize]'"),
    ("OptimizationLoop", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("LoopResult", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("SubspaceObjective", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("SensitivityResult", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("compute_sensitivity", "q2mm.optimizers.cycling", "pip install 'q2mm[optimize]'"),
    ("OptaxOptimizer", "q2mm.optimizers.optax", "pip install 'q2mm[jax]'"),
    ("JaxOptOptimizer", "q2mm.optimizers.jaxopt_opt", "pip install 'q2mm[jax]'"),
    (
        "JaxMultiStartOptimizer",
        "q2mm.optimizers.jax_multistart",
        "pip install 'q2mm[jax]'",
    ),
    (
        "BasinHoppingOptimizer",
        "q2mm.optimizers.basinhopping",
        "pip install 'q2mm[optimize]'",
    ),
    (
        "MultiStartOptimizer",
        "q2mm.optimizers.multistart",
        "pip install 'q2mm[optimize]'",
    ),
)

# attribute → (install_hint, original ImportError) for every optimizer
# whose backing module failed to import.  Populated by ``_discover``.
_FAILED: dict[str, tuple[str, ImportError]] = {}


def _discover() -> None:
    """Eagerly import each optional optimizer; record ImportError details.

    ``ImportError`` is the expected failure mode when an optional dep is
    missing — silence is correct, but we record *why* so we can surface it
    later when the user actually tries to use the optimizer.

    Any *other* exception is logged at WARNING with traceback, since it
    indicates a real bug rather than a missing dependency.
    """
    for attr, module_path, hint in _OPTIONAL_OPTIMIZERS:
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            _FAILED[attr] = (hint, exc)
            logger.debug("Optimizer %s unavailable: %s (%s)", attr, exc, hint)
            continue
        except Exception as exc:
            logger.warning(
                "Unexpected error importing optimizer module %s: %s",
                module_path,
                exc,
                exc_info=True,
            )
            _FAILED[attr] = (hint, ImportError(str(exc)))
            continue
        if hasattr(module, attr):
            globals()[attr] = getattr(module, attr)
            __all__.append(attr)
        else:
            logger.warning(
                "Optimizer module %s loaded but exports no %s",
                module_path,
                attr,
            )


_discover()


def available_optimizers() -> list[str]:
    """Return names of optimizer attributes that imported successfully.

    The returned names are guaranteed to be importable directly from
    :mod:`q2mm.optimizers`.
    """
    return [attr for attr, *_ in _OPTIONAL_OPTIMIZERS if attr in globals()]


def get_optimizer(name: str) -> Any:
    """Look up an optimizer by name with descriptive error handling.

    Args:
        name: The public attribute name (e.g. ``"JaxOptOptimizer"``).

    Returns:
        The class or function exported by the corresponding submodule.

    Raises:
        ImportError: The optimizer's optional dependency is not installed.
            The message includes the original error and the install hint.
        KeyError: ``name`` is not a known optimizer.

    """
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
    """Provide descriptive errors for ``from q2mm.optimizers import X``.

    Without this fallback, importing an unavailable optimizer raises a
    generic ``ImportError: cannot import name 'X'`` with no clue why.
    Now the user gets the install hint at the import site.
    """
    if name in _FAILED:
        hint, exc = _FAILED[name]
        raise ImportError(
            f"{name!r} requires an optional dependency that is not installed: {exc}. Install with: {hint}"
        ) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
