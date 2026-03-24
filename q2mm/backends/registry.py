"""Central engine registry for Q2MM backends.

Provides decorator-based registration and lazy discovery of MM and QM
engines.  Import this module and use :func:`get_engine` to obtain an
engine instance by name, or :func:`available_engines` to list engines
whose dependencies are installed.

Example::

    from q2mm.backends.registry import get_engine, available_engines

    engine = get_engine("openmm")
    print(available_engines())  # ["openmm", "jax", ...]
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.backends.base import MMEngine, QMEngine

logger = logging.getLogger(__name__)


class EngineNotAvailable(RuntimeError):
    """Raised when a requested engine name is not found in the registry."""

    def __init__(self, name: str, *, available: list[str] | None = None) -> None:
        self.name = name
        self.available = available or []
        msg = f"Engine {name!r} is not registered."
        if self.available:
            msg += f" Registered engines: {', '.join(sorted(self.available))}"
        super().__init__(msg)


# ---- Internal state ----------------------------------------------------------

_MM_ENGINES: dict[str, type[MMEngine]] = {}
_QM_ENGINES: dict[str, type[QMEngine]] = {}
_discovered = False

# Modules containing engine classes decorated with @register_mm / @register_qm.
# Each module guards its own third-party imports with try/except so importing
# the module always succeeds — is_available() handles runtime checks.
_ENGINE_MODULES = [
    "q2mm.backends.mm.openmm",
    "q2mm.backends.mm.tinker",
    "q2mm.backends.mm.jax_engine",
    "q2mm.backends.mm.jax_md_engine",
    "q2mm.backends.qm.psi4",
]


# ---- Decorators --------------------------------------------------------------


def register_mm(name: str) -> Callable[[type[MMEngine]], type[MMEngine]]:
    """Class decorator that registers an :class:`MMEngine` subclass.

    Args:
        name: Short lowercase key (e.g. ``"openmm"``, ``"jax-md"``).

    Usage::

        @register_mm("openmm")
        class OpenMMEngine(MMEngine):
            ...

    """

    def decorator(cls: type[MMEngine]) -> type[MMEngine]:
        _MM_ENGINES[name] = cls
        return cls

    return decorator


def register_qm(name: str) -> Callable[[type[QMEngine]], type[QMEngine]]:
    """Class decorator that registers a :class:`QMEngine` subclass.

    Args:
        name: Short lowercase key (e.g. ``"psi4"``).

    Usage::

        @register_qm("psi4")
        class Psi4Engine(QMEngine):
            ...

    """

    def decorator(cls: type[QMEngine]) -> type[QMEngine]:
        _QM_ENGINES[name] = cls
        return cls

    return decorator


# ---- Lazy discovery ----------------------------------------------------------


def _discover() -> None:
    """Import all known engine modules to trigger ``@register_*`` decorators."""
    global _discovered
    if _discovered:
        return
    _discovered = True
    for module_path in _ENGINE_MODULES:
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # pragma: no cover
            logger.debug("Could not import engine module %s: %s", module_path, exc)


def _check_available(cls: type) -> bool:
    """Instantiate an engine and call ``is_available()`` safely."""
    try:
        return cls().is_available()
    except Exception:
        return False


# ---- Public API: retrieval ---------------------------------------------------


def get_engine(name: str, **kwargs: Any) -> MMEngine | QMEngine:
    """Instantiate a registered engine by name.

    Searches both MM and QM registries.

    Args:
        name: Registry key (e.g. ``"openmm"``, ``"psi4"``).
        **kwargs: Forwarded to the engine constructor.

    Returns:
        An engine instance.

    Raises:
        EngineNotAvailable: If *name* is not registered.

    """
    _discover()
    if name in _MM_ENGINES:
        return _MM_ENGINES[name](**kwargs)
    if name in _QM_ENGINES:
        return _QM_ENGINES[name](**kwargs)
    all_names = sorted(set(list(_MM_ENGINES) + list(_QM_ENGINES)))
    raise EngineNotAvailable(name, available=all_names)


def get_mm_engine(name: str, **kwargs: Any) -> MMEngine:
    """Instantiate a registered MM engine by name.

    Args:
        name: Registry key (e.g. ``"openmm"``).
        **kwargs: Forwarded to the engine constructor.

    Raises:
        EngineNotAvailable: If *name* is not in the MM registry.

    """
    _discover()
    if name not in _MM_ENGINES:
        raise EngineNotAvailable(name, available=sorted(_MM_ENGINES))
    return _MM_ENGINES[name](**kwargs)


def get_qm_engine(name: str, **kwargs: Any) -> QMEngine:
    """Instantiate a registered QM engine by name.

    Args:
        name: Registry key (e.g. ``"psi4"``).
        **kwargs: Forwarded to the engine constructor.

    Raises:
        EngineNotAvailable: If *name* is not in the QM registry.

    """
    _discover()
    if name not in _QM_ENGINES:
        raise EngineNotAvailable(name, available=sorted(_QM_ENGINES))
    return _QM_ENGINES[name](**kwargs)


# ---- Public API: introspection -----------------------------------------------


def available_engines() -> list[str]:
    """Return names of all engines whose dependencies are installed.

    This instantiates each engine to call ``is_available()``, catching
    any exceptions silently.
    """
    _discover()
    return sorted(name for name, cls in {**_MM_ENGINES, **_QM_ENGINES}.items() if _check_available(cls))


def available_mm_engines() -> list[str]:
    """Return names of available MM engines."""
    _discover()
    return sorted(name for name, cls in _MM_ENGINES.items() if _check_available(cls))


def available_qm_engines() -> list[str]:
    """Return names of available QM engines."""
    _discover()
    return sorted(name for name, cls in _QM_ENGINES.items() if _check_available(cls))


def registered_engines() -> dict[str, type]:
    """Return all registered engine classes (regardless of availability)."""
    _discover()
    return {**_MM_ENGINES, **_QM_ENGINES}


def registered_mm_engines() -> dict[str, type[MMEngine]]:
    """Return all registered MM engine classes."""
    _discover()
    return dict(_MM_ENGINES)


def registered_qm_engines() -> dict[str, type[QMEngine]]:
    """Return all registered QM engine classes."""
    _discover()
    return dict(_QM_ENGINES)
