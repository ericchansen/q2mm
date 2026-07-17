"""Benchmark system registry.

A small, explicit key -> module mapping. Importing this module never
loads scientific data, optional backends, or a specific system's
dependencies — each system's ``load`` function is resolved and imported
lazily, on first use, via :func:`load_system`.

Usage::

    from q2mm.benchmarks.systems import SYSTEM_KEYS, load_system

    case = load_system("rh-enamide")

"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

from q2mm.benchmarks.cases import BenchmarkCase

__all__ = ["SYSTEM_KEYS", "SystemMetadata", "load_system", "system_metadata"]

# key -> module name under q2mm.benchmarks.systems.  Every module listed
# here exposes exactly one public ``load(...) -> BenchmarkCase`` function;
# no data or optional backend is imported until load_system() actually
# calls it.
_REGISTRY: dict[str, str] = {
    "ch3f": "q2mm.benchmarks.systems.ch3f",
    "ch3f-sn2": "q2mm.benchmarks.systems.ch3f_sn2",
    "rh-enamide": "q2mm.benchmarks.systems.rh_enamide",
    "heck-relay": "q2mm.benchmarks.systems.heck_relay",
    "pd-allyl": "q2mm.benchmarks.systems.pd_allyl",
    "pd-conjugate": "q2mm.benchmarks.systems.pd_conjugate",
    "rh-conjugate": "q2mm.benchmarks.systems.rh_conjugate",
    "ferrocene": "q2mm.benchmarks.systems.ferrocene",
}

SYSTEM_KEYS: tuple[str, ...] = tuple(_REGISTRY)
"""Registered benchmark system keys, in registry order."""


def load_system(key: str, **kwargs: Any) -> BenchmarkCase:
    """Build the :class:`~q2mm.benchmarks.cases.BenchmarkCase` for *key*.

    Lazily imports the system's module (see :data:`SYSTEM_KEYS` for the
    available keys) and calls its ``load(**kwargs)`` function. Keyword
    arguments are forwarded verbatim — see each system module's ``load``
    signature for what it accepts (e.g. ``ch3f``/``ch3f-sn2`` require an
    ``backend=``; the published-FF systems accept ``data_roots=``,
    ``starting_point=``, ``qfuerza_replace_with=``, ``functional_form=``).

    Args:
        key: Registered system key, e.g. ``"rh-enamide"``.
        **kwargs: Forwarded to the system module's ``load`` function.

    Returns:
        A fully-populated :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Raises:
        KeyError: If *key* is not in :data:`SYSTEM_KEYS`.

    """
    if key not in _REGISTRY:
        raise KeyError(f"Unknown benchmark system {key!r}; available: {sorted(_REGISTRY)}")
    module = importlib.import_module(_REGISTRY[key])
    case: BenchmarkCase = module.load(**kwargs)
    return case


@dataclass(frozen=True)
class SystemMetadata:
    """Cheap, load-free descriptive metadata for one registered system.

    Attributes:
        key: Registry key (e.g. ``"rh-enamide"``).
        name: Human-readable system name.
        description: One-line human-readable description.
        default_forms: Functional forms to benchmark by default.

    """

    key: str
    name: str
    description: str
    default_forms: tuple[str, ...]


def system_metadata(key: str) -> SystemMetadata:
    """Return *key*'s declarative metadata without loading any data.

    Imports the system's module (cheap — no scientific data, optional
    backend, or file I/O happens at import time; only the ``load(...)``
    call in :func:`load_system` triggers that) and reads its
    ``NAME``/``DESCRIPTION``/``DEFAULT_FORMS`` module-level constants.

    Use this for CLI listing/filtering that must not eagerly load a
    system's (potentially large, external) training data; use
    :func:`load_system` to actually build a :class:`BenchmarkCase`.

    Args:
        key: Registered system key, e.g. ``"rh-enamide"``.

    Returns:
        The system's :class:`SystemMetadata`.

    Raises:
        KeyError: If *key* is not in :data:`SYSTEM_KEYS`.

    """
    if key not in _REGISTRY:
        raise KeyError(f"Unknown benchmark system {key!r}; available: {sorted(_REGISTRY)}")
    module = importlib.import_module(_REGISTRY[key])
    return SystemMetadata(
        key=key,
        name=module.NAME,
        description=module.DESCRIPTION,
        default_forms=tuple(module.DEFAULT_FORMS),
    )
