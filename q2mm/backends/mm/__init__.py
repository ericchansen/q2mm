"""Molecular mechanics engine backends.

Provides :class:`~q2mm.backends.mm.openmm.OpenMMEngine`,
:class:`~q2mm.backends.mm.tinker.TinkerEngine`, and (optionally)
:class:`~q2mm.backends.mm.jax_engine.JaxEngine` and
:class:`~q2mm.backends.mm.jax_md_engine.JaxMDEngine` for MM energy
evaluations.

Engine availability is managed by the central
:mod:`~q2mm.backends.registry`.  Engines that depend on optional
packages (JAX, jax-md) guard their own imports and report availability
through :meth:`~q2mm.backends.base.MMEngine.is_available`.

JAX-based engines are lazily imported to avoid triggering JAX global
config (x64 mode) as a side effect when only OpenMM/Tinker are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.backends.mm.tinker import TinkerEngine

if TYPE_CHECKING:
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.backends.mm.jax_md_engine import JaxMDEngine

__all__ = ["JaxEngine", "JaxMDEngine", "OpenMMEngine", "TinkerEngine"]

_LAZY_IMPORTS: dict[str, str] = {
    "JaxEngine": "q2mm.backends.mm.jax_engine",
    "JaxMDEngine": "q2mm.backends.mm.jax_md_engine",
}


def __getattr__(name: str) -> type:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
