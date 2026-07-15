"""Molecular mechanics backends.

Provides :class:`~q2mm.backends.mm.openmm.OpenMMBackend`,
:class:`~q2mm.backends.mm.tinker.TinkerBackend`, and (optionally)
:class:`~q2mm.backends.mm.jax_engine.JaxBackend` and
:class:`~q2mm.backends.mm.jax_md_engine.JaxMdBackend` for MM energy
evaluations.

Backend construction and availability are managed by the descriptor-based
:mod:`~q2mm.backends.registry`.  JAX-based backends are lazily imported to
avoid triggering JAX global config (x64 mode) as a side effect when only
OpenMM/Tinker are needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from q2mm.backends.mm.openmm import OpenMMBackend
from q2mm.backends.mm.tinker import TinkerBackend

if TYPE_CHECKING:
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.backends.mm.jax_md_engine import JaxMdBackend

__all__ = ["JaxBackend", "JaxMdBackend", "OpenMMBackend", "TinkerBackend"]

_LAZY_IMPORTS: dict[str, str] = {
    "JaxBackend": "q2mm.backends.mm.jax_engine",
    "JaxMdBackend": "q2mm.backends.mm.jax_md_engine",
}


def __getattr__(name: str) -> type:
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
