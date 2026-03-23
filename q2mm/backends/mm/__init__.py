"""Molecular mechanics engine backends.

Provides :class:`~q2mm.backends.mm.openmm.OpenMMEngine`,
:class:`~q2mm.backends.mm.tinker.TinkerEngine`, and (optionally)
:class:`~q2mm.backends.mm.jax_engine.JaxEngine` for MM energy evaluations.
"""

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.backends.mm.tinker import TinkerEngine

try:
    from q2mm.backends.mm.jax_engine import JaxEngine

    _HAS_JAX_ENGINE = True
except ImportError:
    _HAS_JAX_ENGINE = False

__all__ = ["OpenMMEngine", "TinkerEngine"]
if _HAS_JAX_ENGINE:
    __all__ = [*__all__, "JaxEngine"]
