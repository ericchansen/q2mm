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
"""

from q2mm.backends.mm.jax_engine import JaxEngine
from q2mm.backends.mm.jax_md_engine import JaxMDEngine
from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.backends.mm.tinker import TinkerEngine

__all__ = ["JaxEngine", "JaxMDEngine", "OpenMMEngine", "TinkerEngine"]
