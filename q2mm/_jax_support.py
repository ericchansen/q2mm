"""Foundational, dependency-free lazy JAX import guard.

The single canonical place that knows how to check for JAX's presence,
import it, and configure float64 precision. Every other q2mm layer that
needs optional JAX access (:mod:`q2mm.models.hessian`,
:mod:`q2mm.backends.mm._jax_common`, ...) depends on this module rather
than re-implementing the same import-guard/float64-configuration logic
— one canonical location, per the alpha-discipline "no duplicate
implementations" rule.

This module lives at the top of ``q2mm`` (a sibling of ``constants.py``
and ``geometry.py``), not under ``q2mm.models`` or ``q2mm.backends``,
and imports nothing from the rest of ``q2mm``. That is what lets the
foundational ``q2mm.models`` layer depend on it without creating a
layering violation: it is genuinely *below* every layer, not merely
attached to one of them.

Availability checking (:func:`has_jax`) is a cheap
``importlib.util.find_spec`` call — it never imports JAX or allocates
GPU memory. The actual import and float64 configuration are deferred to
:func:`load_jax`'s first call.
"""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType

# Cheap availability check — does NOT import JAX.
_HAS_JAX: bool = importlib.util.find_spec("jax") is not None

# Populated lazily by load_jax().
_jax_module: ModuleType | None = None
_jnp_module: ModuleType | None = None
_initialized: bool = False


def has_jax() -> bool:
    """Return whether the ``jax`` package is importable.

    Side-effect-free: backed by ``importlib.util.find_spec`` computed
    once at import time. Never imports JAX or triggers CUDA
    initialization.
    """
    return _HAS_JAX


def load_jax(caller_name: str) -> tuple[ModuleType, ModuleType]:
    """Import JAX (+ ``jax.numpy``) and configure float64 on first call.

    Subsequent calls are cheap no-ops that return the cached modules.
    This is the single entry point, shared by every q2mm layer, that
    triggers ``import jax`` and any associated XLA/CUDA initialization.

    Args:
        caller_name: Name of the function/engine requesting JAX, used
            in the error message.

    Returns:
        ``(jax, jax.numpy)`` module references.

    Raises:
        ImportError: If the ``jax`` package cannot be imported.

    """
    global _jax_module, _jnp_module, _initialized  # noqa: PLW0603

    if _initialized:
        assert _jax_module is not None
        assert _jnp_module is not None
        return _jax_module, _jnp_module
    if not _HAS_JAX:
        raise ImportError(f"JAX is required for {caller_name}. Install with: pip install jax jaxlib")

    import jax as _jax
    import jax.numpy as _jnp

    # JAX defaults to float32. q2mm needs float64 throughout (MM
    # parameter optimization and Hessian eigenvalue/frequency analysis
    # are both precision-sensitive: energy differences ~1e-6 kcal/mol
    # matter).
    #
    # Honour the standard JAX_ENABLE_X64 env-var: when the user has set
    # it explicitly, we do NOT override JAX's own interpretation.
    # Otherwise we enable float64 (standard practice in JAX-based
    # chemistry packages).
    #
    # ``jax.config`` exposes registered flags as dynamic attributes (no
    # static stub declares them), so read via ``getattr`` rather than
    # direct attribute access to stay mypy-clean without ``type: ignore``.
    user_set_jax_enable_x64 = "JAX_ENABLE_X64" in os.environ
    if not getattr(_jax.config, "jax_enable_x64", False) and not user_set_jax_enable_x64:
        _jax.config.update("jax_enable_x64", True)

    _jax_module = _jax
    _jnp_module = _jnp
    _initialized = True
    return _jax_module, _jnp_module
