"""Dependency-neutral, strict snapshots of actual configured components."""

import json
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, Protocol, runtime_checkable

from q2mm._provenance import freeze_json_mapping

_BUILTINS = {
    ("q2mm.optimizers.scipy_opt", "ScipyOptimizer"),
    ("q2mm.optimizers.optax", "OptaxOptimizer"),
    ("q2mm.optimizers.jaxopt_opt", "JaxOptOptimizer"),
    ("q2mm.optimizers.basinhopping", "BasinHoppingOptimizer"),
    ("q2mm.optimizers.multistart", "MultiStartOptimizer"),
    ("q2mm.optimizers.jax_multistart", "JaxMultiStartOptimizer"),
    ("q2mm.optimizers.cycling", "CyclingOptimizer"),
    ("q2mm.workflows.single_stage", "SingleStageWorkflow"),
    ("q2mm.workflows.method_e2", "MethodE2Workflow"),
}
_CAPTURE_PATH: ContextVar[frozenset[int]] = ContextVar("q2mm_component_capture_path", default=frozenset())


class ConfigurationCaptureError(ValueError):
    """A component did not provide complete, safe configuration."""


@runtime_checkable
class ConfigurationProvider(Protocol):
    """A complete finite JSON description of the actual concrete component.

    Subclasses of built-ins must define their own provider to describe all
    consequential subclass controls. Capture must not execute or mutate.
    """

    def configuration_settings(self) -> Mapping[str, Any]:
        """Return complete settings, including actual nested components."""
        ...


def _mapping_for_json(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported configuration JSON value: {type(value).__name__}.")


def capture_component_settings(value: object) -> Mapping[str, Any]:
    """Capture one provider graph, preserving and restoring its active ancestor path."""
    cls = type(value)
    ancestors = _CAPTURE_PATH.get()
    if id(value) in ancestors:
        raise ConfigurationCaptureError(f"Recursive component configuration for {cls.__qualname__}.")
    token = _CAPTURE_PATH.set(ancestors | {id(value)})
    try:
        builtin = (cls.__module__, cls.__name__) in _BUILTINS
        inherited_builtin = any((base.__module__, base.__name__) in _BUILTINS for base in cls.__mro__[1:])
        if inherited_builtin and "configuration_settings" not in cls.__dict__:
            raise ConfigurationCaptureError(
                f"{cls.__qualname__} must implement ConfigurationProvider.configuration_settings(); "
                "built-in capture does not apply to subclasses."
            )
        provider = getattr(value, "configuration_settings")
        if not callable(provider):
            raise TypeError("configuration_settings must be callable.")
        captured = provider()
        if not isinstance(captured, Mapping):
            raise TypeError("ConfigurationProvider.configuration_settings() must return a mapping.")
        json.dumps(captured, allow_nan=False, default=_mapping_for_json)
        identity = {"class": cls.__qualname__, "module": cls.__module__}
        if builtin:
            from q2mm import __version__

            identity["version"] = __version__
        return freeze_json_mapping({**identity, "parameters": captured}, path="component.configuration")
    except ConfigurationCaptureError:
        raise
    except Exception as exc:
        raise ConfigurationCaptureError(
            f"Cannot capture configuration for {cls.__module__}.{cls.__qualname__} via "
            f"ConfigurationProvider.configuration_settings(): {exc}"
        ) from exc
    finally:
        _CAPTURE_PATH.reset(token)


__all__ = ["ConfigurationCaptureError", "ConfigurationProvider", "capture_component_settings"]
