"""Explicit configuration capture for supplied optimizer and workflow objects."""

from __future__ import annotations

import inspect
import json
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .models import ApplicationConfigurationError, _safe_mapping

__all__ = ["ConfigurationProvider"]


@runtime_checkable
class ConfigurationProvider(Protocol):
    """Optional configuration contract for custom application components.

    This is separate from optimizer, workflow, and backend execution
    protocols. A provider must describe all consequential configuration of
    its concrete object, including subclass and nested settings. Return
    JSON-native values with string keys and finite numbers; include any
    relevant custom implementation version explicitly. Capture must not
    execute an optimization or mutate the component.
    """

    def configuration_settings(self) -> Mapping[str, Any]:
        """Return the component's complete JSON-safe configuration."""
        ...


def _mapping_for_json(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported configuration JSON value: {type(value).__name__}.")


def _builtin_for_json(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return _mapping_for_json(value)


def _provider_settings(value: object) -> Mapping[str, Any]:
    # Attribute lookup can execute a user-defined descriptor, just like the hook.
    try:
        provider = getattr(value, "configuration_settings")
        if not callable(provider):
            raise ApplicationConfigurationError(
                f"Cannot capture configuration for {type(value).__qualname__}; "
                "implement ConfigurationProvider.configuration_settings(). "
                "Automatic built-in capture does not apply to subclasses."
            )
        settings = provider()
    except ApplicationConfigurationError:
        raise
    except Exception as exc:
        cls = type(value)
        raise ApplicationConfigurationError(
            f"Cannot capture configuration for {cls.__module__}.{cls.__qualname__} via configuration_settings(): {exc}"
        ) from exc
    if not isinstance(settings, Mapping):
        raise ApplicationConfigurationError("ConfigurationProvider.configuration_settings() must return a mapping.")
    # Validate the provider's original values, before canonical normalization
    # could turn nonfinite numbers into string sentinels.
    json.dumps(settings, allow_nan=False, default=_mapping_for_json)
    return _safe_mapping(settings, path="configuration_settings")


def _workflow_settings(value: object) -> dict[str, Any]:
    """Read supported workflow controls without rebuilding the supplied object."""
    from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow

    if type(value) is SingleStageWorkflow:
        return {"name": value.name}
    if type(value) is MethodE2Workflow:
        return {
            "name": value.name,
            **{name: getattr(value, name) for name in inspect.signature(MethodE2Workflow).parameters},
        }
    raise ApplicationConfigurationError(f"No built-in workflow configuration capture for {type(value).__qualname__}.")


def _builtin_settings(value: object, gradient_mode: str, ancestors: frozenset[int]) -> dict[str, Any] | None:
    from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
    from q2mm.optimizers.catalog import (
        _CyclingOptimizer,
        _constructor_arguments,
        _constructor_settings,
        _cycling_nested_settings,
    )
    from q2mm.optimizers.cycling import OptimizationLoop
    from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
    from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
    from q2mm.optimizers.multistart import MultiStartOptimizer
    from q2mm.optimizers.optax import OptaxOptimizer
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow

    cls = type(value)
    if cls is SingleStageWorkflow or cls is MethodE2Workflow:
        return _workflow_settings(value)
    if cls is _CyclingOptimizer:
        parameters = _constructor_arguments(OptimizationLoop, getattr(value, "_kwargs"))
        return {**parameters, **_cycling_nested_settings(parameters, gradient_mode)}
    if not any(
        cls is known
        for known in (
            ScipyOptimizer,
            OptaxOptimizer,
            JaxOptOptimizer,
            BasinHoppingOptimizer,
            MultiStartOptimizer,
            JaxMultiStartOptimizer,
        )
    ):
        return None
    parameters = {
        name: getattr(value, "optimizer_name" if cls is OptaxOptimizer and name == "optimizer" else name)
        for name in inspect.signature(cls).parameters
    }
    if cls is MultiStartOptimizer:
        parameters["optimizer"] = _component_settings(
            parameters["optimizer"], gradient_mode=gradient_mode, ancestors=ancestors
        )
    elif cls is JaxMultiStartOptimizer:
        # This adapter refreshes its inner solver on execution; the stored
        # inner instance may predate changes to the public constructor knobs.
        parameters["optimizer"] = _constructor_settings(
            JaxOptOptimizer,
            "jaxopt",
            method=parameters["method"],
            maxiter=parameters["maxiter"],
            tol=parameters["tol"],
            verbose=False,
        )
    return parameters


def _component_settings(
    value: object, *, gradient_mode: str = "none", ancestors: frozenset[int] = frozenset()
) -> Mapping[str, Any]:
    """Snapshot an exact built-in or an explicit configuration provider."""
    cls = type(value)
    if id(value) in ancestors:
        raise ApplicationConfigurationError(f"Recursive component configuration for {cls.__qualname__}.")
    identity = {"class": cls.__qualname__, "module": cls.__module__}
    try:
        parameters = _builtin_settings(value, gradient_mode, ancestors | {id(value)})
        if parameters is None:
            captured = _provider_settings(value)
        else:
            from q2mm import __version__

            json.dumps(parameters, allow_nan=False, default=_builtin_for_json)
            identity["version"] = __version__
            captured = parameters
        return _safe_mapping({**identity, "parameters": captured}, path="component.configuration")
    except ApplicationConfigurationError:
        raise
    except (AttributeError, LookupError, TypeError, ValueError, RuntimeError) as exc:
        raise ApplicationConfigurationError(
            f"Cannot capture configuration for {cls.__module__}.{cls.__qualname__}: {exc}"
        ) from exc
