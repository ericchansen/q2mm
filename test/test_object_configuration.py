"""Supplied components must declare reproducible, immutable configuration."""

from __future__ import annotations

import inspect
import subprocess
import sys
from collections.abc import Callable, Mapping
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import numpy as np

import q2mm
import q2mm.application as application
from q2mm._canonical import canonical_fingerprint
from q2mm.application import ApplicationConfigurationError, optimize
from q2mm.application.optimization import _resolve_optimizer, _resolve_workflow
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.catalog import resolve_optimizer
from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.optax import OptaxOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow
from test.test_application import _EnergyBackend, _problem, _result


def _optimizer_configuration(optimizer: Any, **kwargs: Any) -> Any:
    resolved = _resolve_optimizer(
        optimizer,
        None,
        executor=kwargs.get("executor", "python"),
        requested_gradient_mode=kwargs.get("gradient_mode"),
        requested_fd_step=kwargs.get("fd_step"),
    )
    assert resolved[0] is optimizer
    return resolved


class ConfiguredOptimizer:
    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.calls = 0

    def configuration_settings(self) -> Mapping[str, Any]:
        return self.settings

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        self.calls += 1
        self.settings["during_execution"] = True
        return _result(_problem(), gradient_mode=evaluator.gradient_mode.value)


class ConfiguredWorkflow:
    name = "custom-workflow"

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.calls = 0

    def configuration_settings(self) -> Mapping[str, Any]:
        return self.settings

    def run(self, problem: Any, make_evaluator: Any, optimizer: Any, **kwargs: Any) -> OptimizationResult:
        self.calls += 1
        return SingleStageWorkflow().run(problem, make_evaluator, optimizer, **kwargs)


@pytest.mark.parametrize("maxiter", [0, 2, 3])
def test_builtin_optimizer_captures_current_values_without_reconstruction(
    maxiter: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    optimizer = ScipyOptimizer(method="Nelder-Mead", maxiter=10, verbose=False, use_bounds=False)
    optimizer.maxiter = maxiter
    original = ScipyOptimizer.__init__

    @wraps(original)
    def forbidden_constructor(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Capture reconstructed a supplied optimizer")

    monkeypatch.setattr(ScipyOptimizer, "__init__", forbidden_constructor)
    config = _optimizer_configuration(optimizer)[1]
    assert config.settings["parameters"]["maxiter"] == maxiter
    assert config.settings["parameters"]["use_bounds"] is False
    assert config.settings["parameters"]["verbose"] is False
    assert config.settings["version"] == q2mm.__version__
    assert optimizer.maxiter == maxiter


def test_distinct_iteration_limits_have_distinct_configuration() -> None:
    configs = [_optimizer_configuration(ScipyOptimizer(method="Nelder-Mead", maxiter=n))[1].settings for n in (2, 3)]
    assert canonical_fingerprint(configs[0]) != canonical_fingerprint(configs[1])


@pytest.mark.parametrize("constructor", [ScipyOptimizer, OptaxOptimizer, JaxOptOptimizer, BasinHoppingOptimizer])
def test_exact_builtin_constructor_fields_are_captured(constructor: type) -> None:
    optimizer = constructor()
    captured = _optimizer_configuration(optimizer)[1].settings["parameters"]
    for name in inspect.signature(constructor).parameters:
        attribute = "optimizer_name" if name == "optimizer" and constructor is OptaxOptimizer else name
        assert captured[name] == getattr(optimizer, attribute), name


def test_nested_multistart_captures_the_live_child_and_freezes_it() -> None:
    inner = ScipyOptimizer(method="Powell", maxiter=7, use_bounds=False, verbose=False)
    optimizer = MultiStartOptimizer(inner, n_starts=2, seed=0, perturbation_pct=0.0, verbose=False)
    parameters = _optimizer_configuration(optimizer)[1].settings["parameters"]
    assert parameters["n_starts"] == 2
    assert parameters["seed"] == parameters["perturbation_pct"] == 0
    child = parameters["optimizer"]["parameters"]
    assert child["maxiter"] == 7
    assert child["use_bounds"] is False
    inner.maxiter = 11
    assert child["maxiter"] == 7
    assert optimizer.optimizer is inner


def test_jax_adapter_captures_effective_not_stale_inner_settings() -> None:
    optimizer = JaxMultiStartOptimizer(n_starts=2, maxiter=9, verbose=False)
    optimizer.maxiter = 0
    optimizer.tol = 1e-4
    stale_inner = optimizer.optimizer
    parameters = _optimizer_configuration(optimizer, executor="jax")[1].settings["parameters"]
    assert parameters["maxiter"] == parameters["optimizer"]["maxiter"] == 0
    assert parameters["tol"] == parameters["optimizer"]["tol"] == 1e-4
    assert optimizer.optimizer is stale_inner
    assert stale_inner.maxiter == 9


def test_catalog_cycling_object_reuses_effective_nested_capture() -> None:
    optimizer, _ = resolve_optimizer("grad-simp-multi", {"maxiter": 0, "max_cycles": 0})
    captured = _optimizer_configuration(optimizer)[1].settings["parameters"]
    assert captured["max_cycles"] == captured["full_maxiter"] == captured["simp_maxiter"] == 0
    assert captured["full_optimizer"]["optimizer"]["maxiter"] == 0
    assert captured["simplex_optimizer"]["method"] == "Nelder-Mead"
    optimizer._kwargs["full_maxiter"] = 5
    updated = _optimizer_configuration(optimizer)[1].settings["parameters"]
    assert updated["full_optimizer"]["optimizer"]["maxiter"] == 5
    assert captured["full_maxiter"] == 0


@pytest.mark.parametrize("threshold", [0.0, 0.02])
def test_workflow_settings_are_current_distinct_and_immutable(threshold: float) -> None:
    workflow = MethodE2Workflow(
        negative_fc_threshold=threshold, allow_negative=False, near_zero_replace_with={"bond_k": 0.0}
    )
    returned, config = _resolve_workflow(workflow, None)
    assert returned is workflow
    captured = config.settings["parameters"]
    assert captured["negative_fc_threshold"] == threshold
    assert captured["replace_with_round2"] == workflow.replace_with_round2
    assert captured["allow_negative"] is False
    assert captured["near_zero_replace_with"]["bond_k"] == 0.0
    workflow.near_zero_replace_with["bond_k"] = 7.0
    assert captured["near_zero_replace_with"]["bond_k"] == 0.0
    with pytest.raises(TypeError):
        captured["near_zero_replace_with"]["bond_k"] = 9.0
    other = _resolve_workflow(MethodE2Workflow(negative_fc_threshold=threshold + 0.01), None)[1]
    assert canonical_fingerprint(config.settings) != canonical_fingerprint(other.settings)


def test_single_stage_object_is_supported_without_a_provider() -> None:
    workflow = SingleStageWorkflow()
    returned, config = _resolve_workflow(workflow, None)
    assert returned is workflow
    assert config.settings["parameters"] == {"name": "single-stage"}


def test_workflow_expanded_defaults_and_explicit_empty_mapping_are_captured() -> None:
    default = MethodE2Workflow()
    default_settings = _resolve_workflow(default, None)[1].settings
    empty_settings = _resolve_workflow(MethodE2Workflow(near_zero_replace_with={}), None)[1].settings
    assert default_settings["parameters"]["near_zero_replace_with"] == default.near_zero_replace_with
    assert empty_settings["parameters"]["near_zero_replace_with"] == {}
    assert canonical_fingerprint(default_settings) != canonical_fingerprint(empty_settings)


@pytest.mark.parametrize(
    ("field", "left", "right"),
    [
        ("replace_with_round2", 1.0, 2.0),
        ("allow_negative", False, True),
        ("near_zero_replace_with", {"bond_k": 0.0}, {"bond_k": 1.0}),
    ],
)
def test_consequential_workflow_controls_are_distinguished(field: str, left: Any, right: Any) -> None:
    first = _resolve_workflow(MethodE2Workflow(**{field: left}), None)[1]
    second = _resolve_workflow(MethodE2Workflow(**{field: right}), None)[1]
    assert canonical_fingerprint(first.settings) != canonical_fingerprint(second.settings)


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
def test_custom_provider_returns_detached_json_snapshot(kind: str) -> None:
    payload = {"limit": 0, "enabled": False, "nested": {"values": [1, 2]}, "implementation_version": "1"}
    component = ConfiguredOptimizer(payload) if kind == "optimizer" else ConfiguredWorkflow(payload)
    config = _optimizer_configuration(component)[1] if kind == "optimizer" else _resolve_workflow(component, None)[1]
    captured = config.settings["parameters"]
    assert captured["limit"] == 0 and captured["enabled"] is False
    payload["nested"]["values"].append(3)
    assert captured["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        captured["nested"]["other"] = 0
    assert component.calls == 0


def test_custom_provider_protocol_is_separate_and_public() -> None:
    protocol = getattr(application, "ConfigurationProvider", None)
    assert protocol is not None
    assert isinstance(ConfiguredOptimizer({}), protocol)
    assert isinstance(ConfiguredWorkflow({}), protocol)


def test_nested_explicit_provider_and_mapping_snapshots() -> None:
    payload = {"limit": 0, "nested": MappingProxyType({"enabled": False})}
    inner = ConfiguredOptimizer(MappingProxyType(payload))
    optimizer = MultiStartOptimizer(inner, n_starts=2)
    captured = _optimizer_configuration(optimizer)[1].settings["parameters"]["optimizer"]
    assert captured["class"] == "ConfiguredOptimizer"
    assert "version" not in captured
    assert captured["parameters"]["nested"]["enabled"] is False
    payload["limit"] = 8
    assert captured["parameters"]["limit"] == 0
    assert optimizer.optimizer is inner


def test_builtin_subclass_with_explicit_complete_provider_is_supported() -> None:
    class DeclaredScipy(ScipyOptimizer):
        extra_control = 0

        def configuration_settings(self) -> Mapping[str, Any]:
            return {
                **{name: getattr(self, name) for name in inspect.signature(ScipyOptimizer).parameters},
                "extra_control": self.extra_control,
            }

    optimizer = DeclaredScipy(maxiter=0, use_bounds=False)
    config = _optimizer_configuration(optimizer)[1]
    assert config.settings["parameters"]["maxiter"] == 0
    assert config.settings["parameters"]["extra_control"] == 0
    assert config.settings["parameters"]["use_bounds"] is False


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"value": object()},
        {"value": Path("local.dat")},
        {"value": {1, 2}},
        {"value": float("nan")},
        {"value": float("inf")},
        {1: "non-string-key"},
        {"nested": {"api_key": "do-not-record"}},
    ],
    ids=["not-mapping", "list", "object", "path", "set", "nan", "infinity", "key", "sensitive-field"],
)
def test_invalid_provider_fails_before_execution(kind: str, payload: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    component = ConfiguredOptimizer(payload) if kind == "optimizer" else ConfiguredWorkflow(payload)
    _assert_preflight_failure(component, kind, monkeypatch)


def _assert_preflight_failure(
    component: Any,
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expected_error: type[BaseException] = ApplicationConfigurationError,
) -> BaseException:
    calls = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        pytest.fail("Invalid configuration reached preparation or optimization")

    backend = _EnergyBackend()
    monkeypatch.setattr(backend, "prepare", forbidden)
    monkeypatch.setattr("q2mm.application.optimization.execute_optimization", forbidden)
    with pytest.raises(expected_error, match="configuration|ConfigurationProvider|JSON") as caught:
        optimize(
            _problem(),
            backend,
            recipe="explicit",
            optimizer=component if kind == "optimizer" else ScipyOptimizer(maxiter=0),
            workflow=component if kind == "workflow" else SingleStageWorkflow(),
            executor="python",
            n_evals=0,
        )
    assert calls == []
    return caught.value


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize(
    ("factory", "attribute", "kind"),
    [
        (ScipyOptimizer, "ftol", "optimizer"),
        (OptaxOptimizer, "learning_rate", "optimizer"),
        (JaxOptOptimizer, "tol", "optimizer"),
        (BasinHoppingOptimizer, "T", "optimizer"),
        pytest.param(lambda: MultiStartOptimizer(ScipyOptimizer()), "perturbation_pct", "optimizer", id="multistart"),
        (JaxMultiStartOptimizer, "tol", "optimizer"),
        (MethodE2Workflow, "negative_fc_threshold", "workflow"),
    ],
)
def test_builtin_nonfinite_controls_fail_before_preparation(
    factory: Callable[[], object], attribute: str, kind: str, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    component = factory()
    setattr(component, attribute, value)
    error = _assert_preflight_failure(component, kind, monkeypatch)
    assert isinstance(error.__cause__, ValueError)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "negative-inf"])
@pytest.mark.parametrize("case", ["multistart-inner", "cycling", "workflow-replacements"])
def test_nested_builtin_nonfinite_controls_fail_before_preparation(
    case: str, value: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    if case == "multistart-inner":
        inner = ScipyOptimizer()
        inner.ftol = value
        component = MultiStartOptimizer(inner)
    elif case == "cycling":
        component, _ = resolve_optimizer("grad-simp")
        component._kwargs["eps"] = value
    else:
        component = MethodE2Workflow()
        component.near_zero_replace_with["bond_k"] = value
    kind = "workflow" if case == "workflow-replacements" else "optimizer"
    error = _assert_preflight_failure(component, kind, monkeypatch)
    assert isinstance(error.__cause__, ValueError)


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [
        ("maxiter", np.int64(7), 7),
        ("ftol", np.float32(0.125), 0.125),
        ("eps", np.array([0.125, 0.25]), (0.125, 0.25)),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_finite_numpy_builtin_controls_keep_canonical_snapshot(
    attribute: str, value: Any, expected: Any, nested: bool
) -> None:
    if isinstance(value, np.ndarray):
        value = value.copy()
    inner = ScipyOptimizer()
    setattr(inner, attribute, value)
    optimizer = MultiStartOptimizer(inner, n_starts=np.int64(2)) if nested else inner
    config = _optimizer_configuration(optimizer)[1].settings["parameters"]
    captured = config["optimizer"]["parameters"] if nested else config
    assert captured[attribute] == expected
    assert getattr(inner, attribute) is value
    if isinstance(value, np.ndarray):
        value[:] = 9.0
        assert captured[attribute] == expected


def test_finite_numpy_workflow_controls_are_supported() -> None:
    workflow = MethodE2Workflow()
    workflow.negative_fc_threshold = np.float32(0.125)
    config = _resolve_workflow(workflow, None)[1].settings["parameters"]
    assert config["negative_fc_threshold"] == 0.125
    assert isinstance(workflow.negative_fc_threshold, np.float32)


@pytest.mark.parametrize("value", [np.float32("nan"), np.float32("inf"), np.array([0.1, np.inf])])
def test_nonfinite_numpy_builtin_controls_fail_before_normalization(
    value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    error = _assert_preflight_failure(ScipyOptimizer(eps=value), "optimizer", monkeypatch)
    assert isinstance(error.__cause__, ValueError)


@pytest.mark.parametrize("value", [np.int64(2), np.float32(0.125), np.array([0.125, 0.25])])
def test_numpy_custom_provider_contract_is_not_broadened(value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    error = _assert_preflight_failure(ConfiguredOptimizer({"value": value}), "optimizer", monkeypatch)
    assert isinstance(error.__cause__, TypeError)


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
def test_unknown_objects_and_unconfigured_subclasses_are_rejected(kind: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class UnknownOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Unknown optimizer ran")

    class UnknownWorkflow:
        name = "unknown"

        def run(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("Unknown workflow ran")

    class DerivedScipy(ScipyOptimizer):
        extra_limit = 3

    class DerivedWorkflow(MethodE2Workflow):
        extra_limit = 3

    for component in (
        (UnknownOptimizer(), DerivedScipy()) if kind == "optimizer" else (UnknownWorkflow(), DerivedWorkflow())
    ):
        error = _assert_preflight_failure(component, kind, monkeypatch)
        assert "implement ConfigurationProvider.configuration_settings()" in str(error)
        assert "does not apply to subclasses" in str(error)
        assert isinstance(error.__cause__, AttributeError)


def test_nested_unsupported_optimizer_and_recursive_configuration_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnknownOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Unsupported child ran")

    _assert_preflight_failure(MultiStartOptimizer(UnknownOptimizer()), "optimizer", monkeypatch)
    recursive = MultiStartOptimizer(ScipyOptimizer())
    recursive.optimizer = recursive
    _assert_preflight_failure(recursive, "optimizer", monkeypatch)
    payload = {}
    payload["self"] = payload
    _assert_preflight_failure(ConfiguredOptimizer(payload), "optimizer", monkeypatch)


def test_noncallable_and_raising_providers_fail_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    class Noncallable(ConfiguredOptimizer):
        configuration_settings = None

    class Raising(ConfiguredOptimizer):
        def configuration_settings(self) -> Mapping[str, Any]:
            raise ValueError("configuration unavailable")

    _assert_preflight_failure(Noncallable({}), "optimizer", monkeypatch)
    _assert_preflight_failure(Raising({}), "optimizer", monkeypatch)


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
@pytest.mark.parametrize("boundary", ["lookup", "call"])
@pytest.mark.parametrize(
    "error_type",
    [
        OSError,
        AttributeError,
        ArithmeticError,
        Exception,
        ApplicationConfigurationError,
        KeyboardInterrupt,
        SystemExit,
        BaseException,
    ],
)
def test_provider_failure_boundary_preserves_causes_and_interrupts(
    kind: str, boundary: str, error_type: type[BaseException], monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = error_type("configuration provider unavailable")
    if isinstance(failure, ApplicationConfigurationError):
        failure.__cause__ = OSError("original provider cause")
    original_cause = failure.__cause__

    class RaisingProvider:
        @property
        def configuration_settings(self) -> Callable[[], Mapping[str, object]]:
            if boundary == "lookup":
                raise failure

            def unavailable() -> Mapping[str, object]:
                raise failure

            return unavailable

    class RaisingOptimizer(RaisingProvider, ConfiguredOptimizer):
        pass

    class RaisingWorkflow(RaisingProvider, ConfiguredWorkflow):
        pass

    component = RaisingOptimizer({}) if kind == "optimizer" else RaisingWorkflow({})
    expected_error = ApplicationConfigurationError if isinstance(failure, Exception) else error_type
    caught = _assert_preflight_failure(component, kind, monkeypatch, expected_error=expected_error)
    if isinstance(failure, ApplicationConfigurationError) or not isinstance(failure, Exception):
        assert caught is failure
        assert caught.__cause__ is original_cause
    else:
        assert caught.__cause__ is failure
        assert f"{type(component).__module__}.{type(component).__qualname__}" in str(caught)
        assert "configuration_settings" in str(caught)
        if boundary == "call":
            assert "implement ConfigurationProvider" not in str(caught)
    assert component.calls == 0


def test_dynamic_provider_attribute_error_preserves_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = AttributeError("configuration provider storage unavailable")

    class DynamicOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Optimizer ran before configuration capture")

        def __getattr__(self, name: str) -> Any:
            if name == "configuration_settings":
                raise failure
            raise AttributeError(name)

    caught = _assert_preflight_failure(DynamicOptimizer(), "optimizer", monkeypatch)
    assert caught.__cause__ is failure


def test_publication_sdk_fixture_captures_fixed_configuration_and_enters() -> None:
    from test.integration.test_publication_sdk_matrix import _OneEvaluationOptimizer

    optimizer = _OneEvaluationOptimizer()
    before = _optimizer_configuration(optimizer)[1].settings
    assert before["parameters"] == {"mode": "one-evaluation"}
    assert optimizer.entered is False
    problem = _problem()
    run = optimize(
        problem,
        _EnergyBackend(),
        recipe="explicit",
        optimizer=optimizer,
        workflow="single-stage",
        executor="python",
        n_evals=0,
    )
    assert optimizer.entered is True
    assert run.result.n_iterations == run.result.n_evaluations == 1
    assert run.result.method == "single-stage"
    assert run.result.stages[0].message == "bounded publication SDK path entered"
    assert (
        run.result.initial_params.tolist() == run.result.final_params.tolist() == problem.active_space.baseline.tolist()
    )
    assert run.optimizer_configuration.settings == before
    assert _optimizer_configuration(optimizer)[1].settings == before
    assert "entered" not in run.optimizer_configuration.settings["parameters"]


def test_unknown_values_and_workflow_names_are_not_stringified(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unknown:
        def __str__(self) -> str:
            pytest.fail("Configuration stringified an unknown value")

    _assert_preflight_failure(ConfiguredOptimizer({"value": Unknown()}), "optimizer", monkeypatch)
    workflow = ConfiguredWorkflow({})
    workflow.name = Unknown()
    _assert_preflight_failure(workflow, "workflow", monkeypatch)


def test_supplied_objects_execute_unchanged_and_snapshot_precedes_execution() -> None:
    optimizer = ConfiguredOptimizer({"limit": 0, "enabled": False})
    workflow = ConfiguredWorkflow({"nested": {"stages": 1}})
    run = optimize(
        _problem(),
        _EnergyBackend(),
        recipe="explicit",
        optimizer=optimizer,
        workflow=workflow,
        executor="python",
        gradient_mode="finite_difference",
        fd_step=0.02,
        n_evals=0,
    )
    assert optimizer.calls == workflow.calls == 1
    assert run.executor_configuration.fd_step == 0.02
    assert run.result.gradient_mode == "finite_difference"
    assert run.optimizer_configuration.settings["parameters"]["limit"] == 0
    assert optimizer.settings["during_execution"] is True
    assert "during_execution" not in run.optimizer_configuration.settings["parameters"]
    assert run.workflow_configuration.settings["parameters"]["nested"]["stages"] == 1


def test_builtin_capture_and_protocol_export_do_not_import_optional_runtimes() -> None:
    script = """
import builtins
import sys
blocked = {"jax", "jaxlib", "jaxopt", "optax", "scipy"}
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"Unexpected optional runtime import: {name}")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
from q2mm.application import ConfigurationProvider
from q2mm.application.optimization import _resolve_optimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer
_resolve_optimizer(ScipyOptimizer(), None, executor="python", requested_gradient_mode=None, requested_fd_step=None)
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
