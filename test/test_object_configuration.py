"""Actual supplied components declare reproducible pre-execution snapshots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import q2mm
from q2mm._canonical import canonical_fingerprint
from q2mm.application import (
    ApplicationConfigurationError,
    OptimizationRun,
    capture_execution_configuration,
    optimize,
)
from q2mm.configuration import ConfigurationProvider
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.catalog import build_optimizer, optimizer_spec
from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.optax import OptaxOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.settings import (
    BasinHoppingSettings,
    OptimizationSettings,
    JaxMultiStartSettings,
    JaxOptSettings,
    MethodE2Settings,
    MultiStartSettings,
    ObjectiveSettings,
    OptaxSettings,
    ScipySettings,
)
from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow
from test.test_application import _EnergyBackend, _factory, _problem, _result


def _capture(
    optimizer: Any,
    workflow: Any = None,
    *,
    objective: ObjectiveSettings = ObjectiveSettings(),
) -> Any:
    return capture_execution_configuration(
        OptimizationSettings(objective=objective, n_evals=0),
        backend=_EnergyBackend(),
        optimizer=optimizer,
        workflow=SingleStageWorkflow() if workflow is None else workflow,
    )


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
def test_builtin_capture_uses_actual_settings_without_reconstruction(
    maxiter: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = ScipySettings(method="Nelder-Mead", maxiter=maxiter, verbose=False, use_bounds=False)
    optimizer = ScipyOptimizer(settings)

    def forbidden_constructor(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Capture reconstructed the supplied optimizer.")

    monkeypatch.setattr(ScipyOptimizer, "__init__", forbidden_constructor)
    config = _capture(optimizer).optimizer
    assert config.settings["parameters"] == settings.model_dump(mode="json")
    assert config.settings["parameters"]["maxiter"] == maxiter
    assert config.settings["parameters"]["use_bounds"] is False
    assert config.settings["version"] == q2mm.__version__
    assert optimizer.settings is settings
    with pytest.raises(ValueError, match="frozen"):
        settings.maxiter = 99


def test_distinct_iteration_limits_have_distinct_configuration() -> None:
    configs = [_capture(ScipyOptimizer(ScipySettings(maxiter=n))).optimizer.settings for n in (2, 3)]
    assert canonical_fingerprint(configs[0]) != canonical_fingerprint(configs[1])


@pytest.mark.parametrize(
    ("constructor", "settings"),
    [
        (ScipyOptimizer, ScipySettings()),
        (OptaxOptimizer, OptaxSettings()),
        (JaxOptOptimizer, JaxOptSettings()),
        (BasinHoppingOptimizer, BasinHoppingSettings()),
    ],
)
def test_builtin_provider_describes_complete_typed_settings(constructor: type, settings: Any) -> None:
    optimizer = constructor(settings)
    captured = _capture(optimizer).optimizer.settings["parameters"]
    assert captured == settings.model_dump(mode="json")
    assert optimizer.settings is settings


def test_nested_multistart_captures_live_child_identity_and_detaches_snapshot() -> None:
    inner = ScipyOptimizer(ScipySettings(method="Powell", maxiter=7, use_bounds=False, verbose=False))
    optimizer = MultiStartOptimizer(
        MultiStartSettings(n_starts=2, seed=0, perturbation_pct=0.0, verbose=False), optimizer=inner
    )
    captured = _capture(optimizer).optimizer.settings["parameters"]
    assert captured["n_starts"] == 2
    assert captured["seed"] == captured["perturbation_pct"] == 0
    child = captured["optimizer"]
    assert child["class"] == "ScipyOptimizer"
    assert child["parameters"]["maxiter"] == 7
    assert child["parameters"]["use_bounds"] is False
    assert optimizer.optimizer is inner
    with pytest.raises(TypeError):
        child["parameters"]["maxiter"] = 11


def test_jax_adapter_captures_actual_built_once_child(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = JaxMultiStartSettings(optimizer=JaxOptSettings(maxiter=0, tol=1e-4), n_starts=2, verbose=False)
    optimizer = JaxMultiStartOptimizer(settings)
    inner = optimizer.optimizer

    def forbidden_constructor(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Configuration capture refreshed the JAX adapter child.")

    monkeypatch.setattr(JaxOptOptimizer, "__init__", forbidden_constructor)
    config = _capture(optimizer, objective=ObjectiveSettings(executor="jax", gradient_mode="analytical"))
    captured = config.optimizer.settings["parameters"]
    assert captured["optimizer"]["parameters"] == settings.optimizer.model_dump(mode="json")
    assert optimizer.optimizer is inner
    assert inner.settings is settings.optimizer
    assert config.optimizer.settings["class"] == "JaxMultiStartOptimizer"


def test_cycling_provider_captures_explicit_nested_settings_without_solver_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optimizer = build_optimizer(optimizer_spec("grad-simp-multi").settings)

    def forbidden_constructor(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Cycling capture constructed an execution phase.")

    monkeypatch.setattr(MultiStartOptimizer, "__init__", forbidden_constructor)
    captured = _capture(optimizer).optimizer.settings["parameters"]
    assert captured == optimizer.settings.model_dump(mode="json")
    assert captured["full_optimizer"]["optimizer"]["maxiter"] == 200
    assert captured["simplex_optimizer"]["method"] == "Nelder-Mead"


@pytest.mark.parametrize("threshold", [0.0, 0.02])
def test_workflow_settings_are_distinct_and_deeply_immutable(threshold: float) -> None:
    source = {"bond_k": 0.0}
    settings = MethodE2Settings(negative_fc_threshold=threshold, allow_negative=False, near_zero_replace_with=source)
    workflow = MethodE2Workflow(settings)
    config = _capture(ScipyOptimizer(), workflow).workflow
    captured = config.settings["parameters"]
    assert captured["negative_fc_threshold"] == threshold
    assert captured["replace_with_round2"] == settings.replace_with_round2
    assert captured["allow_negative"] is False
    source["bond_k"] = 7.0
    assert captured["near_zero_replace_with"]["bond_k"] == settings.near_zero_replace_with["bond_k"] == 0.0
    with pytest.raises(TypeError):
        captured["near_zero_replace_with"]["bond_k"] = 9.0
    with pytest.raises(TypeError):
        settings.near_zero_replace_with["bond_k"] = 9.0
    other = _capture(
        ScipyOptimizer(), MethodE2Workflow(MethodE2Settings(negative_fc_threshold=threshold + 0.01))
    ).workflow
    assert canonical_fingerprint(config.settings) != canonical_fingerprint(other.settings)


def test_single_stage_provider_and_explicit_empty_workflow_mapping() -> None:
    workflow = SingleStageWorkflow()
    assert isinstance(workflow, ConfigurationProvider)
    assert _capture(ScipyOptimizer(), workflow).workflow.settings["parameters"] == {"name": "single-stage"}
    default = _capture(ScipyOptimizer(), MethodE2Workflow()).workflow.settings
    empty = _capture(ScipyOptimizer(), MethodE2Workflow(MethodE2Settings(near_zero_replace_with={}))).workflow.settings
    assert default["parameters"]["near_zero_replace_with"]
    assert empty["parameters"]["near_zero_replace_with"] == {}
    assert canonical_fingerprint(default) != canonical_fingerprint(empty)


@pytest.mark.parametrize(
    ("field", "left", "right"),
    [
        ("replace_with_round2", 1.0, 2.0),
        ("allow_negative", False, True),
        ("near_zero_replace_with", {"bond_k": 0.0}, {"bond_k": 1.0}),
    ],
)
def test_consequential_workflow_controls_are_distinguished(field: str, left: Any, right: Any) -> None:
    first = _capture(ScipyOptimizer(), MethodE2Workflow(MethodE2Settings(**{field: left}))).workflow
    second = _capture(ScipyOptimizer(), MethodE2Workflow(MethodE2Settings(**{field: right}))).workflow
    assert canonical_fingerprint(first.settings) != canonical_fingerprint(second.settings)


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
def test_custom_provider_returns_detached_json_snapshot(kind: str) -> None:
    payload = {"limit": 0, "enabled": False, "nested": {"values": [1, 2]}, "implementation_version": "1"}
    component = ConfiguredOptimizer(payload) if kind == "optimizer" else ConfiguredWorkflow(payload)
    execution = _capture(component) if kind == "optimizer" else _capture(ScipyOptimizer(), component)
    config = execution.optimizer if kind == "optimizer" else execution.workflow
    captured = config.settings["parameters"]
    assert captured["limit"] == 0 and captured["enabled"] is False
    payload["nested"]["values"].append(3)
    assert captured["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        captured["nested"]["other"] = 0
    assert component.calls == 0
    assert isinstance(component, ConfigurationProvider)


def test_nested_explicit_provider_and_mapping_snapshots() -> None:
    payload = {"limit": 0, "nested": MappingProxyType({"enabled": False})}
    inner = ConfiguredOptimizer(MappingProxyType(payload))
    optimizer = MultiStartOptimizer(MultiStartSettings(n_starts=2), optimizer=inner)
    captured = _capture(optimizer).optimizer.settings["parameters"]["optimizer"]
    assert captured["class"] == "ConfiguredOptimizer"
    assert captured["module"] == __name__
    assert "version" not in captured
    assert captured["parameters"]["nested"]["enabled"] is False
    payload["limit"] = 8
    assert captured["parameters"]["limit"] == 0
    assert optimizer.optimizer is inner


def test_builtin_subclass_with_complete_own_provider_is_supported() -> None:
    class DeclaredScipy(ScipyOptimizer):
        extra_control = 0

        def configuration_settings(self) -> Mapping[str, Any]:
            return {**self.settings.model_dump(mode="json"), "extra_control": self.extra_control}

    optimizer = DeclaredScipy(ScipySettings(maxiter=0, use_bounds=False))
    config = _capture(optimizer).optimizer
    assert config.settings["parameters"]["maxiter"] == 0
    assert config.settings["parameters"]["extra_control"] == 0
    assert config.settings["parameters"]["use_bounds"] is False
    assert "version" not in config.settings


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
        pytest.fail("Invalid capture reached preparation or numerical execution.")

    backend = _EnergyBackend()
    monkeypatch.setattr(backend, "prepare", forbidden)
    monkeypatch.setattr("q2mm.application.optimization.optimize", forbidden)
    with pytest.raises(expected_error, match="configuration|ConfigurationProvider|JSON") as caught:
        capture_execution_configuration(
            OptimizationSettings(n_evals=0),
            backend=backend,
            optimizer=component if kind == "optimizer" else ScipyOptimizer(),
            workflow=component if kind == "workflow" else SingleStageWorkflow(),
        )
    assert calls == []
    return caught.value


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("settings_type", "attribute"),
    [
        (ScipySettings, "ftol"),
        (OptaxSettings, "learning_rate"),
        (JaxOptSettings, "tol"),
        (BasinHoppingSettings, "T"),
        (MultiStartSettings, "perturbation_pct"),
        (MethodE2Settings, "negative_fc_threshold"),
    ],
)
def test_builtin_nonfinite_controls_are_rejected_at_settings_boundary(
    settings_type: type, attribute: str, value: float
) -> None:
    with pytest.raises(ValueError, match=attribute):
        settings_type(**{attribute: value})


@pytest.mark.parametrize("value", [np.int64(2), np.float32(0.125), np.array([0.125, 0.25])])
def test_numpy_custom_provider_contract_is_not_broadened(value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    error = _assert_preflight_failure(ConfiguredOptimizer({"value": value}), "optimizer", monkeypatch)
    assert isinstance(error.__cause__, TypeError)


@pytest.mark.parametrize("kind", ["optimizer", "workflow"])
def test_unknown_objects_and_unconfigured_builtin_subclasses_are_rejected(
    kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnknownOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Unknown optimizer ran.")

    class UnknownWorkflow:
        name = "unknown"

        def run(self, *args: Any, **kwargs: Any) -> Any:
            pytest.fail("Unknown workflow ran.")

    class DerivedScipy(ScipyOptimizer):
        extra_limit = 3

    class DerivedWorkflow(MethodE2Workflow):
        extra_limit = 3

    class IndirectScipy(DerivedScipy):
        pass

    components = (
        (UnknownOptimizer(), DerivedScipy(), IndirectScipy())
        if kind == "optimizer"
        else (UnknownWorkflow(), DerivedWorkflow())
    )
    for component in components:
        error = _assert_preflight_failure(component, kind, monkeypatch)
        assert "ConfigurationProvider.configuration_settings()" in str(error)
        if isinstance(component, (ScipyOptimizer, MethodE2Workflow)):
            assert "does not apply to subclasses" in str(error)
        else:
            assert isinstance(error.__cause__, AttributeError)


def test_nested_unsupported_optimizer_and_recursive_configuration_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnknownOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Unsupported child ran.")

    _assert_preflight_failure(MultiStartOptimizer(optimizer=UnknownOptimizer()), "optimizer", monkeypatch)
    recursive = MultiStartOptimizer()
    recursive.optimizer = recursive
    _assert_preflight_failure(recursive, "optimizer", monkeypatch)
    payload: dict[str, Any] = {}
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
def test_provider_failures_preserve_original_causes_and_interrupts(
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
    assert component.calls == 0


def test_dynamic_provider_attribute_error_preserves_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = AttributeError("configuration provider storage unavailable")

    class DynamicOptimizer:
        def optimize(self, evaluator: Any, space: Any) -> Any:
            pytest.fail("Optimizer ran before configuration capture.")

        def __getattr__(self, name: str) -> Any:
            if name == "configuration_settings":
                raise failure
            raise AttributeError(name)

    caught = _assert_preflight_failure(DynamicOptimizer(), "optimizer", monkeypatch)
    assert caught.__cause__ is failure


def test_publication_sdk_fixture_captures_fixed_configuration_and_enters() -> None:
    from test.integration.test_publication_sdk_matrix import _OneEvaluationOptimizer

    optimizer = _OneEvaluationOptimizer()
    backend = _EnergyBackend()
    workflow = SingleStageWorkflow()
    configuration = _capture(optimizer, workflow)
    before = configuration.optimizer.settings
    assert before["parameters"] == {"mode": "one-evaluation"}
    assert optimizer.entered is False
    problem = _problem()
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, ObjectiveSettings()),
        regularization=0.0,
        n_evals=0,
    )
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    assert optimizer.entered is True
    assert result.n_iterations == result.n_evaluations == 1
    assert result.method == "single-stage"
    assert result.stages[0].message == "bounded publication SDK path entered"
    assert result.initial_params.tolist() == result.final_params.tolist() == problem.active_space.baseline.tolist()
    assert run.optimizer_configuration.settings == before
    assert _capture(optimizer).optimizer.settings == before
    assert "entered" not in before["parameters"]


def test_unknown_provider_values_are_not_stringified(monkeypatch: pytest.MonkeyPatch) -> None:
    class Unknown:
        def __str__(self) -> str:
            pytest.fail("Configuration stringified an unknown value.")

    _assert_preflight_failure(ConfiguredOptimizer({"value": Unknown()}), "optimizer", monkeypatch)


def test_unknown_workflow_name_is_not_stringified() -> None:
    class Unknown:
        def __str__(self) -> str:
            pytest.fail("Configuration stringified an unknown workflow identity.")

    workflow = ConfiguredWorkflow({})
    workflow.name = Unknown()
    with pytest.raises(ApplicationConfigurationError):
        _capture(ScipyOptimizer(), workflow)


@pytest.mark.parametrize("name", [0, False, [], {}, None, ""])
def test_workflow_identity_requires_a_nonempty_string(name: Any) -> None:
    workflow = ConfiguredWorkflow({})
    workflow.name = name
    with pytest.raises(ApplicationConfigurationError):
        _capture(ScipyOptimizer(), workflow)


def test_backend_identity_records_actual_provider_not_requested_key() -> None:
    configuration = capture_execution_configuration(
        OptimizationSettings(), backend=_EnergyBackend(), optimizer=ScipyOptimizer(), workflow=SingleStageWorkflow()
    )
    assert configuration.backend.key == "synthetic-mm"
    assert configuration.backend.name == _EnergyBackend.info.name
    assert configuration.backend.role == "mm"
    assert configuration.backend.capabilities == ("energy",)
    assert configuration.backend.functional_forms == ("harmonic", "mm3")


def test_supplied_objects_execute_unchanged_and_snapshot_precedes_execution() -> None:
    optimizer = ConfiguredOptimizer({"limit": 0, "enabled": False})
    workflow = ConfiguredWorkflow({"nested": {"stages": 1}})
    problem = _problem()
    objective = ObjectiveSettings(gradient_mode="finite_difference", fd_step=0.02)
    configuration = _capture(optimizer, workflow, objective=objective)
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, _EnergyBackend(), objective),
        regularization=0.0,
        n_evals=0,
    )
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    assert optimizer.calls == workflow.calls == 1
    assert run.executor_configuration.fd_step == 0.02
    assert result.gradient_mode == "finite_difference"
    assert run.optimizer_configuration.settings["parameters"]["limit"] == 0
    assert optimizer.settings["during_execution"] is True
    assert "during_execution" not in run.optimizer_configuration.settings["parameters"]
    assert run.workflow_configuration.settings["parameters"]["nested"]["stages"] == 1
