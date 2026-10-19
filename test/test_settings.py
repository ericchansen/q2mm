"""Shared declarative settings reject invalid values without runtime discovery."""

import json
import subprocess
import sys

import pytest
from pydantic import ValidationError

from q2mm.settings import (
    BackendSettings,
    CyclingSettings,
    FrequencyMatchingSettings,
    JaxMultiStartSettings,
    JaxOptSettings,
    MethodE2Settings,
    MultiStartSettings,
    ObjectiveSettings,
    OptaxSettings,
    OptimizerSettings,
    OptimizationSettings,
    GroundStateOptimizationSettings,
    TransitionStateOptimizationSettings,
    PreparationSettings,
    ScipySettings,
    QFuerzaSettings,
)


def test_nested_options_are_detached_deeply_immutable_and_json_roundtrip() -> None:
    options = {"nested": {"sequence": [None, 0, False, {"limit": 1}]}}
    settings = BackendSettings(key="external-plugin", options=options)
    options["nested"]["sequence"].append(2)
    assert settings.options["nested"]["sequence"] == (None, 0, False, {"limit": 1})
    with pytest.raises(TypeError):
        settings.options["nested"]["sequence"][-1]["limit"] = 3
    with pytest.raises(ValidationError, match="frozen"):
        settings.key = "changed"
    assert BackendSettings.model_validate_json(settings.model_dump_json()) == settings
    assert json.loads(settings.model_dump_json()) == {
        "key": "external-plugin",
        "options": {"nested": {"sequence": [None, 0, False, {"limit": 1}]}},
    }
    supplied = settings.factory_options()
    assert type(supplied) is type(supplied["nested"]) is dict
    assert type(supplied["nested"]["sequence"]) is list
    supplied["nested"]["sequence"][-1]["limit"] = 99
    assert settings.options["nested"]["sequence"][-1]["limit"] == 1
    assert settings.factory_options()["nested"]["sequence"][-1]["limit"] == 1


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), object(), {1: "bad"}])
def test_backend_options_require_strict_finite_json(value: object) -> None:
    with pytest.raises(ValidationError):
        BackendSettings(options={"nested": {"bad": value}})


@pytest.mark.parametrize(
    "options",
    [
        {"auth": "redacted"},
        {"AUTH": "redacted"},
        {"service_auth": "redacted"},
        {"auth-config": {"mode": "token"}},
        {"nested": {"api_key": "redacted"}},
        {"nested": {"clientSecret": "redacted"}},
        {"nested": {"header": "Bearer secret-value"}},
        {"nested": {"endpoint": "https://user:pass@example.com"}},
    ],
)
def test_backend_options_reject_credential_keys_and_values_at_source(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="credential-like"):
        BackendSettings(options=options)
    assert BackendSettings(options={"authoritative_case_ids": ["case-1"]}).options["authoritative_case_ids"] == (
        "case-1",
    )


def test_omission_and_explicit_none_zero_false_default_are_distinct() -> None:
    omitted = ScipySettings()
    explicit = ScipySettings(maxiter=0, ftol=1e-8, fc_fraction=None, use_bounds=False)
    assert not omitted.model_fields_set
    assert explicit.model_fields_set == {"maxiter", "use_bounds", "ftol", "fc_fraction"}
    assert ScipySettings(maxiter=0, use_bounds=False).maxiter == 0
    assert not ScipySettings(maxiter=0, use_bounds=False).use_bounds


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (ScipySettings, {"maxiter": False}),
        (ScipySettings, {"ftol": -1.0}),
        (ScipySettings, {"fc_fraction": False}),
        (ScipySettings, {"maxls": 0}),
        (ScipySettings, {"unknown": 1}),
        (CyclingSettings, {"max_cycles": -1}),
        (FrequencyMatchingSettings, {"threshold": float("inf")}),
        (FrequencyMatchingSettings, {"weight": 0.0}),
        (MethodE2Settings, {"near_zero_replace_with": {"torsion_k": 0.0}}),
        (ObjectiveSettings, {"executor": "jax", "gradient_mode": "none"}),
        (ObjectiveSettings, {"gradient_mode": "analytical"}),
        (PreparationSettings, {"stationary_point": "mixed"}),
    ],
)
def test_invalid_controls_fail_at_settings_boundary(factory: type, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        factory(**kwargs)


def test_nested_solvers_workflows_and_settings_json_are_immutable() -> None:
    values = {"bond_k": 0.0}
    workflow = MethodE2Settings(near_zero_replace_with=values)
    values["bond_k"] = 12.0
    assert workflow.near_zero_replace_with["bond_k"] == 0.0
    with pytest.raises(TypeError):
        workflow.near_zero_replace_with["bond_k"] = 9.0
    execution = OptimizationSettings(optimizer=CyclingSettings(), workflow=workflow)
    restored = OptimizationSettings.model_validate_json(execution.model_dump_json())
    assert restored.model_dump(mode="json") == execution.model_dump(mode="json")
    with pytest.raises(ValidationError):
        execution.optimizer.full_optimizer.maxiter = 999


def test_native_preset_partial_update_preserves_explicit_none_zero_false_and_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RATIO_TOL", "99.0")
    monkeypatch.setenv("OPTIMIZER__FTOL", "0.5")
    default = TransitionStateOptimizationSettings()
    updated = TransitionStateOptimizationSettings(
        optimizer={"maxiter": 0, "fc_fraction": None, "verbose": False},
        ratio_tol=None,
    )
    assert (updated.optimizer.maxiter, updated.optimizer.ftol, updated.optimizer.eq_fraction) == (0, 1e-12, 0.05)
    assert updated.optimizer.fc_fraction is None and updated.optimizer.verbose is False
    assert default.optimizer.fc_fraction == 0.20 and default.optimizer.maxiter == 500
    assert GroundStateOptimizationSettings().ratio_tol == 10.0
    assert GroundStateOptimizationSettings().optimizer.ftol == 1e-8
    copied = default.model_copy(deep=True)
    assert copied == default and copied is not default
    with pytest.raises(ValueError):
        TransitionStateOptimizationSettings(optimizer={"ftol": float("nan")})
    with pytest.raises(ValueError):
        TransitionStateOptimizationSettings(optimizer={"misspelled": 1})


@pytest.mark.parametrize(
    "settings_type",
    [OptimizationSettings, GroundStateOptimizationSettings, TransitionStateOptimizationSettings],
)
def test_backend_and_executor_are_explicit_coherent_typed_choices(settings_type: type) -> None:
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        settings_type(
            backend=BackendSettings(key="openmm"),
            objective=ObjectiveSettings(executor="jax"),
        )
    python = settings_type(
        backend=BackendSettings(key="openmm"),
        objective=ObjectiveSettings(executor="python"),
    )
    assert python.objective.executor == "python" and python.objective.gradient_mode == "none"
    restored = settings_type.model_validate_json(python.model_dump_json())
    assert restored.model_dump() == python.model_dump()
    if settings_type is not OptimizationSettings:
        original = settings_type()
        assert original.objective.executor == "jax" and original.objective.gradient_mode == "analytical"
        assert python.optimizer == original.optimizer


def test_explicit_executor_supplies_native_gradient_default_without_backend_inference() -> None:
    assert ObjectiveSettings(executor="jax").gradient_mode == "analytical"
    assert ObjectiveSettings(executor="python").gradient_mode == "none"
    assert ObjectiveSettings(executor="python", gradient_mode="finite_difference").fd_step == 1e-4
    with pytest.raises(ValueError, match="analytical"):
        ObjectiveSettings(executor="jax", gradient_mode="none")


@pytest.mark.parametrize(
    "optimizer,mode",
    [
        (JaxOptSettings(), "none"),
        (JaxOptSettings(), "finite_difference"),
        (JaxMultiStartSettings(), "none"),
        (JaxMultiStartSettings(), "finite_difference"),
        (OptaxSettings(), "none"),
        (MultiStartSettings(optimizer=JaxOptSettings()), "none"),
        (CyclingSettings(full_optimizer=JaxOptSettings()), "none"),
        (CyclingSettings(full_optimizer=MultiStartSettings(optimizer=OptaxSettings())), "none"),
    ],
)
def test_unexecutable_nested_optimizer_objective_combinations_fail_validation(
    optimizer: OptimizerSettings, mode: str
) -> None:
    with pytest.raises(ValueError, match="requires a"):
        OptimizationSettings(
            backend=BackendSettings(key="openmm"),
            optimizer=optimizer,
            objective=ObjectiveSettings.model_validate({"gradient_mode": mode}),
        )


@pytest.mark.parametrize(
    "optimizer",
    [OptaxSettings(), MultiStartSettings(optimizer=OptaxSettings()), CyclingSettings(full_optimizer=OptaxSettings())],
)
def test_gradient_optimizers_accept_explicit_python_finite_differences(optimizer: OptimizerSettings) -> None:
    settings = OptimizationSettings(
        backend=BackendSettings(key="openmm"),
        optimizer=optimizer,
        objective=ObjectiveSettings(gradient_mode="finite_difference"),
    )
    assert OptimizationSettings.model_validate_json(settings.model_dump_json()) == settings


def test_subclass_post_init_cannot_bypass_inherited_native_component_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from typing import Any
    from pydantic import PrivateAttr

    class CommandSettings(TransitionStateOptimizationSettings):
        _post_init_ran: bool = PrivateAttr(default=False)

        def model_post_init(self, context: Any) -> None:
            self._post_init_ran = True

    monkeypatch.setenv("BACKEND__KEY", "openmm")
    monkeypatch.setenv("OBJECTIVE__EXECUTOR", "python")
    named = CommandSettings()
    assert named._post_init_ran
    assert named.backend.key == "jax" and named.objective.executor == "jax"
    assert named.optimizer.ftol == 1e-12 and named.optimizer.fc_fraction == 0.20
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        CommandSettings(backend=BackendSettings(key="openmm"))
    valid = CommandSettings(
        backend=BackendSettings(key="openmm"),
        objective=ObjectiveSettings(executor="python"),
    )
    assert valid._post_init_ran and valid.objective.gradient_mode == "none"
    assert valid.optimizer == named.optimizer
    restored = CommandSettings.model_validate_json(valid.model_dump_json())
    assert restored.model_dump() == valid.model_dump() and restored._post_init_ran
    invalid = named.model_dump(mode="json")
    invalid["backend"]["key"] = "openmm"
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        CommandSettings.model_validate_json(json.dumps(invalid))


def test_settings_and_catalog_inspection_do_not_import_numerical_implementations() -> None:
    program = """
import importlib.abc
import sys
forbidden = {"numpy", "scipy", "jax", "jaxlib", "openmm", "optax", "jaxopt", "psi4"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split(".")[0] not in forbidden, fullname
sys.meta_path.insert(0, Block())
from q2mm.settings import OptimizationSettings
from q2mm.optimizers import OPTIMIZER_CATALOG, optimizer_spec
assert OptimizationSettings().backend.key == "jax"
for key in OPTIMIZER_CATALOG:
    spec = optimizer_spec(key)
    OptimizationSettings(optimizer=spec.settings, objective=spec.objective).model_dump_json()
assert not forbidden.intersection(sys.modules)
assert not any(m.startswith("q2mm.optimizers.scipy_opt") for m in sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_multinode_configuration_cycles_fail_directly_and_capture_path_is_restored() -> None:
    from q2mm.configuration import ConfigurationCaptureError, capture_component_settings
    from q2mm.application.configuration import _component_settings
    from q2mm.application.models import ApplicationConfigurationError
    from q2mm.optimizers.multistart import MultiStartOptimizer

    first = MultiStartOptimizer()
    second = MultiStartOptimizer(optimizer=first)
    first.optimizer = second
    with pytest.raises(ConfigurationCaptureError, match="Recursive") as neutral:
        capture_component_settings(first)
    assert neutral.value.__cause__ is None
    with pytest.raises(ApplicationConfigurationError, match="Recursive") as application:
        _component_settings(first)
    assert isinstance(application.value.__cause__, ConfigurationCaptureError)
    assert application.value.__cause__.__cause__ is None
    healthy = capture_component_settings(MultiStartOptimizer())
    assert healthy["parameters"]["optimizer"]["class"] == "ScipyOptimizer"


@pytest.mark.parametrize(
    "settings",
    [
        PreparationSettings(stationary_point="ground_state", initialize="provided"),
        PreparationSettings(stationary_point="ground_state", functional_form="harmonic"),
        PreparationSettings(
            stationary_point="ground_state",
            functional_form="harmonic",
            initialize="qfuerza",
            qfuerza=QFuerzaSettings(replace_with=1.5, zero_torsions=False),
        ),
    ],
)
def test_preparation_effective_settings_roundtrip_without_field_origin_rules(
    settings: PreparationSettings,
) -> None:
    from q2mm.preparation import prepare
    from q2mm.models.observations import ObservationSet
    from test.test_preparation import _template, _water

    restored = PreparationSettings.model_validate_json(settings.model_dump_json())
    assert restored == settings
    force_field = _template() if settings.initialize == "provided" else None
    first = prepare(_water(), settings=settings, force_field=force_field, observations=ObservationSet())
    second = prepare(_water(), settings=restored, force_field=force_field, observations=ObservationSet())
    assert first.preparation_provenance == second.preparation_provenance
    assert first.starting_force_field == second.starting_force_field
    if settings.initialize == "provided":
        assert settings.qfuerza is None and restored.qfuerza is None
        assert first.starting_force_field is force_field
    else:
        assert settings.qfuerza is not None and restored.qfuerza is not None


def test_preparation_does_not_silently_ignore_conflicting_initialization_controls() -> None:
    with pytest.raises(ValueError, match="cannot be applied"):
        PreparationSettings(
            stationary_point="ground_state",
            initialize="provided",
            qfuerza=QFuerzaSettings(replace_with=1.5),
        )
    with pytest.raises(ValueError, match="requires qfuerza"):
        PreparationSettings(stationary_point="ground_state", initialize="qfuerza", qfuerza=None)


@pytest.mark.parametrize(
    "mode,executor,expected",
    [("none", "python", None), ("analytical", "jax", None), ("finite_difference", "python", 1e-4)],
)
def test_applicable_fd_controls_have_value_based_defaults_and_roundtrip(
    mode: str,
    executor: str,
    expected: float | None,
) -> None:
    settings = ObjectiveSettings.model_validate({"executor": executor, "gradient_mode": mode})
    assert settings.fd_step == expected
    assert ObjectiveSettings.model_validate_json(settings.model_dump_json()) == settings
    if expected is None:
        assert (
            ObjectiveSettings.model_validate(
                {"executor": executor, "gradient_mode": mode, "fd_step": None},
            )
            == settings
        )
        for step in (1e-4, 0.001):
            with pytest.raises(ValueError, match="applies only"):
                ObjectiveSettings.model_validate({"executor": executor, "gradient_mode": mode, "fd_step": step})
    else:
        for step in (1e-4, 0.001):
            explicit = ObjectiveSettings(gradient_mode="finite_difference", fd_step=step)
            assert explicit.fd_step == step
            assert ObjectiveSettings.model_validate_json(explicit.model_dump_json()) == explicit
        with pytest.raises(ValueError, match="requires"):
            ObjectiveSettings(gradient_mode="finite_difference", fd_step=None)


def test_scipy_zero_function_tolerance_preserves_native_execution() -> None:
    pytest.importorskip("scipy")
    import numpy as np
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from test._shared import QuadraticEvaluator

    evaluator = QuadraticEvaluator(np.array([1.0, 2.0]), initial=np.array([3.0, 2.5]))
    settings = ScipySettings(ftol=0.0, maxiter=2, use_bounds=False, verbose=False)
    result = ScipyOptimizer(settings).optimize(evaluator, evaluator.space)
    assert settings.ftol == 0.0
    assert np.isfinite(result.final_score) and result.final_score < result.initial_score
    np.testing.assert_allclose(result.final_params, [1.0, 2.0], atol=1e-8)
    with pytest.raises(ValueError):
        ScipySettings(gtol=0.0)


def test_zero_hop_width_and_zero_cycle_budget_preserve_native_controls() -> None:
    pytest.importorskip("scipy")
    import numpy as np
    from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
    from q2mm.optimizers.cycling import CyclingOptimizer
    from q2mm.settings import BasinHoppingSettings
    from test._shared import QuadraticEvaluator

    evaluator = QuadraticEvaluator(np.array([1.0, 2.0]), initial=np.array([3.0, 2.5]))
    basin = BasinHoppingOptimizer(BasinHoppingSettings(stepsize=0.0, niter=1, seed=0, verbose=False))
    result = basin.optimize(evaluator, evaluator.space)
    assert np.isfinite(result.final_score) and result.final_score <= result.initial_score
    evaluator = QuadraticEvaluator(np.array([1.0, 2.0]), initial=np.array([3.0, 2.5]))
    loop = CyclingOptimizer(CyclingSettings(max_cycles=0, verbose=False))
    result = loop.optimize(evaluator, evaluator.space)
    assert result.n_iterations == result.n_evaluations == 0 and not result.stages
    assert result.gradient_mode == "none"
    np.testing.assert_array_equal(result.final_params, evaluator.space.baseline)
