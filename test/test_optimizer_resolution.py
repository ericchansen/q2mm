"""Native nested defaults, explicit typed choices, and strict catalog overrides."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from q2mm.benchmarks.profiles import RunProfile, recommended_publication_profile
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG, optimizer_settings, optimizer_spec
from q2mm.settings import (
    BackendSettings,
    GroundStateOptimizationSettings,
    MethodE2Settings,
    ObjectiveSettings,
    OptimizationSettings,
    OptaxSettings,
    OptimizerOverrides,
    ScipySettings,
    SingleStageSettings,
    TransitionStateOptimizationSettings,
)


@pytest.mark.parametrize("key", tuple(OPTIMIZER_CATALOG))
def test_explicit_zero_iteration_cap_is_not_a_default(key: str) -> None:
    settings = optimizer_settings(key, OptimizerOverrides(maxiter=0))
    if settings.kind == "cycling":
        assert settings.simplex_optimizer.maxiter == 0
        full = settings.full_optimizer
        assert (full.optimizer.maxiter if full.kind == "multistart" else full.maxiter) == 0
    elif settings.kind == "multistart":
        assert settings.optimizer.maxiter == 0
    else:
        field = (
            "max_steps"
            if settings.kind == "optax"
            else "local_maxiter"
            if settings.kind == "basinhopping"
            else "maxiter"
        )
        assert getattr(settings, field) == 0


@pytest.mark.parametrize("key", tuple(OPTIMIZER_CATALOG))
def test_omitted_iteration_cap_preserves_named_policy(key: str) -> None:
    assert optimizer_settings(key, OptimizerOverrides()) == optimizer_spec(key).settings
    assert optimizer_settings(key, OptimizerOverrides(maxiter=None)) == optimizer_spec(key).settings


@pytest.mark.parametrize(
    ("key", "overrides"),
    [
        ("scipy-nm", OptimizerOverrides(learning_rate=0.01)),
        ("optax-adam", OptimizerOverrides(ftol=1e-8)),
        ("jaxopt-lbfgs", OptimizerOverrides(seed=0)),
        ("basinhopping", OptimizerOverrides(eq_fraction=None)),
        ("multi-lbfgsb-5", OptimizerOverrides(fc_fraction=None)),
        ("grad-simp", OptimizerOverrides(ftol=1e-8)),
    ],
)
def test_inapplicable_fields_are_errors_even_when_explicit_default_or_none(
    key: str, overrides: OptimizerOverrides
) -> None:
    with pytest.raises(ValueError, match="Unknown options"):
        optimizer_settings(key, overrides)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maxiter", False),
        ("maxiter", -1),
        ("maxiter", "0"),
        ("ftol", 0.0),
        ("ftol", False),
        ("ftol", float("nan")),
        ("ftol", float("inf")),
        ("fc_fraction", False),
        ("fc_fraction", 0.0),
        ("fc_fraction", 1.01),
        ("eq_fraction", False),
        ("seed", False),
        ("unknown_option", 0),
    ],
)
def test_invalid_typed_overrides_fail_at_trust_boundary(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        OptimizerOverrides(**{field: value})


def test_valid_false_zero_and_explicit_null_are_not_replaced() -> None:
    settings = ScipySettings(maxiter=0, verbose=False, use_bounds=False, divergence_factor=None)
    assert settings.maxiter == 0
    assert settings.verbose is settings.use_bounds is False
    assert settings.divergence_factor is None
    zero_seed = optimizer_settings("basinhopping", OptimizerOverrides(maxiter=0, seed=0))
    assert zero_seed.local_maxiter == zero_seed.seed == 0
    assert OptimizerOverrides(fc_fraction=None).model_fields_set == {"fc_fraction"}
    assert OptimizerOverrides().model_fields_set == set()


def test_explicit_settings_defaults_do_not_select_an_optimizer() -> None:
    settings = OptimizationSettings()
    assert settings.recipe_id == "explicit-v1"
    assert settings.backend == BackendSettings()
    assert settings.optimizer is None
    assert settings.workflow == SingleStageSettings()
    assert settings.objective == ObjectiveSettings()
    assert settings.n_evals == 1 and settings.ratio_tol is None
    assert settings.requested_fields == ()


@pytest.mark.parametrize(
    ("settings_type", "recipe_id", "ftol", "fc_fraction", "eq_fraction", "ratio_tol"),
    [
        (GroundStateOptimizationSettings, "recommended-jax-gs-v1", 1e-8, None, None, 10.0),
        (TransitionStateOptimizationSettings, "recommended-jax-ts-v1", 1e-12, 0.20, 0.05, None),
    ],
)
def test_sdk_default_numerical_policies_are_preserved(
    settings_type: type,
    recipe_id: str,
    ftol: float,
    fc_fraction: float | None,
    eq_fraction: float | None,
    ratio_tol: float | None,
) -> None:
    settings = settings_type()
    assert settings.backend.key == "jax"
    assert settings.optimizer.method == "L-BFGS-B"
    assert settings.optimizer.maxiter == 500
    assert settings.optimizer.verbose is False
    assert settings.workflow == SingleStageSettings()
    assert settings.objective == ObjectiveSettings(executor="jax", gradient_mode="analytical")
    assert settings.recipe_id == recipe_id
    assert settings.optimizer.ftol == ftol
    assert settings.optimizer.fc_fraction == fc_fraction
    assert settings.optimizer.eq_fraction == eq_fraction
    assert settings.ratio_tol == ratio_tol
    assert settings.requested_fields == ()


def test_native_partial_dictionary_updates_retain_unspecified_ts_defaults() -> None:
    optimizer = {"maxiter": 0, "ftol": 1e-8, "fc_fraction": None, "use_bounds": False}
    settings = TransitionStateOptimizationSettings(
        optimizer=optimizer,
        objective={"regularization": 0.02},
        backend={"options": {"threads": 0, "enabled": False}},
        n_evals=0,
        ratio_tol=0.0,
    )
    assert isinstance(settings.optimizer, ScipySettings)
    assert settings.optimizer.method == "L-BFGS-B"
    assert settings.optimizer.ftol == 1e-8
    assert settings.optimizer.fc_fraction is None
    assert settings.optimizer.eq_fraction == 0.05
    assert settings.optimizer.maxiter == settings.n_evals == settings.ratio_tol == 0
    assert settings.optimizer.use_bounds is settings.optimizer.verbose is False
    assert settings.objective.executor == "jax" and settings.objective.gradient_mode == "analytical"
    assert settings.objective.regularization == 0.02
    assert settings.backend.key == "jax"
    assert settings.backend.options == {"threads": 0, "enabled": False}
    assert settings.requested_fields == ("backend", "n_evals", "objective", "optimizer", "ratio_tol")
    optimizer["maxiter"] = 999
    assert settings.optimizer.maxiter == 0
    with pytest.raises(ValueError, match="frozen"):
        settings.optimizer.maxiter = 7
    with pytest.raises(TypeError):
        settings.backend.options["threads"] = 99


def test_complete_typed_nested_components_replace_defaults_and_preserve_values() -> None:
    solver = ScipySettings(method="Powell", maxiter=0, use_bounds=False, verbose=False)
    workflow = MethodE2Settings(negative_fc_threshold=0.0, near_zero_replace_with={})
    objective = ObjectiveSettings(gradient_mode="finite_difference", fd_step=0.02)
    settings = TransitionStateOptimizationSettings(
        optimizer=solver,
        workflow=workflow,
        objective=objective,
        backend=BackendSettings(key="synthetic"),
        n_evals=0,
    )
    assert settings.optimizer == solver
    assert settings.workflow == workflow
    assert settings.objective == objective
    assert settings.optimizer.fc_fraction is settings.optimizer.eq_fraction is None
    assert settings.objective.executor == "python"
    assert settings.recipe_id == "recommended-jax-ts-v1"


def test_different_typed_solver_family_is_not_merged_with_scipy_defaults() -> None:
    solver = OptaxSettings(optimizer="sgd", max_steps=0, schedule="cosine", verbose=False)
    settings = TransitionStateOptimizationSettings(optimizer=solver)
    assert settings.optimizer == solver
    assert settings.optimizer.kind == "optax"
    assert settings.optimizer.max_steps == 0
    assert settings.optimizer.schedule == "cosine"


@pytest.mark.parametrize("step", [1e-4, 0.02, 0.04])
def test_explicit_objective_fd_step_is_retained(step: float) -> None:
    settings = OptimizationSettings(
        optimizer=optimizer_settings("scipy-lbfgsb-fd"),
        objective={"gradient_mode": "finite_difference", "fd_step": step},
    )
    assert settings.objective.executor == "python"
    assert settings.objective.fd_step == step


@pytest.mark.parametrize(
    "values",
    [
        {"n_evals": False},
        {"n_evals": -1},
        {"ratio_tol": -1.0},
        {"ratio_tol": float("inf")},
        {"optimizer": "scipy-nm"},
        {"objective": {"fd_step": 0.0}},
        {"objective": {"executor": "python", "gradient_mode": "analytical"}},
        {"objective": {"executor": "jax", "gradient_mode": "none"}},
        {"unknown_option": 0},
    ],
)
def test_final_settings_validate_explicit_execution_choices(values: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        OptimizationSettings(**values)


@pytest.mark.parametrize(
    ("field", "value"),
    [("ftol", -1.0), ("ftol", float("nan")), ("fc_fraction", False), ("maxiter", False)],
)
def test_native_partial_solver_updates_use_the_same_strict_boundary(field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        TransitionStateOptimizationSettings(optimizer={field: value})


def test_native_settings_ignore_ambient_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "N_EVALS": "999",
        "RATIO_TOL": "0.0",
        "RECIPE_ID": "ambient",
        "BACKEND": '{"key":"ambient"}',
        "OPTIMIZER": '{"kind":"scipy","maxiter":999}',
        "OBJECTIVE": '{"executor":"python","gradient_mode":"none"}',
    }.items():
        monkeypatch.setenv(key, value)
    settings = TransitionStateOptimizationSettings()
    assert settings.recipe_id == "recommended-jax-ts-v1"
    assert settings.optimizer.maxiter == 500
    assert settings.backend.key == "jax"
    assert settings.objective.executor == "jax"
    assert settings.n_evals == 1 and settings.ratio_tol is None
    assert settings.requested_fields == ()


def test_requested_fields_are_raw_python_presence_not_native_merged_defaults() -> None:
    omitted = TransitionStateOptimizationSettings()
    explicit = TransitionStateOptimizationSettings(optimizer={}, ratio_tol=None, n_evals=1)
    assert explicit.optimizer == omitted.optimizer
    assert explicit.requested_fields == ("n_evals", "optimizer", "ratio_tol")
    assert omitted.requested_fields == ()
    assert "requested_fields" not in explicit.model_dump()
    assert "_requested_fields" not in explicit.model_dump()


def test_sdk_cli_and_heck_presets_keep_their_existing_values() -> None:
    from q2mm.benchmarks.cli import _build_parser

    gs = GroundStateOptimizationSettings()
    ts = TransitionStateOptimizationSettings()
    cli = _build_parser().parse_args(["single", "--system", "rh-enamide"])
    assert gs.recipe_id == "recommended-jax-gs-v1" and gs.optimizer.ftol == 1e-8 and gs.ratio_tol == 10.0
    assert ts.recipe_id == "recommended-jax-ts-v1"
    assert (ts.optimizer.ftol, ts.optimizer.fc_fraction, ts.optimizer.eq_fraction) == (1e-12, 0.20, 0.05)
    assert ts.ratio_tol is None
    assert (cli.ftol, cli.fc_fraction, cli.eq_fraction) == (1e-8, None, None)
    heck = recommended_publication_profile("heck-relay")
    ordinary_ts = recommended_publication_profile("rh-enamide")
    assert (heck.ftol, heck.fc_fraction, heck.eq_fraction) == (1e-12, 0.05, 0.05)
    assert ordinary_ts.fc_fraction == 0.20
    override = recommended_publication_profile("heck-relay", fc_fraction=0.12)
    assert override.fc_fraction == 0.12
    assert RunProfile(system="rh-enamide").workflow == "single-stage"


def test_cli_workflow_choices_are_not_changed_by_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import q2mm.benchmarks.cli as cli

    parser = cli._build_parser()
    assert parser.parse_args(["single", "--system", "ch3f"]).workflow == "single-stage"
    assert parser.parse_args(["batch", "--system", "rh-enamide"]).workflow == "method-e2"
    assert parser.parse_args(["batch", "--workflow", "single-stage"]).workflow == "single-stage"
    profiles = []

    def capture(values: Any, **kwargs: Any) -> Any:
        profiles.extend(values)
        return SimpleNamespace(ok=True, candidates=(), by_status=lambda _status: ())

    monkeypatch.setattr(cli, "run_profiles", capture)
    args = parser.parse_args(
        [
            "matrix",
            "--system",
            "ch3f",
            "--backend",
            "jax",
            "--form",
            "harmonic",
            "--optimizer",
            "scipy-nm",
            "--output",
            str(tmp_path),
        ]
    )
    assert args.func(args) == 0
    assert len(profiles) == 1 and profiles[0].workflow == "single-stage"
    assert list(tmp_path.iterdir()) == []


def test_native_settings_do_not_import_or_build_optional_algorithms() -> None:
    script = """
import builtins
import sys
blocked = {"jax", "jaxlib", "jaxopt", "optax", "scipy"}
original = builtins.__import__
def guarded(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(name)
    return original(name, globals, locals, fromlist, level)
builtins.__import__ = guarded
from q2mm.settings import TransitionStateOptimizationSettings
settings = TransitionStateOptimizationSettings(optimizer={"maxiter":0})
assert settings.optimizer.ftol == 1e-12
assert settings.optimizer.maxiter == 0
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
