"""Native nested defaults, explicit typed choices, and strict catalog overrides."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from q2mm.benchmarks.profiles import RunProfile, recommended_publication_profile
from q2mm.optimizers.catalog import optimizer_spec
from q2mm.settings import (
    BackendSettings,
    BasinHoppingSettings,
    GroundStateOptimizationSettings,
    MethodE2Settings,
    ObjectiveSettings,
    OptimizationSettings,
    OptaxSettings,
    ScipySettings,
    SingleStageSettings,
    TransitionStateOptimizationSettings,
)


def test_valid_false_zero_and_explicit_null_are_not_replaced() -> None:
    settings = ScipySettings(maxiter=0, verbose=False, use_bounds=False, divergence_factor=None)
    assert settings.maxiter == 0
    assert settings.verbose is settings.use_bounds is False
    assert settings.divergence_factor is None
    zero_seed = BasinHoppingSettings(local_maxiter=0, seed=0)
    assert zero_seed.local_maxiter == zero_seed.seed == 0


def test_explicit_settings_defaults_do_not_select_an_optimizer() -> None:
    settings = OptimizationSettings()
    assert settings.recipe_id == "explicit-v1"
    assert settings.backend == BackendSettings()
    assert settings.optimizer is None
    assert settings.workflow == SingleStageSettings()
    assert settings.objective == ObjectiveSettings()
    assert settings.n_evals == 1 and settings.ratio_tol is None


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
        optimizer=optimizer_spec("scipy-lbfgsb-fd").settings,
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


def test_sdk_cli_and_heck_presets_keep_their_existing_values() -> None:
    from test.test_benchmarks_cli import _parse

    gs = GroundStateOptimizationSettings()
    ts = TransitionStateOptimizationSettings()
    cli = _parse(["single", "--system", "rh-enamide"])
    assert gs.recipe_id == "recommended-jax-gs-v1" and gs.optimizer.ftol == 1e-8 and gs.ratio_tol == 10.0
    assert ts.recipe_id == "recommended-jax-ts-v1"
    assert (ts.optimizer.ftol, ts.optimizer.fc_fraction, ts.optimizer.eq_fraction) == (1e-12, 0.20, 0.05)
    assert ts.ratio_tol is None
    assert (cli.optimizer.ftol, cli.optimizer.fc_fraction, cli.optimizer.eq_fraction) == (1e-8, None, None)
    heck = recommended_publication_profile("heck-relay")
    ordinary_ts = recommended_publication_profile("rh-enamide")
    assert (heck.optimizer.ftol, heck.optimizer.fc_fraction, heck.optimizer.eq_fraction) == (1e-12, 0.05, 0.05)
    assert ordinary_ts.optimizer.fc_fraction == 0.20
    override = recommended_publication_profile("heck-relay", optimizer=ScipySettings(fc_fraction=0.12))
    assert override.optimizer.fc_fraction == 0.12
    assert RunProfile(system="rh-enamide").workflow.name == "single-stage"


def test_cli_workflow_choices_are_not_changed_by_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import q2mm.benchmarks.cli as cli

    from test.test_benchmarks_cli import _parse

    assert _parse(["single", "--system", "ch3f"]).workflow.name == "single-stage"
    assert _parse(["batch", "--system", "rh-enamide"]).workflow.name == "method-e2"
    assert _parse(["batch", "--workflow.name", "single-stage"]).workflow.name == "single-stage"
    profiles = []

    def capture(values: Any, **kwargs: Any) -> Any:
        profiles.extend(values)
        return SimpleNamespace(ok=True, candidates=(), by_status=lambda _status: ())

    monkeypatch.setattr(cli, "run_profiles", capture)
    args = [
        "matrix",
        "--system",
        "ch3f",
        "--backends",
        "jax",
        "--forms",
        "harmonic",
        "--presets",
        "scipy-nm",
        "--output",
        str(tmp_path),
    ]
    assert cli.main(args) == 0
    assert len(profiles) == 1 and profiles[0].workflow.name == "single-stage"
    assert list(tmp_path.iterdir()) == []
