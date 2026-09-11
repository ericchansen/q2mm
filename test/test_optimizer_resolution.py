"""Construction-path characterization and intentional entry-point policies."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from q2mm.application.optimization import _recommended_defaults, _resolve_optimizer
from q2mm.benchmarks.profiles import RunProfile, recommended_publication_profile
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG, OptimizerSpec, resolve_optimizer
from q2mm.optimizers.cycling import OptimizationLoop
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.optax import OptaxOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer


def _loop(method: str, **options: Any) -> OptimizationLoop:
    return OptimizationLoop(
        MagicMock(spec=ObjectiveEvaluator), MagicMock(spec=ActiveParameterSpace), full_method=method, **options
    )


def _full(loop: OptimizationLoop) -> Any:
    return loop._build_full_optimizer()


def _view(optimizer: Any) -> dict[str, Any]:
    return {
        name: _view(optimizer.optimizer)
        if name == "optimizer" and type(optimizer) is MultiStartOptimizer
        else getattr(optimizer, "optimizer_name" if name == "optimizer" and type(optimizer) is OptaxOptimizer else name)
        for name in inspect.signature(type(optimizer)).parameters
    }


@pytest.mark.parametrize(
    ("method", "constructor", "options"),
    [
        ("L-BFGS-B", ScipyOptimizer, {"method": "L-BFGS-B", "maxiter": 7, "eps": 0.02}),
        ("Nelder-Mead", ScipyOptimizer, {"method": "Nelder-Mead", "maxiter": 7, "eps": 0.02}),
        ("BFGS", ScipyOptimizer, {"method": "BFGS", "maxiter": 7, "eps": 0.02}),
        ("optax:adam", OptaxOptimizer, {"optimizer": "adam", "max_steps": 7, "schedule": None}),
        ("optax:adam+cosine", OptaxOptimizer, {"optimizer": "adam", "max_steps": 7, "schedule": "cosine"}),
        ("optax:sgd+exponential", OptaxOptimizer, {"optimizer": "sgd", "max_steps": 7, "schedule": "exponential"}),
        ("jaxopt:lbfgs", JaxOptOptimizer, {"method": "lbfgs", "maxiter": 7}),
        ("jaxopt:lbfgsb", JaxOptOptimizer, {"method": "lbfgsb", "maxiter": 7}),
        ("jaxopt:gradient_descent", JaxOptOptimizer, {"method": "gradient_descent", "maxiter": 7}),
        ("basinhopping", BasinHoppingOptimizer, {"local_method": "L-BFGS-B", "local_maxiter": 7}),
        ("basinhopping:", BasinHoppingOptimizer, {"local_method": "L-BFGS-B", "local_maxiter": 7}),
        ("basinhopping:  Powell  ", BasinHoppingOptimizer, {"local_method": "Powell", "local_maxiter": 7}),
        ("basinhopping-cold", BasinHoppingOptimizer, {"local_method": "L-BFGS-B", "local_maxiter": 7}),
    ],
)
def test_cycling_spellings_construct_the_same_solver(method: str, constructor: type, options: dict[str, Any]) -> None:
    actual = _full(_loop(method, full_maxiter=7, eps=0.02))
    expected = constructor(**options, verbose=False)
    assert type(actual) is constructor
    assert _view(actual) == _view(expected)


@pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead", "Powell", "BFGS"])
def test_cycling_multi_inner_inherits_only_its_declared_policy(method: str) -> None:
    optimizer = _full(_loop(f"multi:{method}", full_maxiter=0, eps=0.02))
    assert type(optimizer) is MultiStartOptimizer
    assert _view(optimizer) == {
        "optimizer": _view(ScipyOptimizer(method=method, maxiter=0, eps=0.02, verbose=False)),
        "n_starts": 5,
        "perturbation_pct": 0.1,
        "seed": None,
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("key", "method", "attribute", "catalog_value", "cycling_value"),
    [
        ("scipy-lbfgsb", "L-BFGS-B", "maxiter", 500, 200),
        ("optax-adam", "optax:adam", "max_steps", 2000, 200),
        ("basinhopping", "basinhopping", "niter", 25, 50),
        ("basinhopping", "basinhopping", "seed", 0, None),
        ("multi-lbfgsb-5", "multi:L-BFGS-B", "seed", 0, None),
    ],
)
def test_named_catalog_and_cycling_defaults_remain_visibly_different(
    key: str, method: str, attribute: str, catalog_value: Any, cycling_value: Any
) -> None:
    catalog, _ = resolve_optimizer(key)
    cycling = _full(_loop(method))
    assert getattr(catalog, attribute) == catalog_value
    assert getattr(cycling, attribute) == cycling_value


def test_catalog_schedule_and_basin_spelling_policy_is_preserved() -> None:
    spec = OptimizerSpec(
        key="inline", label="Inline", method="optax:adam+cosine", evaluator="jax", gradient_mode="analytical"
    )
    with pytest.raises(ValueError, match="Unknown optimizer"):
        resolve_optimizer(spec)
    named, _ = resolve_optimizer("optax-adam-cosine")
    assert named.schedule == "cosine"
    basin_spec = replace(OPTIMIZER_CATALOG["basinhopping"], method="basinhopping:Powell")
    basin, _ = resolve_optimizer(basin_spec)
    assert basin.local_method == "L-BFGS-B"
    assert _full(_loop("basinhopping:Powell")).local_method == "Powell"


@pytest.mark.parametrize("key", tuple(OPTIMIZER_CATALOG))
def test_unknown_catalog_options_remain_errors(key: str) -> None:
    with pytest.raises(ValueError, match="Unknown options"):
        resolve_optimizer(key, {"not_an_option": False})


@pytest.mark.parametrize("key", ["scipy-lbfgsb", "optax-adam", "jaxopt-lbfgs", "basinhopping", "multi-lbfgsb-5"])
def test_explicit_zero_iteration_caps_are_not_defaults(key: str) -> None:
    optimizer, settings = resolve_optimizer(key, {"maxiter": 0})
    if type(optimizer) is MultiStartOptimizer:
        assert optimizer.optimizer.maxiter == settings["optimizer"]["maxiter"] == 0
    else:
        attribute = (
            "max_steps"
            if type(optimizer) is OptaxOptimizer
            else "local_maxiter"
            if type(optimizer) is BasinHoppingOptimizer
            else "maxiter"
        )
        assert getattr(optimizer, attribute) == settings[attribute] == 0


def test_application_named_resolution_matches_catalog_and_preserves_objects() -> None:
    options = {"maxiter": 0, "ftol": 0.0, "fc_fraction": False, "eq_fraction": 0.05}
    expected, expected_settings = resolve_optimizer("scipy-lbfgsb-jax", options)
    actual, config, *_ = _resolve_optimizer(
        "scipy-lbfgsb-jax", options, executor="jax", requested_gradient_mode=None, requested_fd_step=None
    )
    assert _view(actual) == _view(expected)
    assert config.settings == expected_settings
    supplied = ScipyOptimizer(maxiter=2, use_bounds=False)
    returned, *_ = _resolve_optimizer(
        supplied, None, executor="python", requested_gradient_mode=None, requested_fd_step=None
    )
    assert returned is supplied
    assert supplied.maxiter == 2 and supplied.use_bounds is False


def test_sdk_cli_and_heck_presets_keep_their_existing_values() -> None:
    from q2mm.benchmarks.cli import _build_parser
    from q2mm.benchmarks.runner import resolve_optimizer as resolve_profile_optimizer

    gs_name, gs_options, gs_ratio = _recommended_defaults(StationaryPointKind.GROUND_STATE)
    ts_name, ts_options, ts_ratio = _recommended_defaults(StationaryPointKind.TRANSITION_STATE)
    cli = _build_parser().parse_args(["single", "--system", "rh-enamide"])
    assert gs_name == "recommended-jax-gs-v1" and gs_options == {"ftol": 1e-8} and gs_ratio == 10.0
    assert ts_name == "recommended-jax-ts-v1"
    assert ts_options == {"ftol": 1e-12, "fc_fraction": 0.20, "eq_fraction": 0.05} and ts_ratio is None
    assert (cli.ftol, cli.fc_fraction, cli.eq_fraction) == (1e-8, None, None)
    heck = recommended_publication_profile("heck-relay")
    ordinary_ts = recommended_publication_profile("rh-enamide")
    assert (heck.ftol, heck.fc_fraction, heck.eq_fraction) == (1e-12, 0.05, 0.05)
    assert ordinary_ts.fc_fraction == 0.20
    override = recommended_publication_profile("heck-relay", fc_fraction=0.12)
    assert override.fc_fraction == 0.12
    assert RunProfile(system="rh-enamide").workflow == "single-stage"
    _, sdk_settings = resolve_optimizer("scipy-lbfgsb-jax", ts_options)
    _, cli_settings = resolve_profile_optimizer(RunProfile(system="rh-enamide"))
    _, heck_settings = resolve_profile_optimizer(heck)
    assert (sdk_settings["ftol"], sdk_settings["fc_fraction"], sdk_settings["eq_fraction"]) == (1e-12, 0.20, 0.05)
    assert (cli_settings["ftol"], cli_settings["fc_fraction"], cli_settings["eq_fraction"]) == (1e-8, None, None)
    assert (heck_settings["ftol"], heck_settings["fc_fraction"], heck_settings["eq_fraction"]) == (1e-12, 0.05, 0.05)


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


@pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "optax:adam"])
def test_simplex_keeps_its_scipy_constructor_policy(method: str) -> None:
    loop = _loop("L-BFGS-B", simp_method=method, simp_maxiter=0, eps=0.02)
    actual = loop._build_simplex_optimizer()
    assert type(actual) is ScipyOptimizer
    assert _view(actual) == _view(ScipyOptimizer(method=method, maxiter=0, eps=0.02, verbose=False))


def test_bound_plan_drives_build_and_snapshot_without_aliasing() -> None:
    from q2mm.optimizers.catalog import _optimizer_construction

    options = {"n_starts": 2, "seed": 0, "verbose": False}
    inner_options = {"maxiter": 7, "use_bounds": False, "verbose": False}
    plan = _optimizer_construction("multi:Powell", options, inner_options=inner_options)
    snapshot = plan.settings()
    options["n_starts"] = 99
    inner_options["maxiter"] = 99
    snapshot["optimizer"]["maxiter"] = 99
    with pytest.raises(TypeError):
        plan.arguments["n_starts"] = 99
    optimizer = plan.build()
    assert type(optimizer) is MultiStartOptimizer
    assert optimizer.n_starts == 2
    assert optimizer.optimizer.maxiter == 7
    assert optimizer.optimizer.use_bounds is False
    assert plan.settings()["optimizer"]["maxiter"] == 7


@pytest.mark.parametrize("method", ["L-BFGS-B", "optax:adam", "jaxopt:lbfgs", "basinhopping", "multi:Powell"])
def test_constructor_options_are_rejected_by_the_shared_owner(method: str) -> None:
    from q2mm.optimizers.catalog import _optimizer_construction

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        _optimizer_construction(method, {"unknown_option": False})


def test_runtime_and_application_paths_use_the_same_build_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers.catalog import _Construction, _cycling_nested_settings

    original = _Construction.build
    calls = []

    def capture(plan: Any) -> Any:
        calls.append(plan.kind)
        return original(plan)

    monkeypatch.setattr(_Construction, "build", capture)
    loop = _loop("multi:Powell", full_maxiter=7, simp_maxiter=0)
    _cycling_nested_settings(
        {
            "full_method": loop.full_method,
            "simp_method": loop.simp_method,
            "full_maxiter": loop.full_maxiter,
            "simp_maxiter": loop.simp_maxiter,
            "eps": loop.eps,
        },
        "none",
    )
    assert calls == []
    loop._build_full_optimizer()
    loop._build_simplex_optimizer()
    assert calls == ["multistart", "scipy", "scipy"]
    _resolve_optimizer(
        "scipy-nm", {"maxiter": 0}, executor="python", requested_gradient_mode=None, requested_fd_step=None
    )
    assert calls == ["multistart", "scipy", "scipy", "scipy"]
    supplied = ScipyOptimizer(maxiter=3)
    _resolve_optimizer(supplied, None, executor="python", requested_gradient_mode=None, requested_fd_step=None)
    assert len(calls) == 4


def test_cycling_phase_execution_and_counts_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from q2mm.models.results import OptimizationResult
    from q2mm.optimizers.catalog import _Construction
    from q2mm.optimizers.cycling import SensitivityResult
    from test.test_multistart import QuadraticEvaluator

    calls = []
    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([3.0, 4.0]))
    space = evaluator.space.with_active_indices([0])

    class OneEvaluation:
        def __init__(self, method: str) -> None:
            self.method = method

        def optimize(self, objective: Any, active_space: ActiveParameterSpace) -> OptimizationResult:
            calls.append((self.method, active_space.active_indices.tolist()))
            baseline = active_space.baseline
            score = objective.value(baseline)
            return OptimizationResult(
                success=True,
                message=self.method,
                initial_score=score,
                final_score=score,
                n_iterations=2 if self.method == "L-BFGS-B" else 3,
                n_evaluations=1,
                n_params=active_space.n_full,
                layout_fingerprint=active_space.layout.fingerprint,
                initial_params=baseline,
                final_params=baseline,
                history=(score,),
                gradient_mode="analytical",
            )

    def build(plan: Any) -> OneEvaluation:
        assert plan.kind == "scipy"
        return OneEvaluation(plan.arguments["method"])

    monkeypatch.setattr(_Construction, "build", build)
    monkeypatch.setattr(
        "q2mm.optimizers.cycling.compute_sensitivity",
        lambda *args, **kwargs: SensitivityResult(
            d1=np.ones(2),
            d2=np.ones(2),
            simp_var=np.ones(2),
            ranking=np.array([0, 1]),
            metric="simp_var",
            n_evals=0,
        ),
    )
    result = OptimizationLoop(evaluator, space, max_cycles=1, max_params=1, verbose=False).run()
    assert calls == [("L-BFGS-B", [0]), ("Nelder-Mead", [0])]
    assert result.n_iterations == 1
    assert result.stages[0].n_iterations == 5
    assert result.n_evaluations == result.stages[0].n_evaluations == 2
    assert evaluator.n_evaluations == 3
    np.testing.assert_array_equal(result.final_params, space.baseline)
