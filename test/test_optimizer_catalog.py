"""Constructor-only checks for effective built-in optimizer provenance."""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any
from unittest.mock import MagicMock

import pytest

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG, OptimizerSpec, resolve_optimizer
from q2mm.optimizers.cycling import OptimizationLoop
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.optax import OptaxOptimizer
from q2mm.optimizers.protocols import _Optimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer


def _assert_constructor_settings(optimizer: Any, recorded: dict[str, Any]) -> None:
    for name in inspect.signature(type(optimizer)).parameters:
        if name == "optimizer" and isinstance(optimizer, MultiStartOptimizer):
            _assert_constructor_settings(optimizer.optimizer, recorded[name])
        else:
            attribute = "optimizer_name" if name == "optimizer" and isinstance(optimizer, OptaxOptimizer) else name
            assert name in recorded, f"{type(optimizer).__name__}.{name} is unrecorded"
            assert recorded[name] == getattr(optimizer, attribute), name


def _cycling_loop(optimizer: Any) -> OptimizationLoop:
    evaluator = MagicMock(spec=ObjectiveEvaluator)
    space = MagicMock(spec=ActiveParameterSpace)
    return OptimizationLoop(evaluator, space, **{"verbose": False, **optimizer._kwargs})


def _assert_cycling_settings(optimizer: Any, recorded: dict[str, Any]) -> None:
    loop = _cycling_loop(optimizer)
    for name in inspect.signature(OptimizationLoop).parameters:
        if name not in ("evaluator", "space"):
            assert name in recorded, f"OptimizationLoop.{name} is unrecorded"
            assert recorded[name] == getattr(loop, name), name
    full = loop._build_full_optimizer()
    _assert_constructor_settings(full, recorded["full_optimizer"])
    simplex = loop._build_simplex_optimizer()
    _assert_constructor_settings(simplex, recorded["simplex_optimizer"])


@pytest.mark.parametrize("key", tuple(OPTIMIZER_CATALOG))
@pytest.mark.parametrize("options", [{}, {"maxiter": 0}])
def test_all_catalog_constructor_defaults_are_recorded(key: str, options: dict[str, Any]) -> None:
    optimizer, recorded = resolve_optimizer(key, options)
    if recorded["kind"] == "cycling":
        _assert_cycling_settings(optimizer, recorded)
    else:
        _assert_constructor_settings(optimizer, recorded)
    json.dumps(recorded, allow_nan=False)


@pytest.mark.parametrize("temperature", [None, 0.0, 0.5])
def test_basinhopping_default_and_explicit_temperature(temperature: float | None) -> None:
    spec = OPTIMIZER_CATALOG["basinhopping"]
    if temperature is not None:
        spec = replace(spec, extra={"T": temperature, "niter": 0})
    optimizer, recorded = resolve_optimizer(spec, {"maxiter": 0, "seed": 0})
    expected_temperature = 1.0 if temperature is None else temperature
    assert expected_temperature == optimizer.T
    assert recorded["T"] == optimizer.T
    assert recorded["niter"] == optimizer.niter
    assert recorded["local_maxiter"] == optimizer.local_maxiter == 0
    assert recorded["seed"] == optimizer.seed == 0


@pytest.mark.parametrize("method", ["basinhopping", "multi:L-BFGS-B"])
def test_family_defaults_without_catalog_extra(method: str) -> None:
    spec = OptimizerSpec(key="constructor-defaults", label="Defaults", method=method, evaluator="python")
    optimizer, recorded = resolve_optimizer(spec)
    _assert_constructor_settings(optimizer, recorded)


@pytest.mark.parametrize(
    "method",
    [
        "L-BFGS-B",
        "BFGS",
        "optax:adam+cosine",
        "optax:sgd",
        "optax:sgd+exponential",
        "jaxopt:lbfgsb",
        "jaxopt:gradient_descent",
        "basinhopping:Powell",
        "basinhopping:  Powell  ",
        "basinhopping",
        "basinhopping:",
        "basinhopping-cold",
        "multi:Powell",
        "multi:L-BFGS-B",
    ],
)
@pytest.mark.parametrize("maxiter", [None, 0, 7])
def test_deferred_cycling_nested_settings_match_existing_builders(method: str, maxiter: int | None) -> None:
    spec = replace(OPTIMIZER_CATALOG["grad-simp"], extra={"full_method": method})
    options = {"max_params": 0, "max_cycles": 0, "convergence": 0.0}
    if maxiter is not None:
        options["maxiter"] = maxiter
    optimizer, recorded = resolve_optimizer(spec, options)
    _assert_cycling_settings(optimizer, recorded)
    assert recorded["max_cycles"] == recorded["max_params"] == 0
    assert recorded["convergence"] == 0.0
    assert recorded["full_maxiter"] == recorded["simp_maxiter"] == (200 if maxiter is None else maxiter)


def test_explicit_false_zero_and_policy_values_are_not_replaced() -> None:
    optimizer, recorded = resolve_optimizer(
        "scipy-lbfgsb-jax",
        {"maxiter": 0, "ftol": 0.0, "fc_fraction": False, "eq_fraction": 0.05},
    )
    assert optimizer.maxiter == recorded["maxiter"] == 0
    assert optimizer.ftol == recorded["ftol"] == 0.0
    assert optimizer.fc_fraction is recorded["fc_fraction"] is False
    assert optimizer.eq_fraction == recorded["eq_fraction"] == 0.05
    assert optimizer.verbose is recorded["verbose"] is False
    assert optimizer.use_bounds is recorded["use_bounds"] is True


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("scipy-lbfgsb", "none"),
        ("scipy-lbfgsb-jax", "bound-normalized"),
        ("scipy-lbfgsb-fd", "bound-normalized"),
        ("scipy-nm", "none"),
        ("scipy-powell", "none"),
    ],
)
def test_scaling_policy_is_not_claimed_for_inapplicable_methods(key: str, expected: str) -> None:
    _, recorded = resolve_optimizer(key)
    assert recorded["analytical_parameter_scaling"] == expected


def test_catalog_policy_choices_and_explicit_overrides_are_unchanged() -> None:
    scipy, _ = resolve_optimizer("scipy-lbfgsb")
    optax, _ = resolve_optimizer("optax-adam")
    jaxopt, _ = resolve_optimizer("jaxopt-lbfgs")
    basin, _ = resolve_optimizer("basinhopping")
    multi, _ = resolve_optimizer("multi-lbfgsb-5")
    cycling, _ = resolve_optimizer("grad-simp")
    assert isinstance(scipy, ScipyOptimizer) and scipy.maxiter == 500
    assert isinstance(optax, OptaxOptimizer) and optax.max_steps == 2000
    assert isinstance(jaxopt, JaxOptOptimizer) and jaxopt.maxiter == 200
    assert isinstance(basin, BasinHoppingOptimizer) and (basin.niter, basin.local_maxiter) == (25, 200)
    assert isinstance(multi, MultiStartOptimizer) and (multi.n_starts, multi.optimizer.maxiter) == (5, 500)
    assert _cycling_loop(cycling).full_maxiter == 200
    _, settings = resolve_optimizer("scipy-lbfgsb-jax", {"ftol": 1e-12, "fc_fraction": 0.20, "eq_fraction": 0.05})
    assert (settings["ftol"], settings["fc_fraction"], settings["eq_fraction"]) == (1e-12, 0.20, 0.05)


def test_recorded_settings_do_not_alias_constructor_configuration() -> None:
    optimizer, recorded = resolve_optimizer("grad-simp-multi")
    original = dict(optimizer._kwargs)
    recorded["full_maxiter"] = 999
    assert optimizer._kwargs == original
    multi, multi_record = resolve_optimizer("multi-lbfgsb-5")
    multi_record["optimizer"]["maxiter"] = 999
    assert multi.optimizer.maxiter == 500


@pytest.mark.parametrize(
    ("key", "constructor"),
    [
        ("scipy-lbfgsb", ScipyOptimizer),
        ("optax-adam", OptaxOptimizer),
        ("jaxopt-lbfgs", JaxOptOptimizer),
        ("basinhopping", BasinHoppingOptimizer),
        ("multi-lbfgsb-5", MultiStartOptimizer),
    ],
)
def test_snapshot_reuses_the_one_actual_constructor_call(
    key: str, constructor: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = constructor.__init__
    calls = []

    @wraps(original)
    def capture(self: Any, **kwargs: Any) -> None:
        calls.append(dict(kwargs))
        original(self, **kwargs)

    monkeypatch.setattr(constructor, "__init__", capture)
    optimizer, recorded = resolve_optimizer(key)
    assert len(calls) == 1
    assert set(calls[0]) == set(inspect.signature(constructor).parameters)
    for name, value in calls[0].items():
        if name == "optimizer" and isinstance(optimizer, MultiStartOptimizer):
            assert optimizer.optimizer is value
            _assert_constructor_settings(value, recorded[name])
        else:
            assert recorded[name] == value


def test_deferred_cycling_constructs_with_its_recorded_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    optimizer, recorded = resolve_optimizer("grad-simp-multi", {"maxiter": 0, "max_cycles": 0})
    evaluator = MagicMock(spec=ObjectiveEvaluator)
    space = MagicMock(spec=ActiveParameterSpace)
    expected = MagicMock(spec=OptimizationResult)
    calls = []

    def inspect_without_running(loop: OptimizationLoop) -> OptimizationResult:
        calls.append(loop)
        assert loop.evaluator is evaluator
        assert loop.space is space
        for name in inspect.signature(OptimizationLoop).parameters:
            if name not in ("evaluator", "space"):
                assert getattr(loop, name) == recorded[name]
        return expected

    monkeypatch.setattr(OptimizationLoop, "run", inspect_without_running)
    assert optimizer.optimize(evaluator, space) is expected
    assert len(calls) == 1
    assert evaluator.mock_calls == space.mock_calls == []


def test_constructor_capture_retains_its_declared_argument_shape() -> None:
    from q2mm.optimizers.catalog import _constructor_settings

    recorded = _constructor_settings(ScipyOptimizer, "scipy", maxiter=0, eps=0.02, verbose=False)
    _assert_constructor_settings(ScipyOptimizer(maxiter=0, eps=0.02, verbose=False), recorded)
    assert set(recorded) == {"kind", *inspect.signature(ScipyOptimizer).parameters}


def test_cycling_snapshot_and_execution_use_the_same_construction_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers import catalog

    inner = catalog._leaf(ScipyOptimizer, "scipy", method="Powell", maxiter=7, eps=0.02, verbose=False)
    plans = {
        "full_optimizer": catalog._leaf(
            MultiStartOptimizer, "multistart", optimizer=inner, n_starts=2, seed=0, verbose=False
        ),
        "simplex_optimizer": catalog._leaf(ScipyOptimizer, "scipy", method="Nelder-Mead", maxiter=0, verbose=False),
    }
    decisions: list[str] = []
    builds: list[str] = []
    original_build = catalog._Construction.build

    def shared(settings: Mapping[str, object]) -> dict[str, catalog._Construction]:
        assert settings["full_method"] == "shared-test-method"
        decisions.append("shared")
        return plans

    def build(plan: catalog._Construction) -> _Optimizer:
        builds.append(plan.kind)
        return original_build(plan)

    monkeypatch.setattr(catalog, "_cycling_constructions", shared)
    monkeypatch.setattr(catalog._Construction, "build", build)
    spec = replace(OPTIMIZER_CATALOG["grad-simp"], extra={"full_method": "shared-test-method"})
    optimizer, recorded = resolve_optimizer(spec)
    assert decisions == ["shared"]
    assert builds == ["cycling"]
    assert recorded["full_optimizer"] == plans["full_optimizer"].settings()
    assert recorded["simplex_optimizer"] == plans["simplex_optimizer"].settings()
    recorded["full_optimizer"]["optimizer"]["maxiter"] = 999
    with pytest.raises(TypeError):
        inner.arguments["maxiter"] = 999

    loop = _cycling_loop(optimizer)
    full = loop._build_full_optimizer()
    simplex = loop._build_simplex_optimizer()
    assert decisions == ["shared"] * 3
    assert builds == ["cycling", "multistart", "scipy", "scipy"]
    assert isinstance(full, MultiStartOptimizer)
    assert full.n_starts == 2 and full.seed == 0
    assert isinstance(full.optimizer, ScipyOptimizer)
    assert (full.optimizer.method, full.optimizer.maxiter, full.optimizer.eps) == ("Powell", 7, 0.02)
    assert isinstance(simplex, ScipyOptimizer)
    assert simplex.method == "Nelder-Mead" and simplex.maxiter == 0
    assert plans["full_optimizer"].settings()["optimizer"]["maxiter"] == 7


@pytest.mark.parametrize(
    ("method", "constructor"),
    [
        ("L-BFGS-B", ScipyOptimizer),
        ("optax:adam+cosine", OptaxOptimizer),
        ("jaxopt:lbfgsb", JaxOptOptimizer),
        ("basinhopping:Powell", BasinHoppingOptimizer),
        ("multi:Powell", MultiStartOptimizer),
    ],
)
def test_deferred_cycling_does_not_instantiate_phase_solvers(
    method: str, constructor: type, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = constructor.__init__

    @wraps(original)
    def unexpected(self: object, *args: object, **kwargs: object) -> None:
        raise AssertionError("Resolving deferred settings must not construct a phase solver.")

    monkeypatch.setattr(constructor, "__init__", unexpected)
    spec = replace(OPTIMIZER_CATALOG["grad-simp"], extra={"full_method": method})
    _, recorded = resolve_optimizer(spec, {"maxiter": 0})
    assert recorded["full_method"] == method
    json.dumps(recorded, allow_nan=False)


@pytest.mark.parametrize("simp_method", ["Nelder-Mead", "Powell", "optax:adam"])
def test_cycling_phase_policies_keep_eps_and_scipy_simplex(simp_method: str) -> None:
    loop = OptimizationLoop(
        MagicMock(spec=ObjectiveEvaluator),
        MagicMock(spec=ActiveParameterSpace),
        full_method="multi:Powell",
        simp_method=simp_method,
        full_maxiter=0,
        simp_maxiter=7,
        eps=0.02,
    )
    full = loop._build_full_optimizer()
    simplex = loop._build_simplex_optimizer()
    assert isinstance(full, MultiStartOptimizer) and isinstance(full.optimizer, ScipyOptimizer)
    assert (full.n_starts, full.seed, full.verbose) == (5, None, False)
    assert (full.optimizer.method, full.optimizer.maxiter, full.optimizer.eps) == ("Powell", 0, 0.02)
    assert isinstance(simplex, ScipyOptimizer)
    assert (simplex.method, simplex.maxiter, simplex.eps) == (simp_method, 7, 0.02)
    assert full.optimizer.verbose is simplex.verbose is False
    assert full.optimizer.ftol == simplex.ftol == 1e-8
    assert full.optimizer.fc_fraction is simplex.fc_fraction is None
    assert full.optimizer.eq_fraction is simplex.eq_fraction is None


def test_cycling_run_builds_both_shared_phase_plans(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from q2mm.optimizers.catalog import _Construction
    from q2mm.optimizers.cycling import SensitivityResult
    from test.test_multistart import QuadraticEvaluator

    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([3.0, 4.0]))
    space = evaluator.space.with_active_indices([0])
    builds: list[object] = []
    original_build = _Construction.build

    def build(plan: _Construction) -> _Optimizer:
        builds.append(plan.arguments["method"])
        return original_build(plan)

    def one_evaluation(
        self: ScipyOptimizer, objective: ObjectiveEvaluator, active_space: ActiveParameterSpace
    ) -> OptimizationResult:
        baseline = active_space.baseline
        score = objective.value(baseline)
        return OptimizationResult(
            success=True,
            message="bounded phase",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            n_params=active_space.n_full,
            layout_fingerprint=active_space.layout.fingerprint,
            initial_params=baseline,
            final_params=baseline,
        )

    monkeypatch.setattr(_Construction, "build", build)
    monkeypatch.setattr(ScipyOptimizer, "optimize", one_evaluation)
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
    assert builds == ["L-BFGS-B", "Nelder-Mead"]
    assert result.n_iterations == 1 and result.stages[0].n_iterations == 2
    assert result.n_evaluations == result.stages[0].n_evaluations == 2
    assert evaluator.n_evaluations == 3
    np.testing.assert_array_equal(result.final_params, space.baseline)


def test_catalog_constructor_inspection_remains_optional_runtime_lazy() -> None:
    script = """
import builtins
import sys
blocked = {"jax", "jaxlib", "jaxopt", "optax", "scipy"}
original_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0 and name.split(".")[0] in blocked:
        raise AssertionError(f"Unexpected runtime import: {name}")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG, resolve_optimizer
from dataclasses import replace
for key in OPTIMIZER_CATALOG:
    resolve_optimizer(key)
for method in ("optax:adam+cosine", "jaxopt:lbfgsb", "basinhopping:Powell", "multi:Powell"):
    resolve_optimizer(replace(OPTIMIZER_CATALOG["grad-simp"], extra={"full_method": method}))
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
