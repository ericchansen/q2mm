"""Pure preset policy, explicit construction, and fresh cycling ownership."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.catalog import (
    OPTIMIZER_CATALOG,
    OptimizerSpec,
    build_optimizer,
    expected_result_gradient,
    optimizer_settings,
    optimizer_spec,
)
from q2mm.optimizers.cycling import CyclingOptimizer, SensitivityResult
from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.optax import OptaxOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.settings import (
    BasinHoppingSettings,
    CyclingSettings,
    JaxMultiStartSettings,
    JaxOptSettings,
    MultiStartSettings,
    OptaxSettings,
    OptimizerOverrides,
    ScipySettings,
)


def test_shared_quadratic_support_needs_no_optional_runtime() -> None:
    code = """
import importlib.abc
import importlib.util
import sys
optional = {"scipy", "jax", "jaxopt", "optax", "openmm", "ase", "qcengine", "qcelemental"}
find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, package=None: (
    None if name.split(".")[0] in optional else find_spec(name, package)
)
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        assert fullname.split(".")[0] not in optional, fullname
sys.meta_path.insert(0, BlockOptional())
import numpy as np
from test._shared import QuadraticEvaluator
evaluator = QuadraticEvaluator(
    np.array([1.0, 4.0]), initial=np.array([0.0, 6.0]), active_indices=np.array([0])
)
assert evaluator.value(evaluator.space.baseline) == 5.0
assert evaluator.space.expand(np.array([1.0])).tolist() == [1.0, 6.0]
assert not optional.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


_LABELS = {
    "scipy-lbfgsb": "SciPy L-BFGS-B (SciPy FD)",
    "scipy-lbfgsb-jax": "SciPy L-BFGS-B (JAX grad)",
    "scipy-lbfgsb-fd": "SciPy L-BFGS-B (executor FD)",
    "scipy-nm": "Nelder-Mead",
    "scipy-powell": "Powell",
    "grad-simp": "Grad-Simp",
    "grad-simp-auto": "Grad-Simp (JAX grad)",
    "optax-adam": "Optax Adam",
    "optax-adam-cosine": "Optax Adam+cosine",
    "optax-adagrad": "Optax AdaGrad",
    "optax-sgd": "Optax SGD",
    "basinhopping": "Basin-hopping (T=1.0)",
    "basinhopping-cold": "Basin-hopping (T=0.5)",
    "multi-lbfgsb-5": "Multi-start n=5",
    "multi-lbfgsb-10": "Multi-start n=10",
    "scipy-lbfgsb-l2": "SciPy L-BFGS-B+L2 (SciPy FD)",
    "optax-adam-l2": "Optax Adam+L2",
    "jaxopt-lbfgs": "JaxOpt L-BFGS",
    "jaxopt-lbfgsb": "JaxOpt L-BFGS-B",
    "grad-simp-multi": "Grad-Simp (multi inner)",
}


def test_catalog_labels_are_preserved_exactly() -> None:
    assert {key: spec.label for key, spec in OPTIMIZER_CATALOG.items()} == _LABELS


@pytest.mark.parametrize("key", tuple(_LABELS))
def test_preset_specs_have_typed_settings_and_json_safe_policy(key: str) -> None:
    spec = optimizer_spec(key)
    assert spec is OPTIMIZER_CATALOG[key]
    assert isinstance(spec, OptimizerSpec)
    assert spec.key == key
    assert set(spec.model_dump()) == {
        "key",
        "label",
        "settings",
        "evaluator",
        "gradient_mode",
        "fd_step",
        "regularization",
    }
    json.dumps(spec.model_dump(mode="json"), allow_nan=False)
    settings = optimizer_settings(key)
    assert settings == spec.settings
    built = build_optimizer(settings)
    assert built.settings is settings


@pytest.mark.parametrize(
    ("settings", "constructor"),
    [
        (ScipySettings(method="BFGS", maxiter=7), ScipyOptimizer),
        (OptaxSettings(optimizer="sgd", max_steps=7, schedule="exponential"), OptaxOptimizer),
        (JaxOptSettings(method="gradient_descent", maxiter=7), JaxOptOptimizer),
        (BasinHoppingSettings(local_method="Powell", T=0.0, niter=0), BasinHoppingOptimizer),
        (MultiStartSettings(optimizer=ScipySettings(method="Powell", maxiter=7)), MultiStartOptimizer),
        (JaxMultiStartSettings(optimizer=JaxOptSettings(method="lbfgsb", maxiter=7)), JaxMultiStartOptimizer),
        (CyclingSettings(full_optimizer=OptaxSettings(schedule="cosine")), CyclingOptimizer),
    ],
)
def test_explicit_builder_selects_only_declared_typed_algorithm(settings: Any, constructor: type) -> None:
    optimizer = build_optimizer(settings)
    assert type(optimizer) is constructor
    assert optimizer.settings is settings
    if isinstance(optimizer, MultiStartOptimizer):
        assert optimizer.optimizer.settings is settings.optimizer


@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0])
def test_explicit_basin_temperature_preserves_zero(temperature: float) -> None:
    settings = BasinHoppingSettings(T=temperature, niter=0, local_maxiter=0, seed=0, verbose=False)
    optimizer = build_optimizer(settings)
    assert temperature == optimizer.settings.T
    assert optimizer.settings.niter == optimizer.settings.local_maxiter == optimizer.settings.seed == 0
    assert optimizer.configuration_settings() == settings.model_dump(mode="json")


@pytest.mark.parametrize(
    ("key", "field", "expected"),
    [
        ("scipy-lbfgsb", "maxiter", 500),
        ("optax-adam", "max_steps", 2000),
        ("jaxopt-lbfgs", "maxiter", 200),
        ("basinhopping", "niter", 25),
        ("basinhopping", "local_maxiter", 200),
        ("basinhopping", "T", 1.0),
        ("basinhopping-cold", "T", 0.5),
        ("multi-lbfgsb-5", "n_starts", 5),
        ("multi-lbfgsb-10", "n_starts", 10),
        ("optax-adam-cosine", "schedule", "cosine"),
        ("optax-adagrad", "optimizer", "adagrad"),
        ("optax-sgd", "optimizer", "sgd"),
    ],
)
def test_named_preset_values_are_independent_oracles(key: str, field: str, expected: Any) -> None:
    assert getattr(optimizer_settings(key), field) == expected


def test_nested_catalog_numerical_defaults_remain_distinct() -> None:
    multi = optimizer_settings("multi-lbfgsb-5")
    cycling = optimizer_settings("grad-simp")
    nested = optimizer_settings("grad-simp-multi")
    assert multi.optimizer.maxiter == 500 and multi.seed == 0
    assert cycling.full_optimizer.maxiter == cycling.simplex_optimizer.maxiter == 200
    assert nested.full_optimizer.optimizer.maxiter == 200
    assert nested.full_optimizer.n_starts == 5 and nested.full_optimizer.seed is None
    assert cycling.simplex_optimizer.method == "Nelder-Mead"


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("scipy-lbfgsb", "finite_difference"),
        ("scipy-lbfgsb-jax", "analytical"),
        ("scipy-lbfgsb-fd", "finite_difference"),
        ("scipy-nm", "none"),
        ("scipy-powell", "none"),
    ],
)
def test_named_gradient_contract_is_preserved(key: str, expected: str) -> None:
    assert expected_result_gradient(optimizer_spec(key)) == expected


def test_catalog_and_nested_settings_cannot_be_mutated_or_aliased() -> None:
    settings = optimizer_settings("grad-simp-multi")
    snapshot = settings.model_dump(mode="json")
    snapshot["full_optimizer"]["optimizer"]["maxiter"] = 999
    assert settings.full_optimizer.optimizer.maxiter == 200
    with pytest.raises(ValueError, match="frozen"):
        settings.full_optimizer.optimizer.maxiter = 999
    with pytest.raises(TypeError):
        OPTIMIZER_CATALOG["other"] = optimizer_spec("scipy-nm")
    with pytest.raises(ValueError, match="frozen"):
        optimizer_spec("scipy-nm").label = "changed"


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
def test_builder_constructs_actual_component_once(key: str, constructor: type, monkeypatch: pytest.MonkeyPatch) -> None:
    original = constructor.__init__
    calls = []

    def capture(self: Any, settings: Any) -> None:
        calls.append(settings)
        original(self, settings)

    monkeypatch.setattr(constructor, "__init__", capture)
    settings = optimizer_settings(key)
    optimizer = build_optimizer(settings)
    assert calls == [settings]
    assert calls[0] is optimizer.settings


def test_resolving_cycling_policy_does_not_construct_phase_solvers(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Pure policy resolution or cycling construction allocated a phase solver.")

    monkeypatch.setattr(ScipyOptimizer, "__init__", forbidden)
    monkeypatch.setattr(MultiStartOptimizer, "__init__", forbidden)
    settings = optimizer_settings("grad-simp-multi", OptimizerOverrides(maxiter=0))
    optimizer = build_optimizer(settings)
    assert optimizer.settings.full_optimizer.optimizer.maxiter == 0
    assert optimizer.settings.simplex_optimizer.maxiter == 0
    assert optimizer.configuration_settings() == settings.model_dump(mode="json")


def test_cycling_phase_execution_and_counts_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    from test._shared import QuadraticEvaluator

    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([3.0, 4.0]))
    space = evaluator.space.with_active_indices([0])
    calls = []

    def one_evaluation(
        self: ScipyOptimizer, objective: ObjectiveEvaluator, active_space: ActiveParameterSpace
    ) -> OptimizationResult:
        calls.append((self.settings.method, active_space.active_indices.tolist()))
        baseline = active_space.baseline
        score = objective.value(baseline)
        return OptimizationResult(
            success=True,
            message="bounded phase",
            initial_score=score,
            final_score=score,
            n_iterations=2 if self.settings.method == "L-BFGS-B" else 3,
            n_evaluations=1,
            n_params=active_space.n_full,
            layout_fingerprint=active_space.layout.fingerprint,
            initial_params=baseline,
            final_params=baseline,
            gradient_mode="analytical",
        )

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
    result = CyclingOptimizer(CyclingSettings(max_cycles=1, max_params=1, verbose=False)).optimize(evaluator, space)
    assert calls == [("L-BFGS-B", [0]), ("Nelder-Mead", [0])]
    assert result.n_iterations == 1 and result.stages[0].n_iterations == 5
    assert result.n_evaluations == result.stages[0].n_evaluations == 2
    assert evaluator.n_evaluations == 3
    np.testing.assert_array_equal(result.final_params, space.baseline)


def test_every_cycle_and_every_call_own_fresh_phase_solvers(monkeypatch: pytest.MonkeyPatch) -> None:
    from test._shared import QuadraticEvaluator

    built = []

    class Phase:
        def __init__(self, settings: ScipySettings) -> None:
            self.settings = settings
            built.append(self)

        def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
            initial = np.array(space.baseline)
            final = initial.copy()
            final[space.active_indices] -= 0.25
            initial_score = evaluator.value(initial)
            final_score = evaluator.value(final)
            return OptimizationResult(
                success=True,
                message="phase",
                initial_score=initial_score,
                final_score=final_score,
                n_iterations=1,
                n_evaluations=2,
                n_params=space.n_full,
                layout_fingerprint=space.layout.fingerprint,
                initial_params=initial,
                final_params=final,
            )

    monkeypatch.setattr("q2mm.optimizers.catalog.build_optimizer", Phase)
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
    optimizer = CyclingOptimizer(CyclingSettings(max_cycles=2, max_params=1, verbose=False))
    for _ in range(2):
        evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([3.0, 4.0]))
        result = optimizer.optimize(evaluator, evaluator.space.with_active_indices([0]))
        assert result.n_iterations == 2
        assert result.n_evaluations == 8 and evaluator.n_evaluations == 9
        assert result.initial_score == 13.0 and result.final_score == 10.0
        np.testing.assert_array_equal(result.final_params, [2.0, 4.0])
    assert len(built) == len({id(phase) for phase in built}) == 8
    assert [phase.settings.method for phase in built] == ["L-BFGS-B", "Nelder-Mead"] * 4


def test_catalog_policy_and_builder_remain_optional_runtime_lazy() -> None:
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
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG, optimizer_settings, optimizer_spec, build_optimizer
for key in OPTIMIZER_CATALOG:
    optimizer_spec(key)
    build_optimizer(optimizer_settings(key))
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
