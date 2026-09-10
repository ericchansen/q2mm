"""Shared sequential multi-start behavior and JaxOpt construction compatibility."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

import q2mm.optimizers.jax_multistart as jax_multistart
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult, StageRecord
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator
from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from test.test_multistart import QuadraticEvaluator


class ScriptedOptimizer:
    def __init__(self, outcomes: list[tuple[bool, float] | Exception], *, rich: bool = False) -> None:
        self.outcomes = outcomes
        self.rich = rich
        self.starts: list[np.ndarray] = []
        self.results: list[OptimizationResult] = []
        self.settings: list[tuple[str, int, float, bool]] = []

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        index = len(self.starts)
        baseline = space.baseline.copy()
        self.starts.append(baseline)
        initial_score = evaluator.value(baseline)
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        success, final_score = outcome
        evaluator.record_evaluation(final_score)
        result = OptimizationResult(
            success=success,
            message=f"candidate-{index}",
            initial_score=initial_score,
            final_score=final_score,
            n_iterations=index + 3,
            n_evaluations=2,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=baseline,
            final_params=baseline,
            history=(initial_score, final_score),
            method="jaxopt:lbfgs",
            gradient_mode=evaluator.gradient_mode.value,
            fd_step=evaluator.finite_difference_step,
        )
        if self.rich:
            stage = StageRecord(
                name=f"stage-{index}",
                n_params=space.n_full,
                layout_fingerprint=space.layout.fingerprint,
                initial_score=initial_score,
                final_score=final_score,
                n_iterations=result.n_iterations,
                n_evaluations=2,
                converged=success,
                message=result.message,
                gradient_mode=result.gradient_mode,
            )
            result = replace(
                result,
                stages=(stage,),
                initial_samples=(initial_score,),
                final_samples=(final_score,),
                category_metrics={"energy": {"rmsd": float(index + 1)}},
            )
        self.results.append(result)
        return result


def _wrapper(
    flavor: str,
    inner: ScriptedOptimizer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    perturbation_pct: float = 0.3,
) -> MultiStartOptimizer | JaxMultiStartOptimizer:
    if flavor == "generic":
        return MultiStartOptimizer(
            inner, n_starts=len(inner.outcomes), perturbation_pct=perturbation_pct, seed=17, verbose=False
        )

    def scripted_jax(
        optimizer: JaxOptOptimizer, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace
    ) -> OptimizationResult:
        inner.settings.append((optimizer.method, optimizer.maxiter, optimizer.tol, optimizer.verbose))
        return inner.optimize(evaluator, space)

    monkeypatch.setattr(JaxOptOptimizer, "optimize", scripted_jax)
    monkeypatch.setattr(jax_multistart, "_require_jax_executor", lambda evaluator: None)
    return JaxMultiStartOptimizer(
        n_starts=len(inner.outcomes), perturbation_pct=perturbation_pct, seed=17, verbose=False
    )


@pytest.mark.parametrize("flavor", ["generic", "jax"])
@pytest.mark.parametrize("perturbation_pct", [0.0, 0.3])
@pytest.mark.parametrize("active_indices", [[], [0, 2]])
def test_seeded_start_distribution_and_projection(
    flavor: str, perturbation_pct: float, active_indices: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = np.array([0.0, 8.0, -3.0])
    obj = QuadraticEvaluator(np.zeros(3), initial=baseline, bounds=[(-0.1, 0.1), (7.0, 9.0), (-2.0, -1.0)])
    space = obj.space.with_active_indices(active_indices)
    inner = ScriptedOptimizer([(True, 1.0)] * 4)
    wrapper = _wrapper(flavor, inner, monkeypatch, perturbation_pct=perturbation_pct)
    result = wrapper.optimize(obj, space)
    rng = np.random.default_rng(17)
    active = baseline[active_indices]
    expected = [baseline.copy()]
    for _ in range(3):
        scale = np.maximum(np.abs(active) * perturbation_pct, 1e-6)
        generated = np.clip(active + rng.uniform(-scale, scale), space.bounds[:, 0], space.bounds[:, 1])
        expected.append(space.expand(generated))
    for actual, wanted, candidate in zip(inner.starts, expected, result.candidates, strict=True):
        np.testing.assert_array_equal(actual, wanted)
        np.testing.assert_array_equal(candidate.initial_params, wanted)
        np.testing.assert_array_equal(candidate.final_params, wanted)
        assert candidate.seed == 17
        assert not candidate.initial_params.flags.writeable
    np.testing.assert_array_equal(space.baseline, baseline)
    np.testing.assert_array_equal(obj.plan.layout.vector(obj.forcefield), baseline)
    assert len(result.candidates) == 4
    assert result.n_evaluations == 9


@pytest.mark.parametrize("flavor", ["generic", "jax"])
@pytest.mark.parametrize(
    ("outcomes", "winner", "statuses"),
    [
        (
            [(False, 0.0), RuntimeError("boom"), (True, 5.0), (True, 5.0)],
            1,
            ["failure", "failure", "success", "success"],
        ),
        ([(False, 8.0), (False, 2.0), (False, 2.0)], 1, ["failure", "failure", "failure"]),
        ([RuntimeError("one"), ValueError("two")], None, ["failure", "failure"]),
    ],
    ids=["converged-preference-and-tie", "nonconverged-and-tie", "all-raised"],
)
def test_selection_failures_labels_and_count_semantics(
    flavor: str,
    outcomes: list[tuple[bool, float] | Exception],
    winner: int | None,
    statuses: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obj = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    obj.value(obj.space.baseline)
    count_before = obj.n_evaluations
    inner = ScriptedOptimizer(outcomes)
    result = _wrapper(flavor, inner, monkeypatch).optimize(obj, obj.space)
    assert [c.index for c in result.candidates] == list(range(len(outcomes)))
    assert [c.status for c in result.candidates] == statuses
    assert result.initial_score == 5.0
    np.testing.assert_array_equal(result.initial_params, obj.space.baseline)
    assert result.n_evaluations == obj.n_evaluations - count_before
    assert result.n_evaluations == 1 + sum(1 if isinstance(outcome, Exception) else 2 for outcome in outcomes)
    for candidate, outcome in zip(result.candidates, outcomes, strict=True):
        if isinstance(outcome, Exception):
            assert candidate.message == f"{type(outcome).__name__}: {outcome}"
            assert np.isnan(candidate.initial_score)
            assert candidate.final_score == float("inf")
            np.testing.assert_array_equal(candidate.final_params, candidate.initial_params)
    if winner is None:
        assert not result.success
        assert result.final_score == float("inf")
        assert result.n_iterations == 0
        assert result.history == (5.0,)
        assert result.gradient_mode == ("none" if flavor == "generic" else "analytical")
        np.testing.assert_array_equal(result.final_params, obj.space.baseline)
    else:
        selected = inner.results[winner]
        assert result.success == selected.success
        assert result.final_score == selected.final_score
        assert result.n_iterations == selected.n_iterations
        assert result.history == selected.history
        assert result.gradient_mode == selected.gradient_mode
        assert result.fd_step == selected.fd_step
        np.testing.assert_array_equal(result.final_params, selected.final_params)
    assert result.method == (
        "jaxopt-multi:lbfgs" if flavor == "jax" else "multi-start" if winner is None else "multi-start(jaxopt:lbfgs)"
    )
    assert result.message.startswith("jaxopt-multi" if flavor == "jax" else "multi-start")


@pytest.mark.parametrize("flavor", ["generic", "jax"])
def test_selected_rich_metadata_is_preserved(flavor: str, monkeypatch: pytest.MonkeyPatch) -> None:
    obj = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    inner = ScriptedOptimizer([(True, 2.0), (False, 1.0)], rich=True)
    result = _wrapper(flavor, inner, monkeypatch).optimize(obj, obj.space)
    selected = inner.results[0]
    assert result.stages == selected.stages
    assert result.initial_samples == selected.initial_samples
    assert result.final_samples == selected.final_samples
    assert result.category_metrics == selected.category_metrics


def test_jax_constructor_settings_remain_effective(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    inner = ScriptedOptimizer([(True, 2.0)] * 2)
    wrapper = _wrapper("jax", inner, monkeypatch)
    assert isinstance(wrapper, JaxMultiStartOptimizer)
    wrapper.method = "gradient_descent"
    wrapper.maxiter = 7
    wrapper.tol = 1e-4
    wrapper.optimize(obj, obj.space)
    assert inner.settings == [("gradient_descent", 7, 1e-4, False)] * 2
    assert MultiStartOptimizer(inner).n_starts == 5
    assert JaxMultiStartOptimizer().n_starts == 10


@pytest.mark.parametrize("mode", list(GradientMode))
def test_generic_preserves_explicit_gradient_provenance(mode: GradientMode) -> None:
    obj = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    obj._gradient_mode = mode
    inner = ScriptedOptimizer([(True, 2.0)])
    result = MultiStartOptimizer(inner, n_starts=1, verbose=False).optimize(obj, obj.space)
    assert result.gradient_mode == mode.value
    assert result.fd_step == obj.finite_difference_step


def test_jax_adapter_delegates_execution_and_start_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    obj = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    expected = ScriptedOptimizer([(True, 2.0)]).optimize(obj, obj.space)
    calls = []

    def shared_optimize(
        wrapper: MultiStartOptimizer, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace
    ) -> OptimizationResult:
        calls.append((wrapper.optimizer, evaluator, space))
        return expected

    monkeypatch.setattr(MultiStartOptimizer, "optimize", shared_optimize)
    monkeypatch.setattr(jax_multistart, "_require_jax_executor", lambda evaluator: None)
    wrapper = JaxMultiStartOptimizer(method="gradient_descent", n_starts=2, maxiter=7, tol=1e-4)
    assert wrapper.optimize(obj, obj.space) is expected
    assert len(calls) == 1
    inner, evaluator, space = calls[0]
    assert isinstance(inner, JaxOptOptimizer)
    assert (inner.method, inner.maxiter, inner.tol, inner.verbose) == ("gradient_descent", 7, 1e-4, False)
    assert evaluator is obj
    assert space is obj.space
    assert JaxMultiStartOptimizer._generate_starts is MultiStartOptimizer._generate_starts


def test_multistart_construction_does_not_import_optional_runtimes() -> None:
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
from q2mm.optimizers import MultiStartOptimizer, JaxMultiStartOptimizer, get_optimizer
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
assert get_optimizer("JaxMultiStartOptimizer") is JaxMultiStartOptimizer
MultiStartOptimizer(JaxOptOptimizer(), n_starts=2)
JaxMultiStartOptimizer(n_starts=2)
assert not blocked.intersection(sys.modules)
"""
    completed = subprocess.run([sys.executable, "-c", script], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
