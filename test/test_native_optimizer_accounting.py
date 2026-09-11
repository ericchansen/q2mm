"""Independent execution counters for bounded JaxOpt and multi-start runs."""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from q2mm.backends.registry import load_backend
from q2mm.models.observations import ObservationSet
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from test._shared import make_diatomic
from test.test_jax_executor import _h2_ff, _make_objective

pytestmark = [
    pytest.mark.jax,
    pytest.mark.skipif(
        importlib.util.find_spec("jax") is None or importlib.util.find_spec("jaxopt") is None,
        reason="JAX/JaxOpt not installed",
    ),
]


def _executor() -> tuple[JaxObjectiveExecutor, list[float], list[float]]:
    import jax

    ff = _h2_ff(bond_k=100.0, bond_r0=0.8)
    backend = load_backend("jax")
    objective = _make_objective(
        ff, backend, [make_diatomic(distance=0.9, bond_tolerance=1.5)], ObservationSet().with_energy(0.0)
    )
    evaluator = JaxObjectiveExecutor(objective.plan, backend, ff)
    native_values: list[float] = []
    host_values: list[float] = []

    def loss(params: Any) -> Any:
        displacement = params[1] - 1.0
        return displacement**2 + displacement**4

    def observed_vag(params: Any) -> Any:
        value, gradient = jax.value_and_grad(loss)(params)
        jax.debug.callback(lambda score: native_values.append(float(score)), value, ordered=True)
        return value, gradient

    evaluator._compiled_value_fns = [jax.jit(loss)]
    evaluator._compiled_vag_fns = [jax.jit(observed_vag)]
    total = evaluator._total

    def observed_total(params: np.ndarray) -> float:
        value = total(params)
        host_values.append(value)
        return value

    evaluator._total = observed_total
    return evaluator, native_values, host_values


@pytest.mark.parametrize("method", ["lbfgs", "lbfgsb", "gradient_descent"])
@pytest.mark.parametrize("maxiter", [0, 1, 2])
def test_real_native_callbacks_match_reported_evaluations(
    method: str, maxiter: int, record_property: Callable[[str, object], None]
) -> None:
    import jax

    evaluator, native_values, host_values = _executor()
    space = evaluator.plan.active_space.with_active_indices([1])
    baseline = space.baseline.copy()
    result = JaxOptOptimizer(method=method, maxiter=maxiter, tol=1e-12, verbose=False).optimize(evaluator, space)
    jax.effects_barrier()
    assert native_values
    assert len(host_values) == 2
    assert result.n_evaluations == evaluator.n_evaluations == len(native_values) + len(host_values)
    assert len(result.history) == result.n_evaluations
    assert result.history == evaluator.history
    assert result.history[0] == host_values[0]
    assert result.history[-1] == host_values[-1]
    assert evaluator.n_gradient_evaluations == 0
    assert result.final_params[0] == baseline[0]
    record_property("native_callbacks", len(native_values))
    record_property("host_evaluations", len(host_values))
    record_property("reported_evaluations", result.n_evaluations)
    if maxiter == 0:
        assert result.n_iterations == 0
        np.testing.assert_array_equal(result.final_params, baseline)


def test_tracing_is_not_an_evaluation_and_native_api_stays_uncounted(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax
    import jax.numpy as jnp

    evaluator, native_values, host_values = _executor()
    space = evaluator.plan.active_space.with_active_indices([1])
    optimizer = JaxOptOptimizer(maxiter=0, verbose=False)

    def build(_module: object, callback: Any) -> Any:
        before = evaluator.n_evaluations
        jax.make_jaxpr(callback)(jnp.asarray(space.pack(space.baseline)))
        jax.effects_barrier()
        assert evaluator.n_evaluations == before
        assert native_values == []

        class ZeroStep:
            def run(self, params: Any) -> Any:
                return params, SimpleNamespace(error=0.0, iter_num=0, num_fun_eval=9999, num_grad_eval=8888)

        return ZeroStep()

    monkeypatch.setattr(optimizer, "_build_solver", build)
    result = optimizer.optimize(evaluator, space)
    jax.effects_barrier()
    assert len(native_values) == 1  # The existing final native probe, not tracing.
    assert result.n_evaluations == len(native_values) + len(host_values) == 3
    before = evaluator.n_evaluations
    evaluator.value_and_grad_jax(space.baseline)
    jax.effects_barrier()
    assert evaluator.n_evaluations == before


@pytest.mark.parametrize("method", ["lbfgs", "lbfgsb", "gradient_descent"])
def test_no_active_parameters_count_only_the_existing_host_call(method: str) -> None:
    evaluator, native_values, host_values = _executor()
    space = evaluator.plan.active_space.with_active_indices([])
    result = JaxOptOptimizer(method=method, maxiter=2, verbose=False).optimize(evaluator, space)
    assert result.n_evaluations == 1
    assert native_values == []
    assert len(host_values) == 1


@pytest.mark.parametrize("n_starts", [1, 2])
def test_native_multistart_aggregate_and_selected_history_are_explicit(n_starts: int) -> None:
    import jax

    evaluator, native_values, host_values = _executor()
    space = evaluator.plan.active_space.with_active_indices([1])
    result = MultiStartOptimizer(
        JaxOptOptimizer(maxiter=1, tol=1e-12, verbose=False), n_starts=n_starts, seed=3, verbose=False
    ).optimize(evaluator, space)
    jax.effects_barrier()
    assert result.n_evaluations == evaluator.n_evaluations == len(native_values) + len(host_values)
    assert "n_evaluations=aggregate" in result.message
    assert "history=selected-run" in result.message
    selected = next(i for i in range(n_starts) if f"selected_candidate={i}" in result.message)
    assert len(result.history) < result.n_evaluations
    np.testing.assert_array_equal(result.final_params, result.candidates[selected].final_params)


def test_nonfinite_native_results_remain_invalid_but_completed_calls_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax
    import jax.numpy as jnp

    evaluator, native_values, host_values = _executor()

    def invalid(_params: Any) -> Any:
        value = jnp.float64(jnp.nan)
        jax.debug.callback(lambda score: native_values.append(float(score)), value, ordered=True)
        return value, jnp.full((2,), jnp.nan)

    # This is the invalid native outcome used by the independent GEO-03 fix.
    monkeypatch.setattr(evaluator, "value_and_grad_jax", jax.jit(invalid))
    optimizer = JaxOptOptimizer(maxiter=1, verbose=False)
    space = evaluator.plan.active_space.with_active_indices([1])
    with pytest.raises(ValueError, match="full_vector must be finite"):
        optimizer.optimize(evaluator, space)
    jax.effects_barrier()
    assert native_values and all(np.isnan(value) for value in native_values)
    assert evaluator.n_evaluations == len(native_values) + len(host_values)
    assert any(np.isnan(value) for value in evaluator.history)


@pytest.mark.parametrize("method", ["lbfgs", "lbfgsb", "gradient_descent"])
def test_bookkeeping_does_not_change_native_numerical_path(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    import jax

    counted, counted_native, counted_host = _executor()
    uncounted, uncounted_native, uncounted_host = _executor()
    monkeypatch.setattr(uncounted, "record_evaluation", lambda _score: None)
    counted_result = JaxOptOptimizer(method=method, maxiter=2, verbose=False).optimize(
        counted, counted.plan.active_space.with_active_indices([1])
    )
    uncounted_result = JaxOptOptimizer(method=method, maxiter=2, verbose=False).optimize(
        uncounted, uncounted.plan.active_space.with_active_indices([1])
    )
    jax.effects_barrier()
    np.testing.assert_array_equal(counted_result.final_params, uncounted_result.final_params)
    np.testing.assert_array_equal(counted_native, uncounted_native)
    np.testing.assert_array_equal(counted_host, uncounted_host)
    assert counted_result.final_score == uncounted_result.final_score
    assert counted_result.success == uncounted_result.success
    assert counted_result.n_iterations == uncounted_result.n_iterations
    assert uncounted_result.n_evaluations == 2
    assert counted_result.n_evaluations == len(counted_native) + len(counted_host)


def test_failed_native_starts_keep_completed_work_and_invalidity(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax
    import jax.numpy as jnp

    evaluator, native_values, host_values = _executor()

    def invalid(_params: Any) -> Any:
        value = jnp.float64(jnp.nan)
        jax.debug.callback(lambda score: native_values.append(float(score)), value, ordered=True)
        return value, jnp.full((2,), jnp.nan)

    monkeypatch.setattr(evaluator, "value_and_grad_jax", jax.jit(invalid))
    space = evaluator.plan.active_space.with_active_indices([1])
    result = MultiStartOptimizer(JaxOptOptimizer(maxiter=1, verbose=False), n_starts=2, seed=3, verbose=False).optimize(
        evaluator, space
    )
    jax.effects_barrier()
    assert not result.success
    assert all(candidate.status == "failure" for candidate in result.candidates)
    assert result.n_evaluations == len(native_values) + len(host_values)
    assert "history=initial-baseline" in result.message
    assert "n_evaluations=aggregate" in result.message
    np.testing.assert_array_equal(result.final_params, space.baseline)


def test_one_full_objective_call_is_not_counted_per_compiled_contribution() -> None:
    import jax

    evaluator, native_values, host_values = _executor()
    evaluator._compiled_value_fns *= 2
    evaluator._compiled_vag_fns *= 2
    result = JaxOptOptimizer(method="gradient_descent", maxiter=1, verbose=False).optimize(
        evaluator, evaluator.plan.active_space.with_active_indices([1])
    )
    jax.effects_barrier()
    assert native_values and len(native_values) % 2 == 0
    assert result.n_evaluations == len(native_values) // 2 + len(host_values)


def test_exception_before_native_scalar_return_does_not_fabricate_a_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluator, native_values, host_values = _executor()

    def unavailable(_params: Any) -> Any:
        raise RuntimeError("no scalar returned")

    monkeypatch.setattr(evaluator, "value_and_grad_jax", unavailable)
    with pytest.raises(RuntimeError, match="no scalar returned"):
        JaxOptOptimizer(maxiter=1, verbose=False).optimize(evaluator, evaluator.plan.active_space)
    assert evaluator.n_evaluations == len(host_values) == 1
    assert native_values == []
    assert evaluator.history == tuple(host_values)
