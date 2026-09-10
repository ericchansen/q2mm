"""Numerical derivative failures must never look like stationary objectives."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from q2mm.backends.registry import load_backend
from q2mm.models.observations import ObservationSet
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.protocols import ObjectiveGradientError
from test._shared import make_diatomic
from test.test_jax_executor import _h2_ff, _make_objective

pytestmark = [
    pytest.mark.jax,
    pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="JAX not installed"),
]


def _executor(*, frequency: bool = False) -> JaxObjectiveExecutor:
    ff = _h2_ff()
    backend = load_backend("jax")
    refs = ObservationSet().with_frequency(4400.0, data_idx=5) if frequency else ObservationSet().with_energy(0.0)
    objective = _make_objective(ff, backend, [make_diatomic(distance=0.74)], refs)
    return JaxObjectiveExecutor(objective.plan, backend, ff)


@pytest.mark.parametrize("method", ["loss_and_grad", "value_and_gradient", "gradient"])
def test_actual_nan_frequency_derivative_raises(method: str) -> None:
    executor = _executor(frequency=True)
    params = executor.plan.active_space.baseline
    raw_loss, raw_grad = executor._compiled_vag_fns[0](params)
    assert np.isfinite(raw_loss)
    assert np.isnan(raw_grad).any()
    assert np.isfinite(executor.sample(params))

    with pytest.raises(ObjectiveGradientError, match="non-finite"):
        getattr(executor, method)(params)
    assert executor.n_evaluations == 0
    assert executor.history == ()


@pytest.mark.parametrize("traced", [False, True])
def test_native_nan_derivative_invalidates_value_and_full_gradient(traced: bool) -> None:
    import jax

    executor = _executor(frequency=True)
    params = executor.plan.active_space.baseline
    # A single tiny case exercises tracing, not an all-molecule production JIT.
    call = jax.jit(executor.value_and_grad_jax) if traced else executor.value_and_grad_jax
    loss, gradient = call(params)
    assert isinstance(loss, jax.Array)
    assert isinstance(gradient, jax.Array)
    assert np.isnan(loss)
    assert np.isnan(gradient).all()
    assert executor.n_evaluations == 0


def test_actual_finite_energy_with_overflowing_derivative_is_invalid() -> None:
    executor = _executor()
    params = np.array([1e163, 0.74001])
    raw_loss, raw_grad = executor._compiled_vag_fns[0](params)
    assert np.isfinite(raw_loss)
    assert np.isinf(raw_grad).any()
    assert np.isfinite(executor.sample(params))
    with pytest.raises(ObjectiveGradientError, match="non-finite"):
        executor.value_and_gradient(params)
    loss, gradient = executor.value_and_grad_jax(params)
    assert np.isnan(loss)
    assert np.isnan(gradient).all()


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_loss_with_zero_derivative_is_invalid(invalid: float) -> None:
    import jax
    import jax.numpy as jnp

    executor = _executor()

    def constant_loss(params: Any) -> Any:
        return jnp.float64(invalid)

    executor._compiled_vag_fns = [jax.jit(jax.value_and_grad(constant_loss))]
    params = executor.plan.active_space.baseline
    _, raw_gradient = executor._compiled_vag_fns[0](params)
    np.testing.assert_array_equal(raw_gradient, np.zeros_like(params))
    loss, gradient = executor.value_and_grad_jax(params)
    assert np.isnan(loss)
    assert np.isnan(gradient).all()
    with pytest.raises(ObjectiveGradientError, match="non-finite"):
        executor.value_and_gradient(params)


@pytest.mark.parametrize("second_term", ["case", "regularization"])
@pytest.mark.parametrize("overflow", ["value", "gradient"])
def test_aggregation_overflow_is_invalid(second_term: str, overflow: str) -> None:
    import jax
    import jax.numpy as jnp

    executor = _executor()
    params = executor.plan.active_space.baseline

    def large_finite_term(p: Any) -> Any:
        if overflow == "value":
            return jnp.float64(1e308)
        return jnp.float64(1e308) * (p[0] - params[0])

    compiled = jax.jit(jax.value_and_grad(large_finite_term))
    value, gradient = compiled(params)
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()
    executor._compiled_vag_fns = [compiled]
    if second_term == "case":
        executor._compiled_vag_fns.append(compiled)
    else:
        executor._compiled_reg_vag_fn = compiled

    value, gradient = executor.value_and_grad_jax(params)
    assert np.isnan(value)
    assert np.isnan(gradient).all()
    with pytest.raises(ObjectiveGradientError, match="non-finite"):
        executor.value_and_gradient(params)
    assert executor.history == ()


@pytest.mark.parametrize("optimizer_name", ["scipy", "optax"])
def test_host_optimizer_cannot_report_nonfinite_derivative_as_convergence(optimizer_name: str) -> None:
    if optimizer_name == "scipy":
        pytest.importorskip("scipy")
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        optimizer = ScipyOptimizer(maxiter=1, verbose=False)
    else:
        pytest.importorskip("optax")
        from q2mm.optimizers.optax import OptaxOptimizer

        optimizer = OptaxOptimizer(max_steps=1, verbose=False)
    executor = _executor(frequency=True)
    with pytest.raises(ObjectiveGradientError, match="non-finite"):
        optimizer.optimize(executor, executor.plan.active_space)
    assert executor.n_evaluations == 1
    assert executor.history == (executor.sample(executor.plan.active_space.baseline),)


def test_native_optimizer_cannot_hide_invalid_frozen_derivative() -> None:
    pytest.importorskip("jaxopt")
    import jax
    import jax.numpy as jnp

    from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

    executor = _executor()
    baseline = executor.plan.active_space.baseline.copy()

    def singular_loss(params: Any) -> Any:
        return 1.0 + jnp.sqrt(jnp.abs(params[1] - baseline[1]))

    executor._compiled_value_fns = [jax.jit(singular_loss)]
    executor._compiled_vag_fns = [jax.jit(jax.value_and_grad(singular_loss))]
    space = executor.plan.active_space.with_active_indices([0])
    raw_loss, raw_gradient = executor._compiled_vag_fns[0](baseline)
    assert float(raw_loss) == 1.0
    assert raw_gradient[0] == 0.0
    assert not np.isfinite(raw_gradient[1])
    with pytest.raises(ValueError, match="full_vector must be finite"):
        JaxOptOptimizer(maxiter=1, verbose=False).optimize(executor, space)
    np.testing.assert_array_equal(space.baseline, baseline)
    np.testing.assert_array_equal(executor.plan.layout.vector(executor.base_force_field), baseline)


def test_valid_per_case_gradient_and_regularization_unchanged() -> None:
    import jax

    ff = _h2_ff(bond_k=100.0)
    backend = load_backend("jax")
    distances = (0.9, 1.0)
    refs = ObservationSet().with_energy(1.0, case_id="0", weight=2.0).with_energy(2.0, case_id="1", weight=3.0)
    objective = _make_objective(
        ff,
        backend,
        [make_diatomic(distance=d, bond_tolerance=2.0) for d in distances],
        refs,
        regularization=0.1,
    )
    executor = JaxObjectiveExecutor(objective.plan, backend, ff)
    baseline = executor.plan.active_space.baseline.copy()
    params = baseline + np.array([0.5, 0.01])
    expected_loss = 0.1 * np.sum((params - baseline) ** 2)
    expected_gradient = 0.2 * (params - baseline)
    for distance, ref, weight in zip(distances, (1.0, 2.0), (2.0, 3.0), strict=True):
        displacement = distance - params[1]
        error = params[0] * displacement**2 - ref
        expected_loss += (weight * error) ** 2
        expected_gradient += 2 * weight**2 * error * np.array([displacement**2, -2 * params[0] * displacement])
    assert len(executor._compiled_vag_fns) == 2
    assert executor._compiled_reg_vag_fn is not None
    native_loss, native_gradient = executor.value_and_grad_jax(params)
    assert isinstance(native_loss, jax.Array)
    assert isinstance(native_gradient, jax.Array)
    assert float(native_loss) == pytest.approx(expected_loss)
    np.testing.assert_allclose(native_gradient, expected_gradient)
    value, gradient = executor.value_and_gradient(params)
    assert value == pytest.approx(expected_loss)
    np.testing.assert_allclose(gradient, expected_gradient)
    assert executor.n_evaluations == 1
    np.testing.assert_array_equal(executor.plan.active_space.baseline, baseline)


@pytest.mark.parametrize("optimizer_name", ["scipy", "optax", "jaxopt"])
def test_valid_zero_gradient_still_allows_stationary_success(optimizer_name: str) -> None:
    pytest.importorskip(optimizer_name)
    if optimizer_name == "scipy":
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        optimizer = ScipyOptimizer(maxiter=1, verbose=False)
    elif optimizer_name == "optax":
        from q2mm.optimizers.optax import OptaxOptimizer

        optimizer = OptaxOptimizer(max_steps=1, verbose=False)
    else:
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        optimizer = JaxOptOptimizer(maxiter=1, verbose=False)
    executor = _executor()
    space = executor.plan.active_space.with_active_indices([0])
    baseline = space.baseline.copy()
    for loss, gradient in (
        executor.value_and_gradient(baseline),
        executor.value_and_grad_jax(baseline),
    ):
        assert float(loss) == 0.0
        np.testing.assert_array_equal(gradient, np.zeros_like(baseline))
    result = optimizer.optimize(executor, space)
    assert result.success
    assert result.final_score == 0.0
    np.testing.assert_allclose(result.final_params, baseline, rtol=0.0, atol=1e-12)
    assert result.final_params[1] == baseline[1]
