"""Requested FD coordinates, current optimizer spaces, and probe accounting."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import q2mm.objectives as objectives
from q2mm.backends.contracts import EnergyRequest, PreparationRequest
from q2mm.models.observations import ObservationSet
from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.basinhopping import BasinHoppingOptimizer
from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from test.test_application import _EnergyBackend
from test.test_multistart import _synthetic_plan


def _executor(*, step: float = 0.01) -> tuple[PythonObjectiveExecutor, list[np.ndarray]]:
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    ff, _, plan = _synthetic_plan(baseline, [(0.1, 10.0)] * 4)
    plan = replace(
        plan,
        observations=ObservationSet().with_energy(0.0, weight=2.0).with_energy(5.0, weight=0.5),
        regularization=0.25,
        reference_params=np.zeros(4),
    )
    probes: list[np.ndarray] = []

    class CountedBackend(_EnergyBackend):
        def prepare(self, request: PreparationRequest) -> Any:
            prepared = super().prepare(request)
            energy = prepared._energy

            def observed(query: EnergyRequest) -> Any:
                probes.append(np.array(query.parameters, copy=True))
                return energy(query)

            prepared._energy = observed
            return prepared

    return PythonObjectiveExecutor(
        plan, CountedBackend(), ff, gradient_mode=GradientMode.FINITE_DIFFERENCE, fd_step=step
    ), probes


class _LegacyEvaluator:
    def __init__(self, evaluator: PythonObjectiveExecutor) -> None:
        self.evaluator = evaluator

    def __getattr__(self, name: str) -> Any:
        if name == "value_and_gradient_selected":
            raise AttributeError(name)
        return getattr(self.evaluator, name)


@pytest.mark.parametrize("indices", [[3, 1], [0], [], [0, 1, 2, 3]])
def test_selected_gradient_matches_full_without_unrequested_probes(indices: list[int]) -> None:
    full, full_probes = _executor()
    selected, probes = _executor()
    params = full.plan.active_space.baseline.copy()
    full_value, full_gradient = full.value_and_gradient(params)
    value, gradient = selected.value_and_gradient_selected(params, indices)
    assert value == full_value
    assert gradient.shape == (len(indices),)
    np.testing.assert_allclose(gradient, full_gradient[indices], rtol=0.0, atol=1e-10)
    assert len(full_probes) == 1 + 2 * len(params)
    assert len(probes) == 1 + 2 * len(indices)
    assert selected.n_gradient_evaluations == 2 * len(indices)
    assert selected.n_evaluations == 1
    assert selected.history == (value,)
    np.testing.assert_array_equal(probes[0], params)
    for column, index in enumerate(indices):
        expected_plus, expected_minus = params.copy(), params.copy()
        expected_plus[index] += selected.finite_difference_step
        expected_minus[index] -= selected.finite_difference_step
        np.testing.assert_array_equal(probes[1 + 2 * column], expected_plus)
        np.testing.assert_array_equal(probes[2 + 2 * column], expected_minus)
    np.testing.assert_array_equal(params, full.plan.active_space.baseline)
    assert np.all(full_gradient != 0.0)


@pytest.mark.parametrize(
    "indices",
    [None, 1, "1", {1, 2}, [True], [np.bool_(False)], [1.0], ["1"], [-1], [4], [1, 1], [[1]], np.array(1)],
)
def test_invalid_selectors_reject_before_evaluation(indices: Any) -> None:
    evaluator, probes = _executor()
    with pytest.raises((TypeError, ValueError), match="indices"):
        evaluator.value_and_gradient_selected(evaluator.plan.active_space.baseline, indices)
    assert probes == []
    assert evaluator.n_evaluations == evaluator.n_gradient_evaluations == 0


@pytest.mark.parametrize("dtype", [float, bool, complex, object, str, np.dtype([("index", "i4")])])
def test_empty_noninteger_arrays_reject_before_evaluation(dtype: Any) -> None:
    evaluator, probes = _executor()
    indices = np.array([], dtype=dtype)
    with pytest.raises(TypeError, match="indices.*integer"):
        evaluator.value_and_gradient_selected(evaluator.plan.active_space.baseline, indices)
    assert probes == []
    assert evaluator.n_evaluations == evaluator.n_gradient_evaluations == 0
    assert evaluator.history == ()


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint64])
def test_empty_integer_arrays_preserve_empty_selection_contract(dtype: Any) -> None:
    evaluator, probes = _executor()
    indices = np.array([], dtype=dtype)
    indices.setflags(write=False)
    value, gradient = evaluator.value_and_gradient_selected(evaluator.plan.active_space.baseline, indices)
    assert gradient.shape == (0,)
    assert len(probes) == 1
    assert evaluator.n_evaluations == 1
    assert evaluator.n_gradient_evaluations == 0
    assert evaluator.history == (value,)
    assert indices.dtype == dtype and not indices.flags.writeable


def test_numpy_integer_selector_order_and_readonly_input() -> None:
    evaluator, _ = _executor()
    params = evaluator.plan.active_space.baseline
    indices = np.array([3, 1], dtype=np.int32)
    indices.setflags(write=False)
    _, gradient = evaluator.value_and_gradient_selected(params, indices)
    np.testing.assert_allclose(gradient, np.array([84.5, 83.5]), atol=1e-9)
    np.testing.assert_array_equal(indices, [3, 1])


@pytest.mark.parametrize("operation", ["value_and_gradient", "gradient"])
def test_default_full_gradient_stays_full_when_plan_has_frozen_coordinates(operation: str) -> None:
    original, probes = _executor()
    plan = replace(original.plan, active_space=original.plan.active_space.with_active_indices([1, 3]))
    evaluator = PythonObjectiveExecutor(
        plan,
        original.backend,
        original.base_force_field,
        gradient_mode=GradientMode.FINITE_DIFFERENCE,
        fd_step=0.01,
    )
    returned = getattr(evaluator, operation)(plan.active_space.baseline)
    gradient = returned[1] if operation == "value_and_gradient" else returned
    assert gradient.shape == (plan.n_params,)
    assert np.all(gradient != 0.0)
    assert evaluator.n_gradient_evaluations == 2 * plan.n_params
    assert len(probes) == 1 + 2 * plan.n_params


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("fail_at", [2, 3, 4, 5])
def test_completed_fd_probe_is_counted_when_later_probe_fails(
    selected: bool, fail_at: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, probes = _executor()
    total = evaluator._total
    calls = 0

    def fail_probe(params: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError("probe failed")
        return total(params)

    monkeypatch.setattr(evaluator, "_total", fail_probe)
    with pytest.raises(RuntimeError, match="probe failed"):
        if selected:
            evaluator.value_and_gradient_selected(evaluator.plan.active_space.baseline, [1, 3])
        else:
            evaluator.value_and_gradient(evaluator.plan.active_space.baseline)
    assert evaluator.n_gradient_evaluations == fail_at - 2
    assert len(probes) == fail_at - 1
    assert evaluator.n_evaluations == 0
    assert evaluator.history == ()


@pytest.mark.parametrize("mode", [GradientMode.NONE, GradientMode.ANALYTICAL])
def test_selected_api_requires_explicit_executor_fd(mode: GradientMode) -> None:
    evaluator, probes = _executor()
    evaluator._gradient_mode = mode
    with pytest.raises(ObjectiveGradientError, match="finite_difference"):
        evaluator.value_and_gradient_selected(evaluator.plan.active_space.baseline, [1])
    assert probes == []


def _stub_minimize(probes: list[np.ndarray], groups: list[tuple[np.ndarray, list[np.ndarray], np.ndarray]]) -> Any:
    from scipy.optimize import OptimizeResult

    def minimize(fun: Any, x0: np.ndarray, **kwargs: Any) -> Any:
        assert kwargs["jac"] is True
        before = len(probes)
        value, gradient = fun(x0)
        current = probes[before:]
        groups.append((current[0], current[1:], np.asarray(gradient)))
        return OptimizeResult(x=x0, fun=value, nit=0, success=True, message="one callback")

    return minimize


@pytest.mark.parametrize("use_bounds", [False, True])
def test_scipy_gradient_callback_requests_current_coordinates(
    use_bounds: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3]).with_baseline(np.array([2.0, 5.0, 7.0, 8.0]))
    groups = []
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, groups))
    result = ScipyOptimizer(maxiter=1, use_bounds=use_bounds, verbose=False).optimize(evaluator, space)
    assert len(groups) == 1
    baseline, perturbations, gradient = groups[0]
    assert len(perturbations) == 2 * space.n_active
    assert evaluator.n_gradient_evaluations == 2 * space.n_active
    np.testing.assert_array_equal(baseline[[0, 2]], space.baseline[[0, 2]])
    assert all(np.array_equal(point[[0, 2]], space.baseline[[0, 2]]) for point in perturbations)
    reference, _ = _executor()
    _, full_gradient = reference.value_and_gradient(baseline)
    scales = (space.bounds[:, 1] - space.bounds[:, 0]) / 2 if use_bounds else np.ones(space.n_active)
    np.testing.assert_allclose(gradient, full_gradient[space.active_indices] * scales, rtol=1e-10)
    assert result.fd_step == evaluator.finite_difference_step
    assert result.n_evaluations == 2


def test_rebased_multistart_uses_each_actual_space(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3]).with_baseline(np.array([2.0, 5.0, 7.0, 8.0]))
    groups = []
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, groups))
    result = MultiStartOptimizer(
        ScipyOptimizer(maxiter=1, use_bounds=False, verbose=False), n_starts=2, seed=3, verbose=False
    ).optimize(evaluator, space)
    assert len(groups) == 2
    for (baseline, perturbations, _), candidate in zip(groups, result.candidates, strict=True):
        np.testing.assert_array_equal(baseline, candidate.initial_params)
        assert len(perturbations) == 2 * space.n_active
        assert all(np.array_equal(point[[0, 2]], space.baseline[[0, 2]]) for point in perturbations)
    assert evaluator.n_gradient_evaluations == 2 * space.n_active * 2
    np.testing.assert_array_equal(space.baseline, [2.0, 5.0, 7.0, 8.0])


def test_cycling_gradient_subspace_is_not_the_stale_plan_space(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers.cycling import OptimizationLoop, SensitivityResult

    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([0, 1, 3])
    groups = []
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, groups))
    monkeypatch.setattr(
        "q2mm.optimizers.cycling.compute_sensitivity",
        lambda *args, **kwargs: SensitivityResult(
            d1=np.ones(4),
            d2=np.ones(4),
            simp_var=np.ones(4),
            ranking=np.array([3, 1, 0, 2]),
            metric="simp_var",
            n_evals=0,
        ),
    )
    OptimizationLoop(
        evaluator, space, max_params=1, max_cycles=1, full_method="L-BFGS-B", simp_method="L-BFGS-B", verbose=False
    ).run()
    assert [len(group[1]) for group in groups] == [6, 2]
    assert evaluator.n_gradient_evaluations == 8
    assert all(point[2] == space.baseline[2] for point in probes)


def test_basinhopping_jacobian_uses_requested_gradient_coordinates(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3])

    def basin(fun: Any, x0: np.ndarray, **kwargs: Any) -> Any:
        value = fun(x0)
        before = len(probes)
        gradient = kwargs["minimizer_kwargs"]["jac"](x0)
        assert gradient.shape == (space.n_active,)
        assert len(probes) - before == 1 + 2 * space.n_active
        return SimpleNamespace(
            x=x0, fun=value, message=["entry"], lowest_optimization_result=SimpleNamespace(success=True)
        )

    monkeypatch.setattr("scipy.optimize.basinhopping", basin)
    BasinHoppingOptimizer(niter=0, verbose=False).optimize(evaluator, space)
    assert evaluator.n_gradient_evaluations == 2 * space.n_active


@pytest.mark.jax
def test_optax_one_step_uses_selected_executor_fd() -> None:
    pytest.importorskip("optax")
    from q2mm.optimizers.optax import OptaxOptimizer

    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3])
    result = OptaxOptimizer(max_steps=1, verbose=False).optimize(evaluator, space)
    assert evaluator.n_gradient_evaluations == 2 * space.n_active
    assert len(probes) == 4 + 2 * space.n_active
    assert result.n_evaluations == 4
    assert result.n_iterations == 1
    np.testing.assert_array_equal(probes[-2], result.final_params)
    np.testing.assert_array_equal(probes[-1], result.final_params)
    assert all(np.array_equal(point[[0, 2]], space.baseline[[0, 2]]) for point in probes)


def test_legacy_executor_without_optional_interface_keeps_full_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3])

    groups = []
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, groups))
    ScipyOptimizer(maxiter=1, use_bounds=False, verbose=False).optimize(_LegacyEvaluator(evaluator), space)
    assert evaluator.n_gradient_evaluations == 2 * space.n_full
    assert groups[0][2].shape == (space.n_active,)


@pytest.mark.parametrize("consumer", ["scipy", "basinhopping", pytest.param("optax", marks=pytest.mark.jax)])
def test_bounded_consumers_match_full_fallback_with_less_fd_work(
    consumer: str, record_property: Callable[[str, object], None]
) -> None:
    if consumer == "optax":
        pytest.importorskip("optax")
        from q2mm.optimizers.optax import OptaxOptimizer

        optimizer = OptaxOptimizer(max_steps=1, verbose=False)
    elif consumer == "basinhopping":
        optimizer = BasinHoppingOptimizer(niter=0, local_maxiter=1, seed=3, verbose=False)
    else:
        optimizer = ScipyOptimizer(maxiter=1, verbose=False)
    selected, selected_probes = _executor()
    full, full_probes = _executor()
    baseline = np.array([2.0, 5.0, 7.0, 8.0])
    space = selected.plan.active_space.with_active_indices([3, 1]).with_baseline(baseline)
    result = optimizer.optimize(selected, space)
    reference = optimizer.optimize(_LegacyEvaluator(full), space)

    np.testing.assert_array_equal(result.final_params, reference.final_params)
    assert result.final_score == reference.final_score
    assert result.initial_score == reference.initial_score
    assert result.success == reference.success
    assert result.n_iterations == reference.n_iterations
    assert result.history == reference.history
    assert result.n_evaluations == reference.n_evaluations
    assert result.gradient_mode == reference.gradient_mode == "finite_difference"
    assert result.fd_step == reference.fd_step == selected.finite_difference_step
    assert selected.n_gradient_evaluations > 0
    requests, remainder = divmod(selected.n_gradient_evaluations, 2 * space.n_active)
    assert remainder == 0
    assert full.n_gradient_evaluations == requests * 2 * space.n_full
    assert len(selected_probes) == result.n_evaluations + selected.n_gradient_evaluations
    assert len(full_probes) == reference.n_evaluations + full.n_gradient_evaluations
    assert all(np.array_equal(point[[0, 2]], baseline[[0, 2]]) for point in selected_probes)
    np.testing.assert_array_equal(space.active_indices, [1, 3])
    np.testing.assert_array_equal(space.baseline, baseline)
    np.testing.assert_array_equal(selected.plan.active_space.baseline, [1.0, 2.0, 3.0, 4.0])
    record_property("gradient_requests", requests)
    record_property("selected_fd_probes", selected.n_gradient_evaluations)
    record_property("full_fd_probes", full.n_gradient_evaluations)
    record_property("selected_total_calls", len(selected_probes))
    record_property("full_total_calls", len(full_probes))
    record_property("recorded_evaluations", result.n_evaluations)


@pytest.mark.parametrize("params", [np.ones(3), np.ones((4, 1)), np.array([1.0, np.nan, 3.0, 4.0])])
def test_selected_invalid_full_vector_rejects_before_evaluation(params: np.ndarray) -> None:
    evaluator, probes = _executor()
    before = params.copy()
    with pytest.raises(ValueError, match="full_vector"):
        evaluator.value_and_gradient_selected(params, [1])
    assert probes == []
    assert evaluator.n_evaluations == evaluator.n_gradient_evaluations == 0
    np.testing.assert_array_equal(params, before)


def test_optional_provider_failure_is_not_silently_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers.protocols import _active_value_and_gradient

    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1, 3])

    def selected_failure(params: np.ndarray, indices: np.ndarray) -> Any:
        raise RuntimeError("selected derivative failed")

    def forbidden_full(params: np.ndarray) -> Any:
        pytest.fail("A failed selected request must not retry a full gradient")

    monkeypatch.setattr(evaluator, "value_and_gradient_selected", selected_failure)
    monkeypatch.setattr(evaluator, "value_and_gradient", forbidden_full)
    with pytest.raises(RuntimeError, match="selected derivative failed"):
        _active_value_and_gradient(evaluator, space, space.baseline)
    assert probes == []


def test_analytical_full_gradient_validation_is_not_bypassed(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, probes = _executor()
    evaluator._gradient_mode = GradientMode.ANALYTICAL
    space = evaluator.plan.active_space.with_active_indices([1])

    def full_guard(_params: np.ndarray) -> Any:
        raise ObjectiveGradientError("nonfinite full gradient, including frozen coordinates")

    def forbidden_selection(*args: Any) -> Any:
        pytest.fail("Analytical full-gradient validity was bypassed")

    monkeypatch.setattr(evaluator, "value_and_gradient", full_guard)
    monkeypatch.setattr(evaluator, "value_and_gradient_selected", forbidden_selection, raising=False)
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, []))
    with pytest.raises(ObjectiveGradientError, match="including frozen"):
        ScipyOptimizer(maxiter=1, verbose=False).optimize(evaluator, space)


def test_selected_provider_cannot_return_a_zero_padded_full_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    evaluator, probes = _executor()
    space = evaluator.plan.active_space.with_active_indices([1])
    monkeypatch.setattr(
        evaluator,
        "value_and_gradient_selected",
        lambda params, indices: (evaluator.value(params), np.zeros(4)),
        raising=False,
    )
    monkeypatch.setattr("scipy.optimize.minimize", _stub_minimize(probes, []))
    with pytest.raises(ObjectiveGradientError, match="shape"):
        ScipyOptimizer(maxiter=1, verbose=False).optimize(evaluator, space)


def test_optional_protocol_does_not_change_the_full_evaluator_protocol() -> None:
    from q2mm.objectives.protocols import ObjectiveEvaluator

    assert getattr(objectives, "SelectedGradientEvaluator", None) is not None
    assert "value_and_gradient_selected" not in ObjectiveEvaluator.__dict__


@pytest.mark.parametrize("indices", [[0, 1, 2, 3], [1, 3]])
def test_scipy_owned_residual_jacobian_keeps_active_probes_and_relative_step(
    indices: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    from scipy import optimize as scipy_optimize

    evaluator, probes = _executor(step=0.001)
    space = evaluator.plan.active_space.with_active_indices(indices)
    original = scipy_optimize.least_squares
    jacobians = []

    def inspect_ls(*args: Any, **kwargs: Any) -> Any:
        assert "jac" not in kwargs
        assert kwargs["diff_step"] == 0.02
        result = original(*args, **kwargs)
        jacobians.append(result.jac)
        return result

    monkeypatch.setattr(scipy_optimize, "least_squares", inspect_ls)
    result = ScipyOptimizer(method="least_squares", maxiter=1, eps=0.02, verbose=False).optimize(evaluator, space)
    expected = np.vstack([-2 * np.ones(4), -0.5 * np.ones(4), 0.5 * np.eye(4)])
    np.testing.assert_allclose(jacobians[0], expected[:, indices], atol=1e-12)
    assert jacobians[0].shape == (6, len(indices))
    assert len(probes) == 2 + len(indices)
    assert evaluator.n_gradient_evaluations == 0
    assert result.fd_step == 0.02
    frozen = [i for i in range(4) if i not in indices]
    assert all(np.array_equal(point[frozen], space.baseline[frozen]) for point in probes)
    for point, index in zip(probes[2:], indices, strict=True):
        expected_point = space.baseline.copy()
        expected_point[index] += 0.02 * expected_point[index]
        np.testing.assert_allclose(point, expected_point, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("execution", ["multi-start", "cycling"])
def test_least_squares_jacobians_use_rebased_and_derived_spaces(
    execution: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scipy import optimize as scipy_optimize
    from q2mm.optimizers.cycling import OptimizationLoop, SensitivityResult

    evaluator, probes = _executor(step=0.001)
    baseline = np.array([2.0, 5.0, 7.0, 8.0])
    indices = [1, 3] if execution == "multi-start" else [0, 1, 3]
    space = evaluator.plan.active_space.with_active_indices(indices).with_baseline(baseline)
    original = scipy_optimize.least_squares
    groups = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        before = len(probes)
        result = original(*args, **kwargs)
        groups.append((list(probes[before:]), result.jac))
        return result

    monkeypatch.setattr(scipy_optimize, "least_squares", observe)
    if execution == "multi-start":
        MultiStartOptimizer(
            ScipyOptimizer(method="least_squares", maxiter=1, eps=0.02, verbose=False),
            n_starts=2,
            seed=3,
            verbose=False,
        ).optimize(evaluator, space)
        requested = [[1, 3], [1, 3]]
    else:
        monkeypatch.setattr(
            "q2mm.optimizers.cycling.compute_sensitivity",
            lambda *args, **kwargs: SensitivityResult(
                d1=np.ones(4),
                d2=np.ones(4),
                simp_var=np.ones(4),
                ranking=np.array([3, 1, 0, 2]),
                metric="simp_var",
                n_evals=0,
            ),
        )
        OptimizationLoop(
            evaluator,
            space,
            max_params=1,
            max_cycles=1,
            full_method="least_squares",
            simp_method="least_squares",
            full_maxiter=1,
            simp_maxiter=1,
            eps=0.02,
            verbose=False,
        ).run()
        requested = [[0, 1, 3], [3]]
    expected = np.vstack([-2 * np.ones(4), -0.5 * np.ones(4), 0.5 * np.eye(4)])
    for (points, jacobian), coordinates in zip(groups, requested, strict=True):
        assert len(points) == 1 + len(coordinates)
        np.testing.assert_allclose(jacobian, expected[:, coordinates], atol=1e-12)
        unchanged = [i for i in range(4) if i not in coordinates]
        assert all(np.array_equal(point[unchanged], points[0][unchanged]) for point in points)
        assert points[0][2] == baseline[2]
    assert evaluator.n_gradient_evaluations == 0
    np.testing.assert_array_equal(space.baseline, baseline)
