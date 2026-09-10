"""Tests for executor-driven SciPy gradient behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from test._shared import make_diatomic
from test.test_multistart import QuadraticEvaluator


def _mock_engine(supports_grad: bool) -> MagicMock:
    """Return a MagicMock backend whose ``.info`` declares gradient capabilities."""
    from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

    caps: set[Capability] = {Capability.ENERGY, Capability.HESSIAN, Capability.FREQUENCIES}
    if supports_grad:
        caps |= {Capability.PARAMETER_GRADIENT, Capability.HESSIAN_PARAMETER_JACOBIAN}
    backend = MagicMock()
    backend.info = BackendInfo(
        name="mock",
        role=BackendRole.MM,
        capabilities=frozenset(caps),
        functional_forms=frozenset({"harmonic"}),
        provenance=BackendProvenance(backend="mock", role=BackendRole.MM),
    )
    backend.prepare.return_value.info = backend.info
    return backend


@dataclass(frozen=True)
class MockForceField:
    """Minimal immutable force field for optimizer tests."""

    params: tuple[float, ...]


class MockLayout:
    """Minimal layout exposing vector/replace over MockForceField."""

    def __init__(self, n_params: int) -> None:
        self.n_params = n_params

    def __len__(self) -> int:
        return self.n_params

    @property
    def fingerprint(self) -> str:
        return f"mock:{self.n_params}"

    def vector(self, forcefield: MockForceField) -> np.ndarray:
        return np.asarray(forcefield.params, dtype=np.float64)

    def replace(self, forcefield: MockForceField, vector: np.ndarray) -> MockForceField:
        values = np.asarray(vector, dtype=np.float64)
        return MockForceField(tuple(values.tolist()))


class MockSpace:
    """Active/full parameter projection used by optimizer tests."""

    def __init__(
        self,
        baseline: np.ndarray,
        bounds: list[tuple[float, float]],
        active_indices: np.ndarray | None = None,
    ) -> None:
        self.baseline = np.asarray(baseline, dtype=np.float64).copy()
        self.layout = MockLayout(self.baseline.size)
        self.active_indices = (
            np.arange(self.baseline.size, dtype=int)
            if active_indices is None
            else np.asarray(active_indices, dtype=int)
        )
        self._full_bounds = np.asarray(bounds, dtype=np.float64)

    @property
    def n_active(self) -> int:
        return int(self.active_indices.size)

    @property
    def n_full(self) -> int:
        return int(self.baseline.size)

    @property
    def bounds(self) -> np.ndarray:
        return self._full_bounds[self.active_indices]

    def pack(self, full_vector: np.ndarray) -> np.ndarray:
        full = np.asarray(full_vector, dtype=np.float64)
        return full[self.active_indices].copy()

    def expand(self, active_vector: np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
        full = self.baseline.copy() if base is None else np.asarray(base, dtype=np.float64).copy()
        full[self.active_indices] = np.asarray(active_vector, dtype=np.float64)
        return full

    def with_baseline(self, vector: np.ndarray) -> MockSpace:
        return MockSpace(np.asarray(vector, dtype=float), self._full_bounds.tolist(), self.active_indices)


class _MockObjective:
    """Lightweight objective evaluator for testing executor-driven gradient modes."""

    def __init__(self, *, gradient_mode: GradientMode = GradientMode.NONE) -> None:
        baseline = np.array([1.0, 2.0], dtype=np.float64)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(2)
        self.space = MockSpace(baseline, bounds=[(0.0, 10.0), (0.0, 10.0)])
        self.plan = SimpleNamespace(categories=frozenset({"energy"}))
        self.history: list[float] = []
        self._n_eval = 0
        self._gradient_mode = gradient_mode

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    @property
    def finite_difference_step(self) -> float | None:
        return None

    @property
    def n_evaluations(self) -> int:
        return self._n_eval

    def record_evaluation(self, score: float) -> None:
        self._n_eval += 1
        self.history.append(float(score))

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        score = float(np.sum((x - np.array([0.5, 1.5])) ** 2))
        self._n_eval += 1
        self.history.append(score)
        return score

    def value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.gradient_mode is GradientMode.NONE:
            raise ObjectiveGradientError("No evaluator gradient available")
        value = self.value(x)
        return value, 2.0 * (np.asarray(x, dtype=float) - np.array([0.5, 1.5]))

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - np.array([0.5, 1.5])

    def least_squares_residuals(self, x: np.ndarray) -> np.ndarray:
        return self.residuals(x)


class _MockFrozenObjective:
    """Quadratic objective over a full parameter vector with frozen entries."""

    def __init__(self, *, gradient_mode: GradientMode = GradientMode.NONE) -> None:
        baseline = np.array([0.0, 5.0, 0.0], dtype=float)
        self.target = np.array([1.0, 4.0, 3.0], dtype=float)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(3)
        self.space = MockSpace(
            baseline,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            active_indices=np.array([0, 2]),
        )
        self.plan = SimpleNamespace(categories=frozenset({"energy"}))
        self.history: list[float] = []
        self._n_eval = 0
        self._gradient_mode = gradient_mode

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    @property
    def finite_difference_step(self) -> float | None:
        return None

    @property
    def n_evaluations(self) -> int:
        return self._n_eval

    def record_evaluation(self, score: float) -> None:
        self._n_eval += 1
        self.history.append(float(score))

    def value(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=float) - self.target) ** 2))
        self._n_eval += 1
        self.history.append(score)
        return score

    def value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.gradient_mode is GradientMode.NONE:
            raise ObjectiveGradientError("No evaluator gradient available")
        x = np.asarray(x, dtype=float)
        return self.value(x), 2.0 * (x - self.target)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - self.target

    def least_squares_residuals(self, x: np.ndarray) -> np.ndarray:
        return self.residuals(x)


def _run_ignoring_errors(opt: ScipyOptimizer, obj: _MockObjective) -> None:
    opt.optimize(obj, obj.space)


def _h2_ff() -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=359.7, equilibrium=0.74)],
        functional_form=FunctionalForm.HARMONIC,
    )


def _plan_for_kinds(kinds: tuple[str, ...]) -> ObjectivePlan:
    ff = _h2_ff()
    mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
    ref = ObservationSet()
    for kind in kinds:
        if kind == "energy":
            ref = ref.with_energy(0.0, case_id="0")
        elif kind == "frequency":
            ref = ref.with_frequency(100.0, data_idx=0, case_id="0")
        elif kind == "bond_length":
            ref = ref.with_bond_length(1.5, atom_indices=(0, 1), case_id="0")
        elif kind == "hessian_element":
            ref = ref.with_hessian_element(0.1, row=0, col=0, case_id="0")
    layout = ParameterLayout.from_force_field(ff)
    return ObjectivePlan(
        case_ids=("0",),
        molecules=(mol,),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ref,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, ff),
    )


class TestJacAutoDetection:
    """Verify the optimizer follows executor-declared gradient support."""

    def test_lbfgsb_auto_enables_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_analytical_lbfgsb_scales_bounds_and_recovers_best(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scipy import optimize

        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)

        def fake_minimize(fun, x0, *, method, jac, bounds, options, callback):  # noqa: ANN001, ANN202
            np.testing.assert_allclose(x0, [-0.8, -0.6])
            assert method == "L-BFGS-B"
            assert jac is True
            assert bounds == [(-1.0, 1.0), (-1.0, 1.0)]
            assert options["gtol"] == pytest.approx(1e-5)
            assert options["maxls"] == 100
            best_x = np.array([-0.9, -0.7])
            best_value, _best_gradient = fun(best_x)
            assert best_value == pytest.approx(0.0)
            terminal_x = np.array([1.0, 1.0])
            terminal_value, terminal_gradient = fun(terminal_x)
            return optimize.OptimizeResult(
                x=terminal_x,
                fun=terminal_value,
                jac=terminal_gradient,
                nit=2,
                nfev=2,
                njev=2,
                success=True,
                message="synthetic terminal penalty",
            )

        monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=2, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.final_params, [0.5, 1.5])
        assert result.final_score == pytest.approx(0.0)
        assert result.success is False
        assert result.message.startswith("Recovered best evaluated point")

    @pytest.mark.parametrize("terminal_score", [float("nan"), float("inf")])
    def test_analytical_lbfgsb_recovers_best_from_nonfinite_terminal(
        self,
        monkeypatch: pytest.MonkeyPatch,
        terminal_score: float,
    ) -> None:
        from scipy import optimize

        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)

        def fake_minimize(fun, x0, *, method, jac, bounds, options, callback):  # noqa: ANN001, ANN202
            best_x = np.array([-0.9, -0.7])
            fun(best_x)
            return optimize.OptimizeResult(
                x=np.array([1.0, 1.0]),
                fun=terminal_score,
                jac=np.array([np.nan, np.nan]),
                nit=2,
                success=False,
                message="synthetic non-finite terminal",
            )

        monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=2, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.final_params, [0.5, 1.5])
        assert result.final_score == pytest.approx(0.0)
        assert result.success is False
        assert result.message.startswith("Recovered best evaluated point")

    def test_lbfgsb_no_analytical_when_unsupported(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_lbfgsb_default_jac_none_uses_fd(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_nelder_mead_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Nelder-Mead", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_powell_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Powell", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_explicit_analytical_overrides_auto(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_derivative_free_methods_set(self) -> None:
        assert "Nelder-Mead" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "Powell" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "L-BFGS-B" not in ScipyOptimizer.DERIVATIVE_FREE_METHODS


class TestFrozenParameterSupport:
    """Frozen parameters are excluded from optimizer updates."""

    def test_frozen_values_outside_active_bounds_are_not_rejected(self) -> None:
        obj = _MockFrozenObjective(gradient_mode=GradientMode.ANALYTICAL)
        obj.space = MockSpace(
            obj.space.baseline,
            bounds=[(-10.0, 10.0), (-1.0, 1.0), (-10.0, 10.0)],
            active_indices=np.array([0, 2]),
        )
        result = ScipyOptimizer(verbose=False).optimize(obj, obj.space)
        assert result.final_params[1] == 5.0
        assert result.final_score < result.initial_score

    def test_no_active_parameters_preserves_baseline(self) -> None:
        obj = _MockObjective()
        obj.space = MockSpace(obj.space.baseline, bounds=[(0.0, 0.5)] * 2, active_indices=np.array([], dtype=int))
        result = ScipyOptimizer(verbose=False).optimize(obj, obj.space)
        assert result.success
        assert result.n_iterations == 0
        assert result.final_score == result.initial_score
        np.testing.assert_array_equal(result.final_params, obj.space.baseline)

    def test_lbfgsb_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_params[0] != pytest.approx(result.initial_params[0])
        assert result.final_params[2] != pytest.approx(result.initial_params[2])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])

    def test_least_squares_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="least_squares", maxiter=50, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])


class TestBoundedExecution:
    @pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead", "Powell", "trust-constr", "least_squares"])
    @pytest.mark.parametrize("fraction", [None, 0.2])
    def test_infeasible_start_is_rejected_before_evaluation(self, method: str, fraction: float | None) -> None:
        obj = QuadraticEvaluator(np.array([10.0]), bounds=[(0.0, 5.0)], initial=np.array([10.0]))
        with pytest.raises(ValueError, match="Initial active parameters"):
            ScipyOptimizer(method=method, fc_fraction=fraction, verbose=False).optimize(obj, obj.space)
        assert obj.n_evaluations == 0
        np.testing.assert_array_equal(obj.space.baseline, [10.0])

    def test_unsupported_bounds_fail_before_evaluation(self) -> None:
        obj = _MockObjective()
        with pytest.raises(ValueError, match="use_bounds=False"):
            ScipyOptimizer(method="BFGS", verbose=False).optimize(obj, obj.space)
        assert obj.n_evaluations == 0

    @pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "L-BFGS-B"])
    @pytest.mark.parametrize("fraction", [None, 0.2])
    def test_external_target_stays_within_effective_bounds(self, method: str, fraction: float | None) -> None:
        obj = QuadraticEvaluator(np.array([10.0]), bounds=[(0.0, 5.0)], initial=np.array([1.0]))
        result = ScipyOptimizer(method=method, fc_fraction=fraction, verbose=False).optimize(obj, obj.space)

        lower, upper = (0.0, 5.0) if fraction is None else (0.8, 1.2)
        assert lower <= result.final_params[0] <= upper
        assert result.final_params[0] == pytest.approx(upper, abs=1e-5)
        assert result.final_score == pytest.approx(np.sum((result.final_params - obj.target) ** 2))

    @pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "L-BFGS-B"])
    def test_disabled_bounds_remain_unbounded(self, method: str) -> None:
        obj = QuadraticEvaluator(np.array([10.0]), bounds=[(0.0, 5.0)], initial=np.array([1.0]))
        result = ScipyOptimizer(method=method, use_bounds=False, fc_fraction=0.2, verbose=False).optimize(
            obj, obj.space
        )

        assert result.final_params[0] == pytest.approx(10.0)
        assert result.final_score == pytest.approx(np.sum((result.final_params - obj.target) ** 2))

    def test_unbounded_method_can_start_outside_sanity_bounds(self) -> None:
        obj = QuadraticEvaluator(np.array([10.0]), bounds=[(0.0, 5.0)], initial=np.array([8.0]))
        result = ScipyOptimizer(method="BFGS", use_bounds=False, verbose=False).optimize(obj, obj.space)
        np.testing.assert_array_equal(result.initial_params, [8.0])
        np.testing.assert_allclose(result.final_params, [10.0])

    def test_normalized_endpoint_roundoff_does_not_discard_feasible_optimum(self) -> None:
        lower, upper = 1.791445485312492, 1.9836087943897656
        obj = QuadraticEvaluator(
            np.array([10.0]),
            bounds=[(lower, upper)],
            initial=np.array([(lower + upper) / 2.0]),
        )
        result = ScipyOptimizer(verbose=False).optimize(obj, obj.space)
        np.testing.assert_array_equal(result.final_params, [upper])
        assert result.final_score == pytest.approx((upper - 10.0) ** 2)

    @pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "L-BFGS-B"])
    @pytest.mark.parametrize("terminal", [2.0, 10.0])
    def test_infeasible_trial_cannot_be_recovered(
        self, monkeypatch: pytest.MonkeyPatch, method: str, terminal: float
    ) -> None:
        from scipy.optimize import OptimizeResult

        obj = QuadraticEvaluator(np.array([10.0]), bounds=[(0.0, 5.0)], initial=np.array([1.0]))

        def fake_minimize(fun, x0, *, method, jac, bounds, options, callback):  # noqa: ANN001, ANN202
            def trial(x):  # noqa: ANN001, ANN202
                solver_x = (x - 2.5) / 2.5 if jac else x
                value = fun(np.array([solver_x]))
                return value[0] if jac else value

            trial(4.0)
            trial(10.0)
            score = trial(terminal)
            return OptimizeResult(
                x=np.array([(terminal - 2.5) / 2.5 if jac else terminal]),
                fun=score,
                nit=3,
                success=True,
                message="terminal",
            )

        monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
        result = ScipyOptimizer(method=method, verbose=False).optimize(obj, obj.space)

        np.testing.assert_array_equal(result.final_params, [4.0])
        assert result.final_score == 36.0
        assert not result.success
        assert "Recovered best evaluated point" in result.message

    def test_fractional_scaling_preserves_gradient_chain_rule(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scipy.optimize import OptimizeResult

        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        ff = ForceField(
            bonds=(BondParam(("H", "H"), force_constant=100.0, equilibrium=0.75),),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        space = ActiveParameterSpace.all_active(layout, ff)

        def fake_minimize(fun, x0, *, method, jac, bounds, options, callback):  # noqa: ANN001, ANN202
            np.testing.assert_allclose(x0, [0.0, 0.0])
            assert bounds == [(-1.0, 1.0), (-1.0, 1.0)]
            value, gradient = fun(np.array([-0.5, 0.5]))
            physical = np.array([90.0, 0.76875])
            np.testing.assert_allclose(gradient, 2.0 * (physical - [0.5, 1.5]) * [20.0, 0.0375])
            assert value == pytest.approx(np.sum((physical - [0.5, 1.5]) ** 2))
            return OptimizeResult(x=np.array([-0.5, 0.5]), fun=value, nit=1, success=True, message="done")

        monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
        result = ScipyOptimizer(fc_fraction=0.2, eq_fraction=0.05, verbose=False).optimize(obj, space)
        np.testing.assert_allclose(result.final_params, [90.0, 0.76875])


class _RosenbrockEvaluator(QuadraticEvaluator):
    def _total(self, x: np.ndarray) -> float:
        return float((1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2)

    def _data_gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] ** 2), 200.0 * (x[1] - x[0] ** 2)])


class TestDivergenceTermination:
    @pytest.mark.parametrize(("factor", "initial_score"), [(None, 1.0), (3.0, 0.0), (3.0, -1.0)])
    def test_inactive_x_only_callback_does_not_evaluate(self, factor: float | None, initial_score: float) -> None:
        obj = _MockObjective()
        value_at = MagicMock(wraps=obj.value)
        callback = ScipyOptimizer(divergence_factor=factor, verbose=False)._make_callback(obj, initial_score, value_at)

        for x in ([1.0, 2.0], [0.5, 1.5]):
            callback(np.asarray(x))

        value_at.assert_not_called()
        assert obj.n_evaluations == 0
        assert obj.history == []

    def test_x_only_callback_still_evaluates_for_verbose_logging(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective()
        for _ in range(9):
            obj.record_evaluation(0.5)
        value_at = MagicMock(wraps=obj.value)
        callback = ScipyOptimizer(divergence_factor=None, verbose=True)._make_callback(obj, 1.0, value_at)
        caplog.set_level("INFO", logger="q2mm.optimizers.scipy_opt")

        callback(np.array([0.5, 1.5]))

        value_at.assert_called_once()
        assert obj.n_evaluations == 10
        assert obj.history[-1] == 0.0
        assert "eval   10  score 0.000000" in caplog.text

    @pytest.mark.parametrize("method", ["L-BFGS-B", "Nelder-Mead", "Powell", "trust-constr"])
    def test_real_solver_stops_before_iteration_limit(self, method: str) -> None:
        obj = _RosenbrockEvaluator(np.zeros(2), bounds=[(-5.0, 5.0)] * 2, initial=np.array([-1.2, 1.3]))
        result = ScipyOptimizer(
            method=method,
            maxiter=10,
            divergence_factor=1e-6,
            divergence_patience=1,
            verbose=False,
        ).optimize(obj, obj.space)

        assert result.n_iterations < 10
        assert not result.success
        assert "Abandoned" in result.message
        assert np.all(np.abs(result.final_params) <= 5.0)
        assert result.final_score == pytest.approx(obj._total(result.final_params))

    def test_tnc_unbounded_x_only_callback_stops(self) -> None:
        obj = _RosenbrockEvaluator(np.zeros(2), initial=np.array([-1.2, 1.3]))
        result = ScipyOptimizer(
            method="TNC",
            use_bounds=False,
            divergence_factor=1e-6,
            divergence_patience=1,
            verbose=False,
        ).optimize(obj, obj.space)
        assert result.n_iterations == 1
        assert not result.success
        assert "Abandoned" in result.message
        assert result.final_score == pytest.approx(obj._total(result.final_params))

    def test_objective_stop_iteration_is_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        obj = _MockObjective()

        def stop_from_objective(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            raise StopIteration("objective interruption")

        monkeypatch.setattr("scipy.optimize.minimize", stop_from_objective)
        with pytest.raises(StopIteration, match="objective interruption"):
            ScipyOptimizer(verbose=False).optimize(obj, obj.space)

    @pytest.mark.parametrize("through_workflow", [False, True])
    def test_old_fixed_variable_callback_preserves_objective_interruption(
        self, monkeypatch: pytest.MonkeyPatch, through_workflow: bool
    ) -> None:
        from scipy import optimize

        from q2mm.models.problem import OptimizationProblem, TrainingCase
        from q2mm.workflows import SingleStageWorkflow

        obj = QuadraticEvaluator(np.array([0.5, 1.5]), bounds=[(1.0, 1.0), (0.0, 10.0)], initial=np.array([1.0, 2.0]))
        obj._gradient_mode = GradientMode.NONE
        interruption = StopIteration("objective interruption during accepted-iterate evaluation")
        original_value = obj.value
        calls = 0
        in_callback = False

        def interrupted_value(x: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            if calls == 8:
                assert in_callback
                raise interruption
            return original_value(x)

        monkeypatch.setattr(obj, "value", interrupted_value)
        sample = MagicMock(side_effect=AssertionError("endpoint sampling must not run"))
        evaluate = MagicMock(side_effect=AssertionError("endpoint evaluation must not run"))
        monkeypatch.setattr(obj, "sample", sample)
        monkeypatch.setattr(obj, "evaluate", evaluate)
        original_minimize = optimize.minimize

        def old_fixed_variable_minimize(fun, x0, *, callback, **kwargs):  # noqa: ANN001, ANN003, ANN202
            np.testing.assert_array_equal(kwargs["bounds"], [(1.0, 1.0), (0.0, 10.0)])
            assert kwargs["jac"] is None

            # SciPy 1.15's fixed-variable adapter hides the rich callback
            # signature; emulate it while keeping the real solver/stop handler.
            def x_only(xk: np.ndarray) -> None:
                nonlocal in_callback
                in_callback = True
                try:
                    callback(xk)
                finally:
                    in_callback = False

            return original_minimize(fun, x0, callback=x_only, **kwargs)

        monkeypatch.setattr(optimize, "minimize", old_fixed_variable_minimize)
        optimizer = ScipyOptimizer(verbose=False)
        with pytest.raises(StopIteration) as caught:
            if through_workflow:
                problem = OptimizationProblem(
                    cases=(
                        TrainingCase(
                            case_id="0",
                            molecule=obj.plan.molecules[0],
                            stationary_point=StationaryPointKind.GROUND_STATE,
                        ),
                    ),
                    starting_force_field=obj.forcefield,
                    layout=obj.space.layout,
                    active_space=obj.space,
                    observations=obj.plan.observations,
                )
                SingleStageWorkflow().run(problem, lambda _plan: obj, optimizer)
            else:
                optimizer.optimize(obj, obj.space)
        assert caught.value is interruption
        assert calls == 8
        sample.assert_not_called()
        evaluate.assert_not_called()

    def test_callback_uses_accepted_score_and_resets_patience(self) -> None:
        from scipy.optimize import OptimizeResult

        obj = _MockObjective()
        callback = ScipyOptimizer(divergence_factor=3.0, divergence_patience=2, verbose=False)._make_callback(obj, 1.0)
        obj.record_evaluation(1000.0)
        callback(OptimizeResult(x=np.array([1.0, 2.0]), fun=1.0))
        callback(OptimizeResult(x=np.array([1.0, 2.0]), fun=4.0))
        callback(OptimizeResult(x=np.array([1.0, 2.0]), fun=1.0))
        obj.record_evaluation(0.0)
        callback(OptimizeResult(x=np.array([1.0, 2.0]), fun=4.0))
        with pytest.raises(StopIteration):
            callback(OptimizeResult(x=np.array([1.0, 2.0]), fun=4.0))

    def test_abandonment_cannot_report_solver_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from scipy.optimize import OptimizeResult

        obj = _MockObjective()

        def fake_minimize(fun, x0, *, method, jac, bounds, options, callback):  # noqa: ANN001, ANN202
            accepted = np.array([0.5, 1.6])
            value = fun(accepted)
            with pytest.raises(StopIteration):
                callback(intermediate_result=OptimizeResult(x=accepted, fun=value))
            return OptimizeResult(x=accepted, fun=value, nit=1, success=True, message="incorrect solver success")

        monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)
        result = ScipyOptimizer(divergence_factor=0.01, divergence_patience=1, verbose=False).optimize(obj, obj.space)
        assert not result.success
        assert result.message == "Abandoned: sustained divergence from initial score"
        np.testing.assert_array_equal(result.final_params, [0.5, 1.6])
        assert result.final_score == pytest.approx(0.01)


class TestOptimizationResultFields:
    """Verify gradient_mode and fd_step are set correctly on OptimizationResult."""

    def test_lbfgsb_auto_with_support_sets_eps_none(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_lbfgsb_fd_sets_eps(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_derivative_free_sets_eps_none(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Powell", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_custom_eps_value(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, eps=5e-4).optimize(obj, obj.space)
        assert result.fd_step == 5e-4


class TestPerEvaluatorGradientSupport:
    """Verify explicit Python executor analytical-gradient support checks."""

    @staticmethod
    def _make_objective(*, engine_supports_grad: bool, kinds: tuple[str, ...]) -> PythonObjectiveExecutor:
        ff = _h2_ff()
        return PythonObjectiveExecutor(
            _plan_for_kinds(kinds),
            _mock_engine(engine_supports_grad),
            ff,
            gradient_mode=GradientMode.ANALYTICAL,
        )

    def test_energy_and_frequency_with_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "frequency"))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_energy_and_frequency_without_analytical_engine(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="PARAMETER_GRADIENT|HESSIAN_PARAMETER_JACOBIAN"):
            self._make_objective(engine_supports_grad=False, kinds=("energy", "frequency"))

    def test_energy_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_frequency_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_geometry_always_false(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="geometry references"):
            self._make_objective(engine_supports_grad=True, kinds=("bond_length",))

    def test_hessian_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("hessian_element",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL
