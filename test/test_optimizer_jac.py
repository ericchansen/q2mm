"""Tests for ScipyOptimizer auto-detection of analytical gradients."""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.diagnostics.benchmark import _resolve_gradients
from q2mm.models.observations import ObservationSet
from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import ScipyOptimizer


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


class _MockObjective:
    """Lightweight mock of ObjectiveFunction for testing jac resolution."""

    def __init__(self, *, engine_supports_grad: bool = False) -> None:
        baseline = np.array([1.0, 2.0], dtype=np.float64)
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = engine_supports_grad
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(2)
        self.space = MockSpace(baseline, bounds=[(0.0, 10.0), (0.0, 10.0)])
        self.history: list[float] = []
        self.n_eval = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_eval += 1
        self.history.append(1.0)
        return 1.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.2])


class _MockFrozenObjective:
    """Quadratic objective over a full parameter vector with frozen entries."""

    def __init__(self, *, method: str) -> None:
        baseline = np.array([0.0, 5.0, 0.0], dtype=float)
        self.target = np.array([1.0, 4.0, 3.0], dtype=float)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(3)
        self.space = MockSpace(
            baseline,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            active_indices=np.array([0, 2]),
        )
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = method != "least_squares"
        self.history: list[float] = []
        self.n_eval = 0

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=float) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 2.0 * (x - self.target)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        residuals = x - self.target
        self.n_eval += 1
        self.history.append(float(np.sum(residuals**2)))
        return residuals


def _run_ignoring_errors(opt: ScipyOptimizer, obj: _MockObjective) -> None:
    """Run optimizer, suppressing errors from mock returning non-standard types."""
    with contextlib.suppress(Exception):
        opt.optimize(obj, obj.space)


class TestJacAutoDetection:
    """Verify the optimizer auto-detects analytical gradient support."""

    def test_lbfgsb_auto_enables_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected analytical gradient support" in caplog.text

    def test_lbfgsb_no_analytical_when_unsupported(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text

    def test_lbfgsb_default_jac_none_uses_fd(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True)

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text
        assert "analytical" not in caplog.text.lower()

    def test_nelder_mead_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text
        assert "analytical" not in caplog.text.lower()

    def test_powell_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Powell", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text

    def test_explicit_analytical_overrides_auto(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="analytical")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Using analytical gradients (jac='analytical')" in caplog.text
        assert "Auto-detected" not in caplog.text

    def test_derivative_free_methods_set(self) -> None:
        assert "Nelder-Mead" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "Powell" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "L-BFGS-B" not in ScipyOptimizer.DERIVATIVE_FREE_METHODS


class TestFrozenParameterSupport:
    """Frozen parameters are excluded from optimizer updates."""

    def test_lbfgsb_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(method="L-BFGS-B")
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False, jac="analytical")
        result = opt.optimize(obj, obj.space)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_params[0] != pytest.approx(result.initial_params[0])
        assert result.final_params[2] != pytest.approx(result.initial_params[2])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])

    def test_least_squares_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(method="least_squares")
        opt = ScipyOptimizer(method="least_squares", maxiter=50, verbose=False)
        result = opt.optimize(obj, obj.space)

        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])


class TestOptimizationResultFields:
    """Verify jac_mode and eps are set correctly on OptimizationResult."""

    def test_lbfgsb_auto_with_support_sets_eps_none(self) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac="auto")
        result = opt.optimize(obj, obj.space)
        assert result.jac_mode == "auto"
        assert result.eps is None

    def test_lbfgsb_fd_sets_eps(self) -> None:
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac=None)
        result = opt.optimize(obj, obj.space)
        assert result.jac_mode is None
        assert result.eps == 1e-3

    def test_derivative_free_sets_eps_none(self) -> None:
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Powell", maxiter=1, jac="auto")
        result = opt.optimize(obj, obj.space)
        assert result.jac_mode == "auto"
        assert result.eps is None

    def test_custom_eps_value(self) -> None:
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac=None, eps=5e-4)
        result = opt.optimize(obj, obj.space)
        assert result.eps == 5e-4


class TestResolveGradients:
    """Verify _resolve_gradients produces correct per-evaluator gradient maps."""

    @staticmethod
    def _make_objective(
        *, engine_supports_grad: bool, kinds: tuple[str, ...] = ("energy", "frequency")
    ) -> ObjectiveFunction:
        engine = MagicMock()
        engine.supports_analytical_gradients.return_value = engine_supports_grad
        engine.supports_analytical_hessian_gradients.return_value = engine_supports_grad
        ref = ObservationSet()
        for kind in kinds:
            if kind == "energy":
                ref = ref.with_energy(0.0)
            elif kind == "frequency":
                ref = ref.with_frequency(100.0, data_idx=0)
            elif kind == "bond_length":
                ref = ref.with_bond_length(1.5, atom_indices=(0, 1))
            elif kind == "hessian_element":
                ref = ref.with_hessian_element(0.1, row=0, col=0)
        return ObjectiveFunction(None, engine, [], ref)

    def test_auto_with_analytical_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "frequency": "analytical"}

    def test_auto_without_analytical_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=False)
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "finite-diff", "frequency": "finite-diff"}

    def test_jac_none_is_fd(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients(None, obj)
        assert result == {"energy": "finite-diff", "frequency": "finite-diff"}

    def test_analytical_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("analytical", obj)
        assert result == {"energy": "analytical", "frequency": "analytical"}

    def test_derivative_free_method_overrides_jac(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj, method="Powell")
        assert result == {"energy": "n/a", "frequency": "n/a"}

    def test_nelder_mead_is_derivative_free(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj, method="Nelder-Mead")
        assert result == {"energy": "n/a", "frequency": "n/a"}

    def test_energy_only_objective(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical"}

    def test_frequency_only_objective(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        result = _resolve_gradients("auto", obj)
        assert result == {"frequency": "analytical"}

    def test_geometry_refs_always_fd(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "bond_length"))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "geometry": "finite-diff"}

    def test_hessian_refs_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "hessian_element"))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "hessian": "analytical"}


class TestPerEvaluatorGradientSupport:
    """Verify ObjectiveFunction.per_evaluator_gradient_support()."""

    @staticmethod
    def _make_objective(*, engine_supports_grad: bool, kinds: tuple[str, ...]) -> ObjectiveFunction:
        engine = MagicMock()
        engine.supports_analytical_gradients.return_value = engine_supports_grad
        engine.supports_analytical_hessian_gradients.return_value = engine_supports_grad
        ref = ObservationSet()
        for kind in kinds:
            if kind == "energy":
                ref = ref.with_energy(0.0)
            elif kind == "frequency":
                ref = ref.with_frequency(100.0, data_idx=0)
            elif kind == "bond_length":
                ref = ref.with_bond_length(1.5, atom_indices=(0, 1))
            elif kind == "hessian_element":
                ref = ref.with_hessian_element(0.1, row=0, col=0)
        return ObjectiveFunction(None, engine, [], ref)

    def test_energy_and_frequency_with_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "frequency"))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": True, "frequency": True}

    def test_energy_and_frequency_without_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=False, kinds=("energy", "frequency"))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": False, "frequency": False}

    def test_energy_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": True}

    def test_frequency_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"frequency": True}

    def test_geometry_always_false(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("bond_length",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"geometry": False}

    def test_hessian_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("hessian_element",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"hessian": True}

    def test_result_is_sorted_by_category(self) -> None:
        obj = self._make_objective(
            engine_supports_grad=True,
            kinds=("frequency", "energy", "hessian_element", "bond_length"),
        )
        result = obj.per_evaluator_gradient_support()
        assert list(result.keys()) == ["energy", "frequency", "geometry", "hessian"]
