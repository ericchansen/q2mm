"""Unit tests for OptaxOptimizer (no backend required)."""

from __future__ import annotations
from test.backend_fixtures import mock_backend_info

from dataclasses import dataclass
from importlib.util import find_spec
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    find_spec("optax") is None or find_spec("jax") is None,
    reason="optax and jax are required",
)


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
        bounds: list[tuple[float, float]] | None = None,
        active_indices: np.ndarray | None = None,
    ) -> None:
        self.baseline = np.asarray(baseline, dtype=np.float64).copy()
        self.active_indices = (
            np.arange(self.baseline.size, dtype=int)
            if active_indices is None
            else np.asarray(active_indices, dtype=int)
        )
        full_bounds = bounds if bounds is not None else [(-1_000.0, 1_000.0)] * self.baseline.size
        self._full_bounds = np.asarray(full_bounds, dtype=np.float64)

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
        return MockSpace(
            baseline=np.asarray(vector, dtype=np.float64),
            bounds=self._full_bounds.tolist(),
            active_indices=self.active_indices.copy(),
        )


class MockObjective:
    """Minimal ObjectiveFunction mock with a quadratic loss."""

    def __init__(
        self,
        target: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        initial: np.ndarray | None = None,
    ) -> None:
        self.target = target.astype(np.float64)
        baseline = np.zeros_like(self.target) if initial is None else np.asarray(initial, dtype=np.float64)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(self.target.size)
        self.space = MockSpace(baseline, bounds=bounds)
        self.n_eval = 0
        self.history: list[float] = []
        self.backend = MagicMock()
        self.backend.info = mock_backend_info(param_grad=True, hess_jac=True)

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=np.float64) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x, dtype=np.float64) - self.target)


class MockDivergentObjective:
    """Objective that always returns increasing scores."""

    def __init__(self, n_params: int = 3) -> None:
        self._call_count = 0
        baseline = np.ones(n_params, dtype=np.float64)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(n_params)
        self.space = MockSpace(baseline)
        self.n_eval = 0
        self.history: list[float] = []
        self.backend = MagicMock()
        self.backend.info = mock_backend_info(param_grad=True, hess_jac=True)

    def __call__(self, x: np.ndarray) -> float:
        self._call_count += 1
        score = 10.0 * self._call_count
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(x, dtype=np.float64)) * 100.0


class MockFrozenObjective:
    """Quadratic objective with one frozen full-vector coordinate."""

    def __init__(self) -> None:
        baseline = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        self.target = np.array([1.0, 4.0, 3.0], dtype=np.float64)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(3)
        self.space = MockSpace(
            baseline,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            active_indices=np.array([0, 2]),
        )
        self.n_eval = 0
        self.history: list[float] = []
        self.backend = MagicMock()
        self.backend.info = mock_backend_info(param_grad=True, hess_jac=True)

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=np.float64) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (x - self.target)


class TestOptaxOptimizerCreation:
    """Test optimizer instantiation and validation."""

    def test_create_adam(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adam")
        assert opt.optimizer_name == "adam"

    def test_create_adagrad(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adagrad")
        assert opt.optimizer_name == "adagrad"

    def test_create_sgd(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="sgd", momentum=0.9)
        assert opt.optimizer_name == "sgd"
        assert opt.momentum == 0.9

    def test_create_adamw(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adamw")
        assert opt.optimizer_name == "adamw"

    def test_invalid_optimizer(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptaxOptimizer(optimizer="rmsprop")

    def test_invalid_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(schedule="invalid")
        obj = MockObjective(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="Unknown schedule"):
            opt.optimize(obj, obj.space)


class TestOptaxConvergence:
    """Test that the optimizer converges on simple problems."""

    def test_adam_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert result.final_score < 0.01
        np.testing.assert_allclose(result.final_params, target, atol=0.1)

    def test_sgd_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, -1.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="sgd", learning_rate=0.1, momentum=0.9, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 0.01

    def test_adagrad_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([5.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="adagrad", learning_rate=1.0, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 0.1

    def test_gradient_norm_convergence(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([0.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=1000,
            grad_norm_tol=1e-4,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.success
        assert "gradient norm" in result.message

    def test_score_plateau_convergence(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.05,
            max_steps=2000,
            ftol=1e-8,
            patience=20,
            grad_norm_tol=1e-12,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.success
        assert result.n_iterations < 2000


class TestOptaxFrozenParams:
    """Frozen parameters remain fixed throughout optimization."""

    def test_adam_updates_only_active_params(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = MockFrozenObjective()
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=300, verbose=False)
        result = opt.optimize(obj, obj.space)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])


class TestOptaxBounds:
    """Test parameter bounds enforcement."""

    def test_bounds_enforced(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([10.0, 10.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        obj = MockObjective(target, bounds=bounds)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=200, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert np.all(result.final_params >= 0.0 - 1e-10)
        assert np.all(result.final_params <= 5.0 + 1e-10)

    def test_no_bounds(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([100.0])
        obj = MockObjective(target, bounds=None)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=1.0,
            max_steps=500,
            use_bounds=False,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 1.0


class TestOptaxDivergence:
    """Test divergence detection and early stopping."""

    def test_divergence_stops_early(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = MockDivergentObjective(n_params=3)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=1000,
            divergence_factor=3.0,
            divergence_patience=5,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert not result.success
        assert "Abandoned" in result.message
        assert result.n_iterations < 1000


class TestOptaxFinalScoreUnits:
    """The reported final_score must be in ObjectiveFunction units."""

    def test_final_score_matches_true_objective(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=200, verbose=False)
        result = opt.optimize(obj, obj.space)

        true_score = float(np.sum((result.final_params - target) ** 2))
        assert result.final_score == pytest.approx(true_score, rel=1e-9, abs=1e-9)


class TestOptaxSchedules:
    """Test learning rate schedules."""

    def test_cosine_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            schedule="cosine",
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert "cosine" in result.method

    def test_exponential_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            schedule="exponential",
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert "exponential" in result.method


class TestOptaxResult:
    """Test OptimizationResult fields."""

    def test_result_fields(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=100, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.method.startswith("optax:")
        assert result.jac_mode in ("analytical", "auto")
        assert result.eps is None
        assert result.n_evaluations > 0
        assert result.n_iterations > 0
        assert len(result.history) > 0
        assert result.initial_params is not None
        assert result.final_params is not None
        assert isinstance(result.improvement, float)
        assert len(result.summary()) > 0

    def test_history_tracked(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=50, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert len(result.history) >= 2
        assert result.history[0] == result.initial_score

    def test_forcefield_is_not_mutated(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        original_ff = obj.forcefield
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=100, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert obj.forcefield is original_ff
        np.testing.assert_array_equal(obj.layout.vector(obj.forcefield), np.array([0.0, 0.0]))
        final_ff = obj.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.layout.vector(final_ff), result.final_params)


class TestOptaxImport:
    """Test lazy import and registration."""

    def test_importable_from_package(self) -> None:
        from q2mm.optimizers import OptaxOptimizer

        assert OptaxOptimizer is not None

    def test_in_all(self) -> None:
        import q2mm.optimizers

        assert "OptaxOptimizer" in q2mm.optimizers.__all__
