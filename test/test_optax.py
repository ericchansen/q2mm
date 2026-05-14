"""Unit tests for OptaxOptimizer (no backend required).

These tests use a mock ObjectiveFunction with a simple quadratic loss
to verify optimizer mechanics: convergence, bounds, schedules, and
divergence detection.
"""

from __future__ import annotations

from importlib.util import find_spec
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    find_spec("optax") is None or find_spec("jax") is None,
    reason="optax and jax are required",
)


class MockObjective:
    """Minimal ObjectiveFunction mock with a quadratic loss.

    ``f(x) = sum((x - target)**2)``

    Gradient: ``2 * (x - target)``
    """

    def __init__(
        self,
        target: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self.target = target.astype(np.float64)
        self._bounds = bounds
        self.n_eval = 0
        self.history: list[float] = []
        self.forcefield = MagicMock()
        self.forcefield.get_param_vector.return_value = np.zeros_like(self.target)
        self.forcefield.get_active_param_vector.return_value = np.zeros_like(self.target)
        self.forcefield.get_bounds.return_value = self._bounds
        self.forcefield.get_active_bounds.return_value = (
            None if self._bounds is None else np.asarray(self._bounds, dtype=np.float64)
        )
        self.forcefield.active_mask = np.ones_like(self.target, dtype=bool)
        self.forcefield.n_params = int(self.target.size)
        self.forcefield.n_active_params = int(self.target.size)
        self.forcefield.set_param_vector = MagicMock()

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((x - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.target)


class MockDivergentObjective:
    """Objective that always returns increasing scores."""

    def __init__(self, n_params: int = 3) -> None:
        self._call_count = 0
        self.n_eval = 0
        self.history: list[float] = []
        self.forcefield = MagicMock()
        self.forcefield.get_param_vector.return_value = np.ones(n_params)
        self.forcefield.get_active_param_vector.return_value = np.ones(n_params)
        self.forcefield.get_bounds.return_value = None
        self.forcefield.get_active_bounds.return_value = None
        self.forcefield.active_mask = np.ones(n_params, dtype=bool)
        self.forcefield.n_params = n_params
        self.forcefield.n_active_params = n_params
        self.forcefield.set_param_vector = MagicMock()

    def __call__(self, x: np.ndarray) -> float:
        self._call_count += 1
        # First call returns 10.0, then escalates rapidly
        score = 10.0 * self._call_count
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x) * 100.0  # large gradient pushing params away


class MockFrozenForceField:
    """Mock force field exposing the active-parameter API."""

    def __init__(self) -> None:
        self._full = np.array([0.0, 5.0, 0.0], dtype=np.float64)
        self._mask = np.array([True, False, True])
        self._bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
        self.set_param_vector = MagicMock(side_effect=self._set_param_vector)

    @property
    def n_params(self) -> int:
        return int(self._full.size)

    @property
    def n_active_params(self) -> int:
        return int(self._mask.sum())

    @property
    def active_mask(self) -> np.ndarray:
        return self._mask.copy()

    def get_param_vector(self) -> np.ndarray:
        return self._full.copy()

    def get_active_param_vector(self) -> np.ndarray:
        return self._full[self._mask].copy()

    def get_bounds(self) -> list[tuple[float, float]]:
        return list(self._bounds)

    def get_active_bounds(self) -> np.ndarray:
        return np.asarray(self._bounds, dtype=np.float64)[self._mask]

    def _set_param_vector(self, vec: np.ndarray) -> None:
        self._full = np.asarray(vec, dtype=np.float64).copy()


class MockFrozenObjective:
    """Quadratic objective with one frozen full-vector coordinate."""

    def __init__(self) -> None:
        self.target = np.array([1.0, 4.0, 3.0], dtype=np.float64)
        self.n_eval = 0
        self.history: list[float] = []
        self.forcefield = MockFrozenForceField()
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = True

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=np.float64) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        return 2.0 * (x - self.target)


# ---- Tests ----


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
            opt.optimize(obj)


class TestOptaxConvergence:
    """Test that the optimizer converges on simple problems."""

    def test_adam_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.final_score < result.initial_score
        assert result.final_score < 0.01
        np.testing.assert_allclose(result.final_params, target, atol=0.1)

    def test_sgd_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, -1.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="sgd",
            learning_rate=0.1,
            momentum=0.9,
            max_steps=500,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.final_score < 0.01

    def test_adagrad_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([5.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adagrad",
            learning_rate=1.0,
            max_steps=500,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.final_score < 0.1

    def test_gradient_norm_convergence(self) -> None:
        """Optimizer should stop when gradient norm is small enough."""
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([0.0])  # start at 0, target at 0 → zero gradient
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=1000,
            grad_norm_tol=1e-4,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.success
        assert "gradient norm" in result.message

    def test_score_plateau_convergence(self) -> None:
        """Optimizer should stop when score plateaus."""
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.05,
            max_steps=2000,
            ftol=1e-8,
            patience=20,
            grad_norm_tol=1e-12,  # don't trigger grad norm convergence
            verbose=False,
        )
        result = opt.optimize(obj)

        # Should have converged via plateau or grad norm
        assert result.success
        assert result.n_iterations < 2000


class TestOptaxFrozenParams:
    """Frozen parameters remain fixed throughout optimization."""

    def test_adam_updates_only_active_params(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = MockFrozenObjective()
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=300,
            verbose=False,
        )
        result = opt.optimize(obj)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[~obj.forcefield.active_mask], [5.0])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.forcefield.get_param_vector(), result.final_params)


class TestOptaxBounds:
    """Test parameter bounds enforcement."""

    def test_bounds_enforced(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        # Target is at [10, 10] but bounds restrict to [0, 5]
        target = np.array([10.0, 10.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        obj = MockObjective(target, bounds=bounds)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=200,
            verbose=False,
        )
        result = opt.optimize(obj)

        # Final params should be at upper bound (closest feasible to target)
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
        result = opt.optimize(obj)

        # Should converge without bound constraints
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
        result = opt.optimize(obj)

        assert not result.success
        assert "Abandoned" in result.message
        assert result.n_iterations < 1000


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
        result = opt.optimize(obj)

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
        result = opt.optimize(obj)

        assert result.final_score < result.initial_score
        assert "exponential" in result.method


class TestOptaxResult:
    """Test OptimizationResult fields."""

    def test_result_fields(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=100,
            verbose=False,
        )
        result = opt.optimize(obj)

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
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=50,
            verbose=False,
        )
        result = opt.optimize(obj)

        # History should have one entry per step + initial
        assert len(result.history) >= 2
        # First entry should be the initial score
        assert result.history[0] == result.initial_score

    def test_forcefield_updated(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=100,
            verbose=False,
        )
        opt.optimize(obj)

        # set_param_vector should have been called with final params
        obj.forcefield.set_param_vector.assert_called_once()


class TestOptaxImport:
    """Test lazy import and registration."""

    def test_importable_from_package(self) -> None:
        from q2mm.optimizers import OptaxOptimizer

        assert OptaxOptimizer is not None

    def test_in_all(self) -> None:
        import q2mm.optimizers

        assert "OptaxOptimizer" in q2mm.optimizers.__all__
