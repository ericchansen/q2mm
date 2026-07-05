"""Tests for BasinHoppingOptimizer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.optimizers.basinhopping import BasinHoppingOptimizer, _BoundedStep


class MockObjective:
    """Quadratic objective: f(x) = sum((x - target)^2)."""

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
        self.forcefield.get_bounds.return_value = self._bounds
        self.forcefield.set_param_vector = MagicMock()
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = False

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((x - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (x - self.target)


class TestBoundedStep:
    """Test the bounded perturbation step."""

    def test_respects_bounds(self) -> None:
        bounds = [(0.0, 1.0), (0.0, 1.0)]
        rng = np.random.default_rng(42)
        step = _BoundedStep(stepsize=10.0, bounds=bounds, rng=rng)
        x = np.array([0.5, 0.5])
        for _ in range(100):
            x_new = step(x)
            assert np.all(x_new >= 0.0)
            assert np.all(x_new <= 1.0)

    def test_no_bounds(self) -> None:
        rng = np.random.default_rng(42)
        step = _BoundedStep(stepsize=1.0, bounds=None, rng=rng)
        x = np.array([0.0, 0.0])
        x_new = step(x)
        assert x_new.shape == x.shape


class TestBasinHoppingOptimizer:
    """Basin-hopping optimizer tests."""

    def test_converges_on_quadratic(self) -> None:
        """Should find the global minimum of a simple quadratic."""
        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)
        opt = BasinHoppingOptimizer(niter=10, verbose=False, seed=42)
        result = opt.optimize(obj)
        np.testing.assert_allclose(result.final_params, target, atol=0.1)
        assert result.final_score < result.initial_score

    def test_returns_optimization_result(self) -> None:
        """Should return a proper OptimizationResult."""
        obj = MockObjective(np.array([1.0]))
        opt = BasinHoppingOptimizer(niter=5, verbose=False, seed=0)
        result = opt.optimize(obj)
        assert hasattr(result, "success")
        assert hasattr(result, "final_params")
        assert hasattr(result, "history")
        assert result.method.startswith("basinhopping")

    def test_respects_bounds(self) -> None:
        """Final params should stay within bounds."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        target = np.array([10.0, 10.0])  # outside bounds
        obj = MockObjective(target, bounds=bounds)
        opt = BasinHoppingOptimizer(niter=10, verbose=False, seed=42)
        result = opt.optimize(obj)
        for i, (lo, hi) in enumerate(bounds):
            assert result.final_params[i] >= lo - 1e-10
            assert result.final_params[i] <= hi + 1e-10

    def test_seed_reproducibility(self) -> None:
        """Same seed should give same result."""
        target = np.array([1.0, 2.0])
        obj1 = MockObjective(target)
        obj2 = MockObjective(target)
        opt1 = BasinHoppingOptimizer(niter=5, verbose=False, seed=123)
        opt2 = BasinHoppingOptimizer(niter=5, verbose=False, seed=123)
        r1 = opt1.optimize(obj1)
        r2 = opt2.optimize(obj2)
        np.testing.assert_array_equal(r1.final_params, r2.final_params)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (usually) give different trajectories."""
        target = np.array([1.0, 2.0, 3.0])
        obj1 = MockObjective(target)
        obj2 = MockObjective(target)
        opt1 = BasinHoppingOptimizer(niter=5, verbose=False, seed=1)
        opt2 = BasinHoppingOptimizer(niter=5, verbose=False, seed=99)
        r1 = opt1.optimize(obj1)
        r2 = opt2.optimize(obj2)
        # Trajectories should differ even if both converge
        assert len(r1.history) != len(r2.history) or not np.allclose(r1.history, r2.history)

    def test_niter_controls_hops(self) -> None:
        """More hops should produce more evaluations."""
        target = np.array([1.0])
        obj_few = MockObjective(target)
        obj_many = MockObjective(target)
        r_few = BasinHoppingOptimizer(niter=2, verbose=False, seed=0).optimize(obj_few)
        r_many = BasinHoppingOptimizer(niter=20, verbose=False, seed=0).optimize(obj_many)
        assert r_many.n_evaluations > r_few.n_evaluations

    def test_applies_final_params(self) -> None:
        """Should call forcefield.set_param_vector with final params."""
        obj = MockObjective(np.array([1.0, 2.0]))
        opt = BasinHoppingOptimizer(niter=3, verbose=False, seed=42)
        result = opt.optimize(obj)
        obj.forcefield.set_param_vector.assert_called_once()
        call_args = obj.forcefield.set_param_vector.call_args[0][0]
        np.testing.assert_array_equal(call_args, result.final_params)

    def test_summary(self) -> None:
        """OptimizationResult.summary() should work."""
        obj = MockObjective(np.array([1.0]))
        result = BasinHoppingOptimizer(niter=3, verbose=False, seed=0).optimize(obj)
        summary = result.summary()
        assert "basinhopping" in summary
        assert "Score" in summary
