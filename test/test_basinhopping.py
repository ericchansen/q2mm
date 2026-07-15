"""Tests for BasinHoppingOptimizer."""

from __future__ import annotations
from test.backend_fixtures import mock_backend_info

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.optimizers.basinhopping import BasinHoppingOptimizer, _BoundedStep


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
        full_bounds = bounds if bounds is not None else [(-100.0, 100.0)] * self.baseline.size
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


class MockObjective:
    """Quadratic objective: f(x) = sum((x - target)^2)."""

    def __init__(
        self,
        target: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self.target = target.astype(np.float64)
        baseline = np.zeros_like(self.target)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(self.target.size)
        self.space = MockSpace(baseline, bounds=bounds)
        self.n_eval = 0
        self.history: list[float] = []
        self.backend = MagicMock()
        self.backend.info = mock_backend_info(param_grad=False)

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=np.float64) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x, dtype=np.float64) - self.target)


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
        result = opt.optimize(obj, obj.space)
        np.testing.assert_allclose(result.final_params, target, atol=0.1)
        assert result.final_score < result.initial_score

    def test_returns_optimization_result(self) -> None:
        """Should return a proper OptimizationResult."""
        obj = MockObjective(np.array([1.0]))
        opt = BasinHoppingOptimizer(niter=5, verbose=False, seed=0)
        result = opt.optimize(obj, obj.space)
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
        result = opt.optimize(obj, obj.space)
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
        r1 = opt1.optimize(obj1, obj1.space)
        r2 = opt2.optimize(obj2, obj2.space)
        np.testing.assert_array_equal(r1.final_params, r2.final_params)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (usually) give different trajectories."""
        target = np.array([1.0, 2.0, 3.0])
        obj1 = MockObjective(target)
        obj2 = MockObjective(target)
        opt1 = BasinHoppingOptimizer(niter=5, verbose=False, seed=1)
        opt2 = BasinHoppingOptimizer(niter=5, verbose=False, seed=99)
        r1 = opt1.optimize(obj1, obj1.space)
        r2 = opt2.optimize(obj2, obj2.space)
        assert len(r1.history) != len(r2.history) or not np.allclose(r1.history, r2.history)

    def test_niter_controls_hops(self) -> None:
        """More hops should produce more evaluations."""
        target = np.array([1.0])
        obj_few = MockObjective(target)
        obj_many = MockObjective(target)
        r_few = BasinHoppingOptimizer(niter=2, verbose=False, seed=0).optimize(obj_few, obj_few.space)
        r_many = BasinHoppingOptimizer(niter=20, verbose=False, seed=0).optimize(obj_many, obj_many.space)
        assert r_many.n_evaluations > r_few.n_evaluations

    def test_does_not_mutate_forcefield(self) -> None:
        """Caller materializes the optimized ForceField explicitly."""
        obj = MockObjective(np.array([1.0, 2.0]))
        initial_ff = obj.forcefield
        opt = BasinHoppingOptimizer(niter=3, verbose=False, seed=42)
        result = opt.optimize(obj, obj.space)
        assert obj.forcefield is initial_ff
        np.testing.assert_array_equal(obj.layout.vector(obj.forcefield), np.zeros(2))
        final_ff = obj.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.layout.vector(final_ff), result.final_params)

    def test_summary(self) -> None:
        """OptimizationResult.summary() should work."""
        obj = MockObjective(np.array([1.0]))
        result = BasinHoppingOptimizer(niter=3, verbose=False, seed=0).optimize(obj, obj.space)
        summary = result.summary()
        assert "basinhopping" in summary
        assert "Score" in summary
