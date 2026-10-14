"""Tests for BasinHoppingOptimizer."""

from __future__ import annotations

from q2mm.settings import BasinHoppingSettings

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.optimizers.basinhopping import BasinHoppingOptimizer, _BoundedStep
from test._shared import QuadraticEvaluator


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
        obj = QuadraticEvaluator(target)
        opt = BasinHoppingOptimizer(BasinHoppingSettings(niter=10, verbose=False, seed=42))
        result = opt.optimize(obj, obj.space)
        np.testing.assert_allclose(result.final_params, target, atol=0.1)
        assert result.final_score < result.initial_score

    def test_returns_optimization_result(self) -> None:
        """Should return a proper OptimizationResult."""
        obj = QuadraticEvaluator(np.array([1.0]))
        opt = BasinHoppingOptimizer(BasinHoppingSettings(niter=5, verbose=False, seed=0))
        result = opt.optimize(obj, obj.space)
        assert hasattr(result, "success")
        assert hasattr(result, "final_params")
        assert hasattr(result, "history")
        assert result.method.startswith("basinhopping")

    def test_respects_bounds(self) -> None:
        """Final params should stay within bounds."""
        bounds = [(-5.0, 5.0), (-5.0, 5.0)]
        target = np.array([10.0, 10.0])  # outside bounds
        obj = QuadraticEvaluator(target, bounds=bounds)
        opt = BasinHoppingOptimizer(BasinHoppingSettings(niter=10, verbose=False, seed=42))
        result = opt.optimize(obj, obj.space)
        for i, (lo, hi) in enumerate(bounds):
            assert result.final_params[i] >= lo - 1e-10
            assert result.final_params[i] <= hi + 1e-10

    def test_seed_reproducibility(self) -> None:
        """Same seed should give same result."""
        target = np.array([1.0, 2.0])
        obj1 = QuadraticEvaluator(target)
        obj2 = QuadraticEvaluator(target)
        opt1 = BasinHoppingOptimizer(BasinHoppingSettings(niter=5, verbose=False, seed=123))
        opt2 = BasinHoppingOptimizer(BasinHoppingSettings(niter=5, verbose=False, seed=123))
        r1 = opt1.optimize(obj1, obj1.space)
        r2 = opt2.optimize(obj2, obj2.space)
        np.testing.assert_array_equal(r1.final_params, r2.final_params)

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (usually) give different trajectories."""
        target = np.array([1.0, 2.0, 3.0])
        obj1 = QuadraticEvaluator(target)
        obj2 = QuadraticEvaluator(target)
        opt1 = BasinHoppingOptimizer(BasinHoppingSettings(niter=5, verbose=False, seed=1))
        opt2 = BasinHoppingOptimizer(BasinHoppingSettings(niter=5, verbose=False, seed=99))
        r1 = opt1.optimize(obj1, obj1.space)
        r2 = opt2.optimize(obj2, obj2.space)
        assert len(r1.history) != len(r2.history) or not np.allclose(r1.history, r2.history)

    def test_niter_controls_hops(self) -> None:
        """More hops should produce more evaluations."""
        target = np.array([1.0])
        obj_few = QuadraticEvaluator(target)
        obj_many = QuadraticEvaluator(target)
        r_few = BasinHoppingOptimizer(BasinHoppingSettings(niter=2, verbose=False, seed=0)).optimize(
            obj_few, obj_few.space
        )
        r_many = BasinHoppingOptimizer(BasinHoppingSettings(niter=20, verbose=False, seed=0)).optimize(
            obj_many, obj_many.space
        )
        assert r_many.n_evaluations > r_few.n_evaluations

    def test_does_not_mutate_forcefield(self) -> None:
        """Caller materializes the optimized ForceField explicitly."""
        obj = QuadraticEvaluator(np.array([1.0, 2.0]))
        initial_baseline = obj.space.baseline.copy()
        opt = BasinHoppingOptimizer(BasinHoppingSettings(niter=3, verbose=False, seed=42))
        result = opt.optimize(obj, obj.space)
        # REMOVED (Phase 4): objective is no longer mutated by optimizers.
        np.testing.assert_array_equal(obj.space.baseline, initial_baseline)
        final_ff = obj.plan.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.plan.layout.vector(final_ff), result.final_params)

    def test_summary(self) -> None:
        """OptimizationResult.summary() should work."""
        obj = QuadraticEvaluator(np.array([1.0]))
        result = BasinHoppingOptimizer(BasinHoppingSettings(niter=3, verbose=False, seed=0)).optimize(obj, obj.space)
        summary = result.summary()
        assert "basinhopping" in summary
        assert "Score" in summary
