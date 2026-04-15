"""Tests for MultiStartOptimizer."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.scipy_opt import OptimizationResult


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


class StubOptimizer:
    """Optimizer stub that always returns the objective's current score."""

    def __init__(self) -> None:
        self.call_count = 0

    def optimize(self, objective: MockObjective) -> OptimizationResult:
        self.call_count += 1
        x0 = objective.forcefield.get_param_vector().copy()
        score = objective(x0)
        return OptimizationResult(
            success=True,
            message="stub",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            initial_params=x0,
            final_params=x0,
            history=[score],
            method="stub",
        )


class TestMultiStartOptimizer:
    """Multi-start meta-optimizer tests."""

    def test_runs_n_starts(self) -> None:
        """Should call the inner optimizer n_starts times."""
        obj = MockObjective(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=5, verbose=False, seed=42)
        opt.optimize(obj)
        assert inner.call_count == 5

    def test_keeps_best_result(self) -> None:
        """Should return the lowest-scoring run."""
        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        result = opt.optimize(obj)
        # Best should have found a score close to 0
        assert result.final_score < 1.0

    def test_first_start_is_original(self) -> None:
        """First start should use original parameters (no perturbation)."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=0)
        opt.optimize(obj)
        # First call should have used x0 = [0.0] (from get_param_vector)
        first_call_params = obj.forcefield.set_param_vector.call_args_list[0][0][0]
        np.testing.assert_array_equal(first_call_params, np.array([0.0]))

    def test_perturbation_bounds(self) -> None:
        """Perturbed starts should respect parameter bounds."""
        bounds = [(0.0, 2.0), (0.0, 2.0)]
        obj = MockObjective(np.array([1.0, 1.0]), bounds=bounds)
        obj.forcefield.get_param_vector.return_value = np.array([1.0, 1.0])
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=20, perturbation_pct=0.5, verbose=False, seed=42)
        opt.optimize(obj)
        # All set_param_vector calls should be within bounds
        for call in obj.forcefield.set_param_vector.call_args_list:
            params = call[0][0]
            assert np.all(params >= 0.0 - 1e-10)
            assert np.all(params <= 2.0 + 1e-10)

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same result."""
        target = np.array([1.0, 2.0])
        obj1 = MockObjective(target)
        obj2 = MockObjective(target)
        inner1 = StubOptimizer()
        inner2 = StubOptimizer()
        r1 = MultiStartOptimizer(inner1, n_starts=3, seed=99, verbose=False).optimize(obj1)
        r2 = MultiStartOptimizer(inner2, n_starts=3, seed=99, verbose=False).optimize(obj2)
        assert r1.final_score == r2.final_score

    def test_method_name(self) -> None:
        """Result method should include 'multi-start'."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        result = MultiStartOptimizer(inner, n_starts=2, verbose=False, seed=0).optimize(obj)
        assert "multi-start" in result.method

    def test_applies_best_params(self) -> None:
        """Should set forcefield to the best params at the end."""
        target = np.array([5.0])
        obj = MockObjective(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False)
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        result = opt.optimize(obj)
        # Last set_param_vector call should be the best params
        final_call = obj.forcefield.set_param_vector.call_args_list[-1][0][0]
        np.testing.assert_array_equal(final_call, result.final_params)

    def test_total_eval_count(self) -> None:
        """n_evaluations should reflect all starts plus initial eval."""
        obj = MockObjective(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=5, verbose=False, seed=42)
        result = opt.optimize(obj)
        # 1 upfront eval for initial_score + 5 per-start evals = 6
        assert result.n_evaluations == 6

    def test_best_history_returned(self) -> None:
        """History should come from the best run, not the last."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        result = opt.optimize(obj)
        # History should have exactly 1 entry (from best run's stub)
        assert len(result.history) == 1
        assert result.history[0] == result.final_score

    def test_survives_failed_start(self) -> None:
        """Should skip failed starts and return best of successful ones."""

        class FailOnSecondOptimizer:
            def __init__(self) -> None:
                self.call_count = 0

            def optimize(self, objective: MockObjective) -> OptimizationResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Simulated failure")
                x0 = objective.forcefield.get_param_vector().copy()
                score = objective(x0)
                return OptimizationResult(
                    success=True,
                    message="ok",
                    initial_score=score,
                    final_score=score,
                    n_iterations=1,
                    n_evaluations=1,
                    initial_params=x0,
                    final_params=x0,
                    history=[score],
                    method="fail-test",
                )

        obj = MockObjective(np.array([1.0]))
        inner = FailOnSecondOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=0)
        result = opt.optimize(obj)
        # Should succeed — 2 of 3 starts completed
        assert "2/3" in result.message

    def test_all_starts_fail_raises(self) -> None:
        """Should raise RuntimeError if every start fails."""

        class AlwaysFailOptimizer:
            def optimize(self, objective: MockObjective) -> OptimizationResult:
                raise RuntimeError("boom")

        obj = MockObjective(np.array([1.0]))
        inner = AlwaysFailOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=0)
        with pytest.raises(RuntimeError, match="All 3 multi-start runs failed"):
            opt.optimize(obj)

    def test_n_starts_zero_raises(self) -> None:
        """n_starts < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="n_starts must be >= 1"):
            MultiStartOptimizer(MagicMock(), n_starts=0)

    def test_negative_perturbation_raises(self) -> None:
        """Negative perturbation_pct must raise ValueError."""
        with pytest.raises(ValueError, match="perturbation_pct must be >= 0"):
            MultiStartOptimizer(MagicMock(), perturbation_pct=-0.1)

    def test_initial_score_matches_original_params(self) -> None:
        """Initial_score must correspond to x0_original, not a perturbed start."""
        target = np.array([1.0, 2.0])
        obj = MockObjective(target)
        obj.forcefield.get_param_vector.return_value = np.array([0.0, 0.0])

        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, perturbation_pct=0.5, verbose=False, seed=42)
        result = opt.optimize(obj)

        # initial_score should be obj(x0_original) = (0-1)^2 + (0-2)^2 = 5.0
        assert result.initial_score == pytest.approx(5.0, abs=1e-10)
        np.testing.assert_array_equal(result.initial_params, np.array([0.0, 0.0]))
