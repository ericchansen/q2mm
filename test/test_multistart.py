"""Tests for MultiStartOptimizer."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.optimizers.multistart import MultiStartOptimizer
from q2mm.optimizers.scipy_opt import OptimizationResult


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

    def with_baseline(self, vector: np.ndarray) -> MockSpace:
        return MockSpace(
            baseline=np.asarray(vector, dtype=np.float64),
            bounds=self._full_bounds.tolist(),
            active_indices=self.active_indices.copy(),
        )


class MockObjective:
    """Quadratic objective: f(x) = sum((x - target)^2)."""

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
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = False

    def __call__(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=np.float64) - self.target) ** 2))
        self.n_eval += 1
        self.history.append(score)
        return score

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(x, dtype=np.float64) - self.target)


class StubOptimizer:
    """Optimizer stub that always returns the objective's current score."""

    def __init__(self) -> None:
        self.call_count = 0
        self.start_params: list[np.ndarray] = []

    def optimize(self, objective: MockObjective, space: MockSpace) -> OptimizationResult:
        self.call_count += 1
        x0 = objective.layout.vector(objective.forcefield).copy()
        self.start_params.append(x0.copy())
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
        opt.optimize(obj, obj.space)
        assert inner.call_count == 5

    def test_keeps_best_result(self) -> None:
        """Should return the lowest-scoring run."""
        target = np.array([1.0, 2.0, 3.0])
        obj = MockObjective(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        result = opt.optimize(obj, obj.space)
        assert result.final_score < 1.0

    def test_first_start_is_original(self) -> None:
        """First start should use original parameters (no perturbation)."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=0)
        original_ff = obj.forcefield
        opt.optimize(obj, obj.space)
        np.testing.assert_array_equal(inner.start_params[0], np.array([0.0]))
        assert obj.forcefield is original_ff

    def test_perturbation_bounds(self) -> None:
        """Perturbed starts should respect parameter bounds."""
        bounds = [(0.0, 2.0), (0.0, 2.0)]
        obj = MockObjective(np.array([1.0, 1.0]), bounds=bounds, initial=np.array([1.0, 1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=20, perturbation_pct=0.5, verbose=False, seed=42)
        opt.optimize(obj, obj.space)
        for params in inner.start_params:
            assert np.all(params >= 0.0 - 1e-10)
            assert np.all(params <= 2.0 + 1e-10)

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same result."""
        target = np.array([1.0, 2.0])
        obj1 = MockObjective(target)
        obj2 = MockObjective(target)
        inner1 = StubOptimizer()
        inner2 = StubOptimizer()
        r1 = MultiStartOptimizer(inner1, n_starts=3, seed=99, verbose=False).optimize(obj1, obj1.space)
        r2 = MultiStartOptimizer(inner2, n_starts=3, seed=99, verbose=False).optimize(obj2, obj2.space)
        assert r1.final_score == r2.final_score

    def test_method_name(self) -> None:
        """Result method should include 'multi-start'."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        result = MultiStartOptimizer(inner, n_starts=2, verbose=False, seed=0).optimize(obj, obj.space)
        assert "multi-start" in result.method

    def test_returns_best_params_without_mutating_forcefield(self) -> None:
        """The caller materializes the best force field explicitly."""
        target = np.array([5.0])
        obj = MockObjective(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False)
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        original_ff = obj.forcefield
        result = opt.optimize(obj, obj.space)
        assert obj.forcefield is original_ff
        np.testing.assert_array_equal(obj.layout.vector(obj.forcefield), np.array([0.0]))
        final_ff = obj.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.layout.vector(final_ff), result.final_params)

    def test_total_eval_count(self) -> None:
        """n_evaluations should reflect all starts plus initial eval."""
        obj = MockObjective(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=5, verbose=False, seed=42)
        result = opt.optimize(obj, obj.space)
        assert result.n_evaluations == 6

    def test_best_history_returned(self) -> None:
        """History should come from the best run, not the last."""
        obj = MockObjective(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=42)
        result = opt.optimize(obj, obj.space)
        assert len(result.history) == 1
        assert result.history[0] == result.final_score

    def test_survives_failed_start(self) -> None:
        """Should skip failed starts and return best of successful ones."""

        class FailOnSecondOptimizer:
            def __init__(self) -> None:
                self.call_count = 0

            def optimize(self, objective: MockObjective, space: MockSpace) -> OptimizationResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Simulated failure")
                x0 = objective.layout.vector(objective.forcefield).copy()
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
        result = opt.optimize(obj, obj.space)
        assert "2/3" in result.message

    def test_all_starts_fail_raises(self) -> None:
        """Should raise RuntimeError if every start fails."""

        class AlwaysFailOptimizer:
            def optimize(self, objective: MockObjective, space: MockSpace) -> OptimizationResult:
                raise RuntimeError("boom")

        obj = MockObjective(np.array([1.0]))
        inner = AlwaysFailOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, verbose=False, seed=0)
        with pytest.raises(RuntimeError, match="All 3 multi-start runs failed"):
            opt.optimize(obj, obj.space)

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
        obj = MockObjective(target, initial=np.array([0.0, 0.0]))

        inner = StubOptimizer()
        opt = MultiStartOptimizer(inner, n_starts=3, perturbation_pct=0.5, verbose=False, seed=42)
        result = opt.optimize(obj, obj.space)

        assert result.initial_score == pytest.approx(5.0, abs=1e-10)
        np.testing.assert_array_equal(result.initial_params, np.array([0.0, 0.0]))
