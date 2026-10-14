"""Tests for MultiStartOptimizer."""

from __future__ import annotations

from q2mm.settings import MultiStartSettings, ScipySettings

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult
from q2mm.optimizers.multistart import MultiStartOptimizer
from test._shared import QuadraticEvaluator


class StubOptimizer:
    """Optimizer stub that always returns the objective's current score."""

    def __init__(self) -> None:
        self.call_count = 0
        self.start_params: list[np.ndarray] = []

    def optimize(self, evaluator: QuadraticEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        self.call_count += 1
        x0 = np.asarray(space.baseline, dtype=np.float64).copy()
        self.start_params.append(x0.copy())
        score = evaluator.value(x0)
        return OptimizationResult(
            success=True,
            message="stub",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=x0,
            final_params=x0,
            history=[score],
            method="stub",
            gradient_mode=evaluator.gradient_mode.value,
        )


class TestMultiStartOptimizer:
    """Multi-start meta-optimizer tests."""

    def test_runs_n_starts(self) -> None:
        """Should call the inner optimizer n_starts times."""
        obj = QuadraticEvaluator(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=5, verbose=False, seed=42), optimizer=inner)
        opt.optimize(obj, obj.space)
        assert inner.call_count == 5

    def test_keeps_best_result(self) -> None:
        """Should return the lowest-scoring run."""
        target = np.array([1.0, 2.0, 3.0])
        obj = QuadraticEvaluator(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(ScipySettings(method="L-BFGS-B", maxiter=100, verbose=False))
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=42), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert result.final_score < 1.0

    def test_first_start_is_original(self) -> None:
        """First start should use original parameters (no perturbation)."""
        obj = QuadraticEvaluator(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=0), optimizer=inner)
        original_baseline = obj.space.baseline.copy()
        opt.optimize(obj, obj.space)
        np.testing.assert_array_equal(inner.start_params[0], np.array([0.0]))
        np.testing.assert_array_equal(obj.space.baseline, original_baseline)

    def test_perturbation_bounds(self) -> None:
        """Perturbed starts should respect parameter bounds."""
        bounds = [(0.0, 2.0), (0.0, 2.0)]
        obj = QuadraticEvaluator(np.array([1.0, 1.0]), bounds=bounds, initial=np.array([1.0, 1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(
            MultiStartSettings(n_starts=20, perturbation_pct=0.5, verbose=False, seed=42), optimizer=inner
        )
        opt.optimize(obj, obj.space)
        for params in inner.start_params:
            assert np.all(params >= 0.0 - 1e-10)
            assert np.all(params <= 2.0 + 1e-10)

    def test_seed_reproducibility(self) -> None:
        """Same seed should produce same result."""
        target = np.array([1.0, 2.0])
        obj1 = QuadraticEvaluator(target)
        obj2 = QuadraticEvaluator(target)
        inner1 = StubOptimizer()
        inner2 = StubOptimizer()
        r1 = MultiStartOptimizer(MultiStartSettings(n_starts=3, seed=99, verbose=False), optimizer=inner1).optimize(
            obj1, obj1.space
        )
        r2 = MultiStartOptimizer(MultiStartSettings(n_starts=3, seed=99, verbose=False), optimizer=inner2).optimize(
            obj2, obj2.space
        )
        assert r1.final_score == r2.final_score

    def test_method_name(self) -> None:
        """Result method should include 'multi-start'."""
        obj = QuadraticEvaluator(np.array([1.0]))
        inner = StubOptimizer()
        result = MultiStartOptimizer(MultiStartSettings(n_starts=2, verbose=False, seed=0), optimizer=inner).optimize(
            obj, obj.space
        )
        assert "multi-start" in result.method

    def test_returns_best_params_without_mutating_forcefield(self) -> None:
        """The caller materializes the best force field explicitly."""
        target = np.array([5.0])
        obj = QuadraticEvaluator(target)

        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        inner = ScipyOptimizer(ScipySettings(method="L-BFGS-B", maxiter=50, verbose=False))
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=42), optimizer=inner)
        original_baseline = obj.space.baseline.copy()
        result = opt.optimize(obj, obj.space)
        # REMOVED (Phase 4): objective is no longer mutated by optimizers.
        np.testing.assert_array_equal(obj.space.baseline, original_baseline)
        final_ff = obj.plan.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.plan.layout.vector(final_ff), result.final_params)

    def test_total_eval_count(self) -> None:
        """Evaluator n_evaluations should reflect all starts plus initial eval."""
        obj = QuadraticEvaluator(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=5, verbose=False, seed=42), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert len(result.candidates) == 5
        assert obj.n_evaluations == 6

    def test_best_history_returned(self) -> None:
        """History should come from the best run, not the last."""
        obj = QuadraticEvaluator(np.array([1.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=42), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert len(result.history) == 1
        assert result.history[0] == result.final_score

    def test_survives_failed_start(self) -> None:
        """Should skip failed starts and return best of successful ones."""

        class FailOnSecondOptimizer:
            def __init__(self) -> None:
                self.call_count = 0

            def optimize(self, evaluator: QuadraticEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
                self.call_count += 1
                if self.call_count == 2:
                    raise RuntimeError("Simulated failure")
                x0 = np.asarray(space.baseline, dtype=np.float64).copy()
                score = evaluator.value(x0)
                return OptimizationResult(
                    success=True,
                    message="ok",
                    initial_score=score,
                    final_score=score,
                    n_iterations=1,
                    n_evaluations=1,
                    n_params=space.n_full,
                    layout_fingerprint=space.layout.fingerprint,
                    initial_params=x0,
                    final_params=x0,
                    history=[score],
                    method="fail-test",
                    gradient_mode=evaluator.gradient_mode.value,
                )

        obj = QuadraticEvaluator(np.array([1.0]))
        inner = FailOnSecondOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=0), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert "2/3" in result.message

    def test_all_starts_fail_returns_failure_result(self) -> None:
        """Every start failing returns a canonical failure result (not raises).

        Phase 4: the multi-start optimizer never loses candidate records by
        raising — it returns ``OptimizationResult(success=False)`` whose
        candidates capture every failed start.
        """

        class AlwaysFailOptimizer:
            def optimize(self, evaluator: QuadraticEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
                raise RuntimeError("boom")

        obj = QuadraticEvaluator(np.array([1.0]))
        inner = AlwaysFailOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=0), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert result.success is False
        assert "all 3" in result.message
        assert result.final_score == float("inf")
        assert len(result.candidates) == 3
        assert all(c.status == "failure" for c in result.candidates)
        # Full-length baseline is returned even when everything failed.
        assert result.final_params.shape == (obj.plan.n_params,)
        np.testing.assert_array_equal(result.final_params, obj.space.baseline)

    def test_n_starts_zero_raises(self) -> None:
        """n_starts < 1 must raise ValueError."""
        with pytest.raises(ValueError, match="n_starts"):
            MultiStartOptimizer(MultiStartSettings(n_starts=0), optimizer=MagicMock())

    def test_negative_perturbation_raises(self) -> None:
        """Negative perturbation_pct must raise ValueError."""
        with pytest.raises(ValueError, match="perturbation_pct"):
            MultiStartOptimizer(MultiStartSettings(perturbation_pct=-0.1), optimizer=MagicMock())

    def test_initial_score_matches_original_params(self) -> None:
        """Initial_score must correspond to x0_original, not a perturbed start."""
        target = np.array([1.0, 2.0])
        obj = QuadraticEvaluator(target, initial=np.array([0.0, 0.0]))

        inner = StubOptimizer()
        opt = MultiStartOptimizer(
            MultiStartSettings(n_starts=3, perturbation_pct=0.5, verbose=False, seed=42), optimizer=inner
        )
        result = opt.optimize(obj, obj.space)

        assert result.initial_score == pytest.approx(5.0, abs=1e-10)
        np.testing.assert_array_equal(result.initial_params, np.array([0.0, 0.0]))

    # -- Failure / nonconvergence semantics --------------------------------

    @staticmethod
    def _scripted(outcomes: list[tuple[bool, float]]) -> object:
        """Inner optimizer replaying (success, final_score) per call."""

        class _ScriptedOptimizer:
            def __init__(self) -> None:
                self.call = 0

            def optimize(self, evaluator: QuadraticEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
                success, score = outcomes[self.call]
                self.call += 1
                x0 = np.asarray(space.baseline, dtype=float).copy()
                return OptimizationResult(
                    success=success,
                    message="ok" if success else "nonconverged",
                    initial_score=float(evaluator.value(x0)),
                    final_score=score,
                    n_iterations=1,
                    n_evaluations=1,
                    n_params=space.n_full,
                    layout_fingerprint=space.layout.fingerprint,
                    initial_params=x0,
                    final_params=x0,
                    history=[score],
                    method="scripted",
                    gradient_mode=evaluator.gradient_mode.value,
                )

        return _ScriptedOptimizer()

    def test_nonconverged_inner_is_failure_candidate(self) -> None:
        """A ran-but-nonconverged start is a failure candidate; a converged one wins."""
        obj = QuadraticEvaluator(np.array([1.0]))
        # converged 5.0, nonconverged 2.0 (lower!), converged 3.0.
        inner = self._scripted([(True, 5.0), (False, 2.0), (True, 3.0)])
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=1), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        # Converged is preferred even though a nonconverged start scored lower.
        assert result.success is True
        assert result.final_score == pytest.approx(3.0)
        statuses = [c.status for c in result.candidates]
        assert statuses.count("success") == 2
        assert statuses.count("failure") == 1
        # The nonconverged candidate keeps its finite score.
        failed = next(c for c in result.candidates if c.status == "failure")
        assert failed.final_score == pytest.approx(2.0)

    def test_all_nonconverged_returns_unsuccessful_best(self) -> None:
        """All starts ran but none converged -> success=False, best kept."""
        obj = QuadraticEvaluator(np.array([1.0]))
        inner = self._scripted([(False, 5.0), (False, 2.0), (False, 3.0)])
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=1), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert result.success is False
        assert result.final_score == pytest.approx(2.0)  # best nonconverged
        assert all(c.status == "failure" for c in result.candidates)
        assert len(result.candidates) == 3

    def test_best_selection_prefers_lowest_converged(self) -> None:
        """Among converged starts, the lowest final_score is selected."""
        obj = QuadraticEvaluator(np.array([1.0]))
        inner = self._scripted([(True, 5.0), (True, 1.0), (True, 3.0)])
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=3, verbose=False, seed=1), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        assert result.success is True
        assert result.final_score == pytest.approx(1.0)
        winner = min(result.candidates, key=lambda c: c.final_score)
        np.testing.assert_array_equal(result.final_params, winner.final_params)

    def test_candidate_vectors_are_readonly_and_full_length(self) -> None:
        """Every candidate carries full-length read-only start/final vectors."""
        obj = QuadraticEvaluator(np.array([1.0, 2.0]))
        inner = StubOptimizer()
        opt = MultiStartOptimizer(MultiStartSettings(n_starts=4, verbose=False, seed=7), optimizer=inner)
        result = opt.optimize(obj, obj.space)
        n = obj.plan.n_params
        assert len(result.candidates) == 4
        for cand in result.candidates:
            assert cand.initial_params.shape == (n,)
            assert cand.final_params.shape == (n,)
            assert cand.initial_params.flags.writeable is False
            assert cand.final_params.flags.writeable is False
            assert cand.index >= 0

    def test_deterministic_candidate_vectors_and_seeds(self) -> None:
        """Same seed -> identical candidate start/final vectors and seed metadata."""
        target = np.array([1.0, 2.0])
        opt_a = MultiStartOptimizer(
            MultiStartSettings(n_starts=5, perturbation_pct=0.3, verbose=False, seed=99), optimizer=StubOptimizer()
        )
        opt_b = MultiStartOptimizer(
            MultiStartSettings(n_starts=5, perturbation_pct=0.3, verbose=False, seed=99), optimizer=StubOptimizer()
        )
        obj_a = QuadraticEvaluator(target)
        obj_b = QuadraticEvaluator(target)
        r_a = opt_a.optimize(obj_a, obj_a.space)
        r_b = opt_b.optimize(obj_b, obj_b.space)
        assert [c.seed for c in r_a.candidates] == [99] * 5
        for ca, cb in zip(r_a.candidates, r_b.candidates, strict=True):
            np.testing.assert_array_equal(ca.initial_params, cb.initial_params)
            np.testing.assert_array_equal(ca.final_params, cb.final_params)
            assert ca.index == cb.index
