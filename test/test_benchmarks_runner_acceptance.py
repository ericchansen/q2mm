"""Tests for :mod:`q2mm.benchmarks.acceptance`.

Cover the closed status vocabulary, the single no-progress decision, the
non-finite / worsening rejections, and numeric validation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from q2mm.benchmarks.acceptance import (
    AcceptancePolicy,
    CandidateStatus,
    NoProgressPolicy,
    improvement_percent,
)


class TestCandidateStatus:
    def test_closed_vocabulary(self) -> None:
        assert {s.value for s in CandidateStatus} == {"accepted", "rejected", "skipped", "error"}


class TestImprovementPercent:
    def test_matches_fractional_convention(self) -> None:
        assert improvement_percent(100.0, 50.0) == pytest.approx(50.0)
        assert improvement_percent(100.0, 100.0) == pytest.approx(0.0)
        assert improvement_percent(0.0, 5.0) == pytest.approx(0.0)


class TestNoProgressPolicy:
    def test_progress_when_many_iterations(self) -> None:
        assert NoProgressPolicy().made_progress(n_iterations=10, improvement_pct=0.0) is True

    def test_no_progress_when_stalled(self) -> None:
        assert NoProgressPolicy().made_progress(n_iterations=1, improvement_pct=0.5) is False

    def test_progress_when_big_improvement_despite_few_iterations(self) -> None:
        assert NoProgressPolicy().made_progress(n_iterations=1, improvement_pct=40.0) is True

    def test_negative_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoProgressPolicy(max_iterations=-1)
        with pytest.raises(ValueError):
            NoProgressPolicy(min_improvement_pct=-1.0)


class TestAcceptancePolicy:
    def test_accept_on_progress(self) -> None:
        decision = AcceptancePolicy().evaluate(n_iterations=10, initial_score=100.0, final_score=75.0, converged=True)
        assert decision.status is CandidateStatus.ACCEPTED and decision.is_accepted

    def test_reject_on_no_progress(self) -> None:
        decision = AcceptancePolicy().evaluate(n_iterations=1, initial_score=100.0, final_score=99.9, converged=True)
        assert decision.status is CandidateStatus.REJECTED

    def test_reject_worsened_run_even_with_many_iterations(self) -> None:
        # A 50% WORSENED run has |improvement|=50 (> min) but must be rejected.
        decision = AcceptancePolicy().evaluate(n_iterations=50, initial_score=100.0, final_score=150.0, converged=True)
        assert decision.status is CandidateStatus.REJECTED
        assert "worsened" in decision.reason

    def test_reject_non_finite_scores(self) -> None:
        for final in (float("nan"), float("inf")):
            decision = AcceptancePolicy().evaluate(
                n_iterations=50, initial_score=100.0, final_score=final, converged=True
            )
            assert decision.status is CandidateStatus.REJECTED
            assert "non-finite" in decision.reason

    def test_worsening_tolerance_absorbs_tiny_noise(self) -> None:
        policy = AcceptancePolicy(worsening_tolerance_pct=1e-6)
        decision = policy.evaluate(n_iterations=50, initial_score=100.0, final_score=100.0 + 1e-7, converged=True)
        assert decision.status is CandidateStatus.ACCEPTED

    def test_require_convergence_rejects_unconverged(self) -> None:
        policy = AcceptancePolicy(require_convergence=True)
        decision = policy.evaluate(n_iterations=50, initial_score=100.0, final_score=70.0, converged=False)
        assert decision.status is CandidateStatus.REJECTED

    def test_default_accepts_unconverged_progress(self) -> None:
        decision = AcceptancePolicy().evaluate(n_iterations=50, initial_score=100.0, final_score=70.0, converged=False)
        assert decision.status is CandidateStatus.ACCEPTED

    def test_reject_zero_baseline_worsening(self) -> None:
        # 0 -> positive is a real worsening even though percent improvement is 0.
        decision = AcceptancePolicy().evaluate(n_iterations=50, initial_score=0.0, final_score=5.0, converged=True)
        assert decision.status is CandidateStatus.REJECTED
        assert "zero baseline" in decision.reason

    def test_accept_zero_baseline_zero_final(self) -> None:
        # 0 -> 0 is not a worsening; the perfect objective is preserved.
        decision = AcceptancePolicy().evaluate(n_iterations=50, initial_score=0.0, final_score=0.0, converged=True)
        assert decision.status is CandidateStatus.ACCEPTED

    def test_reject_negative_objective_score(self) -> None:
        decision = AcceptancePolicy().evaluate(n_iterations=50, initial_score=1.0, final_score=-0.5, converged=True)
        assert decision.status is CandidateStatus.REJECTED
        assert "negative" in decision.reason

    def test_negative_iteration_count_rejected(self) -> None:
        with pytest.raises(ValueError):
            AcceptancePolicy().evaluate(n_iterations=-1, initial_score=1.0, final_score=0.5, converged=True)

    def test_skip_and_error_builders(self) -> None:
        assert AcceptancePolicy.skipped("no data").status is CandidateStatus.SKIPPED
        assert AcceptancePolicy.errored("boom").status is CandidateStatus.ERROR

    def test_bad_worsening_tolerance_rejected(self) -> None:
        with pytest.raises(ValueError):
            AcceptancePolicy(worsening_tolerance_pct=-1.0)


def test_single_no_progress_decision_source() -> None:
    """Acceptance is the only caller of NoProgressPolicy.made_progress."""
    root = Path(__file__).resolve().parent.parent / "q2mm" / "benchmarks"

    def _calls(path: Path) -> int:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return sum(1 for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "made_progress")

    assert _calls(root / "acceptance.py") == 1
    assert _calls(root / "runner.py") == 0
