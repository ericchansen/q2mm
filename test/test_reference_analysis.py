"""Tests for q2mm.diagnostics.reference_analysis (issue #122)."""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.diagnostics.reference_analysis import (
    EigenvalueAnalysis,
    FrequencyComparison,
    ModeCouplingAnalysis,
    SymmetryCheck,
    analyze_eigenvalues,
    analyze_mode_coupling,
    check_symmetry,
    compare_frequencies,
    format_hessian_report,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def symmetric_3x3() -> np.ndarray:
    """Return a 3×3 symmetric matrix with known eigenvalues [-5.51, 3.66, 5.85]."""
    return np.array([[-2.0, -4.0, 2.0], [-4.0, 1.0, 2.0], [2.0, 2.0, 5.0]], dtype=float)


@pytest.fixture()
def minimum_hessian_6x6() -> np.ndarray:
    """Return a 6×6 H₂ minimum Hessian with no negative eigenvalues."""
    k = 0.5
    hess = np.zeros((6, 6))
    hess[2, 2] = k
    hess[5, 5] = k
    hess[2, 5] = -k
    hess[5, 2] = -k
    return hess


# ---------------------------------------------------------------------------
# analyze_eigenvalues
# ---------------------------------------------------------------------------


class TestAnalyzeEigenvalues:
    def test_ts_one_negative(self, symmetric_3x3: np.ndarray) -> None:
        """Matrix with one negative eigenvalue recognized as valid TS."""
        result = analyze_eigenvalues(symmetric_3x3, is_transition_state=True)
        assert isinstance(result, EigenvalueAnalysis)
        assert result.n_negative == 1
        assert result.is_consistent is True
        assert result.expected_negatives == 1
        assert len(result.negative_values) == 1
        assert result.negative_values[0] < 0

    def test_minimum_no_negatives(self, minimum_hessian_6x6: np.ndarray) -> None:
        """Positive-definite Hessian recognized as valid minimum."""
        result = analyze_eigenvalues(minimum_hessian_6x6, is_transition_state=False)
        assert result.n_negative == 0
        assert result.is_consistent is True

    def test_ts_flag_mismatch(self, minimum_hessian_6x6: np.ndarray) -> None:
        """Positive-definite Hessian flagged as inconsistent when TS expected."""
        result = analyze_eigenvalues(minimum_hessian_6x6, is_transition_state=True)
        assert result.is_consistent is False
        assert result.expected_negatives == 1
        assert result.n_negative == 0

    def test_condition_number_positive(self, symmetric_3x3: np.ndarray) -> None:
        """Condition number computed from positive eigenvalues."""
        result = analyze_eigenvalues(symmetric_3x3)
        assert result.condition_number > 1.0
        assert np.isfinite(result.condition_number)

    def test_condition_number_identity(self) -> None:
        """Identity matrix has condition number 1."""
        result = analyze_eigenvalues(np.eye(4))
        assert result.condition_number == pytest.approx(1.0, abs=1e-10)

    def test_near_zero_count(self) -> None:
        """Near-zero eigenvalues are counted correctly."""
        # Diagonal matrix with some near-zeros
        hess = np.diag([0.0, 1e-8, 1e-5, 0.5, 1.0])
        result = analyze_eigenvalues(hess, is_transition_state=False, zero_tol=1e-6)
        assert result.n_zero == 2  # 0.0 and 1e-8

    def test_eigenvalues_sorted(self, symmetric_3x3: np.ndarray) -> None:
        """Returned eigenvalues are sorted ascending."""
        result = analyze_eigenvalues(symmetric_3x3)
        for i in range(len(result.eigenvalues) - 1):
            assert result.eigenvalues[i] <= result.eigenvalues[i + 1]

    def test_non_square_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        with pytest.raises(ValueError, match="square"):
            analyze_eigenvalues(np.zeros((3, 4)))


# ---------------------------------------------------------------------------
# check_symmetry
# ---------------------------------------------------------------------------


class TestCheckSymmetry:
    def test_symmetric_passes(self, symmetric_3x3: np.ndarray) -> None:
        result = check_symmetry(symmetric_3x3)
        assert isinstance(result, SymmetryCheck)
        assert result.is_symmetric is True
        assert result.max_deviation == pytest.approx(0.0, abs=1e-15)

    def test_asymmetric_fails(self) -> None:
        hess = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = check_symmetry(hess)
        assert result.is_symmetric is False
        assert result.max_deviation == pytest.approx(1.0)

    def test_near_symmetric_with_tolerance(self) -> None:
        hess = np.array([[1.0, 2.0], [2.0 + 1e-10, 3.0]])
        result = check_symmetry(hess, tolerance=1e-8)
        assert result.is_symmetric is True

    def test_near_symmetric_tight_tolerance(self) -> None:
        hess = np.array([[1.0, 2.0], [2.0 + 1e-10, 3.0]])
        result = check_symmetry(hess, tolerance=1e-12)
        assert result.is_symmetric is False

    def test_identity_symmetric(self) -> None:
        result = check_symmetry(np.eye(5))
        assert result.is_symmetric is True
        assert result.max_deviation == 0.0

    def test_mean_deviation(self) -> None:
        # Specific asymmetry in one off-diagonal pair
        hess = np.array([[1.0, 2.0, 0.0], [2.5, 3.0, 0.0], [0.0, 0.0, 4.0]])
        result = check_symmetry(hess)
        assert result.max_deviation == pytest.approx(0.5)
        # Mean of upper triangle off-diag: (0.5, 0.0, 0.0) / 3
        assert result.mean_deviation == pytest.approx(0.5 / 3.0, abs=1e-10)

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            check_symmetry(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# compare_frequencies
# ---------------------------------------------------------------------------


class TestCompareFrequencies:
    def test_identical_frequencies(self) -> None:
        freqs = [100.0, 200.0, 300.0, 400.0]
        result = compare_frequencies(freqs, freqs)
        assert isinstance(result, FrequencyComparison)
        assert result.rmsd == pytest.approx(0.0, abs=1e-10)
        assert result.mae == pytest.approx(0.0, abs=1e-10)
        assert result.max_deviation == pytest.approx(0.0, abs=1e-10)
        assert result.n_modes == 4

    def test_known_differences(self) -> None:
        qm = [100.0, 200.0, 300.0]
        mm = [110.0, 190.0, 320.0]
        result = compare_frequencies(qm, mm)
        assert result.n_modes == 3
        assert result.mae == pytest.approx(np.mean([10.0, 10.0, 20.0]))
        assert result.max_deviation == pytest.approx(20.0)
        expected_rmsd = np.sqrt(np.mean([100.0, 100.0, 400.0]))
        assert result.rmsd == pytest.approx(expected_rmsd)

    def test_filters_low_frequencies(self) -> None:
        qm = [10.0, 20.0, 100.0, 200.0]  # first two below default threshold
        mm = [15.0, 25.0, 110.0, 210.0]
        result = compare_frequencies(qm, mm)
        assert result.n_modes == 2  # only 100 and 200

    def test_custom_threshold(self) -> None:
        qm = [10.0, 20.0, 100.0]
        mm = [15.0, 25.0, 110.0]
        result = compare_frequencies(qm, mm, threshold=5.0)
        assert result.n_modes == 3

    def test_mismatched_lengths(self) -> None:
        qm = [100.0, 200.0, 300.0]
        mm = [110.0, 210.0]
        result = compare_frequencies(qm, mm)
        assert result.n_modes == 2

    def test_empty_after_threshold(self) -> None:
        result = compare_frequencies([10.0], [20.0], threshold=50.0)
        assert result.n_modes == 0
        assert result.rmsd == 0.0

    def test_per_mode_detail(self) -> None:
        qm = [100.0, 200.0]
        mm = [120.0, 180.0]
        result = compare_frequencies(qm, mm)
        assert len(result.per_mode) == 2
        assert result.per_mode[0]["diff"] == pytest.approx(20.0)
        assert result.per_mode[0]["pct_err"] == pytest.approx(20.0)
        assert result.per_mode[1]["diff"] == pytest.approx(-20.0)
        assert result.per_mode[1]["pct_err"] == pytest.approx(-10.0)

    def test_unsorted_input(self) -> None:
        """Frequencies are sorted before comparison."""
        qm = [300.0, 100.0, 200.0]
        mm = [110.0, 310.0, 210.0]
        result = compare_frequencies(qm, mm)
        assert result.per_mode[0]["qm"] == pytest.approx(100.0)
        assert result.per_mode[0]["other"] == pytest.approx(110.0)


# ---------------------------------------------------------------------------
# analyze_mode_coupling
# ---------------------------------------------------------------------------


class TestAnalyzeModeCoupling:
    def test_self_projection_no_coupling(self) -> None:
        """Projecting a matrix onto its own eigenvectors → diagonal → no coupling."""
        hess = np.diag([1.0, 2.0, 3.0, 4.0])
        _, evecs = np.linalg.eigh(hess)
        result = analyze_mode_coupling(hess, evecs)
        assert isinstance(result, ModeCouplingAnalysis)
        assert result.max_coupling == pytest.approx(0.0, abs=1e-10)
        assert result.mean_coupling == pytest.approx(0.0, abs=1e-10)
        assert len(result.strongly_coupled_pairs) == 0

    def test_coupled_modes_detected(self) -> None:
        """Projecting a different matrix detects coupling."""
        # QM eigenvectors from a diagonal matrix
        qm_hess = np.diag([1.0, 2.0, 3.0, 4.0])
        _, qm_evecs = np.linalg.eigh(qm_hess)

        # MM Hessian with off-diagonal coupling
        mm_hess = np.diag([1.1, 1.9, 3.2, 3.8])
        mm_hess[0, 1] = mm_hess[1, 0] = 0.5  # mode coupling
        mm_hess[2, 3] = mm_hess[3, 2] = 0.3

        result = analyze_mode_coupling(mm_hess, qm_evecs, coupling_threshold=0.01)
        assert result.max_coupling > 0.01
        assert len(result.strongly_coupled_pairs) > 0

    def test_skip_modes(self) -> None:
        """Skipping modes excludes them from coupling analysis."""
        hess = np.eye(6) * np.arange(1, 7)
        _, evecs = np.linalg.eigh(hess)
        result_skip = analyze_mode_coupling(hess, evecs, skip_modes=3)
        result_full = analyze_mode_coupling(hess, evecs, skip_modes=0)
        # Both should be zero coupling for diagonal, but skip_modes changes
        # the number of analyzed pairs
        assert result_skip.coupling_matrix.shape == (6, 6)

    def test_coupling_matrix_symmetric(self) -> None:
        """Coupling matrix should be symmetric."""
        hess = np.array([[1.0, 0.3], [0.3, 2.0]])
        _, evecs = np.linalg.eigh(np.eye(2))
        result = analyze_mode_coupling(hess, evecs)
        np.testing.assert_allclose(result.coupling_matrix, result.coupling_matrix.T, atol=1e-15)


# ---------------------------------------------------------------------------
# format_hessian_report
# ---------------------------------------------------------------------------


class TestFormatHessianReport:
    def test_basic_report(self, symmetric_3x3: np.ndarray) -> None:
        """Smoke test: report runs without error and contains expected sections."""
        report = format_hessian_report(symmetric_3x3)
        assert "Hessian Reference Data Analysis" in report
        assert "Symmetry Check" in report
        assert "Eigenvalue Spectrum" in report

    def test_report_with_frequencies(self, minimum_hessian_6x6: np.ndarray) -> None:
        """Report includes frequency section when symbols provided."""
        report = format_hessian_report(
            minimum_hessian_6x6,
            symbols=["H", "H"],
            is_transition_state=False,
        )
        assert "QM Frequencies" in report
        assert "Real modes" in report

    def test_report_with_comparison(self, minimum_hessian_6x6: np.ndarray) -> None:
        """Report includes comparison section when MM frequencies provided."""
        qm_freqs = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
        mm_freqs = [110.0, 210.0, 310.0, 410.0, 510.0, 610.0]
        report = format_hessian_report(
            minimum_hessian_6x6,
            symbols=["H", "H"],
            is_transition_state=False,
            mm_frequencies=mm_freqs,
        )
        assert "Frequency Comparison" in report
        assert "RMSD" in report

    def test_report_with_mode_coupling(self, symmetric_3x3: np.ndarray) -> None:
        """Report includes mode coupling section when MM Hessian provided."""
        mm_hess = symmetric_3x3 + np.random.default_rng(42).normal(0, 0.01, symmetric_3x3.shape)
        mm_hess = (mm_hess + mm_hess.T) / 2  # ensure symmetric
        report = format_hessian_report(symmetric_3x3, mm_hessian=mm_hess)
        assert "Mode Coupling" in report
