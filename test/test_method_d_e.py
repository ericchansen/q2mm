"""Tests for Method D/E eigenvalue fitting (Limé & Norrby 2015).

Validates the three TSFF eigenvalue strategies:
- Method C: force reaction coordinate to large positive value (existing)
- Method D: keep natural (negative) eigenvalue (new)
- Method E: hybrid — D first, lock problematic params, reoptimize with C (new)

Reference: Limé & Norrby, J. Comput. Chem. 2015, 36, 1130.
DOI: 10.1002/jcc.23797

Covers issue #75.
"""

import numpy as np
import pytest

from test._shared import SN2_HESSIAN

from q2mm.models.hessian import (
    decompose,
    detect_problematic_params,
    invert_ts_curvature,
    keep_natural_eigenvalue,
    lock_params,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField


# ---------------------------------------------------------------------------
# Fixtures: synthetic TS Hessian with one negative eigenvalue
# ---------------------------------------------------------------------------
@pytest.fixture
def ts_eigenvalues():
    """Eigenvalues typical of a TS: one negative (reaction coordinate)."""
    return np.array([-2.5, 0.1, 0.5, 1.0, 2.0, 3.0])


@pytest.fixture
def multi_neg_eigenvalues():
    """Eigenvalues with multiple negatives (higher-order saddle)."""
    return np.array([-5.0, -2.0, 0.1, 0.5, 1.0, 3.0])


@pytest.fixture
def positive_eigenvalues():
    """All-positive eigenvalues (minimum, not TS)."""
    return np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def synthetic_ts_hessian():
    """Build a synthetic symmetric TS Hessian with exactly 1 negative eigenvalue."""
    rng = np.random.default_rng(42)
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    # Eigenvalues: one negative (reaction coordinate), rest positive
    evals = np.array([-3.0, 0.2, 0.8, 1.5, 2.5, 4.0])
    H = Q @ np.diag(evals) @ Q.T
    return H, evals, Q


# ---------------------------------------------------------------------------
# Method D: keep_natural_eigenvalue
# ---------------------------------------------------------------------------
class TestMethodD:
    """Method D: keep natural eigenvalue without replacement."""

    def test_returns_copy(self, ts_eigenvalues):
        result = keep_natural_eigenvalue(ts_eigenvalues)
        assert result is not ts_eigenvalues
        np.testing.assert_array_equal(result, ts_eigenvalues)

    def test_preserves_negative(self, ts_eigenvalues):
        result = keep_natural_eigenvalue(ts_eigenvalues)
        assert result[0] < 0
        assert result[0] == ts_eigenvalues[0]

    def test_all_positive_unchanged(self, positive_eigenvalues):
        result = keep_natural_eigenvalue(positive_eigenvalues)
        np.testing.assert_array_equal(result, positive_eigenvalues)

    def test_multi_neg_preserved(self, multi_neg_eigenvalues):
        result = keep_natural_eigenvalue(multi_neg_eigenvalues)
        assert np.sum(result < 0) == 2


# ---------------------------------------------------------------------------
# Method C vs D via invert_ts_curvature
# ---------------------------------------------------------------------------
class TestInvertTSCurvature:
    """Compare Method C and D through the unified interface."""

    def test_method_c_no_negative_eigenvalues(self, synthetic_ts_hessian):
        H, _, _ = synthetic_ts_hessian
        result = invert_ts_curvature(H, method="C")
        evals = np.linalg.eigvalsh(result)
        assert np.all(evals >= -1e-10), f"Method C left negative eigenvalue: {min(evals)}"

    def test_method_d_preserves_negative_eigenvalue(self, synthetic_ts_hessian):
        H, original_evals, _ = synthetic_ts_hessian
        result = invert_ts_curvature(H, method="D")
        evals = np.sort(np.linalg.eigvalsh(result))
        # Method D should preserve the negative eigenvalue
        assert evals[0] < 0
        np.testing.assert_allclose(evals, np.sort(original_evals), atol=1e-10)

    def test_method_d_is_identity_transform(self, synthetic_ts_hessian):
        """Method D should return a matrix equivalent to the original."""
        H, _, _ = synthetic_ts_hessian
        result = invert_ts_curvature(H, method="D")
        np.testing.assert_allclose(result, H, atol=1e-10)

    def test_default_is_method_c(self, synthetic_ts_hessian):
        H, _, _ = synthetic_ts_hessian
        default_result = invert_ts_curvature(H)
        c_result = invert_ts_curvature(H, method="C")
        np.testing.assert_allclose(default_result, c_result, atol=1e-10)

    def test_method_c_replaces_with_large_positive(self, synthetic_ts_hessian):
        H, _, _ = synthetic_ts_hessian
        result = invert_ts_curvature(H, method="C")
        evals = np.sort(np.linalg.eigvalsh(result))
        # The replaced eigenvalue should be the largest by far
        assert evals[-1] > 1000  # 1 a.u. → ~9376 kJ/mol/A²/amu


# ---------------------------------------------------------------------------
# detect_problematic_params
# ---------------------------------------------------------------------------
class TestDetectProblematicParams:
    """Detect zero/negative force constants after Method D fitting."""

    def _make_ff(self, bond_fcs, angle_fcs):
        bonds = [BondParam(elements=("C", "H"), force_constant=fc, equilibrium=1.09) for fc in bond_fcs]
        angles = [AngleParam(elements=("H", "C", "H"), force_constant=fc, equilibrium=109.5) for fc in angle_fcs]
        return ForceField(name="test", bonds=bonds, angles=angles)

    def test_no_problems(self):
        ff = self._make_ff([359.7, 215.8], [43.2, 57.6])
        result = detect_problematic_params(ff)
        assert result["bonds"] == set()
        assert result["angles"] == set()

    def test_negative_bond_fc(self):
        ff = self._make_ff([359.7, -7.2], [43.2, 57.6])
        result = detect_problematic_params(ff)
        assert ("C", "H") in result["bonds"]
        assert result["angles"] == set()

    def test_zero_angle_fc(self):
        ff = self._make_ff([359.7, 215.8], [43.2, 0.0])
        result = detect_problematic_params(ff)
        assert result["bonds"] == set()
        assert ("H", "C", "H") in result["angles"]

    def test_custom_threshold(self):
        ff = self._make_ff([359.7, 3.6], [43.2, 0.7])
        result = detect_problematic_params(ff, fc_threshold=7.2)
        assert ("C", "H") in result["bonds"]
        assert ("H", "C", "H") in result["angles"]

    def test_all_problematic(self):
        """Multiple distinct problematic param types are all reported."""
        bonds = [
            BondParam(elements=("C", "H"), force_constant=-71.9, equilibrium=1.09),
            BondParam(elements=("C", "C"), force_constant=-143.9, equilibrium=1.53),
        ]
        angles = [
            AngleParam(elements=("H", "C", "H"), force_constant=0.0, equilibrium=109.5),
            AngleParam(elements=("C", "C", "H"), force_constant=-36.0, equilibrium=111.0),
        ]
        ff = ForceField(name="test", bonds=bonds, angles=angles)
        result = detect_problematic_params(ff)
        assert result["bonds"] == {("C", "H"), ("C", "C")}
        assert result["angles"] == {("C", "C", "H"), ("H", "C", "H")}

    def test_invalid_method_raises(self):
        """Unsupported method string must raise ValueError, not silently fall through."""
        H = np.eye(6)
        with pytest.raises(ValueError, match="Unknown method"):
            invert_ts_curvature(H, method="E")
        with pytest.raises(ValueError, match="Unknown method"):
            invert_ts_curvature(H, method="X")


# ---------------------------------------------------------------------------
# lock_params
# ---------------------------------------------------------------------------
class TestLockParams:
    """Lock problematic parameters to reference values."""

    def _make_ff(self, bond_fc, bond_eq, angle_fc, angle_eq):
        return ForceField(
            name="test",
            bonds=[BondParam(elements=("C", "H"), force_constant=bond_fc, equilibrium=bond_eq)],
            angles=[AngleParam(elements=("H", "C", "H"), force_constant=angle_fc, equilibrium=angle_eq)],
        )

    def test_lock_bond(self):
        target = self._make_ff(-36.0, 1.09, 43.2, 109.5)
        source = self._make_ff(359.7, 1.10, 50.4, 108.0)
        lock_params(target, {"bonds": {("C", "H")}, "angles": set()}, source)
        assert target.bonds[0].force_constant == 359.7
        assert target.bonds[0].equilibrium == 1.10

    def test_lock_angle(self):
        target = self._make_ff(359.7, 1.09, -7.2, 109.5)
        source = self._make_ff(215.8, 1.10, 50.4, 108.0)
        lock_params(target, {"bonds": set(), "angles": {("H", "C", "H")}}, source)
        assert target.angles[0].force_constant == 50.4
        assert target.angles[0].equilibrium == 108.0

    def test_lock_preserves_unlocked(self):
        target = self._make_ff(5.0, 1.09, 0.6, 109.5)
        source = self._make_ff(215.8, 1.10, 50.4, 108.0)
        lock_params(target, {"bonds": set(), "angles": set()}, source)
        assert target.bonds[0].force_constant == 5.0
        assert target.angles[0].force_constant == 0.6

    def test_lock_empty_keys(self):
        target = self._make_ff(5.0, 1.09, 0.6, 109.5)
        source = self._make_ff(215.8, 1.10, 50.4, 108.0)
        lock_params(target, {}, source)
        assert target.bonds[0].force_constant == 5.0


# ---------------------------------------------------------------------------
# Integration: Method D on real SN2 data
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SN2_HESSIAN.exists(), reason="SN2 hessian not found")
class TestMethodDOnSN2:
    """Method D on real SN2 TS Hessian."""

    @pytest.fixture(scope="class")
    def sn2_hessian(self):
        return np.load(str(SN2_HESSIAN))

    def test_sn2_has_negative_eigenvalues(self, sn2_hessian):
        """SN2 TS Hessian has at least one negative eigenvalue (reaction coordinate)."""
        evals, _ = decompose(sn2_hessian)
        n_neg = int(np.sum(evals < 0))
        assert n_neg >= 1, f"Expected at least 1 negative eigenvalue, got {n_neg}"

    def test_sn2_most_negative_is_reaction_coordinate(self, sn2_hessian):
        """The most negative eigenvalue should be significantly more negative than others."""
        evals, _ = decompose(sn2_hessian)
        neg_evals = sorted(evals[evals < 0])
        # The reaction coordinate eigenvalue should dominate
        assert neg_evals[0] < -0.01, f"Most negative eigenvalue too small: {neg_evals[0]}"

    def test_method_c_removes_negative(self, sn2_hessian):
        result = invert_ts_curvature(sn2_hessian, method="C")
        evals = np.linalg.eigvalsh(result)
        assert np.all(evals >= -1e-10)

    def test_method_d_preserves_negative(self, sn2_hessian):
        result = invert_ts_curvature(sn2_hessian, method="D")
        evals = np.linalg.eigvalsh(result)
        assert np.any(evals < 0)

    def test_method_d_eigenvalues_match_original(self, sn2_hessian):
        original_evals = np.sort(np.linalg.eigvalsh(sn2_hessian))
        result = invert_ts_curvature(sn2_hessian, method="D")
        result_evals = np.sort(np.linalg.eigvalsh(result))
        np.testing.assert_allclose(result_evals, original_evals, atol=1e-8)

    def test_method_c_vs_d_most_negative_differs(self, sn2_hessian):
        """Method C replaces the most negative eigenvalue; D keeps it."""
        evals_orig = np.sort(np.linalg.eigvalsh(sn2_hessian))
        result_c = invert_ts_curvature(sn2_hessian, method="C")
        evals_c = np.sort(np.linalg.eigvalsh(result_c))

        result_d = invert_ts_curvature(sn2_hessian, method="D")
        evals_d = np.sort(np.linalg.eigvalsh(result_d))

        # D should preserve the most negative eigenvalue
        np.testing.assert_allclose(evals_d[0], evals_orig[0], atol=1e-10)
        # C should have replaced it with something positive
        assert evals_c[-1] > abs(evals_orig[0])
