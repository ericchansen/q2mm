"""Tests for q2mm.models.units — type-safe unit conversions.

Covers round-trip identity, known-value correctness, and all engine
boundary conversion pairs.
"""

from __future__ import annotations

import math

import pytest

from q2mm.constants import (
    BOHR_TO_ANG,
    HARTREE_TO_KCALMOL,
    HESSIAN_AU_TO_KJMOLA2,
    KCAL_TO_KJ,
    KJMOLA2_TO_HESSIAN_AU,
    KJMOLNM2_TO_HESSIAN_AU,
)
from q2mm.models.units import (
    # MM3 boundary
    mm3_bond_k_to_canonical,
    canonical_to_mm3_bond_k,
    mm3_angle_k_to_canonical,
    canonical_to_mm3_angle_k,
    # OpenMM boundary — custom force
    canonical_to_openmm_bond_k,
    openmm_to_canonical_bond_k,
    canonical_to_openmm_angle_k,
    openmm_to_canonical_angle_k,
    canonical_to_openmm_torsion_k,
    openmm_to_canonical_torsion_k,
    canonical_to_openmm_epsilon,
    openmm_to_canonical_epsilon,
    # OpenMM boundary — harmonic forces
    canonical_to_openmm_harmonic_bond_k,
    openmm_to_canonical_harmonic_bond_k,
    canonical_to_openmm_harmonic_angle_k,
    openmm_to_canonical_harmonic_angle_k,
    # Tinker boundary (aliases)
    canonical_to_tinker_bond_k,
    tinker_to_canonical_bond_k,
    canonical_to_tinker_angle_k,
    tinker_to_canonical_angle_k,
    # QM boundary
    qm_to_canonical_bond_k,
    canonical_to_qm_bond_k,
    qm_to_canonical_angle_k,
    canonical_to_qm_angle_k,
    qm_to_canonical_energy,
    canonical_to_qm_energy,
    # Length / angle
    ang_to_nm,
    nm_to_ang,
    bohr_to_ang,
    ang_to_bohr,
    rmin_half_to_sigma_nm,
    rmin_half_to_sigma,
    deg_to_rad,
    rad_to_deg,
    # Hessian
    hessian_kcalmola2_to_au,
    hessian_au_to_kcalmola2,
    hessian_kjmolnm2_to_au,
    hessian_au_to_kjmolnm2,
    hessian_au_to_kjmola2,
    hessian_kjmola2_to_au,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from collections.abc import Callable, Sequence


_TEST_VALUES: list[float] = [0.0, 1.0, -2.5, 100.0, 1e-6, 1e6]


def _assert_roundtrip(
    forward: Callable[[float], float],
    inverse: Callable[[float], float],
    values: Sequence[float] = _TEST_VALUES,
    rtol: float = 1e-12,
) -> None:
    """Assert that inverse(forward(x)) ≈ x for every x in *values*."""
    for x in values:
        result = inverse(forward(x))
        if x == 0.0:
            assert result == pytest.approx(0.0, abs=1e-30)
        else:
            assert result == pytest.approx(x, rel=rtol), f"Round-trip failed for x={x}: got {result}"


# =====================================================================
# Round-trip tests: canonical → engine → canonical = identity
# =====================================================================


class TestMM3RoundTrips:
    """MM3 / Tinker boundary round-trips."""

    def test_bond_k(self) -> None:
        _assert_roundtrip(canonical_to_mm3_bond_k, mm3_bond_k_to_canonical)

    def test_angle_k(self) -> None:
        _assert_roundtrip(canonical_to_mm3_angle_k, mm3_angle_k_to_canonical)

    def test_tinker_bond_k_alias(self) -> None:
        _assert_roundtrip(canonical_to_tinker_bond_k, tinker_to_canonical_bond_k)

    def test_tinker_angle_k_alias(self) -> None:
        _assert_roundtrip(canonical_to_tinker_angle_k, tinker_to_canonical_angle_k)


class TestOpenMMRoundTrips:
    """OpenMM boundary round-trips."""

    def test_bond_k(self) -> None:
        _assert_roundtrip(canonical_to_openmm_bond_k, openmm_to_canonical_bond_k)

    def test_angle_k(self) -> None:
        _assert_roundtrip(canonical_to_openmm_angle_k, openmm_to_canonical_angle_k)

    def test_harmonic_bond_k(self) -> None:
        _assert_roundtrip(
            canonical_to_openmm_harmonic_bond_k,
            openmm_to_canonical_harmonic_bond_k,
        )

    def test_harmonic_angle_k(self) -> None:
        _assert_roundtrip(
            canonical_to_openmm_harmonic_angle_k,
            openmm_to_canonical_harmonic_angle_k,
        )

    def test_torsion_k(self) -> None:
        _assert_roundtrip(canonical_to_openmm_torsion_k, openmm_to_canonical_torsion_k)

    def test_epsilon(self) -> None:
        _assert_roundtrip(canonical_to_openmm_epsilon, openmm_to_canonical_epsilon)


class TestQMRoundTrips:
    """QM ↔ canonical round-trips."""

    def test_bond_k(self) -> None:
        _assert_roundtrip(canonical_to_qm_bond_k, qm_to_canonical_bond_k)

    def test_angle_k(self) -> None:
        _assert_roundtrip(canonical_to_qm_angle_k, qm_to_canonical_angle_k)

    def test_energy(self) -> None:
        _assert_roundtrip(canonical_to_qm_energy, qm_to_canonical_energy)


class TestLengthAngleRoundTrips:
    """Length and angle conversion round-trips."""

    def test_ang_nm(self) -> None:
        _assert_roundtrip(ang_to_nm, nm_to_ang)

    def test_bohr_ang(self) -> None:
        _assert_roundtrip(ang_to_bohr, bohr_to_ang)

    def test_deg_rad(self) -> None:
        _assert_roundtrip(deg_to_rad, rad_to_deg)


class TestHessianRoundTrips:
    """Hessian unit conversion round-trips."""

    def test_kcalmola2_au(self) -> None:
        _assert_roundtrip(hessian_kcalmola2_to_au, hessian_au_to_kcalmola2)

    def test_kjmolnm2_au(self) -> None:
        _assert_roundtrip(hessian_kjmolnm2_to_au, hessian_au_to_kjmolnm2)

    def test_kjmola2_au(self) -> None:
        _assert_roundtrip(hessian_kjmola2_to_au, hessian_au_to_kjmola2)


# =====================================================================
# Known-value tests — hand-computed reference values
# =====================================================================


class TestKnownValues:
    """Verify specific conversions against hand-computed values."""

    def test_openmm_bond_k_is_kcal_times_kj_factor(self) -> None:
        """canonical_to_openmm_bond_k multiplies by KCAL_TO_KJ (4.184)."""
        k = 100.0  # kcal/mol/Å²
        result = canonical_to_openmm_bond_k(k)
        assert result == pytest.approx(100.0 * KCAL_TO_KJ, rel=1e-14)

    def test_openmm_harmonic_bond_k_includes_half_and_nm(self) -> None:
        """HarmonicBondForce k = 2 * k_canonical * 4.184 * 100."""
        k = 50.0  # kcal/mol/Å²
        result = canonical_to_openmm_harmonic_bond_k(k)
        expected = 2.0 * 50.0 * KCAL_TO_KJ * 100.0
        assert result == pytest.approx(expected, rel=1e-14)

    def test_openmm_harmonic_angle_k_includes_half(self) -> None:
        """HarmonicAngleForce k = 2 * k_canonical * 4.184."""
        k = 80.0  # kcal/mol/rad²
        result = canonical_to_openmm_harmonic_angle_k(k)
        expected = 2.0 * 80.0 * KCAL_TO_KJ
        assert result == pytest.approx(expected, rel=1e-14)

    def test_ang_to_nm(self) -> None:
        assert ang_to_nm(10.0) == pytest.approx(1.0, rel=1e-14)

    def test_nm_to_ang(self) -> None:
        assert nm_to_ang(1.0) == pytest.approx(10.0, rel=1e-14)

    def test_bohr_to_ang_codata(self) -> None:
        """Verify against CODATA 2018 Bohr radius."""
        assert bohr_to_ang(1.0) == pytest.approx(BOHR_TO_ANG, rel=1e-10)

    def test_deg_to_rad(self) -> None:
        assert deg_to_rad(180.0) == pytest.approx(math.pi, rel=1e-14)

    def test_rad_to_deg(self) -> None:
        assert rad_to_deg(math.pi) == pytest.approx(180.0, rel=1e-14)

    def test_rmin_half_to_sigma_nm(self) -> None:
        """Sigma = 2*Rmin/2 / 2^(1/6) * 0.1, for Rmin/2 = 1.0 Å."""
        expected = 2.0 / (2.0 ** (1.0 / 6.0)) * 0.1
        assert rmin_half_to_sigma_nm(1.0) == pytest.approx(expected, rel=1e-14)

    def test_rmin_half_to_sigma(self) -> None:
        """Sigma = 2*Rmin/2 / 2^(1/6), for Rmin/2 = 1.5 Å."""
        expected = 2.0 * 1.5 / (2.0 ** (1.0 / 6.0))
        assert rmin_half_to_sigma(1.5) == pytest.approx(expected, rel=1e-14)

    def test_qm_to_canonical_energy(self) -> None:
        """1 Hartree ≈ 627.51 kcal/mol."""
        assert qm_to_canonical_energy(1.0) == pytest.approx(HARTREE_TO_KCALMOL, rel=1e-6)

    def test_hessian_au_to_kjmola2_matches_constant(self) -> None:
        """The function should match the raw constant from constants.py."""
        assert hessian_au_to_kjmola2(1.0) == pytest.approx(HESSIAN_AU_TO_KJMOLA2, rel=1e-12)

    def test_hessian_kjmolnm2_to_au_matches_constant(self) -> None:
        """The function should match the raw constant from constants.py."""
        assert hessian_kjmolnm2_to_au(1.0) == pytest.approx(KJMOLNM2_TO_HESSIAN_AU, rel=1e-12)

    def test_hessian_kcalmola2_to_au_matches_chain(self) -> None:
        """kcal/mol/Å² → au should equal KCAL_TO_KJ * KJMOLA2_TO_HESSIAN_AU."""
        expected = KCAL_TO_KJ * KJMOLA2_TO_HESSIAN_AU
        assert hessian_kcalmola2_to_au(1.0) == pytest.approx(expected, rel=1e-12)


# =====================================================================
# Consistency: Tinker aliases equal MM3 functions
# =====================================================================


class TestTinkerAliases:
    """Tinker aliases must produce the same results as MM3 functions."""

    def test_bond_k_forward(self) -> None:
        for v in _TEST_VALUES:
            assert canonical_to_tinker_bond_k(v) == canonical_to_mm3_bond_k(v)

    def test_bond_k_inverse(self) -> None:
        for v in _TEST_VALUES:
            assert tinker_to_canonical_bond_k(v) == mm3_bond_k_to_canonical(v)

    def test_angle_k_forward(self) -> None:
        for v in _TEST_VALUES:
            assert canonical_to_tinker_angle_k(v) == canonical_to_mm3_angle_k(v)

    def test_angle_k_inverse(self) -> None:
        for v in _TEST_VALUES:
            assert tinker_to_canonical_angle_k(v) == mm3_angle_k_to_canonical(v)
