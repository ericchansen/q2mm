"""Tests for q2mm.geometry — canonical bond length, angle, and dihedral calculations."""

import math

import numpy as np
import pytest

from q2mm.geometry import bond_angle, bond_length, dihedral_angle


# ---------------------------------------------------------------------------
# bond_length
# ---------------------------------------------------------------------------


class TestBondLength:
    def test_unit_x(self) -> None:
        assert bond_length([0, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_diagonal(self) -> None:
        assert bond_length([0, 0, 0], [1, 1, 1]) == pytest.approx(math.sqrt(3))

    def test_same_point(self) -> None:
        assert bond_length([2, 3, 4], [2, 3, 4]) == pytest.approx(0.0)

    def test_negative_coords(self) -> None:
        assert bond_length([-1, 0, 0], [1, 0, 0]) == pytest.approx(2.0)

    def test_numpy_arrays(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([3.0, 4.0, 0.0])
        assert bond_length(p0, p1) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# bond_angle
# ---------------------------------------------------------------------------


class TestBondAngle:
    def test_right_angle(self) -> None:
        assert bond_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(90.0)

    def test_linear(self) -> None:
        assert bond_angle([-1, 0, 0], [0, 0, 0], [1, 0, 0]) == pytest.approx(180.0)

    def test_acute_60(self) -> None:
        # Equilateral triangle vertices in 2D
        p0 = np.array([1.0, 0.0, 0.0])
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([0.5, math.sqrt(3) / 2, 0.0])
        assert bond_angle(p0, p1, p2) == pytest.approx(60.0)

    def test_symmetric(self) -> None:
        # Angle is the same regardless of which outer atom is first.
        p0, p1, p2 = [1, 0, 0], [0, 0, 0], [0, 1, 0]
        assert bond_angle(p0, p1, p2) == pytest.approx(bond_angle(p2, p1, p0))


# ---------------------------------------------------------------------------
# dihedral_angle
# ---------------------------------------------------------------------------


class TestDihedralAngle:
    def test_cis_zero(self) -> None:
        # cis / eclipsed -> 0 degrees
        p0 = [1, 0, 0]
        p1 = [0, 0, 0]
        p2 = [0, 1, 0]
        p3 = [1, 1, 0]
        assert dihedral_angle(p0, p1, p2, p3) == pytest.approx(0.0, abs=1e-6)

    def test_trans_180(self) -> None:
        # trans / anti -> +/-180 degrees
        p0 = [1, 0, 0]
        p1 = [0, 0, 0]
        p2 = [0, 1, 0]
        p3 = [-1, 1, 0]
        assert abs(dihedral_angle(p0, p1, p2, p3)) == pytest.approx(180.0, abs=1e-6)

    def test_gauche_90(self) -> None:
        p0 = [1, 0, 0]
        p1 = [0, 0, 0]
        p2 = [0, 1, 0]
        p3 = [0, 1, 1]
        assert dihedral_angle(p0, p1, p2, p3) == pytest.approx(90.0, abs=1e-6)

    def test_degenerate_collinear_returns_zero(self) -> None:
        # Collinear atoms: degenerate, should return 0.0
        p0, p1, p2, p3 = [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]
        assert dihedral_angle(p0, p1, p2, p3) == pytest.approx(0.0, abs=1e-6)

    def test_sign_convention(self) -> None:
        # Positive and negative rotations should differ in sign
        p0 = [1, 0, 0]
        p1 = [0, 0, 0]
        p2 = [0, 1, 0]
        p3_pos = [0, 1, 1]
        p3_neg = [0, 1, -1]
        d_pos = dihedral_angle(p0, p1, p2, p3_pos)
        d_neg = dihedral_angle(p0, p1, p2, p3_neg)
        assert d_pos == pytest.approx(-d_neg, abs=1e-6)
