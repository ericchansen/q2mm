"""Tests for q2mm.parsers.param.Param."""

from __future__ import annotations

import pytest

from q2mm.parsers.param import Param, ParamError


class TestParamEquality:
    """Param identity is defined by (ptype, ff_row, ff_col)."""

    def test_equal_params(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=999.0)
        assert a == b

    def test_different_ptype(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="be", ff_row=5, ff_col=1)
        assert a != b

    def test_different_row(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="bf", ff_row=6, ff_col=1)
        assert a != b

    def test_different_col(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="bf", ff_row=5, ff_col=2)
        assert a != b

    def test_hash_equal(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=2.0)
        assert hash(a) == hash(b)

    def test_usable_in_set(self):
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=2.0)
        c = Param(ptype="be", ff_row=5, ff_col=2, value=1.5)
        assert len({a, b, c}) == 2

    def test_not_equal_to_non_param(self):
        p = Param(ptype="bf", ff_row=5, ff_col=1)
        assert p != "not a param"
        assert p != 42


class TestParamConstruction:
    """Verify default construction and None handling."""

    def test_default_construction(self):
        p = Param()
        assert p.value is None
        assert p.ptype is None

    def test_value_none_is_valid(self):
        p = Param(ptype="bf")
        assert p.value is None

    def test_set_value_after_construction(self):
        p = Param(ptype="bf")
        p.value = 1.5
        assert p.value == 1.5


class TestAngleNormalization:
    """Equilibrium angle values fold into [0, 180] on assignment."""

    def test_normal_angle(self):
        p = Param(ptype="ae", value=109.5)
        assert p.value == 109.5

    def test_angle_above_180(self):
        p = Param(ptype="ae", value=200.0)
        assert p.value == pytest.approx(160.0)

    def test_angle_at_360(self):
        p = Param(ptype="ae", value=360.0)
        assert p.value == pytest.approx(0.0)

    def test_angle_540_no_longer_negative(self):
        p = Param(ptype="ae", value=540.0)
        assert p.value == pytest.approx(180.0)

    def test_non_ae_no_normalization(self):
        p = Param(ptype="bf", value=200.0)
        assert p.value == 200.0

    def test_getter_is_pure(self):
        """Reading .value multiple times should not change the result."""
        p = Param(ptype="ae", value=170.0)
        _ = p.value
        _ = p.value
        assert p.value == 170.0


class TestParamRange:
    """Range validation for parameter types."""

    def test_force_constant_allows_negative(self):
        p = Param(ptype="bf", value=-5.0)
        assert p.value == -5.0

    def test_equilibrium_rejects_negative(self):
        with pytest.raises(ParamError):
            Param(ptype="be", value=-1.0)

    def test_charge_allows_negative(self):
        p = Param(ptype="q", value=-0.5)
        assert p.value == -0.5
