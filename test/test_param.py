"""Tests for q2mm.parsers.param.Param."""

import pytest

from q2mm.parsers.param import Param, ParamError


class TestParamEquality:
    """Param identity is defined by (ptype, ff_row, ff_col)."""

    def test_equal_params(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=999.0)
        assert a == b

    def test_different_ptype(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="be", ff_row=5, ff_col=1)
        assert a != b

    def test_different_row(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="bf", ff_row=6, ff_col=1)
        assert a != b

    def test_different_col(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1)
        b = Param(ptype="bf", ff_row=5, ff_col=2)
        assert a != b

    def test_hash_equal(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=2.0)
        assert hash(a) == hash(b)

    def test_usable_in_set(self) -> None:
        a = Param(ptype="bf", ff_row=5, ff_col=1, value=1.0)
        b = Param(ptype="bf", ff_row=5, ff_col=1, value=2.0)
        c = Param(ptype="be", ff_row=5, ff_col=2, value=1.5)
        assert len({a, b, c}) == 2

    def test_not_equal_to_non_param(self) -> None:
        p = Param(ptype="bf", ff_row=5, ff_col=1)
        assert p != "not a param"
        assert p != 42

    def test_identity_shortcut(self) -> None:
        p = Param(ptype="bf", ff_row=5, ff_col=1)
        assert p == p  # noqa: PLR0124

    def test_incomplete_identity_not_comparable(self) -> None:
        """Params with None in identity fields fall back to identity comparison."""
        a = Param(ptype="ae")
        b = Param(ptype="ae")
        # NotImplemented from __eq__ causes Python to fall back to `is`
        assert not (a == b)  # noqa: SIM201 — intentionally testing __eq__ fallback
        assert a == a  # noqa: PLR0124

    def test_incomplete_identity_not_hashable(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            hash(Param())
        with pytest.raises(TypeError, match="unhashable"):
            hash(Param(ptype="bf"))


class TestParamConstruction:
    """Verify default construction and None handling."""

    def test_default_construction(self) -> None:
        p = Param()
        assert p.value is None
        assert p.ptype is None

    def test_value_none_is_valid(self) -> None:
        p = Param(ptype="bf")
        assert p.value is None

    def test_set_value_after_construction(self) -> None:
        p = Param(ptype="bf")
        p.value = 1.5
        assert p.value == 1.5


class TestAngleNormalization:
    """Equilibrium angle values fold into [0, 180] on assignment."""

    def test_normal_angle(self) -> None:
        p = Param(ptype="ae", value=109.5)
        assert p.value == 109.5

    def test_angle_above_180(self) -> None:
        p = Param(ptype="ae", value=200.0)
        assert p.value == pytest.approx(160.0)

    def test_angle_at_360(self) -> None:
        p = Param(ptype="ae", value=360.0)
        assert p.value == pytest.approx(0.0)

    def test_angle_540_no_longer_negative(self) -> None:
        p = Param(ptype="ae", value=540.0)
        assert p.value == pytest.approx(180.0)

    def test_non_ae_no_normalization(self) -> None:
        p = Param(ptype="bf", value=200.0)
        assert p.value == 200.0

    def test_getter_is_pure(self) -> None:
        """Reading .value multiple times should not change the result."""
        p = Param(ptype="ae", value=170.0)
        _ = p.value
        _ = p.value
        assert p.value == 170.0


class TestParamRange:
    """Range validation for parameter types."""

    def test_force_constant_allows_negative(self) -> None:
        p = Param(ptype="bf", value=-5.0)
        assert p.value == -5.0

    def test_equilibrium_rejects_negative(self) -> None:
        with pytest.raises(ParamError):
            Param(ptype="be", value=-1.0)

    def test_charge_allows_negative(self) -> None:
        p = Param(ptype="q", value=-0.5)
        assert p.value == -0.5
