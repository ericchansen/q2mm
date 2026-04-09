"""Tests for _strip_pint() helper in q2mm.models.molecule."""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.molecule import _strip_pint


def _has_pint() -> bool:
    try:
        import pint  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


class TestStripPint:
    """Unit tests for the _strip_pint guard function."""

    def test_none_passthrough(self) -> None:
        assert _strip_pint(None) is None

    def test_bare_ndarray_passthrough(self) -> None:
        arr = np.eye(3)
        result = _strip_pint(arr)
        assert result is arr

    def test_pint_quantity_stripped(self) -> None:
        """Use a lightweight fake to avoid requiring pint."""

        class FakeQuantity:
            """Mimics pint.Quantity with .magnitude and .to()."""

            def __init__(self, data: np.ndarray, unit: str) -> None:
                self._data = data
                self._unit = unit

            @property
            def magnitude(self) -> np.ndarray:
                return self._data

            def to(self, target_unit: str) -> FakeQuantity:
                assert target_unit == "hartree/bohr**2"
                return self

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        fake_q = FakeQuantity(arr, "hartree/bohr**2")
        result = _strip_pint(fake_q)
        np.testing.assert_array_equal(result, arr)
        assert isinstance(result, np.ndarray)

    def test_object_with_magnitude_but_no_to(self) -> None:
        """Objects with .magnitude but no .to() should pass through unchanged."""

        class PartialMagnitude:
            magnitude = 42.0

        obj = PartialMagnitude()
        result = _strip_pint(obj)
        assert result is obj

    @pytest.mark.skipif(
        not _has_pint(),
        reason="pint not installed",
    )
    def test_real_pint_quantity(self) -> None:
        """Integration test with real pint (only runs if installed)."""
        import pint  # noqa: PLC0415

        ureg = pint.UnitRegistry()
        arr = np.array([[0.5, 0.1], [0.1, 0.3]])
        q = ureg.Quantity(arr, "hartree/bohr**2")
        result = _strip_pint(q)
        np.testing.assert_array_equal(result, arr)
        assert isinstance(result, np.ndarray)
