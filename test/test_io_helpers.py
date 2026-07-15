"""Tests for shared, pure q2mm.io._helpers functions and per-format staging rows.

Covers ``_normalize_equilibrium_angle`` and ``_torsion_file_value`` (both
pure, format-agnostic) in isolation, plus the privacy and field isolation
of the per-format staging records ``q2mm.io.mm3._Mm3ParameterRow`` and
``q2mm.io.tinker._TinkerParameterRow``. ``test_models.py`` covers that each
format module actually wires ``_normalize_equilibrium_angle`` up only for
angle-equilibrium rows (and not, say, bond force constants) during real
parsing.
"""

import dataclasses

import pytest

from q2mm.io._helpers import _normalize_equilibrium_angle, _torsion_file_value
from q2mm.models.forcefield import TorsionParam


class TestNormalizeEquilibriumAngle:
    """Equilibrium bond-angle values fold into ``[0, 180]``."""

    def test_normal_angle_unchanged(self) -> None:
        assert _normalize_equilibrium_angle(109.5) == 109.5

    def test_angle_at_180_unchanged(self) -> None:
        assert _normalize_equilibrium_angle(180.0) == 180.0

    def test_angle_above_180_folds(self) -> None:
        assert _normalize_equilibrium_angle(200.0) == pytest.approx(160.0)

    def test_angle_at_360_folds_to_zero(self) -> None:
        assert _normalize_equilibrium_angle(360.0) == pytest.approx(0.0)

    def test_angle_540_no_longer_negative(self) -> None:
        assert _normalize_equilibrium_angle(540.0) == pytest.approx(180.0)

    def test_is_pure(self) -> None:
        """Calling repeatedly with the same input must not change the result."""
        value = 170.0
        first = _normalize_equilibrium_angle(value)
        second = _normalize_equilibrium_angle(value)
        assert first == second == 170.0


class TestTorsionFileValue:
    """``_torsion_file_value`` resolves the file-convention (``2*k``) torsion value."""

    def test_matched_row_returns_2x_force_constant(self) -> None:
        torsions = [
            TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=1.5, ff_row=42),
        ]
        value = _torsion_file_value(torsions, ff_row=42, periodicity=2)
        assert value == pytest.approx(3.0)

    def test_unmatched_ff_row_returns_none(self) -> None:
        torsions = [
            TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=1.5, ff_row=42),
        ]
        assert _torsion_file_value(torsions, ff_row=99, periodicity=2) is None

    def test_unmatched_periodicity_returns_none(self) -> None:
        torsions = [
            TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=1.5, ff_row=42),
        ]
        assert _torsion_file_value(torsions, ff_row=42, periodicity=3) is None

    def test_empty_torsions_returns_none(self) -> None:
        assert _torsion_file_value([], ff_row=42, periodicity=1) is None

    def test_input_torsion_is_unchanged(self) -> None:
        """Pure: resolving a value must not mutate the source TorsionParam."""
        torsion = TorsionParam(("C", "C", "C", "C"), periodicity=1, force_constant=2.25, ff_row=7)
        before = dataclasses.replace(torsion)

        _torsion_file_value([torsion], ff_row=7, periodicity=1)

        assert torsion == before
        assert torsion.force_constant == pytest.approx(2.25)

    def test_matches_first_row_with_multiple_periodicities(self) -> None:
        """Matching is by (ff_row, periodicity), not list position alone."""
        torsions = [
            TorsionParam(("C", "C", "C", "C"), periodicity=1, force_constant=1.0, ff_row=1),
            TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=2.0, ff_row=1),
            TorsionParam(("C", "C", "C", "C"), periodicity=3, force_constant=3.0, ff_row=1),
        ]
        assert _torsion_file_value(torsions, ff_row=1, periodicity=2) == pytest.approx(4.0)
        assert _torsion_file_value(torsions, ff_row=1, periodicity=3) == pytest.approx(6.0)


class TestPureHelpersNotExportedFromIoPackage:
    """``q2mm.io`` re-exports only the intended public I/O API, not ``_helpers`` internals."""

    def test_normalize_equilibrium_angle_not_a_package_attribute(self) -> None:
        import q2mm.io

        assert not hasattr(q2mm.io, "_normalize_equilibrium_angle")

    def test_torsion_file_value_not_a_package_attribute(self) -> None:
        import q2mm.io

        assert not hasattr(q2mm.io, "_torsion_file_value")


class TestFormatStagingRowsArePrivate:
    """Per-format parameter-row staging records are private and non-overlapping.

    ``q2mm.io.mm3._Mm3ParameterRow`` and ``q2mm.io.tinker._TinkerParameterRow``
    are each private to their own format module (leading underscore, not
    re-exported from ``q2mm.io``), ``slots=True`` so no arbitrary
    attribute can be attached to either, and Tinker's row must not carry
    the MM3-only ``bond_order``/``context`` columns.
    """

    def test_mm3_row_not_exported_from_io_package(self) -> None:
        import q2mm.io

        assert not hasattr(q2mm.io, "_Mm3ParameterRow")

    def test_tinker_row_not_exported_from_io_package(self) -> None:
        import q2mm.io

        assert not hasattr(q2mm.io, "_TinkerParameterRow")

    def test_row_class_names_use_private_naming_convention(self) -> None:
        from q2mm.io.mm3 import _Mm3ParameterRow
        from q2mm.io.tinker import _TinkerParameterRow

        assert _Mm3ParameterRow.__name__.startswith("_")
        assert _TinkerParameterRow.__name__.startswith("_")

    def test_mm3_row_has_no_dict_and_rejects_arbitrary_attributes(self) -> None:
        from q2mm.io.mm3 import _Mm3ParameterRow

        row = _Mm3ParameterRow(ptype="be", value=1.5, ff_row=1, ff_col=1)
        assert not hasattr(row, "__dict__")
        with pytest.raises(AttributeError):
            row.some_arbitrary_attribute = 1.0  # type: ignore[attr-defined]

    def test_tinker_row_has_no_dict_and_rejects_arbitrary_attributes(self) -> None:
        from q2mm.io.tinker import _TinkerParameterRow

        row = _TinkerParameterRow(ptype="bf", value=1.0, ff_row=1, ff_col=1)
        assert not hasattr(row, "__dict__")
        with pytest.raises(AttributeError):
            row.some_arbitrary_attribute = 1.0  # type: ignore[attr-defined]

    def test_tinker_row_has_no_mm3_only_fields(self) -> None:
        """Tinker rows must not carry MM3-only bond_order/context columns."""
        from q2mm.io.mm3 import _Mm3ParameterRow
        from q2mm.io.tinker import _TinkerParameterRow

        mm3_fields = {f.name for f in dataclasses.fields(_Mm3ParameterRow)}
        tinker_fields = {f.name for f in dataclasses.fields(_TinkerParameterRow)}

        assert {"bond_order", "context"} <= mm3_fields
        assert not ({"bond_order", "context"} & tinker_fields)
        # Every Tinker field must be one MM3 also uses (no format-unique
        # fields either direction beyond MM3's own bond_order/context).
        assert tinker_fields <= mm3_fields

    def test_tinker_row_rejects_mm3_only_fields_at_construction(self) -> None:
        """Slots reject bond_order/context outright, not just post-construction assignment."""
        from q2mm.io.tinker import _TinkerParameterRow

        with pytest.raises(TypeError):
            _TinkerParameterRow(ptype="bf", value=1.0, ff_row=1, ff_col=1, bond_order="-")  # type: ignore[call-arg]

    def test_tinker_row_rejects_bond_order_assignment(self) -> None:
        from q2mm.io.tinker import _TinkerParameterRow

        row = _TinkerParameterRow(ptype="bf", value=1.0, ff_row=1, ff_col=1)
        with pytest.raises(AttributeError):
            row.bond_order = "-"  # type: ignore[attr-defined]

    def test_tinker_row_rejects_context_assignment(self) -> None:
        from q2mm.io.tinker import _TinkerParameterRow

        row = _TinkerParameterRow(ptype="bf", value=1.0, ff_row=1, ff_col=1)
        with pytest.raises(AttributeError):
            row.context = "0000 0000"  # type: ignore[attr-defined]
