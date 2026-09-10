"""MM3 template updates must preserve the source scope of parameter subsets."""

from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.io.mm3 import (
    COM_POS_START,
    P_1_END,
    P_1_START,
    _format_mm3_angle_line,
    _format_mm3_bond_line,
    load_mm3_fld,
    save_mm3_fld,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from test.test_ffs import FF_PATH, _substructure_row

_FAMILIES = ("bonds", "angles", "stretch_bends")


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("in_place", [False, True])
def test_complete_import_changes_only_one_bonded_source_line(tmp_path: Path, family: str, in_place: bool) -> None:
    source = tmp_path / "complete.fld"
    source.write_bytes(FF_PATH.read_bytes())
    before = source.read_bytes().splitlines(keepends=True)
    ff = load_mm3_fld(source)
    parameters = getattr(ff, family)
    selected = parameters[0]
    modified = replace(selected, force_constant=selected.force_constant * 2.0 + 1.0)
    ff = replace(ff, **{family: (modified, *parameters[1:])})
    output = source if in_place else tmp_path / "output.fld"
    save_mm3_fld(ff, output)
    after = output.read_bytes().splitlines(keepends=True)
    assert len(after) == len(before)
    assert selected.ff_row is not None
    # vdW uses a separate exporter; every other bonded source row must retain
    # its spelling even though the full field, not a selected subset, is saved.
    bonded_rows = {
        param.ff_row
        for terms in (ff.bonds, ff.angles, ff.stretch_bends, ff.torsions)
        for param in terms
        if param.ff_row is not None
    }
    assert len(bonded_rows) > 100
    for row in bonded_rows:
        if row == selected.ff_row:
            assert after[row - 1] != before[row - 1]
        else:
            assert after[row - 1] == before[row - 1], f"Unchanged bonded source row {row} was rewritten"
    actual = getattr(load_mm3_fld(output), family)[0]
    assert actual.force_constant == pytest.approx(modified.force_constant, abs=0.02)


def _collision_template(tmp_path: Path, family: str) -> Path:
    if family == "bonds":
        standard = _format_mm3_bond_line(["C1", "C1"], 1.1, 2.0)
        frozen = _substructure_row(" 1", ("1", "2"), 1.2, 3.0)
        selected = _substructure_row(" 1", ("1", "2"), 1.3, 4.0)
    elif family == "angles":
        standard = _format_mm3_angle_line(["C1", "C1", "C1"], 101.0, 0.2)
        frozen = _substructure_row(" 2", ("1", "2", "3"), 102.0, 0.3)
        selected = _substructure_row(" 2", ("1", "2", "3"), 103.0, 0.4)
    else:
        standard = _format_mm3_angle_line(["C1", "C1", "C1"], 0.1, 0.0).replace(" 2", " 3", 1)
        frozen = _substructure_row(" 3", ("1", "2", "3"), 0.2)
        selected = _substructure_row(" 3", ("1", "2", "3"), 0.3)

    def source_spelling(line: str, label: str) -> str:
        value = float(line[P_1_START:P_1_END])
        line = line[:P_1_START] + f"{value:010.4f}" + line[P_1_END:]
        return line.rstrip("\n").ljust(COM_POS_START) + f" {label}\n"

    lines = [
        source_spelling(standard, "standard before"),
        " C  FROZEN\n",
        " 9  C1-C1-C1 \n",
        source_spelling(frozen, "non-OPT source"),
        "-3\n",
        " C  OPT selected\n",
        " 9  C1-C1-C1 \n",
        source_spelling(selected, "selected source"),
        "-3\n",
        source_spelling(standard, "standard after"),
        "-2\n",
    ]
    path = tmp_path / "source.fld"
    path.write_text("".join(lines), encoding="utf-8")
    return path


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("selected_scope", ["OPT", "FROZEN"])
@pytest.mark.parametrize("in_place", [False, True])
def test_source_subset_updates_only_its_exact_rows(
    tmp_path: Path, family: str, selected_scope: str, in_place: bool
) -> None:
    source = _collision_template(tmp_path, family)
    before = source.read_bytes().splitlines(keepends=True)
    if selected_scope == "OPT":
        ff = load_mm3_fld(source, include_standard=False)
        selected = getattr(ff, family)[0]
    else:
        ff = load_mm3_fld(source)
        selected = next(param for param in getattr(ff, family) if param.ff_row == 4)
    modified = replace(selected, force_constant=selected.force_constant * 2.0)
    if isinstance(modified, (BondParam, AngleParam)):
        modified = replace(modified, equilibrium=modified.equilibrium + 0.25)
    ff = replace(ff, **{family: (modified,)})
    output = source if in_place else tmp_path / "output.fld"
    save_mm3_fld(ff, output)

    after = output.read_bytes().splitlines(keepends=True)
    assert len(after) == len(before)
    assert selected.ff_row is not None
    for index, (old_line, new_line) in enumerate(zip(before, after, strict=True), start=1):
        if index == selected.ff_row:
            assert new_line != old_line
        else:
            assert new_line == old_line, f"Unselected source row {index} changed"
    reloaded = getattr(load_mm3_fld(output), family)
    actual = next(param for param in reloaded if param.ff_row == modified.ff_row)
    assert actual.force_constant == pytest.approx(modified.force_constant)
    if isinstance(modified, (BondParam, AngleParam)):
        assert actual.equilibrium == pytest.approx(modified.equilibrium)


@pytest.mark.parametrize("family", _FAMILIES)
def test_explicit_template_keeps_parameter_row_identity_without_file_metadata(tmp_path: Path, family: str) -> None:
    source = _collision_template(tmp_path, family)
    before = source.read_bytes().splitlines(keepends=True)
    ff = load_mm3_fld(source, include_standard=False)
    selected = getattr(ff, family)[0]
    modified = replace(selected, force_constant=selected.force_constant * 2.0)
    ff = replace(ff, source_path=None, source_format=None, **{family: (modified,)})
    output = tmp_path / "explicit.fld"
    save_mm3_fld(ff, output, template_path=source)
    after = output.read_bytes().splitlines(keepends=True)
    assert [line for index, line in enumerate(after, 1) if index != modified.ff_row] == [
        line for index, line in enumerate(before, 1) if index != modified.ff_row
    ]
    assert after[modified.ff_row - 1] != before[modified.ff_row - 1]


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("include_sourced", [False, True])
@pytest.mark.parametrize("reverse_parameters", [False, True])
def test_unsourced_template_updates_keep_environment_fallback_and_row_priority(
    tmp_path: Path, family: str, include_sourced: bool, reverse_parameters: bool
) -> None:
    source = _collision_template(tmp_path, family)
    selected = getattr(load_mm3_fld(source, include_standard=False), family)[0]
    unsourced = replace(selected, ff_row=None, force_constant=selected.force_constant * 2.0)
    sourced = replace(selected, force_constant=selected.force_constant * 3.0)
    parameters = (unsourced, sourced) if include_sourced else (unsourced,)
    if reverse_parameters:
        parameters = tuple(reversed(parameters))
    ff = ForceField(functional_form=FunctionalForm.MM3, **{family: parameters})
    output = tmp_path / "programmatic.fld"
    save_mm3_fld(ff, output, template_path=source)
    reloaded = getattr(load_mm3_fld(output), family)
    assert len(reloaded) == 4
    for actual in reloaded:
        expected = sourced if include_sourced and actual.ff_row == sourced.ff_row else unsourced
        assert actual.force_constant == pytest.approx(expected.force_constant)


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("existing_output", [False, True])
def test_missing_source_row_rejects_before_destination_change(
    tmp_path: Path, family: str, existing_output: bool
) -> None:
    source = _collision_template(tmp_path, family)
    ff = load_mm3_fld(source, include_standard=False)
    selected = getattr(ff, family)[0]
    ff = replace(ff, **{family: (replace(selected, ff_row=999),)})
    output = tmp_path / "output.fld"
    if existing_output:
        output.write_bytes(b"existing destination\n")
    with pytest.raises(ValueError, match=r"source\.fld.*source row.*999"):
        save_mm3_fld(ff, output)
    if existing_output:
        assert output.read_bytes() == b"existing destination\n"
    else:
        assert not output.exists()
