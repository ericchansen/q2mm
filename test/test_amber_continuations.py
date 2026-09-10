"""AMBER continuation encoding tests, not AMBER engine/runtime tests."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.io.amber import load_amber_frcmod, save_amber_frcmod
from q2mm.models.forcefield import ForceField, FunctionalForm, TorsionParam


def _term(types: str, n: int, k: float = 1.0, phase: float = 37.0, **kwargs: object) -> TorsionParam:
    return TorsionParam(("H", "C", "C", "H"), periodicity=n, force_constant=k, phase=phase, env_id=types, **kwargs)


def _records(path: Path, section: str = "DIHE") -> list[list[str]]:
    active = False
    records = []
    for line in path.read_text().splitlines():
        if line == section:
            active = True
        elif not line.strip():
            active = False
        elif active:
            records.append([line[:11], *line[11:].split()])
    return records


def _assert_destination(path: Path, before: bytes | None) -> None:
    if before is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == before


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "multi.frcmod"
    path.write_text(
        "IO-10 synthetic\n\nDIHE\n"
        "h1-c3-c2-h2  2 2.46913578 37.123456789 -4.0 first component\n"
        "h2-c2-c3-h1  3 6.70370367 -47.987654321 2.0 last component\n"
        "h2-c3-c2-h2  1 0.0 180.0 6.0 independent type\n\n"
        "IMPROPER\nh1-c3-c2-h2 0.5 180.0 2.0 improper control\n\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def improper_source(source: Path) -> Path:
    source.write_text(
        source.read_text(encoding="utf-8").replace(
            "0.5 180.0 2.0 improper control\n\n",
            "0.5 180.0 2.0 improper control\nh2-c3-c2-h1 0.7 0.0 3.0 second improper\n\n",
        ),
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize(
    "change", ["remove", "add", "missing-row", "duplicate-row", "changed-types", "proper-row", "rowless-new-fold"]
)
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_improper_template_requires_complete_one_to_one_binding(
    improper_source: Path, tmp_path: Path, change: str, explicit_template: bool, destination: str
) -> None:
    ff = load_amber_frcmod(improper_source)
    first, second = ff.improper_torsions
    updates = {
        "remove": (first,),
        "add": (first, second, replace(first, ff_row=None, env_id="x1-c3-c2-h2")),
        "missing-row": (replace(first, ff_row=999), second),
        "duplicate-row": (first, replace(first, force_constant=0.9)),
        "changed-types": (replace(first, env_id="h1-n3-c2-h2", elements=("H", "N", "C", "H")), second),
        "proper-row": (replace(first, ff_row=ff.proper_torsions[0].ff_row), second),
        "rowless-new-fold": (replace(first, ff_row=None, periodicity=4), second),
    }[change]
    ff = replace(ff, torsions=(*ff.proper_torsions, *updates))
    if explicit_template:
        ff = replace(ff, source_path=None, source_format=None)
    output = improper_source if destination == "source" else tmp_path / "invalid-improper-binding.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER IMPROPER.*(component|binding)"):
        save_amber_frcmod(ff, output, template_path=improper_source if explicit_template else None)
    _assert_destination(output, before)


@pytest.mark.parametrize("clear_rows", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("list_elements", [False, True])
def test_improper_template_binds_every_edit_without_changing_native_order(
    improper_source: Path,
    tmp_path: Path,
    clear_rows: bool,
    reverse_order: bool,
    explicit_template: bool,
    list_elements: bool,
) -> None:
    original = load_amber_frcmod(improper_source)
    updates = tuple(
        replace(
            term,
            elements=list(term.elements) if list_elements else term.elements,
            ff_row=None if clear_rows else term.ff_row,
            force_constant=term.force_constant + 0.25,
            phase=term.phase + 10.0,
        )
        for term in original.improper_torsions
    )
    ff = replace(original, torsions=(*original.proper_torsions, *(updates[::-1] if reverse_order else updates)))
    if explicit_template:
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(
        ff, tmp_path / "bound-improper.frcmod", template_path=improper_source if explicit_template else None
    )
    actual = load_amber_frcmod(output)
    assert len(actual.improper_torsions) == len(updates)
    for expected, emitted in zip(updates, actual.improper_torsions, strict=True):
        assert (emitted.env_id, emitted.elements, emitted.periodicity) == (
            expected.env_id,
            tuple(expected.elements),
            expected.periodicity,
        )
        assert emitted.force_constant == pytest.approx(expected.force_constant)
        assert emitted.phase == expected.phase
    assert "improper control" in output.read_text(encoding="utf-8")
    assert "second improper" in output.read_text(encoding="utf-8")
    if list_elements:
        assert all(isinstance(term.elements, list) for term in updates)


def test_original_negative_pn_regression(tmp_path: Path) -> None:
    source = tmp_path / "original.frcmod"
    source.write_text(
        "Synthetic\nDIHE\nX -c3-c3-X    1 1.0 0.0 -1.0\nX -c3-c3-X    1 2.0 180.0 2.0\n\n",
        encoding="utf-8",
    )
    ff = load_amber_frcmod(source)
    assert [t.periodicity for t in ff.torsions] == [1, 2]
    output = save_amber_frcmod(ff, tmp_path / "out.frcmod")
    assert [float(row[4]) for row in _records(output)] == [-1.0, 2.0]


@pytest.mark.parametrize("invalid_type", ["hydrogen", "c 3", "c\n3", "\u03b1", "#"])
@pytest.mark.parametrize("from_elements", [False, True])
@pytest.mark.parametrize("amplitude", [0.0, 1.0])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_standalone_improper_type_fields_reject_before_writing(
    source: Path,
    tmp_path: Path,
    invalid_type: str,
    from_elements: bool,
    amplitude: float,
    destination: str,
) -> None:
    term = _term("h1-c3-c2-h2", 2, amplitude, is_improper=True)
    term = (
        replace(term, env_id="", elements=(invalid_type, "C", "C", "H"))
        if from_elements
        else replace(term, env_id=f"{invalid_type}-c3-c2-h2")
    )
    ff = ForceField(functional_form=FunctionalForm.HARMONIC, torsions=(term,))
    output = source if destination == "source" else tmp_path / "invalid-improper-types.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER IMPROPER.*atom types"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, before)


@pytest.mark.parametrize("arity", [0, 3, 5])
@pytest.mark.parametrize("improper", [False, True])
def test_torsion_element_arity_cannot_disagree_with_native_type_quad(
    tmp_path: Path, arity: int, improper: bool
) -> None:
    term = replace(_term("h1-c3-c2-h2", 2, is_improper=improper), elements=("H",) * arity)
    ff = ForceField(functional_form=FunctionalForm.HARMONIC, torsions=(term,))
    output = tmp_path / "bad-arity.frcmod"
    output.write_bytes(b"preserve destination")
    with pytest.raises(ValueError, match="AMBER.*four.*elements"):
        save_amber_frcmod(ff, output)
    assert output.read_bytes() == b"preserve destination"


@pytest.mark.parametrize("divisor", [1e100, 1e308, sys.float_info.max])
@pytest.mark.parametrize("existing", [False, True])
def test_large_finite_source_idivf_keeps_float_conversion_bounded(
    tmp_path: Path, divisor: float, existing: bool
) -> None:
    source = tmp_path / "large-divisor.frcmod"
    source.write_text(
        f"Synthetic\nDIHE\nh1-c3-c2-h2 {divisor:.0f} {divisor!r} 180.0 2.0\n\n",
        encoding="utf-8",
    )
    ff = load_amber_frcmod(source)
    assert ff.proper_torsions[0].force_constant == 1.0
    output = save_amber_frcmod(ff, tmp_path / "large-divisor-out.frcmod")
    assert load_amber_frcmod(output).proper_torsions[0].force_constant == 1.0
    changed = replace(ff, torsions=(replace(ff.proper_torsions[0], force_constant=sys.float_info.max),))
    rejected = tmp_path / "overflow.frcmod"
    if existing:
        rejected.write_bytes(b"preserve existing output")
    before = rejected.read_bytes() if existing else None
    with pytest.raises(ValueError, match="AMBER DIHE.*finite"):
        save_amber_frcmod(changed, rejected)
    _assert_destination(rejected, before)


@pytest.mark.parametrize("template", [False, True])
def test_single_component_and_improper_control(tmp_path: Path, template: bool) -> None:
    source = tmp_path / "single.frcmod"
    source.write_text(
        "Synthetic\nDIHE\nh1-c3-c2-h2 2 3.0 180.0 3.0\n\nIMPROPER\nh1-c3-c2-h2 0.5 180.0 2.0\n\n",
        encoding="utf-8",
    )
    ff = load_amber_frcmod(source)
    if not template:
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(ff, tmp_path / "single-out.frcmod")
    assert [float(row[4]) for row in _records(output)] == [3.0]
    actual = load_amber_frcmod(output)
    assert [(t.periodicity, t.force_constant, t.phase, t.is_improper) for t in actual.torsions] == [
        (3, 1.5, 180.0, False),
        (2, 0.5, 180.0, True),
    ]


@pytest.mark.parametrize("field", ["force_constant", "phase"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_nonfinite_improper_rejects_before_writing(
    source: Path, tmp_path: Path, field: str, value: float, mode: str, destination: str
) -> None:
    ff = load_amber_frcmod(source)
    ff = replace(ff, torsions=tuple(replace(t, **{field: value}) if t.is_improper else t for t in ff.torsions))
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "invalid-improper.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER IMPROPER.*finite"):
        save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
    _assert_destination(output, before)


@pytest.mark.parametrize("mode", ["implicit-template", "explicit-template", "standalone"])
@pytest.mark.parametrize("edited", [False, True])
def test_roundtrips_keep_groups_amplitudes_and_explicit_phases(
    source: Path, tmp_path: Path, mode: str, edited: bool
) -> None:
    ff = load_amber_frcmod(source)
    if edited:
        ff = replace(
            ff,
            torsions=tuple(
                replace(t, force_constant=t.force_constant + 0.123456789, phase=t.phase + 12.3456789)
                if not t.is_improper
                else t
                for t in ff.torsions
            ),
        )
    expected = [(t.env_id, t.periodicity, t.force_constant, t.phase, t.is_improper) for t in ff.torsions]
    output = tmp_path / "roundtrip.frcmod"
    for _ in range(3):
        if mode != "implicit-template":
            ff = replace(ff, source_path=None, source_format=None)
        save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
        records = _records(output)
        assert [float(row[4]) for row in records] == [-4.0, 2.0, 6.0]
        assert [int(row[1]) for row in records] == ([1, 1, 1] if mode == "standalone" else [2, 3, 1])
        assert [row[0] for row in records] == ["h1-c3-c2-h2", "h2-c2-c3-h1", "h2-c3-c2-h2"]
        ff = load_amber_frcmod(output)
        actual = [(t.env_id, t.periodicity, t.force_constant, t.phase, t.is_improper) for t in ff.torsions]
        for before, after in zip(expected, actual, strict=True):
            assert before[:2] == after[:2] and before[4] == after[4]
            assert after[2:4] == pytest.approx(before[2:4], rel=1e-14, abs=1e-14)
        assert float(_records(output, "IMPROPER")[0][3]) == 2.0


@pytest.mark.parametrize("clear_rows", [False, True])
def test_template_reorders_bind_by_component_without_using_tuple_position(
    source: Path, tmp_path: Path, clear_rows: bool
) -> None:
    ff = load_amber_frcmod(source)
    ff = replace(
        ff,
        torsions=tuple(
            replace(
                t, ff_row=None if clear_rows and not t.is_improper else t.ff_row, force_constant=t.force_constant + 1
            )
            for t in reversed(ff.torsions)
        ),
    )
    output = save_amber_frcmod(ff, tmp_path / "reordered.frcmod")
    assert [float(row[4]) for row in _records(output)] == [-4, 2, 6]
    actual = load_amber_frcmod(output)
    expected = {t.env_id: t.force_constant for t in ff.proper_torsions}
    assert {t.env_id: t.force_constant for t in actual.proper_torsions} == pytest.approx(expected, rel=1e-14)
    assert "first component" in output.read_text()
    assert "last component" in output.read_text()


@pytest.mark.parametrize("clear_rows", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("list_elements", [False, True])
def test_template_accepts_full_reversal_without_changing_native_values(
    source: Path, tmp_path: Path, clear_rows: bool, reverse_order: bool, list_elements: bool
) -> None:
    source.write_text(source.read_text().replace("c2", "n2").replace("h2", "o1"), encoding="utf-8")
    original = load_amber_frcmod(source)
    assert all(t.elements != t.elements[::-1] for t in original.proper_torsions)
    torsions = tuple(
        replace(
            t,
            elements=list(t.elements[::-1]) if list_elements else t.elements[::-1],
            env_id="-".join(reversed(t.env_id.split("-"))),
            ff_row=None if clear_rows else t.ff_row,
            force_constant=t.force_constant + 0.123456789,
            phase=t.phase + 12.3456789,
        )
        if not t.is_improper
        else t
        for t in original.torsions
    )
    if reverse_order:
        torsions = tuple(reversed(torsions))
    output = save_amber_frcmod(replace(original, torsions=torsions), tmp_path / "reversed.frcmod")
    rows = _records(output)
    assert [float(row[4]) for row in rows] == [-4.0, 2.0, 6.0]
    assert [int(row[1]) for row in rows] == [2, 3, 1]
    assert [row[0] for row in rows] == [row[0] for row in _records(source)]
    actual = load_amber_frcmod(output)
    for before, after in zip(original.proper_torsions, actual.proper_torsions, strict=True):
        assert after.env_id == before.env_id
        assert after.elements == before.elements
        assert after.periodicity == before.periodicity
        assert after.force_constant == pytest.approx(before.force_constant + 0.123456789, rel=1e-14)
        assert after.phase == pytest.approx(before.phase + 12.3456789, rel=1e-14)
    assert actual.improper_torsions == original.improper_torsions
    assert "first component" in output.read_text()
    assert "last component" in output.read_text()


@pytest.mark.parametrize("mismatch", ["source-row", "partial-types", "partial-elements"])
@pytest.mark.parametrize("existing", [False, True])
def test_reversal_binding_still_rejects_unrelated_identity(
    source: Path, tmp_path: Path, mismatch: str, existing: bool
) -> None:
    source.write_text(source.read_text().replace("c2", "n2").replace("h2", "o1"), encoding="utf-8")
    ff = load_amber_frcmod(source)
    first = ff.proper_torsions[0]
    changes = {
        "source-row": {"ff_row": 999},
        "partial-types": {"ff_row": None, "env_id": "h1-n2-c3-o1"},
        "partial-elements": {"ff_row": None, "elements": ("H", "N", "C", "O")},
    }
    ff = replace(ff, torsions=(replace(first, **changes[mismatch]), *ff.torsions[1:]))
    output = tmp_path / "invalid-binding.frcmod"
    if existing:
        output.write_bytes(b"preserve output")
    with pytest.raises(ValueError, match="AMBER DIHE.*binding"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, b"preserve output" if existing else None)


@pytest.fixture
def oriented_source(tmp_path: Path) -> Path:
    path = tmp_path / "oriented.frcmod"
    path.write_text(
        "Distinct elements and source orientations\n\nDIHE\n"
        "h1-c3-n2-o2 2 2.5 37.125 -4.0 SCEE=1.2 first component\n"
        "o2-n2-c3-h1 3 6.75 -47.875 2.0 SCNB=2.0 last component\n"
        "h2-c3-n2-o2 1 0.5 180.0 4.0 independent type, same fold\n\n"
        "IMPROPER\nh1-c3-n2-o2     0.5000    180.0      2.0 improper control\n\n"
        "NONBON\nh1     1.0000     0.0150 unchanged metadata\n\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("clear_rows", [False, True])
@pytest.mark.parametrize("reverse", ["types", "elements", "both"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_template_reversal_binding_preserves_source_orientation(
    oriented_source: Path, tmp_path: Path, clear_rows: bool, reverse: str, destination: str
) -> None:
    original = load_amber_frcmod(oriented_source)
    source_rows = _records(oriented_source)
    source_text = oriented_source.read_text()
    ff = replace(
        original,
        torsions=tuple(
            replace(
                t,
                env_id="-".join(t.env_id.split("-")[::-1]) if reverse != "elements" else t.env_id,
                elements=t.elements[::-1] if reverse != "types" else t.elements,
                ff_row=None if clear_rows else t.ff_row,
                force_constant=t.force_constant + 0.125,
                phase=t.phase + 0.25,
            )
            if not t.is_improper
            else t
            for t in reversed(original.torsions)
        ),
    )
    output = oriented_source if destination == "source" else tmp_path / "out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace supported output")
    save_amber_frcmod(ff, output)
    rows = _records(output)
    assert [(row[0], row[1], row[4:]) for row in rows] == [(row[0], row[1], row[4:]) for row in source_rows]
    actual = load_amber_frcmod(output)
    for before, after in zip(original.proper_torsions, actual.proper_torsions, strict=True):
        assert (after.env_id, after.elements, after.ff_row, after.periodicity) == (
            before.env_id,
            before.elements,
            before.ff_row,
            before.periodicity,
        )
        assert after.force_constant == before.force_constant + 0.125
        assert after.phase == before.phase + 0.25
    assert output.read_text().split("IMPROPER")[1] == source_text.split("IMPROPER")[1]


@pytest.mark.parametrize("edit", ["wrong-row", "missing-row", "missing-fold", "changed-type", "permuted-elements"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_reversed_template_binding_does_not_reassign_components(
    oriented_source: Path, tmp_path: Path, edit: str, destination: str
) -> None:
    ff = load_amber_frcmod(oriented_source)
    tor = ff.torsions[0]
    tor = replace(tor, env_id="-".join(tor.env_id.split("-")[::-1]), elements=tor.elements[::-1])
    if edit == "wrong-row":
        tor = replace(tor, ff_row=ff.torsions[2].ff_row)
    elif edit == "missing-row":
        tor = replace(tor, ff_row=999)
    elif edit == "missing-fold":
        tor = replace(tor, ff_row=None, periodicity=3)
    elif edit == "changed-type":
        tor = replace(tor, ff_row=None, env_id="o2-n2-c3-h3")
    else:
        tor = replace(tor, ff_row=None, elements=("O", "C", "N", "H"))
    ff = replace(ff, torsions=(tor, *ff.torsions[1:]))
    output = oriented_source if destination == "source" else tmp_path / "out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER DIHE.*source-row binding"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, before)


@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_template_barrier_overflow_is_idivf_conversion(source: Path, tmp_path: Path, destination: str) -> None:
    ff = load_amber_frcmod(source)
    amplitude = 1e308
    divisor = int(_records(source)[0][1])
    assert divisor == 2 and math.isfinite(amplitude)
    assert not math.isfinite(amplitude * divisor)
    ff = replace(ff, torsions=(replace(ff.torsions[0], force_constant=amplitude), *ff.torsions[1:]))
    output = source if destination == "source" else tmp_path / "out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER DIHE exported amplitude and phase must be finite"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, before)


@pytest.mark.parametrize("template", [False, True])
@pytest.mark.parametrize(
    "value", [1e308, -1e308, sys.float_info.max, sys.float_info.min, math.ulp(0.0), -math.ulp(0.0)]
)
def test_finite_decimal_extremes_roundtrip(tmp_path: Path, template: bool, value: float) -> None:
    source = tmp_path / "finite.frcmod"
    source.write_text("Finite decimal control\n\nDIHE\nh1-c3-c2-h2 1 1.0 37.0 4.0\n\n", encoding="utf-8")
    ff = load_amber_frcmod(source)
    ff = replace(ff, torsions=(replace(ff.torsions[0], force_constant=value, phase=-value),))
    if not template:
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(ff, tmp_path / "out.frcmod")
    record = _records(output)[0]
    assert record[1] == "1"
    assert all("e" not in token.lower() for token in record[2:4])
    assert float(record[2]) == value
    assert float(record[3]) == -value
    actual = load_amber_frcmod(output).proper_torsions[0]
    assert actual.force_constant == value
    assert actual.phase == -value
    assert actual.periodicity == 4


def test_template_periodicity_edit_keeps_only_the_continuation_sign(source: Path, tmp_path: Path) -> None:
    ff = load_amber_frcmod(source)
    ff = replace(ff, torsions=(replace(ff.torsions[0], periodicity=3), *ff.torsions[1:]))
    output = save_amber_frcmod(ff, tmp_path / "changed-fold.frcmod")
    assert [float(row[4]) for row in _records(output)] == [-3.0, 2.0, 6.0]
    actual = load_amber_frcmod(output)
    assert [(t.periodicity, t.phase) for t in actual.proper_torsions] == [
        (3, 37.123456789),
        (2, -47.987654321),
        (6, 180.0),
    ]


@pytest.mark.parametrize("existing", [False, True])
def test_standalone_regroups_interleaved_types_not_source_metadata(tmp_path: Path, existing: bool) -> None:
    a = "h1-c3-c2-h2"
    ar = "h2-c2-c3-h1"
    b = "h2-c3-c2-h2"
    ff = ForceField(
        functional_form=FunctionalForm.HARMONIC,
        torsions=(
            _term(a, 4, 1.23456789, 30.123456789, ff_row=12, label="first"),
            _term(b, 3, 2.0, -45.0, ff_row=12, label="same source metadata"),
            _term(ar, 2, -3.0, 67.987654321, ff_row=999),
            _term(a, 1, 0.0, 0.0),
            _term(b, 1, 4.0, 180.0),
            _term(a, 2, 0.5, 180.0, is_improper=True),
        ),
    )
    output = tmp_path / "grouped.frcmod"
    if existing:
        output.write_bytes(b"replace supported output")
    save_amber_frcmod(ff, output)
    rows = _records(output)
    assert [row[0] for row in rows] == [a, ar, a, b, b]
    assert [float(row[4]) for row in rows] == [-4, -2, 1, -3, 1]
    assert [float(row[3]) for row in rows] == [30.123456789, 67.987654321, 0.0, -45.0, 180.0]
    assert [float(row[2]) for row in rows] == [1.23456789, -3.0, 0.0, 2.0, 4.0]
    assert len(load_amber_frcmod(output).proper_torsions) == 5
    assert float(_records(output, "IMPROPER")[0][3]) == 2.0


_BAD_SOURCES = [
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n", id="unterminated"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n\n", id="blank-before-terminator"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\nIMPROPER\nh1-c3-c2-h2 1 180 2\n", id="section-before-terminator"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\nh2-c3-c2-h2 1 2 30 2\n", id="interleaved-chain"),
    pytest.param("h1-c3-c2-h2 1 1 0 1\nh2-c2-c3-h1 1 2 30 2\n", id="redefined-reversed-group"),
    pytest.param(
        "h1-c3-c2-h2 1 1 0 1\nh2-c3-c2-h2 1 2 30 2\nh1-c3-c2-h2 1 3 60 3\n",
        id="noncontiguous-redefinition",
    ),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\nh1-c3-c2-h2 1 2 30 1\n", id="duplicate-periodicity"),
    pytest.param("h1-c3-c2-h2 1 1 0 0\n", id="zero-periodicity"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1.5\nh1-c3-c2-h2 1 2 30 2\n", id="fractional-periodicity"),
    pytest.param("           1 2 30 2\n", id="orphan-implicit-continuation"),
    pytest.param("1 2 30 2\n", id="orphan-compact-continuation"),
    pytest.param("1 -2 -3 -4 metadata\n", id="orphan-signed-compact-continuation"),
    pytest.param("h1-c3-c2-h2 1 1 0 1\n           1 2 30 2\n", id="implicit-after-positive-pn"),
    pytest.param("h1-c3-c2-h2 1 1 0 1\n1 -2 -3 -4 metadata\n", id="signed-compact-after-positive-pn"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n           1 2 30 1\n", id="implicit-duplicate-fold"),
    pytest.param("h1-c3-c2-h2 1 1 0 -4\n1 -2 -3 -4 metadata\n1 1 0 5\n", id="signed-compact-duplicate-fold"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n           1 2 30 -2\n", id="implicit-unterminated"),
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n           junk 2 30 2\n", id="implicit-malformed-values"),
    *[
        pytest.param(f"h1-c3-c2-h2 {divisor} 1 0 1\n", id=f"invalid-divisor-{divisor}")
        for divisor in ("1.5", "-1.5", "nan", "inf", "-inf")
    ],
]


@pytest.mark.parametrize("records", _BAD_SOURCES)
def test_load_rejects_ambiguous_or_unsupported_chains(tmp_path: Path, records: str) -> None:
    source = tmp_path / "invalid.frcmod"
    source.write_text("Synthetic\nDIHE\n" + records, encoding="utf-8")
    with pytest.raises(ValueError, match="AMBER.*DIHE"):
        load_amber_frcmod(source)


@pytest.mark.parametrize("records", _BAD_SOURCES)
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_invalid_template_never_replaces_destination(tmp_path: Path, records: str, destination: str) -> None:
    source = tmp_path / "invalid.frcmod"
    source.write_text("Synthetic\nDIHE\n" + records, encoding="utf-8")
    output = source if destination == "source" else tmp_path / "out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER.*DIHE"):
        save_amber_frcmod(ForceField(functional_form=FunctionalForm.HARMONIC), output, template_path=source)
    _assert_destination(output, before)


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda ff: replace(ff, torsions=ff.torsions[1:]), id="removed-component"),
        pytest.param(lambda ff: replace(ff, torsions=(*ff.torsions, _term("h1-c3-c2-h2", 3))), id="added-component"),
        pytest.param(
            lambda ff: replace(ff, torsions=(replace(ff.torsions[0], ff_row=ff.torsions[1].ff_row), *ff.torsions[1:])),
            id="duplicate-source-row",
        ),
        pytest.param(
            lambda ff: replace(ff, torsions=(replace(ff.torsions[0], env_id="h2-c3-c2-h2"), *ff.torsions[1:])),
            id="changed-typed-environment",
        ),
        pytest.param(
            lambda ff: replace(ff, torsions=(replace(ff.torsions[0], is_improper=True), *ff.torsions[1:])),
            id="changed-interaction-kind",
        ),
        pytest.param(
            lambda ff: replace(ff, torsions=(replace(ff.torsions[0], periodicity=2), *ff.torsions[1:])),
            id="duplicate-edited-periodicity",
        ),
        pytest.param(
            # The first source component has IDIVF=2, so the exported PK overflows.
            lambda ff: replace(ff, torsions=(replace(ff.torsions[0], force_constant=1e308), *ff.torsions[1:])),
            id="barrier-overflow",
        ),
    ],
)
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_unrepresentable_template_edits_preserve_destination(
    source: Path, tmp_path: Path, edit: Callable[[ForceField], ForceField], destination: str
) -> None:
    ff = edit(load_amber_frcmod(source))
    output = source if destination == "source" else tmp_path / "out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER.*DIHE"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, before)


@pytest.mark.parametrize(
    "torsions",
    [
        pytest.param((_term("h1-c3-c2-h2", -1),), id="negative-canonical-periodicity"),
        pytest.param((_term("h1-c3-c2-h2", 0),), id="zero-canonical-periodicity"),
        pytest.param((_term("h1-c3-c2-h2", 1), _term("h2-c2-c3-h1", 1, phase=90)), id="duplicate-fold"),
        pytest.param((_term("h1-c3-c2-h2", 1, k=float("inf")),), id="nonfinite-amplitude"),
        pytest.param((_term("h1-c3-c2-h2", 1, phase=float("nan")),), id="nonfinite-phase"),
        pytest.param((_term("hydrogen-c3-c2-h2", 1),), id="unrepresentable-types"),
        pytest.param((_term("h1-c3", 1),), id="incomplete-typed-environment"),
        pytest.param((_term("h1-c3--h2", 1),), id="empty-typed-component"),
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_standalone_rejects_unrepresentable_groups_before_writing(
    tmp_path: Path, torsions: tuple[TorsionParam, ...], existing: bool
) -> None:
    ff = ForceField(functional_form=FunctionalForm.HARMONIC, torsions=torsions)
    output = tmp_path / "out.frcmod"
    if existing:
        output.write_bytes(b"preserve output")
    with pytest.raises(ValueError, match="AMBER.*DIHE"):
        save_amber_frcmod(ff, output)
    _assert_destination(output, b"preserve output" if existing else None)


@pytest.mark.parametrize(
    ("amplitude", "phase"),
    [
        (0.123456789, 37.123456789),
        (math.ulp(0.0), -math.ulp(0.0)),
        (-math.ulp(0.0), 47.125),
        (sys.float_info.max, -sys.float_info.max),
        (-1e308, 180.125),
        (0.0, 37.125),
    ],
)
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_improper_decimal_precision_and_native_numeric_offset(
    source: Path, tmp_path: Path, amplitude: float, phase: float, mode: str, destination: str
) -> None:
    ff = load_amber_frcmod(source)
    ff = replace(
        ff,
        torsions=tuple(replace(t, force_constant=amplitude, phase=phase) if t.is_improper else t for t in ff.torsions),
    )
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "precise-improper.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace supported output")
    save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
    for _ in range(2):
        actual = load_amber_frcmod(output)
        tor = actual.improper_torsions[0]
        assert (tor.force_constant, tor.phase, tor.periodicity) == (amplitude, phase, 2)
        assert tor.ff_row is not None
        line = output.read_text().splitlines()[tor.ff_row - 1]
        assert line[:11] == "h1-c3-c2-h2"
        assert all("e" not in token.lower() for token in line[11:].split()[:3])
        assert [float(token) for token in line[15:].split()[:3]] == [amplitude, phase, 2.0]
        save_amber_frcmod(actual, output)


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize("periodicity", [True, False, 0, -1, 2.5, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_all_torsion_periodicities_reject_before_writing(
    source: Path, tmp_path: Path, improper: bool, periodicity: float, mode: str, destination: str
) -> None:
    ff = load_amber_frcmod(source)
    target = next(t for t in ff.torsions if t.is_improper == improper)
    ff = replace(ff, torsions=tuple(replace(t, periodicity=periodicity) if t is target else t for t in ff.torsions))
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "invalid-periodicity.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve destination")
    before = output.read_bytes() if output.exists() else None
    section = "IMPROPER" if improper else "DIHE"
    with pytest.raises(ValueError, match=f"AMBER {section} canonical periodicity must be a positive integer"):
        save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
    _assert_destination(output, before)


@pytest.mark.parametrize("periodicity", [1, 2.0, 7])
@pytest.mark.parametrize("template", [False, True])
def test_positive_integral_improper_periodicity_remains_supported(
    source: Path, tmp_path: Path, periodicity: float, template: bool
) -> None:
    ff = load_amber_frcmod(source)
    ff = replace(ff, torsions=tuple(replace(t, periodicity=periodicity) if t.is_improper else t for t in ff.torsions))
    if not template:
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(ff, tmp_path / "integral.frcmod")
    assert load_amber_frcmod(output).improper_torsions[0].periodicity == periodicity


@pytest.mark.parametrize("divisor", [0, -2, 2])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_encoded_divisor_uses_native_effective_value(tmp_path: Path, divisor: int, mode: str, destination: str) -> None:
    source = tmp_path / "divisor.frcmod"
    effective = divisor or 1
    source.write_text(
        f"Divisor control\n\nDIHE\nh1-c3-c2-h2 {divisor} {1.25 * effective} 37.125 2.0 metadata\n\n",
        encoding="utf-8",
    )
    ff = load_amber_frcmod(source)
    assert ff.proper_torsions[0].force_constant == 1.25
    ff = replace(ff, torsions=(replace(ff.torsions[0], force_constant=-2.375),))
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "divisor-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace output")
    save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
    record = _records(output)[0]
    assert int(record[1]) == (1 if mode == "standalone" else divisor)
    assert float(record[2]) == -2.375 * (1 if mode == "standalone" else effective)
    assert load_amber_frcmod(output).proper_torsions[0].force_constant == -2.375
    if mode != "standalone":
        assert record[5:] == ["metadata"]


@pytest.fixture
def implicit_source(tmp_path: Path) -> Path:
    source = tmp_path / "implicit.frcmod"
    source.write_text(
        "Implicit orientation control\n\nDIHE\n"
        "h1-c3-n2-o2 2 2.5 37.125 -4.0 first SCEE=1.2\n"
        "o2-n2-c3-h1 3 6.75 -47.875 -2.0 reversed\n"
        "           0 0.125 12.375 -1.0 inherited reversed\n"
        "           2 1.75 -19.625 3.0 final component\n"
        "h2-c3-n2-o2 1 0.5 180.0 6.0 independent\n\n"
        "IMPROPER\nh1-c3-n2-o2     0.5 180.0 2.0 improper control\n\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("clear_rows", [False, True])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_implicit_continuations_preserve_orientation_and_template_prefix(
    implicit_source: Path, tmp_path: Path, clear_rows: bool, mode: str, destination: str
) -> None:
    original = load_amber_frcmod(implicit_source)
    expected_types = ["h1-c3-n2-o2", "o2-n2-c3-h1", "o2-n2-c3-h1", "o2-n2-c3-h1", "h2-c3-n2-o2"]
    assert [t.env_id for t in original.proper_torsions] == expected_types
    assert original.proper_torsions[2].elements == ("O", "N", "C", "H")
    updates = tuple(
        replace(
            t,
            elements=list(t.elements),
            ff_row=None if clear_rows else t.ff_row,
            force_constant=t.force_constant + 0.25,
            phase=t.phase + 0.125,
        )
        for t in original.torsions
    )
    ff = replace(original, torsions=updates if mode == "standalone" else updates[::-1])
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = implicit_source if destination == "source" else tmp_path / "implicit-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace supported output")
    save_amber_frcmod(ff, output, template_path=implicit_source if mode == "explicit-template" else None)
    records = _records(output)
    expected_prefixes = (
        expected_types if mode == "standalone" else [*expected_types[:2], " " * 11, " " * 11, expected_types[4]]
    )
    assert [row[0] for row in records] == expected_prefixes
    assert [int(row[1]) for row in records] == ([1] * 5 if mode == "standalone" else [2, 3, 0, 2, 1])
    assert [float(row[4]) for row in records] == [-4.0, -2.0, -1.0, 3.0, 6.0]
    actual = load_amber_frcmod(output)
    for expected, emitted in zip(updates, actual.torsions, strict=True):
        assert (emitted.env_id, emitted.elements, emitted.periodicity) == (
            expected.env_id,
            tuple(expected.elements),
            expected.periodicity,
        )
        assert emitted.force_constant == expected.force_constant
        assert emitted.phase == expected.phase
        assert isinstance(expected.elements, list)
    if mode != "standalone":
        assert "inherited reversed" in output.read_text()
        assert "first SCEE=1.2" in output.read_text()


@pytest.mark.parametrize("indent", ["", "  ", " " * 11, " " * 15])
def test_compact_implicit_numeric_rows_inherit_the_last_orientation(tmp_path: Path, indent: str) -> None:
    source = tmp_path / "compact.frcmod"
    source.write_text(f"Compact\nDIHE\no2-n2-c3-h1 1 1.25 37.125 -1\n{indent}2 3.5 -12.125 2\n\n")
    ff = load_amber_frcmod(source)
    assert [t.env_id for t in ff.torsions] == ["o2-n2-c3-h1"] * 2
    output = save_amber_frcmod(ff, tmp_path / "compact-out.frcmod")
    assert output.read_text().splitlines()[3].lstrip()[0].isdigit()
    assert [t.force_constant for t in load_amber_frcmod(output).torsions] == [1.25, 1.75]


@pytest.mark.parametrize("comment_prefix", ["#", "   #", "\t#"])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_dihe_comments_are_not_parameters_and_templates_preserve_them(
    implicit_source: Path, tmp_path: Path, comment_prefix: str, mode: str, destination: str
) -> None:
    original = load_amber_frcmod(implicit_source)
    lines = []
    active = False
    for line in implicit_source.read_text().splitlines(keepends=True):
        lines.append(line)
        if line.strip() == "DIHE":
            active = True
        elif not line.strip():
            active = False
        if active:
            lines.append(f"{comment_prefix} 1 -2 -3 -4 metadata after {len(lines)}\n")
    implicit_source.write_text("".join(lines), encoding="utf-8")
    before = implicit_source.read_bytes()
    comments = [line for line in before.splitlines(keepends=True) if line.lstrip().startswith(b"#")]
    ff = load_amber_frcmod(implicit_source)
    assert len(ff.proper_torsions) == 5
    assert [replace(t, ff_row=None, label="") for t in ff.torsions] == [
        replace(t, ff_row=None, label="") for t in original.torsions
    ]
    assert all(not lines[t.ff_row - 1].lstrip().startswith("#") for t in ff.torsions)
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = implicit_source if destination == "source" else tmp_path / "comments-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace output")
    save_amber_frcmod(ff, output, template_path=implicit_source if mode == "explicit-template" else None)
    actual = load_amber_frcmod(output)
    assert [replace(t, ff_row=None, label="") for t in actual.torsions] == [
        replace(t, ff_row=None, label="") for t in ff.torsions
    ]
    emitted_comments = [
        line for line in output.read_bytes().splitlines(keepends=True) if line.lstrip().startswith(b"#")
    ]
    assert emitted_comments == ([] if mode == "standalone" else comments)
    if destination != "source":
        assert implicit_source.read_bytes() == before


@pytest.mark.parametrize("divisor", [0, 1, -1])
@pytest.mark.parametrize("indent", ["", "  ", " " * 11])
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_compact_signed_values_are_not_mistaken_for_atom_types(
    tmp_path: Path, divisor: int, indent: str, mode: str, destination: str
) -> None:
    source = tmp_path / "signed-compact.frcmod"
    source.write_text(
        f"Signed compact\nDIHE\no2-n2-c3-h1 1 1.25 37.125 -1\n"
        f"{indent}{divisor} -2 -3 -4 metadata\n"
        "           2 7 45 5 final\n\n",
        encoding="utf-8",
    )
    ff = load_amber_frcmod(source)
    assert [t.env_id for t in ff.torsions] == ["o2-n2-c3-h1"] * 3
    assert [(t.periodicity, t.force_constant, t.phase) for t in ff.torsions] == [
        (1, 1.25, 37.125),
        (4, -2.0 / (divisor or 1), -3.0),
        (5, 3.5, 45.0),
    ]
    if mode != "implicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "signed-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace output")
    save_amber_frcmod(ff, output, template_path=source if mode == "explicit-template" else None)
    assert [replace(t, ff_row=None, label="") for t in load_amber_frcmod(output).torsions] == [
        replace(t, ff_row=None, label="") for t in ff.torsions
    ]
    if mode != "standalone":
        line = next(line for line in output.read_text().splitlines() if line.endswith("metadata"))
        assert line[: len(indent)] == indent
        assert [float(token) for token in line.split()[:4]] == [divisor, -2.0, -3.0, -4.0]


@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_ambiguous_numeric_types_and_compact_values_reject_without_guessing(tmp_path: Path, destination: str) -> None:
    source = tmp_path / "ambiguous.frcmod"
    source.write_text("Ambiguous\nDIHE\n1 -2 -3 -4   1 2 30 -1\n1 -2 -3 -4   1 2 30 2\n\n")
    with pytest.raises(ValueError, match="AMBER DIHE.*ambiguous"):
        load_amber_frcmod(source)
    output = source if destination == "source" else tmp_path / "ambiguous-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER DIHE.*ambiguous"):
        save_amber_frcmod(ForceField(functional_form=FunctionalForm.HARMONIC), output, template_path=source)
    _assert_destination(output, before)


def test_numeric_atom_types_at_group_start_remain_explicit(tmp_path: Path) -> None:
    source = tmp_path / "numeric-types.frcmod"
    source.write_text("Numeric types\nDIHE\n1 -2 -3 -4   1 2 30 2\n\n")
    ff = load_amber_frcmod(source)
    assert (ff.torsions[0].env_id, ff.torsions[0].force_constant, ff.torsions[0].phase) == ("1-2-3-4", 2.0, 30.0)
    output = save_amber_frcmod(ff, tmp_path / "numeric-types-out.frcmod")
    assert load_amber_frcmod(output).torsions == ff.torsions


_UNSUPPORTED_IMPROPER_PN = ("0", "-0.0", "-1", "-2", "2.5", "-2.5", "nan", "inf", "-inf")


@pytest.mark.parametrize("pn", _UNSUPPORTED_IMPROPER_PN)
@pytest.mark.parametrize("amplitude", [0.0, 0.5])
def test_raw_improper_periodicity_is_not_silently_coerced(tmp_path: Path, pn: str, amplitude: float) -> None:
    source = tmp_path / "raw-improper.frcmod"
    source.write_text(f"Raw improper\nIMPROPER\nh1-c3-n2-o2    {amplitude} 37.125 {pn} metadata\n\n")
    with pytest.raises(ValueError, match="AMBER IMPROPER.*PN"):
        load_amber_frcmod(source)


@pytest.mark.parametrize("pn", _UNSUPPORTED_IMPROPER_PN)
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_unsupported_raw_improper_never_reaches_unchanged_copy(
    tmp_path: Path, pn: str, explicit_template: bool, destination: str
) -> None:
    source = tmp_path / "raw-improper.frcmod"
    old_effective = 1 if pn in ("0", "-0.0", "-1") else 2
    source.write_text(f"Raw improper\nIMPROPER\nh1-c3-n2-o2    0.5 37.125 {old_effective}.0 metadata\n\n")
    ff = load_amber_frcmod(source)
    source.write_text(f"Raw improper\nIMPROPER\nh1-c3-n2-o2    0.5 37.125 {pn} metadata\n\n")
    original = source.read_bytes()
    if explicit_template:
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "raw-improper-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"preserve destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="AMBER IMPROPER.*PN"):
        save_amber_frcmod(ff, output, template_path=source if explicit_template else None)
    _assert_destination(output, before)
    assert source.read_bytes() == original


@pytest.mark.parametrize("pn", ["1", "2.0", "+2.0000", "0002.0", "7"])
@pytest.mark.parametrize("explicit_template", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_valid_raw_improper_spelling_still_copies_unchanged(
    tmp_path: Path, pn: str, explicit_template: bool, destination: str
) -> None:
    source = tmp_path / "valid-improper.frcmod"
    source.write_text(f"Valid improper\nIMPROPER\nh1-c3-n2-o2    0.123456789 37.125 {pn} metadata\n\n")
    original = source.read_bytes()
    ff = load_amber_frcmod(source)
    if explicit_template:
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "valid-improper-out.frcmod"
    if destination == "existing":
        output.write_bytes(b"replace output")
    save_amber_frcmod(ff, output, template_path=source if explicit_template else None)
    assert output.read_bytes() == original
    assert load_amber_frcmod(output).improper_torsions == ff.improper_torsions


@pytest.mark.parametrize("mode", ["template", "standalone"])
def test_parmed_implicit_nonzero_divisors_and_precise_improper(
    implicit_source: Path, tmp_path: Path, mode: str
) -> None:
    amber = pytest.importorskip("parmed.amber", reason="Optional ParmEd parser is not installed")
    # ParmEd 4.3.1 does not implement LEaP's zero-IDIVF-as-one convention.
    implicit_source.write_text(implicit_source.read_text().replace("           0 ", "           1 "))
    ff = load_amber_frcmod(implicit_source)
    ff = replace(
        ff,
        torsions=tuple(
            replace(t, force_constant=math.ulp(0.0), phase=-47.123456789) if t.is_improper else t for t in ff.torsions
        ),
    )
    if mode == "standalone":
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(ff, tmp_path / "parmed-implicit.frcmod")
    parsed = amber.AmberParameterSet(str(output))
    assert [(t.per, t.phi_k, t.phase) for t in parsed.dihedral_types[("h1", "c3", "n2", "o2")]] == [
        (t.periodicity, t.force_constant, t.phase) for t in ff.proper_torsions[:4]
    ]
    improper = next(iter(parsed.improper_periodic_types.values()))
    assert (improper.phi_k, improper.phase, improper.per) == (math.ulp(0.0), -47.123456789, 2)


@pytest.mark.parametrize("mode", ["template", "standalone"])
def test_parmed_reader_retains_all_components(source: Path, tmp_path: Path, mode: str) -> None:
    amber = pytest.importorskip("parmed.amber", reason="Optional ParmEd parser is not installed")

    ff = load_amber_frcmod(source)
    if mode == "standalone":
        ff = replace(ff, source_path=None, source_format=None)
    output = save_amber_frcmod(ff, tmp_path / "parmed.frcmod")
    parsed = amber.AmberParameterSet(str(output))
    types = ("h1", "c3", "c2", "h2")
    assert parsed.dihedral_types[types] is parsed.dihedral_types[types[::-1]]
    assert [(t.per, t.phi_k, t.phase) for t in parsed.dihedral_types[types]] == [
        (4, pytest.approx(1.23456789, rel=1e-14), 37.123456789),
        (2, pytest.approx(2.23456789, rel=1e-14), -47.987654321),
    ]
    assert len(parsed.dihedral_types[("h2", "c3", "c2", "h2")]) == 1
