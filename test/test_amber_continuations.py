"""AMBER continuation encoding tests, not AMBER engine/runtime tests."""

from __future__ import annotations

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
    pytest.param("h1-c3-c2-h2 1 1 0 -1\n           1 2 30 2\n", id="implicit-type-continuation"),
    pytest.param("h1-c3-c2-h2 0 1 0 1\n", id="zero-divisor"),
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
