"""Distinguish native -3 substructure titles from ordinary C comments."""

from pathlib import Path

import pytest

from q2mm.io.mm3 import _format_mm3_angle_line, _mm3_import_ff, load_mm3_fld
from test.test_ffs import FF_PATH, _substructure_row


def _selected_block() -> str:
    return "-3\n C  OPT valid\n 9  C1-C2-C3 \n-2\n" + _substructure_row(" 2", ("1", "2", "3"), 110.0, 0.5) + "-3\n"


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("has_valid_prefix", [False, True])
@pytest.mark.parametrize("following", ["eof", "format", "parameter", "comment", "title"])
def test_declared_selected_title_requires_pattern(
    tmp_path: Path, include_standard: bool, has_valid_prefix: bool, following: str
) -> None:
    prefix = _selected_block() if has_valid_prefix else "-3\n"
    title = " C  OPT malformed\n"
    tails = {
        "eof": "",
        "format": "-2\n-3\n",
        "parameter": _substructure_row(" 2", ("1", "2", "3"), 120.0, 0.7) + "-3\n",
        "comment": " C  missing pattern here\n-3\n",
        "title": " C  OPT next\n 9  C1-C2-C3 \n-2\n-3\n",
    }
    path = tmp_path / "malformed.fld"
    path.write_text(prefix + title + tails[following], encoding="utf-8")
    title_row = len(prefix.splitlines()) + 1
    error_row = title_row if following == "eof" else title_row + 1
    with pytest.raises(ValueError, match=rf"malformed\.fld: row {error_row}.*OPT malformed.*pattern"):
        load_mm3_fld(path, include_standard=include_standard)


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("pattern", [" 9\n", " 9  \n", " 9  ??? \n"])
def test_selected_title_with_empty_or_invalid_pattern_still_rejects(
    tmp_path: Path, include_standard: bool, pattern: str
) -> None:
    path = tmp_path / "pattern.fld"
    path.write_text("-3\n C  OPT invalid\n" + pattern + "-3\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"OPT invalid.*pattern"):
        load_mm3_fld(path, include_standard=include_standard)


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("trailing", [" ???", " C3", " # annotation"])
def test_selected_pattern_rejects_trailing_non_whitespace(
    tmp_path: Path, include_standard: bool, trailing: str
) -> None:
    path = tmp_path / "trailing-pattern.fld"
    path.write_text(
        "-3\n C  OPT selected\n 9  C1-C2"
        + trailing
        + "\n-2\n"
        + _substructure_row(" 1", ("1", "2"), 1.1, 2.0)
        + "-3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"OPT selected.*pattern"):
        load_mm3_fld(path, include_standard=include_standard)


@pytest.mark.parametrize("trailing", ["", " ", " \t "])
def test_selected_pattern_accepts_only_trailing_whitespace(tmp_path: Path, trailing: str) -> None:
    path = tmp_path / "valid-pattern.fld"
    path.write_text(
        "-3\n C  OPT selected\n 9  C1-C2"
        + trailing
        + "\n-2\n"
        + _substructure_row(" 1", ("1", "2"), 1.1, 2.0)
        + "-3\n",
        encoding="utf-8",
    )
    assert len(load_mm3_fld(path).bonds) == 1


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("prefix", ["", "-2\n", "-3\n-2\n"], ids=["no-format", "parameter-format", "explicit-reset"])
def test_opt_comment_outside_header_format_is_not_a_title(tmp_path: Path, include_standard: bool, prefix: str) -> None:
    standard = _format_mm3_angle_line(["C1", "C2", "C3"], 109.0, 0.3)
    path = tmp_path / "comments.fld"
    path.write_text(
        prefix + " C  OPT is ordinary comment text here\n" + standard + " C  OPT trailing comment",
        encoding="utf-8",
    )
    ff = load_mm3_fld(path, include_standard=include_standard)
    assert len(ff.angles) == int(include_standard)
    if include_standard:
        assert ff.angles[0].equilibrium == 109.0
        assert ff.angles[0].env_id == "C1-C2-C3"


@pytest.mark.parametrize("prefix", [" ", "\t"], ids=["space", "tab"])
@pytest.mark.parametrize("include_standard", [False, True])
def test_abbreviated_pattern_lookahead_matches_parser_whitespace(
    tmp_path: Path, prefix: str, include_standard: bool
) -> None:
    path = tmp_path / "abbreviated.fld"
    path.write_text(
        " C  OPT abbreviated\n" + prefix + "9  C1-C2 \n" + _substructure_row(" 1", ("1", "2"), 1.1, 2.0) + "-3\n",
        encoding="utf-8",
    )
    ff = load_mm3_fld(path, include_standard=include_standard)
    assert len(ff.bonds) == 1
    assert ff.bonds[0].elements == ("C", "C")
    assert ff.bonds[0].env_id == "C1-C2"
    assert ff.bonds[0].ff_row == 3


def test_native_title_format_persists_until_an_explicit_format_change(tmp_path: Path) -> None:
    path = tmp_path / "persistent-format.fld"
    path.write_text("-3\n\n C  OPT declared title\nnot a pattern\n-3\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"OPT declared title.*pattern"):
        load_mm3_fld(path)
    path.write_text(
        _selected_block()
        + "-2\n C  ordinary post-block comment\n"
        + _format_mm3_angle_line(["C1", "C2", "C3"], 109.0, 0.3),
        encoding="utf-8",
    )
    assert len(load_mm3_fld(path).angles) == 2


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("explicit_parameter_format", [False, True])
def test_comment_inside_selected_block_does_not_open_another_block(
    tmp_path: Path, include_standard: bool, explicit_parameter_format: bool
) -> None:
    path = tmp_path / "inside.fld"
    parameter_format = "-2\n" if explicit_parameter_format else ""
    path.write_text(
        "-3\n C  OPT selected\n 9  C1-C2-C3 \n"
        + parameter_format
        + " C  ordinary OPT note, not another title\n"
        + _substructure_row(" 2", ("1", "2", "3"), 110.0, 0.5)
        + "-3\n",
        encoding="utf-8",
    )
    ff = load_mm3_fld(path, include_standard=include_standard)
    assert len(ff.angles) == 1
    assert ff.angles[0].env_id == "C1-C2-C3"
    assert ff.angles[0].equilibrium == 110.0


def test_unselected_bad_pattern_keeps_existing_selection_policy(tmp_path: Path) -> None:
    path = tmp_path / "filtered.fld"
    path.write_text("-3\n C  FROZEN\nnot a pattern\n-3\n" + _selected_block(), encoding="utf-8")
    ff = load_mm3_fld(path, include_standard=False)
    assert len(ff.angles) == 1
    assert ff.angles[0].equilibrium == 110.0
    with pytest.raises(ValueError, match=r"FROZEN.*pattern"):
        load_mm3_fld(path)


@pytest.mark.parametrize("include_standard", [False, True])
@pytest.mark.parametrize("has_valid_prefix", [False, True])
@pytest.mark.parametrize("has_valid_suffix", [False, True])
def test_unselected_missing_pattern_does_not_consume_closing_marker(
    tmp_path: Path, include_standard: bool, has_valid_prefix: bool, has_valid_suffix: bool
) -> None:
    prefix = _selected_block() if has_valid_prefix else ""
    suffix = _selected_block().removeprefix("-3\n") if has_valid_suffix else ""
    path = tmp_path / "missing-unselected-pattern.fld"
    path.write_text(prefix + "-3\n C  FROZEN\n-3\n" + suffix, encoding="utf-8")
    if include_standard:
        with pytest.raises(ValueError, match=r"FROZEN.*pattern"):
            load_mm3_fld(path)
    else:
        ff = load_mm3_fld(path, include_standard=False)
        assert len(ff.angles) == int(has_valid_prefix) + int(has_valid_suffix)
        assert all(angle.equilibrium == 110.0 for angle in ff.angles)


@pytest.mark.parametrize("include_standard", [False, True])
def test_closing_format_marker_without_another_title_is_valid(tmp_path: Path, include_standard: bool) -> None:
    path = tmp_path / "closed.fld"
    path.write_text(_selected_block(), encoding="utf-8")
    ff = load_mm3_fld(path, include_standard=include_standard)
    assert len(ff.angles) == 1


def test_real_source_titles_use_declared_substructure_format() -> None:
    lines = FF_PATH.read_text(encoding="utf-8").splitlines()
    current_format = ""
    titles = []
    comments = []
    for index, line in enumerate(lines):
        if line.startswith("-"):
            current_format = line[:2]
        if line.startswith(" C"):
            if index + 1 < len(lines) and lines[index + 1].startswith(" 9"):
                assert current_format == "-3", f"Source title at row {index + 1} lacks -3 format"
                titles.append(index + 1)
            else:
                assert current_format != "-3"
                comments.append(index + 1)
    assert len(titles) == 18
    assert len(comments) == 73
    assert len(_mm3_import_ff(FF_PATH)[0]) > len(_mm3_import_ff(FF_PATH, include_standard=False)[0]) > 0
