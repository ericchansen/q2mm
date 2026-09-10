import copy
import json
import logging
import tempfile
import unittest
from pathlib import Path

import pytest

from q2mm._canonical import canonical_fingerprint
from q2mm.io.mm3 import (
    P_1_END,
    P_1_START,
    P_2_END,
    P_2_START,
    P_3_END,
    P_3_START,
    _Mm3ParameterRow,
    _format_mm3_angle_line,
    _mm3_export_ff,
    _mm3_import_ff,
    _splice_fixed,
    load_mm3_fld,
)
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout, opt_substructure_membership
from q2mm.models.units import mm3_angle_k_to_canonical, mm3_bond_k_to_canonical, mm3_sb_k_to_canonical

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FF_PATH = REPO_ROOT / "examples" / "publication" / "rh-enamide" / "mm3.fld"


class TestMM3Import(unittest.TestCase):
    def setUp(self) -> None:
        self.params, self.lines = _mm3_import_ff(str(FF_PATH))

    def test_has_params(self) -> None:
        self.assertGreater(len(self.params), 0, "No parameters parsed")

    def test_published_golden_preserves_complete_source_vector(self) -> None:
        fixture_dir = REPO_ROOT / "test" / "fixtures"
        golden = json.loads((fixture_dir / "published_ff" / "rh_enamide_donoghue2008.json").read_text(encoding="utf-8"))
        compatibility = json.loads((fixture_dir / "publication_problem_compatibility.json").read_text(encoding="utf-8"))
        published = next(
            row
            for row in compatibility["rows"]
            if row["system"] == "rh-enamide" and row["starting_point"] == "published"
        )
        ff = load_mm3_fld(FF_PATH)
        layout = ParameterLayout.from_force_field(ff)
        vector = layout.vector(ff)

        self.assertEqual(golden["summary"]["n_params"], len(layout))
        self.assertEqual(golden["param_vector"], vector.tolist())
        self.assertEqual(layout.fingerprint, published["layout"]["fingerprint"])
        self.assertEqual(canonical_fingerprint(vector), published["starting_vector"]["fingerprint"])
        self.assertEqual(golden["summary"]["n_molecules"], len(published["case_ids"]))
        self.assertEqual(len(golden["per_molecule"]), len(published["case_ids"]))

    def test_non_opt_source_angle_is_retained(self) -> None:
        ff = load_mm3_fld(FF_PATH)
        matches = [angle for angle in ff.angles if angle.ff_row == 1134]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].env_id, "AA-C2-C2")
        self.assertIs(ff.match_angle(("C", "C", "C"), env_id="C2-C2-C2", ff_row=1134), matches[0])
        opt_only = load_mm3_fld(FF_PATH, include_standard=False)
        self.assertFalse(any(angle.ff_row == 1134 for angle in opt_only.angles))
        layout = ParameterLayout.from_force_field(ff)
        membership = opt_substructure_membership(ff, opt_only)
        self.assertNotIn(ff.angles.index(matches[0]), membership.angles)
        space = ActiveParameterSpace.from_membership(layout, ff, membership)
        opt_layout = ParameterLayout.from_force_field(opt_only)
        self.assertEqual(space.active_ids, opt_layout.ids)
        self.assertEqual(list(space.pack(space.baseline)), list(opt_layout.vector(opt_only)))

    def test_non_opt_bond_identity_preserves_blank_source_order(self) -> None:
        ff = load_mm3_fld(FF_PATH)
        for row_number in (1172, 1177):
            with self.subTest(row=row_number):
                self.assertEqual(self.lines[row_number - 1][6], " ")
                matches = [bond for bond in ff.bonds if bond.ff_row == row_number]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0].env_id, "C3-C3")
                self.assertEqual(matches[0].bond_order, "")


def _substructure_row(label: str, atoms: tuple[str, ...], *values: float) -> str:
    prefix = (label + "  " + "  ".join(f"{atom:>2}" for atom in atoms)).ljust(P_1_START)
    return prefix + " ".join(f"{value:10.4f}" for value in values) + "\n"


def _substructure(name: str, pattern: str) -> list[str]:
    return [
        f" C  {name}\n",
        f" 9  {pattern} \n",
        _substructure_row(" 1", ("1", "2"), 1.4, 4.0, 0.2),
        "\n",
        " C  comment mentioning OPT, not a new block\n",
        _substructure_row(" 2", ("1", "2", "3"), 110.0, 0.5),
        _substructure_row(" 3", ("1", "2", "3"), 0.7),
        _substructure_row(" 4", ("1", "2", "3", "4"), 0.2, 0.4, 0.6),
        _substructure_row("54", (), 0.8, 1.0, 1.2),
        _substructure_row(" 5", ("1", "2", "3", "4"), 0.4, 0.8),
        "-3\n",
    ]


@pytest.mark.parametrize("reverse_blocks", [False, True])
def test_mm3_full_and_filtered_physical_blocks(tmp_path: Path, reverse_blocks: bool) -> None:
    blocks = [_substructure("FROZEN", "C2-C2-AA-C3"), _substructure("OPT selected", "N1-C3-H1-C3")]
    if reverse_blocks:
        blocks.reverse()
    standard = _format_mm3_angle_line(["H1", "C3", "H1"], 109.0, 0.4)
    lines = [" C  ordinary OPT comment\n", standard, *blocks[0], *blocks[1], standard]
    path = tmp_path / "mixed.fld"
    path.write_text("".join(lines), encoding="utf-8")
    full = load_mm3_fld(path)
    assert len(full.bonds) == 2
    assert len(full.angles) == 4
    assert len(full.stretch_bends) == 2
    assert len(full.torsions) == 16
    assert full.angles[0].env_id == full.angles[-1].env_id == "H1-C3-H1"
    assert full.angles[-1].ff_row == len(lines)

    for name, env in (("OPT", "H1-C3-N1"), ("FROZEN", "AA-C2-C2")):
        rows, original_lines = _mm3_import_ff(path, sub_search=name, include_standard=False)
        assert original_lines == lines
        assert len(rows) == 14
        angle_row = next(row for row in rows if row.ptype == "af")
        angle = next(param for param in full.angles if param.ff_row == angle_row.ff_row)
        assert angle.env_id == env
        assert angle.equilibrium == 110.0
        assert angle.force_constant == mm3_angle_k_to_canonical(0.5)
        assert all(row.ff_row > 2 and row.ff_row < len(lines) for row in rows)
        assert all(row.atom_types == rows[-1].atom_types[: len(row.atom_types)] for row in rows)
        full_rows, _ = _mm3_import_ff(path, sub_search=name)
        assert full_rows == _mm3_import_ff(path)[0]

    assert _mm3_import_ff(path, sub_search="absent", include_standard=False)[0] == []
    assert _mm3_import_ff(path, sub_search="opt", include_standard=False)[0] == []
    opt = load_mm3_fld(path, include_standard=False)
    assert [angle.env_id for angle in opt.angles] == ["H1-C3-N1"]
    assert opt.bonds[0].force_constant == mm3_bond_k_to_canonical(4.0)
    assert opt.bonds[0].dipole_moment == 0.2
    assert opt.stretch_bends[0].force_constant == mm3_sb_k_to_canonical(0.7)
    assert [t.periodicity for t in opt.torsions] == [1, 2, 3, 4, 5, 6, 1, 2]
    assert [t.phase for t in opt.torsions] == [0.0, 180.0, 0.0, 180.0, 0.0, 180.0, 0.0, 180.0]
    assert [t.force_constant for t in opt.torsions] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.2, 0.4]
    assert [t.is_improper for t in opt.torsions] == [False] * 6 + [True, True]

    layout = ParameterLayout.from_force_field(full)
    space = ActiveParameterSpace.from_membership(layout, full, opt_substructure_membership(full, opt))
    opt_layout = ParameterLayout.from_force_field(opt)
    assert space.active_ids == opt_layout.ids
    assert list(space.pack(space.baseline)) == list(opt_layout.vector(opt))
    updated = space.expand(space.pack(space.baseline) + 1.0)
    for slot in layout:
        if slot.index not in space.active_indices:
            assert updated[slot.index] == space.baseline[slot.index]


def test_mm3_filter_ignores_row_comments_and_keeps_vdw_policy(tmp_path: Path) -> None:
    block = _substructure("FROZEN", "C2-C2-AA-C3")
    block[5] = block[5].rstrip("\n") + " OPT comment\n"
    standard = _format_mm3_angle_line(["H1", "C3", "H1"], 109.0, 0.4).rstrip("\n") + " OPT\n"
    path = tmp_path / "filtered.fld"
    path.write_text("-6\n  C3  1.8  0.1  0.0\n-2\n" + standard + "".join(block), encoding="utf-8")
    assert _mm3_import_ff(path, include_standard=False)[0] == []
    opt = load_mm3_fld(path, include_standard=False)
    assert opt.angles == ()
    assert opt.vdws == load_mm3_fld(path).vdws
    assert len(opt.vdws) == 1


def test_mm3_literal_pattern_and_substructure_bond_order(tmp_path: Path) -> None:
    block = _substructure("FROZEN", "C2-C2-AA-C3")
    block[2] = block[2][:6] + "=" + block[2][7:]
    path = tmp_path / "literal.fld"
    path.write_text("".join(block), encoding="utf-8")
    ff = load_mm3_fld(path)
    assert ff.bonds[0].bond_order == "="
    angle = ff.angles[0]
    assert angle.env_id == "AA-C2-C2"
    assert ff.match_angle(("C", "C", "C"), env_id="C2-C2-C2", ff_row=angle.ff_row) is angle
    assert ff.match_angle(("C", "C", "C"), env_id="C2-C2-C2") is None


@pytest.mark.parametrize(
    ("pattern", "angle_labels"),
    [
        ("C2/C2", ("1", "2", "3")),
        ("C2-C2-AA", ("1", "2", "99")),
        ("C2-C2-AA", ("1", "2", "0")),
        ("C2-C2-AA", ("1", "2", "")),
        ("C2-C2-99", ("1", "2", "3")),
    ],
)
def test_mm3_invalid_selected_block_has_row_context(
    tmp_path: Path, pattern: str, angle_labels: tuple[str, ...]
) -> None:
    path = tmp_path / "invalid.fld"
    path.write_text(
        f" C  FROZEN\n 9  {pattern} \n" + _substructure_row(" 2", angle_labels, 110.0, 0.5) + "-3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"invalid\.fld.*(?:row|L)[ :]*[23].*FROZEN"):
        load_mm3_fld(path)
    assert _mm3_import_ff(path, include_standard=False)[0] == []


@pytest.mark.parametrize("boundary", ["new_block", "excluded_block", "standard"])
def test_mm3_torsion_continuation_cannot_cross_scope(tmp_path: Path, boundary: str) -> None:
    first = _substructure("OPT first", "C2-C2-AA-C3")[:9] + ["-3\n"]
    if boundary == "excluded_block":
        first += _substructure("FROZEN", "N1-C3-H1-C3")
    tail = [_substructure_row("54", (), 0.8, 1.0, 1.2)]
    if boundary != "standard":
        tail = [" C  OPT next\n", " 9  N1-C3-H1-C3 \n", *tail, "-3\n"]
    path = tmp_path / "continuation.fld"
    path.write_text("".join(first + tail), encoding="utf-8")
    with pytest.raises(ValueError, match=r"continuation\.fld.*(?:row|L).*continuation"):
        _mm3_import_ff(path, include_standard=boundary == "standard")


@pytest.mark.parametrize("location", ["standard_start", "standard_after_block", "non_opt_substructure"])
def test_mm3_torsion_continuation_reports_encountered_scope(tmp_path: Path, location: str) -> None:
    prefix: list[str] = []
    scope = "standard section"
    if location == "standard_after_block":
        prefix = _substructure("OPT completed", "C2-C2-AA-C3")
    elif location == "non_opt_substructure":
        prefix = [" C  FROZEN actual\n", " 9  C2-C2-AA-C3 \n"]
        scope = "substructure 'FROZEN actual'"
    lines = [*prefix, _substructure_row("54", (), 0.8, 1.0, 1.2)]
    if location == "non_opt_substructure":
        lines.append("-3\n")
    path = tmp_path / "scope.fld"
    path.write_text("".join(lines), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        _mm3_import_ff(path, include_standard=True)
    assert str(raised.value) == (
        f"{path}: row {len(prefix) + 1}, {scope}: torsion continuation has no lower torsion in this scope."
    )


def test_mm3_unterminated_block_is_not_partial_success(tmp_path: Path) -> None:
    path = tmp_path / "unterminated.fld"
    path.write_text("".join(_substructure("FROZEN", "C2-C2-AA-C3")[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"unterminated\.fld.*FROZEN.*unterminated"):
        load_mm3_fld(path)


def test_mm3_next_block_requires_previous_terminator(tmp_path: Path) -> None:
    path = tmp_path / "boundary.fld"
    path.write_text(
        "".join(_substructure("FROZEN", "C2-C2-AA-C3")[:-1] + _substructure("OPT next", "N1-C3-H1-C3")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"boundary\.fld.*FROZEN.*missing -3"):
        _mm3_import_ff(path, include_standard=False)


def test_mm3_excluded_pattern_does_not_leak_into_selected_blocks(tmp_path: Path) -> None:
    path = tmp_path / "selection.fld"
    path.write_text(
        "".join(
            _substructure("FROZEN", "unsupported/pattern")
            + _substructure("OPT first", "C2-C2-AA-C3")
            + _substructure("OPT second", "N1-C3-H1-C3")
        ),
        encoding="utf-8",
    )
    opt = load_mm3_fld(path, include_standard=False)
    assert [angle.env_id for angle in opt.angles] == ["AA-C2-C2", "H1-C3-N1"]


class TestMM3Export(unittest.TestCase):
    def setUp(self) -> None:
        self.params, self.lines = _mm3_import_ff(str(FF_PATH))
        self.mod_params = copy.deepcopy(self.params)
        self.mod_params[0].value = 999.0
        self._tmpdir = tempfile.TemporaryDirectory()
        self.test_fld = Path(self._tmpdir.name) / "test_output.fld"
        _mm3_export_ff(str(self.test_fld), self.mod_params, list(self.lines))

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_export_roundtrip(self) -> None:
        mod_params, _ = _mm3_import_ff(str(self.test_fld))
        self.assertEqual(mod_params[0].value, 999.0)

    def test_export_preserves_other_params(self) -> None:
        mod_params, _ = _mm3_import_ff(str(self.test_fld))
        for orig, exported in zip(self.params[1:], mod_params[1:]):
            self.assertAlmostEqual(orig.value, exported.value, places=4)


class TestMM3ExportHigherTorsion(unittest.TestCase):
    """Higher-order torsions (V4/V5/V6, ff_col 4/5/6) must round-trip.

    Regression: ``_mm3_export_ff`` only handled ff_col 1/2/3, so V4-V6
    values updated in memory were silently dropped when writing a ``54``
    continuation line back to the template.
    """

    def _build_54_line(self, v1: float, v2: float, v3: float) -> str:
        line = "54".ljust(P_1_START)
        line += f"{v1:10.4f}" + " "
        line += f"{v2:10.4f}" + " "
        line += f"{v3:10.4f}"
        line += "  TAILBYTES\n"
        return line

    def test_higher_torsion_values_written_back(self) -> None:
        lines = [self._build_54_line(1.0, 2.0, 3.0)]
        params = [
            _Mm3ParameterRow(ptype="df", ff_col=4, ff_row=1, value=11.0),
            _Mm3ParameterRow(ptype="df", ff_col=5, ff_row=1, value=22.0),
            _Mm3ParameterRow(ptype="df", ff_col=6, ff_row=1, value=33.0),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "higher.fld"
            _mm3_export_ff(str(out), params, list(lines))
            written = out.read_text().splitlines()[0]

        self.assertAlmostEqual(float(written[P_1_START:P_1_END]), 11.0, places=4)
        self.assertAlmostEqual(float(written[P_2_START:P_2_END]), 22.0, places=4)
        self.assertAlmostEqual(float(written[P_3_START:P_3_END]), 33.0, places=4)
        # Trailing non-numeric bytes must survive untouched.
        self.assertIn("TAILBYTES", written)


class TestSpliceFixed(unittest.TestCase):
    """``_splice_fixed`` must preserve byte-stability of other columns."""

    def test_fits_within_width(self) -> None:
        line = "AB" + " " * 8 + "TAIL"
        out = _splice_fixed(line, 2, 8, 1.5)
        self.assertEqual(out[2:10], f"{1.5:8.4f}")
        self.assertTrue(out.endswith("TAIL"))
        self.assertEqual(len(out), len(line))

    def test_overflow_leaves_line_unchanged(self) -> None:
        # 1234567.0 formatted as .4f needs 12 chars; a width-8 field cannot
        # hold it without shifting every trailing byte.
        line = "AB" + " " * 8 + "TAIL"
        out = _splice_fixed(line, 2, 8, 1234567.0)
        self.assertEqual(out, line)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
