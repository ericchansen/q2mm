import copy
import json
import logging
import math
import tempfile
import unittest
from dataclasses import replace
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
    _format_mm3_torsion_line,
    _format_mm3_vdw_line,
    _mm3_export_ff,
    _mm3_import_ff,
    _splice_fixed,
    load_mm3_fld,
    save_mm3_fld,
)
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    CmapGrid,
    ForceField,
    FunctionalForm,
    StretchBendParam,
    TorsionParam,
    VdwParam,
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


def _improper_template(scope: str) -> str:
    if scope == "standard":
        proper = _format_mm3_torsion_line(["H1", "C1", "C1", "H1"], 0.2, 0.4, 0.6)
        improper = " 5" + proper[2:].rstrip("\n") + "  UNCHANGED TAIL\n"
        return "".join([" C  synthetic template\n", proper, improper, improper, "-2\n"])
    block = _substructure(scope, "C2-C2-AA-C3")
    block[-2] = block[-2].rstrip("\n") + "             UNCHANGED TAIL\n"
    block.insert(-1, block[-2])
    return "".join(block)


class TestMM3OutputFidelity:
    @pytest.mark.parametrize("scope", ["standard", "OPT selected", "FROZEN"])
    @pytest.mark.parametrize("periodicity", [1, 2], ids=["imp1", "imp2"])
    @pytest.mark.parametrize("value", [9.0, 0.0, -7.5, 1234.5, -1234.5])
    @pytest.mark.parametrize("explicit_template", [False, True])
    def test_template_improper_scalar_roundtrip(
        self, tmp_path: Path, scope: str, periodicity: int, value: float, explicit_template: bool
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_improper_template(scope), encoding="utf-8")
        original_bytes = source.read_bytes()
        ff = load_mm3_fld(source)
        target = next(t for t in ff.torsions if t.is_improper and t.periodicity == periodicity)
        changed = replace(
            ff,
            torsions=tuple(replace(t, force_constant=value) if t is target else t for t in ff.torsions),
        )
        if explicit_template:
            changed = replace(changed, source_path=None, source_format=None)
        output = tmp_path / "updated.fld"
        save_mm3_fld(changed, output, template_path=source if explicit_template else None)

        assert load_mm3_fld(output).torsions == changed.torsions
        assert source.read_bytes() == original_bytes
        expected_lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
        assert target.ff_row is not None
        line = expected_lines[target.ff_row - 1]
        start, end = (P_1_START, P_1_END) if periodicity == 1 else (P_2_START, P_2_END)
        expected_lines[target.ff_row - 1] = line[:start] + f"{2.0 * value:10.4f}" + line[end:]
        assert output.read_text(encoding="utf-8") == "".join(expected_lines)

    @pytest.mark.parametrize("periodicity", [1, 2], ids=["imp1", "imp2"])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    @pytest.mark.parametrize(
        "changes",
        [
            pytest.param({"force_constant": 50000.0}, id="positive-overflow"),
            pytest.param({"force_constant": -5000.0}, id="negative-overflow"),
            pytest.param({"force_constant": float("nan")}, id="nan"),
            pytest.param({"force_constant": float("inf")}, id="infinity"),
            pytest.param({"force_constant": float("-inf")}, id="negative-infinity"),
            pytest.param({"phase": 90.0}, id="phase"),
            pytest.param({"periodicity": 3}, id="order"),
            pytest.param({"ff_row": None}, id="no-source-row"),
            pytest.param({"ff_row": 999}, id="missing-source-row"),
            pytest.param({"env_id": "N1-C1-C1-H1"}, id="environment"),
            pytest.param({"elements": ("N", "C", "C", "H")}, id="elements"),
            pytest.param({"is_improper": False}, id="interaction-kind"),
        ],
    )
    def test_template_rejects_unrepresentable_improper_before_write(
        self, tmp_path: Path, periodicity: int, destination: str, changes: dict[str, object]
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_improper_template("FROZEN"), encoding="utf-8")
        original = source.read_bytes()
        ff = load_mm3_fld(source)
        target = next(t for t in ff.torsions if t.is_improper and t.periodicity == periodicity)
        changed = replace(ff, torsions=tuple(replace(t, **changes) if t is target else t for t in ff.torsions))
        output = source if destination == "source" else tmp_path / "output.fld"
        before = original if destination == "source" else b"existing destination\r\n\xff"
        if destination == "existing":
            output.write_bytes(before)

        with pytest.raises(ValueError, match="MM3.*improper"):
            save_mm3_fld(changed, output)
        assert source.read_bytes() == original
        if destination == "absent":
            assert not output.exists()
        else:
            assert output.read_bytes() == before

    @pytest.mark.parametrize("existing", [False, True])
    def test_template_rejects_duplicate_improper_column(self, tmp_path: Path, existing: bool) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_improper_template("standard"), encoding="utf-8")
        ff = load_mm3_fld(source)
        target = next(t for t in ff.torsions if t.is_improper)
        changed = replace(ff, torsions=(*ff.torsions, replace(target, force_constant=9.0)))
        output = tmp_path / "output.fld"
        if existing:
            output.write_bytes(b"unchanged")
        with pytest.raises(ValueError, match="MM3.*improper.*duplicate"):
            save_mm3_fld(changed, output)
        if existing:
            assert output.read_bytes() == b"unchanged"
        else:
            assert not output.exists()

    @pytest.mark.parametrize("periodicity", [1, 2], ids=["imp1", "imp2"])
    @pytest.mark.parametrize(("value", "phase_offset"), [(9.0, 360.0), (-9.0, -360.0), (0.0, 90.0)])
    def test_template_improper_equivalent_phase_in_place(
        self, tmp_path: Path, periodicity: int, value: float, phase_offset: float
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_improper_template("OPT selected"), encoding="utf-8")
        ff = load_mm3_fld(source)
        target = next(t for t in ff.torsions if t.is_improper and t.periodicity == periodicity)
        changed = replace(
            ff,
            torsions=tuple(
                replace(t, force_constant=value, phase=t.phase + phase_offset) if t is target else t
                for t in ff.torsions
            ),
        )
        save_mm3_fld(changed, source)
        actual = next(
            t for t in load_mm3_fld(source).torsions if t.ff_row == target.ff_row and t.periodicity == periodicity
        )
        assert actual == replace(target, force_constant=value)

    @pytest.mark.parametrize("existing", [False, True])
    @pytest.mark.parametrize(
        ("changes", "feature"),
        [
            *[
                pytest.param(
                    {"cmaps": (CmapGrid(("C",) * 4, ("C",) * 4, 2, (value,) * 4),)},
                    "CMAP",
                    id=f"CMAP-{value}",
                )
                for value in (0.0, 1.0)
            ],
            *[
                pytest.param(
                    {"torsions": (TorsionParam(("H", "C", "C", "H"), n, k, phase=phase),)},
                    "phase",
                    id=f"proper-phase-{n}-{k}-{phase}",
                )
                for n in (1, 2, 3)
                for k in (-0.5, 0.5)
                for phase in (90.0, 180.0 if n % 2 else 0.0)
            ],
            *[
                pytest.param(
                    {"torsions": (TorsionParam(("H", "C", "C", "H"), 1, k, phase=phase),)},
                    "phase",
                    id=f"nonfinite-phase-{k}-{phase}",
                )
                for k in (0.0, 0.5)
                for phase in (float("nan"), float("inf"), float("-inf"))
            ],
            pytest.param({"stretch_bends": (StretchBendParam(("H", "C", "H"), 5.0),)}, "stretch-bend", id="IO-04a"),
            pytest.param(
                {"stretch_bends": (StretchBendParam(("H", "C", "H"), 0.0),)}, "stretch-bend", id="IO-04a-zero"
            ),
            *[
                pytest.param(
                    {"angles": (AngleParam(("H", "C", "H"), 109.5, 20.0, ub_force_constant=k, ub_equilibrium=r),)},
                    "Urey-Bradley",
                    id=f"IO-04b-{k}-{r}",
                )
                for k, r in [(30.0, 2.0), (0.0, 0.0), (30.0, None), (None, 2.0)]
            ],
            *[
                pytest.param(
                    {"torsions": (TorsionParam(("H", "C", "C", "H"), n, k),)},
                    "periodicity",
                    id=f"IO-04c-{n}-{k}",
                )
                for n, k in [(4, 6.0), (5, 6.0), (6, 6.0), (7, 6.0), (4, 0.0)]
            ],
            *[
                pytest.param(
                    {"torsions": (TorsionParam(("H", "C", "C", "H"), n, 2.0, is_improper=True),)},
                    "improper",
                    id=f"IO-04d-{n}",
                )
                for n in (1, 2)
            ],
            *[
                pytest.param(
                    {"bonds": (BondParam(("C", "C"), 1.3, 300.0, bond_order=order),)},
                    "bond order",
                    id=f"IO-04e-{order}",
                )
                for order in ("=", "*", "%")
            ],
            pytest.param(
                {"bonds": (BondParam(("C", "C"), 1.3, 300.0, context="O200 0000"),)},
                "bond context",
                id="IO-04f",
            ),
            *[
                pytest.param(
                    {"bonds": (BondParam(("C", "C"), 1.3, 300.0, dipole_moment=dipole),)},
                    "bond dipole",
                    id=f"IO-04g-{dipole}",
                )
                for dipole in (0.4, -0.4)
            ],
        ],
    )
    def test_standalone_rejects_loss_before_write(
        self, tmp_path: Path, existing: bool, changes: dict[str, object], feature: str
    ) -> None:
        ff = ForceField(
            bonds=(BondParam(("C", "F"), 1.38, 300.0, env_id="C1-F1"),),
            functional_form=FunctionalForm.MM3,
        )
        ff = replace(ff, **changes)
        output = tmp_path / "output.fld"
        if existing:
            output.write_bytes(b"existing destination\r\n\xff")
        with pytest.raises(ValueError, match=f"MM3.*{feature}"):
            save_mm3_fld(ff, output)
        if existing:
            assert output.read_bytes() == b"existing destination\r\n\xff"
        else:
            assert not output.exists()

    @pytest.mark.parametrize("order", ["", "-"])
    @pytest.mark.parametrize("context", ["", "0000 0000"])
    def test_standalone_supported_generic_values(self, tmp_path: Path, order: str, context: str) -> None:
        ff = ForceField(
            bonds=(BondParam(("C", "F"), 1.38, 300.0, env_id="C1-F1", bond_order=order, context=context),),
            angles=(AngleParam(("H", "C", "F"), 109.5, 40.0, env_id="H1-C1-F1"),),
            torsions=tuple(
                TorsionParam(("H", "C", "C", "F"), n, k, phase=phase, env_id="H1-C1-C1-F1")
                for n, k, phase in [(1, -0.5, 0.0), (2, 1.2, 180.0), (3, 0.3, 0.0)]
            ),
            functional_form=FunctionalForm.MM3,
        )
        output = tmp_path / "supported.fld"
        save_mm3_fld(ff, output)
        actual = load_mm3_fld(output)
        assert actual.bonds[0].force_constant == pytest.approx(300.0, rel=1e-3)
        assert actual.bonds[0].equilibrium == 1.38
        assert actual.bonds[0].bond_order == "-"
        assert actual.bonds[0].context == ""
        assert actual.bonds[0].dipole_moment == 0.0
        assert actual.angles[0].force_constant == pytest.approx(40.0, rel=1e-3)
        assert actual.angles[0].ub_force_constant is actual.angles[0].ub_equilibrium is None
        assert [(t.periodicity, t.force_constant, t.phase, t.is_improper) for t in actual.torsions] == [
            (t.periodicity, t.force_constant, t.phase, t.is_improper) for t in ff.torsions
        ]

    @pytest.mark.parametrize("periodicity", [1, 2, 3])
    @pytest.mark.parametrize(("value", "phase_offset"), [(0.5, 360.0), (-0.5, -360.0), (0.0, 90.0)])
    def test_standalone_proper_equivalent_and_zero_phases(
        self, tmp_path: Path, periodicity: int, value: float, phase_offset: float
    ) -> None:
        canonical_phase = 180.0 if periodicity % 2 == 0 else 0.0
        ff = ForceField(
            torsions=(
                TorsionParam(
                    ("H", "C", "C", "H"),
                    periodicity,
                    value,
                    phase=canonical_phase + phase_offset,
                    env_id="H1-C1-C1-H1",
                ),
            ),
            functional_form=FunctionalForm.MM3,
        )
        output = tmp_path / "proper.fld"
        save_mm3_fld(ff, output)
        actual = next(t for t in load_mm3_fld(output).torsions if t.periodicity == periodicity)
        assert actual.force_constant == value
        assert actual.phase == canonical_phase
        assert not actual.is_improper


def _vdw_template(*, generated: bool = False) -> str:
    vdws = (VdwParam("C1", 1.2345, 0.0123, reduction=0.5678), VdwParam("C1", 2.3456, 0.0456, reduction=0.6789))
    lines = [_improper_template("FROZEN"), "-6\n"]
    for vdw in vdws:
        if generated:
            lines.append(_format_mm3_vdw_line(vdw))
        else:
            lines.append(
                f"  {vdw.atom_type:<2} {vdw.radius:10.4f} {vdw.epsilon:10.4f} {vdw.reduction:10.4f}"
                "  UNCHANGED VDW TAIL\n"
            )
    return "".join([*lines, " END OF NONBONDED INTERACTIONS\n", "-2\n"])


class TestMM3VdwOutputFidelity:
    @pytest.mark.parametrize("generated", [False, True])
    @pytest.mark.parametrize("source_bound", [False, True])
    @pytest.mark.parametrize("column", [1, 2, 3], ids=["radius", "epsilon", "reduction"])
    @pytest.mark.parametrize("token", ["123456789.0", "000001.2345"])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_overlong_source_token_rejected_without_leading_digit_leftover(
        self, tmp_path: Path, generated: bool, source_bound: bool, column: int, token: str, destination: str
    ) -> None:
        fields = [f"{value:10.4f}" for value in (1.2345, 0.0123, 0.5678)]
        assert len(token) > 10
        fields[column - 1] = token
        prefix = "  C1  " if generated else "  C1 "
        source = tmp_path / "source.fld"
        source.write_text(_improper_template("FROZEN") + "-6\n" + prefix + " ".join(fields) + "\n-2\n")
        original = source.read_bytes()
        loaded = load_mm3_fld(source)
        replacement = replace(loaded.vdws[0], radius=2.3456, epsilon=0.1234, reduction=0.6789)
        if source_bound:
            ff = replace(loaded, vdws=(replacement,))
        else:
            ff = ForceField(vdws=(replace(replacement, ff_row=None),), functional_form=FunctionalForm.MM3)
        output = source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"preserve destination")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*vdW.*10-character"):
            save_mm3_fld(ff, output, template_path=source)
        assert source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()

    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    @pytest.mark.parametrize(
        ("line", "field"),
        [
            pytest.param("  C1 1.2 0.1 0.0\n", "radius", id="compact-fields"),
            pytest.param(f"  C1 {1.2:9.4f} {0.1:10.4f} {0.0:10.4f}\n", "radius", id="narrow-radius-field"),
            pytest.param(f"  C1 {1.2:10.4f} {0.1:10.4f}\n", "reduction", id="missing-reduction"),
            pytest.param(f"  C1 {1.2:10.4f} {0.1:10.4f} {'not-float':>10}\n", "reduction", id="nonnumeric-reduction"),
        ],
    )
    def test_template_type_match_rejects_unwritable_fields_before_write(
        self, tmp_path: Path, destination: str, line: str, field: str
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_improper_template("FROZEN") + "-6\n" + line + "-2\n", encoding="utf-8")
        original = source.read_bytes()
        ff = ForceField(vdws=(VdwParam("C1", 1.2345, 0.1234, reduction=0.5678),), functional_form=FunctionalForm.MM3)
        output = source if destination == "source" else tmp_path / "output.fld"
        before = original if destination == "source" else b"existing destination\r\n\xff"
        if destination == "existing":
            output.write_bytes(before)
        with pytest.raises(ValueError, match=rf"MM3.*vdW.*{field}"):
            save_mm3_fld(ff, output, template_path=source)
        assert source.read_bytes() == original
        if destination == "absent":
            assert not output.exists()
        else:
            assert output.read_bytes() == before

    @pytest.mark.parametrize("field", ["radius", "epsilon", "reduction"])
    @pytest.mark.parametrize(
        "value",
        [100000.0, -10000.0, 99999.99996, -9999.99996, float("nan"), float("inf"), float("-inf")],
    )
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_template_rejects_invalid_vdw_before_any_write(
        self, tmp_path: Path, field: str, value: float, destination: str
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_vdw_template(), encoding="utf-8")
        original = source.read_bytes()
        ff = load_mm3_fld(source)
        changed = replace(
            ff,
            vdws=(replace(ff.vdws[0], **{field: value}), *ff.vdws[1:]),
            torsions=tuple(replace(t, force_constant=2.5) if t.is_improper else t for t in ff.torsions),
        )
        output = source if destination == "source" else tmp_path / "output.fld"
        before = original if destination == "source" else b"existing destination\r\n\xff"
        if destination == "existing":
            output.write_bytes(before)
        with pytest.raises(ValueError, match=rf"MM3.*vdW.*{field}"):
            save_mm3_fld(changed, output)
        assert source.read_bytes() == original
        if destination == "absent":
            assert not output.exists()
        else:
            assert output.read_bytes() == before

    @pytest.mark.parametrize("field", ["radius", "epsilon", "reduction"])
    @pytest.mark.parametrize("value", [0.0, 1.23456, 99999.9999, -9999.9999])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_template_vdw_precision_preserves_other_fields_and_rows(
        self, tmp_path: Path, field: str, value: float, destination: str
    ) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_vdw_template(), encoding="utf-8")
        original = source.read_bytes()
        ff = load_mm3_fld(source)
        target = replace(ff.vdws[0], **{field: value})
        changed = replace(ff, vdws=(target, *ff.vdws[1:]))
        output = source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"replace destination")
        save_mm3_fld(changed, output)
        actual = load_mm3_fld(output)
        rounded = replace(target, **{field: float(f"{value:.4f}")})
        assert actual.vdws == (rounded, *ff.vdws[1:])
        assert actual.bonds == ff.bonds
        assert actual.angles == ff.angles
        assert actual.stretch_bends == ff.stretch_bends
        assert actual.torsions == ff.torsions
        start = {"radius": 5, "epsilon": 16, "reduction": 27}[field]
        lines = original.splitlines(keepends=True)
        assert target.ff_row is not None
        line = lines[target.ff_row - 1]
        lines[target.ff_row - 1] = line[:start] + f"{value:10.4f}".encode("ascii") + line[start + 10 :]
        assert output.read_bytes() == b"".join(lines)
        if destination != "source":
            assert source.read_bytes() == original

    def test_generated_vdw_template_preserves_four_decimal_precision(self, tmp_path: Path) -> None:
        source = tmp_path / "generated.fld"
        source.write_text(_vdw_template(generated=True), encoding="utf-8")
        ff = load_mm3_fld(source)
        changed = replace(ff, vdws=(replace(ff.vdws[0], radius=3.4567, epsilon=0.1234, reduction=0.7654), *ff.vdws[1:]))
        output = tmp_path / "updated.fld"
        save_mm3_fld(changed, output)
        assert load_mm3_fld(output).vdws == changed.vdws

    def test_template_type_match_updates_only_vdw_fields(self, tmp_path: Path) -> None:
        source = tmp_path / "source.fld"
        source.write_text(_vdw_template(), encoding="utf-8")
        original = source.read_bytes()
        source_ff = load_mm3_fld(source)
        vdw = VdwParam("C1", 3.4567, 0.1234, reduction=0.7654)
        ff = ForceField(vdws=(vdw,), functional_form=FunctionalForm.MM3)
        output = tmp_path / "updated.fld"
        save_mm3_fld(ff, output, template_path=source)
        actual = load_mm3_fld(output)
        assert actual.vdws == tuple(
            replace(param, radius=vdw.radius, epsilon=vdw.epsilon, reduction=vdw.reduction) for param in source_ff.vdws
        )
        assert actual.torsions == source_ff.torsions
        assert source.read_bytes() == original
        original_lines = original.splitlines(keepends=True)
        actual_lines = output.read_bytes().splitlines(keepends=True)
        vdw_rows = {param.ff_row for param in source_ff.vdws}
        assert len(actual_lines) == len(original_lines)
        for row, (before, after) in enumerate(zip(original_lines, actual_lines, strict=True), start=1):
            if row in vdw_rows:
                assert before[:5] == after[:5]
                assert before[37:] == after[37:]
            else:
                assert before == after


_MM3_NUMERIC_FIELDS = [
    ("bonds", "equilibrium"),
    ("bonds", "force_constant"),
    ("angles", "equilibrium"),
    ("angles", "force_constant"),
    ("torsions", "force_constant"),
    ("vdws", "radius"),
    ("vdws", "epsilon"),
    ("vdws", "reduction"),
]


def _edit_mm3_native_scalar(ff: ForceField, family: str, field: str, value: float) -> ForceField:
    if field == "force_constant":
        if family == "bonds":
            value = mm3_bond_k_to_canonical(value)
        elif family == "angles":
            value = mm3_angle_k_to_canonical(value)
        elif family == "stretch_bends":
            value = mm3_sb_k_to_canonical(value)
        else:
            value /= 2.0
    params = getattr(ff, family)
    return replace(ff, **{family: (replace(params[0], **{field: value}), *params[1:])})


@pytest.fixture
def numeric_source(tmp_path: Path) -> Path:
    source = tmp_path / "source.fld"
    source.write_text(_vdw_template(), encoding="utf-8")
    return source


class TestMM3NumericOutputFidelity:
    @pytest.mark.parametrize(("family", "field"), _MM3_NUMERIC_FIELDS)
    @pytest.mark.parametrize("value", [1000.0, -1000.0, 99999.9999, -9999.9999, 0.00004])
    def test_standalone_fitting_scalars_roundtrip_at_file_precision(
        self, numeric_source: Path, tmp_path: Path, family: str, field: str, value: float
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        ff = replace(
            ff,
            bonds=tuple(replace(b, dipole_moment=0.0) for b in ff.bonds),
            stretch_bends=(),
            torsions=tuple(t for t in ff.torsions if not t.is_improper and t.periodicity <= 3),
            source_path=None,
            source_format=None,
        )
        output = save_mm3_fld(_edit_mm3_native_scalar(ff, family, field, value), tmp_path / "output.fld")
        actual = getattr(load_mm3_fld(output), family)[0]
        expected = getattr(_edit_mm3_native_scalar(ff, family, field, float(f"{value:.4f}")), family)[0]
        expected_value = getattr(expected, field)
        if family == "angles" and field == "equilibrium" and expected_value > 180.0:
            folded = expected_value % 360.0
            expected_value = 360.0 - folded if folded > 180.0 else folded
        assert getattr(actual, field) == pytest.approx(expected_value, rel=1e-14, abs=0.0)
        if family != "vdws":
            start = P_2_START if field == "force_constant" and family in ("bonds", "angles") else P_1_START
            assert actual.ff_row is not None
            assert output.read_text().splitlines()[actual.ff_row - 1][start : start + 10] == f"{value:10.4f}"

    @pytest.mark.parametrize(("family", "field"), _MM3_NUMERIC_FIELDS)
    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -10000.0, -9999.99996])
    @pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_invalid_scalar_preserves_destination(
        self, numeric_source: Path, tmp_path: Path, family: str, field: str, value: float, mode: str, destination: str
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        if mode == "standalone":
            ff = replace(
                ff,
                bonds=tuple(replace(b, dipole_moment=0.0) for b in ff.bonds),
                stretch_bends=(),
                torsions=tuple(t for t in ff.torsions if not t.is_improper and t.periodicity <= 3),
            )
        ff = _edit_mm3_native_scalar(ff, family, field, value)
        if mode != "implicit-template":
            ff = replace(ff, source_path=None, source_format=None)
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"existing destination\r\n\xff")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*(finite|10-character)"):
            save_mm3_fld(ff, output, template_path=numeric_source if mode == "explicit-template" else None)
        assert numeric_source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()

    @pytest.mark.parametrize(("family", "field"), _MM3_NUMERIC_FIELDS + [("stretch_bends", "force_constant")])
    @pytest.mark.parametrize("value", [1000.0, -1000.0, 99999.9999, -9999.9999, 99999.99994, -9999.99994])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_fitting_template_scalar_changes_only_its_source_field(
        self, numeric_source: Path, tmp_path: Path, family: str, field: str, value: float, destination: str
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        changed = _edit_mm3_native_scalar(ff, family, field, value)
        target = getattr(ff, family)[0]
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"replace destination")
        save_mm3_fld(changed, output)

        expected = value
        if family == "angles" and field == "equilibrium" and value > 180.0:
            folded = value % 360.0
            expected = 360.0 - folded if folded > 180.0 else folded
        if family == "vdws":
            start = {"radius": 5, "epsilon": 16, "reduction": 27}[field]
        else:
            start = P_2_START if field == "force_constant" and family in ("bonds", "angles") else P_1_START
        lines = original.splitlines(keepends=True)
        assert target.ff_row is not None
        line = lines[target.ff_row - 1]
        serialized = f"{expected:10.4f}".encode("ascii")
        assert len(serialized) == 10
        lines[target.ff_row - 1] = line[:start] + serialized + line[start + 10 :]
        assert output.read_bytes() == b"".join(lines)
        if destination != "source":
            assert numeric_source.read_bytes() == original
        actual = getattr(load_mm3_fld(output), family)[0]
        expected_param = getattr(_edit_mm3_native_scalar(ff, family, field, float(serialized)), family)[0]
        assert getattr(actual, field) == pytest.approx(getattr(expected_param, field), rel=1e-14)

    @pytest.mark.parametrize(("family", "field"), _MM3_NUMERIC_FIELDS)
    @pytest.mark.parametrize("value", [100000.0, 99999.99996])
    @pytest.mark.parametrize("existing", [False, True])
    def test_standalone_positive_width_and_rounding_overflow(
        self, numeric_source: Path, tmp_path: Path, family: str, field: str, value: float, existing: bool
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        ff = replace(
            ff,
            bonds=tuple(replace(b, dipole_moment=0.0) for b in ff.bonds),
            stretch_bends=(),
            torsions=tuple(t for t in ff.torsions if not t.is_improper and t.periodicity <= 3),
            source_path=None,
            source_format=None,
        )
        output = tmp_path / "output.fld"
        if existing:
            output.write_bytes(b"preserve destination")
        with pytest.raises(ValueError, match="MM3.*10-character"):
            save_mm3_fld(_edit_mm3_native_scalar(ff, family, field, value), output)
        assert output.read_bytes() == b"preserve destination" if existing else not output.exists()

    @pytest.mark.parametrize(
        ("family", "field"),
        [
            ("bonds", "equilibrium"),
            ("bonds", "force_constant"),
            ("angles", "force_constant"),
            ("torsions", "force_constant"),
            ("stretch_bends", "force_constant"),
        ],
    )
    @pytest.mark.parametrize("value", [100000.0, 99999.99996, float("nan"), float("inf"), -float("inf")])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_template_positive_overflow_and_stretch_bend_nonfinite_do_not_silently_skip(
        self, numeric_source: Path, tmp_path: Path, family: str, field: str, value: float, destination: str
    ) -> None:
        ff = _edit_mm3_native_scalar(load_mm3_fld(numeric_source), family, field, value)
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"preserve destination")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*(finite|10-character)"):
            save_mm3_fld(ff, output)
        assert numeric_source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()

    @pytest.mark.parametrize("periodicity", [1, 2, 3])
    @pytest.mark.parametrize("amplitude", [1e308, -1e308])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_standalone_finite_amplitude_conversion_overflow_preserves_destination(
        self, numeric_source: Path, tmp_path: Path, periodicity: int, amplitude: float, destination: str
    ) -> None:
        assert math.isfinite(amplitude) and not math.isfinite(amplitude * 2.0)
        ff = ForceField(
            torsions=(
                TorsionParam(
                    ("H", "C", "C", "H"), periodicity, amplitude, phase=180.0 if periodicity % 2 == 0 else 0.0
                ),
            ),
            functional_form=FunctionalForm.MM3,
        )
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"preserve destination")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*finite"):
            save_mm3_fld(ff, output)
        assert numeric_source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()

    @pytest.mark.parametrize("periodicity", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize(
        ("amplitude", "phase"),
        [(0.5, 90.0), (-0.5, 90.0)]
        + [(k, phase) for k in (0.0, 0.5) for phase in (float("nan"), float("inf"), -float("inf"))],
    )
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_template_proper_phase_rejected_before_staging(
        self, numeric_source: Path, tmp_path: Path, periodicity: int, amplitude: float, phase: float, destination: str
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        target = next(t for t in ff.proper_torsions if t.periodicity == periodicity)
        changed = replace(
            ff,
            torsions=tuple(
                replace(t, force_constant=amplitude, phase=phase) if t is target else t for t in ff.torsions
            ),
            vdws=(replace(ff.vdws[0], radius=3.4567), *ff.vdws[1:]),
        )
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"preserve destination")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*proper.*phase"):
            save_mm3_fld(changed, output)
        assert numeric_source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()

    @pytest.mark.parametrize("periodicity", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize(("amplitude", "offset"), [(1234.5, 360.0), (-1234.5, -360.0), (0.0, 90.0)])
    def test_template_proper_equivalent_phase_keeps_all_six_source_columns(
        self, numeric_source: Path, periodicity: int, amplitude: float, offset: float
    ) -> None:
        ff = load_mm3_fld(numeric_source)
        target = next(t for t in ff.proper_torsions if t.periodicity == periodicity)
        changed = replace(
            ff,
            torsions=tuple(
                replace(t, force_constant=amplitude, phase=t.phase + offset) if t is target else t for t in ff.torsions
            ),
        )
        expected = numeric_source.read_bytes().splitlines(keepends=True)
        assert target.ff_row is not None
        start = (P_1_START, P_2_START, P_3_START)[(periodicity - 1) % 3]
        line = expected[target.ff_row - 1]
        expected[target.ff_row - 1] = line[:start] + f"{2.0 * amplitude:10.4f}".encode("ascii") + line[start + 10 :]
        save_mm3_fld(changed, numeric_source)
        assert numeric_source.read_bytes() == b"".join(expected)
        actual = next(t for t in load_mm3_fld(numeric_source).proper_torsions if t.periodicity == periodicity)
        assert actual == replace(target, force_constant=amplitude)

    @pytest.mark.parametrize("periodicity", [1, 2, 3, 4, 5, 6])
    @pytest.mark.parametrize("amplitude", [float("nan"), float("inf"), -float("inf"), 1e308, -1e308])
    @pytest.mark.parametrize("destination", ["absent", "existing", "source"])
    def test_template_proper_raw_and_converted_nonfinite_amplitude(
        self, numeric_source: Path, tmp_path: Path, periodicity: int, amplitude: float, destination: str
    ) -> None:
        if math.isfinite(amplitude):
            assert not math.isfinite(amplitude * 2.0)
        ff = load_mm3_fld(numeric_source)
        target = next(t for t in ff.proper_torsions if t.periodicity == periodicity)
        changed = replace(
            ff, torsions=tuple(replace(t, force_constant=amplitude) if t is target else t for t in ff.torsions)
        )
        original = numeric_source.read_bytes()
        output = numeric_source if destination == "source" else tmp_path / "output.fld"
        if destination == "existing":
            output.write_bytes(b"preserve destination")
        before = output.read_bytes() if output.exists() else None
        with pytest.raises(ValueError, match="MM3.*finite"):
            save_mm3_fld(changed, output)
        assert numeric_source.read_bytes() == original
        assert output.read_bytes() == before if before is not None else not output.exists()


class TestSpliceFixed(unittest.TestCase):
    """``_splice_fixed`` must preserve byte-stability of other columns."""

    def test_fits_within_width(self) -> None:
        line = "AB" + " " * 8 + "TAIL"
        out = _splice_fixed(line, 2, 8, 1.5)
        self.assertEqual(out[2:10], f"{1.5:8.4f}")
        self.assertTrue(out.endswith("TAIL"))
        self.assertEqual(len(out), len(line))

    def test_overflow_rejects_before_splicing(self) -> None:
        # 1234567.0 formatted as .4f needs 12 chars; a width-8 field cannot
        # hold it without shifting every trailing byte.
        line = "AB" + " " * 8 + "TAIL"
        with self.assertRaisesRegex(ValueError, "MM3.*8-character"):
            _splice_fixed(line, 2, 8, 1234567.0)

    def test_missing_field_rejects_before_splicing(self) -> None:
        with self.assertRaisesRegex(ValueError, "MM3.*field"):
            _splice_fixed("AB 1.0", 2, 10, 1.5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
