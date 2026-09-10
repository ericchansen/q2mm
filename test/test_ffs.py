import copy
import logging
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pytest

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
    _mm3_export_ff,
    _mm3_import_ff,
    _splice_fixed,
    load_mm3_fld,
    save_mm3_fld,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, StretchBendParam, TorsionParam
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
