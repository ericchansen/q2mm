import copy
import logging
import tempfile
import unittest
from pathlib import Path

from q2mm.io.mm3 import (
    P_1_END,
    P_1_START,
    P_2_END,
    P_2_START,
    P_3_END,
    P_3_START,
    _Mm3ParameterRow,
    _mm3_export_ff,
    _mm3_import_ff,
    _splice_fixed,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FF_PATH = REPO_ROOT / "examples" / "rh-enamide" / "mm3.fld"


class TestMM3Import(unittest.TestCase):
    def setUp(self) -> None:
        self.params, self.lines = _mm3_import_ff(str(FF_PATH))

    def test_has_params(self) -> None:
        self.assertGreater(len(self.params), 0, "No parameters parsed")


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
