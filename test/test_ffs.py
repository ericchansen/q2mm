import copy
import logging
import tempfile
import unittest
from pathlib import Path

from q2mm.io.mm3 import _mm3_import_ff, _mm3_export_ff

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
