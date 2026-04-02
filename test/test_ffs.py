import copy
import logging
import tempfile
import unittest
from pathlib import Path

from q2mm.parsers.mm3 import MM3

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
FF_PATH = REPO_ROOT / "examples" / "rh-enamide" / "mm3.fld"


class TestMM3Import(unittest.TestCase):
    def setUp(self) -> None:
        self.ff = MM3(str(FF_PATH))
        self.ff.import_ff()

    def test_has_substructures(self) -> None:
        self.assertGreater(len(self.ff.sub_names), 0, "No substructures parsed")

    def test_has_smiles(self) -> None:
        self.assertGreater(len(self.ff.smiles), 0, "No SMILES parsed")

    def test_has_atom_types(self) -> None:
        self.assertGreater(len(self.ff.atom_types), 0, "No atom types parsed")

    def test_has_params(self) -> None:
        self.assertGreater(len(self.ff.params), 0, "No parameters parsed")


class TestMM3Export(unittest.TestCase):
    def setUp(self) -> None:
        self.ff = MM3(str(FF_PATH))
        self.ff.import_ff()
        with open(FF_PATH) as f:
            self.ff.lines = f.readlines()
        self.mod_params = copy.deepcopy(self.ff.params)
        self.mod_params[0].value = 999.0
        self._tmpdir = tempfile.TemporaryDirectory()
        self.test_fld = Path(self._tmpdir.name) / "test_output.fld"
        self.ff.export_ff(path=str(self.test_fld), params=self.mod_params, lines=self.ff.lines)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_export_roundtrip(self) -> None:
        mod_ff = MM3(str(self.test_fld))
        mod_ff.import_ff()
        self.assertEqual(mod_ff.params[0].value, 999.0)

    def test_export_preserves_other_params(self) -> None:
        mod_ff = MM3(str(self.test_fld))
        mod_ff.import_ff()
        for orig, exported in zip(self.ff.params[1:], mod_ff.params[1:]):
            self.assertAlmostEqual(orig.value, exported.value, places=4)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
