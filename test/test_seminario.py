import copy
import logging
import logging.config
import os
import unittest
from pathlib import Path
import numpy as np
from q2mm import linear_algebra

from q2mm.schrod_indep_filetypes import MM3, MacroModelLog, GaussLog, Mol2, mass_weight_hessian
from q2mm.seminario import seminario
from q2mm import constants as co
from q2mm import utilities

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ETHANE_DIR = REPO_ROOT / "q2mm_example" / "amber" / "Ethane"
SEMINARIO_DIR = REPO_ROOT / "q2mm_example" / "amber" / "seminario"
RH_SEMINARIO_DIR = REPO_ROOT / "rh-seminario"


@unittest.skipUnless(
    (ETHANE_DIR / "GS.mol2").exists() and (ETHANE_DIR / "GS.log").exists(),
    "Ethane fixture files not found"
)
class TestGaussLogParsing(unittest.TestCase):
    """Test that GaussLog can parse ethane Gaussian output."""

    def test_parse_gs_log(self):
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        self.assertGreater(len(log.structures), 0, "No structures parsed from GS.log")

    def test_gs_log_has_atoms(self):
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertGreater(len(struct.atoms), 0, "No atoms in parsed structure")

    def test_gs_log_has_hessian(self):
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertIsNotNone(struct.hess, "No Hessian parsed from GS.log")

    def test_parse_ts_log(self):
        log = GaussLog(str(ETHANE_DIR / "TS.log"))
        self.assertGreater(len(log.structures), 0, "No structures parsed from TS.log")


@unittest.skipUnless(
    (ETHANE_DIR / "GS.mol2").exists(),
    "Ethane mol2 fixture not found"
)
class TestMol2Parsing(unittest.TestCase):
    """Test that Mol2 can parse ethane structure."""

    def test_parse_mol2(self):
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        self.assertGreater(len(mol2.structures), 0, "No structures parsed from mol2")

    def test_mol2_atom_count(self):
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: C2H6 = 8 atoms
        self.assertEqual(len(struct.atoms), 8, "Ethane should have 8 atoms")

    def test_mol2_bond_count(self):
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: 7 bonds (1 C-C + 6 C-H)
        self.assertEqual(len(struct.bonds), 7, "Ethane should have 7 bonds")


@unittest.skipUnless(
    (RH_SEMINARIO_DIR / "mm3.fld").exists(),
    "rh-seminario fixture not found"
)
class TestMM3FFParsing(unittest.TestCase):
    """Test MM3 force field parsing from schrod_indep_filetypes."""

    def setUp(self):
        self.ff = MM3(str(RH_SEMINARIO_DIR / "mm3.fld"))
        self.ff.import_ff()

    def test_parse_mm3(self):
        self.assertGreater(len(self.ff.params), 0, "No parameters parsed")

    def test_mm3_has_bonds(self):
        bond_params = [p for p in self.ff.params if p.ptype in ('bf', 'be')]
        self.assertGreater(len(bond_params), 0, "No bond parameters found")

    def test_mm3_has_angles(self):
        angle_params = [p for p in self.ff.params if p.ptype in ('af', 'ae')]
        self.assertGreater(len(angle_params), 0, "No angle parameters found")


@unittest.skipUnless(
    (ETHANE_DIR / "GS.log").exists() and (ETHANE_DIR / "GS.mol2").exists(),
    "Ethane fixture files not found"
)
class TestHessianMassWeighting(unittest.TestCase):
    """Test mass-weighting of Hessians."""

    def test_mass_weight_roundtrip(self):
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        hess = log.structures[0].hess.copy()
        original = hess.copy()
        # Mass-weight then un-weight should give back original
        mass_weight_hessian(hess, struct.atoms)
        mass_weight_hessian(hess, struct.atoms, reverse=True)
        np.testing.assert_allclose(
            original, hess, rtol=1e-10,
            err_msg="Mass-weight roundtrip did not preserve Hessian"
        )


if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    unittest.main()
