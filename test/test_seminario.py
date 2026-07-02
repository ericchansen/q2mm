import unittest
from pathlib import Path
import numpy as np

from q2mm.io.mm3 import _mm3_import_ff
from q2mm.io import GaussLog, Mol2
from q2mm.models.hessian import mass_weight_hessian

REPO_ROOT = Path(__file__).resolve().parent.parent
ETHANE_DIR = REPO_ROOT / "examples" / "ethane"
RH_SEMINARIO_DIR = REPO_ROOT / "examples" / "rh-enamide"


@unittest.skipUnless(
    (ETHANE_DIR / "GS.mol2").exists() and (ETHANE_DIR / "GS.log").exists() and (ETHANE_DIR / "TS.log").exists(),
    "Ethane fixture files not found",
)
class TestGaussLogParsing(unittest.TestCase):
    """Test that GaussLog can parse ethane Gaussian output."""

    def test_parse_gs_log(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        self.assertGreater(len(log.structures), 0, "No structures parsed from GS.log")

    def test_gs_log_has_atoms(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertGreater(len(struct.atoms), 0, "No atoms in parsed structure")

    def test_gs_log_has_hessian(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertIsNotNone(struct.hess, "No Hessian parsed from GS.log")

    def test_parse_ts_log(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "TS.log"))
        self.assertGreater(len(log.structures), 0, "No structures parsed from TS.log")

    def test_archive_hessian_reproduces_gaussian_frequencies(self) -> None:
        """The archive Cartesian Hessian must reproduce Gaussian's frequencies.

        Regression guard for the mass-weighting ingestion bug: both loaders
        used to override ``mol.hessian`` with ``reform_hessian(evals, evecs)``
        reconstructed from Gaussian's *mass-weighted* frequency analysis,
        corrupting every heavy-atom force constant by ~√(mᵢmⱼ).  The archive
        Hessian (``au_hessian=True``) is plain Cartesian (Hartree/Bohr²) and
        must round-trip to the Gaussian-reported vibrational frequencies to
        well under 1 cm⁻¹.
        """
        from q2mm.models.hessian import hessian_to_frequencies

        log = GaussLog(str(ETHANE_DIR / "GS.log"), au_hessian=True)
        mol = log.molecules[-1]
        self.assertIsNotNone(mol.hessian, "Archive Hessian not attached to molecule")

        reported = np.sort(np.asarray(log.frequencies, dtype=float))
        all_freqs = np.asarray(hessian_to_frequencies(mol.hessian, list(mol.symbols), sort=True))
        # Drop the 6 lowest-magnitude (rigid-body) modes before comparing.
        vibrational = np.sort(all_freqs[np.argsort(np.abs(all_freqs))[6:]])
        n = min(len(reported), len(vibrational))
        max_dev = float(np.abs(vibrational[:n] - reported[:n]).max())
        self.assertLess(
            max_dev,
            1.0,
            f"Archive Hessian frequencies deviate {max_dev:.3f} cm⁻¹ from "
            "Gaussian's reported values — mass-weighting override may have "
            "returned.",
        )


@unittest.skipUnless((ETHANE_DIR / "GS.mol2").exists(), "Ethane mol2 fixture not found")
class TestMol2Parsing(unittest.TestCase):
    """Test that Mol2 can parse ethane structure."""

    def test_parse_mol2(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        self.assertGreater(len(mol2.structures), 0, "No structures parsed from mol2")

    def test_mol2_atom_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: C2H6 = 8 atoms
        self.assertEqual(len(struct.atoms), 8, "Ethane should have 8 atoms")

    def test_mol2_bond_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: 7 bonds (1 C-C + 6 C-H)
        self.assertEqual(len(struct.bonds), 7, "Ethane should have 7 bonds")


@unittest.skipUnless((RH_SEMINARIO_DIR / "mm3.fld").exists(), "rh-enamide fixture not found")
class TestMM3FFParsing(unittest.TestCase):
    """Test MM3 force field parsing via q2mm.io.mm3."""

    def setUp(self) -> None:
        self.params, _ = _mm3_import_ff(str(RH_SEMINARIO_DIR / "mm3.fld"))

    def test_parse_mm3(self) -> None:
        self.assertGreater(len(self.params), 0, "No parameters parsed")

    def test_mm3_has_bonds(self) -> None:
        bond_params = [p for p in self.params if p.ptype in ("bf", "be")]
        self.assertGreater(len(bond_params), 0, "No bond parameters found")

    def test_mm3_has_angles(self) -> None:
        angle_params = [p for p in self.params if p.ptype in ("af", "ae")]
        self.assertGreater(len(angle_params), 0, "No angle parameters found")


@unittest.skipUnless(
    (ETHANE_DIR / "GS.log").exists() and (ETHANE_DIR / "GS.mol2").exists(), "Ethane fixture files not found"
)
class TestHessianMassWeighting(unittest.TestCase):
    """Test mass-weighting of Hessians."""

    def test_mass_weight_roundtrip(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        symbols = [a.element for a in struct.atoms]
        hess = log.structures[0].hess.copy()
        original = hess.copy()
        # Mass-weight then un-weight should give back original
        mass_weight_hessian(hess, symbols)
        mass_weight_hessian(hess, symbols, reverse=True)
        np.testing.assert_allclose(original, hess, rtol=1e-10, err_msg="Mass-weight roundtrip did not preserve Hessian")


if __name__ == "__main__":
    unittest.main()
