"""Integration tests for Psi4Engine.

These tests require Psi4 to be installed (conda install psi4 -c conda-forge).
Tests that only validate saved fixtures (TestPsi4HessianFixture) run without Psi4.
Tests that call Psi4 directly are skipped if Psi4 is not available.
"""
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "examples" / "sn2-test"
QM_REF = FIXTURE_DIR / "qm-reference"

try:
    import psi4  # noqa: F401
    from q2mm.backends.qm.psi4 import Psi4Engine
    HAS_PSI4 = True
except ImportError:
    HAS_PSI4 = False


@unittest.skipUnless(HAS_PSI4, "Psi4 not installed")
class TestPsi4EngineAvailability(unittest.TestCase):
    def test_name(self):
        engine = Psi4Engine()
        self.assertIn("Psi4", engine.name)

    def test_is_available(self):
        engine = Psi4Engine()
        self.assertTrue(engine.is_available())


@unittest.skipUnless(HAS_PSI4, "Psi4 not installed")
@unittest.skipUnless(
    (QM_REF / "ch3f-optimized.xyz").exists(),
    "CH3F fixture not found"
)
class TestPsi4EnergyCH3F(unittest.TestCase):
    """Test Psi4 energy calculation on CH3F.

    Compares against the saved reference energy to verify reproducibility.
    """

    def setUp(self):
        self.engine = Psi4Engine(charge=0, multiplicity=1)
        self.xyz = str(QM_REF / "ch3f-optimized.xyz")

    def test_energy_returns_float(self):
        energy = self.engine.energy(self.xyz)
        self.assertIsInstance(energy, float)

    def test_energy_matches_reference(self):
        """Energy should match the saved reference within 1e-5 Ha."""
        energy = self.engine.energy(self.xyz)
        # Reference from generate_qm_data_v2.py
        ref_energy = -139.751112913417
        self.assertAlmostEqual(energy, ref_energy, places=5,
                               msg=f"Energy {energy} differs from reference {ref_energy}")


@unittest.skipUnless(HAS_PSI4, "Psi4 not installed")
class TestPsi4EngineLoadMolecule(unittest.TestCase):
    """Test that Psi4Engine can load molecules from different sources."""

    def test_load_from_xyz_file(self):
        engine = Psi4Engine(charge=0)
        xyz = str(QM_REF / "ch3f-optimized.xyz")
        if not Path(xyz).exists():
            self.skipTest("CH3F fixture not found")
        energy = engine.energy(xyz)
        self.assertTrue(np.isfinite(energy))

    def test_load_from_atoms_coords(self):
        engine = Psi4Engine(charge=0)
        atoms = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        energy = engine.energy((atoms, coords))
        self.assertTrue(np.isfinite(energy))
        # H2 energy should be around -1.17 Ha at B3LYP/6-31+G(d)
        self.assertAlmostEqual(energy, -1.17, delta=0.05)


@unittest.skipUnless(
    (QM_REF / "sn2-ts-hessian.npy").exists(),
    "SN2 TS Hessian fixture not found"
)
class TestPsi4HessianFixture(unittest.TestCase):
    """Verify the saved Hessian fixture is valid."""

    def test_hessian_shape(self):
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        # 6 atoms * 3 = 18 DOF
        self.assertEqual(hess.shape, (18, 18))

    def test_hessian_symmetric(self):
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        np.testing.assert_allclose(hess, hess.T, atol=1e-10,
                                   err_msg="Hessian should be symmetric")

    def test_hessian_has_negative_eigenvalue(self):
        """TS Hessian should have exactly 1 negative eigenvalue
        (ignoring 6 translation/rotation near-zero eigenvalues)."""
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        eigenvalues = np.linalg.eigvalsh(hess)
        # Significantly negative eigenvalues (not just numerical noise)
        n_negative = sum(1 for ev in eigenvalues if ev < -0.001)
        self.assertEqual(n_negative, 1,
                         f"Expected 1 negative eigenvalue, got {n_negative}")


if __name__ == "__main__":
    unittest.main()
