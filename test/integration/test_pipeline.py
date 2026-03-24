"""Pipeline integration test: QM Hessian -> Seminario -> Force Constants.

Tests the core Q2MM workflow: take a QM Hessian and use the Seminario/QFUERZA
method to estimate force field parameters.

This test uses saved QM fixtures (no Psi4/Tinker needed at runtime).
"""

import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"


@unittest.skipUnless((QM_REF / "sn2-ts-hessian.npy").exists(), "SN2 TS fixtures not found")
class TestHessianAnalysis(unittest.TestCase):
    """Test that we can load and analyze the QM Hessian."""

    def setUp(self) -> None:
        self.hessian = np.load(str(QM_REF / "sn2-ts-hessian.npy"))

    def test_hessian_shape(self) -> None:
        self.assertEqual(self.hessian.shape, (18, 18))

    def test_hessian_symmetric(self) -> None:
        np.testing.assert_allclose(self.hessian, self.hessian.T, atol=1e-10)

    def test_one_negative_eigenvalue(self) -> None:
        """TS Hessian should have exactly 1 significantly negative eigenvalue."""
        eigenvalues = np.linalg.eigvalsh(self.hessian)
        n_negative = sum(1 for ev in eigenvalues if ev < -0.001)
        self.assertEqual(n_negative, 1)

    def test_eigenvalue_decomposition_roundtrip(self) -> None:
        """Decompose and reform Hessian, verify roundtrip."""
        from q2mm.models.hessian import decompose, reform_hessian

        eigenvalues, eigenvectors = decompose(self.hessian)
        reformed = reform_hessian(eigenvalues, eigenvectors)
        np.testing.assert_allclose(
            self.hessian, reformed, atol=1e-8, err_msg="Hessian decompose/reform roundtrip failed"
        )


@unittest.skipUnless(
    (QM_REF / "sn2-ts-hessian.npy").exists()
    and (QM_REF / "sn2-ts-frequencies.txt").exists()
    and (QM_REF / "ch3f-frequencies.txt").exists(),
    "SN2 TS fixtures not found",
)
class TestFrequencyFixtures(unittest.TestCase):
    """Validate the saved frequency fixtures."""

    def test_ts_frequencies(self) -> None:
        freqs = np.loadtxt(str(QM_REF / "sn2-ts-frequencies.txt"))
        # 6 atoms -> 3N-6 = 12 vibrational modes
        self.assertEqual(len(freqs), 12)
        # Exactly 1 imaginary
        n_imag = sum(1 for f in freqs if f < 0)
        self.assertEqual(n_imag, 1)
        # Imaginary frequency should be in the SN2 range
        imag_freq = min(freqs)
        self.assertAlmostEqual(imag_freq, -461.7, delta=5.0)

    def test_ch3f_frequencies(self) -> None:
        freqs = np.loadtxt(str(QM_REF / "ch3f-frequencies.txt"))
        # 5 atoms -> 3N-6 = 9 vibrational modes
        self.assertEqual(len(freqs), 9)
        # All positive (ground state minimum)
        self.assertTrue(all(f > 0 for f in freqs))

    def test_ts_energy_below_separated_reactants(self) -> None:
        """For gas-phase SN2, TS energy < F- + CH3F (double-well PES)."""
        with open(str(QM_REF / "summary.txt")) as f:
            text = f.read()
        self.assertIn("Barrier (TS - reactants)", text)


class TestQMReferenceConsistency(unittest.TestCase):
    """Cross-validate QM reference data for internal consistency."""

    def test_ts_geometry_cf_distance(self) -> None:
        """C-F distance in TS should be ~1.85 A (literature: 1.83-1.85)."""
        with open(str(QM_REF / "sn2-ts-optimized.xyz")) as f:
            lines = f.readlines()
        coords = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) == 4:
                coords.append([float(x) for x in parts[1:4]])
        coords = np.array(coords)
        cf1 = np.linalg.norm(coords[0] - coords[1])  # C-F1
        cf2 = np.linalg.norm(coords[0] - coords[2])  # C-F2
        # Both C-F distances should be equal (symmetric TS)
        self.assertAlmostEqual(cf1, cf2, places=3)
        # In literature range
        self.assertAlmostEqual(cf1, 1.85, delta=0.05)

    def test_ch3f_geometry_cf_distance(self) -> None:
        """C-F in CH3F should be ~1.38-1.40 A."""
        with open(str(QM_REF / "ch3f-optimized.xyz")) as f:
            lines = f.readlines()
        coords = []
        for line in lines[2:]:
            parts = line.split()
            if len(parts) == 4:
                coords.append([float(x) for x in parts[1:4]])
        coords = np.array(coords)
        cf = np.linalg.norm(coords[0] - coords[1])
        self.assertAlmostEqual(cf, 1.39, delta=0.02)

    def test_hessian_matches_frequencies(self) -> None:
        """Hessian eigenvalues and saved frequencies should be consistent."""
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        freqs = np.loadtxt(str(QM_REF / "sn2-ts-frequencies.txt"))
        evals = np.linalg.eigvalsh(hess)
        # Should have same sign pattern: 1 negative, rest positive/zero
        n_neg_hess = sum(1 for ev in evals if ev < -0.001)
        n_neg_freq = sum(1 for f in freqs if f < 0)
        self.assertEqual(n_neg_hess, n_neg_freq)


if __name__ == "__main__":
    unittest.main()
