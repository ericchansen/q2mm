"""Integration tests for TinkerEngine.

These tests require Tinker to be installed locally.
Skipped automatically if Tinker is not found.
"""
import unittest
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "examples" / "sn2-test"
QM_REF = FIXTURE_DIR / "qm-reference"

try:
    from q2mm.backends.mm.tinker import TinkerEngine
    _engine = TinkerEngine()
    HAS_TINKER = _engine.is_available()
except (ImportError, FileNotFoundError):
    HAS_TINKER = False


@unittest.skipUnless(HAS_TINKER, "Tinker not installed")
class TestTinkerEngineAvailability(unittest.TestCase):
    def test_name(self):
        engine = TinkerEngine()
        self.assertIn("Tinker", engine.name)

    def test_is_available(self):
        engine = TinkerEngine()
        self.assertTrue(engine.is_available())


@unittest.skipUnless(HAS_TINKER, "Tinker not installed")
@unittest.skipUnless(
    (QM_REF / "ch3f-optimized.xyz").exists(),
    "CH3F fixture not found"
)
class TestTinkerEnergyCH3F(unittest.TestCase):
    """Test Tinker MM3 energy calculation on CH3F."""

    def setUp(self):
        self.engine = TinkerEngine()
        self.xyz = str(QM_REF / "ch3f-optimized.xyz")

    def test_energy_returns_float(self):
        energy = self.engine.energy(self.xyz)
        self.assertIsInstance(energy, float)

    def test_energy_is_finite(self):
        energy = self.engine.energy(self.xyz)
        self.assertTrue(np.isfinite(energy))


@unittest.skipUnless(HAS_TINKER, "Tinker not installed")
@unittest.skipUnless(
    (QM_REF / "sn2-ts-optimized.xyz").exists(),
    "SN2 TS fixture not found"
)
class TestTinkerFrequenciesSN2(unittest.TestCase):
    """Test Tinker vibrational analysis on SN2 TS."""

    def setUp(self):
        self.engine = TinkerEngine()
        self.xyz = str(QM_REF / "sn2-ts-optimized.xyz")

    def test_frequencies_returns_list(self):
        freqs = self.engine.frequencies(self.xyz)
        self.assertIsInstance(freqs, list)

    def test_correct_number_of_modes(self):
        freqs = self.engine.frequencies(self.xyz)
        # 6 atoms -> 3*6 = 18 modes (including translation/rotation)
        self.assertEqual(len(freqs), 18)

    def test_has_imaginary_frequencies(self):
        """SN2 TS is not an MM minimum — expect imaginary frequencies."""
        freqs = self.engine.frequencies(self.xyz)
        n_imaginary = sum(1 for f in freqs if f < -1.0)
        self.assertGreater(n_imaginary, 0,
                           "TS should have imaginary frequencies on MM surface")


@unittest.skipUnless(HAS_TINKER, "Tinker not installed")
@unittest.skipUnless(
    (QM_REF / "ch3f-optimized.xyz").exists(),
    "CH3F fixture not found"
)
class TestTinkerMinimize(unittest.TestCase):
    """Test Tinker energy minimization."""

    def setUp(self):
        self.engine = TinkerEngine()
        self.xyz = str(QM_REF / "ch3f-optimized.xyz")

    def test_minimize_returns_tuple(self):
        result = self.engine.minimize(self.xyz, rms_grad=0.1)
        energy, atoms, coords = result
        self.assertIsInstance(energy, float)
        self.assertIsInstance(atoms, list)
        self.assertIsInstance(coords, np.ndarray)

    def test_minimize_preserves_atom_count(self):
        _, atoms, coords = self.engine.minimize(self.xyz, rms_grad=0.1)
        self.assertEqual(len(atoms), 5)  # CH3F = 5 atoms
        self.assertEqual(coords.shape, (5, 3))

    def test_minimize_lowers_energy(self):
        initial_energy = self.engine.energy(self.xyz)
        final_energy, _, _ = self.engine.minimize(self.xyz, rms_grad=0.1)
        self.assertLessEqual(final_energy, initial_energy + 0.01)


if __name__ == "__main__":
    unittest.main()
