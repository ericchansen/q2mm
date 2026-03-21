"""Integration tests for Psi4Engine.

These tests require Psi4 to be installed (conda install psi4 -c conda-forge).
Tests that only validate saved fixtures (TestPsi4HessianFixture) run without Psi4.
Tests that call Psi4 directly are marked with ``@pytest.mark.psi4``.
"""

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO_ROOT / "examples" / "sn2-test"
QM_REF = FIXTURE_DIR / "qm-reference"

try:
    from q2mm.backends.qm.psi4 import Psi4Engine

    HAS_PSI4 = True
except ImportError:
    HAS_PSI4 = False


@pytest.mark.psi4
class TestPsi4EngineAvailability:
    def test_name(self):
        engine = Psi4Engine()
        assert "Psi4" in engine.name

    def test_is_available(self):
        engine = Psi4Engine()
        assert engine.is_available()


@pytest.mark.psi4
@pytest.mark.skipif(not (QM_REF / "ch3f-optimized.xyz").exists(), reason="CH3F fixture not found")
class TestPsi4EnergyCH3F:
    """Test Psi4 energy calculation on CH3F.

    Compares against the saved reference energy to verify reproducibility.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.engine = Psi4Engine(charge=0, multiplicity=1)
        self.xyz = str(QM_REF / "ch3f-optimized.xyz")

    def test_energy_returns_float(self):
        energy = self.engine.energy(self.xyz)
        assert isinstance(energy, float)

    def test_energy_matches_reference(self):
        """Energy should match the saved reference within 1e-5 Ha."""
        energy = self.engine.energy(self.xyz)
        ref_energy = -139.751112913417
        assert energy == pytest.approx(ref_energy, abs=1e-5), f"Energy {energy} differs from reference {ref_energy}"


@pytest.mark.psi4
class TestPsi4EngineLoadMolecule:
    """Test that Psi4Engine can load molecules from different sources."""

    def test_load_from_xyz_file(self):
        engine = Psi4Engine(charge=0)
        xyz = str(QM_REF / "ch3f-optimized.xyz")
        if not Path(xyz).exists():
            pytest.skip("CH3F fixture not found")
        energy = engine.energy(xyz)
        assert np.isfinite(energy)

    def test_load_from_atoms_coords(self):
        engine = Psi4Engine(charge=0)
        atoms = ["H", "H"]
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        energy = engine.energy((atoms, coords))
        assert np.isfinite(energy)
        # H2 energy should be around -1.17 Ha at B3LYP/6-31+G(d)
        assert energy == pytest.approx(-1.17, abs=0.05)


@pytest.mark.skipif(not (QM_REF / "sn2-ts-hessian.npy").exists(), reason="SN2 TS Hessian fixture not found")
class TestPsi4HessianFixture:
    """Verify the saved Hessian fixture is valid (no Psi4 needed)."""

    def test_hessian_shape(self):
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        assert hess.shape == (18, 18)

    def test_hessian_symmetric(self):
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        np.testing.assert_allclose(hess, hess.T, atol=1e-10, err_msg="Hessian should be symmetric")

    def test_hessian_has_negative_eigenvalue(self):
        """TS Hessian should have exactly 1 negative eigenvalue."""
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        eigenvalues = np.linalg.eigvalsh(hess)
        n_negative = sum(1 for ev in eigenvalues if ev < -0.001)
        assert n_negative == 1, f"Expected 1 negative eigenvalue, got {n_negative}"
