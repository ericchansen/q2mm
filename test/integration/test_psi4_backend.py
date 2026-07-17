"""Integration tests for Psi4Backend.

These tests require Psi4 to be installed (conda install psi4 -c conda-forge).
Tests that only validate saved fixtures (TestPsi4HessianFixture) run without Psi4.
Tests that call Psi4 directly are marked with ``@pytest.mark.psi4``.
"""

from q2mm.backends.contracts import (
    ReferenceEnergyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import qm_prepare_case
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from q2mm.resources import sn2_reference_dir

FIXTURE_DIR = REPO_ROOT / "examples" / "ch3f-sn2"
QM_REF = sn2_reference_dir()

try:
    from q2mm.backends.qm.psi4 import Psi4Backend  # noqa: F401

    HAS_PSI4 = True
except ImportError:
    HAS_PSI4 = False


def _load(xyz: str, charge: int = 0) -> object:
    from q2mm.io.xyz import load_xyz

    return load_xyz(xyz, charge=charge, bond_tolerance=1.5)


@pytest.mark.psi4
class TestPsi4BackendAvailability:
    def test_name(self) -> None:
        backend = load_backend("psi4")
        assert "Psi4" in backend.info.name

    def test_is_available(self) -> None:
        from q2mm.backends.registry import available_backends

        assert "psi4" in available_backends()


@pytest.mark.psi4
class TestPsi4EnergyCH3F:
    """Test Psi4 energy calculation on CH3F.

    Compares against the saved reference energy to verify reproducibility.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.backend = load_backend("psi4", charge=0, multiplicity=1)
        self.mol = _load(str(QM_REF / "ch3f-optimized.xyz"))

    def test_energy_returns_float(self) -> None:
        energy = qm_prepare_case(self.backend, self.mol).energy(ReferenceEnergyRequest()).energy
        assert isinstance(energy, float)

    def test_energy_matches_reference(self) -> None:
        """Energy should match the saved reference within 1e-5 Ha."""
        energy = qm_prepare_case(self.backend, self.mol).energy(ReferenceEnergyRequest()).energy
        ref_energy = -139.751112913417
        assert energy == pytest.approx(ref_energy, abs=1e-5), f"Energy {energy} differs from reference {ref_energy}"


@pytest.mark.psi4
class TestPsi4BackendLoadMolecule:
    """Test that Psi4Backend can evaluate energies for molecules."""

    def test_energy_from_molecule(self) -> None:
        backend = load_backend("psi4", charge=0)
        xyz = str(QM_REF / "ch3f-optimized.xyz")
        if not Path(xyz).exists():
            pytest.skip("CH3F fixture not found")
        energy = qm_prepare_case(backend, _load(xyz)).energy(ReferenceEnergyRequest()).energy
        assert np.isfinite(energy)

    def test_energy_h2(self) -> None:
        from q2mm.models.molecule import Molecule

        backend = load_backend("psi4", charge=0)
        mol = Molecule(
            symbols=["H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
            atom_types=["H", "H"],
        )
        energy = qm_prepare_case(backend, mol).energy(ReferenceEnergyRequest()).energy
        assert np.isfinite(energy)
        # H2 energy should be around -1.17 Ha at B3LYP/6-31+G(d)
        assert energy == pytest.approx(-1.17, abs=0.05)


class TestPsi4HessianFixture:
    """Verify the saved Hessian fixture is valid (no Psi4 needed)."""

    def test_hessian_shape(self) -> None:
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        assert hess.shape == (18, 18)

    def test_hessian_symmetric(self) -> None:
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        np.testing.assert_allclose(hess, hess.T, atol=1e-10, err_msg="Hessian should be symmetric")

    def test_hessian_has_negative_eigenvalue(self) -> None:
        """TS Hessian should have exactly 1 negative eigenvalue."""
        hess = np.load(str(QM_REF / "sn2-ts-hessian.npy"))
        eigenvalues = np.linalg.eigvalsh(hess)
        n_negative = sum(1 for ev in eigenvalues if ev < -0.001)
        assert n_negative == 1, f"Expected 1 negative eigenvalue, got {n_negative}"
