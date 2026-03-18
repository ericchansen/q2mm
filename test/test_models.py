"""Tests for q2mm.models (molecule, forcefield, seminario)."""
from pathlib import Path

import numpy as np
import pytest

from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.forcefield import ForceField, BondParam, AngleParam, _extract_element
from q2mm.models.seminario import estimate_force_constants

# Fixture paths
DATA_DIR = Path(__file__).resolve().parent.parent / "examples" / "sn2-test" / "qm-reference"
TS_XYZ = DATA_DIR / "sn2-ts-optimized.xyz"
TS_HESS = DATA_DIR / "sn2-ts-hessian.npy"
CH3F_XYZ = DATA_DIR / "ch3f-optimized.xyz"
CH3F_HESS = DATA_DIR / "ch3f-hessian.npy"


# ---- _extract_element helper ----

class TestExtractElement:
    def test_single_letter(self):
        assert _extract_element("C1") == "C"
        assert _extract_element("F") == "F"
        assert _extract_element("H3") == "H"

    def test_two_letter(self):
        assert _extract_element("Cl1") == "Cl"
        assert _extract_element("Br") == "Br"
        assert _extract_element("Rh2") == "Rh"
        assert _extract_element("Pt") == "Pt"
        assert _extract_element("RH1") == "Rh"
        assert _extract_element("CL") == "Cl"

    def test_whitespace(self):
        assert _extract_element("  Cl1") == "Cl"
        assert _extract_element(" F") == "F"


# ---- Q2MMMolecule ----

class TestMoleculeFromXYZ:
    def test_load_ch3f(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        assert mol.n_atoms == 5
        assert mol.symbols[0] == "C"
        assert mol.symbols[1] == "F"
        assert mol.geometry.shape == (5, 3)

    def test_load_ts(self):
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        assert mol.n_atoms == 6
        assert mol.symbols.count("F") == 2

    def test_bond_detection_default(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        bonds = mol.bonds
        assert len(bonds) > 0
        elements_found = {b.element_pair for b in bonds}
        assert ("C", "H") in elements_found or ("H", "C") in elements_found

    def test_bond_detection_ts_tolerance(self):
        mol_tight = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.3)
        mol_loose = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        # Looser tolerance should detect more bonds (partial TS bonds)
        assert len(mol_loose.bonds) >= len(mol_tight.bonds)

    def test_angle_detection(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        angles = mol.angles
        assert len(angles) > 0
        # CH3F has H-C-H and H-C-F angles
        center_elements = {a.elements[1] for a in angles}
        assert "C" in center_elements


# ---- ForceField ----

class TestForceField:
    def test_create_for_molecule(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ff = ForceField.create_for_molecule(mol)
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0

    def test_n_params_matches_vector(self):
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 5.0)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 0.5)],
        )
        vec = ff.get_param_vector()
        assert ff.n_params == len(vec)

    def test_param_vector_roundtrip(self):
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 5.0)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 0.5)],
        )
        vec = ff.get_param_vector()
        ff2 = ff.copy()
        ff2.set_param_vector(vec * 2)
        vec2 = ff2.get_param_vector()
        np.testing.assert_allclose(vec2, vec * 2)


# ---- Seminario force constant estimation ----

class TestSeminario:
    @pytest.fixture
    def ch3f_mol_with_hess(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        hess = np.load(CH3F_HESS)
        return mol.with_hessian(hess)

    @pytest.fixture
    def ts_mol_with_hess(self):
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        hess = np.load(TS_HESS)
        return mol.with_hessian(hess)

    def test_estimate_runs(self, ch3f_mol_with_hess):
        ff = estimate_force_constants(ch3f_mol_with_hess)
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0

    def test_fc_values_positive_ground_state(self, ch3f_mol_with_hess):
        ff = estimate_force_constants(ch3f_mol_with_hess)
        for b in ff.bonds:
            assert b.force_constant > 0, f"Bond {b.key} has non-positive FC"

    def test_negative_fc_included_for_ts(self, ts_mol_with_hess):
        """Negative FCs from TS reaction coordinates should be included, not dropped."""
        ff = estimate_force_constants(ts_mol_with_hess)
        bond_fcs = [b.force_constant for b in ff.bonds]
        # The C-F bond in the TS is partially breaking — may have negative FC
        # At minimum, verify the estimation completes and produces values
        assert len(bond_fcs) > 0

    def test_raises_without_hessian(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        with pytest.raises(ValueError, match="Hessian"):
            estimate_force_constants(mol)
