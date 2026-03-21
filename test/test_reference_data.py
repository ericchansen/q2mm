"""Tests for ReferenceData auto-population (issue #63)."""

import numpy as np
import pytest

from test._shared import (
    CH3F_XYZ,
    GS_FCHK,
    SN2_XYZ as TS_XYZ,
    TS_FCHK,
)

from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceData, _parse_fchk


# ---- from_molecule ----


class TestFromMolecule:
    def test_basic_ch3f(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol)

        n_bonds = len(mol.bonds)
        n_angles = len(mol.angles)
        assert n_bonds > 0
        assert n_angles > 0
        assert ref.n_observations == n_bonds + n_angles

        # Check that bond and angle entries are present
        kinds = {v.kind for v in ref.values}
        assert "bond_length" in kinds
        assert "bond_angle" in kinds
        assert "frequency" not in kinds

    def test_default_weights(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol)

        bond_vals = [v for v in ref.values if v.kind == "bond_length"]
        angle_vals = [v for v in ref.values if v.kind == "bond_angle"]

        assert all(v.weight == 10.0 for v in bond_vals)
        assert all(v.weight == 5.0 for v in angle_vals)

    def test_custom_weights(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol, weights={"bond_length": 50.0, "bond_angle": 25.0})

        bond_vals = [v for v in ref.values if v.kind == "bond_length"]
        angle_vals = [v for v in ref.values if v.kind == "bond_angle"]

        assert all(v.weight == 50.0 for v in bond_vals)
        assert all(v.weight == 25.0 for v in angle_vals)

    def test_atom_indices_populated(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol)

        for v in ref.values:
            assert v.atom_indices is not None
            if v.kind == "bond_length":
                assert len(v.atom_indices) == 2
            elif v.kind == "bond_angle":
                assert len(v.atom_indices) == 3

    def test_bond_values_match_molecule(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol)

        ref_bonds = [v for v in ref.values if v.kind == "bond_length"]
        for rb, mb in zip(ref_bonds, mol.bonds):
            assert abs(rb.value - mb.length) < 1e-10

    def test_angle_values_match_molecule(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol)

        ref_angles = [v for v in ref.values if v.kind == "bond_angle"]
        for ra, ma in zip(ref_angles, mol.angles):
            assert abs(ra.value - ma.value) < 1e-10

    def test_with_frequencies(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        freqs = np.array([100.0, 200.0, -50.0, 300.0])
        ref = ReferenceData.from_molecule(mol, frequencies=freqs)

        freq_vals = [v for v in ref.values if v.kind == "frequency"]
        assert len(freq_vals) == 4

    def test_with_frequencies_skip_imaginary(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        freqs = np.array([100.0, 200.0, -50.0, 300.0])
        ref = ReferenceData.from_molecule(mol, frequencies=freqs, skip_imaginary=True)

        freq_vals = [v for v in ref.values if v.kind == "frequency"]
        assert len(freq_vals) == 3
        assert all(v.value >= 0 for v in freq_vals)

    def test_molecule_idx_propagated(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ref = ReferenceData.from_molecule(mol, molecule_idx=3)

        assert all(v.molecule_idx == 3 for v in ref.values)

    def test_ts_molecule(self):
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        ref = ReferenceData.from_molecule(mol)

        # SN2 TS should have C-F bonds detected with loose tolerance
        bond_labels = [v.label for v in ref.values if v.kind == "bond_length"]
        assert any("F" in lbl for lbl in bond_labels)


# ---- from_molecules ----


class TestFromMolecules:
    def test_two_molecules(self):
        mol1 = Q2MMMolecule.from_xyz(CH3F_XYZ)
        mol2 = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        ref = ReferenceData.from_molecules([mol1, mol2])

        # Should have data from both
        mol0_vals = [v for v in ref.values if v.molecule_idx == 0]
        mol1_vals = [v for v in ref.values if v.molecule_idx == 1]
        assert len(mol0_vals) > 0
        assert len(mol1_vals) > 0
        assert ref.n_observations == len(mol0_vals) + len(mol1_vals)

    def test_with_per_molecule_frequencies(self):
        mol1 = Q2MMMolecule.from_xyz(CH3F_XYZ)
        mol2 = Q2MMMolecule.from_xyz(CH3F_XYZ)
        freqs1 = np.array([100.0, 200.0])
        freqs2 = np.array([150.0, 250.0, 350.0])

        ref = ReferenceData.from_molecules([mol1, mol2], frequencies_list=[freqs1, freqs2])

        freq_vals_0 = [v for v in ref.values if v.kind == "frequency" and v.molecule_idx == 0]
        freq_vals_1 = [v for v in ref.values if v.kind == "frequency" and v.molecule_idx == 1]
        assert len(freq_vals_0) == 2
        assert len(freq_vals_1) == 3

    def test_mismatched_frequencies_raises(self):
        mol1 = Q2MMMolecule.from_xyz(CH3F_XYZ)
        with pytest.raises(ValueError, match="frequencies_list length"):
            ReferenceData.from_molecules([mol1], frequencies_list=[np.array([1.0]), np.array([2.0])])


# ---- add_frequencies_from_array ----


class TestAddFrequenciesFromArray:
    def test_basic(self):
        ref = ReferenceData()
        added = ref.add_frequencies_from_array([100.0, 200.0, 300.0])
        assert added == 3
        assert ref.n_observations == 3

    def test_skip_imaginary(self):
        ref = ReferenceData()
        added = ref.add_frequencies_from_array([100.0, -50.0, 300.0], skip_imaginary=True)
        assert added == 2

    def test_data_idx_preserved(self):
        ref = ReferenceData()
        ref.add_frequencies_from_array([100.0, -50.0, 300.0])
        indices = [v.data_idx for v in ref.values]
        assert indices == [0, 1, 2]

    def test_data_idx_with_skip(self):
        """data_idx should reflect original position even when skipping."""
        ref = ReferenceData()
        ref.add_frequencies_from_array([100.0, -50.0, 300.0], skip_imaginary=True)
        indices = [v.data_idx for v in ref.values]
        # Original positions 0 and 2 (skipped position 1)
        assert indices == [0, 2]

    def test_weight_propagated(self):
        ref = ReferenceData()
        ref.add_frequencies_from_array([100.0, 200.0], weight=3.5)
        assert all(v.weight == 3.5 for v in ref.values)

    def test_numpy_array(self):
        ref = ReferenceData()
        ref.add_frequencies_from_array(np.array([100.0, 200.0]))
        assert ref.n_observations == 2


# ---- _parse_fchk ----


class TestParseFchk:
    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_parse_gs(self):
        symbols, coords, hessian, charge, mult = _parse_fchk(GS_FCHK)
        assert len(symbols) == 8
        assert symbols.count("H") == 6
        assert symbols.count("C") == 2
        assert coords.shape == (8, 3)
        assert charge == 0
        assert mult == 1
        assert hessian is not None
        assert hessian.shape == (24, 24)
        # Hessian should be symmetric
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    @pytest.mark.skipif(not TS_FCHK.exists(), reason="Ethane fixture not found")
    def test_parse_ts(self):
        symbols, coords, hessian, charge, mult = _parse_fchk(TS_FCHK)
        assert len(symbols) == 8
        assert hessian is not None
        assert hessian.shape == (24, 24)

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_coordinates_reasonable(self):
        """Coordinates should be in Angstrom and within reasonable range."""
        symbols, coords, _, _, _ = _parse_fchk(GS_FCHK)
        # All coordinates should be within ~5 Angstrom of origin
        assert np.all(np.abs(coords) < 5.0)
        # C-C bond should be ~1.5 Angstrom
        c_indices = [i for i, s in enumerate(symbols) if s == "C"]
        cc_dist = np.linalg.norm(coords[c_indices[0]] - coords[c_indices[1]])
        assert 1.3 < cc_dist < 1.7, f"C-C distance {cc_dist:.3f} Å out of range"


# ---- from_fchk ----


class TestFromFchk:
    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_basic(self):
        ref, mol = ReferenceData.from_fchk(GS_FCHK)

        assert mol.n_atoms == 8
        assert mol.hessian is not None
        assert mol.hessian.shape == (24, 24)
        assert ref.n_observations > 0

        kinds = {v.kind for v in ref.values}
        assert "bond_length" in kinds
        assert "bond_angle" in kinds

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_bond_count(self):
        ref, mol = ReferenceData.from_fchk(GS_FCHK)
        n_bonds = len(mol.bonds)
        bond_refs = [v for v in ref.values if v.kind == "bond_length"]
        assert len(bond_refs) == n_bonds

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_custom_weights(self):
        ref, _ = ReferenceData.from_fchk(GS_FCHK, weights={"bond_length": 100.0, "bond_angle": 50.0})
        bond_vals = [v for v in ref.values if v.kind == "bond_length"]
        angle_vals = [v for v in ref.values if v.kind == "bond_angle"]
        assert all(v.weight == 100.0 for v in bond_vals)
        assert all(v.weight == 50.0 for v in angle_vals)

    @pytest.mark.skipif(
        not GS_FCHK.exists() or not TS_FCHK.exists(),
        reason="Ethane fixtures not found",
    )
    def test_gs_vs_ts_different_geometries(self):
        """GS (staggered) and TS (eclipsed) should have different angles."""
        ref_gs, mol_gs = ReferenceData.from_fchk(GS_FCHK)
        ref_ts, mol_ts = ReferenceData.from_fchk(TS_FCHK)

        # Both are ethane with 8 atoms
        assert mol_gs.n_atoms == mol_ts.n_atoms == 8

        # Bond counts should be the same (same connectivity)
        assert len(mol_gs.bonds) == len(mol_ts.bonds)

        # But at least some angle values should differ (staggered vs eclipsed)
        gs_angles = sorted(v.value for v in ref_gs.values if v.kind == "bond_angle")
        ts_angles = sorted(v.value for v in ref_ts.values if v.kind == "bond_angle")
        # Not all angles will be identical
        diffs = [abs(a - b) for a, b in zip(gs_angles, ts_angles)]
        assert max(diffs) > 0.1, "GS and TS angles should differ"
