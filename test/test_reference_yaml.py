"""Tests for the YAML reference data parser (``q2mm.parsers.reference_yaml``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml not installed")

from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceData, ReferenceValue
from q2mm.parsers.reference_yaml import (
    ReferenceYAMLError,
    _reference_value_to_dict,
    load_reference_yaml,
    save_reference_yaml,
)
from test._shared import make_water

FIXTURES = Path(__file__).resolve().parent / "fixtures"
WATER_YAML = FIXTURES / "reference_water.yaml"
CH3F_XYZ_YAML = FIXTURES / "reference_ch3f_xyz.yaml"


# ---------------------------------------------------------------------------
# Loading from fixture files
# ---------------------------------------------------------------------------


class TestLoadFixture:
    def test_load_water_inline(self) -> None:
        ref, mols = load_reference_yaml(WATER_YAML)
        assert len(mols) == 1
        mol = mols[0]
        assert mol.name == "water"
        assert mol.n_atoms == 3
        assert mol.charge == 0
        assert mol.multiplicity == 1
        assert mol.symbols == ["O", "H", "H"]
        np.testing.assert_allclose(mol.geometry[0], [0.0, 0.0, 0.0])

        # Check reference values
        kinds = [v.kind for v in ref.values]
        assert kinds.count("energy") == 1
        assert kinds.count("bond_length") == 2
        assert kinds.count("bond_angle") == 1
        assert kinds.count("frequency") == 3

        # Frequency values
        freq_vals = sorted(v.value for v in ref.values if v.kind == "frequency")
        assert freq_vals == pytest.approx([1648.0, 3832.0, 3943.0])

    def test_load_ch3f_xyz(self) -> None:
        ref, mols = load_reference_yaml(CH3F_XYZ_YAML)
        assert len(mols) == 1
        mol = mols[0]
        assert mol.name == "ch3f"
        assert mol.n_atoms == 5
        assert mol.charge == 0

        energy_vals = [v for v in ref.values if v.kind == "energy"]
        assert len(energy_vals) == 1
        assert energy_vals[0].value == pytest.approx(-139.7621)

        bond_vals = [v for v in ref.values if v.kind == "bond_length"]
        assert len(bond_vals) == 1
        assert bond_vals[0].atom_indices == (0, 1)


# ---------------------------------------------------------------------------
# Round-trip: save → load → compare
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_basic_round_trip(self, tmp_path: Path) -> None:
        mol = make_water()
        ref = ReferenceData()
        ref.add_energy(-76.0267, weight=1.0, label="water energy")
        ref.add_bond_length(0.96, atom_indices=(0, 1), weight=10.0)
        ref.add_bond_angle(104.5, atom_indices=(1, 0, 2), weight=5.0)
        ref.add_frequency(1648.0, data_idx=0, weight=1.0)

        out_path = tmp_path / "round_trip.yaml"
        save_reference_yaml(out_path, ref, [mol])
        assert out_path.exists()

        ref2, mols2 = load_reference_yaml(out_path)
        assert len(mols2) == 1
        assert mols2[0].name == mol.name
        np.testing.assert_allclose(mols2[0].geometry, mol.geometry, atol=1e-10)
        assert mols2[0].symbols == mol.symbols

        assert ref2.n_observations == ref.n_observations
        for orig, loaded in zip(ref.values, ref2.values):
            assert orig.kind == loaded.kind
            assert orig.value == pytest.approx(loaded.value)
            assert orig.weight == pytest.approx(loaded.weight)
            assert orig.atom_indices == loaded.atom_indices

    def test_multi_molecule_round_trip(self, tmp_path: Path) -> None:
        mol0 = make_water(name="water_gs")
        mol1 = make_water(angle_deg=109.0, name="water_ts")

        ref = ReferenceData()
        ref.add_energy(-76.0267, molecule_idx=0, label="GS energy")
        ref.add_energy(-75.9, molecule_idx=1, label="TS energy")
        ref.add_bond_length(0.96, atom_indices=(0, 1), molecule_idx=0, weight=10.0)
        ref.add_bond_angle(109.0, atom_indices=(1, 0, 2), molecule_idx=1, weight=5.0)

        out_path = tmp_path / "multi.yaml"
        save_reference_yaml(out_path, ref, [mol0, mol1])

        ref2, mols2 = load_reference_yaml(out_path)
        assert len(mols2) == 2
        assert mols2[0].name == "water_gs"
        assert mols2[1].name == "water_ts"
        assert ref2.n_observations == 4

        # Verify molecule_idx preserved (values grouped by molecule)
        mol_indices = [v.molecule_idx for v in ref2.values]
        assert mol_indices == [0, 0, 1, 1]

    def test_torsion_angle_round_trip(self, tmp_path: Path) -> None:
        from test._shared import make_ethane

        mol = make_ethane()
        ref = ReferenceData()
        ref.add_torsion_angle(60.0, atom_indices=(2, 0, 1, 5), weight=3.0, label="HCCH")

        out_path = tmp_path / "torsion.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, mols2 = load_reference_yaml(out_path)

        assert ref2.n_observations == 1
        v = ref2.values[0]
        assert v.kind == "torsion_angle"
        assert v.atom_indices == (2, 0, 1, 5)
        assert v.value == pytest.approx(60.0)
        assert v.weight == pytest.approx(3.0)

    def test_eig_diagonal_round_trip(self, tmp_path: Path) -> None:
        mol = make_water()
        ref = ReferenceData()
        ref.add_hessian_eigenvalue(0.5, mode_idx=0, weight=0.1)
        ref.add_hessian_eigenvalue(1.2, mode_idx=1, weight=0.1)

        out_path = tmp_path / "eig.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, _ = load_reference_yaml(out_path)

        assert ref2.n_observations == 2
        for orig, loaded in zip(ref.values, ref2.values):
            assert orig.kind == loaded.kind
            assert orig.data_idx == loaded.data_idx
            assert orig.value == pytest.approx(loaded.value)

    def test_eig_offdiagonal_round_trip(self, tmp_path: Path) -> None:
        mol = make_water()
        ref = ReferenceData()
        ref.add_hessian_offdiagonal(0.01, row=0, col=1, weight=0.05)

        out_path = tmp_path / "eig_off.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, _ = load_reference_yaml(out_path)

        assert ref2.n_observations == 1
        v = ref2.values[0]
        assert v.kind == "eig_offdiagonal"
        assert v.atom_indices == (0, 1)
        assert v.value == pytest.approx(0.01)

    def test_bulk_frequencies_round_trip(self, tmp_path: Path) -> None:
        mol = make_water()
        ref = ReferenceData()
        ref.add_frequencies_from_array([1648.0, 3832.0, 3943.0], weight=2.0)

        out_path = tmp_path / "freqs.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, _ = load_reference_yaml(out_path)

        assert ref2.n_observations == 3
        freq_vals = sorted(v.value for v in ref2.values)
        assert freq_vals == pytest.approx([1648.0, 3832.0, 3943.0])
        assert all(v.weight == pytest.approx(2.0) for v in ref2.values)

    def test_geometry_data_idx_round_trip(self, tmp_path: Path) -> None:
        """Bond/angle refs identified by data_idx (no atom_indices) should round-trip."""
        mol = make_water()
        ref = ReferenceData()
        ref.add_bond_length(0.96, data_idx=0, weight=10.0)
        ref.add_bond_angle(104.5, data_idx=1, weight=5.0)

        out_path = tmp_path / "data_idx.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, _ = load_reference_yaml(out_path)

        assert ref2.n_observations == 2
        bl = ref2.values[0]
        assert bl.kind == "bond_length"
        assert bl.atom_indices is None
        assert bl.data_idx == 0
        assert bl.value == pytest.approx(0.96)

        ba = ref2.values[1]
        assert ba.kind == "bond_angle"
        assert ba.atom_indices is None
        assert ba.data_idx == 1
        assert ba.value == pytest.approx(104.5)


# ---------------------------------------------------------------------------
# from_yaml / to_yaml on ReferenceData
# ---------------------------------------------------------------------------


class TestReferenceDataMethods:
    def test_from_yaml(self) -> None:
        ref, mols = ReferenceData.from_yaml(WATER_YAML)
        assert isinstance(ref, ReferenceData)
        assert len(mols) == 1
        assert ref.n_observations > 0

    def test_to_yaml(self, tmp_path: Path) -> None:
        mol = make_water()
        ref = ReferenceData()
        ref.add_energy(-76.0, weight=1.0)

        out_path = tmp_path / "to_yaml.yaml"
        ref.to_yaml(out_path, [mol])
        assert out_path.exists()

        ref2, mols2 = ReferenceData.from_yaml(out_path)
        assert ref2.n_observations == 1
        assert ref2.values[0].value == pytest.approx(-76.0)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_reference_yaml("/nonexistent/path.yaml")

    def test_bad_top_level_type(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("- just a list\n")
        with pytest.raises(ReferenceYAMLError, match="Top-level"):
            load_reference_yaml(p)

    def test_molecules_not_a_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("molecules: not_a_list\n")
        with pytest.raises(ReferenceYAMLError, match="must be a list"):
            load_reference_yaml(p)

    def test_molecule_entry_not_a_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("molecules:\n  - just a string\n")
        with pytest.raises(ReferenceYAMLError, match="must be a mapping"):
            load_reference_yaml(p)

    def test_missing_kind(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="Missing required key 'kind'"):
            load_reference_yaml(p)

    def test_unknown_kind(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: dipole\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="Unknown kind 'dipole'"):
            load_reference_yaml(p)

    def test_bad_value_type(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: energy\n"
            "        value: not_a_number\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a number"):
            load_reference_yaml(p)

    def test_bond_wrong_atom_count(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        atoms: [0, 1, 2]\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="requires exactly 2 atom indices"):
            load_reference_yaml(p)

    def test_missing_xyz_file(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    xyz: nonexistent.xyz\n"
            "    data:\n"
            "      - kind: energy\n"
            "        value: -1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="XYZ file not found"):
            load_reference_yaml(p)

    def test_bad_coordinates_shape(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n  - name: test\n    geometry:\n      symbols: [H, H]\n      coordinates: [[0,0],[1,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="Nx3"):
            load_reference_yaml(p)

    def test_symbols_coords_mismatch(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n  - name: test\n    geometry:\n      symbols: [H]\n      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="Number of symbols"):
            load_reference_yaml(p)

    def test_data_not_a_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data: not_a_list\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a list"):
            load_reference_yaml(p)

    def test_data_entry_not_a_dict(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - just a string\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a mapping"):
            load_reference_yaml(p)

    def test_missing_bond_atoms_and_data_idx(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="requires either 'atoms' or 'data_idx'"):
            load_reference_yaml(p)

    def test_frequency_indices_values_mismatch(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: frequency\n"
            "        values: [100.0, 200.0]\n"
            "        indices: [0]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="indices.*must match.*values"):
            load_reference_yaml(p)


# ---------------------------------------------------------------------------
# Relative path resolution
# ---------------------------------------------------------------------------


class TestRelativePaths:
    def test_xyz_relative_to_yaml(self) -> None:
        """The ch3f fixture uses a relative path to the xyz file."""
        ref, mols = load_reference_yaml(CH3F_XYZ_YAML)
        assert len(mols) == 1
        assert mols[0].n_atoms == 5
        assert mols[0].symbols[0] == "C"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_molecules_list(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("molecules: []\n")
        ref, mols = load_reference_yaml(p)
        assert len(mols) == 0
        assert ref.n_observations == 0

    def test_molecule_without_data(self, tmp_path: Path) -> None:
        p = tmp_path / "no_data.yaml"
        p.write_text(
            "molecules:\n  - name: bare\n    geometry:\n      symbols: [H, H]\n      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        ref, mols = load_reference_yaml(p)
        assert len(mols) == 1
        assert ref.n_observations == 0

    def test_molecule_without_geometry_single_line(self, tmp_path: Path) -> None:
        """A molecule entry without geometry should raise (single-line variant)."""
        p = tmp_path / "data_only.yaml"
        p.write_text("molecules:\n  - name: abstract\n    data:\n      - kind: energy\n        value: -10.0\n")
        with pytest.raises(ReferenceYAMLError, match="has no geometry"):
            load_reference_yaml(p)

    def test_default_weight_is_one(self, tmp_path: Path) -> None:
        p = tmp_path / "defaults.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: energy\n"
            "        value: -1.0\n"
        )
        ref, _ = load_reference_yaml(p)
        assert ref.values[0].weight == pytest.approx(1.0)

    def test_custom_atom_types(self, tmp_path: Path) -> None:
        mol = Q2MMMolecule(
            symbols=["O", "H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            atom_types=["OW", "HW", "HW"],
            name="water_tip3p",
        )
        ref = ReferenceData()
        ref.add_energy(-76.0)

        out_path = tmp_path / "custom_types.yaml"
        save_reference_yaml(out_path, ref, [mol])
        ref2, mols2 = load_reference_yaml(out_path)

        assert mols2[0].atom_types == ["OW", "HW", "HW"]

    def test_eigenmatrix_loading(self, tmp_path: Path) -> None:
        """kind: eigenmatrix should bulk-create ReferenceValues from a hessian."""
        # Create a tiny 3x3 symmetric positive-definite Hessian.
        hess = np.array([[2.0, 0.5, 0.1], [0.5, 3.0, 0.2], [0.1, 0.2, 1.5]])
        hess_path = tmp_path / "tiny_hessian.npy"
        np.save(str(hess_path), hess)

        p = tmp_path / "eigenmatrix.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: tiny\n"
            "    geometry:\n"
            "      symbols: [H]\n"
            "      coordinates: [[0,0,0]]\n"
            f"    hessian: {hess_path.name}\n"
            "    data:\n"
            "      - kind: eigenmatrix\n"
            "        diagonal_weight: 0.1\n"
            "        offdiagonal_weight: 0.05\n"
            "        skip_first: false\n"
        )

        ref, mols = load_reference_yaml(p)
        assert len(mols) == 1
        # 3x3 hessian → 3 eigenvalues + 3 off-diagonal lower-tri = 6 entries
        assert ref.n_observations > 0
        kinds = {v.kind for v in ref.values}
        assert "eig_diagonal" in kinds

    def test_eigenmatrix_missing_hessian(self, tmp_path: Path) -> None:
        """kind: eigenmatrix without a hessian file should error."""
        p = tmp_path / "no_hess.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: tiny\n"
            "    geometry:\n"
            "      symbols: [H]\n"
            "      coordinates: [[0,0,0]]\n"
            "    hessian: nonexistent.npy\n"
            "    data:\n"
            "      - kind: eigenmatrix\n"
        )
        with pytest.raises(ReferenceYAMLError, match="Hessian file not found"):
            load_reference_yaml(p)

    def test_eigenmatrix_no_hessian_key(self, tmp_path: Path) -> None:
        """kind: eigenmatrix when molecule has no hessian should error."""
        p = tmp_path / "no_hess_key.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: tiny\n"
            "    geometry:\n"
            "      symbols: [H]\n"
            "      coordinates: [[0,0,0]]\n"
            "    data:\n"
            "      - kind: eigenmatrix\n"
        )
        with pytest.raises(ReferenceYAMLError, match="eigenmatrix.*requires.*hessian"):
            load_reference_yaml(p)

    def test_as_int_rejects_bool(self, tmp_path: Path) -> None:
        """Boolean values should not be silently accepted as integers."""
        p = tmp_path / "bool_charge.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    charge: true\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be an integer"):
            load_reference_yaml(p)

    def test_as_int_rejects_non_integer_float(self, tmp_path: Path) -> None:
        """Non-integer floats should not be silently truncated."""
        p = tmp_path / "float_charge.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    charge: 1.5\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be an integer"):
            load_reference_yaml(p)

    def test_negative_data_idx_geometry(self, tmp_path: Path) -> None:
        """Negative data_idx for geometry kinds should be rejected."""
        p = tmp_path / "neg_idx.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        data_idx: -1\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)

    def test_negative_data_idx_frequency(self, tmp_path: Path) -> None:
        """Negative data_idx for frequency kind should be rejected."""
        p = tmp_path / "neg_freq_idx.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: frequency\n"
            "        data_idx: -1\n"
            "        value: 100.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)

    def test_symbols_must_be_list(self, tmp_path: Path) -> None:
        """'symbols' as a string should be rejected."""
        p = tmp_path / "str_symbols.yaml"
        p.write_text(
            "molecules:\n  - name: test\n    geometry:\n      symbols: HH\n      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a list"):
            load_reference_yaml(p)

    def test_atom_types_must_be_list(self, tmp_path: Path) -> None:
        """'atom_types' as a string should be rejected."""
        p = tmp_path / "str_types.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      atom_types: HH\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a list"):
            load_reference_yaml(p)

    def test_eig_offdiagonal_serialize_missing_atom_indices(self) -> None:
        """eig_offdiagonal with None atom_indices should error on serialization."""
        rv = ReferenceValue(
            kind="eig_offdiagonal",
            value=0.01,
            weight=0.05,
            molecule_idx=0,
            atom_indices=None,
        )
        with pytest.raises(ReferenceYAMLError, match="eig_offdiagonal requires row/col"):
            _reference_value_to_dict(rv)

    def test_missing_molecules_key(self, tmp_path: Path) -> None:
        """A YAML file without a 'molecules' key should be rejected."""
        p = tmp_path / "no_molecules.yaml"
        p.write_text("settings:\n  something: 1\n")
        with pytest.raises(ReferenceYAMLError, match="top-level 'molecules' key"):
            load_reference_yaml(p)

    def test_negative_bulk_frequency_indices(self, tmp_path: Path) -> None:
        """Negative indices in bulk frequency 'indices' should be rejected."""
        p = tmp_path / "neg_bulk_idx.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: frequency\n"
            "        values: [100.0, 200.0]\n"
            "        indices: [0, -1]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)

    def test_non_default_bond_tolerance(self, tmp_path: Path) -> None:
        p = tmp_path / "tol.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: ts\n"
            "    bond_tolerance: 1.5\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
        )
        _, mols = load_reference_yaml(p)
        assert mols[0].bond_tolerance == pytest.approx(1.5)

    def test_as_int_list_rejects_booleans(self, tmp_path: Path) -> None:
        """Booleans in an atom-index list should be rejected, not silently coerced."""
        p = tmp_path / "bool_atoms.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        atoms: [true, 1]\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be an integer"):
            load_reference_yaml(p)

    def test_as_int_list_rejects_non_integer_floats(self, tmp_path: Path) -> None:
        """Non-integer floats in an atom-index list should be rejected."""
        p = tmp_path / "float_atoms.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        atoms: [0, 1.9]\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be an integer"):
            load_reference_yaml(p)

    def test_negative_atom_indices_rejected(self, tmp_path: Path) -> None:
        """Negative atom indices should be rejected."""
        p = tmp_path / "neg_atoms.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        atoms: [-1, 0]\n"
            "        value: 1.0\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)

    def test_non_numeric_coordinates_rejected(self, tmp_path: Path) -> None:
        """Non-numeric coordinate entries should raise a clear error."""
        p = tmp_path / "bad_coords.yaml"
        p.write_text(
            "molecules:\n  - name: test\n    geometry:\n      symbols: [H]\n      coordinates: [[0, 0, abc]]\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-numeric"):
            load_reference_yaml(p)

    def test_as_float_rejects_booleans(self, tmp_path: Path) -> None:
        """Boolean values should not be silently coerced to float."""
        p = tmp_path / "bool_value.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: bond_length\n"
            "        atoms: [0, 1]\n"
            "        value: true\n"
        )
        with pytest.raises(ReferenceYAMLError, match="must be a number"):
            load_reference_yaml(p)

    def test_xyz_non_string_rejected(self, tmp_path: Path) -> None:
        """Non-string xyz values should raise ReferenceYAMLError."""
        p = tmp_path / "bad_xyz.yaml"
        p.write_text("molecules:\n  - name: test\n    xyz: 123\n    data: []\n")
        with pytest.raises(ReferenceYAMLError, match="must be a string"):
            load_reference_yaml(p)

    def test_hessian_non_string_rejected(self, tmp_path: Path) -> None:
        """Non-string hessian values should raise ReferenceYAMLError."""
        xyz = tmp_path / "mol.xyz"
        xyz.write_text("2\n\nH 0 0 0\nH 1 0 0\n")
        p = tmp_path / "bad_hess.yaml"
        p.write_text("molecules:\n  - name: test\n    xyz: mol.xyz\n    hessian: 42\n    data: []\n")
        with pytest.raises(ReferenceYAMLError, match="must be a string"):
            load_reference_yaml(p)

    def test_negative_eig_diagonal_mode_idx(self, tmp_path: Path) -> None:
        """Negative mode_idx in eig_diagonal should be rejected."""
        p = tmp_path / "neg_eig.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: eig_diagonal\n"
            "        mode_idx: -1\n"
            "        value: 0.5\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)

    def test_negative_eig_offdiagonal_row_col(self, tmp_path: Path) -> None:
        """Negative row/col in eig_offdiagonal should be rejected."""
        p = tmp_path / "neg_offdiag.yaml"
        p.write_text(
            "molecules:\n"
            "  - name: test\n"
            "    geometry:\n"
            "      symbols: [H, H]\n"
            "      coordinates: [[0,0,0],[1,0,0]]\n"
            "    data:\n"
            "      - kind: eig_offdiagonal\n"
            "        row: -1\n"
            "        col: 0\n"
            "        value: 0.1\n"
        )
        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            load_reference_yaml(p)
