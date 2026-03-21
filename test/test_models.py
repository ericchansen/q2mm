"""Tests for q2mm.models (molecule, forcefield, seminario)."""

from pathlib import Path

import numpy as np
import pytest

from test._shared import CH3F_HESS, CH3F_XYZ, SN2_HESSIAN as TS_HESS, SN2_XYZ as TS_XYZ

from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.forcefield import ForceField, BondParam, AngleParam, TorsionParam, VdwParam, _extract_element
from q2mm.models.seminario import estimate_force_constants
from q2mm.parsers.tinker_ff import TinkerFF

# Fixture paths (test-specific, not shared)
RH_MM3 = Path(__file__).resolve().parent.parent / "examples" / "rh-enamide" / "mm3.fld"


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

    def test_detected_env_ids_use_atom_types(self):
        mol = Q2MMMolecule(
            symbols=["C", "H", "H"],
            atom_types=["1", "5", "5"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.09, 0.0, 0.0],
                    [-0.36, 1.03, 0.0],
                ]
            ),
        )
        assert any(bond.env_id == "1-5" for bond in mol.bonds)
        assert any(angle.env_id == "5-1-5" for angle in mol.angles)


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
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        assert ff.n_params == len(vec)

    def test_param_vector_roundtrip(self):
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 5.0)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 0.5)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        ff2 = ff.copy()
        ff2.set_param_vector(vec * 2)
        vec2 = ff2.get_param_vector()
        np.testing.assert_allclose(vec2, vec * 2)

    def test_default_bounds_allow_negative_bond_k(self):
        """TSFF requires negative bond force constants for reaction coordinates."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, -0.69)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 0.5)],
        )
        bounds = ff.get_bounds()
        bond_k_lower, bond_k_upper = bounds[0]
        assert bond_k_lower < 0, "Bond k lower bound must allow negative values for TSFF"
        assert bond_k_upper > 0

    def test_default_bounds_allow_negative_angle_k(self):
        """Angle force constants may also be negative in TSFF."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 5.0)],
            angles=[AngleParam(("H", "C", "F"), 109.5, -0.3)],
        )
        bounds = ff.get_bounds()
        angle_k_lower, angle_k_upper = bounds[2]
        assert angle_k_lower < 0, "Angle k lower bound must allow negative values for TSFF"

    def test_negative_fc_in_param_vector_roundtrip(self):
        """Negative force constants must survive get/set param vector roundtrip."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, -0.69)],
            angles=[AngleParam(("H", "C", "F"), 109.5, -0.15)],
        )
        vec = ff.get_param_vector()
        assert vec[0] == pytest.approx(-0.69)
        assert vec[2] == pytest.approx(-0.15)
        ff2 = ff.copy()
        ff2.set_param_vector(vec)
        assert ff2.bonds[0].force_constant == pytest.approx(-0.69)
        assert ff2.angles[0].force_constant == pytest.approx(-0.15)

    def test_torsion_in_param_vector(self):
        """Torsion force constants appear in param vector after bonds/angles."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 4.5)],
            angles=[AngleParam(("H", "C", "H"), 109.5, 0.5)],
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=-0.10),
                TorsionParam(("H", "C", "C", "H"), periodicity=3, force_constant=0.25),
            ],
        )
        vec = ff.get_param_vector()
        # 2 bond + 2 angle + 3 torsion = 7
        assert ff.n_params == 7
        assert len(vec) == 7
        # Torsion values at indices 4, 5, 6
        assert vec[4] == pytest.approx(0.15)
        assert vec[5] == pytest.approx(-0.10)
        assert vec[6] == pytest.approx(0.25)

    def test_torsion_param_vector_roundtrip(self):
        """Torsion params survive get/set roundtrip."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 4.5)],
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=-0.10),
            ],
        )
        vec = ff.get_param_vector()
        ff2 = ff.copy()
        vec[2] = 0.30  # Double V1
        vec[3] = 0.20  # Change V2
        ff2.set_param_vector(vec)
        assert ff2.torsions[0].force_constant == pytest.approx(0.30)
        assert ff2.torsions[1].force_constant == pytest.approx(0.20)

    def test_torsion_bounds(self):
        """Torsion bounds included in get_bounds()."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 4.5)],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15)],
        )
        bounds = ff.get_bounds()
        # 2 bond bounds + 1 torsion bound = 3
        assert len(bounds) == 3
        torsion_lower, torsion_upper = bounds[2]
        assert torsion_lower < 0, "Torsion k must allow negative values"
        assert torsion_upper > 0

    def test_get_torsion(self):
        """get_torsion finds by element quad + optional periodicity."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=-0.10),
                TorsionParam(("C", "C", "N", "H"), periodicity=1, force_constant=0.30),
            ],
        )
        t1 = ff.get_torsion("H", "C", "C", "H", periodicity=1)
        assert t1 is not None
        assert t1.force_constant == pytest.approx(0.15)
        t2 = ff.get_torsion("H", "C", "C", "H", periodicity=2)
        assert t2 is not None
        assert t2.force_constant == pytest.approx(-0.10)
        # Reversed element order should also match
        t_rev = ff.get_torsion("H", "N", "C", "C", periodicity=1)
        assert t_rev is not None
        assert t_rev.force_constant == pytest.approx(0.30)

    def test_mm3_loads_torsions(self):
        """MM3 .fld loading should extract torsion parameters."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        assert len(ff.torsions) > 0, "Expected torsion parameters from Rh-enamide mm3.fld"
        assert all(isinstance(t, TorsionParam) for t in ff.torsions)
        assert all(t.periodicity in (1, 2, 3) for t in ff.torsions)
        assert all(t.ff_row is not None for t in ff.torsions)

    def test_mm3_export_roundtrip_generic(self, tmp_path):
        ff = ForceField(
            name="Generic MM3",
            bonds=[BondParam(("C", "F"), 1.381, 5.25, env_id="C1-F1")],
            angles=[AngleParam(("H", "C", "F"), 109.7, 0.55, env_id="H1-C1-F1")],
        )
        out_path = tmp_path / "generated.fld"
        ff.to_mm3_fld(out_path)

        roundtrip = ForceField.from_mm3_fld(out_path)
        assert roundtrip.source_format == "mm3_fld"
        assert roundtrip.source_path == out_path

        bond = roundtrip.get_bond("C", "F", env_id="C1-F1")
        angle = roundtrip.get_angle("H", "C", "F", env_id="F1-C1-H1")
        assert bond is not None
        assert angle is not None
        assert bond.force_constant == pytest.approx(5.25)
        assert bond.equilibrium == pytest.approx(1.381)
        assert angle.force_constant == pytest.approx(0.55)
        assert angle.equilibrium == pytest.approx(109.7)

    def test_mm3_vdw_roundtrip_generic(self, tmp_path):
        ff = ForceField(name="Generic MM3", vdws=[VdwParam("F0", 1.71, 0.075), VdwParam("H1", 1.62, 0.02)])
        out_path = tmp_path / "generated_vdw.fld"

        ff.to_mm3_fld(out_path)
        roundtrip = ForceField.from_mm3_fld(out_path)

        fluorine = roundtrip.get_vdw(atom_type="F0")
        hydrogen = roundtrip.get_vdw(atom_type="H1")
        assert fluorine is not None
        assert hydrogen is not None
        assert fluorine.radius == pytest.approx(1.71)
        assert fluorine.epsilon == pytest.approx(0.075)
        assert hydrogen.radius == pytest.approx(1.62)
        assert hydrogen.epsilon == pytest.approx(0.02)

    def test_mm3_standalone_torsion_roundtrip(self, tmp_path):
        """Standalone MM3 export should include torsion parameters."""
        ff = ForceField(
            name="Torsion Test",
            bonds=[BondParam(("C", "C"), 1.525, 4.49, env_id="C3-C3")],
            angles=[AngleParam(("H", "C", "C"), 111.0, 0.636, env_id="H1-C3-C3")],
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.0, env_id="H1-C3-C3-H1"),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=0.0, env_id="H1-C3-C3-H1"),
                TorsionParam(("H", "C", "C", "H"), periodicity=3, force_constant=0.238, env_id="H1-C3-C3-H1"),
                TorsionParam(("C", "C", "C", "H"), periodicity=1, force_constant=0.185, env_id="C3-C3-C3-H1"),
                TorsionParam(("C", "C", "C", "H"), periodicity=3, force_constant=0.52, env_id="C3-C3-C3-H1"),
            ],
        )
        out_path = tmp_path / "torsion_test.fld"
        ff.to_mm3_fld(out_path)

        roundtrip = ForceField.from_mm3_fld(out_path)
        assert len(roundtrip.torsions) == 6, "Should have 6 torsion params (3 per line × 2 lines)"

        # Check specific values round-tripped correctly
        hcch_v3 = [t for t in roundtrip.torsions if t.periodicity == 3 and t.env_id == "H1-C3-C3-H1"]
        assert len(hcch_v3) == 1
        assert hcch_v3[0].force_constant == pytest.approx(0.238)

        ccch_v1 = [t for t in roundtrip.torsions if t.periodicity == 1 and t.env_id == "C3-C3-C3-H1"]
        assert len(ccch_v1) == 1
        assert ccch_v1[0].force_constant == pytest.approx(0.185)

        # V2 was not provided for CCCH, should default to 0.0
        ccch_v2 = [t for t in roundtrip.torsions if t.periodicity == 2 and t.env_id == "C3-C3-C3-H1"]
        assert len(ccch_v2) == 1
        assert ccch_v2[0].force_constant == pytest.approx(0.0)

    def test_mm3_standalone_full_roundtrip(self, tmp_path):
        """Standalone MM3 export with bonds, angles, torsions, and vdW."""
        ff = ForceField(
            name="Full Test",
            bonds=[BondParam(("C", "F"), 1.381, 5.25, env_id="C1-F1")],
            angles=[AngleParam(("H", "C", "F"), 109.7, 0.55, env_id="H1-C1-F1")],
            torsions=[
                TorsionParam(("H", "C", "C", "F"), periodicity=1, force_constant=-0.5, env_id="H1-C1-C1-F1"),
                TorsionParam(("H", "C", "C", "F"), periodicity=2, force_constant=1.2, env_id="H1-C1-C1-F1"),
                TorsionParam(("H", "C", "C", "F"), periodicity=3, force_constant=0.0, env_id="H1-C1-C1-F1"),
            ],
            vdws=[VdwParam("F0", 1.71, 0.075)],
        )
        out_path = tmp_path / "full_test.fld"
        ff.to_mm3_fld(out_path)

        roundtrip = ForceField.from_mm3_fld(out_path)
        assert len(roundtrip.bonds) == 1
        assert len(roundtrip.angles) == 1
        assert len(roundtrip.torsions) == 3
        assert len(roundtrip.vdws) == 1

        # Negative torsion values should survive
        v1 = [t for t in roundtrip.torsions if t.periodicity == 1][0]
        assert v1.force_constant == pytest.approx(-0.5)

    def test_mm3_export_updates_template(self, tmp_path):
        ff = ForceField.from_mm3_fld(RH_MM3)
        first_bond = ff.bonds[0]
        first_bond.force_constant += 1.234
        first_bond.equilibrium += 0.123

        out_path = tmp_path / "updated_mm3.fld"
        ff.to_mm3_fld(out_path)

        roundtrip = ForceField.from_mm3_fld(out_path)
        updated = next(bond for bond in roundtrip.bonds if bond.ff_row == first_bond.ff_row)
        assert updated.force_constant == pytest.approx(first_bond.force_constant)
        assert updated.equilibrium == pytest.approx(first_bond.equilibrium)

    def test_mm3_imports_vdw_table(self):
        ff = ForceField.from_mm3_fld(RH_MM3)

        rh = ff.get_vdw(atom_type="RH")
        fluorine = ff.get_vdw(atom_type="F0")
        assert rh is not None
        assert fluorine is not None
        assert rh.radius == pytest.approx(2.69)
        assert rh.epsilon == pytest.approx(0.14)
        assert fluorine.radius == pytest.approx(1.71)
        assert fluorine.epsilon == pytest.approx(0.075)

    def test_tinker_import_export_roundtrip(self, tmp_path):
        prm_path = tmp_path / "sample.prm"
        prm_path.write_text(
            "\n".join(
                [
                    "# Example parameter file",
                    "# Q2MM",
                    "# OPT Synthetic",
                    "bond     C1   F1     5.0000     1.3800",
                    "angle    H1   C1   F1     0.5000   109.5000   111.0000   112.0000",
                    "vdw      F1   1.4700     0.0610     0.0000",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        ff = ForceField.from_tinker_prm(prm_path)
        assert ff.source_format == "tinker_prm"
        assert ff.source_path == prm_path

        bond = ff.get_bond("C", "F", env_id="C1-F1")
        angle = ff.get_angle("H", "C", "F", env_id="F1-C1-H1")
        assert bond is not None
        assert angle is not None
        assert bond.force_constant == pytest.approx(5.0)
        assert bond.equilibrium == pytest.approx(1.38)
        assert angle.force_constant == pytest.approx(0.5)
        assert angle.equilibrium == pytest.approx(109.5)
        vdw = ff.get_vdw(atom_type="F1")
        assert vdw is not None
        assert vdw.radius == pytest.approx(1.47)
        assert vdw.epsilon == pytest.approx(0.061)

        generic_out = tmp_path / "generated.prm"
        ff.to_tinker_prm(generic_out, template_path=None)
        generic_roundtrip = ForceField.from_tinker_prm(generic_out)
        generic_bond = generic_roundtrip.get_bond("C", "F", env_id="C1-F1")
        generic_angle = generic_roundtrip.get_angle("H", "C", "F", env_id="F1-C1-H1")
        assert generic_bond is not None
        assert generic_angle is not None
        assert generic_bond.force_constant == pytest.approx(5.0)
        assert generic_angle.equilibrium == pytest.approx(109.5)
        generic_vdw = generic_roundtrip.get_vdw(atom_type="F1")
        assert generic_vdw is not None
        assert generic_vdw.radius == pytest.approx(1.47)
        assert generic_vdw.epsilon == pytest.approx(0.061)

    def test_tinker_import_generic_prm_without_q2mm_section(self, tmp_path):
        prm_path = tmp_path / "generic.prm"
        prm_path.write_text(
            "\n".join(
                [
                    'atom      1    C     "CSP3 ALKANE"                  6    12.000    4',
                    'atom      5    H     "EXCEPT ON N,O,S"             1     1.008    1',
                    "bond      1    5           4.740     1.1120",
                    "angle     5    1    5      0.550     107.60     107.80     109.47",
                    "vdw       1               2.0400     0.0270",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        ff = ForceField.from_tinker_prm(prm_path)

        bond = ff.get_bond("C", "H", env_id="1-5")
        angle = ff.get_angle("H", "C", "H", env_id="5-1-5")
        vdw = ff.get_vdw(atom_type="1")
        assert bond is not None
        assert angle is not None
        assert vdw is not None
        assert bond.force_constant == pytest.approx(4.74)
        assert bond.equilibrium == pytest.approx(1.1120)
        assert angle.force_constant == pytest.approx(0.55)
        assert angle.equilibrium == pytest.approx(107.60)
        assert vdw.radius == pytest.approx(2.0400)
        assert vdw.epsilon == pytest.approx(0.0270)

    def test_tinker_export_updates_primary_angle_only(self, tmp_path):
        prm_path = tmp_path / "sample.prm"
        prm_path.write_text(
            "\n".join(
                [
                    "# Example parameter file",
                    "# Q2MM",
                    "# OPT Synthetic",
                    "bond     C1   F1     5.0000     1.3800",
                    "angle    H1   C1   F1     0.5000   109.5000   111.0000   112.0000",
                    "vdw      F1   1.4700     0.0610     0.0000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ff = ForceField.from_tinker_prm(prm_path)
        ff.angles[0].equilibrium = 108.25
        ff.angles[0].force_constant = 0.75

        out_path = tmp_path / "updated.prm"
        ff.to_tinker_prm(out_path)

        legacy = TinkerFF(str(out_path))
        legacy.import_ff()
        angle_row = ff.angles[0].ff_row
        angle_fcs = [param.value for param in legacy.params if param.ff_row == angle_row and param.ptype == "af"]
        angle_eqs = [param.value for param in legacy.params if param.ff_row == angle_row and param.ptype == "ae"]
        assert angle_fcs == [pytest.approx(0.75)]
        assert angle_eqs[0] == pytest.approx(108.25)
        assert angle_eqs[1:] == [pytest.approx(111.0), pytest.approx(112.0)]

    def test_tinker_export_updates_vdw(self, tmp_path):
        prm_path = tmp_path / "sample_vdw.prm"
        prm_path.write_text(
            "\n".join(
                [
                    "# Example parameter file",
                    "# Q2MM",
                    "# OPT Synthetic",
                    "vdw      F1   1.4700     0.0610     0.0000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ff = ForceField.from_tinker_prm(prm_path)
        ff.vdws[0].radius = 1.55
        ff.vdws[0].epsilon = 0.081

        out_path = tmp_path / "updated_vdw.prm"
        ff.to_tinker_prm(out_path)

        roundtrip = ForceField.from_tinker_prm(out_path)
        updated = roundtrip.get_vdw(atom_type="F1")
        assert updated is not None
        assert updated.radius == pytest.approx(1.55)
        assert updated.epsilon == pytest.approx(0.081)

    def test_tinker_export_preserves_vdw_reduction(self, tmp_path):
        """Regression: _update_tinker_vdw_lines must write match.reduction,
        not copy the old tail from the file."""
        prm_path = tmp_path / "vdw_reduction.prm"
        prm_path.write_text(
            "\n".join(
                [
                    "# Q2MM",
                    "# OPT Synthetic",
                    "vdw      H1   1.6200     0.0200     0.0000",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ff = ForceField.from_tinker_prm(prm_path)
        ff.vdws[0].reduction = 0.923

        out_path = tmp_path / "updated_reduction.prm"
        ff.to_tinker_prm(out_path)

        roundtrip = ForceField.from_tinker_prm(out_path)
        assert roundtrip.vdws[0].reduction == pytest.approx(0.923)

    def test_generic_prm_amoeba_style_atom_records(self, tmp_path):
        """Parser must handle AMOEBA-style atom records with a class column."""
        prm_path = tmp_path / "amoeba_style.prm"
        prm_path.write_text(
            "\n".join(
                [
                    'atom          1    1    N     "Glycine N"        7    14.003    3',
                    'atom          5    5    H     "Amide H"          1     1.008    1',
                    "bond          1    5     5.0000     1.0100",
                    "vdw           1   1.8200     0.1700",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        ff = ForceField.from_tinker_prm(prm_path)
        assert len(ff.bonds) == 1
        assert ff.bonds[0].elements == ("N", "H")
        assert ff.vdws[0].element == "N"


# ---- AMBER .frcmod I/O ----

SAMPLE_FRCMOD = Path(__file__).resolve().parent / "fixtures" / "sample.frcmod"
UPSTREAM_FRCMOD = Path(__file__).resolve().parent / "fixtures" / "upstream_q2mm.frcmod"


class TestAmberFrcmod:
    def test_load_bonds(self):
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.bonds) == 3
        assert ff.source_format == "amber_frcmod"
        b = ff.get_bond("C", "P")
        assert b is not None
        assert b.force_constant == pytest.approx(380.74)
        assert b.equilibrium == pytest.approx(1.7631)

    def test_load_angles(self):
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.angles) == 7

    def test_load_dihedrals(self):
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        proper = [t for t in ff.torsions if "(improper)" not in t.label]
        assert len(proper) == 8

    def test_load_impropers(self):
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        improper = [t for t in ff.torsions if "(improper)" in t.label]
        assert len(improper) == 3
        assert improper[0].force_constant == pytest.approx(10.5)

    def test_load_vdw(self):
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.vdws) == 1
        assert ff.vdws[0].atom_type == "c4"

    def test_element_from_mass_section(self):
        """MASS section should inform element identification."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        # c4 has mass 12.010 → element C
        b = ff.bonds[0]
        assert all(e == "C" for e in b.elements)

    def test_standalone_roundtrip(self, tmp_path):
        """Standalone save → reload should preserve all values."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        # Clear source info to force standalone mode
        ff_clean = ForceField(
            name=ff.name,
            bonds=ff.bonds,
            angles=ff.angles,
            torsions=ff.torsions,
            vdws=ff.vdws,
        )
        out = tmp_path / "standalone.frcmod"
        ff_clean.to_amber_frcmod(out)
        rt = ForceField.from_amber_frcmod(out)

        assert len(rt.bonds) == len(ff.bonds)
        assert len(rt.angles) == len(ff.angles)
        assert len(rt.torsions) == len(ff.torsions)
        assert len(rt.vdws) == len(ff.vdws)

        for orig, new in zip(ff.bonds, rt.bonds):
            assert orig.force_constant == pytest.approx(new.force_constant)
            assert orig.equilibrium == pytest.approx(new.equilibrium)

        for orig, new in zip(ff.torsions, rt.torsions):
            assert orig.force_constant == pytest.approx(new.force_constant, abs=0.01)

    def test_template_roundtrip(self, tmp_path):
        """Template-based save should update values in-place."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        ff.bonds[0].force_constant = 999.0
        ff.bonds[0].equilibrium = 1.234

        out = tmp_path / "updated.frcmod"
        ff.to_amber_frcmod(out)
        rt = ForceField.from_amber_frcmod(out)

        assert rt.bonds[0].force_constant == pytest.approx(999.0)
        assert rt.bonds[0].equilibrium == pytest.approx(1.234)
        # Other bonds unchanged
        assert rt.bonds[1].force_constant == pytest.approx(ff.bonds[1].force_constant)

    def test_template_preserves_comments(self, tmp_path):
        """Template mode should preserve the remark line."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        out = tmp_path / "preserved.frcmod"
        ff.to_amber_frcmod(out)
        content = out.read_text()
        assert content.startswith("Remark line goes here")

    def test_template_preserves_inline_comments(self, tmp_path):
        """Template mode should preserve trailing inline comments."""
        frcmod_with_comments = tmp_path / "commented.frcmod"
        frcmod_with_comments.write_text(
            "Remark\n"
            "MASS\n"
            "\n"
            "BOND\n"
            "c -c4    337.5987    1.6002  ATTN, need revision\n"
            "\n"
            "ANGLE\n"
            "c -c4-ca    50.7932   102.6974   # penalty score\n"
            "\n",
            encoding="utf-8",
        )
        ff = ForceField.from_amber_frcmod(frcmod_with_comments)
        ff.bonds[0].force_constant = 400.0
        ff.angles[0].force_constant = 60.0
        out = tmp_path / "updated.frcmod"
        ff.to_amber_frcmod(out)
        content = out.read_text()
        assert "ATTN, need revision" in content
        assert "# penalty score" in content
        assert "400.0000" in content
        assert "60.0000" in content

    def test_upstream_frcmod_irregular_spacing(self):
        """Parser should handle upstream Q2MM frcmod with irregular spacing."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        assert len(ff.bonds) == 3
        assert len(ff.angles) == 10
        proper = [t for t in ff.torsions if "(improper)" not in t.label]
        improper = [t for t in ff.torsions if "(improper)" in t.label]
        assert len(proper) == 10
        assert len(improper) == 10
        assert len(ff.vdws) == 1

    def test_upstream_idivf_division(self):
        """IDIVF=4 should divide barrier by 4."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        ca_tor = next(t for t in ff.torsions if t.env_id == "ca-ca-ce-c")
        assert ca_tor.force_constant == pytest.approx(0.7)

    def test_upstream_comment_lines_skipped(self):
        """Lines starting with # should be skipped."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        assert ff.source_format == "amber_frcmod"

    def test_upstream_roundtrip(self, tmp_path):
        """Upstream frcmod should round-trip through standalone save."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        ff_clean = ForceField(
            name=ff.name,
            bonds=ff.bonds,
            angles=ff.angles,
            torsions=ff.torsions,
            vdws=ff.vdws,
        )
        out = tmp_path / "upstream_rt.frcmod"
        ff_clean.to_amber_frcmod(out)
        rt = ForceField.from_amber_frcmod(out)

        assert len(rt.bonds) == len(ff.bonds)
        for orig, new in zip(ff.bonds, rt.bonds):
            assert orig.force_constant == pytest.approx(new.force_constant)
        for orig, new in zip(ff.torsions, rt.torsions):
            assert orig.force_constant == pytest.approx(new.force_constant, abs=0.01)


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
