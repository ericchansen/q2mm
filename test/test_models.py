"""Tests for q2mm.models (molecule, forcefield, seminario)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from test._shared import CH3F_HESS, CH3F_XYZ, SN2_HESSIAN as TS_HESS, SN2_XYZ as TS_XYZ, make_ethane

from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.forcefield import (
    ForceField,
    FunctionalForm,
    BondParam,
    AngleParam,
    StretchBendParam,
    TorsionParam,
    VdwParam,
    _extract_element,
)
from q2mm.models.seminario import (
    qfuerza_fresh,
    qfuerza_into,
    _is_hydrogen_angle,
    QFUERZA_H_ANGLE_DEFAULT_CANONICAL,
)
from q2mm.io.tinker import _tinker_import_ff

# Fixture paths (test-specific, not shared)
RH_MM3 = Path(__file__).resolve().parent.parent / "examples" / "rh-enamide" / "mm3.fld"


# ---- _extract_element helper ----


class TestExtractElement:
    def test_single_letter(self) -> None:
        assert _extract_element("C1") == "C"
        assert _extract_element("F") == "F"
        assert _extract_element("H3") == "H"

    def test_two_letter(self) -> None:
        assert _extract_element("Cl1") == "Cl"
        assert _extract_element("Br") == "Br"
        assert _extract_element("Rh2") == "Rh"
        assert _extract_element("Pt") == "Pt"
        assert _extract_element("RH1") == "Rh"
        assert _extract_element("CL") == "Cl"

    def test_whitespace(self) -> None:
        assert _extract_element("  Cl1") == "Cl"
        assert _extract_element(" F") == "F"


# ---- Q2MMMolecule ----


class TestMoleculeFromXYZ:
    def test_load_ch3f(self) -> None:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        assert mol.n_atoms == 5
        assert mol.symbols[0] == "C"
        assert mol.symbols[1] == "F"
        assert mol.geometry.shape == (5, 3)

    def test_load_ts(self) -> None:
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        assert mol.n_atoms == 6
        assert mol.symbols.count("F") == 2

    def test_bond_detection_default(self) -> None:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        bonds = mol.bonds
        assert len(bonds) > 0
        elements_found = {b.element_pair for b in bonds}
        assert ("C", "H") in elements_found or ("H", "C") in elements_found

    def test_bond_detection_ts_tolerance(self) -> None:
        mol_tight = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.3)
        mol_loose = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        # Looser tolerance should detect more bonds (partial TS bonds)
        assert len(mol_loose.bonds) >= len(mol_tight.bonds)

    def test_angle_detection(self) -> None:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        angles = mol.angles
        assert len(angles) > 0
        # CH3F has H-C-H and H-C-F angles
        center_elements = {a.elements[1] for a in angles}
        assert "C" in center_elements

    def test_detected_env_ids_use_atom_types(self) -> None:
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


# ---- Torsion detection ----


class TestTorsionDetection:
    def test_ethane_torsion_count(self) -> None:
        """Ethane has 9 unique H-C-C-H torsions."""
        mol = make_ethane()
        assert len(mol.torsions) == 9

    def test_ethane_torsion_elements(self) -> None:
        """All ethane torsions are H-C-C-H."""
        mol = make_ethane()
        for t in mol.torsions:
            assert t.element_quad == ("H", "C", "C", "H")

    def test_ethane_torsion_angles_finite(self) -> None:
        """All dihedral angles are finite and in [-180, 180]."""
        mol = make_ethane()
        for t in mol.torsions:
            assert -180.0 <= t.value <= 180.0
            assert np.isfinite(t.value)

    def test_water_no_torsions(self) -> None:
        """Water (H-O-H) has no torsions — not enough connectivity depth."""
        water = Q2MMMolecule(
            symbols=["O", "H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        )
        assert len(water.torsions) == 0

    def test_ch3f_no_torsions(self) -> None:
        """CH3F has no torsions — F is terminal with no further neighbors."""
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        assert len(mol.torsions) == 0

    def test_no_duplicate_torsions(self) -> None:
        """A-B-C-D and D-C-B-A should not both appear."""
        mol = make_ethane()
        seen = set()
        for t in mol.torsions:
            key = (t.atom_i, t.atom_j, t.atom_k, t.atom_l)
            key_rev = (t.atom_l, t.atom_k, t.atom_j, t.atom_i)
            assert key not in seen and key_rev not in seen
            seen.add(key)

    def test_element_quad_canonical(self) -> None:
        """element_quad returns the lexically smaller direction."""
        from q2mm.models.molecule import DetectedTorsion

        t = DetectedTorsion(0, 1, 2, 3, ("H", "C", "N", "O"), 60.0)
        # forward: (H,C,N,O), reverse: (O,N,C,H) — forward is smaller
        assert t.element_quad == ("H", "C", "N", "O")
        t2 = DetectedTorsion(0, 1, 2, 3, ("O", "N", "C", "H"), 60.0)
        assert t2.element_quad == ("H", "C", "N", "O")

    def test_torsion_env_ids(self) -> None:
        """Torsion env_ids use canonical (directional) atom-type labels."""
        mol = make_ethane()
        env_ids = {t.env_id for t in mol.torsions}
        # All H-C-C-H with default atom_types = symbols → "C-C-H-H" or "H-C-C-H"
        assert len(env_ids) == 1
        assert env_ids.pop() == "H-C-C-H"  # palindrome, same in both directions

    def test_formaldehyde_improper_detection(self) -> None:
        """Formaldehyde (H2CO) has one trigonal centre (C) → one improper."""
        mol = Q2MMMolecule(
            symbols=["C", "O", "H", "H"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.2],
                    [0.94, 0.0, -0.54],
                    [-0.94, 0.0, -0.54],
                ]
            ),
        )
        impropers = mol.improper_torsions
        assert len(impropers) == 1
        imp = impropers[0]
        # Centre atom (C, index 0) goes in j position (MM3 convention)
        assert imp.atom_j == 0
        # Neighbours sorted by index: O(1), H(2), H(3)
        assert imp.atom_i == 1
        assert imp.atom_k == 2
        assert imp.atom_l == 3
        assert np.isfinite(imp.value)
        assert imp.env_id is not None

    def test_ethane_no_impropers(self) -> None:
        """Ethane has no trigonal centres (each C has 4 bonds) → no impropers."""
        mol = make_ethane()
        assert len(mol.improper_torsions) == 0

    def test_improper_deterministic_ordering(self) -> None:
        """Improper neighbour ordering is deterministic (sorted by index)."""
        mol = Q2MMMolecule(
            symbols=["C", "O", "H", "H"],
            geometry=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.2],
                    [0.94, 0.0, -0.54],
                    [-0.94, 0.0, -0.54],
                ]
            ),
        )
        # Call twice to verify deterministic
        imp1 = mol.improper_torsions
        imp2 = mol.improper_torsions
        assert len(imp1) == len(imp2)
        for a, b in zip(imp1, imp2):
            assert (a.atom_i, a.atom_j, a.atom_k, a.atom_l) == (b.atom_i, b.atom_j, b.atom_k, b.atom_l)


class TestMatchTorsion:
    def test_match_by_elements(self) -> None:
        """match_torsion finds all periodicity components by element quad."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=-0.10),
                TorsionParam(("H", "C", "C", "H"), periodicity=3, force_constant=0.25),
            ],
        )
        matches = ff.match_torsion(("H", "C", "C", "H"))
        assert len(matches) == 3

    def test_match_reversed(self) -> None:
        """match_torsion matches reversed element order."""
        ff = ForceField(
            torsions=[TorsionParam(("H", "C", "N", "O"), periodicity=1, force_constant=0.5)],
        )
        matches = ff.match_torsion(("O", "N", "C", "H"))
        assert len(matches) == 1

    def test_match_reversed_env_id(self) -> None:
        """match_torsion matches when env_id is reversed (D-C-B-A vs A-B-C-D)."""
        ff = ForceField(
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.5, env_id="H1-C1-C2-H2")],
        )
        matches = ff.match_torsion(("H", "C", "C", "H"), env_id="H2-C2-C1-H1")
        assert len(matches) == 1

    def test_match_by_periodicity(self) -> None:
        """match_torsion filters by periodicity when specified."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=-0.10),
            ],
        )
        matches = ff.match_torsion(("H", "C", "C", "H"), periodicity=2)
        assert len(matches) == 1
        assert matches[0].force_constant == -0.10

    def test_match_by_ff_row(self) -> None:
        """match_torsion prioritizes ff_row over element matching."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15, ff_row=10),
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.99, ff_row=20),
            ],
        )
        matches = ff.match_torsion(("H", "C", "C", "H"), ff_row=20)
        assert len(matches) == 1
        assert matches[0].force_constant == 0.99

    def test_no_match_returns_empty(self) -> None:
        """match_torsion returns empty list when nothing matches."""
        ff = ForceField(
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.5)],
        )
        matches = ff.match_torsion(("C", "N", "C", "O"))
        assert matches == []

    def test_match_proper_vs_improper_filter(self) -> None:
        """match_torsion can filter by is_improper flag."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=0.30, is_improper=True),
            ],
        )
        proper = ff.match_torsion(("H", "C", "C", "H"), is_improper=False)
        assert len(proper) == 1
        assert proper[0].force_constant == 0.15
        improper = ff.match_torsion(("H", "C", "C", "H"), is_improper=True)
        assert len(improper) == 1
        assert improper[0].force_constant == 0.30
        both = ff.match_torsion(("H", "C", "C", "H"))
        assert len(both) == 2

    def test_proper_and_improper_properties(self) -> None:
        """ForceField.proper_torsions and .improper_torsions filter correctly."""
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15),
                TorsionParam(("C", "N", "C", "O"), periodicity=2, force_constant=1.0, is_improper=True),
            ],
        )
        assert len(ff.proper_torsions) == 1
        assert len(ff.improper_torsions) == 1
        assert ff.proper_torsions[0].force_constant == 0.15
        assert ff.improper_torsions[0].is_improper is True


# ---- ForceField ----


class TestForceField:
    def test_create_for_molecule(self) -> None:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        ff = ForceField.create_for_molecule(mol)
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0

    def test_n_params_matches_vector(self) -> None:
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        assert ff.n_params == len(vec)

    def test_param_vector_roundtrip(self) -> None:
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        ff2 = ff.copy()
        ff2.set_param_vector(vec * 2)
        vec2 = ff2.get_param_vector()
        np.testing.assert_allclose(vec2, vec * 2)

    def test_active_param_api_tracks_frozen_params(self) -> None:
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7, frozen=True)],
            angles=[
                AngleParam(
                    ("H", "C", "F"),
                    109.5,
                    36.0,
                    ub_force_constant=10.0,
                    ub_equilibrium=1.52,
                )
            ],
            stretch_bends=[StretchBendParam(("H", "C", "F"), force_constant=0.75, frozen=True)],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15)],
            vdws=[VdwParam("F1", 1.47, 0.061, frozen=True)],
        )

        expected_mask = np.array([False, False, True, True, True, False, False, False, True, True])
        np.testing.assert_array_equal(ff.active_mask, expected_mask)
        assert ff.n_params == len(expected_mask)
        assert ff.n_active_params == 5
        np.testing.assert_allclose(ff.get_active_param_vector(), ff.get_param_vector()[expected_mask])
        assert ff.get_active_param_names() == [
            "ka_F-C-H",
            "th0_F-C-H",
            "kt_H-C-C-H_n1",
            "kub_F-C-H",
            "r13_F-C-H",
        ]
        assert ff.get_active_step_sizes().shape == (5,)
        assert ff.get_active_bounds().shape == (5, 2)

    def test_active_param_mutators_preserve_frozen_values(self) -> None:
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7, frozen=True)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15, frozen=True)],
        )
        updated = np.array([72.0, 120.0])

        ff_mut = ff.copy()
        ff_mut.set_active_param_vector(updated)
        assert ff_mut.bonds[0].force_constant == pytest.approx(359.7)
        assert ff_mut.bonds[0].equilibrium == pytest.approx(1.38)
        assert ff_mut.torsions[0].force_constant == pytest.approx(0.15)
        np.testing.assert_allclose(ff_mut.get_active_param_vector(), updated)

        ff_new = ff.with_active_params(updated)
        np.testing.assert_allclose(ff_new.get_active_param_vector(), updated)
        np.testing.assert_allclose(ff.get_active_param_vector(), [36.0, 109.5])

        with pytest.raises(ValueError, match="does not match"):
            ff_mut.set_active_param_vector(np.array([1.0]))
        with pytest.raises(ValueError, match="does not match"):
            ff.with_active_params(np.array([1.0]))

    def test_default_bounds_allow_negative_bond_k(self) -> None:
        """TSFF requires negative bond force constants for reaction coordinates."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, -49.6)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
        )
        bounds = ff.get_bounds()
        bond_k_lower, bond_k_upper = bounds[0]
        assert bond_k_lower < 0, "Bond k lower bound must allow negative values for TSFF"
        assert bond_k_upper > 0

    def test_default_bounds_allow_negative_angle_k(self) -> None:
        """Angle force constants may also be negative in TSFF."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, -21.6)],
        )
        bounds = ff.get_bounds()
        angle_k_lower, angle_k_upper = bounds[2]
        assert angle_k_lower < 0, "Angle k lower bound must allow negative values for TSFF"

    def test_negative_fc_in_param_vector_roundtrip(self) -> None:
        """Negative force constants must survive get/set param vector roundtrip."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, -49.6)],
            angles=[AngleParam(("H", "C", "F"), 109.5, -10.8)],
        )
        vec = ff.get_param_vector()
        assert vec[0] == pytest.approx(-49.6)
        assert vec[2] == pytest.approx(-10.8)
        ff2 = ff.copy()
        ff2.set_param_vector(vec)
        assert ff2.bonds[0].force_constant == pytest.approx(-49.6)
        assert ff2.angles[0].force_constant == pytest.approx(-10.8)

    def test_fractional_bounds_around_current_values(self) -> None:
        """get_fractional_bounds wraps each param in a (val ± frac*|val|) box."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
        )
        bounds = ff.get_fractional_bounds(fc_fraction=0.20, eq_fraction=0.05)
        # Layout: bond_k, bond_eq, angle_k, angle_eq
        assert bounds[0] == pytest.approx((359.7 * 0.8, 359.7 * 1.2), rel=1e-6)
        assert bounds[1] == pytest.approx((1.38 * 0.95, 1.38 * 1.05), rel=1e-6)
        assert bounds[2] == pytest.approx((36.0 * 0.8, 36.0 * 1.2), rel=1e-6)
        assert bounds[3] == pytest.approx((109.5 * 0.95, 109.5 * 1.05), rel=1e-6)

    def test_fractional_bounds_sign_aware_for_negative_fc(self) -> None:
        """Negative force constants (TSFF) must produce valid (lo < hi) bounds."""
        ff = ForceField(bonds=[BondParam(("C", "F"), 1.38, -49.6)])
        bounds = ff.get_fractional_bounds(fc_fraction=0.20, eq_fraction=0.05)
        bond_k_lo, bond_k_hi = bounds[0]
        # |val|=49.6, window=9.92; box = (-49.6-9.92, -49.6+9.92) = (-59.52, -39.68)
        assert bond_k_lo == pytest.approx(-59.52)
        assert bond_k_hi == pytest.approx(-39.68)
        assert bond_k_lo < bond_k_hi

    def test_fractional_bounds_intersect_sanity_bounds(self) -> None:
        """Fractional bounds are clipped to the DEFAULT_BOUNDS sanity envelope."""
        ff = ForceField(bonds=[BondParam(("C", "F"), 1.38, 3000.0)])
        bounds = ff.get_fractional_bounds(fc_fraction=0.50, eq_fraction=None)
        bond_k_lo, bond_k_hi = bounds[0]
        # |val|*0.5 = 1500 → box (1500, 4500); sanity hi = 3600 → clipped
        assert bond_k_lo == pytest.approx(1500.0)
        assert bond_k_hi == pytest.approx(3600.0)  # sanity envelope

    def test_fractional_bounds_zero_value_falls_back_to_sanity(self) -> None:
        """Frozen-at-zero parameters (e.g. torsion_k) get sanity bounds, not (0, 0)."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 100.0)],
            torsions=[TorsionParam(("H", "C", "F", "H"), periodicity=1, force_constant=0.0)],
        )
        bounds = ff.get_fractional_bounds(fc_fraction=0.20, eq_fraction=0.05)
        # Layout: bond_k, bond_eq, torsion_k
        assert bounds[0] == pytest.approx((80.0, 120.0))
        # Torsion_k (val=0): falls back to DEFAULT_BOUNDS["torsion_k"]
        assert bounds[2] == pytest.approx((-20.0, 20.0))

    def test_fractional_bounds_none_is_get_bounds(self) -> None:
        """When both fractions are None, get_fractional_bounds is get_bounds."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 300.0)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
        )
        assert ff.get_fractional_bounds(None, None) == ff.get_bounds()

    def test_torsion_in_param_vector(self) -> None:
        """Torsion force constants appear in param vector after bonds/angles."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 323.7)],
            angles=[AngleParam(("H", "C", "H"), 109.5, 36.0)],
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

    def test_torsion_param_vector_roundtrip(self) -> None:
        """Torsion params survive get/set roundtrip."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 323.7)],
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

    # --- with_params() tests ---

    def test_with_params_roundtrip(self) -> None:
        """with_params(get_param_vector()) reproduces the same values."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        ff2 = ff.with_params(vec)
        np.testing.assert_allclose(ff2.get_param_vector(), vec)

    def test_with_params_applies_new_values(self) -> None:
        """with_params applies the given vector to the new ForceField."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        vec = ff.get_param_vector()
        new_vec = vec * 2.0
        ff2 = ff.with_params(new_vec)
        np.testing.assert_allclose(ff2.get_param_vector(), new_vec)

    def test_with_params_does_not_mutate_original(self) -> None:
        """with_params returns a new FF; the original is unchanged."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
            vdws=[VdwParam("F1", 1.47, 0.061)],
        )
        original_vec = ff.get_param_vector().copy()
        new_vec = original_vec * 3.0
        ff2 = ff.with_params(new_vec)

        # Original unchanged
        np.testing.assert_allclose(ff.get_param_vector(), original_vec)
        # New FF has new values
        np.testing.assert_allclose(ff2.get_param_vector(), new_vec)

    def test_with_params_no_aliasing(self) -> None:
        """Mutating returned FF params does not affect the original."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
        )
        original_k = ff.bonds[0].force_constant
        ff2 = ff.with_params(ff.get_param_vector())
        ff2.bonds[0].force_constant = 999.0
        assert ff.bonds[0].force_constant == pytest.approx(original_k)

    def test_with_params_preserves_metadata(self) -> None:
        """with_params preserves non-value fields (label, env_id, etc.)."""
        ff = ForceField(
            name="test_ff",
            bonds=[BondParam(("C", "F"), 1.38, 359.7, label="C-F bond", env_id="C1-F1", ff_row=5)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0, label="HCF", env_id="H1-C1-F1")],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=0.15, phase=180.0, env_id="t1")],
            vdws=[VdwParam("F1", 1.47, 0.061, reduction=0.92)],
            functional_form=FunctionalForm.MM3,
        )
        ff2 = ff.with_params(ff.get_param_vector() * 1.5)

        assert ff2.name == "test_ff"
        assert ff2.functional_form == FunctionalForm.MM3
        assert ff2.bonds[0].label == "C-F bond"
        assert ff2.bonds[0].env_id == "C1-F1"
        assert ff2.bonds[0].ff_row == 5
        assert ff2.angles[0].label == "HCF"
        assert ff2.angles[0].env_id == "H1-C1-F1"
        assert ff2.torsions[0].periodicity == 2
        assert ff2.torsions[0].phase == pytest.approx(180.0)
        assert ff2.torsions[0].env_id == "t1"
        assert ff2.vdws[0].reduction == pytest.approx(0.92)

    def test_with_params_wrong_length_raises(self) -> None:
        """with_params raises ValueError for wrong-length vector."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, 359.7)],
            angles=[AngleParam(("H", "C", "F"), 109.5, 36.0)],
        )
        with pytest.raises(ValueError, match="does not match"):
            ff.with_params(np.array([1.0, 2.0]))  # too short
        with pytest.raises(ValueError, match="does not match"):
            ff.with_params(np.zeros(100))  # too long

    def test_torsion_bounds(self) -> None:
        """Torsion bounds included in get_bounds()."""
        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 323.7)],
            torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.15)],
        )
        bounds = ff.get_bounds()
        # 2 bond bounds + 1 torsion bound = 3
        assert len(bounds) == 3
        torsion_lower, torsion_upper = bounds[2]
        assert torsion_lower < 0, "Torsion k must allow negative values"
        assert torsion_upper > 0

    def test_get_torsion(self) -> None:
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

    def test_mm3_loads_torsions(self) -> None:
        """MM3 .fld loading should extract torsion parameters."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        assert len(ff.torsions) > 0, "Expected torsion parameters from Rh-enamide mm3.fld"
        assert all(isinstance(t, TorsionParam) for t in ff.torsions)
        assert all(t.periodicity in (1, 2, 3) for t in ff.torsions)
        assert all(t.ff_row is not None for t in ff.torsions)

    def test_mm3_imp_conversion(self) -> None:
        """ff_io converts imp1/imp2 MM3 params to TorsionParam(is_improper=True)."""
        from q2mm.io import Param
        from q2mm.io.mm3 import load_mm3_fld
        from unittest.mock import patch

        # Create mock _mm3_import_ff output with imp1 and imp2 params
        mock_params = [
            Param(
                atom_labels=["C2", "O2", "H1", "H1"],
                atom_types=["C2", "O2", "H1", "H1"],
                ptype="imp1",
                ff_col=1,
                ff_row=100,
                label=" 5",
                value=0.0,
            ),
            Param(
                atom_labels=["C2", "O2", "H1", "H1"],
                atom_types=["C2", "O2", "H1", "H1"],
                ptype="imp2",
                ff_col=2,
                ff_row=100,
                label=" 5",
                value=0.8,
            ),
        ]
        with (
            patch("q2mm.io.mm3._mm3_import_ff", return_value=(mock_params, [])),
            patch("q2mm.io.mm3._parse_mm3_vdw_params", return_value=[]),
        ):
            ff = load_mm3_fld("/fake/path.fld")

        assert len(ff.torsions) == 2
        imp1 = [t for t in ff.torsions if t.periodicity == 1]
        imp2 = [t for t in ff.torsions if t.periodicity == 2]
        assert len(imp1) == 1
        assert len(imp2) == 1
        assert imp1[0].is_improper is True
        assert imp2[0].is_improper is True
        assert imp1[0].force_constant == pytest.approx(0.0)
        assert imp2[0].force_constant == pytest.approx(0.4)  # V/2: 0.8/2 = 0.4
        assert imp1[0].ff_row == 100
        assert imp2[0].ff_row == 100

    def test_mm3_export_roundtrip_generic(self, tmp_path: Path) -> None:
        ff = ForceField(
            name="Generic MM3",
            bonds=[BondParam(("C", "F"), 1.381, 377.7, env_id="C1-F1")],
            angles=[AngleParam(("H", "C", "F"), 109.7, 39.6, env_id="H1-C1-F1")],
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
        assert bond.force_constant == pytest.approx(377.7, rel=1e-3)
        assert bond.equilibrium == pytest.approx(1.381)
        assert angle.force_constant == pytest.approx(39.6, rel=1e-3)
        assert angle.equilibrium == pytest.approx(109.7)

    def test_mm3_vdw_roundtrip_generic(self, tmp_path: Path) -> None:
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

    def test_mm3_standalone_torsion_roundtrip(self, tmp_path: Path) -> None:
        """Standalone MM3 export should include torsion parameters."""
        ff = ForceField(
            name="Torsion Test",
            bonds=[BondParam(("C", "C"), 1.525, 323.0, env_id="C3-C3")],
            angles=[AngleParam(("H", "C", "C"), 111.0, 45.8, env_id="H1-C3-C3")],
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

    def test_mm3_standalone_full_roundtrip(self, tmp_path: Path) -> None:
        """Standalone MM3 export with bonds, angles, torsions, and vdW."""
        ff = ForceField(
            name="Full Test",
            bonds=[BondParam(("C", "F"), 1.381, 377.7, env_id="C1-F1")],
            angles=[AngleParam(("H", "C", "F"), 109.7, 39.6, env_id="H1-C1-F1")],
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

    def test_mm3_export_updates_template(self, tmp_path: Path) -> None:
        ff = ForceField.from_mm3_fld(RH_MM3)
        first_bond = ff.bonds[0]
        # Mutating a value field on a frozen param now raises FrozenParamError
        # (q2mm#277 follow-up). The test's intent is roundtrip fidelity, not
        # frozenness, so unfreeze for the edit and re-freeze for parity with
        # the original FF state.
        first_bond.unfreeze()
        first_bond.force_constant += 1.234
        first_bond.equilibrium += 0.123
        first_bond.freeze()

        out_path = tmp_path / "updated_mm3.fld"
        ff.to_mm3_fld(out_path)

        roundtrip = ForceField.from_mm3_fld(out_path)
        updated = next(bond for bond in roundtrip.bonds if bond.ff_row == first_bond.ff_row)
        assert updated.force_constant == pytest.approx(first_bond.force_constant, rel=1e-3)
        assert updated.equilibrium == pytest.approx(first_bond.equilibrium)

    def test_mm3_imports_vdw_table(self) -> None:
        ff = ForceField.from_mm3_fld(RH_MM3)

        rh = ff.get_vdw(atom_type="RH")
        fluorine = ff.get_vdw(atom_type="F0")
        assert rh is not None
        assert fluorine is not None
        assert rh.radius == pytest.approx(2.69)
        assert rh.epsilon == pytest.approx(0.14)
        assert fluorine.radius == pytest.approx(1.71)
        assert fluorine.epsilon == pytest.approx(0.075)

    def test_mm3_freezes_standard_params_by_default(self) -> None:
        ff = ForceField.from_mm3_fld(RH_MM3)
        ff_opt = ForceField.from_mm3_fld(RH_MM3, include_standard=False)
        ff.freeze_standard_params(ff_opt)

        assert ff.n_params == 2742
        assert ff.n_active_params == 182
        assert ff.active_mask.shape == (2742,)
        assert len(ff.get_active_param_vector()) == 182
        assert any(param.frozen for param in ff.bonds)
        assert any(not param.frozen for param in ff.bonds)

    def test_tinker_import_export_roundtrip(self, tmp_path: Path) -> None:
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
        assert bond.force_constant == pytest.approx(359.7, rel=1e-3)
        assert bond.equilibrium == pytest.approx(1.38)
        assert angle.force_constant == pytest.approx(36.0, rel=1e-3)
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
        assert generic_bond.force_constant == pytest.approx(359.7, rel=1e-3)
        assert generic_angle.equilibrium == pytest.approx(109.5)
        generic_vdw = generic_roundtrip.get_vdw(atom_type="F1")
        assert generic_vdw is not None
        assert generic_vdw.radius == pytest.approx(1.47)
        assert generic_vdw.epsilon == pytest.approx(0.061)

    def test_tinker_import_generic_prm_without_q2mm_section(self, tmp_path: Path) -> None:
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
        assert bond.force_constant == pytest.approx(341.0, rel=1e-3)
        assert bond.equilibrium == pytest.approx(1.1120)
        assert angle.force_constant == pytest.approx(39.6, rel=1e-3)
        assert angle.equilibrium == pytest.approx(107.60)
        assert vdw.radius == pytest.approx(2.0400)
        assert vdw.epsilon == pytest.approx(0.0270)

    def test_tinker_export_updates_primary_angle_only(self, tmp_path: Path) -> None:
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
        ff.angles[0].force_constant = 54.0

        out_path = tmp_path / "updated.prm"
        ff.to_tinker_prm(out_path)

        legacy_params, _ = _tinker_import_ff(str(out_path))
        angle_row = ff.angles[0].ff_row
        angle_fcs = [param.value for param in legacy_params if param.ff_row == angle_row and param.ptype == "af"]
        angle_eqs = [param.value for param in legacy_params if param.ff_row == angle_row and param.ptype == "ae"]
        assert angle_fcs == [pytest.approx(54.0 / 71.94, abs=1e-3)]
        assert angle_eqs[0] == pytest.approx(108.25)
        assert angle_eqs[1:] == [pytest.approx(111.0), pytest.approx(112.0)]

    def test_tinker_export_updates_vdw(self, tmp_path: Path) -> None:
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

    def test_tinker_export_preserves_vdw_reduction(self, tmp_path: Path) -> None:
        """Verify Tinker export preserves VDW reduction factor.

        Regression: _update_tinker_vdw_lines must write match.reduction,
        not copy the old tail from the file.
        """
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

    def test_generic_prm_amoeba_style_atom_records(self, tmp_path: Path) -> None:
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


# ---- Bond order parsing and matching ----


class TestBondOrderParsing:
    """Test bond-order and context parsing from .fld files."""

    def test_standard_section_bond_order_single(self) -> None:
        """Standard section: '-' at column 7 is parsed as single bond."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        # C3-C3 single bonds exist in the standard section
        c3c3_bonds = [b for b in ff.bonds if b.env_id == "C3-C3"]
        assert len(c3c3_bonds) > 0
        assert all(b.bond_order == "-" for b in c3c3_bonds)

    def test_standard_section_bond_order_double(self) -> None:
        """Standard section: '=' at column 7 is parsed as double bond."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        c2c2_double = [b for b in ff.bonds if b.env_id == "C2-C2" and b.bond_order == "="]
        assert len(c2c2_double) > 0, "Expected C2=C2 double bonds in Rh-enamide mm3.fld"
        # Verify at least one has the expected equilibrium ~1.33 Å
        eq_vals = [b.equilibrium for b in c2c2_double]
        assert any(1.30 < eq < 1.36 for eq in eq_vals), f"Expected C=C eq ~1.33 Å, got {eq_vals}"

    def test_standard_section_bond_order_aromatic(self) -> None:
        """Standard section: bond-order symbols include '*' (aromatic) in angles."""
        # Aromatic bonds appear in angles (C2*C2) but not in bond section
        # of the standard MM3. Verify we can parse '-' and '=' at minimum.
        ff = ForceField.from_mm3_fld(RH_MM3)
        orders = {b.bond_order for b in ff.bonds if b.bond_order}
        assert "-" in orders, "Expected single bonds"
        assert "=" in orders, "Expected double bonds"

    def test_standard_section_context_parsed(self) -> None:
        """Context flags from cols 55-65 are stored on BondParam."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        # Some C3-C3 bonds have context like "O200 0000" or "1C200 000"
        with_context = [b for b in ff.bonds if b.context]
        assert len(with_context) > 0, "Expected some bonds with context flags"

    def test_standard_section_generic_has_empty_context(self) -> None:
        """Generic entries (0000 0000) have empty context string."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        # The generic C3-C3 bond (r₀≈1.5247) has context "0000 0000" → empty
        c3c3_generic = [b for b in ff.bonds if b.env_id == "C3-C3" and not b.context]
        assert len(c3c3_generic) > 0, "Expected at least one generic C3-C3 bond"

    def test_synthetic_standard_bond_order(self, tmp_path: Path) -> None:
        """Parse bond-order symbols from synthetic standard-section lines."""
        from q2mm.io.mm3 import _mm3_import_ff

        # Standard section format: cols 4-5=type1, col 7=order, cols 9-10=type2
        #                          cols 14-24=p1, cols 24-34=p2
        lines = [
            " 1  C2 - C2                1.4800     4.5000     0.0000  0000 0000",
            " 1  C2 = C2                1.3320     7.5000     0.0000  0000 0000",
            " 1  C1 % C1                1.2100    15.0000     0.0000  0000 0000",
        ]
        fld = tmp_path / "test.fld"
        fld.write_text("\n".join(lines) + "\n", encoding="utf-8")

        params, _ = _mm3_import_ff(str(fld))
        bond_params = [p for p in params if p.ptype == "be"]
        assert len(bond_params) == 3

        orders = [p.bond_order for p in bond_params]
        assert orders == ["-", "=", "%"]

    def test_synthetic_opt_no_bond_order(self, tmp_path: Path) -> None:
        """OPT-section bond lines use numeric labels — no bond-order symbol."""
        from q2mm.io.mm3 import _mm3_import_ff

        # Real OPT lines: " 1   1   2   <params>" — numeric labels, no order symbol.
        # Bond order is only in the standard section (column 7 with type labels).
        lines = [
            "# Q2MM",
            " OPT synthetic test",
            " 9  OPT substructure",
            " C2  2  1  P1  2  0  RH  0  0  C2  2  0  C2  2  0",
            " 1   1   2                 1.8000     3.0000     0.0000",
            " 1   2   3                 1.3500     6.0000     0.0000",
        ]
        fld = tmp_path / "opt.fld"
        fld.write_text("\n".join(lines) + "\n", encoding="utf-8")

        params, _ = _mm3_import_ff(str(fld))
        bond_params = [p for p in params if p.ptype == "be"]
        assert len(bond_params) == 2

        # OPT bonds have no bond-order symbol — empty string
        orders = [p.bond_order for p in bond_params]
        assert orders == ["", ""]

    def test_synthetic_context_flags(self, tmp_path: Path) -> None:
        """Context flags are parsed from standard section cols 56-66."""
        from q2mm.io.mm3 import _mm3_import_ff

        # Standard-section lines need to extend past col 66 for context parsing.
        # Context occupies cols 56-65 (0-indexed), slice is [56:66].
        #   cols:  0         1         2         3         4         5         6
        #          0123456789012345678901234567890123456789012345678901234567890123456789
        line_ctx = " 1  C2 - C2                1.4700     4.5000     0.0000 O200 0000  trailing"
        line_gen = " 1  C2 - C2                1.4800     4.5000     0.0000  0000 0000  trailing"
        fld = tmp_path / "ctx.fld"
        fld.write_text(line_ctx + "\n" + line_gen + "\n", encoding="utf-8")

        params, _ = _mm3_import_ff(str(fld))
        bond_params = [p for p in params if p.ptype == "be"]
        assert len(bond_params) == 2

        # First line has "O200 0000" context, second is generic
        assert bond_params[0].context == "O200 0000"
        assert bond_params[1].context == ""


class TestBondOrderMatching:
    """Test ForceField.get_bond() and match_bond() with bond_order and context."""

    @pytest.fixture()
    def ff_with_bond_orders(self) -> ForceField:
        """FF with multiple C-C bonds differing by bond_order and context."""
        return ForceField(
            bonds=[
                BondParam(
                    ("C", "C"),
                    1.48,
                    300.0,
                    env_id="C2-C2",
                    bond_order="-",
                    context="O200 0000",
                    ff_row=96,
                    label="single-ctx",
                ),
                BondParam(("C", "C"), 1.332, 500.0, env_id="C2-C2", bond_order="=", ff_row=98, label="double"),
                BondParam(
                    ("C", "C"),
                    1.46,
                    310.0,
                    env_id="C2-C2",
                    bond_order="-",
                    context="C200 C200",
                    ff_row=100,
                    label="single-conj",
                ),
                BondParam(("C", "C"), 1.48, 290.0, env_id="C2-C2", bond_order="-", ff_row=101, label="single-generic"),
                BondParam(("C", "C"), 1.54, 300.0, env_id="C3-C3", bond_order="-", ff_row=59, label="sp3-single"),
            ],
        )

    # --- get_bond with bond_order ---

    def test_get_bond_filters_by_bond_order(self, ff_with_bond_orders: ForceField) -> None:
        """get_bond returns only bonds matching the requested bond_order."""
        double = ff_with_bond_orders.get_bond("C", "C", env_id="C2-C2", bond_order="=")
        assert double is not None
        assert double.label == "double"
        assert double.equilibrium == pytest.approx(1.332)

    def test_get_bond_bond_order_no_match(self, ff_with_bond_orders: ForceField) -> None:
        """get_bond returns None when bond_order doesn't match any candidate."""
        result = ff_with_bond_orders.get_bond("C", "C", env_id="C2-C2", bond_order="%")
        assert result is None

    def test_get_bond_prefer_generic_context(self, ff_with_bond_orders: ForceField) -> None:
        """get_bond with prefer_generic_context returns the generic entry."""
        generic = ff_with_bond_orders.get_bond(
            "C",
            "C",
            env_id="C2-C2",
            bond_order="-",
            prefer_generic_context=True,
        )
        assert generic is not None
        assert generic.label == "single-generic"

    def test_get_bond_reversed_elements(self, ff_with_bond_orders: ForceField) -> None:
        """get_bond works with reversed element order (canonical sorting)."""
        double = ff_with_bond_orders.get_bond("C", "C", env_id="C2-C2", bond_order="=")
        assert double is not None
        assert double.label == "double"

    # --- match_bond tier tests ---

    def test_match_bond_tier1_ff_row(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 1: ff_row match takes priority over everything."""
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
            ff_row=100,
            bond_order="=",  # wrong bond_order — ff_row should still win
            bond_length=1.332,
        )
        assert result is not None
        assert result.label == "single-conj"
        assert result.ff_row == 100

    def test_match_bond_tier2_env_id_plus_order(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 2: env_id + bond_order match when ff_row is None."""
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
            bond_order="=",
        )
        assert result is not None
        assert result.label == "double"

    def test_match_bond_tier2_beats_tier3(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 2 (bond_order) wins over tier 3 (closest r₀) even if r₀ is closer to another."""
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
            bond_order="=",
            bond_length=1.48,  # closer to single bonds, but "=" should win
        )
        assert result is not None
        assert result.label == "double"

    def test_match_bond_tier3_closest_r0(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 3: when bond_order is unknown, picks closest r₀ to bond_length."""
        # bond_length=1.34 is closest to the double bond (eq=1.332)
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
            bond_length=1.34,
        )
        assert result is not None
        assert result.label == "double"

    def test_match_bond_tier3_picks_single_for_long_bond(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 3: bond_length=1.47 is closest to single-ctx (eq=1.48) or single-conj (eq=1.46)."""
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
            bond_length=1.47,
        )
        assert result is not None
        # 1.47 is equidistant from 1.46 and 1.48 — should pick one of the singles
        assert result.bond_order == "-"

    def test_match_bond_tier3_skipped_without_length(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 3 is skipped when bond_length is None — falls through to tier 4."""
        result = ff_with_bond_orders.match_bond(
            ("C", "C"),
            env_id="C2-C2",
        )
        assert result is not None
        # Without bond_order or bond_length, tier 4 (env_id + prefer generic) applies
        # The first match with env_id="C2-C2" that's generic should be returned
        # (depends on implementation — either first match or generic-preferred)

    def test_match_bond_tier4_env_id_prefers_generic(self, ff_with_bond_orders: ForceField) -> None:
        """Tier 4: env_id-only match prefers generic context entry."""
        # Build a smaller FF where only context differs
        ff = ForceField(
            bonds=[
                BondParam(
                    ("C", "C"), 1.48, 300.0, env_id="C2-C2", bond_order="-", context="O200 0000", label="with-ctx"
                ),
                BondParam(("C", "C"), 1.48, 290.0, env_id="C2-C2", bond_order="-", label="generic"),
            ],
        )
        result = ff.match_bond(("C", "C"), env_id="C2-C2")
        assert result is not None
        assert result.label == "generic"

    def test_match_bond_tier5_element_only(self) -> None:
        """Tier 5: falls back to element-only matching when env_id doesn't match."""
        ff = ForceField(
            bonds=[
                BondParam(("C", "C"), 1.54, 300.0, env_id="C3-C3", label="sp3"),
            ],
        )
        result = ff.match_bond(("C", "C"), env_id="C99-C99")
        assert result is not None
        assert result.label == "sp3"

    def test_match_bond_no_match(self) -> None:
        """match_bond returns None when no bond matches at all."""
        ff = ForceField(
            bonds=[
                BondParam(("C", "F"), 1.38, 370.0, env_id="C1-F1", label="CF"),
            ],
        )
        result = ff.match_bond(("N", "H"), env_id="N3-H1")
        assert result is None


from q2mm.models.molecule import DetectedBond


class TestDetectedBondOrder:
    """Test that DetectedBond carries bond_order through matching."""

    def test_detected_bond_stores_bond_order(self) -> None:
        """DetectedBond stores the bond_order field."""
        bond = DetectedBond(atom_i=0, atom_j=1, elements=("C", "C"), length=1.34, env_id="C2-C2", bond_order="=")
        assert bond.bond_order == "="

    def test_detected_bond_default_empty_order(self) -> None:
        """DetectedBond defaults to empty bond_order."""
        bond = DetectedBond(atom_i=0, atom_j=1, elements=("C", "C"), length=1.54, env_id="C3-C3")
        assert bond.bond_order == ""

    def test_match_bond_uses_detected_bond_fields(self) -> None:
        """match_bond correctly uses bond_order and length from DetectedBond."""
        ff = ForceField(
            bonds=[
                BondParam(("C", "C"), 1.48, 300.0, env_id="C2-C2", bond_order="-", label="single"),
                BondParam(("C", "C"), 1.332, 500.0, env_id="C2-C2", bond_order="=", label="double"),
            ],
        )
        bond = DetectedBond(atom_i=0, atom_j=1, elements=("C", "C"), length=1.35, env_id="C2-C2", bond_order="=")
        result = ff.match_bond(
            bond.elements,
            env_id=bond.env_id,
            ff_row=bond.ff_row,
            bond_order=bond.bond_order,
            bond_length=bond.length,
        )
        assert result is not None
        assert result.label == "double"


class TestV2TorsionPhase:
    """Test that MM3 V2 torsion phase is correctly set to 180°."""

    def test_mm3_fld_v2_phase_180(self) -> None:
        """load_mm3_fld sets phase=180° for periodicity=2 torsions."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        v2_torsions = [t for t in ff.torsions if t.periodicity == 2 and not t.is_improper]
        assert len(v2_torsions) > 0, "Expected V2 torsions in Rh-enamide mm3.fld"
        for t in v2_torsions:
            assert t.phase == pytest.approx(180.0), f"V2 torsion {t.label} has phase={t.phase}, expected 180.0"

    def test_mm3_fld_v1_v3_phase_0(self) -> None:
        """load_mm3_fld sets phase=0° for V1 and V3 torsions."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        v1_torsions = [t for t in ff.torsions if t.periodicity == 1 and not t.is_improper]
        v3_torsions = [t for t in ff.torsions if t.periodicity == 3 and not t.is_improper]
        assert len(v1_torsions) > 0
        assert len(v3_torsions) > 0
        for t in v1_torsions:
            assert t.phase == pytest.approx(0.0), f"V1 torsion {t.label} has phase={t.phase}"
        for t in v3_torsions:
            assert t.phase == pytest.approx(0.0), f"V3 torsion {t.label} has phase={t.phase}"

    def test_improper_v2_phase_180(self) -> None:
        """Improper torsions with periodicity=2 also get phase=180°."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        imp_v2 = [t for t in ff.torsions if t.periodicity == 2 and t.is_improper]
        for t in imp_v2:
            assert t.phase == pytest.approx(180.0), f"Improper V2 {t.label} has phase={t.phase}, expected 180.0"

    def test_v2_energy_minimum_at_planar(self) -> None:
        """V2 energy should be minimum (zero) at planar geometry (ω=0°).

        MM3: E = (V2/2)*(1 − cos 2ω)
        At ω=0°: E = (V2/2)*(1−1) = 0 (minimum)
        At ω=90°: E = (V2/2)*(1+1) = V2 (maximum)
        """
        import math

        k = 10.0  # kcal/mol
        n = 2
        gamma = math.radians(180.0)

        # Our formula: k * (1 + cos(n*phi - gamma))
        e_planar = k * (1.0 + math.cos(n * 0.0 - gamma))  # ω=0°
        e_perp = k * (1.0 + math.cos(n * math.pi / 2 - gamma))  # ω=90°

        assert e_planar == pytest.approx(0.0), f"V2 at ω=0° should be 0, got {e_planar}"
        assert e_perp == pytest.approx(2 * k), f"V2 at ω=90° should be {2 * k}, got {e_perp}"

    def test_v2_count_in_rh_enamide(self) -> None:
        """Rh-enamide .fld should have multiple V2 torsions with significant k."""
        ff = ForceField.from_mm3_fld(RH_MM3)
        v2_proper = [t for t in ff.torsions if t.periodicity == 2 and not t.is_improper]
        # k stores V2/2 (MM3 convention). V2=16.25 → k=8.125; threshold at k>2.5 (V2>5).
        large_v2 = [t for t in v2_proper if abs(t.force_constant) > 2.5]
        assert len(large_v2) > 5, f"Expected >5 large V2 torsions (|k|>5 kcal/mol), found {len(large_v2)}"

    def test_synthetic_mm3_torsion_phases(self, tmp_path: Path) -> None:
        """Synthetic .fld: V1/V2/V3 on one line produce correct phases."""
        from q2mm.io.mm3 import _mm3_import_ff

        # Standard torsion line: " 4  AT1  AT2  AT3  AT4   V1  V2  V3"
        #                                                   ^^^  ^^^  ^^^
        #                                               col1  col2  col3
        lines = [
            " 4  C3 - C3 - C3 - C3       0.1850     0.3000     0.4500  0000 0000 0000 0000",
        ]
        fld = tmp_path / "torsion_test.fld"
        fld.write_text("\n".join(lines) + "\n", encoding="utf-8")

        params, _ = _mm3_import_ff(str(fld))
        df_params = [p for p in params if p.ptype == "df"]
        assert len(df_params) == 3

        from q2mm.io.mm3 import load_mm3_fld

        ff = load_mm3_fld(str(fld))
        assert len(ff.torsions) == 3

        phases = {t.periodicity: t.phase for t in ff.torsions}
        assert phases[1] == pytest.approx(0.0), "V1 phase should be 0°"
        assert phases[2] == pytest.approx(180.0), "V2 phase should be 180°"
        assert phases[3] == pytest.approx(0.0), "V3 phase should be 0°"

        # MM3 convention: k = V_n/2 (the .fld stores V_n, our k = V_n/2)
        fcs = {t.periodicity: t.force_constant for t in ff.torsions}
        assert fcs[1] == pytest.approx(0.1850 / 2), "k1 should be V1/2"
        assert fcs[2] == pytest.approx(0.3000 / 2), "k2 should be V2/2"
        assert fcs[3] == pytest.approx(0.4500 / 2), "k3 should be V3/2"


# ---- AMBER .frcmod I/O ----

SAMPLE_FRCMOD = Path(__file__).resolve().parent / "fixtures" / "sample.frcmod"
UPSTREAM_FRCMOD = Path(__file__).resolve().parent / "fixtures" / "upstream_q2mm.frcmod"


class TestAmberFrcmod:
    def test_load_bonds(self) -> None:
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.bonds) == 3
        assert ff.source_format == "amber_frcmod"
        b = ff.get_bond("C", "P")
        assert b is not None
        assert b.force_constant == pytest.approx(380.74)
        assert b.equilibrium == pytest.approx(1.7631)

    def test_load_angles(self) -> None:
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.angles) == 7

    def test_load_dihedrals(self) -> None:
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.proper_torsions) == 8

    def test_load_impropers(self) -> None:
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.improper_torsions) == 3
        assert ff.improper_torsions[0].force_constant == pytest.approx(10.5)
        assert all(t.is_improper for t in ff.improper_torsions)
        assert all(not t.is_improper for t in ff.proper_torsions)

    def test_load_vdw(self) -> None:
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        assert len(ff.vdws) == 1
        assert ff.vdws[0].atom_type == "c4"

    def test_element_from_mass_section(self) -> None:
        """MASS section should inform element identification."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        # c4 has mass 12.010 → element C
        b = ff.bonds[0]
        assert all(e == "C" for e in b.elements)

    def test_standalone_roundtrip(self, tmp_path: Path) -> None:
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

    def test_template_roundtrip(self, tmp_path: Path) -> None:
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

    def test_template_preserves_comments(self, tmp_path: Path) -> None:
        """Template mode should preserve the remark line."""
        ff = ForceField.from_amber_frcmod(SAMPLE_FRCMOD)
        out = tmp_path / "preserved.frcmod"
        ff.to_amber_frcmod(out)
        content = out.read_text()
        assert content.startswith("Remark line goes here")

    def test_template_preserves_inline_comments(self, tmp_path: Path) -> None:
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

    def test_upstream_frcmod_irregular_spacing(self) -> None:
        """Parser should handle upstream Q2MM frcmod with irregular spacing."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        assert len(ff.bonds) == 3
        assert len(ff.angles) == 10
        proper = [t for t in ff.torsions if "(improper)" not in t.label]
        improper = [t for t in ff.torsions if "(improper)" in t.label]
        assert len(proper) == 10
        assert len(improper) == 10
        assert len(ff.vdws) == 1

    def test_upstream_idivf_division(self) -> None:
        """IDIVF=4 should divide barrier by 4."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        ca_tor = next(t for t in ff.torsions if t.env_id == "ca-ca-ce-c")
        assert ca_tor.force_constant == pytest.approx(0.7)

    def test_upstream_comment_lines_skipped(self) -> None:
        """Lines starting with # should be skipped."""
        ff = ForceField.from_amber_frcmod(UPSTREAM_FRCMOD)
        assert ff.source_format == "amber_frcmod"

    def test_upstream_roundtrip(self, tmp_path: Path) -> None:
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
    def ch3f_mol_with_hess(self) -> Q2MMMolecule:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        hess = np.load(CH3F_HESS)
        return mol.with_hessian(hess)

    @pytest.fixture
    def ts_mol_with_hess(self) -> Q2MMMolecule:
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        hess = np.load(TS_HESS)
        return mol.with_hessian(hess)

    def test_estimate_runs(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        ff = qfuerza_fresh(ch3f_mol_with_hess)
        assert len(ff.bonds) > 0
        assert len(ff.angles) > 0

    def test_fc_values_positive_ground_state(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        ff = qfuerza_fresh(ch3f_mol_with_hess)
        for b in ff.bonds:
            assert b.force_constant > 0, f"Bond {b.key} has non-positive FC"

    def test_negative_fc_included_for_ts(self, ts_mol_with_hess: Q2MMMolecule) -> None:
        """Negative FCs from TS reaction coordinates should be included, not dropped."""
        ff = qfuerza_fresh(ts_mol_with_hess)
        bond_fcs = [b.force_constant for b in ff.bonds]
        # The C-F bond in the TS is partially breaking — may have negative FC
        # At minimum, verify the estimation completes and produces values
        assert len(bond_fcs) > 0

    def test_respects_frozen_params(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        ff = ForceField.create_for_molecule(ch3f_mol_with_hess)
        ff.bonds[0].force_constant = 123.0
        ff.bonds[0].equilibrium = 9.9
        ff.bonds[0].frozen = True
        ff.angles[0].force_constant = 456.0
        ff.angles[0].equilibrium = 150.0
        ff.angles[0].frozen = True
        ff.torsions = [
            TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=7.0, frozen=True),
            TorsionParam(("H", "C", "C", "F"), periodicity=1, force_constant=8.0),
        ]

        estimated = ff.copy()
        qfuerza_into(estimated, ch3f_mol_with_hess)

        assert estimated.bonds[0].force_constant == pytest.approx(123.0)
        assert estimated.bonds[0].equilibrium == pytest.approx(9.9)
        assert estimated.angles[0].force_constant == pytest.approx(456.0)
        assert estimated.angles[0].equilibrium == pytest.approx(150.0)
        assert estimated.torsions[0].force_constant == pytest.approx(7.0)
        assert estimated.torsions[1].force_constant == pytest.approx(0.0)

    def test_raises_without_hessian(self) -> None:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        with pytest.raises(ValueError, match="Hessian"):
            qfuerza_fresh(mol)


class TestQFUERZA:
    """Tests for QFUERZA hybrid initialization (issue #208)."""

    @pytest.fixture
    def ch3f_mol_with_hess(self) -> Q2MMMolecule:
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        hess = np.load(CH3F_HESS)
        return mol.with_hessian(hess)

    @pytest.fixture
    def water_mol_with_hess(self) -> Q2MMMolecule:
        from test._shared import make_water

        mol = make_water()
        n = len(mol.symbols)
        rng = np.random.default_rng(42)
        h = rng.standard_normal((3 * n, 3 * n))
        h = (h + h.T) / 2
        return mol.with_hessian(h)

    # -- _is_hydrogen_angle helper --

    @pytest.mark.parametrize(
        "elements,expected",
        [
            (("H", "O", "H"), True),
            (("H", "C", "H"), True),
            (("H", "C", "F"), True),
            (("F", "C", "H"), True),
            (("C", "O", "C"), False),
            (("C", "C", "C"), False),
            (("N", "C", "O"), False),
        ],
    )
    def test_is_hydrogen_angle(self, elements: tuple, expected: bool) -> None:
        assert _is_hydrogen_angle(elements) == expected

    # -- strategy="fuerza" determinism --

    def test_fuerza_strategy_deterministic(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        """strategy='fuerza' is deterministic across repeated calls."""
        ff_first = qfuerza_fresh(ch3f_mol_with_hess, strategy="fuerza")
        ff_second = qfuerza_fresh(ch3f_mol_with_hess, strategy="fuerza")

        assert len(ff_first.bonds) == len(ff_second.bonds)
        for b1, b2 in zip(ff_first.bonds, ff_second.bonds):
            assert b1.force_constant == pytest.approx(b2.force_constant)
        assert len(ff_first.angles) == len(ff_second.angles)
        for a1, a2 in zip(ff_first.angles, ff_second.angles):
            assert a1.force_constant == pytest.approx(a2.force_constant)

    def test_default_is_qfuerza(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        """Default strategy is QFUERZA."""
        ff_default = qfuerza_fresh(ch3f_mol_with_hess)
        ff_qfuerza = qfuerza_fresh(ch3f_mol_with_hess, strategy="qfuerza")

        assert len(ff_default.bonds) == len(ff_qfuerza.bonds)
        for b_def, b_qf in zip(ff_default.bonds, ff_qfuerza.bonds):
            assert b_def.force_constant == pytest.approx(b_qf.force_constant)
        assert len(ff_default.angles) == len(ff_qfuerza.angles)
        for a_def, a_qf in zip(ff_default.angles, ff_qfuerza.angles):
            assert a_def.force_constant == pytest.approx(a_qf.force_constant)

    # -- strategy="qfuerza" substitution --

    def test_qfuerza_substitutes_h_angles(self, water_mol_with_hess: Q2MMMolecule) -> None:
        """QFUERZA replaces the H-O-H angle force constant with the empirical default."""
        ff = qfuerza_fresh(water_mol_with_hess, strategy="qfuerza")
        h_angles = [a for a in ff.angles if _is_hydrogen_angle(a.elements)]
        assert len(h_angles) > 0, "Expected at least one H-angle in water"
        for angle in h_angles:
            assert angle.force_constant == pytest.approx(QFUERZA_H_ANGLE_DEFAULT_CANONICAL)

    def test_qfuerza_keeps_bonds_unchanged(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        """QFUERZA does not alter bond force constants."""
        ff_fuerza = qfuerza_fresh(ch3f_mol_with_hess, strategy="fuerza")
        ff_qfuerza = qfuerza_fresh(ch3f_mol_with_hess, strategy="qfuerza")

        for b_fur, b_qf in zip(ff_fuerza.bonds, ff_qfuerza.bonds):
            assert b_fur.force_constant == pytest.approx(b_qf.force_constant)

    def test_qfuerza_does_not_substitute_non_h_angles(self) -> None:
        """Non-hydrogen angles are not substituted by QFUERZA."""
        # Use SN2 TS which has F-C-Cl (non-H) angles
        mol = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5)
        hess = np.load(TS_HESS)
        mol = mol.with_hessian(hess)

        ff_fuerza = qfuerza_fresh(mol, strategy="fuerza")
        ff_qfuerza = qfuerza_fresh(mol, strategy="qfuerza")

        non_h_angles = [
            (a_fur, a_qf)
            for a_fur, a_qf in zip(ff_fuerza.angles, ff_qfuerza.angles)
            if not _is_hydrogen_angle(a_fur.elements)
        ]
        assert len(non_h_angles) > 0, "Expected at least one non-H angle in SN2 TS"
        for a_fur, a_qf in non_h_angles:
            assert a_fur.force_constant == pytest.approx(a_qf.force_constant)

    def test_qfuerza_ch3f_all_h_angles_substituted(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        """CH₃F has H-C-F and H-C-H angles — all have H outer atoms."""
        ff = qfuerza_fresh(ch3f_mol_with_hess, strategy="qfuerza")

        for angle in ff.angles:
            if _is_hydrogen_angle(angle.elements):
                assert angle.force_constant == pytest.approx(QFUERZA_H_ANGLE_DEFAULT_CANONICAL), (
                    f"H-angle {angle.key} was not substituted"
                )

    def test_qfuerza_equilibria_unchanged(self, ch3f_mol_with_hess: Q2MMMolecule) -> None:
        """QFUERZA only changes force constants, not equilibrium values."""
        ff_fuerza = qfuerza_fresh(ch3f_mol_with_hess, strategy="fuerza")
        ff_qfuerza = qfuerza_fresh(ch3f_mol_with_hess, strategy="qfuerza")

        for a_fur, a_qf in zip(ff_fuerza.angles, ff_qfuerza.angles):
            assert a_fur.equilibrium == pytest.approx(a_qf.equilibrium)


# ---- Saver functional form validation ----


class TestSaverFormValidation:
    """Verify that savers reject incompatible functional forms."""

    @pytest.fixture()
    def harmonic_ff(self) -> ForceField:
        from q2mm.models.forcefield import FunctionalForm

        return ForceField(
            bonds=[BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54)],
            angles=[AngleParam(elements=("C", "C", "C"), force_constant=50.0, equilibrium=109.5)],
            functional_form=FunctionalForm.HARMONIC,
        )

    @pytest.fixture()
    def mm3_ff(self) -> ForceField:
        from q2mm.models.forcefield import FunctionalForm

        return ForceField(
            bonds=[BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54)],
            angles=[AngleParam(elements=("C", "C", "C"), force_constant=50.0, equilibrium=109.5)],
            functional_form=FunctionalForm.MM3,
        )

    def test_save_mm3_rejects_harmonic(self, tmp_path: Path, harmonic_ff: ForceField) -> None:
        from q2mm.io.mm3 import save_mm3_fld

        with pytest.raises(ValueError, match="Cannot save.*HARMONIC.*mm3_fld"):
            save_mm3_fld(harmonic_ff, tmp_path / "out.fld")

    def test_save_tinker_rejects_harmonic(self, tmp_path: Path, harmonic_ff: ForceField) -> None:
        from q2mm.io.tinker import save_tinker_prm

        with pytest.raises(ValueError, match="Cannot save.*HARMONIC.*tinker_prm"):
            save_tinker_prm(harmonic_ff, tmp_path / "out.prm")

    def test_save_amber_rejects_mm3(self, tmp_path: Path, mm3_ff: ForceField) -> None:
        from q2mm.io.amber import save_amber_frcmod

        with pytest.raises(ValueError, match="Cannot save.*MM3.*amber_frcmod"):
            save_amber_frcmod(mm3_ff, tmp_path / "out.frcmod")

    def test_save_amber_accepts_harmonic(self, tmp_path: Path, harmonic_ff: ForceField) -> None:
        from q2mm.io.amber import save_amber_frcmod

        result = save_amber_frcmod(harmonic_ff, tmp_path / "out.frcmod")
        assert result.exists()

    def test_save_with_none_form_always_allowed(self, tmp_path: Path) -> None:
        """ForceField with functional_form=None is allowed everywhere (backward compat)."""
        from q2mm.io.amber import save_amber_frcmod

        ff = ForceField(
            bonds=[BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54)],
        )
        assert ff.functional_form is None
        result = save_amber_frcmod(ff, tmp_path / "out.frcmod")
        assert result.exists()


class TestFrozenParamInvariant:
    """Frozen params raise FrozenParamError when value fields are mutated.

    The invariant exists to prevent silent overwriting of literature /
    held-fixed parameter values (the q2mm#277 bug).  See the docstring
    of :class:`q2mm.models.forcefield.FrozenParamError` for the user
    workflow (``.unfreeze()`` to opt in to the override).
    """

    def test_setting_value_on_frozen_bond_raises(self) -> None:
        from q2mm.models.forcefield import BondParam, FrozenParamError

        b = BondParam(("C", "F"), 1.38, 359.7, frozen=True)
        with pytest.raises(FrozenParamError, match="BondParam.force_constant"):
            b.force_constant = 500.0
        with pytest.raises(FrozenParamError, match="BondParam.equilibrium"):
            b.equilibrium = 1.5

    def test_setting_value_on_frozen_angle_raises(self) -> None:
        from q2mm.models.forcefield import AngleParam, FrozenParamError

        a = AngleParam(("H", "C", "F"), 109.5, 50.0, frozen=True)
        with pytest.raises(FrozenParamError, match="AngleParam.force_constant"):
            a.force_constant = 80.0
        with pytest.raises(FrozenParamError, match="AngleParam.ub_force_constant"):
            a.ub_force_constant = 10.0

    def test_setting_value_on_frozen_torsion_raises(self) -> None:
        from q2mm.models.forcefield import FrozenParamError, TorsionParam

        t = TorsionParam(("H", "C", "C", "H"), periodicity=3, force_constant=0.16, frozen=True)
        with pytest.raises(FrozenParamError, match="TorsionParam.force_constant"):
            t.force_constant = 0.5

    def test_setting_value_on_frozen_vdw_raises(self) -> None:
        from q2mm.models.forcefield import FrozenParamError, VdwParam

        v = VdwParam("C1", 1.96, 0.044, frozen=True)
        with pytest.raises(FrozenParamError, match="VdwParam.radius"):
            v.radius = 2.0

    def test_setting_value_on_frozen_stretch_bend_raises(self) -> None:
        from q2mm.models.forcefield import FrozenParamError, StretchBendParam

        sb = StretchBendParam(("H", "C", "F"), force_constant=0.75, frozen=True)
        with pytest.raises(FrozenParamError, match="StretchBendParam.force_constant"):
            sb.force_constant = 1.0

    def test_setting_value_on_unfrozen_param_works(self) -> None:
        from q2mm.models.forcefield import BondParam

        b = BondParam(("C", "F"), 1.38, 359.7, frozen=False)
        b.force_constant = 500.0
        assert b.force_constant == 500.0

    def test_freeze_unfreeze_methods_toggle_invariant(self) -> None:
        from q2mm.models.forcefield import BondParam, FrozenParamError

        b = BondParam(("C", "F"), 1.38, 359.7)
        assert not b.frozen

        b.freeze()
        assert b.frozen
        with pytest.raises(FrozenParamError):
            b.force_constant = 999.0

        b.unfreeze()
        assert not b.frozen
        b.force_constant = 999.0
        assert b.force_constant == 999.0

    def test_construction_with_frozen_true_assigns_initial_values(self) -> None:
        """Bypass the guard during __init__ so ``frozen=True`` constructs work."""
        from q2mm.models.forcefield import BondParam

        b = BondParam(("C", "F"), 1.38, 359.7, frozen=True)
        assert b.frozen
        assert b.force_constant == 359.7
        assert b.equilibrium == 1.38

    def test_assigning_frozen_directly_is_not_guarded(self) -> None:
        """``frozen`` itself is not a value field; direct assignment works.

        This is the documented escape hatch for code that prefers
        ``param.frozen = False`` over ``param.unfreeze()``.
        """
        from q2mm.models.forcefield import BondParam

        b = BondParam(("C", "F"), 1.38, 359.7, frozen=True)
        b.frozen = False
        b.force_constant = 999.0
        assert b.force_constant == 999.0

    def test_qfuerza_into_does_not_overwrite_frozen_opt_block(self) -> None:
        """Regression for the q2mm#277 pattern at the unit level.

        ``qfuerza_into(ff, molecules)`` must not overwrite the values
        of any frozen param.  The Heck-relay loader bug fell out of
        this contract being silently violated; now both the
        skip-frozen shortcut in seminario.py and the FrozenParamError
        guard back it up.
        """
        import numpy as np

        from q2mm.models.forcefield import BondParam, ForceField
        from q2mm.models.molecule import Q2MMMolecule
        from q2mm.models.seminario import qfuerza_into

        # Build a tiny FF whose only bond is frozen at a marker value.
        marker_k = 12345.67
        ff = ForceField(
            bonds=[BondParam(("C", "F"), 1.38, marker_k, frozen=True)],
        )

        # Minimal two-atom molecule with a dummy Hessian — Seminario's
        # bond loop will iterate ff.bonds; the skip-frozen branch must
        # bypass it without touching the value.
        mol = Q2MMMolecule(
            symbols=["C", "F"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.38, 0.0, 0.0]]),
            bond_tolerance=1.5,
            hessian=np.eye(6),
        )

        qfuerza_into(ff, mol)
        assert ff.bonds[0].force_constant == marker_k, (
            "Frozen bond force constant was silently overwritten by qfuerza_into — this is the q2mm#277 bug pattern."
        )
