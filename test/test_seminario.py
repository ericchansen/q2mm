import tempfile
import unittest
from pathlib import Path
import numpy as np

from q2mm.io.mm3 import _mm3_import_ff
from q2mm.io import GaussLog, JaguarIn, JaguarOut, MacroModel, Mol2
from q2mm.models.hessian import HessianUnits, mass_weight_hessian
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule

REPO_ROOT = Path(__file__).resolve().parent.parent
ETHANE_DIR = REPO_ROOT / "test" / "fixtures" / "ethane"
RH_SEMINARIO_DIR = REPO_ROOT / "examples" / "publication" / "rh-enamide"
RH_TRAINING_DIR = RH_SEMINARIO_DIR / "rh_enamide_training_set"
RH_MMO = RH_TRAINING_DIR / "rh_enamide_training_set.mmo"
RH_JAGUAR_IN = RH_TRAINING_DIR / "jaguar_spe_freq_in_out" / "1ZDMPfromJCTCSI_loner1.01.in"
RH_JAGUAR_OUT = RH_TRAINING_DIR / "jaguar_spe_freq_in_out" / "1ZDMPfromJCTCSI_loner1.out"


@unittest.skipUnless(
    (ETHANE_DIR / "GS.mol2").exists() and (ETHANE_DIR / "GS.log").exists() and (ETHANE_DIR / "TS.log").exists(),
    "Ethane fixture files not found",
)
class TestGaussLogParsing(unittest.TestCase):
    """Test that GaussLog can parse ethane Gaussian output."""

    def test_parse_gs_log(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        self.assertGreater(len(log.molecules), 0, "No molecules parsed from GS.log")

    def test_gs_log_has_atoms(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        molecule = log.molecules[0]
        self.assertGreater(molecule.n_atoms, 0, "No atoms in parsed molecule")

    def test_gs_log_has_hessian(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        molecule = log.molecules[0]
        self.assertIsNotNone(molecule.hessian, "No Hessian parsed from GS.log")

    def test_gaussian_molecule_preserves_archive_fields_and_infers_topology(self) -> None:
        molecule = GaussLog(str(ETHANE_DIR / "GS.log"), au_hessian=True).molecules[-1]

        self.assertEqual(molecule.charge, 0)
        self.assertEqual(molecule.multiplicity, 1)
        self.assertIsNotNone(molecule.hessian)
        self.assertEqual(len(molecule.bonds), 7)
        self.assertEqual(len(molecule.angles), 12)
        self.assertIsNone(molecule.partial_charges)

    def test_gaussian_hessian_units_normalize_at_domain_boundary(self) -> None:
        default_log = GaussLog(str(ETHANE_DIR / "GS.log"))
        atomic_log = GaussLog(str(ETHANE_DIR / "GS.log"), au_hessian=True)

        default_molecule = default_log.molecules[-1]
        atomic_molecule = atomic_log.molecules[-1]
        self.assertEqual(default_molecule.hessian_provenance.units, HessianUnits.KJ_MOL_ANGSTROM2)
        self.assertEqual(atomic_molecule.hessian_provenance.units, HessianUnits.ATOMIC)

        # Both are converted (once) to canonical Hartree/Bohr² regardless of
        # the source archive's unit provenance.
        np.testing.assert_allclose(default_molecule.hessian, atomic_molecule.hessian, rtol=1e-12, atol=1e-12)

    def test_parse_ts_log(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "TS.log"))
        self.assertGreater(len(log.molecules), 0, "No molecules parsed from TS.log")

    def test_archive_hessian_reproduces_gaussian_frequencies(self) -> None:
        """The archive Cartesian Hessian must reproduce Gaussian's frequencies.

        Regression guard for the mass-weighting ingestion bug: both loaders
        used to override ``mol.hessian`` with ``reform_hessian(evals, evecs)``
        reconstructed from Gaussian's *mass-weighted* frequency analysis,
        corrupting every heavy-atom force constant by ~√(mᵢmⱼ).  The archive
        Hessian (``au_hessian=True``) is plain Cartesian (Hartree/Bohr²) and
        must round-trip to the Gaussian-reported vibrational frequencies to
        well under 1 cm⁻¹.
        """
        from q2mm.models.hessian import hessian_to_frequencies

        log = GaussLog(str(ETHANE_DIR / "GS.log"), au_hessian=True)
        mol = log.molecules[-1]
        self.assertIsNotNone(mol.hessian, "Archive Hessian not attached to molecule")

        reported = np.sort(np.asarray(log.frequencies, dtype=float))
        all_freqs = np.asarray(hessian_to_frequencies(mol.hessian, list(mol.symbols), sort=True))
        # Drop the 6 lowest-magnitude (rigid-body) modes before comparing.
        vibrational = np.sort(all_freqs[np.argsort(np.abs(all_freqs))[6:]])
        n = min(len(reported), len(vibrational))
        max_dev = float(np.abs(vibrational[:n] - reported[:n]).max())
        self.assertLess(
            max_dev,
            1.0,
            f"Archive Hessian frequencies deviate {max_dev:.3f} cm⁻¹ from "
            "Gaussian's reported values — mass-weighting override may have "
            "returned.",
        )


@unittest.skipUnless((ETHANE_DIR / "GS.mol2").exists(), "Ethane mol2 fixture not found")
class TestMol2Parsing(unittest.TestCase):
    """Test that Mol2 can parse ethane structure."""

    def test_parse_mol2(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        self.assertGreater(len(mol2.molecules), 0, "No molecules parsed from mol2")

    def test_mol2_atom_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        molecule = mol2.molecules[0]
        # Ethane: C2H6 = 8 atoms
        self.assertEqual(molecule.n_atoms, 8, "Ethane should have 8 atoms")

    def test_mol2_bond_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        molecule = mol2.molecules[0]
        # Ethane: 7 bonds (1 C-C + 6 C-H)
        self.assertEqual(len(molecule.bonds), 7, "Ethane should have 7 bonds")
        self.assertTrue(molecule.bonds_explicit)

    def test_mol2_molecule_preserves_bonds_and_infers_higher_topology(self) -> None:
        molecule = Mol2(str(ETHANE_DIR / "GS.mol2")).molecules[0]

        self.assertEqual(len(molecule.bonds), 7)
        self.assertEqual(len(molecule.angles), 12)
        self.assertEqual(len(molecule.torsions), 9)
        self.assertEqual(molecule.bonds[0].bond_order, "-")
        self.assertEqual(molecule.bonds[0].source_bond_order, "1")
        self.assertEqual(molecule.partial_charges[0], 0.0227)
        self.assertIsNone(molecule.hessian)

    _MOL2_TWO_ATOM_TEMPLATE = """@<TRIPOS>MOLECULE
bond-order
2 1 0 0 0
SMALL
NO_CHARGES

@<TRIPOS>ATOM
      1 {name1:<9}0.0000    0.0000    0.0000 {type1:<9}1  <1>        0.0000
      2 {name2:<9}1.3000    0.0000    0.0000 {type2:<9}1  <1>        0.0000
@<TRIPOS>BOND
     1    1    2 {order}
"""

    def _molecule_from_two_atom_mol2(
        self, order: str, type1: str = "C.2", type2: str = "C.2", name1: str = "C1", name2: str = "C2"
    ) -> Molecule:
        """Parse a minimal two-atom/one-bond mol2 file through the real ``Mol2`` parser.

        Exercises bond-order canonicalization through the actual public
        ``Mol2`` API (no private staging records touched by the test).
        """
        text = self._MOL2_TWO_ATOM_TEMPLATE.format(name1=name1, type1=type1, name2=name2, type2=type2, order=order)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "bond_order.mol2"
            path.write_text(text, encoding="utf-8")
            return Mol2(str(path)).molecules[0]

    def test_mol2_bond_orders_use_canonical_force_field_symbols(self) -> None:
        for source_order, canonical_order in (("2", "="), ("ar", "*"), ("3", "%")):
            with self.subTest(source_order=source_order):
                molecule = self._molecule_from_two_atom_mol2(source_order)
                bond = molecule.bonds[0]

                self.assertEqual(bond.bond_order, canonical_order)
                self.assertEqual(bond.source_bond_order, source_order)

                forcefield = ForceField(
                    bonds=[
                        BondParam(("C", "C"), 1.54, 300.0, env_id=bond.env_id, bond_order="-", label="single"),
                        BondParam(
                            ("C", "C"),
                            1.30,
                            500.0,
                            env_id=bond.env_id,
                            bond_order=canonical_order,
                            label=source_order,
                        ),
                    ],
                    functional_form=FunctionalForm.HARMONIC,
                )
                matched = forcefield.match_bond(
                    bond.elements,
                    env_id=bond.env_id,
                    bond_order=bond.bond_order,
                    bond_length=bond.length,
                )
                self.assertIsNotNone(matched)
                self.assertEqual(matched.label, source_order)

    def test_unknown_mol2_bond_order_is_not_canonicalized(self) -> None:
        molecule = self._molecule_from_two_atom_mol2("am", type2="N.am", name2="N1")
        bond = molecule.bonds[0]

        self.assertEqual(bond.bond_order, "")
        self.assertEqual(bond.source_bond_order, "am")

    def test_mol2_element_comes_from_type_not_atom_name(self) -> None:
        for atom_type, element in (
            ("C.3", "C"),
            ("C.ar", "C"),
            ("C.cat", "C"),
            ("N.am", "N"),
            ("N.pl3", "N"),
            ("O.co2", "O"),
            ("S.o2", "S"),
            ("Cl", "Cl"),
            ("Br", "Br"),
            ("Ca", "Ca"),
            ("Co", "Co"),
            ("Si", "Si"),
            ("Rh", "Rh"),
            ("C1", "C"),
            ("C2", "C"),
            ("C3", "C"),
            ("H1", "H"),
            ("H3", "H"),
            ("N1", "N"),
            ("N2", "N"),
            ("O2", "O"),
            ("P3", "P"),
            ("HC", "H"),
            ("HX", "H"),
            ("PX", "P"),
            ("RH", "Rh"),
        ):
            with self.subTest(atom_type=atom_type):
                molecule = self._molecule_from_two_atom_mol2("1", type1=atom_type, name1="CA")
                self.assertEqual(molecule.symbols, (element, "C"))
                self.assertEqual(molecule.atom_types, (atom_type, "C.2"))
                self.assertEqual(molecule.bonds[0].elements, (element, "C"))
                self.assertEqual(molecule.partial_charges, (0.0, 0.0))

    def test_mol2_unsupported_atom_types_fail_without_name_fallback(self) -> None:
        for atom_type in ("Du", "Du.C", "X", "LP", "Any", "Hal", "Het", "Unknown", "ca", "na", "os", "CA", "CO", "123"):
            with self.subTest(atom_type=atom_type), self.assertRaisesRegex(ValueError, "Unsupported Mol2 atom type"):
                self._molecule_from_two_atom_mol2("1", type1=atom_type, name1="C1")

    def test_mol2_structure_names_are_stable_and_distinct(self) -> None:
        chunks = [
            self._MOL2_TWO_ATOM_TEMPLATE.format(name1="CA", type1=atom_type, name2="H1", type2="H", order="1")
            for atom_type in ("C.3", "N.3", "O.3")
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "multi.mol2"
            path.write_text("# Multiple structures with the same header name\n" + "".join(chunks), encoding="utf-8")
            parser = Mol2(str(path))
            molecules = parser.molecules
            expected_names = ["multi.mol20", "multi.mol21", "multi.mol22"]
            self.assertEqual([m.name for m in molecules], expected_names)
            self.assertEqual([m.symbols[0] for m in molecules], ["C", "N", "O"])
            self.assertEqual([m.name for m in parser.molecules], expected_names)
            self.assertEqual([m.name for m in Mol2(str(path)).molecules], expected_names)
            renamed = molecules[1].with_overrides(name="explicit-case").with_hessian(np.eye(6))
            self.assertEqual(renamed.name, "explicit-case")
            self.assertEqual(molecules[1].name, "multi.mol21")

            path.write_text(chunks[0], encoding="utf-8")
            self.assertEqual(Mol2(str(path)).molecules[0].name, "multi.mol2")

    def test_mol2_invalid_input_does_not_cache_partial_structures(self) -> None:
        valid = self._MOL2_TWO_ATOM_TEMPLATE.format(name1="C1", type1="C.3", name2="H1", type2="H", order="1")
        invalid = valid.replace("C.3", "Du ")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "invalid.mol2"
            for text, error in (
                (valid + invalid, "Unsupported Mol2 atom type"),
                ("# No molecule records\n", "No @<TRIPOS>MOLECULE"),
            ):
                path.write_text(text, encoding="utf-8")
                parser = Mol2(str(path))
                for _ in range(2):
                    with self.assertRaisesRegex(ValueError, error):
                        _ = parser.molecules


@unittest.skipUnless(RH_JAGUAR_OUT.exists(), "Jaguar fixture not found")
class TestJaguarConversion(unittest.TestCase):
    def test_jaguar_molecule_infers_missing_topology_only(self) -> None:
        parser = JaguarOut(str(RH_JAGUAR_OUT))
        molecule = parser.molecules[-1]

        self.assertFalse(molecule.bonds_explicit)
        self.assertGreater(len(molecule.bonds), 0)
        self.assertGreater(len(molecule.angles), 0)
        self.assertIsNone(molecule.hessian)
        self.assertIsNone(molecule.partial_charges)

    def test_jaguar_hessian_override_is_preserved(self) -> None:
        base_molecule = MacroModel(str(RH_MMO)).molecules[0]
        jaguar = JaguarIn(str(RH_JAGUAR_IN))
        hessian = jaguar.get_hessian(base_molecule.n_atoms)

        molecule = jaguar.attach_hessian(base_molecule)

        np.testing.assert_array_equal(molecule.hessian, hessian)
        self.assertEqual(molecule.hessian_provenance.source, "jaguar")
        self.assertEqual(molecule.hessian_provenance.path, str(RH_JAGUAR_IN.resolve()))


@unittest.skipUnless(RH_MMO.exists(), "MacroModel fixture not found")
class TestMacroModelConversion(unittest.TestCase):
    def test_macromodel_molecule_preserves_all_supplied_topology(self) -> None:
        """MacroModel always supplies explicit bonds/angles/torsions.

        Counts and the first ``ff_row`` of each are pinned to the known
        content of the rh-enamide training-set fixture (via the public
        ``molecules`` output only — no private staging is inspected).
        """
        molecule = MacroModel(str(RH_MMO)).molecules[0]

        self.assertTrue(molecule.bonds_explicit)
        self.assertTrue(molecule.angles_explicit)
        self.assertTrue(molecule.torsions_explicit)
        self.assertEqual(len(molecule.bonds), 38)
        self.assertEqual(len(molecule.angles), 78)
        self.assertEqual(len(molecule.torsions), 62)
        self.assertEqual(molecule.bonds[0].ff_row, 1865)
        self.assertEqual(molecule.angles[0].ff_row, 1881)
        self.assertEqual(molecule.torsions[0].ff_row, 950)
        self.assertEqual(molecule.bonds[0].bond_order, "")
        self.assertIsNone(molecule.partial_charges)


@unittest.skipUnless((RH_SEMINARIO_DIR / "mm3.fld").exists(), "rh-enamide fixture not found")
class TestMM3FFParsing(unittest.TestCase):
    """Test MM3 force field parsing via q2mm.io.mm3."""

    def setUp(self) -> None:
        self.params, _ = _mm3_import_ff(str(RH_SEMINARIO_DIR / "mm3.fld"))

    def test_parse_mm3(self) -> None:
        self.assertGreater(len(self.params), 0, "No parameters parsed")

    def test_mm3_has_bonds(self) -> None:
        bond_params = [p for p in self.params if p.ptype in ("bf", "be")]
        self.assertGreater(len(bond_params), 0, "No bond parameters found")

    def test_mm3_has_angles(self) -> None:
        angle_params = [p for p in self.params if p.ptype in ("af", "ae")]
        self.assertGreater(len(angle_params), 0, "No angle parameters found")


@unittest.skipUnless(
    (ETHANE_DIR / "GS.log").exists() and (ETHANE_DIR / "GS.mol2").exists(), "Ethane fixture files not found"
)
class TestHessianMassWeighting(unittest.TestCase):
    """Test mass-weighting of Hessians."""

    def test_mass_weight_roundtrip(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        symbols = list(mol2.molecules[0].symbols)
        hess = log.molecules[0].hessian.copy()
        original = hess.copy()
        # Mass-weight then un-weight should give back original
        mass_weight_hessian(hess, symbols)
        mass_weight_hessian(hess, symbols, reverse=True)
        np.testing.assert_allclose(original, hess, rtol=1e-10, err_msg="Mass-weight roundtrip did not preserve Hessian")


if __name__ == "__main__":
    unittest.main()
