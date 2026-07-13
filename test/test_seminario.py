import unittest
from pathlib import Path
import numpy as np

from q2mm.io.mm3 import _mm3_import_ff
from q2mm.io import GaussLog, JaguarIn, JaguarOut, MacroModel, Mol2
from q2mm.models.hessian import mass_weight_hessian
from q2mm.models.forcefield import BondParam, ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.structure import Atom, HessianUnits, Structure

REPO_ROOT = Path(__file__).resolve().parent.parent
ETHANE_DIR = REPO_ROOT / "examples" / "ethane"
RH_SEMINARIO_DIR = REPO_ROOT / "examples" / "rh-enamide"
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
        self.assertGreater(len(log.structures), 0, "No structures parsed from GS.log")

    def test_gs_log_has_atoms(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertGreater(len(struct.atoms), 0, "No atoms in parsed structure")

    def test_gs_log_has_hessian(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "GS.log"))
        struct = log.structures[0]
        self.assertIsNotNone(struct.hess, "No Hessian parsed from GS.log")

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

        default_structure = default_log.structures[-1]
        atomic_structure = atomic_log.structures[-1]
        self.assertEqual(default_structure.hessian_units, HessianUnits.KJ_MOL_ANGSTROM2)
        self.assertEqual(atomic_structure.hessian_units, HessianUnits.ATOMIC)

        direct_default = Q2MMMolecule.from_structure(default_structure)
        direct_atomic = Q2MMMolecule.from_structure(atomic_structure)
        np.testing.assert_allclose(direct_default.hessian, direct_atomic.hessian, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            default_log.molecules[-1].hessian, atomic_log.molecules[-1].hessian, rtol=1e-12, atol=1e-12
        )

    def test_parse_ts_log(self) -> None:
        log = GaussLog(str(ETHANE_DIR / "TS.log"))
        self.assertGreater(len(log.structures), 0, "No structures parsed from TS.log")

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
        self.assertGreater(len(mol2.structures), 0, "No structures parsed from mol2")

    def test_mol2_atom_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: C2H6 = 8 atoms
        self.assertEqual(len(struct.atoms), 8, "Ethane should have 8 atoms")

    def test_mol2_bond_count(self) -> None:
        mol2 = Mol2(str(ETHANE_DIR / "GS.mol2"))
        struct = mol2.structures[0]
        # Ethane: 7 bonds (1 C-C + 6 C-H)
        self.assertEqual(len(struct.bonds), 7, "Ethane should have 7 bonds")

    def test_mol2_molecule_preserves_bonds_and_infers_higher_topology(self) -> None:
        molecule = Mol2(str(ETHANE_DIR / "GS.mol2")).molecules[0]

        self.assertEqual(len(molecule.bonds), 7)
        self.assertEqual(len(molecule.angles), 12)
        self.assertEqual(len(molecule.torsions), 9)
        self.assertEqual(molecule.bonds[0].bond_order, "-")
        self.assertEqual(molecule.bonds[0].source_bond_order, "1")
        self.assertEqual(molecule.partial_charges[0], 0.0227)
        self.assertIsNone(molecule.hessian)

    def test_mol2_bond_orders_use_canonical_force_field_symbols(self) -> None:
        parser = Mol2(str(ETHANE_DIR / "GS.mol2"))
        for source_order, canonical_order in (("2", "="), ("ar", "*"), ("3", "%")):
            with self.subTest(source_order=source_order):
                structure = Structure("bond-order")
                structure.atoms.extend(
                    [
                        Atom(element="C", atom_type_name="C.2", coords=[0.0, 0.0, 0.0]),
                        Atom(element="C", atom_type_name="C.2", coords=[1.3, 0.0, 0.0]),
                    ]
                )
                structure.bonds = parser.parse_bonds([f"1 1 2 {source_order}"], structure)
                molecule = Q2MMMolecule.from_structure(structure)
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
                    ]
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
        parser = Mol2(str(ETHANE_DIR / "GS.mol2"))
        structure = Structure("unknown-order")
        structure.atoms.extend(
            [
                Atom(element="C", atom_type_name="C.2", coords=[0.0, 0.0, 0.0]),
                Atom(element="N", atom_type_name="N.am", coords=[1.3, 0.0, 0.0]),
            ]
        )
        structure.bonds = parser.parse_bonds(["1 1 2 am"], structure)

        bond = Q2MMMolecule.from_structure(structure).bonds[0]

        self.assertEqual(bond.bond_order, "")
        self.assertEqual(bond.source_bond_order, "am")


@unittest.skipUnless(RH_JAGUAR_OUT.exists(), "Jaguar fixture not found")
class TestJaguarConversion(unittest.TestCase):
    def test_jaguar_molecule_infers_missing_topology_only(self) -> None:
        parser = JaguarOut(str(RH_JAGUAR_OUT))
        structure = parser.structures[-1]
        molecule = parser.molecules[-1]

        self.assertFalse(structure.has_explicit_bonds)
        self.assertGreater(len(molecule.bonds), 0)
        self.assertGreater(len(molecule.angles), 0)
        self.assertIsNone(molecule.hessian)
        self.assertIsNone(molecule.partial_charges)

    def test_jaguar_hessian_override_is_preserved(self) -> None:
        structure = MacroModel(str(RH_MMO)).structures[0]
        hessian = JaguarIn(str(RH_JAGUAR_IN)).get_hessian(len(structure.atoms))

        from q2mm.models.molecule import Q2MMMolecule

        molecule = Q2MMMolecule.from_structure(structure, hessian=hessian)

        np.testing.assert_array_equal(molecule.hessian, hessian)


@unittest.skipUnless(RH_MMO.exists(), "MacroModel fixture not found")
class TestMacroModelConversion(unittest.TestCase):
    def test_macromodel_molecule_preserves_all_supplied_topology(self) -> None:
        parser = MacroModel(str(RH_MMO))
        structure = parser.structures[0]
        molecule = parser.molecules[0]

        self.assertEqual(len(molecule.bonds), len(structure.bonds))
        self.assertEqual(len(molecule.angles), len(structure.angles))
        self.assertEqual(len(molecule.torsions), len(structure.torsions))
        self.assertEqual(molecule.bonds[0].ff_row, structure.bonds[0].ff_row)
        self.assertEqual(molecule.angles[0].ff_row, structure.angles[0].ff_row)
        self.assertEqual(molecule.torsions[0].ff_row, structure.torsions[0].ff_row)
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
        struct = mol2.structures[0]
        symbols = [a.element for a in struct.atoms]
        hess = log.structures[0].hess.copy()
        original = hess.copy()
        # Mass-weight then un-weight should give back original
        mass_weight_hessian(hess, symbols)
        mass_weight_hessian(hess, symbols, reverse=True)
        np.testing.assert_allclose(original, hess, rtol=1e-10, err_msg="Mass-weight roundtrip did not preserve Hessian")


if __name__ == "__main__":
    unittest.main()
