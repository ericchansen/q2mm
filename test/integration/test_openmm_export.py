"""Integration tests for OpenMM XML export functionality."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import SN2_XYZ as TS_XYZ, SN2_HESSIAN as TS_HESS, make_diatomic, make_water

from openmm import openmm as mm

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, TorsionParam, VdwParam
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.ff_io import save_openmm_xml


# ---------------------------------------------------------------------------
# Molecule factory wrappers (preserve original call signatures)
# ---------------------------------------------------------------------------


def _diatomic(distance: float = 0.74) -> Q2MMMolecule:
    return make_diatomic(distance=distance)


def _water(angle_deg: float = 109.5, bond_length: float = 0.96) -> Q2MMMolecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length, name="water-like")


# ---------------------------------------------------------------------------
# System XML round-trip tests
# ---------------------------------------------------------------------------


class TestSystemXMLExport:
    """Test OpenMMEngine.export_system_xml() round-trip serialization."""

    def setup_method(self) -> None:
        self.engine = OpenMMEngine()

    def test_export_creates_valid_xml_file(self, tmp_path: Path) -> None:
        molecule = _diatomic()
        ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        out = self.engine.export_system_xml(tmp_path / "system.xml", molecule, ff)

        assert out.exists()
        tree = ET.parse(out)
        assert tree.getroot().tag == "System"

    def test_system_xml_round_trip_preserves_energy(self, tmp_path: Path) -> None:
        molecule = _diatomic(0.84)
        ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        original_energy = self.engine.energy(molecule, ff)

        xml_path = tmp_path / "system.xml"
        self.engine.export_system_xml(xml_path, molecule, ff)

        # Deserialize and compute energy with the loaded system
        loaded_system = OpenMMEngine.load_system_xml(xml_path)
        assert isinstance(loaded_system, mm.System)
        assert loaded_system.getNumParticles() == 2

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-8)

    def test_system_xml_round_trip_water_with_angles(self, tmp_path: Path) -> None:
        molecule = _water(angle_deg=120.0)
        ff = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        original_energy = self.engine.energy(molecule, ff)

        xml_path = tmp_path / "water_system.xml"
        self.engine.export_system_xml(xml_path, molecule, ff)

        loaded_system = OpenMMEngine.load_system_xml(xml_path)
        assert loaded_system.getNumParticles() == 3

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-6)

    def test_system_xml_with_vdw(self, tmp_path: Path) -> None:
        molecule = Q2MMMolecule(
            symbols=["He", "He"],
            atom_types=["He", "He"],
            geometry=np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]]),
            name="He2",
            bond_tolerance=0.5,
        )
        ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)])

        original_energy = self.engine.energy(molecule, ff)
        xml_path = tmp_path / "vdw_system.xml"
        self.engine.export_system_xml(xml_path, molecule, ff)

        loaded_system = OpenMMEngine.load_system_xml(xml_path)

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-8)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_system_xml_round_trip(self, tmp_path: Path) -> None:
        from q2mm.models.seminario import estimate_force_constants

        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        ff = estimate_force_constants(molecule)

        original_energy = self.engine.energy(molecule, ff)

        xml_path = tmp_path / "sn2_system.xml"
        self.engine.export_system_xml(xml_path, molecule, ff)

        loaded_system = OpenMMEngine.load_system_xml(xml_path)

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-4)


# ---------------------------------------------------------------------------
# ForceField XML export tests
# ---------------------------------------------------------------------------


class TestForceFieldXMLExport:
    """Test ForceField.to_openmm_xml() standalone XML generation."""

    def test_produces_valid_xml_with_bonds(self, tmp_path: Path) -> None:
        ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        out = ff.to_openmm_xml(tmp_path / "ff.xml")

        assert out.exists()
        tree = ET.parse(out)
        root = tree.getroot()
        assert root.tag == "ForceField"

        # Should have CustomBondForce
        bond_forces = root.findall("CustomBondForce")
        assert len(bond_forces) == 1
        bonds = bond_forces[0].findall("Bond")
        assert len(bonds) == 1

    def test_produces_valid_xml_with_angles(self, tmp_path: Path) -> None:
        ff = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        out = ff.to_openmm_xml(tmp_path / "ff.xml")

        tree = ET.parse(out)
        root = tree.getroot()

        angle_forces = root.findall("CustomAngleForce")
        assert len(angle_forces) == 1
        angles = angle_forces[0].findall("Angle")
        assert len(angles) == 1

    def test_produces_valid_xml_with_vdw(self, tmp_path: Path) -> None:
        ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)])

        out = ff.to_openmm_xml(tmp_path / "ff.xml")

        tree = ET.parse(out)
        root = tree.getroot()

        nb_forces = root.findall("CustomNonbondedForce")
        assert len(nb_forces) == 1
        atoms = nb_forces[0].findall("Atom")
        assert len(atoms) == 1

    def test_produces_valid_xml_with_torsions(self, tmp_path: Path) -> None:
        ff = ForceField(
            torsions=[
                TorsionParam(("H", "C", "C", "H"), periodicity=1, force_constant=0.5, phase=0.0),
                TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=0.3, phase=180.0),
            ]
        )

        out = ff.to_openmm_xml(tmp_path / "ff.xml")

        tree = ET.parse(out)
        root = tree.getroot()
        torsion_forces = root.findall("CustomTorsionForce")
        assert len(torsion_forces) == 1
        torsions = torsion_forces[0].findall("Torsion")
        assert len(torsions) == 2

    def test_with_molecule_generates_atom_types_and_residues(self, tmp_path: Path) -> None:
        molecule = _water()
        ff = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        out = ff.to_openmm_xml(tmp_path / "ff.xml", molecule=molecule)

        tree = ET.parse(out)
        root = tree.getroot()

        # Should have AtomTypes and Residues
        atom_types = root.findall("AtomTypes")
        assert len(atom_types) == 1
        types = atom_types[0].findall("Type")
        assert len(types) >= 2  # O and H

        residues = root.findall("Residues")
        assert len(residues) == 1
        residue = residues[0].findall("Residue")
        assert len(residue) == 1
        atoms = residue[0].findall("Atom")
        assert len(atoms) == 3  # O, H, H

    def test_unit_conversions_are_correct(self, tmp_path: Path) -> None:
        """Verify that exported parameters use correct OpenMM units."""
        ff = ForceField(
            bonds=[BondParam(("C", "F"), equilibrium=1.38, force_constant=359.7)],
            angles=[AngleParam(("F", "C", "F"), equilibrium=108.0, force_constant=86.3)],
            vdws=[VdwParam("C", radius=1.94, epsilon=0.027), VdwParam("F", radius=1.47, epsilon=0.075)],
        )

        out = ff.to_openmm_xml(tmp_path / "ff.xml")
        tree = ET.parse(out)
        root = tree.getroot()

        # Bond: r0 should be in nm (1.38 Å = 0.138 nm)
        bond_el = root.find(".//CustomBondForce/Bond")
        assert float(bond_el.get("r0")) == pytest.approx(0.138, abs=1e-4)

        # Angle: theta0 should be in radians
        import math

        angle_el = root.find(".//CustomAngleForce/Angle")
        assert float(angle_el.get("theta0")) == pytest.approx(math.radians(108.0), abs=1e-4)

        # vdW: radius in nm, epsilon in kJ/mol
        vdw_atoms = root.findall(".//CustomNonbondedForce/Atom")
        for atom in vdw_atoms:
            r = float(atom.get("radius"))
            assert r < 0.3  # nm, not Å

    def test_save_openmm_xml_function_directly(self, tmp_path: Path) -> None:
        """Test the standalone save_openmm_xml function."""
        ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        out = save_openmm_xml(ff, tmp_path / "direct.xml")

        assert out.exists()
        tree = ET.parse(out)
        assert tree.getroot().tag == "ForceField"

    def test_source_format_updated(self) -> None:
        """Verify that 'openmm_xml' is a valid source_format value."""
        ff = ForceField(source_format="openmm_xml")
        assert ff.source_format == "openmm_xml"

    def test_forcefield_xml_loadable_by_openmm_app(self, tmp_path: Path) -> None:
        """Verify exported XML is loadable by openmm.app.ForceField and can create a System."""
        from openmm import app, unit

        molecule = _water()
        ff = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
            vdws=[VdwParam("O", radius=1.52, epsilon=0.21), VdwParam("H", radius=1.20, epsilon=0.02)],
        )

        xml_path = ff.to_openmm_xml(tmp_path / "loadable.xml", molecule=molecule)

        # Load via openmm.app.ForceField
        omm_ff = app.ForceField(str(xml_path))

        # Build topology
        topology = app.Topology()
        chain = topology.addChain()
        residue = topology.addResidue("Q2MM", chain)
        atoms = []
        for i, symbol in enumerate(molecule.symbols):
            elem = app.Element.getBySymbol(symbol)
            atoms.append(topology.addAtom(f"{symbol}{i + 1}", elem, residue))
        for bond in molecule.bonds:
            topology.addBond(atoms[bond.atom_i], atoms[bond.atom_j])

        # createSystem should succeed
        system = omm_ff.createSystem(topology, nonbondedMethod=app.NoCutoff)
        assert system.getNumParticles() == 3

        # Compute energy and verify it's finite
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        assert np.isfinite(energy)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_forcefield_xml_export(self, tmp_path: Path) -> None:
        """Export Seminario-estimated SN2 force field to ForceField XML."""
        from q2mm.models.seminario import estimate_force_constants

        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        ff = estimate_force_constants(molecule)

        out = ff.to_openmm_xml(tmp_path / "sn2_ff.xml", molecule=molecule)

        assert out.exists()
        tree = ET.parse(out)
        root = tree.getroot()

        # Should have all sections
        assert root.find("AtomTypes") is not None
        assert root.find("Residues") is not None
        assert root.find("CustomBondForce") is not None
        assert root.find("CustomAngleForce") is not None
