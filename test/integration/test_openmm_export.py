"""Integration tests for OpenMM XML export functionality."""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import param_vector, prepare_case

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


HAS_OPENMM = importlib.util.find_spec("openmm") is not None

pytestmark = [
    pytest.mark.openmm,
    pytest.mark.skipif(not HAS_OPENMM, reason="openmm not installed"),
]

from test._shared import SN2_HESSIAN as TS_HESS, SN2_XYZ as TS_XYZ, make_diatomic, make_noble_gas_pair, make_water

from q2mm.io.openmm import load_openmm_system_xml, save_openmm_system_xml, save_openmm_xml
from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, TorsionParam, VdwParam, FunctionalForm
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule

if HAS_OPENMM:
    from openmm import openmm as mm
else:  # pragma: no cover - exercised when OpenMM is not installed
    mm = None


# ---------------------------------------------------------------------------
# Molecule factory wrappers (preserve original call signatures)
# ---------------------------------------------------------------------------


def _diatomic(distance: float = 0.74) -> Molecule:
    return make_diatomic(distance=distance)


def _water(angle_deg: float = 109.5, bond_length: float = 0.96) -> Molecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length, name="water-like")


def _sn2_ts_molecule() -> Molecule:
    molecule = load_xyz(TS_XYZ, charge=-1, bond_tolerance=1.5)
    return molecule.with_hessian(
        np.load(TS_HESS),
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source="test-fixture",
            path=str(TS_HESS),
        ),
    )


# ---------------------------------------------------------------------------
# System XML round-trip tests
# ---------------------------------------------------------------------------


class TestSystemXMLExport:
    """Test OpenMM System XML round-trip serialization via q2mm.io.openmm."""

    def setup_method(self) -> None:
        self.backend = load_backend("openmm")

    def test_export_creates_valid_xml_file(self, tmp_path: Path) -> None:
        molecule = _diatomic()
        ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )

        out = save_openmm_system_xml(prepare_case(self.backend, molecule, ff), tmp_path / "system.xml")

        assert out.exists()
        tree = ET.parse(out)
        assert tree.getroot().tag == "System"

    def test_system_xml_round_trip_preserves_energy(self, tmp_path: Path) -> None:
        molecule = _diatomic(0.84)
        ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )

        original_energy = (
            prepare_case(self.backend, molecule, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )

        xml_path = tmp_path / "system.xml"
        save_openmm_system_xml(prepare_case(self.backend, molecule, ff), xml_path)

        # Deserialize and compute energy with the loaded system
        loaded_system = load_openmm_system_xml(xml_path)
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
            functional_form=FunctionalForm.MM3,
        )

        original_energy = (
            prepare_case(self.backend, molecule, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )

        xml_path = tmp_path / "water_system.xml"
        save_openmm_system_xml(prepare_case(self.backend, molecule, ff), xml_path)

        loaded_system = load_openmm_system_xml(xml_path)
        assert loaded_system.getNumParticles() == 3

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-6)

    def test_system_xml_with_vdw(self, tmp_path: Path) -> None:
        molecule = make_noble_gas_pair(distance=3.5)
        ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)], functional_form=FunctionalForm.MM3)

        original_energy = (
            prepare_case(self.backend, molecule, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )
        xml_path = tmp_path / "vdw_system.xml"
        save_openmm_system_xml(prepare_case(self.backend, molecule, ff), xml_path)

        loaded_system = load_openmm_system_xml(xml_path)

        from openmm import unit

        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        context = mm.Context(loaded_system, integrator)
        context.setPositions(np.asarray(molecule.geometry, dtype=float) * unit.angstrom)
        state = context.getState(getEnergy=True)
        loaded_energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

        assert loaded_energy == pytest.approx(original_energy, abs=1e-8)

    def test_sn2_system_xml_round_trip(self, tmp_path: Path) -> None:
        from q2mm.models.seminario import qfuerza_fresh

        molecule = _sn2_ts_molecule()
        ff = qfuerza_fresh(molecule, functional_form=FunctionalForm.MM3)

        original_energy = (
            prepare_case(self.backend, molecule, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )

        xml_path = tmp_path / "sn2_system.xml"
        save_openmm_system_xml(prepare_case(self.backend, molecule, ff), xml_path)

        loaded_system = load_openmm_system_xml(xml_path)

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
        ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )

        out = save_openmm_xml(ff, tmp_path / "ff.xml")

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
            functional_form=FunctionalForm.MM3,
        )

        out = save_openmm_xml(ff, tmp_path / "ff.xml")

        tree = ET.parse(out)
        root = tree.getroot()

        angle_forces = root.findall("CustomAngleForce")
        assert len(angle_forces) == 1
        angles = angle_forces[0].findall("Angle")
        assert len(angles) == 1

    def test_produces_valid_xml_with_vdw(self, tmp_path: Path) -> None:
        ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)], functional_form=FunctionalForm.MM3)

        out = save_openmm_xml(ff, tmp_path / "ff.xml")

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
            ],
            functional_form=FunctionalForm.MM3,
        )

        out = save_openmm_xml(ff, tmp_path / "ff.xml")

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
            functional_form=FunctionalForm.MM3,
        )

        out = save_openmm_xml(ff, tmp_path / "ff.xml", molecule=molecule)

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
            functional_form=FunctionalForm.MM3,
        )

        out = save_openmm_xml(ff, tmp_path / "ff.xml")
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
        ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )

        out = save_openmm_xml(ff, tmp_path / "direct.xml")

        assert out.exists()
        tree = ET.parse(out)
        assert tree.getroot().tag == "ForceField"

    def test_source_format_updated(self) -> None:
        """Verify that 'openmm_xml' is a valid source_format value."""
        ff = ForceField(source_format="openmm_xml", functional_form=FunctionalForm.MM3)
        assert ff.source_format == "openmm_xml"

    def test_forcefield_xml_loadable_by_openmm_app(self, tmp_path: Path) -> None:
        """Verify exported XML is loadable by openmm.app.ForceField and can create a System."""
        from openmm import app, unit

        molecule = _water()
        ff = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
            vdws=[VdwParam("O", radius=1.52, epsilon=0.21), VdwParam("H", radius=1.20, epsilon=0.02)],
            functional_form=FunctionalForm.MM3,
        )

        xml_path = save_openmm_xml(ff, tmp_path / "loadable.xml", molecule=molecule)

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

    def test_sn2_forcefield_xml_export(self, tmp_path: Path) -> None:
        """Export Seminario-estimated SN2 force field to ForceField XML."""
        from q2mm.models.seminario import qfuerza_fresh

        molecule = _sn2_ts_molecule()
        ff = qfuerza_fresh(molecule, functional_form=FunctionalForm.MM3)

        out = save_openmm_xml(ff, tmp_path / "sn2_ff.xml", molecule=molecule)

        assert out.exists()
        tree = ET.parse(out)
        root = tree.getroot()

        # Should have all sections
        assert root.find("AtomTypes") is not None
        assert root.find("Residues") is not None
        assert root.find("CustomBondForce") is not None
        assert root.find("CustomAngleForce") is not None
