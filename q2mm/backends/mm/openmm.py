"""OpenMM molecular mechanics backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.constants import (
    AMU_TO_KG,
    SPEED_OF_LIGHT_MS,
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    MM3_ANGLE_C6,
    RAD_TO_DEG,
    KCAL_TO_KJ,
    MDYNA_TO_KJMOLA2,
    MM3_STR,
    AVO,
    MASSES,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam
from q2mm.models.molecule import Q2MMMolecule

try:
    from openmm import openmm as mm
    from openmm import unit

    _HAS_OPENMM = True
except ImportError:  # pragma: no cover - exercised when OpenMM is not installed
    mm = None
    unit = None
    _HAS_OPENMM = False


@dataclass
class _BondTerm:
    force_index: int
    atom_i: int
    atom_j: int
    elements: tuple[str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _AngleTerm:
    force_index: int
    atom_i: int
    atom_j: int
    atom_k: int
    elements: tuple[str, str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _VdwTerm:
    particle_index: int
    atom_type: str = ""
    element: str = ""
    ff_row: int | None = None


@dataclass
class OpenMMHandle:
    """Reusable OpenMM system/context pair for fast parameter updates."""

    molecule: Q2MMMolecule
    system: object
    integrator: object
    context: object
    bond_force: object | None
    angle_force: object | None
    vdw_force: object | None
    bond_terms: list[_BondTerm]
    angle_terms: list[_AngleTerm]
    vdw_terms: list[_VdwTerm]


def _ensure_openmm():
    if not _HAS_OPENMM:
        raise ImportError('OpenMM is not installed. Install with `pip install openmm` or `pip install -e ".[openmm]"`.')


def _as_molecule(structure) -> Q2MMMolecule:
    if isinstance(structure, Q2MMMolecule):
        return structure
    if isinstance(structure, (str, Path)):
        return Q2MMMolecule.from_xyz(structure)
    raise TypeError("OpenMMEngine expects a Q2MMMolecule or path to an XYZ file.")


def _coerce_forcefield(forcefield: ForceField | None, molecule: Q2MMMolecule) -> ForceField:
    if forcefield is not None:
        return forcefield
    return ForceField.create_for_molecule(molecule)


def _bond_k_to_openmm(force_constant: float) -> float:
    """Convert MM3 bond constants to Tinker-scaled kJ/mol/A^2."""
    return 0.5 * float(force_constant) * MDYNA_TO_KJMOLA2


def _angle_k_to_openmm(force_constant: float) -> float:
    """Convert MM3 angle constants to Tinker-scaled kJ/mol/rad^2."""
    return 0.5 * float(force_constant) * MM3_STR


def _vdw_radius_to_openmm(radius: float) -> float:
    return float(radius) * 0.1


def _vdw_epsilon_to_openmm(epsilon: float) -> float:
    return float(epsilon) * KCAL_TO_KJ


def _match_bond(
    forcefield: ForceField, elements: tuple[str, str], env_id: str = "", ff_row: int | None = None
) -> BondParam | None:
    for bond in forcefield.bonds:
        if ff_row is not None and bond.ff_row == ff_row:
            return bond
    if env_id:
        matched = forcefield.get_bond(elements[0], elements[1], env_id=env_id)
        if matched is not None:
            return matched
    return forcefield.get_bond(elements[0], elements[1])


def _match_angle(
    forcefield: ForceField, elements: tuple[str, str, str], env_id: str = "", ff_row: int | None = None
) -> AngleParam | None:
    for angle in forcefield.angles:
        if ff_row is not None and angle.ff_row == ff_row:
            return angle
    if env_id:
        matched = forcefield.get_angle(elements[0], elements[1], elements[2], env_id=env_id)
        if matched is not None:
            return matched
    return forcefield.get_angle(elements[0], elements[1], elements[2])


def _match_vdw(
    forcefield: ForceField, atom_type: str = "", element: str = "", ff_row: int | None = None
) -> VdwParam | None:
    for vdw in forcefield.vdws:
        if ff_row is not None and vdw.ff_row == ff_row:
            return vdw
    matched = forcefield.get_vdw(atom_type=atom_type, element=element)
    if matched is not None:
        return matched
    if element:
        return forcefield.get_vdw(element=element)
    return None


class OpenMMEngine(MMEngine):
    """Molecular mechanics backend powered by OpenMM."""

    def __init__(self, platform_name: str | None = None):
        _ensure_openmm()
        self._platform_name = platform_name

    @property
    def name(self) -> str:
        return "OpenMM (MM3 bonded + vdW terms)"

    def is_available(self) -> bool:
        return _HAS_OPENMM

    def supports_runtime_params(self) -> bool:
        return True

    def _positions(self, molecule: Q2MMMolecule):
        return np.asarray(molecule.geometry, dtype=float) * unit.angstrom

    def _create_context(self, system):
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        if self._platform_name:
            platform = mm.Platform.getPlatformByName(self._platform_name)
            context = mm.Context(system, integrator, platform)
        else:
            context = mm.Context(system, integrator)
        return integrator, context

    def create_context(self, structure, forcefield: ForceField | None = None) -> OpenMMHandle:
        molecule = _as_molecule(structure)
        forcefield = _coerce_forcefield(forcefield, molecule)

        system = mm.System()
        for symbol in molecule.symbols:
            if symbol not in MASSES:
                raise KeyError(f"No atomic mass is defined for element '{symbol}'.")
            system.addParticle(MASSES[symbol] * unit.dalton)

        bond_force = mm.CustomBondForce(
            f"k*(10*(r-r0))^2*(1-c3*(10*(r-r0))+c4*(10*(r-r0))^2);c3={MM3_BOND_C3};c4={MM3_BOND_C4}"
        )
        bond_force.addPerBondParameter("k")
        bond_force.addPerBondParameter("r0")

        angle_force = mm.CustomAngleForce(
            "k*(theta-theta0)^2*("
            "1+a3*((theta-theta0)*deg)"
            "+a4*((theta-theta0)*deg)^2"
            "+a5*((theta-theta0)*deg)^3"
            "+a6*((theta-theta0)*deg)^4"
            ");"
            f"a3={MM3_ANGLE_C3};"
            f"a4={MM3_ANGLE_C4};"
            f"a5={MM3_ANGLE_C5};"
            f"a6={MM3_ANGLE_C6};"
            f"deg={RAD_TO_DEG}"
        )
        angle_force.addPerAngleParameter("k")
        angle_force.addPerAngleParameter("theta0")
        vdw_force = mm.CustomNonbondedForce(
            "epsilon*(-2.25*(rv/r)^6 + 184000*exp(-12*r/rv));rv=radius1+radius2;epsilon=sqrt(epsilon1*epsilon2)"
        )
        vdw_force.addPerParticleParameter("radius")
        vdw_force.addPerParticleParameter("epsilon")
        vdw_force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)

        bond_terms: list[_BondTerm] = []
        for bond in molecule.bonds:
            param = _match_bond(forcefield, bond.elements, env_id=bond.env_id, ff_row=bond.ff_row)
            if param is None:
                continue
            force_index = bond_force.addBond(
                bond.atom_i,
                bond.atom_j,
                [_bond_k_to_openmm(param.force_constant), float(param.equilibrium) * 0.1],
            )
            bond_terms.append(
                _BondTerm(
                    force_index=force_index,
                    atom_i=bond.atom_i,
                    atom_j=bond.atom_j,
                    elements=bond.elements,
                    env_id=bond.env_id,
                    ff_row=bond.ff_row,
                )
            )

        angle_terms: list[_AngleTerm] = []
        for angle in molecule.angles:
            param = _match_angle(forcefield, angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if param is None:
                continue
            force_index = angle_force.addAngle(
                angle.atom_i,
                angle.atom_j,
                angle.atom_k,
                [_angle_k_to_openmm(param.force_constant), np.deg2rad(float(param.equilibrium))],
            )
            angle_terms.append(
                _AngleTerm(
                    force_index=force_index,
                    atom_i=angle.atom_i,
                    atom_j=angle.atom_j,
                    atom_k=angle.atom_k,
                    elements=angle.elements,
                    env_id=angle.env_id,
                    ff_row=angle.ff_row,
                )
            )

        vdw_terms: list[_VdwTerm] = []
        if forcefield.vdws:
            for atom_index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=False)):
                param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
                if param is None:
                    raise ValueError(f"Missing vdW parameter for atom {atom_index + 1} ({atom_type or symbol}).")
                vdw_force.addParticle([_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)])
                vdw_terms.append(
                    _VdwTerm(
                        particle_index=atom_index,
                        atom_type=atom_type,
                        element=symbol,
                        ff_row=param.ff_row,
                    )
                )
            vdw_force.createExclusionsFromBonds([(bond.atom_i, bond.atom_j) for bond in molecule.bonds], 2)

        if not bond_terms and not angle_terms and not vdw_terms:
            raise ValueError(
                "No OpenMM terms were created. Force field did not match any detected bonds, angles, or vdW types."
            )

        if bond_terms:
            system.addForce(bond_force)
        else:
            bond_force = None

        if angle_terms:
            system.addForce(angle_force)
        else:
            angle_force = None

        if vdw_terms:
            system.addForce(vdw_force)
        else:
            vdw_force = None

        integrator, context = self._create_context(system)
        context.setPositions(self._positions(molecule))

        return OpenMMHandle(
            molecule=copy.deepcopy(molecule),
            system=system,
            integrator=integrator,
            context=context,
            bond_force=bond_force,
            angle_force=angle_force,
            vdw_force=vdw_force,
            bond_terms=bond_terms,
            angle_terms=angle_terms,
            vdw_terms=vdw_terms,
        )

    def update_forcefield(self, handle: OpenMMHandle, forcefield: ForceField):
        """Update per-term parameters in an existing OpenMM Context."""
        if handle.bond_force is not None:
            for term in handle.bond_terms:
                param = _match_bond(forcefield, term.elements, env_id=term.env_id, ff_row=term.ff_row)
                if param is None:
                    raise ValueError(f"Updated force field is missing bond parameter for {term.elements}.")
                handle.bond_force.setBondParameters(
                    term.force_index,
                    term.atom_i,
                    term.atom_j,
                    [_bond_k_to_openmm(param.force_constant), float(param.equilibrium) * 0.1],
                )
            handle.bond_force.updateParametersInContext(handle.context)

        if handle.angle_force is not None:
            for term in handle.angle_terms:
                param = _match_angle(forcefield, term.elements, env_id=term.env_id, ff_row=term.ff_row)
                if param is None:
                    raise ValueError(f"Updated force field is missing angle parameter for {term.elements}.")
                handle.angle_force.setAngleParameters(
                    term.force_index,
                    term.atom_i,
                    term.atom_j,
                    term.atom_k,
                    [_angle_k_to_openmm(param.force_constant), np.deg2rad(float(param.equilibrium))],
                )
            handle.angle_force.updateParametersInContext(handle.context)

        if handle.vdw_force is not None:
            for term in handle.vdw_terms:
                param = _match_vdw(forcefield, atom_type=term.atom_type, element=term.element, ff_row=term.ff_row)
                if param is None:
                    raise ValueError(
                        f"Updated force field is missing vdW parameter for {term.atom_type or term.element}."
                    )
                handle.vdw_force.setParticleParameters(
                    term.particle_index,
                    [_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)],
                )
            handle.vdw_force.updateParametersInContext(handle.context)

    def _prepare_handle(self, structure, forcefield: ForceField | None = None) -> OpenMMHandle:
        if isinstance(structure, OpenMMHandle):
            handle = structure
            if forcefield is not None:
                self.update_forcefield(handle, forcefield)
            return handle
        return self.create_context(structure, forcefield)

    def energy(self, structure, forcefield: ForceField | None = None) -> float:
        handle = self._prepare_handle(structure, forcefield)
        state = handle.context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def minimize(
        self, structure, forcefield: ForceField | None = None, tolerance: float = 1.0, max_iterations: int = 200
    ):
        handle = self._prepare_handle(structure, forcefield)
        mm.LocalEnergyMinimizer.minimize(handle.context, tolerance, max_iterations)
        state = handle.context.getState(getEnergy=True, getPositions=True)
        coords = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        handle.molecule.geometry = coords
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        return energy, list(handle.molecule.symbols), coords

    def hessian(self, structure, forcefield: ForceField | None = None, step: float = 1.0e-4) -> np.ndarray:
        """Finite-difference Hessian in kJ/mol/nm^2."""
        handle = self._prepare_handle(structure, forcefield)
        positions = np.array(
            handle.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        )
        n_atoms = positions.shape[0]
        hessian = np.zeros((3 * n_atoms, 3 * n_atoms))

        for atom_index in range(n_atoms):
            for coord_index in range(3):
                column = 3 * atom_index + coord_index

                displaced_plus = positions.copy()
                displaced_minus = positions.copy()
                displaced_plus[atom_index, coord_index] += step
                displaced_minus[atom_index, coord_index] -= step

                handle.context.setPositions(displaced_plus * unit.nanometer)
                forces_plus = np.array(
                    handle.context.getState(getForces=True)
                    .getForces(asNumpy=True)
                    .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                )

                handle.context.setPositions(displaced_minus * unit.nanometer)
                forces_minus = np.array(
                    handle.context.getState(getForces=True)
                    .getForces(asNumpy=True)
                    .value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
                )

                hessian[:, column] = -((forces_plus - forces_minus) / (2.0 * step)).reshape(-1)

        handle.context.setPositions(positions * unit.nanometer)
        return 0.5 * (hessian + hessian.T)

    def frequencies(self, structure, forcefield: ForceField | None = None) -> list[float]:
        """Approximate harmonic frequencies in cm^-1 from the numerical Hessian."""
        handle = self._prepare_handle(structure, forcefield)
        hessian = self.hessian(handle)

        hessian_si = hessian * (1000.0 / AVO) * 1.0e18
        masses = np.array([MASSES[symbol] * AMU_TO_KG for symbol in handle.molecule.symbols], dtype=float)
        mass_vector = np.repeat(masses, 3)
        mass_weighted = hessian_si / np.sqrt(np.outer(mass_vector, mass_vector))

        eigenvalues = np.linalg.eigvalsh(mass_weighted)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        frequencies /= 2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0
        return frequencies.tolist()
