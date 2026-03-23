"""OpenMM molecular mechanics backend."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.constants import (
    AMU_TO_KG,
    BOHR_TO_ANG,
    HARTREE_TO_J,
    SPEED_OF_LIGHT_MS,
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    MM3_ANGLE_C6,
    RAD_TO_DEG,
    KCAL_TO_KJ,
    KJMOLNM2_TO_HESSIAN_AU,
    AVO,
    MASSES,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, VdwParam
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
    functional_form: FunctionalForm = FunctionalForm.MM3


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
    """Convert canonical bond force constant (kcal/mol/Å²) to kJ/mol/Å²."""
    return float(force_constant) * KCAL_TO_KJ


def _angle_k_to_openmm(force_constant: float) -> float:
    """Convert canonical angle force constant (kcal/mol/rad²) to kJ/mol/rad²."""
    return float(force_constant) * KCAL_TO_KJ


def _bond_k_to_harmonic(force_constant: float) -> float:
    """Canonical bond k (kcal/mol/Å²) → HarmonicBondForce k (kJ/mol/nm²).

    OpenMM's HarmonicBondForce uses E = ½·k·(r−r₀)² while the canonical
    convention is E = k·(r−r₀)², so k_openmm = 2·k.  Additionally convert
    kcal→kJ (×4.184) and Å⁻²→nm⁻² (×100).
    """
    return 2.0 * float(force_constant) * KCAL_TO_KJ * 100.0


def _angle_k_to_harmonic(force_constant: float) -> float:
    """Canonical angle k (kcal/mol/rad²) → HarmonicAngleForce k (kJ/mol/rad²).

    OpenMM's HarmonicAngleForce uses E = ½·k·(θ−θ₀)² while the canonical
    convention is E = k·(θ−θ₀)², so k_openmm = 2·k.  Convert kcal→kJ.
    """
    return 2.0 * float(force_constant) * KCAL_TO_KJ


def _vdw_sigma_nm(radius: float) -> float:
    """Convert Rmin/2 (Å) to LJ sigma (nm) for standard 12-6 NonbondedForce."""
    return float(radius) * 2.0 / (2.0 ** (1.0 / 6.0)) * 0.1


def _vdw_radius_to_openmm(radius: float) -> float:
    return float(radius) * 0.1


def _vdw_epsilon_to_openmm(epsilon: float) -> float:
    return float(epsilon) * KCAL_TO_KJ


def _match_bond(
    forcefield: ForceField, elements: tuple[str, str], env_id: str = "", ff_row: int | None = None
) -> BondParam | None:
    return forcefield.match_bond(elements, env_id=env_id, ff_row=ff_row)


def _match_angle(
    forcefield: ForceField, elements: tuple[str, str, str], env_id: str = "", ff_row: int | None = None
) -> AngleParam | None:
    return forcefield.match_angle(elements, env_id=env_id, ff_row=ff_row)


def _match_vdw(
    forcefield: ForceField, atom_type: str = "", element: str = "", ff_row: int | None = None
) -> VdwParam | None:
    return forcefield.match_vdw(atom_type=atom_type, element=element, ff_row=ff_row)


class OpenMMEngine(MMEngine):
    """Molecular mechanics backend powered by OpenMM."""

    def __init__(self, platform_name: str | None = None):
        _ensure_openmm()
        self._platform_name = platform_name

    @property
    def name(self) -> str:
        return "OpenMM"

    def supported_functional_forms(self) -> frozenset[str]:
        return frozenset({"harmonic", "mm3"})

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
        self._validate_forcefield(forcefield)

        # Default to MM3 for backward compatibility when functional_form is None
        ff_form = forcefield.functional_form or FunctionalForm.MM3
        use_harmonic = ff_form == FunctionalForm.HARMONIC

        system = mm.System()
        for symbol in molecule.symbols:
            if symbol not in MASSES:
                raise KeyError(f"No atomic mass is defined for element '{symbol}'.")
            system.addParticle(MASSES[symbol] * unit.dalton)

        # --- Create force objects based on functional form ---
        if use_harmonic:
            bond_force = mm.HarmonicBondForce()
        else:
            bond_force = mm.CustomBondForce(
                f"k*(10*(r-r0))^2*(1-c3*(10*(r-r0))+c4*(10*(r-r0))^2);c3={MM3_BOND_C3};c4={MM3_BOND_C4}"
            )
            bond_force.addPerBondParameter("k")
            bond_force.addPerBondParameter("r0")

        if use_harmonic:
            angle_force = mm.HarmonicAngleForce()
        else:
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

        if use_harmonic:
            vdw_force = mm.NonbondedForce()
            vdw_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
        else:
            # MM3 Buckingham exp-6 with short-range repulsive wall.
            # Below r < 0.34·rv the attractive r^-6 dominates the exponential,
            # causing divergence to -∞.  Switch to a hard repulsive form
            # at that boundary using step() to prevent collapse.
            vdw_force = mm.CustomNonbondedForce(
                "step(r - rc) * epsilon*(-2.25*(rv/r)^6 + 184000*exp(-12*r/rv))"
                " + step(rc - r) * epsilon*184000*exp(-12*rc/rv) * (rc/r)^12;"
                "rc=0.34*rv;"
                "rv=radius1+radius2;"
                "epsilon=sqrt(epsilon1*epsilon2)"
            )
            vdw_force.addPerParticleParameter("radius")
            vdw_force.addPerParticleParameter("epsilon")
            vdw_force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)

        # --- Assign bond parameters ---
        bond_terms: list[_BondTerm] = []
        for bond in molecule.bonds:
            param = _match_bond(forcefield, bond.elements, env_id=bond.env_id, ff_row=bond.ff_row)
            if param is None:
                continue
            if use_harmonic:
                force_index = bond_force.addBond(
                    bond.atom_i,
                    bond.atom_j,
                    float(param.equilibrium) * 0.1,
                    _bond_k_to_harmonic(param.force_constant),
                )
            else:
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

        # --- Assign angle parameters ---
        angle_terms: list[_AngleTerm] = []
        for angle in molecule.angles:
            param = _match_angle(forcefield, angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
            if param is None:
                continue
            if use_harmonic:
                force_index = angle_force.addAngle(
                    angle.atom_i,
                    angle.atom_j,
                    angle.atom_k,
                    np.deg2rad(float(param.equilibrium)),
                    _angle_k_to_harmonic(param.force_constant),
                )
            else:
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

        # --- Assign vdW parameters ---
        vdw_terms: list[_VdwTerm] = []
        if forcefield.vdws:
            for atom_index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=False)):
                param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
                if param is None:
                    raise ValueError(f"Missing vdW parameter for atom {atom_index + 1} ({atom_type or symbol}).")
                if use_harmonic:
                    vdw_force.addParticle(0.0, _vdw_sigma_nm(param.radius), _vdw_epsilon_to_openmm(param.epsilon))
                else:
                    vdw_force.addParticle([_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)])
                vdw_terms.append(
                    _VdwTerm(
                        particle_index=atom_index,
                        atom_type=atom_type,
                        element=symbol,
                        ff_row=param.ff_row,
                    )
                )
            if use_harmonic:
                # Exclude 1-2 and 1-3 pairs; scale 1-4 pairs by 1/2 (AMBER scnb=2.0)
                excluded_12: set[tuple[int, int]] = set()
                for bond in molecule.bonds:
                    excluded_12.add((min(bond.atom_i, bond.atom_j), max(bond.atom_i, bond.atom_j)))

                excluded_13: set[tuple[int, int]] = set()
                for angle in molecule.angles:
                    excluded_13.add((min(angle.atom_i, angle.atom_k), max(angle.atom_i, angle.atom_k)))
                excluded_13 -= excluded_12  # pure 1-3 only

                # Build adjacency for 1-4 detection
                neighbors: dict[int, set[int]] = {}
                for bond in molecule.bonds:
                    neighbors.setdefault(bond.atom_i, set()).add(bond.atom_j)
                    neighbors.setdefault(bond.atom_j, set()).add(bond.atom_i)

                pairs_14: set[tuple[int, int]] = set()
                for angle in molecule.angles:
                    # 1-4 pairs: atoms bonded to angle endpoints but not in the angle
                    for nb in neighbors.get(angle.atom_i, ()):
                        if nb != angle.atom_j and nb != angle.atom_k:
                            pairs_14.add((min(nb, angle.atom_k), max(nb, angle.atom_k)))
                    for nb in neighbors.get(angle.atom_k, ()):
                        if nb != angle.atom_j and nb != angle.atom_i:
                            pairs_14.add((min(nb, angle.atom_i), max(nb, angle.atom_i)))
                pairs_14 -= excluded_12
                pairs_14 -= excluded_13

                # Fully exclude 1-2 and 1-3
                for p1, p2 in sorted(excluded_12 | excluded_13):
                    vdw_force.addException(p1, p2, 0.0, 1.0, 0.0)

                # Scale 1-4 pairs (AMBER: scnb=2.0 → epsilon/2, scee=1.2 → no charges here)
                SCNB = 2.0
                for p1, p2 in sorted(pairs_14):
                    sig1, eps1 = vdw_force.getParticleParameters(p1)[1:]
                    sig2, eps2 = vdw_force.getParticleParameters(p2)[1:]
                    sig_14 = 0.5 * (sig1 + sig2)
                    eps_14 = (eps1 * eps2) ** 0.5 / SCNB
                    vdw_force.addException(p1, p2, 0.0, sig_14, eps_14)
            else:
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
            functional_form=ff_form,
        )

    def update_forcefield(self, handle: OpenMMHandle, forcefield: ForceField):
        """Update per-term parameters in an existing OpenMM Context."""
        incoming_form = forcefield.functional_form
        if incoming_form is not None and incoming_form != handle.functional_form:
            raise ValueError(
                f"Force field functional form {incoming_form!r} does not match "
                f"the handle's form {handle.functional_form!r}. "
                f"Create a new context instead of reusing this handle."
            )
        use_harmonic = handle.functional_form == FunctionalForm.HARMONIC

        if handle.bond_force is not None:
            for term in handle.bond_terms:
                param = _match_bond(forcefield, term.elements, env_id=term.env_id, ff_row=term.ff_row)
                if param is None:
                    raise ValueError(f"Updated force field is missing bond parameter for {term.elements}.")
                if use_harmonic:
                    handle.bond_force.setBondParameters(
                        term.force_index,
                        term.atom_i,
                        term.atom_j,
                        float(param.equilibrium) * 0.1,
                        _bond_k_to_harmonic(param.force_constant),
                    )
                else:
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
                if use_harmonic:
                    handle.angle_force.setAngleParameters(
                        term.force_index,
                        term.atom_i,
                        term.atom_j,
                        term.atom_k,
                        np.deg2rad(float(param.equilibrium)),
                        _angle_k_to_harmonic(param.force_constant),
                    )
                else:
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
                if use_harmonic:
                    handle.vdw_force.setParticleParameters(
                        term.particle_index,
                        0.0,
                        _vdw_sigma_nm(param.radius),
                        _vdw_epsilon_to_openmm(param.epsilon),
                    )
                else:
                    handle.vdw_force.setParticleParameters(
                        term.particle_index,
                        [_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)],
                    )
            handle.vdw_force.updateParametersInContext(handle.context)

    def export_system_xml(
        self,
        path: str | Path,
        structure,
        forcefield: ForceField | None = None,
    ) -> Path:
        """Serialize the OpenMM System to XML.

        Produces a topology-specific XML file containing the force objects
        (``HarmonicBondForce``/``CustomBondForce``, etc. depending on the
        functional form) with all per-term parameters.  The file can be
        loaded back with ``openmm.XmlSerializer.deserialize()``.

        Args:
            path: Output file path.
            structure: A :class:`~q2mm.models.molecule.Q2MMMolecule`,
                path to an XYZ file, or an existing :class:`OpenMMHandle`.
            forcefield: Force field to apply.  When *structure* is not an
                :class:`OpenMMHandle`, this is used to build the OpenMM
                system.  When *structure* is an existing
                :class:`OpenMMHandle`, providing a non-None *forcefield*
                updates the per-term parameters of that handle; if
                *forcefield* is None, the handle's current parameters are
                used unchanged.

        Returns:
            The resolved output path.
        """
        handle = self._prepare_handle(structure, forcefield)
        xml_string = mm.XmlSerializer.serialize(handle.system)
        output = Path(path)
        output.write_text(xml_string, encoding="utf-8")
        return output

    @staticmethod
    def load_system_xml(path: str | Path):
        """Deserialize an OpenMM System from XML.

        Returns:
            An ``openmm.System`` object.
        """
        _ensure_openmm()
        xml_string = Path(path).read_text(encoding="utf-8")
        return mm.XmlSerializer.deserialize(xml_string)

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
        """Finite-difference Hessian in canonical units (Hartree/Bohr²).

        Internally computed in kJ/mol/nm² (OpenMM native) and converted
        to Hartree/Bohr² before returning, matching the canonical unit
        contract defined in :class:`~q2mm.backends.base.MMEngine`.
        """
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
        hessian_symmetric = 0.5 * (hessian + hessian.T)

        # Convert from OpenMM native kJ/mol/nm² to canonical Hartree/Bohr²
        return hessian_symmetric * KJMOLNM2_TO_HESSIAN_AU

    def frequencies(self, structure, forcefield: ForceField | None = None) -> list[float]:
        """Approximate harmonic frequencies in cm^-1 from the numerical Hessian."""
        handle = self._prepare_handle(structure, forcefield)
        hessian_au = self.hessian(handle)  # Hartree/Bohr²

        # Convert Hartree/Bohr² → J/m² (per molecule, not per mol)
        bohr_to_m = BOHR_TO_ANG * 1e-10
        hessian_si = hessian_au * HARTREE_TO_J / (bohr_to_m**2)

        masses = np.array([MASSES[symbol] * AMU_TO_KG for symbol in handle.molecule.symbols], dtype=float)
        mass_vector = np.repeat(masses, 3)
        mass_weighted = hessian_si / np.sqrt(np.outer(mass_vector, mass_vector))

        eigenvalues = np.linalg.eigvalsh(mass_weighted)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        frequencies /= 2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0
        return frequencies.tolist()
