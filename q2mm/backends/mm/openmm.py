"""OpenMM molecular mechanics backend.

Provides a full-featured MM engine using OpenMM for energy, minimization,
Hessian, and frequency calculations.  Supports both harmonic and MM3
functional forms with runtime parameter updates via :class:`OpenMMHandle`.
"""

from __future__ import annotations

import copy
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.backends.registry import register_mm
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
    MASSES,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, TorsionParam, VdwParam
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
    """Internal record mapping a molecule bond to its OpenMM force index.

    Attributes:
        force_index: Index of this bond in the OpenMM bond force object.
        atom_i: First atom index.
        atom_j: Second atom index.
        elements: Element symbols for the two atoms.
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.

    """

    force_index: int
    atom_i: int
    atom_j: int
    elements: tuple[str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _AngleTerm:
    """Internal record mapping a molecule angle to its OpenMM force index.

    Attributes:
        force_index: Index of this angle in the OpenMM angle force object.
        atom_i: First atom index.
        atom_j: Central atom index.
        atom_k: Third atom index.
        elements: Element symbols for the three atoms.
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.

    """

    force_index: int
    atom_i: int
    atom_j: int
    atom_k: int
    elements: tuple[str, str, str]
    env_id: str = ""
    ff_row: int | None = None


@dataclass
class _VdwTerm:
    """Internal record mapping a molecule atom to its OpenMM vdW particle.

    Attributes:
        particle_index: Index of this particle in the OpenMM vdW force.
        atom_type: Atom type label for parameter matching.
        element: Element symbol.
        ff_row: Row index in the source force field file, if applicable.

    """

    particle_index: int
    atom_type: str = ""
    element: str = ""
    ff_row: int | None = None


@dataclass
class _Exception14:
    """A 1-4 nonbonded exception whose parameters must track particle updates.

    Attributes:
        exception_index: Index of this exception in the OpenMM NonbondedForce.
        particle_i: First particle index.
        particle_j: Second particle index.

    """

    exception_index: int
    particle_i: int
    particle_j: int


@dataclass
class _TorsionTerm:
    """Internal record mapping a molecule torsion to its OpenMM force index.

    Attributes:
        force_index: Index of this torsion in the OpenMM torsion force object.
        atom_i: First atom index.
        atom_j: Second atom index.
        atom_k: Third atom index.
        atom_l: Fourth atom index.
        elements: Element symbols for the four atoms.
        periodicity: Fourier component periodicity (1, 2, or 3).
        env_id: Chemical environment identifier for parameter matching.
        ff_row: Row index in the source force field file, if applicable.
        is_improper: Whether this term is an improper torsion.

    """

    force_index: int
    atom_i: int
    atom_j: int
    atom_k: int
    atom_l: int
    elements: tuple[str, str, str, str]
    periodicity: int = 1
    env_id: str = ""
    ff_row: int | None = None
    is_improper: bool = False


@dataclass
class OpenMMHandle:
    """Reusable OpenMM system/context pair for fast parameter updates.

    Attributes:
        molecule: Deep copy of the input molecule.
        system: The ``openmm.System`` object.
        integrator: The ``openmm.Integrator`` used by the context.
        context: The ``openmm.Context`` for energy evaluation.
        bond_force: The OpenMM bond force object, or ``None`` if no bonds.
        angle_force: The OpenMM angle force object, or ``None`` if no angles.
        torsion_force: The OpenMM torsion force object, or ``None`` if no torsions.
        vdw_force: The OpenMM vdW force object, or ``None`` if no vdW terms.
        bond_terms: Mapping of molecule bonds to force indices.
        angle_terms: Mapping of molecule angles to force indices.
        torsion_terms: Mapping of molecule torsions to force indices.
        vdw_terms: Mapping of atoms to vdW particle indices.
        exceptions_14: 1-4 nonbonded exceptions (harmonic form only).
        functional_form: The functional form used when the handle was created.

    """

    molecule: Q2MMMolecule
    system: object
    integrator: object
    context: object
    bond_force: object | None
    angle_force: object | None
    torsion_force: object | None
    vdw_force: object | None
    bond_terms: list[_BondTerm]
    angle_terms: list[_AngleTerm]
    torsion_terms: list[_TorsionTerm]
    vdw_terms: list[_VdwTerm]
    exceptions_14: list[_Exception14] = field(default_factory=list)
    functional_form: FunctionalForm = FunctionalForm.MM3


def _ensure_openmm() -> None:
    """Raise ``ImportError`` if OpenMM is not installed.

    Raises:
        ImportError: If the ``openmm`` package cannot be imported.

    """
    if not _HAS_OPENMM:
        raise ImportError('OpenMM is not installed. Install with `pip install openmm` or `pip install -e ".[openmm]"`.')


def _as_molecule(structure: Q2MMMolecule | str | Path) -> Q2MMMolecule:
    """Coerce *structure* to a :class:`Q2MMMolecule`.

    Args:
        structure: A :class:`Q2MMMolecule`, file path (``str`` or ``Path``),
            or any other type (which will raise ``TypeError``).

    Returns:
        Q2MMMolecule: The coerced molecule object.

    Raises:
        TypeError: If *structure* is not a recognised type.

    """
    if isinstance(structure, Q2MMMolecule):
        return structure
    if isinstance(structure, (str, Path)):
        return Q2MMMolecule.from_xyz(structure)
    raise TypeError("OpenMMEngine expects a Q2MMMolecule or path to an XYZ file.")


def _coerce_forcefield(forcefield: ForceField | None, molecule: Q2MMMolecule) -> ForceField:
    """Return *forcefield* or create a default one from *molecule*.

    Args:
        forcefield: An explicit force field, or ``None`` to auto-generate.
        molecule: Molecule used when generating a default force field.

    Returns:
        ForceField: The provided or auto-generated force field.

    """
    if forcefield is not None:
        return forcefield
    return ForceField.create_for_molecule(molecule)


def _bond_k_to_openmm(force_constant: float) -> float:
    """Convert canonical bond force constant (kcal/mol/Å²) to kJ/mol/Å².

    Args:
        force_constant: Bond force constant in kcal/mol/Å².

    Returns:
        float: Bond force constant in kJ/mol/Å².

    """
    return float(force_constant) * KCAL_TO_KJ


def _angle_k_to_openmm(force_constant: float) -> float:
    """Convert canonical angle force constant (kcal/mol/rad²) to kJ/mol/rad².

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad².

    """
    return float(force_constant) * KCAL_TO_KJ


def _bond_k_to_harmonic(force_constant: float) -> float:
    """Canonical bond k (kcal/mol/Å²) → HarmonicBondForce k (kJ/mol/nm²).

    OpenMM's HarmonicBondForce uses E = ½·k·(r−r₀)² while the canonical
    convention is E = k·(r−r₀)², so k_openmm = 2·k.  Additionally convert
    kcal→kJ (×4.184) and Å⁻²→nm⁻² (×100).

    Args:
        force_constant: Bond force constant in kcal/mol/Å².

    Returns:
        float: Bond force constant in kJ/mol/nm² with the ½ convention.

    """
    return 2.0 * float(force_constant) * KCAL_TO_KJ * 100.0


def _angle_k_to_harmonic(force_constant: float) -> float:
    """Canonical angle k (kcal/mol/rad²) → HarmonicAngleForce k (kJ/mol/rad²).

    OpenMM's HarmonicAngleForce uses E = ½·k·(θ−θ₀)² while the canonical
    convention is E = k·(θ−θ₀)², so k_openmm = 2·k.  Convert kcal→kJ.

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad² with the ½ convention.

    """
    return 2.0 * float(force_constant) * KCAL_TO_KJ


def _vdw_sigma_nm(radius: float) -> float:
    """Convert Rmin/2 (Å) to LJ sigma (nm) for standard 12-6 NonbondedForce.

    Args:
        radius: Van der Waals radius (Rmin/2) in Å.

    Returns:
        float: LJ sigma in nm.

    """
    return float(radius) * 2.0 / (2.0 ** (1.0 / 6.0)) * 0.1


def _vdw_radius_to_openmm(radius: float) -> float:
    """Convert vdW radius from Å to nm for CustomNonbondedForce.

    Args:
        radius: Van der Waals radius in Å.

    Returns:
        float: Van der Waals radius in nm.

    """
    return float(radius) * 0.1


def _vdw_epsilon_to_openmm(epsilon: float) -> float:
    """Convert vdW epsilon from kcal/mol to kJ/mol.

    Args:
        epsilon: Well depth in kcal/mol.

    Returns:
        float: Well depth in kJ/mol.

    """
    return float(epsilon) * KCAL_TO_KJ


def _match_bond(
    forcefield: ForceField, elements: tuple[str, str], env_id: str = "", ff_row: int | None = None
) -> BondParam | None:
    """Look up a bond parameter from the force field.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the two bonded atoms.
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.

    Returns:
        BondParam | None: Matched parameter, or ``None`` if not found.

    """
    return forcefield.match_bond(elements, env_id=env_id, ff_row=ff_row)


def _match_angle(
    forcefield: ForceField, elements: tuple[str, str, str], env_id: str = "", ff_row: int | None = None
) -> AngleParam | None:
    """Look up an angle parameter from the force field.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the three atoms (i, central, k).
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.

    Returns:
        AngleParam | None: Matched parameter, or ``None`` if not found.

    """
    return forcefield.match_angle(elements, env_id=env_id, ff_row=ff_row)


def _match_vdw(
    forcefield: ForceField, atom_type: str = "", element: str = "", ff_row: int | None = None
) -> VdwParam | None:
    """Look up a vdW parameter from the force field.

    Args:
        forcefield: Force field to search.
        atom_type: Atom type label for matching.
        element: Element symbol for fallback matching.
        ff_row: Optional row index hint for matching.

    Returns:
        VdwParam | None: Matched parameter, or ``None`` if not found.

    """
    return forcefield.match_vdw(atom_type=atom_type, element=element, ff_row=ff_row)


def _torsion_k_to_openmm(force_constant: float) -> float:
    """Convert canonical torsion force constant (kcal/mol) to kJ/mol.

    Args:
        force_constant: Torsion force constant in kcal/mol.

    Returns:
        float: Torsion force constant in kJ/mol.

    """
    return float(force_constant) * KCAL_TO_KJ


def _match_torsions(
    forcefield: ForceField,
    elements: tuple[str, str, str, str],
    env_id: str = "",
    ff_row: int | None = None,
    is_improper: bool | None = None,
) -> list[TorsionParam]:
    """Look up torsion parameters from the force field.

    Returns all periodicity components matching the given torsion.

    Args:
        forcefield: Force field to search.
        elements: Element symbols of the four torsion atoms.
        env_id: Chemical environment identifier.
        ff_row: Optional row index hint for matching.
        is_improper: If set, only match proper (False) or improper (True).

    Returns:
        list[TorsionParam]: All matching torsion parameter components.

    """
    return forcefield.match_torsion(elements, env_id=env_id, ff_row=ff_row, is_improper=is_improper)


@register_mm("openmm")
class OpenMMEngine(MMEngine):
    """Molecular mechanics backend powered by OpenMM.

    Supports both harmonic (AMBER-style) and MM3 functional forms.
    Provides reusable :class:`OpenMMHandle` objects for fast parameter
    updates during optimization loops.
    """

    def __init__(self, platform_name: str | None = None) -> None:
        """Initialize the OpenMM engine.

        Args:
            platform_name: OpenMM platform to use (e.g. ``"CPU"``,
                ``"CUDA"``). Auto-selected if ``None``.

        Raises:
            ImportError: If OpenMM is not installed.

        """
        _ensure_openmm()
        self._platform_name = platform_name

    @property
    def name(self) -> str:
        """Human-readable engine name.

        Returns:
            str: ``"OpenMM"``.

        """
        return "OpenMM"

    def supported_functional_forms(self) -> frozenset[str]:
        """Functional forms this engine can evaluate.

        Returns:
            frozenset[str]: ``{"harmonic", "mm3"}``.

        """
        return frozenset({"harmonic", "mm3"})

    def is_available(self) -> bool:
        """Check if OpenMM is installed.

        Returns:
            bool: ``True`` if the ``openmm`` package is importable.

        """
        return _HAS_OPENMM

    def supports_runtime_params(self) -> bool:
        """Whether parameters can be updated without rebuilding the system.

        Returns:
            bool: Always ``True`` for OpenMM.

        """
        return True

    def _positions(self, molecule: Q2MMMolecule) -> Any:
        """Convert molecule geometry to OpenMM position array.

        Args:
            molecule: Molecule whose geometry to convert.

        Returns:
            Positions as an OpenMM ``Quantity`` in Å.

        """
        return np.asarray(molecule.geometry, dtype=float) * unit.angstrom

    def _create_context(self, system: Any) -> tuple[Any, Any]:
        """Create an OpenMM integrator and context for *system*.

        Args:
            system: An ``openmm.System`` object.

        Returns:
            tuple: ``(integrator, context)`` pair.

        """
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        if self._platform_name:
            platform = mm.Platform.getPlatformByName(self._platform_name)
            context = mm.Context(system, integrator, platform)
        else:
            context = mm.Context(system, integrator)
        return integrator, context

    def create_context(
        self, structure: Q2MMMolecule | str | Path, forcefield: ForceField | None = None
    ) -> OpenMMHandle:
        """Build an OpenMM system and context for a molecule.

        Creates force objects (bond, angle, vdW) matching the force field's
        functional form and assigns per-term parameters from *forcefield*.

        Args:
            structure (Q2MMMolecule | str | Path): A
                :class:`~q2mm.models.molecule.Q2MMMolecule` or path to an
                XYZ file.
            forcefield: Force field to apply. Auto-generated from the
                molecule if ``None``.

        Returns:
            OpenMMHandle: Reusable handle for energy evaluation and parameter
                updates.

        Raises:
            KeyError: If an atom's element has no defined mass.
            ValueError: If no OpenMM terms could be created from the force
                field, or if a vdW parameter is missing for an atom.

        """
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

        # Torsion force: both harmonic (AMBER) and MM3 use PeriodicTorsionForce
        # E = k * (1 + cos(n*θ − phase))
        torsion_force = mm.PeriodicTorsionForce()

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

        # --- Assign proper torsion parameters ---
        torsion_terms: list[_TorsionTerm] = []
        for torsion in molecule.torsions:
            params = _match_torsions(
                forcefield,
                torsion.element_quad,
                env_id=torsion.env_id,
                ff_row=torsion.ff_row,
                is_improper=False,
            )
            for param in params:
                force_index = torsion_force.addTorsion(
                    torsion.atom_i,
                    torsion.atom_j,
                    torsion.atom_k,
                    torsion.atom_l,
                    param.periodicity,
                    np.deg2rad(float(param.phase)),
                    _torsion_k_to_openmm(param.force_constant),
                )
                torsion_terms.append(
                    _TorsionTerm(
                        force_index=force_index,
                        atom_i=torsion.atom_i,
                        atom_j=torsion.atom_j,
                        atom_k=torsion.atom_k,
                        atom_l=torsion.atom_l,
                        elements=torsion.element_quad,
                        periodicity=param.periodicity,
                        env_id=torsion.env_id,
                        ff_row=param.ff_row,
                    )
                )

        # NOTE: Improper torsions are not yet supported.  Proper improper
        # detection requires identifying trigonal centres (not bond-graph
        # walks), which is tracked as future work.  The improper params in
        # the ForceField are preserved but not assigned to OpenMM forces.

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
                exceptions_14: list[_Exception14] = []
                for p1, p2 in sorted(pairs_14):
                    sig1, eps1 = vdw_force.getParticleParameters(p1)[1:]
                    sig2, eps2 = vdw_force.getParticleParameters(p2)[1:]
                    sig_14 = 0.5 * (sig1 + sig2)
                    eps_14 = (eps1 * eps2) ** 0.5 / SCNB
                    exc_idx = vdw_force.addException(p1, p2, 0.0, sig_14, eps_14)
                    exceptions_14.append(_Exception14(exception_index=exc_idx, particle_i=p1, particle_j=p2))
            else:
                vdw_force.createExclusionsFromBonds([(bond.atom_i, bond.atom_j) for bond in molecule.bonds], 2)

        if not bond_terms and not angle_terms and not torsion_terms and not vdw_terms:
            raise ValueError(
                "No OpenMM terms were created. Force field did not match any detected bonds, angles, torsions, or vdW types."
            )

        if bond_terms:
            system.addForce(bond_force)
        else:
            bond_force = None

        if angle_terms:
            system.addForce(angle_force)
        else:
            angle_force = None

        if torsion_terms:
            system.addForce(torsion_force)
        else:
            torsion_force = None

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
            torsion_force=torsion_force,
            vdw_force=vdw_force,
            bond_terms=bond_terms,
            angle_terms=angle_terms,
            torsion_terms=torsion_terms,
            vdw_terms=vdw_terms,
            exceptions_14=exceptions_14 if use_harmonic and vdw_terms else [],
            functional_form=ff_form,
        )

    def update_forcefield(self, handle: OpenMMHandle, forcefield: ForceField) -> None:
        """Update per-term parameters in an existing OpenMM Context.

        Modifies bond, angle, and vdW parameters in-place, then pushes
        changes to the OpenMM context.  Much faster than rebuilding the
        system from scratch.

        Args:
            handle: An existing :class:`OpenMMHandle` to update.
            forcefield: New force field parameters to apply.

        Raises:
            ValueError: If the force field's functional form does not match
                the handle's form, or if a required parameter is missing.

        """
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

        if handle.torsion_force is not None:
            for term in handle.torsion_terms:
                params = _match_torsions(
                    forcefield, term.elements, env_id=term.env_id, ff_row=term.ff_row, is_improper=term.is_improper
                )
                matched = [p for p in params if p.periodicity == term.periodicity]
                if not matched:
                    raise ValueError(
                        f"Updated force field is missing torsion parameter for "
                        f"{term.elements} periodicity={term.periodicity}."
                    )
                param = matched[0]
                handle.torsion_force.setTorsionParameters(
                    term.force_index,
                    term.atom_i,
                    term.atom_j,
                    term.atom_k,
                    term.atom_l,
                    param.periodicity,
                    np.deg2rad(float(param.phase)),
                    _torsion_k_to_openmm(param.force_constant),
                )
            handle.torsion_force.updateParametersInContext(handle.context)

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

            # Recompute 1-4 exception params from updated particle params
            if use_harmonic and handle.exceptions_14:
                SCNB = 2.0
                for exc in handle.exceptions_14:
                    _, sig1, eps1 = handle.vdw_force.getParticleParameters(exc.particle_i)
                    _, sig2, eps2 = handle.vdw_force.getParticleParameters(exc.particle_j)
                    sig_14 = 0.5 * (sig1 + sig2)
                    eps_14 = (eps1 * eps2) ** 0.5 / SCNB
                    handle.vdw_force.setExceptionParameters(
                        exc.exception_index, exc.particle_i, exc.particle_j, 0.0, sig_14, eps_14
                    )

            handle.vdw_force.updateParametersInContext(handle.context)

    def export_system_xml(
        self,
        path: str | Path,
        structure: Q2MMMolecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
    ) -> Path:
        """Serialize the OpenMM System to XML.

        Produces a topology-specific XML file containing the force objects
        (``HarmonicBondForce``/``CustomBondForce``, etc. depending on the
        functional form) with all per-term parameters.  The file can be
        loaded back with ``openmm.XmlSerializer.deserialize()``.

        Args:
            path: Output file path.
            structure (Q2MMMolecule | str | Path | OpenMMHandle): A
                :class:`~q2mm.models.molecule.Q2MMMolecule`, path to an XYZ
                file, or an existing :class:`OpenMMHandle`.
            forcefield: Force field to apply.  When *structure* is not an
                :class:`OpenMMHandle`, this is used to build the OpenMM
                system.  When *structure* is an existing
                :class:`OpenMMHandle`, providing a non-None *forcefield*
                updates the per-term parameters of that handle; if
                *forcefield* is ``None``, the handle's current parameters
                are used unchanged.

        Returns:
            Path: The resolved output path.

        """
        handle = self._prepare_handle(structure, forcefield)
        xml_string = mm.XmlSerializer.serialize(handle.system)
        output = Path(path)
        output.write_text(xml_string, encoding="utf-8")
        return output

    @staticmethod
    def load_system_xml(path: str | Path) -> object:
        """Deserialize an OpenMM System from XML.

        Args:
            path: Path to the XML file.

        Returns:
            object: An ``openmm.System`` object.

        """
        _ensure_openmm()
        xml_string = Path(path).read_text(encoding="utf-8")
        return mm.XmlSerializer.deserialize(xml_string)

    def _prepare_handle(
        self, structure: Q2MMMolecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None
    ) -> OpenMMHandle:
        """Get or create an :class:`OpenMMHandle`.

        If *structure* is already an :class:`OpenMMHandle`, optionally update
        its parameters.  Otherwise, build a new handle.

        Args:
            structure (Q2MMMolecule | str | Path | OpenMMHandle): A
                :class:`Q2MMMolecule`, XYZ path, or existing
                :class:`OpenMMHandle`.
            forcefield: Force field to apply (used for creation or update).

        Returns:
            OpenMMHandle: Ready-to-use handle.

        """
        if isinstance(structure, OpenMMHandle):
            handle = structure
            if forcefield is not None:
                self.update_forcefield(handle, forcefield)
            return handle
        return self.create_context(structure, forcefield)

    def energy(
        self, structure: Q2MMMolecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None
    ) -> float:
        """Calculate MM energy in kcal/mol.

        Args:
            structure (Q2MMMolecule | str | Path | OpenMMHandle): Molecule,
                XYZ path, or :class:`OpenMMHandle`.
            forcefield: Force field to apply. Auto-generated if ``None``.

        Returns:
            float: Potential energy in kcal/mol.

        """
        handle = self._prepare_handle(structure, forcefield)
        state = handle.context.getState(getEnergy=True)
        return float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))

    def minimize(
        self,
        structure: Q2MMMolecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
        tolerance: float = 1.0,
        max_iterations: int = 200,
    ) -> tuple:
        """Energy-minimize structure using L-BFGS.

        Args:
            structure (Q2MMMolecule | str | Path | OpenMMHandle): Molecule,
                XYZ path, or :class:`OpenMMHandle`.
            forcefield: Force field to apply. Auto-generated if ``None``.
            tolerance: Energy convergence tolerance in kJ/mol.
            max_iterations: Maximum minimization steps.

        Returns:
            tuple[float, list[str], np.ndarray]: ``(energy, atoms, coords)``
                where energy is in kcal/mol and coords are in Å.

        """
        handle = self._prepare_handle(structure, forcefield)
        mm.LocalEnergyMinimizer.minimize(handle.context, tolerance, max_iterations)
        state = handle.context.getState(getEnergy=True, getPositions=True)
        coords = np.array(state.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
        handle.molecule.geometry = coords
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        return energy, list(handle.molecule.symbols), coords

    def hessian(
        self,
        structure: Q2MMMolecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
        step: float = 1.0e-4,
    ) -> np.ndarray:
        """Finite-difference Hessian in canonical units (Hartree/Bohr²).

        Internally computed in kJ/mol/nm² (OpenMM native) and converted
        to Hartree/Bohr² before returning, matching the canonical unit
        contract defined in :class:`~q2mm.backends.base.MMEngine`.

        Args:
            structure (Q2MMMolecule | str | Path | OpenMMHandle): Molecule,
                XYZ path, or :class:`OpenMMHandle`.
            forcefield: Force field to apply. Auto-generated if ``None``.
            step: Finite-difference displacement in nm.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian in Hartree/Bohr².

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

    def frequencies(
        self, structure: Q2MMMolecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None
    ) -> list[float]:
        """Approximate harmonic frequencies in cm⁻¹ from the numerical Hessian.

        Args:
            structure (Q2MMMolecule | str | Path | OpenMMHandle): Molecule,
                XYZ path, or :class:`OpenMMHandle`.
            forcefield: Force field to apply. Auto-generated if ``None``.

        Returns:
            list[float]: Vibrational frequencies in cm⁻¹.

        """
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
