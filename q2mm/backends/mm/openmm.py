"""OpenMM molecular mechanics backend.

Provides a full-featured MM engine using OpenMM for energy, minimization,
Hessian, and frequency calculations.  Supports both harmonic and MM3
functional forms with runtime parameter updates via :class:`OpenMMHandle`.
"""

from __future__ import annotations

import copy
import logging
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from q2mm.backends.base import MMEngine
from q2mm.backends.registry import register_mm
from q2mm.constants import (
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    MM3_ANGLE_C6,
    MASSES,
)
from q2mm.models.units import (
    RAD_TO_DEG,
    ang_to_nm,
    canonical_to_openmm_angle_k,
    canonical_to_openmm_bond_k,
    canonical_to_openmm_bond_k_nm,
    canonical_to_openmm_epsilon,
    canonical_to_openmm_harmonic_angle_k,
    canonical_to_openmm_harmonic_bond_k,
    canonical_to_openmm_torsion_k,
    hessian_kjmolnm2_to_au,
    kj_to_kcal,
    rmin_half_to_sigma_nm,
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


@dataclass
class _DiffHandle:
    """Handle for differentiable OpenMM evaluation with global parameters.

    Unlike :class:`OpenMMHandle`, this uses global parameters so that
    ``addEnergyParameterDerivative()`` can compute exact dE/d(param).

    Attributes:
        integrator: The ``openmm.Integrator`` used by the context.  Must
            remain alive for the lifetime of the context to prevent
            use-after-free.
        context: The ``openmm.Context`` for energy evaluation.
        param_names: Global parameter names registered for derivatives.
        param_vector_indices: Indices into the flat param vector.
        grad_unit_factors: Chain-rule conversion factors (dp_openmm/dp_canonical).
        functional_form: The functional form used when the handle was created.

    """

    integrator: object
    context: object
    param_names: list[str]
    param_vector_indices: list[int]
    grad_unit_factors: list[float]
    functional_form: FunctionalForm


def _ensure_openmm() -> None:
    """Raise ``ImportError`` if OpenMM is not installed.

    Raises:
        ImportError: If the ``openmm`` package cannot be imported.

    """
    if not _HAS_OPENMM:
        raise ImportError('OpenMM is not installed. Install with `pip install openmm` or `pip install -e ".[openmm]"`.')


_PLATFORM_PRIORITY = ("CUDA", "OpenCL", "CPU", "Reference")


def detect_best_platform() -> str:
    """Return the name of the fastest available OpenMM platform.

    Platform preference order: CUDA > OpenCL > CPU > Reference.

    Returns:
        str: Name of the best available platform.

    Raises:
        ImportError: If OpenMM is not installed.

    """
    _ensure_openmm()
    available = {mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
    for name in _PLATFORM_PRIORITY:
        if name in available:
            return name
    # Fallback — shouldn't happen since OpenMM always has Reference
    return mm.Platform.getPlatform(0).getName()  # pragma: no cover


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
    return canonical_to_openmm_bond_k(force_constant)


def _angle_k_to_openmm(force_constant: float) -> float:
    """Convert canonical angle force constant (kcal/mol/rad²) to kJ/mol/rad².

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad².

    """
    return canonical_to_openmm_angle_k(force_constant)


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
    return canonical_to_openmm_harmonic_bond_k(force_constant)


def _angle_k_to_harmonic(force_constant: float) -> float:
    """Canonical angle k (kcal/mol/rad²) → HarmonicAngleForce k (kJ/mol/rad²).

    OpenMM's HarmonicAngleForce uses E = ½·k·(θ−θ₀)² while the canonical
    convention is E = k·(θ−θ₀)², so k_openmm = 2·k.  Convert kcal→kJ.

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad² with the ½ convention.

    """
    return canonical_to_openmm_harmonic_angle_k(force_constant)


def _vdw_sigma_nm(radius: float) -> float:
    """Convert Rmin/2 (Å) to LJ sigma (nm) for standard 12-6 NonbondedForce.

    Args:
        radius: Van der Waals radius (Rmin/2) in Å.

    Returns:
        float: LJ sigma in nm.

    """
    return rmin_half_to_sigma_nm(radius)


def _vdw_radius_to_openmm(radius: float) -> float:
    """Convert vdW radius from Å to nm for CustomNonbondedForce.

    Args:
        radius: Van der Waals radius in Å.

    Returns:
        float: Van der Waals radius in nm.

    """
    return ang_to_nm(radius)


def _vdw_epsilon_to_openmm(epsilon: float) -> float:
    """Convert vdW epsilon from kcal/mol to kJ/mol.

    Args:
        epsilon: Well depth in kcal/mol.

    Returns:
        float: Well depth in kJ/mol.

    """
    return canonical_to_openmm_epsilon(epsilon)


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
    return canonical_to_openmm_torsion_k(force_constant)


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

    def __init__(
        self,
        platform_name: str | None = None,
        precision: str | None = None,
    ) -> None:
        """Initialize the OpenMM engine.

        Args:
            platform_name: OpenMM platform to use (e.g. ``"CPU"``,
                ``"CUDA"``, ``"OpenCL"``).  When ``None``, the fastest
                available platform is auto-detected
                (CUDA > OpenCL > CPU > Reference).
            precision: Floating-point precision for GPU platforms
                (``"single"``, ``"mixed"``, or ``"double"``).  Ignored
                for CPU/Reference platforms.  Defaults to ``"mixed"``
                when a GPU platform is selected.

        Raises:
            ImportError: If OpenMM is not installed.

        """
        _ensure_openmm()
        if platform_name is None:
            platform_name = detect_best_platform()
        self._platform_name = platform_name

        _VALID_PRECISIONS = {"single", "mixed", "double"}
        if precision is not None:
            precision = precision.strip().lower()
            if precision not in _VALID_PRECISIONS:
                raise ValueError(
                    f"Invalid precision {precision!r}. Allowed values: {', '.join(sorted(_VALID_PRECISIONS))}."
                )
        self._precision = precision
        logger.info("OpenMM platform: %s", self._platform_name)

    @property
    def name(self) -> str:
        """Human-readable engine name including the active platform.

        Returns:
            str: e.g. ``"OpenMM (CUDA)"``.

        """
        return f"OpenMM ({self._platform_name})"

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

    def supports_analytical_gradients(self) -> bool:
        """Whether this engine provides analytical parameter gradients.

        Both HARMONIC and MM3 functional forms use ``CustomBondForce``,
        ``CustomAngleForce``, and ``CustomTorsionForce`` with global
        parameters, so ``addEnergyParameterDerivative()`` provides exact
        dE/d(param) for bond, angle, and torsion parameters.

        vdW parameters use per-particle values and are supplemented
        with central finite differences inside ``energy_and_param_grad``,
        so the returned gradient is always complete.

        Returns:
            bool: Always ``True``.

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
        platform = mm.Platform.getPlatformByName(self._platform_name)
        # Set precision for GPU platforms (CUDA/OpenCL).
        gpu_platforms = {"CUDA", "OpenCL"}
        if self._platform_name in gpu_platforms:
            precision = self._precision or "mixed"
            prop_key = "CudaPrecision" if self._platform_name == "CUDA" else "OpenCLPrecision"
            context = mm.Context(system, integrator, platform, {prop_key: precision})
        else:
            context = mm.Context(system, integrator, platform)
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
                    ang_to_nm(param.equilibrium),
                    _bond_k_to_harmonic(param.force_constant),
                )
            else:
                force_index = bond_force.addBond(
                    bond.atom_i,
                    bond.atom_j,
                    [_bond_k_to_openmm(param.force_constant), ang_to_nm(param.equilibrium)],
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

        # --- Assign improper torsion parameters ---
        for imp_torsion in molecule.improper_torsions:
            params = _match_torsions(
                forcefield,
                imp_torsion.element_quad,
                env_id=imp_torsion.env_id,
                ff_row=imp_torsion.ff_row,
                is_improper=True,
            )
            for param in params:
                force_index = torsion_force.addTorsion(
                    imp_torsion.atom_i,
                    imp_torsion.atom_j,
                    imp_torsion.atom_k,
                    imp_torsion.atom_l,
                    param.periodicity,
                    np.deg2rad(float(param.phase)),
                    _torsion_k_to_openmm(param.force_constant),
                )
                torsion_terms.append(
                    _TorsionTerm(
                        force_index=force_index,
                        atom_i=imp_torsion.atom_i,
                        atom_j=imp_torsion.atom_j,
                        atom_k=imp_torsion.atom_k,
                        atom_l=imp_torsion.atom_l,
                        elements=imp_torsion.element_quad,
                        periodicity=param.periodicity,
                        env_id=imp_torsion.env_id,
                        ff_row=param.ff_row,
                        is_improper=True,
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
                        ang_to_nm(param.equilibrium),
                        _bond_k_to_harmonic(param.force_constant),
                    )
                else:
                    handle.bond_force.setBondParameters(
                        term.force_index,
                        term.atom_i,
                        term.atom_j,
                        [_bond_k_to_openmm(param.force_constant), ang_to_nm(param.equilibrium)],
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

    # ------------------------------------------------------------------
    # Analytical parameter gradients via addEnergyParameterDerivative
    # ------------------------------------------------------------------

    def _create_diff_handle(self, molecule: Q2MMMolecule, forcefield: ForceField) -> _DiffHandle:
        """Build an OpenMM system with global parameters for analytical gradients.

        Each unique FF parameter becomes a named global parameter on the
        appropriate ``CustomForce``.  ``addEnergyParameterDerivative()`` is
        called for every global parameter so that
        ``getState(getParameterDerivatives=True)`` returns exact dE/dp.

        Supports both HARMONIC and MM3 functional forms.  Both use
        ``CustomBondForce`` / ``CustomAngleForce`` (rather than the built-in
        ``HarmonicBondForce`` etc.) so that ``addEnergyParameterDerivative``
        is available.

        Args:
            molecule: Molecular structure.
            forcefield: Force field with canonical-unit parameters.

        Returns:
            _DiffHandle with the context and parameter mapping.

        """
        self._validate_forcefield(forcefield)
        ff_form = forcefield.functional_form or FunctionalForm.MM3
        use_harmonic = ff_form == FunctionalForm.HARMONIC

        system = mm.System()
        for symbol in molecule.symbols:
            system.addParticle(MASSES[symbol] * unit.dalton)

        param_names: list[str] = []
        param_vector_indices: list[int] = []
        grad_unit_factors: list[float] = []
        param_vector = forcefield.get_param_vector()
        pv_idx = 0  # tracks position in flat param vector

        # --- Bonds: each bond param contributes (k, r0) ---
        bond_global_map: dict[int, tuple[str, str]] = {}
        # Conversion factors differ: HARMONIC CustomBondForce uses
        # kJ/mol/nm² directly; MM3 uses kJ/mol/Å² (expression handles nm→Å).
        bond_k_factor = canonical_to_openmm_bond_k_nm(1.0) if use_harmonic else _bond_k_to_openmm(1.0)
        for bp_idx, bp in enumerate(forcefield.bonds):
            k_name = f"bond_k_{bp_idx}"
            r0_name = f"bond_r0_{bp_idx}"
            bond_global_map[bp_idx] = (k_name, r0_name)
            param_names.extend([k_name, r0_name])
            param_vector_indices.extend([pv_idx, pv_idx + 1])
            grad_unit_factors.extend(
                [
                    bond_k_factor,  # dk_openmm/dk_canonical
                    0.1,  # dr0_openmm/dr0_canonical (Å → nm)
                ]
            )
            pv_idx += 2

        # Build one CustomBondForce per bond-param type so each force has
        # just two global parameters (k, r0) — avoids the select() limit.
        for bp_idx, bp in enumerate(forcefield.bonds):
            k_name, r0_name = bond_global_map[bp_idx]
            k_val = (
                canonical_to_openmm_bond_k_nm(bp.force_constant)
                if use_harmonic
                else _bond_k_to_openmm(bp.force_constant)
            )
            if use_harmonic:
                expr = f"{k_name}*(r-{r0_name})^2"
            else:
                expr = f"{k_name}*dr10^2*(1-c3*dr10+c4*dr10^2);dr10=10*(r-{r0_name});c3={MM3_BOND_C3};c4={MM3_BOND_C4}"
            bf = mm.CustomBondForce(expr)
            bf.setForceGroup(0)
            bf.addGlobalParameter(k_name, k_val)
            bf.addGlobalParameter(r0_name, ang_to_nm(bp.equilibrium))
            bf.addEnergyParameterDerivative(k_name)
            bf.addEnergyParameterDerivative(r0_name)

            for bond in molecule.bonds:
                param = _match_bond(
                    forcefield,
                    bond.elements,
                    env_id=bond.env_id,
                    ff_row=bond.ff_row,
                )
                if param is bp:
                    bf.addBond(bond.atom_i, bond.atom_j)
            system.addForce(bf)

        # --- Angles: each angle param contributes (k, theta0) ---
        angle_global_map: dict[int, tuple[str, str]] = {}
        # CustomAngleForce uses E=k·(θ−θ₀)² (no ½), same conversion for both forms.
        angle_k_factor = _angle_k_to_openmm(1.0)
        for ap_idx, ap in enumerate(forcefield.angles):
            k_name = f"angle_k_{ap_idx}"
            t0_name = f"angle_t0_{ap_idx}"
            angle_global_map[ap_idx] = (k_name, t0_name)
            param_names.extend([k_name, t0_name])
            param_vector_indices.extend([pv_idx, pv_idx + 1])
            grad_unit_factors.extend(
                [
                    angle_k_factor,  # dk_openmm/dk_canonical
                    np.deg2rad(1.0),  # dtheta0_openmm/dtheta0_canonical (deg → rad)
                ]
            )
            pv_idx += 2

        # Build one CustomAngleForce per angle-param type.
        for ap_idx, ap in enumerate(forcefield.angles):
            k_name, t0_name = angle_global_map[ap_idx]
            k_val = _angle_k_to_openmm(ap.force_constant)
            if use_harmonic:
                expr = f"{k_name}*(theta-{t0_name})^2"
            else:
                expr = (
                    f"{k_name}*(theta-{t0_name})^2*("
                    f"1+a3*((theta-{t0_name})*deg)"
                    f"+a4*((theta-{t0_name})*deg)^2"
                    f"+a5*((theta-{t0_name})*deg)^3"
                    f"+a6*((theta-{t0_name})*deg)^4"
                    f");"
                    f"a3={MM3_ANGLE_C3};"
                    f"a4={MM3_ANGLE_C4};"
                    f"a5={MM3_ANGLE_C5};"
                    f"a6={MM3_ANGLE_C6};"
                    f"deg={RAD_TO_DEG}"
                )
            af = mm.CustomAngleForce(expr)
            af.setForceGroup(1)
            af.addGlobalParameter(k_name, k_val)
            af.addGlobalParameter(t0_name, np.deg2rad(float(ap.equilibrium)))
            af.addEnergyParameterDerivative(k_name)
            af.addEnergyParameterDerivative(t0_name)

            for angle in molecule.angles:
                param = _match_angle(
                    forcefield,
                    angle.elements,
                    env_id=angle.env_id,
                    ff_row=angle.ff_row,
                )
                if param is ap:
                    af.addAngle(angle.atom_i, angle.atom_j, angle.atom_k)
            system.addForce(af)

        # --- Torsions: each torsion param (proper and improper) contributes (k,) ---
        # Use CustomTorsionForce with global parameters so that
        # addEnergyParameterDerivative() provides exact dE/dk.
        # One CustomTorsionForce per (torsion_param, periodicity) to keep
        # each force object's global params small — same pattern as bonds/angles.
        torsion_global_map: dict[int, str] = {}
        torsion_k_factor = _torsion_k_to_openmm(1.0)
        for tp_idx, tp in enumerate(forcefield.torsions):
            k_name = f"torsion_k_{tp_idx}"
            torsion_global_map[tp_idx] = k_name
            param_names.append(k_name)
            param_vector_indices.append(pv_idx)
            grad_unit_factors.append(torsion_k_factor)
            pv_idx += 1

        # Build one CustomTorsionForce per torsion param type.
        for tp_idx, tp in enumerate(forcefield.torsions):
            k_name = torsion_global_map[tp_idx]
            k_val = _torsion_k_to_openmm(tp.force_constant)
            phase_rad = np.deg2rad(float(tp.phase))
            n = tp.periodicity
            # Periodic torsion: E = k·(1 + cos(n·θ − φ))
            expr = f"{k_name}*(1+cos({n}*theta-{phase_rad:.15g}))"
            tf = mm.CustomTorsionForce(expr)
            tf.setForceGroup(1)
            tf.addGlobalParameter(k_name, k_val)
            tf.addEnergyParameterDerivative(k_name)

            # Match proper torsions from molecule topology
            if not tp.is_improper:
                for torsion in molecule.torsions:
                    params = _match_torsions(
                        forcefield,
                        torsion.element_quad,
                        env_id=torsion.env_id,
                        ff_row=torsion.ff_row,
                        is_improper=False,
                    )
                    for param in params:
                        if param is tp:
                            tf.addTorsion(
                                torsion.atom_i,
                                torsion.atom_j,
                                torsion.atom_k,
                                torsion.atom_l,
                                [],
                            )
            else:
                # Match improper torsions from trigonal centres
                for imp in molecule.improper_torsions:
                    params = _match_torsions(
                        forcefield,
                        imp.element_quad,
                        env_id=imp.env_id,
                        ff_row=imp.ff_row,
                        is_improper=True,
                    )
                    for param in params:
                        if param is tp:
                            tf.addTorsion(
                                imp.atom_i,
                                imp.atom_j,
                                imp.atom_k,
                                imp.atom_l,
                                [],
                            )
            system.addForce(tf)

        # --- vdW: advance pv_idx past vdW params ---
        # vdW uses per-particle parameters (no global-param derivatives).
        # Gradients are computed via finite differences in energy_and_param_grad().
        pv_idx += 2 * len(forcefield.vdws)

        if forcefield.vdws:
            if use_harmonic:
                # HARMONIC: standard NonbondedForce with LJ 12-6, no charges
                vdw_force = mm.NonbondedForce()
                vdw_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                for symbol, atom_type in zip(molecule.symbols, molecule.atom_types, strict=False):
                    param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
                    if param is None:
                        raise ValueError(f"Missing vdW parameter for {atom_type or symbol}.")
                    vdw_force.addParticle(0.0, _vdw_sigma_nm(param.radius), _vdw_epsilon_to_openmm(param.epsilon))

                # Exclude 1-2 and 1-3; scale 1-4 (AMBER scnb=2.0)
                excluded_12: set[tuple[int, int]] = set()
                for bond in molecule.bonds:
                    excluded_12.add((min(bond.atom_i, bond.atom_j), max(bond.atom_i, bond.atom_j)))

                excluded_13: set[tuple[int, int]] = set()
                for angle in molecule.angles:
                    excluded_13.add((min(angle.atom_i, angle.atom_k), max(angle.atom_i, angle.atom_k)))
                excluded_13 -= excluded_12

                neighbors: dict[int, set[int]] = {}
                for bond in molecule.bonds:
                    neighbors.setdefault(bond.atom_i, set()).add(bond.atom_j)
                    neighbors.setdefault(bond.atom_j, set()).add(bond.atom_i)

                pairs_14: set[tuple[int, int]] = set()
                for angle in molecule.angles:
                    for nb in neighbors.get(angle.atom_i, ()):
                        if nb != angle.atom_j and nb != angle.atom_k:
                            pairs_14.add((min(nb, angle.atom_k), max(nb, angle.atom_k)))
                    for nb in neighbors.get(angle.atom_k, ()):
                        if nb != angle.atom_j and nb != angle.atom_i:
                            pairs_14.add((min(nb, angle.atom_i), max(nb, angle.atom_i)))
                pairs_14 -= excluded_12
                pairs_14 -= excluded_13

                for p1, p2 in sorted(excluded_12 | excluded_13):
                    vdw_force.addException(p1, p2, 0.0, 1.0, 0.0)

                SCNB = 2.0
                for p1, p2 in sorted(pairs_14):
                    _, sig1, eps1 = vdw_force.getParticleParameters(p1)
                    _, sig2, eps2 = vdw_force.getParticleParameters(p2)
                    sig_14 = 0.5 * (sig1 + sig2)
                    eps_14 = (eps1 * eps2) ** 0.5 / SCNB
                    vdw_force.addException(p1, p2, 0.0, sig_14, eps_14)
            else:
                # MM3: Buckingham exp-6 with per-particle params
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
                for symbol, atom_type in zip(molecule.symbols, molecule.atom_types, strict=False):
                    param = _match_vdw(forcefield, atom_type=atom_type, element=symbol)
                    if param is None:
                        raise ValueError(f"Missing vdW parameter for {atom_type or symbol}.")
                    vdw_force.addParticle([_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)])
                vdw_force.createExclusionsFromBonds([(b.atom_i, b.atom_j) for b in molecule.bonds], 2)

            system.addForce(vdw_force)

        integrator, context = self._create_context(system)
        context.setPositions(self._positions(molecule))

        return _DiffHandle(
            integrator=integrator,
            context=context,
            param_names=param_names,
            param_vector_indices=param_vector_indices,
            grad_unit_factors=grad_unit_factors,
            functional_form=forcefield.functional_form,
        )

    def energy_and_param_grad(self, structure: Q2MMMolecule, forcefield: ForceField) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. FF parameters.

        Uses OpenMM's ``addEnergyParameterDerivative()`` on ``CustomForce``
        objects to get exact dE/d(param) for bond, angle, and torsion
        parameters.  vdW parameters use per-particle values that cannot
        be differentiated via global parameters, so their gradients are
        computed via central finite differences automatically.

        Args:
            structure (Q2MMMolecule): Molecular structure.
            forcefield (ForceField): Force field parameters.

        Returns:
            tuple[float, np.ndarray]: ``(energy, grad)`` where ``energy``
                is in kcal/mol and ``grad`` has the same length as
                ``forcefield.get_param_vector()``.

        """
        molecule = _as_molecule(structure)
        diff = self._create_diff_handle(molecule, forcefield)

        state = diff.context.getState(getEnergy=True, getParameterDerivatives=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        derivs = state.getEnergyParameterDerivatives()

        param_vector = forcefield.get_param_vector()
        grad = np.zeros(len(param_vector))

        for name, pv_idx, unit_factor in zip(
            diff.param_names, diff.param_vector_indices, diff.grad_unit_factors, strict=True
        ):
            deriv_openmm = derivs[name]  # dE_kJ/dp_openmm
            grad[pv_idx] = kj_to_kcal(deriv_openmm * unit_factor)

        # vdW parameters use per-particle values without global-parameter
        # derivatives.  Supplement with central finite differences so the
        # returned gradient is complete.
        if forcefield.vdws:
            vdw_start = (
                2 * len(forcefield.bonds)
                + 2 * len(forcefield.angles)
                + len(forcefield.torsions)
            )
            vdw_end = vdw_start + 2 * len(forcefield.vdws)
            step = 1e-4
            for i in range(vdw_start, vdw_end):
                pv_plus = param_vector.copy()
                pv_plus[i] += step
                pv_minus = param_vector.copy()
                pv_minus[i] -= step
                e_plus = self.energy(molecule, forcefield.with_params(pv_plus))
                e_minus = self.energy(molecule, forcefield.with_params(pv_minus))
                grad[i] = (e_plus - e_minus) / (2.0 * step)

        return energy, grad

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
        return hessian_symmetric * hessian_kjmolnm2_to_au(1.0)

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
        from q2mm.models.hessian import hessian_to_frequencies

        handle = self._prepare_handle(structure, forcefield)
        hessian_au = self.hessian(handle)  # Hartree/Bohr²
        return hessian_to_frequencies(hessian_au, list(handle.molecule.symbols))
