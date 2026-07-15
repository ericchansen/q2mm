"""OpenMM molecular mechanics backend.

Provides a full-featured MM engine using OpenMM for energy, minimization,
Hessian, and frequency calculations.  Supports both harmonic and MM3
functional forms with runtime parameter updates via :class:`OpenMMHandle`.
"""

from __future__ import annotations

import logging
from typing import Any
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

from q2mm.backends.base import MMEngine
from q2mm.backends.registry import register_mm
from q2mm.backends.mm._openmm_terms import (
    _AngleTerm,
    _BondTerm,
    _CmapTerm,
    _Exception14,
    _TorsionTerm,
    _UBTerm,
    _VdwTerm,
)
from q2mm.backends.mm._openmm_units import (
    _angle_k_to_harmonic,
    _angle_k_to_openmm,
    _bond_k_to_harmonic,
    _bond_k_to_openmm,
    _vdw_epsilon_to_openmm,
    _vdw_radius_to_openmm,
    _vdw_sigma_nm,
)
from q2mm.constants import (
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    MM3_ANGLE_C6,
    MM3_VDW_A,
    MM3_VDW_B,
    MM3_VDW_C,
    MASSES,
)
from q2mm.models.units import (
    KCAL_TO_KJ,
    RAD_TO_DEG,
    ang_to_nm,
    canonical_to_openmm_bond_k_nm,
    canonical_to_openmm_torsion_k,
    hessian_kjmolnm2_to_au,
    kj_to_kcal,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, TorsionParam, VdwParam
from q2mm.models.molecule import Molecule

try:
    from openmm import openmm as mm
    from openmm import unit

    _HAS_OPENMM = True
except ImportError:  # pragma: no cover - exercised when OpenMM is not installed
    mm = None
    unit = None
    _HAS_OPENMM = False


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
        ub_force: The OpenMM HarmonicBondForce for Urey-Bradley terms, or ``None``.
        cmap_force: The OpenMM CMAPTorsionForce, or ``None`` if no CMAP terms.
        bond_terms: Mapping of molecule bonds to force indices.
        angle_terms: Mapping of molecule angles to force indices.
        torsion_terms: Mapping of molecule torsions to force indices.
        vdw_terms: Mapping of atoms to vdW particle indices.
        ub_terms: Mapping of Urey-Bradley 1-3 pairs to force indices.
        cmap_terms: Mapping of CMAP corrections to force indices.
        exceptions_14: 1-4 nonbonded exceptions (harmonic form only).
        functional_form: The functional form used when the handle was created.

    """

    molecule: Molecule
    system: object
    integrator: object
    context: object
    bond_force: object | None
    angle_force: object | None
    torsion_force: object | None
    vdw_force: object | None
    ub_force: object | None
    cmap_force: object | None
    bond_terms: list[_BondTerm]
    angle_terms: list[_AngleTerm]
    torsion_terms: list[_TorsionTerm]
    vdw_terms: list[_VdwTerm]
    ub_terms: list[_UBTerm] = field(default_factory=list)
    cmap_terms: list[_CmapTerm] = field(default_factory=list)
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

    If the ``OPENMM_DEFAULT_PLATFORM`` environment variable is set, its
    value is returned directly (no validation against installed
    platforms).  This allows test harnesses to force CPU-only execution.

    Otherwise, platform preference order: CUDA > OpenCL > CPU > Reference.

    Logs a warning when CUDA is unavailable and the function falls back
    to OpenCL on a system with an NVIDIA GPU — OpenCL on modern NVIDIA
    GPUs gives very poor utilisation (~14%).  The warning is suppressed
    on non-NVIDIA systems where OpenCL may be the intended backend.

    Returns:
        str: Name of the best available platform.

    Raises:
        ImportError: If OpenMM is not installed.

    """
    _ensure_openmm()
    import os

    env_platform = os.environ.get("OPENMM_DEFAULT_PLATFORM", "").strip()
    if env_platform:
        return env_platform
    available = {mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())}
    for name in _PLATFORM_PRIORITY:
        if name in available:
            if name == "OpenCL" and "CUDA" not in available:
                # Only warn on NVIDIA GPUs where CUDA should be available
                _nvidia_present = False
                try:
                    import subprocess

                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    _nvidia_present = result.returncode == 0 and bool(result.stdout.strip())
                except Exception:
                    pass
                if _nvidia_present:
                    logger.warning(
                        "CUDA platform not available, falling back to OpenCL. "
                        "GPU utilization will be poor (~14%%). "
                        "Consider installing OpenMM-CUDA-12 or using WSL2."
                    )
            return name
    # Fallback — shouldn't happen since OpenMM always has Reference
    return mm.Platform.getPlatform(0).getName()  # pragma: no cover


def _as_molecule(structure: Molecule | str | Path) -> Molecule:
    """Coerce *structure* to a :class:`Molecule`.

    Args:
        structure: A :class:`Molecule`, file path (``str`` or ``Path``),
            or any other type (which will raise ``TypeError``).

    Returns:
        Molecule: The coerced molecule object.

    Raises:
        TypeError: If *structure* is not a recognised type.

    """
    if isinstance(structure, Molecule):
        return structure
    if isinstance(structure, (str, Path)):
        from q2mm.io.xyz import load_xyz

        return load_xyz(structure)
    raise TypeError("OpenMMEngine expects a Molecule or path to an XYZ file.")


def _coerce_forcefield(forcefield: ForceField | None, molecule: Molecule) -> ForceField:
    """Return *forcefield* or create a default one from *molecule*.

    Args:
        forcefield: An explicit force field, or ``None`` to auto-generate.
        molecule: Molecule used when generating a default force field.

    Returns:
        ForceField: The provided or auto-generated force field.

    """
    if forcefield is not None:
        return forcefield
    return ForceField.create_for_molecule(molecule, functional_form=FunctionalForm.MM3)


def _collect_bond_assignments(
    molecule: Molecule,
    forcefield: ForceField,
) -> list[tuple[Any, BondParam]]:
    """Match each molecule bond to a force field bond parameter.

    Returns a list of ``(detected_bond, matched_param)`` pairs, preserving
    molecule traversal order.  Bonds with no matching parameter are skipped.
    """
    assignments: list[tuple[Any, BondParam]] = []
    for bond in molecule.bonds:
        param = forcefield.match_bond(
            bond.elements,
            env_id=bond.env_id,
            ff_row=bond.ff_row,
            bond_order=getattr(bond, "bond_order", ""),
            bond_length=bond.length,
        )
        if param is not None:
            assignments.append((bond, param))
    return assignments


def _collect_angle_assignments(
    molecule: Molecule,
    forcefield: ForceField,
) -> list[tuple[Any, AngleParam]]:
    """Match each molecule angle to a force field angle parameter.

    Returns a list of ``(detected_angle, matched_param)`` pairs, preserving
    molecule traversal order.  Angles with no matching parameter are skipped.
    """
    assignments: list[tuple[Any, AngleParam]] = []
    for angle in molecule.angles:
        param = forcefield.match_angle(angle.elements, env_id=angle.env_id, ff_row=angle.ff_row)
        if param is not None:
            assignments.append((angle, param))
    return assignments


def _collect_torsion_assignments(
    molecule: Molecule,
    forcefield: ForceField,
    *,
    is_improper: bool,
) -> list[tuple[Any, TorsionParam]]:
    """Match each molecule torsion to its force field torsion parameter(s).

    Each torsion may match multiple parameters (one per periodicity), so
    the result is flattened: one entry per ``(torsion, param)`` pair.

    Args:
        molecule: Molecular structure.
        forcefield: Force field to match against.
        is_improper: If ``True``, match improper torsions; otherwise proper.

    """
    source = molecule.improper_torsions if is_improper else molecule.torsions
    assignments: list[tuple[Any, TorsionParam]] = []
    for torsion in source:
        params = forcefield.match_torsion(
            torsion.element_quad,
            env_id=torsion.env_id,
            ff_row=torsion.ff_row,
            is_improper=is_improper,
        )
        for param in params:
            assignments.append((torsion, param))
    return assignments


def _collect_vdw_assignments(
    molecule: Molecule,
    forcefield: ForceField,
) -> list[tuple[int, str, str, VdwParam]]:
    """Match each atom to a force field vdW parameter.

    Returns a list of ``(atom_index, symbol, atom_type, param)`` tuples.

    Raises:
        ValueError: If any atom has no matching vdW parameter.

    """
    assignments: list[tuple[int, str, str, VdwParam]] = []
    for atom_index, (symbol, atom_type) in enumerate(zip(molecule.symbols, molecule.atom_types, strict=False)):
        param = forcefield.match_vdw(atom_type=atom_type, element=symbol)
        if param is None:
            raise ValueError(f"Missing vdW parameter for atom {atom_index + 1} ({atom_type or symbol}).")
        assignments.append((atom_index, symbol, atom_type, param))
    return assignments


def _index_by_param_id(
    assignments: list[tuple[Any, Any]],
) -> dict[int, list[Any]]:
    """Build a reverse index from ``id(param)`` to topology items.

    Given a list of ``(topology_item, param)`` pairs, returns a dict mapping
    ``id(param)`` → list of topology items that matched that param.  Preserves
    the original traversal order within each group.
    """
    by_param: dict[int, list[Any]] = {}
    for item, param in assignments:
        key = id(param)
        if key not in by_param:
            by_param[key] = []
        by_param[key].append(item)
    return by_param


def _build_harmonic_exclusions(
    molecule: Molecule,
    vdw_force: Any,
) -> list[_Exception14]:
    """Add 1-2/1-3 exclusions and scaled 1-4 exceptions to a NonbondedForce.

    Follows AMBER conventions: full exclusion for 1-2 and 1-3 pairs,
    ``scnb=2.0`` scaling for 1-4 pairs (epsilon divided by 2).

    Args:
        molecule: Molecular structure for bond/angle topology.
        vdw_force: An ``openmm.NonbondedForce`` to modify in-place.

    Returns:
        List of :class:`_Exception14` records for later update.

    """
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
    exceptions_14: list[_Exception14] = []
    for p1, p2 in sorted(pairs_14):
        sig1, eps1 = vdw_force.getParticleParameters(p1)[1:]
        sig2, eps2 = vdw_force.getParticleParameters(p2)[1:]
        sig_14 = 0.5 * (sig1 + sig2)
        eps_14 = (eps1 * eps2) ** 0.5 / SCNB
        exc_idx = vdw_force.addException(p1, p2, 0.0, sig_14, eps_14)
        exceptions_14.append(_Exception14(exception_index=exc_idx, particle_i=p1, particle_j=p2))

    return exceptions_14


def _build_atom_type_index(molecule: Molecule) -> dict[str, list[int]]:
    """Build a mapping from atom type to atom indices in the molecule.

    Args:
        molecule: Molecule to index.

    Returns:
        dict mapping atom type strings to lists of 0-based atom indices.

    """
    index: dict[str, list[int]] = {}
    for i, atype in enumerate(molecule.atom_types):
        index.setdefault(atype, []).append(i)
    return index


def _find_dihedral_atoms(
    type_to_indices: dict[str, list[int]],
    atom_types: tuple[str, str, str, str],
    molecule: Molecule,
) -> list[tuple[int, int, int, int]]:
    """Find all atom-index quadruples matching a dihedral type pattern.

    Only returns quadruples where consecutive atoms are bonded.
    Uses pre-enumerated torsions from the molecule when available
    (O(n_torsions) instead of O(n^4) Cartesian product).

    Args:
        type_to_indices: Atom type to index mapping.
        atom_types: Four atom types defining the dihedral.
        molecule: Molecule for bond connectivity and torsion enumeration.

    Returns:
        List of (i, j, k, l) atom index tuples.

    """
    results: list[tuple[int, int, int, int]] = []
    t1, t2, t3, t4 = atom_types
    atom_types_array = molecule.atom_types

    # Use pre-enumerated torsions (bonded 1-4 paths) when available
    for torsion in molecule.torsions:
        a, b, c, d = torsion.atom_i, torsion.atom_j, torsion.atom_k, torsion.atom_l
        if (
            atom_types_array[a] == t1
            and atom_types_array[b] == t2
            and atom_types_array[c] == t3
            and atom_types_array[d] == t4
        ):
            results.append((a, b, c, d))

    return results


def _build_cmap_force(molecule: Molecule, forcefield: ForceField) -> tuple[object | None, list[_CmapTerm]]:
    """Build a ``CMAPTorsionForce`` for the molecule's matching phi/psi pairs.

    Shared by ``create_context`` (scalar energy) and ``_create_diff_handle``
    (analytical-gradient) so both include identical CMAP energy.  CMAP grids
    carry no tunable parameters, so this contributes to the potential energy
    only — not to the parameter-gradient vector.

    Args:
        molecule: Molecular structure.
        forcefield: Force field, possibly carrying CMAP grids.

    Returns:
        ``(cmap_force, cmap_terms)``.  ``cmap_force`` is ``None`` when the
        force field has no CMAP grids or none of them match the molecule.

    """
    if not forcefield.has_cmap:
        return None, []

    cmap_force = mm.CMAPTorsionForce()
    type_to_indices = _build_atom_type_index(molecule)
    cmap_terms: list[_CmapTerm] = []

    for grid in forcefield.cmaps:
        # Add the 2D energy grid (convert kcal/mol → kJ/mol for OpenMM)
        energy_kj = [e * KCAL_TO_KJ for e in grid.energy]
        map_index = cmap_force.addMap(grid.resolution, energy_kj)

        phi_matches = _find_dihedral_atoms(type_to_indices, grid.atom_types_phi, molecule)
        psi_matches = _find_dihedral_atoms(type_to_indices, grid.atom_types_psi, molecule)

        # Pair phi/psi dihedrals sharing 3 overlapping atoms.
        psi_index: dict[tuple[int, ...], list[tuple[int, int, int, int]]] = {}
        for psi_atoms in psi_matches:
            key = tuple(psi_atoms[:3])
            psi_index.setdefault(key, []).append(psi_atoms)

        for phi_atoms in phi_matches:
            key = tuple(phi_atoms[1:])
            for psi_atoms in psi_index.get(key, ()):
                torsion_index = cmap_force.addTorsion(
                    map_index,
                    phi_atoms[0],
                    phi_atoms[1],
                    phi_atoms[2],
                    phi_atoms[3],
                    psi_atoms[0],
                    psi_atoms[1],
                    psi_atoms[2],
                    psi_atoms[3],
                )
                cmap_terms.append(
                    _CmapTerm(
                        torsion_index=torsion_index,
                        map_index=map_index,
                        phi_atoms=phi_atoms,
                        psi_atoms=psi_atoms,
                        phi_types=grid.atom_types_phi,
                        psi_types=grid.atom_types_psi,
                    )
                )

    if not cmap_terms:
        return None, []
    return cmap_force, cmap_terms


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
                available platform is auto-detected via
                :func:`detect_best_platform` (CUDA > OpenCL > CPU >
                Reference).  WSL2 is recommended for CUDA on modern
                GPUs (e.g. RTX 5090 Blackwell) when running on Windows
                hardware.
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

    @classmethod
    def deps_available(cls) -> bool:
        """Check if OpenMM is importable without platform detection."""
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

    def _positions(self, molecule: Molecule) -> Any:
        """Convert molecule geometry to OpenMM position array.

        Args:
            molecule: Molecule whose geometry to convert.

        Returns:
            Positions as an OpenMM ``Quantity`` in Å.

        """
        return np.asarray(molecule.geometry, dtype=float) * unit.angstrom

    def _create_context(self, system: Any, *, precision: str | None = None) -> tuple[Any, Any]:
        """Create an OpenMM integrator and context for *system*.

        Attempts to create a context on the engine's selected platform.
        If the platform is a GPU platform (CUDA or OpenCL) and context
        creation fails (e.g. PTX version mismatch, missing CUDA plugin,
        or unsupported GPU architecture), the method logs a warning and
        **silently falls back to CPU**. The fallback chain is:

        1. Try the selected platform (e.g. CUDA) with the configured
           precision.
        2. On failure, mutate ``self._platform_name`` to ``"CPU"`` and
           create a new context on the CPU platform.

        No exception is raised on GPU failure — callers should check
        ``self._platform_name`` after context creation if they need to
        know which platform is in use.

        Args:
            system: An ``openmm.System`` object.
            precision: Override GPU precision (``"single"``, ``"mixed"``,
                ``"double"``).  When ``None`` (default) uses the
                engine-level setting or ``"mixed"``.

        Returns:
            tuple: ``(integrator, context)`` pair.

        """
        integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = mm.Platform.getPlatformByName(self._platform_name)
        # Set precision for GPU platforms (CUDA/OpenCL).
        gpu_platforms = {"CUDA", "OpenCL"}
        if self._platform_name in gpu_platforms:
            precision = precision or self._precision or "mixed"
            _VALID_PRECISIONS = {"single", "mixed", "double"}
            if precision not in _VALID_PRECISIONS:
                raise ValueError(
                    f"Invalid precision {precision!r}. Expected one of: {', '.join(sorted(_VALID_PRECISIONS))}."
                )
            prop_key = "CudaPrecision" if self._platform_name == "CUDA" else "OpenCLPrecision"
            try:
                context = mm.Context(system, integrator, platform, {prop_key: precision})
            except mm.OpenMMException as e:
                import logging

                logging.getLogger(__name__).warning(
                    "%s platform failed (%s), falling back to CPU. "
                    "This often means the GPU architecture is not supported "
                    "by the installed OpenMM CUDA plugin.",
                    self._platform_name,
                    e,
                )
                self._platform_name = "CPU"
                integrator = mm.VerletIntegrator(1.0 * unit.femtoseconds)
                platform = mm.Platform.getPlatformByName("CPU")
                context = mm.Context(system, integrator, platform)
        else:
            context = mm.Context(system, integrator, platform)
        return integrator, context

    def create_context(
        self,
        structure: Molecule | str | Path,
        forcefield: ForceField | None = None,
        *,
        precision: str | None = None,
    ) -> OpenMMHandle:
        """Build an OpenMM system and context for a molecule.

        Creates force objects (bond, angle, vdW) matching the force field's
        functional form and assigns per-term parameters from *forcefield*.

        Args:
            structure (Molecule | str | Path): A
                :class:`~q2mm.models.molecule.Molecule` or path to an
                XYZ file.
            forcefield: Force field to apply. Auto-generated from the
                molecule if ``None``.
            precision: Override GPU precision (``"single"``, ``"mixed"``,
                ``"double"``).  When ``None`` (default) uses the
                engine-level setting.

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

        if forcefield.stretch_bends:
            raise NotImplementedError(
                "OpenMMEngine does not support stretch-bend cross terms. "
                "Use JaxEngine for force fields with stretch-bend parameters."
            )

        # Functional form is always explicit on the (possibly auto-generated,
        # always MM3-tagged by _coerce_forcefield) force field.
        ff_form = forcefield.functional_form
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
                f"step(r - rc) * epsilon*(-{MM3_VDW_C}*(rv/r)^6 + {MM3_VDW_A}*exp(-{MM3_VDW_B}*r/rv))"
                f" + step(rc - r) * epsilon*{MM3_VDW_A}*exp(-{MM3_VDW_B}*rc/rv) * (rc/r)^12;"
                "rc=0.34*rv;"
                "rv=radius1+radius2;"
                "epsilon=sqrt(epsilon1*epsilon2)"
            )
            vdw_force.addPerParticleParameter("radius")
            vdw_force.addPerParticleParameter("epsilon")
            vdw_force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)

        # --- Assign bond parameters ---
        bond_assignments = _collect_bond_assignments(molecule, forcefield)
        bond_terms: list[_BondTerm] = []
        for bond, param in bond_assignments:
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
        angle_assignments = _collect_angle_assignments(molecule, forcefield)
        angle_terms: list[_AngleTerm] = []
        for angle, param in angle_assignments:
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

        # --- Assign Urey-Bradley terms (1-3 distance for CHARMM angles) ---
        # Reuse precomputed angle assignments to avoid rematching.
        ub_force = None
        ub_terms: list[_UBTerm] = []
        for (angle, param), angle_term in zip(angle_assignments, angle_terms, strict=True):
            if param.ub_force_constant is None and param.ub_equilibrium is None:
                continue
            if param.ub_force_constant is None or param.ub_equilibrium is None:
                raise ValueError(
                    "Inconsistent Urey-Bradley parameters for angle "
                    f"{angle_term.elements} (env_id={angle_term.env_id}, ff_row={angle_term.ff_row}): "
                    "both 'ub_force_constant' and 'ub_equilibrium' must be set or both must be None."
                )
            if ub_force is None:
                ub_force = mm.HarmonicBondForce()
            force_index = ub_force.addBond(
                angle_term.atom_i,
                angle_term.atom_k,
                ang_to_nm(param.ub_equilibrium),
                _bond_k_to_harmonic(param.ub_force_constant),
            )
            ub_terms.append(
                _UBTerm(
                    force_index=force_index,
                    atom_i=angle_term.atom_i,
                    atom_k=angle_term.atom_k,
                    elements=angle_term.elements,
                    env_id=angle_term.env_id,
                    ff_row=angle_term.ff_row,
                )
            )

        # --- Assign proper and improper torsion parameters ---
        proper_assignments = _collect_torsion_assignments(molecule, forcefield, is_improper=False)
        improper_assignments = _collect_torsion_assignments(molecule, forcefield, is_improper=True)

        torsion_terms: list[_TorsionTerm] = []
        for torsion, param in proper_assignments:
            force_index = torsion_force.addTorsion(
                torsion.atom_i,
                torsion.atom_j,
                torsion.atom_k,
                torsion.atom_l,
                param.periodicity,
                np.deg2rad(float(param.phase)),
                canonical_to_openmm_torsion_k(param.force_constant),
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

        for imp_torsion, param in improper_assignments:
            force_index = torsion_force.addTorsion(
                imp_torsion.atom_i,
                imp_torsion.atom_j,
                imp_torsion.atom_k,
                imp_torsion.atom_l,
                param.periodicity,
                np.deg2rad(float(param.phase)),
                canonical_to_openmm_torsion_k(param.force_constant),
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
            vdw_assignments = _collect_vdw_assignments(molecule, forcefield)
            for atom_index, symbol, atom_type, param in vdw_assignments:
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
                exceptions_14 = _build_harmonic_exclusions(molecule, vdw_force)
            else:
                vdw_force.createExclusionsFromBonds([(bond.atom_i, bond.atom_j) for bond in molecule.bonds], 2)

        # --- Assign CMAP correction terms (CHARMM backbone corrections) ---
        cmap_force, cmap_terms = _build_cmap_force(molecule, forcefield)
        if forcefield.has_cmap:
            if cmap_terms:
                logger.info("Created %d CMAP correction term(s).", len(cmap_terms))
            else:
                logger.warning("CMAP grids present but no matching atom pairs found.")

        if not bond_terms and not angle_terms and not torsion_terms and not vdw_terms and not cmap_terms:
            raise ValueError(
                "No OpenMM terms were created. Force field did not match any "
                "detected bonds, angles, torsions, vdW types, or CMAP pairs."
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

        if ub_terms:
            system.addForce(ub_force)
        else:
            ub_force = None

        if cmap_terms:
            system.addForce(cmap_force)
        else:
            cmap_force = None

        integrator, context = self._create_context(system, precision=precision)
        context.setPositions(self._positions(molecule))

        return OpenMMHandle(
            molecule=molecule,
            system=system,
            integrator=integrator,
            context=context,
            bond_force=bond_force,
            angle_force=angle_force,
            torsion_force=torsion_force,
            vdw_force=vdw_force,
            ub_force=ub_force,
            cmap_force=cmap_force,
            bond_terms=bond_terms,
            angle_terms=angle_terms,
            torsion_terms=torsion_terms,
            vdw_terms=vdw_terms,
            ub_terms=ub_terms,
            cmap_terms=cmap_terms,
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
        if incoming_form != handle.functional_form:
            raise ValueError(
                f"Force field functional form {incoming_form!r} does not match "
                f"the handle's form {handle.functional_form!r}. "
                f"Create a new context instead of reusing this handle."
            )
        use_harmonic = handle.functional_form == FunctionalForm.HARMONIC

        if handle.bond_force is not None:
            for term in handle.bond_terms:
                param = forcefield.match_bond(
                    term.elements,
                    env_id=term.env_id,
                    ff_row=term.ff_row,
                    bond_order=getattr(term, "bond_order", ""),
                    bond_length=getattr(term, "length", None),
                )
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
                param = forcefield.match_angle(term.elements, env_id=term.env_id, ff_row=term.ff_row)
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
                params = forcefield.match_torsion(
                    term.elements, env_id=term.env_id, ff_row=term.ff_row, is_improper=term.is_improper
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
                    canonical_to_openmm_torsion_k(param.force_constant),
                )
            handle.torsion_force.updateParametersInContext(handle.context)

        if handle.vdw_force is not None:
            for term in handle.vdw_terms:
                param = forcefield.match_vdw(atom_type=term.atom_type, element=term.element, ff_row=term.ff_row)
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

        if handle.ub_force is not None:
            for term in handle.ub_terms:
                param = forcefield.match_angle(term.elements, env_id=term.env_id, ff_row=term.ff_row)
                if param is None:
                    raise ValueError(f"Updated force field is missing UB parameter for angle {term.elements}.")
                if param.ub_force_constant is None and param.ub_equilibrium is None:
                    raise ValueError(f"Updated force field is missing UB parameter for angle {term.elements}.")
                if param.ub_force_constant is None or param.ub_equilibrium is None:
                    raise ValueError(
                        "Inconsistent Urey-Bradley parameters for angle "
                        f"{term.elements} (env_id={term.env_id}, ff_row={term.ff_row}): "
                        "both 'ub_force_constant' and 'ub_equilibrium' must be set or both must be None."
                    )
                handle.ub_force.setBondParameters(
                    term.force_index,
                    term.atom_i,
                    term.atom_k,
                    ang_to_nm(param.ub_equilibrium),
                    _bond_k_to_harmonic(param.ub_force_constant),
                )
            handle.ub_force.updateParametersInContext(handle.context)

    def export_system_xml(
        self,
        path: str | Path,
        structure: Molecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
    ) -> Path:
        """Serialize the OpenMM System to XML.

        Produces a topology-specific XML file containing the force objects
        (``HarmonicBondForce``/``CustomBondForce``, etc. depending on the
        functional form) with all per-term parameters.  The file can be
        loaded back with ``openmm.XmlSerializer.deserialize()``.

        Args:
            path: Output file path.
            structure (Molecule | str | Path | OpenMMHandle): A
                :class:`~q2mm.models.molecule.Molecule`, path to an XYZ
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
        self, structure: Molecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None
    ) -> OpenMMHandle:
        """Get or create an :class:`OpenMMHandle`.

        If *structure* is already an :class:`OpenMMHandle`, optionally update
        its parameters.  Otherwise, build a new handle.

        Args:
            structure (Molecule | str | Path | OpenMMHandle): A
                :class:`Molecule`, XYZ path, or existing
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

    def _create_diff_handle(self, molecule: Molecule, forcefield: ForceField) -> _DiffHandle:
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
        ff_form = forcefield.functional_form
        use_harmonic = ff_form == FunctionalForm.HARMONIC

        system = mm.System()
        for symbol in molecule.symbols:
            system.addParticle(MASSES[symbol] * unit.dalton)

        param_names: list[str] = []
        param_vector_indices: list[int] = []
        grad_unit_factors: list[float] = []
        pv_idx = 0  # tracks position in flat param vector

        # Precompute assignments and build reverse indices for O(n) lookups.
        bond_assignments = _collect_bond_assignments(molecule, forcefield)
        bonds_by_param = _index_by_param_id(bond_assignments)

        # --- Bonds: each bond param contributes (k, r0) ---
        bond_global_map: dict[int, tuple[str, str]] = {}
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

            for bond in bonds_by_param.get(id(bp), []):
                bf.addBond(bond.atom_i, bond.atom_j)
            system.addForce(bf)

        # Precompute angle assignments and build reverse index.
        angle_assignments = _collect_angle_assignments(molecule, forcefield)
        angles_by_param = _index_by_param_id(angle_assignments)

        # --- Angles: each angle param contributes (k, theta0) ---
        angle_global_map: dict[int, tuple[str, str]] = {}
        angle_k_factor = _angle_k_to_openmm(1.0)
        for ap_idx, ap in enumerate(forcefield.angles):
            k_name = f"angle_k_{ap_idx}"
            t0_name = f"angle_t0_{ap_idx}"
            angle_global_map[ap_idx] = (k_name, t0_name)
            param_names.extend([k_name, t0_name])
            param_vector_indices.extend([pv_idx, pv_idx + 1])
            grad_unit_factors.extend(
                [
                    angle_k_factor,
                    np.deg2rad(1.0),
                ]
            )
            pv_idx += 2

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

            for angle in angles_by_param.get(id(ap), []):
                af.addAngle(angle.atom_i, angle.atom_j, angle.atom_k)
            system.addForce(af)

        # Precompute torsion assignments and build reverse indices.
        proper_assignments = _collect_torsion_assignments(molecule, forcefield, is_improper=False)
        improper_assignments = _collect_torsion_assignments(molecule, forcefield, is_improper=True)
        all_torsion_assignments = proper_assignments + improper_assignments
        torsions_by_param = _index_by_param_id(all_torsion_assignments)

        # --- Torsions: each torsion param (proper and improper) contributes (k,) ---
        torsion_global_map: dict[int, str] = {}
        torsion_k_factor = canonical_to_openmm_torsion_k(1.0)
        for tp_idx, tp in enumerate(forcefield.torsions):
            k_name = f"torsion_k_{tp_idx}"
            torsion_global_map[tp_idx] = k_name
            param_names.append(k_name)
            param_vector_indices.append(pv_idx)
            grad_unit_factors.append(torsion_k_factor)
            pv_idx += 1

        for tp_idx, tp in enumerate(forcefield.torsions):
            k_name = torsion_global_map[tp_idx]
            k_val = canonical_to_openmm_torsion_k(tp.force_constant)
            phase_rad = np.deg2rad(float(tp.phase))
            n = tp.periodicity
            expr = f"{k_name}*(1+cos({n}*theta-{phase_rad:.15g}))"
            tf = mm.CustomTorsionForce(expr)
            tf.setForceGroup(1)
            tf.addGlobalParameter(k_name, k_val)
            tf.addEnergyParameterDerivative(k_name)

            for torsion in torsions_by_param.get(id(tp), []):
                tf.addTorsion(
                    torsion.atom_i,
                    torsion.atom_j,
                    torsion.atom_k,
                    torsion.atom_l,
                    [],
                )
            system.addForce(tf)

        # --- vdW: advance pv_idx past vdW params ---
        # vdW uses per-particle parameters (no global-param derivatives).
        # Gradients are computed via finite differences in energy_and_param_grad().
        pv_idx += 2 * len(forcefield.vdws)

        if forcefield.vdws:
            vdw_assignments = _collect_vdw_assignments(molecule, forcefield)
            if use_harmonic:
                vdw_force = mm.NonbondedForce()
                vdw_force.setNonbondedMethod(mm.NonbondedForce.NoCutoff)
                for _atom_index, _symbol, _atom_type, param in vdw_assignments:
                    vdw_force.addParticle(0.0, _vdw_sigma_nm(param.radius), _vdw_epsilon_to_openmm(param.epsilon))
                _build_harmonic_exclusions(molecule, vdw_force)
            else:
                vdw_force = mm.CustomNonbondedForce(
                    f"step(r - rc) * epsilon*(-{MM3_VDW_C}*(rv/r)^6 + {MM3_VDW_A}*exp(-{MM3_VDW_B}*r/rv))"
                    f" + step(rc - r) * epsilon*{MM3_VDW_A}*exp(-{MM3_VDW_B}*rc/rv) * (rc/r)^12;"
                    "rc=0.34*rv;"
                    "rv=radius1+radius2;"
                    "epsilon=sqrt(epsilon1*epsilon2)"
                )
                vdw_force.addPerParticleParameter("radius")
                vdw_force.addPerParticleParameter("epsilon")
                vdw_force.setNonbondedMethod(mm.CustomNonbondedForce.NoCutoff)
                for _atom_index, _symbol, _atom_type, param in vdw_assignments:
                    vdw_force.addParticle([_vdw_radius_to_openmm(param.radius), _vdw_epsilon_to_openmm(param.epsilon)])
                vdw_force.createExclusionsFromBonds([(b.atom_i, b.atom_j) for b in molecule.bonds], 2)

            system.addForce(vdw_force)

        # --- Urey-Bradley: each UB angle contributes (ub_k, ub_r0) on the
        # 1-3 pair (atom_i, atom_k).  These live at the tail of the param
        # vector (after vdW), mirroring ParameterLayout ordering.  Modeled as a
        # harmonic bond so the energy and derivatives match energy()'s
        # HarmonicBondForce term. ---
        ub_k_factor = canonical_to_openmm_bond_k_nm(1.0)
        for ub_idx, ap in enumerate(forcefield._ub_angles):
            k_name = f"ub_k_{ub_idx}"
            r0_name = f"ub_r0_{ub_idx}"
            param_names.extend([k_name, r0_name])
            param_vector_indices.extend([pv_idx, pv_idx + 1])
            grad_unit_factors.extend(
                [
                    ub_k_factor,  # dk_openmm/dk_canonical
                    0.1,  # dr0_openmm/dr0_canonical (Å → nm)
                ]
            )
            pv_idx += 2

            ubf = mm.CustomBondForce(f"{k_name}*(r-{r0_name})^2")
            ubf.setForceGroup(0)
            ubf.addGlobalParameter(k_name, canonical_to_openmm_bond_k_nm(ap.ub_force_constant))
            ubf.addGlobalParameter(r0_name, ang_to_nm(ap.ub_equilibrium))
            ubf.addEnergyParameterDerivative(k_name)
            ubf.addEnergyParameterDerivative(r0_name)
            for angle in angles_by_param.get(id(ap), []):
                ubf.addBond(angle.atom_i, angle.atom_k)
            system.addForce(ubf)

        # --- CMAP: correction grids carry no tunable parameters, so they
        # contribute to the scalar energy only.  Add them here so that
        # energy_and_param_grad's energy agrees with energy(). ---
        cmap_force, _cmap_terms = _build_cmap_force(molecule, forcefield)
        if cmap_force is not None:
            system.addForce(cmap_force)

        # Use double precision on GPU so that analytical derivatives
        # (getParameterDerivatives) are computed in float64.
        integrator, context = self._create_context(system, precision="double")
        context.setPositions(self._positions(molecule))

        return _DiffHandle(
            integrator=integrator,
            context=context,
            param_names=param_names,
            param_vector_indices=param_vector_indices,
            grad_unit_factors=grad_unit_factors,
            functional_form=forcefield.functional_form,
        )

    def energy_and_param_grad(self, structure: Molecule, forcefield: ForceField) -> tuple[float, np.ndarray]:
        """Compute energy and analytical gradient w.r.t. FF parameters.

        Uses OpenMM's ``addEnergyParameterDerivative()`` on ``CustomForce``
        objects to get exact dE/d(param) for bond, angle, and torsion
        parameters.  vdW parameters use per-particle values that cannot
        be differentiated via global parameters, so their gradients are
        computed via central finite differences automatically.

        Args:
            structure (Molecule): Molecular structure.
            forcefield (ForceField): Force field parameters.

        Returns:
            tuple[float, np.ndarray]: ``(energy, grad)`` where ``energy``
                is in kcal/mol and ``grad`` has the same length as
                ``ParameterLayout.from_force_field(forcefield).vector(forcefield)``.

        """
        molecule = _as_molecule(structure)
        if forcefield.stretch_bends:
            raise NotImplementedError(
                "OpenMMEngine does not support stretch-bend cross terms. "
                "Use JaxEngine for force fields with stretch-bend parameters."
            )
        diff = self._create_diff_handle(molecule, forcefield)

        state = diff.context.getState(getEnergy=True, getParameterDerivatives=True)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        derivs = state.getEnergyParameterDerivatives()

        from q2mm.models.parameters import ParameterKind, ParameterLayout

        layout = ParameterLayout.from_force_field(forcefield)
        param_vector = layout.vector(forcefield)
        grad = np.zeros(len(param_vector))

        for name, pv_idx, unit_factor in zip(
            diff.param_names, diff.param_vector_indices, diff.grad_unit_factors, strict=True
        ):
            deriv_openmm = derivs[name]  # dE_kJ/dp_openmm
            grad[pv_idx] = kj_to_kcal(deriv_openmm * unit_factor)

        # vdW parameters use per-particle values without global-parameter
        # derivatives.  Supplement with central finite differences.
        # Reuse a single OpenMMHandle to avoid rebuilding the OpenMM
        # context for each perturbation.  Use double precision on GPU
        # so the finite differences are not lost to float32 rounding.
        if forcefield.vdws:
            vdw_radius_indices = layout.indices_by_kind.get(ParameterKind.VDW_RADIUS, ())
            vdw_start = min(vdw_radius_indices)
            vdw_end = vdw_start + 2 * len(forcefield.vdws)
            step = 1e-4
            handle = self.create_context(molecule, forcefield, precision="double")
            for i in range(vdw_start, vdw_end):
                pv_plus = param_vector.copy()
                pv_plus[i] += step
                pv_minus = param_vector.copy()
                pv_minus[i] -= step
                e_plus = self.energy(handle, layout.replace(forcefield, pv_plus))
                e_minus = self.energy(handle, layout.replace(forcefield, pv_minus))
                grad[i] = (e_plus - e_minus) / (2.0 * step)

        return energy, grad

    def energy(self, structure: Molecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None) -> float:
        """Calculate MM energy in kcal/mol.

        Args:
            structure (Molecule | str | Path | OpenMMHandle): Molecule,
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
        structure: Molecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
        tolerance: float = 1.0,
        max_iterations: int = 200,
    ) -> tuple:
        """Energy-minimize structure using L-BFGS.

        Args:
            structure (Molecule | str | Path | OpenMMHandle): Molecule,
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
        handle.molecule = handle.molecule.with_geometry(coords)
        energy = float(state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole))
        return energy, list(handle.molecule.symbols), coords

    def hessian(
        self,
        structure: Molecule | str | Path | OpenMMHandle,
        forcefield: ForceField | None = None,
        step: float = 1.0e-4,
    ) -> np.ndarray:
        """Finite-difference Hessian in canonical units (Hartree/Bohr²).

        Internally computed in kJ/mol/nm² (OpenMM native) and converted
        to Hartree/Bohr² before returning, matching the canonical unit
        contract defined in :class:`~q2mm.backends.base.MMEngine`.

        Args:
            structure (Molecule | str | Path | OpenMMHandle): Molecule,
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
        self, structure: Molecule | str | Path | OpenMMHandle, forcefield: ForceField | None = None, **kwargs: Any
    ) -> list[float]:
        """Approximate harmonic frequencies in cm⁻¹ from the numerical Hessian.

        Args:
            structure (Molecule | str | Path | OpenMMHandle): Molecule,
                XYZ path, or :class:`OpenMMHandle`.
            forcefield: Force field to apply. Auto-generated if ``None``.
            **kwargs: Forwarded to
                :func:`~q2mm.models.hessian.hessian_to_frequencies`
                (e.g. ``on_error="penalty"``).

        Returns:
            list[float]: Vibrational frequencies in cm⁻¹.

        """
        from q2mm.models.hessian import hessian_to_frequencies

        handle = self._prepare_handle(structure, forcefield)
        hessian_au = self.hessian(handle)  # Hartree/Bohr²
        return hessian_to_frequencies(hessian_au, list(handle.molecule.symbols), **kwargs)
