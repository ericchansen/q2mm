"""Optional non-periodic ASE reference adapter.

The adapter requires a caller-supplied ASE calculator.  It snapshots that
calculator with :func:`copy.deepcopy` during construction and makes another
independent copy for every prepared session.  Calculators that cannot be
deep-copied are rejected with a typed configuration error rather than being
shared across sessions.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from q2mm._provenance import freeze_json_mapping
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendConfigurationError,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendUnavailableError,
    Capability,
    CoordinateGradientResult,
    CoordinateGradientUnit,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
    PreparationError,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    readonly_array,
)
from q2mm.models.molecule import Molecule

_ase: Any = None
_Atoms: Any = None
_BOHR_ANGSTROM: Any = None
_HARTREE_EV: Any = None

try:
    import ase as _ase_import
    from ase import Atoms as _Atoms_import
    from ase.units import Bohr as _BOHR_ANGSTROM_IMPORT
    from ase.units import Hartree as _HARTREE_EV_IMPORT
except ImportError:
    pass
else:
    _ase = _ase_import
    _Atoms = _Atoms_import
    _BOHR_ANGSTROM = _BOHR_ANGSTROM_IMPORT
    _HARTREE_EV = _HARTREE_EV_IMPORT

_CONVERSION_VERSION = 1
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ASEEvaluationError(EvaluationError):
    """An ASE calculator evaluation failed."""


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _safe_calculator_parameters(calculator: object) -> dict[str, object]:
    """Return only individually JSON-safe, non-sensitive calculator parameters."""
    try:
        parameters = getattr(calculator, "parameters", None)
    except Exception:
        return {}
    if not isinstance(parameters, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or _contains_absolute_path(key) or _contains_absolute_path(value):
            continue
        try:
            frozen = freeze_json_mapping({key: value}, path="calculator.parameters")
        except ValueError:
            continue
        safe[key] = frozen[key]
    return safe


def _safe_identity(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or _contains_absolute_path(value):
        raise BackendConfigurationError(f"ASE calculator {name} is not a safe provenance identifier.")
    try:
        freeze_json_mapping({"value": value}, path=f"calculator.{name}")
    except ValueError as exc:
        raise BackendConfigurationError(f"ASE calculator {name} is not safe for provenance.") from exc
    return value


def _runtime_capabilities(calculator: object) -> tuple[frozenset[Capability], tuple[str, ...]]:
    """Validate calculator structure and derive its exact supported subset."""
    try:
        raw_properties = getattr(calculator, "implemented_properties")
    except Exception as exc:
        raise BackendConfigurationError("ASE calculator implemented_properties could not be read.") from exc
    if isinstance(raw_properties, str) or not isinstance(raw_properties, Iterable):
        raise BackendConfigurationError("ASE calculator implemented_properties must be an iterable of strings.")
    try:
        properties = tuple(raw_properties)
    except Exception as exc:
        raise BackendConfigurationError("ASE calculator implemented_properties could not be enumerated.") from exc
    if not all(isinstance(item, str) for item in properties):
        raise BackendConfigurationError("ASE calculator implemented_properties must contain only strings.")

    implemented = tuple(sorted(set(properties)))
    capabilities: set[Capability] = set()
    if "energy" in implemented:
        if not callable(getattr(calculator, "get_potential_energy", None)):
            raise BackendConfigurationError(
                "ASE calculator declares 'energy' but has no callable get_potential_energy method."
            )
        capabilities.add(Capability.ENERGY)
    if "forces" in implemented:
        if not callable(getattr(calculator, "get_forces", None)):
            raise BackendConfigurationError("ASE calculator declares 'forces' but has no callable get_forces method.")
        capabilities.add(Capability.COORDINATE_GRADIENT)
    if not capabilities:
        raise BackendConfigurationError(
            "ASE calculator must implement at least one of the 'energy' or 'forces' properties."
        )
    if not callable(getattr(calculator, "reset", None)):
        raise BackendConfigurationError("ASE calculator must expose a callable reset method for isolated sessions.")
    return frozenset(capabilities), implemented


def _periodic_state_reason(atoms: object) -> str:
    """Return a reason when an Atoms-like object carries PBC or a nonzero cell."""
    pbc = np.asarray(getattr(atoms, "pbc"), dtype=bool)
    cell = np.asarray(getattr(atoms, "cell"), dtype=float)
    if np.any(pbc):
        return "periodic boundary conditions are enabled"
    if cell.size and np.any(cell != 0.0):
        return "a nonzero simulation cell is present"
    return ""


def _per_atom_state_reason(atoms: object) -> str:
    """Return a reason when unsupported charge or spin arrays are present."""
    for name in ("initial_charges", "initial_magmoms"):
        if getattr(atoms, "has")(name):
            return f"unsupported per-atom {name} state is present"
    return ""


def _validate_calculator_owned_atoms(calculator: object) -> None:
    try:
        atoms = getattr(calculator, "atoms", None)
        reason = "" if atoms is None else _periodic_state_reason(atoms) or _per_atom_state_reason(atoms)
    except Exception as exc:
        raise BackendConfigurationError("ASE calculator-owned Atoms state could not be validated.") from exc
    if reason:
        raise BackendConfigurationError(f"ASE v1 is non-periodic; calculator-owned Atoms state has {reason}.")


def _deepcopy_calculator(calculator: object, *, preparing: bool) -> object:
    error_type: type[BackendConfigurationError] | type[PreparationError]
    error_type = PreparationError if preparing else BackendConfigurationError
    try:
        cloned = copy.deepcopy(calculator)
    except Exception as exc:
        raise error_type(
            "ASE calculators must support copy.deepcopy so each prepared session owns isolated calculator state."
        ) from exc
    if cloned is calculator:
        raise error_type("ASE calculator copy returned the caller's mutable calculator instance.")
    try:
        getattr(cloned, "reset")()
    except Exception as exc:
        raise error_type("ASE calculator copy could not be reset for isolated session state.") from exc
    return cloned


class ASEBackend:
    """Reference backend for a required, user-supplied non-periodic ASE calculator."""

    def __init__(self, *, calculator: object) -> None:
        if _ase is None or _Atoms is None or _HARTREE_EV is None or _BOHR_ANGSTROM is None:
            raise BackendUnavailableError("ASE reference backend requires 'ase'. Install q2mm[ase].")

        capabilities, properties = _runtime_capabilities(calculator)
        _validate_calculator_owned_atoms(calculator)
        template = _deepcopy_calculator(calculator, preparing=False)
        copied_capabilities, copied_properties = _runtime_capabilities(template)
        if copied_capabilities != capabilities or copied_properties != properties:
            raise BackendConfigurationError("Copying the ASE calculator changed its implemented properties.")
        _validate_calculator_owned_atoms(template)

        calculator_type = type(calculator)
        self._calculator_template = template
        self._implemented_properties = properties
        self._capabilities = capabilities
        self._ase_version = str(getattr(_ase, "__version__", ""))
        self._calculator_class = _safe_identity(calculator_type.__name__, name="class")
        self._calculator_module = _safe_identity(calculator_type.__module__, name="module")
        self._calculator_parameters = _safe_calculator_parameters(template)
        provenance = BackendProvenance(
            backend="ase",
            role=BackendRole.REFERENCE,
            version=self._ase_version,
            details=self._provenance_details(driver=None, molecule=None),
        )
        self._info = BackendInfo(
            name=f"ASE ({self._calculator_class})",
            role=BackendRole.REFERENCE,
            capabilities=capabilities,
            functional_forms=frozenset(),
            provenance=provenance,
        )

    @property
    def info(self) -> BackendInfo:
        """Immutable runtime declaration derived from implemented_properties."""
        return self._info

    def prepare(self, request: PreparationRequest) -> PreparedASE:
        """Build an isolated non-periodic ASE Atoms/calculator session."""
        if request.force_field is not None:
            raise PreparationError("ASE reference preparation does not accept a force_field.")
        if request.options:
            raise PreparationError("ASE reference preparation does not accept per-case options.")
        if not isinstance(request.molecule, Molecule):
            raise PreparationError("ASE reference preparation requires a q2mm Molecule.")

        calculator = _deepcopy_calculator(self._calculator_template, preparing=True)
        try:
            capabilities, properties = _runtime_capabilities(calculator)
        except BackendConfigurationError as exc:
            raise PreparationError(f"Copied ASE calculator is invalid: {exc}") from exc
        if capabilities != self._capabilities or properties != self._implemented_properties:
            raise PreparationError("Copying the ASE calculator changed its runtime capabilities.")

        molecule = request.molecule
        try:
            atoms = _Atoms(
                symbols=list(molecule.symbols),
                positions=np.array(molecule.geometry, dtype=float, copy=True),
                cell=None,
                pbc=False,
                info={
                    "q2mm_charge": molecule.charge,
                    "q2mm_multiplicity": molecule.multiplicity,
                },
            )
            reason = _periodic_state_reason(atoms)
            if reason:
                raise ValueError(reason)
            reason = _per_atom_state_reason(atoms)
            if reason:
                raise ValueError(reason)
            atoms.calc = calculator
            reason = _periodic_state_reason(atoms)
            if reason:
                raise ValueError(f"calculator attachment introduced {reason}")
            reason = _per_atom_state_reason(atoms)
            if reason:
                raise ValueError(f"calculator attachment introduced {reason}")
            owned_atoms = getattr(calculator, "atoms", None)
            if owned_atoms is not None:
                reason = _periodic_state_reason(owned_atoms) or _per_atom_state_reason(owned_atoms)
                if reason:
                    raise ValueError(f"calculator attachment introduced {reason} in calculator-owned state")
        except Exception as exc:
            raise PreparationError(f"Could not construct a non-periodic ASE prepared session: {exc}") from exc
        return PreparedASE(backend=self, case_id=request.case_id, molecule=molecule, atoms=atoms, calculator=calculator)

    def _provenance_details(self, *, driver: str | None, molecule: Molecule | None) -> dict[str, object]:
        calculator: dict[str, object] = {
            "class": self._calculator_class,
            "module": self._calculator_module,
        }
        if self._calculator_parameters:
            calculator["parameters"] = self._calculator_parameters
        details: dict[str, object] = {
            "adapter": {
                "name": "q2mm-ase",
                "backend": "ase",
                "class": "ASEBackend",
                "conversion_version": _CONVERSION_VERSION,
                "periodicity": "non-periodic-only",
            },
            "implementation": {"name": "ASE", "version": self._ase_version},
            "calculator": calculator,
            "runtime": {
                "implemented_properties": self._implemented_properties,
                "capabilities": tuple(sorted(capability.value for capability in self._capabilities)),
            },
        }
        if molecule is not None:
            details["structure"] = {
                "charge": molecule.charge,
                "multiplicity": molecule.multiplicity,
                "pbc": False,
                "cell": "zero",
            }
        if driver == "energy":
            details["driver"] = {"property": "energy", "ase_property": "energy"}
            details["units"] = {
                "native": "eV",
                "canonical": "hartree",
                "formula": "energy_hartree = energy_eV / ase.units.Hartree",
                "hartree_eV": float(_HARTREE_EV),
                "conversion_version": _CONVERSION_VERSION,
            }
        elif driver == "forces":
            details["driver"] = {"property": "coordinate_gradient", "ase_property": "forces"}
            details["units"] = {
                "native": "eV/angstrom",
                "canonical": "hartree/bohr",
                "formula": ("gradient_hartree_per_bohr = -forces_eV_per_angstrom * ase.units.Bohr / ase.units.Hartree"),
                "bohr_angstrom": float(_BOHR_ANGSTROM),
                "hartree_eV": float(_HARTREE_EV),
                "conversion_version": _CONVERSION_VERSION,
            }
        return details

    def _result_provenance(self, *, driver: str, molecule: Molecule) -> BackendProvenance:
        return BackendProvenance(
            backend="ase",
            role=BackendRole.REFERENCE,
            version=self._ase_version,
            details=self._provenance_details(driver=driver, molecule=molecule),
        )


class PreparedASE(AbstractPreparedBackend):
    """Reusable ASE session owning one Atoms object and calculator copy."""

    def __init__(
        self,
        *,
        backend: ASEBackend,
        case_id: str,
        molecule: Molecule,
        atoms: Any,
        calculator: object,
    ) -> None:
        super().__init__(
            info=backend.info,
            case_id=case_id,
            molecule=molecule,
            force_field=None,
            layout=None,
        )
        self._backend = backend
        self._atoms = atoms
        self._calculator = calculator
        self._symbols = tuple(molecule.symbols)
        self._geometry = np.array(molecule.geometry, dtype=float, copy=True)

    def _validate_native_state(self) -> None:
        try:
            if tuple(self._atoms.get_chemical_symbols()) != self._symbols:
                raise ValueError("atom symbols/order changed")
            if not np.array_equal(np.asarray(self._atoms.positions, dtype=float), self._geometry):
                raise ValueError("atom geometry changed")
            if self._atoms.info.get("q2mm_charge") != self.molecule.charge:
                raise ValueError("q2mm charge metadata changed")
            if self._atoms.info.get("q2mm_multiplicity") != self.molecule.multiplicity:
                raise ValueError("q2mm multiplicity metadata changed")
            reason = _periodic_state_reason(self._atoms)
            if reason:
                raise ValueError(reason)
            reason = _per_atom_state_reason(self._atoms)
            if reason:
                raise ValueError(reason)
            owned_atoms = getattr(self._calculator, "atoms", None)
            if owned_atoms is not None:
                reason = _periodic_state_reason(owned_atoms) or _per_atom_state_reason(owned_atoms)
                if reason:
                    raise ValueError(f"calculator-owned Atoms state has {reason}")
        except Exception as exc:
            raise ASEEvaluationError(f"ASE session violated the non-periodic structure contract: {exc}") from exc

    def _energy(self, request: EnergyRequest | ReferenceEnergyRequest) -> EnergyResult:
        self._validate_native_state()
        try:
            energy_ev = float(self._atoms.get_potential_energy())
        except Exception as exc:
            raise ASEEvaluationError(f"ASE energy evaluation failed: {exc}") from exc
        self._validate_native_state()
        if not np.isfinite(energy_ev):
            raise ASEEvaluationError("ASE energy evaluation returned a non-finite value.")
        return EnergyResult(
            energy=energy_ev / float(_HARTREE_EV),
            unit=EnergyUnit.HARTREE,
            provenance=self._backend._result_provenance(driver="energy", molecule=self.molecule),
        )

    def _coordinate_gradient(
        self,
        request: ReferenceCoordinateGradientRequest,
    ) -> CoordinateGradientResult:
        self._validate_native_state()
        try:
            forces = np.asarray(self._atoms.get_forces(), dtype=float)
        except Exception as exc:
            raise ASEEvaluationError(f"ASE force evaluation failed: {exc}") from exc
        self._validate_native_state()
        expected = (len(self._symbols), 3)
        if forces.shape != expected:
            raise ASEEvaluationError(f"ASE forces have shape {forces.shape}, expected {expected}.")
        if not np.all(np.isfinite(forces)):
            raise ASEEvaluationError("ASE forces contain non-finite values.")
        gradient = -forces * float(_BOHR_ANGSTROM) / float(_HARTREE_EV)
        return CoordinateGradientResult(
            gradient=readonly_array(gradient),
            unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
            provenance=self._backend._result_provenance(driver="forces", molecule=self.molecule),
        )
