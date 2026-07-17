"""Lazy QCEngine adapter for atomic reference calculations.

The adapter uses QCSchema v2 atomic inputs and intentionally exposes only
single-point energy, Cartesian gradient, and Cartesian Hessian drivers.
QCEngine procedures are not used.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
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
    HessianRequest,
    HessianResult,
    HessianUnit,
    PreparationError,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceHessianRequest,
    readonly_array,
)
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.molecule import Molecule

try:
    import qcengine as _qcengine
    import qcelemental as _qcelemental
    from qcelemental.models.v2 import AtomicInput as _AtomicInput
    from qcelemental.models.v2 import AtomicSpecification as _AtomicSpecification
    from qcelemental.models.v2 import Molecule as _QCSchemaMolecule
except ImportError:
    _qcengine = None
    _qcelemental = None
    _AtomicInput = None
    _AtomicSpecification = None
    _QCSchemaMolecule = None

_CAPABILITIES = frozenset(
    {
        Capability.ENERGY,
        Capability.COORDINATE_GRADIENT,
        Capability.HESSIAN,
    }
)
_TASK_RESOURCE_FIELDS = frozenset({"ncores", "nnodes", "memory", "retries", "cores_per_rank"})
_PROTOCOL_FIELDS = frozenset({"stdout", "wavefunction", "native_files", "error_correction"})
_WAVEFUNCTION_PROTOCOLS = frozenset(
    {"all", "orbitals_and_eigenvalues", "occupations_and_eigenvalues", "return_results", "none"}
)
_NATIVE_FILES_PROTOCOLS = frozenset({"all", "input", "none"})
_SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_CONVERSION_SCHEMA_VERSION = 1


class QCEngineEvaluationError(EvaluationError):
    """An atomic QCEngine computation failed."""

    def __init__(self, message: str, *, error_type: str = "") -> None:
        self.error_type = error_type
        super().__init__(message)


def _require_nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise BackendConfigurationError(f"{name} must be a non-empty string without surrounding whitespace.")
    if any(ord(character) < 32 for character in value):
        raise BackendConfigurationError(f"{name} must not contain control characters.")
    if _contains_absolute_path(value):
        raise BackendConfigurationError(f"{name} must not be an absolute path.")
    try:
        freeze_json_mapping({"value": value}, path=name)
    except ValueError as exc:
        raise BackendConfigurationError(str(exc)) from exc
    return value


def _contains_absolute_path(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith(("/", "\\")) or _WINDOWS_ABSOLUTE_PATH.match(value) is not None
    if isinstance(value, Mapping):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_absolute_path(item) for item in value)
    return False


def _freeze_safe_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BackendConfigurationError(f"{name} must be a mapping.")
    try:
        frozen = freeze_json_mapping(value, path=name)
    except ValueError as exc:
        raise BackendConfigurationError(str(exc)) from exc
    if _contains_absolute_path(frozen):
        raise BackendConfigurationError(f"{name} must not contain absolute paths.")
    return frozen


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _validate_task_config(value: object) -> Mapping[str, object]:
    frozen = _freeze_safe_mapping(value, name="task_config")
    unknown = set(frozen) - _TASK_RESOURCE_FIELDS
    if unknown:
        raise BackendConfigurationError(
            f"task_config contains unsupported resource field(s): {', '.join(sorted(unknown))}."
        )
    for name in ("ncores", "nnodes", "cores_per_rank"):
        item = frozen.get(name)
        if item is not None and (isinstance(item, bool) or not isinstance(item, int) or item <= 0):
            raise BackendConfigurationError(f"task_config.{name} must be a positive integer.")
    retries = frozen.get("retries")
    if retries is not None and (isinstance(retries, bool) or not isinstance(retries, int) or retries < 0):
        raise BackendConfigurationError("task_config.retries must be a non-negative integer.")
    memory = frozen.get("memory")
    if memory is not None and (
        isinstance(memory, bool) or not isinstance(memory, (int, float)) or not np.isfinite(memory) or memory <= 0
    ):
        raise BackendConfigurationError("task_config.memory must be a positive finite number of GiB.")
    return frozen


def _validate_protocols(value: object) -> Mapping[str, object]:
    frozen = _freeze_safe_mapping(value, name="protocols")
    unknown = set(frozen) - _PROTOCOL_FIELDS
    if unknown:
        raise BackendConfigurationError(f"protocols contains unsupported field(s): {', '.join(sorted(unknown))}.")
    stdout = frozen.get("stdout")
    if stdout is not None and not isinstance(stdout, bool):
        raise BackendConfigurationError("protocols.stdout must be a boolean.")
    for name, allowed in (
        ("wavefunction", _WAVEFUNCTION_PROTOCOLS),
        ("native_files", _NATIVE_FILES_PROTOCOLS),
    ):
        item = frozen.get(name)
        if item is not None and (not isinstance(item, str) or item not in allowed):
            raise BackendConfigurationError(f"protocols.{name} must be one of: {', '.join(sorted(allowed))}.")
    correction = frozen.get("error_correction")
    if correction is not None:
        if not isinstance(correction, Mapping):
            raise BackendConfigurationError("protocols.error_correction must be a mapping.")
        correction_unknown = set(correction) - {"default_policy", "policies"}
        if correction_unknown:
            raise BackendConfigurationError(
                f"protocols.error_correction contains unsupported field(s): {', '.join(sorted(correction_unknown))}."
            )
        default_policy = correction.get("default_policy")
        if default_policy is not None and not isinstance(default_policy, bool):
            raise BackendConfigurationError("protocols.error_correction.default_policy must be a boolean.")
        policies = correction.get("policies")
        if policies is not None:
            if not isinstance(policies, Mapping) or not all(
                isinstance(key, str) and isinstance(enabled, bool) for key, enabled in policies.items()
            ):
                raise BackendConfigurationError(
                    "protocols.error_correction.policies must map string names to booleans."
                )
    return frozen


def _safe_version(value: object) -> str:
    text = value if isinstance(value, str) else ""
    return text if _SAFE_VERSION.fullmatch(text) else ""


def _is_safe_provenance_string(value: str) -> bool:
    if not value or _contains_absolute_path(value):
        return False
    try:
        freeze_json_mapping({"value": value}, path="native_provenance")
    except ValueError:
        return False
    return True


def _native_provenance(result: object) -> dict[str, object]:
    native = getattr(result, "provenance", None)
    if native is None:
        return {}
    dumped = native.model_dump() if hasattr(native, "model_dump") else {}
    safe: dict[str, object] = {}
    for field in (
        "creator",
        "version",
        "routine",
        "ncores",
        "nnodes",
        "memory",
        "retries",
        "wall_time",
        "qcengine_version",
    ):
        value = dumped.get(field)
        if (
            (isinstance(value, str) and _is_safe_provenance_string(value))
            or isinstance(value, (bool, int))
            or (isinstance(value, float) and np.isfinite(value))
        ):
            safe[field] = value
    return safe


def _coerce_return_result(result: object, *, driver: str) -> np.ndarray:
    """Return a finite numeric result array or raise a typed adapter error."""
    try:
        values = np.asarray(getattr(result, "return_result", None), dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QCEngineEvaluationError(
            f"QCEngine {driver} returned non-numeric return_result data.",
            error_type="invalid_result",
        ) from exc
    if not np.all(np.isfinite(values)):
        raise QCEngineEvaluationError(
            f"QCEngine {driver} returned non-finite return_result data.",
            error_type="invalid_result",
        )
    return values


class QCEngineBackend:
    """Reference backend executing QCSchema atomic inputs through QCEngine."""

    def __init__(
        self,
        *,
        program: str,
        method: str,
        basis: str | None = None,
        keywords: Mapping[str, object] | None = None,
        protocols: Mapping[str, object] | None = None,
        task_config: Mapping[str, object] | None = None,
    ) -> None:
        if _qcengine is None or _qcelemental is None:
            raise BackendUnavailableError(
                "QCEngine reference backend requires both 'qcengine' and 'qcelemental'. Install q2mm[qcengine]."
            )
        self._program = _require_nonempty_string(program, name="program")
        self._method = _require_nonempty_string(method, name="method")
        if basis is not None:
            basis = _require_nonempty_string(basis, name="basis")
        self._basis = basis
        self._keywords = _freeze_safe_mapping({} if keywords is None else keywords, name="keywords")
        self._protocols = _validate_protocols({} if protocols is None else protocols)
        self._task_config = _validate_task_config({} if task_config is None else task_config)

        try:
            self._build_specification("energy")
        except BackendConfigurationError:
            raise
        except Exception as exc:
            raise BackendConfigurationError(f"Invalid QCEngine atomic configuration: {exc}") from exc

        try:
            self._harness = _qcengine.get_program(self._program, check=True)
        except Exception as exc:
            raise BackendUnavailableError(
                f"QCEngine program {self._program!r} is not registered or available."
            ) from exc

        try:
            program_version = _safe_version(self._harness.get_version())
        except Exception:
            program_version = ""
        qcengine_version = _safe_version(getattr(_qcengine, "__version__", ""))
        qcelemental_version = _safe_version(getattr(_qcelemental, "__version__", ""))
        self._runtime_provenance = BackendProvenance(
            backend="qcengine",
            role=BackendRole.REFERENCE,
            version=qcengine_version,
            details=self._provenance_details(
                driver=None,
                schema=None,
                native_provenance={},
                program_version=program_version,
                qcengine_version=qcengine_version,
                qcelemental_version=qcelemental_version,
            ),
        )
        model_name = self._method if self._basis is None else f"{self._method}/{self._basis}"
        self._info = BackendInfo(
            name=f"QCEngine ({self._program}: {model_name})",
            role=BackendRole.REFERENCE,
            capabilities=_CAPABILITIES,
            functional_forms=frozenset(),
            provenance=self._runtime_provenance,
        )
        self._program_version = program_version
        self._qcengine_version = qcengine_version
        self._qcelemental_version = qcelemental_version

    @property
    def info(self) -> BackendInfo:
        """Immutable runtime declaration for the configured adapter."""
        return self._info

    def prepare(self, request: PreparationRequest) -> PreparedQCEngine:
        """Create a reusable reference session without executing a job."""
        if request.force_field is not None:
            raise PreparationError("QCEngine reference preparation does not accept a force_field.")
        if request.options:
            raise PreparationError("QCEngine reference preparation does not accept per-case options.")
        if not isinstance(request.molecule, Molecule):
            raise PreparationError("QCEngine reference preparation requires a q2mm Molecule.")
        return PreparedQCEngine(backend=self, case_id=request.case_id, molecule=request.molecule)

    def _build_specification(self, driver: str) -> Any:
        model: dict[str, object] = {"method": self._method}
        if self._basis is not None:
            model["basis"] = self._basis
        try:
            return _AtomicSpecification(
                program=self._program,
                driver=driver,
                model=model,
                keywords=_thaw_json(self._keywords),
                protocols=_thaw_json(self._protocols),
                extras={},
            )
        except Exception as exc:
            raise BackendConfigurationError(f"Invalid QCSchema atomic specification: {exc}") from exc

    def _build_input(self, *, case_id: str, molecule: Molecule, driver: str) -> Any:
        geometry_bohr = np.asarray(molecule.geometry, dtype=float) / BOHR_TO_ANG
        try:
            schema_molecule = _QCSchemaMolecule(
                symbols=list(molecule.symbols),
                geometry=geometry_bohr,
                molecular_charge=molecule.charge,
                molecular_multiplicity=molecule.multiplicity,
                fix_com=True,
                fix_orientation=True,
            )
            return _AtomicInput(
                id=case_id,
                molecule=schema_molecule,
                specification=self._build_specification(driver),
            )
        except BackendConfigurationError:
            raise
        except Exception as exc:
            raise BackendConfigurationError(f"Could not construct QCSchema atomic input: {exc}") from exc

    def _provenance_details(
        self,
        *,
        driver: str | None,
        schema: Mapping[str, object] | None,
        native_provenance: Mapping[str, object],
        program_version: str,
        qcengine_version: str,
        qcelemental_version: str,
    ) -> dict[str, object]:
        model: dict[str, object] = {"method": self._method}
        if self._basis is not None:
            model["basis"] = self._basis
        details: dict[str, object] = {
            "adapter": {
                "name": "q2mm-qcengine",
                "backend": "qcengine",
                "class": "QCEngineBackend",
                "conversion_schema_version": _CONVERSION_SCHEMA_VERSION,
            },
            "implementation": {"name": "QCEngine", "version": qcengine_version},
            "qcelemental": {"name": "QCElemental", "version": qcelemental_version},
            "program": {"name": self._program, "version": program_version},
            "model": model,
            "configuration": {
                "keywords": _thaw_json(self._keywords),
                "protocols": _thaw_json(self._protocols),
                "task_config": _thaw_json(self._task_config),
            },
            "units": {
                "input_geometry": "angstrom",
                "qcschema_geometry": "bohr",
                "return_result": "atomic_units",
            },
        }
        if driver is not None:
            property_name = "coordinate_gradient" if driver == "gradient" else driver
            details["driver"] = {"property": property_name, "qcschema_driver": driver}
        if schema is not None:
            details["schema"] = dict(schema)
        if native_provenance:
            details["native_provenance"] = dict(native_provenance)
        return details

    def _result_provenance(self, result: object, *, driver: str) -> BackendProvenance:
        schema = {
            "input_name": "qcschema_atomic_input",
            "input_version": 2,
            "molecule_version": 3,
            "result_name": str(getattr(result, "schema_name", "")),
            "result_version": int(getattr(result, "schema_version", 2)),
        }
        return BackendProvenance(
            backend="qcengine",
            role=BackendRole.REFERENCE,
            version=self._qcengine_version,
            details=self._provenance_details(
                driver=driver,
                schema=schema,
                native_provenance=_native_provenance(result),
                program_version=self._program_version,
                qcengine_version=self._qcengine_version,
                qcelemental_version=self._qcelemental_version,
            ),
        )

    def _compute(self, *, case_id: str, molecule: Molecule, driver: str) -> tuple[object, BackendProvenance]:
        atomic_input = self._build_input(case_id=case_id, molecule=molecule, driver=driver)
        try:
            result = _qcengine.compute(
                atomic_input,
                self._program,
                raise_error=False,
                task_config=_thaw_json(self._task_config),
            )
        except Exception as exc:
            raise QCEngineEvaluationError(
                f"QCEngine {driver} execution raised {type(exc).__name__}: {exc}",
                error_type="native_exception",
            ) from exc

        if getattr(result, "success", None) is False:
            error = getattr(result, "error", None)
            error_type = getattr(error, "error_type", "")
            error_message = getattr(error, "error_message", "")
            if not isinstance(error_type, str) or not isinstance(error_message, str):
                raise QCEngineEvaluationError(
                    f"QCEngine {driver} returned a malformed FailedOperation.",
                    error_type="malformed_failed_operation",
                )
            raise QCEngineEvaluationError(
                f"QCEngine {driver} failed ({error_type}): {error_message}",
                error_type=error_type,
            )
        if getattr(result, "success", None) is not True:
            raise QCEngineEvaluationError(
                f"QCEngine {driver} returned an unexpected result type {type(result).__name__}.",
                error_type="invalid_result",
            )
        return result, self._result_provenance(result, driver=driver)


class PreparedQCEngine(AbstractPreparedBackend):
    """Reusable QCEngine session for one immutable molecule."""

    def __init__(self, *, backend: QCEngineBackend, case_id: str, molecule: Molecule) -> None:
        super().__init__(
            info=backend.info,
            case_id=case_id,
            molecule=molecule,
            force_field=None,
            layout=None,
        )
        self._backend = backend

    def _energy(self, request: EnergyRequest | ReferenceEnergyRequest) -> EnergyResult:
        result, provenance = self._backend._compute(
            case_id=self.case_id,
            molecule=self.molecule,
            driver="energy",
        )
        values = _coerce_return_result(result, driver="energy")
        if values.size != 1:
            raise QCEngineEvaluationError(
                f"QCEngine energy return_result must contain one value; got shape {values.shape}.",
                error_type="invalid_result_shape",
            )
        return EnergyResult(
            energy=float(values.reshape(-1)[0]),
            unit=EnergyUnit.HARTREE,
            provenance=provenance,
        )

    def _coordinate_gradient(
        self,
        request: ReferenceCoordinateGradientRequest,
    ) -> CoordinateGradientResult:
        result, provenance = self._backend._compute(
            case_id=self.case_id,
            molecule=self.molecule,
            driver="gradient",
        )
        values = _coerce_return_result(result, driver="gradient")
        expected = 3 * len(self.molecule.symbols)
        if values.size != expected:
            raise QCEngineEvaluationError(
                f"QCEngine gradient return_result has {values.size} values, expected {expected}.",
                error_type="invalid_result_shape",
            )
        return CoordinateGradientResult(
            gradient=readonly_array(values.reshape((-1, 3))),
            unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
            provenance=provenance,
        )

    def _hessian(self, request: HessianRequest | ReferenceHessianRequest) -> HessianResult:
        result, provenance = self._backend._compute(
            case_id=self.case_id,
            molecule=self.molecule,
            driver="hessian",
        )
        values = _coerce_return_result(result, driver="hessian")
        expected = 3 * len(self.molecule.symbols)
        if values.size != expected * expected:
            raise QCEngineEvaluationError(
                f"QCEngine Hessian return_result has {values.size} values, expected {expected * expected}.",
                error_type="invalid_result_shape",
            )
        return HessianResult(
            hessian=readonly_array(values.reshape((expected, expected))),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=provenance,
        )
