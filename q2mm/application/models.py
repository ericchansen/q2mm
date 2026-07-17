"""Immutable application-service result and configuration models."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from q2mm._canonical import canonical_fingerprint, json_value
from q2mm.models.forcefield import ForceField
from q2mm.models.problem import OptimizationProblem
from q2mm.models.results import OptimizationResult, deep_freeze

APPLICATION_SCHEMA_VERSION = 1
PROBLEM_FINGERPRINT_VERSION = 1


class ApplicationError(RuntimeError):
    """Base class for application-service contract failures."""


class ApplicationConfigurationError(ApplicationError, ValueError):
    """Raised when an application request is contradictory or incomplete."""


class ApplicationEvaluationError(ApplicationError):
    """Raised when generic evaluation cannot be performed."""


class ApplicationOptimizationError(ApplicationError):
    """Raised when generic optimization cannot be performed."""


class PersistenceError(ApplicationError):
    """Base class for application persistence failures."""


class OutputExistsError(PersistenceError, FileExistsError):
    """Raised when save would overwrite an existing output."""


class OutputFormatError(PersistenceError, ValueError):
    """Raised for unknown or incompatible force-field formats."""


def _safe_mapping(value: Mapping[str, Any], *, path: str) -> Mapping[str, Any]:
    try:
        normalized = json_value(value, strict=True, screen_secrets=True)
    except ValueError as exc:
        raise ApplicationConfigurationError(f"Unsafe resolved configuration at {path}: {exc}") from exc
    return deep_freeze(normalized)  # type: ignore[return-value]


@dataclass(frozen=True, eq=False)
class ResolvedBackendConfiguration:
    """Versioned immutable identity of the configured backend."""

    key: str
    name: str
    role: str
    version: str = ""
    capabilities: tuple[str, ...] = ()
    functional_forms: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.key or not self.name or not self.role:
            raise ApplicationConfigurationError("Resolved backend key, name, and role must be non-empty.")
        object.__setattr__(self, "capabilities", tuple(sorted(str(item) for item in self.capabilities)))
        object.__setattr__(self, "functional_forms", tuple(sorted(str(item) for item in self.functional_forms)))
        object.__setattr__(self, "details", _safe_mapping(self.details, path="backend.details"))


@dataclass(frozen=True, eq=False)
class ResolvedOptimizerConfiguration:
    """Versioned immutable optimizer identity and effective settings."""

    key: str
    label: str
    method: str
    settings: Mapping[str, Any]
    expected_result_gradient_mode: str
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.key or not self.method:
            raise ApplicationConfigurationError("Resolved optimizer key and method must be non-empty.")
        object.__setattr__(self, "settings", _safe_mapping(self.settings, path="optimizer.settings"))


@dataclass(frozen=True, eq=False)
class ResolvedWorkflowConfiguration:
    """Versioned immutable workflow identity and settings."""

    key: str
    settings: Mapping[str, Any]
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.key:
            raise ApplicationConfigurationError("Resolved workflow key must be non-empty.")
        object.__setattr__(self, "settings", _safe_mapping(self.settings, path="workflow.settings"))


@dataclass(frozen=True, eq=False)
class ResolvedExecutorConfiguration:
    """Versioned immutable objective-executor selection."""

    kind: str
    gradient_mode: str
    fd_step: float | None = None
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.kind not in {"python", "jax"}:
            raise ApplicationConfigurationError(f"Unknown executor kind {self.kind!r}.")
        if self.gradient_mode not in {"analytical", "finite_difference", "none"}:
            raise ApplicationConfigurationError(f"Unknown gradient mode {self.gradient_mode!r}.")
        if self.fd_step is not None and (not np.isfinite(self.fd_step) or self.fd_step <= 0):
            raise ApplicationConfigurationError("Executor fd_step must be positive and finite.")


@dataclass(frozen=True, eq=False)
class ResolvedExecutionConfiguration:
    """Complete resolved execution policy for an optimization run."""

    recipe_id: str
    backend: ResolvedBackendConfiguration
    optimizer: ResolvedOptimizerConfiguration
    workflow: ResolvedWorkflowConfiguration
    executor: ResolvedExecutorConfiguration
    overrides: tuple[str, ...] = ()
    regularization: float = 0.0
    n_evals: int = 1
    ratio_tol: float | None = None
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.recipe_id:
            raise ApplicationConfigurationError("Resolved recipe_id must be non-empty.")
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ApplicationConfigurationError("regularization must be non-negative and finite.")
        if not isinstance(self.n_evals, int) or self.n_evals < 0:
            raise ApplicationConfigurationError("n_evals must be a non-negative integer.")
        if self.ratio_tol is not None and (not np.isfinite(self.ratio_tol) or self.ratio_tol < 0):
            raise ApplicationConfigurationError("ratio_tol must be non-negative and finite or None.")
        object.__setattr__(self, "overrides", tuple(sorted(set(self.overrides))))


def _path_identity(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return Path(value).name


def _scientific_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        payload: dict[str, Any] = {}
        for item in dataclasses.fields(value):
            if not item.init and item.name in {
                "bonds_explicit",
                "angles_explicit",
                "torsions_explicit",
                "improper_torsions",
            }:
                continue
            field_value = getattr(value, item.name)
            if item.name in {"source_path", "path"}:
                payload[item.name] = _path_identity(field_value)
            else:
                payload[item.name] = _scientific_value(field_value)
        return payload
    if isinstance(value, Mapping):
        return {str(key): _scientific_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scientific_value(item) for item in value]
    return value


def molecule_fingerprint_payload(molecule: Any) -> dict[str, Any]:
    """Return all canonical scientific molecule inputs in stable order."""
    return {
        "symbols": list(molecule.symbols),
        "geometry": np.asarray(molecule.geometry, dtype=float).tolist(),
        "atom_types": list(molecule.atom_types),
        "charge": int(molecule.charge),
        "multiplicity": int(molecule.multiplicity),
        "name": str(molecule.name),
        "bond_tolerance": float(molecule.bond_tolerance),
        "partial_charges": None
        if molecule.partial_charges is None
        else [None if value is None else float(value) for value in molecule.partial_charges],
        "hessian": None if molecule.hessian is None else np.asarray(molecule.hessian, dtype=float).tolist(),
        "hessian_provenance": _scientific_value(molecule.hessian_provenance),
        "bonds": _scientific_value(molecule.bonds),
        "angles": _scientific_value(molecule.angles),
        "torsions": _scientific_value(molecule.torsions),
        "bonds_explicit": bool(molecule.bonds_explicit),
        "angles_explicit": bool(molecule.angles_explicit),
        "torsions_explicit": bool(molecule.torsions_explicit),
    }


def force_field_fingerprint_payload(force_field: ForceField, vector: np.ndarray) -> dict[str, Any]:
    """Return force-field structure, values, form, and source identity."""
    return {
        "name": force_field.name,
        "functional_form": force_field.functional_form.value,
        "source_format": force_field.source_format,
        "source_identity": _path_identity(force_field.source_path),
        "vector": np.asarray(vector, dtype=float).tolist(),
        "bonds": _scientific_value(force_field.bonds),
        "angles": _scientific_value(force_field.angles),
        "stretch_bends": _scientific_value(force_field.stretch_bends),
        "torsions": _scientific_value(force_field.torsions),
        "vdws": _scientific_value(force_field.vdws),
        "cmaps": _scientific_value(force_field.cmaps),
    }


def problem_fingerprint_payload(problem: OptimizationProblem) -> dict[str, Any]:
    """Return the versioned complete identity payload for *problem*."""
    layout = problem.layout
    vector = layout.vector(problem.starting_force_field)
    return {
        "schema_version": PROBLEM_FINGERPRINT_VERSION,
        "cases": [
            {
                "case_id": case.case_id,
                "stationary_point": case.stationary_point.value,
                "molecule": molecule_fingerprint_payload(case.molecule),
            }
            for case in problem.cases
        ],
        "starting_force_field": force_field_fingerprint_payload(problem.starting_force_field, vector),
        "layout": {
            "fingerprint": layout.fingerprint,
            "slots": [
                {
                    "index": slot.index,
                    "id": {
                        "family": slot.id.family,
                        "identity": list(slot.id.identity),
                        "occurrence": slot.id.occurrence,
                        "field": slot.id.field,
                    },
                    "kind": slot.kind.value,
                    "unit": slot.unit.value,
                    "name": slot.name,
                    "bounds": list(slot.bounds),
                    "step": slot.step,
                }
                for slot in layout.slots
            ],
        },
        "active_space": {
            "active_indices": [int(index) for index in problem.active_space.active_indices],
            "baseline": np.asarray(problem.active_space.baseline, dtype=float).tolist(),
        },
        "observations": [
            {
                "kind": observation.kind,
                "value": float(observation.value),
                "weight": float(observation.weight),
                "label": observation.label,
                "case_id": observation.case_id,
                "data_idx": int(observation.data_idx),
                "atom_indices": None
                if observation.atom_indices is None
                else [int(index) for index in observation.atom_indices],
            }
            for observation in problem.observations.values
        ],
    }


def problem_fingerprint(problem: OptimizationProblem) -> str:
    """Return a deterministic fingerprint over all scientific problem inputs."""
    return canonical_fingerprint(problem_fingerprint_payload(problem), screen_secrets=True)


def problem_input_fingerprints(problem: OptimizationProblem) -> Mapping[str, str]:
    """Return stable per-input fingerprints used by run provenance."""
    vector = problem.layout.vector(problem.starting_force_field)
    values = {
        f"molecule:{case.case_id}": canonical_fingerprint(
            molecule_fingerprint_payload(case.molecule), screen_secrets=True
        )
        for case in problem.cases
    }
    values["starting_force_field"] = canonical_fingerprint(
        force_field_fingerprint_payload(problem.starting_force_field, vector), screen_secrets=True
    )
    values["observations"] = canonical_fingerprint(problem_fingerprint_payload(problem)["observations"])
    values["active_space"] = canonical_fingerprint(problem_fingerprint_payload(problem)["active_space"])
    return MappingProxyType(values)


@dataclass(frozen=True, eq=False)
class OptimizationRun:
    """Application envelope around one canonical optimization result."""

    result: OptimizationResult
    final_force_field: ForceField
    configuration: ResolvedExecutionConfiguration
    problem_fingerprint: str
    layout_fingerprint: str
    input_fingerprints: Mapping[str, str]
    active_indices: tuple[int, ...]
    baseline: np.ndarray
    provenance: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = APPLICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        result = self.result
        if result.layout_fingerprint != self.layout_fingerprint:
            raise ApplicationOptimizationError("Result layout fingerprint does not match the run layout.")
        if not self.problem_fingerprint.startswith("sha256:"):
            raise ApplicationOptimizationError("problem_fingerprint must be a canonical SHA-256 fingerprint.")
        if result.n_params != len(result.final_params):
            raise ApplicationOptimizationError("Result final vector is not full-length.")
        active_indices = tuple(sorted(set(int(index) for index in self.active_indices)))
        if any(index < 0 or index >= result.n_params for index in active_indices):
            raise ApplicationOptimizationError("active_indices contain an out-of-range parameter index.")
        baseline = np.array(self.baseline, dtype=float, copy=True)
        if baseline.shape != (result.n_params,):
            raise ApplicationOptimizationError("Run baseline must be a full-length parameter vector.")
        inactive = np.setdiff1d(np.arange(result.n_params), np.asarray(active_indices, dtype=int))
        if not np.array_equal(result.initial_params, baseline):
            raise ApplicationOptimizationError("Result initial vector does not match the run baseline.")
        if not np.array_equal(result.final_params[inactive], baseline[inactive]):
            raise ApplicationOptimizationError("Result final vector changes frozen parameter slots.")
        baseline.setflags(write=False)
        object.__setattr__(self, "active_indices", active_indices)
        object.__setattr__(self, "baseline", baseline)
        final_layout = type(self)._layout_for(self.final_force_field)
        if final_layout.fingerprint != self.layout_fingerprint or len(final_layout) != result.n_params:
            raise ApplicationOptimizationError("Final force field structure does not match the result layout.")
        if not np.array_equal(final_layout.vector(self.final_force_field), result.final_params):
            raise ApplicationOptimizationError("Final force field values do not match result.final_params.")
        fingerprints = dict(self.input_fingerprints)
        if not fingerprints or any(not value.startswith("sha256:") for value in fingerprints.values()):
            raise ApplicationOptimizationError("input_fingerprints must contain canonical SHA-256 values.")
        object.__setattr__(self, "input_fingerprints", MappingProxyType(dict(sorted(fingerprints.items()))))
        object.__setattr__(self, "provenance", _safe_mapping(self.provenance, path="run.provenance"))

    @staticmethod
    def _layout_for(force_field: ForceField) -> Any:
        from q2mm.models.parameters import ParameterLayout

        return ParameterLayout.from_force_field(force_field)

    @property
    def optimization_result(self) -> OptimizationResult:
        """Alias emphasizing that the envelope contains one canonical result."""
        return self.result

    @property
    def backend_configuration(self) -> ResolvedBackendConfiguration:
        """Resolved backend configuration."""
        return self.configuration.backend

    @property
    def optimizer_configuration(self) -> ResolvedOptimizerConfiguration:
        """Resolved optimizer configuration."""
        return self.configuration.optimizer

    @property
    def workflow_configuration(self) -> ResolvedWorkflowConfiguration:
        """Resolved workflow configuration."""
        return self.configuration.workflow

    @property
    def executor_configuration(self) -> ResolvedExecutorConfiguration:
        """Resolved executor configuration."""
        return self.configuration.executor


@dataclass(frozen=True)
class SavedOutput:
    """Paths and semantic format produced by :func:`q2mm.application.save`."""

    path: Path
    format: str
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if self.manifest_path is not None:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path))

    @property
    def force_field_path(self) -> Path:
        """Path to the serialized force field."""
        return self.path


__all__ = [
    "APPLICATION_SCHEMA_VERSION",
    "ApplicationConfigurationError",
    "ApplicationError",
    "ApplicationEvaluationError",
    "ApplicationOptimizationError",
    "OptimizationRun",
    "OutputExistsError",
    "OutputFormatError",
    "PersistenceError",
    "ResolvedBackendConfiguration",
    "ResolvedExecutionConfiguration",
    "ResolvedExecutorConfiguration",
    "ResolvedOptimizerConfiguration",
    "ResolvedWorkflowConfiguration",
    "SavedOutput",
    "force_field_fingerprint_payload",
    "molecule_fingerprint_payload",
    "problem_fingerprint",
    "problem_fingerprint_payload",
    "problem_input_fingerprints",
]
