"""Dependency-light support used by the executable source-tree examples."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyResult,
    EnergyUnit,
    GeometryResult,
    HessianResult,
    HessianUnit,
    LengthUnit,
    PreparationRequest,
)
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import Evaluation, ObjectiveEvaluator

_PROVENANCE = BackendProvenance(backend="example-bounded-echo", role=BackendRole.MM)
_INFO = BackendInfo(
    name="Bounded example echo backend",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY, Capability.MINIMIZE, Capability.HESSIAN}),
    functional_forms=frozenset({"harmonic", "mm3"}),
    provenance=_PROVENANCE,
)


class ExampleConfigurationError(ValueError):
    """An example was invoked with missing or contradictory paths/options."""


class _EchoPrepared(AbstractPreparedBackend):
    def __init__(self, request: PreparationRequest) -> None:
        if request.force_field is None:
            raise ExampleConfigurationError("The bounded MM echo backend requires a prepared force field.")
        super().__init__(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )

    def _energy(self, request: object) -> EnergyResult:
        return EnergyResult(energy=0.0, unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROVENANCE)

    def _minimize(self, request: object) -> GeometryResult:
        return GeometryResult(
            energy=0.0,
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=self.molecule.symbols,
            coordinates=self.molecule.geometry,
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=_PROVENANCE,
        )

    def _hessian(self, request: object) -> HessianResult:
        if self.molecule.hessian is None:
            raise ExampleConfigurationError(f"Case {self.case_id!r} has no Hessian for bounded evaluation.")
        return HessianResult(
            hessian=self.molecule.hessian,
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROVENANCE,
        )


class BoundedEchoBackend:
    """Deterministic MM backend for one-evaluation installed-wheel smoke runs."""

    info = _INFO

    def prepare(self, request: PreparationRequest) -> _EchoPrepared:
        """Prepare a deterministic case without changing its scientific inputs."""
        return _EchoPrepared(request)


class BoundedExampleOptimizer:
    """Enter the real objective once and return its unchanged full vector."""

    def __init__(self) -> None:
        self.entered = False

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        """Evaluate once, proving optimizer entry without claiming convergence."""
        self.entered = True
        initial = np.array(space.baseline, copy=True)
        score = evaluator.value(initial)
        return OptimizationResult(
            success=True,
            message="bounded CI optimizer entry; no convergence claim",
            initial_score=score,
            final_score=score,
            n_iterations=1,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=initial,
            final_params=initial,
            history=(score,),
            method="bounded-example-one-evaluation",
            gradient_mode="none",
        )


def json_safe(value: Any) -> Any:
    """Convert immutable/domain values to deterministic JSON-safe values."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {field.name: json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    return value


def evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    """Return objective totals and categories without inventing derived metrics."""
    return {
        "total": float(evaluation.total),
        "data_value": float(evaluation.data_value),
        "regularization": float(evaluation.regularization),
        "categories": {key: float(value) for key, value in evaluation.category_scores.items()},
    }


def parameter_counts(problem: OptimizationProblem) -> dict[str, dict[str, int]]:
    """Count active and frozen scalar slots by canonical parameter kind."""
    active = set(int(index) for index in problem.active_space.active_indices)
    totals = Counter(slot.kind.value for slot in problem.layout.slots)
    active_counts = Counter(problem.layout.slots[index].kind.value for index in active)
    return {
        kind: {
            "active": active_counts[kind],
            "frozen": totals[kind] - active_counts[kind],
            "total": totals[kind],
        }
        for kind in sorted(totals)
    }


def with_final_force_field(problem: OptimizationProblem, final_vector: np.ndarray) -> OptimizationProblem:
    """Return a problem whose baseline is the supplied full final vector."""
    vector = np.array(final_vector, dtype=float, copy=True)
    force_field = problem.layout.replace(problem.starting_force_field, vector)
    active_space = ActiveParameterSpace(
        layout=problem.layout,
        baseline=vector,
        active_indices=problem.active_space.active_indices,
    )
    return dataclasses.replace(problem, starting_force_field=force_field, active_space=active_space)
