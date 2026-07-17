"""Small, lazy public facade for Q2MM workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from q2mm.application.models import OptimizationRun
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.problem import OptimizationProblem, StationaryPointKind
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import Evaluation

if TYPE_CHECKING:
    from q2mm.application.models import SavedOutput
    from q2mm.backends.contracts import (
        Backend,
        Capability,
        CoordinateGradientResult,
        EnergyResult,
        FrequencyResult,
        GeometryResult,
        HessianResult,
    )
    from q2mm.models.observations import ObservationSet
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.optimizers.catalog import OptimizerSpec
    from q2mm.optimizers.protocols import _Optimizer
    from q2mm.preparation import ObservationRecipe, QFuerzaConfig
    from q2mm.workflows import Workflow

try:
    __version__ = version("q2mm")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

Property = Literal["energy", "coordinate_gradient", "hessian", "frequencies", "geometry_optimization"]


def prepare(
    molecules: Molecule | Sequence[Molecule],
    *,
    stationary_point: str | StationaryPointKind,
    force_field: ForceField | None = None,
    active_parameters: Literal["all"] | ForceField | ActiveParameterSpace = "all",
    observations: ObservationSet | ObservationRecipe | None = None,
    case_ids: Sequence[str] | None = None,
    functional_form: str | FunctionalForm | None = None,
    initialize: Literal["qfuerza", "provided"] | None = None,
    qfuerza: QFuerzaConfig | None = None,
) -> OptimizationProblem:
    """Lazily delegate generic immutable problem construction."""
    from q2mm.preparation import prepare as prepare_problem

    return prepare_problem(
        molecules,
        stationary_point=stationary_point,
        force_field=force_field,
        active_parameters=active_parameters,
        observations=observations,
        case_ids=case_ids,
        functional_form=functional_form,
        initialize=initialize,
        qfuerza=qfuerza,
    )


@overload
def evaluate(
    target: OptimizationProblem,
    *,
    backend: Backend | str,
    executor: Literal["auto", "python", "jax"] = "auto",
    **options: Any,
) -> Evaluation: ...


@overload
def evaluate(
    target: Molecule,
    *,
    backend: Backend,
    property: Property | None = None,
    capability: Capability | str | None = None,
    **options: Any,
) -> EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult: ...


def evaluate(
    target: OptimizationProblem | Molecule,
    *,
    backend: Backend | str,
    property: Property | None = None,
    capability: Capability | str | None = None,
    executor: Literal["auto", "python", "jax"] = "auto",
    **options: Any,
) -> Evaluation | EnergyResult | CoordinateGradientResult | HessianResult | FrequencyResult | GeometryResult:
    """Lazily delegate objective or reference-property evaluation."""
    from q2mm.application.evaluation import evaluate as evaluate_target

    return evaluate_target(
        target,
        backend,
        property=property,
        capability=capability,
        executor=executor,
        **options,
    )


def optimize(
    problem: OptimizationProblem,
    *,
    backend: Backend | str,
    recipe: Literal["recommended", "explicit"] = "recommended",
    optimizer: str | OptimizerSpec | _Optimizer | None = None,
    optimizer_options: Mapping[str, Any] | None = None,
    workflow: str | Workflow | None = None,
    workflow_options: Mapping[str, Any] | None = None,
    executor: Literal["auto", "python", "jax"] = "auto",
    backend_options: Mapping[str, object] | None = None,
    **options: Any,
) -> OptimizationRun:
    """Lazily delegate canonical optimization execution."""
    from q2mm.application.optimization import optimize as optimize_problem

    return optimize_problem(
        problem,
        backend,
        recipe=recipe,
        optimizer=optimizer,
        optimizer_options=optimizer_options,
        workflow=workflow,
        workflow_options=workflow_options,
        executor=executor,
        backend_options=backend_options,
        **options,
    )


def save(
    value: OptimizationRun | ForceField,
    path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> SavedOutput:
    """Lazily delegate semantic force-field persistence."""
    from q2mm.application.persistence import save as save_output

    return save_output(value, path, format=format, overwrite=overwrite)


__all__ = [
    "__version__",
    "Molecule",
    "ForceField",
    "OptimizationProblem",
    "Evaluation",
    "OptimizationResult",
    "OptimizationRun",
    "prepare",
    "evaluate",
    "optimize",
    "save",
]
