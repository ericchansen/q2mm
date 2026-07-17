"""Generic, immutable construction of Q2MM optimization problems."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, cast

import numpy as np

from q2mm._canonical import canonical_fingerprint
from q2mm._provenance import freeze_json_mapping
from q2mm.backends.contracts import (
    Backend,
    BackendRole,
    Capability,
    FrequencyRequest,
    PreparationRequest,
)
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet, observation_payload
from q2mm.models.parameters import (
    ActiveParameterSpace,
    ParameterLayout,
    opt_substructure_membership,
)
from q2mm.models.problem import (
    OptimizationProblem,
    PreparationProvenance,
    StationaryPointKind,
    TrainingCase,
)
from q2mm.models.seminario import qfuerza_fresh, qfuerza_into

_COMPATIBILITY_PROFILE = "repository-geometry-eigenmatrix-v1"
_EXPLICIT_PROFILE = "explicit-observation-set-v1"
_MATCHED_FREQUENCY_PROFILE = "matched-frequency-v1"
_BEND_FIELDS = frozenset({"force_constant", "equilibrium"})

Initialize = Literal["provided", "qfuerza"]
ActiveParameters = Literal["all"] | ForceField | ActiveParameterSpace


class PreparationError(ValueError):
    """Raised when a convenience preparation request is invalid."""


@dataclass(frozen=True)
class QFuerzaConfig:
    """Scientifically meaningful QFUERZA initialization settings."""

    strategy: Literal["fuerza", "qfuerza"] = "qfuerza"
    zero_torsions: bool = True
    au_hessian: bool = True
    invalid_policy: Literal["keep", "skip"] = "keep"
    replace_with: float = 1.0

    def __post_init__(self) -> None:
        if self.strategy not in {"fuerza", "qfuerza"}:
            raise PreparationError("QFuerzaConfig.strategy must be 'fuerza' or 'qfuerza'.")
        if self.invalid_policy not in {"keep", "skip"}:
            raise PreparationError("QFuerzaConfig.invalid_policy must be 'keep' or 'skip'.")
        if not isinstance(self.zero_torsions, bool) or not isinstance(self.au_hessian, bool):
            raise PreparationError("QFuerzaConfig boolean settings must be bool values.")
        if not math.isfinite(self.replace_with) or self.replace_with <= 0.0:
            raise PreparationError("QFuerzaConfig.replace_with must be positive and finite.")


@dataclass(frozen=True)
class ObservationRecipe:
    """Marker for the closed set of supported observation recipes."""


@dataclass(frozen=True)
class MoleculeObservations(ObservationRecipe):
    """Geometry plus full-eigenmatrix observations from canonical molecules."""

    name: str = field(default="MoleculeObservations", init=False)
    profile: str = field(default=_COMPATIBILITY_PROFILE, init=False)


@dataclass(frozen=True)
class MatchedFrequencyObservations(ObservationRecipe):
    """Match sorted real QM modes to a configured MM backend's real modes."""

    qm_frequencies: Sequence[float]
    backend: Backend | str
    threshold: float = 50.0
    weight: float = 0.001
    name: str = field(default="MatchedFrequencyObservations", init=False)
    profile: str = field(default=_MATCHED_FREQUENCY_PROFILE, init=False)

    def __post_init__(self) -> None:
        frequencies = tuple(float(value) for value in self.qm_frequencies)
        if not frequencies or any(not math.isfinite(value) for value in frequencies):
            raise PreparationError("MatchedFrequencyObservations.qm_frequencies must be finite and non-empty.")
        if not math.isfinite(self.threshold):
            raise PreparationError("MatchedFrequencyObservations.threshold must be finite.")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise PreparationError("MatchedFrequencyObservations.weight must be positive and finite.")
        if isinstance(self.backend, str) and not self.backend:
            raise PreparationError("MatchedFrequencyObservations.backend must be non-empty.")
        object.__setattr__(self, "qm_frequencies", frequencies)


def _scientific_value(value: object) -> object:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if dataclasses.is_dataclass(value):
        return {
            item.name: _scientific_value(getattr(value, item.name))
            for item in dataclasses.fields(value)
            if item.name not in {"bonds_explicit", "angles_explicit", "torsions_explicit", "improper_torsions"}
        }
    if isinstance(value, Mapping):
        return {str(key): _scientific_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_scientific_value(item) for item in value]
    return value


def _fingerprint(value: object) -> str:
    return canonical_fingerprint(_scientific_value(value), screen_secrets=True)


def _normalize_molecules(value: Molecule | Sequence[Molecule]) -> tuple[Molecule, ...]:
    if isinstance(value, Molecule):
        return (value,)
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PreparationError("molecules must be a Molecule or a sequence of Molecule values.")
    molecules = tuple(value)
    if not molecules:
        raise PreparationError("prepare requires at least one molecule.")
    if any(not isinstance(molecule, Molecule) for molecule in molecules):
        raise PreparationError("Every item in molecules must be a Molecule.")
    return molecules


def _stationary_point(value: str | StationaryPointKind) -> StationaryPointKind:
    if isinstance(value, StationaryPointKind):
        return value
    if not isinstance(value, str):
        raise PreparationError(
            "stationary_point must name one kind for the complete set; use direct OptimizationProblem "
            "construction for mixed ground-state/transition-state sets."
        )
    normalized = value.strip().lower().replace("-", "_")
    try:
        return StationaryPointKind(normalized)
    except ValueError as exc:
        raise PreparationError(
            "stationary_point must be 'ground_state' or 'transition_state'; mixed sets require direct "
            "OptimizationProblem construction."
        ) from exc


def _functional_form(value: str | FunctionalForm | None) -> FunctionalForm | None:
    if value is None or isinstance(value, FunctionalForm):
        return value
    try:
        return FunctionalForm(value)
    except ValueError as exc:
        raise PreparationError(f"Unknown functional_form {value!r}.") from exc


def _resolve_case_ids(
    molecules: tuple[Molecule, ...],
    case_ids: Sequence[str] | None,
    observations: ObservationSet | ObservationRecipe | None,
) -> tuple[str, ...]:
    if case_ids is None:
        if len(molecules) > 1 and isinstance(observations, ObservationSet):
            raise PreparationError("Explicit observations for multiple molecules require explicit case_ids.")
        return tuple(str(index) for index in range(len(molecules)))
    if isinstance(case_ids, (str, bytes)):
        raise PreparationError("case_ids must be a sequence of complete identifiers, not one string.")
    resolved = tuple(str(case_id) for case_id in case_ids)
    if len(resolved) != len(molecules):
        raise PreparationError(f"case_ids has length {len(resolved)}; expected {len(molecules)}.")
    if any(not case_id for case_id in resolved) or len(set(resolved)) != len(resolved):
        raise PreparationError("case_ids must be non-empty and unique.")
    return resolved


def _resolve_backend(recipe: MatchedFrequencyObservations, form: FunctionalForm) -> Backend:
    backend: Backend
    if isinstance(recipe.backend, str):
        from q2mm.backends.registry import load_backend

        try:
            backend = load_backend(recipe.backend)
        except Exception as exc:
            raise PreparationError(f"Could not load matched-frequency backend {recipe.backend!r}: {exc}") from exc
    else:
        backend = recipe.backend
    if not hasattr(backend, "info") or not hasattr(backend, "prepare"):
        raise PreparationError("MatchedFrequencyObservations.backend must implement the Backend protocol.")
    if backend.info.role is not BackendRole.MM:
        raise PreparationError("MatchedFrequencyObservations requires an MM backend.")
    if Capability.FREQUENCIES not in backend.info.capabilities:
        raise PreparationError("Matched-frequency backend must declare the FREQUENCIES capability.")
    if not backend.info.supports_form(form.value):
        raise PreparationError(f"Matched-frequency backend does not support functional form {form.value!r}.")
    return backend


def _active_space(
    active_parameters: ActiveParameters,
    layout: ParameterLayout,
    force_field: ForceField,
) -> ActiveParameterSpace:
    if active_parameters == "all":
        return ActiveParameterSpace.all_active(layout, force_field)
    if isinstance(active_parameters, ForceField):
        membership = opt_substructure_membership(force_field, active_parameters)
        return ActiveParameterSpace.from_membership(layout, force_field, membership)
    if isinstance(active_parameters, ActiveParameterSpace):
        if active_parameters.layout != layout:
            raise PreparationError("Explicit ActiveParameterSpace does not match the generated ParameterLayout.")
        return ActiveParameterSpace(
            layout=layout,
            baseline=layout.vector(force_field),
            active_indices=active_parameters.active_indices,
        )
    raise PreparationError("active_parameters must be 'all', a ForceField subset, or an ActiveParameterSpace.")


def _qfuerza_settings(config: QFuerzaConfig, stationary_point: StationaryPointKind) -> dict[str, object]:
    return {
        "strategy": config.strategy,
        "zero_torsions": config.zero_torsions,
        "au_hessian": config.au_hessian,
        "invalid_policy": config.invalid_policy,
        "replace_with": config.replace_with,
        "invert_ts_curvature": stationary_point is StationaryPointKind.TRANSITION_STATE,
    }


def _project_template(
    force_field: ForceField,
    molecules: tuple[Molecule, ...],
    space: ActiveParameterSpace,
    config: QFuerzaConfig,
    stationary_point: StationaryPointKind,
) -> ForceField:
    layout = space.layout
    before = layout.vector(force_field)
    projected = qfuerza_into(
        force_field,
        molecules,
        active_bonds=space.active_owner_indices("bonds", fields=_BEND_FIELDS),
        active_angles=space.active_owner_indices("angles", fields=_BEND_FIELDS),
        active_torsions=space.active_owner_indices("torsions"),
        strategy=config.strategy,
        zero_torsions=config.zero_torsions,
        au_hessian=config.au_hessian,
        invalid_policy=config.invalid_policy,
        invert_ts_curvature=stationary_point is StationaryPointKind.TRANSITION_STATE,
        replace_with=config.replace_with,
    )
    projected_vector = layout.vector(projected)
    merged = before.copy()
    merged[space.active_indices] = projected_vector[space.active_indices]
    if not np.array_equal(
        merged[np.setdiff1d(np.arange(len(layout)), space.active_indices)],
        before[np.setdiff1d(np.arange(len(layout)), space.active_indices)],
    ):
        raise RuntimeError("QFUERZA template merge changed an inactive scalar.")
    return layout.replace(force_field, merged)


def _parameter_counts(
    layout: ParameterLayout,
    active_indices: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> dict[str, dict[str, int]]:
    active = set(int(index) for index in active_indices)
    result: dict[str, dict[str, int]] = {}
    for slot in layout:
        bucket = result.setdefault(slot.kind.value, {"overwritten": 0, "retained": 0, "frozen": 0})
        if slot.index not in active:
            bucket["frozen"] += 1
            continue
        before_bits = np.asarray(before[slot.index], dtype=np.float64).tobytes()
        after_bits = np.asarray(after[slot.index], dtype=np.float64).tobytes()
        bucket["retained" if before_bits == after_bits else "overwritten"] += 1
    return result


def _observation_payload(observations: ObservationSet) -> list[dict[str, object]]:
    return [observation_payload(observation) for observation in observations.values]


def _matched_observations(
    recipe: MatchedFrequencyObservations,
    backend: Backend,
    molecule: Molecule,
    case_id: str,
    force_field: ForceField,
    layout: ParameterLayout,
) -> tuple[ObservationSet, dict[str, object]]:
    prepared = backend.prepare(PreparationRequest(case_id=case_id, molecule=molecule, force_field=force_field))
    result = prepared.frequencies(FrequencyRequest(parameters=layout.vector(force_field)))
    qm_real = sorted(value for value in recipe.qm_frequencies if value > recipe.threshold)
    mm_real_indices = sorted(index for index, value in enumerate(result.frequencies) if float(value) > recipe.threshold)
    count = min(len(qm_real), len(mm_real_indices))
    observations = ObservationSet()
    for index in range(count):
        observations = observations.with_frequency(
            qm_real[index],
            data_idx=mm_real_indices[index],
            weight=recipe.weight,
            case_id=case_id,
        )
    provenance = backend.info.provenance
    backend_key = provenance.backend if provenance is not None else backend.info.name
    return observations, {
        "name": recipe.name,
        "profile": recipe.profile,
        "threshold": recipe.threshold,
        "weight": recipe.weight,
        "qm_frequency_count": len(recipe.qm_frequencies),
        "qm_frequency_fingerprint": _fingerprint(recipe.qm_frequencies),
        "matched_count": count,
        "backend": {
            "key": backend_key,
            "name": backend.info.name,
            "role": backend.info.role.value,
            "version": "" if provenance is None else provenance.version,
            "capabilities": sorted(capability.value for capability in backend.info.capabilities),
            "functional_forms": sorted(backend.info.functional_forms),
        },
    }


def prepare(
    molecules: Molecule | Sequence[Molecule],
    *,
    stationary_point: str | StationaryPointKind,
    force_field: ForceField | None = None,
    active_parameters: ActiveParameters = "all",
    observations: ObservationSet | ObservationRecipe | None = None,
    case_ids: Sequence[str] | None = None,
    functional_form: str | FunctionalForm | None = None,
    initialize: Initialize | None = None,
    qfuerza: QFuerzaConfig | None = None,
) -> OptimizationProblem:
    """Build one canonical immutable optimization problem from user inputs."""
    molecule_values = _normalize_molecules(molecules)
    point = _stationary_point(stationary_point)
    requested_form = _functional_form(functional_form)
    ids = _resolve_case_ids(molecule_values, case_ids, observations)
    if observations is not None and not isinstance(observations, (ObservationSet, ObservationRecipe)):
        raise PreparationError("observations must be an ObservationSet or a supported ObservationRecipe.")
    if isinstance(observations, ObservationRecipe) and type(observations) not in {
        MoleculeObservations,
        MatchedFrequencyObservations,
    }:
        raise PreparationError(f"Unsupported observation recipe {type(observations).__name__}.")

    config = QFuerzaConfig() if qfuerza is None else qfuerza
    if not isinstance(config, QFuerzaConfig):
        raise PreparationError("qfuerza must be a QFuerzaConfig.")

    matched_recipe = (
        cast(MatchedFrequencyObservations, observations)
        if isinstance(observations, MatchedFrequencyObservations)
        else None
    )
    if matched_recipe is not None and len(molecule_values) != 1:
        raise PreparationError("MatchedFrequencyObservations currently requires exactly one molecule.")

    if force_field is None:
        if requested_form is None:
            raise PreparationError("functional_form is required when no force_field is supplied.")
        if len(molecule_values) != 1:
            raise PreparationError(
                "Fresh QFUERZA preparation requires exactly one molecule; supply a shared force_field "
                "template for multi-molecule averaging."
            )
        if initialize not in {None, "qfuerza"}:
            raise PreparationError("Fresh preparation implies initialize='qfuerza'.")
        if active_parameters != "all":
            raise PreparationError("Fresh preparation makes all generated parameters active.")
        initialize_source: Initialize = "qfuerza"
        resolved_form = requested_form
    else:
        if not isinstance(force_field, ForceField):
            raise PreparationError("force_field must be a ForceField or None.")
        resolved_form = force_field.functional_form
        if requested_form is not None and requested_form is not resolved_form:
            raise PreparationError(
                f"functional_form={requested_form.value!r} conflicts with supplied force field form "
                f"{resolved_form.value!r}."
            )
        if initialize not in {"provided", "qfuerza"}:
            raise PreparationError("A supplied force_field requires initialize='provided' or 'qfuerza'.")
        initialize_source = initialize
    if initialize_source == "provided" and qfuerza is not None:
        raise PreparationError("qfuerza settings cannot be applied with initialize='provided'.")

    backend = _resolve_backend(matched_recipe, resolved_form) if matched_recipe is not None else None
    needs_hessian = (
        initialize_source == "qfuerza" or observations is None or isinstance(observations, MoleculeObservations)
    )
    if needs_hessian:
        missing = [ids[index] for index, molecule in enumerate(molecule_values) if molecule.hessian is None]
        if missing:
            raise PreparationError(f"Molecules missing required canonical Hessians: {missing}.")

    if force_field is None:
        template = ForceField.create_for_molecule(
            molecule_values[0],
            name=f"QFUERZA FF for {molecule_values[0].name}",
            functional_form=resolved_form,
        )
        pre_layout = ParameterLayout.from_force_field(template)
        before_vector = pre_layout.vector(template)
        starting_force_field = qfuerza_fresh(
            molecule_values[0],
            functional_form=resolved_form,
            strategy=config.strategy,
            zero_torsions=config.zero_torsions,
            au_hessian=config.au_hessian,
            invalid_policy=config.invalid_policy,
            invert_ts_curvature=point is StationaryPointKind.TRANSITION_STATE,
            replace_with=config.replace_with,
        )
        layout = ParameterLayout.from_force_field(starting_force_field)
        if layout != pre_layout:
            raise RuntimeError("Fresh QFUERZA changed force-field layout.")
        initial_space = ActiveParameterSpace.all_active(layout, starting_force_field)
    else:
        layout = ParameterLayout.from_force_field(force_field)
        initial_space = _active_space(active_parameters, layout, force_field)
        before_vector = layout.vector(force_field)
        starting_force_field = (
            force_field
            if initialize_source == "provided"
            else _project_template(force_field, molecule_values, initial_space, config, point)
        )
    after_vector = layout.vector(starting_force_field)
    active_space = ActiveParameterSpace(
        layout=layout,
        baseline=after_vector,
        active_indices=initial_space.active_indices,
    )

    if observations is None or isinstance(observations, MoleculeObservations):
        resolved_observations = ObservationSet.from_molecules(
            molecule_values,
            ids,
            eigenmatrix_diagonal_only=False,
        )
        recipe_details: dict[str, object] = {
            "name": "MoleculeObservations",
            "profile": _COMPATIBILITY_PROFILE,
            "geometry": True,
            "eigenmatrix": "full",
        }
        profile = _COMPATIBILITY_PROFILE
    elif isinstance(observations, ObservationSet):
        resolved_observations = observations
        recipe_details = {"name": "explicit", "profile": _EXPLICIT_PROFILE}
        profile = _EXPLICIT_PROFILE
    else:
        assert matched_recipe is not None and backend is not None
        resolved_observations, recipe_details = _matched_observations(
            matched_recipe,
            backend,
            molecule_values[0],
            ids[0],
            starting_force_field,
            layout,
        )
        profile = _MATCHED_FREQUENCY_PROFILE

    unknown_case_ids = sorted({observation.case_id for observation in resolved_observations.values}.difference(ids))
    if unknown_case_ids:
        raise PreparationError(f"Observations reference unknown case IDs: {unknown_case_ids}.")

    input_fingerprints: dict[str, str] = {
        f"molecule:{case_id}": _fingerprint(molecule) for case_id, molecule in zip(ids, molecule_values, strict=True)
    }
    input_fingerprints["force_field"] = _fingerprint(
        force_field if force_field is not None else {"generated_from": ids[0], "form": resolved_form.value}
    )
    input_fingerprints["active_indices"] = _fingerprint(active_space.active_indices)
    input_fingerprints["observations"] = _fingerprint(_observation_payload(resolved_observations))
    provenance = PreparationProvenance(
        profile=profile,
        initialize_source=initialize_source,
        functional_form=resolved_form.value,
        qfuerza_settings=(_qfuerza_settings(config, point) if initialize_source == "qfuerza" else {}),
        pre_qfuerza_vector_fingerprint=_fingerprint(before_vector),
        parameter_counts=_parameter_counts(
            layout,
            active_space.active_indices,
            before_vector,
            after_vector,
        ),
        stationary_points=tuple(point.value for _ in molecule_values),
        case_ids=ids,
        input_fingerprints=input_fingerprints,
        observation_recipe=freeze_json_mapping(
            recipe_details,
            path="prepare.observation_recipe",
        ),
    )
    return OptimizationProblem(
        cases=tuple(
            TrainingCase(case_id=case_id, molecule=molecule, stationary_point=point)
            for case_id, molecule in zip(ids, molecule_values, strict=True)
        ),
        starting_force_field=starting_force_field,
        layout=layout,
        active_space=active_space,
        observations=resolved_observations,
        preparation_provenance=provenance,
    )


__all__ = [
    "MatchedFrequencyObservations",
    "MoleculeObservations",
    "ObservationRecipe",
    "PreparationError",
    "QFuerzaConfig",
    "prepare",
]
