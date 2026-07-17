"""Immutable training-case and optimization-problem model for Q2MM.

:class:`OptimizationProblem` is the one immutable bundle produced by generic
preparation, a benchmark-system loader, or direct advanced construction and
consumed by objective/optimizer/workflow code: training cases (each a canonical
:class:`~q2mm.models.molecule.Molecule` with a stable ID and explicit
ground-state/transition-state kind), the starting force field, its
:class:`~q2mm.models.parameters.ParameterLayout`, an
:class:`~q2mm.models.parameters.ActiveParameterSpace`, and the
:class:`~q2mm.models.observations.ObservationSet` to fit against.

There is no ``HessianData`` wrapper here — a case's Hessian and its
provenance are read directly off ``case.molecule.hessian`` /
``case.molecule.hessian_provenance`` (the PR #309 model, preserved as
the one Hessian/provenance vocabulary).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from q2mm._provenance import freeze_json_mapping
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterKind, ParameterLayout
from q2mm.models.publication import PublicationMetadata

__all__ = ["StationaryPointKind", "TrainingCase", "PreparationProvenance", "OptimizationProblem"]


class StationaryPointKind(str, Enum):
    """Whether a training case is a ground state or a transition state.

    This field — not a backend, filename, or heuristic — is what drives
    TS curvature inversion (see
    :func:`q2mm.models.hessian.invert_ts_curvature`) and QFUERZA's TS
    projection semantics.
    """

    GROUND_STATE = "ground_state"
    TRANSITION_STATE = "transition_state"


@dataclass(frozen=True)
class TrainingCase:
    """One training-set molecule with a stable identity and stationary-point kind.

    Attributes:
        case_id: Stable, non-empty, unique-within-problem identifier
            (e.g. a filename stem or system-specific label). Observations
            reference training cases by this explicit, stable ID via
            :attr:`~q2mm.models.observations.Observation.case_id` — never
            by a positional index into :attr:`OptimizationProblem.cases`.
        molecule: The canonical, immutable :class:`Molecule` for this
            case, including its Hessian/provenance when applicable.
        stationary_point: Ground-state or transition-state — see
            :class:`StationaryPointKind`.
        source_id: Optional authoritative source membership ID. Publication
            problems use this to preserve semantic case membership while
            retaining legacy compatibility ``case_id`` values.

    """

    case_id: str
    molecule: Molecule
    stationary_point: StationaryPointKind
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("TrainingCase.case_id must be a non-empty string.")
        if not isinstance(self.molecule, Molecule):
            raise TypeError(f"TrainingCase.molecule must be a Molecule, got {type(self.molecule).__name__}.")
        if self.source_id is not None:
            source_id = str(self.source_id).strip()
            if not source_id:
                raise ValueError("TrainingCase.source_id must be non-empty when provided.")
            object.__setattr__(self, "source_id", source_id)


@dataclass(frozen=True)
class PreparationProvenance:
    """Immutable, path-free audit of how a convenience problem was prepared."""

    profile: str
    initialize_source: str
    functional_form: str
    pre_qfuerza_vector_fingerprint: str
    qfuerza_settings: Mapping[str, object] = field(default_factory=dict)
    parameter_counts: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    stationary_points: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    input_fingerprints: Mapping[str, str] = field(default_factory=dict)
    observation_recipe: Mapping[str, object] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("PreparationProvenance.profile must be non-empty.")
        if self.initialize_source not in {"provided", "qfuerza"}:
            raise ValueError("PreparationProvenance.initialize_source must be 'provided' or 'qfuerza'.")
        if not self.functional_form:
            raise ValueError("PreparationProvenance.functional_form must be non-empty.")
        fingerprint = self.pre_qfuerza_vector_fingerprint
        if not fingerprint.startswith("sha256:"):
            raise ValueError("pre_qfuerza_vector_fingerprint must be a canonical SHA-256 fingerprint.")
        valid_kinds = {kind.value for kind in ParameterKind}
        for kind, counts in self.parameter_counts.items():
            if kind not in valid_kinds:
                raise ValueError(f"PreparationProvenance.parameter_counts has unknown kind {kind!r}.")
            if set(counts) != {"overwritten", "retained", "frozen"}:
                raise ValueError(
                    "Each PreparationProvenance.parameter_counts entry must contain overwritten, retained, and frozen."
                )
            if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
                raise ValueError("PreparationProvenance parameter counts must be non-negative integers.")
        if not self.input_fingerprints or any(
            not isinstance(value, str) or not value.startswith("sha256:") for value in self.input_fingerprints.values()
        ):
            raise ValueError("PreparationProvenance.input_fingerprints must contain canonical SHA-256 values.")
        stationary_points = tuple(str(value) for value in self.stationary_points)
        valid_stationary_points = {kind.value for kind in StationaryPointKind}
        if any(value not in valid_stationary_points for value in stationary_points):
            raise ValueError("PreparationProvenance.stationary_points contains an unknown stationary-point kind.")
        case_ids = tuple(str(value) for value in self.case_ids)
        if not case_ids or any(not value for value in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("PreparationProvenance.case_ids must be non-empty and unique.")
        object.__setattr__(self, "stationary_points", stationary_points)
        object.__setattr__(self, "case_ids", case_ids)
        for name in ("qfuerza_settings", "parameter_counts", "input_fingerprints", "observation_recipe"):
            object.__setattr__(
                self,
                name,
                freeze_json_mapping(getattr(self, name), path=f"PreparationProvenance.{name}"),
            )


@dataclass(frozen=True, eq=False)
class OptimizationProblem:
    """Immutable bundle of training cases, starting force field, layout, active space, and observations.

    Construction validates:

    - at least one :class:`TrainingCase`, with unique, non-empty case IDs;
    - every :class:`~q2mm.models.observations.Observation`'s ``case_id``
      resolves to exactly one :class:`TrainingCase` in :attr:`cases`;
    - :attr:`layout` structurally matches :attr:`starting_force_field`
      (``layout.vector(starting_force_field)`` has length
      ``len(layout)``); and
    - :attr:`active_space` was built over this exact :attr:`layout`
      (same slot structure), with a baseline of matching length.

    """

    cases: tuple[TrainingCase, ...]
    starting_force_field: ForceField
    layout: ParameterLayout
    active_space: ActiveParameterSpace
    observations: ObservationSet
    preparation_provenance: PreparationProvenance | None = None
    publication_metadata: PublicationMetadata | None = None

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        object.__setattr__(self, "cases", cases)
        if not cases:
            raise ValueError("OptimizationProblem requires at least one TrainingCase.")

        seen_ids: set[str] = set()
        for case in cases:
            if case.case_id in seen_ids:
                raise ValueError(f"Duplicate TrainingCase.case_id: {case.case_id!r}")
            seen_ids.add(case.case_id)

        for obs in self.observations.values:
            if obs.case_id not in seen_ids:
                raise ValueError(
                    f"Observation {obs.label!r} (kind={obs.kind!r}) references "
                    f"case_id={obs.case_id!r}, which does not match any of this "
                    f"problem's training case IDs: {sorted(seen_ids)}."
                )

        expected_len = len(self.layout)
        try:
            starting_vector = self.layout.vector(self.starting_force_field)
        except (IndexError, AttributeError) as exc:
            raise ValueError("starting_force_field structure does not match layout.") from exc
        if starting_vector.shape != (expected_len,):
            raise ValueError(
                f"starting_force_field produces a {starting_vector.shape} vector; expected ({expected_len},)."
            )
        if self.active_space.layout != self.layout:
            raise ValueError("active_space must be built over this problem's layout.")
        if self.active_space.n_full != expected_len:
            raise ValueError(
                f"active_space baseline length ({self.active_space.n_full}) "
                f"does not match layout length ({expected_len})."
            )
        if self.preparation_provenance is not None:
            if not isinstance(self.preparation_provenance, PreparationProvenance):
                raise TypeError("preparation_provenance must be PreparationProvenance or None.")
            if self.preparation_provenance.case_ids != self.case_ids:
                raise ValueError("preparation_provenance.case_ids must match the problem's case IDs.")
        if self.publication_metadata is not None:
            if not isinstance(self.publication_metadata, PublicationMetadata):
                raise TypeError("publication_metadata must be PublicationMetadata or None.")
            if self.publication_metadata.provisionable:
                authoritative = self.publication_metadata.authoritative_case_ids
                source_ids = tuple(case.source_id or case.case_id for case in cases)
                if authoritative != source_ids:
                    raise ValueError(
                        "publication_metadata.authoritative_case_ids must exactly match problem source IDs and order."
                    )

    @property
    def case_ids(self) -> tuple[str, ...]:
        """Case IDs in case order."""
        return tuple(case.case_id for case in self.cases)

    @property
    def molecules(self) -> tuple[Molecule, ...]:
        """Training-case molecules in case order."""
        return tuple(case.molecule for case in self.cases)

    def case_by_id(self, case_id: str) -> TrainingCase:
        """Look up a training case by its stable ID.

        Raises:
            KeyError: If no case has this ID.

        """
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)
