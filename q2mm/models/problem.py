"""Immutable training-case and optimization-problem model for Q2MM.

:class:`OptimizationProblem` is the one immutable bundle produced by a
benchmark-system loader (see ``q2mm.benchmarks.systems``) and consumed by
objective/optimizer/workflow code: training cases (each a canonical
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

from dataclasses import dataclass
from enum import Enum

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout

__all__ = ["StationaryPointKind", "TrainingCase", "OptimizationProblem"]


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

    """

    case_id: str
    molecule: Molecule
    stationary_point: StationaryPointKind

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("TrainingCase.case_id must be a non-empty string.")
        if not isinstance(self.molecule, Molecule):
            raise TypeError(f"TrainingCase.molecule must be a Molecule, got {type(self.molecule).__name__}.")


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
