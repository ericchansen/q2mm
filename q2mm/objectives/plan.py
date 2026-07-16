"""Immutable, backend-neutral objective plan.

An :class:`ObjectivePlan` is the compiled, backend-neutral description of
*what* to fit: the training-case molecules with their stable case IDs and
stationary-point kinds, the canonical :class:`ObservationSet` to fit
against, the :class:`ParameterLayout`, the :class:`ActiveParameterSpace`
projection, and the L2 regularization reference/strength.

It is compiled from an :class:`~q2mm.models.problem.OptimizationProblem`
via :meth:`ObjectivePlan.from_problem`.  It contains **no** backend or
native imports — a plan is shared unchanged by the Python and JAX
executors, which attach the concrete backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

import numpy as np

from q2mm.models.molecule import Molecule
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind

__all__ = ["ObjectivePlan", "KIND_TO_CATEGORY"]


# Map every observation kind to its evaluator category.  This is the one
# canonical, immutable kind→category vocabulary shared by both executors, the
# metrics module, and diagnostics.
KIND_TO_CATEGORY: MappingProxyType[str, str] = MappingProxyType(
    {
        "energy": "energy",
        "frequency": "frequency",
        "bond_length": "geometry",
        "bond_angle": "geometry",
        "torsion_angle": "geometry",
        "eig_diagonal": "eigenmatrix",
        "eig_offdiagonal": "eigenmatrix",
        "hessian_element": "hessian",
    }
)


@dataclass(frozen=True, eq=False)
class ObjectivePlan:
    """Immutable, backend-neutral compiled objective description.

    Attributes:
        case_ids: Stable case IDs, in case order.  Observations bind to
            cases by this ID, never by position.
        molecules: Training-case molecules, in case order (parallel to
            ``case_ids``).
        stationary_points: Ground-state/transition-state kind per case.
        observations: Canonical reference observations to fit against.
        layout: The full-vector parameter layout.
        active_space: The one active/frozen projection over ``layout``.
        regularization: L2 penalty strength (λ ≥ 0).
        reference_params: Read-only full-length L2 anchor vector.  Defaults
            to ``active_space.baseline`` (the starting force field vector).

    """

    case_ids: tuple[str, ...]
    molecules: tuple[Molecule, ...]
    stationary_points: tuple[StationaryPointKind, ...]
    observations: ObservationSet
    layout: ParameterLayout
    active_space: ActiveParameterSpace
    regularization: float = 0.0
    reference_params: np.ndarray = field(default_factory=lambda: np.zeros(0))
    _case_index: Mapping[str, int] = field(init=False, default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        case_ids = tuple(self.case_ids)
        molecules = tuple(self.molecules)
        stationary = tuple(self.stationary_points)
        object.__setattr__(self, "case_ids", case_ids)
        object.__setattr__(self, "molecules", molecules)
        object.__setattr__(self, "stationary_points", stationary)

        if not case_ids:
            raise ValueError("ObjectivePlan requires at least one case.")
        if len(molecules) != len(case_ids):
            raise ValueError(f"molecules length ({len(molecules)}) must match case_ids length ({len(case_ids)}).")
        if len(stationary) != len(case_ids):
            raise ValueError(
                f"stationary_points length ({len(stationary)}) must match case_ids length ({len(case_ids)})."
            )
        for cid in case_ids:
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"ObjectivePlan case_ids must be non-empty strings, got {cid!r}.")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError(f"case_ids must be unique, got {case_ids}.")
        for mol in molecules:
            if not isinstance(mol, Molecule):
                raise TypeError(f"ObjectivePlan molecules must be Molecule instances, got {type(mol).__name__}.")
        for sp in stationary:
            if not isinstance(sp, StationaryPointKind):
                raise TypeError(f"stationary_points must be StationaryPointKind, got {type(sp).__name__}.")

        known = set(case_ids)
        for obs in self.observations.values:
            if obs.case_id not in known:
                raise ValueError(
                    f"Observation {obs.label!r} (kind={obs.kind!r}) references case_id={obs.case_id!r}, "
                    f"which is not among this plan's case IDs: {sorted(known)}."
                )

        n_full = len(self.layout)
        if self.active_space.layout != self.layout:
            raise ValueError("active_space must be built over this plan's layout.")
        if self.active_space.n_full != n_full:
            raise ValueError(
                f"active_space baseline length ({self.active_space.n_full}) does not match layout length ({n_full})."
            )

        reg = float(self.regularization)
        if not np.isfinite(reg) or reg < 0:
            raise ValueError(f"regularization must be finite and non-negative, got {self.regularization!r}.")
        object.__setattr__(self, "regularization", reg)

        ref_in = self.reference_params
        use_default = ref_in is None or (isinstance(ref_in, np.ndarray) and ref_in.size == 0 and n_full != 0)
        ref = np.array(self.active_space.baseline if use_default else ref_in, dtype=float, copy=True)
        if ref.shape != (n_full,):
            raise ValueError(f"reference_params must have shape ({n_full},), got {ref.shape}.")
        if not np.all(np.isfinite(ref)):
            raise ValueError("reference_params must be finite.")
        ref.setflags(write=False)
        object.__setattr__(self, "reference_params", ref)

        # Stable, immutable case_id -> index map.
        object.__setattr__(self, "_case_index", MappingProxyType({cid: i for i, cid in enumerate(case_ids)}))

    # -- Derived, backend-neutral views -----------------------------------

    @property
    def n_params(self) -> int:
        """Length of the full parameter vector (``len(layout)``)."""
        return len(self.layout)

    def case_index(self, case_id: str) -> int:
        """Return the case-order index of *case_id*.

        Raises:
            KeyError: If *case_id* is not part of this plan.

        """
        index_map = self._case_index
        try:
            return index_map[case_id]
        except KeyError:
            raise KeyError(f"case_id {case_id!r} is not part of this plan: {self.case_ids}.") from None

    @property
    def categories(self) -> frozenset[str]:
        """Set of evaluator categories present among the observations."""
        return frozenset(KIND_TO_CATEGORY[obs.kind] for obs in self.observations.values)

    def observations_for_case(self, case_id: str) -> tuple[Observation, ...]:
        """Return the observations bound to *case_id*, in order."""
        return tuple(obs for obs in self.observations.values if obs.case_id == case_id)

    def with_observations(self, observations: ObservationSet) -> ObjectivePlan:
        """Return a copy with different observations (same cases/layout)."""
        return replace(self, observations=observations)

    def with_active_space(self, active_space: ActiveParameterSpace) -> ObjectivePlan:
        """Return a copy over a different active/frozen projection.

        The L2 reference is rebased to the new space's baseline.
        """
        return replace(self, active_space=active_space, reference_params=np.array(active_space.baseline, dtype=float))

    @classmethod
    def from_problem(
        cls,
        problem: OptimizationProblem,
        *,
        regularization: float = 0.0,
        reference_params: np.ndarray | None = None,
    ) -> ObjectivePlan:
        """Compile a plan from an immutable :class:`OptimizationProblem`.

        Args:
            problem: The loaded optimization problem.
            regularization: L2 penalty strength.
            reference_params: Optional explicit L2 anchor; defaults to the
                problem's active-space baseline (starting-FF vector).

        """
        ref = reference_params if reference_params is not None else problem.active_space.baseline
        return cls(
            case_ids=problem.case_ids,
            molecules=problem.molecules,
            stationary_points=tuple(case.stationary_point for case in problem.cases),
            observations=problem.observations,
            layout=problem.layout,
            active_space=problem.active_space,
            regularization=float(regularization),
            reference_params=np.array(ref, dtype=float),
        )
