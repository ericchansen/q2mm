"""The one canonical optimization-result model for Q2MM.

Every optimizer and every workflow returns exactly one shape —
:class:`OptimizationResult`.  It always carries validated, read-only
**full-length** initial/final parameter vectors bound to an explicit
layout identity (:attr:`n_params` + :attr:`layout_fingerprint`), the
initial/final scores, the explicit success/failure state and message,
non-negative iteration/evaluation counts, the explicit gradient
mode/provenance the run used, multi-start :class:`CandidateRecord`
entries (each with its own full initial/final candidate vectors,
including failures), and workflow :class:`StageRecord` metadata.

The result carries **no** :class:`~q2mm.models.forcefield.ForceField`
snapshot and **no** mutable backend/native state — callers materialise
the optimised force field explicitly via
``layout.replace(base_ff, result.final_params)``.

Every mapping/sequence/array reachable from a result is deeply frozen, so
results are safe to share and cannot be mutated after construction.  This
module lives in :mod:`q2mm.models` and depends only on NumPy; the gradient
mode is stored as a plain string (the ``value`` of
:class:`q2mm.objectives.protocols.GradientMode`) to keep that dependency
direction clean.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

__all__ = [
    "GRADIENT_MODES",
    "CandidateRecord",
    "StageRecord",
    "OptimizationResult",
    "deep_freeze",
]

#: The closed vocabulary for ``gradient_mode`` strings.
GRADIENT_MODES = frozenset({"analytical", "finite_difference", "none"})


def deep_freeze(value: object) -> object:
    """Recursively freeze *value* into an immutable structure.

    Mappings become :class:`~types.MappingProxyType` (keys preserved
    unchanged, values recursively frozen), sets/frozensets become
    :class:`frozenset`, sequences (except strings/bytes and NumPy arrays)
    become tuples, and NumPy arrays become read-only defensive copies.
    Scalars pass through unchanged.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        arr = np.array(value, copy=True)
        arr.setflags(write=False)
        return arr
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(v) for v in value)
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    return value


def _readonly_vector(values: object, *, n_params: int, name: str) -> np.ndarray:
    """Return a read-only 1-D float copy validated to length *n_params*."""
    array = np.array(values, dtype=float, copy=True)
    if array.shape != (n_params,):
        raise ValueError(f"{name} must have shape ({n_params},), got {array.shape}.")
    array.setflags(write=False)
    return array


def _validate_identity(n_params: int, layout_fingerprint: str) -> None:
    if not isinstance(n_params, int) or n_params < 0:
        raise ValueError(f"n_params must be a non-negative int, got {n_params!r}.")
    if not isinstance(layout_fingerprint, str) or not layout_fingerprint:
        raise ValueError("layout_fingerprint must be a non-empty string.")


def _validate_fd_step(fd_step: float | None, *, where: str) -> None:
    """Validate that a finite-difference step, when present, is positive and finite."""
    if fd_step is None:
        return
    if not math.isfinite(fd_step) or fd_step <= 0.0:
        raise ValueError(f"{where}.fd_step must be positive and finite, got {fd_step!r}.")


@dataclass(frozen=True, eq=False)
class CandidateRecord:
    """One candidate evaluated by a multi-start optimizer.

    Every generated candidate — successful, failed, or skipped — is
    recorded with its full-length starting and final vectors so a
    multi-start run never silently drops a start.

    Attributes:
        index: 0-based position in the deterministic candidate sequence.
        status: ``"success"``, ``"failure"``, or ``"skipped"``.
        n_params: Full-vector length; matches the owning result.
        layout_fingerprint: Layout identity; matches the owning result.
        initial_params: Read-only full-length generated starting vector.
        final_params: Read-only full-length final vector.  For a failed or
            skipped candidate this equals the generated start.
        initial_score: Objective at the start, or ``nan`` if unevaluated.
        final_score: Objective after optimizing, or ``nan``/``inf`` for a
            failed/skipped candidate.
        message: Human-readable outcome / error summary.
        seed: Perturbation/RNG seed used to generate the candidate.

    """

    index: int
    status: str
    n_params: int
    layout_fingerprint: str
    initial_params: np.ndarray
    final_params: np.ndarray
    initial_score: float = float("nan")
    final_score: float = float("nan")
    message: str = ""
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.status not in ("success", "failure", "skipped"):
            raise ValueError(f"CandidateRecord.status must be success/failure/skipped, got {self.status!r}.")
        if not isinstance(self.index, int) or self.index < 0:
            raise ValueError(f"CandidateRecord.index must be a non-negative int, got {self.index!r}.")
        _validate_identity(self.n_params, self.layout_fingerprint)
        # A converged (success) candidate must carry finite scores; a failed
        # or skipped candidate may legitimately record NaN/Inf.
        if self.status == "success" and not (math.isfinite(self.initial_score) and math.isfinite(self.final_score)):
            raise ValueError("A successful CandidateRecord must have finite initial and final scores.")
        object.__setattr__(
            self, "initial_params", _readonly_vector(self.initial_params, n_params=self.n_params, name="initial_params")
        )
        object.__setattr__(
            self, "final_params", _readonly_vector(self.final_params, n_params=self.n_params, name="final_params")
        )


@dataclass(frozen=True, eq=False)
class StageRecord:
    """One optimization stage within a workflow.

    Stage metadata lives here so a workflow can aggregate stages into the
    one canonical :class:`OptimizationResult` without inventing a second
    result shape.  Stage records stay focused — they do **not** duplicate
    the full parameter vectors, which live on the result.

    Attributes:
        name: Human-readable stage label.
        n_params: Full-vector length; matches the owning result.
        layout_fingerprint: Layout identity; matches the owning result.
        initial_score: Optimizer-reported objective value at stage start.
        final_score: Optimizer-reported objective value at stage end.
        n_iterations: Optimizer iteration count for this stage.
        n_evaluations: Objective/gradient call count for this stage.
        converged: Optimizer convergence flag for this stage.
        message: Optimizer convergence message.
        gradient_mode: Resolved gradient mode string for this stage.
        fd_step: Finite-difference step for this stage, or ``None``.
        elapsed_s: Wall time for the optimizer call.
        locked_param_indices: Full-vector indices frozen by the workflow
            itself at the start of this stage.  Unique and in ``[0, n_params)``.
        notes: Deeply-frozen free-form per-stage metadata.

    """

    name: str
    n_params: int
    layout_fingerprint: str
    initial_score: float
    final_score: float
    n_iterations: int
    n_evaluations: int
    converged: bool
    message: str
    gradient_mode: str
    fd_step: float | None = None
    elapsed_s: float = 0.0
    locked_param_indices: tuple[int, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identity(self.n_params, self.layout_fingerprint)
        if self.n_iterations < 0 or self.n_evaluations < 0:
            raise ValueError("StageRecord counts must be non-negative.")
        if self.elapsed_s < 0:
            raise ValueError("StageRecord.elapsed_s must be non-negative.")
        if self.gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"StageRecord.gradient_mode must be one of {sorted(GRADIENT_MODES)}.")
        _validate_fd_step(self.fd_step, where="StageRecord")
        locked = tuple(int(i) for i in self.locked_param_indices)
        if len(set(locked)) != len(locked):
            raise ValueError(f"StageRecord.locked_param_indices must be unique, got {locked}.")
        for i in locked:
            if not 0 <= i < self.n_params:
                raise ValueError(f"StageRecord.locked_param_indices out of range [0, {self.n_params}): {i}.")
        object.__setattr__(self, "locked_param_indices", locked)
        object.__setattr__(self, "notes", deep_freeze(dict(self.notes)))


@dataclass(frozen=True, eq=False)
class OptimizationResult:
    """The one canonical result returned by every optimizer and workflow.

    Attributes:
        success: Whether the run converged.
        message: Human-readable convergence/failure message.
        initial_score / final_score: Objective before/after optimization.
        n_iterations / n_evaluations: Non-negative counters.
        n_params: Full parameter-vector length (== ``len(layout)``).
        layout_fingerprint: Immutable layout identity the vectors belong to.
        initial_params / final_params: Read-only full-length vectors; frozen
            slots exactly equal the baseline.
        history: Per-evaluation objective history.
        method: Method identifier.
        gradient_mode: Explicit gradient provenance the run used — one of
            :data:`GRADIENT_MODES`.
        fd_step: Finite-difference step when finite differences were used.
        candidates: Multi-start candidate records (successes and failures).
        stages: Workflow stage records.
        initial_samples / final_samples: Repeated real-objective samples.
        category_metrics: Deeply-frozen per-category fit metrics.

    """

    success: bool
    message: str
    initial_score: float
    final_score: float
    n_iterations: int
    n_evaluations: int
    n_params: int
    layout_fingerprint: str
    initial_params: np.ndarray
    final_params: np.ndarray
    history: tuple[float, ...] = ()
    method: str = ""
    gradient_mode: str = "none"
    fd_step: float | None = None
    candidates: tuple[CandidateRecord, ...] = ()
    stages: tuple[StageRecord, ...] = ()
    initial_samples: tuple[float, ...] = ()
    final_samples: tuple[float, ...] = ()
    category_metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_identity(self.n_params, self.layout_fingerprint)
        if self.n_iterations < 0 or self.n_evaluations < 0:
            raise ValueError("OptimizationResult counts must be non-negative.")
        if self.gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"gradient_mode must be one of {sorted(GRADIENT_MODES)}, got {self.gradient_mode!r}.")
        _validate_fd_step(self.fd_step, where="OptimizationResult")
        # A converged run must carry finite scores; an explicit failure may
        # report inf/nan (e.g. all multi-start candidates failed).
        if self.success and not (math.isfinite(self.initial_score) and math.isfinite(self.final_score)):
            raise ValueError("A successful OptimizationResult must have finite initial and final scores.")
        object.__setattr__(
            self, "initial_params", _readonly_vector(self.initial_params, n_params=self.n_params, name="initial_params")
        )
        object.__setattr__(
            self, "final_params", _readonly_vector(self.final_params, n_params=self.n_params, name="final_params")
        )
        object.__setattr__(self, "history", tuple(float(x) for x in self.history))
        object.__setattr__(self, "initial_samples", tuple(float(x) for x in self.initial_samples))
        object.__setattr__(self, "final_samples", tuple(float(x) for x in self.final_samples))

        candidates = tuple(self.candidates)
        for cand in candidates:
            if not isinstance(cand, CandidateRecord):
                raise TypeError(f"candidates must be CandidateRecord, got {type(cand).__name__}.")
            if cand.n_params != self.n_params or cand.layout_fingerprint != self.layout_fingerprint:
                raise ValueError("CandidateRecord layout identity must match the owning result.")
        object.__setattr__(self, "candidates", candidates)

        stages = tuple(self.stages)
        for stage in stages:
            if not isinstance(stage, StageRecord):
                raise TypeError(f"stages must be StageRecord, got {type(stage).__name__}.")
            if stage.n_params != self.n_params or stage.layout_fingerprint != self.layout_fingerprint:
                raise ValueError("StageRecord layout identity must match the owning result.")
        object.__setattr__(self, "stages", stages)

        object.__setattr__(self, "category_metrics", deep_freeze(dict(self.category_metrics)))

    @property
    def improvement(self) -> float:
        """Fractional improvement ``(initial - final) / initial`` (0 if init 0)."""
        if self.initial_score == 0:
            return 0.0
        return (self.initial_score - self.final_score) / self.initial_score

    def with_full_params(self, initial_params: np.ndarray, final_params: np.ndarray) -> OptimizationResult:
        """Return a copy with new full-length initial/final vectors."""
        return replace(self, initial_params=initial_params, final_params=final_params)

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        return (
            f"Method: {self.method}\n"
            f"Success: {self.success} — {self.message}\n"
            f"Score: {self.initial_score:.6f} → {self.final_score:.6f} "
            f"({self.improvement:.1%} improvement)\n"
            f"Iterations: {self.n_iterations}, Evaluations: {self.n_evaluations}"
        )
