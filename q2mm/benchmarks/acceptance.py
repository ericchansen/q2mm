"""Canonical candidate-acceptance policy for the benchmark runner.

Every requested candidate — whether from a single run, a convergence
batch, or a backend x form x optimizer matrix — ends in exactly one of a
closed, explicit set of states:

- ``accepted`` — the run produced a force field worth promoting.
- ``rejected`` — the run executed but did not clear the acceptance bar
  (non-finite objective, a worsened objective, or no measurable progress).
- ``skipped`` — the run was deliberately not executed (backend/form
  unavailable, ratio gate closed, or ``skip_optimization`` requested).
- ``error`` — the run raised before producing a result.

There is exactly one no-progress decision in the codebase:
:meth:`NoProgressPolicy.made_progress`.  The runner never re-implements
it.  Improvement is computed solely via
:func:`q2mm.objectives.metrics.fractional_improvement`; acceptance never
invents its own metric.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from q2mm.objectives.metrics import fractional_improvement

__all__ = [
    "CandidateStatus",
    "NoProgressPolicy",
    "AcceptanceDecision",
    "AcceptancePolicy",
    "improvement_percent",
]


class CandidateStatus(str, Enum):
    """The closed vocabulary of terminal candidate states."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    ERROR = "error"


def improvement_percent(initial_score: float, final_score: float) -> float:
    """Percentage objective improvement using the shared metric convention.

    Delegates to :func:`q2mm.objectives.metrics.fractional_improvement`
    (``(initial - final) / initial``) so acceptance and reporting never
    disagree.  Returns ``0.0`` when *initial* is zero.
    """
    return 100.0 * fractional_improvement(initial_score, final_score)


@dataclass(frozen=True)
class NoProgressPolicy:
    """The one no-progress rule shared by every runner path.

    An optimization "made progress" unless it both stopped in
    ``<= max_iterations`` iterations *and* moved the objective by
    ``< min_improvement_pct`` percent.  This is the single guard against
    the silent ``nfev<=2`` non-optimization documented in AGENTS.md.

    Attributes:
        max_iterations: Iteration ceiling below which a negligible score
            change counts as no progress.
        min_improvement_pct: Absolute percentage objective change that must
            be cleared to count as progress.

    """

    max_iterations: int = 2
    min_improvement_pct: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int) or self.max_iterations < 0:
            raise ValueError("NoProgressPolicy.max_iterations must be a non-negative int.")
        if not (math.isfinite(self.min_improvement_pct) and self.min_improvement_pct >= 0):
            raise ValueError("NoProgressPolicy.min_improvement_pct must be non-negative and finite.")

    def made_progress(self, *, n_iterations: int, improvement_pct: float) -> bool:
        """Return ``True`` when the run cleared the no-progress bar."""
        stalled = n_iterations <= self.max_iterations and abs(improvement_pct) < self.min_improvement_pct
        return not stalled

    def describe(self) -> str:
        """Human-readable one-line summary of the policy thresholds."""
        return f"n_iterations>{self.max_iterations} or |improvement|>={self.min_improvement_pct:g}%"


@dataclass(frozen=True)
class AcceptanceDecision:
    """One terminal decision about a candidate, with a human-readable reason."""

    status: CandidateStatus
    reason: str

    @property
    def is_accepted(self) -> bool:
        """``True`` only for :attr:`CandidateStatus.ACCEPTED`."""
        return self.status is CandidateStatus.ACCEPTED


@dataclass(frozen=True)
class AcceptancePolicy:
    """The canonical acceptance policy applied to every executed candidate.

    Attributes:
        no_progress: The single shared no-progress rule.
        worsening_tolerance_pct: The only tolerance by which the objective
            may drift negative and still be accepted (absorbs float noise;
            ``0.0`` rejects any real worsening).
        require_convergence: When ``True``, an optimizer that ran but did
            not report ``success`` is rejected even if it improved.

    """

    no_progress: NoProgressPolicy = field(default_factory=NoProgressPolicy)
    worsening_tolerance_pct: float = 0.0
    require_convergence: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.worsening_tolerance_pct) and self.worsening_tolerance_pct >= 0):
            raise ValueError("AcceptancePolicy.worsening_tolerance_pct must be non-negative and finite.")

    def evaluate(
        self,
        *,
        n_iterations: int,
        initial_score: float,
        final_score: float,
        converged: bool,
    ) -> AcceptanceDecision:
        """Accept or reject a candidate that produced an optimization result.

        Order of checks: non-finite scores/improvement, then objective
        worsening, then the single no-progress rule, then convergence.  The
        only caller of :meth:`NoProgressPolicy.made_progress`.
        """
        if not isinstance(n_iterations, int) or n_iterations < 0:
            raise ValueError(f"n_iterations must be a non-negative int, got {n_iterations!r}.")
        if not (math.isfinite(initial_score) and math.isfinite(final_score)):
            return AcceptanceDecision(
                CandidateStatus.REJECTED,
                f"non-finite objective score (initial={initial_score!r}, final={final_score!r})",
            )
        if initial_score < 0.0 or final_score < 0.0:
            return AcceptanceDecision(
                CandidateStatus.REJECTED,
                f"invalid negative objective score (initial={initial_score!r}, final={final_score!r})",
            )
        # A zero baseline makes percentage improvement undefined (it is 0 by the
        # shared convention); reject any actual increase in the raw objective.
        if initial_score == 0.0 and final_score > 0.0:
            return AcceptanceDecision(
                CandidateStatus.REJECTED, f"objective worsened from a zero baseline (0 -> {final_score:g})"
            )
        improvement_pct = improvement_percent(initial_score, final_score)
        if not math.isfinite(improvement_pct):
            return AcceptanceDecision(CandidateStatus.REJECTED, "non-finite objective improvement")
        if improvement_pct < -self.worsening_tolerance_pct:
            return AcceptanceDecision(CandidateStatus.REJECTED, f"objective worsened by {-improvement_pct:.3f}%")
        if not self.no_progress.made_progress(n_iterations=n_iterations, improvement_pct=improvement_pct):
            return AcceptanceDecision(
                CandidateStatus.REJECTED,
                (
                    f"no measurable progress (n_iterations={n_iterations}, "
                    f"improvement={improvement_pct:.3f}%); policy requires {self.no_progress.describe()}"
                ),
            )
        if self.require_convergence and not converged:
            return AcceptanceDecision(
                CandidateStatus.REJECTED,
                f"optimizer did not converge (improvement={improvement_pct:.3f}%)",
            )
        return AcceptanceDecision(
            CandidateStatus.ACCEPTED,
            f"optimizer made progress (n_iterations={n_iterations}, improvement={improvement_pct:.3f}%)",
        )

    @staticmethod
    def skipped(reason: str) -> AcceptanceDecision:
        """Build a :attr:`CandidateStatus.SKIPPED` decision."""
        return AcceptanceDecision(CandidateStatus.SKIPPED, reason)

    @staticmethod
    def errored(reason: str) -> AcceptanceDecision:
        """Build a :attr:`CandidateStatus.ERROR` decision."""
        return AcceptanceDecision(CandidateStatus.ERROR, reason)
