"""Immutable benchmark outcomes and their provenance/JSON projections.

Scientific identity and optimization-result fields come from the canonical
application and result-projection APIs. This module owns only the benchmark
envelopes and scalar coercions, not execution or file installation.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from q2mm._canonical import json_value
from q2mm._result_serialization import result_payload, stage_payload
from q2mm.benchmarks.acceptance import CandidateStatus
from q2mm.models.results import deep_freeze

if TYPE_CHECKING:
    from q2mm.backends.contracts import BackendInfo
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.benchmarks.profiles import ResolvedProfile, RunProfile
    from q2mm.models.forcefield import ForceField
    from q2mm.models.results import OptimizationResult

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def sanitize_for_json(value: Any) -> Any:
    """Recursively coerce *value* into strict-JSON-safe primitives.

    Non-finite floats become the sentinel strings ``"NaN"`` /
    ``"Infinity"`` / ``"-Infinity"`` (valid strict JSON, still readable),
    NumPy scalars/arrays become Python scalars/lists, and read-only
    mappings become plain dicts.
    """
    return json_value(value, strict=False, coerce_keys=True)


def _git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"git_sha": sha, "git_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def build_run_provenance(*, generator: str, output_dir: Path) -> dict[str, Any]:
    """Build the run-level provenance block (timestamp/command are non-identity)."""
    return {
        "generator": generator,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": shlex.join(sys.argv),
        "q2mm": _git_info(REPO_ROOT),
        "output_dir": str(output_dir),
    }


def _data_provenance(case: BenchmarkCase, resolved_roots: Mapping[str, str]) -> dict[str, Any]:
    """Record source metadata and the SDK's versioned scientific input digests.

    The additive ``scientific_problem`` entry participates in resolved candidate
    identity, so new runs no longer reuse metadata-only candidate IDs. Existing
    records remain readable as written; absent input digests are not backfilled.
    Only digests are persisted, not the SDK's full scientific identity payload.
    """
    from q2mm.application.models import (
        PROBLEM_FINGERPRINT_VERSION,
        _problem_fingerprints,
    )

    problem = case.problem
    fingerprint, input_fingerprints = _problem_fingerprints(problem)
    cases = [{"case_id": c.case_id, "stationary_point": c.stationary_point.value} for c in problem.cases]
    hessians: list[dict[str, Any]] = []
    for c in problem.cases:
        hp = c.molecule.hessian_provenance
        hessians.append(
            {
                "case_id": c.case_id,
                "units": None if hp is None else hp.units.value,
                "source": None if hp is None else hp.source,
                "path": None if hp is None else hp.path,
            }
        )
    return {
        "scientific_problem": {
            "fingerprint_version": PROBLEM_FINGERPRINT_VERSION,
            "fingerprint": fingerprint,
            "input_fingerprints": dict(input_fingerprints),
        },
        "metadata": dict(case.metadata),
        "objective_profile": (
            problem.publication_metadata.objective_profile.identifier
            if problem.publication_metadata is not None
            else None
            if problem.preparation_provenance is None
            else problem.preparation_provenance.profile
        ),
        "publication_metadata": (
            None if problem.publication_metadata is None else problem.publication_metadata.to_dict()
        ),
        "publication_metadata_fingerprint": (
            None if problem.publication_metadata is None else problem.publication_metadata.fingerprint
        ),
        "cases": cases,
        "hessians": hessians,
        "default_forms": list(case.default_forms),
        "description": case.description,
        "resolved_data_roots": dict(resolved_roots),
    }


def _benchmark_scalars(record: dict[str, Any], *, stage: bool = False) -> dict[str, Any]:
    """Apply the same benchmark scalar policy to full and stage-only projections."""
    scalar_types: dict[str, Callable[[Any], Any]] = {
        "success": bool,
        "converged": bool,
        "message": str,
        "initial_score": float,
        "final_score": float,
        "n_iterations": int,
        "n_evaluations": int,
        "n_params": int,
        "index": int,
        "elapsed_s": float,
    }
    for key, coerce in scalar_types.items():
        if key in record:
            record[key] = coerce(record[key])
    if stage:
        record["gradient_mode"] = str(record["gradient_mode"])
    return record


def result_to_dict(result: OptimizationResult) -> dict[str, Any]:
    """Project the canonical result with benchmark scalar coercions.

    Includes layout identity, full initial/final vectors, counts, history,
    gradient mode / FD step, multi-start candidate records, workflow stage
    records and notes, endpoint samples, and per-category metrics. Nested
    diagnostics and nonfinite scalars are normalized by :func:`sanitize_for_json`
    at the JSON-safe summary and :meth:`CandidateResult.record` boundaries.
    """
    payload = _benchmark_scalars(result_payload(result))
    for candidate in payload["candidates"]:
        _benchmark_scalars(candidate)
    for stage in payload["stages"]:
        _benchmark_scalars(stage, stage=True)
    return payload


def _resolved_summary(
    *,
    profile: RunProfile,
    backend_info: BackendInfo,
    case: BenchmarkCase,
    form: str,
    spec: Any,
    evaluator_kind: str,
    gradient_mode: str,
    expected_grad: str,
    success_spec: Any | None,
) -> dict[str, Any]:
    """Project resolved settings without rebuilding any configured component."""
    problem = case.problem
    return {
        "system": profile.system,
        "backend": profile.backend,
        "backend_name": backend_info.name,
        "functional_form": form,
        "workflow": profile.workflow,
        "optimizer": profile.optimizer,
        "optimizer_method": spec.method,
        "optimizer_label": spec.label,
        "evaluator": evaluator_kind,
        "gradient_mode": gradient_mode,
        "expected_result_gradient_mode": expected_grad,
        "effective_regularization": profile.effective_regularization,
        "starting_point": profile.starting_point,
        "objective_profile": profile.effective_objective_profile,
        "reproduction_status": (
            None if problem.publication_metadata is None else problem.publication_metadata.status.value
        ),
        "publication_metadata_fingerprint": (
            None if problem.publication_metadata is None else problem.publication_metadata.fingerprint
        ),
        "publication_success_spec": None if success_spec is None else success_spec.to_dict(),
        "starting_point_audit": case.metadata.get("starting_point_audit"),
        "n_molecules": len(problem.molecules),
        "n_active_params": problem.active_space.n_active,
    }


def _initial_summary(
    base_summary: Mapping[str, Any],
    initial_score: float,
    initial_category_scores: Mapping[str, float],
    seminario_categories: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach objective-of-record baseline metrics to resolved metadata."""
    return {
        **base_summary,
        "initial_obj_score": initial_score,
        "initial_category_scores": initial_category_scores,
        "seminario": seminario_categories,
    }


def _optimization_summary(
    *,
    result: OptimizationResult,
    final_score: float,
    final_category_scores: Mapping[str, float],
    improvement_pct: float,
    actual_grad: str,
    expected_grad: str,
    elapsed: float,
    optimized_categories: Mapping[str, Any],
    final_executor_ratio: float | None,
) -> dict[str, Any]:
    """Project terminal scores and canonical result diagnostics into the summary."""
    summary = {
        "final_obj_score": final_score,
        "final_category_scores": final_category_scores,
        "improvement_pct": improvement_pct,
        "initial_optimizer_score": float(result.initial_score),
        "final_optimizer_score": float(result.final_score),
        "n_iterations": int(result.n_iterations),
        "n_evaluations": int(result.n_evaluations),
        "converged": bool(result.success),
        "message": str(result.message),
        "result_gradient_mode": actual_grad,
        "expected_result_gradient_mode": expected_grad,
        "result_fd_step": result.fd_step,
        "opt_time_s": elapsed,
        "optimized": optimized_categories,
        "stages": sanitize_for_json([_benchmark_scalars(stage_payload(stage), stage=True) for stage in result.stages]),
    }
    if len(result.stages) <= 1:
        summary["final_executor_ratio"] = final_executor_ratio
    else:
        summary["final_executor_ratio_omitted"] = "multiple_workflow_stages"
    return summary


@dataclass(frozen=True, eq=False)
class CandidateResult:
    """One immutable, terminal outcome of :func:`q2mm.benchmarks.runner.run_profile`.

    Attributes:
        candidate_id: Stable, filesystem-safe identity (readable prefix +
            deterministic fingerprint suffix).
        status: Terminal :class:`~q2mm.benchmarks.acceptance.CandidateStatus`.
        reason: Human-readable explanation of the status.
        profile: The originating :class:`RunProfile`.
        resolved: The provenance-complete
            :class:`~q2mm.benchmarks.profiles.ResolvedProfile`, or ``None``
            when the run failed before resolution.
        summary: Deeply-frozen JSON-safe metrics/scores for the run.
        optimization_result: The one canonical ``OptimizationResult`` for an
            accepted *or* rejected run (``None`` for skipped/errored).
        final_force_field: The materialized force field for an accepted *or*
            rejected run (``None`` for skipped/errored); only accepted
            candidates are ever promoted.

    """

    candidate_id: str
    status: CandidateStatus
    reason: str
    profile: RunProfile
    resolved: ResolvedProfile | None
    summary: Mapping[str, Any] = field(default_factory=dict)
    optimization_result: OptimizationResult | None = None
    final_force_field: ForceField | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", deep_freeze(dict(self.summary)))
        if not self.reason:
            raise ValueError("CandidateResult.reason must be non-empty.")
        has_run = self.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED)
        if has_run and (self.optimization_result is None or self.final_force_field is None):
            raise ValueError(
                f"{self.status.value} candidate must carry both an OptimizationResult and a final force field."
            )
        if not has_run and (self.optimization_result is not None or self.final_force_field is not None):
            raise ValueError(f"{self.status.value} candidate must not carry an OptimizationResult or force field.")
        if self.resolved is not None and self.candidate_id != self.resolved.candidate_id():
            raise ValueError("CandidateResult.candidate_id must equal resolved.candidate_id() when resolved.")
        if self.resolved is None and self.candidate_id != self.profile.candidate_id():
            raise ValueError(
                "CandidateResult.candidate_id must equal the requested profile candidate_id() before resolution."
            )

    @property
    def accepted(self) -> bool:
        """``True`` only for an accepted candidate."""
        return self.status is CandidateStatus.ACCEPTED

    def record(self) -> dict[str, Any]:
        """Return the JSON-safe persisted record (without run provenance)."""
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "reason": self.reason,
            "profile": {**self.profile.canonical_dict(), "label": self.profile.label},
            "profile_fingerprint": self.profile.fingerprint(),
            "resolved": self.resolved.to_dict() if self.resolved is not None else None,
            "resolved_fingerprint": self.resolved.fingerprint() if self.resolved is not None else None,
            "summary": sanitize_for_json(dict(self.summary)),
            "optimization_result": (
                sanitize_for_json(result_to_dict(self.optimization_result))
                if self.optimization_result is not None
                else None
            ),
        }


@dataclass(frozen=True, eq=False)
class LoadedCandidate:
    """A deeply-frozen candidate record loaded from disk (incl. failures)."""

    candidate_id: str
    status: CandidateStatus
    reason: str
    path: Path
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", deep_freeze(dict(self.record)))

    @property
    def summary(self) -> Mapping[str, Any]:
        """The persisted metrics/scores summary."""
        summary = self.record.get("summary", {})
        return summary if isinstance(summary, Mapping) else MappingProxyType({})


@dataclass(frozen=True, eq=False)
class RunOutcome:
    """Immutable aggregate outcome of :func:`q2mm.benchmarks.runner.run_profiles`."""

    candidates: tuple[CandidateResult, ...] = ()
    promoted: Mapping[str, Mapping[str, Path]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "promoted", MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.promoted.items()})
        )

    def by_status(self, status: CandidateStatus) -> tuple[CandidateResult, ...]:
        """Return candidates in a given terminal status."""
        return tuple(c for c in self.candidates if c.status is status)

    @property
    def accepted(self) -> tuple[CandidateResult, ...]:
        """Accepted candidates."""
        return self.by_status(CandidateStatus.ACCEPTED)

    @property
    def optimized_candidates(self) -> tuple[CandidateResult, ...]:
        """Candidates that actually ran an optimization (accepted or rejected)."""
        return tuple(c for c in self.candidates if c.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED))

    @property
    def ok(self) -> bool:
        """False on any error, or when optimizations ran but none was accepted.

        A run of only skips (e.g. registered-but-unavailable backends, or a
        matrix/smoke with no runnable combos) is still ``ok``.
        """
        if self.by_status(CandidateStatus.ERROR):
            return False
        ran = self.optimized_candidates
        return not (ran and not self.accepted)
