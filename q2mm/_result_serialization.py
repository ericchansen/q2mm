"""Complete canonical result projection, independent of persistence envelopes.

Scalar values and nested diagnostic metadata are preserved here. Callers own
JSON normalization, including nonfinite sentinels and strictness policies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.models.results import CandidateRecord, OptimizationResult, StageRecord


def stage_payload(stage: StageRecord) -> dict[str, Any]:
    """Project one stage without visiting result or candidate vectors."""
    return {
        "name": stage.name,
        "n_params": stage.n_params,
        "layout_fingerprint": stage.layout_fingerprint,
        "initial_score": stage.initial_score,
        "final_score": stage.final_score,
        "n_iterations": stage.n_iterations,
        "n_evaluations": stage.n_evaluations,
        "converged": stage.converged,
        "message": stage.message,
        "gradient_mode": stage.gradient_mode,
        "fd_step": stage.fd_step,
        "elapsed_s": stage.elapsed_s,
        "locked_param_indices": list(stage.locked_param_indices),
        "notes": dict(stage.notes),
    }


def _candidate_payload(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "index": candidate.index,
        "status": candidate.status,
        "n_params": candidate.n_params,
        "layout_fingerprint": candidate.layout_fingerprint,
        "initial_params": candidate.initial_params.tolist(),
        "final_params": candidate.final_params.tolist(),
        "initial_score": candidate.initial_score,
        "final_score": candidate.final_score,
        "message": candidate.message,
        "seed": candidate.seed,
    }


def result_payload(result: OptimizationResult) -> dict[str, Any]:
    """Project every canonical result field without changing scalar values."""
    return {
        "success": result.success,
        "message": result.message,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "n_iterations": result.n_iterations,
        "n_evaluations": result.n_evaluations,
        "n_params": result.n_params,
        "layout_fingerprint": result.layout_fingerprint,
        "initial_params": result.initial_params.tolist(),
        "final_params": result.final_params.tolist(),
        "history": list(result.history),
        "method": result.method,
        "gradient_mode": result.gradient_mode,
        "fd_step": result.fd_step,
        "initial_samples": list(result.initial_samples),
        "final_samples": list(result.final_samples),
        "category_metrics": {key: dict(value) for key, value in result.category_metrics.items()},
        "candidates": [_candidate_payload(candidate) for candidate in result.candidates],
        "stages": [stage_payload(stage) for stage in result.stages],
    }
