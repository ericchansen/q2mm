"""Shared residual, regularization, sampling, and fit-metric helpers.

One implementation of the weighted/unweighted residual convention, L2
regularization, per-category fit metrics (R² / RMSD / MAE), and post-hoc
sampling — used by both executors, the workflows, diagnostics, and the
benchmark runner.  Keeping this in one module guarantees the Python and
JAX executors share identical residual/regularization/metric semantics.

Residual convention (matches the historical objective):

    raw_i      = ref_i - calc_i                    (torsion-wrapped)
    weighted_i = weight_i * raw_i
    score      = sum_i weighted_i**2 + lambda * ||p - p_ref||**2
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.protocols import Evaluation, ObjectiveEvaluator

__all__ = [
    "torsion_wrap",
    "raw_residual",
    "weighted_residual",
    "regularization_value",
    "regularization_gradient",
    "regularization_residuals",
    "fractional_improvement",
    "r2",
    "category_stats",
    "category_metrics",
    "evaluate_samples",
]


def torsion_wrap(diff: float) -> float:
    """Wrap a torsion-angle difference into ``[-180, 180)`` degrees."""
    return (diff + 180.0) % 360.0 - 180.0


def raw_residual(kind: str, ref_value: float, calc_value: float) -> float:
    """Return ``ref - calc``, wrapping torsion-angle differences."""
    diff = ref_value - calc_value
    if kind == "torsion_angle":
        diff = torsion_wrap(diff)
    return diff


def weighted_residual(kind: str, ref_value: float, calc_value: float, weight: float) -> float:
    """Return ``weight * (ref - calc)`` with torsion wrapping."""
    return weight * raw_residual(kind, ref_value, calc_value)


def regularization_value(params: np.ndarray, reference_params: np.ndarray, lam: float) -> float:
    """L2 penalty ``lambda * ||params - reference_params||**2``."""
    if lam <= 0:
        return 0.0
    diff = np.asarray(params, dtype=float) - np.asarray(reference_params, dtype=float)
    return float(lam) * float(np.dot(diff, diff))


def regularization_gradient(params: np.ndarray, reference_params: np.ndarray, lam: float) -> np.ndarray:
    """Gradient of the L2 penalty: ``2 * lambda * (params - reference_params)``."""
    params = np.asarray(params, dtype=float)
    if lam <= 0:
        return np.zeros_like(params)
    return np.asarray(2.0 * float(lam) * (params - np.asarray(reference_params, dtype=float)), dtype=float)


def regularization_residuals(params: np.ndarray, reference_params: np.ndarray, lam: float) -> np.ndarray:
    """Return ``sqrt(lambda) * (params - reference_params)`` for least squares.

    Appending these to the weighted data residuals makes
    ``sum(residuals**2) == data_loss + lambda * ||p - p_ref||**2``.
    """
    if lam <= 0:
        return np.array([], dtype=float)
    diff = np.asarray(params, dtype=float) - np.asarray(reference_params, dtype=float)
    return np.asarray(np.sqrt(float(lam)) * diff, dtype=float)


def fractional_improvement(initial: float, final: float) -> float:
    """Fractional improvement of a score (0 = no change, 1 = perfect).

    Returns ``(initial - final) / initial``, or ``0.0`` when *initial* is
    zero.
    """
    if initial == 0:
        return 0.0
    return (initial - final) / initial


def r2(ref: np.ndarray, calc: np.ndarray) -> float:
    """Q2MM coefficient of determination; ``nan`` when undefined."""
    ref = np.asarray(ref, dtype=float)
    calc = np.asarray(calc, dtype=float)
    if ref.size < 2:
        return float("nan")
    ss_res = float(np.sum((ref - calc) ** 2))
    mean = float(np.mean(ref))
    ss_tot = float(np.sum((ref - mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def category_stats(ref_values: np.ndarray, calc_values: np.ndarray) -> dict[str, float]:
    """Per-category ``n_refs`` / ``r2`` / ``rmsd`` / ``mae``."""
    ref_values = np.asarray(ref_values, dtype=float)
    calc_values = np.asarray(calc_values, dtype=float)
    n = int(ref_values.size)
    if n == 0:
        return {"n_refs": 0, "r2": float("nan"), "rmsd": float("nan"), "mae": float("nan")}
    residuals = ref_values - calc_values
    return {
        "n_refs": n,
        "r2": r2(ref_values, calc_values),
        "rmsd": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
    }


def category_metrics(plan: ObjectivePlan, evaluation: Evaluation) -> dict[str, dict[str, float]]:
    """Bucket calculated-vs-reference pairs by observation kind, then stat.

    References with ``weight == 0.0`` are skipped (e.g. the imaginary mode
    in TS eigenmatrix fits).  Grouping is by ``observation.kind`` — the
    same buckets the historical ``_per_category_metrics`` produced.
    """
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for obs, calc in zip(plan.observations.values, evaluation.calculated, strict=True):
        if obs.weight == 0.0:
            continue
        buckets[obs.kind].append((float(obs.value), float(calc)))
    return {
        kind: category_stats(
            np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]),
        )
        for kind, pairs in buckets.items()
    }


def evaluate_samples(evaluator: ObjectiveEvaluator, full_vector: np.ndarray, n_evals: int) -> tuple[float, ...]:
    """Sample the real objective ``n_evals`` times at *full_vector*.

    Quantifies per-call engine non-determinism.  Uses the evaluator's
    :meth:`~q2mm.objectives.protocols.ObjectiveEvaluator.sample`, which
    re-evaluates without touching the evaluation counter or history, so
    optimizer bookkeeping is never polluted by post-hoc sampling.
    """
    if n_evals <= 0:
        return ()
    return tuple(float(evaluator.sample(full_vector)) for _ in range(n_evals))
