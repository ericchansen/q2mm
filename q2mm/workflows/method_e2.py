"""Limé & Norrby 2015 Method E2 — two-stage TSFF parameterization.

The published TSFF protocol that addresses the negative bond/angle
force-constant problem (Limé & Norrby, *J. Comput. Chem.* 2015, 36, 244,
DOI:10.1002/jcc.23797).

1. **Round 1 (Method D)**: optimize all active parameters against the
   *unmodified* QM Hessian eigenmatrix (imaginary mode weight 0).
2. **Identify candidates**: bond/angle/UB force constants that drifted to
   zero / went negative in Round 1.
3. **Derive a Round 2 active space** over the same layout, rebased to the
   Round 1 vector, with the candidate rows removed from the active set.
4. **Round 2 (Method C)**: optimize the remaining active parameters
   against an ``invert_ts_curvature``-modified Hessian eigenmatrix.

If Round 1 produces no candidates, the workflow short-circuits to the
Round 1 result.  The original problem/space/force field are never mutated —
Round 2's locked candidates are expressed as a derived, rebased
:class:`~q2mm.models.parameters.ActiveParameterSpace`.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.results import OptimizationResult, StageRecord
from q2mm.models.units import MDYNA_RAD2_TO_KCALMOLRAD2, MDYNA_TO_KCALMOLA2
from q2mm.objectives.metrics import category_metrics, evaluate_samples
from q2mm.objectives.plan import ObjectivePlan

if TYPE_CHECKING:
    from collections.abc import Callable

    from q2mm.models.parameters import ActiveParameterSpace, ParameterId, ParameterKind, ParameterLayout
    from q2mm.models.observations import ObservationSet
    from q2mm.models.problem import OptimizationProblem
    from q2mm.objectives.protocols import ObjectiveEvaluator
    from q2mm.optimizers.protocols import _Optimizer

logger = logging.getLogger(__name__)

_PHYSICAL_FC_KINDS = frozenset({"bond_k", "angle_k", "ub_k"})

APPROXN_BOND_K_MDYNA = 5.0
APPROXN_ANGLE_K_MDYNA_RAD2 = 0.5
APPROXN_DEFAULTS: dict[str, float] = {
    "bond_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,
    "angle_k": APPROXN_ANGLE_K_MDYNA_RAD2 * MDYNA_RAD2_TO_KCALMOLRAD2,
    "ub_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,
}


def _iter_active_force_constants(
    layout: ParameterLayout, active_space: ActiveParameterSpace
) -> list[tuple[int, ParameterKind]]:
    active = {int(i) for i in active_space.active_indices}
    return [
        (slot.index, slot.kind)
        for slot in layout.slots
        if slot.index in active and slot.kind.value in _PHYSICAL_FC_KINDS
    ]


def _identify_method_e2_candidates(
    layout: ParameterLayout,
    active_space: ActiveParameterSpace,
    vector: np.ndarray,
    *,
    threshold: float,
    allow_negative: bool,
) -> list[tuple[int, str, float]]:
    candidates: list[tuple[int, str, float]] = []
    for full_i, kind in _iter_active_force_constants(layout, active_space):
        value = float(vector[full_i])
        is_negative = (not allow_negative) and value < 0.0
        is_near_zero = abs(value) < threshold
        if is_negative or is_near_zero:
            candidates.append((full_i, kind.value, value))
    return candidates


def _row_key(parameter_id: ParameterId) -> tuple[str, tuple[str, ...], int]:
    return (parameter_id.family, parameter_id.identity, parameter_id.occurrence)


def _lock_candidate_rows(layout: ParameterLayout, candidate_indices: set[int]) -> frozenset[int]:
    rows: dict[tuple[str, tuple[str, ...], int], set[int]] = {}
    for slot in layout.slots:
        rows.setdefault(_row_key(slot.id), set()).add(slot.index)
    locked: set[int] = set()
    for full_i in candidate_indices:
        locked.update(rows[_row_key(layout.slots[full_i].id)])
    return frozenset(locked)


def _build_round2_observations(problem: OptimizationProblem, *, replace_with: float) -> ObservationSet:
    """Build Method C observations: geometry + modified-Hessian eigenmatrix."""
    from q2mm.models.hessian import invert_ts_curvature
    from q2mm.models.observations import ObservationSet

    molecules = list(problem.molecules)
    inverted_hessians = [invert_ts_curvature(mol.hessian, replace_with=replace_with) for mol in molecules]
    return ObservationSet.from_molecules(
        molecules,
        case_ids=list(problem.case_ids),
        eigenmatrix_diagonal_only=True,
        eigenmatrix_hessians=inverted_hessians,
    )


class MethodE2Workflow:
    """Limé & Norrby 2015 Method E2 two-stage TSFF parameterization."""

    name: str = "method-e2"

    def __init__(
        self,
        *,
        negative_fc_threshold: float = 1e-3,
        replace_with_round2: float = 1.0,
        allow_negative: bool = False,
        near_zero_replace_with: dict[str, float] | None = None,
    ) -> None:
        if not np.isfinite(negative_fc_threshold) or negative_fc_threshold < 0:
            raise ValueError(f"negative_fc_threshold must be finite and ≥ 0; got {negative_fc_threshold!r}")
        if not np.isfinite(replace_with_round2) or replace_with_round2 <= 0:
            raise ValueError(f"replace_with_round2 must be finite and > 0; got {replace_with_round2!r}")
        self.negative_fc_threshold = float(negative_fc_threshold)
        self.replace_with_round2 = float(replace_with_round2)
        self.allow_negative = bool(allow_negative)
        if near_zero_replace_with is None:
            self.near_zero_replace_with: dict[str, float] = dict(APPROXN_DEFAULTS)
        else:
            for kind_value, val in near_zero_replace_with.items():
                if kind_value not in _PHYSICAL_FC_KINDS:
                    raise ValueError(
                        f"near_zero_replace_with: unsupported kind {kind_value!r}; must be one of "
                        f"{sorted(_PHYSICAL_FC_KINDS)}"
                    )
                if not np.isfinite(val) or val < 0:
                    raise ValueError(f"near_zero_replace_with[{kind_value!r}]={val!r} must be finite and ≥ 0")
            self.near_zero_replace_with = dict(near_zero_replace_with)

    def run(
        self,
        problem: OptimizationProblem,
        make_evaluator: Callable[[ObjectivePlan], ObjectiveEvaluator],
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> OptimizationResult:
        """Execute the two-stage Method E2 protocol; return the canonical result."""
        layout = problem.layout
        n_params = len(layout)
        fingerprint = layout.fingerprint

        # --- Round 1: Method D (unmodified Hessian) ---
        plan_round1 = ObjectivePlan.from_problem(problem)
        obj_round1 = make_evaluator(plan_round1)
        t0 = time.perf_counter()
        round1_result = optimizer.optimize(obj_round1, problem.active_space)
        round1_elapsed = time.perf_counter() - t0
        round1_vector = np.asarray(round1_result.final_params, dtype=float)

        candidates = _identify_method_e2_candidates(
            layout,
            problem.active_space,
            round1_vector,
            threshold=self.negative_fc_threshold,
            allow_negative=self.allow_negative,
        )
        # Build the Round 1 notes fully before constructing the frozen stage.
        round1_notes: dict[str, Any] = {
            "method": "D",
            "imaginary_mode_weight": 0.0,
            "method_e2_candidates": [
                {"full_idx": full_i, "type": kind_value, "round1_value": value, "name": layout.slots[full_i].name}
                for (full_i, kind_value, value) in candidates
            ],
        }

        def _round1_stage() -> StageRecord:
            return StageRecord(
                name="round-1-method-d",
                n_params=n_params,
                layout_fingerprint=fingerprint,
                initial_score=float(round1_result.initial_score),
                final_score=float(round1_result.final_score),
                n_iterations=int(round1_result.n_iterations),
                n_evaluations=int(round1_result.n_evaluations),
                converged=bool(round1_result.success),
                message=str(round1_result.message),
                gradient_mode=round1_result.gradient_mode,
                fd_step=round1_result.fd_step,
                elapsed_s=round1_elapsed,
                notes=round1_notes,
            )

        if not candidates:
            logger.info("MethodE2Workflow: 0 candidates identified, short-circuiting after Round 1")
            return self._finalize(
                obj_round1,
                plan_round1,
                round1_result,
                [_round1_stage()],
                initial_full=round1_result.initial_params,
                final_full=round1_vector,
                n_evals=n_evals,
                n_params=n_params,
                fingerprint=fingerprint,
            )

        logger.info(
            "MethodE2Workflow: identified %d Method E2 candidate(s) after Round 1; running Round 2", len(candidates)
        )

        # --- Near-zero replacements before locking ---
        round2_baseline = round1_vector.copy()
        replaced: list[dict[str, Any]] = []
        for full_i, kind_value, old_value in candidates:
            if kind_value not in self.near_zero_replace_with:
                continue
            new_val = self.near_zero_replace_with[kind_value]
            round2_baseline[full_i] = new_val
            replaced.append({"full_idx": full_i, "type": kind_value, "from": old_value, "to": new_val})
        if replaced:
            round1_notes["near_zero_replacements"] = replaced

        # --- Derive Round 2 active space (candidate rows removed, rebased) ---
        candidate_indices = {full_i for full_i, _kind, _value in candidates}
        locked_row_indices = _lock_candidate_rows(layout, candidate_indices)
        active_before_lock = {int(i) for i in problem.active_space.active_indices}
        round2_active_indices = sorted(active_before_lock - locked_row_indices)
        locked_param_indices = sorted(active_before_lock & locked_row_indices)
        round2_space = problem.active_space.with_baseline(round2_baseline).with_active_indices(round2_active_indices)

        if not round2_active_indices:
            logger.warning("MethodE2Workflow: all candidates locked, no active params for Round 2; skipping Round 2.")
            round1_notes["round_2_skipped"] = "no_active_params_after_lock"
            return self._finalize(
                obj_round1,
                plan_round1,
                round1_result,
                [_round1_stage()],
                initial_full=round1_result.initial_params,
                final_full=round2_baseline,
                n_evals=n_evals,
                n_params=n_params,
                fingerprint=fingerprint,
            )

        round1_stage = _round1_stage()

        # --- Round 2: Method C (modified Hessian; locked candidates) ---
        round2_obs = _build_round2_observations(problem, replace_with=self.replace_with_round2)
        plan_round2 = plan_round1.with_observations(round2_obs).with_active_space(round2_space)
        obj_round2 = make_evaluator(plan_round2)
        t0 = time.perf_counter()
        round2_result = optimizer.optimize(obj_round2, round2_space)
        round2_elapsed = time.perf_counter() - t0
        final_full = np.asarray(round2_result.final_params, dtype=float)

        round2_stage = StageRecord(
            name="round-2-method-c",
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_score=float(round2_result.initial_score),
            final_score=float(round2_result.final_score),
            n_iterations=int(round2_result.n_iterations),
            n_evaluations=int(round2_result.n_evaluations),
            converged=bool(round2_result.success),
            message=str(round2_result.message),
            gradient_mode=round2_result.gradient_mode,
            fd_step=round2_result.fd_step,
            elapsed_s=round2_elapsed,
            locked_param_indices=tuple(locked_param_indices),
            notes={"method": "C", "replace_with": self.replace_with_round2},
        )

        initial_samples = evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
        final_samples = evaluate_samples(obj_round2, final_full, n_evals)
        categories = category_metrics(plan_round2, obj_round2.evaluate(final_full))

        return OptimizationResult(
            success=bool(round2_result.success),
            message=str(round2_result.message),
            initial_score=float(round1_result.initial_score),
            final_score=float(round2_result.final_score),
            n_iterations=int(round1_result.n_iterations) + int(round2_result.n_iterations),
            n_evaluations=int(round1_result.n_evaluations) + int(round2_result.n_evaluations),
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=round1_result.initial_params,
            final_params=final_full,
            history=tuple(round1_result.history) + tuple(round2_result.history),
            method=self.name,
            gradient_mode=round2_result.gradient_mode,
            fd_step=round2_result.fd_step,
            candidates=tuple(round1_result.candidates) + tuple(round2_result.candidates),
            stages=(round1_stage, round2_stage),
            initial_samples=tuple(initial_samples),
            final_samples=tuple(final_samples),
            category_metrics=categories,
        )

    def _finalize(
        self,
        evaluator: ObjectiveEvaluator,
        plan: ObjectivePlan,
        round1_result: OptimizationResult,
        stages: list[StageRecord],
        *,
        initial_full: np.ndarray,
        final_full: np.ndarray,
        n_evals: int,
        n_params: int,
        fingerprint: str,
    ) -> OptimizationResult:
        """Build the canonical result for the short-circuit / all-locked paths.

        The final score is re-evaluated at *final_full* so the reported score
        always corresponds to the returned final vector (which, on the
        all-locked path, differs from the Round 1 final vector because of the
        near-zero replacements).
        """
        final_score = float(evaluator.sample(final_full))
        initial_samples = evaluate_samples(evaluator, initial_full, n_evals)
        final_samples = evaluate_samples(evaluator, final_full, n_evals)
        categories = category_metrics(plan, evaluator.evaluate(final_full))
        return OptimizationResult(
            success=bool(round1_result.success),
            message=str(round1_result.message),
            initial_score=float(round1_result.initial_score),
            final_score=final_score,
            n_iterations=int(round1_result.n_iterations),
            n_evaluations=int(round1_result.n_evaluations),
            n_params=n_params,
            layout_fingerprint=fingerprint,
            initial_params=initial_full,
            final_params=final_full,
            history=round1_result.history,
            method=self.name,
            gradient_mode=round1_result.gradient_mode,
            fd_step=round1_result.fd_step,
            candidates=round1_result.candidates,
            stages=tuple(stages),
            initial_samples=tuple(initial_samples),
            final_samples=tuple(final_samples),
            category_metrics=categories,
        )
