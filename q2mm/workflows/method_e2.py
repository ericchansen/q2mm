"""Limé & Norrby 2015 Method E2 — two-stage TSFF parameterization.

The published TSFF protocol that addresses the negative bond/angle
force-constant problem documented in Limé & Norrby (J. Comput. Chem.
2015, 36, 244, DOI:10.1002/jcc.23797).

The protocol — paraphrased from paper §97 + Conclusion:

1. **Round 1 (Method D)**: optimize ALL active parameters against the
   *unmodified* QM Hessian eigenmatrix, with the imaginary
   (reaction-coordinate) mode excluded from the fit (weight = 0).
   This is what the existing :class:`~q2mm.workflows.SingleStageWorkflow`
   already does for TSFFs: ``ObservationSet.from_molecules`` uses
   ``mol.hessian`` (the unmodified Hessian) and
   ``add_eigenmatrix_from_hessian(skip_first=True)`` assigns weight 0
   to the imaginary mode.

2. **Identify candidates**: scan the Round 1 final full-vector for
   bond/angle/UB force constants that drifted to zero / went negative.
   These are the parameters Limé & Norrby's paper explicitly calls out
   as needing protection (¶97: "we were troubled to see that the FACAF
   bend constant went to zero in the optimization, and would have
   become negative if allowed").

3. **Derive a round-2 active space**: build a new
   :class:`~q2mm.models.parameters.ActiveParameterSpace` over the same
   layout, rebased to the Round 1 final vector, with the candidates'
   full-vector indices removed from the active set (never mutating the
   original problem's force field or active space).

4. **Round 2 (Method C)**: optimize only the *remaining* active
   parameters against a fresh reference set built with the
   ``invert_ts_curvature``-modified Hessian eigenmatrix.  This
   preserves correct steric response along the reaction coordinate
   (paper ¶98) while the locked Method-D values keep the problematic
   FCs from drifting back into the unphysical region.

If Round 1 produces no candidates, the workflow short-circuits to
``SingleStageWorkflow`` behavior: returns the Round 1 result with one
stage and a ``notes["method_e2_candidates"]: []`` entry.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.units import MDYNA_RAD2_TO_KCALMOLRAD2, MDYNA_TO_KCALMOLA2
from q2mm.workflows.base import StageResult, WorkflowResult
from q2mm.workflows.single_stage import _evaluate_samples, _per_category_metrics

if TYPE_CHECKING:
    from q2mm.models.parameters import ActiveParameterSpace, ParameterId, ParameterKind, ParameterLayout
    from q2mm.models.problem import OptimizationProblem
    from q2mm.workflows.base import _Optimizer

logger = logging.getLogger(__name__)


# Force-constant parameter kinds that are physically required to be
# non-negative (Hooke's-law springs); torsions, stretch-bends, and
# bend-bend cross terms can legitimately be negative.
_PHYSICAL_FC_KINDS = frozenset({"bond_k", "angle_k", "ub_k"})


# --- Q2MM Approxn defaults (Farrugia 2025, JCTC 22, 469) ----------
# Empirical lower-bound force constants used when a parameter has no
# QM-derived value (or has drifted to an unphysical near-zero value
# during optimization).  Numbers are the paper's "Q2MM Approxn"
# standards in MM3 units; we convert once at import time so callers
# get the canonical kcal/(mol·Å²) and kcal/(mol·rad²) values used by
# :class:`~q2mm.models.parameters.ParameterLayout`.
APPROXN_BOND_K_MDYNA = 5.0  # 5 mdyn/Å
APPROXN_ANGLE_K_MDYNA_RAD2 = 0.5  # 0.5 mdyn·Å/rad²
APPROXN_DEFAULTS: dict[str, float] = {
    "bond_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,
    "angle_k": APPROXN_ANGLE_K_MDYNA_RAD2 * MDYNA_RAD2_TO_KCALMOLRAD2,
    "ub_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,  # UB is a bond-like spring
}


def _iter_active_force_constants(
    layout: ParameterLayout, active_space: ActiveParameterSpace
) -> list[tuple[int, ParameterKind]]:
    """Yield ``(full_idx, kind)`` for every active bond_k/angle_k/ub_k slot."""
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
    """Return active force constants that need Method E2 protection.

    A parameter is a candidate when its current force-constant value
    (``vector[full_idx]``) falls below ``threshold`` in magnitude
    (drifted toward zero) or is strictly negative (when
    ``allow_negative=False``).  Returns a list of
    ``(full_idx, kind_value, current_value)`` tuples.
    """
    candidates: list[tuple[int, str, float]] = []
    for full_i, kind in _iter_active_force_constants(layout, active_space):
        value = float(vector[full_i])
        is_negative = (not allow_negative) and value < 0.0
        is_near_zero = abs(value) < threshold
        if is_negative or is_near_zero:
            candidates.append((full_i, kind.value, value))
    return candidates


def _row_key(parameter_id: ParameterId) -> tuple[str, tuple[str, ...], int]:
    """Semantic row identity for a slot: family + chemical identity + occurrence.

    Two ``ParameterSlot`` objects belong to the same physical parameter
    row — and thus must be locked together — iff this key matches; only
    :attr:`ParameterId.field` differs between them (e.g. a bond's
    ``force_constant`` vs. its ``equilibrium``). An angle's
    Urey-Bradley ``ub_k``/``ub_eq`` slots do NOT share a row with that
    same angle's own bending ``angle_k``/``angle_eq`` slots even though
    both live on the same ``owner="angles"``/``owner_index`` —
    ``ParameterId.family`` is ``"urey_bradley"`` for the former and
    ``"angle"`` for the latter.
    """
    return (parameter_id.family, parameter_id.identity, parameter_id.occurrence)


def _lock_candidate_rows(layout: ParameterLayout, candidate_indices: set[int]) -> frozenset[int]:
    """Expand *candidate_indices* to every full-vector slot sharing a row.

    Limé & Norrby ¶104's "lock at the Method D values" applies to the
    *whole* physical parameter a candidate force constant belongs to,
    not just the force-constant scalar itself: locking a ``bond_k``/
    ``angle_k``/``ub_k`` candidate must also lock its paired
    ``bond_eq``/``angle_eq``/``ub_eq`` slot at its Round-1 (QM-derived
    geometry) value, exactly as the pre-Phase-2 row-scoped
    ``BondParam.freeze()``/``AngleParam.freeze()`` API did.

    Expansion uses only :class:`~q2mm.models.parameters.ParameterLayout`
    / :class:`~q2mm.models.parameters.ParameterId` metadata — the
    semantic row identity from :func:`_row_key` — never manual block
    arithmetic or tuple-position assumptions, so one implementation
    covers bonds/angles/Urey-Bradley uniformly without a hardcoded
    per-kind pairing table.
    """
    rows: dict[tuple[str, tuple[str, ...], int], set[int]] = {}
    for slot in layout.slots:
        rows.setdefault(_row_key(slot.id), set()).add(slot.index)
    locked: set[int] = set()
    for full_i in candidate_indices:
        locked.update(rows[_row_key(layout.slots[full_i].id)])
    return frozenset(locked)


def _build_round2_problem(problem: OptimizationProblem, *, replace_with: float) -> OptimizationProblem:
    """Build a Method C problem: geometry + modified-Hessian eigenmatrix.

    Mirrors the reference construction every published-FF system
    loader uses (``ObservationSet.from_molecules``) except eigenmatrix
    references use each molecule's Hessian with the reaction-coordinate
    eigenvalue replaced via
    :func:`~q2mm.models.hessian.invert_ts_curvature`.
    """
    import dataclasses

    from q2mm.models.hessian import invert_ts_curvature
    from q2mm.models.observations import ObservationSet

    molecules = list(problem.molecules)
    inverted_hessians = [invert_ts_curvature(mol.hessian, replace_with=replace_with) for mol in molecules]
    observations = ObservationSet.from_molecules(
        molecules,
        case_ids=list(problem.case_ids),
        eigenmatrix_diagonal_only=True,
        eigenmatrix_hessians=inverted_hessians,
    )
    return dataclasses.replace(problem, observations=observations)


class MethodE2Workflow:
    """Limé & Norrby 2015 Method E2 two-stage TSFF parameterization.

    See the module docstring for the full protocol.  Constructor
    knobs control the candidate-identification threshold and the
    Round 2 inversion magnitude; the inner ``optimizer`` is the same
    primitive (typically :class:`~q2mm.optimizers.scipy_opt.ScipyOptimizer`)
    used by ``SingleStageWorkflow`` — Method E2 is a workflow over
    optimizers, not a new search algorithm.
    """

    name: str = "method-e2"

    def __init__(
        self,
        *,
        negative_fc_threshold: float = 1e-3,
        replace_with_round2: float = 1.0,
        allow_negative: bool = False,
        near_zero_replace_with: dict[str, float] | None = None,
    ) -> None:
        """Configure the protocol.

        Args:
            negative_fc_threshold: Magnitude threshold (internal units —
                kcal mol⁻¹ Å⁻² for ``bond_k``/``ub_k``; kcal mol⁻¹ rad⁻²
                for ``angle_k``) below which a Round 1 force constant
                is flagged as a Method E2 candidate.  Default ``1e-3``
                is effectively "drifted to zero".
            replace_with_round2: Replacement value (Hartree/Bohr²) for
                the most-negative TS-Hessian eigenvalue when building
                Round 2 references.  Default ``1.0`` matches Limé &
                Norrby Method C.
            allow_negative: If ``False`` (default), strictly negative
                force constants are flagged as candidates regardless of
                magnitude.  If ``True``, only the ``< threshold``
                check applies.  Set ``True`` only when you have
                specifically reasoned about why a negative bond/angle
                FC is acceptable for your system.
            near_zero_replace_with: Per-kind replacement values applied
                to candidate force constants *before* locking them
                for Round 2.  When ``None`` (default), uses
                :data:`APPROXN_DEFAULTS` — the Q2MM Approxn standards
                from Farrugia 2025 (5 mdyn/Å for bond/UB, 0.5
                mdyn·Å/rad² for angles).  This addresses the case
                where Round 1 plus the new non-negative bounds parks
                a force constant at exactly ``0.0``: locking at ``0``
                gives a Hooke's-law spring with no restoring force
                (unphysical), so we substitute a small empirical
                positive value that keeps the MM potential
                well-defined for downstream production use.

                When a dict of ``{kind_value: value}`` in canonical
                units (kcal/(mol·Å²) for ``bond_k``/``ub_k``,
                kcal/(mol·rad²) for ``angle_k``) is provided,
                candidates of the listed kinds are reset to the given
                value before locking. Keys not present in the dict
                are *not* replaced — the paper-literal
                lock-at-Round-1-value applies to those kinds. Pass
                ``{}`` for the strict paper-literal behavior with no
                replacements (Limé & Norrby ¶104: lock at Round 1 /
                Method D values).

        """
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
                        f"near_zero_replace_with: unsupported kind {kind_value!r}; "
                        f"must be one of {sorted(_PHYSICAL_FC_KINDS)}"
                    )
                if not np.isfinite(val) or val < 0:
                    raise ValueError(f"near_zero_replace_with[{kind_value!r}]={val!r} must be finite and ≥ 0")
            self.near_zero_replace_with = dict(near_zero_replace_with)

    def run(
        self,
        problem: OptimizationProblem,
        engine: Any,  # noqa: ANN401
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> WorkflowResult:
        """Execute the two-stage Method E2 protocol.

        Args:
            problem: Loaded optimization problem.  ``problem.starting_force_field``
                is never mutated — Round 2's locked candidates are
                expressed as a derived, rebased
                :class:`~q2mm.models.parameters.ActiveParameterSpace`,
                not by freezing rows in place.
            engine: MM backend used to evaluate both rounds.
            optimizer: Pre-configured optimizer.  The same instance
                is used for both Round 1 and Round 2; for a single
                run this is fine, but be aware the optimizer's
                internal state may carry between rounds (e.g.
                cached engine handles).
            n_evals: Real-objective samples at initial and final
                parameter vectors for noise quantification.

        Returns:
            :class:`WorkflowResult` with up to 2 :class:`StageResult`
            entries (``"round-1-method-d"`` and, if any candidates
            were identified, ``"round-2-method-c"``).

        """
        from q2mm.optimizers.objective import ObjectiveFunction

        initial_ff = problem.starting_force_field
        layout = problem.layout
        molecules = list(problem.molecules)

        # --- Round 1: Method D (unmodified Hessian; existing behaviour)
        obj_round1 = ObjectiveFunction(
            initial_ff, engine, molecules, problem.observations, case_ids=list(problem.case_ids), layout=layout
        )
        t0 = time.perf_counter()
        round1_result = optimizer.optimize(obj_round1, problem.active_space)
        round1_elapsed = time.perf_counter() - t0
        round1_vector = np.asarray(round1_result.final_params, dtype=float)
        round1_ff = layout.replace(initial_ff, round1_vector)

        round1_stage = StageResult(
            name="round-1-method-d",
            initial_score=float(round1_result.initial_score),
            final_score=float(round1_result.final_score),
            n_iterations=int(round1_result.n_iterations),
            n_evaluations=int(round1_result.n_evaluations),
            converged=bool(round1_result.success),
            message=str(round1_result.message),
            jac_mode=str(round1_result.jac_mode) if round1_result.jac_mode is not None else "unknown",
            elapsed_s=round1_elapsed,
            notes={"method": "D", "imaginary_mode_weight": 0.0},
        )

        # --- Identify Method E2 candidates --------------------------
        candidates = _identify_method_e2_candidates(
            layout,
            problem.active_space,
            round1_vector,
            threshold=self.negative_fc_threshold,
            allow_negative=self.allow_negative,
        )
        round1_stage.notes["method_e2_candidates"] = [
            {"full_idx": full_i, "type": kind_value, "round1_value": value, "name": layout.slots[full_i].name}
            for (full_i, kind_value, value) in candidates
        ]

        if not candidates:
            # No problematic FCs — short-circuit to the Round 1 result.
            logger.info("MethodE2Workflow: 0 candidates identified, short-circuiting after Round 1")
            initial_samples = _evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
            final_samples = _evaluate_samples(obj_round1, round1_result.final_params, n_evals)
            categories = _per_category_metrics(obj_round1, round1_ff)
            return WorkflowResult(
                workflow_name=self.name,
                final_ff=round1_ff,
                initial_ff=initial_ff,
                stages=[round1_stage],
                initial_obj_samples=initial_samples,
                final_obj_samples=final_samples,
                optimized_categories=categories,
            )

        logger.info(
            "MethodE2Workflow: identified %d Method E2 candidate(s) after Round 1; "
            "locking at Round 1 values and running Round 2 against modified Hessian",
            len(candidates),
        )

        # --- Apply near-zero replacements before locking -----------
        # When Round 1 (with the new non-negative bounds on bond_k /
        # angle_k / ub_k) parks a candidate at exactly 0.0, locking
        # the value at that value gives a Hooke's-law spring with no
        # restoring force (unphysical), an artefact of the bounded
        # search.  ``near_zero_replace_with`` lets the caller (default
        # ``APPROXN_DEFAULTS``) substitute a small empirical positive
        # value for the kind so the locked Round 2 force constant is
        # production-usable.  Pass ``near_zero_replace_with={}`` to
        # opt out and recover the strict paper-literal Method E2.
        round2_baseline = round1_vector.copy()
        replaced: list[dict[str, Any]] = []
        for full_i, kind_value, old_value in candidates:
            if kind_value not in self.near_zero_replace_with:
                continue
            new_val = self.near_zero_replace_with[kind_value]
            round2_baseline[full_i] = new_val
            replaced.append({"full_idx": full_i, "type": kind_value, "from": old_value, "to": new_val})
        if replaced:
            round1_stage.notes["near_zero_replacements"] = replaced
            logger.info(
                "MethodE2Workflow: substituted %d candidate FC(s) with Approxn-style defaults before Round 2 lock",
                len(replaced),
            )

        # --- Derive a Round 2 active space: candidate rows removed, ---
        # --- rebased to the (possibly Approxn-adjusted) Round 1 values.
        # Per Limé & Norrby ¶104, "all force constants that go to zero
        # in the method C refinement should be set to the values found
        # in the method D force field, and subsequently left out of the
        # refinement (method E2)."  This never mutates the original
        # problem's force field or active space — it derives a new
        # ActiveParameterSpace by exact full-vector index.
        #
        # Locking is row-scoped, not scalar-scoped: each candidate
        # *_k is expanded (via ParameterLayout's semantic row identity
        # — see _lock_candidate_rows) to include its paired *_eq slot,
        # matching the pre-Phase-2 row-scoped BondParam.freeze()/
        # AngleParam.freeze() semantics (freezing a row froze both its
        # force constant and its equilibrium value together). Both the
        # candidate and its paired eq are removed from Round 2's active
        # set and both are reported in locked_param_indices — reporting
        # only the *_k index would under-report what Round 2 actually
        # skips.
        candidate_indices = {full_i for full_i, _kind, _value in candidates}
        locked_row_indices = _lock_candidate_rows(layout, candidate_indices)
        active_before_lock = {int(i) for i in problem.active_space.active_indices}
        round2_active_indices = sorted(active_before_lock - locked_row_indices)
        locked_param_indices = sorted(active_before_lock & locked_row_indices)
        round2_space = problem.active_space.with_baseline(round2_baseline).with_active_indices(round2_active_indices)

        if not round2_active_indices:
            # Pathological: all active rows had a candidate, leaving
            # nothing for Round 2 to optimize.  Skip Round 2 and return
            # the (Approxn-adjusted) Round 1 result with a notes flag.
            logger.warning(
                "MethodE2Workflow: locking all %d candidate(s) left 0 active params; "
                "skipping Round 2.  Try a tighter ``negative_fc_threshold``.",
                len(candidates),
            )
            round1_stage.notes["round_2_skipped"] = "no_active_params_after_lock"
            adjusted_ff = layout.replace(initial_ff, round2_baseline)
            initial_samples = _evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
            final_samples = _evaluate_samples(obj_round1, round2_baseline, n_evals)
            categories = _per_category_metrics(obj_round1, adjusted_ff)
            return WorkflowResult(
                workflow_name=self.name,
                final_ff=adjusted_ff,
                initial_ff=initial_ff,
                stages=[round1_stage],
                initial_obj_samples=initial_samples,
                final_obj_samples=final_samples,
                optimized_categories=categories,
            )

        # --- Round 2: Method C (modified Hessian; locked candidates) ---
        round2_problem = _build_round2_problem(problem, replace_with=self.replace_with_round2)
        round2_start_ff = layout.replace(initial_ff, round2_baseline)
        obj_round2 = ObjectiveFunction(
            round2_start_ff,
            engine,
            molecules,
            round2_problem.observations,
            case_ids=list(round2_problem.case_ids),
            layout=layout,
        )
        t0 = time.perf_counter()
        round2_result = optimizer.optimize(obj_round2, round2_space)
        round2_elapsed = time.perf_counter() - t0

        final_full = np.asarray(round2_result.final_params, dtype=float)
        round2_ff = layout.replace(initial_ff, final_full)

        round2_stage = StageResult(
            name="round-2-method-c",
            initial_score=float(round2_result.initial_score),
            final_score=float(round2_result.final_score),
            n_iterations=int(round2_result.n_iterations),
            n_evaluations=int(round2_result.n_evaluations),
            converged=bool(round2_result.success),
            message=str(round2_result.message),
            jac_mode=str(round2_result.jac_mode) if round2_result.jac_mode is not None else "unknown",
            elapsed_s=round2_elapsed,
            locked_param_indices=locked_param_indices,
            notes={"method": "C", "replace_with": self.replace_with_round2},
        )

        # --- Post-hoc samples + per-category metrics (against Round 2 obj)
        # Use obj_round2 (modified-Hessian reference) for consistency with
        # the final stage — Method E2's defining feature is that final
        # validation uses the Method C reference data.
        initial_samples = _evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
        final_samples = _evaluate_samples(obj_round2, final_full, n_evals)
        categories = _per_category_metrics(obj_round2, round2_ff)

        return WorkflowResult(
            workflow_name=self.name,
            final_ff=round2_ff,
            initial_ff=initial_ff,
            stages=[round1_stage, round2_stage],
            initial_obj_samples=initial_samples,
            final_obj_samples=final_samples,
            optimized_categories=categories,
        )
