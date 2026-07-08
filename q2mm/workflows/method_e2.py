"""Limé & Norrby 2015 Method E2 — two-stage TSFF parameterization.

The published TSFF protocol that addresses the negative bond/angle
force-constant problem documented in Limé & Norrby (J. Comput. Chem.
2015, 36, 244, DOI:10.1002/jcc.23797).

The protocol — paraphrased from paper §97 + Conclusion:

1. **Round 1 (Method D)**: optimize ALL active parameters against the
   *unmodified* QM Hessian eigenmatrix, with the imaginary
   (reaction-coordinate) mode excluded from the fit (weight = 0).
   This is what the existing :class:`~q2mm.workflows.SingleStageWorkflow`
   already does for TSFFs: ``ReferenceData.from_molecules`` uses
   ``mol.hessian`` (the unmodified Hessian) and
   ``add_eigenmatrix_from_hessian(skip_first=True)`` assigns weight 0
   to the imaginary mode.

2. **Identify candidates**: scan the Round 1 final FF for bond/angle/UB
   force constants that drifted to zero / went negative.  These are
   the parameters Limé & Norrby's paper explicitly calls out as
   needing protection (¶97: "we were troubled to see that the FACAF
   bend constant went to zero in the optimization, and would have
   become negative if allowed").

3. **Lock candidates at Round 1 values**: per paper recommendation,
   freeze these candidates at the values they reached in Round 1
   (the "Method D natural" values).

4. **Round 2 (Method C)**: optimize only the *remaining* active
   parameters against a fresh reference set built with the
   ``invert_ts_curvature``-modified Hessian eigenmatrix.  This
   preserves correct steric response along the reaction coordinate
   (paper ¶98) while the locked Method-D values keep the problematic
   FCs from drifting back into the unphysical region.

5. **Restore active mask**: before returning, unfreeze the candidates
   so the caller sees the same active/frozen partition they passed
   in.  The locked values stay; the freeze state is purely a
   workflow-internal bookkeeping.

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
    from q2mm.systems import SystemData
    from q2mm.models.forcefield import ForceField, _FrozenAwareParam
    from q2mm.workflows.base import _Optimizer

logger = logging.getLogger(__name__)


# Force-constant parameter types that are physically required to be
# non-negative (Hooke's-law springs); torsions, stretch-bends, and
# bend-bend cross terms can legitimately be negative.
_PHYSICAL_FC_TYPES = frozenset({"bond_k", "angle_k", "ub_k"})


# --- Q2MM Approxn defaults (Farrugia 2025, JCTC 22, 469) ----------
# Empirical lower-bound force constants used when a parameter has no
# QM-derived value (or has drifted to an unphysical near-zero value
# during optimization).  Numbers are the paper's "Q2MM Approxn"
# standards in MM3 units; we convert once at import time so callers
# get the canonical kcal/(mol·Å²) and kcal/(mol·rad²) values used by
# :class:`~q2mm.models.forcefield.ForceField`.
APPROXN_BOND_K_MDYNA = 5.0  # 5 mdyn/Å
APPROXN_ANGLE_K_MDYNA_RAD2 = 0.5  # 0.5 mdyn·Å/rad²
APPROXN_DEFAULTS: dict[str, float] = {
    "bond_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,
    "angle_k": APPROXN_ANGLE_K_MDYNA_RAD2 * MDYNA_RAD2_TO_KCALMOLRAD2,
    "ub_k": APPROXN_BOND_K_MDYNA * MDYNA_TO_KCALMOLA2,  # UB is a bond-like spring
}


def _iter_active_force_constants(ff: ForceField) -> list[tuple[int, str, _FrozenAwareParam, str]]:
    """Walk active force-constant params; yield ``(full_idx, type, row, attr)``.

    Returns one tuple per active ``bond_k`` / ``angle_k`` / ``ub_k``
    slot, paired with the underlying parameter object (``BondParam``
    or ``AngleParam``) so callers can mutate the FC value and toggle
    the freeze flag.  Cursor logic mirrors
    ``q2mm/workflows/single_stage.py`` and the Phase 9.1 diff script.
    """
    labels = ff.get_param_type_labels()
    active = ff.active_mask
    coll_iters: dict[str, list] = {
        "bonds": list(ff.bonds),
        "angles": list(ff.angles),
        "torsions": list(getattr(ff, "torsions", [])),
        "stretch_bends": list(getattr(ff, "stretch_bends", [])),
        "vdw": list(getattr(ff, "vdws", [])),
        "ub_angles": list(getattr(ff, "_ub_angles", [])),
    }
    type_to_collection = {
        "bond_k": ("bonds", "k"),
        "bond_eq": ("bonds", "eq"),
        "angle_k": ("angles", "k"),
        "angle_eq": ("angles", "eq"),
        "torsion_k": ("torsions", "k"),
        "sb_k": ("stretch_bends", "k"),
        "vdw_radius": ("vdw", "r"),
        "vdw_epsilon": ("vdw", "epsilon"),
        "ub_k": ("ub_angles", "k"),
        "ub_eq": ("ub_angles", "eq"),
    }
    cursor: dict[str, int] = {}
    results: list[tuple[int, str, _FrozenAwareParam, str]] = []

    for full_i, lbl in enumerate(labels):
        coll, attr = type_to_collection.get(lbl, ("?", "?"))
        c = cursor.get(coll, 0)
        row = coll_iters.get(coll, [None])[c] if coll in coll_iters and c < len(coll_iters[coll]) else None
        # Advance cursor on every label (active and frozen) so positions
        # stay aligned with the underlying FF collections.
        if attr == "eq" or coll in {"torsions", "stretch_bends"} or (coll == "vdw" and attr == "epsilon"):
            cursor[coll] = c + 1
        if not active[full_i] or lbl not in _PHYSICAL_FC_TYPES or row is None:
            continue
        results.append((full_i, lbl, row, attr))
    return results


def _identify_method_e2_candidates(
    ff: ForceField, *, threshold: float, allow_negative: bool
) -> list[tuple[int, str, _FrozenAwareParam, float]]:
    """Return active force constants that need Method E2 protection.

    A parameter is a candidate when its current force-constant value
    falls below ``threshold`` in magnitude (drifted toward zero) or
    is strictly negative (when ``allow_negative=False``).  Returns a
    list of ``(full_idx, type, param_row, current_value)`` tuples.
    """
    candidates: list[tuple[int, str, _FrozenAwareParam, float]] = []
    for full_i, lbl, row, _attr in _iter_active_force_constants(ff):
        # ``BondParam`` and ``AngleParam`` both expose ``force_constant``;
        # ``ub_k`` lives on ``AngleParam.ub_force_constant``.
        if lbl == "ub_k":
            value = float(row.ub_force_constant) if row.ub_force_constant is not None else 0.0
        else:
            value = float(row.force_constant)
        is_negative = (not allow_negative) and value < 0.0
        is_near_zero = abs(value) < threshold
        if is_negative or is_near_zero:
            candidates.append((full_i, lbl, row, value))
    return candidates


def _build_round2_references(system: SystemData, *, replace_with: float) -> Any:  # noqa: ANN401 — returns ReferenceData but TYPE_CHECKING circular
    """Build a Method C reference set: geometry + modified-Hessian eigenmatrix.

    Mirrors the call ``load_system`` makes for published-FF strategies
    (``ReferenceData.from_molecules(... eigenmatrix_diagonal_only=True)``)
    except eigenmatrix references use each molecule's Hessian with the
    reaction-coordinate eigenvalue replaced via
    :func:`~q2mm.models.hessian.invert_ts_curvature`.
    """
    from q2mm.models.hessian import invert_ts_curvature
    from q2mm.optimizers.objective import ReferenceData

    inverted_hessians = [invert_ts_curvature(mol.hessian, replace_with=replace_with) for mol in system.molecules]
    return ReferenceData.from_molecules(
        system.molecules,
        eigenmatrix_diagonal_only=True,
        eigenmatrix_hessians=inverted_hessians,
    )


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
            near_zero_replace_with: Per-type replacement values applied
                to candidate force constants *before* freezing them
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

                When a dict of ``{label: value}`` in canonical units
                (kcal/(mol·Å²) for ``bond_k``/``ub_k``,
                kcal/(mol·rad²) for ``angle_k``) is provided,
                candidates of the listed types are reset to the given
                value before freezing.  Keys not present in the dict
                are *not* replaced — the paper-literal
                lock-at-Round-1-value applies to those types.  Pass
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
            for lbl, val in near_zero_replace_with.items():
                if lbl not in _PHYSICAL_FC_TYPES:
                    raise ValueError(
                        f"near_zero_replace_with: unsupported label {lbl!r}; "
                        f"must be one of {sorted(_PHYSICAL_FC_TYPES)}"
                    )
                if not np.isfinite(val) or val < 0:
                    raise ValueError(f"near_zero_replace_with[{lbl!r}]={val!r} must be finite and ≥ 0")
            self.near_zero_replace_with = dict(near_zero_replace_with)

    def run(
        self,
        system: SystemData,
        engine: Any,  # noqa: ANN401
        optimizer: _Optimizer,
        *,
        n_evals: int = 1,
    ) -> WorkflowResult:
        """Execute the two-stage Method E2 protocol.

        .. warning::

           ``system.forcefield`` is mutated in place by both rounds.
           Use ``WorkflowResult.initial_ff`` / ``final_ff`` for stable
           before/after snapshots.  The active/frozen partition on
           ``system.forcefield`` is restored before returning — any
           freezes the workflow applies are reverted.

        Args:
            system: Loaded benchmark system.  Reference data must
                already be built (typically via ``load_system``).
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

        # --- Snapshot initial state ---------------------------------
        initial_params = system.forcefield.get_param_vector().copy()
        initial_ff = system.forcefield.with_params(initial_params)

        # --- Round 1: Method D (unmodified Hessian; existing behaviour)
        obj_round1 = ObjectiveFunction(system.forcefield, engine, system.molecules, system.reference)
        t0 = time.perf_counter()
        round1_result = optimizer.optimize(obj_round1)
        round1_elapsed = time.perf_counter() - t0
        # ``optimizer.optimize`` mutates the FF in place to the final
        # parameter values.  This is the "Method D natural" FF.
        round1_ff = system.forcefield

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
            round1_ff, threshold=self.negative_fc_threshold, allow_negative=self.allow_negative
        )
        round1_stage.notes["method_e2_candidates"] = [
            {"full_idx": full_i, "type": lbl, "round1_value": value, "atoms": "-".join(row.elements)}
            for (full_i, lbl, row, value) in candidates
        ]

        if not candidates:
            # No problematic FCs — short-circuit to the Round 1 result.
            logger.info("MethodE2Workflow: 0 candidates identified, short-circuiting after Round 1")
            initial_samples = _evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
            final_samples = _evaluate_samples(obj_round1, round1_result.final_params, n_evals)
            categories = _per_category_metrics(obj_round1, round1_ff)
            return WorkflowResult(
                workflow_name=self.name,
                final_ff=initial_ff.with_params(round1_result.final_params),
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
        # the row at that value gives a Hooke's-law spring with no
        # restoring force — an unphysical artefact of the bounded
        # search.  ``near_zero_replace_with`` lets the caller (default
        # ``APPROXN_DEFAULTS``) substitute a small empirical positive
        # value for the type so the locked Round 2 force constant is
        # production-usable.  Pass ``near_zero_replace_with={}`` to
        # opt out and recover the strict paper-literal Method E2.
        replaced: list[dict[str, Any]] = []
        for full_i, lbl, row, _value in candidates:
            if lbl not in self.near_zero_replace_with:
                continue
            new_val = self.near_zero_replace_with[lbl]
            old_val = float(row.ub_force_constant) if lbl == "ub_k" else float(row.force_constant)
            if lbl == "ub_k":
                row.ub_force_constant = new_val
            else:
                row.force_constant = new_val
            replaced.append({"full_idx": full_i, "type": lbl, "from": old_val, "to": new_val})
        if replaced:
            round1_stage.notes["near_zero_replacements"] = replaced
            logger.info(
                "MethodE2Workflow: substituted %d candidate FC(s) with Approxn-style defaults before Round 2 lock",
                len(replaced),
            )

        # --- Lock candidates at Round 1 (or replacement) values ----
        # Per Limé & Norrby ¶104, "all force constants that go to zero
        # in the method C refinement should be set to the values found
        # in the method D force field, and subsequently left out of the
        # refinement (method E2)."  Round 1 values ARE the Method D
        # values; just freeze the rows in place.
        #
        # NOTE: ``BondParam.freeze`` / ``AngleParam.freeze`` operate on
        # the whole row (the ``frozen`` flag is per-param, not per-
        # field), so freezing a bond_k candidate also freezes the
        # paired bond_eq.  This matches the paper's "lock at Method D
        # values" semantics in spirit (eq values came from QM geometry
        # and shouldn't drift either) but be aware of the coupling
        # when reasoning about which slots are active in Round 2.
        # Take an active-mask snapshot BEFORE freezing so we can
        # report every slot the freeze actually locked (the freeze is
        # row-scoped, so both ``bond_k`` and the paired ``bond_eq``
        # transition from active to frozen — recording only the
        # ``bond_k`` index would under-report what Round 2 is
        # actually skipping).
        active_before_lock = system.forcefield.active_mask.copy()
        for full_i, lbl, row, _value in candidates:
            row.freeze()
        active_after_lock = system.forcefield.active_mask
        locked_param_indices = [int(i) for i in np.flatnonzero(active_before_lock & ~active_after_lock).tolist()]

        n_active_after_lock = int(system.forcefield.active_mask.sum())
        if n_active_after_lock == 0:
            # Pathological: all active rows had a candidate, leaving
            # nothing for Round 2 to optimize.  Skip Round 2, unfreeze,
            # and return Round 1 result with a notes flag.
            logger.warning(
                "MethodE2Workflow: locking all %d candidate(s) left 0 active params; "
                "skipping Round 2.  Try a tighter ``negative_fc_threshold``.",
                len(candidates),
            )
            for _full_i, _lbl, row, _value in candidates:
                row.unfreeze()
            round1_stage.notes["round_2_skipped"] = "no_active_params_after_lock"
            initial_samples = _evaluate_samples(obj_round1, round1_result.initial_params, n_evals)
            final_samples = _evaluate_samples(obj_round1, round1_result.final_params, n_evals)
            categories = _per_category_metrics(obj_round1, round1_ff)
            return WorkflowResult(
                workflow_name=self.name,
                final_ff=initial_ff.with_params(round1_result.final_params),
                initial_ff=initial_ff,
                stages=[round1_stage],
                initial_obj_samples=initial_samples,
                final_obj_samples=final_samples,
                optimized_categories=categories,
            )

        # --- Round 2: Method C (modified Hessian; locked candidates) ---
        round2_ref = _build_round2_references(system, replace_with=self.replace_with_round2)
        obj_round2 = ObjectiveFunction(system.forcefield, engine, system.molecules, round2_ref)
        try:
            t0 = time.perf_counter()
            round2_result = optimizer.optimize(obj_round2)
            round2_elapsed = time.perf_counter() - t0
        finally:
            # --- Restore active mask: unfreeze the candidates -------
            # The locked values stay (they're the Method D values per
            # the paper); the freeze state is workflow-internal.
            for _full_i, _lbl, row, _value in candidates:
                row.unfreeze()

        round2_ff = system.forcefield

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
        final_full = system.forcefield.get_param_vector()
        initial_samples = _evaluate_samples(obj_round1, initial_params, n_evals)
        final_samples = _evaluate_samples(obj_round2, final_full, n_evals)
        categories = _per_category_metrics(obj_round2, round2_ff)

        # Snapshot final FF (with_params copies; round2_ff is the caller's
        # mutated reference and may still get modified by subsequent code).
        final_ff = initial_ff.with_params(final_full)

        return WorkflowResult(
            workflow_name=self.name,
            final_ff=final_ff,
            initial_ff=initial_ff,
            stages=[round1_stage, round2_stage],
            initial_obj_samples=initial_samples,
            final_obj_samples=final_samples,
            optimized_categories=categories,
        )
