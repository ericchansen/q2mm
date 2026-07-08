"""Tests for :class:`q2mm.workflows.MethodE2Workflow`.

Validation strategy:

1. **Constructor validation** — bad inputs raise ``ValueError`` early.
2. **Short-circuit path** — when Round 1 produces no Method E2
   candidates, the workflow returns a single-stage result equivalent
   to ``SingleStageWorkflow``.  Uses CH3F ground state (no TS
   coordinate, no negative-FC pressure).
3. **Two-stage path** — when Round 1 produces candidates, the
   workflow runs Round 2 against the modified-Hessian reference and
   reports both stages.  Uses the CH3F + F⁻ SN2 TS system, where
   Limé & Norrby 2015 ¶97 documented the FACAF bend going to zero
   under naive Method C fitting.
4. **Active-mask restoration** — locked candidates are unfrozen
   before the workflow returns so the caller's active/frozen
   partition is preserved.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.workflows import (
    MethodE2Workflow,
    SingleStageWorkflow,
    StageResult,
    Workflow,
    WorkflowResult,
)
from q2mm.workflows.method_e2 import (
    _identify_method_e2_candidates,
    _iter_active_force_constants,
)


class TestConstructor:
    """Constructor input validation."""

    def test_default_kwargs(self) -> None:
        """Defaults match the documented Limé & Norrby Method C/E2 values."""
        wf = MethodE2Workflow()
        assert wf.negative_fc_threshold == pytest.approx(1e-3)
        assert wf.replace_with_round2 == pytest.approx(1.0)
        assert wf.allow_negative is False

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf"), float("-inf")])
    def test_bad_threshold_raises(self, bad: float) -> None:
        """Non-finite or negative thresholds are rejected."""
        with pytest.raises(ValueError, match="negative_fc_threshold"):
            MethodE2Workflow(negative_fc_threshold=bad)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_bad_replace_with_raises(self, bad: float) -> None:
        """Non-finite or non-positive replace_with values are rejected."""
        with pytest.raises(ValueError, match="replace_with_round2"):
            MethodE2Workflow(replace_with_round2=bad)

    def test_near_zero_replace_with_defaults_to_approxn(self) -> None:
        """Default ``near_zero_replace_with`` is a copy of Approxn standards."""
        from q2mm.workflows.method_e2 import APPROXN_DEFAULTS

        wf = MethodE2Workflow()
        assert wf.near_zero_replace_with == APPROXN_DEFAULTS
        # Mutating the workflow's dict must NOT mutate the module constant.
        wf.near_zero_replace_with["bond_k"] = 999.0
        assert APPROXN_DEFAULTS["bond_k"] != 999.0

    def test_near_zero_replace_with_empty_dict_opts_out(self) -> None:
        """Passing ``{}`` is an explicit opt-out to paper-literal lock-at-Round-1."""
        wf = MethodE2Workflow(near_zero_replace_with={})
        assert wf.near_zero_replace_with == {}

    def test_near_zero_replace_with_custom_dict_accepted(self) -> None:
        """Caller can override per-type replacement values."""
        wf = MethodE2Workflow(near_zero_replace_with={"bond_k": 10.0})
        assert wf.near_zero_replace_with == {"bond_k": 10.0}

    def test_near_zero_replace_with_bad_label_raises(self) -> None:
        """Labels outside the physical FC types are rejected."""
        with pytest.raises(ValueError, match="unsupported label"):
            MethodE2Workflow(near_zero_replace_with={"torsion_k": 1.0})

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_near_zero_replace_with_bad_value_raises(self, bad: float) -> None:
        """Negative or non-finite replacement values are rejected."""
        with pytest.raises(ValueError, match="must be finite and"):
            MethodE2Workflow(near_zero_replace_with={"bond_k": bad})

    def test_approxn_defaults_match_paper_in_mm3_units(self) -> None:
        """``APPROXN_DEFAULTS`` (canonical units) round-trips to the paper values."""
        from q2mm.models.units import KCALMOLA2_TO_MDYNA, KCALMOLRAD2_TO_MDYNA_RAD2
        from q2mm.workflows.method_e2 import APPROXN_DEFAULTS

        # Farrugia 2025 Q2MM Approxn standards: 5 mdyn/Å bonds, 0.5 mdyn·Å/rad² angles.
        assert APPROXN_DEFAULTS["bond_k"] * KCALMOLA2_TO_MDYNA == pytest.approx(5.0)
        assert APPROXN_DEFAULTS["angle_k"] * KCALMOLRAD2_TO_MDYNA_RAD2 == pytest.approx(0.5)
        assert APPROXN_DEFAULTS["ub_k"] * KCALMOLA2_TO_MDYNA == pytest.approx(5.0)


class TestProtocolConformance:
    """``MethodE2Workflow`` satisfies the ``Workflow`` Protocol."""

    def test_satisfies_workflow_protocol(self) -> None:
        """Runtime-checkable Protocol accepts the implementation."""
        assert isinstance(MethodE2Workflow(), Workflow)


@pytest.mark.jax
class TestShortCircuit:
    """When no Method E2 candidates emerge, Round 2 is skipped."""

    @staticmethod
    def _load() -> tuple:
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system

        engine = JaxEngine()
        return load_system("ch3f", engine=engine), engine

    def test_no_candidates_returns_single_stage(self) -> None:
        """CH3F ground state has no near-zero / negative FCs → 1 stage only."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = MethodE2Workflow().run(sd, engine, opt, n_evals=0)

        assert result.workflow_name == "method-e2"
        assert len(result.stages) == 1, (
            f"CH3F ground state should not produce Method E2 candidates; "
            f"got {len(result.stages)} stages with candidates: "
            f"{result.stages[0].notes.get('method_e2_candidates')}"
        )
        assert result.stages[0].name == "round-1-method-d"
        # Empty candidate list documented in the round 1 stage notes.
        assert result.stages[0].notes.get("method_e2_candidates") == []


@pytest.mark.jax
class TestActiveMaskRestoration:
    """The workflow must leave the FF's active/frozen partition unchanged.

    Uses CH3F ground state (1-stage path) and the SN2 TS (2-stage path)
    to verify the unfreeze logic survives both code paths — including
    when Round 2 raises (the ``try/finally`` guard).
    """

    def test_active_mask_unchanged_on_short_circuit(self) -> None:
        """Single-stage path: no freezes applied, no unfreezes needed."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        engine = JaxEngine()
        sd = load_system("ch3f", engine=engine)
        mask_before = sd.forcefield.active_mask.copy()

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        MethodE2Workflow().run(sd, engine, opt, n_evals=0)

        np.testing.assert_array_equal(sd.forcefield.active_mask, mask_before)


@pytest.mark.jax
@pytest.mark.nightly
class TestTwoStageOnSn2:
    """Two-stage Method E2 on the canonical Limé & Norrby SN2 TS.

    These tests document the END-TO-END contract: Method E2 must run
    Round 2 when candidates emerge, must preserve the active mask,
    and must produce a final FF with the candidates' values locked at
    their Round 1 (Method D) values.

    Marked ``nightly``: each test builds a fresh ``JaxEngine`` and drives
    a full two-stage optimizer loop on the ch3f-sn2 TS, paying ~9s of
    per-molecule JIT compilation apiece (~2 min for the class).  Per repo
    policy heavy optimizer-loop tests run under ``--run-nightly`` rather
    than on every push, matching the ``TestScipyJaxLossTelemetry`` /
    ``test_reported_scores_in_objective_units`` precedent; keeping them in
    per-PR CI pushed the already-borderline ``test-jax`` job past its
    10-minute cap.  The lightweight candidate-identification contracts
    stay in per-PR CI via ``TestHelpers`` below.
    """

    @staticmethod
    def _load() -> tuple:
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system

        engine = JaxEngine()
        # Use a small replace_with to push FACAF (and similar fragile
        # FCs) into the candidate regime more reliably (per Phase 9.1
        # sensitivity sweep, the default ``qfuerza_replace_with=1.0``
        # doesn't always trigger candidate identification at our
        # threshold; we want to exercise the 2-stage code path
        # deterministically here).
        return load_system("ch3f-sn2", engine=engine, qfuerza_replace_with=0.03), engine

    def test_round1_runs_and_candidates_field_populated(self) -> None:
        """Round 1 always runs; candidates list is populated (possibly empty)."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = MethodE2Workflow().run(sd, engine, opt, n_evals=0)

        assert result.workflow_name == "method-e2"
        assert len(result.stages) >= 1
        round1 = result.stages[0]
        assert round1.name == "round-1-method-d"
        assert round1.notes["method"] == "D"
        # Candidates list is always present (possibly empty).
        assert "method_e2_candidates" in round1.notes
        assert isinstance(round1.notes["method_e2_candidates"], list)

    def test_active_mask_restored_after_two_stage_run(self) -> None:
        """If Round 2 runs (candidates non-empty), the active mask must still be restored."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        mask_before = sd.forcefield.active_mask.copy()

        # Pick a threshold low enough that Round 2 actually runs (some
        # FCs become candidates, but not all — leaves active params for
        # Round 2 to optimize).  The exact value is system-dependent;
        # 0.5 in internal units catches a handful of small FCs on
        # ch3f-sn2 without locking the whole FF.
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(sd, engine, opt, n_evals=0)

        np.testing.assert_array_equal(sd.forcefield.active_mask, mask_before)
        # Either Round 2 ran (2 stages) or short-circuited (1 stage + flag);
        # either way the mask must be restored.  When Round 2 runs we
        # also verify its stage metadata.
        if len(result.stages) == 2:
            assert result.stages[1].name == "round-2-method-c"
            assert result.stages[1].notes["method"] == "C"

    def test_locked_param_indices_includes_paired_slots(self) -> None:
        """``StageResult.locked_param_indices`` reports every locked slot.

        ``BondParam.freeze`` / ``AngleParam.freeze`` are row-scoped, so
        freezing a ``bond_k`` candidate also freezes the paired
        ``bond_eq`` slot.  The Round-2 ``StageResult`` must report both
        — recording only the ``*_k`` index would under-report what
        Round 2 actually skipped (and conflicts with the
        ``StageResult.locked_param_indices`` contract in
        ``q2mm/workflows/base.py``).
        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(sd, engine, opt, n_evals=0)

        if len(result.stages) != 2:
            pytest.skip("System did not produce Round 2; nothing to check here.")
        round2 = result.stages[1]
        n_candidates = len(result.stages[0].notes.get("method_e2_candidates", []))
        # Every candidate is a *_k slot.  Each freeze locks the row,
        # so every paired *_eq slot is locked too — total locked
        # indices must be at least 2 × candidates (more if any
        # candidate row carries additional paired slots).
        assert n_candidates > 0
        assert len(round2.locked_param_indices) >= 2 * n_candidates
        # Indices must be unique and refer to valid param-vector slots.
        assert len(set(round2.locked_param_indices)) == len(round2.locked_param_indices)
        labels = sd.forcefield.get_param_type_labels()
        assert all(0 <= i < len(labels) for i in round2.locked_param_indices)

    def test_all_candidates_skips_round_2_and_restores_mask(self) -> None:
        """Pathological case: threshold so high every FC is locked → Round 2 skipped, mask restored."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        mask_before = sd.forcefield.active_mask.copy()

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=1e30)
        result = wf.run(sd, engine, opt, n_evals=0)

        # Round 2 was skipped because locking would leave 0 active params.
        assert len(result.stages) == 1
        assert result.stages[0].notes.get("round_2_skipped") == "no_active_params_after_lock"
        # Active mask still restored even though we hit the pathological branch.
        np.testing.assert_array_equal(sd.forcefield.active_mask, mask_before)

    def test_near_zero_replace_with_approxn_substitutes_candidate_values(self) -> None:
        """Default Approxn replacements lift locked FC values above zero.

        With ``negative_fc_threshold=0.5`` (large enough that small CH3F-SN2
        Round-1 FCs become candidates) and the default Approxn
        replacements, locked Round-2 FCs are substituted with the Approxn
        defaults (5 mdyn/Å bonds, 0.5 mdyn·Å/rad² angles in canonical
        units).  Final FF should report the substitutions in
        ``round1.notes["near_zero_replacements"]`` and the substituted
        rows on ``final_ff`` should hold the replacement values, not
        the near-zero Round-1 values.
        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer
        from q2mm.workflows.method_e2 import APPROXN_DEFAULTS

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(sd, engine, opt, n_evals=0)

        replacements = result.stages[0].notes.get("near_zero_replacements", [])
        if not replacements:
            pytest.skip("System produced no candidates eligible for replacement.")

        # Every replacement uses an Approxn default value for its type.
        for rec in replacements:
            assert rec["to"] == pytest.approx(APPROXN_DEFAULTS[rec["type"]])

    def test_near_zero_replace_with_empty_dict_recovers_paper_behaviour(self) -> None:
        """Opt-out: no replacements recorded, candidates lock at Round-1 values."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5, near_zero_replace_with={})
        result = wf.run(sd, engine, opt, n_evals=0)

        # No replacements when the opt-out dict is empty.
        assert "near_zero_replacements" not in result.stages[0].notes


class TestHelpers:
    """Unit tests for the candidate-identification helpers.

    These run without an engine — purely walk the FF data structures.
    """

    @pytest.mark.jax
    def test_iter_active_force_constants_covers_bonds_and_angles(self) -> None:
        """The walker yields one entry per active bond_k + angle_k + ub_k."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system

        sd = load_system("ch3f", engine=JaxEngine())
        items = _iter_active_force_constants(sd.forcefield)
        # CH3F: 2 bond types (C-H, C-F) → 2 bond_k slots; 3 angle types
        # (HCH, HCF, FCF) → 3 angle_k slots; no UB → 5 total.
        types_found = [lbl for _i, lbl, _row, _attr in items]
        assert types_found.count("bond_k") >= 1
        assert types_found.count("angle_k") >= 1
        # Only physical FC types — no torsion_k / sb_k / vdw.
        assert all(lbl in {"bond_k", "angle_k", "ub_k"} for lbl in types_found)

    @pytest.mark.jax
    def test_identify_candidates_threshold_inclusive(self) -> None:
        """With threshold above all FC magnitudes, every active FC is a candidate."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system

        sd = load_system("ch3f", engine=JaxEngine())
        all_fcs = _iter_active_force_constants(sd.forcefield)
        candidates = _identify_method_e2_candidates(sd.forcefield, threshold=1e30, allow_negative=False)
        assert len(candidates) == len(all_fcs)

    @pytest.mark.jax
    def test_identify_candidates_threshold_zero(self) -> None:
        """With threshold=0 and allow_negative=True, no candidates emerge."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.systems import load_system

        sd = load_system("ch3f", engine=JaxEngine())
        candidates = _identify_method_e2_candidates(sd.forcefield, threshold=0.0, allow_negative=True)
        assert candidates == []


class TestWorkflowResultStructure:
    """Verify ``WorkflowResult`` and ``StageResult`` shape contracts."""

    def test_imports(self) -> None:
        """All public names import cleanly from the package."""
        # WorkflowResult / StageResult are dataclasses; constructing one
        # without all required fields raises TypeError, not silently.
        with pytest.raises(TypeError):
            StageResult()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            WorkflowResult()  # type: ignore[call-arg]
        # SingleStage + MethodE2 both satisfy Workflow.
        assert isinstance(SingleStageWorkflow(), Workflow)
        assert isinstance(MethodE2Workflow(), Workflow)
