"""Tests for :class:`q2mm.workflows.MethodE2Workflow`."""

from __future__ import annotations
from q2mm.backends.registry import load_backend

import numpy as np
import pytest

from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow, StageResult, Workflow, WorkflowResult
from q2mm.workflows.method_e2 import APPROXN_DEFAULTS, _identify_method_e2_candidates, _iter_active_force_constants


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
        wf = MethodE2Workflow()
        assert wf.near_zero_replace_with == APPROXN_DEFAULTS
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
        with pytest.raises(ValueError, match="unsupported kind"):
            MethodE2Workflow(near_zero_replace_with={"torsion_k": 1.0})

    @pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
    def test_near_zero_replace_with_bad_value_raises(self, bad: float) -> None:
        """Negative or non-finite replacement values are rejected."""
        with pytest.raises(ValueError, match="must be finite and"):
            MethodE2Workflow(near_zero_replace_with={"bond_k": bad})

    def test_approxn_defaults_match_paper_in_mm3_units(self) -> None:
        """``APPROXN_DEFAULTS`` (canonical units) round-trips to the paper values."""
        from q2mm.models.units import KCALMOLA2_TO_MDYNA, KCALMOLRAD2_TO_MDYNA_RAD2

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
    def _load() -> tuple[object, object]:
        from q2mm.benchmarks.systems import load_system

        backend = load_backend("jax")
        return load_system("ch3f", backend=backend, functional_form="harmonic"), backend

    def test_no_candidates_returns_single_stage(self) -> None:
        """CH3F ground state has no near-zero / negative FCs → 1 stage only."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = MethodE2Workflow().run(case.problem, backend, opt, n_evals=0)

        assert result.workflow_name == "method-e2"
        assert len(result.stages) == 1, (
            f"CH3F ground state should not produce Method E2 candidates; "
            f"got {len(result.stages)} stages with candidates: "
            f"{result.stages[0].notes.get('method_e2_candidates')}"
        )
        assert result.stages[0].name == "round-1-method-d"
        assert result.stages[0].notes.get("method_e2_candidates") == []


@pytest.mark.jax
class TestProblemImmutability:
    """The workflow must leave the input problem unchanged."""

    def test_active_space_unchanged_on_short_circuit(self) -> None:
        """Single-stage path derives no new active space on the problem object."""
        from q2mm.benchmarks.systems import load_system
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        backend = load_backend("jax")
        case = load_system("ch3f", backend=backend, functional_form="harmonic")
        problem = case.problem
        active_before = np.array(problem.active_space.active_indices, copy=True)
        baseline_before = np.array(problem.active_space.baseline, copy=True)

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = MethodE2Workflow().run(problem, backend, opt, n_evals=0)

        np.testing.assert_array_equal(problem.active_space.active_indices, active_before)
        np.testing.assert_array_equal(problem.active_space.baseline, baseline_before)
        assert result.initial_ff is problem.starting_force_field


@pytest.mark.jax
@pytest.mark.nightly
class TestTwoStageOnSn2:
    """Two-stage Method E2 on the canonical Limé & Norrby SN2 TS."""

    @staticmethod
    def _load() -> tuple[object, object]:
        from q2mm.benchmarks.systems import load_system

        backend = load_backend("jax")
        return load_system("ch3f-sn2", backend=backend, qfuerza_replace_with=0.03, functional_form="harmonic"), backend

    def test_round1_runs_and_candidates_field_populated(self) -> None:
        """Round 1 always runs; candidates list is populated (possibly empty)."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = MethodE2Workflow().run(case.problem, backend, opt, n_evals=0)

        assert result.workflow_name == "method-e2"
        assert len(result.stages) >= 1
        round1 = result.stages[0]
        assert round1.name == "round-1-method-d"
        assert round1.notes["method"] == "D"
        assert "method_e2_candidates" in round1.notes
        assert isinstance(round1.notes["method_e2_candidates"], list)

    def test_problem_active_space_preserved_after_two_stage_run(self) -> None:
        """Running Method E2 does not mutate the caller's active space."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        active_before = np.array(problem.active_space.active_indices, copy=True)
        baseline_before = np.array(problem.active_space.baseline, copy=True)

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(problem, backend, opt, n_evals=0)

        np.testing.assert_array_equal(problem.active_space.active_indices, active_before)
        np.testing.assert_array_equal(problem.active_space.baseline, baseline_before)
        if len(result.stages) == 2:
            assert result.stages[1].name == "round-2-method-c"
            assert result.stages[1].notes["method"] == "C"

    def test_locked_param_indices_includes_paired_slots(self) -> None:
        """``StageResult.locked_param_indices`` reports every locked slot.

        Locking a Method E2 candidate force constant (``bond_k``/
        ``angle_k``/``ub_k``) is row-scoped, not scalar-scoped: the
        paired equilibrium slot (``bond_eq``/``angle_eq``/``ub_eq``) of
        the *same* physical parameter is locked too, at its Round-1
        (QM-derived geometry) value — exactly what the pre-Phase-2
        row-scoped ``BondParam.freeze()``/``AngleParam.freeze()`` API
        did (freezing a row froze both its force constant and its
        equilibrium value together). Reporting only the ``*_k`` index
        would under-report what Round 2 actually skips (and conflicts
        with the ``StageResult.locked_param_indices`` contract in
        ``q2mm/workflows/base.py``).
        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(problem, backend, opt, n_evals=0)

        if len(result.stages) != 2:
            pytest.skip("System did not produce Round 2; nothing to check here.")
        round1 = result.stages[0]
        round2 = result.stages[1]
        candidates = round1.notes.get("method_e2_candidates", [])
        n_candidates = len(candidates)
        assert n_candidates > 0

        # Every candidate is a *_k slot. Locking is row-scoped, so every
        # paired *_eq slot is locked too — total locked indices must be
        # at least 2x candidates (more if any candidate row carries
        # additional paired slots).
        assert len(round2.locked_param_indices) >= 2 * n_candidates
        # Indices must be unique and refer to valid param-vector slots.
        assert len(set(round2.locked_param_indices)) == len(round2.locked_param_indices)
        layout = problem.layout
        assert all(0 <= i < len(layout) for i in round2.locked_param_indices)

        # Explicitly verify: each candidate's own sibling *_eq slot (same
        # family + chemical identity + occurrence — i.e. the same row,
        # per ParameterId, differing only in `field`) is locked. This is
        # computed independently of q2mm.workflows.method_e2's own
        # row-grouping helper, so it validates the *outcome*, not just
        # that the implementation agrees with itself.
        for record in candidates:
            candidate_slot = layout.slots[record["full_idx"]]
            siblings = [
                slot
                for slot in layout.slots
                if slot.index != candidate_slot.index
                and slot.id.family == candidate_slot.id.family
                and slot.id.identity == candidate_slot.id.identity
                and slot.id.occurrence == candidate_slot.id.occurrence
            ]
            assert siblings, f"Candidate {candidate_slot.name!r} has no paired eq slot in the layout."
            for sibling in siblings:
                assert sibling.index in round2.locked_param_indices, (
                    f"Candidate {candidate_slot.name!r} (full_idx={candidate_slot.index}) was locked, "
                    f"but its paired slot {sibling.name!r} (full_idx={sibling.index}) was not."
                )

    def test_all_candidates_skips_round_2_and_preserves_problem(self) -> None:
        """If every active FC is locked, Round 2 is skipped without mutating the problem."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        active_before = np.array(problem.active_space.active_indices, copy=True)
        baseline_before = np.array(problem.active_space.baseline, copy=True)

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=1e30)
        result = wf.run(problem, backend, opt, n_evals=0)

        assert len(result.stages) == 1
        assert result.stages[0].notes.get("round_2_skipped") == "no_active_params_after_lock"
        np.testing.assert_array_equal(problem.active_space.active_indices, active_before)
        np.testing.assert_array_equal(problem.active_space.baseline, baseline_before)

    def test_near_zero_replace_with_approxn_substitutes_candidate_values(self) -> None:
        """Default Approxn replacements lift locked FC values above zero."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        result = wf.run(problem, backend, opt, n_evals=0)

        replacements = result.stages[0].notes.get("near_zero_replacements", [])
        if not replacements:
            pytest.skip("System produced no candidates eligible for replacement.")

        final_vector = problem.layout.vector(result.final_ff)
        for rec in replacements:
            assert rec["to"] == pytest.approx(APPROXN_DEFAULTS[rec["type"]])
            assert final_vector[rec["full_idx"]] == pytest.approx(rec["to"])

    def test_near_zero_replace_with_empty_dict_recovers_paper_behaviour(self) -> None:
        """Opt-out: no replacements recorded, candidates lock at Round-1 values."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        wf = MethodE2Workflow(negative_fc_threshold=0.5, near_zero_replace_with={})
        result = wf.run(problem, backend, opt, n_evals=0)

        assert "near_zero_replacements" not in result.stages[0].notes
        candidates = result.stages[0].notes.get("method_e2_candidates", [])
        if not candidates:
            pytest.skip("System produced no Method E2 candidates.")

        final_vector = problem.layout.vector(result.final_ff)
        for rec in candidates:
            assert final_vector[rec["full_idx"]] == pytest.approx(rec["round1_value"])


class TestHelpers:
    """Unit tests for the candidate-identification helpers."""

    @pytest.mark.jax
    def test_iter_active_force_constants_covers_bonds_and_angles(self) -> None:
        """The walker yields one entry per active bond_k + angle_k + ub_k."""
        from q2mm.benchmarks.systems import load_system

        case = load_system("ch3f", backend=load_backend("jax"), functional_form="harmonic")
        items = _iter_active_force_constants(case.problem.layout, case.problem.active_space)
        types_found = [kind.value for _idx, kind in items]
        assert types_found.count("bond_k") >= 1
        assert types_found.count("angle_k") >= 1
        assert all(label in {"bond_k", "angle_k", "ub_k"} for label in types_found)

    @pytest.mark.jax
    def test_identify_candidates_threshold_inclusive(self) -> None:
        """With threshold above all FC magnitudes, every active FC is a candidate."""
        from q2mm.benchmarks.systems import load_system

        case = load_system("ch3f", backend=load_backend("jax"), functional_form="harmonic")
        problem = case.problem
        all_fcs = _iter_active_force_constants(problem.layout, problem.active_space)
        candidates = _identify_method_e2_candidates(
            problem.layout,
            problem.active_space,
            problem.layout.vector(problem.starting_force_field),
            threshold=1e30,
            allow_negative=False,
        )
        assert len(candidates) == len(all_fcs)

    @pytest.mark.jax
    def test_identify_candidates_threshold_zero(self) -> None:
        """With threshold=0 and allow_negative=True, no candidates emerge."""
        from q2mm.benchmarks.systems import load_system

        case = load_system("ch3f", backend=load_backend("jax"), functional_form="harmonic")
        problem = case.problem
        candidates = _identify_method_e2_candidates(
            problem.layout,
            problem.active_space,
            problem.layout.vector(problem.starting_force_field),
            threshold=0.0,
            allow_negative=True,
        )
        assert candidates == []


class TestWorkflowResultStructure:
    """Verify ``WorkflowResult`` and ``StageResult`` shape contracts."""

    def test_imports(self) -> None:
        """All public names import cleanly from the package."""
        with pytest.raises(TypeError):
            StageResult()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            WorkflowResult()  # type: ignore[call-arg]
        assert isinstance(SingleStageWorkflow(), Workflow)
        assert isinstance(MethodE2Workflow(), Workflow)
