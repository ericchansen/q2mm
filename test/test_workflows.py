"""Tests for :mod:`q2mm.workflows`.

The acceptance criterion for ``SingleStageWorkflow`` is **bit-identical
parity** with the inline ``ObjectivePlan → ObjectiveExecutor → ScipyOptimizer.optimize``
pattern that the codebase has used to date. These tests run both code
paths on the smallest available system (CH3F) and assert the workflow
produces the same final score, parameter vector, and per-category
metrics as the direct call.
"""

from __future__ import annotations
from q2mm.backends.registry import load_backend

import numpy as np
import pytest

from q2mm.models.results import CandidateRecord, OptimizationResult, StageRecord
from q2mm.objectives.plan import ObjectivePlan
from q2mm.optimizers.protocols import _Optimizer
from q2mm.workflows import SingleStageWorkflow, Workflow, make_evaluator_factory


class TestWorkflowProtocol:
    """The Protocol must accept conforming implementations."""

    def test_single_stage_satisfies_workflow_protocol(self) -> None:
        """``SingleStageWorkflow`` is runtime-checkable as ``Workflow``."""
        assert isinstance(SingleStageWorkflow(), Workflow)

    def test_optimizer_protocol_accepts_scipy_optimizer(self) -> None:
        """``ScipyOptimizer`` satisfies the workflows' ``_Optimizer`` Protocol."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        assert isinstance(ScipyOptimizer(), _Optimizer)


class TestStageRecordDataclass:
    """Field defaults and dataclass semantics."""

    def test_stage_record_required_fields(self) -> None:
        """All non-default fields must be supplied."""
        sr = StageRecord(
            name="opt",
            n_params=2,
            layout_fingerprint="sha256:test",
            initial_score=1.0,
            final_score=0.5,
            n_iterations=10,
            n_evaluations=20,
            converged=True,
            message="ok",
            gradient_mode="analytical",
            elapsed_s=1.5,
        )
        assert sr.locked_param_indices == ()
        assert sr.notes == {}

    def test_stage_record_locked_indices_validated(self) -> None:
        """Locked indices must be unique and within [0, n_params)."""
        common = dict(
            name="opt",
            n_params=3,
            layout_fingerprint="sha256:test",
            initial_score=1.0,
            final_score=0.5,
            n_iterations=1,
            n_evaluations=1,
            converged=True,
            message="ok",
            gradient_mode="analytical",
        )
        with pytest.raises(ValueError, match="unique"):
            StageRecord(locked_param_indices=(0, 0), **common)
        with pytest.raises(ValueError, match="out of range"):
            StageRecord(locked_param_indices=(3,), **common)

    def test_stage_record_fd_step_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="fd_step"):
            StageRecord(
                name="opt",
                n_params=1,
                layout_fingerprint="sha256:test",
                initial_score=1.0,
                final_score=0.5,
                n_iterations=1,
                n_evaluations=1,
                converged=True,
                message="ok",
                gradient_mode="finite_difference",
                fd_step=0.0,
            )

    def test_optimization_result_category_metrics_frozen(self) -> None:
        """Default category_metrics are independent and deeply immutable."""
        common = dict(
            success=True,
            initial_score=1.0,
            final_score=1.0,
            n_iterations=0,
            n_evaluations=0,
            n_params=1,
            layout_fingerprint="sha256:test",
            initial_params=np.array([1.0]),
            final_params=np.array([1.0]),
        )
        wr1 = OptimizationResult(message="x", **common)
        wr2 = OptimizationResult(message="y", **common)
        # Independent per-instance defaults (not a shared mutable dict).
        assert wr1.category_metrics == {}
        assert wr2.category_metrics == {}
        assert wr1.category_metrics is not wr2.category_metrics
        # Deeply frozen: assignment is rejected.
        with pytest.raises(TypeError):
            wr1.category_metrics["x"] = {"n_refs": 1.0}  # type: ignore[index]
        assert wr2.category_metrics == {}

    def test_result_records_use_identity_equality(self) -> None:
        candidate_one = CandidateRecord(
            index=0,
            status="success",
            n_params=3,
            layout_fingerprint="sha256:test",
            initial_params=np.array([1.0, 2.0, 3.0]),
            final_params=np.array([1.1, 2.1, 3.1]),
            initial_score=2.0,
            final_score=1.0,
        )
        candidate_two = CandidateRecord(
            index=0,
            status="success",
            n_params=3,
            layout_fingerprint="sha256:test",
            initial_params=np.array([1.0, 2.0, 3.0]),
            final_params=np.array([1.1, 2.1, 3.1]),
            initial_score=2.0,
            final_score=1.0,
        )
        stage_one = StageRecord(
            name="stage",
            n_params=3,
            layout_fingerprint="sha256:test",
            initial_score=2.0,
            final_score=1.0,
            n_iterations=1,
            n_evaluations=2,
            converged=True,
            message="ok",
            gradient_mode="analytical",
            notes={"vector": np.array([1.0, 2.0, 3.0])},
        )
        stage_two = StageRecord(
            name="stage",
            n_params=3,
            layout_fingerprint="sha256:test",
            initial_score=2.0,
            final_score=1.0,
            n_iterations=1,
            n_evaluations=2,
            converged=True,
            message="ok",
            gradient_mode="analytical",
            notes={"vector": np.array([1.0, 2.0, 3.0])},
        )
        common = {
            "success": True,
            "message": "ok",
            "initial_score": 2.0,
            "final_score": 1.0,
            "n_iterations": 1,
            "n_evaluations": 2,
            "n_params": 3,
            "layout_fingerprint": "sha256:test",
            "initial_params": np.array([1.0, 2.0, 3.0]),
            "final_params": np.array([1.1, 2.1, 3.1]),
        }
        result_one = OptimizationResult(**common)
        result_two = OptimizationResult(**common)

        assert candidate_one != candidate_two
        assert stage_one != stage_two
        assert result_one != result_two
        assert len({candidate_one, candidate_two, stage_one, stage_two, result_one, result_two}) == 6


@pytest.mark.jax
class TestSingleStageWorkflowParity:
    """``SingleStageWorkflow.run()`` must match the direct optimizer call."""

    @staticmethod
    def _load() -> tuple[object, object]:
        """Fresh ``(BenchmarkCase, backend)`` per test."""
        from q2mm.benchmarks.systems import load_system

        backend = load_backend("jax")
        return load_system("ch3f", backend=backend, functional_form="harmonic"), backend

    def test_workflow_matches_direct_call(self) -> None:
        """Same final score and parameter vector as the inline pattern."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case_direct, backend = self._load()
        problem_direct = case_direct.problem
        make_evaluator_direct = make_evaluator_factory(
            backend,
            problem_direct.starting_force_field,
            executor="jax",
        )
        obj_direct = make_evaluator_direct(ObjectivePlan.from_problem(problem_direct))
        opt_direct = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        direct_result = opt_direct.optimize(obj_direct, problem_direct.active_space)

        case_workflow, backend = self._load()
        problem_workflow = case_workflow.problem
        make_evaluator_workflow = make_evaluator_factory(
            backend,
            problem_workflow.starting_force_field,
            executor="jax",
        )
        opt_workflow = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        workflow_result = SingleStageWorkflow().run(problem_workflow, make_evaluator_workflow, opt_workflow, n_evals=0)

        np.testing.assert_array_equal(direct_result.final_params, workflow_result.final_params)
        assert direct_result.final_score == workflow_result.stages[0].final_score
        assert direct_result.n_iterations == workflow_result.stages[0].n_iterations
        assert direct_result.n_evaluations == workflow_result.stages[0].n_evaluations

    def test_workflow_returns_optimized_categories(self) -> None:
        """Per-category metrics dict is populated for systems with eigenmatrix refs."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        make_evaluator = make_evaluator_factory(backend, problem.starting_force_field, executor="jax")
        result = SingleStageWorkflow().run(problem, make_evaluator, opt, n_evals=0)

        assert result.category_metrics, "expected at least one residual category"
        for kind, stats in result.category_metrics.items():
            assert {"n_refs", "r2", "rmsd", "mae"} <= set(stats.keys()), f"missing keys for {kind}"

    def test_workflow_result_records_problem_starting_vector(self) -> None:
        """``OptimizationResult.initial_params`` is the problem starting force-field vector."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        make_evaluator = make_evaluator_factory(backend, problem.starting_force_field, executor="jax")
        result = SingleStageWorkflow().run(problem, make_evaluator, opt, n_evals=0)

        np.testing.assert_array_equal(result.initial_params, problem.layout.vector(problem.starting_force_field))

    def test_n_evals_samples_objective(self) -> None:
        """``n_evals > 0`` populates the sample lists with that many entries."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        make_evaluator = make_evaluator_factory(backend, problem.starting_force_field, executor="jax")
        result = SingleStageWorkflow().run(problem, make_evaluator, opt, n_evals=3)
        assert len(result.initial_samples) == 3
        assert len(result.final_samples) == 3
        assert all(isinstance(s, float) for s in result.initial_samples)

    def test_n_evals_zero_skips_sampling(self) -> None:
        """``n_evals=0`` produces empty sample lists."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, backend = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        make_evaluator = make_evaluator_factory(backend, problem.starting_force_field, executor="jax")
        result = SingleStageWorkflow().run(problem, make_evaluator, opt, n_evals=0)
        assert result.initial_samples == ()
        assert result.final_samples == ()

    def test_post_hoc_sampling_does_not_pollute_optimizer_bookkeeping(self) -> None:
        """``evaluate_samples`` restores n_evaluations and truncates history."""
        from q2mm.objectives.metrics import evaluate_samples

        case, backend = self._load()
        problem = case.problem
        make_evaluator = make_evaluator_factory(
            backend,
            problem.starting_force_field,
            executor="jax",
        )
        obj = make_evaluator(ObjectivePlan.from_problem(problem))
        params = problem.layout.vector(problem.starting_force_field)
        _ = obj.value(params)
        n_eval_baseline = obj.n_evaluations
        history_len_baseline = len(obj.history)

        samples = evaluate_samples(obj, params, 5)
        assert len(samples) == 5
        assert obj.n_evaluations == n_eval_baseline, (
            f"Sampling leaked into n_evaluations: baseline={n_eval_baseline}, after sampling={obj.n_evaluations}"
        )
        assert len(obj.history) == history_len_baseline, (
            f"Sampling leaked into history: baseline len={history_len_baseline}, after sampling len={len(obj.history)}"
        )
