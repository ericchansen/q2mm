"""Tests for :mod:`q2mm.workflows`.

The acceptance criterion for ``SingleStageWorkflow`` is **bit-identical
parity** with the inline ``ObjectiveFunction → ScipyOptimizer.optimize``
pattern that the codebase has used to date. These tests run both code
paths on the smallest available system (CH3F) and assert the workflow
produces the same final score, parameter vector, and per-category
metrics as the direct call.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.workflows import SingleStageWorkflow, StageResult, Workflow, WorkflowResult
from q2mm.workflows.base import _Optimizer


class TestWorkflowProtocol:
    """The Protocol must accept conforming implementations."""

    def test_single_stage_satisfies_workflow_protocol(self) -> None:
        """``SingleStageWorkflow`` is runtime-checkable as ``Workflow``."""
        assert isinstance(SingleStageWorkflow(), Workflow)

    def test_optimizer_protocol_accepts_scipy_optimizer(self) -> None:
        """``ScipyOptimizer`` satisfies the workflows' ``_Optimizer`` Protocol."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        assert isinstance(ScipyOptimizer(), _Optimizer)


class TestStageResultDataclass:
    """Field defaults and dataclass semantics."""

    def test_stage_result_required_fields(self) -> None:
        """All non-default fields must be supplied."""
        sr = StageResult(
            name="opt",
            initial_score=1.0,
            final_score=0.5,
            n_iterations=10,
            n_evaluations=20,
            converged=True,
            message="ok",
            jac_mode="analytical",
            elapsed_s=1.5,
        )
        assert sr.locked_param_indices == []
        assert sr.notes == {}

    @pytest.mark.jax
    def test_workflow_result_default_lists(self) -> None:
        """Default lists/dicts are per-instance (not shared)."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.benchmarks.systems import load_system

        case = load_system("ch3f", engine=JaxEngine(), functional_form="harmonic")
        ff = case.problem.starting_force_field
        wr1 = WorkflowResult(
            workflow_name="x",
            final_ff=ff,
            initial_ff=ff,
            stages=[],
        )
        wr2 = WorkflowResult(
            workflow_name="y",
            final_ff=ff,
            initial_ff=ff,
            stages=[],
        )
        wr1.initial_obj_samples.append(1.0)
        assert wr2.initial_obj_samples == []


@pytest.mark.jax
class TestSingleStageWorkflowParity:
    """``SingleStageWorkflow.run()`` must match the direct optimizer call."""

    @staticmethod
    def _load() -> tuple[object, object]:
        """Fresh ``(BenchmarkCase, engine)`` per test."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.benchmarks.systems import load_system

        engine = JaxEngine()
        return load_system("ch3f", engine=engine, functional_form="harmonic"), engine

    def test_workflow_matches_direct_call(self) -> None:
        """Same final score and parameter vector as the inline pattern."""
        from q2mm.optimizers.objective import ObjectiveFunction
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case_direct, engine = self._load()
        problem_direct = case_direct.problem
        obj_direct = ObjectiveFunction(
            problem_direct.starting_force_field,
            engine,
            list(problem_direct.molecules),
            problem_direct.observations,
            case_ids=list(problem_direct.case_ids),
            layout=problem_direct.layout,
        )
        opt_direct = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        direct_result = opt_direct.optimize(obj_direct, problem_direct.active_space)

        case_workflow, engine = self._load()
        problem_workflow = case_workflow.problem
        opt_workflow = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        workflow_result = SingleStageWorkflow().run(problem_workflow, engine, opt_workflow, n_evals=0)

        np.testing.assert_array_equal(
            direct_result.final_params,
            problem_workflow.layout.vector(workflow_result.final_ff),
        )
        assert direct_result.final_score == workflow_result.stages[0].final_score
        assert direct_result.n_iterations == workflow_result.stages[0].n_iterations
        assert direct_result.n_evaluations == workflow_result.stages[0].n_evaluations

    def test_workflow_returns_optimized_categories(self) -> None:
        """Per-category metrics dict is populated for systems with eigenmatrix refs."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, engine = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(problem, engine, opt, n_evals=0)

        assert result.optimized_categories, "expected at least one residual category"
        for kind, stats in result.optimized_categories.items():
            assert {"n_refs", "r2", "rmsd", "mae"} <= set(stats.keys()), f"missing keys for {kind}"

    def test_workflow_result_reuses_problem_starting_force_field(self) -> None:
        """``WorkflowResult.initial_ff`` is the immutable problem starting force field."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, engine = self._load()
        problem = case.problem

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(problem, engine, opt, n_evals=0)

        assert result.initial_ff is problem.starting_force_field
        assert result.initial_ff == problem.starting_force_field

    def test_n_evals_samples_objective(self) -> None:
        """``n_evals > 0`` populates the sample lists with that many entries."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, engine = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(problem, engine, opt, n_evals=3)
        assert len(result.initial_obj_samples) == 3
        assert len(result.final_obj_samples) == 3
        assert all(isinstance(s, float) for s in result.initial_obj_samples)

    def test_n_evals_zero_skips_sampling(self) -> None:
        """``n_evals=0`` produces empty sample lists."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        case, engine = self._load()
        problem = case.problem
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(problem, engine, opt, n_evals=0)
        assert result.initial_obj_samples == []
        assert result.final_obj_samples == []

    def test_post_hoc_sampling_does_not_pollute_optimizer_bookkeeping(self) -> None:
        """``_evaluate_samples`` restores n_eval and truncates history."""
        from q2mm.optimizers.objective import ObjectiveFunction
        from q2mm.workflows.single_stage import _evaluate_samples

        case, engine = self._load()
        problem = case.problem
        obj = ObjectiveFunction(
            problem.starting_force_field,
            engine,
            list(problem.molecules),
            problem.observations,
            case_ids=list(problem.case_ids),
            layout=problem.layout,
        )
        params = problem.layout.vector(problem.starting_force_field)
        _ = obj(params)
        n_eval_baseline = obj.n_eval
        history_len_baseline = len(obj.history)

        samples = _evaluate_samples(obj, params, 5)
        assert len(samples) == 5
        assert obj.n_eval == n_eval_baseline, (
            f"Sampling leaked into n_eval: baseline={n_eval_baseline}, after sampling={obj.n_eval}"
        )
        assert len(obj.history) == history_len_baseline, (
            f"Sampling leaked into history: baseline len={history_len_baseline}, after sampling len={len(obj.history)}"
        )
