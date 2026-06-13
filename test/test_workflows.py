"""Tests for :mod:`q2mm.workflows`.

The acceptance criterion for ``SingleStageWorkflow`` is **bit-identical
parity** with the inline ``ObjectiveFunction → ScipyOptimizer.optimize``
pattern that the codebase has used to date.  These tests run both code
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
        from q2mm.diagnostics.systems import load_system

        sd = load_system("ch3f", engine=JaxEngine())
        wr1 = WorkflowResult(
            workflow_name="x",
            final_ff=sd.forcefield,
            initial_ff=sd.forcefield,
            stages=[],
        )
        wr2 = WorkflowResult(
            workflow_name="y",
            final_ff=sd.forcefield,
            initial_ff=sd.forcefield,
            stages=[],
        )
        wr1.initial_obj_samples.append(1.0)
        assert wr2.initial_obj_samples == []


@pytest.mark.jax
class TestSingleStageWorkflowParity:
    """``SingleStageWorkflow.run()`` must match the direct optimizer call.

    Each test loads its own SystemData; the optimizer mutates the FF
    inside the ObjectiveFunction in place, so a class-scope fixture
    would leak state between tests.
    """

    @staticmethod
    def _load() -> tuple:
        """Fresh ``(SystemData, engine)`` per test."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.diagnostics.systems import load_system

        engine = JaxEngine()
        return load_system("ch3f", engine=engine), engine

    def test_workflow_matches_direct_call(self) -> None:
        """Same final score and parameter vector as the inline pattern."""
        from q2mm.optimizers.objective import ObjectiveFunction
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        # Path 1: inline pattern (the historical default)
        sd_direct, engine = self._load()
        obj_direct = ObjectiveFunction(sd_direct.forcefield, engine, sd_direct.molecules, sd_direct.reference)
        opt_direct = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        direct_result = opt_direct.optimize(obj_direct)

        # Path 2: SingleStageWorkflow with the same optimizer config, fresh system
        sd_workflow, engine = self._load()
        opt_workflow = ScipyOptimizer(method="L-BFGS-B", maxiter=5, ftol=1e-6, verbose=False)
        workflow_result = SingleStageWorkflow().run(sd_workflow, engine, opt_workflow, n_evals=0)

        np.testing.assert_array_equal(
            direct_result.final_params,
            workflow_result.final_ff.get_param_vector()[sd_workflow.forcefield.active_mask],
        )
        assert direct_result.final_score == workflow_result.stages[0].final_score
        assert direct_result.n_iterations == workflow_result.stages[0].n_iterations
        assert direct_result.n_evaluations == workflow_result.stages[0].n_evaluations

    def test_workflow_returns_optimized_categories(self) -> None:
        """Per-category metrics dict is populated for systems with eigenmatrix refs."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(sd, engine, opt, n_evals=0)

        # CH3F has eigenmatrix-diagonal refs by default
        assert result.optimized_categories, "expected at least one residual category"
        for kind, stats in result.optimized_categories.items():
            assert {"n_refs", "r2", "rmsd", "mae"} <= set(stats.keys()), f"missing keys for {kind}"

    def test_workflow_result_preserves_initial_ff_snapshot(self) -> None:
        """``WorkflowResult.initial_ff`` snapshots starting params even after the optimizer mutates the original.

        The current ``ScipyOptimizer`` mutates the FF inside the
        ObjectiveFunction in place as it searches.  The workflow
        guards against that by snapshotting the initial parameter
        vector before invoking the optimizer, so consumers comparing
        before/after have a stable reference.
        """
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        before = sd.forcefield.get_param_vector().copy()

        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=2, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(sd, engine, opt, n_evals=0)

        np.testing.assert_array_equal(result.initial_ff.get_param_vector(), before)

    def test_n_evals_samples_objective(self) -> None:
        """``n_evals > 0`` populates the sample lists with that many entries."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(sd, engine, opt, n_evals=3)
        assert len(result.initial_obj_samples) == 3
        assert len(result.final_obj_samples) == 3
        assert all(isinstance(s, float) for s in result.initial_obj_samples)

    def test_n_evals_zero_skips_sampling(self) -> None:
        """``n_evals=0`` produces empty sample lists."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        sd, engine = self._load()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, ftol=1e-6, verbose=False)
        result = SingleStageWorkflow().run(sd, engine, opt, n_evals=0)
        assert result.initial_obj_samples == []
        assert result.final_obj_samples == []

    def test_post_hoc_sampling_does_not_pollute_optimizer_bookkeeping(self) -> None:
        """``_evaluate_samples`` restores n_eval and truncates history.

        Without this, every workflow run would leak ``2 * n_evals``
        spurious evaluations into ``ObjectiveFunction.n_eval`` and
        ``ObjectiveFunction.history`` — confusing downstream
        diagnostics and growing memory unboundedly for callers that
        re-use the same ObjectiveFunction across multiple workflow
        runs.  Mirrors the script's
        ``_evaluate_objective_samples`` contract.
        """
        from q2mm.optimizers.objective import ObjectiveFunction

        sd, engine = self._load()
        obj = ObjectiveFunction(sd.forcefield, engine, sd.molecules, sd.reference)
        # Call once to seed n_eval / history (just like the optimizer would).
        params = sd.forcefield.get_param_vector()
        _ = obj(params)
        n_eval_baseline = obj.n_eval
        history_len_baseline = len(obj.history)

        # Sample 5 times; these must not pollute the counters.
        from q2mm.workflows.single_stage import _evaluate_samples

        samples = _evaluate_samples(obj, params, 5)
        assert len(samples) == 5
        assert obj.n_eval == n_eval_baseline, (
            f"Sampling leaked into n_eval: baseline={n_eval_baseline}, after sampling={obj.n_eval}"
        )
        assert len(obj.history) == history_len_baseline, (
            f"Sampling leaked into history: baseline len={history_len_baseline}, after sampling len={len(obj.history)}"
        )
