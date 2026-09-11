"""Aggregate work must not masquerade as a selected candidate's trajectory."""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.results import OptimizationResult, StageRecord
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.optimizers.multistart import MultiStartOptimizer
from test.test_multistart import QuadraticEvaluator


class Scripted:
    def __init__(self, outcomes: list[tuple[bool, float] | Exception]) -> None:
        self.outcomes = outcomes
        self.call = 0
        self.results: list[OptimizationResult] = []

    def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
        index = self.call
        self.call += 1
        initial = evaluator.value(space.baseline)
        outcome = self.outcomes[index]
        if isinstance(outcome, Exception):
            raise outcome
        success, score = outcome
        evaluator.record_evaluation(score)
        stage = StageRecord(
            name="inner",
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_score=initial,
            final_score=score,
            n_iterations=index + 2,
            n_evaluations=2,
            converged=success,
            message="inner",
            gradient_mode="analytical",
        )
        result = OptimizationResult(
            success=success,
            message=f"candidate-{index}",
            initial_score=initial,
            final_score=score,
            n_iterations=index + 2,
            n_evaluations=999,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=space.baseline,
            final_params=space.baseline,
            history=(initial, score),
            method="scripted",
            gradient_mode="analytical",
            stages=(stage,),
        )
        self.results.append(result)
        return result


@pytest.mark.parametrize(
    ("outcomes", "selected", "counts"),
    [
        ([(True, 5.0), RuntimeError("failed"), (False, 0.0)], 0, (2, 1, 2)),
        ([(False, 5.0), (False, 1.0)], 1, (2, 2)),
        ([RuntimeError("failed"), ValueError("also failed")], None, (1, 1)),
    ],
)
def test_measured_aggregate_scope_and_selected_history(
    outcomes: list[tuple[bool, float] | Exception], selected: int | None, counts: tuple[int, ...]
) -> None:
    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    evaluator.value(evaluator.space.baseline)
    before = evaluator.n_evaluations
    inner = Scripted(outcomes)
    result = MultiStartOptimizer(inner, n_starts=len(outcomes), seed=3, verbose=False).optimize(
        evaluator, evaluator.space
    )
    assert result.n_evaluations == evaluator.n_evaluations - before == 1 + sum(counts)
    assert "n_evaluations=aggregate" in result.message
    if selected is None:
        assert "history=initial-baseline" in result.message
        assert "n_iterations=none" in result.message
        assert result.history == (result.initial_score,)
        assert not result.success
    else:
        selected_result = next(item for item in inner.results if item.message == f"candidate-{selected}")
        assert "history=selected-run" in result.message
        assert "n_iterations=selected-run" in result.message
        assert f"selected_candidate={selected}" in result.message
        assert result.history == selected_result.history
        assert result.n_iterations == selected_result.n_iterations
        assert result.stages == selected_result.stages
    assert len(result.candidates) == len(outcomes)


def test_repeated_equal_result_objects_keep_first_selected_index() -> None:
    evaluator = QuadraticEvaluator(np.ones(2))
    fixed = Scripted([(True, 1.0)]).optimize(evaluator, evaluator.space)

    class Reused:
        def optimize(self, objective: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
            objective.value(space.baseline)
            return fixed

    result = MultiStartOptimizer(Reused(), n_starts=2, seed=0, verbose=False).optimize(evaluator, evaluator.space)
    assert "selected_candidate=0" in result.message


def test_scope_labels_survive_single_stage_workflow() -> None:
    from q2mm.objectives.python import PythonObjectiveExecutor
    from q2mm.workflows import SingleStageWorkflow
    from test.test_application import _EnergyBackend, _problem

    problem = _problem()
    inner = Scripted([(False, 1.0), (True, 3.0)])
    result = SingleStageWorkflow().run(
        problem,
        lambda plan: PythonObjectiveExecutor(plan, _EnergyBackend(), problem.starting_force_field),
        MultiStartOptimizer(inner, n_starts=2, seed=1, verbose=False),
        n_evals=0,
    )
    assert "n_evaluations=aggregate" in result.message
    assert "history=selected-run" in result.message
    assert "selected_candidate=1" in result.message
    assert result.stages[0].message == result.message
    assert result.history == inner.results[1].history


def test_recorded_work_and_labels_remain_compatible_with_cycling(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers.catalog import _Construction
    from q2mm.optimizers.cycling import OptimizationLoop, SensitivityResult

    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    space = evaluator.space.with_active_indices([0])
    full = MultiStartOptimizer(Scripted([(True, 3.0), (False, 1.0)]), n_starts=2, seed=1, verbose=False)
    simplex = Scripted([(True, 2.0)])
    monkeypatch.setattr(
        _Construction, "build", lambda plan: full if plan.arguments["method"] == "L-BFGS-B" else simplex
    )
    monkeypatch.setattr(
        "q2mm.optimizers.cycling.compute_sensitivity",
        lambda *args, **kwargs: SensitivityResult(
            d1=np.ones(2), d2=np.ones(2), simp_var=np.ones(2), ranking=np.array([0, 1]), metric="simp_var", n_evals=0
        ),
    )
    result = OptimizationLoop(evaluator, space, max_cycles=1, max_params=1, verbose=False).run()
    assert result.n_evaluations == result.stages[0].n_evaluations == 7
    assert evaluator.n_evaluations == 8  # Cycling's pre-existing outer-baseline scope.
    assert "n_evaluations=aggregate" in result.stages[0].message
    assert "history=selected-run" in result.stages[0].message
    np.testing.assert_array_equal(result.final_params, space.baseline)
