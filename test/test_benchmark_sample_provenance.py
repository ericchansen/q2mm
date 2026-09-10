"""Keep optimizer-executor samples distinct from the objective of record."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Literal

import pytest

from q2mm.backends.contracts import Backend
from q2mm.benchmarks.acceptance import CandidateStatus
from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.profiles import RunProfile
from q2mm.benchmarks.runner import _score_interval_summary, persist_candidate, run_profile
from q2mm.models.observations import ObservationSet
from test.test_application import _EnergyBackend, _problem, _result


@pytest.mark.parametrize(
    ("optimizer", "executor", "gradient"),
    [
        ("scipy-lbfgsb", "python", "finite_difference"),
        pytest.param("scipy-lbfgsb-jax", "jax", "analytical", marks=pytest.mark.jax),
    ],
)
def test_workflow_samples_identify_their_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optimizer: str, executor: str, gradient: str
) -> None:
    problem = dataclasses.replace(_problem(), observations=ObservationSet().with_energy(90.75, case_id="h2"))
    backend: Backend = _EnergyBackend()
    final_k = 99.0
    if executor == "jax":
        pytest.importorskip("jax")
        from q2mm.backends.mm.jax_engine import JaxBackend

        backend = JaxBackend()
        molecule = problem.molecules[0].with_geometry([[0.0, 0.0, 0.0], [0.85, 0.0, 0.0]])
        problem = dataclasses.replace(
            problem,
            cases=(dataclasses.replace(problem.cases[0], molecule=molecule),),
            observations=ObservationSet().with_energy(-9.0, case_id="h2"),
        )
        final_k = 0.0
    initial_ff = problem.starting_force_field
    final_ff = dataclasses.replace(
        initial_ff, bonds=(dataclasses.replace(initial_ff.bonds[0], force_constant=final_k),)
    )
    result = dataclasses.replace(
        _result(problem, gradient_mode=gradient),
        initial_score=1000.0,
        final_score=500.0,
        final_params=problem.layout.vector(final_ff),
        n_iterations=5,
        initial_samples=(1000.0, 1000.0),
        final_samples=(500.0, 500.0),
    )
    case = BenchmarkCase(key="ch3f", name="synthetic", problem=problem, default_forms=("harmonic",))
    monkeypatch.setattr("q2mm.benchmarks.systems.load_system", lambda *_args, **_kwargs: case)
    monkeypatch.setattr(
        "q2mm.application.optimization.execute_optimization", lambda *_args, **_kwargs: (result, final_ff)
    )

    candidate = run_profile(
        RunProfile(
            system="ch3f",
            backend=backend.info.name,
            functional_form="harmonic",
            optimizer=optimizer,
            n_evals=2,
            regularization=0.0,
            executor_ratio_tol=None,
        ),
        backend=backend,
        analyze=False,
        include_device=False,
    )

    assert candidate.status is CandidateStatus.ACCEPTED, candidate.reason
    summary = candidate.summary
    assert summary["initial_obj_score"] == pytest.approx(100.0)
    assert summary["final_obj_score"] == pytest.approx(81.0)
    assert summary["improvement_pct"] == pytest.approx(19.0)
    assert summary["initial_optimizer_score_mean"] == 1000.0
    assert summary["final_optimizer_score_mean"] == 500.0
    assert summary["optimizer_improvement_pct_mean"] == 50.0
    assert summary["optimizer_improvement_significant"] is True
    assert summary["optimizer_samples_executor"] == executor
    assert summary["optimizer_sample_statistics_version"] == 1
    assert summary["optimizer_initial_sample_count"] == summary["optimizer_final_sample_count"] == 2
    for misleading in (
        "initial_obj_score_mean",
        "initial_obj_score_ci95",
        "final_obj_score_mean",
        "final_obj_score_ci95",
        "improvement_pct_mean",
        "improvement_significant",
    ):
        assert misleading not in summary

    record = json.loads(persist_candidate(tmp_path, candidate, provenance={"generator": "test"}).read_text())
    assert record["summary"]["optimizer_samples_executor"] == executor
    assert record["summary"]["initial_optimizer_score_mean"] == 1000.0
    assert record["optimization_result"]["initial_samples"] == [1000.0, 1000.0]
    assert record["optimization_result"]["final_samples"] == [500.0, 500.0]


@pytest.mark.parametrize("executor", ["python", "jax"])
@pytest.mark.parametrize(("initial", "final"), [((), ()), ((1.0,), ()), ((), (1.0,))])
def test_missing_samples_do_not_invent_statistics(
    executor: Literal["python", "jax"], initial: tuple[float, ...], final: tuple[float, ...]
) -> None:
    assert _score_interval_summary(initial, final, executor=executor) == {}
