"""Full force-field optimization benchmark via the unified runner.

Runs :func:`q2mm.benchmarks.runner.run_profile` end-to-end for each
available backend and validates that the pipeline produces a terminal
candidate with finite baseline/optimized objective scores.

These tests are ``slow`` because each optimization takes 10-60 s depending
on the backend.
"""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.nightly]


def _run_and_validate(backend: object, backend_key: str) -> None:
    """Run the benchmark pipeline and validate the candidate structure."""
    from q2mm.benchmarks.acceptance import CandidateStatus
    from q2mm.benchmarks.profiles import RunProfile
    from q2mm.benchmarks.runner import run_profile
    from test._shared import backend_functional_form

    profile = RunProfile(
        system="ch3f",
        backend=backend_key,
        functional_form=backend_functional_form(backend).value,
        optimizer="scipy-lbfgsb",
        workflow="single-stage",
        n_evals=0,
    )
    candidate = run_profile(profile, backend=backend, analyze=True, include_device=False)

    # The run must terminate in a state that actually executed an optimization.
    assert candidate.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED), candidate.reason
    summary = candidate.summary
    assert "seminario" in summary
    assert "optimized" in summary
    assert math.isfinite(summary["initial_obj_score"]), summary["initial_obj_score"]
    assert math.isfinite(summary["final_obj_score"]), summary["final_obj_score"]
    assert summary["n_iterations"] >= 0
    # Frequency analysis is included when the backend supports it.
    assert math.isfinite(summary["frequencies"]["initial_rmsd"])
    assert math.isfinite(summary["frequencies"]["final_rmsd"])


@pytest.mark.openmm
def test_optimization_openmm(openmm_backend: object) -> None:
    """Full optimization benchmark with OpenMM."""
    _run_and_validate(openmm_backend, "openmm")


@pytest.mark.jax
def test_optimization_jax(jax_backend: object) -> None:
    """Full optimization benchmark with JAX (harmonic)."""
    _run_and_validate(jax_backend, "jax")


@pytest.mark.jax_md
def test_optimization_jax_md(jax_md_backend: object) -> None:
    """Full optimization benchmark with JAX-MD (OPLSAA)."""
    _run_and_validate(jax_md_backend, "jax-md")
