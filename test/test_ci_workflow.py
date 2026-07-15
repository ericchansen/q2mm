"""Contracts for backend CI test selection and runtime budgets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _jobs() -> dict[str, Any]:
    with CI_WORKFLOW.open(encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    return workflow["jobs"]


def _pytest_command(job: dict[str, Any]) -> str:
    commands = [step["run"] for step in job["steps"] if "run" in step and "pytest" in step["run"]]
    assert len(commands) == 1
    return commands[0]


def test_cross_backend_marker_is_registered(pytestconfig: pytest.Config) -> None:
    markers = pytestconfig.getini("markers")
    assert any(marker.startswith("cross_backend:") for marker in markers)


def test_backend_ci_selectors_partition_cross_backend_tests() -> None:
    jobs = _jobs()
    selectors = {
        "test-openmm": "openmm",
        "test-tinker": "tinker",
        "test-jax": "jax",
        "test-jax-md": "jax_md",
    }
    for job_name, marker in selectors.items():
        assert f'-m "{marker} and not cross_backend"' in _pytest_command(jobs[job_name])

    cross_command = _pytest_command(jobs["test-cross-backend"])
    assert "--run-integration -m cross_backend -q" in cross_command
    assert jobs["test-cross-backend"]["timeout-minutes"] == 10


def test_jax_ci_timeout_matches_measured_cpu_budget() -> None:
    assert _jobs()["test-jax"]["timeout-minutes"] == 15
