"""Required responsibilities and native job outcomes, not command snapshots."""

import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import pytest
import yaml

from scripts.ci_paths import JOB_COMPONENTS

ROOT = Path(__file__).resolve().parent.parent


def _jobs(workflow: str = "ci") -> dict[str, Any]:
    return yaml.safe_load((ROOT / ".github" / "workflows" / f"{workflow}.yml").read_text())["jobs"]


def test_required_responsibilities_cannot_hide_failures() -> None:
    jobs = _jobs()
    required = {
        "lint",
        "core-py314",
        "core-py315",
        "docs",
        "release",
        "jax-cpu",
        "qcelemental-io",
        "qcengine-adapter",
        "ase-adapter",
        "parmed-amber",
    } | set(JOB_COMPONENTS)
    assert required <= jobs.keys() and "full" not in jobs
    for job in jobs.values():
        assert not job.get("continue-on-error")
        for step in job["steps"]:
            assert not step.get("continue-on-error") and "|| true" not in step.get("run", "")


@pytest.mark.parametrize("name", JOB_COMPONENTS)
@pytest.mark.parametrize(("affected", "status", "expected"), [("false", 9, 0), ("true", 0, 0), ("true", 9, 9)])
def test_native_path_gate_runs_only_affected_helpers_and_propagates_failure(
    name: str, affected: str, status: int, expected: int
) -> None:
    job = _jobs()[name]
    command = next(step["run"] for step in job["steps"] if "run" in step)
    shell = shutil.which("bash")
    if shell is None:
        pytest.skip("bash is required to execute the native job gate")
    result = subprocess.run(
        [shell, "-c", 'bash() { echo "helper: $*"; return "$STATUS"; }\n' + command],
        env={**os.environ, "AFFECTED": affected, "STATUS": str(status)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == expected, result.stderr
    assert (f"helper: scripts/ci_local.sh --job {name}" in result.stdout) == (affected == "true")


def test_only_image_builds_can_publish_packages() -> None:
    assert _jobs("build-images")["build"]["permissions"]["packages"] == "write"
    assert all(job.get("permissions", {}).get("packages") != "write" for job in _jobs().values())
