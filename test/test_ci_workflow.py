"""Contracts for backend CI test selection and runtime budgets."""

from __future__ import annotations

from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _jobs() -> dict[str, Any]:
    with CI_WORKFLOW.open(encoding="utf-8") as handle:
        workflow = yaml.safe_load(handle)
    return workflow["jobs"]


def _pytest_command(job: dict[str, Any], *, step_name: str | None = None) -> str:
    commands = [
        step["run"]
        for step in job["steps"]
        if "run" in step and "pytest" in step["run"] and (step_name is None or step.get("name") == step_name)
    ]
    assert len(commands) == 1
    return commands[0]


def test_cross_backend_marker_is_registered(pytestconfig: pytest.Config) -> None:
    markers = pytestconfig.getini("markers")
    assert any(marker.startswith("cross_backend:") for marker in markers)


def test_backend_ci_selectors_partition_cross_backend_tests() -> None:
    jobs = _jobs()
    assert '-m "jax and not cross_backend"' in _pytest_command(jobs["test-jax"])

    native = jobs["test-native"]
    selectors = {entry["name"]: entry["args"] for entry in native["strategy"]["matrix"]["backend"]}
    assert selectors["openmm"] == '-m "openmm and not cross_backend"'
    assert selectors["jax-md"] == '-m "jax_md and not cross_backend"'
    assert selectors["full"] == "--run-integration -m cross_backend"
    command = _pytest_command(native, step_name="Required native correctness tests")
    assert "python -m pytest ${{ matrix.backend.args }} -q" in command
    assert native["timeout-minutes"] == 20


@pytest.mark.parametrize(
    ("backend", "nodes"),
    [
        ("tinker", ["test/integration/test_tinker_backend.py"]),
        (
            "psi4",
            [
                "test/integration/test_psi4_backend.py::TestPsi4BackendLoadMolecule",
                "test/integration/test_psi4_backend.py::TestPsi4EnergyCH3F::test_energy_matches_reference",
            ],
        ),
    ],
)
def test_provisioned_native_lanes_select_bounded_real_integration(backend: str, nodes: list[str]) -> None:
    selectors = {entry["name"]: entry["args"] for entry in _jobs()["test-native"]["strategy"]["matrix"]["backend"]}
    args = shlex.split(selectors[backend])
    assert "--run-integration" in args
    assert "--run-validation" not in args and "--run-nightly" not in args
    assert args[args.index("-m") + 1] == (
        f"{backend} and not (cross_backend or nightly or validation or external_data)"
    )
    assert [arg for arg in args if arg.startswith("test/")] == nodes


@pytest.mark.parametrize(
    ("cases", "executed"),
    [
        ("", 0),
        ("<testcase><skipped message='integration disabled'/></testcase>", 0),
        ("<testcase/><testcase><skipped/></testcase>", 1),
        ("<testcase/><testcase/>", 2),
    ],
)
def test_provisioned_native_lane_cannot_succeed_with_all_skipped_cases(
    tmp_path: Path,
    cases: str,
    executed: int,
) -> None:
    native = _jobs()["test-native"]
    steps = {step["name"]: step for step in native["steps"] if "name" in step}
    command = _pytest_command(native, step_name="Required native correctness tests")
    assert "--junitxml=native-results.xml" in command and "-rs" in command
    guard = steps["Require executed native tests"]
    assert guard["if"] == "steps.native-env.outcome == 'success'"
    assert not guard.get("continue-on-error")
    script = guard["run"].removeprefix("python - <<'PY'\n").removesuffix("PY\n")
    (tmp_path / "native-results.xml").write_text(
        f"<testsuites><testsuite>{cases}</testsuite></testsuites>",
        encoding="utf-8",
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert (result.returncode == 0) is bool(executed)
    if executed:
        assert f"Executed {executed} native test cases." in result.stdout
    else:
        assert "executed no tests" in result.stderr


def test_native_correctness_is_required_after_best_effort_provisioning() -> None:
    native = _jobs()["test-native"]
    assert not native.get("continue-on-error")
    softened = {step["id"] for step in native["steps"] if step.get("continue-on-error")}
    assert softened == {"native-env", "interoperability"}
    steps = {step["name"]: step for step in native["steps"] if "name" in step}
    for name in (
        "Verify native interpreter",
        "Install Q2MM without replacing native dependencies",
        "Required native correctness tests",
        "Require executed native tests",
    ):
        assert steps[name]["if"] == "steps.native-env.outcome == 'success'"
        assert not steps[name].get("continue-on-error")
    interoperability = steps["Required interoperability correctness tests"]
    assert interoperability["if"] == "steps.interoperability.outcome == 'success'"
    assert not interoperability.get("continue-on-error")


def test_tinker_runtime_data_matches_the_installed_native_version() -> None:
    native = _jobs()["test-native"]
    steps = {step["name"]: step for step in native["steps"] if "name" in step}
    setup = steps["Configure version-matched Tinker runtime parameters"]
    assert not setup.get("continue-on-error")
    assert setup["if"] == (
        "steps.native-env.outcome == 'success' && (matrix.backend.name == 'tinker' || matrix.backend.name == 'full')"
    )
    command = setup["run"]
    assert "https://dasher.wustl.edu/tinker/downloads/tinker-8.11.3.tar.gz" in command
    assert "8dfbc9fb8f26742d91139187657e2c905744b0243538f81b75bc04cdc2606ff7" in command
    assert "sha256sum --check -" in command
    assert 'tar -xOzf "$archive" tinker/params/mm3.prm' in command
    assert 'TINKER_DIR="$CONDA_PREFIX/bin"' in command
    assert 'TINKER_PRM="$CONDA_PREFIX/params/mm3.prm"' in command
    assert "load_test_backend('tinker')" in command
    for environment in ("tinker", "full"):
        config = yaml.safe_load((REPO_ROOT / ".github" / "envs" / f"{environment}.yml").read_text(encoding="utf-8"))
        assert "tinker=8.11.3" in config["dependencies"]


def test_jax_ci_timeout_matches_measured_cpu_budget() -> None:
    jax = _jobs()["test-jax"]
    assert jax["timeout-minutes"] == 15
    assert not jax.get("continue-on-error")
    assert not any(step.get("continue-on-error") for step in jax["steps"])
