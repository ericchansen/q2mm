"""Generic engine cases have explicit backend selection independent of installed extras."""

from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "backend,marker", [("openmm", "openmm"), ("tinker", "tinker"), ("jax", "jax"), ("jax-md", "jax_md")]
)
def test_engine_contract_selection_is_backend_specific(backend: str, marker: str) -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "test/integration/test_engine_contract.py",
            "--run-integration",
            "--collect-only",
            "-q",
            "-m",
            f"{marker} and not cross_backend",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    cases = [line for line in result.stdout.splitlines() if line.startswith("test/") and "::" in line]
    assert cases
    assert all(line.endswith(f"[{backend}]") for line in cases), cases
