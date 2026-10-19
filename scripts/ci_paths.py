"""Select the eight native/parity CI responsibilities affected by changed paths."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import subprocess

JOB_COMPONENTS = {
    "openmm-cpu": {"openmm"},
    "tinker": {"tinker"},
    "jax-md-cpu": {"jax-md"},
    "psi4-reference": {"psi4"},
    "parity-openmm-jax": {"openmm", "jax"},
    "parity-openmm-tinker": {"openmm", "tinker"},
    "parity-jax-jaxmd": {"jax", "jax-md"},
    "parity-psi4-qcengine": {"psi4", "qcengine"},
}

SHARED = (
    "q2mm/__init__.py",
    "q2mm/_canonical.py",
    "q2mm/_provenance.py",
    "q2mm/constants.py",
    "q2mm/elements.py",
    "q2mm/geometry.py",
    "q2mm/preparation.py",
    "q2mm/resources.py",
    "q2mm/settings.py",
    "q2mm/models/*",
    "q2mm/objectives/*",
    "q2mm/data/sn2/*",
    "q2mm/backends/__init__.py",
    "q2mm/backends/contracts.py",
    "q2mm/backends/discovery.py",
    "q2mm/backends/registry.py",
    "q2mm/io/__init__.py",
    "q2mm/io/_helpers.py",
    "q2mm/io/xyz.py",
    "pyproject.toml",
    ".github/workflows/ci.yml",
    ".github/docker/Dockerfile",
    "scripts/ci_paths.py",
    "scripts/ci_image.py",
    "scripts/ci_local.sh",
    "scripts/check_test_execution.py",
    "test/conftest.py",
    "test/_shared.py",
    "test/backend_fixtures.py",
    "test/integration/test_engine_contract.py",
)

COMPONENT_PATHS = {
    "openmm": (
        "q2mm/backends/mm/__init__.py",
        "q2mm/backends/mm/_term_support.py",
        "q2mm/backends/mm/*openmm*.py",
        "q2mm/io/openmm.py",
        "q2mm/io/cmap.py",
    ),
    "tinker": (
        "q2mm/backends/mm/__init__.py",
        "q2mm/backends/mm/tinker.py",
        "q2mm/io/tinker.py",
    ),
    "jax": (
        "q2mm/backends/mm/__init__.py",
        "q2mm/backends/mm/_term_support.py",
        "q2mm/backends/mm/jax_engine.py",
        "q2mm/backends/mm/_jax_common.py",
        "q2mm/backends/mm/batched.py",
        "q2mm/_jax_support.py",
    ),
    "jax-md": (
        "q2mm/backends/mm/__init__.py",
        "q2mm/backends/mm/_term_support.py",
        "q2mm/backends/mm/jax_md_engine.py",
        "q2mm/backends/mm/_jax_common.py",
        "q2mm/_jax_support.py",
    ),
    "psi4": (
        "q2mm/backends/reference/__init__.py",
        "q2mm/backends/reference/psi4.py",
    ),
    "qcengine": (
        "q2mm/backends/reference/__init__.py",
        "q2mm/backends/reference/qcengine.py",
    ),
}

IMAGE_JOBS = {
    "openmm": {"openmm-cpu"},
    "tinker": {"tinker"},
    "jax-md": {"jax-md-cpu", "parity-jax-jaxmd"},
    "psi4": {"psi4-reference", "parity-psi4-qcengine"},
    "openmm-jax": {"parity-openmm-jax"},
    "openmm-tinker": {"parity-openmm-tinker"},
}

JOB_PATHS = {
    "openmm-cpu": (
        "test/test_openmm*.py",
        "test/integration/test_openmm*.py",
    ),
    "tinker": ("test/test_tinker*.py", "test/integration/test_tinker_backend.py"),
    "jax-md-cpu": (
        "test/test_jaxmd_minimize_controls.py",
        "test/integration/test_jax_md_backend.py",
    ),
    "psi4-reference": ("test/integration/test_psi4_backend.py",),
    "parity-openmm-jax": (
        "test/test_backend_contracts.py",
        "test/test_mm3_jax.py",
        "test/test_urey_bradley.py",
        "test/integration/test_jax_openmm_improper_parity.py",
        "test/benchmarks/test_cross_engine.py",
        "test/benchmarks/conftest.py",
    ),
    "parity-openmm-tinker": ("test/benchmarks/test_cross_engine.py", "test/benchmarks/conftest.py"),
    "parity-jax-jaxmd": ("test/integration/test_jax_md_backend.py",),
    "parity-psi4-qcengine": ("test/integration/test_qcengine_psi4_parity.py",),
}


def classify(paths: Iterable[str]) -> dict[str, bool]:
    """Return fixed, safe output names; backend-specific changes stay pairwise."""
    affected: set[str] = set()
    for raw in paths:
        path = raw.replace("\\", "/").removeprefix("./")
        if any(fnmatchcase(path, pattern) for pattern in SHARED):
            return dict.fromkeys(JOB_COMPONENTS, True)
        components = {
            component
            for component, patterns in COMPONENT_PATHS.items()
            if any(fnmatchcase(path, pattern) for pattern in patterns)
        }
        affected.update(job for job, required in JOB_COMPONENTS.items() if required & components)
        for image, jobs in IMAGE_JOBS.items():
            if path == f".github/envs/{image}.yml":
                affected.update(jobs)
        affected.update(
            job for job, patterns in JOB_PATHS.items() if any(fnmatchcase(path, pattern) for pattern in patterns)
        )
    return {job: job in affected for job in JOB_COMPONENTS}


def changed_paths(base: str, head: str) -> list[str]:
    """Read NUL-delimited changes, including deleted/renamed paths; fail closed."""
    head_sha = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}"], text=True
    ).strip()
    if not base or set(base) == {"0"}:
        command = ["git", "ls-tree", "-r", "--name-only", "-z", head_sha]
    else:
        base_sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}"], text=True
        ).strip()
        command = ["git", "diff", "--name-only", "--no-renames", "-z", base_sha, head_sha, "--"]
    return [path for path in os.fsdecode(subprocess.check_output(command)).split("\0") if path]


def main(argv: list[str] | None = None) -> None:
    """Emit JSON and, when requested by Actions, append all eight boolean outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*")
    parser.add_argument("--base")
    parser.add_argument("--head")
    args = parser.parse_args(argv)
    if (args.base is None) != (args.head is None):
        parser.error("--base and --head must be supplied together")
    paths = args.paths
    if args.head is not None:
        paths += changed_paths(args.base, args.head)
    result = classify(paths)
    if output := os.environ.get("GITHUB_OUTPUT"):
        with Path(output).open("a", encoding="utf-8") as handle:
            handle.writelines(f"{job}={str(affected).lower()}\n" for job, affected in result.items())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
