#!/usr/bin/env python3
"""Check runtime and relevant extra dependencies in the six CI environments.

Image jobs install Q2MM with ``--no-deps`` and require ``pip check``. Each
environment must therefore supply all core requirements plus the extras
actually exercised by its native/pairwise jobs. Standalone JAX uses normal
PyPI installation, not a native image. There is no kitchen-sink environment.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
ENVS_DIR = REPO_ROOT / ".github" / "envs"

# Map a pyproject extra name to the env files it must be a subset of.
# Extras not listed here (e.g. ``amber``, ``qcel``, ``dev``,
# ``docs``) are not enforced because they have no corresponding CI env file.
EXTRA_TO_ENV_FILES: dict[str, tuple[str, ...]] = {
    "openmm": ("openmm.yml", "openmm-jax.yml", "openmm-tinker.yml"),
    "jax": ("openmm-jax.yml",),
    "jax-md": ("jax-md.yml",),
    "qcengine": ("psi4.yml",),
    "optimize": ("openmm.yml", "tinker.yml", "jax-md.yml", "psi4.yml", "openmm-jax.yml", "openmm-tinker.yml"),
}


def _normalize(name: str) -> str:
    """Normalize a package name (lowercase, strip extras, replace _ with -)."""
    name = name.split("[", 1)[0].strip().lower()
    return name.replace("_", "-")


def _strip_marker_and_specifier(spec: str) -> str:
    """Extract the bare package name from a PEP 508 dep string."""
    spec = spec.split(";", 1)[0].strip()
    spec = re.split(r"[<>=!~ ]", spec, maxsplit=1)[0]
    return _normalize(spec)


def load_extras() -> dict[str, set[str]]:
    """Return ``{extra_name: {normalized_pkg, ...}}``."""
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    extras = data.get("project", {}).get("optional-dependencies", {})
    return {name: {_strip_marker_and_specifier(d) for d in deps} for name, deps in extras.items()}


def load_core_deps() -> set[str]:
    """Return the normalized package names in ``[project] dependencies``."""
    with PYPROJECT.open("rb") as f:
        data = tomllib.load(f)
    deps = data.get("project", {}).get("dependencies", [])
    return {_strip_marker_and_specifier(d) for d in deps}


def load_env_packages(env_file: Path) -> set[str]:
    """Return the set of normalized package names in a conda env file."""
    with env_file.open() as f:
        data = yaml.safe_load(f)
    pkgs: set[str] = set()
    for entry in data.get("dependencies", []):
        if isinstance(entry, str):
            # ``python=3.14`` → python
            pkgs.add(_strip_marker_and_specifier(entry))
        elif isinstance(entry, dict) and "pip" in entry:
            for pip_entry in entry["pip"]:
                pkgs.add(_strip_marker_and_specifier(pip_entry))
    return pkgs


def check_parity() -> list[str]:
    """Return a list of human-readable error messages (empty = OK)."""
    extras = load_extras()
    core_deps = load_core_deps()
    errors: list[str] = []

    # Contract 1: each backend extra must be a subset of its env file(s).
    for extra_name, env_filenames in EXTRA_TO_ENV_FILES.items():
        if extra_name not in extras:
            errors.append(f"pyproject.toml extra [{extra_name}] is missing — expected by check_env_dep_parity.py")
            continue

        required = extras[extra_name]
        for env_filename in env_filenames:
            env_path = ENVS_DIR / env_filename
            if not env_path.exists():
                errors.append(f"env file not found: {env_path}")
                continue

            present = load_env_packages(env_path)
            missing = required - present
            if missing:
                errors.append(
                    f"{env_filename} is missing packages from "
                    f"pyproject.toml [{extra_name}] extra: "
                    f"{sorted(missing)}\n"
                    f"  → add them to {env_path.relative_to(REPO_ROOT)} "
                    "so the CI container image picks them up on next build."
                )

    # Contract 2: every core dep must appear in every backend env file.
    # `pip install -e . --no-deps` skips them, so the env must provide them.
    required_core = core_deps
    if required_core:
        for env_path in sorted(ENVS_DIR.glob("*.yml")):
            present = load_env_packages(env_path)
            missing = required_core - present
            if missing:
                errors.append(
                    f"{env_path.name} is missing core deps from "
                    f"pyproject.toml [project] dependencies: "
                    f"{sorted(missing)}\n"
                    f"  → add them to {env_path.relative_to(REPO_ROOT)}; "
                    "backend jobs use `pip install -e . --no-deps` and the "
                    "env must provide every core dep."
                )

    return errors


def main() -> int:
    """Entry point: print parity status and return shell exit code."""
    errors = check_parity()
    if errors:
        print("env-dep parity check FAILED:\n", file=sys.stderr)
        for err in errors:
            print(f"  ✗ {err}\n", file=sys.stderr)
        print(
            "Why this matters: backend CI jobs use `pip install -e . --no-deps` "
            "and rely on the container image to provide deps. A package in a "
            "pyproject.toml extra but not in the matching env file is silently "
            "absent in CI. See scripts/check_env_dep_parity.py for details.",
            file=sys.stderr,
        )
        return 1
    print("env-dep parity OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
