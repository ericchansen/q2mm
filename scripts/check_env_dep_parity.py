#!/usr/bin/env python3
"""Assert backend extras in pyproject.toml are present in matching CI env files.

Background
----------
Backend test jobs install q2mm with ``pip install -e . --no-deps`` and rely on
the conda env baked into the CI container image to provide native deps. If a
new dep is added to a ``[project.optional-dependencies]`` extra in
``pyproject.toml`` but the matching ``.github/envs/<backend>.yml`` is not
updated, the dep silently goes missing from CI — exactly the bug that bit
PR #250 (jaxopt was in the ``[jax]`` extra but not in ``full.yml``).

This script enforces the contract: every package declared in a backend extra
must appear in both the backend-specific env file and ``full.yml``.

Run manually:
    python scripts/check_env_dep_parity.py

Wired into CI lint job. Exit code 0 = parity OK, non-zero = drift detected.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # Python 3.10 — use the tomli backport (stdlib tomllib lands in 3.11)
    import tomli as tomllib

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
ENVS_DIR = REPO_ROOT / ".github" / "envs"

# Map a pyproject extra name to the env files it must be a subset of.
# Extras not listed here (e.g. ``amber``, ``qcel``, ``optimize``, ``dev``,
# ``docs``) are not enforced because they have no corresponding CI env file.
EXTRA_TO_ENV_FILES: dict[str, tuple[str, ...]] = {
    "openmm": ("openmm.yml", "full.yml"),
    "jax": ("jax.yml", "full.yml"),
    "jax-md": ("jax-md.yml", "full.yml"),
}

# Packages that are always present in every env file as baseline (python,
# numpy, scipy, pytest, ruff). Listing them in extras shouldn't trigger drift.
BASELINE_PACKAGES: frozenset[str] = frozenset({"python", "numpy", "scipy", "pytest", "ruff", "pip"})


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


def load_env_packages(env_file: Path) -> set[str]:
    """Return the set of normalized package names in a conda env file."""
    with env_file.open() as f:
        data = yaml.safe_load(f)
    pkgs: set[str] = set()
    for entry in data.get("dependencies", []):
        if isinstance(entry, str):
            # ``python=3.12`` → python
            pkgs.add(_normalize(entry.split("=", 1)[0].split("<", 1)[0]))
        elif isinstance(entry, dict) and "pip" in entry:
            for pip_entry in entry["pip"]:
                pkgs.add(_strip_marker_and_specifier(pip_entry))
    return pkgs


def check_parity() -> list[str]:
    """Return a list of human-readable error messages (empty = OK)."""
    extras = load_extras()
    errors: list[str] = []

    for extra_name, env_filenames in EXTRA_TO_ENV_FILES.items():
        if extra_name not in extras:
            errors.append(f"pyproject.toml extra [{extra_name}] is missing — expected by check_env_dep_parity.py")
            continue

        required = extras[extra_name] - BASELINE_PACKAGES
        for env_filename in env_filenames:
            env_path = ENVS_DIR / env_filename
            if not env_path.exists():
                errors.append(f"env file not found: {env_path}")
                continue

            present = load_env_packages(env_path)
            missing = required - present - BASELINE_PACKAGES
            if missing:
                errors.append(
                    f"{env_filename} is missing packages from "
                    f"pyproject.toml [{extra_name}] extra: "
                    f"{sorted(missing)}\n"
                    f"  → add them to {env_path.relative_to(REPO_ROOT)} "
                    "so the CI container image picks them up on next build."
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
