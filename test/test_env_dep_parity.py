"""Unit tests for ``scripts/check_env_dep_parity``.

Verifies the parity check correctly detects drift between pyproject.toml
backend extras and the matching ``.github/envs/*.yml`` files. Uses the
real repo state (no temp files) for the green-path test, and monkeypatches
in-memory data structures for the red-path test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_env_dep_parity.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_env_dep_parity", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parity_holds_on_current_repo() -> None:
    """The current repo state must satisfy the env/extras parity contract."""
    mod = _load_module()
    errors = mod.check_parity()
    assert errors == [], "env-dep parity check failed:\n" + "\n".join(errors)


def test_strip_marker_and_specifier_handles_pep508() -> None:
    mod = _load_module()
    cases = {
        "jaxopt": "jaxopt",
        "jaxopt; sys_platform != 'win32'": "jaxopt",
        "jax[cuda12]; sys_platform != 'win32'": "jax",
        "openmm>=8.0": "openmm",
        "jax_md": "jax-md",
        "PyYAML": "pyyaml",
        "python=3.14": "python",
        "pydantic>=2,<3": "pydantic",
        "pydantic-settings>=2,<3": "pydantic-settings",
    }
    for raw, expected in cases.items():
        assert mod._strip_marker_and_specifier(raw) == expected, raw


def test_conda_env_version_ranges_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    monkeypatch.setattr(
        mod.yaml,
        "safe_load",
        lambda stream: {"dependencies": ["python=3.14", "pydantic>=2,<3", "pydantic-settings>=2,<3"]},
    )
    assert mod.load_env_packages(REPO_ROOT / ".github" / "envs" / "openmm.yml") == {
        "python",
        "pydantic",
        "pydantic-settings",
    }


def test_load_extras_contains_known_extras() -> None:
    mod = _load_module()
    extras = mod.load_extras()
    assert {"openmm", "jax", "jax-md"} <= extras.keys()
    assert "jaxopt" in extras["jax"]
    assert "openmm" in extras["openmm"]


def test_drift_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each pair image needs the extras used by its own bounded selectors."""
    mod = _load_module()
    real_loader = mod.load_env_packages

    def lying_loader(env_file: Path) -> set[str]:
        pkgs = real_loader(env_file)
        if env_file.name == "openmm-jax.yml":
            pkgs.discard("jaxopt")
        return pkgs

    monkeypatch.setattr(mod, "load_env_packages", lying_loader)
    errors = mod.check_parity()
    assert errors, "expected parity check to flag missing jaxopt in openmm-jax.yml"
    assert any("jaxopt" in e and "openmm-jax.yml" in e for e in errors), errors


@pytest.mark.parametrize("package", ["numpy", "pyyaml", "pydantic", "pydantic-settings"])
def test_every_core_dependency_is_required(package: str, monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    real_loader = mod.load_env_packages

    def missing_core(env_file: Path) -> set[str]:
        pkgs = real_loader(env_file)
        if env_file.name == "tinker.yml":
            pkgs.discard(package)
        return pkgs

    monkeypatch.setattr(mod, "load_env_packages", missing_core)
    assert any(package in error and "tinker.yml" in error for error in mod.check_parity())


def test_reference_pair_requires_qcengine(monkeypatch: pytest.MonkeyPatch) -> None:
    mod = _load_module()
    real_loader = mod.load_env_packages
    monkeypatch.setattr(mod, "load_env_packages", lambda env: real_loader(env) - {"qcengine"})
    assert any("psi4.yml" in error and "qcengine" in error for error in mod.check_parity())
