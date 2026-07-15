"""Tests for backend availability decisions made by the test harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from q2mm.backends.contracts import BackendConfigurationError, BackendUnavailableError
from q2mm.backends.mm import tinker as tinker_module
from q2mm.backends.registry import load_backend
from test import backend_fixtures


def test_tinker_executable_without_default_parameters_is_typed_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An executable-only Tinker install is not enough to construct a backend."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "analyze").write_text("", encoding="ascii")
    monkeypatch.setattr(tinker_module, "_find_tinker_dir", lambda: str(bin_dir))

    with pytest.raises(BackendUnavailableError, match="MM3 parameter file not found"):
        tinker_module.TinkerBackend()


def test_optional_tinker_load_converts_only_typed_unavailability_to_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection-safe probing converts dependency absence into a skip signal."""
    monkeypatch.setattr(backend_fixtures, "available_backends", lambda: ["tinker"])

    def unavailable(name: str, **kwargs: object) -> object:
        raise BackendUnavailableError("mm3.prm is absent")

    monkeypatch.setattr(backend_fixtures, "load_test_backend", unavailable)

    assert backend_fixtures.optional_test_backend("tinker") is None


def test_optional_tinker_load_does_not_hide_configuration_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid explicit test configuration remains a visible test failure."""
    monkeypatch.setattr(backend_fixtures, "available_backends", lambda: ["tinker"])

    def misconfigured(name: str, **kwargs: object) -> object:
        raise BackendConfigurationError("bad configured path")

    monkeypatch.setattr(backend_fixtures, "load_test_backend", misconfigured)

    with pytest.raises(BackendConfigurationError, match="bad configured path"):
        backend_fixtures.optional_test_backend("tinker")


def test_tinker_test_load_honors_explicit_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Configured Tinker tests pass executable and parameter paths explicitly."""
    bin_dir = tmp_path / "bin"
    params_file = tmp_path / "mm3.prm"
    monkeypatch.setenv("TINKER_DIR", str(bin_dir))
    monkeypatch.setenv("TINKER_PRM", str(params_file))
    sentinel = object()
    captured: dict[str, object] = {}

    def load(name: str, **kwargs: object) -> object:
        captured["name"] = name
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(backend_fixtures, "load_backend", load)

    assert backend_fixtures.load_test_backend("tinker") is sentinel
    assert captured == {
        "name": "tinker",
        "tinker_dir": str(bin_dir),
        "params_file": str(params_file),
    }


def test_optional_tinker_load_bypasses_catalog_probe_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The test loader can use explicit paths even when PATH probing fails."""
    sentinel = object()
    monkeypatch.setattr(backend_fixtures, "available_backends", lambda: [])
    monkeypatch.setattr(backend_fixtures, "load_test_backend", lambda _name: sentinel)

    assert backend_fixtures.optional_test_backend("tinker") is sentinel


def test_explicit_tinker_configuration_bypasses_cheap_probe(tmp_path: Path) -> None:
    """A valid explicit load does not depend on catalog probe health."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    params_file = tmp_path / "mm3.prm"
    params_file.write_text("", encoding="ascii")

    backend = load_backend("tinker", tinker_dir=str(bin_dir), params_file=str(params_file))

    assert backend.info.provenance.backend == "tinker"
