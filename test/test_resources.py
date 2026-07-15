"""Tests for packaged scientific resources."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from q2mm.resources import sn2_reference_dir, validate_sn2_resources


def _load_generator_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "examples" / "sn2-test" / "generate_qm_data.py"
    spec = importlib.util.spec_from_file_location("q2mm_sn2_generator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load generator module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_tree_sn2_resource_is_canonical_package_data() -> None:
    resource_dir = sn2_reference_dir()
    assert resource_dir.parts[-3:] == ("q2mm", "data", "sn2")
    assert (resource_dir / "ch3f-optimized.xyz").is_file()
    assert not (resource_dir.parents[2] / "examples" / "sn2-test" / "qm-reference").exists()

    from q2mm.benchmarks.systems.ch3f import load_molecule as load_ch3f_molecule
    from q2mm.benchmarks.systems.ch3f_sn2 import load_molecule as load_ch3f_sn2_molecule

    assert load_ch3f_molecule().hessian.shape == (15, 15)
    assert load_ch3f_sn2_molecule().hessian.shape == (18, 18)


def test_sn2_resource_manifest_covers_exact_payload() -> None:
    resource_dir = sn2_reference_dir()
    metadata = json.loads((resource_dir / "manifest.json").read_text(encoding="utf-8"))
    declared = {entry["name"] for entry in metadata["files"]}
    actual = {path.name for path in resource_dir.iterdir() if path.name != "manifest.json"}
    assert declared == actual
    assert metadata["license"]["spdx"] == "MIT"
    validate_sn2_resources()


def test_runtime_resource_validation_rejects_duplicate_manifest_entries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import q2mm.resources as resources

    payload = tmp_path / "payload.txt"
    payload.write_text("data", encoding="utf-8")
    entry = {
        "name": payload.name,
        "size": payload.stat().st_size,
        "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps({"files": [entry, entry]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(resources, "sn2_reference_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="duplicate file names"):
        resources.validate_sn2_resources()


@pytest.mark.parametrize(
    ("manifest_names", "payload_names"),
    [
        (["declared.txt"], ["declared.txt", "undeclared.txt"]),
        (["declared.txt", "missing.txt"], ["declared.txt"]),
    ],
)
def test_runtime_resource_validation_rejects_inexact_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest_names: list[str],
    payload_names: list[str],
) -> None:
    import q2mm.resources as resources

    for name in payload_names:
        (tmp_path / name).write_text(name, encoding="utf-8")
    entries = []
    for name in manifest_names:
        content = name.encode()
        entries.append(
            {
                "name": name,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    (tmp_path / "manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")
    monkeypatch.setattr(resources, "sn2_reference_dir", lambda: tmp_path)

    with pytest.raises(RuntimeError, match="manifest coverage differs"):
        resources.validate_sn2_resources()


def test_resource_resolution_uses_importlib_package_location(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import q2mm.resources as resources

    package_root = tmp_path / "installed" / "q2mm"
    resource_dir = package_root / "data" / "sn2"
    resource_dir.mkdir(parents=True)
    monkeypatch.setattr(resources, "files", lambda package: package_root)
    assert resources.sn2_reference_dir() == resource_dir


def test_regenerator_targets_its_source_checkout(tmp_path: Path) -> None:
    generator = _load_generator_module()
    script = tmp_path / "checkout" / "examples" / "sn2-test" / "generate_qm_data.py"
    script.parent.mkdir(parents=True)
    resource_dir = tmp_path / "checkout" / "q2mm" / "data" / "sn2"
    resource_dir.mkdir(parents=True)
    (tmp_path / "checkout" / "pyproject.toml").write_text("", encoding="utf-8")
    (resource_dir / "manifest.json").write_text("{}", encoding="utf-8")

    assert generator.checkout_resource_dir(script) == resource_dir


def test_normal_mode_archive_generation_is_deterministic(tmp_path: Path) -> None:
    generator = _load_generator_module()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    hessian = np.diag([0.1, 0.2, 0.3])

    generator.write_normal_modes(first, hessian, ["H"])
    generator.write_normal_modes(second, hessian, ["H"])

    assert first.read_bytes() == second.read_bytes()
    with np.load(first, allow_pickle=False) as modes:
        assert set(modes.files) == {"eigenvalues", "eigenvectors", "masses_amu", "symbols"}
        assert modes["eigenvectors"].shape == (3, 3)


def test_manifest_refresh_normalizes_text_and_updates_hashes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    generator = _load_generator_module()
    monkeypatch.setattr(generator, "TEXT_RESOURCE_NAMES", ("sample.txt",))
    text_path = tmp_path / "sample.txt"
    binary_path = tmp_path / "sample.npy"
    text_path.write_bytes(b"one\r\ntwo\r\n")
    binary_path.write_bytes(b"\x00binary")
    manifest = {
        "files": [
            {"name": text_path.name, "size": 0, "sha256": ""},
            {"name": binary_path.name, "size": 0, "sha256": ""},
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    generator.refresh_manifest(tmp_path)

    assert text_path.read_bytes() == b"one\ntwo\n"
    refreshed = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    for entry in refreshed["files"]:
        content = (tmp_path / entry["name"]).read_bytes()
        assert entry["size"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
