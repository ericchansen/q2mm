"""Tests for the release-artifact allowlist."""

from __future__ import annotations

from pathlib import Path
import tarfile
import warnings
import zipfile

import pytest

from scripts.check_release_artifacts import (
    INSTALLED_PUBLICATION_CHECK,
    REFERENCE_PLUGIN_DIR,
    ArtifactContractError,
    _sdist_member_allowed,
    _validate_resource_manifest,
    smoke_test_installed_publications,
    _wheel_member_allowed,
    compare_wheel_payload,
    inspect_sdist,
    inspect_wheel,
)


def test_wheel_manifest_rejects_unapproved_data(tmp_path: Path) -> None:
    wheel = tmp_path / "q2mm-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("q2mm/__init__.py", "")
        archive.writestr("q2mm/py.typed", "")
        archive.writestr("q2mm/data/raw-gaussian.log", "not approved")

    with pytest.raises(ArtifactContractError, match="outside the release contract"):
        inspect_wheel(wheel)


def test_sdist_manifest_rejects_repository_content(tmp_path: Path) -> None:
    payload = tmp_path / "workflow.yml"
    payload.write_text("name: must-not-ship", encoding="utf-8")
    sdist = tmp_path / "q2mm-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="q2mm-test/.github/workflows/publish.yml")

    with pytest.raises(ArtifactContractError, match="outside the release contract"):
        inspect_sdist(sdist)


def test_sdist_manifest_rejects_links(tmp_path: Path) -> None:
    sdist = tmp_path / "q2mm-test.tar.gz"
    link = tarfile.TarInfo("q2mm-test/q2mm/linked.py")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../outside"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.addfile(link)

    with pytest.raises(ArtifactContractError, match="links or special files"):
        inspect_sdist(sdist)


@pytest.mark.parametrize(
    "second_name",
    [
        "q2mm-test/q2mm/__init__.py",
        "q2mm-test/q2mm/./__init__.py",
    ],
)
def test_sdist_manifest_rejects_duplicate_members(tmp_path: Path, second_name: str) -> None:
    payload = tmp_path / "__init__.py"
    payload.write_text("", encoding="utf-8")
    sdist = tmp_path / "q2mm-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname="q2mm-test/q2mm/__init__.py")
        archive.add(payload, arcname=second_name)

    with pytest.raises(ArtifactContractError, match="duplicate extraction targets"):
        inspect_sdist(sdist)


def test_sdist_manifest_rejects_single_component_file(tmp_path: Path) -> None:
    payload = tmp_path / "forbidden-top-level.bin"
    payload.write_bytes(b"not allowed")
    sdist = tmp_path / "q2mm-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(payload, arcname=payload.name)

    with pytest.raises(ArtifactContractError, match="outside the release contract"):
        inspect_sdist(sdist)


def test_wheel_payload_comparison_rejects_changed_bytes(tmp_path: Path) -> None:
    direct = tmp_path / "direct.whl"
    rebuilt = tmp_path / "rebuilt.whl"
    with zipfile.ZipFile(direct, "w") as archive:
        archive.writestr("q2mm/__init__.py", "VERSION = 1\n")
    with zipfile.ZipFile(rebuilt, "w") as archive:
        archive.writestr("q2mm/__init__.py", "VERSION = 2\n")

    with pytest.raises(ArtifactContractError, match=r"changed=\['q2mm/__init__\.py'\]"):
        compare_wheel_payload(direct, rebuilt)


def test_wheel_payload_comparison_rejects_duplicate_members(tmp_path: Path) -> None:
    direct = tmp_path / "direct.whl"
    rebuilt = tmp_path / "rebuilt.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(direct, "w") as archive:
            archive.writestr("q2mm/__init__.py", "bad\n")
            archive.writestr("q2mm/__init__.py", "good\n")
    with zipfile.ZipFile(rebuilt, "w") as archive:
        archive.writestr("q2mm/__init__.py", "good\n")

    with pytest.raises(ArtifactContractError, match="duplicate payload member"):
        compare_wheel_payload(direct, rebuilt)


def test_resource_manifest_requires_exact_approved_coverage() -> None:
    incomplete = b'{"files": [{"name": "ch3f-energy.txt"}]}'
    with pytest.raises(ArtifactContractError, match="manifest coverage differs"):
        _validate_resource_manifest(incomplete, artifact="test artifact")


def test_backend_reference_plugin_present_in_repo() -> None:
    # The out-of-tree reference plugin must exist in the repository (it is what the
    # release checker installs to prove entry-point discovery).
    assert (REFERENCE_PLUGIN_DIR / "pyproject.toml").is_file()
    assert (REFERENCE_PLUGIN_DIR / "q2mm_reference_backend" / "descriptor.py").is_file()
    assert (REFERENCE_PLUGIN_DIR / "q2mm_reference_backend" / "backend.py").is_file()
    assert INSTALLED_PUBLICATION_CHECK.is_file()


def test_installed_publication_proof_reports_missing_configuration(tmp_path: Path) -> None:
    assert smoke_test_installed_publications(Path("python"), tmp_path, {}) == "installed-publication-sdk=not-configured"


def test_wheel_allowlist_excludes_reference_plugin() -> None:
    # The reference plugin package must never be an allowed wheel member, while a
    # genuine q2mm package module is.
    assert _wheel_member_allowed("q2mm/backends/discovery.py") is True
    assert _wheel_member_allowed("q2mm_reference_backend/backend.py") is False
    assert _wheel_member_allowed("q2mm_reference_backend/descriptor.py") is False


def test_sdist_allowlist_excludes_reference_plugin_example() -> None:
    # examples/ (including the independently installable plugin) is pruned.
    assert _sdist_member_allowed("q2mm-5.0.0/q2mm/backends/discovery.py") is True
    assert _sdist_member_allowed("q2mm-5.0.0/examples/backend-plugin/pyproject.toml") is False
    assert _sdist_member_allowed("q2mm-5.0.0/examples/backend-plugin/q2mm_reference_backend/backend.py") is False


def test_wheel_manifest_rejects_reference_plugin_package(tmp_path: Path) -> None:
    wheel = tmp_path / "q2mm-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("q2mm/__init__.py", "")
        archive.writestr("q2mm/py.typed", "")
        archive.writestr("q2mm_reference_backend/backend.py", "should never ship")

    with pytest.raises(ArtifactContractError, match="outside the release contract"):
        inspect_wheel(wheel)
