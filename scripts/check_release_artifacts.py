#!/usr/bin/env python3
"""Validate Q2MM release artifacts and their installed behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import venv
import zipfile
from collections import Counter
from collections.abc import Callable


APPROVED_RESOURCE_FILES = frozenset(
    {
        "q2mm/data/sn2/ch3f-energy.txt",
        "q2mm/data/sn2/ch3f-frequencies.txt",
        "q2mm/data/sn2/ch3f-hessian.npy",
        "q2mm/data/sn2/ch3f-normal-modes.npz",
        "q2mm/data/sn2/ch3f-optimized.xyz",
        "q2mm/data/sn2/complex-optimized.xyz",
        "q2mm/data/sn2/manifest.json",
        "q2mm/data/sn2/sn2-ts-energy.txt",
        "q2mm/data/sn2/sn2-ts-frequencies.txt",
        "q2mm/data/sn2/sn2-ts-hessian.npy",
        "q2mm/data/sn2/sn2-ts-normal-modes.npz",
        "q2mm/data/sn2/sn2-ts-optimized.xyz",
        "q2mm/data/sn2/summary.txt",
    }
)

SDIST_ROOT_FILES = frozenset(
    {
        "LICENSE",
        "MANIFEST.in",
        "PKG-INFO",
        "README.md",
        "pyproject.toml",
        "setup.cfg",
    }
)

EGG_INFO_FILES = frozenset(
    {
        "PKG-INFO",
        "SOURCES.txt",
        "dependency_links.txt",
        "entry_points.txt",
        "requires.txt",
        "scm_file_list.json",
        "scm_version.json",
        "top_level.txt",
    }
)

DIST_INFO_FILES = frozenset(
    {
        "METADATA",
        "RECORD",
        "WHEEL",
        "entry_points.txt",
        "licenses/LICENSE",
        "scm_file_list.json",
        "scm_version.json",
        "top_level.txt",
    }
)


class ArtifactContractError(RuntimeError):
    """Raised when a built distribution violates the release contract."""


def _is_package_file(path: str) -> bool:
    return path == "q2mm/py.typed" or (path.startswith("q2mm/") and path.endswith(".py"))


def _wheel_member_allowed(path: str) -> bool:
    if _is_package_file(path) or path in APPROVED_RESOURCE_FILES:
        return True
    parts = PurePosixPath(path).parts
    if len(parts) >= 2 and parts[0].endswith(".dist-info"):
        return "/".join(parts[1:]) in DIST_INFO_FILES
    return False


def _sdist_member_allowed(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if not parts:
        return False
    relative = "/".join(parts[1:])
    if not relative:
        return False
    if relative in SDIST_ROOT_FILES or _is_package_file(relative) or relative in APPROVED_RESOURCE_FILES:
        return True
    relative_parts = PurePosixPath(relative).parts
    if len(relative_parts) >= 2 and relative_parts[0] == "q2mm.egg-info":
        return "/".join(relative_parts[1:]) in EGG_INFO_FILES
    return False


def _raise_for_members(kind: str, members: list[str], predicate: Callable[[str], bool]) -> None:
    unexpected = sorted(path for path in members if not predicate(path))
    if unexpected:
        formatted = "\n".join(f"  - {path}" for path in unexpected)
        raise ArtifactContractError(f"{kind} contains files outside the release contract:\n{formatted}")


def _validate_resource_manifest(content: bytes, *, artifact: str) -> None:
    """Require the integrity manifest to cover the exact approved resource set."""
    try:
        metadata = json.loads(content)
        entries = metadata["files"]
        names = [entry["name"] for entry in entries]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ArtifactContractError(f"{artifact} contains an invalid SN2 resource manifest") from exc
    if not isinstance(entries, list) or not all(isinstance(name, str) for name in names):
        raise ArtifactContractError(f"{artifact} contains an invalid SN2 resource manifest")
    if len(names) != len(set(names)):
        raise ArtifactContractError(f"{artifact} SN2 resource manifest contains duplicate file names")

    expected = {PurePosixPath(path).name for path in APPROVED_RESOURCE_FILES if path != "q2mm/data/sn2/manifest.json"}
    declared = set(names)
    if declared != expected:
        raise ArtifactContractError(
            f"{artifact} SN2 resource manifest coverage differs: "
            f"missing={sorted(expected - declared)}, extra={sorted(declared - expected)}"
        )


def inspect_wheel(path: Path) -> tuple[int, int, frozenset[str]]:
    """Validate a wheel and return member count, bytes, and canonical names."""
    with zipfile.ZipFile(path) as archive:
        files = [info for info in archive.infolist() if not info.is_dir()]
    names = [info.filename for info in files]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ArtifactContractError(f"wheel contains duplicate members: {duplicates}")
    _raise_for_members("wheel", names, _wheel_member_allowed)
    missing = APPROVED_RESOURCE_FILES.difference(names)
    if missing:
        raise ArtifactContractError(f"wheel is missing approved resources: {sorted(missing)}")
    if "q2mm/py.typed" not in names:
        raise ArtifactContractError("wheel is missing q2mm/py.typed")
    metadata_names = {
        "/".join(PurePosixPath(name).parts[1:]) for name in names if PurePosixPath(name).parts[0].endswith(".dist-info")
    }
    missing_metadata = DIST_INFO_FILES.difference(metadata_names)
    if missing_metadata:
        raise ArtifactContractError(f"wheel is missing distribution metadata: {sorted(missing_metadata)}")
    with zipfile.ZipFile(path) as archive:
        manifest_content = archive.read("q2mm/data/sn2/manifest.json")
    _validate_resource_manifest(manifest_content, artifact="wheel")
    return len(files), sum(info.file_size for info in files), _canonical_wheel_names(names)


def inspect_sdist(path: Path) -> tuple[int, int]:
    """Validate an sdist and return member count and uncompressed bytes."""
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
    normalized_names: list[str] = []
    for member in members:
        if "\\" in member.name:
            raise ArtifactContractError(f"sdist contains a non-portable member path: {member.name}")
        normalized = posixpath.normpath(member.name)
        if normalized == ".." or normalized.startswith("../") or normalized.startswith("/"):
            raise ArtifactContractError(f"sdist contains an unsafe member path: {member.name}")
        normalized_names.append(normalized)
    duplicate_names = sorted(name for name, count in Counter(normalized_names).items() if count > 1)
    if duplicate_names:
        raise ArtifactContractError(f"sdist contains duplicate extraction targets: {duplicate_names}")
    unsafe_types = sorted(member.name for member in members if not (member.isdir() or member.isfile()))
    if unsafe_types:
        raise ArtifactContractError(f"sdist contains links or special files: {unsafe_types}")
    files = [member for member in members if member.isfile()]
    names = [member.name for member in files]
    _raise_for_members("sdist", names, _sdist_member_allowed)
    relative_names = {"/".join(PurePosixPath(name).parts[1:]) for name in names}
    missing = APPROVED_RESOURCE_FILES.difference(relative_names)
    if missing:
        raise ArtifactContractError(f"sdist is missing approved resources: {sorted(missing)}")
    manifest_members = [member for member in members if member.name.endswith("/q2mm/data/sn2/manifest.json")]
    if len(manifest_members) != 1:
        raise ArtifactContractError(f"sdist must contain one SN2 resource manifest, found {len(manifest_members)}")
    with tarfile.open(path, "r:gz") as archive:
        manifest_file = archive.extractfile(manifest_members[0])
        if manifest_file is None:
            raise ArtifactContractError("sdist SN2 resource manifest is not a regular file")
        manifest_content = manifest_file.read()
    _validate_resource_manifest(manifest_content, artifact="sdist")
    return len(files), sum(member.size for member in files)


def _canonical_wheel_names(names: list[str]) -> frozenset[str]:
    canonical: set[str] = set()
    for name in names:
        parts = PurePosixPath(name).parts
        if parts and parts[0].endswith(".dist-info"):
            canonical.add("{dist-info}/" + "/".join(parts[1:]))
        else:
            canonical.add(name)
    return frozenset(canonical)


def _wheel_payload_hashes(path: Path) -> dict[str, str]:
    """Return SHA-256 hashes for wheel members outside ``.dist-info``."""
    payload: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            parts = PurePosixPath(info.filename).parts
            if info.is_dir() or (parts and parts[0].endswith(".dist-info")):
                continue
            if info.filename in payload:
                raise ArtifactContractError(f"wheel contains duplicate payload member: {info.filename}")
            payload[info.filename] = hashlib.sha256(archive.read(info)).hexdigest()
    return payload


def compare_wheel_payload(direct: Path, rebuilt: Path) -> None:
    """Assert that direct and sdist-built wheels contain identical payload bytes."""
    direct_hashes = _wheel_payload_hashes(direct)
    rebuilt_hashes = _wheel_payload_hashes(rebuilt)
    missing = sorted(direct_hashes.keys() - rebuilt_hashes.keys())
    extra = sorted(rebuilt_hashes.keys() - direct_hashes.keys())
    changed = sorted(
        name for name in direct_hashes.keys() & rebuilt_hashes.keys() if direct_hashes[name] != rebuilt_hashes[name]
    )
    if missing or extra or changed:
        raise ArtifactContractError(
            f"wheel rebuilt from sdist differs: missing={missing}, extra={extra}, changed={changed}"
        )


def _extract_sdist(path: Path, destination: Path) -> Path:
    with tarfile.open(path, "r:gz") as archive:
        destination_resolved = destination.resolve()
        members = archive.getmembers()
        unsafe_types = sorted(member.name for member in members if not (member.isdir() or member.isfile()))
        if unsafe_types:
            raise ArtifactContractError(f"sdist contains links or special files: {unsafe_types}")
        for member in members:
            target = (destination / member.name).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise ArtifactContractError(f"unsafe path in sdist: {member.name}")
        archive.extractall(destination, members=members)
    roots = [entry for entry in destination.iterdir() if entry.is_dir()]
    if len(roots) != 1:
        raise ArtifactContractError(f"sdist must contain one source root, found: {roots}")
    return roots[0]


def build_wheel_from_sdist(sdist: Path, destination: Path) -> Path:
    """Build a wheel from an sdist in an isolated PEP 517 environment."""
    source_parent = destination / "source"
    source_parent.mkdir()
    source_root = _extract_sdist(sdist, source_parent)
    wheel_dir = destination / "wheel"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(wheel_dir), str(source_root)],
        check=True,
    )
    wheels = list(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise ArtifactContractError(f"expected one wheel rebuilt from sdist, found: {wheels}")
    return wheels[0]


def smoke_test_wheel(wheel: Path, destination: Path) -> str:
    """Install a wheel in a clean venv and exercise import, CLI, and resources."""
    venv_dir = destination / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    if os.name == "nt":
        python = venv_dir / "Scripts" / "python.exe"
        cli = venv_dir / "Scripts" / "q2mm-benchmark.exe"
    else:
        python = venv_dir / "bin" / "python"
        cli = venv_dir / "bin" / "q2mm-benchmark"

    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", str(wheel)],
        check=True,
        cwd=destination,
        env=environment,
    )
    subprocess.run(
        [str(python), "-I", "-c", "import q2mm"],
        check=True,
        cwd=destination,
        env=environment,
    )
    subprocess.run(
        [str(cli), "--help"],
        check=True,
        cwd=destination,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    smoke_code = """
import numpy as np
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    FrequencyResult,
    FrequencyUnit,
)
from q2mm.models.parameters import ParameterLayout
from q2mm.resources import validate_sn2_resources
from q2mm.benchmarks.systems import load_system

_PROV = BackendProvenance(backend="smoke", role=BackendRole.MM)
_INFO = BackendInfo(
    name="smoke",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.FREQUENCIES}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_PROV,
)


class _SmokePrepared(AbstractPreparedBackend):
    def _frequencies(self, request):
        n = 3 * len(self.molecule.symbols)
        return FrequencyResult(
            frequencies=np.full(n, 100.0), unit=FrequencyUnit.INVERSE_CM, provenance=_PROV
        )


class SmokeBackend:
    @property
    def info(self):
        return _INFO

    def prepare(self, request):
        layout = ParameterLayout.from_force_field(request.force_field)
        return _SmokePrepared(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=layout,
        )


validate_sn2_resources()
case = load_system("ch3f", backend=SmokeBackend(), functional_form="harmonic")
assert case.problem.molecules[0].hessian.shape == (15, 15)
print("installed-import=ok cli-help=ok sn2-resource=ok ch3f-system=ok")
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", smoke_code],
        check=True,
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def main() -> int:
    """Validate release artifacts, rebuild from sdist, and smoke-test install."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()

    wheels = list(args.dist_dir.glob("*.whl"))
    sdists = list(args.dist_dir.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ArtifactContractError(
            f"expected one wheel and one sdist in {args.dist_dir}; found wheels={wheels}, sdists={sdists}"
        )

    wheel_count, wheel_size, _ = inspect_wheel(wheels[0])
    sdist_count, sdist_size = inspect_sdist(sdists[0])
    print(f"wheel: {wheel_count} files, {wheel_size} uncompressed bytes")
    print(f"sdist: {sdist_count} files, {sdist_size} uncompressed bytes")

    with TemporaryDirectory(prefix="q2mm-release-") as temp_dir:
        temp = Path(temp_dir)
        rebuilt_wheel = build_wheel_from_sdist(sdists[0], temp)
        rebuilt_count, rebuilt_size, _ = inspect_wheel(rebuilt_wheel)
        compare_wheel_payload(wheels[0], rebuilt_wheel)
        print(f"sdist-wheel: {rebuilt_count} files, {rebuilt_size} uncompressed bytes, payload matches")
        print(smoke_test_wheel(rebuilt_wheel, temp / "smoke"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
