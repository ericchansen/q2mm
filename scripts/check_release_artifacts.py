#!/usr/bin/env python3
"""Validate Q2MM release artifacts and their installed behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import sys
import tarfile
import venv
import zipfile
from collections import Counter
from collections.abc import Callable


#: Canonical external reference plugin. It is independently installable from the
#: source checkout but excluded from both Q2MM release artifacts.
REFERENCE_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "examples" / "backend-plugin"
INSTALLED_PUBLICATION_CHECK = Path(__file__).resolve().parent / "check_installed_publication_sdk.py"


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
    # Force UTF-8 I/O so capturing the CLI's Unicode ``list`` output is portable
    # (Windows consoles/pipes otherwise default to cp1252 and fail to decode).
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
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
    # The `list` subcommand must be side-effect-free (cheap backend catalog
    # probes only, no device/XLA/CUDA init) and work in a fresh install.
    subprocess.run(
        [str(cli), "list"],
        check=True,
        cwd=destination,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    # An installed single run against a registered-but-unavailable backend must
    # terminate as a persisted skipped candidate (exit 0), exercising the full
    # CLI wiring without requiring any optional backend in the smoke venv.
    # (An *unknown* backend is a configuration error and would exit non-zero.)
    single_out = destination / "cli-single"
    subprocess.run(
        [
            str(cli),
            "single",
            "--system",
            "ch3f",
            "--backend",
            "openmm",
            "--optimizer",
            "scipy-lbfgsb",
            "--output",
            str(single_out),
        ],
        check=True,
        cwd=destination,
        env=environment,
        stdout=subprocess.DEVNULL,
    )
    if not list((single_out / "candidates").glob("*.json")):
        raise ArtifactContractError("installed `q2mm-benchmark single` wrote no candidate record")
    smoke_code = """
import numpy as np
from pathlib import Path
import q2mm
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    FrequencyResult,
    FrequencyUnit,
)
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout
from q2mm.models.results import OptimizationResult
from q2mm.preparation import MatchedFrequencyObservations
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


class SmokeOptimizer:
    def optimize(self, evaluator, space):
        parameters = np.array(space.baseline, copy=True)
        score = evaluator.value(parameters)
        return OptimizationResult(
            success=True,
            message="installed optimizer entry",
            initial_score=score,
            final_score=score,
            n_iterations=0,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=parameters,
            final_params=parameters,
            method="installed-smoke",
            gradient_mode="none",
        )


validate_sn2_resources()
case = load_system("ch3f", backend=SmokeBackend(), functional_form="harmonic")
assert case.problem.molecules[0].hessian.shape == (15, 15)
molecule = Molecule(
    symbols=("H", "H"),
    geometry=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
    hessian=np.eye(6) * 0.1,
    name="installed-byo",
)
problem = q2mm.prepare(
    molecule,
    stationary_point="ground_state",
    functional_form="harmonic",
    observations=MatchedFrequencyObservations(
        qm_frequencies=(100.0,),
        backend=SmokeBackend(),
    ),
)
evaluation = q2mm.evaluate(problem, backend=SmokeBackend(), executor="python")
assert evaluation.total >= 0.0
run = q2mm.optimize(
    problem,
    backend=SmokeBackend(),
    recipe="explicit",
    optimizer=SmokeOptimizer(),
    workflow="single-stage",
    executor="python",
)
output = Path("installed-byo.frcmod")
saved = q2mm.save(run, output)
assert output.is_file()
assert saved.manifest_path is not None and saved.manifest_path.is_file()
print("installed-import=ok cli-help=ok cli-list=ok cli-single-skip=ok sn2-resource=ok "
      "ch3f-system=ok generic-prepare=ok generic-evaluate=ok optimizer-entry=ok generic-save=ok")
"""
    completed = subprocess.run(
        [str(python), "-I", "-c", smoke_code],
        check=True,
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    external = smoke_test_external_plugin(python, cli, destination, environment)
    publication = smoke_test_installed_publications(python, destination, environment)
    return f"{completed.stdout.strip()} {external} {publication}"


_EXTERNAL_PLUGIN_PROOF = """
import math
import sys

import q2mm.backends.registry as reg

# The plugin is discovered purely from its installed entry point (no injection).
registered = reg.registered_backends()
assert "harmonic-reference" in registered, registered

# Built-ins remain visible alongside the external plugin.
assert "openmm" in registered and "psi4" in registered, registered

# Cataloging/listing must NOT import the backend implementation module.
reg.catalog()
reg.available_backends()
assert "q2mm_reference_backend.descriptor" in sys.modules
assert "q2mm_reference_backend.backend" not in sys.modules, "catalog imported implementation!"

# Descriptor fields / capabilities / forms are exactly what the manifest declared.
desc = reg.get_descriptor("harmonic-reference")
assert desc.backend_api_version == 1
assert desc.name == "harmonic-reference"
assert desc.role.value == "mm", desc.role
assert {c.value for c in desc.capability_ceiling} == {"energy"}, desc.capability_ceiling
assert set(desc.functional_form_ceiling) == {"harmonic"}, desc.functional_form_ceiling
assert desc.factory == "q2mm_reference_backend.backend:HarmonicReferenceBackend", desc.factory

# Explicit load imports the implementation module.
backend = reg.load_backend("harmonic-reference")
assert "q2mm_reference_backend.backend" in sys.modules, "load did not import implementation!"
assert backend.info.provenance.backend == "harmonic-reference"
assert backend.info.role is desc.role
assert backend.info.capabilities <= desc.capability_ceiling
assert backend.info.functional_forms <= desc.functional_form_ceiling

# Public ENERGY conformance also proves every undeclared public operation is
# gated by UnsupportedCapabilityError.
from q2mm.backends.conformance import MMConformanceCase, run_mm_conformance
from q2mm.benchmarks.systems.ch3f import load_molecule
from q2mm.models.forcefield import FunctionalForm
from q2mm.models.seminario import qfuerza_fresh

molecule = load_molecule()
force_field = qfuerza_fresh(molecule, functional_form=FunctionalForm.HARMONIC, invert_ts_curvature=False)
outcome = run_mm_conformance(
    MMConformanceCase(
        descriptor=desc,
        backend=backend,
        molecule=molecule,
        force_field=force_field,
    )
)
assert [cap.value for cap in outcome.executed] == ["energy"], outcome
assert "hessian" in {cap.value for cap in outcome.unsupported_verified}, outcome

print("external-plugin=ok")
"""


def smoke_test_external_plugin(python: Path, cli: Path, destination: Path, environment: dict[str, str]) -> str:
    """Install the canonical reference plugin and prove entry-point discovery.

    Installs ``examples/backend-plugin`` (``--no-deps``) into the same
    fresh venv the q2mm wheel was installed in, then proves: the plugin is
    discovered from its entry point; catalog/list does not import the
    implementation; descriptor fields/capabilities/forms are correct; explicit
    load imports the implementation; ENERGY conformance works and an undeclared
    capability stays typed-unsupported; the built-ins remain; and
    ``q2mm-benchmark list`` includes the plugin.  Returns ``external-plugin=ok``.
    """
    if not (REFERENCE_PLUGIN_DIR / "pyproject.toml").is_file():
        raise ArtifactContractError(f"backend reference plugin is missing at {REFERENCE_PLUGIN_DIR}")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--disable-pip-version-check", "--no-deps", str(REFERENCE_PLUGIN_DIR)],
        check=True,
        cwd=destination,
        env=environment,
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", _EXTERNAL_PLUGIN_PROOF],
        check=True,
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    marker = completed.stdout.strip()
    if marker != "external-plugin=ok":
        raise ArtifactContractError(f"external plugin proof produced unexpected output: {marker!r}")
    listing = subprocess.run(
        [str(cli), "list"],
        check=True,
        cwd=destination,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if "harmonic-reference" not in listing.stdout:
        raise ArtifactContractError("`q2mm-benchmark list` did not include the discovered plugin")
    return marker


def smoke_test_installed_publications(python: Path, destination: Path, environment: dict[str, str]) -> str:
    """Run provisionable publication rows through the installed wheel when configured."""
    required = {
        "Q2MM_SUPPORTING_INFO": environment.get("Q2MM_SUPPORTING_INFO"),
        "Q2MM_MM3_BASE": environment.get("Q2MM_MM3_BASE"),
        "Q2MM_RH_ENAMIDE": environment.get("Q2MM_RH_ENAMIDE"),
    }
    if not all(required.values()):
        return "installed-publication-sdk=not-configured"
    if not INSTALLED_PUBLICATION_CHECK.is_file():
        raise ArtifactContractError(f"installed publication checker is missing: {INSTALLED_PUBLICATION_CHECK}")
    try:
        completed = subprocess.run(
            [
                str(python),
                "-I",
                str(INSTALLED_PUBLICATION_CHECK),
                "--supporting-info",
                str(required["Q2MM_SUPPORTING_INFO"]),
                "--mm3-base",
                str(required["Q2MM_MM3_BASE"]),
                "--rh-enamide",
                str(required["Q2MM_RH_ENAMIDE"]),
                "--output",
                str(destination / "publication-sdk"),
            ],
            check=True,
            cwd=destination,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        raise ArtifactContractError(
            f"installed publication proof failed:\nstdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    marker = completed.stdout.strip().splitlines()[-1]
    if marker != "installed-publication-sdk=ok":
        raise ArtifactContractError(f"installed publication proof produced unexpected output: {marker!r}")
    return marker


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

    temp = (args.dist_dir.parent / "build" / "release-check").resolve()
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    try:
        rebuilt_wheel = build_wheel_from_sdist(sdists[0], temp)
        rebuilt_count, rebuilt_size, _ = inspect_wheel(rebuilt_wheel)
        compare_wheel_payload(wheels[0], rebuilt_wheel)
        print(f"sdist-wheel: {rebuilt_count} files, {rebuilt_size} uncompressed bytes, payload matches")
        print(smoke_test_wheel(rebuilt_wheel, temp / "smoke"))
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
