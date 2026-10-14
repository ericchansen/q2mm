"""Runtime and dependency contracts independent of optional native backends."""

from pathlib import Path
import json
import tomllib
import zipfile

from packaging.requirements import Requirement
import pytest
import yaml

from scripts.check_release_artifacts import (
    APPROVED_RESOURCE_FILES,
    DIST_INFO_FILES,
    ArtifactContractError,
    _sdist_member_allowed,
    inspect_wheel,
    validate_runtime_metadata,
)

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_METADATA = (
    "Requires-Python: >=3.14\n"
    "Classifier: Programming Language :: Python :: 3.15\n"
    "Requires-Dist: pydantic<3,>=2.14.1\n"
    "Requires-Dist: pydantic-settings<3,>=2.16.0\n"
    'Requires-Dist: openmm; extra == "openmm"\n'
)


def _load_project(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def test_primary_runtime_and_real_extras() -> None:
    project = _load_project(ROOT / "pyproject.toml")
    build = {requirement.name: requirement for requirement in map(Requirement, project["build-system"]["requires"])}
    assert str(build["setuptools-scm"].specifier) == ">=10.3.4"
    metadata = project["project"]
    assert metadata["requires-python"] == ">=3.14"
    assert {"Programming Language :: Python :: 3.14", "Programming Language :: Python :: 3.15"} <= set(
        metadata["classifiers"]
    )
    dependencies = {requirement.name: requirement for requirement in map(Requirement, metadata["dependencies"])}
    for name, floor in (("pydantic", "2.14.1"), ("pydantic-settings", "2.16.0")):
        assert str(dependencies[name].specifier) == f"<3,>={floor}"
        assert dependencies[name].marker is None
    assert "openmm" not in dependencies
    dev = {Requirement(spec).name for spec in metadata["optional-dependencies"]["dev"]}
    assert not dev.intersection({"openmm", "parmed", "qcelemental", "qcengine", "ase", "tomli"})
    openmm = Requirement(metadata["optional-dependencies"]["openmm"][0])
    assert openmm.name == "openmm" and openmm.marker is None
    assert project["tool"]["ruff"]["target-version"] == "py315"
    assert "UP042" not in project["tool"]["ruff"]["lint"]["ignore"]
    assert project["tool"]["ruff"]["lint"]["per-file-ignores"]["q2mm/backends/contracts.py"] == ["UP042"]
    assert project["tool"]["mypy"]["python_version"] == "3.15"
    plugin = _load_project(ROOT / "examples" / "backend-plugin" / "pyproject.toml")
    assert plugin["project"]["requires-python"] == metadata["requires-python"]
    assert "harmonic-reference" in plugin["project"]["entry-points"]["q2mm.backends"]


def test_ci_best_effort_is_provisioning_only() -> None:
    ci = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))
    for name in ("lint", "test-core", "test-jax"):
        job = ci["jobs"][name]
        assert not job.get("continue-on-error")
        setup = next(step for step in job["steps"] if step.get("uses", "").startswith("actions/setup-python@"))
        assert setup["with"]["python-version"] == "3.15"
        assert any(".[dev" in step.get("run", "") for step in job["steps"])
    assert ci["jobs"]["test-jax"]["env"]["JAX_PLATFORMS"] == "cpu"
    native = ci["jobs"]["test-native"]
    assert "BEST-EFFORT Python 3.14" in native["name"]
    assert not native.get("continue-on-error")
    softened = {step["name"] for step in native["steps"] if step.get("continue-on-error")}
    assert softened == {"Provision optional native environment", "Provision optional interoperability extras"}
    correctness = next(step for step in native["steps"] if step.get("name") == "Required native correctness tests")
    assert correctness["if"] == "steps.native-env.outcome == 'success'"
    assert "--run-validation" not in correctness["run"] and "--run-nightly" not in correctness["run"]
    for path in (ROOT / ".github" / "envs").glob("*.yml"):
        environment = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "python=3.14" in environment["dependencies"]
    for filename, job_name in (("docs.yml", "deploy"), ("publish.yml", "build")):
        workflow = yaml.safe_load((ROOT / ".github" / "workflows" / filename).read_text(encoding="utf-8"))
        assert workflow["env"]["PYYAML_FORCE_LIBYAML"] == "0"
        setup = next(
            step
            for step in workflow["jobs"][job_name]["steps"]
            if step.get("uses", "").startswith("actions/setup-python@")
        )
        assert setup["with"]["python-version"] == "3.15"


def test_release_runtime_metadata_requires_real_base_dependencies() -> None:
    metadata = RUNTIME_METADATA
    validate_runtime_metadata(metadata, artifact="test")
    for changed in (
        metadata.replace(">=3.14", ">=3.10"),
        metadata.replace("3.15", "3.13"),
        metadata.replace("Requires-Dist: pydantic<3,>=2.14.1\n", ""),
        metadata.replace("Requires-Dist: pydantic-settings<3,>=2.16.0\n", ""),
        metadata.replace("pydantic<3,>=2.14.1", "pydantic<3,>=2"),
        metadata.replace("pydantic-settings<3,>=2.16.0", "pydantic-settings<3,>=2"),
        metadata.replace('extra == "openmm"', 'extra == "dev"'),
        metadata + "Requires-Dist: tomli\n",
    ):
        with pytest.raises(ArtifactContractError):
            validate_runtime_metadata(changed, artifact="test")


def test_scm_rebuild_records_are_sdist_only(tmp_path: Path) -> None:
    wheel = tmp_path / "q2mm-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("q2mm/__init__.py", "")
        archive.writestr("q2mm/py.typed", "")
        for path in APPROVED_RESOURCE_FILES:
            if path.endswith("manifest.json"):
                manifest = {"files": [{"name": Path(name).name} for name in APPROVED_RESOURCE_FILES if name != path]}
                archive.writestr(path, json.dumps(manifest))
            else:
                archive.writestr(path, "")
        for name in DIST_INFO_FILES:
            archive.writestr(f"q2mm-test.dist-info/{name}", RUNTIME_METADATA if name == "METADATA" else "")
    inspect_wheel(wheel)
    for name in ("scm_file_list.json", "scm_version.json"):
        assert _sdist_member_allowed(f"q2mm-test/q2mm.egg-info/{name}")
    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("q2mm-test.dist-info/scm_version.json", "{}")
    with pytest.raises(ArtifactContractError, match="outside the release contract"):
        inspect_wheel(wheel)
