from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from scripts.check_installed_examples import _write_minimal_fchk
from test._shared import REPO_ROOT

EXAMPLES = REPO_ROOT / "examples"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import example {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", ("ch3f", "ch3f-sn2"))
def test_small_examples_have_callable_root_workflow_and_confined_outputs(key: str, tmp_path: Path) -> None:
    module = _load(EXAMPLES / key / "run.py", f"example_{key.replace('-', '_')}")
    output = tmp_path / key
    result = module.run(output_root=output, bounded_ci=True)

    assert callable(module.run)
    assert callable(module.main)
    assert result["schema"] == "q2mm.example-result"
    assert result["case_count"] == 1
    assert result["bounded_ci"] is True
    assert result["optimization"]["iterations"] == 1
    assert result["optimization"]["convergence_claim"] is False
    assert result["qfuerza"]["settings"]["invert_ts_curvature"] is (key == "ch3f-sn2")
    assert {path.name for path in output.iterdir()} == {
        f"{key}.frcmod",
        f"{key}.frcmod.manifest.json",
    }


def test_ch3f_example_accepts_caller_fchk_through_public_loader(tmp_path: Path) -> None:
    module = _load(EXAMPLES / "ch3f" / "run.py", "example_ch3f_fchk")
    fchk = tmp_path / "input" / "molecule.fchk"
    fchk.parent.mkdir()
    _write_minimal_fchk(fchk)
    output = tmp_path / "output"

    result = module.run(
        fchk=fchk,
        stationary_point="ground_state",
        output_root=output,
        bounded_ci=True,
    )

    assert result["input"] == {"kind": "caller_fchk", "name": "molecule.fchk"}
    assert result["choices"]["stationary_point"] == "ground_state"
    assert (output / "ch3f.frcmod").is_file()


def test_sn2_example_rejects_ground_state_override(tmp_path: Path) -> None:
    module = _load(EXAMPLES / "ch3f-sn2" / "run.py", "example_ch3f_sn2_stationary_point")
    with pytest.raises(Exception, match="fixed 'transition_state' semantics"):
        module.run(output_root=tmp_path, stationary_point="ground_state", bounded_ci=True)


def test_small_example_cli_prints_json_and_writes_only_output_root(tmp_path: Path) -> None:
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "ch3f" / "run.py"),
            "--bounded-ci",
            "--output-root",
            str(output),
        ],
        check=True,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    result = json.loads(completed.stdout)
    assert result["example"] == "ch3f"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]


def test_publication_entry_scripts_only_configure_canonical_differences() -> None:
    expected = {
        "rh-enamide": ("partial_repository_reproduction", 0.20),
        "heck-relay": ("executable_archive_reproduction", 0.05),
        "pd-allyl": ("partial_repository_reproduction", 0.20),
        "pd-conjugate": ("partial_repository_reproduction", 0.20),
        "rh-conjugate": ("sdk_software_path_demonstration", 0.20),
        "ferrocene": ("partial_repository_reproduction", None),
    }
    for key, (status, fc_fraction) in expected.items():
        module = _load(EXAMPLES / "publication" / key / "run.py", f"publication_{key.replace('-', '_')}")
        assert callable(module.run)
        assert callable(module.main)
        assert module.CONFIG.key == key
        assert module.CONFIG.expected_status == status
        assert module.CONFIG.fc_fraction == fc_fraction
    assert sys.modules["publication_ferrocene"].CONFIG.default_starting_point == "published"
    assert sys.modules["publication_ferrocene"].CONFIG.objective_profile == "wahlers-ferrocene-seven-structure-v1"


def test_publication_cli_reports_actionable_typed_missing_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "publication" / "rh-enamide" / "run.py"),
            "--bounded-ci",
            "--output-root",
            str(tmp_path / "output"),
        ],
        check=False,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    error = json.loads(completed.stderr)
    assert completed.returncode == 2
    assert error["error_type"] == "ExampleConfigurationError"
    assert "--rh-enamide" in error["message"]


def test_example_tree_is_canonical_and_free_of_generated_artifacts() -> None:
    directories = {path.name for path in EXAMPLES.iterdir() if path.is_dir() and not path.name.startswith("__")}
    assert directories == {"backend-plugin", "ch3f", "ch3f-sn2", "publication"}
    publication = {
        path.name for path in (EXAMPLES / "publication").iterdir() if path.is_dir() and not path.name.startswith("__")
    }
    assert publication == {
        "rh-enamide",
        "heck-relay",
        "pd-allyl",
        "pd-conjugate",
        "rh-conjugate",
        "ferrocene",
    }
    tracked = subprocess.run(
        ["git", "ls-files", "examples"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.splitlines()
    offenders = [
        path
        for path in tracked
        if "__pycache__" in path or ".egg-info/" in path or "/build/" in path or path.endswith(".pyc")
    ]
    assert not offenders
