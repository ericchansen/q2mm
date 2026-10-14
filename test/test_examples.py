from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Any

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
    assert result["choices"]["backend"] == "example-bounded-echo"
    assert result["choices"]["executor"] == "python"
    assert result["qfuerza"]["settings"]["invert_ts_curvature"] is (key == "ch3f-sn2")
    assert {path.name for path in output.iterdir()} == {
        f"{key}.frcmod",
        f"{key}.frcmod.manifest.json",
    }
    manifest = json.loads((output / f"{key}.frcmod.manifest.json").read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    assert configuration["backend"]["key"] == "example-bounded-echo"
    assert configuration["optimizer"]["settings"]["parameters"]["convergence_claim"] is False


def test_example_captures_actual_components_before_optimizer_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import q2mm

    module = _load(EXAMPLES / "ch3f" / "run.py", "example_ch3f_capture")
    capture = q2mm.capture_execution_configuration
    optimize = q2mm.optimize
    captured = []

    def capture_before_run(settings: Any, *, backend: Any, optimizer: Any, workflow: Any) -> Any:
        assert optimizer.entered is False
        captured.append((optimizer, workflow))
        return capture(settings, backend=backend, optimizer=optimizer, workflow=workflow)

    def optimize_same_objects(problem: Any, **kwargs: Any) -> Any:
        assert (kwargs["optimizer"], kwargs["workflow"]) == captured[-1]
        return optimize(problem, **kwargs)

    monkeypatch.setattr(q2mm, "capture_execution_configuration", capture_before_run)
    monkeypatch.setattr(q2mm, "optimize", optimize_same_objects)
    module.run(output_root=tmp_path, bounded_ci=True)
    assert len(captured) == 1
    assert captured[0][0].entered is True


@pytest.mark.parametrize("key", ("ch3f", "ch3f-sn2"))
def test_small_example_selects_named_settings_from_declared_stationarity(
    key: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import q2mm
    from q2mm.settings import GroundStateOptimizationSettings, TransitionStateOptimizationSettings

    module = _load(EXAMPLES / key / "run.py", f"example_named_settings_{key.replace('-', '_')}")
    runner = module._RUNNER
    backend = runner._SUPPORT.BoundedEchoBackend()
    monkeypatch.setattr(runner, "load_backend", lambda *args, **kwargs: backend)
    expected = GroundStateOptimizationSettings if key == "ch3f" else TransitionStateOptimizationSettings

    def check_capture(settings: Any, *, backend: Any, optimizer: Any, workflow: Any) -> Any:
        assert isinstance(settings, expected)
        assert optimizer.settings is settings.optimizer
        assert workflow.settings is settings.workflow
        assert settings.optimizer.verbose is False
        assert settings.objective.executor == "jax"
        assert settings.objective.gradient_mode == "analytical"
        assert settings.optimizer.fc_fraction == (None if key == "ch3f" else 0.20)
        raise RuntimeError("named-settings construction verified before numerical work")

    monkeypatch.setattr(q2mm, "capture_execution_configuration", check_capture)
    with pytest.raises(RuntimeError, match="named-settings construction verified"):
        module.run(output_root=tmp_path, bounded_ci=False)


@pytest.mark.parametrize("key", ("ch3f", "ch3f-sn2"))
@pytest.mark.parametrize("backend_key", ("openmm", "custom-mm"))
def test_small_example_explicit_python_executor_enters_scipy_and_saves(
    key: str,
    backend_key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    module = _load(EXAMPLES / key / "run.py", f"explicit_python_{key.replace('-', '_')}")
    runner = module._RUNNER
    backend = runner._SUPPORT.BoundedEchoBackend()
    loaded = []
    entered = []
    original = ScipyOptimizer.optimize

    def load(selected: str, **options: Any) -> Any:
        loaded.append(selected)
        return backend

    def optimize(self: ScipyOptimizer, evaluator: Any, space: Any) -> Any:
        assert evaluator.gradient_mode.value == "none"
        entered.append(self.settings)
        return original(self, evaluator, space)

    monkeypatch.setattr(runner, "load_backend", load)
    monkeypatch.setattr(ScipyOptimizer, "optimize", optimize)
    result = module.run(output_root=tmp_path, backend=backend_key, executor="python")
    assert loaded == [backend_key] and len(entered) == 1
    assert result["choices"]["executor"] == "python"
    assert entered[0].ftol == (1e-8 if key == "ch3f" else 1e-12)
    assert entered[0].fc_fraction == (None if key == "ch3f" else 0.20)
    assert entered[0].eq_fraction == (None if key == "ch3f" else 0.05)
    manifest = json.loads((tmp_path / f"{key}.frcmod.manifest.json").read_text())
    assert manifest["configuration"]["executor"]["gradient_mode"] == "none"
    assert manifest["result"]["gradient_mode"] == "finite_difference"
    assert manifest["result"]["n_evals"] == manifest["configuration"]["n_evals"] == 1


@pytest.mark.parametrize("key", ("ch3f", "ch3f-sn2"))
def test_small_example_rejects_implicit_executor_change_before_backend_loading(
    key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EXAMPLES / key / "run.py", f"reject_implicit_{key.replace('-', '_')}")

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Contradictory typed settings must fail before backend construction.")

    monkeypatch.setattr(module._RUNNER, "load_backend", forbidden)
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        module.run(output_root=tmp_path, backend="openmm")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ("rh-enamide", "heck-relay", "ferrocene"))
def test_publication_example_explicit_python_executor_preserves_bounds_and_saves(
    key: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.benchmarks.publications import publication_record
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from test.test_application import _EnergyBackend, _problem

    module = _load(EXAMPLES / "publication" / key / "run.py", f"explicit_publication_{key.replace('-', '_')}")
    runner = module._RUNNER
    record = publication_record(key, module.CONFIG.objective_profile, module.CONFIG.default_starting_point)
    record = replace(record, authoritative_case_ids=("h2",))
    problem = replace(_problem(form=FunctionalForm.MM3, ts=key != "ferrocene"), publication_metadata=record)
    monkeypatch.setattr(runner, "_validate_roots", lambda *args: None)
    monkeypatch.setattr(runner, "publication_record", lambda *args: record)
    monkeypatch.setattr(
        runner, "load_system", lambda *args, **kwargs: BenchmarkCase(key=key, name=key, problem=problem)
    )
    monkeypatch.setattr(runner, "load_backend", lambda name, **options: _EnergyBackend())
    captured = []
    original = q2mm.capture_execution_configuration

    def capture(settings: Any, **components: Any) -> Any:
        assert isinstance(components["optimizer"], ScipyOptimizer)
        captured.append(settings)
        return original(settings, **components)

    monkeypatch.setattr(q2mm, "capture_execution_configuration", capture)
    result = module.run(output_root=tmp_path, backend="openmm", executor="python")
    assert len(captured) == 1
    assert captured[0].backend.key == "openmm"
    assert captured[0].objective.executor == "python"
    assert captured[0].optimizer.fc_fraction == module.CONFIG.fc_fraction
    assert captured[0].optimizer.eq_fraction == module.CONFIG.eq_fraction
    assert captured[0].optimizer.ftol == (1e-8 if key == "ferrocene" else 1e-12)
    assert result["execution"]["executor"] == "python"
    manifest = json.loads(Path(result["saved"]["manifest"]).read_text())
    assert manifest["configuration"]["executor"]["gradient_mode"] == "none"
    assert manifest["result"]["n_evals"] == manifest["configuration"]["n_evals"] == 1
    monkeypatch.setattr(runner, "load_backend", lambda *args, **kwargs: pytest.fail("Unexpected backend load"))
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        module.run(output_root=tmp_path / "invalid", backend="openmm")
    assert not list((tmp_path / "invalid").iterdir())


@pytest.mark.parametrize("kind", ("small", "publication"))
def test_example_cli_requires_explicit_non_jax_executor_choice(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = EXAMPLES / "ch3f" / "run.py" if kind == "small" else EXAMPLES / "publication" / "rh-enamide" / "run.py"
    module = _load(path, f"example_executor_cli_{kind}")
    runner = module._RUNNER
    kwargs = []
    target = "run_small" if kind == "small" else "run_publication"
    monkeypatch.setattr(runner, target, lambda config, **options: kwargs.append(options) or {"ok": True})
    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--output-root", str(tmp_path), "--backend", "openmm", "--executor", "python"],
    )
    assert module.main() == 0
    assert kwargs[0]["backend"] == "openmm" and kwargs[0]["executor"] == "python"
    assert json.loads(capsys.readouterr().out) == {"ok": True}


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
