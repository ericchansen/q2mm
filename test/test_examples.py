from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import ModuleType
from typing import Any

import pytest
import numpy as np

from scripts.check_installed_examples import _write_minimal_fchk
from test._shared import REPO_ROOT

EXAMPLES = REPO_ROOT / "examples"


@pytest.fixture(autouse=True)
def _cpu_only_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import example {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _with_platform(backend: Any, name: str = "CPU") -> Any:
    backend.info = replace(
        backend.info,
        provenance=replace(backend.info.provenance, details={"platform": {"name": name}}),
    )
    return backend


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
    assert result["execution_mode"] == "software_path_only"
    assert result["optimization"]["iterations"] == 1
    assert result["optimization"]["convergence_claim"] is False
    assert result["optimization"]["parameter_update_norm"] == 0
    assert result["choices"]["backend"] == "example-bounded-echo"
    assert result["choices"]["executor"] == "python"
    assert result["qfuerza"]["settings"]["invert_ts_curvature"] is (key == "ch3f-sn2")
    assert {path.name for path in output.iterdir()} == {
        f"{key}.frcmod",
        f"{key}.frcmod.manifest.json",
    }
    manifest = json.loads((output / f"{key}.frcmod.manifest.json").read_text(encoding="utf-8"))
    configuration = manifest["configuration"]
    assert manifest["provenance"]["example"]["convergence_claim"] is False
    assert manifest["provenance"]["example"]["execution_mode"] == "software_path_only"
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


def test_small_example_explicit_python_executor_enters_scipy_and_saves(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    module = _load(EXAMPLES / "ch3f" / "run.py", "explicit_python_ch3f")
    runner = module._RUNNER
    backend = _with_platform(runner._SUPPORT.BoundedEchoBackend())
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
    settings = runner._command_type(module.CONFIG)(
        output_root=tmp_path,
        backend=runner.BackendSettings(key="custom-mm"),
        objective=runner.ObjectiveSettings(executor="python"),
    )
    result = module.run(output_root=tmp_path, settings=settings)
    assert loaded == ["custom-mm"] and len(entered) == 1
    assert result["choices"]["executor"] == "python"
    assert result["choices"]["platform"] == "CPU"
    manifest = json.loads((tmp_path / "ch3f.frcmod.manifest.json").read_text())
    assert manifest["configuration"]["executor"]["gradient_mode"] == "none"
    assert manifest["result"]["gradient_mode"] == "finite_difference"
    assert manifest["result"]["n_evals"] == manifest["configuration"]["n_evals"] == 0
    assert manifest["configuration"]["backend"]["details"]["platform"]["name"] == "CPU"
    assert manifest["provenance"]["example"]["platform"] == "CPU"

    monkeypatch.setattr(runner, "load_backend", lambda *args, **kwargs: pytest.fail("Unexpected backend load"))
    with pytest.raises(ValueError, match="explicitly select the Python objective"):
        runner._command_type(module.CONFIG)(
            output_root=tmp_path / "invalid", backend=runner.BackendSettings(key="openmm")
        )
    assert not (tmp_path / "invalid").exists()


def test_example_factory_receives_native_nested_json_without_mutating_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(EXAMPLES / "ch3f" / "run.py", "example_nested_backend_options")
    runner = module._RUNNER
    backend = _with_platform(runner._SUPPORT.BoundedEchoBackend())
    options = {"config": {"layers": [{"weights": [1, 2]}]}}
    received = []

    def load(key: str, **parameters: Any) -> Any:
        assert key == "custom-mm"
        assert json.loads(json.dumps(parameters)) == options
        assert isinstance(parameters["config"], dict)
        assert isinstance(parameters["config"]["layers"], list)
        received.append(parameters)
        parameters["config"]["layers"][0]["weights"].append(3)
        return backend

    settings = runner._command_type(module.CONFIG)(
        output_root=tmp_path,
        backend=runner.BackendSettings(key="custom-mm", options=options),
        objective=runner.ObjectiveSettings(executor="python"),
    )
    before = settings.model_dump(mode="json")
    monkeypatch.setattr(runner, "load_backend", load)
    result = module.run(output_root=tmp_path, settings=settings)
    assert len(received) == 1
    assert result["choices"]["executor"] == "python"
    assert settings.model_dump(mode="json") == before
    assert settings.backend.options["config"]["layers"][0]["weights"] == (1, 2)
    with pytest.raises(TypeError):
        settings.backend.options["config"]["layers"][0]["weights"][0] = 3


@pytest.mark.parametrize(
    ("options", "environment"),
    [
        ({}, None),
        ({}, "CUDA"),
        ({"platform_name": "CUDA"}, "CPU"),
        ({"platform_name": "OpenCL"}, "CPU"),
    ],
)
def test_jax_cpu_does_not_allow_openmm_gpu_factory_or_data_loading(
    options: dict[str, str],
    environment: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(EXAMPLES / "ch3f" / "run.py", "example_openmm_cpu_request")
    runner = module._RUNNER
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    if environment is None:
        monkeypatch.delenv("OPENMM_DEFAULT_PLATFORM", raising=False)
    else:
        monkeypatch.setenv("OPENMM_DEFAULT_PLATFORM", environment)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("CPU policy must reject before factories, scientific data, or GPU context allocation.")

    monkeypatch.setattr(runner, "load_backend", forbidden)
    monkeypatch.setattr(runner, "_installed_input_root", forbidden)
    settings = runner._command_type(module.CONFIG)(
        output_root=tmp_path / "output",
        backend=runner.BackendSettings(key="openmm", options=options),
        objective=runner.ObjectiveSettings(executor="python"),
    )
    with pytest.raises(ValueError, match="JAX_PLATFORMS does not control OpenMM"):
        module.run(output_root=tmp_path / "output", settings=settings)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("platform", ("CUDA", "OpenCL", None))
def test_explicit_cpu_request_rechecks_selected_backend_before_prepare_or_evaluation(
    platform: str | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm

    module = _load(EXAMPLES / "ch3f" / "run.py", "example_openmm_cpu_selected")
    runner = module._RUNNER
    backend = runner._SUPPORT.BoundedEchoBackend()
    if platform is not None:
        backend = _with_platform(backend, platform)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Non-CPU selected backend must not prepare a context or evaluate.")

    monkeypatch.setattr(backend, "prepare", forbidden)
    monkeypatch.setattr(runner, "load_backend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(q2mm, "make_evaluator_factory", forbidden)
    settings = runner._command_type(module.CONFIG)(
        output_root=tmp_path / "output",
        backend=runner.BackendSettings(key="openmm", options={"platform_name": "CPU"}),
        objective=runner.ObjectiveSettings(executor="python"),
    )
    with pytest.raises(ValueError, match="requires a verified CPU backend"):
        module.run(output_root=tmp_path / "output", settings=settings)
    assert not (tmp_path / "output").exists()


def test_publication_example_python_execution_saves_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.benchmarks.publications import publication_record
    from q2mm.models.forcefield import FunctionalForm
    from test.test_application import _EnergyBackend, _problem

    key = "heck-relay"
    module = _load(EXAMPLES / "publication" / key / "run.py", "explicit_publication")
    runner = module._RUNNER
    record = publication_record(key, module.CONFIG.objective_profile, module.CONFIG.default_starting_point)
    record = replace(record, authoritative_case_ids=("h2",))
    problem = replace(_problem(form=FunctionalForm.MM3, ts=True), publication_metadata=record)
    monkeypatch.setattr(runner, "_validate_roots", lambda *args: None)
    monkeypatch.setattr(runner, "publication_record", lambda *args: record)
    monkeypatch.setattr(
        runner, "load_system", lambda *args, **kwargs: BenchmarkCase(key=key, name=key, problem=problem)
    )
    monkeypatch.setattr(runner, "load_backend", lambda name, **options: _EnergyBackend())
    result = module.run(output_root=tmp_path, backend="openmm", executor="python")
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
    options = (
        ["--backend.key", "openmm", "--objective.executor", "python", "--objective.gradient-mode", "none"]
        if kind == "small"
        else ["--backend", "openmm", "--executor", "python"]
    )
    monkeypatch.setattr(sys, "argv", ["run.py", "--output-root", str(tmp_path), *options])
    assert module.main() == 0
    if kind == "small":
        assert kwargs[0]["settings"].backend.key == "openmm"
        assert kwargs[0]["settings"].objective.executor == "python"
    else:
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


def test_reference_root_override_records_caller_provenance_and_rejects_incomplete_inputs(tmp_path: Path) -> None:
    from q2mm.resources import sn2_reference_dir

    module = _load(EXAMPLES / "ch3f" / "run.py", "example_ch3f_root")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    shutil.copyfile(sn2_reference_dir() / "ch3f-optimized.xyz", inputs / "ch3f-optimized.xyz")
    with pytest.raises(ValueError, match="ch3f-hessian.npy"):
        module.run(input_root=inputs, output_root=tmp_path / "output", bounded_ci=True)
    assert not (tmp_path / "output").exists()
    shutil.copyfile(sn2_reference_dir() / "ch3f-hessian.npy", inputs / "ch3f-hessian.npy")
    result = module.run(input_root=inputs, output_root=tmp_path / "output", bounded_ci=True)
    assert result["input"]["kind"] == "caller_reference_root"
    molecule = module._RUNNER._load_packaged_molecule(module.CONFIG, inputs, source="caller-sn2-reference-root")
    assert molecule.hessian_provenance.source == "caller-sn2-reference-root"


def test_fchk_without_force_constants_fails_without_saved_outputs(tmp_path: Path) -> None:
    module = _load(EXAMPLES / "ch3f" / "run.py", "example_ch3f_no_force_constants")
    fchk = tmp_path / "incomplete.fchk"
    _write_minimal_fchk(fchk)
    fchk.write_text(fchk.read_text(encoding="ascii").split("Cartesian Force Constants", 1)[0], encoding="ascii")
    with pytest.raises(ValueError, match="missing required canonical Hessians"):
        module.run(fchk=fchk, output_root=tmp_path / "output", bounded_ci=True)
    assert not (tmp_path / "output").exists()


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


def test_native_example_requires_explicit_cpu_before_creating_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(EXAMPLES / "ch3f" / "run.py", "example_cpu_guard")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    with pytest.raises(ValueError, match="Set JAX_PLATFORMS=cpu"):
        module.run(output_root=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_example_cli_inspection_is_native_model_data_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pydantic_settings import CliApp

    module = _load(EXAMPLES / "ch3f-sn2" / "run.py", "example_cli_inspection")
    runner = module._RUNNER
    command = CliApp.run(
        runner.TransitionStateExampleCommand,
        cli_args=["--optimizer.maxiter", "2", "--output-root", str(tmp_path / "missing"), "--show-config"],
    )
    restored = runner.TransitionStateExampleCommand.model_validate_json(command.model_dump_json())
    assert restored.model_dump() == command.model_dump()
    assert command.optimizer.maxiter == 2
    assert command.optimizer.maxls == 5
    assert command.optimizer.ftol == 1e-12
    assert command.optimizer.fc_fraction == 0.20
    assert command.optimizer.eq_fraction == 0.05
    assert command.n_evals == 0

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Inspection loaded data or a runtime backend")

    monkeypatch.setattr(runner, "_installed_input_root", forbidden)
    monkeypatch.setattr(runner, "load_backend", forbidden)
    monkeypatch.setattr(sys, "argv", ["run.py", "--output-root", str(tmp_path / "missing"), "--show-config"])
    assert module.main() == 0
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (["--fchk", "missing.fchk"], "FCHK input is not a file"),
        (["--input-root", "missing"], "Input root is not a directory"),
        (["--fchk", "missing.fchk", "--input-root", "missing"], "not both"),
        (["--optimizer.maxiter", "-1"], "greater than or equal to 0"),
    ],
)
def test_small_example_cli_reports_input_errors_without_outputs(
    options: list[str], message: str, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "ch3f" / "run.py"), "--bounded-ci", "--output-root", str(output), *options],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 2
    assert message in json.loads(completed.stderr)["message"]
    assert not output.exists()


@pytest.mark.jax
@pytest.mark.parametrize("key", ("ch3f", "ch3f-sn2"))
def test_documented_native_cpu_command_has_real_budget_and_saved_parameters(key: str, tmp_path: Path) -> None:
    environment = {**os.environ, "JAX_PLATFORMS": "cpu"}
    output = tmp_path / "output"
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / key / "run.py"),
            "--optimizer.maxiter",
            "2",
            "--output-root",
            str(output),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=180,
    )
    result = json.loads(completed.stdout)
    assert result["bounded_ci"] is False
    assert result["execution_mode"] == "budget_limited_native"
    assert result["choices"]["backend"] == "jax"
    assert result["choices"]["executor"] == "jax"
    assert result["choices"]["platform"] == "cpu"
    assert result["optimization"]["iterations"] <= 2
    assert result["optimization"]["convergence_claim"] is False
    assert np.isfinite(result["initial"]["total"])
    assert np.isfinite(result["final"]["total"])
    manifest = json.loads((output / f"{key}.frcmod.manifest.json").read_text(encoding="utf-8"))
    parameters = manifest["configuration"]["optimizer"]["settings"]["parameters"]
    assert manifest["provenance"]["example"]["convergence_claim"] is False
    assert manifest["provenance"]["example"]["execution_mode"] == "budget_limited_native"
    assert manifest["provenance"]["example"]["platform"] == "cpu"
    assert parameters["maxiter"] == 2
    assert parameters["maxls"] == 5
    assert manifest["configuration"]["executor"]["gradient_mode"] == "analytical"
    update = np.linalg.norm(np.array(manifest["result"]["final_params"]) - manifest["result"]["initial_params"])
    assert update == pytest.approx(result["optimization"]["parameter_update_norm"])
    if key == "ch3f":
        assert update > 0
    assert {path.name for path in output.iterdir()} == {f"{key}.frcmod", f"{key}.frcmod.manifest.json"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]


@pytest.mark.openmm
@pytest.mark.parametrize(
    ("options", "platform_environment"),
    [({"platform_name": "CPU"}, "CUDA"), ({}, "CPU")],
)
def test_documented_openmm_cpu_selection_runs_without_jax(
    options: dict[str, str], platform_environment: str, tmp_path: Path
) -> None:
    environment = {**os.environ, "OPENMM_DEFAULT_PLATFORM": platform_environment, "OPENMM_CPU_THREADS": "1"}
    environment.pop("JAX_PLATFORMS", None)
    completed = subprocess.run(
        [
            sys.executable,
            str(EXAMPLES / "ch3f" / "run.py"),
            "--backend.key",
            "openmm",
            "--backend.options",
            json.dumps(options),
            "--objective.executor",
            "python",
            "--objective.gradient-mode",
            "none",
            "--optimizer.maxiter",
            "2",
            "--output-root",
            str(tmp_path / "output"),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=180,
    )
    result = json.loads(completed.stdout)
    assert result["choices"]["backend"] == "openmm"
    assert result["choices"]["executor"] == "python"
    assert result["choices"]["platform"] == "CPU"
    assert result["optimization"]["convergence_claim"] is False
    assert result["optimization"]["iterations"] <= 2
    manifest = json.loads((tmp_path / "output" / "ch3f.frcmod.manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["backend"]["details"]["platform"]["name"] == "CPU"
    assert manifest["provenance"]["example"]["platform"] == "CPU"
    assert {path.name for path in (tmp_path / "output").iterdir()} == {"ch3f.frcmod", "ch3f.frcmod.manifest.json"}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["output"]


@pytest.mark.jax
def test_tutorial_executes_canonical_runner_and_settings_json_from_temporary_checkout(tmp_path: Path) -> None:
    tutorial = (REPO_ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")
    code = tutorial.split("## The same example from Python\n", 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    for relative in ("_small_runner.py", "_support.py", "ch3f/run.py"):
        source = EXAMPLES / relative
        destination = tmp_path / "examples" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert {path.name for path in (tmp_path / "results" / "ch3f").iterdir()} == {
        "ch3f.frcmod",
        "ch3f.frcmod.manifest.json",
    }


@pytest.mark.parametrize("section", ("## Bring your own FCHK", "## XYZ plus an explicit Hessian"))
def test_tutorial_preparation_snippets_use_actual_inputs(section: str, tmp_path: Path) -> None:
    from q2mm.resources import sn2_reference_dir

    tutorial = (REPO_ROOT / "docs" / "tutorial.md").read_text(encoding="utf-8")
    code = tutorial.split(section, 1)[1].split("```python\n", 1)[1].split("```", 1)[0]
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_minimal_fchk(inputs / "minimum.fchk")
    for resource, name in (("ch3f-optimized.xyz", "minimum.xyz"), ("ch3f-hessian.npy", "minimum-hessian.npy")):
        shutil.copyfile(sn2_reference_dir() / resource, inputs / name)
    subprocess.run(
        [sys.executable, "-c", code + "\nassert problem.cases[0].stationary_point.value == 'ground_state'\n"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["inputs"]


def test_tutorial_xyz_requires_a_real_hessian_shape(tmp_path: Path) -> None:
    from q2mm.io.xyz import load_xyz
    from q2mm.models.hessian import HessianProvenance, HessianUnits
    from q2mm.resources import sn2_reference_dir
    from q2mm.settings import PreparationSettings
    import q2mm

    molecule = load_xyz(sn2_reference_dir() / "ch3f-optimized.xyz", charge=0, bond_tolerance=1.5)
    with pytest.raises(ValueError, match="Hessian"):
        q2mm.prepare(
            molecule, settings=PreparationSettings(stationary_point="ground_state", functional_form="harmonic")
        )
    with pytest.raises(ValueError, match="shape"):
        molecule.with_hessian(
            np.eye(3),
            HessianProvenance(units=HessianUnits.ATOMIC, source="caller-quantum-calculation"),
        )


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
