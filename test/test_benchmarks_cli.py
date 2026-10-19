"""Actual-argv contracts for the native Pydantic benchmark command models."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import JsonValue
from pydantic_settings import CliApp, SettingsConfigDict, get_subcommand

from q2mm.backends.contracts import BackendDescriptor
from q2mm.benchmarks import cli
from q2mm.benchmarks.cli import BatchCommand, BenchmarkCLI, MatrixCommand, SingleCommand, main
from q2mm.benchmarks.profiles import RunProfile, catalog_settings
from q2mm.benchmarks.records import RunOutcome
from q2mm.settings import OptaxSettings, ScipySettings
from test.test_benchmarks_profiles import configured_registry as configured_registry


def _parse(argv: list[str]) -> SingleCommand | BatchCommand | MatrixCommand:
    root = CliApp.run(BenchmarkCLI, cli_args=argv, cli_cmd_method_name="parse_only", cli_exit_on_error=False)
    command = get_subcommand(root)
    assert isinstance(command, (SingleCommand, BatchCommand, MatrixCommand))
    return command


@pytest.mark.parametrize("argv", [[], ["unknown"], ["single"], ["single", "--unknown", "1"], ["load"]])
def test_syntax_errors_return_two(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert main(argv) == 2
    assert "Error:" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["single", "batch", "matrix", "load", "list", "preflight"])
def test_generated_help(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main([command, "--help"]) == 0
    assert "usage:" in capsys.readouterr().out


def test_list_and_explicit_preflight(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert all(section in output for section in ("Systems:", "Backends", "Optimizers:", "ch3f"))
    probe = Mock(return_value=0)
    monkeypatch.setattr(cli, "_cmd_preflight", probe)
    assert main(["preflight"]) == 0
    probe.assert_called_once_with()


def test_no_runtime_or_data_import_for_help_list_and_inspection() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys
        forbidden = {"jax", "jaxlib", "openmm", "scipy", "optax", "jaxopt", "psi4"}
        class RejectLoader(importlib.abc.Loader):
            def create_module(self, spec):
                return None
            def exec_module(self, module):
                raise AssertionError("runtime import: " + module.__name__)
        class RejectRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in forbidden:
                    return __import__("importlib.machinery").machinery.ModuleSpec(fullname, RejectLoader())
        sys.meta_path.insert(0, RejectRuntime())
        from q2mm.benchmarks.cli import main
        import q2mm.benchmarks.systems as systems
        def reject(*args, **kwargs):
            raise AssertionError("scientific loading")
        systems.load_system = reject
        for args in (
            ["--help"], ["single", "--help"], ["list"],
            ["single", "--system", "ch3f", "--show-config"],
            ["batch", "--show-config"], ["matrix", "--system", "ch3f", "--show-config"],
        ):
            assert main(args) == 0, args
        assert not forbidden.intersection(sys.modules)
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, encoding="utf-8")
    assert result.returncode == 0, result.stdout + result.stderr


def test_single_defaults_and_native_zero_false_none() -> None:
    command = _parse(
        [
            "single",
            "--system",
            "ch3f",
            "--optimizer.maxiter",
            "0",
            "--optimizer.ftol",
            "0",
            "--optimizer.fc-fraction",
            "null",
            "--no-optimizer.verbose",
            "--no-optimizer.use-bounds",
            "--n-evals",
            "0",
            "--ratio-tol",
            "null",
            "--objective.regularization",
            "0",
            "--objective",
            '{"executor":"python","gradient_mode":"none"}',
            "--backend.key",
            "installed-plugin",
            "--backend.options",
            '{"enabled":false,"threads":0}',
        ]
    )
    assert isinstance(command, SingleCommand)
    assert isinstance(command.optimizer, ScipySettings)
    assert command.optimizer.maxiter == command.optimizer.ftol == command.n_evals == 0
    assert command.optimizer.fc_fraction is command.ratio_tol is command.objective.fd_step is None
    assert command.optimizer.verbose is command.optimizer.use_bounds is False
    assert command.objective.executor == "python" and command.objective.gradient_mode == "none"
    assert command.backend.key == "installed-plugin"
    assert command.backend.options == {"enabled": False, "threads": 0}
    assert command.workflow.name == "single-stage"
    assert command.analyze and command.promote


def test_family_switching_does_not_keep_scipy_fields() -> None:
    command = _parse(
        [
            "single",
            "--system",
            "ch3f",
            "--optimizer.kind",
            "optax",
            "--optimizer.optimizer",
            "sgd",
            "--optimizer.max-steps",
            "0",
            "--optimizer.learning-rate",
            "0",
            "--optimizer.schedule",
            "cosine",
            "--no-optimizer.verbose",
        ]
    )
    assert isinstance(command.optimizer, OptaxSettings)
    assert command.optimizer.max_steps == command.optimizer.learning_rate == 0
    assert command.optimizer.schedule == "cosine"
    assert not {"maxiter", "fc_fraction", "eq_fraction"} & command.optimizer.model_dump().keys()


@pytest.mark.parametrize("command", ["single", "batch"])
def test_shared_backend_executor_guard_is_preserved_before_dispatch(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Mock(side_effect=AssertionError("incoherent settings entered execution"))
    monkeypatch.setattr(cli, "run_profiles", runner)
    assert (
        main(
            [
                command,
                "--system",
                "ch3f",
                "--backend.key",
                "plugin",
                "--output",
                str(tmp_path),
                "--show-config",
            ]
        )
        == 2
    )
    assert "The JAX objective requires backend='jax'" in capsys.readouterr().err
    runner.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "optimizer",
    [
        {"kind": "jaxopt"},
        {"kind": "jax-multistart"},
        {"kind": "optax"},
        {"kind": "multistart", "optimizer": {"kind": "jaxopt"}},
        {"kind": "cycling", "full_optimizer": {"kind": "multistart", "optimizer": {"kind": "optax"}}},
    ],
)
@pytest.mark.parametrize("inspection", [[], ["--show-config"]])
def test_invalid_optimizer_objective_fails_before_backend_or_data_loading(
    optimizer: dict,
    inspection: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import q2mm.backends.registry as registry
    import q2mm.benchmarks.systems as systems

    backend = Mock(side_effect=AssertionError("backend construction"))
    data = Mock(side_effect=AssertionError("scientific loading"))
    monkeypatch.setattr(registry, "load_backend", backend)
    monkeypatch.setattr(systems, "load_system", data)
    assert (
        main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend.key",
                "openmm",
                "--objective.executor",
                "python",
                "--objective.gradient-mode",
                "none",
                "--optimizer",
                json.dumps(optimizer),
                "--output",
                str(tmp_path),
                *inspection,
            ]
        )
        == 2
    )
    assert "requires a" in capsys.readouterr().err
    backend.assert_not_called()
    data.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_explicit_python_objective_uses_native_gradient_default() -> None:
    from q2mm.settings import ObjectiveSettings

    objective = ObjectiveSettings(executor="python")
    assert objective.gradient_mode == "none" and objective.fd_step is None
    command = _parse(
        [
            "single",
            "--system",
            "ch3f",
            "--backend.key",
            "plugin",
            "--objective",
            objective.model_dump_json(),
        ]
    )
    assert command.objective == objective
    assert command.backend.key == "plugin"


@pytest.mark.parametrize("mode", ["none", "finite_difference"])
def test_actual_dotted_gradient_none_does_not_collide_with_null_token(mode: str) -> None:
    args = [
        "single",
        "--system",
        "ch3f",
        "--objective.executor",
        "python",
        "--objective.gradient-mode",
        mode,
        "--objective.fd-step",
        "null" if mode == "none" else "0.002",
        "--ratio-tol",
        "null",
        "--optimizer.fc-fraction",
        "null",
    ]
    command = _parse(args)
    assert command.objective.executor == "python" and command.objective.gradient_mode == mode
    assert command.objective.fd_step == (None if mode == "none" else 0.002)
    assert command.ratio_tol is command.optimizer.fc_fraction is None


def test_actual_dotted_python_scalar_options_reach_factory_as_plain_json(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
) -> None:
    from q2mm.benchmarks.profiles import _classify_backend

    descriptor, backend, factory = configured_registry
    options = {
        "configured_path": "explicit-location",
        "config": {
            "modes": ["fast", {"enabled": False, "limit": 0, "fallback": None}],
        },
    }
    command = _parse(
        [
            "single",
            "--system",
            "ch3f",
            "--backend.key",
            descriptor.name,
            "--backend.options",
            json.dumps(options),
            "--objective.executor",
            "python",
            "--objective.gradient-mode",
            "none",
            "--objective.fd-step",
            "null",
        ]
    )
    profile = RunProfile(**command.model_dump(exclude={"output", "analyze", "promote", "show_config"}))
    assert _classify_backend(profile) == (descriptor, backend)
    assert command.objective.gradient_mode == "none"
    factory.assert_called_once_with(**options)
    supplied = factory.call_args.kwargs["config"]
    assert type(supplied) is dict and type(supplied["modes"]) is list
    assert type(supplied["modes"][1]) is dict


def test_native_nested_defaults_and_complete_policy_round_trip() -> None:
    from q2mm.settings import TransitionStateOptimizationSettings

    class TSSingle(TransitionStateOptimizationSettings):
        model_config = SettingsConfigDict(cli_kebab_case=True, cli_implicit_flags=True, cli_parse_none_str="null")
        system: str

    settings = TransitionStateOptimizationSettings(optimizer={"maxiter": 0, "fc_fraction": None, "verbose": False})
    command = CliApp.run(
        TSSingle,
        cli_args=[
            "--system",
            "rh-enamide",
            "--optimizer.maxiter",
            "0",
            "--optimizer.fc-fraction",
            "null",
            "--no-optimizer.verbose",
        ],
        cli_cmd_method_name="parse_only",
        cli_exit_on_error=False,
    )
    assert command.optimizer == settings.optimizer
    source = SingleCommand(system="ch3f", optimizer=OptaxSettings(max_steps=0, verbose=False))
    restored = _parse(["single", "--system", source.system, "--optimizer", source.optimizer.model_dump_json()])
    assert restored.model_dump() == SingleCommand.model_validate_json(source.model_dump_json()).model_dump()


def test_explicit_fd_controls_and_workflow_are_retained() -> None:
    command = _parse(
        [
            "single",
            "--system",
            "ch3f",
            "--objective",
            '{"executor":"python","gradient_mode":"finite_difference","fd_step":0.02}',
            "--workflow.name",
            "method-e2",
            "--workflow.negative-fc-threshold",
            "0",
            "--workflow.allow-negative",
        ]
    )
    assert command.objective.fd_step == 0.02
    assert command.workflow.name == "method-e2"
    assert command.workflow.negative_fc_threshold == 0 and command.workflow.allow_negative is True


@pytest.mark.parametrize(
    "argv",
    [
        ["--optimizer.maxiter", "-1"],
        ["--optimizer", '{"kind":"scipy","maxiter":false}'],
        ["--optimizer.ftol", "nan"],
        ["--optimizer.ftol", "inf"],
        ["--optimizer.unknown", "0"],
        ["--objective.fd-step", "0.02"],
        ["--objective", '{"executor":"python","gradient_mode":"finite_difference","fd_step":null}'],
        ["--objective", '{"executor":"python","gradient_mode":"analytical"}'],
        ["--data-roots", "bogus=path"],
        ["--data-roots", 'ch3f=""'],
        ["--ratio-tol", "-1"],
        ["--optimizer.kind", "optax", "--optimizer.maxiter", "0"],
    ],
)
def test_invalid_policy_is_syntax_failure(argv: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["single", "--system", "ch3f", "--output", str(tmp_path), *argv]) == 2
    assert "Error:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_repeated_selectors_and_native_key_path_roots() -> None:
    batch = _parse(
        [
            "batch",
            "--system",
            "ch3f",
            "--system",
            "ch3f-sn2",
            "--data-roots",
            r'ch3f="C:\data with spaces"',
            "--data-roots",
            "supporting_info=other",
        ]
    )
    assert isinstance(batch, BatchCommand)
    assert batch.system == ["ch3f", "ch3f-sn2"]
    assert batch.data_roots == {"ch3f": r"C:\data with spaces", "supporting_info": "other"}
    assert batch.workflow.name == "method-e2" and not batch.analyze
    matrix = _parse(
        [
            "matrix",
            "--system",
            "ch3f",
            "--backends",
            "jax",
            "--backends",
            "plugin",
            "--forms",
            "harmonic,mm3",
            "--presets",
            "scipy-lbfgsb",
            "--presets",
            "scipy-nm",
        ]
    )
    assert isinstance(matrix, MatrixCommand)
    assert matrix.backends == ["jax", "plugin"] and matrix.forms == ["harmonic", "mm3"]
    assert matrix.policies == [catalog_settings(key) for key in matrix.presets]
    assert matrix.policies[1].optimizer.method == "Nelder-Mead"
    configured = _parse(
        [
            "matrix",
            "--system",
            "ch3f",
            "--backends",
            '{"key":"openmm","options":{"platform_name":"CPU"}}',
            "--presets",
            "scipy-lbfgsb",
        ]
    )
    assert configured.backends[0].options == {"platform_name": "CPU"}


def test_selector_defaults_expand_only_at_execution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata
    from q2mm.optimizers.catalog import OPTIMIZER_CATALOG

    runner = Mock(return_value=RunOutcome())
    monkeypatch.setattr(cli, "run_profiles", runner)
    assert main(["batch", "--output", str(tmp_path)]) == 0
    batch = runner.call_args.args[0]
    assert [profile.system for profile in batch] == list(SYSTEM_KEYS)
    assert all(profile.workflow.name == "method-e2" for profile in batch)
    monkeypatch.setattr("q2mm.backends.registry.available_mm_backends", lambda: ["jax"])
    assert main(["matrix", "--system", "ch3f", "--output", str(tmp_path)]) == 0
    matrix = runner.call_args.args[0]
    assert len(matrix) == len(system_metadata("ch3f").default_forms) * len(OPTIMIZER_CATALOG)
    assert {profile.recipe_id for profile in matrix} == set(OPTIMIZER_CATALOG)
    assert all(profile.workflow.name == "single-stage" for profile in matrix)


def test_coherent_matrix_runs_every_requested_policy_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Mock(return_value=RunOutcome())
    monkeypatch.setattr(cli, "run_profiles", runner)
    assert (
        main(
            [
                "matrix",
                "--system",
                "ch3f",
                "--backends",
                "jax,plugin",
                "--forms",
                "harmonic",
                "--presets",
                "scipy-lbfgsb,scipy-nm",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    profiles = runner.call_args.args[0]
    assert [(p.backend.key, p.recipe_id, p.objective.executor) for p in profiles] == [
        ("jax", "scipy-lbfgsb", "python"),
        ("jax", "scipy-nm", "python"),
        ("plugin", "scipy-lbfgsb", "python"),
        ("plugin", "scipy-nm", "python"),
    ]
    output = capsys.readouterr()
    assert "4 candidates" in output.out
    assert output.err == ""


def test_all_incompatible_matrix_choices_fail_without_constructing_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runner = Mock(side_effect=AssertionError("invalid selection entered execution"))
    monkeypatch.setattr(cli, "run_profiles", runner)
    assert (
        main(
            [
                "matrix",
                "--system",
                "ch3f",
                "--backends",
                "plugin",
                "--forms",
                "harmonic",
                "--presets",
                "scipy-lbfgsb-jax",
                "--output",
                str(tmp_path),
            ]
        )
        == 2
    )
    runner.assert_not_called()
    assert list(tmp_path.iterdir()) == []
    assert "The JAX objective requires backend='jax'" in capsys.readouterr().err


@pytest.mark.parametrize("selection", ["explicit", "default", "inspection"])
def test_invalid_matrix_selection_after_valid_selection_fails_before_any_work(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    selection: str,
) -> None:
    descriptor, backend, factory = configured_registry
    loader = Mock(side_effect=AssertionError("scientific data loading"))
    monkeypatch.setattr("q2mm.benchmarks.systems.load_system", loader)
    runner = Mock(side_effect=AssertionError("matrix execution"))
    monkeypatch.setattr(cli, "run_profiles", runner)
    output = tmp_path / "never-created"
    args = [
        "matrix",
        "--system",
        "ch3f",
        "--forms",
        "harmonic",
        "--presets",
        "scipy-lbfgsb,scipy-lbfgsb-jax",
        "--output",
        str(output),
    ]
    if selection == "default":
        monkeypatch.setattr("q2mm.backends.registry.available_mm_backends", lambda: ["jax", descriptor.name])
    else:
        args += [
            "--backends",
            json.dumps({"key": descriptor.name, "options": {"configured_path": "explicit-location"}}),
        ]
    if selection == "inspection":
        args += ["--show-config"]
    assert main(args) == 2
    assert "The JAX objective requires backend='jax'" in capsys.readouterr().err
    runner.assert_not_called()
    factory.assert_not_called()
    loader.assert_not_called()
    backend.prepare.assert_not_called()
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_coherent_matrix_persists_all_created_candidates(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
    tmp_path: Path,
) -> None:
    from q2mm.benchmarks.artifacts import load_candidates

    descriptor, backend, factory = configured_registry
    assert (
        main(
            [
                "matrix",
                "--system",
                "ch3f",
                "--forms",
                "harmonic",
                "--backends",
                json.dumps({"key": descriptor.name, "options": {"configured_path": "explicit-location"}}),
                "--presets",
                "scipy-lbfgsb,scipy-nm",
                "--output",
                str(tmp_path),
            ]
        )
        == 0
    )
    records = load_candidates(tmp_path)
    assert len(records) == 2
    assert {record.status.value for record in records} == {"skipped"}
    assert {record.record["profile"]["recipe_id"] for record in records} == {"scipy-lbfgsb", "scipy-nm"}
    assert all(record.record["profile"]["objective"]["executor"] == "python" for record in records)
    assert factory.call_count == 2
    assert all(call.kwargs == {"configured_path": "explicit-location"} for call in factory.call_args_list)
    backend.prepare.assert_not_called()
    assert not (tmp_path / "accepted").exists() and not (tmp_path / "forcefields").exists()


@pytest.mark.parametrize(
    ("options", "expected_status", "expected_code", "reason"),
    [
        ({"configured_path": "explicit-location"}, "skipped", 0, "does not support functional form"),
        (
            {
                "configured_path": "explicit-location",
                "config": {"threads": 2, "modes": ["fast", {"enabled": False, "limit": 0, "fallback": None}]},
            },
            "skipped",
            0,
            "does not support functional form",
        ),
        ({}, "skipped", 0, "configured dependency missing"),
        ({"configured_path": "invalid"}, "error", 1, "configured factory failed"),
        ({"configured_path": "wrong-role"}, "error", 1, "runtime role"),
    ],
)
def test_actual_argv_constructs_configured_backend_despite_failed_generic_probe(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
    options: dict[str, JsonValue],
    expected_status: str,
    expected_code: int,
    reason: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from q2mm.benchmarks.artifacts import load_candidates

    descriptor, backend, factory = configured_registry
    loader = Mock(side_effect=AssertionError("scientific data loading"))
    monkeypatch.setattr("q2mm.benchmarks.systems.load_system", loader)
    assert descriptor.is_available()[0] is False
    assert (
        main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend.key",
                descriptor.name,
                "--backend.options",
                json.dumps(options),
                "--objective",
                '{"executor":"python","gradient_mode":"none"}',
                "--output",
                str(tmp_path),
                "--no-analyze",
            ]
        )
        == expected_code
    )
    factory.assert_called_once_with(**options)
    candidate = load_candidates(tmp_path)[0]
    assert candidate.status.value == expected_status
    assert reason in candidate.reason
    assert json.loads(candidate.path.read_text(encoding="utf-8"))["profile"]["backend"]["options"] == options
    assert json.loads(json.dumps(factory.call_args.kwargs)) == options
    loader.assert_not_called()
    backend.prepare.assert_not_called()


def test_configured_factory_is_not_called_for_inspection_or_list(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
) -> None:
    descriptor, _backend, factory = configured_registry
    assert main(["list"]) == 0
    assert main(["single", "--help"]) == 0
    assert (
        main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend.key",
                descriptor.name,
                "--backend.options",
                '{"configured_path":"explicit-location"}',
                "--objective",
                '{"executor":"python","gradient_mode":"none"}',
                "--show-config",
            ]
        )
        == 0
    )
    factory.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        ["matrix", "--system", "ch3f", "--presets", "bogus"],
        ["matrix", "--system", "ch3f", "--presets", "[]"],
        ["matrix", "--system", "ch3f", "--functional-form", "harmonic"],
    ],
)
def test_invalid_selector_cardinality_or_policy(argv: list[str]) -> None:
    assert main(argv) == 2


def test_single_selector_remains_scalar_on_repeated_flags() -> None:
    command = _parse(["single", "--system", "ch3f", "--system", "ch3f-sn2"])
    assert command.system == "ch3f-sn2"


def test_failed_device_probe_is_not_success(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules,
        "openmm",
        SimpleNamespace(
            Platform=SimpleNamespace(getNumPlatforms=Mock(side_effect=RuntimeError("platform unavailable"))),
        ),
    )
    monkeypatch.setitem(sys.modules, "jax", SimpleNamespace(devices=lambda: []))
    assert main(["preflight"]) == 1
    assert "platform probe failed" in capsys.readouterr().out


def test_execution_interruptions_are_not_caught_as_parser_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "run_profiles", Mock(side_effect=SystemExit(7)))
    with pytest.raises(SystemExit) as caught:
        main(["single", "--system", "ch3f"])
    assert caught.value.code == 7


def test_rejected_candidate_is_persisted_with_failure_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.benchmarks import runner
    from q2mm.benchmarks.acceptance import CandidateStatus
    from q2mm.benchmarks.artifacts import load_candidates
    from test.test_benchmarks_runner import _candidate

    rejected = _candidate(CandidateStatus.REJECTED)
    monkeypatch.setattr(runner, "run_profile", Mock(return_value=rejected))
    assert main(["single", "--system", "ch3f", "--output", str(tmp_path)]) == 1
    assert load_candidates(tmp_path)[0].status is CandidateStatus.REJECTED
    assert not (tmp_path / "accepted").exists() and not (tmp_path / "forcefields").exists()


@pytest.mark.parametrize("command", ["single", "batch", "matrix"])
def test_show_config_ignores_ambient_sources_and_writes_nothing(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("N_EVALS", "999")
    monkeypatch.setenv("OPTIMIZER", '{"kind":"optax","max_steps":999}')
    monkeypatch.setenv("SINGLE", '{"system":"ambient"}')
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text('BACKEND={"key":"ambient"}\n', encoding="utf-8")
    runner = Mock(side_effect=AssertionError("inspection executed"))
    monkeypatch.setattr(cli, "run_profiles", runner)
    args = [command, "--show-config", "--output", str(tmp_path / "absent")]
    if command != "batch":
        args += ["--system", "ch3f"]
    assert main(args) == 0
    output = json.loads(capsys.readouterr().out)
    if command == "matrix":
        assert output["backends"] == [] and output["forms"] == []
        assert output["policies"][0]["n_evals"] == 1
    else:
        assert output["n_evals"] == 1
        assert output["optimizer"]["kind"] == "scipy" and output["backend"]["key"] == "jax"
    assert not (tmp_path / "absent").exists()
    runner.assert_not_called()


@pytest.mark.parametrize("command", ["single", "batch", "matrix"])
def test_commands_dispatch_actual_profiles_and_propagate_rejection(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from q2mm.benchmarks.acceptance import CandidateStatus
    from test.test_benchmarks_runner import _candidate

    outcome = RunOutcome(candidates=(_candidate(CandidateStatus.REJECTED),))
    runner = Mock(return_value=outcome)
    monkeypatch.setattr(cli, "run_profiles", runner)
    args = [command, "--system", "ch3f", "--output", str(tmp_path), "--no-promote"]
    if command == "matrix":
        args += ["--backends", "plugin", "--presets", "scipy-lbfgsb", "--forms", "harmonic"]
    else:
        args += [
            "--backend.key",
            "plugin",
            "--optimizer.maxiter",
            "0",
            "--objective",
            '{"executor":"python","gradient_mode":"none"}',
        ]
    assert main(args) == 1
    profiles = runner.call_args.args[0]
    assert len(profiles) == 1 and isinstance(profiles[0], RunProfile)
    assert profiles[0].backend.key == "plugin"
    assert runner.call_args.kwargs["promote"] is False


def test_skipped_error_and_old_candidate_records_persist_and_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from q2mm.backends.contracts import BackendUnavailableError
    from q2mm.benchmarks import profiles
    from q2mm.benchmarks.artifacts import load_candidates

    assert (
        main(
            [
                "single",
                "--system",
                "ch3f",
                "--backend.key",
                "does-not-exist",
                "--output",
                str(tmp_path),
                "--objective",
                '{"executor":"python","gradient_mode":"none"}',
            ]
        )
        == 1
    )
    loaded = load_candidates(tmp_path)
    assert len(loaded) == 1 and loaded[0].status.value == "error"
    assert loaded[0].record["profile"]["version"] == "benchmark-profile-v2"
    unavailable = Mock(side_effect=BackendUnavailableError("synthetic missing dependency"))
    monkeypatch.setattr(profiles, "_classify_backend", unavailable)
    assert main(["single", "--system", "ch3f", "--output", str(tmp_path)]) == 0
    assert {c.status.value for c in load_candidates(tmp_path)} == {"error", "skipped"}
    assert not (tmp_path / "accepted").exists() and not (tmp_path / "forcefields").exists()
    old = {
        "candidate_id": "old-id",
        "status": "rejected",
        "reason": "no progress",
        "profile": {"optimizer": "scipy-lbfgsb-jax", "maxiter": 0},
        "profile_fingerprint": "old-hash",
    }
    (tmp_path / "candidates" / "old.json").write_text(json.dumps(old), encoding="utf-8")
    capsys.readouterr()
    assert main(["load", str(tmp_path)]) == 0
    assert "old-id" in capsys.readouterr().out
    assert next(c for c in load_candidates(tmp_path) if c.candidate_id == "old-id").record == old
    assert main(["load", str(tmp_path / "absent")]) == 1


def test_console_process_returns_execution_failure(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "q2mm.benchmarks.cli",
            "single",
            "--system",
            "ch3f",
            "--backend.key",
            "does-not-exist",
            "--objective",
            '{"executor":"python","gradient_mode":"none"}',
            "--output",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert len(list((tmp_path / "candidates").glob("*.json"))) == 1


@pytest.mark.jax
@pytest.mark.integration
@pytest.mark.parametrize(
    "objective_json", [None, '{"executor":"python","gradient_mode":"finite_difference","fd_step":0.02}']
)
def test_bounded_real_cpu_run_persists_result_without_acceptance_claim(
    tmp_path: Path, objective_json: str | None
) -> None:
    from q2mm.benchmarks.artifacts import load_candidates

    args = [
        "single",
        "--system",
        "ch3f",
        "--functional-form",
        "harmonic",
        "--optimizer.maxiter",
        "0",
        "--n-evals",
        "0",
        "--output",
        str(tmp_path),
        "--no-analyze",
        "--no-promote",
    ]
    if objective_json is not None:
        args += ["--objective", objective_json]
    rc = main(args)
    loaded = load_candidates(tmp_path)
    assert len(loaded) == 1
    assert loaded[0].status.value in {"accepted", "rejected"}, loaded[0].reason
    assert loaded[0].record["optimization_result"] is not None
    if objective_json is not None:
        assert loaded[0].record["optimization_result"]["fd_step"] == 0.02
        assert loaded[0].record["resolved"]["fd_step"] == 0.02
    assert loaded[0].record["optimization_result"]["n_iterations"] <= 1
    assert loaded[0].record["optimization_result"]["regularization"] == 0.0
    assert loaded[0].record["optimization_result"]["n_evals"] == 0
    assert loaded[0].record["optimization_result"]["initial_samples"] == ()
    assert loaded[0].record["optimization_result"]["final_samples"] == ()
    assert rc == (0 if loaded[0].status.value == "accepted" else 1)
    assert not (tmp_path / "accepted").exists()
