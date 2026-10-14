"""Shared runner for installed-data CH3F and CH3F-SN2 examples."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, replace
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import numpy as np
from pydantic import Field, ValidationError
from pydantic_settings import CliApp, SettingsConfigDict, SettingsError

import q2mm
from q2mm.backends.contracts import Backend
from q2mm.io import load_fchk_molecule
from q2mm.io.xyz import load_xyz
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.backends.registry import load_backend
from q2mm.objectives.plan import ObjectivePlan
from q2mm.settings import (
    BackendSettings,
    GroundStateOptimizationSettings,
    Iterations,
    ObjectiveSettings,
    OptimizationSettings,
    PreparationSettings,
    ScipySettings,
    Settings,
    SingleStageSettings,
    TransitionStateOptimizationSettings,
)


def _load_support() -> ModuleType:
    name = "_q2mm_example_support"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("_support.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load example support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SUPPORT = _load_support()

_BUDGET = {"maxiter": 2, "maxls": 5}


class SmallExampleOptions(Settings):
    """Source-example inputs and output paths; numerical controls stay shared."""

    model_config = SettingsConfigDict(
        cli_kebab_case=True,
        cli_implicit_flags=True,
        cli_enforce_required=True,
        cli_parse_none_str="null",
        cli_use_class_docs_for_groups=True,
    )
    output_root: Path = Field(strict=False)
    input_root: Path | None = Field(default=None, strict=False)
    fchk: Path | None = Field(default=None, strict=False)
    stationary_point: Literal["ground_state", "transition_state"] | None = None
    functional_form: Literal["harmonic"] = "harmonic"
    bounded_ci: bool = False
    show_config: bool = False

    def model_post_init(self, context: Any) -> None:
        """Do not silently ignore one of two supplied input sources."""
        if self.fchk is not None and self.input_root is not None:
            raise ValueError("Choose --fchk or --input-root, not both.")


class GroundStateExampleCommand(GroundStateOptimizationSettings, SmallExampleOptions):
    """Budget-limited ground-state example using the shared native settings."""

    optimizer: ScipySettings = ScipySettings.model_validate(
        GroundStateOptimizationSettings.model_validate({"optimizer": _BUDGET}).optimizer
    )
    n_evals: Iterations = 0
    workflow: SingleStageSettings = SingleStageSettings()


class TransitionStateExampleCommand(TransitionStateOptimizationSettings, SmallExampleOptions):
    """Budget-limited transition-state example using the shared native settings."""

    optimizer: ScipySettings = ScipySettings.model_validate(
        TransitionStateOptimizationSettings.model_validate({"optimizer": _BUDGET}).optimizer
    )
    n_evals: Iterations = 0
    workflow: SingleStageSettings = SingleStageSettings()


@dataclass(frozen=True)
class SmallExample:
    """Scientific choices that differ between the two installed-data examples."""

    key: str
    stationary_point: Literal["ground_state", "transition_state"]
    geometry_name: str
    hessian_name: str
    charge: int
    bond_tolerance: float


def _command_type(config: SmallExample) -> type[GroundStateExampleCommand | TransitionStateExampleCommand]:
    return GroundStateExampleCommand if config.stationary_point == "ground_state" else TransitionStateExampleCommand


def _installed_input_root() -> Path:
    from q2mm.resources import sn2_reference_dir

    return sn2_reference_dir()


def _require_cpu_request(settings: OptimizationSettings) -> None:
    if settings.backend.key == "openmm":
        platform = settings.backend.options.get("platform_name")
        if platform is None:
            platform = os.environ.get("OPENMM_DEFAULT_PLATFORM", "").strip()
        if platform != "CPU":
            raise _SUPPORT.ExampleConfigurationError(
                "This native example is CPU-only. Explicitly set OpenMM backend.options "
                'to {"platform_name":"CPU"} or OPENMM_DEFAULT_PLATFORM=CPU. '
                "JAX_PLATFORMS does not control OpenMM."
            )
    elif os.environ.get("JAX_PLATFORMS") != "cpu":
        raise _SUPPORT.ExampleConfigurationError(
            "This native example is CPU-only. Set JAX_PLATFORMS=cpu before starting Python."
        )


def _require_cpu_backend(backend: Backend) -> str:
    provenance = backend.info.provenance
    platform = None if provenance is None else provenance.details.get("platform")
    name = platform.get("name", platform.get("backend")) if isinstance(platform, Mapping) else None
    if not isinstance(name, str) or name.lower() != "cpu":
        raise _SUPPORT.ExampleConfigurationError(
            f"This native example requires a verified CPU backend before evaluation; selected platform is {name!r}."
        )
    return name


def _load_packaged_molecule(config: SmallExample, input_root: Path, *, source: str) -> Any:
    geometry = input_root / config.geometry_name
    hessian = input_root / config.hessian_name
    missing = [str(path) for path in (geometry, hessian) if not path.is_file()]
    if missing:
        raise _SUPPORT.ExampleConfigurationError(
            f"{config.key} input root is incomplete; missing {missing}. "
            "Pass --input-root pointing at a complete q2mm SN2 reference directory."
        )
    molecule = load_xyz(
        geometry,
        charge=config.charge,
        name=config.key,
        bond_tolerance=config.bond_tolerance,
    )
    return molecule.with_hessian(
        np.load(hessian),
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source=source,
            path=str(hessian.resolve()),
        ),
    )


def run_small(
    config: SmallExample,
    *,
    output_root: Path,
    input_root: Path | None = None,
    fchk: Path | None = None,
    stationary_point: Literal["ground_state", "transition_state"] | None = None,
    functional_form: Literal["harmonic"] = "harmonic",
    settings: OptimizationSettings | None = None,
    bounded_ci: bool = False,
) -> dict[str, Any]:
    """Prepare, evaluate, run a small optimizer budget, and explicitly save."""
    output_root = Path(output_root)
    if fchk is not None and input_root is not None:
        raise _SUPPORT.ExampleConfigurationError("Choose fchk or input_root, not both.")
    if not bounded_ci:
        if settings is None:
            settings = _command_type(config)(output_root=output_root)
        _require_cpu_request(settings)
    selected_platform = None
    if stationary_point is not None and stationary_point != config.stationary_point:
        raise _SUPPORT.ExampleConfigurationError(
            f"{config.key} has fixed {config.stationary_point!r} semantics; "
            f"received conflicting stationary_point={stationary_point!r}."
        )
    if fchk is not None:
        fchk = Path(fchk)
        if not fchk.is_file():
            raise _SUPPORT.ExampleConfigurationError(f"FCHK input is not a file: {fchk}")
        molecule = load_fchk_molecule(fchk, bond_tolerance=config.bond_tolerance)
        selected_stationary_point = config.stationary_point
        input_status = {"kind": "caller_fchk", "name": fchk.name}
    else:
        root = _installed_input_root() if input_root is None else Path(input_root)
        if not root.is_dir():
            raise _SUPPORT.ExampleConfigurationError(f"Input root is not a directory: {root}")
        molecule = _load_packaged_molecule(
            config,
            root,
            source="q2mm-installed-sn2-reference" if input_root is None else "caller-sn2-reference-root",
        )
        selected_stationary_point = config.stationary_point
        input_status = (
            {"kind": "installed_q2mm_reference", "resource_id": "q2mm.sn2-reference.b3lyp-6-31+g(d)"}
            if input_root is None
            else {"kind": "caller_reference_root", "geometry": config.geometry_name, "hessian": config.hessian_name}
        )

    problem = q2mm.prepare(
        molecule,
        settings=PreparationSettings(
            stationary_point=selected_stationary_point,
            functional_form=functional_form,
        ),
    )
    if bounded_ci:
        selected_backend: Any = _SUPPORT.BoundedEchoBackend()
        optimizer = _SUPPORT.BoundedExampleOptimizer()
        settings = OptimizationSettings(
            backend=BackendSettings(key=selected_backend.info.provenance.backend),
            workflow=SingleStageSettings(),
            objective=ObjectiveSettings(),
            n_evals=0,
        )
    else:
        if settings is None:
            settings = _command_type(config)(output_root=output_root)
        selected_backend = load_backend(settings.backend.key, **settings.backend.factory_options())
        selected_platform = _require_cpu_backend(selected_backend)
        optimizer = q2mm.build_optimizer(settings.optimizer)
    workflow = q2mm.build_workflow(settings.workflow)
    configuration = q2mm.capture_execution_configuration(
        settings, backend=selected_backend, optimizer=optimizer, workflow=workflow
    )
    make_evaluator = q2mm.make_evaluator_factory(
        selected_backend, problem.starting_force_field, settings=settings.objective
    )
    plan = ObjectivePlan.from_problem(problem, regularization=settings.objective.regularization)
    evaluator = make_evaluator(plan)
    initial = q2mm.evaluate(evaluator, problem.active_space.baseline)
    result = q2mm.optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=make_evaluator,
        regularization=settings.objective.regularization,
        n_evals=settings.n_evals,
    )
    run = q2mm.OptimizationRun.from_result(problem, result, configuration=configuration)
    execution_mode = "software_path_only" if bounded_ci else "budget_limited_native"
    run = replace(
        run,
        provenance={
            **run.provenance,
            "example": {
                "key": config.key,
                "execution_mode": execution_mode,
                "platform": selected_platform,
                "convergence_claim": False,
            },
        },
    )
    if bounded_ci and not optimizer.entered:
        raise RuntimeError("Bounded optimizer was not entered.")
    final = q2mm.evaluate(evaluator, result.final_params)
    output_root.mkdir(parents=True, exist_ok=True)
    saved = q2mm.save(run, output_root / f"{config.key}.frcmod")

    preparation = problem.preparation_provenance
    return {
        "schema": "q2mm.example-result",
        "schema_version": 1,
        "example": config.key,
        "bounded_ci": bounded_ci,
        "execution_mode": execution_mode,
        "input": input_status,
        "choices": {
            "stationary_point": selected_stationary_point,
            "functional_form": functional_form,
            "backend": run.configuration.backend.key,
            "platform": selected_platform,
            "optimizer": run.configuration.optimizer.key,
            "workflow": run.configuration.workflow.key,
            "executor": run.configuration.executor.kind,
        },
        "case_count": len(problem.cases),
        "case_order": list(problem.case_ids),
        "parameter_counts": _SUPPORT.parameter_counts(problem),
        "qfuerza": None
        if preparation is None
        else {
            "settings": _SUPPORT.json_safe(preparation.qfuerza_settings),
            "audit": _SUPPORT.json_safe(preparation.parameter_counts),
        },
        "initial": _SUPPORT.evaluation_payload(initial),
        "final": _SUPPORT.evaluation_payload(final),
        "optimization": {
            "success": run.result.success,
            "message": run.result.message,
            "iterations": run.result.n_iterations,
            "evaluations": run.result.n_evaluations,
            "convergence_claim": False,
            "parameter_update_norm": float(np.linalg.norm(result.final_params - result.initial_params)),
        },
        "saved": {
            "force_field": str(saved.force_field_path.resolve()),
            "manifest": None if saved.manifest_path is None else str(saved.manifest_path.resolve()),
        },
    }


def main_for(config: SmallExample) -> int:
    """Run one small example CLI and print one JSON document."""
    try:
        args = CliApp.run(_command_type(config), cli_exit_on_error=False)
        if args.show_config:
            print(args.model_dump_json(indent=2))
            return 0
        result = run_small(
            config,
            output_root=args.output_root,
            input_root=args.input_root,
            fchk=args.fchk,
            stationary_point=args.stationary_point,
            functional_form=args.functional_form,
            settings=args,
            bounded_ci=args.bounded_ci,
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 2
    except (SettingsError, ValidationError, ValueError, OSError, RuntimeError, ImportError) as exc:
        print(
            json.dumps(
                {
                    "schema": "q2mm.example-error",
                    "example": config.key,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
