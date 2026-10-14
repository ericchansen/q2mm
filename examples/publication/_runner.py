"""Shared executable runner for source-backed publication case studies."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import q2mm
from q2mm.benchmarks.publications import (
    REPOSITORY_OBJECTIVE_PROFILE,
    PublicationProfileBlockedError,
    publication_record,
    publication_records,
    publication_success_spec,
)
from q2mm.benchmarks.systems import load_system
from q2mm import ExternalDataRoots
from q2mm.backends.registry import load_backend
from q2mm.objectives.plan import ObjectivePlan
from q2mm.settings import (
    BackendSettings,
    GroundStateOptimizationSettings,
    ObjectiveSettings,
    OptimizationSettings,
    SingleStageSettings,
    TransitionStateOptimizationSettings,
)


def _load_support() -> ModuleType:
    name = "_q2mm_example_support"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).parents[1] / "_support.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load example support: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_SUPPORT = _load_support()


@dataclass(frozen=True)
class PublicationExample:
    """Only the source-backed choices that legitimately differ by system."""

    key: str
    expected_status: str
    required_roots: tuple[str, ...]
    default_starting_point: str = "qfuerza"
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE
    fc_fraction: float | None = 0.20
    eq_fraction: float | None = 0.05
    note: str = ""


def _validate_roots(config: PublicationExample, roots: ExternalDataRoots) -> None:
    labels = {
        "supporting_info": (roots.supporting_info, "directory"),
        "mm3_base": (roots.mm3_base, "file"),
        "rh_enamide": (roots.rh_enamide, "directory"),
    }
    missing = []
    for name in config.required_roots:
        value, kind = labels[name]
        valid = value is not None and (value.is_file() if kind == "file" else value.is_dir())
        if not valid:
            missing.append(f"{name}=<{kind}>")
    if missing:
        rendered = ", ".join(missing)
        raise _SUPPORT.ExampleConfigurationError(
            f"{config.key} requires explicit external root(s): {rendered}. "
            "Use --supporting-info, --mm3-base, and/or --rh-enamide as indicated."
        )


def _blocked_rows(system: str) -> list[dict[str, Any]]:
    return [
        {
            "objective_profile": record.objective_profile.identifier,
            "starting_point": record.starting_point,
            "status": record.status.value,
            "blockers": list(record.blockers),
            "case_order": list(record.authoritative_case_ids),
        }
        for record in publication_records(system=system)
        if not record.provisionable
    ]


def run_publication(
    config: PublicationExample,
    *,
    output_root: Path,
    supporting_info: Path | None = None,
    mm3_base: Path | None = None,
    rh_enamide: Path | None = None,
    starting_point: str | None = None,
    backend: str = "jax",
    executor: Literal["python", "jax"] = "jax",
    bounded_ci: bool = False,
) -> dict[str, Any]:
    """Run one real publication problem through root evaluate/optimize/save."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    roots = ExternalDataRoots(
        supporting_info=None if supporting_info is None else Path(supporting_info),
        mm3_base=None if mm3_base is None else Path(mm3_base),
        rh_enamide=None if rh_enamide is None else Path(rh_enamide),
    )
    _validate_roots(config, roots)
    selected_start = starting_point or config.default_starting_point
    try:
        record = publication_record(config.key, config.objective_profile, selected_start)
    except PublicationProfileBlockedError as exc:
        raise _SUPPORT.ExampleConfigurationError(str(exc)) from exc
    if record.status.value != config.expected_status:
        raise RuntimeError(
            f"{config.key} status changed from {config.expected_status!r} to {record.status.value!r}; "
            "review the source-backed example before running it."
        )
    optimization_proof = publication_success_spec(
        config.key,
        config.objective_profile,
        selected_start,
    )

    case = load_system(
        config.key,
        data_roots=roots,
        starting_point=selected_start,
        objective_profile=config.objective_profile,
        functional_form="mm3",
    )
    problem = case.problem
    if tuple(record.authoritative_case_ids) != tuple(source.source_id or source.case_id for source in problem.cases):
        raise RuntimeError(f"{config.key} loader did not preserve authoritative case membership and order.")

    scientific_bounds = {
        "fc_fraction": config.fc_fraction,
        "eq_fraction": config.eq_fraction,
    }
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
        optimizer_options = {key: value for key, value in scientific_bounds.items() if value is not None}
        settings_type = (
            GroundStateOptimizationSettings if config.key == "ferrocene" else TransitionStateOptimizationSettings
        )
        settings = settings_type(
            backend=BackendSettings(key=backend),
            optimizer=optimizer_options,
            objective=ObjectiveSettings(executor=executor),
        )
        selected_backend = load_backend(settings.backend.key, **settings.backend.factory_options())
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
    if bounded_ci and not optimizer.entered:
        raise RuntimeError("Bounded optimizer was not entered.")
    final = q2mm.evaluate(evaluator, result.final_params)
    saved = q2mm.save(run, output_root / f"{config.key}-{selected_start}.fld")

    preparation = problem.preparation_provenance
    publication = problem.publication_metadata
    if publication is None:
        raise RuntimeError(f"{config.key} problem has no publication provenance.")
    publication_payload = publication.to_dict()
    configuration = run.configuration
    return {
        "schema": "q2mm.publication-example-result",
        "schema_version": 1,
        "system": config.key,
        "bounded_ci": bounded_ci,
        "citation": publication_payload["governing_sources"],
        "source_status": {
            "status": publication.status.value,
            "source_artifacts": publication_payload["source_artifacts"],
            "tracked_source_policy": (
                "tracked in the source repository; excluded from wheel and sdist; "
                "redistribution/licensing not established"
                if config.key == "rh-enamide"
                else "caller-supplied external scientific data; not copied into q2mm artifacts"
            ),
        },
        "objective_profile": publication.objective_profile.identifier,
        "case_count": len(problem.cases),
        "case_order": [source.source_id or source.case_id for source in problem.cases],
        "functional_form": problem.starting_force_field.functional_form.value,
        "stationary_point": publication.stationary_point,
        "force_field_composition": {
            "blocks": list(publication.force_field_blocks),
            "starting_point": selected_start,
        },
        "parameter_counts": _SUPPORT.parameter_counts(problem),
        "qfuerza": None
        if preparation is None
        else {
            "initialize_source": preparation.initialize_source,
            "settings": _SUPPORT.json_safe(preparation.qfuerza_settings),
            "audit": _SUPPORT.json_safe(preparation.parameter_counts),
        },
        "initial": _SUPPORT.evaluation_payload(initial),
        "execution": {
            "recipe": configuration.recipe_id,
            "backend": configuration.backend.key,
            "optimizer": configuration.optimizer.key,
            "optimizer_settings": _SUPPORT.json_safe(configuration.optimizer.settings),
            "workflow": configuration.workflow.key,
            "workflow_settings": _SUPPORT.json_safe(configuration.workflow.settings),
            "executor": configuration.executor.kind,
            "gradient_mode": configuration.executor.gradient_mode,
            "scientific_default_bounds": scientific_bounds,
            "resolved_bounds": (
                {"mode": "bounded_ci_no_parameter_update"}
                if bounded_ci
                else {
                    key: value
                    for key, value in configuration.optimizer.settings["parameters"].items()
                    if key in {"fc_fraction", "eq_fraction", "use_bounds"}
                }
            ),
            "bounded_ci_limit": "one optimizer-owned objective call; separate endpoint reporting"
            if bounded_ci
            else None,
        },
        "final": _SUPPORT.evaluation_payload(final),
        "optimization": {
            "success": run.result.success,
            "message": run.result.message,
            "iterations": run.result.n_iterations,
            "evaluations": run.result.n_evaluations,
            "convergence_claim": False if bounded_ci else run.result.success,
            "proof": optimization_proof.to_dict(),
        },
        "saved": {
            "force_field": str(saved.force_field_path.resolve()),
            "manifest": None if saved.manifest_path is None else str(saved.manifest_path.resolve()),
        },
        "objective_targets": publication_payload["targets"],
        "blockers": list(publication.blockers),
        "blocked_rows": _blocked_rows(config.key),
        "note": config.note,
    }


def main_for(config: PublicationExample) -> int:
    """Run one publication CLI and print a structured result or error."""
    parser = argparse.ArgumentParser(description=f"Run the {config.key} publication example.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--supporting-info", type=Path)
    parser.add_argument("--mm3-base", type=Path)
    parser.add_argument("--rh-enamide", type=Path)
    parser.add_argument("--starting-point", choices=("qfuerza", "published"))
    parser.add_argument("--backend", default="jax")
    parser.add_argument(
        "--executor",
        choices=("jax", "python"),
        default="jax",
        help="Objective executor; explicitly choose python with non-JAX backends.",
    )
    parser.add_argument("--bounded-ci", action="store_true")
    args = parser.parse_args()
    try:
        result = run_publication(
            config,
            output_root=args.output_root,
            supporting_info=args.supporting_info,
            mm3_base=args.mm3_base,
            rh_enamide=args.rh_enamide,
            starting_point=args.starting_point,
            backend=args.backend,
            executor=args.executor,
            bounded_ci=args.bounded_ci,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema": "q2mm.example-error",
                    "system": config.key,
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
