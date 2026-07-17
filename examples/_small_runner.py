"""Shared runner for installed-data CH3F and CH3F-SN2 examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

import q2mm
from q2mm.io import load_fchk_molecule
from q2mm.io.xyz import load_xyz
from q2mm.models.hessian import HessianProvenance, HessianUnits


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


@dataclass(frozen=True)
class SmallExample:
    """Scientific choices that differ between the two installed-data examples."""

    key: str
    stationary_point: str
    geometry_name: str
    hessian_name: str
    charge: int
    bond_tolerance: float


def _installed_input_root() -> Path:
    from q2mm.resources import sn2_reference_dir

    return sn2_reference_dir()


def _load_packaged_molecule(config: SmallExample, input_root: Path) -> Any:
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
            source="q2mm-installed-sn2-reference",
            path=str(hessian.resolve()),
        ),
    )


def run_small(
    config: SmallExample,
    *,
    output_root: Path,
    input_root: Path | None = None,
    fchk: Path | None = None,
    stationary_point: str | None = None,
    functional_form: str = "harmonic",
    backend: str = "jax",
    bounded_ci: bool = False,
) -> dict[str, Any]:
    """Prepare, evaluate, optimize, and save one fresh-force-field problem."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
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
        molecule = _load_packaged_molecule(config, root)
        selected_stationary_point = config.stationary_point
        input_status = {"kind": "installed_q2mm_reference", "resource_id": "q2mm.sn2-reference.b3lyp-6-31+g(d)"}

    problem = q2mm.prepare(
        molecule,
        stationary_point=selected_stationary_point,
        functional_form=functional_form,
    )
    if bounded_ci:
        selected_backend: Any = _SUPPORT.BoundedEchoBackend()
        executor = "python"
        optimizer = _SUPPORT.BoundedExampleOptimizer()
        optimize_options = {
            "recipe": "explicit",
            "optimizer": optimizer,
            "workflow": "single-stage",
            "executor": "python",
            "n_evals": 0,
        }
    else:
        selected_backend = backend
        executor = "auto"
        optimizer = None
        optimize_options = {"recipe": "recommended"}

    initial = q2mm.evaluate(problem, backend=selected_backend, executor=executor)
    run = q2mm.optimize(problem, backend=selected_backend, **optimize_options)
    if bounded_ci and (optimizer is None or not optimizer.entered):
        raise RuntimeError("Bounded optimizer was not entered.")
    final_problem = _SUPPORT.with_final_force_field(problem, run.result.final_params)
    final = q2mm.evaluate(final_problem, backend=selected_backend, executor=executor)
    saved = q2mm.save(run, output_root / f"{config.key}.frcmod")

    preparation = problem.preparation_provenance
    return {
        "schema": "q2mm.example-result",
        "schema_version": 1,
        "example": config.key,
        "bounded_ci": bounded_ci,
        "input": input_status,
        "choices": {
            "stationary_point": selected_stationary_point,
            "functional_form": functional_form,
            "backend": run.configuration.backend.key,
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
            "convergence_claim": False if bounded_ci else run.result.success,
        },
        "saved": {
            "force_field": str(saved.force_field_path.resolve()),
            "manifest": None if saved.manifest_path is None else str(saved.manifest_path.resolve()),
        },
    }


def main_for(config: SmallExample) -> int:
    """Run one small example CLI and print one JSON document."""
    parser = argparse.ArgumentParser(description=f"Run the {config.key} q2mm example.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--fchk", type=Path)
    parser.add_argument("--stationary-point", choices=("ground_state", "transition_state"))
    parser.add_argument("--functional-form", choices=("harmonic",), default="harmonic")
    parser.add_argument("--backend", default="jax")
    parser.add_argument("--bounded-ci", action="store_true")
    args = parser.parse_args()
    try:
        result = run_small(
            config,
            output_root=args.output_root,
            input_root=args.input_root,
            fchk=args.fchk,
            stationary_point=args.stationary_point,
            functional_form=args.functional_form,
            backend=args.backend,
            bounded_ci=args.bounded_ci,
        )
    except Exception as exc:
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
