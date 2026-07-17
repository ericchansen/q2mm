"""Atomic semantic persistence for application outputs."""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from q2mm._canonical import canonical_json
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.results import CandidateRecord, OptimizationResult, StageRecord

from .models import (
    OptimizationRun,
    OutputExistsError,
    OutputFormatError,
    PersistenceError,
    ResolvedExecutionConfiguration,
    SavedOutput,
)

MANIFEST_SUFFIX = ".manifest.json"

_EXTENSIONS = {
    ".fld": "mm3_fld",
    ".frcmod": "amber_frcmod",
    ".prm": "tinker_prm",
}
_FORMAT_ALIASES = {
    "fld": "mm3_fld",
    "mm3": "mm3_fld",
    "mm3_fld": "mm3_fld",
    "frcmod": "amber_frcmod",
    "amber": "amber_frcmod",
    "amber_frcmod": "amber_frcmod",
    "prm": "tinker_prm",
    "tinker": "tinker_prm",
    "tinker_prm": "tinker_prm",
}
_REQUIRED_FORMS = {
    "mm3_fld": FunctionalForm.MM3,
    "amber_frcmod": FunctionalForm.HARMONIC,
    "tinker_prm": FunctionalForm.MM3,
}


def _resolve_format(path: Path, requested: str | None) -> str:
    inferred = _EXTENSIONS.get(path.suffix.lower())
    if requested is None:
        if inferred is None:
            raise OutputFormatError(
                f"Cannot infer force-field format from {path.suffix or 'no extension'!r}; "
                "use .fld, .frcmod, or .prm, or pass format explicitly."
            )
        return inferred
    try:
        selected = _FORMAT_ALIASES[requested.strip().lower().lstrip(".")]
    except KeyError:
        raise OutputFormatError(
            f"Unknown format {requested!r}; expected MM3 .fld, AMBER .frcmod, or Tinker .prm."
        ) from None
    if inferred is not None and inferred != selected:
        raise OutputFormatError(f"Requested format {selected!r} conflicts with target extension {path.suffix!r}.")
    return selected


def _serializer(format_name: str) -> Callable[[ForceField, Path], Path]:
    if format_name == "mm3_fld":
        from q2mm.io.mm3 import save_mm3_fld

        return save_mm3_fld
    if format_name == "amber_frcmod":
        from q2mm.io.amber import save_amber_frcmod

        return save_amber_frcmod
    from q2mm.io.tinker import save_tinker_prm

    return save_tinker_prm


def _stage_payload(stage: StageRecord) -> dict[str, Any]:
    return {
        "name": stage.name,
        "n_params": stage.n_params,
        "layout_fingerprint": stage.layout_fingerprint,
        "initial_score": stage.initial_score,
        "final_score": stage.final_score,
        "n_iterations": stage.n_iterations,
        "n_evaluations": stage.n_evaluations,
        "converged": stage.converged,
        "message": stage.message,
        "gradient_mode": stage.gradient_mode,
        "fd_step": stage.fd_step,
        "elapsed_s": stage.elapsed_s,
        "locked_param_indices": list(stage.locked_param_indices),
        "notes": dict(stage.notes),
    }


def _candidate_payload(candidate: CandidateRecord) -> dict[str, Any]:
    return {
        "index": candidate.index,
        "status": candidate.status,
        "n_params": candidate.n_params,
        "layout_fingerprint": candidate.layout_fingerprint,
        "initial_params": candidate.initial_params.tolist(),
        "final_params": candidate.final_params.tolist(),
        "initial_score": candidate.initial_score,
        "final_score": candidate.final_score,
        "message": candidate.message,
        "seed": candidate.seed,
    }


def _result_payload(result: OptimizationResult) -> dict[str, Any]:
    return {
        "success": result.success,
        "message": result.message,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "n_iterations": result.n_iterations,
        "n_evaluations": result.n_evaluations,
        "n_params": result.n_params,
        "layout_fingerprint": result.layout_fingerprint,
        "initial_params": result.initial_params.tolist(),
        "final_params": result.final_params.tolist(),
        "history": list(result.history),
        "method": result.method,
        "gradient_mode": result.gradient_mode,
        "fd_step": result.fd_step,
        "initial_samples": list(result.initial_samples),
        "final_samples": list(result.final_samples),
        "category_metrics": {key: dict(value) for key, value in result.category_metrics.items()},
        "candidates": [_candidate_payload(candidate) for candidate in result.candidates],
        "stages": [_stage_payload(stage) for stage in result.stages],
    }


def _configuration_payload(configuration: ResolvedExecutionConfiguration) -> dict[str, Any]:
    backend = configuration.backend
    optimizer = configuration.optimizer
    workflow = configuration.workflow
    executor = configuration.executor
    return {
        "schema_version": configuration.schema_version,
        "recipe_id": configuration.recipe_id,
        "backend": {
            "schema_version": backend.schema_version,
            "key": backend.key,
            "name": backend.name,
            "role": backend.role,
            "version": backend.version,
            "capabilities": list(backend.capabilities),
            "functional_forms": list(backend.functional_forms),
            "details": dict(backend.details),
        },
        "optimizer": {
            "schema_version": optimizer.schema_version,
            "key": optimizer.key,
            "label": optimizer.label,
            "method": optimizer.method,
            "settings": dict(optimizer.settings),
            "expected_result_gradient_mode": optimizer.expected_result_gradient_mode,
        },
        "workflow": {
            "schema_version": workflow.schema_version,
            "key": workflow.key,
            "settings": dict(workflow.settings),
        },
        "executor": {
            "schema_version": executor.schema_version,
            "kind": executor.kind,
            "gradient_mode": executor.gradient_mode,
            "fd_step": executor.fd_step,
        },
        "overrides": list(configuration.overrides),
        "regularization": configuration.regularization,
        "n_evals": configuration.n_evals,
        "ratio_tol": configuration.ratio_tol,
    }


def _manifest_payload(run: OptimizationRun, format_name: str) -> dict[str, Any]:
    return {
        "schema": "q2mm.optimization-run-manifest",
        "schema_version": 1,
        "force_field_format": format_name,
        "problem_fingerprint": run.problem_fingerprint,
        "layout_fingerprint": run.layout_fingerprint,
        "input_fingerprints": dict(run.input_fingerprints),
        "active_indices": list(run.active_indices),
        "baseline": run.baseline.tolist(),
        "configuration": _configuration_payload(run.configuration),
        "provenance": dict(run.provenance),
        "result": _result_payload(run.result),
    }


def _temp_sibling(path: Path, label: str) -> Path:
    return path.with_name(f".{path.name}.q2mm-{label}-{uuid4().hex}.tmp")


def _write_manifest(path: Path, run: OptimizationRun, format_name: str) -> None:
    blob = canonical_json(_manifest_payload(run, format_name), strict=True, screen_secrets=True)
    path.write_bytes((blob + "\n").encode("ascii"))


def _replace_transaction(
    staged: list[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    backups: list[tuple[Path, Path]] = []
    reservations: list[Path] = []
    installed: list[Path] = []
    try:
        if overwrite:
            for _temporary, target in staged:
                if target.exists():
                    backup = _temp_sibling(target, "backup")
                    os.replace(target, backup)
                    backups.append((backup, target))
        else:
            for _temporary, target in staged:
                try:
                    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    raise OutputExistsError(f"Refusing to overwrite existing output: {target}") from None
                else:
                    os.close(descriptor)
                    reservations.append(target)
        for temporary, target in staged:
            os.replace(temporary, target)
            installed.append(target)
            with suppress(ValueError):
                reservations.remove(target)
    except Exception:
        for target in reversed(installed):
            with suppress(OSError):
                target.unlink(missing_ok=True)
        for reservation in reversed(reservations):
            with suppress(OSError):
                reservation.unlink(missing_ok=True)
        for backup, target in reversed(backups):
            if backup.exists():
                os.replace(backup, target)
        raise
    else:
        for backup, _target in backups:
            backup.unlink(missing_ok=True)


def save(
    value: OptimizationRun | ForceField,
    path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> SavedOutput:
    """Atomically save a force field and, for a run, its deterministic manifest.

    The manifest path is ``<force-field-path>.manifest.json``. Bare force
    fields intentionally produce no manifest.
    """
    if not isinstance(value, (OptimizationRun, ForceField)):
        raise PersistenceError("save accepts an OptimizationRun or ForceField.")
    target = Path(path)
    if not target.name:
        raise PersistenceError("Output path must name a file.")
    if not target.parent.exists() or not target.parent.is_dir():
        raise PersistenceError(f"Output directory does not exist: {target.parent}")
    selected_format = _resolve_format(target, format)
    force_field = value.final_force_field if isinstance(value, OptimizationRun) else value
    required_form = _REQUIRED_FORMS[selected_format]
    if force_field.functional_form is not required_form:
        raise OutputFormatError(
            f"{selected_format!r} requires functional form {required_form.value!r}; "
            f"force field uses {force_field.functional_form.value!r}."
        )
    if force_field.nonbonded_excluded_atom_types and selected_format != "mm3_fld":
        raise OutputFormatError(
            f"{selected_format!r} cannot represent nonbonded_excluded_atom_types; use MM3 .fld output."
        )
    manifest_target = Path(f"{target}{MANIFEST_SUFFIX}") if isinstance(value, OptimizationRun) else None
    invalid_targets = [
        candidate
        for candidate in (target, manifest_target)
        if candidate is not None and candidate.exists() and not candidate.is_file()
    ]
    if invalid_targets:
        raise PersistenceError(f"Output target is not a regular file: {invalid_targets[0]}")
    collisions = [candidate for candidate in (target, manifest_target) if candidate is not None and candidate.exists()]
    if collisions and not overwrite:
        raise OutputExistsError(f"Refusing to overwrite existing output(s): {', '.join(map(str, collisions))}")

    ff_temporary = _temp_sibling(target, "output")
    manifest_temporary = _temp_sibling(manifest_target, "manifest") if manifest_target is not None else None
    temporaries = [item for item in (ff_temporary, manifest_temporary) if item is not None]
    try:
        _serializer(selected_format)(force_field, ff_temporary)
        if manifest_temporary is not None:
            assert isinstance(value, OptimizationRun)
            _write_manifest(manifest_temporary, value, selected_format)
        staged = [(ff_temporary, target)]
        if manifest_temporary is not None and manifest_target is not None:
            staged.append((manifest_temporary, manifest_target))
        _replace_transaction(staged, overwrite=overwrite)
    except (OutputFormatError, OutputExistsError):
        raise
    except Exception as exc:
        raise PersistenceError(f"Could not save {target}: {exc}") from exc
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)
    return SavedOutput(path=target, format=selected_format, manifest_path=manifest_target)


__all__ = ["MANIFEST_SUFFIX", "save"]
