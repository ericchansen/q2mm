"""Staged semantic persistence with rollback for application outputs."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from q2mm._canonical import canonical_json
from q2mm._result_serialization import result_payload
from q2mm.models.forcefield import ForceField, FunctionalForm

from .models import (
    OptimizationRun,
    OutputExistsError,
    OutputFormatError,
    PersistenceError,
    ResolvedExecutionConfiguration,
    SavedOutput,
)

MANIFEST_SUFFIX = ".manifest.json"
_INTERNAL_ARTIFACT_NAME = re.compile(r"\..*\.q2mm-(?:reservation|.*-[0-9a-f]{32}\.tmp)", re.IGNORECASE | re.DOTALL)
logger = logging.getLogger(__name__)

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


def _validate_force_field_format(force_field: ForceField, format_name: str) -> None:
    required_form = _REQUIRED_FORMS[format_name]
    if force_field.functional_form is not required_form:
        raise OutputFormatError(
            f"{format_name!r} requires functional form {required_form.value!r}; "
            f"force field uses {force_field.functional_form.value!r}."
        )
    if force_field.nonbonded_excluded_atom_types and format_name != "mm3_fld":
        raise OutputFormatError(f"{format_name!r} cannot represent nonbonded_excluded_atom_types; use MM3 .fld output.")


@contextmanager
def _persistence_errors(target: Path) -> Iterator[None]:
    """Share the existing write-error policy without changing interruption handling."""
    try:
        yield
    except (OutputFormatError, OutputExistsError):
        raise
    except Exception as exc:
        raise PersistenceError(f"Could not save {target}: {exc}") from exc


def _write_staged_force_field(force_field: ForceField, path: Path, format_name: str) -> Path:
    """Serialize caller-owned staging content without treating it as a public output."""
    with _persistence_errors(path):
        _validate_force_field_format(force_field, format_name)
        return _serializer(format_name)(force_field, path)


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
        "result": result_payload(run.result),
    }


def _temp_sibling(path: Path, label: str) -> Path:
    return path.with_name(f".{path.name}.q2mm-{label}-{uuid4().hex}.tmp")


def _cleanup_files(paths: Iterable[Path], *, phase: str) -> None:
    """Remove only caller-owned paths, reporting cleanup without rollback."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("%s failed for %s: %s", phase, path, exc)
        except BaseException:
            logger.error("%s interrupted at %s; cleanup does not roll back installation", phase, path)
            raise


def _write_manifest(path: Path, run: OptimizationRun, format_name: str) -> None:
    blob = canonical_json(_manifest_payload(run, format_name), strict=True, screen_secrets=True)
    path.write_bytes((blob + "\n").encode("ascii"))


def _require_absent_manifest(target: Path) -> None:
    sidecar = Path(f"{target}{MANIFEST_SUFFIX}")
    if sidecar.exists() or sidecar.is_symlink():
        raise OutputExistsError(
            f"Refusing bare force-field save beside existing manifest {sidecar}; "
            "use a different output path or save an OptimizationRun to replace the pair."
        )


def _require_user_output_path(target: Path, *, force_field: bool = False) -> None:
    """Keep data out of transaction artifacts and force fields out of manifest paths."""
    names = [target.name]
    if target.is_symlink() or target.exists():
        try:
            names.append(target.resolve(strict=False).name)
        except (OSError, RuntimeError) as exc:
            raise OutputExistsError(f"Cannot resolve output path or alias: {target}") from exc
    if os.name == "nt":
        names += [name.partition(":")[0] for name in names]
    windows_alias = os.name == "nt" and any(name != name.rstrip(" .") for name in names)
    names = [name.rstrip(" .") for name in names]
    if any(_INTERNAL_ARTIFACT_NAME.fullmatch(name) for name in names):
        raise OutputExistsError(f"Output path uses Q2MM's reserved transaction-artifact namespace: {target}")
    if force_field and any(name.casefold().endswith(MANIFEST_SUFFIX) for name in names):
        raise OutputExistsError(f"Force-field output path uses Q2MM's reserved manifest namespace: {target}")
    if os.name == "nt" and ":" in target.name:
        raise OutputExistsError(f"Windows output filenames must not use alternate-data-stream syntax: {target}")
    if windows_alias:
        raise OutputExistsError(f"Windows output filenames must not end in a dot or space: {target}")
    if target.is_symlink():
        raise OutputExistsError(f"Output path must not be a filename-symlink alias: {target}")


@contextmanager
def _reserve_outputs(
    targets: Iterable[Path],
    *,
    bare_targets: Iterable[Path] = (),
    force_field_targets: Iterable[Path] = (),
) -> Iterator[None]:
    """Exclude cooperating processes for these outputs through rollback/cleanup.

    Claims are exclusive-created siblings, not the outputs or manifests.
    Abandoned claims fail closed; they are never automatically retired.
    Force-field roles are separate from their installable manifest targets.
    """
    bare_targets = tuple(bare_targets)
    force_fields = set(force_field_targets) | set(bare_targets)
    protected = set(targets) | force_fields | {Path(f"{target}{MANIFEST_SUFFIX}") for target in bare_targets}
    for target in protected:
        _require_user_output_path(target, force_field=target in force_fields)
    owned: list[Path] = []
    try:
        for target in sorted(protected):
            claim = target.with_name(f".{target.name}.q2mm-reservation")
            try:
                with claim.open("xb"):
                    owned.append(claim)
            except FileExistsError:
                raise OutputExistsError(f"Another save has reserved output {target}: {claim}") from None
        for target in bare_targets:
            _require_absent_manifest(target)
        yield
    finally:
        _cleanup_files(reversed(owned), phase="Transaction reservation cleanup")


def _replace_transaction(
    staged: list[tuple[Path, Path]],
    *,
    overwrite: bool,
    bare_targets: Iterable[Path] = (),
    force_field_targets: Iterable[Path] = (),
) -> None:
    with _reserve_outputs(
        (target for _temporary, target in staged), bare_targets=bare_targets, force_field_targets=force_field_targets
    ):
        _install_staged(staged, overwrite=overwrite)


def _install_staged(
    staged: list[tuple[Path, Path]],
    *,
    overwrite: bool,
) -> None:
    backups: dict[Path, Path] = {}
    reservations: list[Path] = []
    attempted: list[Path] = []
    phase = "snapshot" if overwrite else "reservation"
    try:
        if overwrite:
            for _temporary, target in staged:
                if target.exists():
                    backup = _temp_sibling(target, "backup")
                    backups[target] = backup
                    os.replace(target, backup)
        else:
            for _temporary, target in staged:
                try:
                    descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    raise OutputExistsError(f"Refusing to overwrite existing output: {target}") from None
                else:
                    reservations.append(target)
                    os.close(descriptor)
        phase = "installation"
        for temporary, target in staged:
            # Record intent first so an interrupt after replace still rolls back.
            attempted.append(target)
            os.replace(temporary, target)
    except BaseException as exc:
        logger.error("Save failed during %s; attempting rollback: %s", phase, exc)
        _cleanup_files(
            (target for target in reversed(dict.fromkeys(reservations + attempted)) if target not in backups),
            phase="Save not committed; rollback removal",
        )
        for target, backup in reversed(backups.items()):
            try:
                if backup.exists():
                    os.replace(backup, target)
            except OSError as recovery_error:
                logger.error(
                    "Save not committed; rollback failed for %s; recovery backup %s retained: %s",
                    target,
                    backup,
                    recovery_error,
                )
        raise
    else:
        _cleanup_files(backups.values(), phase="Save committed; backup cleanup")


def save(
    value: OptimizationRun | ForceField,
    path: str | Path,
    *,
    format: str | None = None,
    overwrite: bool = False,
) -> SavedOutput:
    """Save a force field and, for a run, its deterministic manifest.

    The manifest path is ``<force-field-path>.manifest.json``. Bare force
    fields intentionally produce no manifest and are rejected if that sidecar
    already exists, even with ``overwrite=True`` or a missing force-field file.
    Catchable installation failures trigger rollback; failed recovery retains
    backups and is logged. Cleanup errors after installation are logged as
    committed, without rollback. User interruptions propagate. Sequential
    replacements are not crash-atomic. Per-output filesystem reservations
    exclude concurrent cooperating saves and benchmark promotions through
    installation, rollback, and cleanup; contention raises OutputExistsError.
    An abandoned reservation fails closed rather than guessing ownership.
    Internal reservation/temporary names and aliases are not valid data-output
    paths, including when a format is supplied explicitly.
    The case-insensitive ``.manifest.json`` suffix and its normalized aliases
    are reserved for metadata, never a primary force-field output, even for runs.
    Windows filenames ending in dots or spaces are rejected, not redirected,
    so filesystem aliases cannot acquire different output/sidecar reservations.
    Windows alternate-data-stream syntax is not a supported output filename.
    Existing or dangling filename symlinks are also rejected; directory symlinks
    remain supported because their children share the same filesystem claims.
    """
    if not isinstance(value, (OptimizationRun, ForceField)):
        raise PersistenceError("save accepts an OptimizationRun or ForceField.")
    target = Path(path)
    if not target.name:
        raise PersistenceError("Output path must name a file.")
    if not target.parent.exists() or not target.parent.is_dir():
        raise PersistenceError(f"Output directory does not exist: {target.parent}")
    _require_user_output_path(target, force_field=True)
    selected_format = _resolve_format(target, format)
    force_field = value.final_force_field if isinstance(value, OptimizationRun) else value
    _validate_force_field_format(force_field, selected_format)
    sidecar = Path(f"{target}{MANIFEST_SUFFIX}")
    if not isinstance(value, OptimizationRun):
        _require_absent_manifest(target)
    manifest_target = sidecar if isinstance(value, OptimizationRun) else None
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
        with _persistence_errors(target):
            _serializer(selected_format)(force_field, ff_temporary)
            if manifest_temporary is not None:
                assert isinstance(value, OptimizationRun)
                _write_manifest(manifest_temporary, value, selected_format)
            staged = [(ff_temporary, target)]
            if manifest_temporary is not None and manifest_target is not None:
                staged.append((manifest_temporary, manifest_target))
            _replace_transaction(
                staged,
                overwrite=overwrite,
                bare_targets=(target,) if manifest_target is None else (),
                force_field_targets=(target,),
            )
    finally:
        _cleanup_files(temporaries, phase="Save staging cleanup")
    return SavedOutput(path=target, format=selected_format, manifest_path=manifest_target)


__all__ = ["MANIFEST_SUFFIX", "save"]
