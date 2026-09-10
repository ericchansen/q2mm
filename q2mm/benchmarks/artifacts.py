"""Benchmark record storage and accepted-artifact installation.

The coordinator decides when to persist or promote. This owner preserves the
benchmark's copy-snapshot rollback contract while reusing the application's
serializer, temporary-path, and cleanup helpers.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from q2mm.benchmarks.acceptance import CandidateStatus
from q2mm.benchmarks.records import CandidateResult, LoadedCandidate, sanitize_for_json

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField

logger = logging.getLogger("q2mm.benchmarks.runner")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write *payload* to *path* as strict, sorted-key JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(sanitize_for_json(payload), fh, indent=2, allow_nan=False, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON candidate record written by :func:`write_json`."""
    with Path(path).open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


def _candidate_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "candidates" / f"{candidate_id}.json"


def persist_candidate(output_dir: Path, candidate: CandidateResult, provenance: Mapping[str, Any]) -> Path:
    """Write *candidate* to the stable ``candidates/`` location (all statuses).

    Persists the complete canonical result projection for accepted and
    rejected candidates alike; the collision-free candidate ID names the file.
    """
    path = _candidate_path(output_dir, candidate.candidate_id)
    write_json(path, {"provenance": dict(provenance), **candidate.record()})
    return path


def _ff_extension(ff: ForceField) -> str:
    from q2mm.models.forcefield import FunctionalForm

    if ff.functional_form is FunctionalForm.MM3:
        return ".fld"
    if ff.functional_form is FunctionalForm.HARMONIC:
        return ".frcmod"
    raise ValueError(f"no force-field serializer for functional form {ff.functional_form!r}.")


def _serialize_ff(ff: ForceField, path: Path) -> None:
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.application.persistence import save

    if ff.functional_form is FunctionalForm.MM3:
        save(ff, path, format="mm3_fld")
    elif ff.functional_form is FunctionalForm.HARMONIC:
        save(ff, path, format="amber_frcmod")
    else:
        raise ValueError(f"no force-field serializer for functional form {ff.functional_form!r}.")


def _opposite_ext(ext: str) -> str:
    return ".fld" if ext == ".frcmod" else ".frcmod"


def promote_candidate(output_dir: Path, candidate: CandidateResult, provenance: Mapping[str, Any]) -> dict[str, Path]:
    """Promote an accepted candidate with rollback on installation failure.

    Refuses any non-accepted candidate.  The accepted result JSON and the
    optimized force field are serialised to temporary siblings first (a
    serialisation failure changes nothing).  Pre-existing canonical artifacts
    are snapshotted, then installed with ``os.replace``. Catchable failures
    attempt to restore the old pair; failed recovery retains backups and is
    logged without replacing the original exception. After installation,
    cleanup of backups and the stale opposite-form force field only warns on
    OSError and never rolls back the committed pair. User interruptions
    propagate. Sequential replacements are not crash-atomic.
    """
    import shutil

    from q2mm.application.persistence import _cleanup_files, _temp_sibling

    if not candidate.accepted:
        raise ValueError(
            f"refusing to promote non-accepted candidate {candidate.candidate_id!r} ({candidate.status.value})."
        )
    accepted_dir = output_dir / "accepted"
    ff_dir = output_dir / "forcefields"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    result_path = accepted_dir / f"{candidate.candidate_id}.json"
    payload = sanitize_for_json({"provenance": dict(provenance), **candidate.record()})

    ff = candidate.final_force_field
    ext = _ff_extension(ff) if ff is not None else None
    ff_path = ff_dir / f"{candidate.candidate_id}{ext}" if ext is not None else None
    tmp_ff = _temp_sibling(ff_path, "output") if ff_path is not None else None
    tmp_json = _temp_sibling(result_path, "output")
    targets: list[tuple[Path, Path]] = [(tmp_json, result_path)]
    if ff is not None and tmp_ff is not None and ff_path is not None:
        targets.append((tmp_ff, ff_path))

    backups: dict[Path, Path] = {}
    attempted: list[Path] = []
    committed = False
    phase = "serialization"
    try:
        with tmp_json.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False, sort_keys=True)
            fh.write("\n")
        if ff is not None and tmp_ff is not None:
            ff_dir.mkdir(parents=True, exist_ok=True)
            _serialize_ff(ff, tmp_ff)
        phase = "snapshot"
        for _tmp, target in targets:
            if target.exists():
                backup = _temp_sibling(target, "backup")
                # A failed copy can leave a partial file that also needs cleanup.
                backups[target] = backup
                shutil.copy2(target, backup)
        phase = "installation"
        for tmp, target in targets:
            attempted.append(target)
            os.replace(tmp, target)
        committed = True
    except BaseException as exc:
        logger.error("Promotion failed during %s; attempting rollback: %s", phase, exc)
        for target in reversed(attempted):
            recovery_backup = backups.pop(target, None)
            try:
                if recovery_backup is not None:
                    os.replace(recovery_backup, target)
                else:
                    target.unlink(missing_ok=True)
            except OSError as recovery_error:
                logger.error(
                    "Promotion not committed; rollback failed for %s (recovery backup: %s): %s",
                    target,
                    recovery_backup,
                    recovery_error,
                )
        _cleanup_files(backups.values(), phase="Promotion not committed; snapshot cleanup")
        raise
    else:
        cleanup = list(backups.values())
        if ff_path is not None and ext is not None:
            cleanup.insert(0, ff_dir / f"{candidate.candidate_id}{_opposite_ext(ext)}")
        _cleanup_files(cleanup, phase="Promotion committed; cleanup")
    finally:
        _cleanup_files(
            (tmp for tmp, _target in targets),
            phase=f"Promotion {'committed' if committed else 'not committed'}; staging cleanup",
        )

    promoted: dict[str, Path] = {"result": result_path}
    if ff_path is not None:
        promoted["force_field"] = ff_path
    return promoted


def load_candidates(directory: Path) -> list[LoadedCandidate]:
    """Load every persisted candidate record under *directory* (all statuses)."""
    directory = Path(directory)
    search = directory / "candidates" if (directory / "candidates").is_dir() else directory
    loaded: list[LoadedCandidate] = []
    for path in sorted(search.glob("*.json")):
        try:
            record = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not load candidate %s: %s", path.name, exc)
            continue
        status_value = str(record.get("status", "error"))
        try:
            status = CandidateStatus(status_value)
        except ValueError:
            status = CandidateStatus.ERROR
        loaded.append(
            LoadedCandidate(
                candidate_id=str(record.get("candidate_id", path.stem)),
                status=status,
                reason=str(record.get("reason", "")),
                path=path,
                record=record,
            )
        )
    return loaded
