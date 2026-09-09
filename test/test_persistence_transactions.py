"""Fault-injection coverage for application paired-file installation."""

from pathlib import Path

import pytest

from q2mm.application import persistence
from q2mm.application.models import PersistenceError
from test.test_application import _problem, _run


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("install_number", [1, 2])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("after_replace", [False, True])
def test_save_install_failure_restores_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    overwrite: bool,
    install_number: int,
    failure_type: type[BaseException],
    after_replace: bool,
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    manifest = Path(f"{target}{persistence.MANIFEST_SUFFIX}")
    old = {target: b"old force field", manifest: b"old manifest"} if overwrite else {}
    for path, content in old.items():
        path.write_bytes(content)
    real_replace = persistence.os.replace
    failure = failure_type("install failed")
    failed_target = (target, manifest)[install_number - 1]

    def fail_install(source: Path, destination: Path) -> None:
        if destination == failed_target and ".q2mm-backup-" not in source.name:
            if after_replace:
                real_replace(source, destination)
            raise failure
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, "replace", fail_install)
        with pytest.raises(PersistenceError if failure_type is OSError else KeyboardInterrupt) as raised:
            persistence.save(run, target, overwrite=overwrite)
    assert (raised.value.__cause__ if failure_type is OSError else raised.value) is failure
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == old
    assert "failed during installation" in caplog.text
    assert "Save committed;" not in caplog.text
    saved = persistence.save(run, target, overwrite=overwrite)
    assert saved.manifest_path == manifest
    assert set(tmp_path.iterdir()) == {target, manifest}


@pytest.mark.parametrize("snapshot_number", [1, 2])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
@pytest.mark.parametrize("after_replace", [False, True])
def test_save_snapshot_failure_restores_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    snapshot_number: int,
    failure_type: type[BaseException],
    after_replace: bool,
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    manifest = Path(f"{target}{persistence.MANIFEST_SUFFIX}")
    old = {target: b"old force field", manifest: b"old manifest"}
    for path, content in old.items():
        path.write_bytes(content)
    real_replace = persistence.os.replace
    failure = failure_type("snapshot failed")
    failed_target = (target, manifest)[snapshot_number - 1]

    def fail_snapshot(source: Path, destination: Path) -> None:
        if source == failed_target and ".q2mm-backup-" in destination.name:
            if after_replace:
                real_replace(source, destination)
            raise failure
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, "replace", fail_snapshot)
        with pytest.raises(PersistenceError if failure_type is OSError else KeyboardInterrupt) as raised:
            persistence.save(run, target, overwrite=True)
    assert (raised.value.__cause__ if failure_type is OSError else raised.value) is failure
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == old
    assert "failed during snapshot" in caplog.text
    persistence.save(run, target, overwrite=True)
    assert set(tmp_path.iterdir()) == {target, manifest}


@pytest.mark.parametrize("backup_number", [1, 2])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_save_cleanup_failure_keeps_committed_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    backup_number: int,
    failure_type: type[BaseException],
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    saved = persistence.save(run, target)
    manifest = saved.manifest_path
    assert manifest is not None
    expected = {path: path.read_bytes() for path in (target, manifest)}
    old = {target: b"old force field", manifest: b"old manifest"}
    for path, content in old.items():
        path.write_bytes(content)
    real_unlink = Path.unlink
    failure = failure_type("backup cleanup failed")
    calls = 0

    def fail_cleanup(path: Path, missing_ok: bool = False) -> None:
        nonlocal calls
        if ".q2mm-backup-" in path.name:
            calls += 1
            if calls == backup_number:
                raise failure
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_cleanup)
        if failure_type is OSError:
            assert persistence.save(run, target, overwrite=True) == saved
        else:
            with pytest.raises(KeyboardInterrupt) as raised:
                persistence.save(run, target, overwrite=True)
            assert raised.value is failure
    assert {path: path.read_bytes() for path in expected} == expected
    recovery = {path: path.read_bytes() for path in tmp_path.iterdir() if path not in expected}
    assert len(recovery) == (1 if failure_type is OSError else 3 - backup_number)
    assert all(".q2mm-backup-" in path.name and data in old.values() for path, data in recovery.items())
    assert "Save committed; backup cleanup" in caplog.text
    assert "attempting rollback" not in caplog.text
    persistence.save(run, target, overwrite=True)
    assert {path: path.read_bytes() for path in recovery} == recovery


@pytest.mark.parametrize("restore_number", [1, 2])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_save_failed_recovery_retains_backup_and_original_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    restore_number: int,
    failure_type: type[BaseException],
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    manifest = Path(f"{target}{persistence.MANIFEST_SUFFIX}")
    old = {target: b"old force field", manifest: b"old manifest"}
    for path, content in old.items():
        path.write_bytes(content)
    real_replace = persistence.os.replace
    failure = failure_type("install failed")
    failed_target = (target, manifest)[restore_number - 1]

    def fail_install_and_recovery(source: Path, destination: Path) -> None:
        if destination == manifest and ".q2mm-manifest-" in source.name:
            raise failure
        if destination == failed_target and ".q2mm-backup-" in source.name:
            raise OSError("restore failed")
        real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, "replace", fail_install_and_recovery)
        with pytest.raises(PersistenceError if failure_type is OSError else KeyboardInterrupt) as raised:
            persistence.save(run, target, overwrite=True)
    assert (raised.value.__cause__ if failure_type is OSError else raised.value) is failure
    recovery = {path: path.read_bytes() for path in tmp_path.iterdir() if path not in old}
    assert len(recovery) == 1
    backup = next(iter(recovery))
    assert recovery[backup] == old[failed_target]
    assert str(backup) in caplog.text
    assert "not committed; rollback failed" in caplog.text
    for path in old:
        if path != failed_target:
            assert path.read_bytes() == old[path]
    persistence.save(run, target, overwrite=True)
    assert backup.read_bytes() == old[failed_target]


def test_save_staging_cleanup_does_not_mask_failure_or_block_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    real_unlink = Path.unlink
    failure = OSError("manifest serialization failed")

    def fail_manifest(*args: object) -> None:
        raise failure

    def fail_cleanup(path: Path, missing_ok: bool = False) -> None:
        if ".q2mm-output-" in path.name:
            raise OSError("staging cleanup failed")
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(persistence, "_write_manifest", fail_manifest)
        patch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(PersistenceError) as raised:
            persistence.save(run, target)
    assert raised.value.__cause__ is failure
    assert not target.exists()
    leftovers = {path: path.read_bytes() for path in tmp_path.iterdir()}
    assert len(leftovers) == 1
    assert "Save staging cleanup failed" in caplog.text
    persistence.save(run, target)
    assert {path: path.read_bytes() for path in leftovers} == leftovers


@pytest.mark.parametrize("reservation_number", [1, 2])
def test_save_reservation_close_failure_removes_owned_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reservation_number: int
) -> None:
    run = _run(_problem())
    target = tmp_path / "run.frcmod"
    real_close = persistence.os.close
    failure = OSError("reservation close failed")
    calls = 0

    def fail_close(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        real_close(descriptor)
        if calls == reservation_number:
            raise failure

    with monkeypatch.context() as patch:
        patch.setattr(persistence.os, "close", fail_close)
        with pytest.raises(PersistenceError) as raised:
            persistence.save(run, target)
    assert raised.value.__cause__ is failure
    assert list(tmp_path.iterdir()) == []
    persistence.save(run, target)
