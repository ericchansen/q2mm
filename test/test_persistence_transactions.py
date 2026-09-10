"""Fault-injection coverage for application paired-file installation."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.application import persistence
from q2mm.application.models import OutputExistsError, PersistenceError
from q2mm.models.forcefield import ForceField
from test.test_application import _problem, _run


def _competing_save(target: Path, *, paired: bool, cwd: Path | None = None, format_name: str | None = None) -> str:
    script = """
import sys
sys.path.insert(0, sys.argv[3])
from q2mm.application.persistence import save
from q2mm.application.models import OutputExistsError
from test.test_application import _problem, _run
problem = _problem()
value = _run(problem) if sys.argv[2] == "paired" else problem.starting_force_field
try:
    save(value, sys.argv[1], format=sys.argv[4] or None, overwrite=True)
except OutputExistsError:
    print("blocked")
else:
    print("installed")
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(target),
            "paired" if paired else "bare",
            str(Path(__file__).resolve().parents[1]),
            format_name or "",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        cwd=cwd,
    )
    return completed.stdout.strip()


@pytest.mark.skipif(os.name != "nt", reason="Windows normalizes trailing dots and spaces in filenames")
@pytest.mark.parametrize("suffix", [".", " ", ".. "])
@pytest.mark.parametrize("paired", [False, True])
def test_windows_filename_alias_cannot_bypass_live_pair_reservations(tmp_path: Path, suffix: str, paired: bool) -> None:
    target = tmp_path / "run.frcmod"
    saved = persistence.save(_run(_problem()), target)
    assert saved.manifest_path is not None
    alias = target.with_name(target.name + suffix)
    assert alias.samefile(target)
    with persistence._reserve_outputs([target, saved.manifest_path], force_field_targets=[target]):
        original = {path: path.read_bytes() for path in tmp_path.iterdir()}
        assert _competing_save(alias, paired=paired, format_name="amber_frcmod") == "blocked"
        assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original
    assert set(tmp_path.iterdir()) == {target, saved.manifest_path}


@pytest.mark.skipif(os.name != "nt", reason="Windows filename normalization is platform-specific")
@pytest.mark.parametrize("suffix", [".", " ", ".. "])
@pytest.mark.parametrize("paired", [False, True])
def test_windows_filename_alias_fails_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, paired: bool
) -> None:
    def unexpected_staging(*args: object, **kwargs: object) -> None:
        pytest.fail("Noncanonical Windows output name reached staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_staging)
    with pytest.raises(OutputExistsError, match="Windows.*filename"):
        problem = _problem()
        persistence.save(
            _run(problem) if paired else problem.starting_force_field,
            tmp_path / ("run.frcmod" + suffix),
            format="amber_frcmod",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows filename normalization is platform-specific")
@pytest.mark.parametrize("suffix", [".", " "])
def test_windows_filename_alias_fails_at_transaction_entry(tmp_path: Path, suffix: str) -> None:
    target = tmp_path / ("result.json" + suffix)
    with pytest.raises(OutputExistsError, match="Windows.*filename"), persistence._reserve_outputs([target]):
        pytest.fail("Noncanonical Windows output acquired a separate claim")
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX permits distinct trailing-dot and trailing-space files")
@pytest.mark.parametrize("suffix", [".", " "])
def test_posix_trailing_filename_characters_remain_distinct(tmp_path: Path, suffix: str) -> None:
    target = tmp_path / "run.frcmod"
    target.write_bytes(b"unchanged ordinary path")
    distinct = target.with_name(target.name + suffix)
    persistence.save(_problem().starting_force_field, distinct, format="amber_frcmod")
    assert distinct.is_file() and not distinct.samefile(target)
    assert target.read_bytes() == b"unchanged ordinary path"


@pytest.mark.parametrize("entry", ["bare", "paired", "transaction"])
@pytest.mark.parametrize("referent_exists", [False, True])
def test_filename_symlink_alias_rejects_before_staging_or_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, referent_exists: bool
) -> None:
    problem = _problem()
    target = tmp_path / "run.frcmod"
    if referent_exists:
        persistence.save(_run(problem), target)
    alias = tmp_path / "alias.frcmod"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows symlink privilege")
        raise
    link_target = alias.readlink()
    paths = set(tmp_path.iterdir())
    original = {path: path.read_bytes() for path in paths if not path.is_symlink()}

    def unexpected_staging(*args: object, **kwargs: object) -> None:
        pytest.fail("Filename-symlink output reached staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_staging)
    with pytest.raises(OutputExistsError, match="filename-symlink"):
        if entry == "transaction":
            with persistence._reserve_outputs([alias]):
                pytest.fail("Filename-symlink output acquired an independent reservation")
        else:
            persistence.save(
                _run(problem) if entry == "paired" else problem.starting_force_field,
                alias,
                format="amber_frcmod",
                overwrite=True,
            )
    assert set(tmp_path.iterdir()) == paths
    assert alias.is_symlink() and alias.readlink() == link_target
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("paired", [False, True])
def test_filename_symlink_competitor_cannot_split_a_live_output_alias(tmp_path: Path, paired: bool) -> None:
    target = tmp_path / "run.frcmod"
    saved = persistence.save(_run(_problem()), target)
    assert saved.manifest_path is not None
    alias = tmp_path / "alias.frcmod"
    try:
        alias.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows symlink privilege")
        raise
    with persistence._reserve_outputs([target, saved.manifest_path], force_field_targets=[target]):
        original = {path: path.read_bytes() for path in tmp_path.iterdir() if not path.is_symlink()}
        assert _competing_save(alias, paired=paired, format_name="amber_frcmod") == "blocked"
        assert alias.is_symlink() and alias.samefile(target)
        assert {path: path.read_bytes() for path in tmp_path.iterdir() if not path.is_symlink()} == original


@pytest.mark.skipif(os.name != "nt", reason="Alternate data streams are Windows-specific")
@pytest.mark.parametrize("suffix", [":payload", "::$DATA", ":payload:$DATA"])
@pytest.mark.parametrize("entry", ["bare", "paired", "transaction"])
def test_windows_ads_rejects_before_staging_or_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, entry: str
) -> None:
    target = tmp_path / ("run.frcmod" + suffix)

    def unexpected_staging(*args: object, **kwargs: object) -> None:
        pytest.fail("Windows stream output reached staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_staging)
    with pytest.raises(OutputExistsError, match="alternate-data-stream"):
        if entry == "transaction":
            with persistence._reserve_outputs([target]):
                pytest.fail("Windows stream output acquired a non-file claim")
        else:
            problem = _problem()
            persistence.save(
                _run(problem) if entry == "paired" else problem.starting_force_field,
                target,
                format="amber_frcmod",
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(os.name == "nt", reason="POSIX colons are filename characters, not stream selectors")
def test_posix_colon_filename_remains_supported(tmp_path: Path) -> None:
    target = tmp_path / "run.frcmod:payload"
    persistence.save(_problem().starting_force_field, target, format="amber_frcmod")
    assert target.is_file()


@pytest.mark.parametrize("length", [1, 2])
def test_filename_symlink_loop_is_a_typed_preflight_rejection(tmp_path: Path, length: int) -> None:
    target = tmp_path / "loop.frcmod"
    other = target if length == 1 else tmp_path / "other.frcmod"
    try:
        target.symlink_to(other)
        if length == 2:
            other.symlink_to(target)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows symlink privilege")
        raise
    links = {path: path.readlink() for path in tmp_path.iterdir()}
    with pytest.raises(OutputExistsError, match="resolve|symlink|alias"):
        persistence.save(_problem().starting_force_field, target, overwrite=True)
    assert {path: path.readlink() for path in tmp_path.iterdir()} == links


@pytest.mark.parametrize("error_type", [RuntimeError, OSError])
def test_output_alias_resolution_errors_keep_the_original_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[Exception]
) -> None:
    target = tmp_path / "run.frcmod"
    target.write_bytes(b"original field")
    failure = error_type("output alias cannot be resolved")
    original_resolve = Path.resolve

    def fail_target(path: Path, *args: object, **kwargs: object) -> Path:
        if path == target:
            raise failure
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fail_target)
    with pytest.raises(OutputExistsError, match="resolve") as caught:
        persistence.save(_problem().starting_force_field, target, overwrite=True)
    assert caught.value.__cause__ is failure
    assert target.read_bytes() == b"original field"
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("paired", [False, True])
def test_primary_force_field_cannot_replace_a_valid_run_manifest(tmp_path: Path, paired: bool) -> None:
    problem = _problem()
    run = _run(problem)
    saved = persistence.save(run, tmp_path / "run.frcmod")
    manifest = saved.manifest_path
    assert manifest is not None
    assert json.loads(manifest.read_text())["schema"] == "q2mm.optimization-run-manifest"
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}

    with pytest.raises(OutputExistsError, match="reserved.*manifest"):
        persistence.save(
            run if paired else problem.starting_force_field, manifest, format="amber_frcmod", overwrite=True
        )

    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original


@pytest.mark.parametrize(
    "name",
    ["run.frcmod.manifest.json", "run.frcmod.MANIFEST.JSON", "run.frcmod.manifest.json.", "run.frcmod.manifest.json "],
)
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_manifest_role_names_fail_before_primary_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, paired: bool, existing: bool
) -> None:
    target = tmp_path / name
    if existing:
        target.write_bytes(b"existing ownership metadata")
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    problem = _problem()

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Manifest-role primary output reached staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    monkeypatch.setattr(persistence, "_serializer", unexpected_write)
    with pytest.raises(OutputExistsError, match="reserved.*manifest"):
        persistence.save(
            _run(problem) if paired else problem.starting_force_field,
            target,
            format="amber_frcmod",
            overwrite=True,
        )
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("manifest_exists", [False, True])
def test_primary_manifest_alias_is_rejected_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paired: bool, manifest_exists: bool
) -> None:
    problem = _problem()
    field = tmp_path / "run.frcmod"
    persistence.save(_run(problem) if manifest_exists else problem.starting_force_field, field)
    manifest = Path(f"{field}{persistence.MANIFEST_SUFFIX}")
    alias = tmp_path / "alias.frcmod"
    try:
        alias.symlink_to(manifest)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows symlink privilege")
        raise
    link_target = alias.readlink()
    paths_before = set(tmp_path.iterdir())
    original = {path: path.read_bytes() for path in paths_before if not path.is_symlink()}

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Manifest alias reached primary staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    monkeypatch.setattr(persistence, "_serializer", unexpected_write)
    with pytest.raises(OutputExistsError, match="reserved.*manifest"):
        persistence.save(
            _run(problem) if paired else problem.starting_force_field,
            alias,
            format="amber_frcmod",
            overwrite=True,
        )
    assert set(tmp_path.iterdir()) == paths_before
    assert alias.is_symlink()
    assert alias.readlink() == link_target
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("paired", [False, True])
def test_primary_manifest_alias_is_rechecked_at_transaction_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paired: bool
) -> None:
    problem = _problem()
    run = _run(problem)
    saved = persistence.save(run, tmp_path / "run.frcmod")
    manifest = saved.manifest_path
    assert manifest is not None
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    alias = tmp_path / "late-alias.frcmod"
    serializer = persistence._serializer("amber_frcmod")
    introduced_links: list[Path] = []

    def introduce_alias_after_staging(force_field: ForceField, temporary: Path) -> Path:
        result = serializer(force_field, temporary)
        try:
            alias.symlink_to(manifest)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Creating symlinks requires Windows symlink privilege")
            raise
        introduced_links.append(alias.readlink())
        return result

    monkeypatch.setattr(persistence, "_serializer", lambda _format: introduce_alias_after_staging)
    with pytest.raises(OutputExistsError, match="reserved.*manifest"):
        persistence.save(run if paired else problem.starting_force_field, alias, format="amber_frcmod", overwrite=True)
    assert set(tmp_path.iterdir()) == set(original) | {alias}
    assert alias.is_symlink()
    assert introduced_links == [alias.readlink()]
    assert alias.samefile(manifest)
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.skipif(sys.platform != "win32", reason="Windows alternate data stream syntax")
@pytest.mark.parametrize("stream", ["::$DATA", ":metadata"])
@pytest.mark.parametrize("paired", [False, True])
def test_primary_manifest_stream_is_rejected_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str, paired: bool
) -> None:
    problem = _problem()
    target = tmp_path / f"run.frcmod.manifest.json{stream}"

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Manifest stream reached primary staging")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    with pytest.raises(OutputExistsError, match="reserved.*manifest"):
        persistence.save(
            _run(problem) if paired else problem.starting_force_field,
            target,
            format="amber_frcmod",
            overwrite=True,
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name", ["ordinary.json", "custom-output", "run.manifest-json", "run.manifest.json.frcmod"])
@pytest.mark.parametrize("paired", [False, True])
def test_ordinary_explicit_format_primary_names_and_paired_metadata_remain_supported(
    tmp_path: Path, name: str, paired: bool
) -> None:
    problem = _problem()
    target = tmp_path / name
    value = _run(problem) if paired else problem.starting_force_field
    saved = persistence.save(value, target, format="amber_frcmod")
    assert target.read_bytes()
    if paired:
        assert saved.manifest_path == Path(f"{target}{persistence.MANIFEST_SUFFIX}")
        assert json.loads(saved.manifest_path.read_text())["schema"] == "q2mm.optimization-run-manifest"
        assert set(tmp_path.iterdir()) == {target, saved.manifest_path}
    else:
        assert saved.manifest_path is None
        assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("claim_for", ["force-field", "manifest"])
def test_another_save_cannot_overwrite_an_owned_claim(tmp_path: Path, paired: bool, claim_for: str) -> None:
    target = tmp_path / "run.frcmod"
    manifest = Path(f"{target}{persistence.MANIFEST_SUFFIX}")
    protected = target if claim_for == "force-field" else manifest
    claim = protected.with_name(f".{protected.name}.q2mm-reservation")

    with persistence._reserve_outputs([target, manifest]):
        original = {path: path.read_bytes() for path in tmp_path.iterdir()}
        assert _competing_save(claim, paired=paired, format_name="amber_frcmod") == "blocked"
        assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original

    assert list(tmp_path.iterdir()) == []


_INTERNAL_OUTPUT_NAMES = [
    ".run.frcmod.q2mm-reservation",
    ".RUN.FRCMOD.Q2MM-RESERVATION",
    ".run.frcmod.q2mm-reservation.",
    *[f".run.frcmod.q2mm-{kind}-{'a' * 32}.tmp" for kind in ("output", "manifest", "backup")],
]


@pytest.mark.parametrize("name", _INTERNAL_OUTPUT_NAMES)
@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_internal_output_names_fail_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, paired: bool, existing: bool
) -> None:
    target = tmp_path / name
    if existing:
        target.write_bytes(b"original internal artifact")
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    problem = _problem()
    value = _run(problem) if paired else problem.starting_force_field

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Internal output name reached staging or serialization")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    monkeypatch.setattr(persistence, "_serializer", unexpected_write)
    with pytest.raises(OutputExistsError, match="reserved.*namespace"):
        persistence.save(value, target, format="amber_frcmod", overwrite=True)

    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original


@pytest.mark.parametrize("name", [_INTERNAL_OUTPUT_NAMES[0], _INTERNAL_OUTPUT_NAMES[-1]])
def test_transaction_helper_rejects_internal_output_names(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    with pytest.raises(OutputExistsError, match="reserved.*namespace"), persistence._reserve_outputs([target]):
        pytest.fail("Internal output name reached the transaction body")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows alternate data stream syntax")
@pytest.mark.parametrize("stream", ["::$DATA", ":metadata"])
def test_transaction_helper_rejects_streams_of_internal_artifacts(tmp_path: Path, stream: str) -> None:
    target = tmp_path / f".run.frcmod.q2mm-reservation{stream}"
    with pytest.raises(OutputExistsError, match="reserved.*namespace"), persistence._reserve_outputs([target]):
        pytest.fail("Internal output stream reached the transaction body")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("artifact_exists", [False, True])
def test_existing_alias_to_internal_artifact_is_not_an_output(tmp_path: Path, artifact_exists: bool) -> None:
    artifact = tmp_path / ".run.frcmod.q2mm-reservation"
    if artifact_exists:
        artifact.write_bytes(b"owned internal artifact")
    alias = tmp_path / "alias.frcmod"
    try:
        alias.symlink_to(artifact)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows symlink privilege")
        raise
    link_target = alias.readlink()

    with pytest.raises(OutputExistsError, match="reserved.*namespace"):
        persistence.save(_problem().starting_force_field, alias, overwrite=True)

    assert alias.is_symlink()
    assert alias.readlink() == link_target
    if artifact_exists:
        assert artifact.read_bytes() == b"owned internal artifact"
    else:
        assert not artifact.exists()


@pytest.mark.parametrize("name", [".ordinary.frcmod", "run.q2mm-reservation", ".run.q2mm-reservation.frcmod"])
def test_noninternal_output_names_remain_supported(tmp_path: Path, name: str) -> None:
    target = tmp_path / name
    saved = persistence.save(_problem().starting_force_field, target, format="amber_frcmod")
    assert saved.path == target
    assert target.read_bytes()
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("spelling", ["absolute", "relative", "parent-traversal", "directory-symlink"])
def test_reservations_are_shared_across_path_spellings(tmp_path: Path, paired: bool, spelling: str) -> None:
    directory = tmp_path / "real"
    directory.mkdir()
    target = directory / "run.frcmod"
    other_target = target
    if spelling == "relative":
        other_target = Path(target.name)
    elif spelling == "parent-traversal":
        (directory / "child").mkdir()
        other_target = Path("child") / ".." / target.name
    elif spelling == "directory-symlink":
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(directory, target_is_directory=True)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Creating directory symlinks requires Windows symlink privilege")
            raise
        other_target = alias / target.name

    with persistence._reserve_outputs([target], bare_targets=[target]):
        original = set(directory.iterdir())
        assert _competing_save(other_target, paired=paired, cwd=directory) == "blocked"
        assert set(directory.iterdir()) == original

    assert not target.exists()
    assert not Path(f"{target}{persistence.MANIFEST_SUFFIX}").exists()
    assert not list(directory.glob("*.q2mm-reservation"))


def test_bare_save_rejects_pair_installed_during_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()
    run = _run(problem)
    target = tmp_path / "run.frcmod"
    vector = run.result.final_params.copy()
    vector[0] += 1.0
    changed = problem.layout.replace(run.final_force_field, vector)
    serializer = persistence._serializer("amber_frcmod")
    paired_bytes: dict[Path, bytes] = {}

    def install_pair_then_serialize(force_field: ForceField, temporary: Path) -> Path:
        with monkeypatch.context() as patch:
            patch.setattr(persistence, "_serializer", lambda _format: serializer)
            persistence.save(run, target, overwrite=True)
        paired_bytes.update({path: path.read_bytes() for path in tmp_path.iterdir()})
        return serializer(force_field, temporary)

    monkeypatch.setattr(persistence, "_serializer", lambda _format: install_pair_then_serialize)
    with pytest.raises(OutputExistsError, match="manifest"):
        persistence.save(changed, target, overwrite=True)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == paired_bytes


@pytest.mark.parametrize("paired", [False, True])
def test_save_reserves_pair_across_processes_during_installation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, paired: bool
) -> None:
    problem = _problem()
    target = tmp_path / "run.frcmod"
    target.write_bytes(b"original bare field")
    real_replace = persistence.os.replace
    competitors = []

    def compete_before_install(source: Path, destination: Path) -> None:
        if destination == target and ".q2mm-output-" in source.name:
            competitors.append(_competing_save(target, paired=not paired))
        real_replace(source, destination)

    monkeypatch.setattr(persistence.os, "replace", compete_before_install)
    saved = persistence.save(_run(problem) if paired else problem.starting_force_field, target, overwrite=True)
    assert competitors == ["blocked"]
    expected = {target}
    if paired:
        assert saved.manifest_path is not None
        expected.add(saved.manifest_path)
    assert set(tmp_path.iterdir()) == expected


def test_reservation_conflict_releases_only_owned_claims(tmp_path: Path) -> None:
    first, second, independent = (tmp_path / name for name in ("a.frcmod", "b.frcmod", "c.frcmod"))
    with persistence._reserve_outputs([second]):
        foreign_claims = set(tmp_path.iterdir())
        with (
            pytest.raises(OutputExistsError, match="reserved output"),
            persistence._reserve_outputs([first, second]),
        ):
            pytest.fail("Overlapping reservations were admitted")
        assert set(tmp_path.iterdir()) == foreign_claims
        persistence.save(_problem().starting_force_field, independent)
        assert set(tmp_path.iterdir()) == foreign_claims | {independent}
    assert set(tmp_path.iterdir()) == {independent}


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_partial_reservation_failure_releases_claims(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_type: type[BaseException]
) -> None:
    first, second = (tmp_path / name for name in ("a.frcmod", "b.frcmod"))
    real_open = Path.open
    failure = failure_type("reservation interrupted")

    def fail_second_claim(path: Path, *args: object, **kwargs: object):  # noqa: ANN202
        if path.name == ".b.frcmod.q2mm-reservation":
            raise failure
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_claim)
    with pytest.raises(failure_type) as raised, persistence._reserve_outputs([first, second]):
        pytest.fail("Failed reservation reached installation")
    assert raised.value is failure
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_failed_reservation_release_is_logged_and_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure_type: type[BaseException]
) -> None:
    target = tmp_path / "bare.frcmod"
    claim = tmp_path / ".bare.frcmod.q2mm-reservation"
    force_field = _problem().starting_force_field
    real_unlink = Path.unlink
    failure = failure_type("reservation release failed")

    def fail_release(path: Path, missing_ok: bool = False) -> None:
        if path == claim:
            raise failure
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_release)
        if failure_type is OSError:
            assert persistence.save(force_field, target).path == target
        else:
            with pytest.raises(KeyboardInterrupt) as raised:
                persistence.save(force_field, target)
            assert raised.value is failure
    committed_bytes = target.read_bytes()
    assert set(tmp_path.iterdir()) == {target, claim}
    assert str(claim) in caplog.text
    assert "Transaction reservation cleanup" in caplog.text
    with pytest.raises(OutputExistsError, match="reserved output"):
        persistence.save(force_field, target, overwrite=True)
    assert target.read_bytes() == committed_bytes
    assert claim.exists()
    claim.unlink()
    persistence.save(force_field, target, overwrite=True)
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("orphan", [False, True])
def test_bare_save_rejects_existing_sidecar_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool, orphan: bool
) -> None:
    problem = _problem()
    run = _run(problem)
    target = tmp_path / "run.frcmod"
    saved = persistence.save(run, target)
    manifest = saved.manifest_path
    assert manifest is not None
    if orphan:
        target.unlink()
        manifest.write_bytes(b"untrusted sidecar: not JSON")
    original = {path: path.read_bytes() for path in tmp_path.iterdir()}
    vector = run.result.final_params.copy()
    vector[0] += 1.0
    changed = problem.layout.replace(run.final_force_field, vector)

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Conflicting bare save reached staging or serialization")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    monkeypatch.setattr(persistence, "_serializer", unexpected_write)
    monkeypatch.setattr(persistence, "_replace_transaction", unexpected_write)
    with pytest.raises(OutputExistsError, match="manifest.*different output path") as raised:
        persistence.save(changed, target, overwrite=overwrite)
    assert str(manifest) in str(raised.value)
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == original


def test_bare_save_without_sidecar_still_supports_fresh_and_overwrite(tmp_path: Path) -> None:
    problem = _problem()
    target = tmp_path / "bare.frcmod"
    saved = persistence.save(problem.starting_force_field, target)
    original = target.read_bytes()
    assert saved.manifest_path is None
    vector = problem.layout.vector(problem.starting_force_field).copy()
    vector[0] += 1.0
    changed = problem.layout.replace(problem.starting_force_field, vector)
    assert persistence.save(changed, target, overwrite=True) == saved
    assert target.read_bytes() != original
    assert set(tmp_path.iterdir()) == {target}


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("existing_target", [False, True])
def test_bare_save_rejects_dangling_manifest_link_before_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overwrite: bool, existing_target: bool
) -> None:
    target = tmp_path / "bare.frcmod"
    if existing_target:
        target.write_bytes(b"original force field")
    manifest = Path(f"{target}{persistence.MANIFEST_SUFFIX}")
    missing = tmp_path / "missing-manifest"
    try:
        manifest.symlink_to(missing)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            pytest.skip("Creating symlinks requires Windows developer mode or symlink privilege")
        raise
    assert manifest.is_symlink()
    assert not manifest.exists()
    original_link_target = manifest.readlink()
    before = set(tmp_path.iterdir())

    def unexpected_write(*args: object, **kwargs: object) -> None:
        pytest.fail("Dangling manifest reached staging or serialization")

    monkeypatch.setattr(persistence, "_temp_sibling", unexpected_write)
    monkeypatch.setattr(persistence, "_serializer", unexpected_write)
    monkeypatch.setattr(persistence, "_replace_transaction", unexpected_write)
    with pytest.raises(OutputExistsError, match="manifest.*different output path"):
        persistence.save(_problem().starting_force_field, target, overwrite=overwrite)

    assert set(tmp_path.iterdir()) == before
    assert manifest.is_symlink()
    assert manifest.readlink() == original_link_target
    assert not missing.exists()
    if existing_target:
        assert target.read_bytes() == b"original force field"
    else:
        assert not target.exists()


def test_run_over_run_save_replaces_both_artifacts(tmp_path: Path) -> None:
    problem = _problem()
    run = _run(problem)
    target = tmp_path / "run.frcmod"
    saved = persistence.save(run, target)
    manifest = saved.manifest_path
    assert manifest is not None
    original = {path: path.read_bytes() for path in (target, manifest)}
    vector = run.result.final_params.copy()
    vector[problem.active_space.active_indices[0]] += 1.0
    changed = replace(
        run,
        result=replace(run.result, final_params=vector, message="updated run"),
        final_force_field=problem.layout.replace(run.final_force_field, vector),
    )
    assert persistence.save(changed, target, overwrite=True) == saved
    assert all(path.read_bytes() != content for path, content in original.items())
    assert json.loads(manifest.read_text())["result"]["final_params"] == vector.tolist()
    assert set(tmp_path.iterdir()) == {target, manifest}


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
