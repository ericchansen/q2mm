from __future__ import annotations

import json
from pathlib import Path

import pytest

from q2mm.benchmarks.systems._paths import ExternalDataRoots
from scripts.freeze_publication_compatibility import _compatibility_row, _write_incremental
from test._shared import REPO_ROOT

_FIXTURE = REPO_ROOT / "test" / "fixtures" / "publication_problem_compatibility.json"
_DOCUMENT = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def test_publication_fixture_is_small_path_free_identity_only() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    assert _DOCUMENT["profile"] == "repository-geometry-eigenmatrix-v1"
    assert _DOCUMENT["incremental_per_row"] is True
    assert len(_DOCUMENT["rows"]) == 10
    assert _FIXTURE.stat().st_size < 50_000
    forbidden = (
        "C:\\",
        "H:\\",
        "/home/",
        "publication-data",
        "mm3_base.fld",
        '"geometry"',
        '"hessian"',
        '"coordinates"',
        "credential",
        "password",
        "api_key",
    )
    assert all(value.lower() not in text.lower() for value in forbidden)
    assert all(row["baseline_evaluation"] is None for row in _DOCUMENT["rows"])
    assert all(row["baseline_evaluation_policy"] == "compare-old-new-in-process" for row in _DOCUMENT["rows"])


def test_fixture_writer_persists_each_completed_row(tmp_path: Path) -> None:
    output = tmp_path / "compatibility.json"
    rows = _DOCUMENT["rows"]

    _write_incremental(output, [rows[0]])
    assert len(json.loads(output.read_text(encoding="utf-8"))["rows"]) == 1
    _write_incremental(output, [rows[0], rows[1]])
    assert len(json.loads(output.read_text(encoding="utf-8"))["rows"]) == 2
    assert not output.with_name(f".{output.name}.tmp").exists()


def _external_roots_or_skip() -> ExternalDataRoots:
    roots = ExternalDataRoots.from_environment()
    missing = []
    if roots.supporting_info is None or not roots.supporting_info.is_dir():
        missing.append("Q2MM_SUPPORTING_INFO")
    if roots.mm3_base is None or not roots.mm3_base.is_file():
        missing.append("Q2MM_MM3_BASE")
    if roots.rh_enamide is None or not roots.rh_enamide.is_dir():
        missing.append("Q2MM_RH_ENAMIDE")
    if missing:
        pytest.skip(f"publication compatibility data unavailable; configure {', '.join(missing)}")
    return roots


@pytest.mark.external_data
@pytest.mark.parametrize(
    "expected",
    _DOCUMENT["rows"],
    ids=lambda row: f"{row['system']}-{row['starting_point']}-{row['functional_form']}",
)
def test_every_publication_problem_matches_frozen_compatibility(expected: dict[str, object]) -> None:
    roots = _external_roots_or_skip()
    actual = _compatibility_row(
        str(expected["system"]),
        str(expected["starting_point"]),
        roots,
    )
    assert actual == expected
