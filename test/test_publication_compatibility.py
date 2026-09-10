from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

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


@pytest.mark.parametrize("system", ["rh-enamide", "ferrocene"])
def test_publication_assembly_explicitly_preserves_compatibility_recipe(system: str) -> None:
    from q2mm.benchmarks.publications import FERROCENE_SEVEN_STRUCTURE_PROFILE, publication_record
    from q2mm.benchmarks.systems import _assembly
    from q2mm.models.observations import ObservationSet
    from q2mm.models.problem import StationaryPointKind
    from q2mm.preparation import MoleculeObservations
    from test._shared import make_harmonic_water
    from test.test_preparation import _template

    profile = FERROCENE_SEVEN_STRUCTURE_PROFILE if system == "ferrocene" else "repository-geometry-eigenmatrix-v1"
    source = publication_record(system, profile, "published")
    metadata = replace(source, authoritative_case_ids=("synthetic-a", "synthetic-b"))
    molecules = [make_harmonic_water(), make_harmonic_water()]
    with (
        patch.object(_assembly, "prepare", wraps=_assembly.prepare) as prepare_spy,
        patch("q2mm.preparation._resolve_linearity", side_effect=AssertionError("generic recipe leaked")),
    ):
        case = _assembly.assemble_published_case(
            key=system,
            name="synthetic assembly-routing proof",
            molecules=molecules,
            composed_ff=_template(),
            opt_only_ff=_template(),
            stationary_point=StationaryPointKind(metadata.stationary_point),
            starting_point="published",
            qfuerza_replace_with=1.0,
            functional_form="harmonic",
            metadata={},
            publication_metadata=metadata,
            case_ids=("a", "b"),
        )
    assert prepare_spy.call_args.kwargs["observations"] == MoleculeObservations()
    assert case.problem.observations == ObservationSet.from_molecules(molecules, ("a", "b"))
    assert case.problem.preparation_provenance.profile == profile
    assert case.problem.preparation_provenance.observation_recipe == {
        "name": "MoleculeObservations",
        "profile": profile,
        "geometry": True,
        "eigenmatrix": "full",
    }
    assert tuple(c.source_id for c in case.problem.cases) == ("synthetic-a", "synthetic-b")
