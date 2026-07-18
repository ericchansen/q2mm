from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from q2mm.benchmarks.publications import (
    FERROCENE_EXACT_SCAN_PROFILE,
    FERROCENE_SEVEN_STRUCTURE_PROFILE,
    HECK_EXACT_PUBLICATION_PROFILE,
    PD_ALLYL_EIGHT_BLOCK_PROFILE,
    REPOSITORY_OBJECTIVE_PROFILE,
    PublicationProfileBlockedError,
    publication_record,
    publication_records,
    publication_success_spec,
)
from q2mm.models.publication import (
    ObjectiveTargetDisposition,
    PublicationMetadata,
    PublicationTargetCategory,
    ReproductionStatus,
)


def _record(system: str, profile: str, start: str) -> PublicationMetadata:
    return next(
        record
        for record in publication_records(system=system)
        if record.objective_profile.identifier == profile and record.starting_point == start
    )


def _source_tree_manifest(root: Path) -> list[dict[str, str]]:
    serialized_paths = (
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.name not in {"README.md", "run.py"}
    )
    return [
        {
            "path": relative_path,
            "sha256": hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
        }
        for relative_path, path in sorted(serialized_paths, key=lambda item: item[0])
    ]


def test_reproduction_status_vocabulary_is_exact() -> None:
    assert {status.value for status in ReproductionStatus} == {
        "exact_publication_reproduction",
        "executable_archive_reproduction",
        "partial_repository_reproduction",
        "sdk_software_path_demonstration",
        "blocked_historical_record",
    }


def test_metadata_is_frozen_path_free_and_deterministic() -> None:
    record = _record("ferrocene", FERROCENE_SEVEN_STRUCTURE_PROFILE, "published")
    payload = record.to_dict()
    text = json.dumps(payload, sort_keys=True)

    assert record.fingerprint == record.fingerprint
    assert "C:\\" not in text and "H:\\" not in text and "/home/" not in text
    assert record.authoritative_case_ids == ("TS1", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.system = "changed"  # type: ignore[misc]


def test_provisionable_metadata_requires_exact_case_membership_and_order() -> None:
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.molecule import Molecule
    from q2mm.models.observations import ObservationSet
    from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
    from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase

    record = _record("ferrocene", FERROCENE_SEVEN_STRUCTURE_PROFILE, "published")
    force_field = ForceField(functional_form=FunctionalForm.MM3)
    layout = ParameterLayout.from_force_field(force_field)
    cases = tuple(
        TrainingCase(
            case_id=f"unrelated-{index}",
            molecule=Molecule(symbols=("H",), geometry=np.zeros((1, 3)), bonds=(), angles=(), torsions=()),
            stationary_point=StationaryPointKind.GROUND_STATE,
        )
        for index in range(7)
    )
    with pytest.raises(ValueError, match="exactly match"):
        OptimizationProblem(
            cases=cases,
            starting_force_field=force_field,
            layout=layout,
            active_space=ActiveParameterSpace.all_active(layout, force_field),
            observations=ObservationSet(),
            publication_metadata=record,
        )


def test_rh_enamide_partial_row_records_source_omissions_and_distribution_conflict() -> None:
    record = _record("rh-enamide", REPOSITORY_OBJECTIVE_PROFILE, "published")
    targets = {target.category: target for target in record.targets}

    assert record.status is ReproductionStatus.PARTIAL_REPOSITORY_REPRODUCTION
    assert len(record.authoritative_case_ids) == 9
    assert targets[PublicationTargetCategory.ATOMIC_PARTIAL_CHARGE].disposition is ObjectiveTargetDisposition.BLOCKED
    assert targets[PublicationTargetCategory.RELATIVE_ENTHALPY].disposition is ObjectiveTargetDisposition.AVAILABLE
    assert targets[PublicationTargetCategory.BOND_LENGTH].profile_weight == 10.0
    assert targets[PublicationTargetCategory.BOND_LENGTH].source_weight == 100.0
    assert any("redistribution/licensing is not established" in blocker for blocker in record.blockers)


def test_rh_enamide_source_tree_fingerprint_is_content_verified() -> None:
    from test._shared import REPO_ROOT

    root = REPO_ROOT / "examples" / "publication" / "rh-enamide"
    entries = _source_tree_manifest(root)
    digest = "sha256:" + hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record = _record("rh-enamide", REPOSITORY_OBJECTIVE_PROFILE, "published")
    assert len(entries) == 66
    assert [entry["path"] for entry in entries] == sorted(entry["path"] for entry in entries)
    assert all("\\" not in entry["path"] for entry in entries)
    assert record.source_artifacts[0].fingerprint == digest


def test_heck_has_separate_23_archive_and_blocked_24_publication_rows() -> None:
    archive = _record("heck-relay", REPOSITORY_OBJECTIVE_PROFILE, "published")
    blocked = _record("heck-relay", HECK_EXACT_PUBLICATION_PROFILE, "published")

    assert archive.status is ReproductionStatus.EXECUTABLE_ARCHIVE_REPRODUCTION
    assert len(archive.authoritative_case_ids) == 23
    assert "prrts1" not in archive.authoritative_case_ids
    assert blocked.status is ReproductionStatus.BLOCKED_HISTORICAL_RECORD
    assert len(blocked.authoritative_case_ids) == 24
    assert "prrts1" in blocked.authoritative_case_ids
    assert blocked.blockers == ("The sole missing member of the publication's 24-case set is prrts1.",)
    with pytest.raises(PublicationProfileBlockedError) as exc:
        publication_record("heck-relay", HECK_EXACT_PUBLICATION_PROFILE, "published")
    assert exc.value.record is blocked


def test_pd_allyl_primary_and_auxiliary_rows_are_distinct() -> None:
    primary = _record("pd-allyl", REPOSITORY_OBJECTIVE_PROFILE, "published")
    blocked = _record("pd-allyl", PD_ALLYL_EIGHT_BLOCK_PROFILE, "published")

    assert len(primary.authoritative_case_ids) == 21
    assert len(primary.force_field_blocks) == 8
    assert primary.authoritative_case_ids[:3] == ("TS1", "TS10", "TS11")
    assert blocked.authoritative_case_ids == tuple(f"TS{index}" for index in range(1, 26))
    assert "TS22-TS25" in blocked.blockers[0]


def test_pd_and_rh_conjugate_claim_boundaries_are_exact() -> None:
    pd_record = _record("pd-conjugate", REPOSITORY_OBJECTIVE_PROFILE, "published")
    rh_record = _record("rh-conjugate", REPOSITORY_OBJECTIVE_PROFILE, "published")

    assert pd_record.authoritative_case_ids == ("TS1", "TS10", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7", "TS8", "TS9")
    assert len(pd_record.force_field_blocks) == 6
    assert "four conceptual groups" in pd_record.notes[0]
    assert rh_record.status is ReproductionStatus.SDK_SOFTWARE_PATH_DEMONSTRATION
    assert len(rh_record.authoritative_case_ids) == 10
    assert rh_record.governing_sources[0].chapter == "Chapter 6"
    assert "2021" in rh_record.governing_sources[0].citation
    assert "developmental" in rh_record.notes[0]


def test_ferrocene_is_ground_state_and_exact_scan_and_qfuerza_rows_are_blocked() -> None:
    partial = _record("ferrocene", FERROCENE_SEVEN_STRUCTURE_PROFILE, "published")
    exact = _record("ferrocene", FERROCENE_EXACT_SCAN_PROFILE, "published")
    qfuerza = _record("ferrocene", REPOSITORY_OBJECTIVE_PROFILE, "qfuerza")

    assert partial.stationary_point == "ground_state"
    assert len(partial.force_field_blocks) == 4
    assert partial.authoritative_case_ids == tuple(f"TS{index}" for index in range(1, 8))
    assert exact.status is ReproductionStatus.BLOCKED_HISTORICAL_RECORD
    assert "four constrained scans" in exact.blockers[0]
    assert "D1 dummy-atom topology" in qfuerza.blockers[0]
    with pytest.raises(PublicationProfileBlockedError):
        publication_record("ferrocene", FERROCENE_SEVEN_STRUCTURE_PROFILE, "qfuerza")


def test_historical_records_use_correct_candidate_sources_and_remain_blocked() -> None:
    records = {
        record.system: record
        for record in publication_records()
        if record.system in {"osmium-dihydroxylation", "ru-ketone-hydrogenation", "sulfone"}
    }
    assert set(records) == {"osmium-dihydroxylation", "ru-ketone-hydrogenation", "sulfone"}
    assert all(record.status is ReproductionStatus.BLOCKED_HISTORICAL_RECORD for record in records.values())
    assert records["osmium-dihydroxylation"].governing_sources[0].zotero_key == "BT3U4GKA"
    assert records["osmium-dihydroxylation"].governing_sources[0].doi == "10.1021/ja992023n"
    assert records["ru-ketone-hydrogenation"].governing_sources[0].zotero_key == "KF2F4U5E"
    assert records["ru-ketone-hydrogenation"].governing_sources[0].doi == "10.1021/ct500178w"
    assert records["sulfone"].governing_sources[0].zotero_key == "RPQ4XDL2"
    assert records["sulfone"].governing_sources[0].doi == "10.1021/acs.jpca.6b02757"
    assert "unverified" in records["sulfone"].blockers[0]


def test_canonical_full_optimization_success_specs_are_measurable_and_bounded() -> None:
    heck = publication_success_spec("heck-relay", REPOSITORY_OBJECTIVE_PROFILE, "qfuerza")
    published = publication_success_spec("heck-relay", REPOSITORY_OBJECTIVE_PROFILE, "published")
    ferrocene = publication_success_spec("ferrocene", FERROCENE_SEVEN_STRUCTURE_PROFILE, "published")

    assert heck.minimum_absolute_improvement_percent == 1.0
    assert heck.executor_ratio_bounds == (0.1, 10.0)
    assert heck.maximum_category_regression_percent_of_initial_total == 1.0
    assert heck.require_optimizer_convergence is True
    assert heck.require_accepted_candidate is True
    assert heck.canonical_full_run is True
    assert heck.proof_status == "blocked_methodology"
    assert heck.methodology_blocker is not None
    assert published.canonical_full_run is False
    assert published.proof_status == "bounded_software_path"
    assert published.methodology_blocker is None
    assert ferrocene.canonical_full_run is True
    assert ferrocene.proof_status == "blocked_methodology"


def test_publication_success_spec_enforces_convergence_and_weighted_category_regression() -> None:
    spec = publication_success_spec("rh-enamide", REPOSITORY_OBJECTIVE_PROFILE, "qfuerza")
    failed = spec.audit(
        improvement_percent=23.0,
        initial_executor_ratio=1.0,
        final_executor_ratio=1.0,
        initial_category_scores={"geometry": 100.0, "eigenmatrix": 1.0},
        final_category_scores={"geometry": 70.0, "eigenmatrix": 3.0},
        optimizer_converged=False,
        accepted=True,
    )
    assert failed["passes"] is False
    assert any("eigenmatrix weighted objective regressed" in failure for failure in failed["failures"])
    assert any("did not report convergence" in failure for failure in failed["failures"])

    passed = spec.audit(
        improvement_percent=42.0,
        initial_executor_ratio=1.0,
        final_executor_ratio=1.01,
        initial_category_scores={"geometry": 100.0, "eigenmatrix": 0.1},
        final_category_scores={"geometry": 57.0, "eigenmatrix": 0.2},
        optimizer_converged=True,
        accepted=True,
    )
    assert passed["passes"] is True
