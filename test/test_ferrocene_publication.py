from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from q2mm.benchmarks.publications import (
    FERROCENE_EXACT_SCAN_PROFILE,
    FERROCENE_SEVEN_STRUCTURE_PROFILE,
    REPOSITORY_OBJECTIVE_PROFILE,
    PublicationProfileBlockedError,
)
from q2mm.benchmarks.systems import load_system
from q2mm.benchmarks.systems._forcefield import compose_opt_with_mm3_base
from q2mm.benchmarks.systems._paths import (
    ExternalDataRoots,
    resolve_mm3_base_path,
    resolve_wahlers_opt_path,
)
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout, opt_substructure_membership
from q2mm.models.problem import StationaryPointKind
from q2mm.models.publication import ObjectiveTargetDisposition, PublicationTargetCategory
from scripts.freeze_publication_compatibility import _ferrocene_profile_row
from test._shared import REPO_ROOT

_FIXTURE = REPO_ROOT / "test" / "fixtures" / "ferrocene_publication_profile.json"
_EXPECTED = json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _external_roots_or_skip() -> ExternalDataRoots:
    roots = ExternalDataRoots.from_environment()
    missing = []
    if roots.supporting_info is None or not roots.supporting_info.is_dir():
        missing.append("Q2MM_SUPPORTING_INFO")
    if roots.mm3_base is None or not roots.mm3_base.is_file():
        missing.append("Q2MM_MM3_BASE")
    if missing:
        pytest.skip(f"Ferrocene publication data unavailable; configure {', '.join(missing)}")
    return roots


def test_ferrocene_fixture_is_path_free_and_contains_no_raw_science() -> None:
    text = _FIXTURE.read_text(encoding="utf-8")
    assert _FIXTURE.stat().st_size < 15_000
    assert _EXPECTED["objective_profile"] == FERROCENE_SEVEN_STRUCTURE_PROFILE
    assert _EXPECTED["reproduction_status"] == "partial_repository_reproduction"
    forbidden = (
        "C:\\",
        "H:\\",
        "/home/",
        '"hessian"',
        '"coordinates"',
        "Crystal Structure Comparision",
        "Diastereomeric Energy Difference",
        "Selectivity Predictions",
    )
    assert all(value.lower() not in text.lower() for value in forbidden)


def test_ferrocene_qfuerza_and_exact_scan_profiles_are_explicitly_blocked_without_loading_data() -> None:
    with pytest.raises(PublicationProfileBlockedError, match="D1 dummy-atom topology"):
        load_system(
            "ferrocene",
            starting_point="qfuerza",
            objective_profile=FERROCENE_SEVEN_STRUCTURE_PROFILE,
        )
    with pytest.raises(PublicationProfileBlockedError, match="four constrained scans"):
        load_system(
            "ferrocene",
            starting_point="published",
            objective_profile=FERROCENE_EXACT_SCAN_PROFILE,
        )


@pytest.mark.external_data
@pytest.mark.jax
def test_ferrocene_published_profile_matches_path_free_preparation_and_evaluation_identity() -> None:
    assert _ferrocene_profile_row(_external_roots_or_skip()) == _EXPECTED


@pytest.mark.external_data
def test_ferrocene_loader_has_numeric_ground_state_membership_and_no_validation_leakage() -> None:
    case = load_system(
        "ferrocene",
        data_roots=_external_roots_or_skip(),
        starting_point="published",
        objective_profile=FERROCENE_SEVEN_STRUCTURE_PROFILE,
        functional_form="mm3",
    )
    problem = case.problem
    publication = problem.publication_metadata
    assert problem.case_ids == ("TS1", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7")
    assert tuple(molecule.name for molecule in problem.molecules) == problem.case_ids
    assert all(case.stationary_point is StationaryPointKind.GROUND_STATE for case in problem.cases)
    assert publication is not None
    assert publication.authoritative_case_ids == problem.case_ids
    assert len(publication.force_field_blocks) == 4
    assert problem.starting_force_field.nonbonded_excluded_atom_types == ("FE",)
    targets = {target.category: target for target in publication.targets}
    assert targets[PublicationTargetCategory.CONSTRAINED_SCAN_ENERGY].disposition is ObjectiveTargetDisposition.BLOCKED
    counts = Counter(observation.kind for observation in problem.observations.values)
    assert counts == {
        "bond_length": 343,
        "bond_angle": 975,
        "eig_diagonal": 828,
        "eig_offdiagonal": 51336,
    }
    assert {observation.weight for observation in problem.observations.values if observation.kind == "bond_length"} == {
        10.0
    }
    assert {observation.weight for observation in problem.observations.values if observation.kind == "bond_angle"} == {
        5.0
    }
    assert "scan_energy" not in counts


@pytest.mark.external_data
def test_ferrocene_active_space_is_exactly_the_four_block_opt_membership() -> None:
    roots = _external_roots_or_skip()
    case = load_system(
        "ferrocene",
        data_roots=roots,
        starting_point="published",
        objective_profile=FERROCENE_SEVEN_STRUCTURE_PROFILE,
        functional_form="mm3",
    )
    composed, opt_only = compose_opt_with_mm3_base(
        resolve_wahlers_opt_path("Chapter 4", "mm3.ferrocene.fld", roots),
        resolve_mm3_base_path(roots),
        metal="FE",
    )
    layout = ParameterLayout.from_force_field(composed)
    expected = ActiveParameterSpace.from_membership(
        layout,
        composed,
        opt_substructure_membership(composed, opt_only),
    )

    assert case.problem.layout.fingerprint == layout.fingerprint
    assert case.problem.active_space.n_active == 134
    assert case.problem.active_space.n_full == 2766
    np.testing.assert_array_equal(case.problem.active_space.active_indices, expected.active_indices)
    assert case.problem.active_space.n_full - case.problem.active_space.n_active == 2632


@pytest.mark.external_data
def test_ferrocene_named_and_compatibility_profiles_change_metadata_not_observations() -> None:
    roots = _external_roots_or_skip()
    named = load_system(
        "ferrocene",
        data_roots=roots,
        starting_point="published",
        objective_profile=FERROCENE_SEVEN_STRUCTURE_PROFILE,
    )
    compatibility = load_system(
        "ferrocene",
        data_roots=roots,
        starting_point="published",
        objective_profile=REPOSITORY_OBJECTIVE_PROFILE,
    )
    named_refs = [(value.kind, value.value, value.weight, value.case_id) for value in named.problem.observations.values]
    compatibility_refs = [
        (value.kind, value.value, value.weight, value.case_id) for value in compatibility.problem.observations.values
    ]
    assert named_refs == compatibility_refs
    np.testing.assert_array_equal(
        named.problem.layout.vector(named.problem.starting_force_field),
        compatibility.problem.layout.vector(compatibility.problem.starting_force_field),
    )
    assert named.metadata["objective_profile"] == FERROCENE_SEVEN_STRUCTURE_PROFILE
    assert compatibility.metadata["objective_profile"] == REPOSITORY_OBJECTIVE_PROFILE
