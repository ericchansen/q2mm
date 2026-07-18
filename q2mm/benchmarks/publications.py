"""Canonical publication-profile records for repository benchmark systems.

The records here are metadata and claim boundaries, not public per-system SDK
types.  Scientific data remain caller-supplied and system loading remains
lazy under :mod:`q2mm.benchmarks.systems`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping

from q2mm.models.publication import (
    ObjectiveProfileIdentity,
    ObjectiveTarget,
    ObjectiveTargetDisposition,
    PublicationCitation,
    PublicationMetadata,
    PublicationTargetCategory,
    ReproductionStatus,
    SourceArtifactIdentity,
)

__all__ = [
    "REPOSITORY_OBJECTIVE_PROFILE",
    "FERROCENE_SEVEN_STRUCTURE_PROFILE",
    "HECK_EXACT_PUBLICATION_PROFILE",
    "PD_ALLYL_EIGHT_BLOCK_PROFILE",
    "FERROCENE_EXACT_SCAN_PROFILE",
    "KNOWN_OBJECTIVE_PROFILES",
    "PUBLICATION_SYSTEM_KEYS",
    "PublicationProfileError",
    "PublicationProfileBlockedError",
    "PublicationOptimizationSuccessSpec",
    "publication_record",
    "publication_records",
    "publication_profile_ids",
    "publication_success_spec",
]

_REPOSITORY_PROFILE = ObjectiveProfileIdentity("repository-geometry-eigenmatrix", 1)
_FERROCENE_SEVEN_PROFILE = ObjectiveProfileIdentity("wahlers-ferrocene-seven-structure", 1)
_HECK_EXACT_PROFILE = ObjectiveProfileIdentity("rosales-heck-publication-24", 1)
_PD_ALLYL_EIGHT_BLOCK_PROFILE = ObjectiveProfileIdentity("wahlers-pd-allyl-eight-block", 1)
_FERROCENE_EXACT_SCAN_PROFILE = ObjectiveProfileIdentity("wahlers-ferrocene-full-scan", 1)
_HISTORICAL_PROFILE = ObjectiveProfileIdentity("historical-force-field-record", 1)

REPOSITORY_OBJECTIVE_PROFILE = _REPOSITORY_PROFILE.identifier
FERROCENE_SEVEN_STRUCTURE_PROFILE = _FERROCENE_SEVEN_PROFILE.identifier
HECK_EXACT_PUBLICATION_PROFILE = _HECK_EXACT_PROFILE.identifier
PD_ALLYL_EIGHT_BLOCK_PROFILE = _PD_ALLYL_EIGHT_BLOCK_PROFILE.identifier
FERROCENE_EXACT_SCAN_PROFILE = _FERROCENE_EXACT_SCAN_PROFILE.identifier
KNOWN_OBJECTIVE_PROFILES: frozenset[str] = frozenset(
    {
        REPOSITORY_OBJECTIVE_PROFILE,
        FERROCENE_SEVEN_STRUCTURE_PROFILE,
        HECK_EXACT_PUBLICATION_PROFILE,
        PD_ALLYL_EIGHT_BLOCK_PROFILE,
        FERROCENE_EXACT_SCAN_PROFILE,
        _HISTORICAL_PROFILE.identifier,
    }
)
PUBLICATION_SYSTEM_KEYS: frozenset[str] = frozenset(
    {"rh-enamide", "heck-relay", "pd-allyl", "pd-conjugate", "rh-conjugate", "ferrocene"}
)
RELAXED_GEOMETRY_METHODOLOGY_BLOCKER = (
    "Full canonical optimization is blocked until q2mm defines and validates "
    "local-basin semantics for relaxed-geometry objectives; preparation, "
    "evaluation, bounded optimizer entry, and persistence remain supported."
)

_IMPLEMENTED = ObjectiveTargetDisposition.IMPLEMENTED
_AVAILABLE = ObjectiveTargetDisposition.AVAILABLE
_OMITTED = ObjectiveTargetDisposition.OMITTED
_BLOCKED = ObjectiveTargetDisposition.BLOCKED
_CATEGORY = PublicationTargetCategory

_RH_CASES = (
    "1ZDMPfromJCTCSI_loner1.01",
    "2DMPEConformation1fromJCTCSI_isomer1.01",
    "3DMPEConformation2fromJCTCSI_isomer2.01",
    "4RR-Me-DuPHOSpro-RfromJCTCSI_isomer3.01",
    "5RR-Me-DuPHOSpro-SfromJCTCSI_isomer4.01",
    "6RR-Me-BPEpro-RConformation1fromJCTCSI_isomer5.01",
    "7RR-Me-BPEpro-RConformation2fromJCTCSI_isomer6.01",
    "8RR-Me-BPEpro-SConformation1fromJCTCSI_isomer7.01",
    "9RR-Me-BPEpro-SConformation2fromJCTCSI_isomer8.01",
)
_HECK_CASES = (
    "aagts1",
    "aagts2",
    "bgts1",
    "bgts2",
    "cf3ets1",
    "cf3ets2",
    "dimeets1",
    "dimeets2",
    "epgts1",
    "epgts2",
    "ipets1",
    "ipets2",
    "meets1",
    "meets2",
    "prrts2",
    "prsts1",
    "prsts2",
    "tbets1",
    "tbets2",
    "tmets1",
    "tmets2",
    "zpgts1",
    "zpgts2",
)
_HECK_PUBLICATION_CASES = (*_HECK_CASES[:14], "prrts1", *_HECK_CASES[14:])
_PD_ALLYL_CASES = (
    "TS1",
    "TS10",
    "TS11",
    "TS12",
    "TS13",
    "TS14",
    "TS15",
    "TS16",
    "TS17",
    "TS18",
    "TS19",
    "TS2",
    "TS20",
    "TS21",
    "TS3",
    "TS4",
    "TS5",
    "TS6",
    "TS7",
    "TS8",
    "TS9",
)
_TEN_CASE_LEXICOGRAPHIC = ("TS1", "TS10", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7", "TS8", "TS9")
_FERROCENE_CASES = ("TS1", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7")

_QFUERZA = PublicationCitation(
    citation="Farrugia, Helquist, Norrby & Wiest, J. Chem. Theory Comput. 2025",
    authoritative_url="https://doi.org/10.1021/acs.jctc.5c01751",
    doi="10.1021/acs.jctc.5c01751",
    zotero_key="XDS9K3C4",
)
_RH_SOURCE = PublicationCitation(
    citation="Donoghue, Helquist, Norrby & Wiest, J. Chem. Theory Comput. 2008, 4, 1313",
    authoritative_url="https://doi.org/10.1021/ct800132a",
    doi="10.1021/ct800132a",
    zotero_key="JXH5HHS6",
)
_HECK_SOURCE = PublicationCitation(
    citation="Rosales, Helquist, Norrby & Wiest, J. Am. Chem. Soc. 2020, 142, 9700",
    authoritative_url="https://doi.org/10.1021/jacs.0c01979",
    doi="10.1021/jacs.0c01979",
    zotero_key="2NHVUNW5",
)
_ROSALES_DISSERTATION = PublicationCitation(
    citation="Rosales, Ph.D. Dissertation, University of Notre Dame, 2019",
    authoritative_url="https://doi.org/10.7274/rj430290902",
    doi="10.7274/rj430290902",
    zotero_key="QCQ6Z5MR",
    chapter="Chapter 3",
)
_PD_ALLYL_SOURCE = PublicationCitation(
    citation="Wahlers et al., Nature Communications 2021, 12, 6508",
    authoritative_url="https://doi.org/10.1038/s41467-021-27065-2",
    doi="10.1038/s41467-021-27065-2",
    zotero_key="QVKE99W3",
)
_PD_CONJUGATE_SOURCE = PublicationCitation(
    citation="Wahlers et al., J. Org. Chem. 2021, 86, 5660",
    authoritative_url="https://doi.org/10.1021/acs.joc.1c00136",
    doi="10.1021/acs.joc.1c00136",
    zotero_key="R62E6EGV",
)
_FERROCENE_SOURCE = PublicationCitation(
    citation="Wahlers et al., J. Org. Chem. 2022, 87, 12334",
    authoritative_url="https://doi.org/10.1021/acs.joc.2c01553",
    doi="10.1021/acs.joc.2c01553",
    zotero_key="SXWNJTQ2",
)
_WAHLERS_DISSERTATION = PublicationCitation(
    citation="Wahlers, Ph.D. Dissertation, University of Notre Dame, 2021",
    authoritative_url="https://doi.org/10.7274/k930bv76q4n",
    doi="10.7274/k930bv76q4n",
    zotero_key="AAZ6I5V3",
)

_WAHLERS_ARCHIVE = SourceArtifactIdentity(
    identity="Wahlers dissertation supporting-information archive",
    role="authoritative external scientific archive",
    fingerprint="md5:47dbdbc26362d7b3487ea9855c0d1718",
)
_ROSALES_ARCHIVE = SourceArtifactIdentity(
    identity="Rosales dissertation supporting-information archive",
    role="authoritative external scientific archive",
    fingerprint="md5:e2d9ed2701f897d0b515089f4edf2e40",
)
_MM3_BASE = SourceArtifactIdentity(
    identity="mm3_base.fld",
    role="licensed external MM3 base; never distributed by q2mm",
)


def _structural_targets(
    *,
    source_bond_weight: float | None,
    source_angle_weight: float | None,
    eigenmatrix_details: str,
) -> tuple[ObjectiveTarget, ...]:
    return (
        ObjectiveTarget(
            _CATEGORY.BOND_LENGTH,
            _IMPLEMENTED,
            profile_weight=10.0,
            source_weight=source_bond_weight,
            unit="angstrom^-1",
            details="Compatibility profile preserves the repository weight byte-for-byte.",
        ),
        ObjectiveTarget(
            _CATEGORY.BOND_ANGLE,
            _IMPLEMENTED,
            profile_weight=5.0,
            source_weight=source_angle_weight,
            unit="degree^-1",
            details="Compatibility profile preserves the repository weight byte-for-byte.",
        ),
        ObjectiveTarget(
            _CATEGORY.TORSION_GEOMETRY,
            _OMITTED,
            source_weight=1.0,
            unit="degree^-1",
            details="The canonical torsion observation exists, but this compatibility profile does not populate it.",
        ),
        ObjectiveTarget(
            _CATEGORY.EIGENMATRIX_DIAGONAL,
            _IMPLEMENTED,
            profile_weight=0.1,
            source_weight=0.1,
            unit="dimensionless",
            details=f"Reaction/negative mode weight is 0. {eigenmatrix_details}",
        ),
        ObjectiveTarget(
            _CATEGORY.EIGENMATRIX_OFFDIAGONAL,
            _IMPLEMENTED,
            profile_weight=0.05,
            source_weight=0.05,
            unit="dimensionless",
            details=eigenmatrix_details,
        ),
    )


_RH_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="The source final fit used a restricted structure/Hessian selection, unlike all-nine compatibility rows.",
    ),
    ObjectiveTarget(
        _CATEGORY.ATOMIC_PARTIAL_CHARGE,
        _BLOCKED,
        source_weight=50.0,
        unit="elementary_charge^-1",
        details="The paper used ESP charges with a 0.02 e tolerance; MM backends do not expose calculated atomic charges.",
    ),
    ObjectiveTarget(
        _CATEGORY.RELATIVE_ENTHALPY,
        _AVAILABLE,
        unit="kJ/mol",
        details="Source conformer/diastereomer enthalpies are tabulated; the compatibility profile intentionally omits them.",
    ),
)
_HECK_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="The deposited archive contains 23 of the 24 structures described in the dissertation.",
    ),
    ObjectiveTarget(
        _CATEGORY.DIRECT_ELECTROSTATIC_POTENTIAL,
        _BLOCKED,
        unit="electrostatic_potential",
        details="Bond dipoles were fitted to direct CHELPG ESP error, not to atomic-charge residuals.",
    ),
    ObjectiveTarget(
        _CATEGORY.EQUILIBRIUM_PARAMETER_TETHER,
        _BLOCKED,
        details="The source used equilibrium-value tethers; authoritative per-slot target values are not reconstructed.",
    ),
)
_PD_ALLYL_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="The compatibility row contains the 21 primary cases only.",
    ),
    ObjectiveTarget(
        _CATEGORY.ATOMIC_PARTIAL_CHARGE,
        _BLOCKED,
        unit="elementary_charge",
        details="The source electrostatic stage is not executable through current MM backend results.",
    ),
    ObjectiveTarget(
        _CATEGORY.EQUILIBRIUM_PARAMETER_TETHER,
        _BLOCKED,
        details="The source used equilibrium-value tethers; exact per-slot targets are unavailable.",
    ),
)
_PD_CONJUGATE_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="All ten primary structures are represented.",
    ),
    ObjectiveTarget(
        _CATEGORY.ATOMIC_PARTIAL_CHARGE,
        _BLOCKED,
        unit="elementary_charge",
        details="CHELPG electrostatics are not executable through current MM backend results.",
    ),
    ObjectiveTarget(
        _CATEGORY.EQUILIBRIUM_PARAMETER_TETHER,
        _BLOCKED,
        details="The source used equilibrium-value tethers; exact per-slot targets are unavailable.",
    ),
)
_RH_CONJUGATE_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="The compatibility row combines the source's staged eight-case and two-case fits.",
    ),
    ObjectiveTarget(
        _CATEGORY.ATOMIC_PARTIAL_CHARGE,
        _BLOCKED,
        unit="elementary_charge",
        details="CHELPG electrostatics are not executable through current MM backend results.",
    ),
)
_FERROCENE_TARGETS = (
    *_structural_targets(
        source_bond_weight=100.0,
        source_angle_weight=2.0,
        eigenmatrix_details="The source describes Hessian eigenvalues; the repository partial profile retains its full eigenmatrix recipe.",
    ),
    ObjectiveTarget(
        _CATEGORY.ATOMIC_PARTIAL_CHARGE,
        _BLOCKED,
        unit="elementary_charge",
        details="Published point-charge/bond-dipole fitting is not executable through current MM backend results.",
    ),
    ObjectiveTarget(
        _CATEGORY.CONSTRAINED_SCAN_ENERGY,
        _BLOCKED,
        unit="kcal/mol",
        details="Numerical data for all four source scans are absent from the recovered archive.",
    ),
)


def _with_start(record: PublicationMetadata, starting_point: str) -> PublicationMetadata:
    sources = record.governing_sources
    if starting_point == "qfuerza" and _QFUERZA not in sources:
        sources = (*sources, _QFUERZA)
    return dataclasses.replace(record, governing_sources=sources, starting_point=starting_point)


def _two_starts(record: PublicationMetadata) -> tuple[PublicationMetadata, PublicationMetadata]:
    return _with_start(record, "published"), _with_start(record, "qfuerza")


_RH_BASE = PublicationMetadata(
    system="rh-enamide",
    status=ReproductionStatus.PARTIAL_REPOSITORY_REPRODUCTION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(_RH_SOURCE,),
    authoritative_case_ids=_RH_CASES,
    stationary_point="transition_state",
    targets=_RH_TARGETS,
    source_artifacts=(
        SourceArtifactIdentity(
            identity="Rh-enamide tracked source tree",
            role="repository-tracked scientific-input content manifest excluded from installed package data",
            fingerprint="sha256:8453fa0d81f58a56278c187d23e18adf9812a7594977f713a23938c5cd9d43fe",
        ),
    ),
    force_field_blocks=("RhH3-E core OPT", "RH-PX OPT"),
    blockers=(
        "Inputs are tracked in the source repository and excluded from distribution artifacts; "
        "redistribution/licensing is not established.",
    ),
    notes=("The published objective also used ESP charges and relative enthalpies.",),
)
_HECK_BASE = PublicationMetadata(
    system="heck-relay",
    status=ReproductionStatus.EXECUTABLE_ARCHIVE_REPRODUCTION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(_HECK_SOURCE, _ROSALES_DISSERTATION),
    authoritative_case_ids=_HECK_CASES,
    stationary_point="transition_state",
    targets=_HECK_TARGETS,
    source_artifacts=(_ROSALES_ARCHIVE,),
    force_field_blocks=("Heck Palladium", "Palladium pyridine", "Palladium oxazoline", "Sqr Plane"),
    notes=("This row preserves the deposited 23-case archive and a partial repository objective.",),
)
_PD_ALLYL_BASE = PublicationMetadata(
    system="pd-allyl",
    status=ReproductionStatus.PARTIAL_REPOSITORY_REPRODUCTION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(_PD_ALLYL_SOURCE, dataclasses.replace(_WAHLERS_DISSERTATION, chapter="Chapter 3")),
    authoritative_case_ids=_PD_ALLYL_CASES,
    stationary_point="transition_state",
    targets=_PD_ALLYL_TARGETS,
    source_artifacts=(_WAHLERS_ARCHIVE, _MM3_BASE),
    force_field_blocks=(
        "PdTS_Core",
        "PdTS_PP",
        "PdTS_PN",
        "PdTS_N3 ligand",
        "PdTS_N2 ligand",
        "PdTS_amine",
        "Palladium oxazoline",
        "PdAllyl Oxazole",
    ),
    blockers=("Complete eight-block rederivation requires auxiliary TS22-TS25 oxazole Hessian data.",),
    notes=("The executable compatibility row contains the 21 primary TS1-TS21 structures.",),
)
_PD_CONJUGATE_BASE = PublicationMetadata(
    system="pd-conjugate",
    status=ReproductionStatus.PARTIAL_REPOSITORY_REPRODUCTION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(_PD_CONJUGATE_SOURCE, dataclasses.replace(_WAHLERS_DISSERTATION, chapter="Chapter 5")),
    authoritative_case_ids=_TEN_CASE_LEXICOGRAPHIC,
    stationary_point="transition_state",
    targets=_PD_CONJUGATE_TARGETS,
    source_artifacts=(_WAHLERS_ARCHIVE, _MM3_BASE),
    force_field_blocks=(
        "Carbonyl bond",
        "Palladium 1,4-conj_add A",
        "Palladium 1,4-conj_add B",
        "Palladium pyridine-extra",
        "Palladium pyridine",
        "Palladium oxazoline",
    ),
    notes=("The source describes four conceptual groups implemented as six physical OPT blocks.",),
)
_RH_CONJUGATE_BASE = PublicationMetadata(
    system="rh-conjugate",
    status=ReproductionStatus.SDK_SOFTWARE_PATH_DEMONSTRATION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(dataclasses.replace(_WAHLERS_DISSERTATION, chapter="Chapter 6"),),
    authoritative_case_ids=_TEN_CASE_LEXICOGRAPHIC,
    stationary_point="transition_state",
    targets=_RH_CONJUGATE_TARGETS,
    source_artifacts=(_WAHLERS_ARCHIVE, _MM3_BASE),
    force_field_blocks=("Rhodium 1,4-addition", "Diphosphine", "Diene", "Enone", "Imide substrates"),
    notes=("This 2021 dissertation force field is developmental, not a mature validation system.",),
)
_FERROCENE_BASE = PublicationMetadata(
    system="ferrocene",
    status=ReproductionStatus.PARTIAL_REPOSITORY_REPRODUCTION,
    objective_profile=_REPOSITORY_PROFILE,
    governing_sources=(_FERROCENE_SOURCE, dataclasses.replace(_WAHLERS_DISSERTATION, chapter="Chapter 4")),
    authoritative_case_ids=_FERROCENE_CASES,
    stationary_point="ground_state",
    targets=_FERROCENE_TARGETS,
    source_artifacts=(
        _WAHLERS_ARCHIVE,
        _MM3_BASE,
        SourceArtifactIdentity(
            identity="Chapter 4 TS1-TS7 Gaussian-log manifest",
            role="seven-case ground-state training set",
            fingerprint="sha256:923625fe4203f1263c5bb2221c07805bef473077488a29cf7d39768adeca8c8f",
        ),
        SourceArtifactIdentity(
            identity="mm3.ferrocene.fld",
            role="four-block OPT-only published force field",
            fingerprint="sha256:93a1f1fe8d8f48921a2884c385da16de03e264e1c3e415c124164e610c8dff41",
        ),
    ),
    force_field_blocks=(
        "Ferrocene_2016",
        "Ferrocene_C2_Ligands_2016",
        "Ferrocene_C3_Ligands_2016",
        "Ferrocene_PX_Ligands_2016",
    ),
    blockers=("Exact reoptimization requires four absent numerical constrained-scan data sets.",),
    notes=(
        "TS-prefixed archive labels do not change the governing ground-state semantics.",
        "External crystal, diastereomer, and selectivity validation structures are not training cases.",
        "The seven-case partial evaluator explicitly excludes the Fe atom type from nonbonded pairs because the source OPT file defines no Fe vdW row; it does not reconstruct the D1 dummy topology.",
    ),
)
_FERROCENE_NAMED = dataclasses.replace(_FERROCENE_BASE, objective_profile=_FERROCENE_SEVEN_PROFILE)

_PROVISIONABLE: tuple[PublicationMetadata, ...] = (
    *_two_starts(_RH_BASE),
    *_two_starts(_HECK_BASE),
    *_two_starts(_PD_ALLYL_BASE),
    *_two_starts(_PD_CONJUGATE_BASE),
    *_two_starts(_RH_CONJUGATE_BASE),
    _with_start(_FERROCENE_BASE, "published"),
    _with_start(_FERROCENE_NAMED, "published"),
)

_BLOCKED_RECORDS: tuple[PublicationMetadata, ...] = (
    dataclasses.replace(
        _HECK_BASE,
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_HECK_EXACT_PROFILE,
        authoritative_case_ids=_HECK_PUBLICATION_CASES,
        starting_point="published",
        provisionable=False,
        blockers=("The sole missing member of the publication's 24-case set is prrts1.",),
        notes=("No synthetic prrts1 structure is permitted.",),
    ),
    dataclasses.replace(
        _PD_ALLYL_BASE,
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_PD_ALLYL_EIGHT_BLOCK_PROFILE,
        authoritative_case_ids=tuple(f"TS{index}" for index in range(1, 26)),
        starting_point="published",
        provisionable=False,
        blockers=(
            "Auxiliary oxazole cases TS22-TS25 lack the Hessian data required for exact eight-block rederivation.",
        ),
    ),
    dataclasses.replace(
        _FERROCENE_BASE,
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_FERROCENE_EXACT_SCAN_PROFILE,
        starting_point="published",
        provisionable=False,
        blockers=("Numerical data are absent for all four constrained scans used by the publication.",),
    ),
    dataclasses.replace(
        _FERROCENE_BASE,
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        starting_point="qfuerza",
        provisionable=False,
        blockers=(
            "QFUERZA initialization is unsupported until the D1 dummy-atom topology is represented and its frozen partition is proven.",
        ),
    ),
    dataclasses.replace(
        _FERROCENE_NAMED,
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        starting_point="qfuerza",
        provisionable=False,
        blockers=(
            "QFUERZA initialization is unsupported until the D1 dummy-atom topology is represented and its frozen partition is proven.",
        ),
    ),
    PublicationMetadata(
        system="osmium-dihydroxylation",
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_HISTORICAL_PROFILE,
        governing_sources=(
            PublicationCitation(
                citation="Norrby et al., J. Am. Chem. Soc. 1999",
                authoritative_url="https://doi.org/10.1021/ja992023n",
                doi="10.1021/ja992023n",
                zotero_key="BT3U4GKA",
            ),
        ),
        authoritative_case_ids=(),
        stationary_point="transition_state",
        targets=(ObjectiveTarget(_CATEGORY.EIGENMATRIX_DIAGONAL, _BLOCKED),),
        source_artifacts=(
            SourceArtifactIdentity(
                identity="os-dihydroxylation-alkene.fld",
                role="legacy force field with unverified publication mapping",
            ),
        ),
        blockers=(
            "No authoritative QM training files are available and the legacy force-field-to-publication mapping is unverified.",
        ),
        provisionable=False,
    ),
    PublicationMetadata(
        system="ru-ketone-hydrogenation",
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_HISTORICAL_PROFILE,
        governing_sources=(
            PublicationCitation(
                citation="Limé et al., J. Chem. Theory Comput. 2014",
                authoritative_url="https://doi.org/10.1021/ct500178w",
                doi="10.1021/ct500178w",
                zotero_key="KF2F4U5E",
            ),
        ),
        authoritative_case_ids=(),
        stationary_point="transition_state",
        targets=(ObjectiveTarget(_CATEGORY.EIGENMATRIX_DIAGONAL, _BLOCKED),),
        source_artifacts=(
            SourceArtifactIdentity(identity="ru-hydrogenation-ketone.fld", role="legacy force field only"),
        ),
        blockers=("No authoritative QM training files are available.",),
        provisionable=False,
    ),
    PublicationMetadata(
        system="sulfone",
        status=ReproductionStatus.BLOCKED_HISTORICAL_RECORD,
        objective_profile=_HISTORICAL_PROFILE,
        governing_sources=(
            PublicationCitation(
                citation="Hansen et al., J. Phys. Chem. A 2016 candidate mapping",
                authoritative_url="https://doi.org/10.1021/acs.jpca.6b02757",
                doi="10.1021/acs.jpca.6b02757",
                zotero_key="RPQ4XDL2",
            ),
        ),
        authoritative_case_ids=(),
        stationary_point="unknown",
        targets=(ObjectiveTarget(_CATEGORY.CONSTRAINED_SCAN_ENERGY, _BLOCKED),),
        source_artifacts=(
            SourceArtifactIdentity(
                identity="sulfone.fld",
                role="legacy force field with unverified candidate publication mapping",
            ),
        ),
        blockers=(
            "The legacy file-to-paper mapping is unverified and no authoritative QM training or scan data are available.",
        ),
        provisionable=False,
    ),
)

_RECORDS = (*_PROVISIONABLE, *_BLOCKED_RECORDS)


@dataclasses.dataclass(frozen=True)
class PublicationOptimizationSuccessSpec:
    """Pre-flight gate for a parent-run canonical publication optimization."""

    system: str
    objective_profile: str
    starting_point: str
    minimum_absolute_improvement_percent: float = 1.0
    executor_ratio_bounds: tuple[float, float] = (0.1, 10.0)
    maximum_category_regression_percent_of_initial_total: float = 1.0
    require_optimizer_convergence: bool = True
    require_accepted_candidate: bool = True
    canonical_full_run: bool = False
    proof_status: str = "bounded_software_path"
    methodology_blocker: str | None = None

    def __post_init__(self) -> None:
        if self.proof_status not in {"bounded_software_path", "blocked_methodology"}:
            raise ValueError(f"Unknown publication optimization proof status {self.proof_status!r}.")
        if self.proof_status == "blocked_methodology":
            if not self.canonical_full_run or not self.methodology_blocker:
                raise ValueError("Blocked methodology proofs require a canonical row and explicit blocker.")
        elif self.methodology_blocker is not None:
            raise ValueError("Bounded software-path proofs cannot carry a methodology blocker.")

    def audit(
        self,
        *,
        improvement_percent: float,
        initial_executor_ratio: float | None,
        final_executor_ratio: float | None,
        initial_category_scores: Mapping[str, float],
        final_category_scores: Mapping[str, float],
        optimizer_converged: bool,
        accepted: bool,
    ) -> dict[str, object]:
        """Evaluate the measurable canonical-run gate."""
        failures: list[str] = []
        if improvement_percent < self.minimum_absolute_improvement_percent:
            failures.append(
                f"improvement={improvement_percent:.3f}% < {self.minimum_absolute_improvement_percent:.3f}%"
            )
        low, high = self.executor_ratio_bounds
        ratios = {
            "initial_executor_ratio": initial_executor_ratio,
            "final_executor_ratio": final_executor_ratio,
        }
        for name, ratio in ratios.items():
            if ratio is None or not math.isfinite(ratio) or not low <= ratio <= high:
                failures.append(f"{name}={ratio!r} outside [{low}, {high}]")
        initial_total = sum(float(value) for value in initial_category_scores.values())
        regression_budget = initial_total * self.maximum_category_regression_percent_of_initial_total / 100.0
        category_regressions: dict[str, dict[str, float | bool]] = {}
        for category in sorted(set(initial_category_scores) | set(final_category_scores)):
            initial = float(initial_category_scores.get(category, 0.0))
            final = float(final_category_scores.get(category, 0.0))
            increase = final - initial
            passes = increase <= regression_budget
            category_regressions[category] = {
                "initial": initial,
                "final": final,
                "increase": increase,
                "allowed_increase": regression_budget,
                "passes": passes,
            }
            if not passes:
                failures.append(
                    f"{category} weighted objective regressed by {increase:.6g}, "
                    f"exceeding {regression_budget:.6g} "
                    f"({self.maximum_category_regression_percent_of_initial_total:g}% of initial total)"
                )
        if self.require_optimizer_convergence and not optimizer_converged:
            failures.append("optimizer did not report convergence")
        if self.require_accepted_candidate and not accepted:
            failures.append("base acceptance policy rejected the candidate")
        return {
            "passes": not failures,
            "failures": failures,
            "executor_ratios": ratios,
            "category_regressions": category_regressions,
        }

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe pre-flight gate."""
        return {
            "system": self.system,
            "objective_profile": self.objective_profile,
            "starting_point": self.starting_point,
            "minimum_absolute_improvement_percent": self.minimum_absolute_improvement_percent,
            "executor_ratio_bounds": list(self.executor_ratio_bounds),
            "maximum_category_regression_percent_of_initial_total": (
                self.maximum_category_regression_percent_of_initial_total
            ),
            "require_optimizer_convergence": self.require_optimizer_convergence,
            "require_accepted_candidate": self.require_accepted_candidate,
            "canonical_full_run": self.canonical_full_run,
            "proof_status": self.proof_status,
            "methodology_blocker": self.methodology_blocker,
        }


class PublicationProfileError(ValueError):
    """A publication system/profile/start combination is unknown."""


class PublicationProfileBlockedError(PublicationProfileError):
    """A known publication profile cannot be constructed from available data."""

    def __init__(self, record: PublicationMetadata) -> None:
        self.record = record
        reasons = "; ".join(record.blockers)
        super().__init__(
            f"{record.system}/{record.objective_profile.identifier}/{record.starting_point or 'none'} is blocked: {reasons}"
        )


def publication_records(*, system: str | None = None) -> tuple[PublicationMetadata, ...]:
    """Return all canonical records, optionally restricted to one system."""
    if system is None:
        return _RECORDS
    return tuple(record for record in _RECORDS if record.system == system)


def publication_profile_ids(system: str, *, provisionable_only: bool = False) -> tuple[str, ...]:
    """Return deterministic objective-profile identifiers known for *system*."""
    values = {
        record.objective_profile.identifier
        for record in _RECORDS
        if record.system == system and (record.provisionable or not provisionable_only)
    }
    return tuple(sorted(values))


def publication_record(system: str, objective_profile: str, starting_point: str) -> PublicationMetadata:
    """Resolve one provisionable publication record or raise a typed blocker."""
    matches = tuple(
        record
        for record in _RECORDS
        if record.system == system
        and record.objective_profile.identifier == objective_profile
        and record.starting_point == starting_point
    )
    if not matches:
        known = publication_profile_ids(system)
        raise PublicationProfileError(
            f"Unknown publication profile/start for {system!r}: {objective_profile!r}/{starting_point!r}; "
            f"known objective profiles: {list(known)}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"Duplicate canonical publication records for {system}/{objective_profile}/{starting_point}."
        )
    record = matches[0]
    if not record.provisionable:
        raise PublicationProfileBlockedError(record)
    return record


def publication_success_spec(
    system: str,
    objective_profile: str,
    starting_point: str,
) -> PublicationOptimizationSuccessSpec:
    """Return the measurable gate for one provisionable publication row."""
    publication_record(system, objective_profile, starting_point)
    canonical = (
        system != "ferrocene" and objective_profile == REPOSITORY_OBJECTIVE_PROFILE and starting_point == "qfuerza"
    ) or (
        system == "ferrocene"
        and objective_profile == FERROCENE_SEVEN_STRUCTURE_PROFILE
        and starting_point == "published"
    )
    return PublicationOptimizationSuccessSpec(
        system=system,
        objective_profile=objective_profile,
        starting_point=starting_point,
        canonical_full_run=canonical,
        proof_status="blocked_methodology" if canonical else "bounded_software_path",
        methodology_blocker=RELAXED_GEOMETRY_METHODOLOGY_BLOCKER if canonical else None,
    )
