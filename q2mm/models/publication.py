"""Immutable publication-reproduction metadata for scientific problem provenance.

This module supplies one path-free vocabulary for saying exactly what a
publication benchmark does and does not reproduce.  It is deliberately a
domain model rather than a catalog of repository systems: benchmark modules
own the concrete records, while :class:`PublicationMetadata` can accompany an
arbitrary :class:`~q2mm.models.problem.OptimizationProblem`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum

from q2mm._canonical import canonical_fingerprint

__all__ = [
    "ReproductionStatus",
    "ObjectiveTargetDisposition",
    "PublicationTargetCategory",
    "ObjectiveProfileIdentity",
    "PublicationCitation",
    "SourceArtifactIdentity",
    "ObjectiveTarget",
    "PublicationMetadata",
]


class ReproductionStatus(str, Enum):
    """Canonical claim boundary for one publication profile."""

    EXACT_PUBLICATION_REPRODUCTION = "exact_publication_reproduction"
    EXECUTABLE_ARCHIVE_REPRODUCTION = "executable_archive_reproduction"
    PARTIAL_REPOSITORY_REPRODUCTION = "partial_repository_reproduction"
    SDK_SOFTWARE_PATH_DEMONSTRATION = "sdk_software_path_demonstration"
    BLOCKED_HISTORICAL_RECORD = "blocked_historical_record"


class ObjectiveTargetDisposition(str, Enum):
    """How one governing-source target is represented by a selected profile."""

    IMPLEMENTED = "implemented"
    AVAILABLE = "available"
    OMITTED = "omitted"
    BLOCKED = "blocked"


class PublicationTargetCategory(str, Enum):
    """Closed publication-objective category vocabulary."""

    BOND_LENGTH = "bond_length"
    BOND_ANGLE = "bond_angle"
    TORSION_GEOMETRY = "torsion_geometry"
    EIGENMATRIX_DIAGONAL = "eigenmatrix_diagonal"
    EIGENMATRIX_OFFDIAGONAL = "eigenmatrix_offdiagonal"
    ATOMIC_PARTIAL_CHARGE = "atomic_partial_charge"
    DIRECT_ELECTROSTATIC_POTENTIAL = "direct_electrostatic_potential"
    RELATIVE_ENERGY = "relative_energy"
    RELATIVE_ENTHALPY = "relative_enthalpy"
    CONSTRAINED_SCAN_ENERGY = "constrained_scan_energy"
    EQUILIBRIUM_PARAMETER_TETHER = "equilibrium_parameter_tether"


def _nonempty(value: str, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string.")
    return normalized


def _path_free(value: str, field_name: str) -> str:
    normalized = _nonempty(value, field_name)
    if re.match(r"^[A-Za-z]:[\\/]", normalized) or normalized.startswith(("/", "\\")):
        raise ValueError(f"{field_name} must be a path-free identity, not an absolute path.")
    return normalized


@dataclass(frozen=True)
class ObjectiveProfileIdentity:
    """Stable name and integer version of an objective profile."""

    name: str
    version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty(self.name, "ObjectiveProfileIdentity.name"))
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("ObjectiveProfileIdentity.version must be an integer >= 1.")

    @property
    def identifier(self) -> str:
        """Return the canonical ``<name>-v<version>`` identifier."""
        return f"{self.name}-v{self.version}"


@dataclass(frozen=True)
class PublicationCitation:
    """One governing article or dissertation citation."""

    citation: str
    authoritative_url: str
    doi: str | None = None
    zotero_key: str | None = None
    chapter: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "citation", _nonempty(self.citation, "PublicationCitation.citation"))
        url = _nonempty(self.authoritative_url, "PublicationCitation.authoritative_url")
        if not url.startswith("https://"):
            raise ValueError("PublicationCitation.authoritative_url must be an HTTPS URL.")
        object.__setattr__(self, "authoritative_url", url)
        if self.doi is not None:
            object.__setattr__(self, "doi", _nonempty(self.doi, "PublicationCitation.doi"))
        if self.zotero_key is not None:
            key = _nonempty(self.zotero_key, "PublicationCitation.zotero_key")
            if not re.fullmatch(r"[A-Z0-9]{8}", key):
                raise ValueError("PublicationCitation.zotero_key must be an eight-character Zotero key.")
            object.__setattr__(self, "zotero_key", key)
        if self.chapter is not None:
            object.__setattr__(self, "chapter", _nonempty(self.chapter, "PublicationCitation.chapter"))


@dataclass(frozen=True)
class SourceArtifactIdentity:
    """Path-free identity and optional content fingerprint for source evidence."""

    identity: str
    role: str
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _path_free(self.identity, "SourceArtifactIdentity.identity"))
        object.__setattr__(self, "role", _nonempty(self.role, "SourceArtifactIdentity.role"))
        if self.fingerprint is not None:
            fingerprint = _nonempty(self.fingerprint, "SourceArtifactIdentity.fingerprint")
            if ":" not in fingerprint:
                raise ValueError("SourceArtifactIdentity.fingerprint must use an '<algorithm>:<digest>' form.")
            algorithm, digest = fingerprint.split(":", 1)
            if not algorithm or not digest or not re.fullmatch(r"[A-Za-z0-9._+-]+", algorithm):
                raise ValueError("SourceArtifactIdentity.fingerprint is malformed.")
            object.__setattr__(self, "fingerprint", fingerprint.lower())


@dataclass(frozen=True)
class ObjectiveTarget:
    """Completeness record for one governing-source objective category."""

    category: PublicationTargetCategory
    disposition: ObjectiveTargetDisposition
    profile_weight: float | None = None
    source_weight: float | None = None
    unit: str | None = None
    details: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.category, PublicationTargetCategory):
            raise TypeError("ObjectiveTarget.category must be a PublicationTargetCategory.")
        if not isinstance(self.disposition, ObjectiveTargetDisposition):
            raise TypeError("ObjectiveTarget.disposition must be an ObjectiveTargetDisposition.")
        for field_name in ("profile_weight", "source_weight"):
            value = getattr(self, field_name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"ObjectiveTarget.{field_name} must be finite and non-negative or None.")
        if self.unit is not None:
            object.__setattr__(self, "unit", _nonempty(self.unit, "ObjectiveTarget.unit"))
        object.__setattr__(self, "details", str(self.details).strip())
        if self.disposition is ObjectiveTargetDisposition.IMPLEMENTED and self.profile_weight is None:
            raise ValueError("Implemented ObjectiveTarget entries require profile_weight.")


@dataclass(frozen=True)
class PublicationMetadata:
    """Complete, immutable claim and objective-completeness record."""

    system: str
    status: ReproductionStatus
    objective_profile: ObjectiveProfileIdentity
    governing_sources: tuple[PublicationCitation, ...]
    authoritative_case_ids: tuple[str, ...]
    stationary_point: str
    targets: tuple[ObjectiveTarget, ...]
    source_artifacts: tuple[SourceArtifactIdentity, ...] = ()
    force_field_blocks: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    starting_point: str | None = None
    provisionable: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "system", _nonempty(self.system, "PublicationMetadata.system"))
        if not isinstance(self.status, ReproductionStatus):
            raise TypeError("PublicationMetadata.status must be a ReproductionStatus.")
        if not isinstance(self.objective_profile, ObjectiveProfileIdentity):
            raise TypeError("PublicationMetadata.objective_profile must be an ObjectiveProfileIdentity.")
        sources = tuple(self.governing_sources)
        if not sources or not all(isinstance(source, PublicationCitation) for source in sources):
            raise ValueError("PublicationMetadata.governing_sources must contain at least one PublicationCitation.")
        case_ids = tuple(
            _nonempty(value, "PublicationMetadata.authoritative_case_ids") for value in self.authoritative_case_ids
        )
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("PublicationMetadata.authoritative_case_ids must be unique.")
        if self.provisionable and not case_ids:
            raise ValueError("Provisionable PublicationMetadata requires authoritative case IDs.")
        stationary_point = _nonempty(self.stationary_point, "PublicationMetadata.stationary_point")
        if stationary_point not in {"ground_state", "transition_state", "unknown"}:
            raise ValueError("PublicationMetadata.stationary_point is not canonical.")
        targets = tuple(self.targets)
        if not targets or not all(isinstance(target, ObjectiveTarget) for target in targets):
            raise ValueError("PublicationMetadata.targets must contain at least one ObjectiveTarget.")
        categories = tuple(target.category for target in targets)
        if len(set(categories)) != len(categories):
            raise ValueError("PublicationMetadata.targets may contain each category only once.")
        artifacts = tuple(self.source_artifacts)
        if not all(isinstance(artifact, SourceArtifactIdentity) for artifact in artifacts):
            raise TypeError("PublicationMetadata.source_artifacts must contain SourceArtifactIdentity values.")
        blocks = tuple(_path_free(value, "PublicationMetadata.force_field_blocks") for value in self.force_field_blocks)
        blockers = tuple(_nonempty(value, "PublicationMetadata.blockers") for value in self.blockers)
        notes = tuple(_nonempty(value, "PublicationMetadata.notes") for value in self.notes)
        if self.starting_point is not None and self.starting_point not in {"published", "qfuerza"}:
            raise ValueError("PublicationMetadata.starting_point must be 'published', 'qfuerza', or None.")
        if not isinstance(self.provisionable, bool):
            raise TypeError("PublicationMetadata.provisionable must be bool.")
        if self.status is ReproductionStatus.BLOCKED_HISTORICAL_RECORD:
            if self.provisionable or not blockers:
                raise ValueError(
                    "Blocked publication records must be non-provisionable and state at least one blocker."
                )
        if not isinstance(self.schema_version, int) or self.schema_version != 1:
            raise ValueError("PublicationMetadata.schema_version must be 1.")
        object.__setattr__(self, "governing_sources", sources)
        object.__setattr__(self, "authoritative_case_ids", case_ids)
        object.__setattr__(self, "stationary_point", stationary_point)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "source_artifacts", artifacts)
        object.__setattr__(self, "force_field_blocks", blocks)
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "notes", notes)

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-safe representation."""
        return {
            "schema_version": self.schema_version,
            "system": self.system,
            "status": self.status.value,
            "objective_profile": {
                "name": self.objective_profile.name,
                "version": self.objective_profile.version,
                "identifier": self.objective_profile.identifier,
            },
            "governing_sources": [
                {
                    "citation": source.citation,
                    "authoritative_url": source.authoritative_url,
                    "doi": source.doi,
                    "zotero_key": source.zotero_key,
                    "chapter": source.chapter,
                }
                for source in self.governing_sources
            ],
            "authoritative_case_ids": list(self.authoritative_case_ids),
            "stationary_point": self.stationary_point,
            "targets": [
                {
                    "category": target.category.value,
                    "disposition": target.disposition.value,
                    "profile_weight": target.profile_weight,
                    "source_weight": target.source_weight,
                    "unit": target.unit,
                    "details": target.details,
                }
                for target in self.targets
            ],
            "source_artifacts": [
                {
                    "identity": artifact.identity,
                    "role": artifact.role,
                    "fingerprint": artifact.fingerprint,
                }
                for artifact in self.source_artifacts
            ],
            "force_field_blocks": list(self.force_field_blocks),
            "blockers": list(self.blockers),
            "notes": list(self.notes),
            "starting_point": self.starting_point,
            "provisionable": self.provisionable,
        }

    @property
    def fingerprint(self) -> str:
        """Return a deterministic SHA-256 fingerprint of this path-free record."""
        return canonical_fingerprint(self.to_dict(), strict=True, screen_secrets=True)
