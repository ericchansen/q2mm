"""Wahlers Chapter 4 Ferrocene ground-state force-field training system."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.publications import REPOSITORY_OBJECTIVE_PROFILE, publication_record
from q2mm.benchmarks.systems._assembly import assemble_published_case
from q2mm.benchmarks.systems._forcefield import compose_opt_with_mm3_base
from q2mm.benchmarks.systems._molecules import load_gaussian_molecules
from q2mm.benchmarks.systems._paths import (
    ExternalDataRoots,
    StartingPoint,
    resolve_mm3_base_path,
    resolve_supporting_info_dir,
    resolve_wahlers_opt_path,
)
from q2mm.models.molecule import Molecule
from q2mm.models.problem import StationaryPointKind

KEY = "ferrocene"
NAME = "Ferrocene ground-state force field"
DESCRIPTION = "7 Ferrocene ground-state structures (Wahlers Chapter 4)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
CASE_IDS: tuple[str, ...] = ("TS1", "TS2", "TS3", "TS4", "TS5", "TS6", "TS7")
METADATA = {
    "level_of_theory": "B3LYP-D3/LANL2DZ(Fe)/6-31G**",
    "publication": "Wahlers et al. J. Org. Chem. 2022, 87, 12334",
    "doi": "10.1021/acs.joc.2c01553",
    "dissertation": "Wahlers, Ph.D. Dissertation, University of Notre Dame, 2021, Ch. 4",
}
METAL = "FE"
_CHAPTER = "Chapter 4"
_OPT_FILENAME = "mm3.ferrocene.fld"


def _training_set_dir(roots: ExternalDataRoots | None) -> Path:
    si = resolve_supporting_info_dir(roots)
    return si / "wahlers" / "Wahlers_Jessica_Supporting_information" / _CHAPTER / "Training Set Structures"


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load exactly the seven source training logs in numeric semantic order."""
    molecules = load_gaussian_molecules(_training_set_dir(data_roots))
    by_name = {molecule.name: molecule for molecule in molecules}
    missing = [case_id for case_id in CASE_IDS if case_id not in by_name]
    extras = sorted(set(by_name).difference(CASE_IDS))
    if missing or extras:
        raise ValueError(
            f"Ferrocene Chapter 4 training membership must be exactly TS1-TS7; missing={missing}, extras={extras}."
        )
    return [by_name[case_id] for case_id in CASE_IDS]


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "published",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE,
) -> BenchmarkCase:
    """Build the provisionable seven-structure published-start partial profile."""
    publication_metadata = publication_record(KEY, objective_profile, starting_point)
    resolved_roots = ExternalDataRoots() if data_roots is None else data_roots
    molecules = load_molecules(data_roots=resolved_roots)
    composed_ff, opt_only_ff = compose_opt_with_mm3_base(
        resolve_wahlers_opt_path(_CHAPTER, _OPT_FILENAME, resolved_roots),
        resolve_mm3_base_path(resolved_roots),
        metal=METAL,
    )
    composed_ff = dataclasses.replace(composed_ff, nonbonded_excluded_atom_types=("FE",))
    return assemble_published_case(
        key=KEY,
        name=NAME,
        molecules=molecules,
        composed_ff=composed_ff,
        opt_only_ff=opt_only_ff,
        stationary_point=StationaryPointKind.GROUND_STATE,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
        functional_form=functional_form,
        metadata=METADATA,
        publication_metadata=publication_metadata,
        metal=METAL,
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
        case_ids=CASE_IDS,
    )
