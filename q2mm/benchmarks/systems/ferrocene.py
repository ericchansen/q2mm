"""Wahlers Chapter 4 Ferrocene ground-state force-field training system."""

from __future__ import annotations

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.publications import REPOSITORY_OBJECTIVE_PROFILE
from q2mm.benchmarks.systems._assembly import assemble_wahlers_case
from q2mm.benchmarks.systems._molecules import load_gaussian_molecules
from q2mm.benchmarks.systems._paths import (
    ExternalDataRoots,
    StartingPoint,
    resolve_wahlers_training_set_dir,
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


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load exactly the seven source training logs in numeric semantic order."""
    molecules = load_gaussian_molecules(resolve_wahlers_training_set_dir(_CHAPTER, data_roots))
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
    return assemble_wahlers_case(
        key=KEY,
        name=NAME,
        description=DESCRIPTION,
        default_forms=DEFAULT_FORMS,
        metadata=METADATA,
        chapter=_CHAPTER,
        opt_filename=_OPT_FILENAME,
        metal=METAL,
        molecules_loader=load_molecules,
        data_roots=data_roots,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
        functional_form=functional_form,
        objective_profile=objective_profile,
        stationary_point=StationaryPointKind.GROUND_STATE,
        nonbonded_excluded_atom_types=("FE",),
        case_ids=CASE_IDS,
    )
