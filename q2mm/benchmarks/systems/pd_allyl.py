"""Pd-allyl amination: 21 Pd-catalyzed allylic amination transition-state structures.

Wahlers et al. *Nat. Commun.* 2021, 12, 6508 (DOI 10.1038/s41467-021-27065-2)
TSFF reference system. Ships as a standalone OPT-substructure .fld that must
be composed with the licensed MM3 base force field (see
:func:`~q2mm.benchmarks.systems._forcefield.compose_opt_with_mm3_base`).
"""

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

KEY = "pd-allyl"
NAME = "Pd-allyl amination"
DESCRIPTION = "21 Pd-allyl TS structures (Wahlers, Nat. Commun. 2021)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
METADATA = {
    "level_of_theory": "M06/gen+pseudo (GD3)",
    "publication": "Wahlers et al. Nat. Commun. 2021, 12, 6508",
    "doi": "10.1038/s41467-021-27065-2",
}
METAL = "PD"
_CHAPTER = "Chapter 3"
_OPT_FILENAME = "mm3.Pd-allyl.fld"


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load the 21 Pd-allyl amination TS molecules (Wahlers Ch 3)."""
    return load_gaussian_molecules(resolve_wahlers_training_set_dir(_CHAPTER, data_roots))


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE,
) -> BenchmarkCase:
    """Build the Pd-allyl :class:`~q2mm.benchmarks.cases.BenchmarkCase`."""
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
    )
