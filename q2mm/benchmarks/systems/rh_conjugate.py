"""Rh 1,4-conjugate addition: 10 Rh-catalyzed transition-state structures.

Wahlers, J. Ph.D. Dissertation, University of Notre Dame, 2021, Ch. 6.
Ships as a standalone OPT-substructure .fld that must be composed with
the licensed MM3 base force field (see
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

KEY = "rh-conjugate"
NAME = "Rh 1,4-conjugate addition"
DESCRIPTION = "10 Rh 1,4-conjugate TS structures (Wahlers thesis)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
METADATA = {
    "level_of_theory": "M06/gen+pseudo (GD3)",
    "publication": "Wahlers, J. Ph.D. Dissertation, U. Notre Dame, 2021, Ch. 6",
    "doi": "10.7274/k930bv76q4n",
}
METAL = "RH"
_CHAPTER = "Chapter 6"
_OPT_FILENAME = "mm3.Rh-1,4.fld"


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load the 10 Rh 1,4-conjugate addition TS molecules (Wahlers Ch 6)."""
    return load_gaussian_molecules(resolve_wahlers_training_set_dir(_CHAPTER, data_roots))


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE,
) -> BenchmarkCase:
    """Build the Rh 1,4-conjugate :class:`~q2mm.benchmarks.cases.BenchmarkCase`."""
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
