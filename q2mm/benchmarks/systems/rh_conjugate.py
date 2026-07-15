"""Rh 1,4-conjugate addition: 10 Rh-catalyzed transition-state structures.

Wahlers, J. Ph.D. Dissertation, University of Notre Dame, 2022, Ch. 6.
Ships as a standalone OPT-substructure .fld that must be composed with
the licensed MM3 base force field (see
:func:`~q2mm.benchmarks.systems._forcefield.compose_opt_with_mm3_base`).
"""

from __future__ import annotations

from pathlib import Path

from q2mm.benchmarks.cases import BenchmarkCase
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

KEY = "rh-conjugate"
NAME = "Rh 1,4-conjugate addition"
DESCRIPTION = "10 Rh 1,4-conjugate TS structures (Wahlers thesis)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
METADATA = {
    "level_of_theory": "M06/gen+pseudo (GD3)",
    "publication": "Wahlers, J. Ph.D. Dissertation, U. Notre Dame, 2022, Ch. 6",
}
METAL = "RH"
_CHAPTER = "Chapter 6"
_OPT_FILENAME = "mm3.Rh-1,4.fld"


def _training_set_dir(roots: ExternalDataRoots | None) -> Path:
    si = resolve_supporting_info_dir(roots)
    chapter = si / "wahlers" / "Wahlers_Jessica_Supporting_information" / _CHAPTER
    ts_dir = chapter / "Training Set Structures"
    if not ts_dir.exists():
        ts_dir = chapter / "DFT-optmized training set structures"
    return ts_dir


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load the 10 Rh 1,4-conjugate addition TS molecules (Wahlers Ch 6)."""
    return load_gaussian_molecules(_training_set_dir(data_roots))


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
) -> BenchmarkCase:
    """Build the Rh 1,4-conjugate :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Args:
        data_roots: Explicit locations for the (non-distributed)
            dissertation supporting information and licensed MM3 base
            force field; falls back to ``Q2MM_SUPPORTING_INFO`` /
            ``Q2MM_MM3_BASE`` when omitted.
        starting_point: ``"qfuerza"`` (default, Farrugia 2025) overwrites
            active OPT bond/angle values with multi-molecule QFUERZA
            estimates; ``"published"`` keeps the literature OPT values
            verbatim.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative TS-Hessian eigenvalue during QFUERZA
            projection (Limé & Norrby Method C; default ``1.0``).
        functional_form: Optional override (``"harmonic"`` or ``"mm3"``).

    Returns:
        A fully-populated :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    """
    resolved_roots = ExternalDataRoots() if data_roots is None else data_roots
    molecules = load_molecules(data_roots=resolved_roots)
    composed_ff, opt_only_ff = compose_opt_with_mm3_base(
        resolve_wahlers_opt_path(_CHAPTER, _OPT_FILENAME, resolved_roots),
        resolve_mm3_base_path(resolved_roots),
        metal=METAL,
    )
    return assemble_published_case(
        key=KEY,
        name=NAME,
        molecules=molecules,
        composed_ff=composed_ff,
        opt_only_ff=opt_only_ff,
        stationary_point=StationaryPointKind.TRANSITION_STATE,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
        functional_form=functional_form,
        metadata=METADATA,
        metal=METAL,
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
    )
