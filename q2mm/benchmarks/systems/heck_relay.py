"""Heck relay: 23 Pd-catalyzed redox-relay Heck transition-state structures.

Rosales, Helquist, Norrby & Wiest 2020 (*J. Am. Chem. Soc.* 142, 9700,
DOI 10.1021/jacs.0c01979) TSFF reference system. Molecules come from
Gaussian M06/HPModes training-set logs.
"""

from __future__ import annotations

from pathlib import Path

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.publications import REPOSITORY_OBJECTIVE_PROFILE, publication_record
from q2mm.benchmarks.systems._assembly import assemble_published_case
from q2mm.benchmarks.systems._forcefield import load_published_opt
from q2mm.benchmarks.systems._molecules import load_gaussian_molecules
from q2mm.benchmarks.systems._paths import ExternalDataRoots, StartingPoint, resolve_supporting_info_dir
from q2mm.models.molecule import Molecule
from q2mm.models.problem import StationaryPointKind

KEY = "heck-relay"
NAME = "Heck relay"
DESCRIPTION = "23 Pd-catalyzed redox-relay Heck TS structures (Gaussian M06/HPModes)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
METADATA = {
    "level_of_theory": "M06/gen+pseudo (GD3)",
    "publication": "Rosales et al. JACS 2020, 142, 9700",
    "doi": "10.1021/jacs.0c01979",
}


def _ff_path(roots: ExternalDataRoots | None) -> Path:
    si = resolve_supporting_info_dir(roots)
    return si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load the 23 Heck relay TS molecules from Gaussian logs.

    Returns:
        List of Molecule with Hessians and MM3 atom types assigned.

    Raises:
        FileNotFoundError: If the training set directory is absent.

    """
    si = resolve_supporting_info_dir(data_roots)
    ts_dir = si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "TrainingSet"
    return load_gaussian_molecules(ts_dir)


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE,
) -> BenchmarkCase:
    """Build the Heck-relay :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Args:
        data_roots: Explicit locations for the (non-distributed)
            dissertation supporting information; falls back to
            ``Q2MM_SUPPORTING_INFO`` when omitted.
        starting_point: ``"qfuerza"`` (default, Farrugia 2025) overwrites
            active OPT bond/angle values with multi-molecule QFUERZA
            estimates; ``"published"`` keeps the literature OPT values
            verbatim.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative TS-Hessian eigenvalue during QFUERZA
            projection (Limé & Norrby Method C; default ``1.0``).
        functional_form: Optional override (``"harmonic"`` or ``"mm3"``).
        objective_profile: Canonical publication objective-profile identifier.

    Returns:
        A fully-populated :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    """
    publication_metadata = publication_record(KEY, objective_profile, starting_point)
    resolved_roots = ExternalDataRoots() if data_roots is None else data_roots
    molecules = load_molecules(data_roots=resolved_roots)
    composed_ff, opt_only_ff = load_published_opt(_ff_path(resolved_roots))
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
        publication_metadata=publication_metadata,
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
    )
