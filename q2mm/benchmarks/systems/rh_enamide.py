"""Rh-enamide: 9 Rh-diphosphine transition-state structures.

Donoghue, Helquist, Norrby & Wiest 2008 (*J. Chem. Theory Comput.* 4,
1313, DOI 10.1021/ct800132a) TSFF reference system. Molecules come from
a MacroModel training set (geometry/atom types) cross-referenced with
Jaguar single-point-energy + frequency calculations (QM Hessians).
"""

from __future__ import annotations


from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.systems._assembly import assemble_published_case
from q2mm.benchmarks.systems._forcefield import load_published_opt
from q2mm.benchmarks.systems._paths import ExternalDataRoots, StartingPoint, natural_sort_key, resolve_rh_enamide_dir
from q2mm.models.molecule import Molecule
from q2mm.models.problem import StationaryPointKind

KEY = "rh-enamide"
NAME = "Rh-enamide"
DESCRIPTION = "9 Rh-diphosphine structures (Jaguar B3LYP/LACVP**)"
DEFAULT_FORMS: tuple[str, ...] = ("mm3",)
METADATA = {
    "level_of_theory": "B3LYP/LACVP**",
    "publication": "Donoghue et al. JCTC 2008, 4, 1313",
    "doi": "10.1021/ct800132a",
}


def load_molecules(*, data_roots: ExternalDataRoots | None = None) -> list[Molecule]:
    """Load 9 rh-enamide structures with Jaguar Hessians.

    Returns:
        list[Molecule]: 9 molecules with Hessian matrices.

    Raises:
        FileNotFoundError: If the rh-enamide dataset is not found.
        ValueError: If the number of MacroModel structures doesn't match
            the number of Jaguar input files.

    """
    from q2mm.io import JaguarIn, MacroModel

    rh_dir = resolve_rh_enamide_dir(data_roots)
    training_set_dir = rh_dir / "rh_enamide_training_set"
    mmo_path = training_set_dir / "rh_enamide_training_set.mmo"
    jag_dir = training_set_dir / "jaguar_spe_freq_in_out"
    if not mmo_path.is_file():
        raise FileNotFoundError(f"Rh-enamide MacroModel training set not found: {mmo_path}")
    if not jag_dir.is_dir():
        raise FileNotFoundError(f"Rh-enamide Jaguar training set not found: {jag_dir}")

    mm = MacroModel(str(mmo_path))
    jag_files = sorted(jag_dir.glob("*.in"), key=natural_sort_key)
    base_molecules = mm.molecules
    n_structures = len(base_molecules)
    n_jag = len(jag_files)
    if n_structures != n_jag:
        raise ValueError(
            f"Rh-enamide dataset inconsistent: {n_structures} MacroModel structures "
            f"but {n_jag} Jaguar .in files in {jag_dir}"
        )

    molecules = []
    for mol, jag_path in zip(base_molecules, jag_files):
        jag = JaguarIn(str(jag_path))
        molecules.append(jag.attach_hessian(mol))
    return molecules


def load(
    *,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str | None = None,
) -> BenchmarkCase:
    """Build the Rh-enamide :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Args:
        data_roots: Explicit locations for the (non-distributed)
            Rh-enamide dataset; falls back to ``Q2MM_RH_ENAMIDE`` when
            omitted — see :class:`~q2mm.benchmarks.systems._paths.ExternalDataRoots`.
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
    ff_path = resolve_rh_enamide_dir(resolved_roots) / "mm3.fld"
    composed_ff, opt_only_ff = load_published_opt(ff_path)
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
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
    )
