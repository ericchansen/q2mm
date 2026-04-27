"""Benchmark system configurations.

Each :class:`BenchmarkSystem` describes a molecular system with its reference
data, force field template, and metadata.  The :data:`SYSTEMS` registry maps
system names to their configs, making it easy to add new benchmark targets.

Usage::

    from q2mm.diagnostics.systems import SYSTEMS, BenchmarkSystem

    system = SYSTEMS["rh-enamide"]
    sys_data = system.loader(engine)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from collections.abc import Callable

import numpy as np

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.optimizers.objective import ReferenceData


@dataclass(frozen=True)
class SystemData:
    """Loaded data for a benchmark system, ready for optimization.

    Attributes:
        molecules: One or more molecules with geometry (and optionally Hessians).
        forcefield: Template force field (from QFUERZA estimation or file).
        freq_ref: Frequency-based reference data for the objective function.
        qm_freqs_per_mol: QM real frequencies per molecule (for reporting).
        metadata: Extra info (level of theory, molecule name, etc.).
        normal_modes: Pre-computed normal modes for PES distortion analysis.
            ``None`` when not available.

    """

    molecules: list[Q2MMMolecule]
    forcefield: ForceField
    freq_ref: ReferenceData
    qm_freqs_per_mol: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    normal_modes: dict[str, np.ndarray] | None = None


def _qm_frequencies_from_hessian(
    hessian_au: np.ndarray,
    symbols: list[str],
) -> np.ndarray:
    """Compute harmonic frequencies (cm⁻¹) from a Cartesian Hessian in AU.

    Delegates to :func:`q2mm.models.hessian.hessian_to_frequencies`.
    """
    from q2mm.models.hessian import hessian_to_frequencies

    return np.array(hessian_to_frequencies(hessian_au, symbols, sort=False))


def _build_frequency_reference(
    qm_freqs: np.ndarray,
    mm_all_freqs: np.ndarray,
    *,
    threshold: float = 50.0,
    weight: float = 0.001,
    molecule_idx: int = 0,
    ref: ReferenceData | None = None,
) -> tuple[ReferenceData, np.ndarray]:
    """Build (or extend) a ReferenceData with frequency observations."""
    from q2mm.optimizers.objective import ReferenceData as RefCls

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))

    if ref is None:
        ref = RefCls()
    for k in range(n):
        ref.add_frequency(
            float(qm_real[k]),
            data_idx=mm_real_idx[k],
            weight=weight,
            molecule_idx=molecule_idx,
        )
    return ref, np.array(qm_real[:n])


# ---------------------------------------------------------------------------
# Loader: CH3F (single molecule, SN2 test reference data)
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_ch3f_data_dir() -> Path:
    """Locate CH3F reference data directory."""
    candidates = [
        _REPO_ROOT / "examples" / "sn2-test" / "qm-reference",
        Path.cwd() / "examples" / "sn2-test" / "qm-reference",
    ]
    for d in candidates:
        if (d / "ch3f-optimized.xyz").exists():
            return d
    raise FileNotFoundError(
        "Cannot find CH3F reference data (ch3f-optimized.xyz). Run from the repo root or use --data-dir."
    )


def load_ch3f(
    engine: Any,
    *,
    data_dir: Path | None = None,
    functional_form: str | None = None,
) -> SystemData:
    """Load the CH3F benchmark system.

    Args:
        engine: MM engine instance (used for frequency computation).
        data_dir: Override for the QM reference data directory.
        functional_form: Override functional form (e.g. ``"harmonic"``,
            ``"mm3"``).  When ``None``, the form is left unset and the
            engine uses its native default.

    Returns:
        SystemData with a single CH3F molecule.

    """
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.models.seminario import estimate_force_constants

    qm_dir = data_dir or _find_ch3f_data_dir()
    xyz = qm_dir / "ch3f-optimized.xyz"
    hess_path = qm_dir / "ch3f-hessian.npy"
    freqs_path = qm_dir / "ch3f-frequencies.txt"
    modes_path = qm_dir / "ch3f-normal-modes.npz"

    molecule = Q2MMMolecule.from_xyz(xyz, bond_tolerance=1.5)
    qm_freqs_all = np.loadtxt(freqs_path)
    qm_hessian = np.load(hess_path)

    mol_h = molecule.with_hessian(qm_hessian)
    ff = estimate_force_constants(mol_h)

    if functional_form is not None:
        from q2mm.models.forcefield import FunctionalForm

        ff.functional_form = FunctionalForm(functional_form)

    mm_all = engine.frequencies(molecule, ff)
    freq_ref, qm_real = _build_frequency_reference(qm_freqs_all, mm_all)

    # Load normal modes for PES distortion analysis (optional)
    normal_modes = None
    if modes_path.exists():
        from q2mm.diagnostics.pes_distortion import load_normal_modes

        normal_modes = load_normal_modes(modes_path)

    # Resolve functional form for metadata: explicit override > ff value
    resolved_form = functional_form
    if resolved_form is None and ff.functional_form is not None:
        resolved_form = ff.functional_form.value

    return SystemData(
        molecules=[molecule],
        forcefield=ff,
        freq_ref=freq_ref,
        qm_freqs_per_mol=[qm_real],
        metadata={
            "molecule_name": "CH3F",
            "level_of_theory": "B3LYP/6-31+G(d)",
            "n_atoms": len(molecule.symbols),
            **({"functional_form": resolved_form} if resolved_form else {}),
        },
        normal_modes=normal_modes,
    )


# ---------------------------------------------------------------------------
# Loader: Rh-enamide (9 molecules, Jaguar reference data)
# ---------------------------------------------------------------------------

_RH_DIR = _REPO_ROOT / "examples" / "rh-enamide"
_TRAINING_SET_DIR = _RH_DIR / "rh_enamide_training_set"
_MMO_PATH = _TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
_JAG_DIR = _TRAINING_SET_DIR / "jaguar_spe_freq_in_out"


def _natural_sort_key(p: Path) -> list:
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)]


def load_rh_enamide_molecules() -> list[Q2MMMolecule]:
    """Load 9 rh-enamide structures with Jaguar Hessians.

    This is the shared loader used by both the benchmark CLI and tests.

    Returns:
        list[Q2MMMolecule]: 9 molecules with Hessian matrices.

    Raises:
        FileNotFoundError: If the rh-enamide dataset is not found.
        ValueError: If the number of MacroModel structures doesn't match
            the number of Jaguar input files.

    """
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.io import JaguarIn, MacroModel

    if not _MMO_PATH.exists():
        raise FileNotFoundError(f"Rh-enamide dataset not found: {_MMO_PATH}")

    mm = MacroModel(str(_MMO_PATH))
    jag_files = sorted(_JAG_DIR.glob("*.in"), key=_natural_sort_key)
    n_structures = len(mm.structures)
    n_jag = len(jag_files)
    if n_structures != n_jag:
        raise ValueError(
            f"Rh-enamide dataset inconsistent: {n_structures} MacroModel structures "
            f"but {n_jag} Jaguar .in files in {_JAG_DIR}"
        )

    molecules = []
    for struct, jag_path in zip(mm.structures, jag_files):
        jag = JaguarIn(str(jag_path))
        hess = jag.get_hessian(len(struct.atoms))
        molecules.append(Q2MMMolecule.from_structure(struct, hessian=hess))
    return molecules


def load_rh_enamide(
    engine: Any,
    *,
    functional_form: str | None = None,
) -> SystemData:
    """Load the Rh-enamide benchmark system (9 molecules).

    Args:
        engine: MM engine instance (used for frequency computation).
        functional_form: Override functional form (e.g. ``"harmonic"``,
            ``"mm3"``).  Defaults to ``"mm3"`` since the template is MM3.

    Returns:
        SystemData with 9 Rh-enamide molecules and frequency references.

    """
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import estimate_force_constants

    mm3_path = _RH_DIR / "mm3.fld"
    if not mm3_path.exists():
        raise FileNotFoundError(f"Rh-enamide force field not found: {mm3_path}")

    molecules = load_rh_enamide_molecules()
    ff_template = ForceField.from_mm3_fld(str(mm3_path))
    ff = estimate_force_constants(molecules, forcefield=ff_template)

    # Set functional form: explicit override > default (mm3)
    if functional_form is not None:
        ff.functional_form = FunctionalForm(functional_form)
    else:
        ff.functional_form = FunctionalForm.MM3

    # Build multi-molecule frequency reference
    freq_ref = None
    qm_freqs_per_mol = []
    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )
        qm_freqs_per_mol.append(qm_real)

    return SystemData(
        molecules=molecules,
        forcefield=ff,
        freq_ref=freq_ref,
        qm_freqs_per_mol=qm_freqs_per_mol,
        metadata={
            "molecule_name": "Rh-enamide",
            "level_of_theory": "B3LYP/LACVP**",
            "n_molecules": len(molecules),
            "n_atoms_per_mol": [len(m.symbols) for m in molecules],
            "functional_form": functional_form or "mm3",
        },
    )


# ---------------------------------------------------------------------------
# Heck relay (Rosales 2020, JACS 142, 9700)
# ---------------------------------------------------------------------------


def load_heck_relay_molecules() -> list[Q2MMMolecule]:
    """Load the 23 Heck relay TS molecules from Gaussian logs.

    Returns:
        List of Q2MMMolecule with Hessians and MM3 atom types assigned.

    Raises:
        FileNotFoundError: If the training set directory is absent.

    """
    si = _resolve_supporting_info_dir()
    ts_dir = si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "TrainingSet"
    return _load_gaussian_molecules(ts_dir)


def load_heck_relay(
    engine: Any,
    *,
    functional_form: str | None = None,
) -> SystemData:
    """Load the Heck relay benchmark system (23 TS molecules).

    Uses the published FF from Rosales et al. *JACS* 2020, 142, 9700-9707.
    Training set: 23 Gaussian 09 logs with HPModes frequency data.

    Args:
        engine: MM engine instance (used for frequency computation).
        functional_form: Override functional form.  Defaults to ``"mm3"``.

    Returns:
        SystemData with 23 Heck TS molecules and frequency references.

    """
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import estimate_force_constants

    si = _resolve_supporting_info_dir()
    ff_path = si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"
    if not ff_path.exists():
        raise FileNotFoundError(f"Heck relay FF not found: {ff_path}")

    molecules = load_heck_relay_molecules()
    ff_template = ForceField.from_mm3_fld(str(ff_path))
    ff = estimate_force_constants(molecules, forcefield=ff_template)

    if functional_form is not None:
        ff.functional_form = FunctionalForm(functional_form)
    else:
        ff.functional_form = FunctionalForm.MM3

    # Build multi-molecule frequency reference
    freq_ref = None
    qm_freqs_per_mol = []
    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )
        qm_freqs_per_mol.append(qm_real)

    return SystemData(
        molecules=molecules,
        forcefield=ff,
        freq_ref=freq_ref,
        qm_freqs_per_mol=qm_freqs_per_mol,
        metadata={
            "molecule_name": "Heck relay",
            "level_of_theory": "M06/gen+pseudo (GD3)",
            "n_molecules": len(molecules),
            "n_atoms_per_mol": [len(m.symbols) for m in molecules],
            "functional_form": functional_form or "mm3",
            "publication": "Rosales et al. JACS 2020, 142, 9700",
            "doi": "10.1021/jacs.0c01979",
        },
    )


# ---------------------------------------------------------------------------
# Shared helper for supporting-info based systems
# ---------------------------------------------------------------------------


def _resolve_supporting_info_dir() -> Path:
    """Return the root of the supporting-info directory."""
    import os

    si_env = os.environ.get("Q2MM_SUPPORTING_INFO")
    if si_env:
        return Path(si_env)
    return Path(__file__).resolve().parent.parent.parent / "validation" / "supporting-info"


def _load_gaussian_molecules(log_dir: Path, *, bond_tolerance: float = 1.3) -> list[Q2MMMolecule]:
    """Load molecules from all Gaussian .log files in a directory.

    Reconstructs Hessians from eigenvalues/eigenvectors, handling the
    (3N-6, 3N) → (3N, 3N-6) transpose that Gaussian logs require.
    Assigns MM3 atom types from element + connectivity (bond count).
    """
    from q2mm.io.gaussian import GaussLog
    from q2mm.models.hessian import reform_hessian

    if not log_dir.exists():
        raise FileNotFoundError(
            f"Training set not found: {log_dir}\n"
            "Extract dissertation supporting info into "
            "validation/supporting-info/ or set Q2MM_SUPPORTING_INFO."
        )

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No Gaussian logs found in {log_dir}")

    molecules = []
    for log_path in log_files:
        log = GaussLog(str(log_path))
        mol = log.molecules[-1]
        if log.evals is not None and log.evecs is not None and log.evals.size and log.evecs.size:
            evecs = log.evecs
            if evecs.shape[0] < evecs.shape[1]:
                evecs = evecs.T
            mol.hessian = reform_hessian(log.evals, evecs)
        mol.name = log_path.stem

        # Detect bonds and assign MM3 atom types from connectivity.
        # Gaussian logs only carry element symbols; MM3 engines need
        # typed atoms (C2/C3, H1, N2, etc.) for vdW matching.
        mol.bond_tolerance = bond_tolerance
        mol._bonds = None  # force re-detection with new tolerance
        _assign_mm3_atom_types(mol)

        molecules.append(mol)

    return molecules


# MM3 atom-type assignment from element + bond count.
_MM3_TYPE_MAP: dict[str, dict[int, str]] = {
    "C": {4: "C3", 3: "C2", 2: "C1", 1: "C1"},
    "H": {1: "H1", 0: "H1"},
    "N": {4: "N3", 3: "N2", 2: "N2", 1: "N1"},
    "O": {2: "O3", 1: "O2"},
    "S": {2: "SX", 1: "SX", 3: "SX", 4: "SX"},
    "P": {3: "PX", 4: "PX"},
    "F": {1: "F0"},
    "Cl": {1: "Cl"},
    "Br": {1: "Br"},
    "I": {1: "I0"},
    "Si": {4: "Si"},
    "B": {3: "B3", 4: "B2"},
    "Pd": {0: "PD"},
    "Rh": {0: "RH"},
    "Ru": {0: "RU"},
    "Ir": {0: "IR"},
    "Fe": {0: "FE"},
}


def _assign_mm3_atom_types(mol: Any) -> None:
    """Assign MM3 atom types from element + bond count."""
    bond_counts: dict[int, int] = {}
    for b in mol.bonds:
        bond_counts[b.atom_i] = bond_counts.get(b.atom_i, 0) + 1
        bond_counts[b.atom_j] = bond_counts.get(b.atom_j, 0) + 1

    for i, elem in enumerate(mol.symbols):
        n_bonds = bond_counts.get(i, 0)
        type_map = _MM3_TYPE_MAP.get(elem, {})
        if n_bonds in type_map:
            mm3_type = type_map[n_bonds]
        elif type_map:
            mm3_type = type_map.get(0, next(iter(type_map.values())))
        else:
            mm3_type = elem
        mol.atom_types[i] = mm3_type


def _load_wahlers_system(
    chapter_dir: str,
    ff_filename: str,
    engine: Any,
    *,
    system_name: str,
    level_of_theory: str = "M06/gen+pseudo (GD3)",
    publication: str = "",
    doi: str = "",
    functional_form: str | None = None,
) -> SystemData:
    """Load a Wahlers dissertation system with FF composition.

    Wahlers FFs are standalone OPT-substructure-only files.  This loader
    composes them with the standard MM3 base to produce a complete FF.
    """
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import estimate_force_constants

    si = _resolve_supporting_info_dir()
    wahlers_base = si / "wahlers" / "Wahlers_Jessica_Supporting_information"
    chapter = wahlers_base / chapter_dir

    # Load training set
    ts_dir = chapter / "Training Set Structures"
    if not ts_dir.exists():
        ts_dir = chapter / "DFT-optmized training set structures"
    molecules = _load_gaussian_molecules(ts_dir)

    # Compose FF: standard base + Wahlers OPT overlay.
    # The base file (mm3_base.fld) is not committed due to copyright.
    # Fall back to the rh-enamide mm3.fld which includes the same base section.
    mm3_base_path = _REPO_ROOT / "validation" / "published_ffs" / "mm3_base.fld"
    if not mm3_base_path.exists():
        mm3_base_path = _REPO_ROOT / "examples" / "rh-enamide" / "mm3.fld"
    if not mm3_base_path.exists():
        raise FileNotFoundError(
            "MM3 base force field not found. Place mm3_base.fld in "
            "validation/published_ffs/ (download from atlas-nano/ATLAS_toolkit) "
            "or ensure examples/rh-enamide/mm3.fld exists."
        )
    base_ff = ForceField.from_mm3_fld(str(mm3_base_path))
    opt_ff = ForceField.from_mm3_fld(str(chapter / ff_filename), include_standard=False)

    composed = ForceField(
        bonds=list(opt_ff.bonds) + list(base_ff.bonds),
        angles=list(opt_ff.angles) + list(base_ff.angles),
        torsions=list(opt_ff.torsions) + list(base_ff.torsions),
        vdws=list(opt_ff.vdws) + list(base_ff.vdws),
        stretch_bends=list(opt_ff.stretch_bends) + list(base_ff.stretch_bends),
        functional_form=FunctionalForm.MM3,
    )
    ff = estimate_force_constants(molecules, forcefield=composed)

    if functional_form is not None:
        ff.functional_form = FunctionalForm(functional_form)

    # Build frequency reference
    freq_ref = None
    qm_freqs_per_mol = []
    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )
        qm_freqs_per_mol.append(qm_real)

    return SystemData(
        molecules=molecules,
        forcefield=ff,
        freq_ref=freq_ref,
        qm_freqs_per_mol=qm_freqs_per_mol,
        metadata={
            "molecule_name": system_name,
            "level_of_theory": level_of_theory,
            "n_molecules": len(molecules),
            "n_atoms_per_mol": [len(m.symbols) for m in molecules],
            "functional_form": functional_form or "mm3",
            "publication": publication,
            "doi": doi,
        },
    )


# ---------------------------------------------------------------------------
# Wahlers systems
# ---------------------------------------------------------------------------


def load_pd_allyl(engine: Any, *, functional_form: str | None = None) -> SystemData:
    """Load the Pd-allyl amination system (Wahlers Ch 3, 21 TS structures)."""
    return _load_wahlers_system(
        "Chapter 3",
        "mm3.Pd-allyl.fld",
        engine,
        system_name="Pd-allyl amination",
        publication="Wahlers et al. Nat. Commun. 2021, 12, 6508",
        doi="10.1038/s41467-021-27065-2",
        functional_form=functional_form,
    )


def load_pd_conjugate(engine: Any, *, functional_form: str | None = None) -> SystemData:
    """Load the Pd 1,4-conjugate addition system (Wahlers Ch 5, 10 TS structures)."""
    return _load_wahlers_system(
        "Chapter 5",
        "mm3.Pd-1,4.fld",
        engine,
        system_name="Pd 1,4-conjugate addition",
        publication="Wahlers et al. J. Org. Chem. 2021, 86, 5660",
        doi="10.1021/acs.joc.0c02918",
        functional_form=functional_form,
    )


def load_rh_conjugate(engine: Any, *, functional_form: str | None = None) -> SystemData:
    """Load the Rh 1,4-conjugate addition system (Wahlers Ch 6, 10 TS structures)."""
    return _load_wahlers_system(
        "Chapter 6",
        "mm3.Rh-1,4.fld",
        engine,
        system_name="Rh 1,4-conjugate addition",
        functional_form=functional_form,
    )


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchmarkSystem:
    """Configuration for a benchmark molecular system.

    Attributes:
        name: Human-readable system name.
        key: CLI key (e.g. ``"ch3f"``, ``"rh-enamide"``).
        loader: Callable that takes an engine and returns :class:`SystemData`.
        description: One-line description for ``--list`` output.
        default_forms: Functional forms to benchmark by default.

    """

    name: str
    key: str
    loader: Callable
    description: str = ""
    default_forms: tuple[str, ...] = ("harmonic", "mm3")


SYSTEMS: dict[str, BenchmarkSystem] = {
    "ch3f": BenchmarkSystem(
        name="CH3F",
        key="ch3f",
        loader=load_ch3f,
        description="Single CH3F molecule (SN2 test, B3LYP/6-31+G(d))",
        default_forms=("harmonic", "mm3"),
    ),
    "rh-enamide": BenchmarkSystem(
        name="Rh-enamide",
        key="rh-enamide",
        loader=load_rh_enamide,
        description="9 Rh-diphosphine structures (Jaguar B3LYP/LACVP**)",
        default_forms=("mm3",),
    ),
    "heck-relay": BenchmarkSystem(
        name="Heck relay",
        key="heck-relay",
        loader=load_heck_relay,
        description="23 Pd-catalyzed redox-relay Heck TS structures (Gaussian M06/HPModes)",
        default_forms=("mm3",),
    ),
    "pd-allyl": BenchmarkSystem(
        name="Pd-allyl amination",
        key="pd-allyl",
        loader=load_pd_allyl,
        description="21 Pd-allyl TS structures (Wahlers, Nat. Commun. 2021)",
        default_forms=("mm3",),
    ),
    "pd-conjugate": BenchmarkSystem(
        name="Pd 1,4-conjugate addition",
        key="pd-conjugate",
        loader=load_pd_conjugate,
        description="10 Pd 1,4-conjugate TS structures (Wahlers, J. Org. Chem. 2021)",
        default_forms=("mm3",),
    ),
    "rh-conjugate": BenchmarkSystem(
        name="Rh 1,4-conjugate addition",
        key="rh-conjugate",
        loader=load_rh_conjugate,
        description="10 Rh 1,4-conjugate TS structures (Wahlers thesis)",
        default_forms=("mm3",),
    ),
}
