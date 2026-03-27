"""Benchmark system configurations.

Each :class:`BenchmarkSystem` describes a molecular system with its reference
data, force field template, and metadata.  The :data:`SYSTEMS` registry maps
system names to their configs, making it easy to add new benchmark targets.

Usage::

    from q2mm.diagnostics.systems import SYSTEMS, BenchmarkSystem

    system = SYSTEMS["rh-enamide"]
    molecules, qm_freqs, ... = system.load()
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
        forcefield: Template force field (from Seminario or file).
        freq_ref: Frequency-based reference data for the objective function.
        qm_freqs_per_mol: QM real frequencies per molecule (for reporting).
        metadata: Extra info (level of theory, molecule name, etc.).

    """

    molecules: list[Q2MMMolecule]
    forcefield: ForceField
    freq_ref: ReferenceData
    qm_freqs_per_mol: list[np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


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


def load_ch3f(engine: Any, *, data_dir: Path | None = None) -> SystemData:
    """Load the CH3F benchmark system.

    Args:
        engine: MM engine instance (used for frequency computation).
        data_dir: Override for the QM reference data directory.

    Returns:
        SystemData with a single CH3F molecule.

    """
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.models.seminario import estimate_force_constants

    qm_dir = data_dir or _find_ch3f_data_dir()
    xyz = qm_dir / "ch3f-optimized.xyz"
    hess_path = qm_dir / "ch3f-hessian.npy"
    freqs_path = qm_dir / "ch3f-frequencies.txt"

    molecule = Q2MMMolecule.from_xyz(xyz, bond_tolerance=1.5)
    qm_freqs_all = np.loadtxt(freqs_path)
    qm_hessian = np.load(hess_path)

    mol_h = molecule.with_hessian(qm_hessian)
    ff = estimate_force_constants(mol_h)

    mm_all = engine.frequencies(molecule, ff)
    freq_ref, qm_real = _build_frequency_reference(qm_freqs_all, mm_all)

    return SystemData(
        molecules=[molecule],
        forcefield=ff,
        freq_ref=freq_ref,
        qm_freqs_per_mol=[qm_real],
        metadata={
            "molecule_name": "CH3F",
            "level_of_theory": "B3LYP/6-31+G(d)",
            "n_atoms": len(molecule.symbols),
        },
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
    from q2mm.parsers import JaguarIn, MacroModel

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


def load_rh_enamide(engine: Any) -> SystemData:
    """Load the Rh-enamide benchmark system (9 molecules).

    Args:
        engine: MM engine instance (used for frequency computation).

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

    # Seminario produces harmonic force constants regardless of the template's
    # functional form.  Switch to HARMONIC so JAX/JAX-MD engines (which only
    # support harmonic) can use the result.  OpenMM handles both forms and
    # defaults to MM3 when functional_form is None, so HARMONIC is safe there too.
    supported = getattr(engine, "supported_functional_forms", lambda: frozenset())()
    if supported and FunctionalForm.MM3.value not in supported:
        ff.functional_form = FunctionalForm.HARMONIC

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
        },
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

    """

    name: str
    key: str
    loader: Callable
    description: str = ""


SYSTEMS: dict[str, BenchmarkSystem] = {
    "ch3f": BenchmarkSystem(
        name="CH3F",
        key="ch3f",
        loader=load_ch3f,
        description="Single CH3F molecule (SN2 test, B3LYP/6-31+G(d))",
    ),
    "rh-enamide": BenchmarkSystem(
        name="Rh-enamide",
        key="rh-enamide",
        loader=load_rh_enamide,
        description="9 Rh-diphosphine structures (Jaguar B3LYP/LACVP**)",
    ),
}
