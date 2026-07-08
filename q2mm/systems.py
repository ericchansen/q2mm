"""Benchmark system configurations.

Each :class:`SystemSpec` declaratively describes one benchmark system —
its molecule source, FF-assembly strategy, and metadata — and the
:data:`SYSTEMS` registry maps a CLI key to that spec.  The single
public entry point is :func:`load_system`, which dispatches to the
right molecule loader and the right FF strategy based on the spec.

Adding a new system = appending a :class:`SystemSpec` entry to
:data:`SYSTEMS`; no new module-level function is required.

Usage::

    from q2mm.systems import load_system, SYSTEMS

    sys_data = load_system("rh-enamide", engine=engine)

The FF-assembly strategies live in :mod:`q2mm.models.loaders` and
correspond to the published-FF workflows in Farrugia, Helquist, Norrby
& Wiest 2025 (the QFUERZA paper — see AGENTS.md "Key Papers").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from collections.abc import Callable, Mapping

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
        reference: Reference data for the objective function.
        qm_freqs_per_mol: QM real frequencies per molecule (for reporting only).
            These frequencies are stored separately from ``reference``.
        metadata: Extra info (level of theory, molecule name, etc.).
        normal_modes: Pre-computed normal modes for PES distortion analysis.
            ``None`` when not available.

    """

    molecules: list[Q2MMMolecule]
    forcefield: ForceField
    reference: ReferenceData
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

_REPO_ROOT = Path(__file__).resolve().parent.parent


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


# ---------------------------------------------------------------------------
# Shared helper for supporting-info based systems
# ---------------------------------------------------------------------------


def _resolve_supporting_info_dir() -> Path:
    """Return the root of the supporting-info directory."""
    import os

    si_env = os.environ.get("Q2MM_SUPPORTING_INFO")
    if si_env:
        return Path(si_env)
    return Path(__file__).resolve().parent.parent / "validation" / "supporting-info"


def _load_gaussian_molecules(log_dir: Path, *, bond_tolerance: float = 1.3) -> list[Q2MMMolecule]:
    """Load molecules from all Gaussian .log files in a directory.

    Reads the archive **Cartesian** Hessian (Hartree/Bohr², full rank 3N,
    imaginary mode intact) in a frame consistent with the geometry.
    Assigns MM3 atom types from element + connectivity (bond count).
    """
    from q2mm.io.gaussian import GaussLog

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
        log = GaussLog(str(log_path), au_hessian=True)
        mol = log.molecules[-1]
        mol.name = log_path.stem

        # Detect bonds and assign MM3 atom types from connectivity.
        # Gaussian logs only carry element symbols; MM3 engines need
        # typed atoms (C2/C3, H1, N2, etc.) for parameter matching.
        mol.bond_tolerance = bond_tolerance
        mol._bonds = None  # force re-detection with new tolerance
        _assign_mm3_atom_types(mol)
        # Invalidate topology caches so that bonds/angles/torsions are
        # re-detected with the updated MM3 atom_types (env_id depends
        # on atom_types).  Without this, cached bonds keep element-only
        # env_id like "C-C" instead of "C2-C3".
        mol.invalidate_topology()

        molecules.append(mol)

    return molecules


# Elements treated as metals for bond-count purposes.  Bonds to metals
# are excluded when counting organic hybridization (a C bonded to Pd +
# 2 C neighbours is sp2, not sp3).
_METAL_ELEMENTS: frozenset[str] = frozenset(
    {
        "Pd",
        "Rh",
        "Ru",
        "Ir",
        "Fe",
        "Os",
        "Pt",
        "Ni",
        "Cu",
        "Zn",
        "Co",
    }
)

# MM3 atom-type assignment from element + bond count.
#
# Keys are element symbols; values map the number of *organic* bonds
# (i.e. bonds to non-metal atoms) to an MM3 type string.  Metal atoms
# always use bond-count 0 because ``_assign_mm3_atom_types`` excludes
# metal-metal and metal-organic bonds from the count.
#
# Examples:
#   C with 4 organic bonds → C3 (sp3)
#   C with 3 organic bonds → C2 (sp2)
#   N with 2 organic bonds → N2
#   Pd (any)               → PD  (0-bond fallback)
#
# Source: standard MM3 atom-type conventions; see Allinger, Yuh & Lii,
# J. Am. Chem. Soc. 1989, 111, 8551.
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
    """Assign MM3 atom types from element + organic bond count.

    Metal bonds are excluded from the bond count for non-metal atoms
    so that e.g. a carbon bonded to Pd + 2 carbons is typed C2 (sp2),
    not C3 (sp3).  Metal atoms themselves use the zero-bond-count
    fallback in ``_MM3_TYPE_MAP``.
    """
    # Build adjacency: for non-metal atoms, count only bonds to other
    # non-metals.  For metal atoms, use 0 (the map fallback).
    organic_bond_counts: dict[int, int] = {}
    for b in mol.bonds:
        sym_i, sym_j = mol.symbols[b.atom_i], mol.symbols[b.atom_j]
        i_metal = sym_i in _METAL_ELEMENTS
        j_metal = sym_j in _METAL_ELEMENTS
        if not i_metal and not j_metal:
            organic_bond_counts[b.atom_i] = organic_bond_counts.get(b.atom_i, 0) + 1
            organic_bond_counts[b.atom_j] = organic_bond_counts.get(b.atom_j, 0) + 1
        elif not i_metal:
            # i is organic, j is metal — don't count for i
            pass
        elif not j_metal:
            # j is organic, i is metal — don't count for j
            pass

    for i, elem in enumerate(mol.symbols):
        n_bonds = organic_bond_counts.get(i, 0)
        type_map = _MM3_TYPE_MAP.get(elem, {})
        if n_bonds in type_map:
            mm3_type = type_map[n_bonds]
        elif type_map:
            mm3_type = type_map.get(0, next(iter(type_map.values())))
        else:
            mm3_type = elem
        mol.atom_types[i] = mm3_type


# ---------------------------------------------------------------------------
# Wahlers-system molecule loaders
# ---------------------------------------------------------------------------


def _load_wahlers_molecules(chapter_subdir: str) -> list[Q2MMMolecule]:
    """Load Wahlers-dissertation Gaussian molecules from a chapter subdirectory."""
    si = _resolve_supporting_info_dir()
    chapter = si / "wahlers" / "Wahlers_Jessica_Supporting_information" / chapter_subdir
    ts_dir = chapter / "Training Set Structures"
    if not ts_dir.exists():
        ts_dir = chapter / "DFT-optmized training set structures"
    return _load_gaussian_molecules(ts_dir)


def load_pd_allyl_molecules() -> list[Q2MMMolecule]:
    """Load the 21 Pd-allyl amination TS molecules (Wahlers Ch 3)."""
    return _load_wahlers_molecules("Chapter 3")


def load_pd_conjugate_molecules() -> list[Q2MMMolecule]:
    """Load the 10 Pd 1,4-conjugate addition TS molecules (Wahlers Ch 5)."""
    return _load_wahlers_molecules("Chapter 5")


def load_rh_conjugate_molecules() -> list[Q2MMMolecule]:
    """Load the 10 Rh 1,4-conjugate addition TS molecules (Wahlers Ch 6)."""
    return _load_wahlers_molecules("Chapter 6")


def _load_ch3f_molecules(*, data_dir: Path | None = None) -> list[Q2MMMolecule]:
    """Load the single CH3F molecule + its QM Hessian (B3LYP/6-31+G(d))."""
    from q2mm.models.molecule import Q2MMMolecule

    qm_dir = data_dir or _find_ch3f_data_dir()
    xyz = qm_dir / "ch3f-optimized.xyz"
    hess_path = qm_dir / "ch3f-hessian.npy"
    molecule = Q2MMMolecule.from_xyz(xyz, bond_tolerance=1.5)
    return [molecule.with_hessian(np.load(hess_path))]


def _load_ch3f_sn2_molecules(*, data_dir: Path | None = None) -> list[Q2MMMolecule]:
    """Load the F⁻ + CH3F SN2 transition state + its QM Hessian.

    The D3h-symmetric TS of the identity SN2 reaction
    F⁻ + CH3F → FCH3 + F⁻ at B3LYP/6-31+G(d) (one imaginary mode
    at ≈ −462 cm⁻¹, corresponding to the asymmetric C-F stretch
    along the reaction coordinate).  Bond tolerance is set to ``1.5``
    (a unitless multiplier on the sum of covalent radii — see
    :meth:`Q2MMMolecule.from_xyz`) to include the partially-formed
    C-F bonds at the TS geometry (~1.85 Å each); the default ``1.3``
    misses them.  Charge is set to −1 to match the anionic complex
    on which the QM Hessian was computed (see
    ``examples/sn2-test/generate_qm_data.py``).

    This is the test case Limé & Norrby 2015 (J. Comput. Chem. 36,
    244) used to demonstrate the FACAF bend force constant going
    negative under naive Method C fitting and to motivate the
    Method E2 hybrid protocol — see ``MethodE2Workflow`` (planned).
    """
    from q2mm.models.molecule import Q2MMMolecule

    qm_dir = data_dir or _find_ch3f_data_dir()
    xyz = qm_dir / "sn2-ts-optimized.xyz"
    hess_path = qm_dir / "sn2-ts-hessian.npy"
    # ``_find_ch3f_data_dir`` only checks for the GS ``ch3f-optimized.xyz``;
    # the SN2 TS files (computed by ``generate_qm_data.py`` after the
    # GS calc) may be absent in partial checkouts.  Surface a targeted
    # error rather than let ``from_xyz`` or ``np.load`` emit a less
    # actionable message downstream.
    missing = [p.name for p in (xyz, hess_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"SN2 TS reference data missing in {qm_dir}: {missing}. "
            "Run ``examples/sn2-test/generate_qm_data.py`` to compute the TS "
            "Hessian + frequencies, or pass ``data_dir=`` pointing at a "
            "complete reference directory."
        )
    molecule = Q2MMMolecule.from_xyz(xyz, charge=-1, bond_tolerance=1.5)
    return [molecule.with_hessian(np.load(hess_path))]


# ---------------------------------------------------------------------------
# Wahlers-system FF paths
# ---------------------------------------------------------------------------


def _wahlers_opt_path(chapter_subdir: str, ff_filename: str) -> Path:
    """Resolve the path to a Wahlers chapter's standalone OPT .fld file."""
    si = _resolve_supporting_info_dir()
    return si / "wahlers" / "Wahlers_Jessica_Supporting_information" / chapter_subdir / ff_filename


def _mm3_base_path() -> Path:
    """Resolve the standard MM3 base .fld file.

    The base file (mm3_base.fld) is not committed due to copyright;
    fall back to examples/rh-enamide/mm3.fld which includes the same
    base section.
    """
    p = _REPO_ROOT / "validation" / "published_ffs" / "mm3_base.fld"
    if p.exists():
        return p
    p = _REPO_ROOT / "examples" / "rh-enamide" / "mm3.fld"
    if p.exists():
        return p
    raise FileNotFoundError(
        "MM3 base force field not found. Place mm3_base.fld in "
        "validation/published_ffs/ (download from atlas-nano/ATLAS_toolkit) "
        "or ensure examples/rh-enamide/mm3.fld exists."
    )


# ---------------------------------------------------------------------------
# SystemSpec and load_system dispatch
# ---------------------------------------------------------------------------


FFStrategy = Literal[
    "qfuerza_fresh",
    "published_opt",
    "published_opt_composed",
]
"""Names of the FF-assembly strategies in :mod:`q2mm.models.loaders`."""


@dataclass(frozen=True)
class SystemSpec:
    """Declarative spec for one benchmark system.

    The :func:`load_system` dispatcher reads this spec to build a
    :class:`SystemData` for the system.  Add new systems by appending
    to :data:`SYSTEMS`; do not write per-system loader functions.

    Attributes:
        key: CLI key (e.g. ``"ch3f"``, ``"rh-enamide"``).
        name: Human-readable system name.
        molecule_loader: Zero-argument callable returning the training-
            set molecules (with QM Hessians for the published-FF
            systems).
        ff_strategy: Name of the FF-assembly strategy in
            :mod:`q2mm.models.loaders`.  One of
            ``"qfuerza_fresh"``, ``"published_opt"``,
            ``"published_opt_composed"``.
        ff_paths: Mapping of strategy-specific path keys to zero-arg
            callables returning a :class:`Path`.  Required keys per
            strategy:

            - ``qfuerza_fresh``: no keys.
            - ``published_opt``: ``"ff_path"`` → published .fld.
            - ``published_opt_composed``: ``"opt_path"`` → standalone
              Wahlers OPT-only .fld; ``"base_path"`` → MM3 base .fld.
        normal_modes_path: Optional callable returning a path to a
            ``.npz`` file with pre-computed normal-mode eigendecomposition
            (used by PES distortion analysis).  Accepts the same
            ``data_dir`` override the CLI may pass via
            :func:`load_system`'s ``molecule_loader_kwargs`` so molecule,
            Hessian, and normal modes stay co-located.  Return ``None``
            (or a non-existent path) to signal "no modes available".
        metadata: Static metadata merged into the returned
            :class:`SystemData.metadata` (level of theory, publication,
            etc.).
        metal: Optional element symbol for vdW injection during
            ``published_opt_composed`` (e.g. ``"PD"``).
        description: One-line CLI description.
        default_forms: Functional forms to benchmark by default.

    """

    key: str
    name: str
    molecule_loader: Callable[..., list[Q2MMMolecule]]
    ff_strategy: FFStrategy
    ff_paths: Mapping[str, Callable[[], Path]] = field(default_factory=dict)
    normal_modes_path: Callable[[Path | None], Path | None] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    metal: str | None = None
    description: str = ""
    default_forms: tuple[str, ...] = ("mm3",)


StartingPoint = Literal["published", "qfuerza"]
"""Choice of starting-parameter values for the optimizer.

- ``"qfuerza"`` (default, canonical): take the FF skeleton from the
  ``.fld`` file (atom-type rows, OPT-substructure topology, frozen/
  active partition, vdW, stretch-bend, Urey-Bradley) and overwrite the
  OPT bond and angle scalars with QFUERZA Hessian-derived estimates
  averaged across the training molecules (Farrugia 2025 protocol).
  This is the standard QFUERZA workflow as defined in the literature:
  the chemist provides the FF skeleton (no tool can automate the
  decisions of which atom types to use or which substructure rows
  need OPT parameters); QFUERZA fills in the Hessian-derivable
  scalars.  Frozen MM3 backbone parameters are untouched; torsions
  are zeroed; OPT parameters that :func:`qfuerza_into` does not
  estimate (stretch-bends, vdW, Urey-Bradley) retain their literature
  values — the audit in :func:`load_system` records this explicitly.
- ``"published"``: use the literature OPT values from the ``.fld``
  file(s) as-is, with no QFUERZA overwrite.  This is the
  publication-baseline path used to reproduce historical convergence
  runs.

For ``qfuerza_fresh`` strategy (CH3F), ``"qfuerza"`` is a no-op
because the FF is already QFUERZA-derived; the audit records this.
"""


def _build_param_type_labels(ff: ForceField) -> list[str]:
    """Generate per-scalar type labels matching :meth:`ForceField.get_param_vector` layout.

    Derived from :data:`ForceField._PARAM_SLOTS` (plus the Urey-Bradley tail)
    so that any future change to the parameter vector layout will be reflected
    here automatically.  Unknown collection or slot attribute names raise
    ``KeyError`` rather than silently producing wrong labels.

    Labels follow the convention ``{singular_collection}_{short_attr}``
    (e.g. ``"bond_fc"``, ``"vdw_radius"``, ``"ub_eq"``).
    """
    collection_prefix = {
        "bonds": "bond",
        "angles": "angle",
        "torsions": "torsion",
        "stretch_bends": "stretch_bend",
        "vdws": "vdw",
    }
    attr_short = {
        "force_constant": "fc",
        "equilibrium": "eq",
        "radius": "radius",
        "epsilon": "epsilon",
    }
    labels: list[str] = []
    for collection_attr, slot_attrs in ff._PARAM_SLOTS:  # noqa: SLF001 — schema is the source of truth
        prefix = collection_prefix[collection_attr]
        per_item = [f"{prefix}_{attr_short[slot]}" for slot in slot_attrs]
        for _ in getattr(ff, collection_attr):
            labels.extend(per_item)
    # Urey-Bradley tail mirrors the ordering inside ForceField.get_param_vector().
    labels.extend(["ub_fc", "ub_eq"] * len(ff._ub_angles))  # noqa: SLF001 — schema is the source of truth
    return labels


def _audit_starting_point(
    ff: ForceField,
    *,
    before_vec: np.ndarray | None,
    starting_point: StartingPoint,
) -> dict[str, Any]:
    """Classify every scalar param as qfuerza/retained_published/frozen.

    Honest accounting of where the starting values come from.  For
    ``starting_point="qfuerza"`` (canonical default) we diff the
    parameter vector before vs after :func:`qfuerza_into` and call any
    active scalar whose value changed ``qfuerza_overwritten``; any
    active scalar whose value did not change is ``retained_published``
    (e.g. an OPT stretch-bend, an active bond/angle that QFUERZA could
    not match to any training molecule).  For
    ``starting_point="published"`` everything active is
    ``retained_published``.

    Args:
        ff: Force field *after* any QFUERZA overwrite.
        before_vec: Param vector snapshot from before the overwrite,
            or ``None`` for the published case.
        starting_point: The starting-point choice.

    Returns:
        A nested dict ``{starting_point, n_active, n_frozen, by_type:
        {bond_fc: {qfuerza_overwritten, retained_published, frozen}, …}}``.

    """
    after_vec = ff.get_param_vector()
    active = ff.active_mask

    type_labels = _build_param_type_labels(ff)

    if len(type_labels) != len(after_vec):
        raise AssertionError(f"type label / param vector length mismatch: {len(type_labels)} vs {len(after_vec)}")

    by_type: dict[str, dict[str, int]] = {}
    for label, is_active, after_val, before_val in zip(
        type_labels,
        active,
        after_vec,
        before_vec if before_vec is not None else after_vec,
        strict=True,
    ):
        bucket = by_type.setdefault(
            label,
            {"qfuerza_overwritten": 0, "retained_published": 0, "frozen": 0},
        )
        if not is_active:
            bucket["frozen"] += 1
        elif before_vec is not None and not np.isclose(before_val, after_val, rtol=0.0, atol=1e-12):
            bucket["qfuerza_overwritten"] += 1
        else:
            bucket["retained_published"] += 1

    return {
        "starting_point": starting_point,
        "n_active": int(active.sum()),
        "n_frozen": int((~active).sum()),
        "by_type": by_type,
    }


def load_system(
    key: str,
    *,
    engine: Any | None = None,
    functional_form: str | None = None,
    molecule_loader_kwargs: dict[str, Any] | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
) -> SystemData:
    """Build a :class:`SystemData` for one benchmark system.

    The single loader entry point.  Dispatches on the system's
    :class:`SystemSpec` to build the molecule list and the force
    field, then constructs the reference data and metadata.

    Reference construction depends on the FF strategy:

    - ``qfuerza_fresh`` (single-molecule, e.g. CH3F): a frequency-only
      reference is built from ``engine.frequencies(molecule, ff)``
      compared against the QM frequencies derived from the Hessian.
      The *engine* must be provided in this case.
    - ``published_opt`` / ``published_opt_composed`` (multi-molecule
      published-FF systems): an eigenmatrix-diagonal reference is built
      via :meth:`ReferenceData.from_molecules`.  The engine is unused.

    Args:
        key: System key from :data:`SYSTEMS`.
        engine: MM engine instance; required for ``qfuerza_fresh``
            systems (CH3F-style).  Unused for other strategies.
        functional_form: Override the FF's functional form
            (``"harmonic"`` or ``"mm3"``).
        molecule_loader_kwargs: Optional kwargs forwarded to the
            system's molecule loader (e.g. ``data_dir`` for CH3F).
        starting_point: Where the starting OPT parameter values come
            from.  See :data:`StartingPoint`.  ``"qfuerza"`` is the
            canonical default: it overwrites OPT bond/angle values with
            multi-molecule QFUERZA estimates after FF assembly (per
            Farrugia 2025).  ``"published"`` retains the literature OPT
            values verbatim — pass this to reproduce the historical
            publication-baseline runs.  CH3F (``qfuerza_fresh``
            strategy) treats ``"qfuerza"`` as a no-op since the FF is
            already QFUERZA-derived.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most negative TS-Hessian eigenvalue during QFUERZA
            projection.  Default ``1.0`` matches Limé & Norrby Method C
            and preserves historical numbers.  Smaller values (e.g.
            ``0.03``, Method D's "natural" value) reduce the chance of
            negative angle force constants in the starting FF but
            produce a softer reaction-coordinate mode.  Applied
            whenever QFUERZA invokes ``invert_ts_curvature`` — i.e.
            for the ``qfuerza_fresh`` strategy regardless of
            ``starting_point``, and for the published-FF strategies
            only when ``starting_point="qfuerza"`` triggers a QFUERZA
            overwrite.  Has no effect for published-FF strategies with
            ``starting_point="published"`` (no QFUERZA invocation
            occurs) or for any ground-state Hessian (no negative
            eigenvalue to replace).

    Returns:
        A fully-populated :class:`SystemData`.  The
        ``metadata["starting_point_audit"]`` block reports, for each
        parameter type, how many active scalars were QFUERZA-overwritten
        vs retained from the published FF vs frozen — see
        :func:`_audit_starting_point`.

    Raises:
        KeyError: If *key* is not in :data:`SYSTEMS`.
        TypeError: If *engine* is required but not provided.
        ValueError: If *starting_point* is not one of ``"published"`` or ``"qfuerza"``.

    """
    from q2mm.models import loaders
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.optimizers.objective import ReferenceData

    if key not in SYSTEMS:
        raise KeyError(f"Unknown system {key!r}; available: {sorted(SYSTEMS)}")
    if starting_point not in ("published", "qfuerza"):
        raise ValueError(f"Unknown starting_point {starting_point!r}; must be one of: 'published', 'qfuerza'")
    spec = SYSTEMS[key]

    # 1. Molecules ---------------------------------------------------------
    molecules = spec.molecule_loader(**(molecule_loader_kwargs or {}))

    # 2. Force field via the named strategy --------------------------------
    if spec.ff_strategy == "qfuerza_fresh":
        if len(molecules) != 1:
            raise ValueError(
                f"qfuerza_fresh strategy requires exactly 1 molecule, got {len(molecules)} for system {key!r}"
            )
        ff = loaders.load_qfuerza_fresh(molecules[0], replace_with=qfuerza_replace_with)
    elif spec.ff_strategy == "published_opt":
        ff = loaders.load_published_opt(spec.ff_paths["ff_path"]())
    elif spec.ff_strategy == "published_opt_composed":
        ff = loaders.load_published_opt_composed(
            spec.ff_paths["opt_path"](),
            spec.ff_paths["base_path"](),
            metal=spec.metal,
        )
    else:  # pragma: no cover — unreachable per FFStrategy literal
        raise ValueError(f"Unknown ff_strategy {spec.ff_strategy!r} for system {key!r}")

    # Functional form: explicit override > strategy default (MM3)
    if functional_form is not None:
        ff.functional_form = FunctionalForm(functional_form)

    # 2b. Optional QFUERZA overwrite of OPT parameter values --------------
    # For ``starting_point="qfuerza"`` we keep the published FF
    # topology + frozen/active partition but replace the active
    # OPT bond/angle values with multi-molecule QFUERZA estimates
    # (Farrugia 2025 §"Methods", per-molecule mean averaging).
    # ``qfuerza_into`` honours the frozen partition, so MM3 backbone
    # rows are never touched.  Active rows that QFUERZA cannot match
    # to any training molecule retain their published values — the
    # audit captures this so the leak is visible downstream.
    from q2mm.models.seminario import qfuerza_into

    before_vec: np.ndarray | None = None
    if starting_point == "qfuerza" and spec.ff_strategy != "qfuerza_fresh":
        before_vec = ff.get_param_vector().copy()
        qfuerza_into(
            ff,
            molecules,
            invert_ts_curvature=True,
            replace_with=qfuerza_replace_with,
        )
    starting_point_audit = _audit_starting_point(ff, before_vec=before_vec, starting_point=starting_point)

    # 3. Reference data ----------------------------------------------------
    if spec.ff_strategy == "qfuerza_fresh":
        # Frequency-only reference: requires the engine to compute MM
        # frequencies against the QM ones derived from the Hessian.
        if engine is None:
            raise TypeError(
                f"System {key!r} uses ff_strategy='qfuerza_fresh' which requires "
                "engine= to build the frequency-only reference."
            )
        molecule = molecules[0]
        qm_freqs_all = _qm_frequencies_from_hessian(molecule.hessian, molecule.symbols)
        mm_all = engine.frequencies(molecule, ff)
        reference, qm_real = _build_frequency_reference(qm_freqs_all, mm_all)
        qm_freqs_per_mol = [qm_real]
    else:
        reference = ReferenceData.from_molecules(molecules, eigenmatrix_diagonal_only=False)
        qm_freqs_per_mol = []
        for mol in molecules:
            qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
            qm_real = np.array(sorted(f for f in qm_freqs if f > 50.0))
            qm_freqs_per_mol.append(qm_real)

    # 4. Optional normal-modes side-load (declared per-spec).  Pass the
    #    CLI's data_dir override (if any) through so molecule, Hessian,
    #    and normal modes stay co-located.
    normal_modes: dict[str, np.ndarray] | None = None
    if spec.normal_modes_path is not None:
        data_dir_override = (molecule_loader_kwargs or {}).get("data_dir")
        modes_path = spec.normal_modes_path(data_dir_override)
        if modes_path is not None and modes_path.exists():
            from q2mm.diagnostics.pes_distortion import load_normal_modes

            normal_modes = load_normal_modes(modes_path)

    # 5. Wrap in SystemData ------------------------------------------------
    resolved_form = functional_form
    if resolved_form is None and ff.functional_form is not None:
        resolved_form = ff.functional_form.value
    metadata = {
        "molecule_name": spec.name,
        "n_molecules": len(molecules),
        "n_atoms_per_mol": [len(m.symbols) for m in molecules],
        "starting_point": starting_point,
        "starting_point_audit": starting_point_audit,
        **dict(spec.metadata),
        **({"functional_form": resolved_form} if resolved_form else {}),
    }
    return SystemData(
        molecules=molecules,
        forcefield=ff,
        reference=reference,
        qm_freqs_per_mol=qm_freqs_per_mol,
        metadata=metadata,
        normal_modes=normal_modes,
    )


# ---------------------------------------------------------------------------
# System registry
# ---------------------------------------------------------------------------


def _heck_relay_ff_path() -> Path:
    """Resolve the Heck-relay Rosales FF file path lazily."""
    si = _resolve_supporting_info_dir()
    return si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"


def _ch3f_normal_modes_path(data_dir_override: Path | None) -> Path:
    """Resolve the CH3F normal-modes ``.npz`` path, honouring a CLI override."""
    base = data_dir_override or _find_ch3f_data_dir()
    return base / "ch3f-normal-modes.npz"


def _ch3f_sn2_normal_modes_path(data_dir_override: Path | None) -> Path:
    """Resolve the F⁻ + CH3F SN2 TS normal-modes ``.npz`` path."""
    base = data_dir_override or _find_ch3f_data_dir()
    return base / "sn2-ts-normal-modes.npz"


SYSTEMS: dict[str, SystemSpec] = {
    "ch3f": SystemSpec(
        key="ch3f",
        name="CH3F",
        molecule_loader=_load_ch3f_molecules,
        ff_strategy="qfuerza_fresh",
        normal_modes_path=_ch3f_normal_modes_path,
        metadata={"level_of_theory": "B3LYP/6-31+G(d)"},
        description="Single CH3F molecule (SN2 test, B3LYP/6-31+G(d))",
        default_forms=("harmonic", "mm3"),
    ),
    "ch3f-sn2": SystemSpec(
        key="ch3f-sn2",
        name="F⁻ + CH3F SN2 TS",
        molecule_loader=_load_ch3f_sn2_molecules,
        ff_strategy="qfuerza_fresh",
        normal_modes_path=_ch3f_sn2_normal_modes_path,
        metadata={
            "level_of_theory": "B3LYP/6-31+G(d)",
            "publication": "Limé & Norrby, J. Comput. Chem. 2015, 36, 244",
            "doi": "10.1002/jcc.23797",
            "is_transition_state": True,
            "imaginary_mode_freq_cm": -461.7,
        },
        description=(
            "F⁻ + CH3F → FCH3 + F⁻ identity SN2 transition state "
            "(D3h, B3LYP/6-31+G(d)). Limé & Norrby 2015's canonical "
            "test case for Method E2 (FACAF bend force constant goes "
            "to zero/negative under naive Method C fitting). One "
            "imaginary mode ≈ −462 cm⁻¹ along the asymmetric C-F stretch."
        ),
        default_forms=("harmonic", "mm3"),
    ),
    "rh-enamide": SystemSpec(
        key="rh-enamide",
        name="Rh-enamide",
        molecule_loader=load_rh_enamide_molecules,
        ff_strategy="published_opt",
        ff_paths={"ff_path": lambda: _RH_DIR / "mm3.fld"},
        metadata={
            "level_of_theory": "B3LYP/LACVP**",
            "publication": "Donoghue et al. JCTC 2008, 4, 1313",
            "doi": "10.1021/ct800132a",
        },
        description="9 Rh-diphosphine structures (Jaguar B3LYP/LACVP**)",
    ),
    "heck-relay": SystemSpec(
        key="heck-relay",
        name="Heck relay",
        molecule_loader=load_heck_relay_molecules,
        ff_strategy="published_opt",
        ff_paths={"ff_path": _heck_relay_ff_path},
        metadata={
            "level_of_theory": "M06/gen+pseudo (GD3)",
            "publication": "Rosales et al. JACS 2020, 142, 9700",
            "doi": "10.1021/jacs.0c01979",
        },
        description="23 Pd-catalyzed redox-relay Heck TS structures (Gaussian M06/HPModes)",
    ),
    "pd-allyl": SystemSpec(
        key="pd-allyl",
        name="Pd-allyl amination",
        molecule_loader=load_pd_allyl_molecules,
        ff_strategy="published_opt_composed",
        ff_paths={
            "opt_path": lambda: _wahlers_opt_path("Chapter 3", "mm3.Pd-allyl.fld"),
            "base_path": _mm3_base_path,
        },
        metadata={
            "level_of_theory": "M06/gen+pseudo (GD3)",
            "publication": "Wahlers et al. Nat. Commun. 2021, 12, 6508",
            "doi": "10.1038/s41467-021-27065-2",
        },
        metal="PD",
        description="21 Pd-allyl TS structures (Wahlers, Nat. Commun. 2021)",
    ),
    "pd-conjugate": SystemSpec(
        key="pd-conjugate",
        name="Pd 1,4-conjugate addition",
        molecule_loader=load_pd_conjugate_molecules,
        ff_strategy="published_opt_composed",
        ff_paths={
            "opt_path": lambda: _wahlers_opt_path("Chapter 5", "mm3.Pd-1,4.fld"),
            "base_path": _mm3_base_path,
        },
        metadata={
            "level_of_theory": "M06/gen+pseudo (GD3)",
            "publication": "Wahlers et al. J. Org. Chem. 2021, 86, 5660",
            "doi": "10.1021/acs.joc.1c00136",
        },
        metal="PD",
        description="10 Pd 1,4-conjugate TS structures (Wahlers, J. Org. Chem. 2021)",
    ),
    "rh-conjugate": SystemSpec(
        key="rh-conjugate",
        name="Rh 1,4-conjugate addition",
        molecule_loader=load_rh_conjugate_molecules,
        ff_strategy="published_opt_composed",
        ff_paths={
            "opt_path": lambda: _wahlers_opt_path("Chapter 6", "mm3.Rh-1,4.fld"),
            "base_path": _mm3_base_path,
        },
        metadata={
            "level_of_theory": "M06/gen+pseudo (GD3)",
            "publication": "Wahlers, J. Ph.D. Dissertation, U. Notre Dame, 2022, Ch. 6",
        },
        metal="RH",
        description="10 Rh 1,4-conjugate TS structures (Wahlers thesis)",
    ),
}
