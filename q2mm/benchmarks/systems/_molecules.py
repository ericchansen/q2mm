"""Molecule loading and MM3 atom-typing helpers for benchmark systems.

Shared by every system module that trains on Gaussian-log molecules
(heck-relay, pd-allyl, pd-conjugate, rh-conjugate): batch-loading a
directory of Gaussian ``.log`` archives into :class:`Molecule` objects,
and assigning MM3 atom types from element + connectivity (needed for
parameter matching against the standard MM3 backbone).
"""

from __future__ import annotations

from pathlib import Path

from q2mm.models.molecule import Molecule

# ---------------------------------------------------------------------------
# MM3 atom typing (element + organic bond count)
# ---------------------------------------------------------------------------

# Elements treated as metals for bond-count purposes.  Bonds to metals
# are excluded when counting organic hybridization (a C bonded to Pd +
# 2 C neighbours is sp2, not sp3).
_METAL_ELEMENTS: frozenset[str] = frozenset({"Pd", "Rh", "Ru", "Ir", "Fe", "Os", "Pt", "Ni", "Cu", "Zn", "Co"})

# MM3 atom-type assignment from element + bond count.
#
# Keys are element symbols; values map the number of *organic* bonds
# (i.e. bonds to non-metal atoms) to an MM3 type string.  Metal atoms
# always use bond-count 0 because ``assign_mm3_atom_types`` excludes
# metal-metal and metal-organic bonds from the count.
#
# Examples:
#   C with 4 organic bonds -> C3 (sp3)
#   C with 3 organic bonds -> C2 (sp2)
#   N with 2 organic bonds -> N2
#   Pd (any)               -> PD  (0-bond fallback)
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


def assign_mm3_atom_types(mol: Molecule) -> list[str]:
    """Compute MM3 atom types from element + organic bond count.

    Metal bonds are excluded from the bond count for non-metal atoms
    so that e.g. a carbon bonded to Pd + 2 carbons is typed C2 (sp2),
    not C3 (sp3).  Metal atoms themselves use the zero-bond-count
    fallback in ``_MM3_TYPE_MAP``.

    Returns:
        list[str]: New atom-type labels, one per atom, in the same order as
        ``mol.symbols``. Does not mutate *mol* — pass the result to
        :meth:`~q2mm.models.molecule.Molecule.with_atom_types`.

    """
    organic_bond_counts: dict[int, int] = {}
    for b in mol.bonds or ():
        sym_i, sym_j = mol.symbols[b.atom_i], mol.symbols[b.atom_j]
        i_metal = sym_i in _METAL_ELEMENTS
        j_metal = sym_j in _METAL_ELEMENTS
        if not i_metal and not j_metal:
            organic_bond_counts[b.atom_i] = organic_bond_counts.get(b.atom_i, 0) + 1
            organic_bond_counts[b.atom_j] = organic_bond_counts.get(b.atom_j, 0) + 1

    atom_types = []
    for i, elem in enumerate(mol.symbols):
        n_bonds = organic_bond_counts.get(i, 0)
        type_map = _MM3_TYPE_MAP.get(elem, {})
        if n_bonds in type_map:
            mm3_type = type_map[n_bonds]
        elif type_map:
            mm3_type = type_map.get(0, next(iter(type_map.values())))
        else:
            mm3_type = elem
        atom_types.append(mm3_type)
    return atom_types


def load_gaussian_molecules(log_dir: Path, *, bond_tolerance: float = 1.3) -> list[Molecule]:
    """Load molecules from all Gaussian .log files in a directory.

    Reads the archive **Cartesian** Hessian (Hartree/Bohr², full rank 3N,
    imaginary mode intact) in a frame consistent with the geometry.
    Assigns MM3 atom types from element + connectivity (bond count).
    """
    from q2mm.io.gaussian import GaussLog

    if not log_dir.exists():
        raise FileNotFoundError(
            f"Training set not found: {log_dir}\n"
            "Check the configured ExternalDataRoots.supporting_info path or Q2MM_SUPPORTING_INFO."
        )

    log_files = sorted(log_dir.glob("*.log"))
    if not log_files:
        raise FileNotFoundError(f"No Gaussian logs found in {log_dir}")

    molecules = []
    for log_path in log_files:
        log = GaussLog(str(log_path), au_hessian=True)
        mol = log.molecules[-1]
        mol = mol.with_overrides(name=log_path.stem, bond_tolerance=bond_tolerance)

        # Detect bonds and assign MM3 atom types from connectivity.
        # Gaussian logs only carry element symbols; MM3 engines need
        # typed atoms (C2/C3, H1, N2, etc.) for parameter matching.
        # ``with_atom_types`` recomputes bonds/angles/torsions with the
        # updated MM3 atom_types (env_id depends on atom_types) — without
        # that, bonds would keep an element-only env_id like "C-C" instead
        # of "C2-C3".
        mol = mol.with_atom_types(assign_mm3_atom_types(mol))

        molecules.append(mol)

    return molecules
