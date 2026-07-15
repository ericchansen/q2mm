"""XYZ file loading for Q2MM.

Reads a plain XYZ file (atom-count line, comment line, then one
``symbol x y z`` line per atom) into a :class:`~q2mm.models.molecule.Molecule`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q2mm.models.molecule import Molecule


def load_xyz(
    path: str | Path,
    *,
    charge: int = 0,
    multiplicity: int = 1,
    name: str = "",
    bond_tolerance: float = 1.3,
) -> Molecule:
    """Load a molecule from an XYZ file.

    Args:
        path: Path to the XYZ file.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.
        name: Display name; defaults to the file's stem.
        bond_tolerance: Multiplier on the sum of covalent radii for bond
            detection. Use 1.3 for ground states, 1.4-1.5 for transition
            states with partially formed/broken bonds.

    Returns:
        A new :class:`~q2mm.models.molecule.Molecule` built from the XYZ
        geometry, with bonds/angles/torsions inferred from that geometry.

    """
    path = Path(path)
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    symbols = []
    coords = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return Molecule(
        symbols=tuple(symbols),
        atom_types=tuple(symbols),
        geometry=np.array(coords),
        charge=charge,
        multiplicity=multiplicity,
        name=name or path.stem,
        bond_tolerance=bond_tolerance,
    )
