"""Clean molecular structure representation for Q2MM.

Built on QCElemental for validated molecular data (symbols, geometry,
charge, multiplicity, connectivity) with Q2MM-specific extensions
(Hessian, detected bonds/angles, element-based matching).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    import qcelemental as qcel
    _HAS_QCEL = True
except ImportError:
    qcel = None
    _HAS_QCEL = False


# Covalent radii for bond detection (Angstrom)
COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84,
    "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07,
    "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
    "Br": 1.20, "I": 1.39,
    # Transition metals relevant to Q2MM
    "Rh": 1.42, "Pd": 1.39, "Ru": 1.46, "Ir": 1.41, "Pt": 1.36,
    "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
}


@dataclass
class DetectedBond:
    """A bond detected from molecular geometry."""
    atom_i: int          # 0-based index
    atom_j: int          # 0-based index
    elements: tuple[str, str]
    length: float        # Angstrom

    @property
    def element_pair(self) -> tuple[str, str]:
        """Sorted element pair for matching (e.g., ('C', 'F'))."""
        return tuple(sorted(self.elements))


@dataclass
class DetectedAngle:
    """An angle detected from molecular bonds."""
    atom_i: int          # 0-based (outer)
    atom_j: int          # 0-based (center)
    atom_k: int          # 0-based (outer)
    elements: tuple[str, str, str]
    value: float         # degrees

    @property
    def element_triple(self) -> tuple[str, str, str]:
        """Canonical element triple: (outer, center, outer) sorted by outer elements."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass
class Q2MMMolecule:
    """Q2MM's internal molecular structure representation.

    Wraps atomic symbols, coordinates, charge, and multiplicity with
    auto-detected bonds and angles. Optionally carries a Hessian matrix.

    Can be created from XYZ files, QCElemental molecules, or raw data.
    """
    symbols: list[str]
    geometry: np.ndarray          # Shape (N, 3), Angstrom
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    bond_tolerance: float = 1.3   # Multiplier for bond detection. 1.4+ for TS.
    hessian: np.ndarray | None = None   # Shape (3N, 3N), Hartree/Bohr^2
    _bonds: list[DetectedBond] | None = field(default=None, repr=False)
    _angles: list[DetectedAngle] | None = field(default=None, repr=False)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def bonds(self) -> list[DetectedBond]:
        """Auto-detected bonds from covalent radii."""
        if self._bonds is None:
            self._bonds = self._detect_bonds(self.bond_tolerance)
        return self._bonds

    @property
    def angles(self) -> list[DetectedAngle]:
        """Auto-detected angles from bonds."""
        if self._angles is None:
            self._angles = self._detect_angles()
        return self._angles

    def _detect_bonds(self, tolerance: float = 1.3) -> list[DetectedBond]:
        """Detect bonds based on covalent radii with tolerance factor."""
        bonds = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                ri = COVALENT_RADII.get(self.symbols[i], 0.76)
                rj = COVALENT_RADII.get(self.symbols[j], 0.76)
                dist = np.linalg.norm(self.geometry[i] - self.geometry[j])
                if dist < tolerance * (ri + rj):
                    bonds.append(DetectedBond(
                        atom_i=i, atom_j=j,
                        elements=(self.symbols[i], self.symbols[j]),
                        length=dist,
                    ))
        return bonds

    def _detect_angles(self) -> list[DetectedAngle]:
        """Detect angles from detected bonds."""
        # Build adjacency from bonds
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in self.bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        angles = []
        for center in range(self.n_atoms):
            neighbors = adj[center]
            for ii in range(len(neighbors)):
                for jj in range(ii + 1, len(neighbors)):
                    a, b = neighbors[ii], neighbors[jj]
                    v1 = self.geometry[a] - self.geometry[center]
                    v2 = self.geometry[b] - self.geometry[center]
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle_val = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    angles.append(DetectedAngle(
                        atom_i=a, atom_j=center, atom_k=b,
                        elements=(self.symbols[a], self.symbols[center], self.symbols[b]),
                        value=angle_val,
                    ))
        return angles

    # ---- Factory methods ----

    @classmethod
    def from_xyz(cls, path: str | Path, charge: int = 0,
                 multiplicity: int = 1, name: str = "",
                 bond_tolerance: float = 1.3) -> Q2MMMolecule:
        """Load from XYZ file.

        Args:
            bond_tolerance: Multiplier on sum of covalent radii for bond detection.
                           Use 1.3 for ground states, 1.4-1.5 for transition states
                           with partially formed/broken bonds.
        """
        path = Path(path)
        with open(path) as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        symbols = []
        coords = []
        for line in lines[2:2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return cls(
            symbols=symbols,
            geometry=np.array(coords),
            charge=charge,
            multiplicity=multiplicity,
            name=name or path.stem,
            bond_tolerance=bond_tolerance,
        )

    @classmethod
    def from_qcel(cls, mol: qcel.models.Molecule,
                  name: str = "") -> Q2MMMolecule:
        """Create from a QCElemental Molecule object."""
        if not _HAS_QCEL:
            raise ImportError("qcelemental required: pip install qcelemental")
        coords_bohr = np.array(mol.geometry).reshape(-1, 3)
        coords_ang = coords_bohr * qcel.constants.bohr2angstroms
        return cls(
            symbols=list(mol.symbols),
            geometry=coords_ang,
            charge=mol.molecular_charge,
            multiplicity=mol.molecular_multiplicity,
            name=name,
        )

    def to_qcel(self) -> qcel.models.Molecule:
        """Convert to QCElemental Molecule."""
        if not _HAS_QCEL:
            raise ImportError("qcelemental required: pip install qcelemental")
        coords_bohr = self.geometry / qcel.constants.bohr2angstroms
        conn = [(b.atom_i, b.atom_j, 1) for b in self.bonds]
        return qcel.models.Molecule(
            symbols=self.symbols,
            geometry=coords_bohr.flatten().tolist(),
            molecular_charge=self.charge,
            molecular_multiplicity=self.multiplicity,
            connectivity=conn,
        )

    def with_hessian(self, hessian: np.ndarray) -> Q2MMMolecule:
        """Return a copy with Hessian attached."""
        return Q2MMMolecule(
            symbols=self.symbols,
            geometry=self.geometry.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            name=self.name,
            bond_tolerance=self.bond_tolerance,
            hessian=hessian,
        )

    def __repr__(self) -> str:
        formula = "".join(f"{s}{self.symbols.count(s)}" for s in dict.fromkeys(self.symbols))
        hess_str = f", hessian={self.hessian.shape}" if self.hessian is not None else ""
        return f"Q2MMMolecule({formula}, {self.n_atoms} atoms, {len(self.bonds)} bonds{hess_str})"
