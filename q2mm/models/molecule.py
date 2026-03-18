"""Clean molecular structure representation for Q2MM.

Built on QCElemental for validated molecular data (symbols, geometry,
charge, multiplicity, connectivity) with Q2MM-specific extensions
(Hessian, detected bonds/angles, element-based matching).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)

try:
    import qcelemental as qcel

    _HAS_QCEL = True
except ImportError:
    qcel = None
    _HAS_QCEL = False


# Covalent radii for bond detection (Angstrom)
COVALENT_RADII = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Br": 1.20,
    "I": 1.39,
    # Transition metals relevant to Q2MM
    "Rh": 1.42,
    "Pd": 1.39,
    "Ru": 1.46,
    "Ir": 1.41,
    "Pt": 1.36,
    "Fe": 1.32,
    "Co": 1.26,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
}


@dataclass
class DetectedBond:
    """A bond detected from molecular geometry."""

    atom_i: int  # 0-based index
    atom_j: int  # 0-based index
    elements: tuple[str, str]
    length: float  # Angstrom
    env_id: str = ""
    ff_row: int | None = None

    @property
    def element_pair(self) -> tuple[str, str]:
        """Sorted element pair for matching (e.g., ('C', 'F'))."""
        return tuple(sorted(self.elements))


@dataclass
class DetectedAngle:
    """An angle detected from molecular bonds."""

    atom_i: int  # 0-based (outer)
    atom_j: int  # 0-based (center)
    atom_k: int  # 0-based (outer)
    elements: tuple[str, str, str]
    value: float  # degrees
    env_id: str = ""
    ff_row: int | None = None

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
    geometry: np.ndarray  # Shape (N, 3), Angstrom
    atom_types: list[str] | None = None
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    bond_tolerance: float = 1.3  # Multiplier for bond detection. 1.4+ for TS.
    hessian: np.ndarray | None = None  # Shape (3N, 3N), Hartree/Bohr^2
    _bonds: list[DetectedBond] | None = field(default=None, repr=False)
    _angles: list[DetectedAngle] | None = field(default=None, repr=False)

    def __post_init__(self):
        self.symbols = [str(symbol) for symbol in self.symbols]
        if self.atom_types is None:
            self.atom_types = list(self.symbols)
        else:
            self.atom_types = [str(atom_type) for atom_type in self.atom_types]
        if len(self.atom_types) != len(self.symbols):
            raise ValueError("atom_types must have the same length as symbols.")
        self.geometry = np.asarray(self.geometry, dtype=float)

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
                    bonds.append(
                        DetectedBond(
                            atom_i=i,
                            atom_j=j,
                            elements=(self.symbols[i], self.symbols[j]),
                            length=dist,
                            env_id=canonicalize_bond_env_id([self.atom_types[i], self.atom_types[j]]),
                        )
                    )
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
                    angles.append(
                        DetectedAngle(
                            atom_i=a,
                            atom_j=center,
                            atom_k=b,
                            elements=(self.symbols[a], self.symbols[center], self.symbols[b]),
                            value=angle_val,
                            env_id=canonicalize_angle_env_id(
                                [self.atom_types[a], self.atom_types[center], self.atom_types[b]]
                            ),
                        )
                    )
        return angles

    # ---- Factory methods ----

    @classmethod
    def from_xyz(
        cls, path: str | Path, charge: int = 0, multiplicity: int = 1, name: str = "", bond_tolerance: float = 1.3
    ) -> Q2MMMolecule:
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
        for line in lines[2 : 2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return cls(
            symbols=symbols,
            atom_types=list(symbols),
            geometry=np.array(coords),
            charge=charge,
            multiplicity=multiplicity,
            name=name or path.stem,
            bond_tolerance=bond_tolerance,
        )

    @classmethod
    def from_structure(
        cls,
        structure,
        *,
        charge: int = 0,
        multiplicity: int = 1,
        name: str = "",
        bond_tolerance: float = 1.3,
        hessian: np.ndarray | None = None,
    ) -> Q2MMMolecule:
        """Create from a legacy Structure while preserving bond/angle metadata."""
        symbols = []
        atom_types = []
        coords = []
        for atom in structure.atoms:
            atom_label = atom.atom_type_name or atom.element or ""
            symbols.append(_extract_element(atom_label))
            atom_types.append(atom_label.strip() or _extract_element(atom_label))
            coords.append(atom.coords)

        bonds = []
        for bond in structure.bonds:
            atoms = structure.get_atoms_in_DOF(bond)
            dof_atom_types = [atom.atom_type_name or atom.element or "" for atom in atoms]
            elements = tuple(_extract_element(atom_type) for atom_type in dof_atom_types[:2])
            length = bond.value
            if length is None:
                length = np.linalg.norm(atoms[0].coords - atoms[1].coords)
            bonds.append(
                DetectedBond(
                    atom_i=bond.atom_nums[0] - 1,
                    atom_j=bond.atom_nums[1] - 1,
                    elements=elements,
                    length=float(length),
                    env_id=canonicalize_bond_env_id(dof_atom_types),
                    ff_row=bond.ff_row,
                )
            )

        angles = []
        for angle in structure.angles:
            atoms = structure.get_atoms_in_DOF(angle)
            dof_atom_types = [atom.atom_type_name or atom.element or "" for atom in atoms]
            elements = tuple(_extract_element(atom_type) for atom_type in dof_atom_types[:3])
            angle_value = angle.value
            if angle_value is None:
                v1 = atoms[0].coords - atoms[1].coords
                v2 = atoms[2].coords - atoms[1].coords
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle_value = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            angles.append(
                DetectedAngle(
                    atom_i=angle.atom_nums[0] - 1,
                    atom_j=angle.atom_nums[1] - 1,
                    atom_k=angle.atom_nums[2] - 1,
                    elements=elements,
                    value=float(angle_value),
                    env_id=canonicalize_angle_env_id(dof_atom_types),
                    ff_row=angle.ff_row,
                )
            )

        return cls(
            symbols=symbols,
            atom_types=atom_types,
            geometry=np.array(coords, dtype=float),
            charge=charge,
            multiplicity=multiplicity,
            name=name or structure.origin_name,
            bond_tolerance=bond_tolerance,
            hessian=hessian,
            _bonds=bonds,
            _angles=angles,
        )

    @classmethod
    def from_qcel(cls, mol: qcel.models.Molecule, name: str = "") -> Q2MMMolecule:
        """Create from a QCElemental Molecule object."""
        if not _HAS_QCEL:
            raise ImportError("qcelemental required: pip install qcelemental")
        coords_bohr = np.array(mol.geometry).reshape(-1, 3)
        coords_ang = coords_bohr * qcel.constants.bohr2angstroms
        return cls(
            symbols=list(mol.symbols),
            atom_types=list(mol.symbols),
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
        kwargs = {
            "symbols": self.symbols,
            "geometry": coords_bohr.flatten().tolist(),
            "molecular_charge": self.charge,
            "molecular_multiplicity": self.multiplicity,
        }
        if conn:
            kwargs["connectivity"] = conn
        return qcel.models.Molecule(**kwargs)

    def with_hessian(self, hessian: np.ndarray) -> Q2MMMolecule:
        """Return a copy with Hessian attached."""
        return Q2MMMolecule(
            symbols=self.symbols,
            atom_types=list(self.atom_types),
            geometry=self.geometry.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            name=self.name,
            bond_tolerance=self.bond_tolerance,
            hessian=hessian,
            _bonds=copy.deepcopy(self._bonds) if self._bonds is not None else None,
            _angles=copy.deepcopy(self._angles) if self._angles is not None else None,
        )

    def __repr__(self) -> str:
        formula = "".join(f"{s}{self.symbols.count(s)}" for s in dict.fromkeys(self.symbols))
        hess_str = f", hessian={self.hessian.shape}" if self.hessian is not None else ""
        return f"Q2MMMolecule({formula}, {self.n_atoms} atoms, {len(self.bonds)} bonds{hess_str})"
