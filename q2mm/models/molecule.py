"""Clean molecular structure representation for Q2MM.

Built on QCElemental for validated molecular data (symbols, geometry,
charge, multiplicity, connectivity) with Q2MM-specific extensions
(Hessian, detected bonds/angles, element-based matching).
"""

from __future__ import annotations


import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
    canonicalize_torsion_env_id,
)

if TYPE_CHECKING:
    from q2mm.models.structure import Structure

try:
    import qcelemental as qcel

    _HAS_QCEL = True
except ImportError:
    qcel = None
    _HAS_QCEL = False


# Covalent radii — imported from the single-source-of-truth element table.
from q2mm.elements import COVALENT_RADII  # noqa: E402


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute signed dihedral angle (degrees) for four points using atan2.

    Delegates to :func:`q2mm.geometry.dihedral_angle`.
    Returns a value in [-180, 180].
    """
    from q2mm.geometry import dihedral_angle

    return dihedral_angle(p0, p1, p2, p3)


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
class DetectedTorsion:
    """A torsion/dihedral detected from molecular bonds."""

    atom_i: int  # 0-based (end)
    atom_j: int  # 0-based (central)
    atom_k: int  # 0-based (central)
    atom_l: int  # 0-based (end)
    elements: tuple[str, str, str, str]
    value: float  # dihedral angle in degrees, [-180, 180]
    env_id: str = ""
    ff_row: int | None = None

    @property
    def element_quad(self) -> tuple[str, str, str, str]:
        """Canonical element quad: forward or reversed, whichever is lexically smaller."""
        fwd = self.elements
        rev = (fwd[3], fwd[2], fwd[1], fwd[0])
        return min(fwd, rev)


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
    bond_tolerance: float = 1.3  # See constants.DEFAULT_BOND_TOLERANCE. 1.4+ for TS.
    hessian: np.ndarray | None = None  # Shape (3N, 3N), Hartree/Bohr^2
    _bonds: list[DetectedBond] | None = field(default=None, repr=False)
    _angles: list[DetectedAngle] | None = field(default=None, repr=False)
    _torsions: list[DetectedTorsion] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate atom_types length and normalize geometry to float."""
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
        """Number of atoms in the molecule."""
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

    @property
    def torsions(self) -> list[DetectedTorsion]:
        """Auto-detected torsion/dihedral angles from bonds."""
        if self._torsions is None:
            self._torsions = self._detect_torsions()
        return self._torsions

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

    def _detect_torsions(self) -> list[DetectedTorsion]:
        """Detect torsion/dihedral angles from detected bonds.

        For each bond B-C, finds all atoms A bonded to B (A ≠ C) and all
        atoms D bonded to C (D ≠ B) to form torsions A-B-C-D.  Deduplicates
        so that A-B-C-D and D-C-B-A are not both stored.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in self.bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        seen: set[tuple[int, int, int, int]] = set()
        torsions: list[DetectedTorsion] = []
        for bond in self.bonds:
            b, c = bond.atom_i, bond.atom_j
            for a in adj[b]:
                if a == c:
                    continue
                for d in adj[c]:
                    if d in (b, a):
                        continue
                    key = (a, b, c, d)
                    key_rev = (d, c, b, a)
                    if key in seen or key_rev in seen:
                        continue
                    seen.add(key)
                    value = _dihedral_angle(self.geometry[a], self.geometry[b], self.geometry[c], self.geometry[d])
                    torsions.append(
                        DetectedTorsion(
                            atom_i=a,
                            atom_j=b,
                            atom_k=c,
                            atom_l=d,
                            elements=(
                                self.symbols[a],
                                self.symbols[b],
                                self.symbols[c],
                                self.symbols[d],
                            ),
                            value=value,
                            env_id=canonicalize_torsion_env_id(
                                [self.atom_types[a], self.atom_types[b], self.atom_types[c], self.atom_types[d]]
                            ),
                        )
                    )
        return torsions

    # ---- Factory methods ----

    @classmethod
    def from_xyz(
        cls, path: str | Path, charge: int = 0, multiplicity: int = 1, name: str = "", bond_tolerance: float = 1.3
    ) -> Q2MMMolecule:
        """Load from XYZ file.

        Args:
            path: Path to the XYZ file.
            charge: Molecular charge.
            multiplicity: Spin multiplicity.
            name: Display name; defaults to filename stem.
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
        structure: Structure,
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
