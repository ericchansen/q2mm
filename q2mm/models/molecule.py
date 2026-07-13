"""Clean molecular structure representation for Q2MM.

Built on QCElemental for validated molecular data (symbols, geometry,
charge, multiplicity, connectivity) with Q2MM-specific extensions
(Hessian, detected bonds/angles, element-based matching).
"""

from __future__ import annotations


import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
    canonicalize_torsion_env_id,
)

if TYPE_CHECKING:
    from q2mm.models.structure import HessianUnits, Structure

try:
    import qcelemental as qcel

    _HAS_QCEL = True
except ImportError:
    qcel = None
    _HAS_QCEL = False


# Covalent radii — imported from the single-source-of-truth element table.
from q2mm.elements import COVALENT_RADII  # noqa: E402


def _structure_atom_element(atom: Any) -> str:
    """Return an atom's authoritative element symbol.

    Prefers the :class:`~q2mm.models.structure.Atom` element (derived from
    ``atomic_num`` or an explicitly-set symbol) over guessing from the
    atom-type label.  ``_extract_element`` alone would misread two-letter
    labels that title-case to a real element — ``"CO"`` (a carbon type) →
    cobalt, ``"CA"`` → calcium — which corrupts every element-keyed match
    downstream.  Falls back to ``_extract_element`` on the type name only
    when the atom carries no derivable element (e.g. type-only FF atoms).
    """
    try:
        element = atom.element
    except (ValueError, AttributeError):
        element = None
    if element:
        # Normalise casing/aliases to the canonical element table form
        # (e.g. a raw ``"RH"`` label becomes ``"Rh"``).  ``_extract_element``
        # is safe here because ``element`` is the authoritative symbol, not
        # the ambiguous atom-type name.
        return _extract_element(element)
    return _extract_element(atom.atom_type_name or "")


def _structure_atom_label(atom: Any) -> str:
    """Return an atom's type-name label, tolerantly falling back to its element.

    Prefers the explicit atom-type name; only when that is absent does it
    consult ``atom.element``, accessed through a guard so dummy / type-only
    atoms (no ``atomic_num`` and no explicit element) yield an empty label
    instead of raising ``ValueError`` — mirroring the tolerant resolution in
    :func:`_structure_atom_element`.
    """
    if atom.atom_type_name:
        return atom.atom_type_name
    try:
        return atom.element or ""
    except (ValueError, AttributeError):
        return ""


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute signed dihedral angle (degrees) for four points using atan2.

    Delegates to :func:`q2mm.geometry.dihedral_angle`.
    Returns a value in [-180, 180].
    """
    from q2mm.geometry import dihedral_angle

    return dihedral_angle(p0, p1, p2, p3)


def _strip_pint(hessian: Any) -> Any:
    """Strip a ``pint.Quantity`` wrapper and convert it to canonical AU.

    When ``pint`` is installed, ``JaguarIn.get_hessian(tag_units=True)``
    returns a ``pint.Quantity`` tagged with units.  This helper converts to
    ``hartree/bohr**2`` (if not already) and extracts the bare
    ``np.ndarray`` so that downstream code (Seminario projection,
    eigendecomposition) sees plain arrays with zero Pint overhead.
    """
    if hessian is None:
        return None
    if hasattr(hessian, "magnitude") and hasattr(hessian, "to"):
        return np.asarray(hessian.to("hartree/bohr**2").magnitude)
    return hessian


_HESSIAN_ATOMIC_UNIT = "hartree/bohr**2"
_HESSIAN_KJ_MOL_ANGSTROM2_UNIT = "kilojoule/(mole*angstrom**2)"


def _hessian_to_atomic_units(hessian: Any, units: HessianUnits | str | None) -> np.ndarray | None:
    """Convert a Hessian with known provenance to a bare canonical-AU array."""
    if hessian is None:
        return None
    if hasattr(hessian, "magnitude") and hasattr(hessian, "to"):
        return np.asarray(hessian.to(_HESSIAN_ATOMIC_UNIT).magnitude)
    array = np.asarray(hessian, dtype=float)
    unit_value = getattr(units, "value", units)
    if unit_value == _HESSIAN_ATOMIC_UNIT:
        return array
    if unit_value == _HESSIAN_KJ_MOL_ANGSTROM2_UNIT:
        from q2mm.constants import KJMOLA2_TO_HESSIAN_AU

        return array * KJMOLA2_TO_HESSIAN_AU
    raise ValueError(
        "Structure Hessian unit provenance is unknown. Set "
        "structure.hessian_units or pass an explicit canonical-AU hessian override."
    )


_CANONICAL_BOND_ORDERS = {
    "-": "-",
    "=": "=",
    "*": "*",
    "%": "%",
    "1": "-",
    "2": "=",
    "ar": "*",
    "3": "%",
}


def _canonicalize_bond_order(order: str | None) -> str:
    """Map supported parser bond-order tokens into the MM3 symbol domain."""
    if order is None:
        return ""
    return _CANONICAL_BOND_ORDERS.get(str(order).strip().lower(), "")


class _HessianUnset:
    """Sentinel distinguishing an omitted Hessian from an explicit ``None``."""


_HESSIAN_UNSET = _HessianUnset()


@dataclass
class DetectedBond:
    """A bond detected from molecular geometry."""

    atom_i: int  # 0-based index
    atom_j: int  # 0-based index
    elements: tuple[str, str]
    length: float  # Angstrom
    env_id: str = ""
    ff_row: int | None = None
    bond_order: str = ""  # "-" single, "=" double, "*" aromatic, "%" triple
    source_bond_order: str | None = None  # Original parser token, if supplied.

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
    _improper_torsions: list[DetectedTorsion] | None = field(default=None, repr=False)
    _bonds_explicit: bool = field(default=False, repr=False)
    _angles_explicit: bool = field(default=False, repr=False)
    _torsions_explicit: bool = field(default=False, repr=False)
    partial_charges: list[float | None] | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        """Invalidate cached topology when the bond-detection tolerance changes."""
        if name == "bond_tolerance":
            value = float(value)
            previous = self.__dict__.get(name)
            object.__setattr__(self, name, value)
            if previous is not None and previous != value:
                self._invalidate_inferred_topology()
            return
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        """Validate per-atom data and normalize geometry to float."""
        self.symbols = [str(symbol) for symbol in self.symbols]
        if self.atom_types is None:
            self.atom_types = list(self.symbols)
        else:
            self.atom_types = [str(atom_type) for atom_type in self.atom_types]
        if len(self.atom_types) != len(self.symbols):
            raise ValueError("atom_types must have the same length as symbols.")
        if self.partial_charges is not None:
            if len(self.partial_charges) != len(self.symbols):
                raise ValueError("partial_charges must have the same length as symbols.")
            self.partial_charges = [
                None if partial_charge is None else float(partial_charge) for partial_charge in self.partial_charges
            ]
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

    @property
    def improper_torsions(self) -> list[DetectedTorsion]:
        """Auto-detected improper torsions at trigonal (sp2) centres.

        A trigonal centre is an atom bonded to exactly 3 neighbours.
        For centre C with neighbours A, B, D, the improper is stored
        as (A, C, B, D) so that C is in position j (same convention as
        MM3 out-of-plane bending).
        """
        if self._improper_torsions is None:
            self._improper_torsions = self._detect_improper_torsions()
        return self._improper_torsions

    def invalidate_topology(self) -> None:
        """Clear all cached bond/angle/torsion data.

        Call this after changing ``atom_types`` so that ``env_id`` values
        in the cached topology are recomputed on next access. This explicitly
        discards parser-provided topology as well as inferred topology.
        """
        self._bonds = None
        self._angles = None
        self._torsions = None
        self._improper_torsions = None
        self._bonds_explicit = False
        self._angles_explicit = False
        self._torsions_explicit = False

    def _invalidate_inferred_topology(self) -> None:
        """Clear only topology that depends on distance-based bond detection."""
        if self._bonds_explicit:
            return
        self._bonds = None
        if not self._angles_explicit:
            self._angles = None
        if not self._torsions_explicit:
            self._torsions = None
        self._improper_torsions = None

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

    def _detect_improper_torsions(self) -> list[DetectedTorsion]:
        """Detect improper torsions at trigonal (sp2) centres.

        For each atom with exactly 3 bonded neighbours, generates an
        improper torsion quad.  The centre atom goes in position j
        (second slot) following the MM3 out-of-plane convention:
        ``(neighbour_0, centre, neighbour_1, neighbour_2)``.

        Neighbours are sorted by index for deterministic ordering.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in self.bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        impropers: list[DetectedTorsion] = []
        for centre in range(self.n_atoms):
            nbrs = sorted(adj[centre])
            if len(nbrs) != 3:
                continue
            a, b, d = nbrs
            value = _dihedral_angle(
                self.geometry[a],
                self.geometry[centre],
                self.geometry[b],
                self.geometry[d],
            )
            impropers.append(
                DetectedTorsion(
                    atom_i=a,
                    atom_j=centre,
                    atom_k=b,
                    atom_l=d,
                    elements=(
                        self.symbols[a],
                        self.symbols[centre],
                        self.symbols[b],
                        self.symbols[d],
                    ),
                    value=value,
                    env_id=canonicalize_torsion_env_id(
                        [self.atom_types[a], self.atom_types[centre], self.atom_types[b], self.atom_types[d]]
                    ),
                )
            )
        return impropers

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

        Returns:
            A new :class:`Q2MMMolecule` built from the XYZ geometry.

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
        charge: int | None = None,
        multiplicity: int | None = None,
        name: str = "",
        bond_tolerance: float = 1.3,
        hessian: np.ndarray | None | _HessianUnset = _HESSIAN_UNSET,
    ) -> Q2MMMolecule:
        """Create from a parser ``Structure`` without losing supplied data.

        Topology is preserved per degree of freedom: a supplied list,
        including an explicitly empty one, is authoritative. Missing bonds are
        inferred from geometry, while missing angles and torsions are inferred
        from the resulting bonds.

        Args:
            structure: Source :class:`~q2mm.models.structure.Structure` instance.
            charge: Molecular charge override. Defaults to the ``Structure``
                ``props["charge"]`` value, then zero.
            multiplicity: Spin multiplicity override. Defaults to the
                ``Structure`` ``props["multiplicity"]`` value, then one.
            name: Display name; defaults to ``structure.origin_name``.
            bond_tolerance: Multiplier on sum of covalent radii for bond
                detection.
            hessian: Cartesian Hessian override. When omitted, preserves
                ``structure.hess`` and converts it from
                ``structure.hessian_units`` to Hartree/Bohr². A bare explicit
                override is interpreted as Hartree/Bohr². Pass ``None``
                explicitly to omit it.

        Returns:
            A new :class:`Q2MMMolecule` with parser-supplied data preserved.

        """
        symbols = []
        atom_types = []
        partial_charges: list[float | None] = []
        coords = []
        for atom in structure.atoms:
            atom_label = _structure_atom_label(atom)
            symbols.append(_structure_atom_element(atom))
            atom_types.append(atom_label.strip() or _extract_element(atom_label))
            partial_charges.append(atom.partial_charge)
            coords.append(atom.coords)

        bonds = None
        if structure.has_explicit_bonds:
            bonds = []
            for bond in structure.bonds:
                atoms = structure.get_atoms_in_DOF(bond)
                dof_atom_types = [_structure_atom_label(a) for a in atoms]
                elements = (
                    _structure_atom_element(atoms[0]),
                    _structure_atom_element(atoms[1]),
                )
                length = (
                    float(bond.value)
                    if bond.value is not None
                    else float(np.linalg.norm(atoms[0].coords - atoms[1].coords))
                )
                bonds.append(
                    DetectedBond(
                        atom_i=bond.atom_nums[0] - 1,
                        atom_j=bond.atom_nums[1] - 1,
                        elements=elements,
                        length=length,
                        env_id=canonicalize_bond_env_id(dof_atom_types),
                        ff_row=bond.ff_row,
                        bond_order=_canonicalize_bond_order(bond.order),
                        source_bond_order=bond.order,
                    )
                )

        angles = None
        if structure.has_explicit_angles:
            angles = []
            for angle in structure.angles:
                atoms = structure.get_atoms_in_DOF(angle)
                dof_atom_types = [_structure_atom_label(a) for a in atoms]
                elements = (
                    _structure_atom_element(atoms[0]),
                    _structure_atom_element(atoms[1]),
                    _structure_atom_element(atoms[2]),
                )
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

        torsions = None
        if structure.has_explicit_torsions:
            torsions = []
            for torsion in structure.torsions:
                atoms = structure.get_atoms_in_DOF(torsion)
                dof_atom_types = [_structure_atom_label(a) for a in atoms]
                elements = (
                    _structure_atom_element(atoms[0]),
                    _structure_atom_element(atoms[1]),
                    _structure_atom_element(atoms[2]),
                    _structure_atom_element(atoms[3]),
                )
                torsion_value = torsion.value
                if torsion_value is None:
                    torsion_value = _dihedral_angle(
                        atoms[0].coords,
                        atoms[1].coords,
                        atoms[2].coords,
                        atoms[3].coords,
                    )
                torsions.append(
                    DetectedTorsion(
                        atom_i=torsion.atom_nums[0] - 1,
                        atom_j=torsion.atom_nums[1] - 1,
                        atom_k=torsion.atom_nums[2] - 1,
                        atom_l=torsion.atom_nums[3] - 1,
                        elements=elements,
                        value=float(torsion_value),
                        env_id=canonicalize_torsion_env_id(dof_atom_types),
                        ff_row=torsion.ff_row,
                    )
                )

        resolved_charge = int(structure.props.get("charge", 0)) if charge is None else int(charge)
        resolved_multiplicity = (
            int(structure.props.get("multiplicity", 1)) if multiplicity is None else int(multiplicity)
        )
        if isinstance(hessian, _HessianUnset):
            resolved_hessian = _hessian_to_atomic_units(structure.hess, structure.hessian_units)
        else:
            resolved_hessian = _hessian_to_atomic_units(hessian, _HESSIAN_ATOMIC_UNIT)

        return cls(
            symbols=symbols,
            atom_types=atom_types,
            partial_charges=partial_charges if any(value is not None for value in partial_charges) else None,
            geometry=np.array(coords, dtype=float),
            charge=resolved_charge,
            multiplicity=resolved_multiplicity,
            name=name or structure.origin_name,
            bond_tolerance=bond_tolerance,
            hessian=resolved_hessian,
            _bonds=bonds,
            _angles=angles,
            _torsions=torsions,
            _bonds_explicit=structure.has_explicit_bonds,
            _angles_explicit=structure.has_explicit_angles,
            _torsions_explicit=structure.has_explicit_torsions,
        )

    @classmethod
    def from_qcel(cls, mol: qcel.models.Molecule, name: str = "") -> Q2MMMolecule:
        """Create from a QCElemental Molecule object.

        Args:
            mol: QCElemental ``Molecule`` with geometry in Bohr.
            name: Display name for the molecule.

        Returns:
            A new :class:`Q2MMMolecule` with geometry converted to Ångströms.

        """
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
        """Convert to QCElemental Molecule.

        Returns:
            A ``qcelemental.models.Molecule`` with geometry in Bohr and
            connectivity derived from detected bonds.

        """
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
        """Return a copy with Hessian attached.

        Args:
            hessian: Cartesian Hessian matrix of shape ``(3N, 3N)``.

        Returns:
            A new :class:`Q2MMMolecule` identical to this one but with the
            given *hessian* attached.

        """
        return Q2MMMolecule(
            symbols=self.symbols,
            atom_types=list(self.atom_types),
            partial_charges=copy.deepcopy(self.partial_charges),
            geometry=self.geometry.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            name=self.name,
            bond_tolerance=self.bond_tolerance,
            hessian=_strip_pint(hessian),
            _bonds=copy.deepcopy(self._bonds) if self._bonds is not None else None,
            _angles=copy.deepcopy(self._angles) if self._angles is not None else None,
            _torsions=copy.deepcopy(self._torsions) if self._torsions is not None else None,
            _improper_torsions=(
                copy.deepcopy(self._improper_torsions) if self._improper_torsions is not None else None
            ),
            _bonds_explicit=self._bonds_explicit,
            _angles_explicit=self._angles_explicit,
            _torsions_explicit=self._torsions_explicit,
        )

    def __repr__(self) -> str:
        formula = "".join(f"{s}{self.symbols.count(s)}" for s in dict.fromkeys(self.symbols))
        hess_str = f", hessian={self.hessian.shape}" if self.hessian is not None else ""
        return f"Q2MMMolecule({formula}, {self.n_atoms} atoms, {len(self.bonds)} bonds{hess_str})"
