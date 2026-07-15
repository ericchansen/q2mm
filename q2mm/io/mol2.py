"""Parser for Tripos Mol2 structure files.

Provides the ``Mol2`` class for reading atom coordinates, bond
connectivity, and other structural data from ``.mol2`` files.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from q2mm.geometry import bond_length
from q2mm.models.identifiers import _extract_element, canonicalize_bond_env_id
from q2mm.models.molecule import Bond, Molecule

logger = logging.getLogger(__name__)

# Mol2 (SYBYL) bond-order tokens mapped onto the MM3 symbol domain.
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
    """Map a Mol2 bond-order token into the MM3 symbol domain."""
    if order is None:
        return ""
    return _CANONICAL_BOND_ORDERS.get(str(order).strip().lower(), "")


@dataclass
class _Mol2Atom:
    """One atom parsed from a Mol2 ``@<TRIPOS>ATOM`` block."""

    index: int
    element: str
    x: float
    y: float
    z: float
    atom_type_name: str
    partial_charge: float | None = None

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates of this atom, as ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z])

    @property
    def symbol(self) -> str:
        """Resolve this atom's element symbol."""
        return _extract_element(self.element) if self.element else ""


@dataclass
class _Mol2Bond:
    """One bond parsed from a Mol2 ``@<TRIPOS>BOND`` block."""

    atom_nums: list[int]
    value: float
    order: str | None = None


@dataclass
class _Mol2Record:
    """One structure staging record parsed from a Mol2 ``MOLECULE`` chunk.

    Mol2 files always supply explicit bonds for every structure; angles and
    torsions are never present in the format and are always inferred by
    :class:`~q2mm.models.molecule.Molecule` from that bond connectivity.
    """

    origin_name: str
    atoms: list[_Mol2Atom] = field(default_factory=list)
    bonds: list[_Mol2Bond] = field(default_factory=list)


def _molecule_from_mol2_record(record: _Mol2Record) -> Molecule:
    """Convert a :class:`_Mol2Record` into a :class:`~q2mm.models.molecule.Molecule`.

    Bonds are always explicit/authoritative; angles/torsions are always
    inferred from that bond connectivity (Mol2 carries no angle/torsion
    data at all).
    """
    symbols = [atom.symbol for atom in record.atoms]
    atom_types = [atom.atom_type_name for atom in record.atoms]
    coords = [atom.coords for atom in record.atoms]
    partial_charges = [atom.partial_charge for atom in record.atoms]

    def _atom(num: int) -> _Mol2Atom:
        return record.atoms[num - 1]

    bonds = tuple(
        Bond(
            atom_i=b.atom_nums[0] - 1,
            atom_j=b.atom_nums[1] - 1,
            elements=(_atom(b.atom_nums[0]).symbol, _atom(b.atom_nums[1]).symbol),
            length=b.value,
            env_id=canonicalize_bond_env_id([_atom(n).atom_type_name for n in b.atom_nums]),
            bond_order=_canonicalize_bond_order(b.order),
            source_bond_order=b.order,
        )
        for b in record.bonds
    )

    return Molecule(
        symbols=tuple(symbols),
        atom_types=tuple(atom_types),
        partial_charges=(tuple(partial_charges) if any(c is not None for c in partial_charges) else None),
        geometry=np.array(coords, dtype=float),
        bonds=bonds,
        name=record.origin_name,
    )


class Mol2:
    """Retrieve structural data from Tripos Mol2 files.

    Please ensure that mol2 atom types match the atom types specified
    in the force field.

    Note:
        Format for the data in the file can be found by searching
        Tripos Mol2 File Format SYBYL.

    """

    TRIPOS_FLAG = "@<TRIPOS>"
    MOLECULE_FLAG = "MOLECULE"
    ATOM_FLAG = "ATOM"
    BOND_FLAG = "BOND"

    __slots__ = ["_lines", "path", "directory", "filename", "_records"]

    def __init__(self, path: str) -> None:
        """Initialize a Mol2 instance.

        Args:
            path (str): Absolute path of the mol2 file.

        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        self._records: list[_Mol2Record] | None = None

    @property
    def lines(self) -> list[str]:
        """Return the lines of the file.

        Returns:
            (list[str]): Lines of the file.

        """
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path: str, lines: list[str] | None = None) -> None:
        """Write lines to file at path.

        Args:
            path (str): Location of file to write.
            lines (list[str] | None): Lines to write. Defaults to
                ``self.lines``.

        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)

    @property
    def _records_parsed(self) -> list[_Mol2Record]:
        """Records extracted from the mol2 file (private staging).

        Lazily parses the file on first access. Not part of the public API
        — see :attr:`molecules`.
        """
        if self._records is None:
            self.parse_lines()
        return self._records

    @property
    def molecules(self) -> list[Molecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Molecule` objects."""
        return [_molecule_from_mol2_record(record) for record in self._records_parsed]

    def parse_lines(self) -> None:
        """Parse file lines to extract records into ``self._records``.

        It is safe to parse this with ``split`` because the mol2 format
        from SYBYL requires consistent data ordering matching the
        standard; otherwise the file is not in valid mol2 format.
        """
        self._records = []
        joined_lines = "".join(self.lines)
        structure_chunks = joined_lines.split(self.TRIPOS_FLAG + self.MOLECULE_FLAG)
        entry_num = 0 if len(structure_chunks) > 2 else None
        for struct_chunk in structure_chunks:
            if struct_chunk != "":
                self._records.append(self.parse_structure(struct_chunk, chunk_index=entry_num))

        if len(structure_chunks) - 1 != len(self._records):
            logger.log(
                logging.WARNING,
                "Only "
                + str(len(self._records))
                + " structures could be parsed from "
                + str(len(structure_chunks) - 1)
                + " MOLECULE entries in the .mol2 file",
            )

    def parse_atoms(self, atom_lines: list[str]) -> list[_Mol2Atom]:
        """Parse atom entries from mol2 atom-section lines.

        Args:
            atom_lines (list[str]): Lines from the mol2 file pertaining
                to the atoms in the structure.

        Returns:
            (list[_Mol2Atom]): Atom records parsed from *atom_lines*.

        """
        atoms = []
        for atom_entry in atom_lines:
            if atom_entry == "" or atom_entry.strip() == self.ATOM_FLAG:
                continue
            atom_split = atom_entry.split()
            # Mol2 column 2 is the atom name (e.g. "C1", "RH1"), not the
            # element symbol.  Strip trailing digits and title-case to get
            # a proper element key that matches constants.MASSES (e.g. "Rh").
            raw_name = atom_split[1]
            element = raw_name.rstrip("0123456789").capitalize()
            # partial_charge (column 9) comes as a string — cast to float
            try:
                charge = float(atom_split[8])
            except (IndexError, ValueError):
                charge = None
            x, y, z = (float(v) for v in atom_split[2:5])
            atoms.append(
                _Mol2Atom(
                    index=int(atom_split[0]),
                    element=element,
                    x=x,
                    y=y,
                    z=z,
                    atom_type_name=atom_split[5],
                    partial_charge=charge,
                )
            )
        return atoms

    def parse_bonds(self, bond_lines: list[str], structure: _Mol2Record) -> list[_Mol2Bond]:
        """Parse bond entries from mol2 bond-section lines.

        Args:
            bond_lines (list[str]): Lines from the mol2 file pertaining
                to the bond connectivity in the structure.
            structure (_Mol2Record): Record to which the bonds pertain,
                used for bond-length measurement.

        Returns:
            (list[_Mol2Bond]): Bond records parsed from *bond_lines*.

        """
        bonds = []
        for bond_entry in bond_lines:
            if bond_entry == "" or bond_entry.strip() == self.BOND_FLAG:
                continue
            bond_split = bond_entry.split()
            a_index = int(bond_split[1])
            b_index = int(bond_split[2])
            bonds.append(
                _Mol2Bond(
                    atom_nums=[a_index, b_index],
                    order=bond_split[3],
                    value=bond_length(
                        structure.atoms[a_index - 1].coords,
                        structure.atoms[b_index - 1].coords,
                    ),
                )
            )

        return bonds

    def parse_structure(self, structure_chunk: str, chunk_index: int | None = None) -> _Mol2Record:
        """Parse a single structure from a mol2 molecule chunk.

        Args:
            structure_chunk (str): String containing the lines which
                pertain to a single structure.
            chunk_index (int | None): Zero-based index of this chunk
                within the file. Appended to the filename to form a
                unique identifier when the file contains multiple
                structures. ``None`` for single-structure files.

        Returns:
            (_Mol2Record): The record parsed from *structure_chunk* data.

        """
        tripos_chunks = structure_chunk.split(self.TRIPOS_FLAG)
        molecule_lines = tripos_chunks[0].split("\n")
        atom_lines = tripos_chunks[1].split("\n")
        bond_chunk = 2
        bond_lines = tripos_chunks[bond_chunk].split("\n")

        # Validate that data was chunked correctly:
        if atom_lines[0].strip() != self.ATOM_FLAG:
            raise ValueError(f"Expected {self.ATOM_FLAG} but got {atom_lines[0].strip()!r}")
        while bond_lines[0].strip() != self.BOND_FLAG:
            bond_chunk += 1
            try:
                bond_lines = tripos_chunks[bond_chunk].split("\n")
            except IndexError:
                logger.log(
                    logging.ERROR,
                    "No BOND flag within mol2 MOLECULE, invalid structure.",
                )
                break

        # parse number of atoms and number of bonds from line 2 below @<TRIPOS>MOLECULE
        molecule_data = molecule_lines[2].split()
        num_atoms = int(molecule_data[0])
        num_bonds = int(molecule_data[1])

        file_identifier = self.filename if chunk_index is None else self.filename + str(chunk_index)

        struct = _Mol2Record(file_identifier)

        # send chunk from @<TRIPOS>ATOM to @<TRIPOS>BOND to parse_atoms
        struct.atoms = self.parse_atoms(atom_lines)

        # use num atoms from @<TRIPOS>MOLECULE to verify parse is correct
        if len(struct.atoms) != num_atoms:
            raise ValueError(f"Parsed {len(struct.atoms)} atoms but expected {num_atoms} atoms based on Mol2 data.")
        if not all(struct.atoms[i].index == i + 1 for i in range(len(struct.atoms))):
            raise ValueError("Mol2 atom index values do not match their ordering.")

        # send chunk from @<TRIPOS>BOND to end-of-file to parse_bonds
        struct.bonds = self.parse_bonds(bond_lines, struct)

        # use num bonds from @<TRIPOS>MOLECULE to verify parse is correct
        if len(struct.bonds) != num_bonds:
            raise ValueError(f"Parsed {len(struct.bonds)} bonds but expected {num_bonds} bonds based on Mol2 data.")

        return struct
