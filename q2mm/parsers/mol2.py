"""Parser for Tripos Mol2 structure files.

Provides the ``Mol2`` class for reading atom coordinates, bond
connectivity, and other structural data from ``.mol2`` files.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from q2mm.geometry import bond_length
from q2mm.parsers.base import File
from q2mm.models.structure import Atom, Bond, Structure

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule

logger = logging.getLogger(__name__)


class Mol2(File):
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

    __slots__ = ["_lines", "path", "directory", "filename", "_structures"]

    def __init__(self, path: str) -> None:
        """Initialize a Mol2 instance.

        Args:
            path (str): Absolute path of the mol2 file.

        """
        super().__init__(path)
        self._structures: list[Structure] = None

    @property
    def structures(self) -> list[Structure]:
        """list[Structure]: Structure objects extracted from the mol2 file.

        Lazily parses the file on first access.

        Returns:
            (list[Structure]): Structure objects extracted from parsing the
                mol2 file.

        .. deprecated::
            Use :attr:`molecules` instead for ``Q2MMMolecule`` objects.

        """
        if self._structures is None:
            self.parse_lines()
        return self._structures

    @property
    def molecules(self) -> list[Q2MMMolecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Q2MMMolecule` objects."""
        from q2mm.models.molecule import Q2MMMolecule

        return [Q2MMMolecule.from_structure(s) for s in self.structures]

    def parse_lines(self) -> None:
        """Parse file lines to extract Structure objects into ``self.structures``.

        It is safe to parse this with ``split`` because the mol2 format
        from SYBYL requires consistent data ordering matching the
        standard; otherwise the file is not in valid mol2 format.
        """
        # TODO this could be amended to use regular expression matching (regex) if slow
        self._structures: list[Structure] = []
        joined_lines = "".join(self.lines)
        structure_chunks = joined_lines.split(self.TRIPOS_FLAG + self.MOLECULE_FLAG)
        entry_num = 0 if len(structure_chunks) > 2 else None
        for struct_chunk in structure_chunks:
            if struct_chunk != "":
                self._structures.append(self.parse_structure(struct_chunk, chunk_index=entry_num))

        if len(structure_chunks) - 1 != len(self._structures):
            logger.log(
                logging.WARNING,
                "Only "
                + str(len(self._structures))
                + " structures could be parsed from "
                + str(len(structure_chunks) - 1)
                + " MOLECULE entries in the .mol2 file",
            )

    def parse_atoms(self, atom_lines: list[str]) -> list[Atom]:
        """Parse atom entries from mol2 atom-section lines.

        Args:
            atom_lines (list[str]): Lines from the mol2 file pertaining
                to the atoms in the structure.

        Returns:
            (list[Atom]): Atom objects parsed from *atom_lines*.

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
            atoms.append(
                Atom(
                    index=int(atom_split[0]),
                    element=element,
                    coords=atom_split[2:5],
                    atom_type_name=atom_split[5],
                    partial_charge=charge,
                )
            )
        return atoms

    def parse_bonds(self, bond_lines: list[str], structure: Structure) -> list[Bond]:
        """Parse bond entries from mol2 bond-section lines.

        Args:
            bond_lines (list[str]): Lines from the mol2 file pertaining
                to the bond connectivity in the structure.
            structure (Structure): Structure to which the bonds pertain,
                used for bond-length measurement.

        Returns:
            (list[Bond]): Bond objects parsed from *bond_lines*.

        """
        bonds = []
        for bond_entry in bond_lines:
            if bond_entry == "" or bond_entry.strip() == self.BOND_FLAG:
                continue
            bond_split = bond_entry.split()
            a_index = int(bond_split[1])
            b_index = int(bond_split[2])
            bonds.append(
                Bond(
                    atom_nums=[a_index, b_index],
                    order=bond_split[3],
                    value=bond_length(
                        structure.atoms[a_index - 1].coords,
                        structure.atoms[b_index - 1].coords,
                    ),
                )
            )

        # TODO: Ideally, the bonds class would measure the bonds and just contain a pointer to an Atom
        # object, but that would require a decent-sized refactor so hold off for now

        return bonds

    def parse_structure(self, structure_chunk: str, chunk_index: int | None = None) -> Structure:
        """Parse a single structure from a mol2 molecule chunk.

        Args:
            structure_chunk (str): String containing the lines which
                pertain to a single structure.
            chunk_index (int | None): Zero-based index of this chunk
                within the file. Appended to the filename to form a
                unique identifier when the file contains multiple
                structures. ``None`` for single-structure files.

        Returns:
            (Structure): The Structure object parsed from
                *structure_chunk* data.

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

        struct = Structure(file_identifier)  # ideally we would gather data, then instantiate a Structure with
        # all the data as arguments, but for now I will follow the precedent within the Q2MM code
        # to avoid significant refactoring since we still don't have test cases or test scripts

        # send chunk from @<TRIPOS>ATOM to @<TRIPOS>BOND to parse_atoms
        struct._atoms = self.parse_atoms(atom_lines)

        # use num atoms from @<TRIPOS>MOLECULE to verify parse is correct
        if len(struct._atoms) != num_atoms:
            raise ValueError(f"Parsed {len(struct.atoms)} atoms but expected {num_atoms} atoms based on Mol2 data.")
        if not all(struct.atoms[i].index == i + 1 for i in range(len(struct.atoms))):
            raise ValueError("Mol2 atom index values do not match their ordering.")

        # send chunk from @<TRIPOS>BOND to end-of-file to parse_bonds
        struct._bonds = self.parse_bonds(bond_lines, struct)

        # use num bonds from @<TRIPOS>MOLECULE to verify parse is correct
        if len(struct._bonds) != num_bonds:
            raise ValueError(f"Parsed {len(struct.bonds)} bonds but expected {num_bonds} bonds based on Mol2 data.")

        return struct
