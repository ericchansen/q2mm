from __future__ import annotations
import logging
import numpy as np
import os
import re
from q2mm import constants as co
from q2mm import utilities
from q2mm.parsers.base import File
from q2mm.parsers.structures import Atom, Bond, Structure

logger = logging.getLogger(__name__)


class Mol2(File):
    """
    Used to retrieve structural data from mol2 files.

    Please ensure that mol2 atom types match the atom types specified in the force field.

    Note:
            Format for the data in the file can be found by searching
            Tripos Mol2 File Format SYBYL.
    """

    TRIPOS_FLAG = "@<TRIPOS>"
    MOLECULE_FLAG = "MOLECULE"
    ATOM_FLAG = "ATOM"
    BOND_FLAG = "BOND"

    __slots__ = ["_lines", "path", "directory", "filename", "_structures"]

    def __init__(self, path: str):
        """Creates a Mol2 object based on the path given, data is only structural.

        Args:
            path (str): Absolute path of mol2 file.
        """
        super().__init__(path)
        self._structures: List[Structure] = None

    @property
    def structures(self) -> List[Structure]:
        """Returns the Structure objects extracted from the mol2 file at self.path.
        If None, indicating no extraction yet, parses the lines from the file to populate
        the structures list with Structures.

        Returns:
            List[Structure]: Structure objects extracted from parsing the mol2 file.
        """
        if self._structures is None:
            self.parse_lines()
        return self._structures

    def parse_lines(self):
        """Parses self.lines() as set by super to extract Structure objects to self.structures.

        It is safe to parse this with split because the mol2 format from SYBYL
         requires consistent data ordering matching the standard, otherwise the
         file is not in valid mol2 format.
        """
        # TODO this could be amended to use regular expression matching (regex) if slow
        self._structures: List[Structure] = []
        joined_lines = "".join(self.lines)
        structure_chunks = joined_lines.split(self.TRIPOS_FLAG + self.MOLECULE_FLAG)
        entry_num = 0 if len(structure_chunks) > 2 else None
        for struct_chunk in structure_chunks:
            if struct_chunk != "":
                self._structures.append(self.parse_structure(struct_chunk, chunk_index=entry_num))

        if len(structure_chunks) - 1 != len(self._structures):
            logger.log(
                logging.WARN,
                "Only "
                + str(len(self._structures))
                + " structures could be parsed from "
                + str(len(structure_chunks) - 1)
                + " MOLECULE entries in the .mol2 file",
            )

    def parse_atoms(self, atom_lines: List[str]) -> List[Atom]:
        """Returns the Atom objects parsed from the atom_lines given.

        Args:
            atom_lines (List[str]): lines from the mol2 file pertaining to the atoms in the structure.

        Returns:
            List[Atom]: Atom objects parsed from atom_lines
        """
        atoms = []
        for atom_entry in atom_lines:
            if atom_entry == "" or atom_entry.strip() == self.ATOM_FLAG:
                continue
            atom_split = atom_entry.split()
            atoms.append(
                Atom(
                    index=int(atom_split[0]),
                    element=atom_split[1],
                    coords=atom_split[2:5],
                    atom_type_name=atom_split[5],
                    partial_charge=atom_split[8],
                )
            )
        return atoms

    def parse_bonds(self, bond_lines: List[str], structure: Structure) -> List[Bond]:
        """Returns the Bond objects parsed from the bond_lines given.

        Args:
            bond_lines (List[str]): lines from the mol2 file pertaining to the bond connectivity in the structure.
            structure (Structure): structure which the bonds pertain to, used for bond measurement.

        Returns:
            List[Bond]: Bond objects parsed from bond_lines
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
                    value=utilities.measure_bond(
                        structure.atoms[a_index - 1].coords,
                        structure.atoms[b_index - 1].coords,
                    ),
                )
            )

        # TODO: Ideally, the bonds class would measure the bonds and just contain a pointer to an Atom
        # object, but that would require a decent-sized refactor so hold off for now

        return bonds

    def parse_structure(self, structure_chunk: str, chunk_index: int = None) -> Structure:
        """Returns the Structure objects parsed from the structure_chunk given.

        Args:
            structure_chunk (str): string containing the lines which pertain to a single structure.

        Returns:
            Structure: the Structure object parsed from structure_chunk data.
        """
        tripos_chunks = structure_chunk.split(self.TRIPOS_FLAG)
        molecule_lines = tripos_chunks[0].split("\n")
        atom_lines = tripos_chunks[1].split("\n")
        bond_chunk = 2
        bond_lines = tripos_chunks[bond_chunk].split("\n")

        # assert that data was chunked correctly:
        assert atom_lines[0].strip() == self.ATOM_FLAG
        while bond_lines[0].strip() != self.BOND_FLAG:
            bond_chunk += 1
            try:
                bond_lines = tripos_chunks[bond_chunk].split("\n")
            except IndexError:
                logger.log(
                    logging.ERROR,
                    "No BOND flag within mol2 MOLECULE, invalid structure.",
                )

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
        assert len(struct._atoms) == num_atoms, (
            f"Parsed {len(struct.atoms)} atoms but only expected {num_atoms} atoms based on Mol2 data."
        )
        assert all(struct.atoms[i].index == i + 1 for i in range(len(struct.atoms))), (
            "Mol2 atom index values do not match their ordering."
        )

        # send chunk from @<TRIPOS>BOND to end-of-file to parse_bonds
        struct._bonds = self.parse_bonds(bond_lines, struct)

        # use num bonds from @<TRIPOS>MOLECULE to verify parse is correct
        assert len(struct._bonds) == num_bonds, (
            f"Parsed {len(struct.bonds)} bonds but only expected {num_bonds} bonds based on Mol2 data."
        )

        return struct

    def value_bonds(
        self,
    ):  # TODO Not currently in use, remove if not needed by March 1, 2024.
        atom_list = self._structures.atoms
        for bond in self._structures.bonds:
            # Indexing atom_list is possible only because this is within the Mol2 class so
            # we can assume that the atoms were added in order of their atom index.
            atom1 = atom_list[bond.atom_nums[0] - 1]
            atom2 = atom_list[bond.atom_nums[1] - 1]
            bond.value = utilities.measure_bond(np.array(atom1.coords), np.array(atom2.coords))
