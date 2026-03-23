"""Parsers for MacroModel ``.mmo`` and log files.

Provides ``MacroModel`` for extracting structural data (bonds, angles,
torsions) from ``.mmo`` files and ``MacroModelLog`` for reading
mass-weighted Hessian matrices from MacroModel log files.
"""

import logging
import numpy as np
import re
from q2mm import constants as co
from q2mm.parsers.base import File
from q2mm.parsers.structures import Atom, Bond, Angle, Torsion, Structure

logger = logging.getLogger(__name__)


class MacroModel(File):
    """Extract structural data from MacroModel ``.mmo`` files.

    Reads bond lengths, angles, and torsions for each structure entry
    in the ``.mmo`` file.
    """

    def __init__(self, path):
        """Initialize a MacroModel instance.

        Args:
            path (str): Path to the MacroModel ``.mmo`` file.
        """
        super().__init__(path)
        self._structures = None

    @property
    def structures(self):
        """list[Structure]: Parsed structures with bonds, angles, and torsions.

        Returns:
            (list[Structure]): Structure objects extracted from the ``.mmo``
                file, each populated with sorted bonds, angles, and
                torsions.
        """
        # TODO: make this read atoms for consistency and bc need num atoms for hessian read
        if self._structures is None or self._structures == []:
            logger.log(10, f"READING: {self.filename}")
            self._structures = []
            with open(self.path) as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
                bonds = []
                angles = []
                torsions = []
                atoms = []
                current_structure = None
                section = None
                for line in f:
                    # This would probably be better as a function in the structure
                    # class but I wanted this as upstream as possible so I didn't
                    # have to worry about other coding issues. The MMO file lists
                    # the bonds, angles, and torsions in some order that I am unsure
                    # of. It seems consistent with the same filename but with two
                    # files with the exact same structure the ordering is off. This
                    # reorders the lists before being added to the structure class.
                    if "Atomic Charges, Coordinates and Connectivity" in line:
                        section = "atoms"
                        continue
                    if section == "atoms":
                        if "(" in line:
                            split = [item.strip() for item in line.split()]
                            atom_num = split[2][:-1]  # same as index
                            ele_name = re.sub(r"[0-9]", "", split[0])
                            atom = Atom(
                                atom_type_name=split[0],
                                element=ele_name,
                                index=int(atom_num),
                                x=float(split[5]),
                                y=float(split[6]),
                                z=float(split[7]),
                            )
                            atoms.append(atom)
                        if "Total" in line:
                            section = None
                            # Sort the bonds, angles, and torsions before the start
                            # of a new structure
                            if bonds:
                                bonds.sort(key=lambda x: (x.atom_nums[0], x.atom_nums[1]))
                                current_structure.bonds.extend(bonds)
                            if angles:
                                angles.sort(key=lambda x: (x.atom_nums[1], x.atom_nums[0], x.atom_nums[2]))
                                current_structure.angles.extend(angles)
                            if torsions:
                                torsions.sort(
                                    key=lambda x: (x.atom_nums[1], x.atom_nums[2], x.atom_nums[0], x.atom_nums[3])
                                )
                                current_structure.torsions.extend(torsions)
                            if atoms:
                                atoms.sort(key=lambda x: x.index)
                                current_structure.atoms.extend(atoms)
                    if "Input filename" in line:
                        count_input += 1
                    if "Input Structure Name" in line:
                        count_structure += 1
                    count_previous = count_current
                    # Sometimes only one of the above ("Input filename" and
                    # "Input Structure Name") is used, sometimes both are used.
                    # count_current will make sure you catch both.
                    count_current = max(count_input, count_structure)
                    # If these don't match, then we reached the end of a
                    # structure.
                    if count_current != count_previous:
                        bonds = []
                        angles = []
                        torsions = []
                        atoms = []
                        current_structure = Structure(self.filename)
                        self._structures.append(current_structure)
                    # For each structure we come across, look for sections that
                    # we are interested in: those pertaining to bonds, angles,
                    # and torsions. Of course more could be added. We set the
                    # section to None to mark the end of a section, and we leave
                    # it None for parts of the file we don't care about.
                    if "BOND LENGTHS AND STRETCH ENERGIES" in line:
                        section = "bond"
                    if "ANGLES, BEND AND STRETCH BEND ENERGIES" in line:
                        section = "angle"
                    if "BEND-BEND ANGLES AND ENERGIES" in line:
                        section = None
                    if "DIHEDRAL ANGLES AND TORSIONAL ENERGIES" in line:
                        section = "torsion"
                    if "DIHEDRAL ANGLES AND TORSIONAL CROSS-TERMS" in line:
                        section = None
                    if section == "bond":
                        bond = self.read_line_for_bond(line)
                        if bond is not None:
                            # current_structure.bonds.append(bond)
                            bonds.append(bond)
                    if section == "angle":
                        angle = self.read_line_for_angle(line)
                        if angle is not None:
                            # current_structure.angles.append(angle)
                            angles.append(angle)
                    if section == "torsion":
                        torsion = self.read_line_for_torsion(line)
                        if torsion is not None:
                            # current_structure.torsions.append(torsion)
                            torsions.append(torsion)
            logger.log(5, f"  -- Imported {len(self._structures)} structure(s).")
        return self._structures

    def read_line_for_bond(self, line):
        """Parse a single line for bond data.

        Args:
            line (str): A line from the bond section of the ``.mmo`` file.

        Returns:
            (Bond | None): A Bond object if the line matches the bond
                pattern, otherwise ``None``.
        """
        match = co.RE_BOND.match(line)
        # TODO: MF find if atom_nums are atomic or index, where index bc need for sub_hessian seminario
        if match:
            atom_nums = [int(x) for x in [match.group(1), match.group(2)]]
            value = float(match.group(3))
            comment = match.group(4).strip()
            ff_row = int(match.group(5))
            return Bond(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
        else:
            return None

    def read_line_for_angle(self, line):
        """Parse a single line for angle data.

        Terminal atoms are reordered so that the lower index comes first.

        Args:
            line (str): A line from the angle section of the ``.mmo`` file.

        Returns:
            (Angle | None): An Angle object if the line matches the angle
                pattern, otherwise ``None``.
        """
        match = co.RE_ANGLE.match(line)
        if match:
            atom_nums = [int(x) for x in [match.group(1), match.group(2), match.group(3)]]
            # Reorder the terminal atoms so that the lower index atom is first.
            if atom_nums[0] > atom_nums[2]:
                atom_nums = [atom_nums[2], atom_nums[1], atom_nums[0]]
            value = float(match.group(4))
            comment = match.group(5).strip()
            ff_row = int(match.group(6))
            return Angle(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
        else:
            return None

    def read_line_for_torsion(self, line):
        """Parse a single line for torsion data.

        Atom indices are reordered so that the lower central-atom index
        comes first.

        Args:
            line (str): A line from the torsion section of the ``.mmo``
                file.

        Returns:
            (Torsion | None): A Torsion object if the line matches the
                torsion pattern, otherwise ``None``.
        """
        match = co.RE_TORSION.match(line)
        if match:
            atom_nums = [int(x) for x in [match.group(1), match.group(2), match.group(3), match.group(4)]]
            if atom_nums[1] > atom_nums[2]:
                atom_nums = [atom_nums[3], atom_nums[2], atom_nums[1], atom_nums[0]]
            value = float(match.group(5))
            comment = match.group(6).strip()
            ff_row = int(match.group(7))
            return Torsion(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
        else:
            return None


class MacroModelLog(File):
    """Retrieve data from MacroModel log files.

    The Hessian matrix read from these files is mass-weighted.
    """

    def __init__(self, path):
        """Initialize a MacroModelLog instance.

        Args:
            path (str): Path to the MacroModel log file.
        """
        super().__init__(path)
        self._hessian = None
        self._structures = None

    @property
    def hessian(self):
        """numpy.ndarray: Mass-weighted Hessian matrix read from the log file.

        Returns:
            (numpy.ndarray): 2-D Hessian of shape ``(N*3, N*3)`` where
                *N* is the number of atoms.
        """
        if self._hessian is None:
            logger.log(10, f"READING: {self.filename}")
            with open(self.path) as f:
                lines = f.read()
            num_atoms = int(re.search(r"Read\s+(\d+)\s+atoms.", lines).group(1))
            logger.log(5, f"  -- Read {num_atoms} atoms.")

            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(5, f"  -- Creating {hessian.shape} Hessian matrix.")
            words = lines.split()
            section_hessian = False
            start_row = False
            start_col = False
            row_num = 0
            col_nums = []
            elements = []
            for i, word in enumerate(words):
                # 1. Start of Hessian section.
                if word == "Mass-weighted":
                    section_hessian = True
                    continue
                # 5. End of Hessian. Add last row of Hessian and break.
                if word == "Eigenvalues:":
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    section_hessian = False
                    break
                # 4. End of a Hessian row. Add to matrix and reset.
                if section_hessian and start_col and word == "Element":
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    start_col = False
                    start_row = True
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    continue
                # 2. Start of a Hessian row.
                if section_hessian and word == "Element":
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    start_row = True
                    continue
                # 3. Okay, made it through the row number. Now look for columns
                #    and elements.
                if section_hessian and start_row and word == ":":
                    start_row = False
                    start_col = True
                    continue
                if section_hessian and start_col and "." not in word and word != "NaN":
                    col_nums.append(int(word))
                    continue
                if section_hessian and start_col and "." in word or word == "NaN":
                    elements.append(float(word))
                    continue
            self._hessian = hessian
            logger.log(5, f"  -- Creating {hessian.shape} Hessian matrix.")
        return self._hessian

    @property
    def structures(self):
        """list[Structure]: Parsed structures from the log file.

        Returns:
            (list[Structure]): Structure objects extracted from the
                MacroModel log file.
        """
        if self._structures is None:
            logger.log(10, f"READING: {self.filename}")
            self._structures = []
            with open(self.path) as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
                atoms = []
                bonds = []
                section = None
                for line in f:
                    if "m_atom" in line:
                        section = "atom"
                    elif "m_bond" in line:
                        section = "bond"
                    elif ":::" in line and "ready" not in section:
                        section = section + "ready"
                    elif ":::" in line and "ready" in section:
                        section = None
                    elif section == "atom ready":
                        # read in atoms to list
                        continue
                    elif section == "bond ready":
                        # read in bond atom numbers, populate later with atoms
                        continue
                    else:
                        continue
                    # This would probably be better as a function in the structure
                    # class but I wanted this as upstream as possible so I didn't
                    # have to worry about other coding issues. The MMO file lists
                    # the bonds, angles, and torsions in some order that I am unsure
                    # of. It seems consistent with the same filename but with two
                    # files with the exact same structure the ordering is off. This
                    # reorders the lists before being added to the structure class.
                    if "Input filename" in line:
                        count_input += 1
                    if "Input Structure Name" in line:
                        count_structure += 1
                    count_previous = count_current
                    # Sometimes only one of the above ("Input filename" and
                    # "Input Structure Name") is used, sometimes both are used.
                    # count_current will make sure you catch both.
                    count_current = max(count_input, count_structure)
                    # If these don't match, then we reached the end of a
                    # structure.
                    if count_current != count_previous:
                        bonds = []
                        angles = []
                        torsions = []
                        current_structure = Structure(self.filename)
                        self._structures.append(current_structure)
                    # For each structure we come across, look for sections that
                    # we are interested in: those pertaining to bonds, angles,
                    # and torsions. Of course more could be added. We set the
                    # section to None to mark the end of a section, and we leave
                    # it None for parts of the file we don't care about.
                    if "BOND LENGTHS AND STRETCH ENERGIES" in line:
                        section = "bond"
                    if "ANGLES, BEND AND STRETCH BEND ENERGIES" in line:
                        section = "angle"
                    if "BEND-BEND ANGLES AND ENERGIES" in line:
                        section = None
                    if "DIHEDRAL ANGLES AND TORSIONAL ENERGIES" in line:
                        section = "torsion"
                    if "DIHEDRAL ANGLES AND TORSIONAL CROSS-TERMS" in line:
                        section = None
                    if section == "bond":
                        bond = self.read_line_for_bond(line)
                        if bond is not None:
                            # current_structure.bonds.append(bond)
                            bonds.append(bond)
                    if section == "angle":
                        angle = self.read_line_for_angle(line)
                        if angle is not None:
                            # current_structure.angles.append(angle)
                            angles.append(angle)
                    if section == "torsion":
                        torsion = self.read_line_for_torsion(line)
                        if torsion is not None:
                            # current_structure.torsions.append(torsion)
                            torsions.append(torsion)
                    if "Connection Table" in line:
                        # Sort the bonds, angles, and torsions before the start
                        # of a new structure
                        if bonds:
                            bonds.sort(key=lambda x: (x.atom_nums[0], x.atom_nums[1]))
                            current_structure.bonds.extend(bonds)
                        if angles:
                            angles.sort(key=lambda x: (x.atom_nums[1], x.atom_nums[0], x.atom_nums[2]))
                            current_structure.angles.extend(angles)
                        if torsions:
                            torsions.sort(
                                key=lambda x: (x.atom_nums[1], x.atom_nums[2], x.atom_nums[0], x.atom_nums[3])
                            )
                            current_structure.torsions.extend(torsions)
            logger.log(5, f"  -- Imported {len(self._structures)} structure(s).")
        return self._structures
