from __future__ import annotations
import logging
import numpy as np
import os
import re
from q2mm import constants as co
from q2mm.parsers.base import File
from q2mm.parsers.structures import Atom, Bond, Angle, Torsion, Structure

logger = logging.getLogger(__name__)

# MacroModel configuration constants (moved from constants.py)
COM_FORM = " {0:4}{1:>8}{2:>7}{3:>7}{4:>7}{5:>11.4f}{6:>11.4f}{7:>11.4f}{8:>11.4f}\n"
LABEL_SUITE = r"SUITE_\w+"
LABEL_MACRO = "MMOD_MACROMODEL"
LIC_SUITE = re.compile(rf"(?<!_){LABEL_SUITE}\s+(\d+)\sof\s\d+\s" r"tokens\savailable")
LIC_MACRO = re.compile(rf"{LABEL_MACRO}\s+(\d+)\sof\s\d+\stokens\s" "available")
MIN_SUITE_TOKENS = 2
MIN_MACRO_TOKENS = 2


class MacroModel(File):
    """
    Extracts data from MacroModel .mmo files.
    """

    def __init__(self, path):
        super().__init__(path)
        self._structures = None

    @property
    def structures(self):  # TODO make this read atoms for consistency and bc need num atoms for hessian read
        if self._structures is None or self._structures == []:
            logger.log(10, f"READING: {self.filename}")
            self._structures = []
            with open(self.path) as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
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


# This could use some documentation. Looks pretty though.
def geo_from_points(*args):
    x1 = args[0][0]
    y1 = args[0][1]
    z1 = args[0][2]
    x2 = args[1][0]
    y2 = args[1][1]
    z2 = args[1][2]
    if len(args) == 2:
        bond = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        return float(bond)
    x3 = args[2][0]
    y3 = args[2][1]
    z3 = args[2][2]
    if len(args) == 3:
        dist_21 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        dist_23 = np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2 + (z2 - z3) ** 2)
        dist_13 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2 + (z1 - z3) ** 2)
        angle = np.acos((dist_21**2 + dist_23**2 - dist_13**2) / (2 * dist_21 * dist_23))
        angle = np.degrees(angle)
        return float(angle)
    x4 = args[3][0]
    y4 = args[3][1]
    z4 = args[3][2]
    if len(args) == 4:
        vect_21 = [x2 - x1, y2 - y1, z2 - z1]
        vect_32 = [x3 - x2, y3 - y2, z3 - z2]
        vect_43 = [x4 - x3, y4 - y3, z4 - z3]
        x_ab = np.cross(vect_21, vect_32)
        x_bc = np.cross(vect_32, vect_43)
        norm_ab = x_ab / (np.sqrt(x_ab[0] ** 2 + x_ab[1] ** 2 + x_ab[2] ** 2))
        norm_bc = x_bc / (np.sqrt(x_bc[0] ** 2 + x_bc[1] ** 2 + x_bc[2] ** 2))
        mag_ab = np.sqrt(norm_ab[0] ** 2 + norm_ab[1] ** 2 + norm_ab[2] ** 2)
        mag_bc = np.sqrt(norm_bc[0] ** 2 + norm_bc[1] ** 2 + norm_bc[2] ** 2)
        angle = np.acos(np.dot(norm_ab, norm_bc) / (mag_ab * mag_bc))
        torsion = angle * (180 / np.pi)
        return torsion


class MacroModelLog(File):
    """
    Used to retrieve data from MacroModel log files. Hessian is Mass weighted.
    """

    def __init__(self, path):
        super().__init__(path)
        self._hessian = None
        self._structures = None

    @property
    def hessian(self):
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
