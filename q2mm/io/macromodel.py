"""Parsers for MacroModel ``.mmo`` and log files.

Provides ``MacroModel`` for extracting structural data (bonds, angles,
torsions) from ``.mmo`` files and ``MacroModelLog`` for reading
mass-weighted Hessian matrices from MacroModel log files.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from q2mm import constants as co
from q2mm.io._helpers import _TextFileBase
from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
    canonicalize_torsion_env_id,
)
from q2mm.models.molecule import Angle, Bond, Molecule, Torsion

logger = logging.getLogger(__name__)


@dataclass
class _MacroModelAtom:
    """One atom parsed from a MacroModel ``.mmo`` connectivity block."""

    index: int
    element: str
    atom_type_name: str
    x: float
    y: float
    z: float

    @property
    def coords(self) -> np.ndarray:
        """Cartesian coordinates of this atom, as ``[x, y, z]``."""
        return np.array([self.x, self.y, self.z])

    @property
    def symbol(self) -> str:
        """Resolve this atom's element symbol."""
        if self.element:
            return _extract_element(self.element)
        return _extract_element(self.atom_type_name) if self.atom_type_name else ""


@dataclass
class _MacroModelTerm:
    """One bond, angle, or torsion parsed from a MacroModel ``.mmo`` energy line."""

    atom_nums: list[int]
    value: float
    ff_row: int
    comment: str = ""


@dataclass
class _MacroModelRecord:
    """One structure/conformer staging record parsed from a MacroModel file.

    MacroModel ``.mmo`` files always supply full bond/angle/torsion
    connectivity for every structure — there is no "topology omitted"
    case for this format — so ``bonds``/``angles``/``torsions`` here are
    always converted as explicit/authoritative, never inferred.
    """

    origin_name: str
    atoms: list[_MacroModelAtom] = field(default_factory=list)
    bonds: list[_MacroModelTerm] = field(default_factory=list)
    angles: list[_MacroModelTerm] = field(default_factory=list)
    torsions: list[_MacroModelTerm] = field(default_factory=list)


def _read_bond_line(line: str) -> _MacroModelTerm | None:
    """Parse a single line for bond data."""
    match = co.RE_BOND.match(line)
    if match:
        atom_nums = [int(x) for x in [match.group(1), match.group(2)]]
        value = float(match.group(3))
        comment = match.group(4).strip()
        ff_row = int(match.group(5))
        return _MacroModelTerm(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
    return None


def _read_angle_line(line: str) -> _MacroModelTerm | None:
    """Parse a single line for angle data, ordering terminal atoms ascending."""
    match = co.RE_ANGLE.match(line)
    if match:
        atom_nums = [int(x) for x in [match.group(1), match.group(2), match.group(3)]]
        if atom_nums[0] > atom_nums[2]:
            atom_nums = [atom_nums[2], atom_nums[1], atom_nums[0]]
        value = float(match.group(4))
        comment = match.group(5).strip()
        ff_row = int(match.group(6))
        return _MacroModelTerm(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
    return None


def _read_torsion_line(line: str) -> _MacroModelTerm | None:
    """Parse a single line for torsion data, ordering central atoms ascending."""
    match = co.RE_TORSION.match(line)
    if match:
        atom_nums = [int(x) for x in [match.group(1), match.group(2), match.group(3), match.group(4)]]
        if atom_nums[1] > atom_nums[2]:
            atom_nums = [atom_nums[3], atom_nums[2], atom_nums[1], atom_nums[0]]
        value = float(match.group(5))
        comment = match.group(6).strip()
        ff_row = int(match.group(7))
        return _MacroModelTerm(atom_nums=atom_nums, comment=comment, value=value, ff_row=ff_row)
    return None


def _molecule_from_macromodel_record(record: _MacroModelRecord) -> Molecule:
    """Convert a :class:`_MacroModelRecord` into a :class:`~q2mm.models.molecule.Molecule`.

    Bonds/angles/torsions are always explicit/authoritative — MacroModel
    ``.mmo`` files always supply full connectivity for every structure.
    """
    symbols = [atom.symbol for atom in record.atoms]
    atom_types = [atom.atom_type_name for atom in record.atoms]
    coords = [atom.coords for atom in record.atoms]

    def _atom(num: int) -> _MacroModelAtom:
        return record.atoms[num - 1]

    bonds = tuple(
        Bond(
            atom_i=b.atom_nums[0] - 1,
            atom_j=b.atom_nums[1] - 1,
            elements=(_atom(b.atom_nums[0]).symbol, _atom(b.atom_nums[1]).symbol),
            length=b.value,
            env_id=canonicalize_bond_env_id([_atom(n).atom_type_name for n in b.atom_nums]),
            ff_row=b.ff_row,
        )
        for b in record.bonds
    )
    angles = tuple(
        Angle(
            atom_i=a.atom_nums[0] - 1,
            atom_j=a.atom_nums[1] - 1,
            atom_k=a.atom_nums[2] - 1,
            elements=(_atom(a.atom_nums[0]).symbol, _atom(a.atom_nums[1]).symbol, _atom(a.atom_nums[2]).symbol),
            value=a.value,
            env_id=canonicalize_angle_env_id([_atom(n).atom_type_name for n in a.atom_nums]),
            ff_row=a.ff_row,
        )
        for a in record.angles
    )
    torsions = tuple(
        Torsion(
            atom_i=t.atom_nums[0] - 1,
            atom_j=t.atom_nums[1] - 1,
            atom_k=t.atom_nums[2] - 1,
            atom_l=t.atom_nums[3] - 1,
            elements=(
                _atom(t.atom_nums[0]).symbol,
                _atom(t.atom_nums[1]).symbol,
                _atom(t.atom_nums[2]).symbol,
                _atom(t.atom_nums[3]).symbol,
            ),
            value=t.value,
            env_id=canonicalize_torsion_env_id([_atom(n).atom_type_name for n in t.atom_nums]),
            ff_row=t.ff_row,
        )
        for t in record.torsions
    )

    return Molecule(
        symbols=tuple(symbols),
        atom_types=tuple(atom_types),
        geometry=np.array(coords, dtype=float),
        bonds=bonds,
        angles=angles,
        torsions=torsions,
        name=record.origin_name,
    )


class MacroModel(_TextFileBase):
    """Extract structural data from MacroModel ``.mmo`` files.

    Reads bond lengths, angles, and torsions for each structure entry
    in the ``.mmo`` file.
    """

    def __init__(self, path: str) -> None:
        """Initialize a MacroModel instance.

        Args:
            path (str): Path to the MacroModel ``.mmo`` file.

        """
        super().__init__(path)
        self._records: list[_MacroModelRecord] | None = None

    @property
    def _records_parsed(self) -> list[_MacroModelRecord]:
        """Parsed records with bonds, angles, and torsions (private staging).

        Not part of the public API — see :attr:`molecules`.
        """
        if not self._records:
            logger.log(10, f"READING: {self.filename}")
            records: list[_MacroModelRecord] = []
            with open(self.path) as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
                bonds: list[_MacroModelTerm] = []
                angles: list[_MacroModelTerm] = []
                torsions: list[_MacroModelTerm] = []
                atoms: list[_MacroModelAtom] = []
                current_record: _MacroModelRecord | None = None
                section: str | None = None
                for line in f:
                    if "Atomic Charges, Coordinates and Connectivity" in line:
                        section = "atoms"
                        continue
                    if section == "atoms":
                        if "(" in line:
                            split = [item.strip() for item in line.split()]
                            atom_num = split[2][:-1]
                            ele_name = re.sub(r"[0-9]", "", split[0])
                            atom = _MacroModelAtom(
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
                            bonds.sort(key=lambda x: (x.atom_nums[0], x.atom_nums[1]))
                            angles.sort(key=lambda x: (x.atom_nums[1], x.atom_nums[0], x.atom_nums[2]))
                            torsions.sort(
                                key=lambda x: (x.atom_nums[1], x.atom_nums[2], x.atom_nums[0], x.atom_nums[3])
                            )
                            assert current_record is not None
                            current_record.bonds = bonds
                            current_record.angles = angles
                            current_record.torsions = torsions
                            if atoms:
                                atoms.sort(key=lambda x: x.index)
                                current_record.atoms.extend(atoms)
                    if "Input filename" in line:
                        count_input += 1
                    if "Input Structure Name" in line:
                        count_structure += 1
                    count_previous = count_current
                    count_current = max(count_input, count_structure)
                    if count_current != count_previous:
                        bonds = []
                        angles = []
                        torsions = []
                        atoms = []
                        current_record = _MacroModelRecord(self.filename)
                        records.append(current_record)
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
                        bond = _read_bond_line(line)
                        if bond is not None:
                            bonds.append(bond)
                    if section == "angle":
                        angle = _read_angle_line(line)
                        if angle is not None:
                            angles.append(angle)
                    if section == "torsion":
                        torsion = _read_torsion_line(line)
                        if torsion is not None:
                            torsions.append(torsion)
            self._records = records
        return self._records

    @property
    def molecules(self) -> list[Molecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Molecule` objects."""
        return [_molecule_from_macromodel_record(record) for record in self._records_parsed]


class MacroModelLog(_TextFileBase):
    """Retrieve data from MacroModel log files.

    The Hessian matrix read from these files is mass-weighted.
    """

    def __init__(self, path: str) -> None:
        """Initialize a MacroModelLog instance.

        Args:
            path (str): Path to the MacroModel log file.

        """
        super().__init__(path)
        self._hessian: np.ndarray | None = None
        self._records: list[_MacroModelRecord] | None = None

    @property
    def hessian(self) -> np.ndarray:
        """numpy.ndarray: Mass-weighted Hessian matrix read from the log file.

        Returns:
            (numpy.ndarray): 2-D Hessian of shape ``(N*3, N*3)`` where
                *N* is the number of atoms.

        """
        if self._hessian is None:
            logger.log(10, f"READING: {self.filename}")
            with open(self.path) as f:
                lines = f.read()
            atoms_match = re.search(r"Read\s+(\d+)\s+atoms.", lines)
            if atoms_match is None:
                raise ValueError(f"Could not find atom count in MacroModel log {self.filename}.")
            num_atoms = int(atoms_match.group(1))
            logger.log(5, f"  -- Read {num_atoms} atoms.")

            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(5, f"  -- Creating {hessian.shape} Hessian matrix.")
            words = lines.split()
            section_hessian = False
            start_row = False
            start_col = False
            row_num = 0
            col_nums: list[int] = []
            elements: list[float] = []
            for i, word in enumerate(words):
                if word == "Mass-weighted":
                    section_hessian = True
                    continue
                if word == "Eigenvalues:":
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    section_hessian = False
                    break
                if section_hessian and start_col and word == "Element":
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    start_col = False
                    start_row = True
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    continue
                if section_hessian and word == "Element":
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    start_row = True
                    continue
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
    def _records_parsed(self) -> list[_MacroModelRecord]:
        """Parsed records from the log file (private staging).

        Not part of the public API — see :attr:`molecules`.
        """
        if self._records is None:
            logger.log(10, f"READING: {self.filename}")
            records: list[_MacroModelRecord] = []
            with open(self.path) as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
                bonds: list[_MacroModelTerm] = []
                angles: list[_MacroModelTerm] = []
                torsions: list[_MacroModelTerm] = []
                current_record: _MacroModelRecord | None = None
                section: str | None = None
                for line in f:
                    if "m_atom" in line:
                        section = "atom"
                    elif "m_bond" in line:
                        section = "bond"
                    elif ":::" in line and section is not None and "ready" not in section:
                        section = section + "ready"
                    elif ":::" in line and section is not None and "ready" in section:
                        section = None
                    elif section in ("atom ready", "bond ready"):
                        continue
                    else:
                        continue
                    if "Input filename" in line:
                        count_input += 1
                    if "Input Structure Name" in line:
                        count_structure += 1
                    count_previous = count_current
                    count_current = max(count_input, count_structure)
                    if count_current != count_previous:
                        bonds = []
                        angles = []
                        torsions = []
                        current_record = _MacroModelRecord(self.filename)
                        records.append(current_record)
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
                        bond = _read_bond_line(line)
                        if bond is not None:
                            bonds.append(bond)
                    if section == "angle":
                        angle = _read_angle_line(line)
                        if angle is not None:
                            angles.append(angle)
                    if section == "torsion":
                        torsion = _read_torsion_line(line)
                        if torsion is not None:
                            torsions.append(torsion)
                    if "Connection Table" in line and current_record is not None:
                        bonds.sort(key=lambda x: (x.atom_nums[0], x.atom_nums[1]))
                        angles.sort(key=lambda x: (x.atom_nums[1], x.atom_nums[0], x.atom_nums[2]))
                        torsions.sort(key=lambda x: (x.atom_nums[1], x.atom_nums[2], x.atom_nums[0], x.atom_nums[3]))
                        current_record.bonds = bonds
                        current_record.angles = angles
                        current_record.torsions = torsions
            self._records = records
            logger.log(5, f"  -- Imported {len(self._records)} structure(s).")
        return self._records

    @property
    def molecules(self) -> list[Molecule]:
        """Parsed structures as :class:`~q2mm.models.molecule.Molecule` objects."""
        return [_molecule_from_macromodel_record(record) for record in self._records_parsed]
