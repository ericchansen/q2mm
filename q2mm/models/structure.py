"""Data classes for molecular structures, atoms, and degrees of freedom.

This module defines the core data structures used to represent molecular
geometry: atoms, bonds, angles, torsions, and full molecular structures.

Note:
    :class:`q2mm.models.molecule.Q2MMMolecule` is the preferred
    representation for new code.  These legacy classes remain in use by
    several parsers and will be retained as long as those parsers exist.

"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from q2mm import constants as co

if TYPE_CHECKING:
    from q2mm.parsers.datum import Datum

logger = logging.getLogger(__name__)


class Atom:
    """Data class for a single atom."""

    __slots__ = [
        "atom_type",
        "atom_type_name",
        "atomic_num",
        "atomic_mass",
        "bonded_atom_indices",
        "coords_type",
        "_element",
        "_exact_mass",
        "index",
        "partial_charge",
        "x",
        "y",
        "z",
        "props",
    ]

    def __init__(
        self,
        atom_type: str | None = None,
        atom_type_name: str | None = None,
        atomic_num: int | None = None,
        atomic_mass: float | None = None,
        bonded_atom_indices: list[int] | None = None,
        coords: list[float] | None = None,
        coords_type: str | None = None,
        element: str | None = None,
        exact_mass: float | None = None,
        index: int | None = None,
        partial_charge: float | None = None,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> None:
        """Atom object containing relevant properties and metadata.

        Units: Angstrom

        Note:
            Values are all optional because of the currently established q2mm
            code, however this is bad practice; any strictly necessary
            properties should be required arguments, such as atom type, index,
            and coordinates.

        Args:
            atom_type (str, optional): The integer atom type according to
                atom.typ file. Defaults to None.
            atom_type_name (str, optional): The name of the atom type
                corresponding to the integer atom type in the atom.typ file.
                Defaults to None.
            atomic_num (int, optional): Atomic number (element number) of
                the atom. Defaults to None.
            atomic_mass (float | None, optional): Atomic mass of the atom
                in atomic mass units. Defaults to None.
            bonded_atom_indices (list[int] | None, optional): Indices of
                atoms bonded to this atom. Defaults to None.
            coords (list[float] | None, optional): Atom coordinates as
                ``[x, y, z]``. Defaults to None.
            coords_type (str | None, optional): Coordinate type label.
                Defaults to None.
            element (str, optional): The atom element (e.g. C, N, O, H).
                Defaults to None.
            exact_mass (float | None, optional): Exact isotopic mass of the
                atom. Defaults to None.
            index (int, optional): The index number of the atom in its
                original structural file. Defaults to None.
            partial_charge (float, optional): Partial charge of the atom.
                Defaults to None.
            x (float, optional): X coordinate of the atom. Defaults to None.
            y (float, optional): Y coordinate of the atom. Defaults to None.
            z (float, optional): Z coordinate of the atom. Defaults to None.

        """
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num
        self.atomic_mass = atomic_mass
        self.bonded_atom_indices = bonded_atom_indices
        self.coords_type = coords_type
        self._element = element
        self._exact_mass = exact_mass
        self.index = index
        self.partial_charge = partial_charge
        self.x = x
        self.y = y
        self.z = z
        if coords is not None:  # coordinates are all in Angstroms and Cartesian
            self.x = float(coords[0])
            self.y = float(coords[1])
            self.z = float(coords[2])
        self.props = {}

    def __repr__(self) -> str:
        return f"{self.atom_type_name}[{self.x},{self.y},{self.z}]"

    @property
    def coords(self) -> np.ndarray:
        """Getter method for coords property.

        Returns:
            (np.ndarray): Array of Cartesian coordinates of atom of form [x, y, z].

        """
        return np.array([self.x, self.y, self.z])

    @coords.setter
    def coords(self, value: list[float] | np.ndarray) -> None:
        """Setter method for coords property.

        Args:
            value (list[float] | np.ndarray): Cartesian coordinates of atom
                of form ``[x, y, z]``.

        """
        try:
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        except TypeError:
            pass

    @property
    def element(self) -> str:
        """Element symbol for this atom.

        Returns:
            (str): Element symbol (e.g. ``'C'``, ``'N'``, ``'O'``).

        Raises:
            ValueError: If ``atomic_num`` is ``None`` or invalid and no
                element was explicitly set.

        """
        if self._element is None:
            if self.atomic_num is None or self.atomic_num < 1:
                raise ValueError(
                    f"Cannot derive element: atomic_num={self.atomic_num!r}. "
                    "Set the element explicitly for dummy or special atoms."
                )
            self._element = list(co.MASSES.keys())[self.atomic_num - 1]
        return self._element

    @element.setter
    def element(self, value: str) -> None:
        """Set the element symbol, overriding automatic lookup."""
        self._element = value

    @property
    def exact_mass(self) -> float:
        """Exact isotopic mass of this atom.

        Returns:
            (float): Exact mass in atomic mass units.

        """
        if self._exact_mass is None:
            self._exact_mass = co.MASSES[self.element]
        return self._exact_mass

    @exact_mass.setter
    def exact_mass(self, value: float) -> None:
        """Set the exact isotopic mass, overriding automatic lookup."""
        self._exact_mass = value

    @property
    def is_dummy(self) -> bool:
        """Return whether this atom is a dummy atom.

        Returns:
            (bool): ``True`` if the atom is a dummy atom, ``False`` otherwise.

        """
        if self.atom_type_name == "Du" or self.atomic_num == -2:
            return True
        try:
            return self.element == "X"
        except ValueError:
            return False


class DOF:
    """Abstract data class for a single degree of freedom."""

    __slots__ = ["atom_nums", "comment", "value", "ff_row"]

    def __init__(
        self,
        atom_nums: list[int] | None = None,
        comment: str | None = None,
        value: float | None = None,
        ff_row: int | None = None,
    ) -> None:
        """Abstract class for a degree of freedom (DOF).

        Contains the bare-bones properties necessary for any DOF.

        Args:
            atom_nums (list[int], optional): Indices of the atoms involved
                in the DOF. Defaults to None.
            comment (str, optional): Any comment associated with the DOF.
                Defaults to None.
            value (float, optional): Value of the DOF. Defaults to None.
            ff_row (int, optional): Row of the FF which models the DOF.
                Defaults to None.

        """
        self.atom_nums: list[int] = atom_nums
        # Note: atom_nums are 1-based atom indices. The name is legacy;
        # atom_indices would be clearer but is used widely in downstream code.
        self.comment = comment
        self.value = value
        self.ff_row = ff_row

    def __repr__(self) -> str:
        return "{}[{}]({})".format(self.__class__.__name__, "-".join(map(str, self.atom_nums)), self.value)

    def as_data(self, **kwargs: object) -> Datum:
        """Convert this DOF into a Datum object for data collection.

        Args:
            **kwargs: Additional attributes to set on the returned
                :class:`~q2mm.parsers.datum.Datum` object.

        Returns:
            (Datum): A Datum representation of this DOF.

        """
        from q2mm.parsers.datum import Datum

        _typ_map = {"bond": "b", "angle": "a", "torsion": "t"}
        cls_name = self.__class__.__name__.lower()
        try:
            typ = _typ_map[cls_name]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported DOF subclass '{self.__class__.__name__}' for as_data(); "
                "expected one of Bond, Angle, or Torsion."
            ) from exc
        datum = Datum(val=self.value, typ=typ, ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, f"atm_{i + 1}", atom_num)
        for k, v in kwargs.items():
            setattr(datum, k, v)
        return datum


class Bond(DOF):
    """Data class for a single bond."""

    __slots__ = ["atom_nums", "comment", "value", "ff_row", "order"]

    def __init__(
        self,
        atom_nums: list[int] | None = None,
        comment: str | None = None,
        value: float | None = None,
        ff_row: int | None = None,
        order: str | None = None,
    ) -> None:
        """Bond object containing the bare-bones properties necessary.

        Note:
            As of yet, there is no need for this class to also track a force
            constant as it is used exclusively in the newer,
            Schrödinger-independent code like seminario.py, but this could be
            a good place to store that data within a FF object if Param were
            to be replaced/condensed.

        Args:
            atom_nums (list[int], optional): Indices of the atoms involved in
                the bond of the form ``[index1, index2]``. Defaults to None.
            comment (str, optional): Any comment associated with the Bond.
                Defaults to None.
            value (float, optional): Bond length in Angstrom. Defaults to
                None.
            ff_row (int, optional): Row of the FF which models the bond.
                Defaults to None.
            order (str, optional): Bond order (e.g. single bond ``'1'``,
                double bond ``'2'``). Defaults to None.

        """
        super().__init__(atom_nums, comment, value, ff_row)
        self.order = order


class Angle(DOF):
    """Data class for a single angle."""

    def __init__(
        self,
        atom_nums: list[int] | None = None,
        comment: str | None = None,
        value: float | None = None,
        ff_row: int | None = None,
    ) -> None:
        """Angle object containing the bare-bones properties necessary.

        Args:
            atom_nums (list[int], optional): Indices of the atoms involved in
                the angle. Defaults to None.
            comment (str, optional): Any comment associated with the angle.
                Defaults to None.
            value (float, optional): Value of the angle in degrees. Defaults
                to None.
            ff_row (int, optional): Row of the FF which models the angle.
                Defaults to None.

        """
        super().__init__(atom_nums, comment, value, ff_row)


class Torsion(DOF):
    """Data class for a single torsion."""

    def __init__(
        self,
        atom_nums: list[int] | None = None,
        comment: str | None = None,
        value: float | None = None,
        ff_row: int | None = None,
    ) -> None:
        """Torsion/dihedral object containing the bare-bones properties necessary.

        Args:
            atom_nums (list[int], optional): Indices of the atoms involved in
                the torsion. Defaults to None.
            comment (str, optional): Any comment associated with the torsion.
                Defaults to None.
            value (float, optional): Value of the torsion angle in degrees.
                Defaults to None.
            ff_row (int, optional): Row of the FF which models the torsion.
                Defaults to None.

        """
        super().__init__(atom_nums, comment, value, ff_row)


class Structure:
    """Data class for a single structure, conformer, or snapshot."""

    __slots__ = ["_atoms", "_bonds", "_angles", "_torsions", "hess", "props", "origin_name"]

    def __init__(self, origin_name: str) -> None:
        """Initialise a Structure.

        Args:
            origin_name (str): Name or path of the file from which this
                structure originates.

        """
        self._atoms: list[Atom] = None
        self._bonds: list[Bond] = None
        self._angles: list[Angle] = None
        self._torsions: list[Torsion] = None
        self.hess = None
        self.props = {}
        self.origin_name: str = origin_name

    @property
    def coords(self) -> list[np.ndarray]:
        """Atomic coordinates for every atom in the structure.

        Returns:
            (list[np.ndarray]): List of coordinate arrays, one per atom.

        """
        return [atom.coords for atom in self.atoms]

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the structure.

        Returns:
            (int): Atom count, or a guess based on bond indices when atoms
                are not yet populated.

        """
        if self._atoms is None or self._atoms == []:
            return self.guess_atoms()
        else:
            return len(self.atoms)

    @property
    def atoms(self) -> list[Atom]:
        """List of atoms in the structure.

        Returns:
            (list[Atom]): Atoms belonging to this structure.

        """
        if self._atoms is None:
            self._atoms: list[Atom] = []
        return self._atoms

    @property
    def bonds(self) -> list[Bond]:
        """List of bonds in the structure.

        Returns:
            (list[Bond]): Bonds belonging to this structure.

        """
        if self._bonds is None:
            self._bonds: list[Bond] = []
        return self._bonds

    @property
    def angles(self) -> list[Angle]:
        """List of angles in the structure.

        Returns:
            (list[Angle]): Angles belonging to this structure.

        """
        if self._angles is None:
            self._angles: list[Angle] = []
        return self._angles

    @property
    def torsions(self) -> list[Torsion]:
        """List of torsions in the structure.

        Returns:
            (list[Torsion]): Torsions belonging to this structure.

        """
        if self._torsions is None:
            self._torsions: list[Torsion] = []
        return self._torsions

    def guess_atoms(self) -> int:
        """Estimate the number of atoms from bond indices.

        Returns:
            (int): The highest atom index found across all bonds.

        """
        max_atom_index = 0
        for bond in self.bonds:
            max_in_bond = np.max(bond.atom_nums)
            if max_in_bond > max_atom_index:
                max_atom_index = max_in_bond
        return max_atom_index

    def format_coords(self, format: str = "latex", indices_use_charge: list[int] | None = None) -> list[str]:
        """Format atomic coordinates for output in various file formats.

        Args:
            format (str, optional): Output format. Supported values are
                ``'latex'`` (LaTeX table), ``'gauss'`` (Gaussian ``.com``
                format), and ``'jaguar'`` (Jaguar input format).
                Defaults to ``'latex'``.
            indices_use_charge (list[int] | None, optional): Atom indices
                for which partial charges should be embedded in the
                Gaussian-style output. Only used when *format* is
                ``'gauss'``. Defaults to None.

        Returns:
            (list[str]): Formatted coordinate lines.

        """
        # Formatted for LaTeX.
        if format == "latex":
            output = ["\\begin{tabular}{l S[table-format=3.6] S[table-format=3.6] S[table-format=3.6]}"]
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = list(co.MASSES.keys())[atom.atomic_num - 1]
                else:
                    ele = atom.element
                output.append(f"{ele}{i + 1} & {atom.x:3.6f} & {atom.y:3.6f} & {atom.z:3.6f}\\\\")
            output.append("\\end{tabular}")
            return output
        # Formatted for Gaussian .com's.
        elif format == "gauss":
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = list(co.MASSES.keys())[atom.atomic_num - 1]
                else:
                    ele = atom.element
                if indices_use_charge:
                    if atom.index in indices_use_charge:
                        output.append(f" {ele:s}--{atom.partial_charge:.5f}{atom.x:>16.6f}{atom.y:16.6f}{atom.z:16.6f}")
                    else:
                        output.append(f" {ele:<8s}{atom.x:>16.6f}{atom.y:>16.6f}{atom.z:>16.6f}")
                else:
                    output.append(f" {ele:<8s}{atom.x:>16.6f}{atom.y:>16.6f}{atom.z:>16.6f}")
            return output
        # Formatted for Jaguar.
        elif format == "jaguar":
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = list(co.MASSES.keys())[atom.atomic_num - 1]
                else:
                    ele = atom.element
                label = f"{ele}{atom.index}"
                output.append(f" {label:<8s}{atom.x:>16.6f}{atom.y:>16.6f}{atom.z:>16.6f}")
            return output

    def identify_angles(self) -> list[Angle]:
        """Identify and measure angles within this structure.

        Pairs of bonds sharing an atom are used to construct angle objects
        with measured values.

        Note:
            Does not yet special-case linear angles (0° / 180°).

        Returns:
            (list[Angle]): Angles found in this structure.

        """
        from q2mm.geometry import bond_angle

        angles: list[Angle] = []
        for i, a in enumerate(self.bonds):
            for b in self.bonds[i + 1 :]:
                a1_index, a2_index = a.atom_nums
                b1_index, b2_index = b.atom_nums
                if a1_index == b1_index:
                    if a2_index != b2_index:
                        angle = bond_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a2_index, a1_index, b2_index], value=angle))
                if a1_index == b2_index:
                    if a2_index != b1_index:
                        angle = bond_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a2_index, a1_index, b1_index], value=angle))
                if a2_index == b2_index:
                    if a1_index != b1_index:
                        angle = bond_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a1_index, a2_index, b1_index], value=angle))
                if a2_index == b1_index:
                    if a1_index != b2_index:
                        angle = bond_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a1_index, a2_index, b2_index], value=angle))
        return angles

    def get_atoms_in_DOF(self, dof: DOF) -> list[Atom]:
        """Return atoms involved in the given degree of freedom.

        Args:
            dof (DOF): Degree of freedom (Bond, Angle, etc.) to query.

        Returns:
            (list[Atom]): Atom objects involved in the DOF.

        """
        return [self.atoms[idx - 1] for idx in dof.atom_nums]
