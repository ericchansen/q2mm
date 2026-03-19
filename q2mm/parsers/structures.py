from __future__ import annotations
import logging
import logging.config
from string import digits
import numpy as np
import os
import re
from q2mm import constants as co
from q2mm import utilities
from q2mm.models.datum import Datum

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


class Atom:
    """
    Data class for a single atom.
    """

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
        atom_type: str = None,
        atom_type_name: str = None,
        atomic_num: int = None,
        atomic_mass: float = None,
        bonded_atom_indices=None,
        coords=None,
        coords_type=None,
        element: str = None,
        exact_mass=None,
        index: int = None,
        partial_charge: float = None,
        x: float = None,
        y: float = None,
        z: float = None,
    ):
        """Atom object containing relevant properties and metadata.

        Units: Angstrom

        Note:
            TODO Values are all optional because of the currently established q2mm code, however this
            is bad practice, any strictly necessary properties should be required arguments, such as atom type, index, and
            coordinates.

        Args:
            atom_type (str, optional): The integer atom type according to atom.typ file. Defaults to None.
            atom_type_name (str, optional): The name of the atom type corresponding to the integer atom type in the atom.typ file. Defaults to None.
            atomic_num (int, optional): Atomic number (element number) of the atom. Defaults to None.
            atomic_mass (float, optional): TODO. Defaults to None.
            bonded_atom_indices (TODO, optional): TODO. Defaults to None.
            coords (TODO maybe np.ndarray, optional): Atom coordinates. Defaults to None.
            coords_type (TODO, optional): TODO Is this even ever used?. Defaults to None.
            element (str, optional): The atom element (e.g. C, N, O, H). Defaults to None.
            exact_mass (_type_, optional): TODO. Defaults to None.
            index (int, optional): The index number of the atom in its original structural file. Defaults to None.
            partial_charge (float, optional): TODO is this even really used?. Defaults to None.
            x (float, optional): X coordinate of the atom. Defaults to None.
            y (float, optional): Y coordinate of the atom. Defaults to None.
            z (float, optional): Z coordinate of the atom. Defaults to None.
        """
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num  # This is the atom index in the original structure file, 1-based NOT 0-based
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
        if coords:  # coordinates are all in Angstroms and Cartesian
            self.x = float(coords[0])
            self.y = float(coords[1])
            self.z = float(coords[2])
        self.props = {}

    def __repr__(self):
        return f"{self.atom_type_name}[{self.x},{self.y},{self.z}]"

    @property
    def coords(self) -> np.ndarray:
        """Getter method for coords property.

        Returns:
            np.ndarray: Array of Cartesian coordinates of atom of form [x, y, z].
        """
        return np.array([self.x, self.y, self.z])

    @coords.setter
    def coords(self, value):
        """Setter method for coords property.

        Args:
            value (TODO): Cartesian coordinates of atom of form [x, y, z]
        """
        try:
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        except TypeError:
            pass

    @property
    def element(self):
        if self._element is None:
            self._element = list(co.MASSES.keys())[self.atomic_num - 1]
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def exact_mass(self):
        if self._exact_mass is None:
            self._exact_mass = co.MASSES[self.element]
        return self._exact_mass

    @exact_mass.setter
    def exact_mass(self, value):
        self._exact_mass = value

    @property
    def is_dummy(self):
        """
        Return True if self is a dummy atom, else return False.

        Returns
        -------
        bool
        """
        # I think 61 is the default dummy atom type in a Schrodinger atom.typ
        # file.
        # Okay, so maybe it's not. Anyway, Tony added an atom type 205 for
        # dummies. It'd be really great if we all used the same atom.typ file
        # someday.
        # Could add in a check for the atom_type number. I removed it.
        if self.atom_type_name == "Du" or self.element == "X" or self.atomic_num == -2:
            return True
        else:
            return False


class DOF:
    """
    Abstract data class for a single degree of freedom.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
    ):
        """Abstract Class for a Degree Of Freedom (DOF) containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the DOF. Defaults to None.
            comment (str, optional): Any comment associated with the DOF TODO Is this used?. Defaults to None.
            value (float, optional): Value of the DOF. Defaults to None.
            ff_row (int, optional): Row of the FF which models the DOF. Defaults to None.
        """
        self.atom_nums: List[int] = atom_nums
        """ TODO atom_indices is a more intuitive name,
        but use of this property is too widespread (with poor referencing) to change atm,
        refactor this name when there is time."""
        self.comment = comment
        self.value = value
        self.ff_row = ff_row

    def __repr__(self):
        return "{}[{}]({})".format(self.__class__.__name__, "-".join(map(str, self.atom_nums)), self.value)

    def as_data(self, **kwargs):
        # Sort of silly to have all this stuff about angles and
        # torsions in here, but they both inherit from this class.
        # I suppose it'd make more sense to create a structural
        # element class that these all inherit from.
        # Warning that I recently changed these labels, and that
        # may have consequences.
        if self.__class__.__name__.lower() == "bond":
            typ = "b"
        elif self.__class__.__name__.lower() == "angle":
            typ = "a"
        elif self.__class__.__name__.lower() == "torsion":
            typ = "t"
        datum = Datum(val=self.value, typ=typ, ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, f"atm_{i + 1}", atom_num)
        for k, v in kwargs.items():
            setattr(datum, k, v)
        return datum

    def is_same_DOF(self, other) -> bool:
        """Comparison operator for DOFs. Returns true if the DOFs are identical
        based on the indices of the atoms involved. Relies on the assumption that
        atom indices are not different from structure to structure, TODO this is a
        fallacy which should be addressed at some point or at least emphasized to
        the user in documentation.

        Args:
            other (DOF): The DOF to which to compare self.

        Returns:
            bool: True if DOF is identical to self, else False.
        """
        assert isinstance(other, DOF)
        return self.atom_nums == other.atom_nums or list(reversed(self.atom_nums)) == other.atom_nums


class Bond(DOF):
    """
    Data class for a single bond.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row", "order"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
        order: str = None,
    ):
        """Bond object containing the bare bones properties necessary.

        Note:
            As of yet, there is no need for this class to also track a force constant as it
            is used exclusively in the newer, schrodinger-independent code like seminario.py,
            but this could be a good place to store that data within a FF object if Param were
            to be replaced/condensed.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the bond of the form [index1, index2]. Defaults to None.
            comment (str, optional): Any comment associated with the Bond TODO Is this used?. Defaults to None.
            value (float, optional): Bond length in Angstrom. Defaults to None.
            ff_row (int, optional): Row of the FF which models the bond. Defaults to None.
            order (int, optional): Bond order (e.g. single bond - 1, double bond - 2...)
        """
        super().__init__(atom_nums, comment, value, ff_row)
        self.order = order


class Angle(DOF):
    """
    Data class for a single angle.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Angle object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the Angle. Defaults to None.
            comment (str, optional): Any comment associated with the Angle TODO Is this used?. Defaults to None.
            value (float, optional): Value of the Angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Angle. Defaults to None.
        """
        super().__init__(atom_nums, comment, value, ff_row)


class Torsion(DOF):
    """
    Data class for a single torsion.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Torsion/Dihedral object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the torsion. Defaults to None.
            comment (str, optional): Any comment associated with the torsion TODO Is this used?. Defaults to None.
            value (float, optional): Value of the torsion angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Torsion. Defaults to None.
        """
        super().__init__(atom_nums, comment, value, ff_row)


class Structure:
    """
    Data for a single structure/conformer/snapshot.
    """

    __slots__ = ["_atoms", "_bonds", "_angles", "_torsions", "hess", "props", "origin_name"]

    def __init__(self, origin_name: str):
        # TODO: This should really be a constructor which accepts a bare minimum of
        # these fields and the rest are optional defaulted to None, good for error-protection
        # and just generally cleaner, more intuitive. An empty structure is never itself used,
        # so why have it as an option which simply complicates error-checking and tracking.
        self._atoms: List[Atom] = None
        self._bonds: List[Bond] = None
        self._angles: List[Angle] = None
        self._torsions: List[Torsion] = None
        self.hess = None
        self.props = {}
        self.origin_name: str = origin_name

    @property
    def coords(self):
        """
        Returns atomic coordinates as a list of lists.
        """
        return [atom.coords for atom in self._atoms]

    @property
    def num_atoms(self):
        if self._atoms is None or self._atoms == []:
            return self.guess_atoms()
        else:
            return len(self.atoms)

    @property
    def atoms(self):
        # if self._atoms == []:
        #     raise Exception(
        #         "structure._atoms is not defined, this must be done on creation."
        #     )
        if self._atoms is None:
            self._atoms: List[Atom] = []
        return self._atoms

    @property
    def bonds(self):
        # if self._bonds == []:
        #     raise Exception(
        #         "structure._bonds is not defined, this must be done on creation."
        #     )
        if self._bonds is None:
            self._bonds: List[Bond] = []
        return self._bonds

    @property
    def angles(self):
        # if self._angles == []:
        #     self._angles = self.identify_angles() TODO move this to Mol2.structures None property if
        if self._angles is None:
            self._angles: List[Angle] = []
        return self._angles

    @property
    def torsions(self):
        # if self._torsions == []:
        #     self._torsions = self.identify_torsions()
        if self._torsions is None:
            self._torsions: List[Torsion] = []
        return self._torsions

    def generalize_to_ff_atom_types(self, equivalency_dict: dict, substr_atom_types: list):
        for atom in self.atoms:
            if atom.atom_type_name not in substr_atom_types and atom.atom_type_name in equivalency_dict:
                atom.atom_type_name = equivalency_dict[atom.atom_type_name]

    def guess_atoms(self) -> int:
        max_atom_index = 0
        for bond in self.bonds:
            max_in_bond = np.max(bond.atom_nums)
            if max_in_bond > max_atom_index:
                max_atom_index = max_in_bond
        return max_atom_index

    # region Methods which ought to be refactored or might be unused but I'm too busy/scared to mess with yet

    def format_coords(self, format="latex", indices_use_charge=None):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.

        latex  - Makes a LaTeX table.
        gauss  - Makes output that matches Gaussian's .com filse.
        jaguar - Just like Gaussian, but include the atom number after the
                 element name in the left column.
        """
        # Formatted for LaTeX.
        if format == "latex":
            output = ["\\begin{tabular}{l S[table-format=3.6] S[table-format=3.6] S[table-format=3.6]}"]
            for i, atom in enumerate(self._atoms):
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
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = list(co.MASSES.keys())[atom.atomic_num - 1]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
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
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = list(co.MASSES.keys())[atom.atomic_num - 1]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                label = f"{ele}{atom.index}"
                output.append(f" {label:<8s}{atom.x:>16.6f}{atom.y:>16.6f}{atom.z:>16.6f}")
            return output

    def select_stuff(self, typ, com_match=None):
        """
        A much simpler version of select_data. It would be nice if select_data
        was a wrapper around this function.
        """
        stuff = []
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or com_match is None:
                stuff.append(thing)
        return stuff

    def select_data(self, typ, com_match=None, **kwargs):
        """
        Selects bonds, angles, or torsions from the structure and returns them
        in the format used as data.

        typ       - 'bonds', 'angles', or 'torsions'.
        com_match - String or None. If None, just returns all of the selected
                    stuff (bonds, angles, or torsions). If a string, selects
                    only those that have this string in their comment.

                    In .mmo files, the comment corresponds to the substructures
                    name. This way, we only fit bonds, angles, and torsions that
                    directly depend on our parameters.
        """
        data = []
        logger.log(1, f">>> typ: {typ}")
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or com_match is None:
                datum = thing.as_data(**kwargs)
                # If it's a torsion we have problems.
                # Have to check whether an angle inside the torsion is near 0 or 180.
                if typ == "torsions":
                    atom_nums = [datum.atm_1, datum.atm_2, datum.atm_3, datum.atm_4]
                    angle_atoms_1 = [atom_nums[0], atom_nums[1], atom_nums[2]]
                    angle_atoms_2 = [atom_nums[1], atom_nums[2], atom_nums[3]]
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_1):
                            angle_1 = angle.value
                            break
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_2):
                            angle_2 = angle.value
                            break
                    try:
                        logger.log(1, f">>> atom_nums: {atom_nums}")
                        logger.log(1, f">>> angle_1: {angle_1} / angle_2: {angle_2}")
                    except UnboundLocalError:
                        logger.error(f">>> atom_nums: {atom_nums}")
                        logger.error(f">>> angle_atoms_1: {angle_atoms_1}")
                        logger.error(f">>> angle_atoms_2: {angle_atoms_2}")
                        if "angle_1" not in locals():
                            logger.error("Can't identify angle_1!")
                        else:
                            logger.error(f">>> angle_1: {angle_1}")
                        if "angle_2" not in locals():
                            logger.error("Can't identify angle_2!")
                        else:
                            logger.error(f">>> angle_2: {angle_2}")
                        logger.warning("WARNING: Using torsion anyway!")
                        data.append(datum)
                    if (
                        -20.0 < angle_1 < 20.0
                        or 160.0 < angle_1 < 200.0
                        or -20.0 < angle_2 < 20.0
                        or 160.0 < angle_2 < 200.0
                    ):
                        logger.log(1, ">>> angle_1 or angle_2 is too close to 0 or 180!")
                        pass
                    else:
                        data.append(datum)
                    # atom_coords = [x.coords for x in atoms]
                    # tor_1 = geo_from_points(
                    #     atom_coords[0], atom_coords[1], atom_coords[2])
                    # tor_2 = geo_from_points(
                    #     atom_coords[1], atom_coords[2], atom_coords[3])
                    # logger.log(1, '>>> tor_1: {} / tor_2: {}'.format(
                    #     tor_1, tor_2))
                    # if -5. < tor_1 < 5. or 175. < tor_1 < 185. or \
                    #         -5. < tor_2 < 5. or 175. < tor_2 < 185.:
                    #     logger.log(
                    #         1,
                    #         '>>> tor_1 or tor_2 is too close to 0 or 180!')
                    #     pass
                    # else:
                    #     data.append(datum)
                else:
                    data.append(datum)
        assert data, "No data actually retrieved!"
        return data

    def get_aliph_hyds(self):
        """
        Returns the atom numbers of aliphatic hydrogens. These hydrogens
        are always assigned a partial charge of zero in MacroModel
        calculations.

        This should be subclassed into something is MM3* specific.
        """
        aliph_hyds = []
        for atom in self._atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = self._atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyds.append(atom)
        logger.log(5, f"  -- {len(aliph_hyds)} aliphatic hydrogen(s).")
        return aliph_hyds

    def get_hyds(self):
        """
        Returns the atom numbers of any default MacroModel type hydrogens.

        This should be subclassed into something is MM3* specific.
        """
        hyds = []
        for atom in self._atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    hyds.append(atom)
        logger.log(5, f"  -- {len(hyds)} hydrogen(s).")
        return hyds

    def get_dummy_atom_indices(self):
        """
        Returns a list of integers where each integer corresponds to an atom
        that is a dummy atom.

        Returns
        -------
        list of integers
        """
        dummies = []
        for atom in self._atoms:
            if atom.is_dummy:
                logger.log(10, f"  -- Identified {atom} as a dummy atom.")
                dummies.append(atom.index)
        return dummies

    # endregion

    def identify_angles(self) -> List[Angle]:
        """Returns angles identified and measured within self Structure.

        Note:
            TODO May need to add same logic of 0 vs 180 as in filetypes.py

        Returns:
            List[Angle]: angles in self Structure
        """
        angles: List[Angle] = []
        i = 0
        for a in self.bonds:
            i += 1
            for b in self.bonds[i:]:
                a1_index, a2_index = a.atom_nums
                b1_index, b2_index = b.atom_nums
                if a1_index == b1_index:
                    if a2_index != b2_index:
                        angle = utilities.measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a2_index, a1_index, b2_index], value=angle))
                if a1_index == b2_index:
                    if a2_index != b1_index:
                        angle = utilities.measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a2_index, a1_index, b1_index], value=angle))
                if a2_index == b2_index:
                    if a1_index != b1_index:
                        angle = utilities.measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a1_index, a2_index, b1_index], value=angle))
                if a2_index == b1_index:
                    if a1_index != b2_index:
                        angle = utilities.measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(Angle(atom_nums=[a1_index, a2_index, b2_index], value=angle))
        return angles

    def identify_torsions(self):  # TODO
        raise NotImplementedError

    def get_atoms_in_DOF(self, dof: DOF) -> List[Atom]:
        """Returns a list of Atom objects which are involved in the DOF as implied by atom indices.

        Args:
            dof (DOF): Degree of Freedom (Bond, Angle, etc.) to query

        Returns:
            List[Atom]: Atom objects involved in the DOF.
        """
        return [self.atoms[idx - 1] for idx in dof.atom_nums]

    def get_DOF_atom_types_dict(self) -> dict:
        """Returns a dictionary of the atom types which correspond to each DOF in self.

        Returns:
            dict: dictionary of the form {DOF: [atom_type_name1, atom_type_name2, ...]}
        """
        dof_atom_type_dict = dict()
        for bond in self.bonds:
            dof_atom_type_dict[bond] = [atom.atom_type_name for atom in self.get_atoms_in_DOF(bond)]
        for angle in self.angles:
            dof_atom_type_dict[angle] = [atom.atom_type_name for atom in self.get_atoms_in_DOF(angle)]
        return dof_atom_type_dict

    def get_eqbm_geom_values(self):
        """
        Gather bonds and angles from structures. Adapted from parameters.py code.

        Ex.:
          bond_dic = {1857: [2.2233, 2.2156, 2.5123],
                      1858: [1.3601, 1.3535, 1.3532]
                     }
        """

        bond_dic = {}
        angle_dic = {}
        torsion_dic = {}
        for bond in self.bonds:
            if bond.ff_row in bond_dic:
                bond_dic[bond.ff_row].append(bond.value)
            else:
                bond_dic[bond.ff_row] = [bond.value]
        for angle in self.angles:
            if angle.ff_row in angle_dic:
                angle_dic[angle.ff_row].append(angle.value)
            else:
                angle_dic[angle.ff_row] = [angle.value]
        for torsion in self.torsions:
            if torsion.ff_row in torsion_dic:
                torsion_dic[torsion.ff_row].append(torsion.value)
            else:
                torsion_dic[torsion.ff_row] = [torsion.value]
        return bond_dic, angle_dic, torsion_dic


def check_mm_dummy(hess, dummy_indices):
    """
    Removes dummy atom rows and columns from the Hessian based upon
    dummy_indices.

    Arguments
    ---------
    hess : np.matrix
    dummy_indices : list of integers
                    Integers correspond to the indices to be removed from the
                    np.matrix of the Hessian.

    Returns
    -------
    np.matrix
    """
    hess = np.delete(hess, dummy_indices, 0)
    hess = np.delete(hess, dummy_indices, 1)
    logger.log(15, f"Created {hess.shape} Hessian w/o dummy atoms.")
    return hess


def get_dummy_hessian_indices(dummy_indices):
    """
    Takes a list of indices for the dummy atoms and returns another list of
    integers corresponding to the rows of the eigenvectors to remove
    for those those dummy atoms.

    Arguments
    ---------
    dummy_indices : list of integers
                    Indices for the dummy atoms.

    Returns
    -------
    list of integers
    """
    hess_dummy_indices = []
    for index in dummy_indices:
        hess_index = (index - 1) * 3
        hess_dummy_indices.append(hess_index)
        hess_dummy_indices.append(hess_index + 1)
        hess_dummy_indices.append(hess_index + 2)
    return hess_dummy_indices
