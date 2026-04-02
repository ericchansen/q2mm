"""Parser for AMBER force field parameter files.

Reads and writes AMBER-format force field files, extracting bond, angle,
dihedral, improper, and van der Waals parameters for Q2MM optimization.
"""

import logging
from q2mm import constants as co
from q2mm.parsers.base import FF
from q2mm.models.structure import DOF, Structure
from q2mm.parsers.param import Param

logger = logging.getLogger(__name__)


class AmberFF(FF):
    """AMBER force field reader and writer.

    Handles import and export of AMBER-format force field parameter files,
    supporting bond, angle, dihedral, improper torsion, and van der Waals
    parameter types.

    Attributes:
        units: Unit system constants for AMBER force fields.
        sub_names (list[str]): Names of substructure sections read from the
            force field file.
        AMBER_STEPS (dict[str, float]): AMBER-specific step sizes that
            differ from the global defaults.

    """

    units = co.AMBERFF

    def __init__(
        self,
        path: str | None = None,
        data: list | None = None,
        method: str | None = None,
        params: list[Param] | None = None,
        score: float | None = None,
    ) -> None:
        """Initialize an AmberFF instance.

        Args:
            path (str | None, optional): Path to the AMBER force field file.
                Defaults to None.
            data (object | None, optional): Pre-loaded data. Defaults to None.
            method (str | None, optional): Calculation method label.
                Defaults to None.
            params (list[Param] | None, optional): Pre-loaded parameters.
                Defaults to None.
            score (float | None, optional): Objective function score.
                Defaults to None.

        """
        super().__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None

    # AMBER-specific step sizes that differ from the global defaults.
    AMBER_STEPS = {"bf": 10.0, "af": 10.0, "df": 10.0}

    @property
    def lines(self) -> list[str]:
        """Lines of the force field file.

        Returns:
            (list[str]): All lines from the force field file, lazily loaded
                from disk on first access.

        """
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x: list[str]) -> None:
        """Set the cached file lines, bypassing disk I/O."""
        self._lines = x

    def import_ff(self, path: str | None = None, sub_search: str = "OPT") -> None:
        """Read parameters from an AMBER force field file.

        Parses bond, angle, dihedral, improper, and van der Waals parameters
        from the file, applying AMBER-specific step sizes.

        Args:
            path (str | None, optional): Path to the force field file.
                Defaults to ``self.path``.
            sub_search (str, optional): Marker string used to identify
                optimizable sections. Defaults to ``"OPT"``.

        """
        if path is None:
            path = self.path
        bonds = ["bond", "bond3", "bond4", "bond5"]
        pibonds = ["pibond", "pibond3", "pibond4", "pibond5"]
        angles = ["angle", "angle3", "angle4", "angle5"]
        torsions = ["torsion", "torsion4", "torsion5"]
        dipoles = ["dipole", "dipole3", "dipole4", "dipole5"]
        self.params: list[Param] = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
        count = 0
        with open(path) as f:
            logger.log(15, f"READING: {path}")
            for i, line in enumerate(f):
                split = line.split()
                if not q2mm_sec and "# Q2MM" in line:
                    q2mm_sec = True
                elif q2mm_sec and "#" in line[0]:
                    self.sub_names.append(line[1:])
                    if "OPT" in line:
                        gather_data = True
                    else:
                        gather_data = False
                if gather_data and split:
                    if "MASS" in line and count == 0:
                        count = 1
                        continue
                    if "BOND" in line and count == 1:
                        count = 2
                        continue
                    elif count == 1 and "ANGL" not in line:
                        # atom symbol:atomic mass:atomic polarizability
                        at = split[0]  # need number if it matters
                        el = split[0]
                        mass = split[1]
                        # no need for atom label
                    # BOND
                    if "ANGL" in line and count == 2:
                        count = 3
                        continue
                    elif count == 2 and "DIHE" not in line:
                        # A1-A2 Force Const in kcal/mol/(A**2): Eq. length in A
                        AA = line[:5].split("-")
                        BB = line[5:].split()
                        at = [AA[0], AA[1]]
                        self.params.extend(
                            (
                                Param(
                                    atom_types=at,
                                    ptype="bf",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                Param(
                                    atom_types=at,
                                    ptype="be",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # ANGLE
                    if "DIHE" in line and count == 3:
                        count = 4
                        continue
                    elif count == 3 and "IMPR" not in line:
                        AA = line[: 2 + 3 * 2].split("-")
                        BB = line[2 + 3 * 2 :].split()
                        at = [AA[0], AA[1], AA[2]]
                        self.params.extend(
                            (
                                Param(
                                    atom_types=at,
                                    ptype="af",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                Param(
                                    atom_types=at,
                                    ptype="ae",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # Dihedral
                    if "IMPR" in line and count == 4:
                        count = 5
                        continue
                    elif count == 4 and "NONB" not in line:
                        # Dihedral energy: (PK/IDIVF) × (1 + cos(PN·φ − PHASE))
                        # A4 IDIVF PK PHASE PN
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            Param(
                                atom_types=at,
                                ptype="df",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(BB[1]),
                            )
                        )

                    # Improper
                    if "NONB" in line and count == 5:
                        count = 6
                        continue
                    elif count == 5:
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            Param(
                                atom_types=at,
                                ptype="imp1",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(BB[0]),
                            )
                        )

                    # NONB
                    if count == 6:
                        continue

                    if split[0] == "vdw":
                        # The first float is the vdw radius, the second has to do
                        # with homoatomic well depths and the last is a reduction
                        # factor for univalent atoms (I don't think we will need
                        # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            Param(
                                atom_types=at,
                                ptype="vdw",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(split[2]),
                            )
                        )
        logger.log(15, f"  -- Read {len(self.params)} parameters.")
        # Apply AMBER-specific step sizes (differ from global defaults).
        for param in self.params:
            if param.ptype in self.AMBER_STEPS:
                param.step = self.AMBER_STEPS[param.ptype]

    def export_ff(
        self, path: str | None = None, params: list[Param] | None = None, lines: list[str] | None = None
    ) -> None:
        """Export the force field to a file.

        Args:
            path (str | None, optional): Output file path. Defaults to
                ``self.path``.
            params (list[Param] | None, optional): Parameters to write.
                Defaults to ``self.params``.
            lines (list[str] | None, optional): Base file lines to modify.
                Defaults to ``self.lines``.

        """
        if path is None:
            path = self.path
        if params is None:
            params: list[Param] = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, f">>> param: {param} param.value: {param.value}")
            line = lines[param.ff_row - 1]
            if abs(param.value) > 1999.0:
                logger.warning(f"Value of {param} is too high! Skipping write.")
            else:
                atoms = ""
                const = ""
                space3 = " " * 3
                col = int(param.ff_col - 1)
                value = f"{param.value:7.4f}"
                tempsplit = line.split("-")
                leng = len(tempsplit)
                AA = None
                BB = None
                if leng == 2:
                    # Bond
                    nl = 2 + 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 5
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng == 3:
                    # Angle
                    nl = 2 + 3 * 2
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 4
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng >= 4:
                    # Dihedral/Improper
                    nl = 2 + 3 * 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 2
                    value = f"{param.value:7.5f}"
                    if param.ptype == "imp1":
                        atoms += space3
                        BB[0] = value
                        const = "".join([format(el, ">12") for el in BB[:3]]) + space3 + " ".join(BB[3:])
                    else:
                        atoms += format(BB[0], ">3")
                        # Dihedral
                        BB[1] = value
                        const = "".join([format(el, ">12") for el in BB[1:4]]) + space3 + " ".join(BB[4:])

                lines[param.ff_row - 1] = atoms + const + "\n"
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, f"WROTE: {path}")

    def get_DOFs_by_atom_type(self, structs: list[Structure]) -> dict:
        """Group degrees of freedom by force field row, keyed by atom type.

        Args:
            structs (list[Structure]): Structures whose DOFs are collected.

        Returns:
            (dict): Mapping of ``ff_row`` to a list of :class:`DOF` instances
                (bonds, angles, and torsions).

        """
        dof_by_param = dict()
        for param in self.params:
            dof_by_param[param.ff_row]: list[DOF] = []
        for struct in structs:
            for bond in struct.bonds:
                dof_by_param[bond.ff_row].append(bond)
            for angle in struct.angles:
                dof_by_param[angle.ff_row].append(angle)
            for dihed in struct.torsions:
                dof_by_param[dihed.ff_row].append(dihed)
        return dof_by_param

    def get_DOFs_by_param(self, structs: list[Structure]) -> dict:
        """Group degrees of freedom by parameter.

        Delegates to :meth:`get_DOFs_by_atom_type`.

        Args:
            structs (list[Structure]): Structures whose DOFs are collected.

        Returns:
            (dict): Mapping of ``ff_row`` to a list of :class:`DOF` instances.

        """
        return self.get_DOFs_by_atom_type(structs)
