from __future__ import annotations
import logging
from q2mm.parsers.base import FF
from q2mm.parsers.param import Param

logger = logging.getLogger(__name__)


class TinkerFF(FF):
    """
    STUFF TO FILL IN LATER
    THE PROBLEM: Depending on the forcefield used, the parameter structures are different.
    mm3.prm (exists)
    amoeba09.prm (development)
    """

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        super().__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        """
        ff.path = self.path
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def import_ff(self, path=None, sub_search="OPT"):
        if path is None:
            path = self.path
        bonds = ["bond", "bond3", "bond4", "bond5"]
        pibonds = ["pibond", "pibond3", "pibond4", "pibond5"]
        angles = ["angle", "angle3", "angle4", "angle5"]
        torsions = ["torsion", "torsion4", "torsion5"]
        dipoles = ["dipole", "dipole3", "dipole4", "dipole5"]
        self.params = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
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
                    if split[0] == "atom":
                        at = split[1]
                        el = split[2]
                        des = split[3][1:-1]
                        atnum = split[4]
                        mass = split[5]
                        # still don't know what this colum does. I don't even
                        # know if its valence
                        # Number of bonds - KJK
                        valence = split[6]
                    if split[0] in bonds:
                        at = [split[1], split[2]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="bf", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                Param(atom_types=at, ptype="be", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in dipoles:
                        at = [split[1], split[2]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="q", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                # I think this second value is the position of the
                                # dipole along the bond. I've only seen 0.5 which
                                # indicates the dipole is positioned at the center
                                # of the bond.
                                Param(atom_types=at, ptype="q_p", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in pibonds:
                        at = [split[1], split[2]]
                        # I'm still not sure how these effect the potential
                        # energy but I believe they are correcting factors for
                        # atoms in a pi system with the pi_b being for the bond
                        # and pi_t being for torsions.
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="pi_b", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                Param(atom_types=at, ptype="pi_t", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in angles:
                        at = [split[1], split[2], split[3]]
                        # TINKER param file might include several equillibrum
                        # bond angles which are for a central atom with 0, 1,
                        # or 2 additional hydrogens on the central atom.
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="af", ff_col=1, ff_row=i + 1, value=float(split[4])),
                                Param(atom_types=at, ptype="ae", ff_col=2, ff_row=i + 1, value=float(split[5])),
                            )
                        )
                        if len(split) == 8:
                            self.params.extend(
                                (
                                    Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6])),
                                    Param(atom_types=at, ptype="ae", ff_col=4, ff_row=i + 1, value=float(split[7])),
                                )
                            )
                        elif len(split) == 7:
                            self.params.append(
                                Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6]))
                            )
                    if split[0] in torsions:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="df", ff_col=1, ff_row=i + 1, value=float(split[5])),
                                Param(atom_types=at, ptype="df", ff_col=2, ff_row=i + 1, value=float(split[8])),
                                Param(atom_types=at, ptype="df", ff_col=3, ff_row=i + 1, value=float(split[11])),
                            )
                        )
                    if split[0] == "opbend":
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.append(
                            Param(atom_types=at, ptype="op_b", ff_col=1, ff_row=i + 1, value=float(split[5]))
                        )
                    if split[0] == "vdw":
                        # The first float is the vdw radius, the second has to do
                        # with homoatomic well depths and the last is a reduction
                        # factor for univalent atoms (I don't think we will need
                        # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            Param(atom_types=at, ptype="vdw", ff_col=1, ff_row=i + 1, value=float(split[2]))
                        )
        logger.log(15, f"  -- Read {len(self.params)} parameters.")

    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, f">>> param: {param} param.value: {param.value}")
            line = lines[param.ff_row - 1]
            if abs(param.value) > 999.0:
                logger.warning(f"Value of {param} is too high! Skipping write.")
            else:
                col = int(param.ff_col - 1)
                linesplit = line.split()
                value = f"{param.value:7.3f}"
                par = format(linesplit[0], "<10")
                space5 = " " * 5

                if "bond" in line:
                    atoms = "".join([format(el, ">5") for el in linesplit[1:3]]) + space5 * 2
                    linesplit[3 + col] = value
                    const = "".join([format(el, ">12") for el in linesplit[3:]])
                elif "angle" in line:
                    atoms = "".join([format(el, ">5") for el in linesplit[1:4]]) + space5
                    linesplit[4 + col] = value
                    const = "".join([format(el, ">12") for el in linesplit[4:]])
                elif "torsion" in line:
                    atoms = "".join([format(el, ">5") for el in linesplit[1:5]]) + space5
                    linesplit[5 + 3 * col] = value
                    const = "".join([format(el, ">8") for el in linesplit[5:]])
                elif "opbend" in line:
                    atoms = "".join([format(el, ">5") for el in linesplit[1:5]]) + space5
                    linesplit[5 + col] = value
                    const = "".join([format(el, ">12") for el in linesplit[5:]])
                elif "vdw" in line:
                    atoms = format(linesplit[1], ">5") + space5 * 3
                    linesplit[2 + col] = value
                    const = "".join([format(el, ">12") for el in linesplit[2:]])
                lines[param.ff_row - 1] = par + atoms + const + "\n"
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, f"WROTE: {path}")


class TinkerMM3A(FF):
    """
    STUFF TO FILL IN LATER
    """

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        super().__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        """
        ff.path = self.path
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def import_ff(self, path=None, sub_search="OPT"):
        if path is None:
            path = self.path
        bonds = ["bond", "bond3", "bond4", "bond5"]
        pibonds = ["pibond", "pibond3", "pibond4", "pibond5"]
        angles = ["angle", "angle3", "angle4", "angle5"]
        torsions = ["torsion", "torsion4", "torsion5"]
        dipoles = ["dipole", "dipole3", "dipole4", "dipole5"]
        self.params = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
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
                    if split[0] == "atom":
                        at = split[1]
                        el = split[2]
                        des = split[3][1:-1]
                        atnum = split[4]
                        mass = split[5]
                        # still don't know what this colum does. I don't even
                        # know if its valence
                        valence = split[6]
                    if split[0] in bonds:
                        at = [split[1], split[2]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="bf", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                Param(atom_types=at, ptype="be", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in dipoles:
                        at = [split[1], split[2]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="q", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                # I think this second value is the position of the
                                # dipole along the bond. I've only seen 0.5 which
                                # indicates the dipole is positioned at the center
                                # of the bond.
                                Param(atom_types=at, ptype="q_p", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in pibonds:
                        at = [split[1], split[2]]
                        # I'm still not sure how these effect the potential
                        # energy but I believe they are correcting factors for
                        # atoms in a pi system with the pi_b being for the bond
                        # and pi_t being for torsions.
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="pi_b", ff_col=1, ff_row=i + 1, value=float(split[3])),
                                Param(atom_types=at, ptype="pi_t", ff_col=2, ff_row=i + 1, value=float(split[4])),
                            )
                        )
                    if split[0] in angles:
                        at = [split[1], split[2], split[3]]
                        # TINKER param file might include several equillibrum
                        # bond angles which are for a central atom with 0, 1,
                        # or 2 additional hydrogens on the central atom.
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="af", ff_col=1, ff_row=i + 1, value=float(split[4])),
                                Param(atom_types=at, ptype="ae", ff_col=2, ff_row=i + 1, value=float(split[5])),
                            )
                        )
                        if len(split) == 8:
                            self.params.extend(
                                (
                                    Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6])),
                                    Param(atom_types=at, ptype="ae", ff_col=4, ff_row=i + 1, value=float(split[7])),
                                )
                            )
                        elif len(split) == 7:
                            self.params.append(
                                Param(atom_types=at, ptype="ae", ff_col=3, ff_row=i + 1, value=float(split[6]))
                            )
                    if split[0] in torsions:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.extend(
                            (
                                Param(atom_types=at, ptype="df", ff_col=1, ff_row=i + 1, value=float(split[5])),
                                Param(atom_types=at, ptype="df", ff_col=2, ff_row=i + 1, value=float(split[8])),
                                Param(atom_types=at, ptype="df", ff_col=3, ff_row=i + 1, value=float(split[11])),
                            )
                        )
                    if split[0] == "opbend":
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.append(
                            Param(atom_types=at, ptype="op_b", ff_col=1, ff_row=i + 1, value=float(split[5]))
                        )
                    if split[0] == "vdw":
                        # The first float is the vdw radius, the second has to do
                        # with homoatomic well depths and the last is a reduction
                        # factor for univalent atoms (I don't think we will need
                        # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            Param(atom_types=at, ptype="vdw", ff_col=1, ff_row=i + 1, value=float(split[2]))
                        )
        logger.log(15, f"  -- Read {len(self.params)} parameters.")

    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, f">>> param: {param} param.value: {param.value}")
            line = lines[param.ff_row - 1]
            if abs(param.value) > 999.0:
                logger.warning(f"Value of {param} is too high! Skipping write.")
            # Currently this isn't to flexible. The prm file (or atleast the
            # parts that are actually being paramterized have to be formatted
            # correctly. This includes the position of the columns and a space
            # at the end of every line.
            else:
                col = int(param.ff_col - 1)
                pos = 12 * (col + 1)
                linesplit = line.split()
                value = f"{param.value:7.4f}"
                par = " " * 12  # (12 * 1)
                n = len(linesplit[0])
                par[:n] = linesplit[0]
                atoms = " " * 5 * 4  # (5 * 4)
                const = " " * 4 * 12  # (4 * 12)

                if "pibond" in lines:
                    0
                # bond A B Kb b (3+(n-1))
                elif "bond" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    atoms[4 - n1 : 4] = linesplit[1]
                    atoms[8 - n2 : 8] = linesplit[2]
                    n3 = len(value)
                    const[pos - n3 : pos] = value
                #                    linesplit[3+col] = value
                # angle A B C (4+(n-1))
                elif "angle" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    n3 = len(linesplit[3])
                    atoms[4 - n1 : 4] = linesplit[1]
                    atoms[8 - n2 : 8] = linesplit[2]
                    atoms[12 - n3 : 12] = linesplit[3]
                    n4 = len(value)
                    const[pos - n4 : pos] = value
                    # linesplit[4+col] = value
                # torsion A B C D (5+3*(n-1))
                elif "torsion" in line:
                    linesplit[5 + 3 * col] = value
                # opbend A B C D (5)
                elif "opbend" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    n3 = len(linesplit[3])
                    n4 = len(linesplit[4])
                    atoms[4 - n1 : 4] = linesplit[1]
                    atoms[8 - n2 : 8] = linesplit[2]
                    atoms[12 - n3 : 12] = linesplit[3]
                    atoms[16 - n4 : 16] = linesplit[4]
                    n5 = len(value)
                    const[pos - n5 : pos] = value
                #                    linesplit[5+col] = value
                #                lines[param.ff_row - 1] = ("\t".join(linesplit)+"\n")
                lines[param.ff_row - 1] = par + atoms + const + "\n"
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, f"WROTE: {path}")
