from __future__ import annotations
import copy
import logging
import numpy as np
import os
import re
import sys
from q2mm import constants as co
from q2mm.parsers.base import FF
from q2mm.parsers.structures import Structure, DOF
from q2mm.models.param import Param, ParamMM3, COM_POS_START, P_1_START, P_1_END, P_2_START, P_2_END, P_3_START, P_3_END
from q2mm.models.datum import Datum

logger = logging.getLogger(__name__)


class MM3(FF):
    """
    Class for Schrodinger MM3* force fields (mm3.fld). TODO

    Attributes
    ----------
    smiles : list of strings
             MM3* SMILES syntax used in a custom parameter section of a
             Schrodinger MM3* force field file.
    sub_names : list of strings
                Strings used to describe each custom parameter section read.
    atom_types : list of strings
                 Atom types derived from the SMILES formula. The smiles
                 formula may have some integers, but this is strictly atom
                 types.
    lines : list of strings
            Every line from the MM3* force field file.
    """

    units = co.MM3FF

    __slots__ = ["smiles", "sub_names", "_atom_types", "_lines", "atom_type_equivalencies"]

    def __init__(self, path=None, data=None, method=None, params: list[Param] | None = None, score=None):
        super().__init__(path, data, method, params, score)
        self.smiles = []
        self.sub_names = []
        self._atom_types = None
        self._lines = None
        self.atom_type_equivalencies = dict()

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        ff : `datatypes.MM3`
        """
        ff.path = self.path
        ff.smiles = self.smiles
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def atom_types(self):
        """
        Uses the SMILES-esque substructure definition (located
        directly below the substructre's name) to determine
        the atom types.
        """
        self._atom_types = []
        for smiles in self.smiles:
            self._atom_types.append(self.convert_smiles_to_types(smiles))
        return self._atom_types

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path) as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def split_smiles(self, smiles):
        """
        Uses the MM3* SMILES substructure definition (located directly below the
        substructure's name) to determine the atom types.
        """
        split_smiles = re.split(co.RE_SPLIT_ATOMS, smiles)
        # I guess this could be an if instead of while since .remove gets rid of
        # all of them, right?
        while "" in split_smiles:
            split_smiles.remove("")
        return split_smiles

    def convert_smiles_to_types(self, smiles):
        atom_types = self.split_smiles(smiles)
        atom_types = self.convert_to_types(atom_types, atom_types)
        return atom_types

    def convert_to_types(self, atom_labels, atom_types):
        """
        Takes a list of atom_labels, which may have digits instead of atom
        types, and converts it into a list of solely atom types.

        For example,
          atom_labels = [1, 2]
          atom_types  = ["Z0", "P1", "P2"]
        would return ["Z0", "P1"].

        atom_labels - List of atom labels, which can be strings like C3, H1,
                      etc. or digits like "1" or 1.
        atom_types  - List of atom types, which are only strings like C3, H1,
                      etc.
        """
        return [atom_types[int(x) - 1] if x.strip().isdigit() and x != "00" else x for x in atom_labels]

    def import_ff(self, path=None, sub_search="OPT"):
        """
        Reads parameters from mm3.fld.
        """
        if path is None:
            path = self.path
        self.params: list[Param] = []
        self.smiles = []
        self.sub_names = []
        with open(path) as f:
            logger.log(15, f"READING: {path}")
            section_sub = False
            section_smiles = False
            section_vdw = False
            section_atm_eqv = False  # atom type equivalencies section
            for i, line in enumerate(f):
                if section_atm_eqv:
                    if line.startswith(" C") and len(self.atom_type_equivalencies.items()) > 0:
                        section_atm_eqv = False
                        continue
                    elif not line.startswith(" C") and not line.startswith("-5"):
                        equivalency = [typ.strip() for typ in line.split()[1:]]
                        for typ in equivalency[1:]:
                            self.atom_type_equivalencies[typ] = equivalency[0]
                        continue

                # These lines are for parameters.
                if not section_sub and sub_search in line and line.startswith(" C"):
                    matched = re.match(rf"\sC\s+({co.RE_SUB})\s+", line)
                    assert matched is not None, f"[L{i + 1}] Can't read substructure name: {line}"
                    if matched is not None:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15,
                            f"[L{i + 1}] Start of substructure: {sub_name}",
                        )
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match(rf"\s9\s+({co.RE_SMILES})\s", line)
                    assert matched is not None, f"[L{i + 1}] Can't read substructure SMILES: {line}"
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, f"  -- SMILES: {self.smiles[-1]}")
                    logger.log(15, "  -- Atom types: {}".format(" ".join(self.atom_types[-1])))
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith("-3"):
                    logger.log(
                        15,
                        f"[L{i}] End of substructure: {self.sub_names[-1]}",
                    )
                    section_sub = False
                    continue
                if "OPT" in line and section_vdw:
                    logger.log(
                        5,
                        "[L{}] Found Van der Waals:\n{}".format(i + 1, line.strip("\n")),
                    )
                    atm = line[2:5]
                    rad = line[5:15]
                    eps = line[16:26]
                    self.params.extend(
                        (
                            ParamMM3(
                                atom_types=atm,
                                ptype="vdwr",
                                ff_col=1,
                                ff_row=i + 1,
                                value=float(rad),
                            ),
                            ParamMM3(
                                atom_types=atm,
                                ptype="vdwe",
                                ff_col=2,
                                ff_row=i + 1,
                                value=float(eps),
                            ),
                        )
                    )
                    continue
                if "OPT" in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n")))
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            atm_typs = [line[4:6], line[9:11]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="be",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="bf",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        try:
                            self.params.append(
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="q",
                                    ff_col=3,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                )
                            )
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(5, "[L{}] Found angle:\n{}".format(i + 1, line.strip("\n")))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="ae",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="af",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                    # Stretch-bends.
                    elif match_mm3_stretch_bend(line):
                        logger.log(
                            5,
                            "[L{}] Found stretch-bend:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.append(
                            ParamMM3(
                                atom_labels=atm_lbls,
                                atom_types=atm_typs,
                                ptype="sb",
                                ff_col=1,
                                ff_row=i + 1,
                                mm3_label=line[:2],
                                value=parm_cols[0],
                            )
                        )
                        continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=3,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                ),
                            )
                        )
                        continue
                    # Higher order torsions.
                    elif match_mm3_higher_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found higher order torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        # Will break if torsions aren't also looked up.
                        atm_lbls = self.params[-1].atom_labels
                        atm_typs = self.params[-1].atom_types
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=3,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                ),
                            )
                        )
                        continue
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp1",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp2",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                    # Bonds.
                    elif match_mm3_vdw(line):
                        logger.log(5, "[L{}] Found vdw:\n{}".format(i + 1, line.strip("\n")))
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="vdwr",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="vdwfc",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith("-6"):
                    section_vdw = True
                    continue
                if "New Atom Type Equivalencies" in line:
                    section_atm_eqv = True
                    continue
        logger.log(15, f"  -- Read {len(self.params)} parameters.")

    def alternate_import_ff(self, path=None, sub_search="OPT"):
        """
        Reads parameters, but doesn't need as particular of formatting.
        TODO Clean up!!!
        """
        if path is None:
            path = self.path
        self.params: list[Param] = []
        self.smiles = []
        self.sub_names = []
        with open(path) as f:
            logger.log(15, f"READING: {path}")
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                cols = line.split()
                # These lines are for parameters.
                if not section_sub and sub_search in line and line.startswith(" C"):
                    matched = re.match(rf"\sC\s+({co.RE_SUB})\s+", line)
                    assert matched is not None, f"[L{i + 1}] Can't read substructure name: {line}"
                    if matched:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15,
                            f"[L{i + 1}] Start of substructure: {sub_name}",
                        )
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match(rf"\s9\s+({co.RE_SMILES})\s", line)
                    assert matched is not None, f"[L{i + 1}] Can't read substructure SMILES: {line}"
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, f"  -- SMILES: {self.smiles[-1]}")
                    logger.log(15, "  -- Atom types: {}".format(" ".join(self.atom_types[-1])))
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith("-3"):
                    logger.log(
                        15,
                        f"[L{i}] End of substructure: {self.sub_names[-1]}",
                    )
                    section_sub = False
                    continue
                # Not implemented.
                # if 'OPT' in line and section_vdw:
                #     logger.log(5, '[L{}] Found Van der Waals:\n{}'.format(
                #             i + 1, line.strip('\n')))
                #     atm = line[2:5]
                #     rad = line[5:15]
                #     eps = line[16:26]
                #     self.params.extend((
                #             ParamMM3(atom_types = atm,
                #                      ptype = 'vdwr',
                #                      ff_col = 1,
                #                      ff_row = i + 1,
                #                      value = float(rad)),
                #             ParamMM3(atom_types = atm,
                #                      ptype = 'vdwe',
                #                      ff_col = 2,
                #                      ff_row = i + 1,
                #                      value = float(eps))))
                #     continue
                if "OPT" in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n")))
                        if section_sub:
                            atm_lbls = [cols[1], cols[2]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        # Not really implemented.
                        else:
                            atm_typs = [cols[1], cols[2]]
                            atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="be",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[3]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="bf",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[4]),
                                ),
                            )
                        )
                        try:
                            self.params.append(
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="q",
                                    ff_col=3,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                )
                            )
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(5, "[L{}] Found angle:\n{}".format(i + 1, line.strip("\n")))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        # Not implemented.
                        else:
                            pass
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="ae",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[4]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="af",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                            )
                        )
                        continue
                    # Stretch-bends.
                    # elif match_mm3_stretch_bend(line):
                    #     logger.log(
                    #         5, '[L{}] Found stretch-bend:\n{}'.format(
                    #             i + 1, line.strip('\n')))
                    #     if section_sub:
                    #         # Do stuff.
                    #         atm_lbls = [line[4:6], line[8:10],
                    #                     line[12:14]]
                    #         atm_typs = self.convert_to_types(
                    #             atm_lbls, self.atom_types[-1])
                    #     else:
                    #         # Do other method.
                    #         atm_typs = [line[4:6], line[9:11],
                    #                     line[14:16]]
                    #         atm_lbls = atm_typs
                    #         comment = line[COM_POS_START:].strip()
                    #         self.sub_names.append(comment)
                    #     parm_cols = line[P_1_START:P_3_END]
                    #     parm_cols = map(float, parm_cols.split())
                    #     self.params.append(
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'sb',
                    #                  ff_col = 1,
                    #                  ff_row = i + 1,
                    #                  mm3_label = line[:2],
                    #                  value = parm_cols[0]))
                    #     continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            pass
                            # Do other method.
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16], line[19:21]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[6]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    ff_col=3,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[7]),
                                ),
                            )
                        )
                        continue
                    # Higher order torsions.
                    # """elif match_mm3_higher_torsion(line):
                    #     logger.log(
                    #         5, '[L{}] Found higher order torsion:\n{}'.format(
                    #             i + 1, line.strip('\n')))
                    #     # Will break if torsions aren't also looked up.
                    #     atm_lbls = self.params[-1].atom_labels
                    #     atm_typs = self.params[-1].atom_types
                    #     parm_cols = line[P_1_START:P_3_END]
                    #     parm_cols = map(float, parm_cols.split())
                    #     self.params.extend((
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  ff_col = 1,
                    #                  ff_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[0]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  ff_col = 2,
                    #                  ff_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[1]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  ff_col = 3,
                    #                  ff_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[2])))
                    #     continue"""
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(atm_lbls, self.atom_types[-1])
                        else:
                            pass
                            # Do other method.
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16], line[19:21]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp1",
                                    ff_col=1,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp2",
                                    ff_col=2,
                                    ff_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[6]),
                                ),
                            )
                        )
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith("-6"):
                    section_vdw = True
                    continue
        logger.log(15, f"  -- Read {len(self.params)} parameters.")

    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.

        Parameters
        ----------
        path : string
               File to be written or overwritten.
        params : list of `datatypes.Param` (or subclass)
        lines : list of strings
                This is what is generated when you read mm3.fld using
                readlines().
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
            # There are some problems with this. Probably an optimization
            # technique gave you these crazy parameter values. Ideally, this
            # entire trial FF should be discarded.
            # Someday export_ff should raise an exception when these values
            # get too rediculous, and this exception should be handled by the
            # optimization techniques appropriately.
            if abs(param.value) > 999.0:
                logger.warning(f"Value of {param} is too high! Skipping write.")
            elif param.ff_col == 1:
                lines[param.ff_row - 1] = line[:P_1_START] + f"{param.value:10.4f}" + line[P_1_END:]
            elif param.ff_col == 2:
                lines[param.ff_row - 1] = line[:P_2_START] + f"{param.value:10.4f}" + line[P_2_END:]
            elif param.ff_col == 3:
                lines[param.ff_row - 1] = line[:P_3_START] + f"{param.value:10.4f}" + line[P_3_END:]
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, f"WROTE: {path}")

    def get_DOFs_by_ff_row(self, structs: list[Structure]) -> dict:
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
        return self.get_DOFs_by_ff_row(structs)

    def alternate_export_ff(self, path=None, params=None):
        """
        Doesn't rely upon needing to read an mm3.fld.
        """
        lines = []
        for param in params:
            pass


def match_mm3_label(mm3_label):
    """
    Makes sure the MM3* label is recognized.

    The label is the 1st 2 characters in the line containing the parameter
    in a Schrodinger mm3.fld file.
    """
    return re.match(r"[\s5a-z][1-5]", mm3_label)


def match_mm3_vdw(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match(r"[\sa-z]6", mm3_label)


def match_mm3_bond(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match(r"[\sa-z]1", mm3_label)


def match_mm3_angle(mm3_label):
    """Matches MM3* label for angles."""
    return re.match(r"[\sa-z]2", mm3_label)


def match_mm3_stretch_bend(mm3_label):
    """Matches MM3* label for stretch-bends."""
    return re.match(r"[\sa-z]3", mm3_label)


def match_mm3_torsion(mm3_label):
    """Matches MM3* label for all orders of torsional parameters."""
    return re.match(r"[\sa-z]4|54", mm3_label)


def match_mm3_lower_torsion(mm3_label):
    """Matches MM3* label for torsions (1st through 3rd order)."""
    return re.match(r"[\sa-z]4", mm3_label)


def match_mm3_higher_torsion(mm3_label):
    """Matches MM3* label for torsions (4th through 6th order)."""
    return re.match("54", mm3_label)


def match_mm3_improper(mm3_label):
    """Matches MM3* label for improper torsions."""
    return re.match(r"[\sa-z]5", mm3_label)
