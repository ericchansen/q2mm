"""
Contains basic data structures used throughout the rest of Q2MM.
"""
from __future__ import print_function
import copy
import logging
import numpy as np
import os
import re

import constants as co
import filetypes

logger = logging.getLogger(__name__)

# Row of mm3.fld where comments start.
COM_POS_START = 96
# Row where standard 3 columns of parameters appear.
P_1_START = 23
P_1_END = 33
P_2_START = 34
P_2_END = 44
P_3_START = 45
P_3_END = 55

class ParamError(Exception):
    pass

class Param(object):
    """
    A single parameter.

    :var _allowed_range: Stored as None if not set, else it's set to True or
      False depending on :func:`allowed_range`.
   :type _allowed_range: None, 'both', 'pos', 'neg'

    :ivar ptype: Parameter type can be one of the following: ae, af, be, bf, df,
      imp1, imp2, sb, or q.
    :type ptype: string

    Attributes
    ----------
    d1 : float
         First derivative of parameter with respect to penalty function.
    d2 : float
         Second derivative of parameter with respect to penalty function.
    step : float
           Step size used during numerical differentiation.
    ptype : {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}
    value : float
            Value of the parameter.
    """
    __slots__ = ['_allowed_range', 'd1', 'd2', '_step', 'ptype', '_value']
    def __init__(self, d1=None, d2=None, ptype=None, value=None):
        self._allowed_range = None
        self.d1 = d1
        self.d2 = d2
        self._step = None 
        self.ptype = ptype
        self._value = None
        self.value = value
    def __repr__(self):
        return '{}[{}]({:7.4f})'.format(
            self.__class__.__name__, self.ptype, self.value)
    @property
    def allowed_range(self):
        """
        Returns True or False, depending on whether the parameter is
        allowed to be negative values.
        """
        if self._allowed_range is None and self.ptype is not None:
            if self.ptype in ['q', 'df']:
                self._allowed_range = [-float('inf'), float('inf')]
            else:
                self._allowed_range = [0., float('inf')]
        return self._allowed_range
    @property
    def step(self):
        """
        Returns a float for the current step size that should be used. If
        _step is a string, return float(_step) * value. If
        _step is a float, simply return that.

        Not sure how well the check for a step size of zero works.
        """
        if self._step is None:
            try:
                self._step = co.STEPS[self.ptype]
            except KeyError:
                logger.warning(
                    "{} doesn't have a default step size and none "
                    "provided!".format(self))
                raise
        if self._step == 0.:
            self._step = 0.1
        if isinstance(self._step, basestring):
            return float(self._step) * self.value
        else:
            return self._step
    @step.setter
    def step(self, x):
        self._step = x
    @property
    def value(self):
        if self.ptype == 'ae' and self._value > 180.:
            self._value = 180. - abs(180 - self._value)
        return self._value
    @value.setter
    def value(self, value):
        """
        When you try to give the parameter a value, make sure that's okay.
        """
        if self.allowed_range[0] <= value <= self.allowed_range[1]:
            self._value = value
        else:
            raise ParamError(
                "{} isn't allowed to have a value of {}! "
                "({} <= x <= {})".format(
                    str(self), value, self.allowed_range[0], self.allowed_range[1]))
    
# Need a general index scheme/method/property to compare the equalness of two
# parameters, rather than having to rely on some expression that compares
# mm3_row and mm3_col.
class ParamMM3(Param):
    '''
    Adds information to Param that is specific to MM3* parameters.
    '''
    __slots__ = ['atom_labels', 'atom_types', 'mm3_col', 'mm3_row', 'mm3_label']
    def __init__(self, atom_labels=None, atom_types=None, mm3_col=None,
                 mm3_row=None, mm3_label=None,
                 d1=None, d2=None, ptype=None, value=None):
        super(ParamMM3, self).__init__(ptype=ptype, value=value)
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.mm3_col = mm3_col
        self.mm3_row = mm3_row
        self.mm3_label = mm3_label
    def __repr__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col,
            self.value)
    def __str__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col,
            self.value)

class Datum(object):
    '''
    Class for a reference or calculated data point.
    '''
    __slots__ = ['_lbl', 'val', 'wht', 'typ', 'com', 'src_1', 'src_2', 'idx_1',
                 'idx_2', 'atm_1', 'atm_2', 'atm_3', 'atm_4']
    def __init__(self, lbl=None, val=None, wht=None, typ=None, com=None,
                 src_1=None, src_2=None,
                 idx_1=None, idx_2=None,
                 atm_1=None, atm_2=None, atm_3=None, atm_4=None):
        self._lbl  = lbl
        self.val   = val
        self.wht   = wht
        self.typ   = typ
        self.com   = com
        self.src_1 = src_1
        self.src_2 = src_2
        self.idx_1 = idx_1
        self.idx_2 = idx_2
        self.atm_1 = atm_1
        self.atm_2 = atm_2
        self.atm_3 = atm_3
        self.atm_4 = atm_4
    def __repr__(self):
        return '{}({:7.4f})'.format(self.lbl, self.val)
    @property
    def lbl(self):
        if self._lbl is None:
            a = self.typ
            if self.src_1:
                b = re.split('[.]+', self.src_1)[0]
            else:
                b = None
            c = '-'.join(map(str, remove_none(
                        self.idx_1, self.idx_2)))
            d = '-'.join(map(str, remove_none(
                        self.atm_1, self.atm_2, self.atm_3, self.atm_4)))
            abcd = remove_none(a, b, c, d)
            return '_'.join(abcd)
        
def remove_none(*args):
    return [x for x in args if x is not None]

def datum_sort_key(datum):
    '''
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    '''
    return (datum.dtype, datum.group, lbl_from_source(datum.source), datum.i, datum.j)

class FF(object):
    """
    Class for any type of force field.
    
    path   - Self explanatory.
    conn   - Connection to a database object. Would contain data obtained
             using this FF.
    method - String describing method used to generate this FF.
    params - List of Param objects.
    score  - Float which is the objective function score.
    """
    def __init__(self, path=None, conn=None, method=None, params=None,
                 score=None):
        self.path = path
        self.conn = conn
        self.method = method
        self.params = params
        self.score = score
    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        ff : `datatypes.FF`
        """
        ff.path = self.path
    def __repr__(self):
        return '{}[{}]({})'.format(
            self.__class__.__name__, self.method, self.score)

class MM3(FF):
    """
    Class for Schrodinger MM3* force fields (mm3.fld).

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
    def __init__(self, path=None, conn=None, method=None, params=None,
                 score=None):
        super(MM3, self).__init__(path, conn, method, params, score)
        self.smiles = []
        self.sub_names = []
        # self.smiles = None
        # self.sub_names = None
        self._atom_types = None
        self._lines = None
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
            # self._atom_types.append(self.split_smiles(smiles))
        return self._atom_types
        # if self._atom_types is None:
        #     atom_types = re.split(co.RE_SPLIT_ATOMS, self.smiles)
        #     if '' in atom_types:
        #         atom_types.remove('')
        #     self._atom_types = atom_types
        # return self._atom_types
    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, 'r') as f:
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
        while '' in split_smiles:
            split_smiles.remove('')
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
        return [atom_types[int(x) - 1] if x.strip().isdigit() and 
                x != '00' 
                else x
                for x in atom_labels]
    def import_ff(self, path=None, sub_search='OPT'):
        """
        Reads parameters from mm3.fld.
        """
        if path is None:
            path = self.path
        self.params = []
        self.smiles = []
        self.sub_names = []
        with open(path, 'r') as f:
            logger.log(15, 'READING: {}'.format(path))
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                # These lines are for parameters.
                if not section_sub and sub_search in line \
                        and line.startswith(' C'):
                    matched = re.match('\sC\s+({})\s+'.format(
                            co.RE_SUB), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure name: {}".format(
                        i + 1, line)
                    if matched != None:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15, '[L{}] Start of substructure: {}'.format(
                                i+1, sub_name))
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match(
                        '\s9\s+({})\s'.format(co.RE_SMILES), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure SMILES: {}".format(
                        i + 1, line)
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, '  -- SMILES: {}'.format(
                            self.smiles[-1]))
                    logger.log(15, '  -- Atom types: {}'.format(
                            ' '.join(self.atom_types[-1])))
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith('-3'):
                    logger.log(15, '[L{}] End of substructure: {}'.format(
                            i, self.sub_names[-1]))
                    section_sub = False
                    continue
                if 'OPT' in line and section_vdw:
                    logger.log(15, '[L{}] Found van der Waals:\n{}'.format(
                            i + 1, line.strip('\n')))
                    atm = line[2:5]
                    rad = line[5:15]
                    eps = line[16:26]
                    self.params.extend((
                            ParamMM3(atom_types = atm,
                                     ptype = 'vdwr',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     value = float(rad)),
                            ParamMM3(atom_types = atm,
                                     ptype = 'vdwe',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     value = float(eps))))
                    continue
                if 'OPT' in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(
                            5, '[L{}] Found bond:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            atm_typs = [line[4:6], line[9:11]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.extend((
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'be',
                                         mm3_col = 1,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[0]),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'bf',
                                         mm3_col = 2,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[1])))
                        try:
                            self.params.append(
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'q',
                                         mm3_col = 3,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[2]))
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(
                            5, '[L{}] Found angle:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'ae',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'af',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1])))
                        continue
                    # Stretch-bends.
                    elif match_mm3_stretch_bend(line):
                        logger.log(
                            5, '[L{}] Found stretch-bend:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.append(
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'sb',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]))
                        continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[2])))
                        continue
                    # Higher order torsions.
                    elif match_mm3_higher_torsion(line):
                        logger.log(
                            5, '[L{}] Found higher order torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        # Will break if torsions aren't also looked up.
                        atm_lbls = self.params[-1].atom_labels
                        atm_typs = self.params[-1].atom_types
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[2])))
                        continue
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = map(float, parm_cols.split())
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp1',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp2',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1])))
                        continue
                # The van der Waals are stored in annoying way.
                if line.startswith('-6'):
                    section_vdw = True
                    continue
        logger.log(15, '  -- Read {} parameters.'.format(len(self.params)))
    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.

        Parameters
        ----------
        path : string
               File to be written or overwritten.
        params : list of `datatypes.Param` (or subclass)
        """
        if path is None:
            path = self.path
        if params is None:
            params = self.params
        if lines is None and self.lines is None:
            with open(path, 'r') as f:
                lines = f.readlines()
            logger.log(10, '  -- Read {} lines from {}.'.format(
                    len(lines), path))
        else:
            lines = self.lines
        for param in params:
            line = lines[param.mm3_row - 1]
            if param.mm3_col == 1:
                lines[param.mm3_row - 1] = (line[:P_1_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_1_END:])
                # line[P_1_END:] +
                # '\n')
            elif param.mm3_col == 2:
                lines[param.mm3_row - 1] = (line[:P_2_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_2_END:])
                # line[P_2_END:] +
                # '\n')
            elif param.mm3_col == 3:
                lines[param.mm3_row - 1] = (line[:P_3_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_3_END:])
                # line[P_3_END:] +
                # '\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(10, 'WROTE: {}'.format(path))


def match_mm3_label(mm3_label):
    """
    Makes sure the MM3* label is recognized.

    The label is the 1st 2 characters in the line containing the parameter
    in a Schrodinger mm3.fld file.
    """
    return re.match('[\s5a-z][1-5]', mm3_label)
def match_mm3_bond(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match('[\sa-z]1', mm3_label)
def match_mm3_angle(mm3_label):
    """Matches MM3* label for angles."""
    return re.match('[\sa-z]2', mm3_label)
def match_mm3_stretch_bend(mm3_label):
    """Matches MM3* label for stretch-bends."""
    return re.match('[\sa-z]3', mm3_label)
def match_mm3_torsion(mm3_label):
    """Matches MM3* label for all orders of torsional parameters."""
    return re.match('[\sa-z]4|54', mm3_label)
def match_mm3_lower_torsion(mm3_label):
    """Matches MM3* label for torsions (1st through 3rd order)."""
    return re.match('[\sa-z]4', mm3_label)
def match_mm3_higher_torsion(mm3_label):
    """Matches MM3* label for torsions (4th through 6th order)."""
    return re.match('54', mm3_label)
def match_mm3_improper(mm3_label):
    """Matches MM3* label for improper torsions."""
    return re.match('[\sa-z]5', mm3_label)

# This should probably be a part of the MM3* class for better
# encapsulation.
def export_ff(path, params, lines=None):
    """
    Exports the force field to a file, typically mm3.fld.

    Parameters
    ----------
    path : string
           File to be written or overwritten.
    params : list of `datatypes.Param` (or subclass)
    """
    assert os.path.splitext(path)[-1] == '.fld', \
        "Can't recognize FF: {}".format(path)
    if lines is None:
        with open(path, 'r') as f:
            lines = f.readlines()
        logger.log(10, '  -- Read {} lines from {}.'.format(
                len(lines), path))
    modified_params = 0
    for param in params:
        cols = lines[param.mm3_row - 1].split()
        if match_mm3_bond(param.mm3_label):
            cols[3:6] = map(float, cols[3:6])
            cols[param.mm3_col + 2] = param.value
            if len(cols) == 6:
                lines[param.mm3_row - 1] = (
                    '{0:>2}{1:>4}{2:>4}{3:>23.4f}{4:>11.4f}'
                    '{5:>11.4f}\n'.format(*cols))
            elif len(cols) == 5:
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>23.4f}{4:>11.4f}\n'.format(*cols)
            modified_params += 1
        elif match_mm3_angle(param.mm3_label):
            cols[4:6] = map(float, cols[4:6])
            cols[param.mm3_col + 3] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}{5:>11.4f}\n'.format(*cols)
            modified_params += 1
        elif match_mm3_stretch_bend(param.mm3_label):
            cols[param.mm3_col + 3] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}\n'.format(*cols)
            modified_params += 1
        elif match_mm3_torsion(param.mm3_label):
            cols[5:8] = map(float, cols[5:8])
            cols[param.mm3_col + 4] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}{7:>11.4f}\n'.format(*cols)
            modified_params += 1
        elif match_mm3_higher_torsion(param.mm3_label):
            cols[1:4] = map(float, cols[1:4])
            cols[param.mm3_col] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>31.4f}{2:>11.4f}{3:>11.4f}\n'.format(*cols)
            modified_params += 1
        elif match_mm3_improper(param.mm3_label):
            cols[5:7] = map(float, cols[5:7])
            cols[param.mm3_col + 4] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}\n'.format(*cols)
            modified_params += 1
        else:
            raise Exception('Unrecognized MM3* parameter label: "{}"'.format(
                    param.mm3_label))
    logger.log(10, '  -- Modified {} parameters.'.format(modified_params))
    with open(path, 'w') as f:
        f.writelines(lines)
    logger.log(10, 'WROTE: {}'.format(path))

def import_ff(path, sub_search='OPT'):
    """
    Reads parameters from mm3.fld.
    """
    # path = os.path.abs(path)
    # directory = os.path.dirname(path)
    # filename = os.path.basename(path)
    # assert os.path.splitext(filename)[-1] == '.fld', \
    assert os.path.splitext(path)[-1] == '.fld', \
        "Can't recognize FF: {}".format(path)
    ff = MM3(path)
    ff.params = []
    with open(ff.path, 'r') as f:
        logger.log(15, 'READING: {}'.format(ff.path))
        section_sub = False
        section_smiles = False
        for i, line in enumerate(f):
            # Search for the string you provided, sub_search, to find the start
            # of an MM3* subsection.
            if not section_sub and sub_search in line:
                matched = re.match('\sC\s+({})\s+'.format(co.RE_SUB), line)
                assert matched is not None, \
                    "[L{}] Can't read substructure name: {}".format(i + 1, line)
                if matched != None:
                    # Oh good, you found your substructure!
                    section_sub = True
                    sub_name = matched.group(1).strip()
                    ff.sub_names.append(sub_name)
                    # if ff.sub_names is None:
                    #     ff.sub_names = [sub_name]
                    # else:
                    #     ff.sub_names.append(sub_name)
                    logger.log(15, '[L{}] Start of substructure: {}'.format(
                            i+1, sub_name))
                    section_smiles = True
                    continue
            elif section_smiles is True:
                matched = re.match('\s9\s+({})\s'.format(co.RE_SMILES), line)
                assert matched is not None, \
                    "[L{}] Can't read substructure SMILES: {}".format(i + 1, line)
                smiles = matched.group(1)
                ff.smiles.append(smiles)
                # if ff.smiles is None:
                #     ff.smiles = [smiles]
                # else:
                #     ff.smiles.append(smiles)
                logger.log(15, '  -- SMILES: {}'.format(
                        ff.smiles[-1]))
                logger.log(15, '  -- Atom types: {}'.format(
                        ' '.join(ff.atom_types[-1])))
                section_smiles = False
                continue
            # Marks the end of a substructure.
            elif section_sub and line.startswith('-3'):
                logger.log(15, '[L{}] End of substructure: {}'.format(
                        i, ff.sub_names[-1]))
                section_sub = False
                continue
            # Look at bonds.
            elif section_sub and re.match('^[a-z\s]1', line):
                cols = line.split()
                atm_lbls = cols[1:3]
                atm_typs = ff.convert_to_types(atm_lbls, ff.atom_types[-1])
                ff.params.extend((
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'be',
                                 mm3_col = 1,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[3])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'bf',
                                 mm3_col = 2,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[4]))))
                try:
                    ff.params.append(
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'q',
                                 mm3_col = 3,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[5])))
                # Some bonds parameters don't use bond dipoles.
                except IndexError:
                    pass
                continue
            # Angles.
            elif section_sub and re.match('^[a-z\s]2', line):
                cols = line.split()
                atm_lbls = cols[1:4]
                atm_typs = ff.convert_to_types(atm_lbls, ff.atom_types[-1])
                ff.params.extend((
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'ae',
                                 mm3_col = 1,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[4])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'af',
                                 mm3_col = 2,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[5]))))
                continue
            # Stretch-bends.
            elif section_sub and re.match('^[a-z\s]3', line):
                cols = line.split()
                atm_lbls = cols[1:4]
                atm_typs = ff.convert_to_types(atm_lbls, ff.atom_types[-1])
                ff.params.append(
                    ParamMM3(atom_labels = atm_lbls,
                             atom_types = atm_typs,
                             ptype = 'sb',
                             mm3_col = 1,
                             mm3_row = i + 1,
                             mm3_label = line[:2],
                             value = float(cols[4])))
                continue
            # Torsions.
            elif section_sub and re.match('^[a-z\s]4', line):
                cols = line.split()
                atm_lbls = cols[1:5]
                atm_typs = ff.convert_to_types(atm_lbls, ff.atom_types[-1])
                ff.params.extend((
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 1,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[5])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 2,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[6])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 3,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[7]))))
                continue
            # Higher order torsions.
            elif section_sub and line.startswith('54'):
                cols = line.split()
                # Will break if the torsion isn't also looked up.
                # Should never happen?
                atm_lbls = ff.params[-1].atom_labels
                atm_typs = ff.params[-1].atom_types
                ff.params.extend((
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 1,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[1])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 2,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[2])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'df',
                                 mm3_col = 3,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[3]))))
                continue
            # Improper torsions.
            elif section_sub and re.match('^[a-z\s]5', line):
                cols = line.split()
                atm_lbls = cols[1:5]
                atm_typs = ff.convert_to_types(atm_lbls, ff.atom_types[-1])
                ff.params.extend((
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'imp1',
                                 mm3_col = 1,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[5])),
                        ParamMM3(atom_labels = atm_lbls,
                                 atom_types = atm_typs,
                                 ptype = 'imp2',
                                 mm3_col = 2,
                                 mm3_row = i + 1,
                                 mm3_label = line[:2],
                                 value = float(cols[6]))))
                continue
    logger.log(15, '  -- Read {} parameters in {} substructure(s).'.format(
            len(ff.params), len(ff.sub_names)))
    return ff

class Hessian(object):
    """
    Contains methods to manipulate a certesian Hessian matrix.
    """
    def __init__(self, *args):
        self.atoms = None
        self.evecs = None
        self.evals = None
        self.hess = None
        for source in args:
            self.import_source(source)
    def diagonalize(self, reverse=False):
        """
        Diagonalizes self.hess using self.evecs. If reverse is True,
        it un-diagonalizes self.hess.
        """
        if reverse:
            self.hess = np.dot(np.dot(
                    self.evecs.T, self.hess), self.evecs)
        else:
            self.hess = np.dot(np.dot(
                    self.evecs, self.hess), self.evecs.T)
    def mass_weight_eigenvectors(self, reverse=False):
        """
        Mass weights self.evecs. If reverse is True, it un-mass weights
        the eigenvectors.
        """
        # masses = [co.MASSES[x.element] for x in self.atoms]
        # changes = []
        # for mass in masses:
        #     changes.extend([np.sqrt(mass)] * 3)
        changes = []
        for atom in self.atoms:
            if not atom.is_dummy:
                changes.extend([np.sqrt(atom.exact_mass)] * 3)
        x, y = self.evecs.shape
        for i in xrange(0, x):
            for j in xrange(0, y):
                if reverse:
                    self.evecs[i, j] /= changes[j]
                else:
                    self.evecs[i, j] *= changes[j]
    def mass_weight_hessian(self, reverse=False):
        """
        Mass weights self.hess. If reverse is True, it un-mass weights
        the Hessian.
        """
        masses = [co.MASSES[x.element] for x in self.atoms if not x.is_dummy]
        changes = []
        for mass in masses:
            changes.extend([1 / np.sqrt(mass)] * 3)
        x, y = self.hess.shape
        for i in xrange(0, x):
            for j in xrange(0, y):
                if reverse:
                    self.hess[i, j] = \
                        self.hess[i, j] / changes[i] / changes[j]
                else:
                    self.hess[i, j] = \
                        self.hess[i, j] * changes[i] * changes[j]
    def replace_minimum(self, array, value=1):
        """
        Replace the minimum vallue in an arbitrary NumPy array. Typically
        the replace value is either 1 or co.HESSIAN_CONVERSION.
        """
        minimum = array.min()
        minimum_index = np.where(array == minimum)
        assert minimum < 0, 'Minimum of array is not negative!'
        # It would be better to address this in a different way. This particular
        # data structure just isn't what we want.
        array.setflags(write=True)
        # Sometimes we use 1, but sometimes we use co.HESSIAN_CONVERSION.
        array[minimum_index] = value
        logger.log(10, '  -- Replaced minimum in array with {}.'.format(value))
    def import_source(self, source):
        """
        source - String for a filename or a filetype object.
        """
        # if isinstance(source, basestring):
        #     ext = os.path.splitext(source)[1]
        #     if ext == '.in':
        #         source = filetypes.JaguarIn(source)
        #     elif ext == '.out':
        #         source = filetypes.JaguarOut(source)
        #     elif ext == '.log':
        #         source = filetypes.MacroModelLog(source)
        assert isinstance(source, filetypes.JaguarIn) or \
            isinstance(source, filetypes.JaguarOut) or \
            isinstance(source, filetypes.MacroModelLog), \
            'Must provide an instance of a class that has Hessian data!'
        if hasattr(source, 'hessian') and source.hessian is not None:
            self.hess = source.hessian
            logger.log(10, '  -- Loaded {} Hessian from {}.'.format(
                    self.hess.shape, source.path))
        if hasattr(source, 'eigenvectors') and source.eigenvectors is not None:
            self.evecs = source.eigenvectors
            logger.log(10, '  -- Loaded {} eigenvectors from {}.'.format(
                    self.evecs.shape, source.path))
        if hasattr(source, 'structures') and source.structures is not None:
            self.atoms = source.structures[0].atoms
            logger.log(10, '  -- Loaded {} atoms from {}.'.format(
                    len(self.atoms), source.path))

def check_mm_dummy(hess, dummy_indices):
    hess = np.delete(hess, dummy_indices, 0)
    hess = np.delete(hess, dummy_indices, 1)
    logger.log(15, 'Created {} Hessian w/o dummy atoms.'.format(hess.shape))
    return hess
