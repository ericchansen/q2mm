'''
Contains basic data structures used throughout the Q2MM code.
'''
import copy
import logging
import numpy as np
import re

import constants as cons

logger = logging.getLogger(__name__)

class Param(object):
    '''
    Class for a single parameter.
    '''
    __slots__ = ['_allow_negative', '_default_value', 'der1', 'der2', 'ptype',
                 'range', 'step' ,'value']
    def __init__(self, ptype, value):
        self._allow_negative = None
        self._default_value = None
        self.der1 = None
        self.der2 = None
        self.ptype = ptype
        self.range = None
        self.step = None 
        self.value = value
    @property
    def allow_negative(self):
        '''
        Returns True or False, depending on whether the parameter is
        allowed to be negative values.
        '''
        if self._allow_negative is None:
            if self.ptype in ['q', 'df']:
                self._allow_negative = True
            else:
                self._allow_negative = False
        return self._allow_negative
    @property
    def default_value(self):
        '''
        Default value of the parameter used when the parameter must
        be reset.
        '''
        if self._default_value is None:
            if self.ptype == 'bf':
                self._default_value = 1.
            elif self.ptype == 'af':
                self._default_value = 0.5
            else:
                raise Exception("can't set default value for unexpected parameter: {}".format(self))
        return self._default_value
    # will expand to include allowed parameter ranges.
    def check_value(self):
        '''
        Use to check if the current parameter value is allowed.
        '''
        if not self.allow_negative and self.value < 0:
            logger.warning('{} attempted to go below zero'.format(self))
            self.value = self.default_value
            return 1
        else:
            return 0
    def __repr__(self):
        return '{}({})'.format(self.ptype, self.value)
    
class ParamMM3(Param):
    '''
    Adds information to Param that is specific to MM3* parameters.
    '''
    __slots__ = ['atom_labels', 'atom_types', 'mm3_col', 'mm3_row', 'mm3_label']
    def __init__(self, atom_labels, atom_types, mm3_col, mm3_row, mm3_label, ptype, value):
        super(ParamMM3, self).__init__(ptype, value)
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.mm3_col = mm3_col
        self.mm3_row = mm3_row
        self.mm3_label = mm3_label
    def __repr__(self):
        return '{}[{},{}]({})'.format(self.ptype, self.mm3_row, self.mm3_col, self.value)

class FF(object):
    '''
    Class for any type of force field.
    '''
    def __init__(self, path=None):
        self.data = None
        self.method = None
        self.path = path
        self.params = None
        self.x2 = None

class MM3(FF):
    '''
    Class for MM3* force fields.
    '''
    def __init__(self, path=None):
        super(MM3, self).__init__(path)
        self.lines = None
        self.smiles = None
        self.sub_search = 'OPT'
        self.sub_name = None
        self.start_row = None
        self._atom_types = None
    @property
    def atom_types(self):
        '''
        Uses the SMILES-esque substructure definition (located
        directly below the substructre's name) to determine
        the atom types.
        '''
        if self._atom_types is None:
            atom_types = re.split(cons.re_split_atoms, self.smiles)
            if '' in atom_types:
                atom_types.remove('')
            self._atom_types = atom_types
        return self._atom_types
    def copy_attributes_to(self, other_ff):
        other_ff.lines = self.lines
        other_ff.path = self.path
        other_ff.smiles = self.smiles
        other_ff.sub_search = self.sub_search
        other_ff.sub_name = self.sub_name
        other_ff.start_row = self.start_row
    def select_atom_types(self, atom_labels):
        return [self._atom_types[int(x) - 1] if x.isdigit() else x for x in atom_labels]
    def import_ff(self, sub_search=None):
        if sub_search is None:
            sub_search = self.sub_search
        else:
            self.sub_search = sub_search
        self.params = []
        with open(self.path, 'r') as f:
            self.lines = f.readlines()
        with open(self.path, 'r') as f:
            logger.log(2, 'reading {}'.format(self.path))
            section_sub = False
            for i, line in enumerate(f):
                if not section_sub and sub_search in line:
                    # may need to be more robust
                    matched = re.match('\sC\s+({})'.format(cons.re_sub), line)
                    if matched != None:
                        section_sub = True
                        self.sub_name = matched.group(1).strip()
                        self.start_row = i + 1
                        logger.log(2, '{} started on line {}'.format(self.sub_name, self.start_row))
                        matched = re.match('\s9\s+({})'.format(cons.re_smiles),
                                           next(f))
                        try:
                            self.smiles = matched.group(1)
                        except:
                            logger.exception("can't read substructure name on line {}".format(i + 2))
                            raise
                        logger.log(2, 'chemical formula: {}'.format(self.smiles))
                        logger.log(2, 'atom types: {}'.format(self.atom_types))
                if section_sub and line.startswith('-3'):
                    logger.log(2, '{} ended on line {}'.format(self.sub_name, i))
                    section_sub = False
                if section_sub and line.startswith(' 1'): # stretches
                    cols = line.split()
                    self.params.extend((
                            ParamMM3(atom_labels = cols[1:3],
                                     atom_types = self.select_atom_types(cols[1:3]),
                                     ptype = 'be',
                                     mm3_col = 1,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[3])),
                            ParamMM3(atom_labels = cols[1:3],
                                     atom_types = self.select_atom_types(cols[1:3]),
                                     ptype = 'bf',
                                     mm3_col = 2,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[4])),
                            ParamMM3(atom_labels = cols[1:3],
                                     atom_types = self.select_atom_types(cols[1:3]),
                                     ptype = 'q',
                                     mm3_col = 3,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[5]))))
                if section_sub and line.startswith(' 2'): # angles
                    cols = line.split()
                    self.params.extend((
                            ParamMM3(atom_labels = cols[1:4],
                                     atom_types = self.select_atom_types(cols[1:4]),
                                     ptype = 'ae',
                                     mm3_col = 1,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[4])),
                            ParamMM3(atom_labels = cols[1:4],
                                     atom_types = self.select_atom_types(cols[1:4]),
                                     ptype = 'af',
                                     mm3_col = 2,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[5]))))
                if section_sub and line.startswith(' 3'): # stretch-bends
                    cols = line.split()
                    self.params.append(
                        ParamMM3(atom_labels = cols[1:4],
                                 atom_types = self.select_atom_types(cols[1:4]),
                                 ptype = 'sb',
                                 mm3_col = 1,
                                 mm3_row = i + 2,
                                 mm3_label = line[:2],
                                 value = float(cols[4])))
                if section_sub and line.startswith(' 4'): # torsions
                    cols = line.split()
                    self.params.extend((
                            ParamMM3(atom_labels = cols[1:5],
                                     atom_types = self.select_atom_types(cols[1:4]),
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = cols[1:5],
                                     atom_types = self.select_atom_types(cols[1:4]),
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[6])),
                            ParamMM3(atom_labels = cols[1:5],
                                     atom_types = self.select_atom_types(cols[1:4]),
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[7]))))
                if section_sub and line.startswith('54'): # higher order torsions
                    cols = line.split()
                    self.params.extend((
                            # will break if torsions aren't also looked up
                            ParamMM3(atom_labels = self.params[-1].atom_labels,
                                     atom_types = self.params[-1].atom_types,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[1])),
                            ParamMM3(atom_labels = self.params[-1].atom_labels,
                                     atom_types = self.params[-1].atom_types,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[2])),
                            ParamMM3(atom_labels = self.params[-1].atom_labels,
                                     atom_types = self.params[-1].atom_types,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[3]))))
                if section_sub and line.startswith(' 5'): # improper torsions
                    cols = line.split()
                    self.params.extend((
                            ParamMM3(atom_labels = cols[1:5],
                                     atom_types = self.select_atom_types(cols[1:5]),
                                     ptype = 'imp1',
                                     mm3_col = 1,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = cols[1:5],
                                     atom_types = self.select_atom_types(cols[1:5]),
                                     ptype = 'imp2',
                                     mm3_col = 2,
                                     mm3_row = i + 2,
                                     mm3_label = line[:2],
                                     value = float(cols[6]))))
        logger.log(2, 'read {} parameters'.format(len(self.params)))
    def export_ff(self, params=None, path=None):
        if params is None:
            params = self.params
        if path is None:
            path = self.path
        lines = copy.deepcopy(self.lines)
        modified_params = 0
        for param in params:
            cols = lines[param.mm3_row - 1].split()
            assert param.mm3_label in [' 1', ' 2', ' 3', ' 4', '54', ' 5'], \
                'unrecognized MM3* parameter label: {}'.format(param.mm3_label)
            if param.mm3_label == ' 1': # stretches
                cols[3:6] = map(float, cols[3:6])
                cols[param.mm3_col + 2] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>23.4f}{4:>11.4f}{5:>11.4f}\n'.format(*cols)
                modified_params += 1
            elif param.mm3_label == ' 2': # angles
                cols[4:6] = map(float, cols[4:6])
                cols[param.mm3_col + 3] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}{5:>11.4f}\n'.format(*cols)
                modified_params += 1
            elif param.mm3_label == ' 3': # stretch-bends
                cols[param.mm3_col + 3] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}\n'.format(*cols)
                modified_params += 1
            elif param.mm3_label == ' 4': # torsions
                cols[5:8] = map(float, cols[5:8])
                cols[param.mm3_col + 4] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}{7:>11.4f}\n'.format(*cols)
                modified_params += 1
            elif param.mm3_label == '54': # higher order torsions
                cols[1:4] = map(float, cols[1:4])
                cols[param.mm3_col] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>31.4f}{2:>11.4f}{3:>11.4f}\n'.format(*cols)
                modified_params += 1
            elif param.mm3_label == ' 5': # improper torsions
                cols[5:7] = map(float, cols[5:7])
                cols[param.mm3_col + 4] = param.value
                lines[param.mm3_row - 1] = \
                    '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}\n'.format(*cols)
                modified_params += 1
        logger.log(2, 'modified {} params'.format(modified_params))
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(5, 'wrote {}'.format(path))

class Datum(object):
    '''
    Class for a reference or calculated data point.
    '''
    __slots__ = ['_name', 'value', 'com', 'dtype', 'source', 'group', 'i', 'j', 'weight']
    def __init__(self, value, com, dtype, source, group=None, i=None, j=None, weight=None):
        self._name = None
        self.value = value
        self.com = com
        self.dtype = dtype
        self.source = source
        self.group = group
        self.i = i
        self.j = j
        self.weight = weight
    def __repr__(self):
        return ('{}({})'.format(self.dtype, self.value))
    @property
    def name(self):
        if self._name is None:
            self._name = '{}_{}_{}_{}'.format(
                self.com, name_from_source(self.source), name_from_index(self.i),
                name_from_index(self.j))
        return self._name
        
def name_from_source(source):
    '''
    Converts Datum.source into a simple, short string.
    '''
    if isinstance(source, tuple):
        source = source[0]
    return re.split('[.]+', source)[0]
def name_from_index(index):
    '''
    Converts either Datum index (i, j) into a short, simple string.
    '''
    if isinstance(index, list):
        return '-'.join(map(str, index))
    else:
        return index
def datum_sort_key(datum):
    '''
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    '''
    return (datum.dtype, datum.group, name_from_source(datum.source), datum.i, datum.j)

class Hessian(object):
    '''
    Contains methods to manipulate a cartesian Hessian matrix.
    '''
    def __init__(self):
        self.atoms = None
        self.hessian = None
        self.eigenvalues = None
        self.eigenvectors = None
    def diagonalize(self, matrix=None, eigenvectors=None, undo=False):
        if matrix is None:
            matrix = self.hessian
            assert matrix is not None, "couldn't load matrix"
        if eigenvectors is None:
            eigenvectors = self.eigenvectors
            assert eigenvectors is not None, "couldn't load eigenvectors"
        if undo:
            logger.log(3, '{} x {} x {}'.format(
                    eigenvectors.T.shape, matrix.shape, eigenvectors.shape))
            new_matrix = np.dot(np.dot(eigenvectors.T, matrix), eigenvectors)
            logger.log(3, '{} reformed matrix'.format(new_matrix.shape))
        else:
            new_matrix = np.dot(np.dot(eigenvectors, matrix), eigenvectors.T)
            logger.log(3, '{} diagonalized matrix'.format(new_matrix.shape))
        return new_matrix
    def mass_weight_eigenvectors(self, eigenvectors=None, atoms=None, undo=False):
        if eigenvectors is None:
            eigenvectors = self.eigenvectors
            assert eigenvectors is not None, "couldn't load eigenvectors"
        if atoms is None:
            atoms = self.atoms
            assert atoms is not None, "couldn't load atoms"
        masses = [cons.masses[atom.element] for atom in atoms]
        scale_factors = []
        for mass in masses:
            scale_factors.extend([np.sqrt(mass)] * 3)
        x, y = eigenvectors.shape
        for i in xrange(0, x):
            for j in xrange(0, y):
                if undo:
                    eigenvectors[i, j] /= scale_factors[j]
                else:
                    eigenvectors[i, j] *= scale_factors[j]
        return eigenvectors
    def mass_weight_hessian(self, hessian=None, atoms=None, undo=False):
        if hessian is None:
            hessian = self.hessian
            assert hessian is not None, "couldn't load hessian"
        if atoms is None:
            atoms = self.atoms
            assert atoms is not None, "couldn't load atoms"
        masses = [cons.masses[atom.element] for atom in atoms]
        scale_factors = []
        for mass in masses:
            scale_factors.extend([1 / np.sqrt(mass)] * 3)
        x, y = hessian.shape
        for i in xrange(0, x):
            for j in xrange(0, y):
                if undo:
                    hessian[i, j] = hessian[i, j] / scale_factors[i] / scale_factors[j]
                else:
                    hessian[i, j] = hessian[i, j] * scale_factors[i] * scale_factors[j]
        return hessian
    def replace_minimum(self, array, value=1):
        minimum = array.min()
        minimum_index = np.where(array == minimum)
        logger.log(3, 'minimum: {}'.format(minimum))
        if minimum >= 0:
            logger.warning("minimum isn't negative")
        # not sure why this has to be done... something i do must make write protect it
        array.setflags(write=True)
        # sometimes we use 1 and sometimes we use cons.hessian_conversion (9375.828222)
        array[minimum_index] = value
        logger.log(3, 'replaced minimum with {}'.format(value))
        return array
    def load_from_jaguar_in(self, file_class=None, file_path=None, get_hessian=True, get_atoms=True):
        assert file_class is not None or file_path is not None, \
            'must provide class object or path to file'
        if not file_class:
            file_class = JaguarIn(file_path)
        if get_hessian:
            self.hessian = file_class.hessian
        if get_atoms:
            # try:
            #     self.atoms = file_class.structures[0].atoms
            # except (TypeError, IndexError):
            #     file_class.import_structures()
            #     self.atoms = file_class.structures[0].atoms
            self.atoms = file_class.structures[0].atoms
    def load_from_jaguar_out(self, file_class=None, file_path=None,
                             get_eigenvalues=False, get_eigenvectors=True, get_atoms=False):
        assert file_class is not None or file_path is not None, \
            'must provide class object or path to file'
        if not file_class:
            file_class = JaguarOut(file_path)
        file_class.import_file()
        if get_eigenvalues:
            self.eigenvalues = file_class.eigenvalues
        if get_eigenvectors:
            self.eigenvectors = file_class.eigenvectors
        if get_atoms:
            self.atoms = file_class.structures[0].atoms
    def load_from_mmo_log(self, file_class=None, file_path=None):
        assert file_class is not None or file_path is not None, \
            'must provide class object or path to file'
        if not file_class:
            file_class = MacroModelLog(file_path)
        self.hessian = file_class.hessian
