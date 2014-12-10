'''
Contains classes for the files that Q2MM will interact with, with the
exception being force field classes, which are in data.py.
'''
from string import digits
import logging
import numpy as np
import re

from schrodinger import structure as schrod_structure
from schrodinger.application.jaguar import input as schrod_jaguar_in
import constants as cons

logger = logging.getLogger(__name__)

class File(object):
    '''
    Base class for all filetypes.
    '''
    def __init__(self, path):
        self.path = path
        
class GaussLog(File):
    '''
    Class used to retrieve data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    '''
    def __init__(self, path):
        super(GaussLog, self).__init__(path)
        self._structures = None
    @property
    def structures(self):
        if self._structures is None:
            self._structures = self.import_optimization(coords_type='both')
        return self._structures
    def get_most_converged(self, structures=None):
        '''
        Used with geometry optimizations that don't succeed. Sometimes
        intermediate geometries obtain better convergence than the
        final geometry. This function returns the class Structure for
        the most converged geometry, which can then be used to output
        the coordinates for the next optimization.
        '''
        if structures is None:
            structures = self.structures
        structures_compared = 0
        best_structure = None
        best_yes_or_no = None
        fields = ['RMS Force', 'RMS Displacement', 'Maximum Force', 'Maximum Displacement']
        for i, structure in reversed(list(enumerate(structures))):
            yes_or_no = [value[2] for key, value in structure.props.items() if key in fields]
            if not structure.atoms:
                logger.warning('no atoms found in structure {}. skipping'.format(i+1))
                continue
            if len(yes_or_no) == 4:
                structures_compared += 1
                if best_structure is None:
                    logger.log(3, 'most converged structure: {}'.format(i+1))
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') > best_yes_or_no.count('YES'):
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') == best_yes_or_no.count('YES'):
                    number_better = 0
                    for field in fields:
                        if structure.props[field][0] < best_structure.props[field][0]:
                            number_better += 1
                    if number_better > 2:
                        best_structure = structure
                        best_yes_or_no = yes_or_no
            elif len(yes_or_no) != 0:
                logger.warning('partial convergence criterion in structure from {}'.format(
                        self.path))
        logger.log(3, 'compared {} out of {} structures'.format(structures_compared, len(self.structures)))
        return best_structure
    def import_optimization(self, coords_type='both'):
        '''
        Finds structures from a Gaussian geometry optimization that
        are listed throughout the log file. Also finds data about
        their convergence.

        coords_type = "input" or "standard" or "both"
                      Using both may cause coordinates in one format
                      to be overwritten by whatever comes later in the
                      log file.
        '''
        logger.log(2, 'reading {}'.format(self.path))
        structures = []
        with open(self.path, 'r') as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):

                if section_optimization and 'Optimization stopped.' in line:
                    section_optimization = False
                    logger.log(2, '{} end optimization section'.format(i+1))
                if not section_optimization and \
                        'Search for a local minimum.' in line:
                    section_optimization = True
                    logger.log(2, '{} start optimization section'.format(i+1))
                    
                if section_optimization:

                    if 'Step number' in line:
                        structures.append(Structure())
                        current_structure = structures[-1]
                        logger.log(2, '{} added structure (currently {})'.format(
                                i+1, len(structures)))

                    if section_convergence and 'GradGradGrad' in line:
                        section_convergence = False
                        logger.log(2, '{} end convergence section'.format(i+1))
                    if section_convergence:
                        match = re.match(
                            '\s(Maximum|RMS)\s+(Force|Displacement)\s+({0})\s+({0})\s+(YES|NO)'.format(
                                cons.re_float), line)
                        if match:
                            current_structure.props['{} {}'.format(match.group(1), match.group(2))] = \
                                (float(match.group(3)), float(match.group(4)), match.group(5))
                    if 'Converged?' in line:
                        section_convergence = True
                        logger.log(2, '{} start convergence section'.format(i+1))
            
                    if coords_type == 'input' or coords_type == 'both':
                        if section_coords_input and 'Distance matrix' in line:
                            section_coords_input = False
                            logger.log(2, '{} end input coordinates section ({} atoms)'.format(
                                    i+1, count_atom))
                        if section_coords_input:
                            match = re.match(
                                '\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+({0})'.format(
                                    cons.re_float), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(match.group(2)), \
                                        ("{} atomic numbers don't match (current != existing) ({} != {})".format(
                                            i+1, int(match.group(2)), current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.coords_type = 'input'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        if not section_coords_input and \
                                'Input orientation:' in line:
                            section_coords_input = True
                            count_atom = 0
                            logger.log(2, '{} start input coordinates section'.format(i+1))

                    if coords_type == 'standard' or coords_type == 'both':
                        if section_coords_standard and \
                                ('Rotational constants' in line or
                                 'Leave Link' in line):
                            section_coords_standard = False
                            logger.log(2, '{} end standard coordinates section ({} atoms)'.format(
                                    i+1, count_atom))
                        if section_coords_standard:
                            match = re.match('\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+({0})'.format(
                                    cons.re_float), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num: 
                                    assert current_atom.atomic_num == int(match.group(2)), \
                                        ("{} atomic numbers don't match (current != existing) ({} != {})".format(
                                            i+1, int(match.group(2)), current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.coords_type = 'standard'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        if not section_coords_standard and \
                                'Standard orientation' in line:
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(2, '{} start standard coordinates section.'.format(i+1))
        return structures
                            
class SchrodingerFile(File):
    '''
    Parent class used for all Schrodinger files.
    '''
    def convert_schrodinger_structure(self, sch_struct):
        '''
        Converts a schrodinger.structure object to my own structure object.
        '''
        my_struct = Structure()
        my_struct.props.update(sch_struct.property)
        for sch_atom in sch_struct.atom:
            my_atom = Atom()
            my_struct.atoms.append(my_atom)
            my_atom.atom_type = sch_atom.atom_type
            my_atom.atom_type_name = sch_atom.atom_type_name
            my_atom.atomic_num = sch_atom.atomic_number
            my_atom.bonded_atom_indices = [x.index for x in sch_atom.bonded_atoms]
            my_atom.element = sch_atom.element
            my_atom.index = sch_atom.index
            my_atom.partial_charge = sch_atom.partial_charge
            my_atom.x, my_atom.y, my_atom.z = sch_atom.x, sch_atom.y, sch_atom.z
        for sch_bond in sch_struct.bond:
            my_bond = Bond()
            my_struct.bonds.append(my_bond)
            my_bond.atom_nums = [sch_bond.atom1, sch_bond.atom2]
            my_bond.order = sch_bond.order
            my_bond.value = sch_bond.length
        return my_struct
    
class JaguarIn(SchrodingerFile):
    '''
    Used to retrieve data from Jaguar in files.
    '''
    def __init__(self, path):
        super(JaguarIn, self).__init__(path)
        self._structures = None
        self._hessian = None
        self._sch_ob = None
        self._sch_struct = None
    @property
    def hessian(self):
        if self._hessian is None:
            num_atoms = len(self.structures[0].atoms)
            assert num_atoms != 0, 'zero atoms while loading hessian from {}'.format(self.path)
            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(2, '{} hessian matrix'.format(hessian.shape))
            with open(self.path, 'r') as f:
                section_hess = False
                for line in f:
                    if section_hess and line.startswith('&'):
                        section_hess = False
                        hessian += np.tril(hessian, -1).T
                    if section_hess:
                        cols = line.split()
                        if len(cols) == 1:
                            hess_col = int(cols[0])
                        elif len(cols) > 1:
                            hess_row = int(cols[0])
                            for i, hess_ele in enumerate(cols[1:]):
                                hessian[hess_row - 1, i + hess_col - 1] = float(hess_ele)
                    if '&hess' in line:
                        section_hess = True
            self._hessian = hessian * cons.hessian_conversion
        return self._hessian
    @property
    def structures(self):
        if self._structures is None:
            logger.log(2, 'reading {}'.format(self.path))
            self._sch_ob = schrod_jaguar_in.read(self.path)
            self._sch_struct = self._sch_ob.getStructure()
            structures = [self.convert_schrodinger_structure(self._sch_struct)]
            logger.log(2, 'imported {} structures'.format(len(structures)))
            # this area is sketch. i added it so i could use hessian data
            # generated from a jaguar calculation that had a dummy atom.
            # no gaurantees this will always work.
            for i, structure in enumerate(structures): 
                empty_atoms = []
                for atom in structure.atoms:
                    if atom.element == '':
                        empty_atoms.append(atom)
                for atom in empty_atoms:
                    structure.atoms.remove(atom)
                if empty_atoms:
                    logger.log(2, 'structure {}: {} empty atoms removed'.format(i + 1, len(empty_atoms)))
            self._structures = structures
        return self._structures

class JaguarOut(File):
    '''
    Used to retrieve data from Schrodinger Jaguar out files.
    '''
    def __init__(self, path):
        super(JaguarOut, self).__init__(path)
        self.structures = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.frequencies = None
        # self.force_constants = None
    def import_file(self):
        logger.log(2, 'reading {}'.format(self.path))
        frequencies = []
        force_constants = []
        eigenvectors = []
        structures = []
        with open(self.path, 'r') as f:
            section_geometry = False
            section_eigenvalues = False
            section_eigenvectors = False
            for line in f:

                if section_geometry:
                    cols = line.split()
                    if len(cols) == 0:
                        section_geometry = False
                        structures.append(current_structure)
                        continue
                    elif len(cols) == 1:
                        pass
                    else:
                        match = re.match('\s+([\d\w]+)\s+({0})\s+({0})\s+({0})'.format(cons.re_float), line)
                        if match != None:
                            current_atom = Atom()
                            current_atom.element = match.group(1).translate(None, digits)
                            current_atom.x = float(match.group(2))
                            current_atom.y = float(match.group(3))
                            current_atom.z = float(match.group(4))
                            current_structure.atoms.append(current_atom)
                            logger.log(2, '{0:<3}{1:>12.6f}{2:>12.6f}{3:>12.6f}'.format(
                                    current_atom.element, current_atom.x, current_atom.y, current_atom.z))
                if 'geometry:' in line:
                    section_geometry = True
                    current_structure = Structure()
                    logger.log(2, 'located geometry')

                if 'Number of imaginary frequencies' in line or \
                        'Writing vibrational' in line or \
                        'Thermochemical properties at' in line:
                    section_eigenvalues = False
                if section_eigenvectors is True:
                    cols = line.split()
                    if len(cols) == 0:
                        section_eigenvectors = False
                        eigenvectors.extend(temp_eigenvectors)
                        continue
                    else:
                        for i, x in enumerate(cols[2:]):
                            if not len(temp_eigenvectors) > i:
                                temp_eigenvectors.append([])
                            temp_eigenvectors[i].append(float(x))
                if section_eigenvalues is True and section_eigenvectors is False:
                    if 'frequencies' in line:
                        cols = line.split()
                        frequencies.extend(map(float, cols[1:]))
                    if 'force const' in line:
                        cols = line.split()
                        force_constants.extend(map(float, cols[2:]))
                        section_eigenvectors = True
                        temp_eigenvectors = [[]]
                if 'IR intensities in' in line:
                    section_eigenvalues = True

        eigenvalues = [- fc / cons.force_conversion if f < 0 else fc / cons.force_conversion
                       for fc, f in zip(force_constants, frequencies)]
        self.structures = structures
        self.eigenvalues = np.array(eigenvalues)
        self.eigenvectors = np.array(eigenvectors)
        self.frequencies = np.array(frequencies)
        # self.force_constants = np.array(force_constants)
        logger.log(2, '{} structures'.format(len(self.structures)))
        logger.log(2, '{} frequencies'.format(len(self.frequencies)))
        logger.log(2, '{} eigenvalues'.format(len(self.eigenvalues)))
        logger.log(2, '{} eigenvectors'.format(self.eigenvectors.shape))
        num_atoms = len(structures[-1].atoms)
        # logger.log(3, '({}, {}) eigenvectors expected for linear molecule'.format(
        #         num_atoms * 3 - 5, num_atoms * 3))
        # logger.log(3, '({}, {}) eigenvectors expected for nonlinear molecule'.format(
        #         num_atoms * 3 - 6, num_atoms * 3))
        
class Mae(SchrodingerFile):
    '''
    Used to retrieve data from Schrodinger mae files.
    '''
    def __init__(self, path):
        super(Mae, self).__init__(path)
        self._structures = None
        # self._sch_structs = None
    @property
    def structures(self):
        if self._structures is None:
            logger.log(2, 'reading {}'.format(self.path))
            sch_structs = list(schrod_structure.StructureReader(self.path))
            structures = [self.convert_schrodinger_structure(sch_struct) for sch_struct in sch_structs]
            logger.log(2, 'imported {} structures'.format(len(structures)))
            # self._sch_structs = sch_structs
            self._structures = structures
        return self._structures
    def get_aliph_hyds(self):
        '''
        Returns the atom numbers of aliphatic hydrogens. These hydrogens
        always receive a charge of zero in MacroModel calculations.
        '''
        aliph_hyd_nums = []
        atoms = self.structures[0].atoms
        bonds = self.structures[0].bonds
        for atom in atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyd_nums.append(atom.index)
        return aliph_hyd_nums

class MacroModelLog(File):
    '''
    Used to retrieve data from MacroModel log files.
    '''
    def __init__(self, path):
        super(MacroModelLog, self).__init__(path)
        self._hessian = None
    @property
    def hessian(self):
        if self._hessian is None:
            logger.log(2, 'reading {}'.format(self.path))
            with open(self.path, 'r') as f:
                lines = f.read()
            num_atoms = int(re.search('Read\s+(\d+)\s+atoms.', lines).group(1))
            logger.log(2, '{} atoms'.format(num_atoms))
            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(2, '{} hessian matrix'.format(hessian.shape))
            words = lines.split()
            section_hessian = False
            start_row = False
            start_col = False
            for i, word in enumerate(words):
                # 1. Start of Hessian section.
                if word == 'Mass-weighted':
                    section_hessian = True
                    continue
                # 5. End of Hessian. Add last row of Hessian and break.
                if word == 'Eigenvalues:':
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    section_hessian = False
                    break
                # 4. End of a Hessian row. Add to matrix and reset.
                if section_hessian and start_col and word == 'Element':
                    for col_num, element in zip(col_nums, elements):
                        hessian[row_num - 1, col_num - 1] = element
                    start_col = False
                    start_row = True
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    continue
                # 2. Start of a Hessian row.
                if section_hessian and word == 'Element':
                    row_num = int(words[i + 1])
                    col_nums = []
                    elements = []
                    start_row = True
                    continue
                # 3. Okay, made it through the row number. Now look for columns
                #    and elements.
                if section_hessian and start_row and word == ':':
                    start_row = False
                    start_col = True
                    continue
                if section_hessian and start_col and '.' not in word:
                    col_nums.append(int(word))
                    continue
                if section_hessian and start_col and '.' in word:
                    elements.append(float(word))
                    continue
            self._hessian = hessian
        return self._hessian

class MacroModel(File):
    '''
    Extracts data from MacroModel mmo files.
    '''
    def __init__(self, path):
        super(MacroModel, self).__init__(path)
        self._structures = None
    @property
    def structures(self):
        if self._structures is None:
            logger.log(2, 'reading {}'.format(self.path))
            structures = []
            with open(self.path, 'r') as f:
                count_current = 0
                count_input = 0
                count_structure = 0
                count_previous = 0
                section = None
                for line in f:
                    if 'Input filename' in line:
                        count_input += 1
                    if 'Input Structure Name' in line:
                        count_structure += 1
                    count_previous = count_current
                    count_current = max(count_input, count_structure) # sometimes both are used
                                                                      # sometimes only one is used
                    if count_current != count_previous:
                        current_structure = Structure()
                        structures.append(current_structure)
                    if 'BOND LENGTHS AND STRETCH ENERGIES' in line:
                        section = 'bond'
                    if 'ANGLES, BEND AND STRETCH BEND ENERGIES' in line:
                        section = 'angle'
                    if 'BEND-BEND ANGLES AND ENERGIES' in line:
                        section = None
                    if section == 'bond':
                        match = cons.re_bond.match(line)
                        if match:
                            atom_nums = map(int, [match.group(1), match.group(2)])
                            comment = match.group(4) # this has a lot of extra white space
                            value = float(match.group(3))
                            current_structure.bonds.append(
                                Bond(atom_nums=atom_nums, comment=comment, value=value))
                    if section == 'angle':
                        match = cons.re_angle.match(line)
                        if match:
                            atom_nums = map(int, [match.group(1), match.group(2), match.group(3)])
                            comment = match.group(5)
                            value = float(match.group(4))
                            current_structure.angles.append(
                                Angle(atom_nums=atom_nums, comment=comment, value=value))
            logger.log(3, 'imported {} structures'.format(len(structures)))
            self._structures = structures
        return self._structures

class Structure(object):
    '''
    Data for a single structure/conformer/snapshot.
    '''
    __slots__ = ['atoms', 'bonds', 'angles', 'props']
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.props = {}
    @property
    def coords(self):
        '''
        Returns atomic coordinates as a list of lists.
        '''
        return [atom.coords for atom in self.atoms]
    def format_coords(self, format='latex'):
        '''
        Returns a list of strings/lines to easily generate coordinate
        lists in various formats.
        '''
        elements = {1: 'H',
                    6: 'C',
                    7: 'N',
                    8: 'O',
                    79: 'Au'}
        if format == 'latex':
            output = ['\\begin{tabular}{l S[table-format=3.6] S[table-format=3.6] S[table-format=3.6]}']
            for i, atom in enumerate(self.atoms):
                output.append('{0}{1} & {2:3.6f} & {3:3.6f} & {4:3.6f}\\\\'.format(
                        elements[atom.atomic_num], i+1, atom.x, atom.y, atom.z))
            output.append('\\end{tabular}')
            return output
        elif format == 'gauss':
            output = []
            for i, atom in enumerate(self.atoms):
                output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                        elements[atom.atomic_num], atom.x, atom.y, atom.z))
            return output

class Atom(object):
    '''
    Data class for a single atom.
    '''
    __slots__ = ['atom_type', 'atom_type_name', 'atomic_num', 'atomic_mass',
                 'bonded_atom_indices', 'coords_type', 'element', 'exact_mass',
                 'index', 'partial_charge', 'x', 'y', 'z']
    def __init__(self):
        self.atom_type = None
        self.atom_type_name = None
        self.atomic_num = None
        self.atomic_mass = None
        self.bonded_atom_indices = None
        self.coords_type = None
        self.element = None
        self.exact_mass = None
        self.index = None
        self.partial_charge = None
        self.x = None
        self.y = None
        self.z = None
    @property
    def coords(self):
        return [self.x, self.y, self.z]
    def __repr__(self):
        return '{}[{},{}]'.format(self.element, self.atom_type, self.atom_type_name)

class Bond(object):
    '''
    Data class for a single bond.
    '''
    __slots__ = ['atom_nums', 'comment', 'order', 'value']
    def __init__(self, atom_nums=None, comment=None, order=None, value=None):
        self.atom_nums = atom_nums
        self.comment = comment
        self.order = order
        self.value = value

class Angle(Bond):
    '''
    Data class for a single angle.
    '''
    def __init__(self, atom_nums=None, comment=None, order=None, value=None):
        super(Angle, self).__init__(atom_nums, comment, order, value)
        
