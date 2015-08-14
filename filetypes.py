"""
Handles importing data from the various filetypes that Q2MM uses.

Schrodinger
-----------
If the atom.typ file is not in the directory where run the Python scripts,
you may see a warning similar to the following when you import Schrodinger
files:

  WARNING mmat_get_atomic_num x is not a valid atom type
  WARNING mmat_get_mmod_name x is not a valid atom type

In this example, x would be the number of the custom atom types you defined,
atom types that you added to atom.typ. The warning can safely be ignored. If
you'd like the warning messages to go away, copy the atom.typ file into the
directory where you execute the Python scripts. Note that the atom.typ file
must be located with your structure files, else the MacroModel jobs will fail.
the atom.typ file
"""
from __future__ import print_function
from string import digits
import logging
import mmap
import numpy as np
import os
import re

from schrodinger import structure as schrod_structure
from schrodinger.application.jaguar import input as schrod_jaguar_in

import constants as co
import datatypes

logger = logging.getLogger(__name__)

class File(object):
    """
    Base class for all filetypes.
    """
    def __init__(self, path):
        # self.path = path
        self.path = os.path.abspath(path)
        self.filename = os.path.basename(self.path)
        # self.directory = os.path.dirname(self.path)
        # self.name = os.path.splitext(self.filename)[0]
        
class GaussLog(File):
    """
    Class used to retrieve data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """
    def __init__(self, path):
        super(GaussLog, self).__init__(path)
        self._structures = None
        self._hessian = None
    @property
    def structures(self):
        if self._structures is None:
            self.populate()
        return self._structures
    def populate(self):
            logger.log(5, 'READING: {}'.format(self.filename))
            struct = Structure()
            arch = re.findall(
                '(\s1\\\\1\\\\(?s).*?[\\\\]+@)', 
                open(self.path, 'r').read())[-1]
            arch = arch.replace('\n ', '')
            stuff = re.search(
                '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
                '\\\\(?P<date>.*?)'
                '\\\\.*?\\\\\\\\(?P<com>.*?)'
                '\\\\\\\\(?P<filename>.*?)'
                '\\\\\\\\(?P<charge>.*?)'
                ',(?P<multiplicity>.*?)'
                '\\\\(?P<atoms>.*?)'
                '\\\\\\\\.*?HF=(?P<hf>.*?)'
                '\\\\NImag=1\\\\\\\\(?P<hess>.*?)'
                '\\\\\\\\(?P<evals>.*?)'
                '\\\\\\\\\\\\',
                arch)
            atoms = stuff.group('atoms')
            atoms = atoms.split('\\')
            for atom in atoms:
                ele, x, y, z = atom.split(',')
                struct.atoms.append(
                    Atom(element=ele, x=x, y=y, z=z))
            logger.log(5, '  -- Read {} atoms.'.format(len(atoms)))
            self._structures = [struct]
            hess_tri = stuff.group('hess')
            hess_tri = hess_tri.split(',')
            logger.log(
                5, '  -- Read {} Hessian elements in triangular form.'.format(
                    len(hess_tri)))
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(
                5, '  -- Created {} Hessian matrix.'.format(hess.shape))
            # hess[np.triu_indices_from(hess)] = hess_tri
            # hess += np.triu(hess, -1).T
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
            np.set_printoptions(threshold=np.nan)
            self._hessian = datatypes.Hessian()
            self._hessian.hessian = hess
    def get_most_converged(self, structures=None):
        """
        Used with geometry optimizations that don't succeed. Sometimes
        intermediate geometries obtain better convergence than the
        final geometry. This function returns the class Structure for
        the most converged geometry, which can then be used to output
        the coordinates for the next optimization.
        """
        if structures is None:
            structures = self.structures
        structures_compared = 0
        best_structure = None
        best_yes_or_no = None
        fields = ['RMS Force', 'RMS Displacement', 'Maximum Force',
                  'Maximum Displacement']
        for i, structure in reversed(list(enumerate(structures))):
            yes_or_no = [value[2] for key, value in structure.props.items()
                         if key in fields]
            if not structure.atoms:
                logger.warning('  -- No atoms found in structure {}. '
                               'Skipping.'.format(i+1))
                continue
            if len(yes_or_no) == 4:
                structures_compared += 1
                if best_structure is None:
                    logger.log(10, '  -- Most converged structure: {}'.format(
                            i+1))
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') > best_yes_or_no.count('YES'):
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') == best_yes_or_no.count('YES'):
                    number_better = 0
                    for field in fields:
                        if structure.props[field][0] < \
                                best_structure.props[field][0]:
                            number_better += 1
                    if number_better > 2:
                        best_structure = structure
                        best_yes_or_no = yes_or_no
            elif len(yes_or_no) != 0:
                logger.warning(
                    '  -- Partial convergence criterion in structure: {}'.format(
                        self.path))
        logger.log(10, '  -- Compared {} out of {} structures.'.format(
                structures_compared, len(self.structures)))
        return best_structure
    def import_any(self, coords_type='both'):
        logger.log(10, 'READING: {}'.format(self.path))
        structures = []
        with open(self.path, 'r') as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                    # Look for input coordinates.
                    if coords_type == 'input' or coords_type == 'both':
                        # Marks end of input coords for a given structure.
                        if section_coords_input and 'Distance matrix' in line:
                            section_coords_input = False
                            logger.log(5, '[L{}] End of input coordinates '
                                       '({} atoms).'.format(
                                    i+1, count_atom))
                        # Add atoms and coordinates to structure.
                        if section_coords_input:
                            match = re.match(
                                '\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+'
                                '({0})'.format(co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.coords_type = 'input'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and \
                                'Input orientation:' in line:
                            current_structure = Structure()
                            structures.append(current_structure)
                            section_coords_input = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start input coordinates '
                                       'section.'.format(i+1))
                    # Look for standard coordinates.
                    if coords_type == 'standard' or coords_type == 'both':
                        # End of coordinates for a given structure.
                        if section_coords_standard and \
                                ('Rotational constants' in line or
                                 'Leave Link' in line):
                            section_coords_standard = False
                            logger.log(5, '[L{}] End standard coordinates '
                                       'section ({} atoms).'.format(
                                    i+1, count_atom))
                        # grab coords for each atom. add atoms to the structure
                        if section_coords_standard:
                            match = re.match('\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+'
                                             '({0})\s+({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num: 
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.coords_type = 'standard'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # start of standard coords
                        if not section_coords_standard and \
                                'Standard orientation' in line:
                            current_structure = Structure()
                            structures.append(current_structure)
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start standard coordinates '
                                       'section.'.format(i+1))
        return structures
    def import_optimization(self, coords_type='both'):
        """
        Finds structures from a Gaussian geometry optimization that
        are listed throughout the log file. Also finds data about
        their convergence.

        coords_type = "input" or "standard" or "both"
                      Using both may cause coordinates in one format
                      to be overwritten by whatever comes later in the
                      log file.
        """
        logger.log(10, 'READING: {}'.format(self.path))
        structures = []
        with open(self.path, 'r') as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                # Look for start of optimization section of log file and
                # set a flag that it has indeed started.
                if section_optimization and 'Optimization stopped.' in line:
                    section_optimization = False
                    logger.log(5, '[L{}] End optimization section.'.format(i+1))
                if not section_optimization and \
                        'Search for a local minimum.' in line:
                    section_optimization = True
                    logger.log(5, '[L{}] Start optimization section.'.format(
                            i+1))
                if section_optimization:
                    # Start of a structure.
                    if 'Step number' in line:
                        structures.append(Structure())
                        current_structure = structures[-1]
                        logger.log(5, '[L{}] Added structure '
                                   '(currently {}).'.format(
                                i+1, len(structures)))
                    # Look for convergence information related to a single
                    # structure.
                    if section_convergence and 'GradGradGrad' in line:
                        section_convergence = False
                        logger.log(5, '[L{}] End convergence section.'.format(
                                i+1))
                    if section_convergence:
                        match = re.match(
                            '\s(Maximum|RMS)\s+(Force|Displacement)\s+({0})\s+'
                            '({0})\s+(YES|NO)'.format(
                                co.RE_FLOAT), line)
                        if match:
                            current_structure.props['{} {}'.format(
                                    match.group(1), match.group(2))] = \
                                (float(match.group(3)),
                                 float(match.group(4)), match.group(5))
                    if 'Converged?' in line:
                        section_convergence = True
                        logger.log(5, '[L{}] Start convergence section.'.format(
                                i+1))
                    # Look for input coords.
                    if coords_type == 'input' or coords_type == 'both':
                        # End of input coords for a given structure.
                        if section_coords_input and 'Distance matrix' in line:
                            section_coords_input = False
                            logger.log(5, '[L{}] End input coordinates section '
                                       '({} atoms).'.format(
                                    i+1, count_atom))
                        # Add atoms and coords to structure.
                        if section_coords_input:
                            match = re.match(
                                '\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+'
                                '({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == \
                                        int(match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = \
                                        int(match.group(2))
                                current_atom.coords_type = 'input'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and \
                                'Input orientation:' in line:
                            section_coords_input = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start input coordinates '
                                       'section.'.format(i+1))
                    # Look for standard coords.
                    if coords_type == 'standard' or coords_type == 'both':
                        # End of coordinates for a given structure.
                        if section_coords_standard and \
                                ('Rotational constants' in line or
                                 'Leave Link' in line):
                            section_coords_standard = False
                            logger.log(5, '[L{}] End standard coordinates '
                                       'section ({} atoms).'.format(
                                    i+1, count_atom))
                        # Grab coords for each atom. Add atoms to the structure.
                        if section_coords_standard:
                            match = re.match('\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+'
                                             '({0})\s+({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num: 
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                            i+1, int(match.group(2)),
                                            current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.coords_type = 'standard'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of standard coords.
                        if not section_coords_standard and \
                                'Standard orientation' in line:
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start standard coordinates '
                                       'section.'.format(i+1))
        return structures
                            
class SchrodingerFile(File):
    """
    Parent class used for all Schrodinger files.
    """
    def convert_schrodinger_structure(self, sch_struct):
        """
        Converts a schrodinger.structure object to my own structure object.
        Sort of pointless. Probably remove soon.
        """
        my_struct = Structure()
        my_struct.props.update(sch_struct.property)
        for sch_atom in sch_struct.atom:
            my_atom = Atom()
            my_struct.atoms.append(my_atom)
            my_atom.atom_type = sch_atom.atom_type
            my_atom.atom_type_name = sch_atom.atom_type_name
            my_atom.atomic_num = sch_atom.atomic_number
            my_atom.bonded_atom_indices = \
                [x.index for x in sch_atom.bonded_atoms]
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
    """
    Used to retrieve data from Jaguar .in files.
    """
    def __init__(self, path):
        super(JaguarIn, self).__init__(path)
        self._structures = None
        self._hessian = None
    @property
    def hessian(self):
        if self._hessian is None:
            num_atoms = len(self.structures[0].atoms)
            assert num_atoms != 0, \
                'Zero atoms found when loading Hessian from {}!'.format(
                self.path)
            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(5, '  -- Created {} Hessian matrix.'.format(
                    hessian.shape))
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
                                hessian[hess_row - 1, i + hess_col - 1] = \
                                    float(hess_ele)
                    if '&hess' in line:
                        section_hess = True
            self._hessian = hessian * co.HESSIAN_CONVERSION
        return self._hessian
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.path))
            sch_ob = schrod_jaguar_in.read(self.path)
            sch_struct = sch_ob.getStructure()
            structures = [self.convert_schrodinger_structure(sch_struct)]
            logger.log(5, '  -- Imported {} structure(s).'.format(
                    len(structures)))
            # This area is sketch. I added it so I could use Hessian data
            # generated from a Jaguar calculation that had a dummy atom.
            # No gaurantees this will always work.
            for i, structure in enumerate(structures): 
                empty_atoms = []
                for atom in structure.atoms:
                    if atom.element == '':
                        empty_atoms.append(atom)
                for atom in empty_atoms:
                    structure.atoms.remove(atom)
                if empty_atoms:
                    logger.log(5, 'Structure {}: {} empty atoms '
                               'removed.'.format(i + 1, len(empty_atoms)))
            self._structures = structures
        return self._structures

class JaguarOut(File):
    """
    Used to retrieve data from Schrodinger Jaguar .out files.
    """
    def __init__(self, path):
        super(JaguarOut, self).__init__(path)
        self._structures = None
        self._eigenvalues = None
        self._eigenvectors = None
        self._frequencies = None
        # self._force_constants = None
    @property
    def structures(self):
        if self._structures is None:
            self.import_file()
        return self._structures
    @property
    def eigenvalues(self):
        if self._eigenvalues is None:
            self.import_file()
        return self._eigenvalues
    @property
    def eigenvectors(self):
        if self._eigenvectors is None:
            self.import_file()
        return self._eigenvectors
    @property
    def frequencies(self):
        if self._frequencies is None:
            self.import_file()
        return self._frequencies
    def import_file(self):
        logger.log(10, 'READING: {}'.format(self.path))
        frequencies = []
        force_constants = []
        eigenvectors = []
        structures = []
        with open(self.path, 'r') as f:
            section_geometry = False
            section_eigenvalues = False
            section_eigenvectors = False
            for i, line in enumerate(f):
                if section_geometry:
                    cols = line.split()
                    if len(cols) == 0:
                        section_geometry = False
                        structures.append(current_structure)
                        continue
                    elif len(cols) == 1:
                        pass
                    else:
                        match = re.match(
                            '\s+([\d\w]+)\s+({0})\s+({0})\s+({0})'.format(
                                co.RE_FLOAT), line)
                        if match != None:
                            current_atom = Atom()
                            current_atom.element = match.group(1).translate(
                                None, digits)
                            current_atom.x = float(match.group(2))
                            current_atom.y = float(match.group(3))
                            current_atom.z = float(match.group(4))
                            current_structure.atoms.append(current_atom)
                            logger.log(0,
                                       '{0:<3}{1:>12.6f}{2:>12.6f}'
                                       '{3:>12.6f}'.format(
                                    current_atom.element, current_atom.x,
                                    current_atom.y, current_atom.z))
                if 'geometry:' in line:
                    section_geometry = True
                    current_structure = Structure()
                    logger.log(5, '[L{}] Located geometry.'.format(i + 1))
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
                if section_eigenvalues is True and \
                        section_eigenvectors is False:
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
        eigenvalues = [- fc / co.FORCE_CONVERSION if f < 0 else
                         fc / co.FORCE_CONVERSION
                         for fc, f in zip(force_constants, frequencies)]
        self._structures = structures
        self._eigenvalues = np.array(eigenvalues)
        self._eigenvectors = np.array(eigenvectors)
        self._frequencies = np.array(frequencies)
        # self._force_constants = np.array(force_constants)
        logger.log(5, '  -- Read {} structures'.format(
                len(self.structures)))
        logger.log(5, '  -- Read {} frequencies.'.format(
                len(self.frequencies)))
        logger.log(5, '  -- Read {} eigenvalues.'.format(
                len(self.eigenvalues)))
        logger.log(5, '  -- Read {} eigenvectors.'.format(
                self.eigenvectors.shape))
        num_atoms = len(structures[-1].atoms)
        # logger.log(5,
        #            '  -- ({}, {}) eigenvectors expected for linear '
        #            'molecule.'.format(
        #         num_atoms * 3 - 5, num_atoms * 3))
        # logger.log(5, '  -- ({}, {}) eigenvectors expected for nonlinear '
        #            'molecule.'.format(
        #         num_atoms * 3 - 6, num_atoms * 3))
        
class Mae(SchrodingerFile):
    """
    Used to retrieve data from Schrodinger .mae files.
    """
    def __init__(self, path):
        super(Mae, self).__init__(path)
        self._structures = None
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.path))
            sch_structs = list(schrod_structure.StructureReader(self.path))
            self._structures = [self.convert_schrodinger_structure(sch_struct)
                                for sch_struct in sch_structs]
            logger.log(5, '  -- Imported {} structure(s).'.format(
                    len(self._structures)))
        return self._structures
    def get_aliph_hyds(self):
        """
        Returns the atom numbers of aliphatic hydrogens. These hydrogens
        always receive a charge of zero in MacroModel calculations.
        """
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
    """
    Used to retrieve data from MacroModel log files.
    """
    def __init__(self, path):
        super(MacroModelLog, self).__init__(path)
        self._hessian = None
    @property
    def hessian(self):
        if self._hessian is None:
            logger.log(10, 'READING: {}'.format(self.path))
            with open(self.path, 'r') as f:
                lines = f.read()
            num_atoms = int(re.search('Read\s+(\d+)\s+atoms.', lines).group(1))
            logger.log(5, '  -- Read {} atoms.'.format(num_atoms))
            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(5, '  -- Read {} Hessian matrix.'.format(hessian.shape))
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
    """
    Extracts data from MacroModel .mmo files.
    """
    def __init__(self, path):
        super(MacroModel, self).__init__(path)
        self._structures = None
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.path))
            self._structures = []
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
                    # Sometimes only one of the above ("Input filename" and
                    # "Input Structure Name") is used, sometimes both are used.
                    # count_current will make sure you catch both.
                    count_current = max(count_input, count_structure)
                    # If these don't match, then we reached the end of a
                    # structure.
                    if count_current != count_previous:
                        current_structure = Structure()
                        self._structures.append(current_structure)
                    # For each structure we come across, look for sections that
                    # we are interested in: those pertaining to bonds, angles,
                    # and torsions. Of course more could be added. We set the
                    # section to None to mark the end of a section, and we leave
                    # it None for parts of the file we don't care about.
                    if 'BOND LENGTHS AND STRETCH ENERGIES' in line:
                        section = 'bond'
                    if 'ANGLES, BEND AND STRETCH BEND ENERGIES' in line:
                        section = 'angle'
                    if 'BEND-BEND ANGLES AND ENERGIES' in line:
                        section = None
                    if 'DIHEDRAL ANGLES AND TORSIONAL ENERGIES' in line:
                        section = 'torsion'
                    if 'DIHEDRAL ANGLES AND TORSIONAL CROSS-TERMS' in line:
                        section = None
                    if section == 'bond':
                        bond = self.read_line_for_bond(line)
                        if bond is not None:
                            current_structure.bonds.append(bond)
                    if section == 'angle':
                        angle = self.read_line_for_angle(line)
                        if angle is not None:
                            current_structure.angles.append(angle)
                    if section == 'torsion':
                        torsion = self.read_line_for_torsion(line)
                        if torsion is not None:
                            current_structure.torsions.append(torsion)
            logger.log(5, '  -- Imported {} structure(s).'.format(
                    len(self._structures)))
        return self._structures
    def read_line_for_bond(self, line):
        match = co.RE_BOND.match(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(2)])
            value = float(match.group(3))
            comment = match.group(4).strip()
            ff_row = int(match.group(5))
            return Bond(atom_nums=atom_nums, comment=comment, value=value,
                        ff_row=ff_row)
        else:
            return None
    def read_line_for_angle(self, line):
        match = co.RE_ANGLE.match(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(2),
                                  match.group(3)])
            value = float(match.group(4))
            comment = match.group(5).strip()
            ff_row = int(match.group(6))
            return Angle(atom_nums=atom_nums, comment=comment, value=value,
                         ff_row=ff_row)
        else:
            return None
    def read_line_for_torsion(self, line):
        match = co.RE_TORSION.match(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(2),
                                  match.group(3), match.group(4)])
            value = float(match.group(5))
            comment = match.group(6).strip()
            ff_row = int(match.group(7))
            return Torsion(atom_nums=atom_nums, comment=comment, value=value,
                           ff_row=ff_row)
        else:
            return None

def select_structures(structures, indices, label):
        """
        Returns a list of structures where the index matches the label. This
        is used with the structures in the class MacroModel (.mmo's) and Mae
        (.mae's of course).

        Basically, you're not sure what structures appear in these files if the
        files were generated using calculate.py and the .com files it writes.
        Fear not! calculate.py keeps track of that for you (using indices) and
        knows which structures to use.

        indices - A list of strings (labels).
        label   - A string. Possible strings include:
                      'opt', 'pre', 'hess' (.mae only), and
                      'stupid_extra_structure'
        """
        selected = []
        idx_iter = iter(indices)
        for str_num, struct in enumerate(structures):
            try:
                idx_curr = idx_iter.next()
            except StopIteration:
                idx_iter = iter(indices)
                idx_curr = idx_iter.next()
            if idx_curr == label:
                selected.append((str_num, struct))
        return selected

class Structure(object):
    """
    Data for a single structure/conformer/snapshot.
    """
    __slots__ = ['atoms', 'bonds', 'angles', 'torsions', 'props']
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.torsions = []
        self.props = {}
    @property
    def coords(self):
        """
        Returns atomic coordinates as a list of lists.
        """
        return [atom.coords for atom in self.atoms]
    def format_coords(self, format='latex'):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.
        """
        # Please expand the supported elements.
        elements = {1: 'H',
                    6: 'C',
                    7: 'N',
                    8: 'O',
                    79: 'Au'}
        # Formatted for LaTeX.
        if format == 'latex':
            output = ['\\begin{tabular}{l S[table-format=3.6] '
                      'S[table-format=3.6] S[table-format=3.6]}']
            for i, atom in enumerate(self.atoms):
                output.append('{0}{1} & {2:3.6f} & {3:3.6f} & '
                              '{4:3.6f}\\\\'.format(
                        elements[atom.atomic_num], i+1, atom.x, atom.y, atom.z))
            output.append('\\end{tabular}')
            return output
        # Formatted for Gaussian .com's.
        elif format == 'gauss':
            output = []
            for i, atom in enumerate(self.atoms):
                output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                        elements[atom.atomic_num], atom.x, atom.y, atom.z))
            return output
    def select_stuff(self, typ, com_match=None, **kwargs):
        """
        Selects bonds, angles, or torsions from the structure and returns them
        in the format used as data in the sqlite3 database.

        typ       - 'Bond', 'Angle', or 'Torsion'.
        com_match - String or None. If None, just returns all of the selected
                    stuff (bonds, angles, or torsions). If a string, selects
                    only those that have this string in their comment.

                    In .mmo files, the comment corresponds to the substructures
                    name. This way, we only fit bonds, angles, and torsions that
                    directly depend on our parameters.
        """
        data = []
        for thing in getattr(self, typ):
            if (com_match and thing.comment in com_match) or \
                    com_match is None:
                datum = thing.as_data(**kwargs)
                # Done now by thing.as_data.
                # datum.update(kwargs)
                # datum = {k: datum.get(k, co.DEFAULTS[k]) for k in co.DEFAULTS}
                data.append(datum)
        assert data, "No data actually retrieved!"
        return data

class Atom(object):
    """
    Data class for a single atom.
    """
    __slots__ = ['atom_type', 'atom_type_name', 'atomic_num', 'atomic_mass',
                 'bonded_atom_indices', 'coords_type', 'element', 'exact_mass',
                 'index', 'partial_charge', 'x', 'y', 'z']
    def __init__(self, atom_type=None, atom_type_name=None, atomic_num=None,
                 atomic_mass=None, bonded_atom_indices=None, coords_type=None,
                 element=None, exact_mass=None, index=None, partial_charge=None,
                 x=None, y=None, z=None):
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num
        self.atomic_mass = atomic_mass
        self.bonded_atom_indices = bonded_atom_indices
        self.coords_type = coords_type
        self.element = element
        self.exact_mass = exact_mass
        self.index = index
        self.partial_charge = partial_charge
        self.x = x
        self.y = y
        self.z = z
    @property
    def coords(self):
        return [self.x, self.y, self.z]
    def __repr__(self):
        return '{}[{},{},{}]'.format(
            self.element, self.x, self.y, self.z)

class Bond(object):
    """
    Data class for a single bond.
    """
    __slots__ = ['atom_nums', 'comment', 'order', 'value', 'ff_row']
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        self.atom_nums = atom_nums
        self.comment = comment
        self.order = order
        self.value = value
        self.ff_row = ff_row
    def __repr__(self):
        return '{}[{}]({})'.format(
            self.__class__.__name__, '-'.join(
                map(str, self.atom_nums)), self.value)
    def as_data(self, **kwargs):
        datum = {'val': self.value, 
                 'typ': self.__class__.__name__
                 }
        for i, atom_num in enumerate(self.atom_nums):
            datum.update({'atm_{}'.format(i + 1): atom_num})
        datum.update(kwargs)
        datum = co.set_data_defaults(datum)
        return datum

class Angle(Bond):
    """
    Data class for a single angle.
    """
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        super(Angle, self).__init__(atom_nums, comment, order, value, ff_row)

class Torsion(Bond):
    """
    Data class for a single torsion.
    """
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        super(Torsion, self).__init__(atom_nums, comment, order, value, ff_row)
