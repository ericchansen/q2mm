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
import itertools
import logging
import mmap
import numpy as np
import os
import re
import subprocess as sp

from schrodinger import structure as sch_str
from schrodinger.application.jaguar import input as jag_in

import constants as co
import datatypes

logger = logging.getLogger(__name__)
np.set_printoptions(threshold=np.nan)

class File(object):
    """
    Base for every other filetype class.
    """
    def __init__(self, path):
        self.path = os.path.abspath(path)
        # self.path = path
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]
        
class GaussFormChk(File):
    """
    Class used to retrieve data from Gaussian formatted checkpoint files.
    """
    def __init__(self, path):
        super(GaussFormChk, self).__init__(path)
        self.atoms = []
        # Not sure these should really be called the eigenvalues.
        self.evals = None
        self._hess = None
    @property
    def hess(self):
        if self._hess is None:
            self.read_self()
        return self._hess
    def read_self(self):
        logger.log(5, 'READING: {}'.format(self.filename))

        # Manual example.
        # with open(self.path, 'r') as f:
        #     s_cart_grad = False
        #     s_cart_fc = False
        #     for line in f:
        #         if s_cart_grad is True:
        #             pass
        #         if 'Cartesian Gradient' in line:
        #             s_cart_grad = True
        #             cols = line.split()
        #             logger.log(
        #                 5,
        #                 '  -- Reading cartesian gradient vector of length {}.'.
        #                 format(cols[-1]))
        #             self.evec = np.zeros(1, int(cols[-1]))
        #         if 'Cartesian Force Constants' in line:
        #             s_cart_grad = False
        #             s_cart_fc = True
        #             cols = line.split()
        #             logger.log(
        #                 5,
        #                 '  -- Reading {} cartesian force constants in lower '
        #                 'triangular form.'.format(cols[-1]))
        #         if 'Dipole Moment' in line:
        #             s_cart_fc = False
       
        # re.finditer example
        # stuff = re.finditer(
        #     'Atomic numbers\s+I\s+N=\s+(?P<num>\d+)\n\s+(?P<anum>.*)Nuclear charges',
        #     # 'Atomic numbers\s+I\s+N=\d+\s+(.*)Nuclear charges',
        #     open(self.path, 'r').read(), flags=re.DOTALL)
        # stuff = [m.groupdict() for m in stuff]
        # print(stuff)

        stuff = re.search(
            'Atomic numbers\s+I\s+N=\s+(?P<num_atoms>\d+)'
            '\n\s+(?P<anums>.*?)'
            'Nuclear charges.*?Current cartesian coordinates.*?\n(?P<coords>.*?)'
            'Force Field'
            '.*?Real atomic weights.*?\n(?P<masses>.*?)'
            'Atom fragment info.*?Cartesian Gradient.*?\n(?P<evals>.*?)'
            'Cartesian Force Constants.*?\n(?P<hess>.*?)'
            'Dipole Moment',
            open(self.path, 'r').read(), flags=re.DOTALL)

        anums = map(int, stuff.group('anums').split())
        masses = map(float, stuff.group('masses').split())
        coords = map(float, stuff.group('coords').split())
        coords = [coords[i:i+3] for i in range(0, len(coords), 3)]
        for anum, mass, coord in itertools.izip(anums, masses, coords):
            self.atoms.append(
                Atom(
                    atomic_num = anum,
                    coords = coord,
                    exact_mass = mass)
                )
        logger.log(5, '  -- Read {} atoms.'.format(len(self.atoms)))
        self.evals = np.array(
            map(float, stuff.group('evals').split()), dtype=float)
        logger.log(5, '  -- Read {} eigenvectors.'.format(len(self.evals)))
        self.low_tri = np.array(
            map(float, stuff.group('hess').split()), dtype=float)
        one_dim = len(anums) * 3
        self._hess = np.empty([one_dim, one_dim], dtype=float)
        self._hess[np.tril_indices_from(self._hess)] = self.low_tri
        self._hess += np.tril(self._hess, -1).T
        # Convert to MacroModel units.
        self._hess *= co.HESSIAN_CONVERSION
        logger.log(5, '  -- Read {} Hessian.'.format(self._hess.shape))

class GaussLog(File):
    """
    Class used to retrieve data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """
    def __init__(self, path):
        super(GaussLog, self).__init__(path)
        self._structures = None
        self._evals = None
        self._evecs = None
    @property
    def evecs(self):
        if self._evecs is None:
            self.read_out()
        return self._evecs
    @property
    def evals(self):
        if self._evals is None:
            self.read_out()
        return self._evals
    @property
    def structures(self):
        if self._structures is None:
            self.read_archive()
        return self._structures
    def read_out(self):
        """
        Read force constant and eigenvector data from a frequency
        calculation.
        """
        import pHessMan
        atoms, evals, evecs = pHessMan.readgauout(self.path)
        evals = np.array(evals)
        evecs = np.array(evecs)
        self._evals = evals
        self._evecs = evecs
        return evals, evecs

        # logger.log(5, 'READING: {}'.format(self.filename))
        # stuffs = re.findall(
        #     'Frequencies --(.*?)'
        #     '\n.*?Z\s*?\n(.*?)Frequencies',
        #     open(self.path, 'r').read(), flags=re.DOTALL)
        # freqs = []
        # # Works with nonlinear molecules only.
        # # Not good to read this for the atoms.
        # evecs = np.zeros((len(self.structures[0].atoms)*3 - 6, len(self.structures[0].atoms)))
        # logger.log(5, '  -- Made {} eigenvector matrix.'.format(evecs.shape))

        # stuff = stuffs[0]
        # # Get the frequencies on this line.
        # freqs.extend(map(float, stuff[0].split()))
        # # Get the eigenvectors.
        # indices = evec_lines
        # evecs_lines = stuff[1].split('\n')[:-3]
        # for line in evecs_lines:
        #     print(line)

        # print(freqs)
        # with open(self.path, 'r') as f:
        #     sec = False
        #     for line in f:
        #         if sec == True:
        #             cols = line.split()
        #             if len(cols) == 11:
        #                 try:
        #                     cols = map(float, cols)
        #                     print(line)
                            
        #                 except ValueError:
        #                     pass
        #         if len(line.split()) == 0:
        #             sec = False
        #         if 'and normal coordinates:' in line:
        #             sec = True
                    
    def read_archive(self):
        """
        Only reads last archive found in the Gaussian .log file.
        """
        logger.log(5, 'READING: {}'.format(self.filename))
        struct = Structure()
        # Some more manual trimming.
        # lines = open(self.path, 'r').readlines()
        # for i, line in enumerate(lines):
        #     if '1\\1\\' in line:
        #         last_arch_start = i
        # lines = ''.join(lines[last_arch_start:])
        # arch = re.findall(
        #     '(\s1\\\\1\\\\(?s).*?[\\\\]+@)', 
        #     lines)[0]
        arch = re.findall(
            '(\s1\\\\1\\\\(?s).*?[\\\\]+@)', 
            open(self.path, 'r').read())[-1]
        logger.log(5, '  -- Found last archive.')
        arch = arch.replace('\n ', '')
        # Watch out with NImag. Sometimes equals 1, sometimes 0,
        # anything else?
        stuff = re.search(
            '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
            '\\\\(?P<date>.*?)'
            '\\\\.*?\\\\\\\\(?P<com>.*?)'
            '\\\\\\\\(?P<filename>.*?)'
            '\\\\\\\\(?P<charge>.*?)'
            ',(?P<multiplicity>.*?)'
            '\\\\(?P<atoms>.*?)'
            '\\\\\\\\.*?HF=(?P<hf>.*?)'
            '\\\\NImag=[10]\\\\\\\\(?P<hess>.*?)'
            '\\\\\\\\(?P<evals>.*?)'
            '\\\\\\\\\\\\',
            arch)
        logger.log(5, '  -- Found stuff in archive.')
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
            5,
            '  -- Read {} Hessian elements in lower triangular '
            'form.'.format(len(hess_tri)))
        hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
        logger.log(
            5, '  -- Created {} Hessian matrix.'.format(hess.shape))
        # Code for if it was in upper triangle, but it's not in
        # upper triangle format. It's lower triangle.
        # hess[np.triu_indices_from(hess)] = hess_tri
        # hess += np.triu(hess, -1).T
        # Lower triangle code.
        hess[np.tril_indices_from(hess)] = hess_tri
        hess += np.tril(hess, -1).T
        hess *= co.HESSIAN_CONVERSION
        struct.hess = hess
        # self._hessian = datatypes.Hessian()
        # self._hessian.hessian = hess
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
    def read_any_coords(self, coords_type='both'):
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
    def read_optimization(self, coords_type='both'):
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
    def conv_sch_str(self, sch_struct):
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
            my_atom.props.update(sch_atom.property)
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
            sch_ob = jag_in.read(self.path)
            sch_struct = sch_ob.getStructure()
            structures = [self.conv_sch_str(sch_struct)]
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
    Used to retrieve and work with data from Schrodinger .mae files.
    """
    def __init__(self, path):
        super(Mae, self).__init__(path)
        self._index_output_mae = None
        self._index_output_mmo = None
        self._structures = None
        self.commands = None
        # Strings for keeping track of this file and output files.
        self.name = os.path.splitext(self.filename)[0]
        self.name_com = self.name + '.q2mm.com'
        self.name_log = self.name + '.q2mm.log'
        self.name_mae = self.name + '.q2mm.mae'
        self.name_mmo = self.name + '.q2mm.mmo'
        self.name_out = self.name + '.q2mm.out'
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.path))
            sch_structs = list(sch_str.StructureReader(self.path))
            self._structures = [self.conv_sch_str(sch_struct)
                                for sch_struct in sch_structs]
            logger.log(5, '  -- Imported {} structure(s).'.format(
                    len(self._structures)))
        return self._structures
    def get_com_opts(self):
        """
        Takes the users arguments from calculate (ex. mb, me, etc.) and
        determines what has to be written to the .com file in order to
        generate the requested data using MacroModel.

        Returns
        -------
        dictionary of options used when writing a .com file
        """
        com_opts = {
            'cs1': False,
            'cs2': False,
            'cs3': False,
            'freq': False,
            'opt': False,
            'opt_mmo': False,
            'sp': False,
            'sp_mmo': False,
            'strs': False,
            'tors': False}
        if len(self.structures) > 1:
            com_opts['strs'] = True
        if any(x in ['ja', 'jb', 'jt'] for x in self.commands):
            com_opts['sp_mmo'] = True
        if any(x in ['me', 'me2', 'mq', 'mqh'] for x in self.commands):
            com_opts['sp'] = True
        # Command meig is depreciated.
        if any(x in ['meig', 'mjeig', 'mgeig', 'mh'] for x in self.commands):
            if com_opts['strs']:
                raise Exception(
                    "Can't obtain the Hessian from a Maestro file "
                    "containing multiple structures!\n"
                    "FILENAME: {}\n"
                    "COMMANDS:{}\n".format(
                        self.path, ' '.join(commands)))
            else:
                com_opts['freq'] = True
        if any(x in ['ma', 'mb', 'meo', 'mt'] for x in self.commands):
            com_opts['opt'] = True
            com_opts['opt_mmo'] = True
        elif any(x in ['ma', 'mb', 'mt'] for x in self.commands):
            com_opts['opt'] = True
        if any(x in ['mt', 'jt'] for x in self.commands):
            com_opts['tors'] = True
        if any(x in ['mcs', 'mcs2', 'mcs3'] for x in self.commands) and \
                any(x for x in com_opts.itervalues()):
            raise Exception(
                "Conformational search methods must be used alone!\n"
                "FILENAME: {}\n"
                "COMMANDS: {}\n".format(
                    self.path, ' '.join(commands)))
        if 'mcs' in self.commands:
            com_opts['cs1'] = True
        elif 'mcs2' in self.commands:
            com_opts['cs2'] = True
        elif 'mcs3' in self.commands:
            com_opts['cs3'] = True
        return com_opts
    def get_debg_opts(self, com_opts):
        """
        Determines what arguments are needed for the DEBG line used inside
        a MacroModel .com file.

        Returns
        -------
        list of integers
        """
        debg_opts = []
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            return None
        else:
            debg_opts.append(57)
        if com_opts['tors']:
            debg_opts.append(56)
        if com_opts['freq']:
            debg_opts.extend((210, 211))
        debg_opts.sort()
        debg_opts.insert(0, 'DEBG')
        while len(debg_opts) < 9:
            debg_opts.append(0)
        return debg_opts
    def write_com(self):
        """
        Writes the .com file with all the right arguments to generate
        the requested data.
        """
        self._index_output_mae = []
        self._index_output_mmo = []
        com_opts = self.get_com_opts()
        debg_opts = self.get_debg_opts(com_opts)
        com = '{}\n{}\n'.format(self.filename, self.name_mae)
        if debg_opts:
            com += co.COM_FORM.format(*debg_opts)
        else:
            com += co.COM_FORM.format('MMOD', 0, 1, 0, 0, 0, 0, 0, 0)
        # May want to turn off arg2 (continuum solvent).
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('FFLD', 2, 1, 0, 0, 0, 0, 0, 0)
        else:
            com += co.COM_FORM.format('FFLD', 2, 0, 0, 0, 0, 0, 0, 0)
        # Also may want to turn off these cutoffs.
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('BDCO', 0, 0, 0, 0, 41.5692, 99999, 0, 0)
        if com_opts['strs']:
            com += co.COM_FORM.format('BGIN', 0, 0, 0, 0, 0, 0, 0, 0)
        # Look into differences.
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('READ', 0, 0, 0, 0, 0, 0, 0, 0)
        else:
            com += co.COM_FORM.format('READ', -1, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['sp'] or com_opts['sp_mmo']:
            com += co.COM_FORM.format('ELST', 1, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mmo.append('pre')
            com += co.COM_FORM.format('WRIT', 0, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mae.append('pre')
        if com_opts['freq']:
            com += co.COM_FORM.format('MINI', 9, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mae.append('stupid_extra_structure')
            # What does arg1 as 3 even do?
            com += co.COM_FORM.format('RRHO', 3, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mae.append('hess')
        if com_opts['opt']:
            # Commented line was used in code from Per-Ola/Elaine.
            # com += co.COM_FORM.format('MINI', 9, 0, 50, 0, 0, 0, 0, 0)

            # TNCG has more risk of not converging, and may print NaN instead
            # of coordinates and forces to output.
            # arg1: 1 = PRCG, 9 = TNCG
            com += co.COM_FORM.format('MINI', 1, 0, 500, 0, 0, 0, 0, 0) 
            self._index_output_mae.append('opt')
        if com_opts['opt_mmo']:
            com += co.COM_FORM.format('ELST', 1, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mmo.append('opt')
        if com_opts['strs']:
            com += co.COM_FORM.format('END', 0, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('CRMS', 0, 0, 0, 0, 0, 0.25, 0, 0)
        if com_opts['cs1']:
            com += co.COM_FORM.format('MCMM', 10000, 0, 0, 0, 0, 0.25, 0, 0)
        if com_opts['cs2']:
            com += co.COM_FORM.format('LMCS', 10000, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['cs3']:
            com += co.COM_FORM.format('LMCS', 4000, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('NANT', 0, 0, 0, 0, 0, 0, 0, 0)
        # if com_opts['cs2'] or com_opts['cs3']:
        #     com += co.COM_FORM.format('MCNV', 1, 5, 0, 0, 0, 0, 0, 0)
        if com_opts['cs1']:
            com += co.COM_FORM.format('MCSS', 2, 0, 0, 0, 50, 0, 0, 0)
            com += co.COM_FORM.format('MCOP', 1, 0, 0, 0, 0, 0, 0, 0)
            com += co.COM_FORM.format('DEMX', 0, 166, 0, 0, 50, 100, 0, 0)
        if com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('MCOP', 1, 0, 0, 0, 0.5, 0, 0, 0)
            com += co.COM_FORM.format('DEMX', 0, 833, 0, 0, 50, 100, 0, 0)
        # I don't think MSYM does anything when all arguments are set to zero.
        if com_opts['cs1'] or com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('MSYM', 0, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['cs2']:
            com += co.COM_FORM.format('AUOP', 0, 0, 0, 0, 400, 0, 0, 0)
        # I'm not sure if this does anything either.
        if com_opts['cs3']:
            com += co.COM_FORM.format('AUOP', 0, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['cs1']:
            com += co.COM_FORM.format('AUTO', 0, 2, 1, 1, 0, -1, 0, 0)
        if com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('AUTO', 0, 3, 1, 2, 1, 1, 4, 3)
        if com_opts['cs1']:
            com += co.COM_FORM.format('CONV', 2, 0, 0, 0, 0.5, 0, 0, 0)
        if com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('CONV', 2, 0, 0, 0, 0.05, 0, 0, 0)
        if com_opts['cs1']:
            com += co.COM_FORM.format('MINI', 9, 0, 500, 0, 0, 0, 0, 0)
        if com_opts['cs2'] or com_opts['cs3']:
            com += co.COM_FORM.format('MINI', 1, 0, 2500, 0, 0.05, 0, 0, 0)
        with open(os.path.join(self.directory, self.name_com), 'w') as f:
            f.write(com)
        logger.log(5, 'WROTE: {}'.format(
                os.path.join(self.directory, self.name_com)))
    def run(self, max_timeout=None, timeout=10, check_tokens=True):
        """
        Runs MacroModel .com files. This has to be more complicated than a
        simple subprocess command due to problems with Schrodinger tokens.
        This script checks the available tokens, and if there's not enough,
        waits to run MacroModel until there are.
 
        Arguments
        ---------
        max_timeout : int
                      Maximum number of attempts to look for Schrodinger
                      license tokens before giving up.
        timeout : float
                  Time waited in between lookups of Schrodinger license
                  tokens.
        """
        current_directory = os.getcwd()
        os.chdir(self.directory)
        current_timeout = 0
        if check_tokens is True:
            logger.log(5, "  -- Checking Schrodinger tokens.")
            while True:
                token_string = sp.check_output(
                    '$SCHRODINGER/utilities/licutil -available', shell=True)
                suite_tokens = re.search(co.LIC_SUITE, token_string)
                macro_tokens = re.search(co.LIC_MACRO, token_string)
                if not suite_tokens or not macro_tokens:
                    raise Exception(
                        'The command "$SCHRODINGER/utilities/licutil '
                        '-available" is not working with the current '
                        'regex in calculate.py.')
                suite_tokens = int(suite_tokens.group(1))
                macro_tokens = int(macro_tokens.group(1))
                if suite_tokens > co.MIN_SUITE_TOKENS and \
                        macro_tokens > co.MIN_MACRO_TOKENS:
                    logger.log(5, 'RUNNING: {}'.format(self.name_com))
                    sp.check_output(
                        'bmin -WAIT {}'.format(
                            os.path.splitext(self.name_com)[0]), shell=True)
                    break
                else:
                    if max_timeout is not None and \
                            current_timeout > max_timeout:
                        pretty_timeout(
                            current_timeout, suite_tokens,
                            macro_tokens, end=True)
                        raise Exception(
                            "Not enough tokens to run {}. Waited {} seconds "
                            "before giving up.".format(
                                self.name_com, current_timeout))
                    pretty_timeout(current_timeout, suite_tokens, macro_tokens)
                    current_timeout += timeout
                    time.sleep(timeout)
        else:
                    logger.log(5, 'RUNNING: {}'.format(self.name_com))
                    sp.check_output(
                        'bmin -WAIT {}'.format(
                            os.path.splitext(self.name_com)[0]), shell=True)
        os.chdir(current_directory)

def pretty_timeout(current_timeout, macro_tokens, suite_tokens, end=False,
                   level=10):
    """
    Logs information about the wait for Schrodinger tokens.

    Arguments
    ---------
    current_timeout : int
                      Number of times waited for Schrodinger tokens.
    macro_tokens : int
                   Current number of available MacroModel tokens.
    suite_tokens : int
                   Current number of available Schrodinger Suite tokens.
    end : bool
          If True, adds a pretty ending border to all these logs.
    level : int
            Logging level of the pretty messages.
    """
    if current_timeout == 0:
        logger.warning('  -- Waiting on tokens to run {}.'.format(
                self.name_com))
        logger.log(level,
                   '--' + ' (s) '.center(8, '-') +
                   '--' + ' {} '.format(LABEL_SUITE).center(17, '-') +
                   '--' + ' {} '.format(LABEL_MACRO).center(17, '-') +
                   '--')
    logger.log(level, '  {:^8d}  {:^17d}  {:^17d}'.format(
            current_timeout, macro_tokens, suite_tokens))
    if end is True:
        logger.log(level, '-' * 50)
        
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
    __slots__ = ['atoms', 'bonds', 'angles', 'torsions', 'hess', 'props']
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.torsions = []
        self.hess = None
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
        # Formatted for LaTeX.
        if format == 'latex':
            output = ['\\begin{tabular}{l S[table-format=3.6] '
                      'S[table-format=3.6] S[table-format=3.6]}']
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.element[atom.atomic_num]
                else:
                    ele = atom.element
                output.append('{0}{1} & {2:3.6f} & {3:3.6f} & '
                              '{4:3.6f}\\\\'.format(
                        ele, i+1, atom.x, atom.y, atom.z))
            output.append('\\end{tabular}')
            return output
        # Formatted for Gaussian .com's.
        elif format == 'gauss':
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.element[atom.atomic_num]
                else:
                    ele = atom.element
                output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                        ele, atom.x, atom.y, atom.z))
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
    def get_aliph_hyds(self):
        """
        Returns the atom numbers of aliphatic hydrogens. These hydrogens
        are always assigned a partial charge of zero in MacroModel
        calculations.
        """
        aliph_hyds = []
        for atom in self.atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = self.atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyds.append(atom)
        logger.log(5, '  -- {} aliphatic hydrogen(s).'.format(len(aliph_hyds)))
        return aliph_hyds

class Atom(object):
    """
    Data class for a single atom.
    """
    __slots__ = ['atom_type', 'atom_type_name', 'atomic_num', 'atomic_mass',
                 'bonded_atom_indices', 'coords_type', '_element',
                 '_exact_mass', 'index', 'partial_charge', 'x', 'y', 'z',
                 'props']
    def __init__(self, atom_type=None, atom_type_name=None, atomic_num=None,
                 atomic_mass=None, bonded_atom_indices=None, coords=None,
                 coords_type=None, element=None, exact_mass=None, index=None,
                 partial_charge=None, x=None, y=None, z=None):
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
        if coords:
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
        self.props = {}
    @property
    def coords(self):
        return [self.x, self.y, self.z]
    @coords.setter
    def coords(self, value):
        try:
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        except TypeError:
            pass
    @property
    def element(self):
        if self._element is None:
            self._element = co.ele[self.atomic_num]
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

class Reference(File):
    """
    Basic class for directly inputting reference data.

    File looks as follows.

    Each row is a data point. The attributes of the data points
    are separated by spaces, tabs, whatever Python's built-in
    string.split() function works on.

    Columns:
    val - Value
    wht - Weight
    typ - Data type (must match the MacroModel type)
    src_1 - 1st data source
    src_2 - 2nd data source
    idx_1 - 1st index
    idx_2 - 2nd index
    atm_1 - 1st atom
    atm_2 - 2nd atom
    atm_3 - 3rd atom
    atm_4 - 4th atom
    """
    def __init__(self, path):
        self.path = path
    def get_data(self):
        """
        Returns a list of data points. Each data point is a dictionary.
        """
        with open(self.path, 'r') as f:
            data = []
            for line in f:
                line = line.partition('#')[0]
                cols = line.split()
                labels = ['val', 'wht', 'com', 'typ', 'src_1', 'src_2', 'idx_1',
                          'idx_2', 'atm_1', 'atm_2', 'atm_3', 'atm_4']
                assert len(labels) == len(cols), \
                    'Column mismatch in reference data file!'
                datum = {}
                for label, col in zip(labels, cols):
                    datum.update({label: self.interpret_col(col)})
                data.append(datum)
        return data
    def interpret_col(self, col):
        """
        How to deal with an individual column.

        Converts the string "None" to actually being None.
        Converts floats to floats.
        Otherwise returns the string.
        """
        if col == 'None':
            return None
        try:
            atr = float(col)
            return atr
        except ValueError:
            return col
