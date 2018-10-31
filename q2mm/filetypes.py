#!/usr/bin/env python
"""
Handles importing data from the various filetypes that Q2MM uses.

Schrodinger
-----------
When importing Schrodinger files, if the atom.typ file isn't in the directory
where you execute the Q2MM Python scripts, you may see this warning:

  WARNING mmat_get_atomic_num x is not a valid atom type
  WARNING mmat_get_mmod_name x is not a valid atom type

In this example, x is the number of a custom atom type defined and added to
atom.typ. The warning can be ignored. If it's bothersome, copy atom.typ into
the directory where you execute the Q2MM Python scripts.

Note that the atom.typ must be located with your structure files, else the
Schrodinger jobs will fail.
"""
from __future__ import print_function
from argparse import RawTextHelpFormatter
from string import digits
import itertools
import logging
import mmap
import numpy as np
import math
import os
import re
import subprocess as sp
import time
import sys

try:
    from schrodinger import structure as sch_str
    from schrodinger.application.jaguar import input as jag_in
except:
    print("Schrodinger not installed, limited functionality")
    pass

import constants as co
import datatypes

logger = logging.getLogger(__name__)
# Print out full matrices rather than having Numpy truncate them.
np.set_printoptions(threshold=np.nan)

class File(object):
    """
    Base for every other filetype class.
    """
    def __init__(self, path):
        self._lines = None
        self.path = os.path.abspath(path)
        # self.path = path
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]
    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, 'r') as f:
                self._lines = f.readlines()
        return self._lines
    def write(self, path, lines=None):
        if lines is None:
            lines = self.lines
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)

class AmberInput(File):
    """
    Some sort of generic class for Amber shell scripts.
    """
    SCRIPT_ENERGY = \
"""Energy of current thingy
 &cntrl
 ig=-1,
 imin=1,
 ncyc=0, maxcyc=0,
 ntb=1,
 &end
"""
    def __init__(self, path):
        super(AmberInput, self).__init__(path)
    def run(self, path=None, inpcrd=None, prmtop=None, **kwargs):
        # Added `**kwargs` to deal with this varying from the MacroModel run
        # command.
        with open('AMBER_TEMP.in', 'w') as f:
            f.writelines(self.SCRIPT_ENERGY)
        if not path:
            # Seriously, this doesn't matter.
            path = self.path
        if not inpcrd:
            inpcrd = os.path.join(self.directory, self.inpcrd)
        if not prmtop:
            prmtop = os.path.join(self.directory, self.prmtop)
        # Could use inpcrd as the basis for the output filename?
        path = os.path.splitext(prmtop)
        path, ext = path[0], path[1]
        self.out = path + '.out'
        self.rst = path + '.rst'
        # sp.call(
        #     'sander -O -i AMBER_TEMP.in -o {} -c {} -p {} -r {} -ref {}'.format(
        #         self.out, inpcrd, prmtop, self.rst, inpcrd),
        #     shell=True)

class AmberOut(File):
    """
    Some sort of generic class for Amber output files.
    """
    LINE_HEADER = '\s+NSTEP\s+ENERGY\s+RMS\s+GMAX\s+NAME\s+NUMBER[\s+]?\n+'
    def __init__(self, path):
        super(AmberOut, self).__init__(path)
    def read_energy(self, path=None):
        if not path:
            path = self.path
        logger.log(1, '>>> path: {}'.format(path))
        with open(path, 'r') as f:
            string = f.read()
        # No idea if this will find more than just this one. Sure hope not!
        # I'd double check those energies.
        # something = re.findall(
        #     '\s+FINAL\sRESULTS\s+\n+{}\s+{}\s+' \
        #     '(?P<energy>{})'.format(self.LINE_HEADER, co.RE_FLOAT, co.RE_FLOAT),
        #     string,
        #     re.MULTILINE)
        # if something:
        #     logger.log(1, '>>> something: {}'.format(something))
        #     energy = float(something[-1])
        #     logger.log(1, '>>> energy: {}'.format(energy))
        #     return energy
        # else:
        #     raise Exception("Awww bummer! I can't find the energy "
        #                     "in {}!".format(path))
        # Here's an iterative version.
        re_compiled = re.compile('FINAL\sRESULTS\s+\n+{}\s+{}\s+'
                                 '(?P<energy>{})'.format(
                        self.LINE_HEADER, co.RE_FLOAT, co.RE_FLOAT))
        somethings = [x.groupdict() for x in re_compiled.finditer(string)]
        if somethings:
            logger.log(1, '>>> somethings: {}'.format(somethings))
            energy = float(somethings[-1]['energy'])
            logger.log(1, '>>> energy: {}'.format(energy))
            return energy
        else:
            raise Exception("Awww bummer! I can't find the energy "
                            "in {}!".format(path))

class TinkerHess(File):
    def __init__(self, path):
        super(TinkerHess, self).__init__(path)
        self._hessian = None
        self.natoms = None
    @property
    def hessian(self):
        if self._hessian is None:
            logger.log(10, 'READING: {}'.format(self.filename))
            hessian = np.zeros([self.natoms * 3, self.natoms * 3], dtype=float)
            logger.log(5, '  -- Creatting {} Hessian Matrix.'.format(
                hessian.shape))
            with open(self.path, 'r') as f:
                lines = f.read()
            words = lines.split()
            diag = True
            row_num = 0
            col_num = 0
            line = -1
            index = 0
            for i, word in enumerate(words):
                match = re.compile('\d+[.]\d+').search(word)
                # First group of values are all of the diagonal elements. So
                # This will grab them first and put them in the correct index
                # of the Hessian.
                if diag and match:
                    hessian[row_num, col_num] = word
                    row_num += 1
                    col_num += 1
                # After the first group of values the line will read
                # 'Off-diagonal'. This signifies when the next elements are
                # H_i, j for section i.
                if word == 'Off-diagonal':
                    diag = False
                    line += 1
                    index = line + 1
                    row_num = 0
                    col_num = 0
                if not diag and match:
                    hessian[line, col_num + index] = word
                    hessian[row_num + index, line] = word
                    row_num += 1
                    col_num += 1
            # Convert hessian units to use kJ/mol instead of kcal/mol.
            self._hessian = hessian / co.HARTREE_TO_KCALMOL \
                * co.HARTREE_TO_KJMOL
            logger.log(5, '  -- Finished Creating {} Hessian matrix.'.format(
                hessian.shape))
            return self._hessian

class TinkerLog(File):
    def __init__(self, path):
        super(TinkerLog, self).__init__(path)
        self._structures = None
        self.name = None
    @property
    def structures(self):
        if self._structures == None:
            logger.log(10, 'READING: {}'.format(self.filename))
            self._structures = []
            with open(self.path, 'r') as f:
                sections = {'sp':1, 'minimization':2, 'hessian':2}
                count_previous = 0
                calc_section = 'sp'
                for line in f:
                    count_current = sections[calc_section]
                    if count_current != count_previous:
                        # Due to TINKER printing to standard error and standard
                        # out the redirection of this printout to the *q2mm.log
                        # does not print the bonds and angles in the correct
                        # order or in a consistent order. Therefore we need to
                        # sort the bonds, angles, and torsions in a standard
                        # way before extending the data in calculate.
                        bonds = []
                        angles = []
                        torsions = []
                        current_structure = Structure()
                        self._structures.append(current_structure)
                        count_previous += 1
                    section = None
                    if "SINGLE POINT" in line:
                        calc_section = 'minimization'
                        for bond in bonds:
                            # Not sure if I have to sort the atom list but
                            # I'm doing it anyway.
                            bond.atom_nums.sort()
                        # Sorts the bonds by the first atom and then by
                        # the second.
                        bonds.sort(key=lambda x: (x.atom_nums[0],
                                                  x.atom_nums[1]))
                        for angle in angles:
                            if angle.atom_nums[0] > angle.atom_nums[2]:
                                angle = [angle.atom_nums[2],
                                         angle.atom_nums[1],
                                         angle.atom_nums[0]]
                        angles.sort(key=lambda x: (x.atom_nums[1],
                                                   x.atom_nums[0],
                                                   x.atom_nums[2]))
                        torsions.sort(key=lambda x: (x.atom_nums[1],
                                                     x.atom_nums[2],
                                                     x.atom_nums[0],
                                                     x.atom_nums[3]))
                        current_structure.bonds.extend(bonds)
                        current_structure.angles.extend(angles)
                        current_structure.torsions.extend(torsions)
                    if 'END OF OPTIMIZED SINGLE POINT' in line:
                        calc_section = 'hessian'
                    if 'Bond' in line:
                        bond = self.read_line_for_bond(line)
                        if bond is not None:
                            bonds.append(bond)
                    if 'Angle' in line:
                        angle = self.read_line_for_angle(line)
                        if angle is not None:
                            angles.append(angle)
                    if 'Torsion' in line:
                        torsion = self.read_line_for_torsion(line)
                        if torsion is not None:
                            torsions.append(torsion)
                    if 'Total Potential Energy' in line:
                        energy = self.read_line_for_energy(line)
                        if energy is not None:
                            current_structure.props['energy']=energy
                    if "END OF CALCULATION" in line and \
                        calc_section != 'hessian':
                        last_ind = len(self._structures) - 1
                        self._structures.remove(self._structures[last_ind])
            logger.log(5, '  -- Imported {} structure(s)'.format(
                len(self._structures)))
        return self._structures

    def read_line_for_bond(self, line):
        # All bond data starts with the string "Bond" and then the rest of the
        # interaction information.
        match = re.compile('Bond\s+(\d+)-(\w+)\s+(\d+)-(\w+)\s+'
            '({0})\s+({0})\s+({0})'.format(co.RE_FLOAT)).search(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(3)])
            value = float(match.group(6))
            return Bond(atom_nums=atom_nums, value=value)
        else:
            return None

    def read_line_for_angle(self, line):
        # All angle data starts with the string "Angle" and then the rest of the
        # interaction information.
        match = re.compile('Angle\s+(\d+)-(\w+)\s+(\d+)-(\w+)\s+(\d+)-(\w+)\s+'
            '({0})\s+({0})\s+({0})'.format(co.RE_FLOAT)).search(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(3),
                match.group(5)])
            value = float(match.group(8))
            return Angle(atom_nums=atom_nums, value=value)
        else:
            return None

    def read_line_for_torsion(self, line):
        # All torsion data starts with the string "torsion" and then the rest of
        # the interaction information.
        match = re.compile('Torsion\s+(\d+)-(\w+)\s+(\d+)-(\w+)\s+'
            '(\d+)-(\w+)\s+(\d+)-(\w+)\s+({0})\s+({0})'.format(
                co.RE_FLOAT)).search(line)
        if match:
            atom_nums = map(int, [match.group(1), match.group(3),
                match.group(5), match.group(7)])
            value = float(match.group(9))
            return Angle(atom_nums=atom_nums, value=value)
        else:
            return None

    def read_line_for_energy(self, line):
        # The TPE is in units of kcal/mol, so we have to convert them to kJ/mol
        # for consistency purposes.
        match = re.compile('Total Potential Energy :\s+({0})'.format(
            co.RE_FLOAT)).search(line)
        if match:
            energy = float(match.group(1))
            energy *= co.HARTREE_TO_KJMOL / co.HARTREE_TO_KCALMOL
            return energy
        else:
            return None

class TinkerXYZ(File):
    def __init__(self, path):
        super(TinkerXYZ, self).__init__(path)
        self._index_output_log = None
        self._structures = None
        self.commands = None
        self.name = os.path.splitext(self.filename)[0]
        # Key file is needed to set the settings for the calculation, including
        # the parameters needed to perform the calculation.
        self.name_key = self.name + '.q2mm.key'
        # The log file is a file that contains the information redirected from
        # the TINKER calculations that are performed with Q2MM. This is not a
        # file setup by TINKER. TINKER will only print to the front end except
        # for select files such as a newly minimized structure. In this case
        # the minimized structure will be saved to *.q2mm.xyz.
        self.name_log = self.name + '.q2mm.log'
        self.name_xyz = self.name + '.q2mm.xyz'
        self.name_hes = self.name + '.q2mm.hes'
        self.name_1st_hess = self.name + '.hes'
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.filename))
            struct = Structure()
            self._structures = [struct]
            with open(self.filename, 'r') as f:
                for line in f:
                    line = line.split()
                    if len(line) == 2:
                        struct.props['total atoms'] = int(line[0])
                        struct.props['title'] = line[1]
                        logger.log(5, '  -- Read {} atoms.'.format(
                            struct.props['total atoms']))
                    if len(line) > 2:
                        indx, ele, x, y, z, at, bonded_atom = line[0], \
                            line[1], line[2], line[3], line[4], \
                            line[5], line[6:]
                        struct.atoms.append(Atom(index=indx,
                            element=ele,
                            x=float(x),
                            y=float(y),
                            z=float(z),
                            atom_type=at,
                            atom_type_name=at,
                            bonded_atom_indices=bonded_atom))
            return self._structures
    def get_com_opts(self):
        com_opts = {'freq': False,
                    'opt': False,
                    'sp': False,
                    'tors': False}
        if any(x in ['tb', 'ta', 'tt', 'te', 'tea'] for x in self.commands):
            com_opts['sp'] = True
        if any(x in ['tbo','tao','tto','teo','teao'] for x in self.commands):
            com_opts['opt'] = True
            com_opts['sp'] = True
        if any(x in ['th', 'tjeig', 'tgeig'] for x in self.commands):
            com_opts['freq'] = True
            com_opts['opt'] = True
            com_opts['sp'] = True
        if any(x in ['tt', 'tto'] for x in self.commands):
            com_opts['tors'] = True
        return com_opts
    def run(self,check_tokens=False):
        logger.log(5, 'RUNNING: {}'.format(self.filename))
        self._index_output_log = []
        com_opts = self.get_com_opts()
        current_directory = os.getcwd()
        os.chdir(self.directory)
        if os.path.isfile(self.name_log):
            os.remove(self.name_log)
        if os.path.isfile(self.name_xyz):
            os.remove(self.name_xyz)
        if os.path.isfile(self.name_hes):
            os.remove(self.name_hes)
        if com_opts['sp']:
            logger.log(1, '  ANALYZE: {}'.format(self.filename))
            with open(self.name_log, 'w') as f:
                sp.call(
                    'analyze {}.xyz -k {} D'.format(self.name,
                    self.name_key), shell=True, stderr=f, stdin=f, stdout=f)
                # Not sure if these print outs are important, but I add in these
                # to define what section of the *q2mm.log file you are in. This
                # is especially helpful when needed to grab all of the bond,
                # angle, and torsional data since these have to be ordered
                # everytime they are grabbed.
                f.write("\n=======================\
                         \n= END OF SINGLE POINT =\
                         \n=======================\n")
        if com_opts['opt']:
            logger.log(1, '  MINIMIZE & ANALYZE: {}'.format(self.filename))
            with open(self.name_log, 'a') as f:
                # The float value is the convergence criteria.
                sp.call(
                    'minimize {}.xyz -k {} 0.01 {}'.format(self.name,
                    self.name_key, self.name_xyz), shell=True, stderr=f,
                    stdin=f, stdout=f)
                sp.call(
                    'analyze {} -k {} D'.format(self.name_xyz,
                    self.name_key), shell=True, stderr=f, stdin=f, stdout=f)
                f.write("\n=================================\
                         \n= END OF OPTIMIZED SINGLE POINT =\
                         \n=================================\n")
        if com_opts['freq']:
            logger.log(1, '  TESTHESS: {}'.format(self.filename))
            with open(self.name_log, 'a') as f:
                # Tinker will not take a file output argument if the there isn't
                # currently a file.hes. For example, file.xyz will write to
                # file.hes if file.hes doesn't already exist otherwise TINKER
                # will ask the user for a new filename.
                if os.path.isfile(self.name + '.hes'):
                    sp.call(
                        'testhess {} -k {} y n {}'.format(self.name,
                        self.name_key,self.name_hes), shell=True,
                        stderr=f, stdin=f, stdout=f)
                else:
                    sp.call(
                        'testhess {} -k {} y n'.format(self.name,
                        self.name_key), shell=True,
                        stderr=f, stdin=f, stdout=f)
                    os.rename(self.name + '.hes', self.name_hes)
                f.write("\n==================\
                         \n= END OF HESSIAN =\
                         \n==================\n")
        with open(self.name_log, 'a') as f:
            f.write("\n======================\
                     \n= END OF CALCULATION =\
                     \n======================\n")
        os.chdir(current_directory)

class GaussFormChk(File):
    """
    Used to retrieve data from Gaussian formatted checkpoint files.
    """
    def __init__(self, path):
        super(GaussFormChk, self).__init__(path)
        self.atoms = []
        # Not sure these should really be called the eigenvalues.
        self.evals = None
        self.low_tri = None
        self._hess = None
    @property
    def hess(self):
        if self._hess is None:
            self.read_self()
        return self._hess
    def read_self(self):
        logger.log(5, 'READING: {}'.format(self.filename))
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
    Used to retrieve data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """
    def __init__(self, path):
        super(GaussLog, self).__init__(path)
        self._evals = None
        self._evecs = None
        self._structures = None
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
            # self.read_out()
            self.read_archive()
        return self._structures
    def read_out(self):
        """
        Read force constant and eigenvector data from a frequency
        calculation.
        """
        logger.log(5, 'READING: {}'.format(self.filename))
        self._evals = []
        self._evecs = []
        self._structures = []
        force_constants = []
        evecs = []
        with open(self.path, 'r') as f:
            # The keyword "harmonic" shows up before the section we're
            # interested in. It can show up multiple times depending on the
            # options in the Gaussian .com file.
            past_first_harm = False
            # High precision mode, turned on by including "freq=hpmodes" in the
            # Gaussian .com file.
            hpmodes = False
            file_iterator = iter(f)
            # This while loop breaks when the end of the file is reached, or
            # if the high quality modes have been read already.
            while True:
                try:
                    line = file_iterator.next()
                except:
                    # End of file.
                    break
                # Gathering some geometric information.
                if 'orientation:' in line:
                    self._structures.append(Structure())
                    file_iterator.next()
                    file_iterator.next()
                    file_iterator.next()
                    file_iterator.next()
                    line = file_iterator.next()
                    while not '---' in line:
                        cols = line.split()
                        self._structures[-1].atoms.append(
                            Atom(atomic_num=int(cols[1]),
                                 x=float(cols[3]),
                                 y=float(cols[4]),
                                 z=float(cols[5])))
                        line = file_iterator.next()
                    logger.log(5, '  -- Found {} atoms.'.format(
                            len(self._structures[-1].atoms)))
                elif 'Harmonic' in line:
                    # The high quality eigenvectors come before the low quality
                    # ones. If you see "Harmonic" again, it means you're at the
                    # low quality ones now, so break.
                    if past_first_harm:
                        break
                    else:
                        past_first_harm = True
                elif 'Frequencies' in line:
                    # We're going to keep reusing these.
                    # We accumulate sets of eigevectors and eigenvalues, add
                    # them to self._evecs and self._evals, and then reuse this
                    # for the next set.
                    del(force_constants[:])
                    del(evecs[:])
                    # Values inside line look like:
                    #     "Frequencies --- xxxx.xxxx xxxx.xxxx"
                    # That's why we remove the 1st two columns. This is
                    # consistent with and without "hpmodes".
                    # For "hpmodes" option, there are 5 of these frequencies.
                    # Without "hpmodes", there are 3.
                    # Thus the eigenvectors and eigenvalues will come in sets of
                    # either 5 or 3.
                    cols = line.split()
                    for frequency in map(float, cols[2:]):
                        # Has 1. or -1. depending on the sign of the frequency.
                        if frequency < 0.:
                            force_constants.append(-1.)
                        else:
                            force_constants.append(1.)
                        # For now this is empty, but we will add to it soon.
                        evecs.append([])

                    # Moving on to the reduced masses.
                    line = file_iterator.next()
                    cols = line.split()
                    # Again, trim the "Reduced masses ---".
                    # It's "Red. masses --" for without "hpmodes".
                    for i, mass in enumerate(map(float, cols[3:])):
                        # +/- 1 / reduced mass
                        force_constants[i] = force_constants[i] / mass

                    # Now we are on the line with the force constants.
                    line = file_iterator.next()
                    cols = line.split()
                    # Trim "Force constants ---". It's "Frc consts --" without
                    # "hpmodes".
                    for i, force_constant in enumerate(map(float, cols[3:])):
                        # co.AU_TO_MDYNA = 15.569141
                        force_constants[i] *= force_constant / co.AU_TO_MDYNA

                    # Force constants were calculated above as follows:
                    #    a = +/- 1 depending on the sign of the frequency
                    #    b = a / reduced mass (obtained from the Gaussian log)
                    #    c = b * force constant / conversion factor (force
                    #         (constant obtained from Gaussian log) (conversion
                    #         factor is inside constants module)

                    # Skip the IR intensities.
                    file_iterator.next()
                    # This is different depending on whether you use "hpmodes".
                    line = file_iterator.next()
                    # "Coord" seems to only appear when the "hpmodes" is used.
                    if 'Coord' in line:
                        hpmodes = True
                    # This is different depending on whether you use
                    # "freq=projected".
                    line = file_iterator.next()
                    # The "projected" keyword seems to add "IRC Coupling".
                    if 'IRC Coupling' in line:
                        line = file_iterator.next()

                    # We're on to the eigenvectors.
                    # Until the end of this section containing the eigenvectors,
                    # the number of columns remains constant. When that changes,
                    # we know we're to the next set of frequencies, force
                    # constants and eigenvectors.
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        # This will come after all the eigenvectors have been
                        # read. We can break out then.
                        if 'Harmonic' in line:
                            break
                        # If "hpmodes" is used, you have an extra column here
                        # that is simply an index.
                        if hpmodes:
                            cols = cols[1:]
                        # cols corresponds to line(s) (maybe only 1st line)
                        # under section "Coord Atom Element:" (at least for
                        # "hpmodes").

                        # Just the square root of the mass from co.MASSES.
                        # co.MASSES currently has the average mass.
                        # Gaussian may use the mass of the most abundant
                        # isotope. This may be a problem.
                        mass_sqrt = np.sqrt(co.MASSES.items()[int(cols[1]) - 1][1])

                        cols = cols[2:]
                        # This corresponds to the same line still, but without
                        # the atom elements.

                        # This loop expands the LoL, evecs, as so.
                        # Iteration 1:
                        # [[x], [x], [x], [x], [x]]
                        # Iteration 2:
                        # [[x, x], [x, x], [x, x], [x, x], [x, x]]
                        # ... etc. until the length of the sublist is equal to
                        # the number of atoms. Remember, for low precision
                        # eigenvectors it only adds in sets of 3, not 5.

                        # Elements of evecs are simply the data under
                        # "Coord Atom Element" multiplied by the square root
                        # of the weight.
                        for i in range(len(evecs)):
                            if hpmodes:
                                # evecs is a LoL. Length of sublist is
                                # equal to # of columns in section "Coord Atom
                                # Element" minus 3, for the 1st 3 columns
                                # (index, atom index, atomic number).
                                evecs[i].append(float(cols[i]) * mass_sqrt)
                            else:
                                # This is fow low precision eigenvectors. It's a
                                # funny way to go in sets of 3. Take a look at
                                # your low precision Gaussian log and it will
                                # make more sense.
                                for useless in range(3):
                                    x = float(cols.pop(0))
                                    evecs[i].append(x * mass_sqrt)
                        line = file_iterator.next()
                        cols = line.split()

                    # Here the overall number of eigenvalues and eigenvectors is
                    # increased by 5 (high precision) or 3 (low precision). The
                    # total number goes to 3N - 6 for non-linear and 3N - 5 for
                    # linear. Same goes for self._evecs.
                    for i in range(len(evecs)):
                        self._evals.append(force_constants[i])
                        self._evecs.append(evecs[i])
                    # We know we're done if this is in the line.
                    if 'Harmonic' in line:
                        break
        for evec in self._evecs:
            # Each evec is a single eigenvector.
            # Add up the sum of squares over an eigenvector.
            sum_of_squares = 0.
            # Appropriately named, element is an element of that single
            # eigenvector.
            for element in evec:
                sum_of_squares += element * element
            # Now x is the inverse of the square root of the sum of squares
            # for an individual eigenvector.
            element = 1 / np.sqrt(sum_of_squares)
            for i in range(len(evec)):
                evec[i] *= element
        self._evals = np.array(self._evals)
        self._evecs = np.array(self._evecs)
        logger.log(1, '>>> self._evals: {}'.format(self._evals))
        logger.log(1, '>>> self._evecs: {}'.format(self._evecs))
        logger.log(5, '  -- {} structures found.'.format(len(self.structures)))
    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self):
        """
        Only reads last archive found in the Gaussian .log file.
        """
        logger.log(5, 'READING: {}'.format(self.filename))
        struct = Structure()
        self._structures = [struct]
        # Matches everything in between the start and end.
        # (?s)  - Flag for re.compile which says that . matches all.
        # \\\\  - One single \
        # Start - " 1\1\".
        # End   - Some number of \ followed by @. Not sure how many \ there
        #         are, so this matches as many as possible. Also, this could
        #         get separated by a line break (which would also include
        #         adding in a space since that's how Gaussian starts new lines
        #         in the archive).
        # We pull out the last one [-1] in case there are multiple archives
        # in a file.
        try:
            arch = re.findall(
                '(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)',
                open(self.path, 'r').read())[-1]
            logger.log(5, '  -- Located last archive.')
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        # Make it into one string.
        arch = arch.replace('\n ', '')
        # Separate it by Gaussian's section divider.
        arch = arch.split('\\\\')
        # Helps us iterate over sections of the archive.
        section_counter = 0
        # SECTION 0
        # General job information.
        arch_general = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
            '\\\\(?P<date>.*?)'
            '\\\\.*?',
            arch_general)
        struct.props['user'] = stuff.group('user')
        struct.props['date'] = stuff.group('date')
        # SECTION 1
        # The commands you wrote.
        arch_commands = arch[section_counter]
        section_counter += 1
        # SECTION 2
        # The comment line.
        arch_comment = arch[section_counter]
        section_counter += 1
        # SECTION 3
        # Actually has charge, multiplicity and coords.
        arch_coords = arch[section_counter]
        section_counter +=1
        stuff = re.search(
            '(?P<charge>.*?)'
            ',(?P<multiplicity>.*?)'
            '\\\\(?P<atoms>.*)',
            arch_coords)
        struct.props['charge'] = stuff.group('charge')
        struct.props['multiplicity'] = stuff.group('multiplicity')
        # We want to do more fancy stuff with the atoms than simply add to
        # the properties dictionary.
        atoms = stuff.group('atoms')
        atoms = atoms.split('\\')
        # Z-matrix coordinates adds another section. We need to be aware of
        # this.
        probably_z_matrix = False
        for atom in atoms:
            stuff = atom.split(',')
            # An atom typically looks like this:
            #    C,0.1135,0.13135,0.63463
            if len(stuff) == 4:
                ele, x, y, z = stuff
            # But sometimes they look like this (notice the extra zero):
            #    C,0,0.1135,0.13135,0.63463
            # I'm not sure what that extra zero is for. Anyway, ignore
            # that extra whatever if it's there.
            elif len(stuff) == 5:
                ele, x, y, z = stuff[0], stuff[2], stuff[3], stuff[4]
            # And this would be really bad. Haven't seen anything else like
            # this yet.
            # 160613 - So, not sure when I wrote that comment, but something
            # like this definitely happens when using scans and z-matrices.
            # I'm going to ignore grabbing any atoms in this case.
            else:
                logger.warning(
                    'Not sure how to read coordinates from Gaussian acrhive!')
                probably_z_matrix = True
                section_counter += 1
                # Let's have it stop looping over atoms, but not fail anymore.
                break
                # raise Exception(
                #     'Not sure how to read coordinates from Gaussian archive!')
            struct.atoms.append(
                Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        logger.log(20, '  -- Read {} atoms.'.format(len(struct.atoms)))
        # SECTION 4
        # All sorts of information here. This area looks like:
        #     prop1=value1\prop2=value2\prop3=value3
        arch_info = arch[section_counter]
        section_counter += 1
        arch_info = arch_info.split('\\')
        for thing in arch_info:
            prop_name, prop_value = thing.split('=')
            struct.props[prop_name] = prop_value
        # SECTION 5
        # The Hessian. Only exists if you did a frequency calculation.
        # Appears in lower triangular form.
        if not arch[section_counter] == '@':
            hess_tri = arch[section_counter]
            hess_tri = hess_tri.split(',')
            logger.log(
                5,
                '  -- Read {} Hessian elements in lower triangular '
                'form.'.format(len(hess_tri)))
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(
                5, '  -- Created {} Hessian matrix.'.format(hess.shape))
            # Code for if it was in upper triangle (it's not).
            # hess[np.triu_indices_from(hess)] = hess_tri
            # hess += np.triu(hess, -1).T
            # Lower triangle code.
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
            hess *= co.HESSIAN_CONVERSION
            struct.hess = hess
            # SECTION 6
            # Not sure what this is.

        # stuff = re.search(
        #     '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
        #     '\\\\(?P<date>.*?)'
        #     '\\\\.*?\\\\\\\\(?P<com>.*?)'
        #     '\\\\\\\\(?P<filename>.*?)'
        #     '\\\\\\\\(?P<charge>.*?)'
        #     ',(?P<multiplicity>.*?)'
        #     '\\\\(?P<atoms>.*?)'
        #     # This marks the end of what always shows up.
        #     '\\\\\\\\'
        #     # This stuff sometimes shows up.
        #     # And it breaks if it doesn't show up.
        #     '.*?HF=(?P<hf>.*?)'
        #     '\\\\.*?ZeroPoint=(?P<zp>.*?)'
        #     '\\\\.*?Thermal=(?P<thermal>.*?)'
        #     '\\\\.*?\\\\NImag=\d+\\\\\\\\(?P<hess>.*?)'
        #     '\\\\\\\\(?P<evals>.*?)'
        #     '\\\\\\\\\\\\',
        #     arch)
        # logger.log(5, '  -- Read archive.')
        # atoms = stuff.group('atoms')
        # atoms = atoms.split('\\')
        # for atom in atoms:
        #     ele, x, y, z = atom.split(',')
        #     struct.atoms.append(
        #         Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        # logger.log(5, '  -- Read {} atoms.'.format(len(atoms)))
        # self._structures = [struct]
        # hess_tri = stuff.group('hess')
        # hess_tri = hess_tri.split(',')
        # logger.log(
        #     5,
        #     '  -- Read {} Hessian elements in lower triangular '
        #     'form.'.format(len(hess_tri)))
        # hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
        # logger.log(
        #     5, '  -- Created {} Hessian matrix.'.format(hess.shape))
        # # Code for if it was in upper triangle, but it's not.
        # # hess[np.triu_indices_from(hess)] = hess_tri
        # # hess += np.triu(hess, -1).T
        # # Lower triangle code.
        # hess[np.tril_indices_from(hess)] = hess_tri
        # hess += np.tril(hess, -1).T
        # hess *= co.HESSIAN_CONVERSION
        # struct.hess = hess
        # # Code to extract energies.
        # # Still not sure exactly what energies we want to use.
        # struct.props['hf'] = float(stuff.group('hf'))
        # struct.props['zp'] = float(stuff.group('zp'))
        # struct.props['thermal'] = float(stuff.group('thermal'))
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
        logger.log(10, 'READING: {}'.format(self.filename))
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
                        # Grab coordinates for each atom.
                        # Add atoms to the structure.
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
                        # Start of standard coordinates.
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
        logger.log(10, 'READING: {}'.format(self.filename))
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

def conv_sch_str(sch_struct):
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
        self._empty_atoms = None
        self._lines = None
    @property
    def hessian(self):
        """
        Reads the Hessian from a Jaguar .in.

        Automatically removes Hessian elements corresponding to dummy atoms.
        """
        if self._hessian is None:
            num = len(self.structures[0].atoms) + len(self._empty_atoms)
            logger.log(5,
                       '  -- {} has {} atoms and {} dummy atoms.'.format(
                    self.filename,
                    len(self.structures[0].atoms),
                    len(self._empty_atoms)))
            assert num != 0, \
                'Zero atoms found when loading Hessian from {}!'.format(
                self.path)
            hessian = np.zeros([num * 3, num * 3], dtype=float)
            logger.log(5, '  -- Created {} Hessian matrix (including dummy '
                       'atoms).'.format(hessian.shape))
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
            for atom in self._empty_atoms:
                logger.log(1, '>>> _empty_atom {}: {}'.format(atom.index, atom))
            # Figure out the indices of the dummy atoms.
            dummy_indices = []
            for atom in self._empty_atoms:
                logger.log(1, '>>> atom.index: {}'.format(atom.index))
                index = (atom.index - 1) * 3
                dummy_indices.append(index)
                dummy_indices.append(index + 1)
                dummy_indices.append(index + 2)
            logger.log(1, '>>> dummy_indices: {}'.format(dummy_indices))
            # Delete these rows and columns.
            logger.log(1, '>>> hessian.shape: {}'.format(hessian.shape))
            logger.log(1, '>>> hessian:\n{}'.format(hessian))
            hessian = np.delete(hessian, dummy_indices, 0)
            hessian = np.delete(hessian, dummy_indices, 1)
            logger.log(1, '>>> hessian:\n{}'.format(hessian))
            logger.log(5, '  -- Created {} Hessian matrix (w/o dummy '
                       'atoms).'.format(hessian.shape))
            self._hessian = hessian * co.HESSIAN_CONVERSION
            logger.log(1, '>>> hessian.shape: {}'.format(hessian.shape))
        return self._hessian
    @property
    def structures(self):
        if self._structures is None:
            logger.log(10, 'READING: {}'.format(self.filename))
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
                    logger.log(1, '>>> atom {}: {}'.format(atom.index, atom))
                    if atom.element == '':
                        empty_atoms.append(atom)
                for atom in empty_atoms:
                    structure.atoms.remove(atom)
                if empty_atoms:
                    logger.log(5, 'Structure {}: {} empty atoms '
                               'removed.'.format(i + 1, len(empty_atoms)))
            self._empty_atoms = empty_atoms
            self._structures = structures
        return self._structures
    def gen_lines(self):
        """
        Attempts to figure out the lines of itself.

        Since it'd be difficult, the written version will be missing much
        of the data in the original. Maybe there's something in the
        Schrodinger API for that.

        However, I do want this to include the ability to write out an
        atomic section with the ESP data that we'd want.
        """
        lines = []
        mae_name = None
        lines.append('MAEFILE: {}'.format(mae_name))
        lines.append('&gen')
        lines.append('&')
        lines.append('&zmat')
        # Just use the 1st structure. I don't imagine a Jaguar input file
        # ever containing more than one structure.
        struct = self.structures[0]
        lines.extend(struct.format_coords(format='gauss'))
        lines.append('&')
        return lines

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
        self._dummy_atom_eigenvector_indices = None
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
    @property
    def dummy_atom_eigenvector_indices(self):
        if self._dummy_atom_eigenvector_indices is None:
            self.import_file()
        return self._dummy_atom_eigenvector_indices
    def import_file(self):
        logger.log(10, 'READING: {}'.format(self.filename))
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
                if 'normal modes in' in line:
                    section_eigenvalues = True
        logger.log(1, '>>> len(frequencies): {}'.format(len(frequencies)))
        logger.log(1, '>>> frequencies:\n{}'.format(frequencies))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x / co.FORCE_CONVERSION for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x * 4.55633e-6 for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x * 1.23981e-4 for x in frequencies]))
        # logger.log(1, '>>> frequencies:\n{}'.format(
        #         [x / 219474.6305 for x in frequencies]))
        eigenvalues = [- fc / co.FORCE_CONVERSION if f < 0 else
                         fc / co.FORCE_CONVERSION
                         for fc, f in zip(force_constants, frequencies)]
        logger.log(1, '>>> eigenvalues:\n{}'.format(eigenvalues))
        # Remove eigenvector components related to dummy atoms.
        # Find the index of the atoms that are dummies.
        dummy_atom_indices = []
        for i, atom in enumerate(structures[-1].atoms):
            if atom.is_dummy:
                dummy_atom_indices.append(i)
        logger.log(10, '  -- Located {} dummy atoms.'.format(len(dummy_atom_indices)))
        # Correlate those indices to the rows in the cartesian eigenvector.
        dummy_atom_eigenvector_indices = []
        for dummy_atom_index in dummy_atom_indices:
            start = dummy_atom_index * 3
            dummy_atom_eigenvector_indices.append(start)
            dummy_atom_eigenvector_indices.append(start + 1)
            dummy_atom_eigenvector_indices.append(start + 2)
        new_eigenvectors = []
        # Create new eigenvectors without the rows corresponding to the
        # dummy atoms.
        for eigenvector in eigenvectors:
            new_eigenvectors.append([])
            for i, eigenvector_row in enumerate(eigenvector):
                if i not in dummy_atom_eigenvector_indices:
                    new_eigenvectors[-1].append(eigenvector_row)
        # Replace old eigenvectors with new where dummy atoms aren't included.
        eigenvectors = np.array(new_eigenvectors)
        self._dummy_atom_eigenvector_indices = dummy_atom_eigenvector_indices
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
        # num_atoms = len(structures[-1].atoms)
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
            logger.log(10, 'READING: {}'.format(self.filename))
            # It would be great if we could leave this as an iter.
            try:
                sch_structs = list(sch_str.StructureReader(self.path))
            except:
                logger.warning('Error reading {}.'.format(self.path))
                raise
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
            'freq': False,
            'opt': False,
            'opt_mmo': False,
            'sp': False,
            'sp_mmo': False,
            'strs': False,
            'tors': False}
        if len(self.structures) > 1:
            com_opts['strs'] = True
        if any(x in ['jb', 'ja', 'jt'] for x in self.commands):
            com_opts['sp_mmo'] = True
        if any(x in ['me', 'mea', 'mq', 'mqh', 'mqa'] for x in self.commands):
            com_opts['sp'] = True
        # Command meig is depreciated.
        if any(x in ['mh', 'meig', 'mjeig', 'mgeig'] for x in self.commands):
            if com_opts['strs']:
                raise Exception(
                    "Can't obtain the Hessian from a Maestro file "
                    "containing multiple structures!\n"
                    "FILENAME: {}\n"
                    "COMMANDS:{}\n".format(
                        self.path, ' '.join(commands)))
            else:
                com_opts['freq'] = True
        if any(x in ['mb', 'ma', 'mt', 'meo', 'meao'] for x in self.commands):
            com_opts['opt'] = True
            com_opts['opt_mmo'] = True
        elif any(x in ['mb', 'ma', 'mt'] for x in self.commands):
            com_opts['opt'] = True
        if any(x in ['mt', 'jt'] for x in self.commands):
            com_opts['tors'] = True
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
        debg_opts.append(57)
        # Leads to problems when an angle inside a torsion is ~ 0 or 180.
        # if com_opts['tors']:
        #     debg_opts.append(56)
        if com_opts['freq']:
            debg_opts.extend((210, 211))
        debg_opts.sort()
        debg_opts.insert(0, 'DEBG')
        while len(debg_opts) < 9:
            debg_opts.append(0)
        return debg_opts
    def write_com(self, sometext=None):
        """
        Writes the .com file with all the right arguments to generate
        the requested data.
        """
        # Setup new filename. User can add additional text.
        if sometext:
            pieces = self.name_com.split('.')
            pieces.insert(-1, sometext)
            self.name_com = '.'.join(pieces)
        # Even if the command file already exists, we still need to
        # determine these indices.
        self._index_output_mae = []
        self._index_output_mmo = []
        com_opts = self.get_com_opts()
        debg_opts = self.get_debg_opts(com_opts)
        com = '{}\n{}\n'.format(self.filename, self.name_mae)
        # Is this right? It seems to work, but looking back at this,
        # I'm not sure why we wouldn't always want to control using
        # MMOD. Also, that 2nd argument of MMOD only affects the color
        # of atoms. I don't think this needs to be included. At some
        # point, I am going to remove it and test everything to make
        # sure it's not essential.
        if debg_opts:
            com += co.COM_FORM.format(*debg_opts)
        else:
            com += co.COM_FORM.format('MMOD', 0, 1, 0, 0, 0, 0, 0, 0)
        # May want to turn on/off arg2 (continuum solvent).
        com += co.COM_FORM.format('FFLD', 2, 0, 0, 0, 0, 0, 0, 0)
        # Also may want to turn on/off cutoffs using BDCO.
        if com_opts['strs']:
            com += co.COM_FORM.format('BGIN', 0, 0, 0, 0, 0, 0, 0, 0)
        # Look into differences.
        com += co.COM_FORM.format('READ', -1, 0, 0, 0, 0, 0, 0, 0)
        if com_opts['sp'] or com_opts['sp_mmo'] or com_opts['freq']:
            com += co.COM_FORM.format('MINI', 9, 0, 0, 0, 0, 0, 0, 0)
            # self._index_output_mae.append('stupid_extra_structure')
            self._index_output_mae.append('pre')
        if com_opts['sp'] or com_opts['sp_mmo']:
            com += co.COM_FORM.format('ELST', 1, 0, 0, 0, 0, 0, 0, 0)
            self._index_output_mmo.append('pre')
            # Replaced by using a pointless MINI statement. For whatever
            # reason, that causes the .mmo file to be written without
            # needing this WRIT statement.
            # com += co.COM_FORM.format('WRIT', 0, 0, 0, 0, 0, 0, 0, 0)
            # self._index_output_mae.append('pre')
        if com_opts['freq']:
            # Now the WRIT is handled above.
            # com += co.COM_FORM.format('MINI', 9, 0, 0, 0, 0, 0, 0, 0)
            # self._index_output_mae.append('stupid_extra_structure')
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
        # If the file already exists, don't rewrite it.
        path_com = os.path.join(self.directory, self.name_com)
        if sometext and os.path.exists(path_com):
            logger.log(5, '  -- {} already exists. Skipping write.'.format(
                    self.name_com))
        else:
            with open(os.path.join(self.directory, self.name_com), 'w') as f:
                f.write(com)
            logger.log(5, 'WROTE: {}'.format(
                    os.path.join(self.name_com)))

    def run(self, max_fails=5, max_timeout=None, timeout=10, check_tokens=True):
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
        max_fails : int
                    Maximum number of times the job can fail.
        timeout : float
                  Time waited in between lookups of Schrodinger license
                  tokens.
        """
        print("Run " + str(self.filename) + " with commands:" + str(self.commands))
        current_directory = os.getcwd()
        os.chdir(self.directory)
        current_timeout = 0
        current_fails = 0
        licenses_available = False
        if check_tokens is True:
            logger.log(5, "  -- Checking Schrodinger tokens.")
            while True:
                token_string = sp.check_output(
                    '$SCHRODINGER/utilities/licutil -available', shell=True)
                if (sys.version_info > (3, 0)):
                  token_string = token_string.decode("utf-8")
                if 'SUITE' not in token_string:
                    licenses_available = True
                    break
                suite_tokens = co.LIC_SUITE.search(token_string)
                macro_tokens = co.LIC_MACRO.search(token_string)
                #suite_tokens = re.search(co.LIC_SUITE, token_string)
                #macro_tokens = re.search(co.LIC_MACRO, token_string)
                if not suite_tokens or not macro_tokens:
                    raise Exception(
                        'The command "$SCHRODINGER/utilities/licutil '
                        '-available" is not working with the current '
                        'regex in calculate.py.\nOUTPUT:\n{}'.format(
                            token_string))
                suite_tokens = int(suite_tokens.group(1))
                macro_tokens = int(macro_tokens.group(1))
                if suite_tokens > co.MIN_SUITE_TOKENS and \
                        macro_tokens > co.MIN_MACRO_TOKENS:
                    licenses_available = True
                    break
                else:
                    if max_timeout is not None and \
                            current_timeout > max_timeout:
                        pretty_timeout(
                            current_timeout, suite_tokens,
                            macro_tokens, end=True, name_com=self.name_com)
                        raise Exception(
                            "Not enough tokens to run {}. Waited {} seconds "
                            "before giving up.".format(
                                self.name_com, current_timeout))
                    pretty_timeout(current_timeout, suite_tokens, macro_tokens,
                                   name_com=self.name_com)
                    current_timeout += timeout
                    time.sleep(timeout)
        else:
            licenses_available = True
        if licenses_available:
            while True:
                try:
                    logger.log(5, 'RUNNING: {}'.format(self.name_com))
                    sp.check_output(
                        '$SCHRODINGER/bmin -WAIT {}'.format(
                            os.path.splitext(self.name_com)[0]), shell=True)
                    break
                except sp.CalledProcessError:
                    logger.warning('Call to MacroModel failed and I have no '
                                   'idea why!')
                    current_fails += 1
                    if current_fails < max_fails:
                        time.sleep(timeout)
                        continue
                    else:
                        raise
        os.chdir(current_directory)

def pretty_timeout(current_timeout, macro_tokens, suite_tokens, end=False,
                   level=10, name_com=None):
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
        if name_com:
            logger.warning('  -- Waiting on tokens to run {}.'.format(
                    name_com))
        logger.log(level,
                   '--' + ' (s) '.center(8, '-') +
                   '--' + ' {} '.format(co.LABEL_SUITE).center(17, '-') +
                   '--' + ' {} '.format(co.LABEL_MACRO).center(17, '-') +
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
            logger.log(10, 'READING: {}'.format(self.filename))
            with open(self.path, 'r') as f:
                lines = f.read()
            num_atoms = int(re.search('Read\s+(\d+)\s+atoms.', lines).group(1))
            logger.log(5, '  -- Read {} atoms.'.format(num_atoms))

            hessian = np.zeros([num_atoms * 3, num_atoms * 3], dtype=float)
            logger.log(5, '  -- Creating {} Hessian matrix.'.format(hessian.shape))
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
                if section_hessian and start_col and '.' not in word and \
                        word != 'NaN':
                    col_nums.append(int(word))
                    continue
                if section_hessian and start_col and '.' in word or \
                        word == 'NaN':
                    elements.append(float(word))
                    continue
            self._hessian = hessian
            logger.log(5, '  -- Creating {} Hessian matrix.'.format(hessian.shape))
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
            logger.log(10, 'READING: {}'.format(self.filename))
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
                if (sys.version_info > (3, 0)):
                    idx_curr = next(idx_iter)
                else:
                    idx_curr = idx_iter.next()
            except StopIteration:
                idx_iter = iter(indices)
                idx_curr = idx_iter.next()
            if idx_curr == label:
                selected.append((str_num, struct))
        return selected

# This could use some documentation. Looks pretty though.
def geo_from_points(*args):
    x1 = args[0][0]
    y1 = args[0][1]
    z1 = args[0][2]
    x2 = args[1][0]
    y2 = args[1][1]
    z2 = args[1][2]
    if len(args) == 2:
        bond = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        return float(bond)
    x3 = args[2][0]
    y3 = args[2][1]
    z3 = args[2][2]
    if len(args) == 3:
        dist_21 = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        dist_23 = math.sqrt((x2 - x3)**2 + (y2 - y3)**2 + (z2 - z3)**2)
        dist_13 = math.sqrt((x1 - x3)**2 + (y1 - y3)**2 + (z1 - z3)**2)
        angle = math.acos((dist_21**2 + dist_23**2 - dist_13**2) /
                          (2*dist_21*dist_23))
        angle = math.degrees(angle)
        return float(angle)
    x4 = args[3][0]
    y4 = args[3][1]
    z4 = args[3][2]
    if len(args) == 4:
        vect_21 = [x2 - x1, y2 - y1, z2 - z1]
        vect_32 = [x3 - x2, y3 - y2, z3 - z2]
        vect_43 = [x4 - x3, y4 - y3, z4 - z3]
        x_ab = np.cross(vect_21,vect_32)
        x_bc = np.cross(vect_32,vect_43)
        norm_ab = x_ab/(math.sqrt(x_ab[0]**2 + x_ab[1]**2 + x_ab[2]**2))
        norm_bc = x_bc/(math.sqrt(x_bc[0]**2 + x_bc[1]**2 + x_bc[2]**2))
        mag_ab = math.sqrt(norm_ab[0]**2 + norm_ab[1]**2 + norm_ab[2]**2)
        mag_bc = math.sqrt(norm_bc[0]**2 + norm_bc[1]**2 + norm_bc[2]**2)
        angle = math.acos(np.dot(norm_ab, norm_bc)/(mag_ab * mag_bc))
        torsion = angle * (180/math.pi)
        return torsion

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
    def format_coords(self, format='latex', indices_use_charge=None):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.

        latex  - Makes a LaTeX table.
        gauss  - Makes output that matches Gaussian's .com filse.
        jaguar - Just like Gaussian, but include the atom number after the
                 element name in the left column.
        """
        # Formatted for LaTeX.
        if format == 'latex':
            output = ['\\begin{tabular}{l S[table-format=3.6] '
                      'S[table-format=3.6] S[table-format=3.6]}']
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
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
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                if indices_use_charge:
                    if atom.index in indices_use_charge:
                        output.append(
                            ' {0:s}--{1:.5f}{2:>16.6f}{3:16.6f}'
                            '{4:16.6f}'.format(
                                ele, atom.partial_charge, atom.x,
                                atom.y, atom.z))
                    else:
                        output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                                ele, atom.x, atom.y, atom.z))
                else:
                    output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                            ele, atom.x, atom.y, atom.z))
            return output
        # Formatted for Jaguar.
        elif format == 'jaguar':
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                label = '{}{}'.format(ele, atom.index)
                output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                        label, atom.x, atom.y, atom.z))
            return output
    def select_stuff(self, typ, com_match=None):
        """
        A much simpler version of select_data. It would be nice if select_data
        was a wrapper around this function.
        """
        stuff = []
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or \
                    com_match is None:
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
        logger.log(1, '>>> typ: {}'.format(typ))
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or \
                    com_match is None:
                datum = thing.as_data(**kwargs)
                # If it's a torsion we have problems.
                # Have to check whether an angle inside the torsion is near 0 or 180.
                if typ == 'torsions':
                    atom_nums = [datum.atm_1, datum.atm_2, datum.atm_3, datum.atm_4]
                    angle_atoms_1 = [atom_nums[0], atom_nums[1], atom_nums[2]]
                    angle_atoms_2 = [atom_nums[1], atom_nums[2], atom_nums[3]]
                    for angle in self.angles:
                        if set(angle.atom_nums) == set(angle_atoms_1):
                            angle_1 = angle.value
                            break
                    for angle in self.angles:
                        if set(angle.atom_nums) == set(angle_atoms_2):
                            angle_2 = angle.value
                            break
                    try:
                        logger.log(1, '>>> atom_nums: {}'.format(atom_nums))
                        logger.log(1, '>>> angle_1: {} / angle_2: {}'.format(
                                angle_1, angle_2))
                    except UnboundLocalError:
                        logger.error('>>> atom_nums: {}'.format(atom_nums))
                        logger.error(
                            '>>> angle_atoms_1: {}'.format(angle_atoms_1))
                        logger.error(
                            '>>> angle_atoms_2: {}'.format(angle_atoms_2))
                        if 'angle_1' not in locals():
                            logger.error("Can't identify angle_1!")
                        else:
                            logger.error(">>> angle_1: {}".format(angle_1))
                        if 'angle_2' not in locals():
                            logger.error("Can't identify angle_2!")
                        else:
                            logger.error(">>> angle_2: {}".format(angle_2))
                        logger.warning('WARNING: Using torsion anyway!')
                        data.append(datum)
                    if -5. < angle_1 < 5. or 175. < angle_1 < 185. or \
                            -5. < angle_2 < 5. or 175. < angle_2 < 185.:
                        logger.log(
                            1, '>>> angle_1 or angle_2 is too close to 0 or 180!')
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
        for atom in self.atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = self.atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyds.append(atom)
        logger.log(5, '  -- {} aliphatic hydrogen(s).'.format(len(aliph_hyds)))
        return aliph_hyds
    def get_hyds(self):
        """
        Returns the atom numbers of any default MacroModel type hydrogens.

        This should be subclassed into something is MM3* specific.
        """
        hyds = []
        for atom in self.atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    hyds.append(atom)
        logger.log(5, '  -- {} hydrogen(s).'.format(len(hyds)))
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
        for atom in self.atoms:
            if atom.is_dummy:
                logger.log(
                    10,'  -- Identified {} as a dummy atom.'.format(atom))
                dummies.append(atom.index)
        return dummies

class Atom(object):
    """
    Data class for a single atom.

    Really, some of this atom type stuff should perhaps be in a MM3*
    specific atom class.
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
    def __repr__(self):
            return '{}[{},{},{}]'.format(
                self.atom_type_name, self.x, self.y, self.z)
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
            self._element = co.MASSES.items()[self.atomic_num - 1][0]
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
    # I have no idea if these atom types are actually correct.
    # Really, the user should specify custom atom types, such as dummies, in a
    # configuration file somewhere.
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
        if self.atom_type_name == 'Du' or \
                self.element == 'X' or \
                self.atomic_num == -2:
            return True
        else:
            return False

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
        # Sort of silly to have all this stuff about angles and
        # torsions in here, but they both inherit from this class.
        # I suppose it'd make more sense to create a structural
        # element class that these all inherit from.
        # Warning that I recently changed these labels, and that
        # may have consequences.
        if self.__class__.__name__.lower() == 'bond':
            typ = 'b'
        elif self.__class__.__name__.lower() == 'angle':
            typ = 'a'
        elif self.__class__.__name__.lower() == 'torsion':
            typ = 't'
        datum = datatypes.Datum(val=self.value, typ=typ,ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, 'atm_{}'.format(i+1), atom_num)
        if (sys.version_info > (3, 0)):
            kwargs_iter = iter(kwargs.items())
        else:
            kwargs_iter = kwargs.iteritems()
        for k, v in kwargs_iter:
            setattr(datum, k, v)
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

def return_filetypes_parser():
    """
    Returns an argument parser for filetypes module.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        '-i', '--input', type=str,
        help='Input filename.')
    parser.add_argument(
        '-o', '--output', type=str,
        help='Output filename.')
    parser.add_argument(
        '-p', '--print', action='store_true',
        help='Print coordinates for each structure.')
    parser.add_argument(
        '-n', '--num', type=int,
        help='Number of structures to display.')
    return parser

def detect_filetype(filename):
    path = os.path.abspath(filename)
    ext = os.path.splitext(path)[1]
    if ext == '.mae' or ext =='.maegz':
        file_ob = Mae(path)
    elif ext == '.log':
        file_ob = GaussLog(path)
        file_ob.read_out()
        # try:
        #     file_ob.read_archive()
        # except IndexError:
        #     pass
    elif ext == '.in':
        file_ob = JaguarIn(path)
    elif ext == '.out':
        file_ob = JaguarOut(path)
    else:
        raise Exception('Filetype not recognized.')
    return file_ob

def main(args):
    parser = return_filetypes_parser()
    opts = parser.parse_args(args)
    file_ob = detect_filetype(opts.input)
    if opts.print:
        if hasattr(file_ob, 'structures'):
            for i, structure in enumerate(file_ob.structures):
                print(' ' + ' STRUCTURE {} '.format(i + 1).center(56, '-'))
                output = structure.format_coords(format='gauss')
                for line in output:
                    print(line)
                if opts.num and i+1 == opts.num:
                    break
        if hasattr(file_ob, 'evals') and file_ob.evals:
            print('EIGENVALUES:')
            print(file_ob.evals)
        if hasattr(file_ob, 'evecs') and file_ob.evecs:
            print('EIGENVECTORS:')
            print(file_ob.evecs)
    if opts.output:
        file_ob.write(opts.output)

if __name__ == '__main__':
    import argparse
    import sys

    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
