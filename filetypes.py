#!/usr/bin/python
'''
Brief description of contents:
------------------------------
CachedProperty - Custom property class for accessing data that
  initially takes a long time to retrieve.
FileType - Base class for any file type.
FileType.atom_data - Generic and convenient means for extracting data
  without having to manually process the raw data.
GaussLogFile - Class responsible for processing data in Gaussian log
  files.
MaeFile - " Schrodinger Maestro (.mae) files.
MaeFile.aliphatic_hydrogens - Figures out which hydrogens are aliphatic.
'''
import collections
import logging
import numpy as np
import os
import re

logger = logging.getLogger(__name__)

class CachedProperty(object):
    '''
    1st call to a class function with this property performs the
    function, which must return an object. The 2nd call to this 
    function retrieves that object from memory without having to
    perform the function again.
    '''
    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__doc__ = doc or func.__doc__
        self.__module__ = func.__module__
        self.func = func
    def __get__(self, inst, owner=None):
        value = inst.__dict__.get(self.__name__)
        if value is None:
            value = self.func(inst)
            inst.__dict__[self.__name__] = value
        return value

class FileType(object):
    def __init__(self, filename, directory=os.getcwd()):
        self.filename = filename
        self.directory = directory
    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, self.filename)
    def atom_data(self, label, atom_label='# First column is atom index #',
                  include_anums=[], exclude_anums=[], calculation_indices=None,
                  calculation_type=None):
        '''
        Generic means of helping you extract data associated with
        individual atoms from a single or multiple structures.
        '''
        data = []
        atom_numbers = []
        if calculation_indices:
            indices_generator = iter(calculation_indices)
        for structure in self.raw_data:
            # Helps with .mae files that have many structures from
            # various types of calculations. You may only want data
            # from one type of calculation.
            use_this_structure = True
            if calculation_type and calculation_indices:
                try:
                    current_calculation_type = indices_generator.next()
                except StopIteration:
                    indices_generator = iter(calculation_indices)
                    current_calculation_type = indices_generator.next()
                if current_calculation_type != calculation_type:
                    use_this_structure = False
            # Only gather data if that check passed.
            if use_this_structure:
                # Check to make sure that the data we are retrieving is
                # a list of values (and also not a string, which is also
                # iterable in Python).
                if isinstance(structure[label], collections.Iterable) and \
                        not isinstance(structure[label], basestring):
                    for datum, atom_number in zip(
                        structure[label], map(int, structure[atom_label])):
                        if not include_anums and not exclude_anums:
                            data.append(datum)
                            atom_numbers.append(atom_number)
                        elif not include_anums and \
                                atom_number not in exclude_anums:
                            data.append(datum)
                            atom_numbers.append(atom_number)
                        elif not exclude_anums and \
                                atom_number in include_anums:
                            data.append(datum)
                            atom_numbers.append(atom_number)
                        elif atom_number in include_anums and \
                                atom_number not in exclude_anums:
                            data.append(datum)
                            atom_numbers.append(atom_number)
                else:
                    data.append(structure[label])
            data = map(float, data)
            atom_numbers = map(int, atom_numbers)
        return data, atom_numbers

class GaussLogFile(FileType):
    '''
    Used to get data from Gaussian .log files.
    '''
    def __init__(self, filename, directory=os.getcwd()):
        FileType.__init__(self, filename, directory)
    @CachedProperty
    def raw_data(self):
        # This dictionary shows everything that is currently
        # extracted from the .log file.
        structure = {'ESP Atom Nums': [],
                     'ESP Elements': [],
                     'ESP Charges': [],
                     'ESP Sum Atom Nums': [],
                     'ESP Sum Elements': [],
                     'ESP Sum Charges': [],
                     'Charge': None,
                     'Multiplicity': None}
        zcoords = False
        esp_section = None
        with open(os.path.join(self.directory, self.filename), 'r') as f:
            for line in f:
                if 'Symbolic Z-matrix' in line:
                    zcoords = True
                if 'Charge =' and 'Multiplicity =' in line:
                    cols = line.split()
                    structure['Charge'] = int(cols[2])
                    structure['Multiplicity'] = int(cols[5])
                # Look for ESP charges. The following numbers indicate
                # the order these things happen as we go through the file
                # line by line.
                # 1st
                if 'ESP charges:' in line:
                    esp_section = 'ESP Charges'
                # 4th
                if 'ESP charges with hydrogens summed into heavy atoms:' \
                        in line:
                    esp_section = 'ESP Charges Summed'
                # 3rd / 6th
                if 'Charge=' in line:
                    esp_section = None
                # 2nd / 5th
                if esp_section is not None:
                    matched = re.match(
                        '\s+(?P<num>[\d]+)\s+(?P<ele>[\A-z]+)' +
                        '\s+(?P<q>[\d\.\-\+\E]+)', line)
                    if matched is not None:
                        if esp_section == 'ESP Charges':
                            structure['ESP Atom Nums'].append(
                                int(matched.group('num')))
                            structure['ESP Elements'].append(
                                matched.group('ele'))
                            structure['ESP Charges'].append(
                                float(matched.group('q')))
                        elif esp_section == 'ESP Charges Summed':
                            structure['ESP Sum Atom Nums'].append(
                                int(matched.group('num')))
                            structure['ESP Sum Elements'].append(
                                matched.group('ele'))
                            structure['ESP Sum Charges'].append(
                                float(matched.group('q')))
        # All filetypes return as lists in case there are multiple
        # structures per file.
        return [structure]

# Work in progress.
class JagInFile(FileType):
    '''
    Extracts data from Jaguar .in files.

    I'm not sure how this works yet with dummy atoms.
    '''
    def __init__(self, filename, directory=os.getcwd()):
        FileType.__init__(self, filename, directory)
    @CachedProperty
    def raw_data(self):
        raw_data = {'Atoms': [],
                    'Hessian': None}
        with open(os.path.join(self.directory, self.filename), 'r') as f:
            zmat = False
            hess = False
            for i, line in enumerate(f):
                if zmat is True and line.startswith('&'):
                    zmat = False
                    # We know # of atoms now, so we can make an array
                    # to hold the lower triangle of the Hessian.
                    raw_data['Hessian'] = np.zeros(
                        (len(raw_data['Atoms']) * 3,
                         len(raw_data['Atoms']) * 3), dtype=float)
                if zmat is True:
                    cols = line.split()
                    raw_data['Atoms'].append(cols[0])
                if '&zmat' in line:
                    zmat = True
                if hess is True and line.startswith('&'):
                    hess = False
                if hess is True:
                    cols = line.split()
                    if len(cols) == 1:
                        col = int(cols[0])
                    elif len(cols) > 1:
                        row = int(cols[0])
                        for i, ele in enumerate(cols[1:]):
                            raw_data['Hessian'][row - 1, i + col - 1] = \
                                float(ele)
                if '&hess' in line:
                    hess = True
        return raw_data
                
class MaeFile(FileType):
    '''
    Extracts data from .mae files.
    '''
    def __init__(self, filename, directory=os.getcwd()):
        FileType.__init__(self, filename, directory)
    def aliphatic_hydrogens(self):
        '''
        Returns the atom numbers of aliphatic hydrogens. This assumes
        that the atom numbers of the aliphatic hydrogens don't change
        between structures in the .mae file, as it only checks the 1st
        structure.
        '''
        atom_numbers = []
        for atom_number, atom_type in zip(
            map(int, self.raw_data[0]['# First column is atom index #']),
            map(int, self.raw_data[0]['i_m_mmod_type'])):
            # This range includes the various hydrogen types.
            if atom_type > 40 and atom_type < 49:
                for bond_from, bond_to in zip(
                    map(int, self.raw_data[0]['i_m_from']),
                    map(int, self.raw_data[0]['i_m_to'])):
                    # Loop through bonds. Check if the bond is connected
                    # to the hydrogen in question. If that connection is
                    # to a tetrahedral carbon (that's a 3 for
                    # atom_type), then that hydrogen is aliphatic.
                    if bond_from == atom_number:
                        if int(self.raw_data[0]['i_m_mmod_type'][bond_to - 1]) \
                                == 3:
                            atom_numbers.append(atom_number)
                            break
                    elif bond_to == atom_number:
                        if int(self.raw_data[0]['i_m_mmod_type'][bond_from -1]) \
                                == 3:
                            atom_numbers.append(atom_number)
                            break
        return atom_numbers
    @CachedProperty
    def raw_data(self):
        '''
        Holds the raw data from the .mae. Returns a list of
        dictionaries, one for each structure in the .mae file.
        '''
        structures = []
        # These are used to distinguish parts of the .mae file.
        horizontal_headers = ['m_depend[', 'm_atom[', 'm_bond']
        vertical_headers = ['f_m_ct {']
        breaks = [':::', '}'] # Strings that indicate a break.
        consecutive_breaks = 0
        section_type = None # 'horizontal' or 'vertical'
        label_section = False
        # Initiate some variables that will hold temporary data.
        labels = []
        values = []
        # Read data from file.
        with open(os.path.join(self.directory, self.filename), 'r') as f:
            for line in f:
                # Start of a structure. Give it a fresh dictionary.
                if 'f_m_ct {' in line:
                    structures.append({})
                # Count how many consecutive lines had break
                # characters.
                if any(x in line for x in breaks):
                    consecutive_breaks += 1
                else:
                    consecutive_breaks = 0
                # Conditions to terminate a section of the .mae file.
                if any(x in line for x in vertical_headers + \
                           horizontal_headers) or consecutive_breaks > 2:
                    # 2 methods here for dealing with data stored in the
                    # horizontal vs. vertical format.
                    if section_type == 'vertical':
                        assert len(labels) == len(values), \
                            'Length of labels and values are not equal in ' + \
                            'vertical data section of structure ' + \
                            '{} in {}.'.format(len(structures), self.filename)
                        for label, value in zip(labels, values):
                            structures[-1].update({label: value})
                        labels = []
                        values = []
                    if section_type == 'horizontal':
                        for label, value in sub_data.iteritems():
                            structures[-1].update({label: value})
                        labels = []
                        values = []
                # Look for the start of the vertical layout.
                if any(x in line for x in vertical_headers):
                    section_type = 'vertical'
                    label_section = True
                # Look for the start of the horizontal layout.
                if any(x in line for x in horizontal_headers):
                    section_type = 'horizontal'
                    label_section = True
                    sub_data = {}
                # Find labels and data in the vertical format.
                if section_type == 'vertical':
                    # Marks the end of the labels.
                    if ':::' in line:
                        label_section = False
                    # Gather labels.
                    if not any(x in line for x in breaks + vertical_headers) \
                            and label_section:
                        labels.append(line.strip())
                    # Gather data.
                    if not any(x in line for x in breaks) and not label_section:
                        values.append(line.strip())
                # Find labels and data in the horizontal format.
                if section_type == 'horizontal':
                    if ':::' in line and sub_data == {}:
                        label_section = False
                        for label in labels:
                            sub_data.update({label: []})
                    if not any(x in line for x in breaks + horizontal_headers) \
                            and label_section:
                        labels.append(line.strip())
                    if not any(x in line for x in breaks) and not label_section:
                        cols = line.split()
                        # It gets tricky to figure out what the columns 
                        # of data represent because some values are " ".
                        # split() results in this consuming 2 columns.
                        # Here's a work around.
                        errors_exist = True
                        while errors_exist:
                            errors = 0
                            new_cols = []
                            for i, col in enumerate(cols):
                                if col.count('"') != 1:
                                    new_cols.append(col)
                                else:
                                    errors += 1
                                    new_cols.append(cols[i] + cols[i+1])
                                    new_cols.extend(cols[i+2:])
                                    cols = new_cols
                                    break
                            if errors == 0:
                                errors_exist = False
                        for i, col in enumerate(cols):
                            sub_data[labels[i]].append(col)
        return structures
        
