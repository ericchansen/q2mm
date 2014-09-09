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
import yaml

logger = logging.getLogger(__name__)

re_float = '[+-]? * (?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
re_substr_name = '[\w\s\-\=\(\)\[\]]+?(?=\s+\d+[\n\r])'
# Used for mmo files.
re_bond = re.compile(
    '\s+(\d+)\s+(\d+)\s+{}\s+{}\s+({})\s+{}'.format(
        re_float, re_float, re_float, re_float) +
    '\s+\w+\s+\d+\s+({})\s+(\d+)'.format(re_substr_name))
re_angle = re.compile(
    '\s+(\d+)\s+(\d+)\s+(\d+)' +
    '\s+{}\s+{}\s+{}\s+({})\s+{}\s+{}'.format(
        re_float, re_float, re_float, re_float, re_float, re_float) +
    '\s+\w+\s+\d+\s+({})\s+(\d+)'.format(re_substr_name))
re_tors = re.compile(
    '\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)' +
    '\s+{}\s+{}\s+{}\s+({})\s+{}'.format(
        re_float, re_float, re_float, re_float, re_float) +
    '\s+\w+\s+\d+\s+({})\s+(\d+)'.format(re_substr_name))

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
    def get_data(self, label, atom_label='# First column is atom index #',
                 include_inds=[], exclude_inds=[], calc_indices=None,
                 calc_type=None, substr=None, comment_label=None,
                 scan_inds=None):
        '''
        Generic means of helping you extract data associated with
        individual atoms from a single or multiple structures. This has
        gotten a bit big for my taste, and should probably be broken
        into smaller functions.
        '''
        logger.debug('{} structure(s) in {}.'.format(
                len(self.raw_data), self.filename))
        if calc_type is not None and calc_indices is not None:
            logger.debug("Looking for '{}' in {}.".format(
                    calc_type, calc_indices))
        # These will be lists of lists. Each outer list corresponds
        # to a structure in the file. The inner list is that structures
        # data or associated atom numbers.
        all_data = []
        all_inds = []
        if calc_indices:
            indices_generator = iter(calc_indices)
        for structure in self.raw_data:
            # Use to select certain structures from the file that
            # pertain to the data you want to use.
            use_this_structure = True
            if calc_type and calc_indices:
                try:
                    current_calc_type = indices_generator.next()
                except StopIteration:
                    indices_generator = iter(calc_indices)
                    current_calc_type = indices_generator.next()
                if current_calc_type != calc_type:
                    use_this_structure = False
            # Only gather data if that check passed.
            if use_this_structure:
                data = []
                inds = []
                # Check to make sure that the data we are retrieving is
                # a list of values (also check to ensure it's not a string,
                # which is also iterable in Python).
                if isinstance(structure[label], collections.Iterable) and \
                        not isinstance(structure[label], basestring):
                    if substr and comment_label:
                        for datum, atom_number, com in zip(
                            structure[label], structure[atom_label],
                            structure[comment_label]):
                            keep = False
                            if not include_inds and not exclude_inds:
                                keep = True
                            elif not include_inds and \
                                    int(atom_number) not in exclude_inds:
                                keep = True
                            elif not exclude_inds and \
                                    int(atom_number) in include_inds:
                                keep = True
                            elif int(atom_number) in include_inds and \
                                    int(atom_number) not in exclude_inds:
                                keep = True
                            if keep is True and comment_label and \
                                    not substr in com:
                                keep = False
                            if keep is True:
                                data.append(datum)
                                inds.append(atom_number)
                    else:
                        # If so, make sure the associated atom numbers are what
                        # you selected using the exclude_inds and include_inds
                        # lists.
                        for datum, atom_number in zip(
                            structure[label], structure[atom_label]):
                            if not include_inds and not exclude_inds:
                                data.append(datum)
                                inds.append(atom_number)
                            elif not include_inds and \
                                    int(atom_number) not in exclude_inds:
                                data.append(datum)
                                inds.append(atom_number)
                            elif not exclude_inds and \
                                    int(atom_number) in include_inds:
                                data.append(datum)
                                inds.append(atom_number)
                            elif int(atom_number) in include_inds and \
                                    int(atom_number) not in exclude_inds:
                                data.append(datum)
                                inds.append(atom_number)
                else:
                    # Just append the data. Don't worry about inds because
                    # you won't use the atom numbers anyway.
                    data.append(structure[label])
                    if scan_inds:
                        inds = []
                        for scan_ind in scan_inds:
                            inds.append(structure[scan_ind])
                all_data.append(data)
                all_inds.append(inds)
        logger.debug('Used {} structure(s) in {}.'.format(
                len(all_data), self.filename))
        return all_data, all_inds
    def get_inv_hess(self, hess=None, replace_value=1):
        if hess is None:
            hess = self.raw_data['Hessian']
        e_val, e_vec = np.linalg.eigh(hess)
        minimum = np.min(e_val)
        logging.debug('Minimum eigen value: {}'.format(minimum))
        assert minimum < 0, 'Minimum eigen value is not negative.'
        min_index = np.where(e_val == minimum)
        e_val[min_index[0][0]] = replace_value
        logger.debug('Set minimum eigen value to {}.'.format(replace_value))
        # Is this the most efficient way to do this?
        return e_vec.dot(np.diag(e_val)).dot(np.linalg.inv(e_vec))
    def get_hess_tril_array(self, hess=None):
        if hess is None:
            hess = self.raw_data['Hessian']
        indices = np.tril_indices_from(hess)
        return hess[indices], indices
    def get_mass_weight_hess(
        self, atom_types, hess=None, mass_yaml='options/masses.yaml'):
        if hess is None:
            hess = self.raw_data['Hessian']
        with open(mass_yaml, 'r') as f:
            masses = yaml.load(f)
        scale_facs = []
        for atom in atom_types:
            scale_facs.extend([1/np.sqrt(masses[atom])] * 3)
        x, y = hess.shape
        for i in range(0, x):
            for j in range(0, y):
                hess[i, j] = hess[i, j] * scale_facs[i] * scale_facs[j]
        return hess

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
                    raw_data['Hessian'] += np.tril(raw_data['Hessian'], -1).T
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
    def get_aliph_hyds(self):
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
        logger.debug('Aliphatic hydrogens in {}: {}'.format(
                self.filename, atom_numbers))
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
        
class MMoFile(FileType):
    '''
    Extracts data from .mmo files.

    Getting raw_data seems to take significantly longer than for
    MaeFile. Maybe we can eliminate the use of re, and that might
    speed it up a bit.
    '''
    def __init__(self, filename, directory=os.getcwd(), substr_name='OPT'):
        FileType.__init__(self, filename, directory)
    @CachedProperty
    def raw_data(self):
        structures = []
        with open(os.path.join(self.directory, self.filename), 'r') as f:
            # Need both to keep track of where we are in the file.
            # These are used to keep track of which structure we're on
            # if there are multiple.
            file_count = 0
            structure_count = 0
            previous = 0
            count = 0
            # This keeps track of what data we're looking at for a
            # given structure.
            section = None
            for line in f:
                # Marks the start of a new strucure.
                if 'Input filename' in line:
                    file_count += 1
                if 'Input Structure Name' in line:
                    structure_count += 1
                previous = count
                count = max(file_count, structure_count)
                # If so, then we are actually at a new structure, not
                # just a division in the file.
                if count != previous:
                    # Add the last structure's data.
                    if count != 1:
                        structures.append(data)
                    # Start a new dictionary.
                    data = {'B. Nums.': [], # Bonds
                            'B.':       [],
                            'B. Com.':  [],
                            'A. Nums.': [], # Angles
                            'A.':       [],
                            'A. Com.':  [],
                            'T. Nums.': [], # Torisons
                            'T.':       [],
                            'T. Com.':  []}
                if 'BOND LENGTHS AND STRETCH ENERGIES' in line:
                    section = 'bond'
                if 'ANGLES, BEND AND STRETCH BEND ENERGIES' in line:
                    section = 'angle'
                if 'BEND-BEND ANGLES AND ENERGIES' in line:
                    section = 'bend-bend'
                if 'DIHEDRAL ANGLES AND TORSIONAL ENERGIES' in line:
                    section = 'torsion'
                if 'Improper torsion interactions present' in line:
                    section = 'improper'
                if 'DIHEDRAL ANGLES AND TORSIONAL CROSS-TERMS' in line:
                    'cross-terms'
                if 'NONBONDED DISTANCES AND ENERGIES' in line:
                    'nonbonded'
                if section == 'bond':
                    m = re_bond.match(line)
                    if m is not None:
                        data['B. Nums.'].append(
                            map(int, (m.group(1), m.group(2))))
                        data['B.'].append(float(m.group(3)))
                        data['B. Com.'].append(m.group(4))
                if section == 'angle':
                    m = re_angle.match(line)
                    if m is not None:
                        data['A. Nums.'].append(
                            map(int, (m.group(1), m.group(2), m.group(3))))
                        data['A.'].append(float(m.group(4)))
                        data['A. Com.'].append(m.group(5))
                if section == 'torsion':
                    m = re_tors.match(line)
                    if m is not None:
                        data['T. Nums.'].append(
                            map(int, (m.group(1), m.group(2), m.group(3),
                                      m.group(4))))
                        data['T.'].append(float(m.group(5)))
                        data['T. Com.'].append(m.group(6))
            structures.append(data)
        return structures

class MMoLogFile(FileType):
    '''
    Extracts data from .log files of MacroModel calculations.
    '''
    def __init__(self, filename, directory=os.getcwd(), substr_name='OPT'):
        FileType.__init__(self, filename, directory)
    @CachedProperty
    def raw_data(self):
        logger.debug('Importing: {}'.format(self.filename))
        raw_data = {'Hessian': None}
        with open(os.path.join(self.directory, self.filename), 'r') as f:
            lines = f.read()
        num_atoms = int(re.search('Read\s+(\d+)\s+atoms.', lines).group(1))
        logger.debug('Read {} atoms in {}.'.format(num_atoms, self.filename))
        # Make appropriately sized matrix.
        raw_data['Hessian'] = np.zeros(
            [num_atoms * 3, num_atoms * 3], dtype=float)
        logger.debug('Creating a {} x {} Hessian matrix.'.format(
                num_atoms * 3, num_atoms * 3))
        split_lines = lines.split()
        hessian_section = False
        start_row = False
        start_col = False
        for i, word in enumerate(split_lines):
            # 1. Start of Hessian.
            if word == 'Mass-weighted':
                hessian_section = True
                continue
            # 5. End of Hessian. Add last row of Hessian and break.
            if word == 'Eigenvalues:':
                for col_num, element in zip(col_nums, elements):
                    raw_data['Hessian'][row_num - 1, col_num - 1] = element
                hessian_section = False
                break
            # 4. End of a Hessian row. Add to matrix and reset.
            if hessian_section and start_col and word == 'Element':
                for col_num, element in zip(col_nums, elements):
                    raw_data['Hessian'][row_num - 1, col_num - 1] = element
                start_col = False
                start_row = True
                row_num = int(split_lines[i + 1])
                col_nums = []
                elements = []
                continue
            # 2. Start of a Hessian row.
            if hessian_section and word == 'Element':
                row_num = int(split_lines[i + 1])
                col_nums = []
                elements = []
                start_row = True
                continue
            # 3. Okay, made it through the row number. Now look for columns
            #    and elements.
            if hessian_section and start_row and word == ':':
                start_row = False
                start_col = True
                continue
            if hessian_section and start_col and '.' not in word:
                col_nums.append(int(word))
                continue
            if hessian_section and start_col and '.' in word:
                elements.append(float(word))
                continue
        return raw_data
