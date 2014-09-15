#!/usr/bin/python
# The export and import methods (or at least the export) should be
# incorporated into the force field class. That way the export method
# can be called without knowledge of what type of force field is in use.
# Of course a unique export function will be defined for each force 
# field type.
'''
BaseParam - Base class for all parameters.
MM3Param - Class for MM3* parameters.
BaseFF - Base class for all force fields. Contains a list of parameters
  amongst other things.
BaseFF.import_step_sizes - Imports step sizes for parameters from a YAML
  file.
MM3FF - Class for MM3* force fields.
MM3FF.atom_types - Converts the SMILES-esque formula into a list of atom
  types.
MM3FF.spawn_child_params - Makes a copy of the FF's parameters.
MM3FF.spawn_child_ff - Makes a near copy of the class. Saves some
  variables that define how to write the parameters to the force field,
  but resets data and penalty function value to None. Can optionally
  modify parameters as well.
MM3FF.remove_other_params - Takes a list of parameters as an argument.
  Removes any parameters from self.params that are not in that list
  (determined by the row and column in the MM3* force field).
MM3FF.validate_params - Ensure that all parameters in the list are
  in self.params.
import_mm3_ff - Returns an instance of MM3FF.
export_mm3_ff - Creates a parameter file in YAML format. Can also export
  the entire substructure in YAML format.
get_mm3_atom_types - Uses a list of atom labels, which are sometimes
  only integers, to create a list of atom types, which can be found in
  the atom types file, atom.typ.
'''
import argparse
import calculate
import copy
import logging
import logging.config
import os
import re
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

# Setup logging.
logger = logging.getLogger(__name__)

re_substr_name = '[\w\s]+'
re_smiles = '[\w\-\=\(\)\.\+\[\]\*]+'
re_atom_types_split = '[\s\-\(\)\=\.\[\]\*]+'

class BaseParam(object):
    '''
    Base class for each parameter. Assists with input and output from
    force field formats.

    value      = The floating point parameter value.
    param_type = An arbitrary string defining the parameter type. See
                 the list of possible values below.
    atom_types = The atom types used by the parameter.
    step_size  = How far to step the parameter for numerical
                 differentiation.
    der1       = 1st derivative with respect to the penalty function as
                 determined by central differentiation.
    der2       = 2nd derivative " " "

    Parameter types:
    af         = Angle/bend force constant
    bf         = Bond/stretch force constant
    df         = Dihedral/torsional force constant
    ea         = Equilibrium angle
    eb         = Equilibrium bond length
    imp1       = 1st improper torsion parameter
    imp2       = 2nd improper torsion parameter
    sb         = Stretch-bend force constant
    q          = Dipole constant for a bond
    '''
    # Does anyone know a way to make the instance variables rigidly
    # defined like this, but somehow cut down on all the space?
    def __init__(self, value=None, param_type=None, atom_types=None,
                 step_size=None, der1=None, der2=None):
        self.value = value
        self.param_type = param_type
        self.atom_types = atom_types
        self.step_size = step_size
        self.der1 = der1
        self.der2 = der2
    def __repr__(self):
        return '{}({} : {} : {})'.format(
            self.__class__.__name__, self.param_type,
            ' '.join(self.atom_types), self.value)
        
class MM3Param(BaseParam):
    '''
    mm3_row     = The row/line of the force field containing the 
                  parameter. The 1st line of the force field is 1 (not
                  0).
    mm3_col     = Parameter values (equilibrium values, force constants,
                  etc.) are specified to be in column 1, 2, or 3. See
                  the MacroModel reference manual for more details on
                  the format.
    mm3_label   = The 1st two characters of the line containing the
                  parameter. MacroModel uses these 2 characters to 
                  determine the type of the parameter.
    atom_labels = A list of the atom labels, found in the force field
                  directly after mm3_label. These can be integers or
                  MM3* atom types, such as H3, N2, etc. This is used to
                  determine the attribute atom_types of the class
                  BaseParam.
    substr_name = The name of the substructure containing the 
                  parameters.
    '''
    def __init__(self, value=None, param_type=None, step_size=None,
                 der1=None, der2=None, atom_types=None, mm3_row=None,
                 mm3_col=None, mm3_label=None, atom_labels=None,
                 substr_name=None):
        # I'm not sure why this doesn't work (assuming I included all
        # the necessary arguments.
        # super(BaseParam, self).__init__(value)
        # Anyway, this works.
        BaseParam.__init__(self, value, param_type, atom_types, step_size,
                           der1, der2)
        self.mm3_row = mm3_row
        self.mm3_col = mm3_col
        self.mm3_label = mm3_label
        self.atom_labels = atom_labels
        self.substr_name = substr_name

class BaseFF(object):
    '''
    filename   = Relative path to FF file.
    params     = A list of parameter objects (BaseParam, MM3Param, etc.).
    data       = A list of Datum objects (defined in
                 calculate.py).
    x2         = Value of the objective function determined
                 from comparing self.data to reference data
                 points (another list of Datum objects).
    gen_method = Method used to generate this parameter set.
    '''
    def __init__(self, filename, params=None, data=None, x2=None,
                 gen_method=None, cent_ffs=None, for_ffs=None):
        self.filename = filename
        if params is None:
            params = []
        self.params = params
        if data is None:
            data = []
        self.data = data
        self.x2 = x2
        self.gen_method = gen_method
        # Still deciding on whether or not I want these to be
        # instance attributes of this class or a separate dictionary
        # somewhere. If they stay like this, then functions to
        # create the residual vector and Jacobian should also be
        # integrated into this class.
        if cent_ffs is None:
            cent_ffs = []
        self.cent_ffs = cent_ffs
        if for_ffs is None:
            for_ffs = []
        self.for_ffs = for_ffs
    def calculate_x2(self, ref_data):
        x2 = 0.
        # Although the calculated energies are on their own relative
        # scale, the reference's minimum should be used for the
        # calculated data's minimum.
        r_energies = sorted([x for x in ref_data if x.data_type == 'Energy'],
                            key=calculate.sort_datum)
        c_energies = sorted([x for x in self.data if x.data_type == 'Energy'],
                            key=calculate.sort_datum)
        if r_energies and c_energies:
            assert len(r_energies) == len(c_energies), \
                "Number of reference and calculated energies don't match."
            r_zero = min([x.value for x in r_energies])
            r_zero_i = [x.value for x in r_energies].index(r_zero)
            c_zero = c_energies[r_zero_i].value
            for e in c_energies:
                e.value -= c_zero
        for r_datum, c_datum in zip(
            sorted(ref_data, key=calculate.sort_datum),
            sorted(self.data, key=calculate.sort_datum)):
            x2 += r_datum.weight**2 * (r_datum.value - c_datum.value)**2
        self.x2 = x2
    def import_step_sizes(self, filename='options/steps.yaml'):
        assert os.path.exists(filename), \
            "File for parameter steps, {}, doesn't exist.".format(filename)
        with open(filename, 'r') as f:
            steps = yaml.load(f) # Dictionary matching parameter type to
                                 # step size.
            logger.debug('Loaded step sizes from {}.'.format(
                    filename))
        for param in self.params:
            param.step_size = steps[param.param_type]
            logger.debug('Set step size of {} to {}.'.format(
                    param, steps[param.param_type]))

class MM3FF(BaseFF):
    def __init__(self, filename, params=None, data=None, x2=None,
                 gen_method=None, cent_ffs=None, for_ffs=None,
                 substr_name=None, starting_row=None, ending_row=None,
                 smiles=None):
        BaseFF.__init__(self, filename, params, data, x2, gen_method, cent_ffs,
                        for_ffs)
        self.substr_name = substr_name
        self.starting_row = starting_row
        self.ending_row = ending_row
        self.smiles = smiles
    @property
    def atom_types(self):
        '''
        Use the SMILES-esque substructure definition (directly below
        the substructure name in mm3.fld) to determine the atom types.
        '''
        # I hope I included all the seperators.
        atom_types = re.split(re_atom_types_split, self.smiles)
        if '' in atom_types:
            atom_types.remove('')
        return atom_types
    def calculate_data(self, calc_args, backup=True):
        '''
        calc_args = String or list. Arguments for calculate.py.
        backup    = If True, backup existing force field and replace it
                    when finished doing calculations.
        '''
        if backup:
            logger.debug('Backing up existing FF in memory.')
            orig_ff = import_mm3_ff(filename=self.filename,
                                    substr_name=self.substr_name)
            orig_ff.remove_other_params(self.params)
        export_mm3_ff(self.params, in_filename=self.filename,
                      out_filename=self.filename)
        if isinstance(calc_args, list):
            self.data = calculate.process_args(calc_args)
        elif isinstance(calc_args, basestring):
            self.data = calculate.process_args(calc_args.split())
        else:
            raise Exception(
                "Can't recognize arguments for calculate.py " +
                "{}: {}".format(type(calc_args), calc_args))
        if backup:
            logger.debug('Restoring backup FF from memory.')
            export_mm3_ff(params=orig_ff.params, in_filename=orig_ff.filename,
                          out_filename=orig_ff.filename)
    def spawn_child_params(self, p_changes=None):
        '''
        Create a copy of the FF's parameters. Increment them if changes
        are provided. Some values are not copied intentionally.
        '''
        if p_changes is None:
            new_params = copy.deepcopy(self.params)
        else:
            assert len(p_changes) == len(self.params), \
                'Number of parameters to change and in FF are not ' + \
                'equal.'
            new_params = [MM3Param(
                    value=p.value + x, param_type=p.param_type,
                    atom_types=p.atom_types, mm3_row=p.mm3_row,
                    mm3_col=p.mm3_col, mm3_label=p.mm3_label,
                    atom_labels=p.atom_labels, substr_name=p.substr_name)
                          for p, x in zip(self.params, p_changes)]
        return new_params
    def spawn_child_ff(self, p_changes=None, gen_method=None):
        '''
        Create a copy of the FF. Some values are not copied
        intentionally.
        '''
        # I can't stress enough how important it is that this does a
        # deep copy.
        new_params = self.spawn_child_params(p_changes=p_changes)
        return MM3FF(
            self.filename, params=new_params, gen_method=gen_method,
            substr_name=self.substr_name, starting_row=self.starting_row,
            ending_row=self.ending_row, smiles=self.smiles)
    def remove_other_params(self, other_params):
        '''
        Remove all parameters from self.params that are not in
        the list. Checks based upon mm3_row and mm3_col.

        other_params = List of MM3Param objects.
        '''
        new_params = []
        for other_param in other_params:
            for param in self.params:
                if other_param.mm3_row == param.mm3_row and \
                        other_param.mm3_col == param.mm3_col:
                    new_params.append(param)
        logger.debug('Trimmed number of parameters from {} to {}.'.format(
                len(self.params), len(new_params)))
        self.params = new_params
    def validate_params(self, other_params):
        '''
        Returns True if all parameters are in self.params, else returns
        False.

        other_params = List of MM3Param objects.
        '''
        for other_param in other_params:
            if other_param.mm3_row not in [x.mm3_row for x in self.params] or \
                    other_param.mm3_col not in [x.mm3_col for x in self.params]:
                # break
                return False
        # else:
        #     return True
        return True
                    
def import_mm3_ff(filename='mm3.fld', substr_name='OPT'):
    '''
    filename    = Filename of MM3* force field including relative
                  directory.
    substr_name = Gathers parameters from the substructure with this
                  string in its name. By default, this is set to OPT,
                  so it will pick up any substructure with OPT in its
                  name.
    '''
    atom_str = '\w+' # What atom names can be.
    value_str = '[\d.-]+' # What values can be.
    with open(filename, 'r') as f:
        logger.debug('Loading force field from {}.'.format(filename))
        found_substr = False
        substr_section = False
        for i, line in enumerate(f):
            # Look for the substructure.
            if not substr_section and substr_name in line:
                matched = re.match('\sC\s+({})'.format(re_substr_name), line)
                if matched != None:
                    found_substr = True
                    substr_section = True
                    # Make the substructure/FF object.
                    substr = MM3FF(
                        filename, substr_name=matched.group(1).strip(),
                        starting_row=i+1)
                    logger.debug('Located "{}" on line {} of {}.'.format(
                            substr.substr_name, substr.starting_row,
                            substr.filename))
                   # Get the chemical formula from the next line.
                    matched = re.match('\s9\s+({})'.format(re_smiles),
                                       next(f))
                    try:
                        substr.smiles = matched.group(1)
                    except:
                        logger.exception("Couldn't read substructure name " +
                                         "on line {} of {}".format(
                                i+2, filename))
                        raise
                    logger.debug('Chemical formula: {}'.format(substr.smiles))
                    logger.debug('Atom types: {}'.format(substr.atom_types))
            # Look for the end of the substructure.
            if substr_section and line.startswith('-3'):
                logger.debug('{} ended on line {}.'.format(
                        substr.substr_name, i))
                substr_section = False
            # Look for stretches.
            if substr_section and line.startswith(' 1'):
                cols = line.split()
                substr.params.extend((
                        MM3Param(value = float(cols[3]),
                                 param_type = 'eb',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:3], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 1,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:3],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[4]),
                                 param_type = 'bf',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:3], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 2,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:3],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[5]),
                                 param_type = 'q',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:3], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 3,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:3],
                                 substr_name = substr.substr_name)
                        ))
            # Look for angles.
            if substr_section and line.startswith(' 2'):
                cols = line.split()
                substr.params.extend((
                        MM3Param(value = float(cols[4]),
                                 param_type = 'ea',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:4], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 1,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:4],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[5]),
                                 param_type = 'af',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:4], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 2,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:4],
                                 substr_name = substr.substr_name)
                        ))
            # Look for stretch-bends.
            if substr_section and line.startswith(' 3'):
                cols = line.split()
                substr.params.append(
                    MM3Param(value = float(cols[4]),
                             param_type = 'sb',
                             atom_types = get_mm3_atom_types(
                                     cols[1:4], substr.atom_types),
                             mm3_row = i+2,
                             mm3_col = 1,
                             mm3_label = line[:2],
                             atom_labels = cols[1:4],
                             substr_name = substr.substr_name))
            # Look for torsions.
            if substr_section and line.startswith(' 4'):
                cols = line.split()
                substr.params.extend((
                        MM3Param(value = float(cols[5]),
                                 param_type = 'df',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:5], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 1,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:5],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[6]),
                                 param_type = 'df',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:5], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 2,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:5],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[7]),
                                 param_type = 'df',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:5], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 3,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:5],
                                 substr_name = substr.substr_name)
                        ))
            # Look for higher order torsions.
            if substr_section and line.startswith('54'):
                cols = line.split()
                substr.params.extend((
                        MM3Param(value = float(cols[1]),
                                 param_type = 'df',
                                 atom_types = substr.params[-1].atom_types,
                                 mm3_row = i+2,
                                 mm3_col = 1,
                                 mm3_label = line[:2],
                                 atom_labels = substr.params[-1].atom_labels,
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[2]),
                                 param_type = 'df',
                                 atom_types = substr.params[-1].atom_types,
                                 mm3_row = i+2,
                                 mm3_col = 2,
                                 mm3_label = line[:2],
                                 atom_labels = substr.params[-1].atom_labels,
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[3]),
                                 param_type = 'df',
                                 atom_types = substr.params[-1].atom_types,
                                 mm3_row = i+2,
                                 mm3_col = 3,
                                 mm3_label = line[:2],
                                 atom_labels = substr.params[-1].atom_labels,
                                 substr_name = substr.substr_name)
                        ))
            # Look for improper torsions.
            if substr_section and line.startswith(' 5'):
                cols = line.split()
                substr.params.extend((
                        MM3Param(value = float(cols[5]),
                                 param_type = 'imp1',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:5], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 1,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:5],
                                 substr_name = substr.substr_name),
                        MM3Param(value = float(cols[5]),
                                 param_type = 'imp2',
                                 atom_types = get_mm3_atom_types(
                                         cols[1:5], substr.atom_types),
                                 mm3_row = i+2,
                                 mm3_col = 2,
                                 mm3_label = line[:2],
                                 atom_labels = cols[1:5],
                                 substr_name = substr.substr_name)
                        ))
        # Just in case we couldn't find the right substructure.
        if not found_substr:
            logger.warning('Could not find substructure containing ' + 
                           '{} in {}.'.format(substr_name, filename))
    logger.debug('Located {} parameters in {}.'.format(
            len(substr.params), substr.substr_name))
    return substr

def export_mm3_ff(params, in_filename='mm3.fld', out_filename=None,
                  starting_row=None):
    '''
    Only updates existing parameters. Can't be used to add new
    parameters.
    
    params        = List of MM3Params.
    in_filename   = Filename of MM3* force field used as a template for
                    the new force field. Only the parameters listed in 
                    params are modified.
    out_filename  = Filename of the new MM3* force field. If None, then
                    the new force field is printed to standard output.
    starting_row  = Row of the FF containing the substructure name.
                    Not necessary. Only used to shorten what is printed
                    to standard output when no output filename is given.
    '''
    logger.debug('Reading {} as a template to export parameters.'.format(
            in_filename))
    with open(in_filename, 'r') as f:
        lines = f.readlines()
    for param in params:
        cols = lines[param.mm3_row - 1].split()
        assert param.mm3_label in [' 1', ' 2', ' 3', ' 4', '54', ' 5'], \
            'Unrecognized parameter label "{}".'.format(param.mm3_label)
        # How to write stretches.
        if param.mm3_label == ' 1':
            cols[3:6] = map(float, cols[3:6])
            # This value being negative used to mean that negative 
            # parameter values were allowed, hence the absolute value.
            cols[abs(param.mm3_col) + 2] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>23.4f}{4:>11.4f}{5:>11.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
        # " angles.
        elif param.mm3_label == ' 2':
            cols[4:6] = map(float, cols[4:6])
            cols[abs(param.mm3_col) + 3] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}{5:>11.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
        # " stretch-bends.
        elif param.mm3_label == ' 3':
            cols[abs(param.mm3_col) + 3] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>19.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
        # " torsions.
        elif param.mm3_label == ' 4':
            cols[5:8] = map(float, cols[5:8])
            cols[abs(param.mm3_col) + 4] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}{7:>11.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
        # " higher order torsions.
        elif param.mm3_label == '54':
            cols[1:4] = map(float, cols[1:4])
            cols[abs(param.mm3_col)] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>31.4f}{2:>11.4f}{3:>11.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
        # " improper torsions.
        elif param.mm3_label == ' 5':
            cols[5:7] = map(float, cols[5:7])
            cols[abs(param.mm3_col) + 4] = param.value
            lines[param.mm3_row - 1] = \
                '{0:>2}{1:>4}{2:>4}{3:>4}{4:>4}{5:>15.4f}{6:>11.4f}\n'.format(*cols)
            logger.debug('Inserted {} at row {}, column {}.'.format(
                    param, param.mm3_row, param.mm3_col))
    if out_filename is None:
        logger.debug('No output force field specified. Printing.')
        for i, line in enumerate(lines):
            if starting_row is None:
                print line.strip('\n')
            else:
                if i > starting_row - 10:
                    print line.strip('\n')
    else:
        logger.debug('Wrote {} parameters to {}.'.format(len(params), out_filename))
        with open(out_filename, 'w') as f:
            f.writelines(lines)


def get_mm3_atom_types(param_atom_labels, substr_atom_types):
    return [substr_atom_types[int(x)-1] if x.isdigit()
            else x for x in param_atom_labels]

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Extracts and imports parameters from force ' +
        'fields. Contains classes for parameter objects.')
    param_opts = parser.add_argument_group('Parameter selection')
    parser.add_argument('filename', type=str, help='Path to force field.')
    parser.add_argument(
        '--input', '-i', type=str, nargs='?', metavar='filename',
        const='options/parameters.yaml', help='Imports YAML parameters from ' +
        'filename into the force field. If not optional argument is given, ' +
        'loads from options/parameters.yaml.')
    parser.add_argument(
        '--name', type=str, metavar='"substructure name"', default='OPT',
        help='Searches for this string to identify which substructure in' +
        'the force field to operate on. The default is "OPT".')
    parser.add_argument(
        '--output', '-o', type=str, nargs='?', metavar='filename',
        const='parameters.yaml',
        help='Outputs parameters from the force field using YAML. The ' +
        'default filename is parameters.yaml.')
    parser.add_argument(
        '--print', '-p', action='store_true', help='Print parameters ' +
        'for inspection instead of writing to a file.')
    parser.add_argument('--substr', action='store_true', help='Export the ' +
                        'entire substructure in YAML instead of only the ' +
                        'parameters.')
    param_opts.add_argument(
        '--all', '-a', action='store_true',
        help='Exports all parameters from substructure.')
    param_opts.add_argument(
        '-af', action='store_true',
        help='Exports bend force constants.')
    param_opts.add_argument(
        '-bf', action='store_true',
        help='Exports stretch force constants.')
    param_opts.add_argument(
        '-df', action='store_true',
        help='Exports dihedral force constants.')
    param_opts.add_argument(
        '-ea', action='store_true',
        help='Exports equilibrium angles.')
    param_opts.add_argument(
        '-eb', action='store_true',
        help='Exports equilibrium bond lengths.')
    param_opts.add_argument(
        '-imp1', action='store_true',
        help='Exports 1st improper dihedral parameter.')
    param_opts.add_argument(
        '-imp2', action='store_true',
        help='Exports 2nd improper dihedral parameter.')
    param_opts.add_argument(
        '-sb', action='store_true',
        help='Exports stretch-bend force constants.')
    param_opts.add_argument(
        '-q', action='store_true',
        help='Exports charge parameters.')
    options = vars(parser.parse_args(args))
    # Import parameters from a YAML file into a force field.
    if options['input']:
        logger.debug('Loading parameters from {}.'.format(options['input']))
        with open(options['input'], 'r') as f:
            params = list(yaml.load_all(f))
        logger.debug('Loaded {} parameters.'.format(len(params)))
        export_mm3_ff(params, in_filename=options['filename'],
                      out_filename=options['filename'])
    # Export the selected parameters from a force field in YAML format.
    elif options['output'] or options['print']:
        param_types = [key for key, value in options.iteritems() 
                       if options['all'] or value is True and key in
                       ['af', 'bf', 'df', 'ea', 'eb', 'imp1', 'imp2', 'sb',
                        'q']]
        logger.debug('Data types to extract: {}'.format(
                ', '.join(param_types)))
        substr = import_mm3_ff(filename=options['filename'],
                               substr_name=options['name'])
        selected_params = [x for x in substr.params if x.param_type in
                           param_types]
        logger.debug('{} parameters were selected.'.format(
                len(selected_params)))
        # Print them.
        if options['print']:
            print '\n---------- YAML Parameters ----------\n'
            if options['substr']:
                print yaml.dump(substr)
            else:
                print yaml.dump_all(selected_params)
        # Write them to a file.
        if options['output']:
            with open(options['output'], 'w') as f:
                if options['substr']:
                    yaml.dump(substr, f)
                else:
                    yaml.dump_all(selected_params, f)
            logger.info('Wrote {} parameters from {} to {}.'.format(
                    len(selected_params), options['filename'],
                    options['output']))
        return selected_params
                
if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        config = yaml.load(f)
    logging.config.dictConfig(config)
    # Execute.
    params = process_args(sys.argv[1:])
