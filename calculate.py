#!/usr/bin/python
'''
Marked areas throughout with "# CHANGES" that will require modification
as new data types are added.

Brief description of contents:
------------------------------
Datum - Class for individual data points. When doing math on data points
  in optimizations, definitely extract them from here to save time.
  However, this class certainly helps with sorting and matching data.
sort_datum - A custom sorting function used to match up reference data
  to calculated data. Relies upon the 1st part of the filename matching
  (or nearly matching) to pair calculated FF data with reference data.
calculate_x2 - Takes a set of reference and calculated data as lists,
  which can be out of order, and determines the penalty function value.
process_args - Interprets arguments and handles all the other functions.
make_macromodel_coms - Sets up MacroModel .com files. Their output will
  produce all the requested data.
extract_data - Pulls data from the output files.
'''
import argparse
import filetypes
import logging
import logging.config
import os
import re
from setup_logging import log_uncaught_exceptions, remove_logs
import subprocess
import sys
import yaml

logger = logging.getLogger(__name__)

class Datum(object):
    def __init__(self, value, data_type=None, weight=None, source=None,
                 index=None, units=None):
        self.value = value
        self.data_type = data_type
        self.weight = weight
        self.index = index
        self.source = source
        self.units = units
    @property
    def name(self):
        return '{}_{}_{}'.format(self.data_type, self.source_name, self.index)
    @property
    def source_name(self):
        return re.split('[-_.]+', self.source)[0]
    def __repr__(self):
        return '{}({} : {})'.format(self.__class__.__name__, self.data_type,
                                    self.value)

def sort_datum(datum):
    '''
    Sorts data points such that the reference data points will be
    matched with the correct calculated data points. The sorting
    goes as follows:
   
    1st - By "data_type", a string. For example, both "-gq" and
          "-mq" have the data type "Charge".
    2nd - By "source_name", which is based upon the filename. The
          filename is split using regex ([-_.]+). The 1st string
          in the list is used as the "source_name".
    3rd - By "index". This is an integer or list of integers.
          For example, a single integer usually refers to the atom
          number. An atomic charge data point would use this.
          The list of integers is used to reference individual
          Hessian elements.
    '''
    return (datum.data_type, datum.source_name, datum.index)

def calculate_x2(ref_data, calc_data):
    '''
    Calculate the objective function. Uses the weights
    stored in the reference data points.
    
    ref_data  = A list of Datum objects representing
                the reference data.
    calc_data = A list of Datum objects representing
                the data calculated by the FF.
    '''
    x2 = 0.
    for r_datum, c_datum in zip(sorted(ref_data, key=sort_datum),
                                sorted(calc_data, key=sort_datum)):
        x2 += r_datum.weight**2 * (r_datum.value - c_datum.value)**2
    return x2
        
def unit_conv(unit_from, unit_to):
    if unit_from == 'Hartree' and unit_to == 'kcal mol^-1':
        return 627.503
    if unit_from == 'Hartree' and unit_to == 'J':
        return float('4.35974434e-18')
    if unit_from == 'Hartree' and unit_to == 'kJ mol^-1':
        return 2625.5
    if unit_from == 'Hatree Bohr^-2' and unit_to == 'kJ mol^-1 A^-2':
        return 9375.829222

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Imports data from QM calculations or runs MM '
        'calculations and imports data from the results.')
    data_opts = parser.add_argument_group('Data gathering/generating arguments')
    parser.add_argument(
        '--dir', type=str, metavar='relative/path/to/data', default=os.getcwd(),
        help='Set the directory where force field calculations will be ' +
        'performed. Directory must include the necessary data files and ' +
        'force field files.')
    parser.add_argument(
        '--norun', action='store_true', help="Don't run MacroModel " +
        'calculations. Assumes the output file from the calculation is ' +
        'already present.')
    parser.add_argument(
        '--output', '-o', type=str, metavar='filename', const='print',
        nargs='?',
        help='Write data to output file. If no filename is given, print ' +
        'the data to standard output. Format is similar to the old par.one ' +
        'and par.ref files.')
    parser.add_argument(
        '--weights', type=str, metavar='relative/path/weights.yaml',
        default='options/weights.yaml',
        help='Relative path to the weights YAML file.')
    data_opts.add_argument(
        '-gq', type=str, nargs='+', metavar='file.log',
        help='Gaussian ESP charges.')
    data_opts.add_argument(
        '-gqh', type=str, nargs='+', metavar='file.mae,file.log', 
        help='Gaussian charges excluding aliphatic hydrogens. Determine ' +
        'which hydrogens are aliphatic from file.mae, and exclude those ' +
        'charges.')
    data_opts.add_argument(
        '-je', type=str, nargs='+', metavar='file.mae', 
        help='Jaguar energy (r_j_Gas_Phase_Energy).')
    data_opts.add_argument(
        '-jh', type=str, nargs='+', metavar='file.in',
        help='Jaguar Hessian elements.')
    data_opts.add_argument(
        '-jq', type=str, nargs='+', metavar='file.mae',
        help='Jaguar charge (r_j_ESP_Charges).')
    data_opts.add_argument(
        '-me', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel single point energy (r_mmod_Potential_Energy-MM3*).')
    data_opts.add_argument(
        '-meo', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel single point energy ' +
        '(r_mmod_Potential_Energy-MM3*) after optimizing.')
    data_opts.add_argument(
        '-mq', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel charges (r_m_charge1).')
    data_opts.add_argument(
        '-mqh', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel charges excluding aliphatic hydrogens (r_m_charge1).')
    options = vars(parser.parse_args(args))
    # ignoring arguments that control settings like --dir, etc.
    # Also ignore unused data types.
    commands = {}
    for key, value in options.iteritems():
        # CHANGES
        if key in ['gq', 'gqh', 'je', 'jh', 'jq', 'me', 'meo', 'mq', 'mqh'] \
                and value is not None: 
            commands.update({key: value})
    logger.debug('Commands: {}'.format(commands))
    # Now make another dictionary where each input file is the key,
    # which is essentially the reverse of the last dictionary. We
    # want to know all of the commands that are to be performed
    # on a given file.
    inputs = {}
    for command, input_file_sets in commands.iteritems():
        # These sets of input files can be a single filename
        # or multiple filenames separated by commas.
        for input_file_set in input_file_sets:
            filenames = input_file_set.split(',')
            for filename in filenames:
                # We use a dictionary to hold the commands associated
                # with a given file rather than a list. This is so
                # the filename containing the output from the given
                # command (they key) can be stored as its value in the
                # dictionary.
                if filename in inputs:
                    inputs[filename].update({command: None})
                else:
                    inputs[filename] = {command: None}
    logger.debug('Inputs: {}'.format(inputs))
    # Create .com files.
    inputs, outputs, coms_to_run = make_macromodel_coms(
        inputs, rel_dir=options['dir'])
    logger.debug('Inputs: {}'.format(inputs))
    logger.debug('Outputs: {}'.format(outputs))
    # Run the .com files.
    if not options['norun']:
        current_directory = os.getcwd()
        logger.debug('Changing directory for calculations: {}'.format(
                options['dir']))
        os.chdir(options['dir'])
        for com in coms_to_run:
            logger.debug('Running {}.'.format(com))
            subprocess.check_output('bmin -WAIT {}'.format(
                    '.'.join(com.split('.')[:-1])), shell=True)
        logger.debug('Returning to original directory: {}'.format(
                current_directory))
        os.chdir(current_directory)
    with open(options['weights'], 'r') as f:
        weights = yaml.load(f)
    data = extract_data(commands, inputs, outputs, weights)
    # Some options for displaying the data.
    if options['output']:
        if options['output'] == 'print':
            for datum in sorted(data, key=sort_datum):
                print('{0:<20}{1:>10.4f}{2:>22.6f}'.format(
                        datum.name, datum.weight, datum.value))
        else:
            with open(options['output'], 'w') as f:
                for datum in sorted(data, key=sort_datum):
                    f.write('{0:<20}{1:>10.4f}{2:>22.6f}\n'.format(
                            datum.name, datum.weight, datum.value))
    return data

def make_macromodel_coms(inputs, rel_dir=os.getcwd()):
    coms_to_run = []
    # Create a dictionary where the output filenames are keys. As
    # extracting the data can take several seconds for large
    # output files containing thousands of structures, we don't
    # want to repeat this operation.
    outputs = {}
    for filename, commands in inputs.iteritems():
        logger.debug('Data types for {}: {}'.format(filename, commands.keys()))
        # MacroModel has to be used for these arguments.
        if set(commands).intersection(['me', 'meo', 'mq', 'mqh']): # CHANGES
            # Setup filenames.
            out_mae_filename = '.'.join(filename.split('.')[:-1]) + '-out.mae'
            com_filename = '.'.join(filename.split('.')[:-1]) + '.com'
            coms_to_run.append(com_filename)
            # Sometimes commands duplicate structures in the output. For
            # example a single point and a optimization will produce 2
            # output structures for each input structure. For this 
            # reason, we use an indexing system to keep track of where
            # data goes in the output file.
            calc_indices = []
            # These booleans control what goes into the .com file.
            multiple_structures = False
            single_point = False
            hessian = False
            optimization = False
            # CHANGES
            if set(commands).intersection(['me', 'mq', 'mqh']):
                single_point = True
            if set(commands).intersection(['meo']):
                optimization = True
            with open(os.path.join(rel_dir, filename), 'r') as f:
                number_structures = 0
                for line in f:
                    # This string marks the start of a new structure in a
                    # .mae file.
                    if 'f_m_ct {' in line:
                        number_structures += 1
            if number_structures > 1:
                logger.debug('Multiple structures detected in {}.'.format(
                        filename))
                multiple_structures = True
            # Write the .com file.
            # Specify the input and output files and load MM3*.
            com_contents = '{}\n{}\n'.format(filename, out_mae_filename) + \
                ' FFLD       2      0      0      0     ' + \
                '0.0000     0.0000     0.0000     0.0000\n'
            # Some debug commands are needed for the Hessian.
            if hessian:
                com_contents += ' DEBG      57    210    211\n'
            # Generic debug commands.
            else:
                com_contents += ' DEBG      57\n'
            # Start loop for multiple structures.
            if multiple_structures:
                com_contents += ' BGIN\n'
            com_contents += ' READ      -1      0      0      0     ' + \
                '0.0000     0.0000     0.0000     0.0000\n'
            # Setup for Hessian calculations. Needed for odd method of extracting
            # the Hessian using debug commands.
            if hessian:
                com_contents += ' MINI       9      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n' + \
                    ' RRHO       3      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n'
                calc_indices.append('Hessian')
            # Setup for single points and Hessian calculations.
            if single_point or hessian:
                com_contents += ' ELST       1      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n' + \
                    ' WRIT\n'
                calc_indices.append('Single Point')
            # Setup for optimizations.
            if optimization:
                com_contents += ' MINI       9      0     50      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n'
                calc_indices.append('Optimization')
            # End the multiple structure loop.
            if multiple_structures:
                com_contents += ' END\n'
            with open(os.path.join(rel_dir, com_filename), 'w') as f:
                f.write(com_contents)
            logger.debug('Wrote {} in {}.'.format(
                    com_filename, rel_dir))
        # Remember where to get the output data.
        # Add the output filenames as values in the input dictionary.
        # Add the output filenames as keys in the output dictionary, and
        # add the class that extracts data from the file as the value.
        # CHANGES
        for command in set(commands).intersection(['je', 'jq']):
            # In this case, the input file contains the data to extract.
            inputs[filename][command] = filename
            if filename not in outputs:
                outputs[filename] = filetypes.MaeFile(
                    filename, directory=rel_dir)
        for command in set(commands).intersection(['jh']):
            inputs[filename][command] = filename
            if filename not in outputs:
                outputs[filename] = filetypes.JagInFile(
                    filename, directory=rel_dir)
        for command in set(commands).intersection(['me', 'meo', 'mq', 'mqh']):
            inputs[filename][command] = (out_mae_filename, calc_indices)
            if out_mae_filename not in outputs:
                outputs[out_mae_filename] = filetypes.MaeFile(
                    out_mae_filename, directory=rel_dir)
        for command in set(commands).intersection(['gq', 'gqh']):
            inputs[filename][command] = filename
            if command == 'gq':
                outputs[filename] = filetypes.GaussLogFile(
                    filename, directory=rel_dir)
            elif command == 'gqh' and filename.endswith('.log'):
                outputs[filename] = filetypes.GaussLogFile(
                    filename, directory=rel_dir)
            elif command == 'gqh' and filename.endswith('.mae'):
                outputs[filename] = filetypes.MaeFile(
                    filename, directory=rel_dir)
    return inputs, outputs, coms_to_run

# CHANGES
def extract_data(commands, inputs, outputs, weights):
    data = []
    for command, input_file_sets in commands.iteritems():
        if command == 'gq':
            for filename in input_file_sets:
                charges, atom_nums = \
                    outputs[inputs[filename][command]].get_data(
                    'ESP Charges', atom_label='ESP Atom Nums')
                charges = [
                    Datum(x, data_type='Charge', weight=weights['Charge'],
                          index=y, source=inputs[filename][command])
                    for x, y in zip(charges, atom_nums)]
                data.extend(charges)
                logger.debug('Got {} charges from {}.'.format(
                        len(charges), filename))
        elif command == 'gqh':
            for input_file_set in input_file_sets:
                filenames = input_file_set.split(',')
                for filename in filenames:
                    if filename.endswith('.log'):
                        log = filename
                    elif filename.endswith('.mae'):
                        mae = filename
                # Same thing.
                # aliph_hyds = outputs[mae].get_aliph_hyds()
                aliph_hyds = outputs[inputs[mae][command]].get_aliph_hyds()
                charges, atom_nums = \
                    outputs[inputs[log][command]].get_data(
                    'ESP Charges', atom_label='ESP Atom Nums',
                    exclude_anums=aliph_hyds)
                charges = [
                    Datum(x, data_type='Charge', index=y,
                          weight=weights['Charge'],
                          source=inputs[filename][command])
                    for x, y in zip(charges, atom_nums)]
                data.extend(charges)
                logger.debug('Got {} charges from {}.'.format(
                        len(charges), filename))
        elif command == 'je':
            for filename in input_file_sets:
                energies = [x['r_j_Gas_Phase_Energy'] for x in 
                            outputs[inputs[filename][command]].raw_data]
                energies = map(float, energies)
                energies = [
                    Datum(x, data_type='Energy', weight=weights['Energy'],
                          index=i + 1, source=inputs[filename][command],
                          units='Hartree')
                    for i, x in enumerate(energies)]
                data.extend(energies)
                logger.debug('Got {} energies from {}.'.format(
                        len(energies), filename))
        elif command == 'jh':
            for filename in input_file_sets:
                hessian, indices = \
                    outputs[inputs[filename][command]].get_hess_tril_array()
                hessian = [
                    Datum(h, data_type='Hessian', weight=weights['Hessian'],
                          index=(x, y), source=inputs[filename][command],
                          units='Hartree Bohr^-2')
                    for h, x, y in zip(hessian, indices[0], indices[1])]
                data.extend(hessian)
                logger.debug('Got {} Hessian elements from {}.'.format(
                        len(hessian), filename))
        elif command == 'jq':
            for filename in input_file_sets:
                # inputs[filename][command] = Filename of output file
                charges, atom_nums = \
                    outputs[inputs[filename][command]].get_data(
                    'r_j_ESP_Charges')
                charges = [
                    Datum(x, data_type='Charge', weight=weights['Charge'],
                          index=y, source=inputs[filename][command])
                    for x, y in zip(charges, atom_nums)]
                data.extend(charges)
                logger.debug('Got {} charges from {}.'.format(
                        len(charges), filename))
        elif command == 'me':
            for filename in input_file_sets:
                # Index 0 corresponds to the filetype class.
                # Index 1 corresponds to the calculation indices.
                energies, atom_nums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_mmod_Potential_Energy-MM3*', calc_type='Single Point',
                    calc_indices=inputs[filename][command][1])
                energies = [
                    Datum(x, data_type='Energy', index=i + 1,
                          weight=weights['Energy'], units='kJ mol^-1',
                          source=inputs[filename][command][0])
                    for i, x in enumerate(energies)]
                data.extend(energies)
                logger.debug('Got {} energies from {}.'.format(
                        len(energies), filename))
        elif command == 'meo':
            for filename in input_file_sets:
                # Index 0 corresponds to the filetype class.
                # Index 1 corresponds to the calculation indices.
                energies, atom_nums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_mmod_Potential_Energy-MM3*', calc_type='Optimization',
                    calc_indices=inputs[filename][command][1])
                energies = [
                    Datum(x, data_type='Energy', index=i + 1,
                          weight=weights['Energy'], units='kJ mol^-1',
                          source=inputs[filename][command][0])
                    for i, x in enumerate(energies)]
                data.extend(energies)
                logger.debug('Got {} energies from {}.'.format(
                        len(energies), filename))
        elif command == 'mq':
            for filename in input_file_sets:
                charges, atom_nums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_m_charge1', calc_type='Single Point',
                    calc_indices=inputs[filename][command][1])
                charges = [
                    Datum(x, data_type='Charge', index=y,
                          weight=weights['Charge'],
                          source=inputs[filename][command][0])
                    for x, y in zip(charges, atom_nums)]
                data.extend(charges)
                logger.debug('Got {} charges from {}.'.format(
                        len(charges), filename))
        elif command == 'mqh':
            for filename in input_file_sets:
                aliph_hyds = \
                    outputs[inputs[filename][command][0]].get_aliph_hyds()
                logger.debug('Aliph. hyds. in {}: {}'.format(
                        filename, aliph_hyds))
                charges, atom_nums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_m_charge1', calc_type='Single Point',
                    calc_indices=inputs[filename][command][1],
                    exclude_anums=aliph_hyds)
                charges = [
                    Datum(x, data_type='Charge', index=y,
                          weight=weights['Charge'],
                          source=inputs[filename][command][0])
                    for x, y in zip(charges, atom_nums)]
                data.extend(charges)
                logger.debug('Got {} charges from {}.'.format(
                        len(charges), filename))
    logger.debug('Got {} total data points.'.format(len(data)))
    return data

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        config = yaml.load(f)
    logging.config.dictConfig(config)
    # Execute.
    process_args(sys.argv[1:])
