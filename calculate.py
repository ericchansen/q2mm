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
import numpy as np
import os
import re
from setup_logging import log_uncaught_exceptions, remove_logs
import subprocess
import sys
import yaml

logger = logging.getLogger(__name__)

class Datum(object):
    def __init__(self, value, data_type=None, weight=None, source=None,
                 index=None, units=None, str_num=None, calc_com=None):
        self.value = value
        self.data_type = data_type
        self.weight = weight
        self.index = index
        self.source = source
        self.units = units
        self.str_num = str_num
        self.calc_com = calc_com
    @property
    def name(self):
        return '{}_{}_{}_{}'.format(
            self.calc_com, self.source_name, self.str_num, self.index)
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
    return (datum.data_type, datum.source_name, datum.str_num, datum.index)

def calculate_x2(ref_data, calc_data):
    '''
    Calculate the objective function. Uses the weights
    stored in the reference data points. Note that this
    now modifies the calculated data to account for
    relative energies.
    
    ref_data  = A list of Datum objects representing
                the reference data.
    calc_data = A list of Datum objects representing
                the data calculated by the FF.
    '''
    x2 = 0.
    # Although the calculated energies are on their own relative
    # scale, the reference's minimum should be used for the
    # calculated data's minimum.
    r_energies = sorted([x for x in ref_data if x.data_type == 'energy'],
                        key=sort_datum)
    c_energies = sorted([x for x in calc_data if x.data_type == 'energy'],
                        key=sort_datum)
    if r_energies and c_energies:
        assert len(r_energies) == len(c_energies), \
            "Number of reference and calculated energies don't match."
        r_zero = min([x.value for x in r_energies])
        r_zero_i = [x.value for x in r_energies].index(r_zero)
        c_zero = c_energies[r_zero_i].value
        for e in c_energies:
            e.value -= c_zero
    for r_datum, c_datum in zip(sorted(ref_data, key=sort_datum),
                                sorted(calc_data, key=sort_datum)):
        x2 += r_datum.weight**2 * (r_datum.value - c_datum.value)**2
    return x2, ref_data, calc_data
        
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
        '--norel', action='store_true', help="Don't use relative energies " +
        'in the output data.')
    parser.add_argument(
        '--norun', action='store_true',
        help="Don't run MacroModel " +
        'calculations. Assumes the output file from the calculation is ' +
        'already present. If an optional argument is given, this is used ' +
        'as the mae indices.')
    parser.add_argument(
        '--output', '-o', type=str, metavar='filename', const='print',
        nargs='?',
        help='Write data to output file. If no filename is given, print ' +
        'the data to standard output. Format is similar to the old par.one ' +
        'and par.ref files.')
    parser.add_argument(
        '--scanind', type=str, metavar='d1,d2',
        help='If you are retrieving energy data, also search for these ' +
        'scan indices to be used as the data indeces.')
    parser.add_argument(
        '--substr', type=str, metavar='"string in substructure title"',
        const='OPT', nargs='?',
        help='When selecting bonds and angles as data, only include those ' +
        'that are directly related to a parameter in the substructure of ' +
        'the FF. If no additional argument is given, it looks for any bonds ' +
        'or angles in a substructure that includes the word "OPT".')
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
        '-ja', type=str, nargs='+', metavar='file.mae',
        help='Jaguar angles.')
    data_opts.add_argument(
        '-jb', type=str, nargs='+', metavar='file.mae',
        help='Jaguar bond lengths.')
    data_opts.add_argument(
        '-je', type=str, nargs='+', metavar='file.mae', 
        help='Jaguar energy (r_j_Gas_Phase_Energy).')
    data_opts.add_argument(
        '-jeig', type=str, nargs='+', metavar='file.in,file.out',
        help='Eigenvalues from Jaguar calculation.')
    data_opts.add_argument(
        '-jh', type=str, nargs='+', metavar='file.in',
        help='Jaguar Hessian elements. Ouputs as the mass-weighted Hessian.')
    data_opts.add_argument(
        '-jhi', type=str, nargs='+', metavar='file.in',
        help='Jaguar Hessian elements. Outputs as inverted and mass-weighted.')
    data_opts.add_argument(
        '-jq', type=str, nargs='+', metavar='file.mae',
        help='Jaguar charge (r_j_ESP_Charges).')
    data_opts.add_argument(
        '-ma', type=str, nargs='+', metavar='file.mae',
        help='MacroModel optimized structure bond lengths.')
    data_opts.add_argument(
        '-mb', type=str, nargs='+', metavar='file.mae',
        help='MacroModel optimized structure bond lengths.')
    data_opts.add_argument(
        '-me', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel single point energy (r_mmod_Potential_Energy-MM3*).')
    data_opts.add_argument(
        '-meo', type=str, nargs='+', metavar='file.mae', 
        help='MacroModel single point energy ' +
        '(r_mmod_Potential_Energy-MM3*) after optimizing.')
    data_opts.add_argument(
        '-meig', type=str, nargs='+', metavar='file.mae,file.out',
        help='MacroModel eigenvalue method.')
    data_opts.add_argument(
        '-mh', type=str, nargs='+', metavar='file.mae',
        help='MacroModel Hessian elements. Ouputs as the mass-weighted ' +
        'Hessian.')
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
    # for key, value in options.iteritems():
    for key, value in options.items():
        # CHANGES
        if key in ['gq', 'gqh', 'ja', 'jb', 'je', 'jeig', 'jh', 'jhi', 'jq',
                   'ma', 'mb', 'me', 'meo', 'meig', 'mh', 'mq', 'mqh'] \
                and value is not None: 
            commands.update({key: value})
    logger.debug('Commands: {}'.format(commands))
    # Now make another dictionary where each input file is the key,
    # which is essentially the reverse of the last dictionary. We
    # want to know all of the commands that are to be performed
    # on a given file.
    inputs = {}
    # for command, input_file_sets in commands.iteritems():
    for command, input_file_sets in commands.items():
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
        inputs, rel_dir=options['dir'], scaninds=options['scanind'])
    logger.debug('Inputs: {}'.format(inputs))
    logger.debug('Outputs: {}'.format(outputs))
    # Run the .com files.
    if not options['norun'] and coms_to_run:
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
    data = extract_data(commands, inputs, outputs, weights, options['norel'],
                        substr=options['substr'], scanind=options['scanind'])
    # Some options for displaying the data.
    if options['output']:
        if options['output'] == 'print':
            for datum in sorted(data, key=sort_datum):
                print('{0:<30}{1:>10.4f}{2:>22.6f}'.format(
                        datum.name, datum.weight, datum.value))
        else:
            with open(options['output'], 'w') as f:
                for datum in sorted(data, key=sort_datum):
                    f.write('{0:<30}{1:>10.4f}{2:>22.6f}\n'.format(
                            datum.name, datum.weight, datum.value))
    return data

def make_macromodel_coms(inputs, rel_dir=os.getcwd(), scaninds=None):
    coms_to_run = []
    # Create a dictionary where the output filenames are keys. As
    # extracting the data can take several seconds for large
    # output files containing thousands of structures, we don't
    # want to repeat this operation.
    outputs = {}
    for filename, commands in inputs.iteritems():
        # Rework flow so this doesn't have to be here.
        if filename.endswith('.out'):
            for command in set(commands).intersection(['meig']):
                if filename not in outputs:
                    if filename.endswith('.out'):
                        outputs[filename] = filetypes.JagOutFile(
                            filename, directory=rel_dir)
            continue
        logger.debug('Data types for {}: {}'.format(filename, commands.keys()))
        # MacroModel has to be used for these arguments.
        # CHANGES
        if set(commands).intersection(
            ['ja', 'jb', 'ma', 'mb', 'me', 'meo', 'meig', 'mh', 'mq', 'mqh']):
            # Setup filenames.
            out_mae_filename = '.'.join(filename.split('.')[:-1]) + '-out.mae'
            out_mmo_filename = '.'.join(filename.split('.')[:-1]) + '-out.mmo'
            out_log_filename = '.'.join(filename.split('.')[:-1]) + '.log'
            com_filename = '.'.join(filename.split('.')[:-1]) + '.com'
            coms_to_run.append(com_filename)
            # Sometimes commands duplicate structures in the output. For
            # example a single point and a optimization will produce 2
            # output structures for each input structure. For this 
            # reason, we use an indexing system to keep track of where
            # data goes in the output file.
            mae_indices = []
            mmo_indices = []
            # These booleans control what goes into the .com file.
            multiple_structures = False
            single_point = False
            hessian = False
            optimization = False
            pre_opt_str = False
            post_opt_str = False
            # CHANGES
            if set(commands).intersection(['ja', 'jb']):
                pre_opt_str = True
            if set(commands).intersection(['me', 'mq', 'mqh']):
                single_point = True
            if set(commands).intersection(['meig', 'mh']):
                hessian = True
            if set(commands).intersection(['ma', 'mb', 'meo']):
                optimization = True
            if set(commands).intersection(['ma', 'mb']):
                post_opt_str = True
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
            # Setup for Hessian calculations. Needed for odd method of
            # extracting the Hessian using debug commands.
            # MINI 9 uses PRCG. Has risk of not converging, but that's okay 
            # because we aren't actually optimizing here. This is just a 
            # necessary workaround (evil?) such that the Hessian will be
            # output properly.
            if hessian:
                com_contents += ' MINI       9      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n' + \
                    ' RRHO       3      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n'
                mae_indices.append('hessian')
            # Setup for single points and Hessian calculations. If you don't
            # have this, the output mae appears to be the same as the input.
            # I think the WRIT line will only be used later for reference
            # structures.
            if single_point or hessian or pre_opt_str:
                com_contents += ' ELST       1      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n'
                com_contents += ' WRIT\n'
                mae_indices.append('pre-opt. energy')
                mmo_indices.append('pre-opt. str.')
            # Setup for optimizations.
            # Changed from PRCG (9) to TNCG (1). Upped iterations from
            # 50 to 500.
            if optimization:
                com_contents += ' MINI       1      0    500      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n'
                mae_indices.append('optimization')
            if post_opt_str or (scaninds and optimization):
                com_contents += ' ELST       1      0      0      0     ' + \
                    '0.0000     0.0000     0.0000     0.0000\n' + \
                    ' WRIT\n'
                mae_indices.append('post-opt. energy')
                mmo_indices.append('post-opt. str.')
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
        for command in set(commands).intersection(['ja', 'jb']):
            inputs[filename][command] = (out_mmo_filename, mmo_indices)
            if out_mmo_filename not in outputs:
                outputs[out_mmo_filename] = filetypes.MMoFile(
                    out_mmo_filename, directory=rel_dir)
        for command in set(commands).intersection(['je', 'jq']):
            # In this case, the input file contains the data to extract.
            inputs[filename][command] = filename
            if filename not in outputs:
                outputs[filename] = filetypes.MaeFile(
                    filename, directory=rel_dir)
        for command in set(commands).intersection(['jeig']):
            if filename not in outputs:
                if filename.endswith('.in'):
                    outputs[filename] = filetypes.JagInFile(
                        filename, directory=rel_dir)
                if filename.endswith('.out'):
                    outputs[filename] = filetypes.JagOutFile(
                        filename, directory=rel_dir)
        for command in set(commands).intersection(['jh', 'jhi']):
            inputs[filename][command] = filename
            if filename not in outputs:
                outputs[filename] = filetypes.JagInFile(
                    filename, directory=rel_dir)
        for command in set(commands).intersection(['ma', 'mb']):
            inputs[filename][command] = (out_mmo_filename, mmo_indices)
            if out_mmo_filename not in outputs:
                outputs[out_mmo_filename] = filetypes.MMoFile(
                    out_mmo_filename, directory=rel_dir)
        for command in set(commands).intersection(['me', 'meo', 'mq', 'mqh']):
            inputs[filename][command] = (out_mae_filename, mae_indices)
            if out_mae_filename not in outputs:
                outputs[out_mae_filename] = filetypes.MaeFile(
                    out_mae_filename, directory=rel_dir)
        for command in set(commands).intersection(['meig']):
            inputs[filename][command] = out_log_filename
            if out_log_filename not in outputs:
                outputs[out_log_filename] = filetypes.MMoLogFile(
                    out_log_filename, directory=rel_dir)
        for command in set(commands).intersection(['mh']):
            inputs[filename][command] = out_log_filename
            if out_log_filename not in outputs:
                outputs[out_log_filename] = filetypes.MMoLogFile(
                    out_log_filename, directory=rel_dir)
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
def extract_data(commands, inputs, outputs, weights, no_rel_energy=False,
                 substr='OPT', scanind=None):
    data = []
    # for command, input_file_sets in commands.iteritems():
    for command, input_file_sets in commands.items():
        if command == 'gq':
            for filename in input_file_sets:
                more_data = []
                all_charges, all_anums = \
                    outputs[inputs[filename][command]].get_data(
                    'ESP Charges', atom_label='ESP Atom Nums')
                for str_num, (charges, anums) in enumerate(zip(
                        all_charges, all_anums)):
                    some_data = [
                        Datum(float(q), data_type='charge',
                              weight=weights['Charge'],
                              index=a, source=inputs[filename][command],
                              calc_com='gq')
                        for q, a in zip(charges, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} charge(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'gqh':
            for input_file_set in input_file_sets:
                more_data = []
                filenames = input_file_set.split(',')
                for filename in filenames:
                    if filename.endswith('.log'):
                        log = filename
                    elif filename.endswith('.mae'):
                        mae = filename
                # aliph_hyds = outputs[mae].get_aliph_hyds()
                aliph_hyds = outputs[inputs[mae][command]].get_aliph_hyds()
                all_charges, all_anums = \
                    outputs[inputs[log][command]].get_data(
                    'ESP Charges', atom_label='ESP Atom Nums',
                    exclude_anums=aliph_hyds)
                for str_num, (charges, anums) in enumerate(zip(
                        all_charges, all_anums)):
                    some_data = [
                        Datum(float(q), data_type='charge', index=int(a),
                              weight=weights['Charge'],
                              source=inputs[filename][command],
                              calc_com='gqh')
                        for q, a in zip(charges, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} charge(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'ja':
            for filename in input_file_sets:
                more_data = []
                all_angs, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'A.', atom_label='A. Nums.', calc_type='pre-opt. str.',
                    calc_indices=inputs[filename][command][1],
                    comment_label='A. Com.', substr=substr)
                for str_num, (angs, anums) in enumerate(zip(
                        all_angs, all_anums)):
                    some_data = [
                        Datum(float(ang), data_type='angle', index=anum,
                              weight=weights['Angle'], units='Degrees',
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='ja')
                        for ang, anum in zip(angs, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} angle(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'jb':
            for filename in input_file_sets:
                more_data = []
                all_bonds, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'B.', atom_label='B. Nums.', calc_type='pre-opt. str.',
                    calc_indices=inputs[filename][command][1],
                    comment_label='B. Com.', substr=substr)
                for str_num, (bonds, anums) in enumerate(zip(
                        all_bonds, all_anums)):
                    some_data = [
                        Datum(float(b), data_type='bond', index=a,
                              weight=weights['Bond'], units='Angstroms',
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='jb')
                        for b, a in zip(bonds, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} bond(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'je':
            if scanind:
                scaninds = scanind.split(',')
                scaninds = ['r_j_' + x for x in scaninds]
            for filename in input_file_sets:
                energies = [x['r_j_Gas_Phase_Energy'] for x in 
                            outputs[inputs[filename][command]].raw_data]
                energies = map(float, energies)
                if scanind:
                    labels = []
                    for x in outputs[inputs[filename][command]].raw_data:
                        subgrouping = []
                        for ind in scaninds:
                            subgrouping.append(x[ind])
                        labels.append(subgrouping)
                    energies = [
                        Datum(e, data_type='energy', weight=weights['Energy'],
                              str_num=str_num, source=inputs[filename][command],
                              units='Hartree', calc_com='je', index=l)
                        for str_num, (e, l) in enumerate(zip(
                                energies, labels))]
                else:
                    energies = [
                        Datum(e, data_type='energy', weight=weights['Energy'],
                              str_num=str_num, source=inputs[filename][command],
                              units='Hartree', calc_com='je')
                        for str_num, e in enumerate(energies)]
                data.extend(energies)
                logger.debug('{} energ(ies/y) from {}.'.format(
                        len(energies), filename))
        elif command == 'jeig':
            for file_set in input_file_sets:
                filenames = file_set.split(',')
                out_file = [x for x in filenames if x.endswith('.out')][0]
                in_file = [x for x in filenames if x.endswith('.in')][0]
                hess = filetypes.Hessian()
                hess.load_from_jaguar_out(outputs[out_file])
                hess.load_from_jaguar_in(outputs[in_file])
                hess.mass_weight_hess()
                hess.mass_weight_evec()
                hess.convert_units_for_mm()
                hess.inv_hess()
                print np.diag(hess.evals)
        elif command == 'jh':
            for filename in input_file_sets:
                hess = filetypes.Hessian()
                hess.load_from_jaguar_in(outputs[inputs[filename][command]])
                hess.mass_weight_hess()
                hess.convert_units_for_mm()
                indices = np.tril_indices_from(hess.matrix)
                used_elements = hess.matrix[indices]
                used_data = [
                    Datum(ele,
                          data_type='hessian',
                          weight=weights['Hessian'],
                          index=(x, y),
                          source=inputs[filename][command],
                          units='kJ mol^-1 A^-2 amu^-1',
                          calc_com='jh')
                    for ele, x, y, in zip(
                        used_elements, indices[0], indices[1])]
                data.extend(used_data)
                logger.debug('{} Hessian elements from {}'.format(
                        len(used_data), filename))
        elif command == 'jhi':
            for filename in input_file_sets:
                hess = filetypes.Hessian()
                hess.load_from_jaguar_in(outputs[inputs[filename][command]])
                hess.mass_weight_hess()
                hess.convert_units_for_mm()
                hess.inv_hess()
                indices = np.tril_indices_from(hess.matrix)
                used_elements = hess.matrix[indices]
                used_data = [
                    Datum(ele,
                          data_type='hessian',
                          weight=weights['Hessian'],
                          index=(x, y),
                          source=inputs[filename][command],
                          units='kJ mol^-1 A^-2 amu^-1',
                          calc_com='jh')
                    for ele, x, y, in zip(
                        used_elements, indices[0], indices[1])]
                data.extend(used_data)
                logger.debug('{} Hessian elements from {}'.format(
                        len(used_data), filename))
        elif command == 'jq':
            for filename in input_file_sets:
                more_data = []
                # inputs[filename][command] = Filename of output file
                all_charges, all_anums = \
                    outputs[inputs[filename][command]].get_data(
                    'r_j_ESP_Charges')
                for str_num, (charges, anums) in enumerate(zip(
                        all_charges, all_anums)):
                    some_data = [
                        Datum(float(q), data_type='charge',
                              weight=weights['Charge'],
                              index=int(a), source=inputs[filename][command],
                              calc_com='jq')
                        for q, a in zip(charges, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} charge(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'ma':
            for filename in input_file_sets:
                more_data = []
                all_angs, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'A.', atom_label='A. Nums.', calc_type='post-opt. str.',
                    calc_indices=inputs[filename][command][1],
                    comment_label='A. Com.', substr=substr)
                for str_num, (angs, anums) in enumerate(zip(
                        all_angs, all_anums)):
                    some_data = [
                        Datum(float(ang), data_type='angle', index=anum,
                              weight=weights['Angle'], units='Degrees',
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='ma')
                        for ang, anum in zip(angs, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} angle(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'mb':
            for filename in input_file_sets:
                more_data = []
                all_bonds, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'B.', atom_label='B. Nums.', calc_type='post-opt. str.',
                    calc_indices=inputs[filename][command][1],
                    comment_label='B. Com.', substr=substr)
                for str_num, (bonds, anums) in enumerate(zip(
                        all_bonds, all_anums)):
                    some_data = [
                        Datum(float(bond), data_type='bond', index=anum,
                              weight=weights['Bond'], units='Angstroms',
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='mb')
                        for bond, anum in zip(bonds, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} bond(s) from {}'.format(
                        len(more_data), filename))
        elif command == 'me':
            if scanind:
                scaninds = scanind.split(',')
                scaninds = ['r_j_' + x for x in scaninds]
            for filename in input_file_sets:
                more_data = []
                if scanind:
                    all_energies, all_indices = \
                        outputs[inputs[filename][command][0]].get_data(
                        'r_mmod_Potential_Energy-MM3*',
                        calc_type='pre-opt. energy',
                        calc_indices=inputs[filename][command][1],
                        scan_inds=scaninds)
                else:
                    all_energies, all_indices = \
                        outputs[inputs[filename][command][0]].get_data(
                        'r_mmod_Potential_Energy-MM3*',
                        calc_type='pre-opt. energy',
                        calc_indices=inputs[filename][command][1])
                # This is a bit inefficient in the case of not using the indices.
                for str_num, (energies, labels) in enumerate(zip(
                        all_energies, all_indices)):
                    if scanind:
                        more_data.append(Datum(
                                float(energies[0]), data_type='energy',
                                weight=weights['Energy'],
                                units='kJ mol^-1',
                                source=inputs[filename][command][0],
                                str_num=str_num, calc_com='me',
                                index=labels))
                    else:
                        more_data.append(Datum(
                                float(energies[0]), data_type='energy',
                                weight=weights['Energy'], units='kJ mol^-1',
                                source=inputs[filename][command][0],
                                str_num=str_num, calc_com='me'))
                data.extend(more_data)
                logger.debug('{} energ(ies/y) from {}.'.format(
                        len(more_data), filename))
        elif command == 'meo':
            if scanind:
                scaninds = scanind.split(',')
                scaninds = ['r_j_' + x for x in scaninds]
            for filename in input_file_sets:
                more_data = []
                if scanind:
                    all_energies, all_indices = \
                        outputs[inputs[filename][command][0]].get_data(
                        'r_mmod_Potential_Energy-MM3*', calc_type='post-opt. energy',
                        calc_indices=inputs[filename][command][1],
                        scan_inds=scaninds)
                else:
                    all_energies, all_indices = \
                        outputs[inputs[filename][command][0]].get_data(
                        'r_mmod_Potential_Energy-MM3*', calc_type='optimization',
                        calc_indices=inputs[filename][command][1])
                for str_num, (energies, labels) in enumerate(zip(
                        all_energies, all_indices)):
                    if scanind:
                        more_data.append(Datum(
                                float(energies[0]), data_type='energy',
                                weight=weights['Energy'], units='kJ mol^-1',
                                source=inputs[filename][command][0],
                                str_num=str_num, calc_com='meo',
                                index=labels))
                    else:
                        more_data.append(Datum(
                                float(energies[0]), data_type='energy',
                                weight=weights['Energy'], units='kJ mol^-1',
                                source=inputs[filename][command][0],
                                str_num=str_num, calc_com='meo'))
                data.extend(more_data)
                logger.debug('{} geo. opt. energ(ies/y) from {}.'.format(
                        len(more_data), filename))
        elif command == 'meig':
            for file_set in input_file_sets:
                filenames = file_set.split(',')
                mae_file = [x for x in filenames if x.endswith('.mae')][0]
                out_file = [x for x in filenames if x.endswith('.out')][0]
                hess = filetypes.Hessian()
                hess.load_from_jaguar_out(outputs[out_file])
                hess.load_from_mmo_log(outputs[inputs[mae_file][command]])
                hess.mass_weight_evec()
                # print '==== EIGENVECTORS ============================'
                # print hess.evecs
                # print '========== HESS ==============================='
                # print hess.matrix
                yay = hess.do_what_i_want()
                yay = np.array(yay)
                # print '============= ANSWER? ======================='
                indices = np.tril_indices_from(yay)
                used_elements = yay[indices]
                used_data = [
                    Datum(ele,
                          data_type='hessian',
                          weight=weights['Hessian'],
                          index=(x, y),
                          source=inputs[mae_file][command],
                          units='kJ mol^-1 A^-2 amu^-1',
                          calc_com='jh')
                    for ele, x, y, in zip(
                        used_elements, indices[0], indices[1])]
                data.extend(used_data)
                logger.debug('{} strange Hessian elements from {}'.format(
                        len(used_data), inputs[mae_file][command]))
        elif command == 'mh':
            for filename in input_file_sets:
                hess = filetypes.Hessian()
                
                hessian, indices = \
                    outputs[inputs[filename][command]].get_hess_tril_array()
                hessian = [
                    Datum(h, data_type='hessian', weight=weights['Hessian'],
                          index=(x, y), source=inputs[filename][command],
                          units='kJ mol^-1 A^-2', calc_com='mh')
                    for h, x, y in zip(hessian, indices[0], indices[1])]
                data.extend(hessian)
                logger.debug('{} Hessian elements from {}.'.format(
                        len(hessian), filename))
        elif command == 'mq':
            for filename in input_file_sets:
                more_data = []
                all_charges, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_m_charge1', calc_type='pre-opt. energy',
                    calc_indices=inputs[filename][command][1])
                for str_num, (charges, anums) in enumerate(zip(
                        all_charges, all_anums)):
                    some_data = [
                        Datum(float(q), data_type='charge', index=int(a),
                              weight=weights['Charge'],
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='mq')
                        for q, a in zip(charges, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} charge(s) from {}.'.format(
                        len(more_data), filename))
        elif command == 'mqh':
            for filename in input_file_sets:
                more_data = []
                aliph_hyds = \
                    outputs[inputs[filename][command][0]].get_aliph_hyds()
                all_charges, all_anums = \
                    outputs[inputs[filename][command][0]].get_data(
                    'r_m_charge1', calc_type='pre-opt. energy',
                    calc_indices=inputs[filename][command][1],
                    exclude_anums=aliph_hyds)
                for str_num, (charges, anums) in enumerate(zip(
                        all_charges, all_anums)):
                    some_data = [
                        Datum(float(q), data_type='charge', index=int(a),
                              weight=weights['Charge'],
                              source=inputs[filename][command][0],
                              str_num=str_num, calc_com='mqh')
                        for q, a in zip(charges, anums)]
                    more_data.extend(some_data)
                data.extend(more_data)
                logger.debug('{} charge(s) from {}.'.format(
                        len(more_data), filename))
    # Convert units.
    for d in data:
        if d.units == 'Hartree':
            d.value *= convert_units(d.units, 'kJ mol^-1')
            d.units = 'kJ mol^-1'
        if d.units == 'Hartree Bohr^-2':
            d.value *= convert_units(d.units, 'kJ mol^-1 A^-2')
            d.units = 'kJ mol^-1 A^-2'
    if no_rel_energy:
        logger.debug('Skipping conversion to relative energies.')
    else:
        energies = [x for x in data if x.data_type == 'energy']
        if energies:
            logger.debug('Converting to relative energies.')
            zero = min([e.value for e in energies])
            for e in energies:
                e.value -= zero
    logger.debug('{} total data point(s).'.format(len(data)))
    return data

def convert_units(unit_from, unit_to):
    if unit_from == 'Hartree' and unit_to == 'kcal mol^-1':
        return 627.503
    if unit_from == 'Hartree' and unit_to == 'J':
        return float('4.35974434e-18')
    if unit_from == 'Hartree' and unit_to == 'kJ mol^-1':
        return 2625.5
    if unit_from == 'Hartree Bohr^-2' and unit_to == 'kJ mol^-1 A^-2':
        return 9375.829222

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        config = yaml.load(f)
    logging.config.dictConfig(config)
    # Execute.
    process_args(sys.argv[1:])
