#!/usr/bin/python
'''
Generates data used in the penalty function.
'''
# there seems to be a small difference in the data produced by jhi and the old
# Hessian inversion scripts

# seems that the value used for inversion randomly switches between 1 and
# 9375.828222. set so that it reproduces elaine's scripts for each function

# concern over RRHO argument 4. it's temperature. used to be 0 K. should it be
# 300 K?

# can speed up by making data a np.ndarray using np.array(dtype=object). then
# data.min/data.amin would be faster, but somehow i'd have to set up datum such
# that it worked.

# a lot of repetition in collect_data could be eliminated to shorten the code.
from collections import defaultdict
from itertools import chain, izip
import argparse
import logging
import logging.config
import numpy as np
import os
import subprocess
import sys
import time
import traceback
import yaml

from datatypes import Datum, datum_sort_key, Hessian, MM3
import constants as cons
import filetypes

logger = logging.getLogger(__name__)

# remember to add in inverse distance
commands_gaussian = [] # gq, gqh
commands_jaguar = ['je', 'je2', 'jeig', 'jeigi', 'jeige', 'jeigz', 'jeigzi', 'jh', 'jhi', 'jq', 'jqh']
commands_macromodel = ['ja', 'jb', 'ma', 'mb', 'mcs', 'mcs2', 'me', 'me2',
                       'meo', 'meig', 'meigz', 'mh', 'mq', 'mqh']
commands_other = ['pm', 'pr', 'r', 'zm', 'zr']
commands_all = commands_gaussian + commands_jaguar + commands_macromodel + commands_other

# these commands require me to import the force field
commands_need_ff = ['ma', 'mb', 'ja', 'jb', 'pm', 'zm']

def return_calculate_parser(add_help=True, parents=[]):
    '''
    Return an argument parser for calculate.
    '''
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    calc_args = parser.add_argument_group('calculate options')
    calc_args.add_argument(
        '--dir', '-d', type=str, metavar='directory', default=os.getcwd(),
        help=('Searches for files (data files like .mae, .log, etc. and force '
              'field files) in this directory. 3rd party calculations are '
              'executed from this directory.'))
    calc_args.add_argument(
        '--norun', '-n', action='store_true',
        help="Don't run 3rd party software.")
    calc_args.add_argument(
        '--printdata', '-pd', action='store_true', help='Print data.')
    data_args = parser.add_argument_group('calculate data types')
    data_args.add_argument(
        '-ma', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel angles (post force field optimization).')
    data_args.add_argument(
        '-mb', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel bond lengths (post force field optimization).')
    data_args.add_argument(
        '-mcs', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Run a MacroModel conformational search. Not designed to work in '
              'conjunction with any other commands.'))
    data_args.add_argument(
        '-mcs2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Run a MacroModel conformational search. Not designed to work in '
              'conjunction with any other commands.'))
    data_args.add_argument(
        '-me', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel energies (pre force field optimization).')
    data_args.add_argument(
        '-me2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('MacroModel energies. Same as -me, but having two options '
              'allows for the weights to be set differently.'))
    data_args.add_argument(
        '-meo', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel energies (post force field optimization).')
    data_args.add_argument(
        '-meig', type=str, nargs='+', action='append', default=[], metavar='file.mae,file.out',
        help='MacroModel eigenmode fitting. Includes diagonal and off diagonal elements.')
    data_args.add_argument(
        '-meigz', type=str, nargs='+', action='append', default=[], metavar='file.mae,file.out',
        help="MacroModel eigenmode fitting. Doesn't include off diagonal elements.")
    data_args.add_argument(
        '-mh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel Hessian.')
    data_args.add_argument(
        '-mq', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel charges.')
    data_args.add_argument(
        '-mqh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel charges, but excludes aliphatic hydrogens.')
    data_args.add_argument(
        '-pm', type=str, nargs='+', action='append', default=[], metavar='parteth',
        help='Uses a tethering file for parameters. Calculated data.')
    data_args.add_argument(
        '-pr', type=str, nargs='+', action='append', default=[], metavar='parteth',
        help='Uses a tethering file for parameters. Reference data.')
    data_args.add_argument(
        '-ja', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar angles.')
    data_args.add_argument(
        '-jb', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar bond lengths.')
    data_args.add_argument(
        '-je', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar energies.')
    data_args.add_argument(
        '-je2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar energies. Can set weight separately from -je.')
    data_args.add_argument(
        '-jeig', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help='Jaguar eigenmode fitting.')
    data_args.add_argument(
        '-jeigi', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help='Jaguar eigenmode fitting with inversion of the 1st eigenvalue.')
    data_args.add_argument(
        '-jeige', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help=('Jaguar eigenmode fitting. Zeros all off diagonal elements. '
              "Equivalent to Elaine's method."))
    data_args.add_argument(
        '-jeigz', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help="Jaguar eigenmode fitting. Don't include off diagonal elements.")
    data_args.add_argument(
        '-jeigzi', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help=("Jaguar eigenmode fitting. Don't include off diagonal elements. "
              "Invert lowest eigenvalue."))
    data_args.add_argument(
        '-jh', type=str, nargs='+', action='append', default=[], metavar='file.in',
        help='Jaguar Hessian.')
    data_args.add_argument(
        '-jhi', type=str, nargs='+', action='append', default=[], metavar='file.in',
        help='Jaguar Hessian with inversion.')
    data_args.add_argument(
        '-jq', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar charges.')
    data_args.add_argument(
        '-jqh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar charges (ignores aliphatic hydrogens).')
    data_args.add_argument(
        '-r', type=str, nargs='+', action='append', default=[], metavar='filename',
        help=('Read data points directly (ex. use with .cal files). '
              'Each row corresponds to a data point. Columns may be '
              'separated by '
              "anything that Python's basic split method recognizes. "
              '1st column '
              'is the data label, 2nd column is the weight, and 3rd column is the '
              'value.'))
    data_args.add_argument(
        '-zm', type=str, nargs='+', action='append', default=[], metavar='parteth',
        help='Tether parameters away from zero. Force field data.')
    data_args.add_argument(
        '-zr', type=str, nargs='+', action='append', default=[], metavar='parteth',
        help='Tether parameters away from zero. Reference data.')
    return parser

def group_commands(commands):
    '''
    Determines how many data types are associated with each mae file. This
    enables us to run only one com file on each mae (instead of a separate com
    for each data type).
    '''
    commands_grouped = defaultdict(list)
    for calc_command, values in commands.iteritems():
        for comma_sep_filenames in chain.from_iterable(values):
            for filename in comma_sep_filenames.split(','):
                if filename.endswith('.mae'):
                    commands_grouped[filename].append(calc_command)
    # comment out later
    for filename, data_types in commands_grouped.iteritems():
        logger.log(1, '{}: {}'.format(filename, data_types))
    return commands_grouped

def make_macromodel_coms(commands_grouped, directory=os.getcwd()):
    '''
    Writes MacroModel com files. Uses the dictionary produced by group_commands.
    '''
    coms_to_run = []
    macromodel_indices = {}
    for filename, commands in commands_grouped.iteritems():
        # if set(commands).intersection(commands_macromodel):
        # should be faster
        if any(x in commands_macromodel for x in commands):
            indices_mae = []
            indices_mmo = []
            name_base = '.'.join(filename.split('.')[:-1])
            name_mae = name_base + '.q2mm.mae'
            name_mmo = name_base + '.q2mm.mmo'
            name_log = name_base + '.q2mm.log'
            name_com = name_base + '.q2mm.com'
            pre_structure = False
            hessian = False
            pre_energy = False
            optimization = False
            post_structure = False
            multiple_structures = False
            conf_search1 = False
            conf_search2 = False
            # would be faster if we had 2 sets of commands: a set for mae
            # files containing only 1 structure, and a set for mae files
            # containing multiple structures
            with open(os.path.join(directory, filename)) as f:
                number_structures = 0
                for line in f:
                    if 'f_m_ct {' in line:
                        number_structures += 1
                    if number_structures > 1:
                        multiple_structures = True
                        break
            # if set(commands).intersection(['ja', 'jb']):
            if any(x in ['ja', 'jb'] for x in commands):
                pre_structure = True
            # if set(commands).intersection(['me', 'me2', 'mq', 'mqh']):
            if any(x in ['me', 'me2', 'mq', 'mqh'] for x in commands):
                pre_energy = True
            # if set(commands).intersection(['meig', 'meigz', 'mh']):
            if any(x in ['meig', 'meigz', 'mh'] for x in commands):
                hessian = True
            # if set(commands).intersection(['ma', 'mb', 'meo']):
            if any(x in ['ma', 'mb', 'meo'] for x in commands):
                optimization = True
            # if set(commands).intersection(['ma', 'mb']):
            if any(x in ['ma', 'mb'] for x in commands):
                post_structure = True
            if any(x in ['mcs'] for x in commands):
                conf_search1 = True
            if any(x in ['mcs2'] for x in commands):
                conf_search2 = True
            com_string = '{}\n{}\n'.format(filename, name_mae)
            if conf_search1:
                com_string += cons.format_macromodel.format('MMOD', 0, 1, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('FFLD', 2, 1, 0, 0, 1, 0, 0, 0)
                com_string += cons.format_macromodel.format('BDCO', 0, 0, 0, 0, 41.5692, 99999., 0, 0)
                com_string += cons.format_macromodel.format('READ', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('CRMS', 0, 0, 0, 0, 0, 0.2500, 0, 0)
                com_string += cons.format_macromodel.format('MCMM', 10000, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('MCSS', 2, 0, 0, 0, 50., 0, 0, 0)
                com_string += cons.format_macromodel.format('MCOP', 1, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('DEMX', 0, 166, 0, 0, 50., 100., 0, 0)
                com_string += cons.format_macromodel.format('MSYM', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('AUTO', 0, 2, 1, 1, 0, -1., 0, 0)
                com_string += cons.format_macromodel.format('CONV', 2, 0, 0, 0, 0.5, 0, 0, 0)
                com_string += cons.format_macromodel.format('MINI', 9, 0, 500, 0, 0, 0, 0, 0)
            elif conf_search2:
                com_string += cons.format_macromodel.format('MMOD', 0, 1, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('DEBG', 55, 179, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('FFLD', 2, 1, 0, 0, 1, 0, 0, 0)
                com_string += cons.format_macromodel.format('BDCO', 0, 0, 0, 0, 41.5692, 99999., 0, 0)
                com_string += cons.format_macromodel.format('READ', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('CRMS', 0, 0, 0, 0, 0, 0.2500, 0, 0)
                com_string += cons.format_macromodel.format('LCMS', 10000, 0, 0, 0, 0, 0, 3, 6)
                com_string += cons.format_macromodel.format('NANT', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('MCNV', 1, 5, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('MCSS', 2, 0, 0, 0, 50, 0, 0, 0)
                com_string += cons.format_macromodel.format('MCOP', 1, 0, 0, 0, 0.5, 0, 0, 0)
                com_string += cons.format_macromodel.format('DEMX', 0, 833, 0, 0, 50, 100, 0, 0)
                com_string += cons.format_macromodel.format('MSYM', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('AUOP', 0, 0, 0, 0, 400, 0, 0, 0)
                com_string += cons.format_macromodel.format('AUTO', 0, 3, 1, 2, 1, 1, 4, 3)
                com_string += cons.format_macromodel.format('CONV', 2, 0, 0, 0, 0.05, 0, 0, 0)
                com_string += cons.format_macromodel.format('MINI', 1, 0, 2500, 0, 0, 0, 0, 0)
            else:
                com_string += cons.format_macromodel.format('FFLD', 2, 0, 0, 0, 0, 0, 0, 0)
                if hessian:
                    com_string += cons.format_macromodel.format('DEBG', 57, 210, 211, 0, 0, 0, 0, 0)
                else:
                    com_string += cons.format_macromodel.format('DEBG', 57, 0, 0, 0, 0, 0, 0, 0)
                if multiple_structures:
                    com_string += cons.format_macromodel.format('BGIN', 0, 0, 0, 0, 0, 0, 0, 0)
                com_string += cons.format_macromodel.format('READ', -1, 0, 0, 0, 0, 0, 0, 0)
                if pre_structure or pre_energy:
                    com_string += cons.format_macromodel.format('ELST', 1, 0, 0, 0, 0, 0, 0, 0)
                    indices_mmo.append('pre')
                    com_string += cons.format_macromodel.format('WRIT', 0, 0, 0, 0, 0, 0, 0, 0)
                    indices_mae.append('pre')
                if hessian:
                    com_string += cons.format_macromodel.format('MINI', 9, 0, 0, 0, 0, 0, 0, 0)
                    indices_mae.append('stupid_extra_structure')
                    com_string += cons.format_macromodel.format('RRHO', 3, 0, 0, 0, 0, 0, 0, 0)
                    indices_mae.append('hess')
                # if pre_structure or pre_energy or hessian:
                if optimization:
                    # this commented line is what was used in the code received from Elaine.
                    # arg1: 9 = TNCG, 1 = PRCG
                    # TNCG has risk that structures never converge, and may print NaN instead
                    # of coordinates and forces.
                    # com_string += cons.format_macromodel.format('MINI', 9, 0, 50, 0, 0, 0, 0, 0)
                    com_string += cons.format_macromodel.format('MINI', 1, 0, 500, 0, 0, 0, 0, 0) 
                    indices_mae.append('opt')
                if post_structure:
                    com_string += cons.format_macromodel.format('ELST', 1, 0, 0, 0, 0, 0, 0, 0)
                    # faily sure this indices_mae shouldn't be here
                    # indices_mae.append('post')
                    indices_mmo.append('opt')
                if multiple_structures:
                    com_string += cons.format_macromodel.format('END', 0, 0, 0, 0, 0, 0, 0, 0)
                macromodel_indices.update({name_mae: indices_mae})
                macromodel_indices.update({name_mmo: indices_mmo})
            with open(os.path.join(directory, name_com), 'w') as f:
                f.write(com_string)
            coms_to_run.append(name_com)
            logger.log(5, 'wrote {}'.format(os.path.join(directory, name_com)))
    # comment later
    for output_filename, data_indices in macromodel_indices.iteritems():
        logger.log(1, '{}: {}'.format(output_filename, data_indices))
    return coms_to_run, macromodel_indices

def run_macromodel(coms_to_run, directory=os.getcwd()):
    current_directory = os.getcwd()
    logger.log(5, 'moving to {}'.format(directory))
    os.chdir(directory)
    for com in coms_to_run:
        logger.log(6, 'running {}'.format(com))
        name = '.'.join(com.split('.')[:-1])
        success = False
        attempts = 0
        while success is False and attempts < 5:
            try:
                subprocess.check_output('bmin {} -WAIT'.format(name), shell=True)
            except subprocess.CalledProcessError as e:
                attempts += 1
                logger.warning('{} failed attempts: bmin {} -WAIT'.format(attempts, name))
                logger.warning('return code: {}'.format(e.returncode))
                logger.warning('output: {}'.format(e.output))
                # logger.warning('current directory: {}'.format(os.listdir(os.getcwd())))
                logger.warning(traceback.format_exc())
                time.sleep(10)
            except OSError as e:
                attempts += 1
                logger.warning('{} failed attempts: bmin {} -WAIT'.format(attempts, name))
                # logger.warning('current directory: {}'.format(os.listdir(os.getcwd())))
                logger.warning(traceback.format_exc())
                time.sleep(10)
            else:
                success = True
        if success is False:
            raise Exception('aborting due to > {} failed attempts'.format(attempts))
    logger.log(5, 'returning to {}'.format(current_directory))
    os.chdir(current_directory)

def collect_data(commands, macromodel_indices, directory=os.getcwd()):
    '''
    Return a list of data points.
    '''
    data = []
    file_storage = {}
    # would be nice if we could accept a force field as an argument
    # or we could only do this for the commands where it's needed
    if any(x in commands_need_ff for x in commands):
        ff = MM3(os.path.join(directory, 'mm3.fld'))
        ff.import_ff()

    if 'ja' in commands:
        for filenames in commands['ja']:
            for filename in filenames:
                data_temp = []
                name_base = '.'.join(filename.split('.')[:-1])
                name_mmo = name_base + '.q2mm.mmo'
                if filename in file_storage:
                    mmo = file_storage[filename]
                else:
                    mmo = filetypes.MacroModel(os.path.join(directory, name_mmo))
                    # mmo.import_structures()
                    file_storage[name_mmo] = mmo
                indices_output = macromodel_indices[name_mmo]
                indices_generator = iter(indices_output)
                for i, structure in enumerate(mmo.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        for angle in structure.angles:
                            if ff.sub_name in angle.comment:
                                data_temp.append(Datum(
                                    angle.value, 'ja', 'angle', filename, i=i, j=angle.atom_nums))
                data.extend(data_temp)
                logger.log(7, '{} ja from {}'.format(len(data_temp), name_mmo))

    if 'jb' in commands:
        for filenames in commands['jb']:
            for filename in filenames:
                data_temp = []
                name_base = '.'.join(filename.split('.')[:-1])
                name_mmo = name_base + '.q2mm.mmo'
                if filename in file_storage:
                    mmo = file_storage[filename]
                else:
                    mmo = filetypes.MacroModel(os.path.join(directory, name_mmo))
                    # mmo.import_structures()
                    file_storage[name_mmo] = mmo
                indices_output = macromodel_indices[name_mmo]
                indices_generator = iter(indices_output)
                for i, structure in enumerate(mmo.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        for bond in structure.bonds:
                            if ff.sub_name in bond.comment:
                                data_temp.append(Datum(
                                    bond.value, 'jb', 'bond', filename, i=i, j=bond.atom_nums))
                data.extend(data_temp)
                logger.log(7, '{} jb from {}'.format(len(data_temp), name_mmo))

    if 'je' in commands:
        for file_num, filenames in enumerate(commands['je']):
            data_temp = []
            for filename in filenames:
                if filename in file_storage:
                    mae = file_storage[filename]
                else:
                    mae = filetypes.Mae(os.path.join(directory, filename))
                    file_storage[filename] = mae
                for structure_num, structure in enumerate(mae.structures):
                    data_temp.append(Datum(
                        structure.props['r_j_Gas_Phase_Energy'] * cons.hartree_to_kjmol,
                        'je', 'energy', filename, group=file_num, i=structure_num))
            minimum = min([x.value for x in data_temp])
            for datum in data_temp:
                datum.value -= minimum
            data.extend(data_temp)
            logger.log(7, '{} je from {}'.format(len(data_temp), filenames))

    if 'je2' in commands:
        for i, filenames in enumerate(commands['je2']):
            data_temp = []
            for filename in filenames:
                if filename in file_storage:
                    mae = file_storage[filename]
                else:
                    mae = filetypes.Mae(os.path.join(directory, filename))
                    file_storage[filename] = mae
                for j, structure in enumerate(mae.structures):
                    data_temp.append(Datum(
                        structure.props['r_j_Gas_Phase_Energy'] * cons.hartree_to_kjmol,
                        'je2', 'energy2', filename, group=i, i=j))
            minimum = min([x.value for x in data_temp])
            for datum in data_temp:
                datum.value -= minimum
            data.extend(data_temp)
            logger.log(7, '{} je2 from {}'.format(len(data_temp), filenames))

    if 'jeig' in commands:
        for comma_filenames in commands['jeig']:
            for comma_filename in comma_filenames:
                name_in, name_out = comma_filename.split(',')
                if name_in in file_storage:
                    jin = file_storage[name_in]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, name_in))
                    file_storage[name_in] = jin
                if name_out in file_storage:
                    jout = file_storage[name_out]
                else:
                    jout = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = jout
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.load_from_jaguar_out(file_class=jout)
                hess.hessian = hess.mass_weight_hessian()
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()
                diagonal = np.diag(diagonal_matrix)
                lower_tri_indices = np.tril_indices_from(diagonal_matrix)
                lower_tri = diagonal_matrix[lower_tri_indices]
                data.extend(
                    [Datum(e, 'jeig', 'eig', (name_in, name_out), i=x, j=y) for e, x, y, in
                    izip(lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} jeig from {}'.format(len(lower_tri), (name_in, name_out)))

    if 'jeigi' in commands:
        for comma_filenames in commands['jeigi']:
            for comma_filename in comma_filenames:
                name_in, name_out = comma_filename.split(',')
                if name_in in file_storage:
                    jin = file_storage[name_in]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, name_in))
                    file_storage[name_in] = jin
                if name_out in file_storage:
                    jout = file_storage[name_out]
                else:
                    jout = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = jout
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.load_from_jaguar_out(file_class=jout)
                hess.hessian = hess.mass_weight_hessian()
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()
                diagonal = np.diag(diagonal_matrix)
                inv_diagonal = hess.replace_minimum(diagonal, value=cons.hessian_conversion)
                np.fill_diagonal(diagonal_matrix, inv_diagonal)
                lower_tri_indices = np.tril_indices_from(diagonal_matrix)
                lower_tri = diagonal_matrix[lower_tri_indices]
                data.extend(
                    [Datum(e, 'jeig', 'eig', (name_in, name_out), i=x, j=y) for e, x, y, in
                    izip(lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} jeig from {}'.format(len(lower_tri), (name_in, name_out)))

    if 'jeige' in commands:
        for comma_filenames in commands['jeige']:
            for comma_filename in comma_filenames:
                name_in, name_out = comma_filename.split(',')
                if name_in in file_storage:
                    jin = file_storage[name_in]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, name_in))
                    file_storage[name_in] = jin
                if name_out in file_storage:
                    jout = file_storage[name_out]
                else:
                    jout = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = jout
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.load_from_jaguar_out(file_class=jout)
                hess.hessian = hess.mass_weight_hessian()
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()

                diagonal_matrix_zero = np.diag(np.diag(diagonal_matrix)) 
                lower_tri_indices = np.tril_indices_from(diagonal_matrix_zero)
                lower_tri = diagonal_matrix_zero[lower_tri_indices]
                data.extend([Datum(e, 'jeige', 'eig', (name_in, name_out), i=x, j=y)
                             for e, x, y in izip(
                            lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} jeige from {}'.format(len(lower_tri), (name_in, name_out)))

    if 'jeigz' in commands:
        for comma_filenames in commands['jeigz']:
            for comma_filename in comma_filenames:
                name_in, name_out = comma_filename.split(',')
                if name_in in file_storage:
                    jin = file_storage[name_in]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, name_in))
                    file_storage[name_in] = jin
                if name_out in file_storage:
                    jout = file_storage[name_out]
                else:
                    jout = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = jout
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.load_from_jaguar_out(file_class=jout)
                hess.hessian = hess.mass_weight_hessian()
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()
                diagonal = np.diag(diagonal_matrix)
                data.extend([Datum(e, 'jeigz', 'eigz', (name_in, name_out), i=i, j=i)
                             for i, e in enumerate(diagonal)])
                logger.log(7, '{} jeigz from {}'.format(len(diagonal), (name_in, name_out)))

    if 'jeigzi' in commands:
        for comma_filenames in commands['jeigzi']:
            for comma_filename in comma_filenames:
                name_in, name_out = comma_filename.split(',')
                if name_in in file_storage:
                    jin = file_storage[name_in]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, name_in))
                    file_storage[name_in] = jin
                if name_out in file_storage:
                    jout = file_storage[name_out]
                else:
                    jout = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = jout
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.load_from_jaguar_out(file_class=jout)
                hess.hessian = hess.mass_weight_hessian()
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()
                diagonal = np.diag(diagonal_matrix)
                inv_diagonal = hess.replace_minimum(diagonal, value=cons.hessian_conversion)
                data.extend([Datum(e, 'jeigz', 'eigz', (name_in, name_out), i=i, j=i)
                             for i, e in enumerate(inv_diagonal)])
                logger.log(7, '{} jeigz from {}'.format(len(inv_diagonal), (name_in, name_out)))

                # Elaine/Per-Ola's code sets the off-diagonal elements to zero.
                # These elements are usually close to zero, but not quite zero.
                # I believe they also set the weights for these elements to zero
                # and/or set the off diagonal elements of the calculated
                # diagonalized matrix to zero. Either way, they didn't end up
                # influencing the penalty function, but they would make the list
                # of data points longer. Instead, this method just returns the
                # diagonal elements and ignores the off diagonal elements. Use
                # jeig if you don't want to ignore the off diagonal elements instead
                # of jeigz.

                # inv_diagonal_matrix_zeroed = np.diag(inv_diagonal)
                # lower_tri_indices = np.tril_indices_from(inv_diagonal_matrix_zeroed)
                # lower_tri = inv_diagonal_matrix_zeroed[lower_tri_indices]
                # data.extend(
                #     [Datum(e, 'jeigz', 'eig', (name_in, name_out), i=x, j=y) for e, x, y, in
                #     izip(lower_tri, lower_tri_indices[0], lower_tri_indices[1])])

    if 'jh' in commands:
        for filenames in commands['jh']:
            for filename in filenames:
                if filename in file_storage:
                    jin = file_storage[filename]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, filename))
                    # jin.import_structures()
                    file_storage[filename] = jin
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                matrix = hess.mass_weight_hessian()
                lower_tri_indices = np.tril_indices_from(matrix)
                lower_tri = matrix[lower_tri_indices]
                data.extend([Datum(e, 'jh', 'hess', filename, i=x, j=y) for e, x, y in izip(
                            lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} jh from {}'.format(len(lower_tri), filename))

    if 'jhi' in commands:
        for filenames in commands['jhi']:
            for filename in filenames:
                if filename in file_storage:
                    jin = file_storage[filename]
                else:
                    jin = filetypes.JaguarIn(os.path.join(directory, filename))
                    # jin.import_structures()
                    file_storage[filename] = jin
                hess = Hessian()
                hess.load_from_jaguar_in(file_class=jin)
                hess.hessian = hess.mass_weight_hessian()
                eigenvalues, eigenvectors = np.linalg.eigh(hess.hessian)
                inv_eigenvalues = hess.replace_minimum(eigenvalues)
                inv_hess = hess.diagonalize(matrix=np.diag(inv_eigenvalues), eigenvectors=eigenvectors)
                # this gets the same answer. sometimes it's nice to see the math
                # diagonal_matrix = hess.diagonalize(eigenvectors=eigenvectors)
                # diagonal = np.diag(diagonal_matrix)
                # inv_diagonal = hess.replace_minimum(diagonal)
                # inv_hess = hess.diagonalize(matrix=np.diag(inv_diagonal), eigenvectors=eigenvectors)
                lower_tri_indices = np.tril_indices_from(inv_hess)
                lower_tri = inv_hess[lower_tri_indices]
                data.extend([Datum(e, 'jhi', 'hess', filename, i=x, j=y) for e, x, y in izip(
                            lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} jhi from {}'.format(len(lower_tri), filename))

    if 'jqh' in commands:
        for filenames in commands['jqh']:
            for filename in filenames:
                data_temp = []
                if filename in file_storage:
                    mae = file_storage[filename]
                else:
                    mae = filetypes.Mae(os.path.join(directory, filename))
                    file_storage[filename] = mae
                aliph_hyd_inds = mae.get_aliph_hyds()
                for i, structure in enumerate(mae.structures):
                    for atom in structure.atoms:
                        if not atom.index in aliph_hyd_inds:
                            data_temp.append(
                                Datum(atom.partial_charge, 'jqh', 'charge', filename, i=i, j=atom.index))
                data.extend(data_temp)
                logger.log(7, '{} jqh from {}'.format(len(data_temp), filename))

    if 'jq' in commands:
        for filenames in commands['jq']:
            for filename in filenames:
                data_temp = []
                if filename in file_storage:
                    mae = file_storage[filename]
                else:
                    mae = filetypes.Mae(os.path.join(directory, filename))
                    file_storage[filename] = mae
                for i, structure in enumerate(mae.structures):
                    for atom in structure.atoms:
                        data_temp.append(Datum(atom.partial_charge, 'jq', 'charge', filename, i=i, j=atom.index))
                data.extend(data_temp)
                logger.log(7, '{} jq from {}'.format(len(data_temp), filename))

    if 'ma' in commands:
        for filenames in commands['ma']:
            for filename in filenames:
                data_temp = []
                name_base = '.'.join(filename.split('.')[:-1])
                name_mmo = name_base + '.q2mm.mmo'
                if name_mmo in file_storage:
                    mmo = file_storage[name_mmo]
                else:
                    mmo = filetypes.MacroModel(os.path.join(directory, name_mmo))
                    # mmo.import_structures()
                    file_storage[name_mmo] = mmo
                indices_output = macromodel_indices[name_mmo]
                indices_generator = iter(indices_output)
                for i, structure in enumerate(mmo.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'opt':
                        for angle in structure.angles:
                            if ff.sub_name in angle.comment:
                                data_temp.append(Datum(angle.value, 'ma', 'angle', filename, i=i, j=angle.atom_nums))
                data.extend(data_temp)
                logger.log(7, '{} ma from {}'.format(len(data_temp), filename))

    if 'mb' in commands:
        for filenames in commands['mb']:
            for filename in filenames:
                data_temp = []
                name_base = '.'.join(filename.split('.')[:-1])
                name_mmo = name_base + '.q2mm.mmo'
                if name_mmo in file_storage:
                    mmo = file_storage[name_mmo]
                else:
                    mmo = filetypes.MacroModel(os.path.join(directory, name_mmo))
                    # mmo.import_structures()
                    file_storage[name_mmo] = mmo
                indices_output = macromodel_indices[name_mmo]
                indices_generator = iter(indices_output)
                for i, structure in enumerate(mmo.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'opt':
                        for bond in structure.bonds:
                            if ff.sub_name in bond.comment:
                                data_temp.append(Datum(bond.value, 'mb', 'bond', filename, i=i, j=bond.atom_nums))
                data.extend(data_temp)
                logger.log(7, '{} mb from {}'.format(len(data_temp), filename))

    if 'me' in commands:
        for i, filenames in enumerate(commands['me']):
            data_temp = []
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_mae = name_base + '.q2mm.mae'
                if name_mae in file_storage:
                    mae = file_storage[name_mae]
                else:
                    mae = filetypes.Mae(os.path.join(directory, name_mae))
                    file_storage[name_mae] = mae
                indices_output = macromodel_indices[name_mae]
                indices_generator = iter(indices_output)
                for j, structure in enumerate(mae.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        data_temp.append(Datum(structure.props['r_mmod_Potential_Energy-MM3*'],
                                               'me', 'energy', name_mae, group=i, i=j))
            # relative energies aren't necessary right now but they can be nice
            # minimum = min([x.value for x in data_temp])
            # for datum in data_temp:
            #     datum.value -= minimum
            data.extend(data_temp)
            logger.log(7, '{} me from {}'.format(len(data_temp), filenames))

    if 'me2' in commands:
        for i, filenames in enumerate(commands['me2']):
            data_temp = []
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_mae = name_base + '.q2mm.mae'
                if name_mae in file_storage:
                    mae = file_storage[name_mae]
                else:
                    mae = filetypes.Mae(os.path.join(directory, name_mae))
                    file_storage[name_mae] = mae
                indices_output = macromodel_indices[name_mae]
                indices_generator = iter(indices_output)
                for j, structure in enumerate(mae.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        data_temp.append(Datum(structure.props['r_mmod_Potential_Energy-MM3*'],
                                               'me2', 'energy2', name_mae, group=i, i=j))
            # minimum = min([x.value for x in data_temp])
            # for datum in data_temp:
            #     datum.value -= minimum
            data.extend(data_temp)
            logger.log(7, '{} me2 from {}'.format(len(data_temp), filenames))

    if 'meo' in commands:
        for i, filenames in enumerate(commands['meo']):
            data_temp = []
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_mae = name_base + '.q2mm.mae'
                if name_mae in file_storage:
                    mae = file_storage[name_mae]
                else:
                    mae = filetypes.Mae(os.path.join(directory, name_mae))
                    file_storage[name_mae] = mae
                indices_output = macromodel_indices[name_mae]
                indices_generator = iter(indices_output)
                for j, structure in enumerate(mae.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'opt':
                        data_temp.append(Datum(structure.props['r_mmod_Potential_Energy-MM3*'],
                                               'meo', 'energy', name_mae, group=i, i=j))
            # relative energies aren't necessary right now but they can be nice
            # minimum = min([x.value for x in data_temp])
            # for datum in data_temp:
            #     datum.value -= minimum
            data.extend(data_temp)
            logger.log(7, '{} meo from {}'.format(len(data_temp), filenames))

    if 'meig' in commands:
        for comma_filenames in commands['meig']:
            for comma_filename in comma_filenames:
                name_mae, name_out = comma_filename.split(',')
                name_base = '.'.join(name_mae.split('.')[:-1])
                name_log = name_base + '.q2mm.log'
                if name_log in file_storage:
                    log = file_storage[name_log]
                else:
                    log = filetypes.MacroModelLog(os.path.join(directory, name_log))
                    file_storage[name_log] = log
                if name_out in file_storage:
                    out = file_storage[name_out]
                else:
                    out = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = out
                hess = Hessian()
                hess.load_from_jaguar_out(file_class=out, get_atoms=True)
                hess.load_from_mmo_log(file_class=log)
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()
                lower_tri_indices = np.tril_indices_from(diagonal_matrix)
                lower_tri = diagonal_matrix[lower_tri_indices]
                data.extend(
                    [Datum(e, 'meig', 'eig', (name_mae, name_out), i=x, j=y) for e, x, y, in
                    izip(lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} meig from {}'.format(len(lower_tri), (name_log, name_out)))

    if 'meigz' in commands:
        for comma_filenames in commands['meigz']:
            for comma_filename in comma_filenames:
                name_mae, name_out = comma_filename.split(',')
                name_base = '.'.join(name_mae.split('.')[:-1])
                name_log = name_base + '.q2mm.log'
                if name_log in file_storage:
                    log = file_storage[name_log]
                else:
                    log = filetypes.MacroModelLog(os.path.join(directory, name_log))
                    file_storage[name_log] = log
                if name_out in file_storage:
                    out = file_storage[name_out]
                else:
                    out = filetypes.JaguarOut(os.path.join(directory, name_out))
                    file_storage[name_out] = out
                hess = Hessian()
                hess.load_from_jaguar_out(file_class=out, get_atoms=True)
                hess.load_from_mmo_log(file_class=log)
                hess.eigenvectors = hess.mass_weight_eigenvectors()
                diagonal_matrix = hess.diagonalize()

                diagonal = np.diag(diagonal_matrix)
                data.extend([Datum(e, 'meigz', 'eigz', (name_mae, name_out), i=i, j=i)
                             for i, e in enumerate(diagonal)])

                # this way generates data similar to elaine's code. it has many off-diagonal
                # data points that are unused. use meig instead of meigz for that.
                # diagonal_matrix_zero = np.diag(np.diag(diagonal_matrix)) # off diagonal = zero
                # lower_tri_indices = np.tril_indices_from(diagonal_matrix_zero)
                # lower_tri = diagonal_matrix_zero[lower_tri_indices]
                # data.extend(
                #     [Datum(e, 'meigz', 'eig', (name_mae, name_out), i=x, j=y) for e, x, y, in
                #     izip(lower_tri, lower_tri_indices[0], lower_tri_indices[1])])

                logger.log(7, '{} meigz from {}'.format(len(diagonal), (name_log, name_out)))

    if 'mh' in commands:
        for filenames in commands['mh']:
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_log = name_base + '.q2mm.log'
                if name_log in file_storage:
                    log = file_storage[name_log]
                else:
                    log = filetypes.MacroModelLog(os.path.join(directory, name_log))
                    file_storage[name_log] = log
                hess = log.hessian
                lower_tri_indices = np.tril_indices_from(hess)
                lower_tri = hess[lower_tri_indices]
                data.extend(
                    [Datum(e, 'mh', 'hess', filename, i=x, j=y) for e, x, y in izip(
                    lower_tri, lower_tri_indices[0], lower_tri_indices[1])])
                logger.log(7, '{} mh from {}'.format(len(lower_tri), name_log))

    if 'mq' in commands:
        for filenames in commands['mq']:
            data_temp = []
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_mae = name_base + '.q2mm.mae'
                if name_mae in file_storage:
                    mae = file_storage[name_mae]
                else:
                    mae = filetypes.Mae(os.path.join(directory, name_mae))
                    file_storage[name_mae] = mae
                indices_output = macromodel_indices[name_mae]
                indices_generator = iter(indices_output)
                for i, structure in enumerate(mae.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        for atom in structure.atoms:
                            data_temp.append(Datum(atom.partial_charge, 'mq', 'charge', name_mae, i=i, j=atom.index))
            data.extend(data_temp)
            logger.log(7, '{} mq from {}'.format(len(data_temp), filenames))

    if 'mqh' in commands:
        for filenames in commands['mqh']:
            data_temp = []
            for filename in filenames:
                name_base = '.'.join(filename.split('.')[:-1])
                name_mae = name_base + '.q2mm.mae'
                if name_mae in file_storage:
                    mae = file_storage[name_mae]
                else:
                    mae = filetypes.Mae(os.path.join(directory, name_mae))
                    file_storage[name_mae] = mae
                indices_output = macromodel_indices[name_mae]
                indices_generator = iter(indices_output)
                aliph_hyd_inds = mae.get_aliph_hyds()
                for i, structure in enumerate(mae.structures):
                    try:
                        index_current = indices_generator.next()
                    except StopIteration:
                        indices_generator = iter(indices_output)
                        index_current = indices_generator.next()
                    if index_current == 'pre':
                        for atom in structure.atoms:
                            if not atom.index in aliph_hyd_inds:
                                data_temp.append(
                                    Datum(atom.partial_charge, 'mqh', 'charge', name_mae, i=i, j=atom.index))
            data.extend(data_temp)
            logger.log(7, '{} mqh from {}'.format(len(data_temp), filenames))

    if 'pm' in commands:
        for parteths in commands['pm']:
            for parteth in parteths:
                mm3_rows = []
                mm3_cols = []
                with open(os.path.join(directory, parteth)) as f:
                    for line in f:
                        line = line.partition('#')[0]
                        cols = line.split()
                        # if not line.startswith('#') and len(cols) > 0:
                        if len(cols) > 0:
                            mm3_rows.append(int(cols[0]))
                            mm3_cols.append(int(cols[1]))
                params_tethered = [x for x in ff.params if x.mm3_row in mm3_rows and x.mm3_col in mm3_cols]
                data_temp = [Datum(x.value, 'pm', 'parteth', parteth, i=x.mm3_row, j=x.mm3_col) for x in params_tethered]
                data.extend(data_temp)
                logger.log(7, '{} pm from {}'.format(len(data_temp), parteth))

    if 'pr' in commands:
        for parteths in commands['pr']:
            for parteth in parteths:
                mm3_rows = []
                mm3_cols = []
                values = []
                with open(os.path.join(directory, parteth)) as f:
                    for line in f:
                        line = line.partition('#')[0]
                        cols = line.split()
                        # if not line.startswith('#') and len(cols) > 0:
                        # if len(cols) > 0:
                        if cols:
                            mm3_rows.append(int(cols[0]))
                            mm3_cols.append(int(cols[1]))
                            values.append(float(cols[2]))
                data_temp = [Datum(value, 'pr', 'parteth', parteth, i=row, j=col) for value, row, col
                             in izip(values, mm3_rows, mm3_cols)]
                data.extend(data_temp)
                logger.log(7, '{} pr from {}'.format(len(data_temp), parteth))

    if 'r' in commands:
        for filenames in commands['r']:
            for filename in filenames:
                data_temp = []
                with open(os.path.join(directory, filename)) as f:
                    for line in f:
                        line = line.partition('#')[0]
                        label, weight, value = line.split()
                        data_temp.append(
                            Datum(float(value), 'r', 'read', filename, i=label, weight=float(weight)))
                data.extend(data_temp)
                logger.log(7, '{} r from {}'.format(len(data_temp), filename))

    if 'zm' in commands:
        for parteths in commands['zm']:
            for parteth in parteths:
                mm3_rows = []
                mm3_cols = []
                ayes = []
                bees = []
                with open(os.path.join(directory, parteth)) as f:
                    for line in f:
                        line = line.partition('#')[0]
                        cols = line.split()
                        # if not line.startswith('#') and len(cols) > 0:
                        if len(cols) > 0:
                            mm3_rows.append(int(cols[0]))
                            mm3_cols.append(int(cols[1]))
                            ayes.append(float(cols[2]))
                            bees.append(float(cols[3]))
                params_tethered = [x for x in ff.params if x.mm3_row in mm3_rows and x.mm3_col in mm3_cols]
                data_temp = [Datum(a * np.exp(- b * x.value * x.value),
                                   'zm', 'zeroteth', parteth, i=x.mm3_row, j=x.mm3_col) for x, a, b
                             in izip(params_tethered, ayes, bees)]
                data.extend(data_temp)
                logger.log(7, '{} zm from {}'.format(len(data_temp), parteth))

    if 'zr' in commands:
        for parteths in commands['zr']:
            for parteth in parteths:
                mm3_rows = []
                mm3_cols = []
                with open(os.path.join(directory, parteth)) as f:
                    for line in f:
                        line = line.partition('#')[0]
                        cols = line.split()
                        # if not line.startswith('#') and len(cols) > 0:
                        if len(cols) > 0:
                            mm3_rows.append(int(cols[0]))
                            mm3_cols.append(int(cols[1]))
                data_temp = [Datum(0., 'zr', 'zeroteth', parteth, i=row, j=col) for row, col in
                             izip(mm3_rows, mm3_cols)]
                data.extend(data_temp)
                logger.log(7, '{} zr from {}'.format(len(data_temp), parteth))

    if 'mcs' in commands:
        logger.log(7, 'not collecting data for mcs')
    logger.log(7, '{} data points'.format(len(data)))
    return data

def run_calculate(args):
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    commands = {key: value for key, value in opts.__dict__.iteritems() if key in commands_all and value}
    commands_grouped = group_commands(commands)
    coms_to_run, macromodel_indices = make_macromodel_coms(commands_grouped, opts.dir)
    if not opts.norun:
        run_macromodel(coms_to_run, opts.dir)
    data = collect_data(commands, macromodel_indices, opts.dir)
    if opts.printdata:
        for datum in sorted(data, key=datum_sort_key):
            print('{0:<25} {1:>22.6f} {2:>30}'.format(datum.name, datum.value, datum.source))
    return data

if __name__ == '__main__':
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    data = run_calculate(sys.argv[1:])
