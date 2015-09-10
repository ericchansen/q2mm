import argparse
import itertools
import logging
import logging.config
import numpy as np
import os
import random
import re
import sqlite3
import subprocess as sp
import sys
import textwrap
import time

import constants as co
import datatypes
import filetypes

# LOCATION OF SQLITE3 DATABASE MUST BE IN MEMORY (FOR NOW AT LEAST)!
DATABASE_LOC = ':memory:'
# Commands where we need to load the force field.
COM_LOAD_FF = ['ma', 'mb', 'mt', 'ja', 'jb', 'jt', 'pm', 'zm']
# Commands related to Gaussian.
COM_GAUSSIAN = []
# Commands related to Jaguar (Schrodinger).
COM_JAGUAR = ['je', 'je2', 'jeo', 'jeig', 'jeigi', 'jeige',
              'jeigz', 'jeigzi', 'jh', 'jhi', 'jq', 'jqh']
# Commands related to MacroModel (Schrodinger).
COM_MACROMODEL = ['ja', 'jb', 'jt', 'ma', 'mb', 'mcs', 'mcs2',
                  'mcs3', 'me', 'me2', 'meo', 'meig', 'meigz',
                  'mh', 'mq', 'mqh', 'mt']
# All other commands.
COM_OTHER = ['pm', 'pr', 'r', 'zm', 'zr']
# A list of all the possible commands.
COM_ALL = COM_GAUSSIAN + COM_JAGUAR + COM_MACROMODEL + COM_OTHER

logger = logging.getLogger(__name__)

def collect_data(commands, inps, ff_dir, sub_names=None):
    if any([x in COM_LOAD_FF for x in commands]):
        coms_need_ff = [x for x in commands if x in COM_LOAD_FF]
        if sub_names is None:
            logger.log(10, '  -- Must read FF for {}.'.format(
                    ' '.join(coms_need_ff)))
            ff = datatypes.import_ff(os.path.join(ff_dir, 'mm3.fld'))
    outs = {}
    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)
    for com, groups_of_filenames in commands.iteritems():
        if com in ['je', 'je2', 'jeo']:
            # Set the type.
            if com == 'je':
                typ = 'energy_1'
            elif com == 'je2':
                typ = 'energy_2'
            elif com == 'jeo':
                typ = 'energy_opt'
            # Move through files. Grouping matters here. Each group (idx_1) is
            # used to separately calculate relative energies.
            for idx_1, group_of_filenames in enumerate(groups_of_filenames):
                for filename in group_of_filenames:
                    # Currently this doesn't exist. inps[filename].filename is
                    # None.
                    # if inps[filename].filename not in outs:
                    if filename not in outs:
                        # Index the file so you don't read it more than once.
                        outs[filename] = \
                            filetypes.Mae(os.path.join(ff_dir, filename))
                    #     outs[inps[filename].filename] = \
                    #         filetypes.Mae(os.path.join(
                    #             inps[filename].directory,
                    #             inps[filename].filename))
                    # mae = outs[inps[filename].filename]
                    mae = outs[filename]
                    for str_num, struct in enumerate(mae.structures):
                        energy = {'val': (struct.props['r_j_Gas_Phase_Energy'] *
                                          co.HARTREE_TO_KJMOL),
                                  'com': com,
                                  'typ': typ,
                                  'src_1': filename,
                                  # 'src_1': inps[filename].filename,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)
        if com == 'jq':
            data_type = 'charge'
            for comma_sep_names in groups_of_filenames:
                for filename in comma_sep_names:
                    if filename in outs:
                        mae = outs[filename]
                    else:
                        mae = filetypes.Mae(os.path.join(
                                ff_dir, filename))
                        outs[filename] = mae
                    for i, structure in enumerate(mae.structures):
                        for atom in structure.atoms:
                            dp = {'val': atom.partial_charge,
                                  'com': com,
                                  'typ': data_type,
                                  'src_1': filename,
                                  'idx_1': i + 1,
                                  'atm_1': atom.index}
                            dp = co.set_data_defaults(dp)
                            c.execute(co.STR_SQLITE3, dp)
        if com == 'mq':
            data_type = 'charge'
            index = 'pre'
            for comma_sep_names in groups_of_filenames:
                for filename in comma_sep_names:
                    mae = inps[filename].name_mae
                    if mae not in outs:
                        outs[mae] = filetypes.Mae(
                            os.path.join(inps[filename].directory, mae))
                    mae = outs[mae]
                    structures = filetypes.select_structures(
                        mae.structures, inps[filename]._index_output_mae, index)
                    for str_num, structure in structures:
                        for atom in structure.atoms:
                            dp = {'val': atom.partial_charge,
                                  'com': com,
                                  'typ': data_type,
                                  'src_1': filename,
                                  'idx_1': str_num + 1,
                                  'atm_1': atom.index}
                            dp = co.set_data_defaults(dp)
                            c.execute(co.STR_SQLITE3, dp)
        if com in ['me', 'me2', 'meo']:
            # Set the type.
            if com == 'me':
                typ = 'energy_1'
            elif com == 'me2':
                typ = 'energy_2'
            elif com == 'meo':
                typ = 'energy_opt'
            # Set the index.
            if com in ['me', 'me2']:
                index = 'pre'
            elif com == 'meo':
                index = 'opt'
            # Move through files. Grouping matters here. Each group (idx_1) is
            # used to separately calculate relative energies.
            for idx_1, group_of_filenames in enumerate(groups_of_filenames):
                for filename in group_of_filenames:
                    if inps[filename].name_mae not in outs:
                        # Index the output file so you don't read it more than
                        # once.
                        outs[inps[filename].name_mae] = \
                                 filetypes.Mae(os.path.join(
                                    inps[filename].directory,
                                    inps[filename].name_mae))
                    mae = outs[inps[filename].name_mae]
                    selected = filetypes.select_structures(
                        mae.structures, inps[filename]._index_output_mae, index)
                    for str_num, struct in selected:
                        energy = {'val': struct.props['r_mmod_Potential_Energy-MM3*'],
                                  'com': com,
                                  'typ': typ,
                                  'src_1': inps[filename].name_mae,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)
        if com in ['ja', 'jb', 'jt', 'ma', 'mb', 'mt']:
            # Set the .mmo index.
            if com in ['ja', 'jb', 'jt']:
                index = 'pre'
            elif com in ['ma', 'mb', 'mt']:
                index = 'opt'
            # Set the type.
            if com in ['ja', 'ma']:
                typ = 'angles'
            elif com in ['jb', 'mb']:
                typ = 'bonds'
            elif com in ['jt', 'mt']:
                typ = 'torsions'
            # Move through files as you specified them on the command line.
            for group_of_filenames in groups_of_filenames:
                for filename in group_of_filenames:
                    # If 1st time accessing file, go ahead and do it. However,
                    # if you've already accessed it's data, don't read it again.
                    # Look it up in the dictionary instead.
                    if inps[filename].name_mmo not in outs:
                        outs[inps[filename].name_mmo] = \
                            filetypes.MacroModel(os.path.join(
                                inps[filename].directory,
                                inps[filename].name_mmo))
                    mmo = outs[inps[filename].name_mmo]
                    selected = filetypes.select_structures(
                        mmo.structures, inps[filename]._index_output_mmo, index)
                    data = []
                    for str_num, struct in selected:
                        temp = struct.select_stuff(
                            typ, com_match=ff.sub_names, com=com,
                            src_1=mmo.filename, idx_1=str_num + 1)
                        data.extend(temp)
                    c.executemany(co.STR_SQLITE3, data)
        if com in ['jeige', 'meig']:
            for group_of_filenames in groups_of_filenames:
                for comma_filenames in group_of_filenames:
                    if com == 'meig':
                        name_mae, name_out = comma_filenames.split(',')
                        name_log = inps[name_mae].name_log
                        if name_log not in outs:
                            outs[name_log] = filetypes.MacroModelLog(
                                os.path.join(inps[name_mae].directory,
                                             inps[name_mae].name_log))
                        log = outs[name_log]
                    elif com == 'jeige':
                        name_in, name_out = comma_filenames.split(',')
                        if name_in not in outs:
                            outs[name_in] = filetypes.JaguarIn(
                                os.path.join(ff_dir, name_in))
                        jin = outs[name_in]
                    if name_out not in outs:
                        outs[name_out] = filetypes.JaguarOut(os.path.join(
                                ff_dir, name_out))
                    out = outs[name_out]
                    if com == 'jeige':
                        hess = datatypes.Hessian(jin, out)
                        hess.mass_weight_hessian()
                    elif com == 'meig':
                        hess = datatypes.Hessian(log, out)
                    hess.mass_weight_eigenvectors()
                    hess.diagonalize()
                    if com == 'jeige':
                        diagonal_matrix = np.diag(np.diag(hess.hessian))
                    else:
                        diagonal_matrix = hess.hessian
                    low_tri_idx = np.tril_indices_from(diagonal_matrix)
                    lower_tri = diagonal_matrix[low_tri_idx]
                    if com == 'jeige':
                        src_1 = name_in
                    elif com == 'meig':
                        src_1 = name_mae
                    data = [{'val': e,
                             'com': com,
                             'typ': 'eig',
                             'src_1': src_1,
                             'src_2': name_out,
                             'idx_1': x + 1,
                             'idx_2': y + 1
                             }
                            for e, x, y in itertools.izip(
                            lower_tri, low_tri_idx[0], low_tri_idx[1])]
                    data = [co.set_data_defaults(x) for x in data]
                    c.executemany(co.STR_SQLITE3, data)
    c.execute('SELECT Count(*) FROM data')
    count_data = c.fetchone()
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(list(count_data)[0]))
    conn.commit()
    return conn

def get_label(row):
    '''Returns a string that serves as the label for a given data point.'''
    atoms = [row['atm_1'], row['atm_2'], row['atm_3'], row['atm_4']]
    atoms = filter(lambda x: x is not None, atoms)
    if atoms:
        atoms = '-'.join(map(str, atoms))
    index = [row['idx_1'], row['idx_2']]
    index = filter(lambda x: x is not None, index)
    if index:
        index = '-'.join(map(str, index))
    if atoms:
        label = '{}_{}_{}_{}'.format(
            row['typ'], row['src_1'].split('.')[0], index, atoms)
    else:
        label = '{}_{}_{}'.format(
            row['typ'], row['src_1'].split('.')[0], index)
    return label

def pretty_conn(conn):
    ''''Prints the data in a table.'''
    logger.log(20, '--' + ' Label '.center(22, '-') +
               '--' + ' Value '.center(22, '-') + '--')
    c = conn.cursor()
    c.execute('SELECT * FROM data ORDER BY typ, src_1, src_2, idx_1, idx_2, '
              'atm_1, atm_2, atm_3, atm_4')
    for row in c.fetchall():
        logger.log(20, '  ' + '{:22s}'.format(get_label(row)) +
                   '  ' + '{:22.4f}'.format(row['val']))
    logger.log(20, '-' * 50)

def pretty_commands_for_files(commands_for_files, level=5):
    '''Pretty verbosity for the .mae commands dictionary.'''
    if logger.getEffectiveLevel() <= level:
        foobar = textwrap.TextWrapper(width=48, subsequent_indent=' '*26)
        logger.log(
            level,
            '--' + ' Filename '.center(22, '-') +
            '--' + ' Commands '.center(22, '-') +
            '--')
        for filename, commands in commands_for_files.iteritems():
            foobar.initial_indent = '  {:22s}  '.format(filename)
            logger.log(level, foobar.fill(' '.join(commands)))
        logger.log(level, '-'*50)

def pretty_commands(commands, level=5):
    '''Pretty verbosity for the commands dictionary.'''
    if logger.getEffectiveLevel() <= level:
        foobar = textwrap.TextWrapper(width=48, subsequent_indent=' '*24)
        logger.log(
            level,
            '--' + ' Command '.center(9, '-') +
            '--' + ' Group # '.center(9, '-') +
            '--' + ' Filenames '.center(24, '-') + 
            '--')
        for command, groups_of_filenames in commands.iteritems():
            for i, filenames in enumerate(groups_of_filenames):
                if i == 0:
                    foobar.initial_indent = \
                        '  {:9s}  {:^9d}  '.format(command, i+1)
                else:
                    foobar.initial_indent = \
                        '  ' + ' '*9 + '  ' + '{:^9d}  '.format(i+1)
                logger.log(level, foobar.fill(' '.join(filenames)))
        logger.log(level, '-'*50)

def sort_commands_by_filename(commands):
    '''
    Takes a dictionary of commands like...

     {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'], ['b1.01.mae', 'b2.01.mae']],
      'mb': [['a1.01.mae'], ['b1.01.mae']],
      'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
     }
    
    ... and turn it into a dictionary that looks like...

    {'a1.01.mae': ['me', 'mb'],
     'a1.01.in': ['jeig'],
     'a1.out': ['jeig'],
     'a2.01.mae': ['me'],
     'a3.01.mae': ['me'],
     'b1.01.mae': ['me', 'mb'],
     'b1.01.in': ['jeig'],
     'b1.out': ['jeig'],
     'b2.01.mae': ['me']
    }
    '''
    sorted_commands = {}
    for command, groups_of_filenames in commands.iteritems():
        for comma_separated in itertools.chain.from_iterable(groups_of_filenames):
            for filename in comma_separated.split(','):
                if filename in sorted_commands:
                    sorted_commands[filename].append(command)
                else:
                    sorted_commands[filename] = [command]
    return sorted_commands

def generate_fake_conn(commands, inps, ff_dir, sub_names=None):
    outs = {}
    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)
    for com, groups_of_filenames in commands.iteritems():
        if com in ['je', 'je2', 'jeo']:
            if com == 'je':
                typ = 'energy_1'
            elif com == 'je2':
                typ = 'energy_2'
            elif com == 'jeo':
                typ = 'energy_opt'
            for idx_1, group_of_filenames in enumerate(groups_of_filenames):
                for filename in group_of_filenames:
                    if filename not in outs:
                        outs[filename] = \
                            filetypes.Mae(os.path.join(ff_dir, filename))
                    mae = outs[filename]
                    for str_num, struct in enumerate(mae.structures):
                        energy = {'val': random.gauss(2, 3),
                                  'com': com,
                                  'typ': typ,
                                  'src_1': filename,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)
        if com in ['me', 'me2', 'meo']:
            if com == 'me':
                typ = 'energy_1'
            elif com == 'me2':
                typ = 'energy_2'
            elif com == 'meo':
                typ = 'energy_opt'
            if com in ['me', 'me2']:
                index = 'pre'
            elif com == 'meo':
                index = 'opt'
            for idx_1, group_of_filenames in enumerate(groups_of_filenames):
                for filename in group_of_filenames:
                    if inps[filename].name_mae not in outs:
                        outs[inps[filename].name_mae] = \
                                 filetypes.Mae(os.path.join(
                                    inps[filename].directory,
                                    inps[filename].name_mae))
                    mae = outs[inps[filename].name_mae]
                    selected = filetypes.select_structures(
                        mae.structures,
                        inps[filename]._index_output_mae, index)
                    for str_num, struct in selected:
                        energy = {'val': random.gauss(2, 3),
                                  'com': com,
                                  'typ': typ,
                                  'src_1': inps[filename].name_mae,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)
    c.execute('SELECT Count(*) FROM data')
    count_data = c.fetchone()
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(list(count_data)[0]))
    conn.commit()
    return conn

def return_calculate_parser(add_help=True, parents=None):
    '''
    Returns an argument parser.
    
    Uses command line arguments to select types of data to generate.
    The data is used to evaluate the objective/penalty function.
    '''
    # Necessary? I know there can sometimes be problems with lists
    # as defaults.
    if parents is None:
        parents = []
    # Whether or not to add help. You may not want to add help if
    # these arguments are being used in another, higher level parser.
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    # General options. Perhaps directory shouldn't be used this way.
    args_calc = parser.add_argument_group("calculate options")
    args_calc.add_argument(
        '--directory', '-d', type=str, metavar='path', default=os.getcwd(),
        help=('Directory to search for files (.mae, .log, mm3.fld, etc.).'
              '3rd party calculations are executed from this directory.'))
    args_calc.add_argument(
        '--doprint', '-p', action='store_true',
        help="Print data.")
    args_calc.add_argument(
        '--norun', '-n', action='store_true',
        help="Don't run 3rd party software.")
    args_calc.add_argument(
        '--fake', '-f', action='store_true',
        help="Generate fake data.")
    # Each option corresponds to a particular type of data.
    args_data = parser.add_argument_group("calculate data types")
    args_data.add_argument(
        '-ma', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel angles (post-FF optimization).')
    args_data.add_argument(
        '-mb', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel bond lengths (post-FF optimization).')
    args_data.add_argument(
        '-mcs', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Run a MacroModel conformational search. For ease of use only. '
              "Doesn't work as a data type for FF optimizations."))
    args_data.add_argument(
        '-mcs2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Run a MacroModel conformational search. For ease of use only. '
              "Doesn't work as a data type for FF optimizations."
              'Uses AUOP cutoff for number of steps.'))
    args_data.add_argument(
        '-mcs3', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Run a MacroModel conformational search. For ease of use only. '
              "Doesn't work as a data type for FF optimizations."
              'Maximum of 4000 steps and no AUOP cutoff.'))
    args_data.add_argument(
        '-me', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel energies (pre-FF optimization).')
    args_data.add_argument(
        '-me2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help=('Same as -me, but uses a separate weight.'))
    args_data.add_argument(
        '-meo', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel energies (post-FF optimization).')
    args_data.add_argument(
        '-meig', type=str, nargs='+', action='append',
        default=[], metavar='file.mae,file.out',
        help='MacroModel eigenmatrix (all elements).')
    # args_data.add_argument(
    #     '-meigz', type=str, nargs='+', action='append',
    #     default=[], metavar='file.mae,file.out',
    #     help="MacroModel eigenmatrix (diagonal elements).")
    # args_data.add_argument(
    #     '-mh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
    #     help='MacroModel Hessian.')
    args_data.add_argument(
        '-mq', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel charges.')
    # args_data.add_argument(
    #     '-mqh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
    #     help='MacroModel charges (excludes aliphatic hydrogens).')
    args_data.add_argument(
        '-mt', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='MacroModel torsions (post-FF optimization).')
    # args_data.add_argument(
    #     '-pm', type=str, nargs='+', action='append', default=[], metavar='parteth',
    #     help='Tethering of parameters for FF data.')
    # args_data.add_argument(
    #     '-pr', type=str, nargs='+', action='append', default=[], metavar='parteth',
    #     help='Tethering of parameters for reference data.')
    args_data.add_argument(
        '-ja', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar angles.')
    args_data.add_argument(
        '-jb', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar bond lengths.')
    args_data.add_argument(
        '-je', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar energies.')
    args_data.add_argument(
        '-je2', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Same as -je, but uses a separate weight.')
    args_data.add_argument(
        '-jeo', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar energies. Same as -je, except the files selected by this '
        'command will have their energies compared to those selected by -meo.')
    # args_data.add_argument(
    #     '-jeig', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
    #     help='Jaguar eigenmatrix (all elements).')
    # args_data.add_argument(
    #     '-jeigi', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
    #     help='Jaguar eigenmatrix (all elements). Invert 1st eigenvalue.')
    args_data.add_argument(
        '-jeige', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
        help=('Jaguar eigenmatrix. Incluldes all elements, but zeroes those '
              'that are off-diagonal.'))
    # args_data.add_argument(
    #     '-jeigz', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
    #     help="Jaguar eigenmatrix (only diagonal elements).")
    # args_data.add_argument(
    #     '-jeigzi', type=str, nargs='+', action='append', default=[], metavar='file.in,file.out',
    #     help="Jaguar eigenmatrix (only diagonal elements). Invert 1st eigenvalue.")
    # args_data.add_argument(
    #     '-jh', type=str, nargs='+', action='append', default=[], metavar='file.in',
    #     help='Jaguar Hessian.')
    # args_data.add_argument(
    #     '-jhi', type=str, nargs='+', action='append', default=[], metavar='file.in',
    #     help='Jaguar Hessian with inversion.')
    args_data.add_argument(
        '-jq', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar partial charges.')
    # args_data.add_argument(
    #     '-jqh', type=str, nargs='+', action='append', default=[], metavar='file.mae',
    #     help='Jaguar charges (excludes aliphatic hydrogens).')
    args_data.add_argument(
        '-jt', type=str, nargs='+', action='append', default=[], metavar='file.mae',
        help='Jaguar torsions.')
    # args_data.add_argument(
    #     '-r', type=str, nargs='+', action='append', default=[], metavar='filename',
    #     help=('Read data points directly (ex. use with .cal files). '
    #           'Each row corresponds to a data point. Columns are separated '
    #           'by spaces. 1st column is the data label, 2nd column is the '
    #           'weight, and 3rd column is the value.'))
    # args_data.add_argument(
    #     '-zm', type=str, nargs='+', action='append', default=[], metavar='parteth',
    #     help='Tether parameters away from zero. FF data.')
    # args_data.add_argument(
    #     '-zr', type=str, nargs='+', action='append', default=[], metavar='parteth',
    #     help='Tether parameters away from zero. Reference data.')
    return parser

def main(args):
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    # commands looks like:
    # {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'], 
    #         ['b1.01.mae', 'b2.01.mae']],
    #  'mb': [['a1.01.mae'], ['b1.01.mae']],
    #  'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
    # }
    commands = {key: value for key, value in opts.__dict__.iteritems() if key
                in COM_ALL and value}
    pretty_commands(commands)
    # commands_for_filenames looks like:
    # {'a1.01.mae': ['me', 'mb'],
    #  'a1.01.in': ['jeig'],
    #  'a1.out': ['jeig'],
    #  'a2.01.mae': ['me'],
    #  'a3.01.mae': ['me'],
    #  'b1.01.mae': ['me', 'mb'],
    #  'b1.01.in': ['jeig'],
    #  'b1.out': ['jeig'],
    #  'b2.01.mae': ['me']
    # }
    commands_for_filenames = sort_commands_by_filename(commands)
    pretty_commands_for_files(commands_for_filenames)
    # inps looks like:
    # {'a1.01.mae': <__main__.Mae object at 0x1110e10>,
    #  'a1.01.in': None,
    #  'a1.out': None,
    #  'a2.01.mae': <__main__.Mae object at 0x1733b23>,
    #  'a3.01.mae': <__main__.Mae object at 0x1853e12>,
    #  'b1.01.mae': <__main__.Mae object at 0x2540e10>,
    #  'b1.01.in': None,
    #  'b1.out': None,
    #  'b2.01.mae': <__main__.Mae object at 0x1353e11>,
    # }
    inps = {}
    for filename, commands_for_filename in commands_for_filenames.iteritems():
        if any(x in COM_MACROMODEL for x in commands_for_filename):
            if os.path.splitext(filename)[1] == '.mae':
                inps[filename] = filetypes.Mae(
                    os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
                inps[filename].write_com()
        else:
            inps[filename] = None
    # Run external software if need be.
    if opts.norun or opts.fake:
        logger.log(15, "  -- Skipping external software calculations. ")
    else:
        for filename, some_class in inps.iteritems():
            if some_class is not None:
                some_class.run_com()
    if opts.fake:
        conn = generate_fake_conn(commands, inps, opts.directory)
        logger.log(15, '  -- Generated fake data.')
    else:
        conn = collect_data(commands, inps, opts.directory)
    if opts.doprint:
        pretty_conn(conn)
    return conn

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
    
