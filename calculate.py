"""
Extracts data from various files based upon the user's input.

This script takes a sequence of keywords corresponding to various
datatypes (ex. mb = MacroModel bond lengths) followed by filenames,
and extracts that particular data type from the file.

Also handles running 3rd party applications to calculate force
field data (ex.  writing and running .com files for MacroModel).
"""
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

def main(args):
    """
    Main control for calculate module.

    Arguments
    ---------
    args : string
           Used by argparse.ArgumentParser to determine
           what to do.
    """
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
    beautiful_commands(commands)
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
    beautiful_commands_for_files(commands_for_filenames)
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
    if opts.norun or opts.fake:
        logger.log(15, "  -- Skipping external software calculations. ")
    else:
        for filename, some_class in inps.iteritems():
            if some_class is not None:
                some_class.run(check_tokens=opts.check)
    if opts.fake:
        conn = gather_fake_data(commands, inps, opts.directory)
        logger.log(15, '  -- Generated fake data.')
    else:
        conn = gather_data(
            commands, inps, opts.directory, opts.ffpath, opts.subnames)
    if opts.doprint:
        beautiful_conn(conn)
    return conn

def gather_data(commands, inps, directory, ff_path=None, sub_names=None):
    """
    Gathers data from files. Knows what to do based upon dictionary
    `commands`. Uses dictionary `inps` to keep track of files that may
    have been generated. Reads files and stores the data in dictionary
    `outs` to prevent having to rereading files.

    Arguments
    ---------
    commands : dictionary
    inps : dictionary
    directory : string
    ff_path : string
    sub_names : list of strings or None

    Returns
    -------
    connection to sqlite3 database
    """
    ff_coms = [x for x in commands if x in COM_LOAD_FF]
    if sub_names is None and ff_coms:
        logger.log(5, '  -- Must read force field for datatypes {}.'.format(
                ', '.join(ff_coms)))
        sub_names = get_sub_names(commands, ff_path)

    outs = {}

    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)

    for com, groups_filenames in commands.iteritems():

        # ----- REFERENCE DATA FILE -----
        if com == 'r':
            for filename in groups_filenames:
                ref = filetypes.Reference(filename[0])
                data = ref.get_data()
                for datum in data:
                    datum = co.set_data_defaults(datum)
                    c.execute(co.STR_SQLITE3, datum)
        # ----- JAGUAR ENERGIES -----
        if com in ['je', 'je2', 'jeo']:
            if com == 'je': typ = 'energy_1'
            elif com == 'je2': typ = 'energy_2'
            elif com == 'jeo': typ = 'energy_opt'
            # Move through files. Grouping matters here. Each group (idx_1)
            # is used to separately calculate relative energies.
            for idx_1, group_filenames in enumerate(groups_filenames):
                for filename in group_filenames:
                    if filename not in outs:
                        outs[filename] = \
                            filetypes.Mae(os.path.join(directory, filename))
                    mae = outs[filename]
                    for str_num, struct in enumerate(mae.structures):
                        energy = {'val': (struct.props['r_j_Gas_Phase_Energy'] *
                                          co.HARTREE_TO_KJMOL),
                                  'com': com,
                                  'typ': typ,
                                  'src_1': filename,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)
        # ----- JAGUAR CHARGES -----
        if com in ['jq', 'jqh']:
            for comma_sep_names in groups_filenames:
                for filename in comma_sep_names:
                    if filename not in outs:
                        outs[filename] = filetypes.Mae(os.path.join(
                                directory, filename))
                    mae = outs[filename]
                    for i, structure in enumerate(mae.structures):
                        if com == 'jqh':
                            aliph_hyds = structure.get_aliph_hyds()
                            aliph_hyd_inds = [x.index for x in aliph_hyds]
                        for atom in structure.atoms:
                            if not 'b_q_use_charge' in atom.props or \
                                    atom.props['b_q_use_charge']:
                                q = atom.partial_charge
                                if com == 'jqh':
                                    for bonded_atom_ind in \
                                            atom.bonded_atom_indices:
                                        if bonded_atom_ind in aliph_hyd_inds:
                                            q += \
                                                structure.atoms[
                                                bonded_atom_ind - 
                                                1].partial_charge
                                if com == 'jq' or not atom in aliph_hyds:
                                    dp = {'val': q,
                                          'com': com,
                                          'typ': 'charge',
                                          'src_1': filename,
                                          'idx_1': i + 1,
                                          'atm_1': atom.index}
                                    dp = co.set_data_defaults(dp)
                                    c.execute(co.STR_SQLITE3, dp)

        # ----- MACROMODEL CHARGES -----
        if com in ['mq', 'mqh']:
            for comma_sep_names in groups_filenames:
                for filename in comma_sep_names:
                    mae = inps[filename].name_mae
                    if mae not in outs:
                        outs[mae] = filetypes.Mae(
                            os.path.join(inps[filename].directory, mae))
                    mae = outs[mae]
                    structures = filetypes.select_structures(
                        mae.structures, inps[filename]._index_output_mae, 'pre')
                    for str_num, structure in structures:
                        if com == 'mqh':
                            aliph_hyds = structure.get_aliph_hyds()
                        for atom in structure.atoms:
                            if not 'b_q_use_charge' in atom.props or \
                                    atom.props['b_q_use_charge']:
                                if com == 'mq' or not atom in aliph_hyds:
                                    dp = {'val': atom.partial_charge,
                                          'com': com,
                                          'typ': 'charge',
                                          'src_1': filename,
                                          'idx_1': str_num + 1,
                                          'atm_1': atom.index}
                                    dp = co.set_data_defaults(dp)
                                    c.execute(co.STR_SQLITE3, dp)

        # ----- MACROMODEL ENERGIES -----
        if com in ['me', 'me2', 'meo']:
            if com == 'me': typ = 'energy_1'
            elif com == 'me2': typ = 'energy_2'
            elif com == 'meo': typ = 'energy_opt'
            if com in ['me', 'me2']: ind = 'pre'
            elif com == 'meo': ind = 'opt'
            for idx_1, group_filenames in enumerate(groups_filenames):
                for filename in group_filenames:
                    if inps[filename].name_mae not in outs:
                        outs[inps[filename].name_mae] = \
                                 filetypes.Mae(os.path.join(
                                    inps[filename].directory,
                                    inps[filename].name_mae))
                    mae = outs[inps[filename].name_mae]
                    selected = filetypes.select_structures(
                        mae.structures, inps[filename]._index_output_mae, ind)
                    for str_num, struct in selected:
                        energy = {'val': struct.props['r_mmod_Potential_Energy-MM3*'],
                                  'com': com,
                                  'typ': typ,
                                  'src_1': inps[filename].name_mae,
                                  'idx_1': idx_1 + 1,
                                  'idx_2': str_num + 1}
                        energy = co.set_data_defaults(energy)
                        c.execute(co.STR_SQLITE3, energy)

        # ----- SCHRODINGER STRUCTURES -----
        if com in ['ja', 'jb', 'jt', 'ma', 'mb', 'mt']:
            if com in ['ja', 'jb', 'jt']: index = 'pre'
            elif com in ['ma', 'mb', 'mt']: index = 'opt'
            if com in ['ja', 'ma']: typ = 'angles'
            elif com in ['jb', 'mb']: typ = 'bonds'
            elif com in ['jt', 'mt']: typ = 'torsions'
            # Move through files as you specified them on the command line.
            for group_filenames in groups_filenames:
                for filename in group_filenames:
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
                            typ, com_match=sub_names, com=com,
                            src_1=mmo.filename, idx_1=str_num + 1)
                        data.extend(temp)
                    c.executemany(co.STR_SQLITE3, data)

        # ------ SCHRODINGER EIGENMATRIX -----
        if com in ['jeigz', 'meig']:
            for group_filenames in groups_filenames:
                for comma_filenames in group_filenames:
                    if com == 'meig':
                        name_mae, name_out = comma_filenames.split(',')
                        name_log = inps[name_mae].name_log
                        if name_log not in outs:
                            outs[name_log] = filetypes.MacroModelLog(
                                os.path.join(inps[name_mae].directory,
                                             inps[name_mae].name_log))
                        log = outs[name_log]
                    elif com == 'jeigz':
                        print(comma_filenames)
                        name_in, name_out = comma_filenames.split(',')
                        if name_in not in outs:
                            outs[name_in] = filetypes.JaguarIn(
                                os.path.join(directory, name_in))
                        jin = outs[name_in] 
                    if name_out not in outs:
                        outs[name_out] = filetypes.JaguarOut(os.path.join(
                                directory, name_out))
                    out = outs[name_out]
                    if com == 'jeigz':
                        hess = datatypes.Hessian(jin, out)
                        hess.mass_weight_hessian()
                    elif com == 'meig':
                        hess = datatypes.Hessian(log, out)
                    hess.mass_weight_eigenvectors()
                    hess.diagonalize()
                    if com == 'jeigz':
                        diagonal_matrix = np.diag(np.diag(hess.hessian))
                    else:
                        diagonal_matrix = hess.hessian
                    low_tri_idx = np.tril_indices_from(diagonal_matrix)
                    lower_tri = diagonal_matrix[low_tri_idx]
                    if com == 'jeigz':
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

def get_sub_names(commands, ff_path):
    """
    Some datatypes only work when we also know the names of the
    substructures containing parameters that are being optimized.
    If the substructure names aren't provided by the user, this will
    read the force field for those names.

    Arguments
    ---------
    commands : dictionary
    ff_path : string
    """
    assert ff_path is not None, \
        "User didn't provide required substructure names " + \
        "or path to force field!"
    ff = datatypes.MM3(ff_path)
    ff.import_ff()
    # ff = datatypes.import_ff(ff_path)
    return ff.sub_names

def beautiful_conn(conn, log_level=20):
    """
    Logs data as a table.

    Arguments
    ---------
    conn : sqlite3 connection object
    log_level : int
            Logging level used.
    """
    logger.log(20, '--' + ' LABEL '.center(22, '-') +
               '--' + ' VALUE '.center(22, '-') + '--')
    c = conn.cursor()
    c.execute('SELECT * FROM data ORDER BY typ, src_1, src_2, '
              'idx_1, idx_2, atm_1, atm_2, atm_3, atm_4')
    for row in c.fetchall():
        logger.log(20, '  ' + '{:22s}'.format(get_label(row)) +
                   '  ' + '{:22.4f}'.format(row['val']))
    logger.log(20, '-' * 50)

def beautiful_commands_for_files(commands_for_files, log_level=5):
    """
    Logs the .mae commands dictionary, or the all of the commands
    used on a particular file.

    Arguments
    ---------
    commands_for_files : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = textwrap.TextWrapper(
            width=48, subsequent_indent=' '*26)
        logger.log(
            log_level,
            '--' + ' FILENAME '.center(22, '-') +
            '--' + ' COMMANDS '.center(22, '-') +
            '--')
        for filename, commands in commands_for_files.iteritems():
            foobar.initial_indent = '  {:22s}  '.format(filename)
            logger.log(log_level, foobar.fill(' '.join(commands)))
        logger.log(log_level, '-'*50)

def beautiful_commands(commands, log_level=5):
    """
    Logs the arguments/commands given to calculate that are used
    to request particular datatypes from particular files.

    Arguments
    ---------
    commands : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = textwrap.TextWrapper(width=48, subsequent_indent=' '*24)
        logger.log(
            log_level,
            '--' + ' COMMAND '.center(9, '-') +
            '--' + ' GROUP # '.center(9, '-') +
            '--' + ' FILENAMES '.center(24, '-') + 
            '--')
        for command, groups_filenames in commands.iteritems():
            for i, filenames in enumerate(groups_filenames):
                if i == 0:
                    foobar.initial_indent = \
                        '  {:9s}  {:^9d}  '.format(command, i+1)
                else:
                    foobar.initial_indent = \
                        '  ' + ' '*9 + '  ' + '{:^9d}  '.format(i+1)
                logger.log(log_level, foobar.fill(' '.join(filenames)))
        logger.log(log_level, '-'*50)

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

    Arguments
    ---------
    commands : dic

    Returns
    -------
    dictionary of the sorted commands
    '''
    sorted_commands = {}
    for command, groups_filenames in commands.iteritems():
        for comma_separated in itertools.chain.from_iterable(
            groups_filenames):
            for filename in comma_separated.split(','):
                if filename in sorted_commands:
                    sorted_commands[filename].append(command)
                else:
                    sorted_commands[filename] = [command]
    return sorted_commands

def gather_fake_data(commands, inps, directory):
    """
    Similar to `gather_data`, but instead generates randomly selected,
    fake data. Used solely for testing purposes, as sometimes generating
    the actual data isn't necessary and takes a long time.

    Arguments
    ---------
    commands : dictionary
    inps : dictionary
    directory : string
    """
    outs = {}
    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)
    for com, groups_filenames in commands.iteritems():

        # ----- FAKE JAGUAR ENERGIES -----
        if com in ['je', 'je2', 'jeo']:
            if com == 'je': typ = 'energy_1'
            elif com == 'je2': typ = 'energy_2'
            elif com == 'jeo': typ = 'energy_opt'
            for idx_1, group_filenames in enumerate(groups_filenames):
                for filename in group_filenames:
                    if filename not in outs:
                        outs[filename] = \
                            filetypes.Mae(os.path.join(directory, filename))
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

        # ----- FAKE MACROMODEL ENERGIES -----
        if com in ['me', 'me2', 'meo']:
            if com == 'me': typ = 'energy_1'
            elif com == 'me2': typ = 'energy_2'
            elif com == 'meo': typ = 'energy_opt'
            if com in ['me', 'me2']: ind = 'pre'
            elif com == 'meo': ind = 'opt'
            for idx_1, group_filenames in enumerate(groups_filenames):
                for filename in group_filenames:
                    if inps[filename].name_mae not in outs:
                        outs[inps[filename].name_mae] = \
                            filetypes.Mae(os.path.join(
                                inps[filename].directory,
                                inps[filename].name_mae))
                    mae = outs[inps[filename].name_mae]
                    selected = filetypes.select_structures(
                        mae.structures,
                        inps[filename]._index_output_mae, ind)
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
    Command line argument parser for calculate. Used to select various
    options and what data types to gather from specific files.

    Arguments
    ---------
    add_help : bool
               Whether or not to add help to the parser. Default
               is True.
    parents : argparse.ArgumentParser
              Parent parser incorporated into this parser. Default
              is None.
    '''
    if parents is None: parents = []
    # Whether or not to add help. You may not want to add help if
    # these arguments are being used in another, higher level parser.
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    # ----- GENERAL OPTIONS -----
    opts = parser.add_argument_group("calculate options")
    opts.add_argument(
        '--directory', '-d', type=str, metavar='somepath', default=os.getcwd(),
        help=('Directory to search for files (.mae, .log, mm3.fld, etc.).'
              '3rd party calculations are executed from this directory.'))
    opts.add_argument(
        '--doprint', '-p', action='store_true',
        help=("Logs the data that was collected. Be forewarned that this "
              " can generate lengthy log files."))
    opts.add_argument(
        '--fake', action='store_true',
        help=("Generate fake data. Works with me, me2, meo, "
              "je, je2, and jeo."))
    opts.add_argument(
        '--ffpath', '-f', type=str,
        help=("Path to force field. Only necessary for certain data types "
              "and if the subgroup names aren't provided."))
    opts.add_argument(
        '--nocheck', '-nc', action='store_false', dest='check', default=True,
        help=("By default, Q2MM checks whether MacroModel tokens are "
              "available before attempting a MacroModel calculation. If this "
              "option is supplied, MacroModel will not check the tokens "
              "first."))
    opts.add_argument(
        '--norun', '-n', action='store_true',
        help="Don't run 3rd party software.")
    opts.add_argument(
        '--subnames',  '-s', type=str, nargs='+',
        metavar='"Substructure Name OPT"',
        help=("Names of all the substructures containing parameters to "
              "optimize in a MacroModel .fld."))
    # ----- DATA TYPES -----
    data_args = parser.add_argument_group("calculate data types")
    data_args.add_argument(
        '-ma', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel angles (post-FF optimization).')
    data_args.add_argument(
        '-mb', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel bond lengths (post-FF optimization).')
    data_args.add_argument(
        '-mcs', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Run a MacroModel conformational search. '
              "Doesn't work as a data type for FF optimizations."))
    data_args.add_argument(
        '-mcs2', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Run a MacroModel conformational search. '
              "Doesn't work as a data type for FF optimizations."
              'Uses AUOP cutoff for number of steps.'))
    data_args.add_argument(
        '-mcs3', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Run a MacroModel conformational search. '
              "Doesn't work as a data type for FF optimizations."
              'Maximum of 4000 steps and no AUOP cutoff.'))
    data_args.add_argument(
        '-me', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (pre-FF optimization).')
    data_args.add_argument(
        '-me2', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Same as -me, but uses a separate weight.'))
    data_args.add_argument(
        '-meo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (post-FF optimization).')
    data_args.add_argument(
        '-meig', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae,somename.out',
        help='MacroModel eigenmatrix (all elements).')
    # data_args.add_argument(
    #     '-meigz', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.mae,somename.out',
    #     help="MacroModel eigenmatrix (diagonal elements).")
    # data_args.add_argument(
    #     '-mh', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.mae',
    #     help='MacroModel Hessian.')
    data_args.add_argument(
        '-mq', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel charges.')
    data_args.add_argument(
        '-mqh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel charges (excludes aliphatic hydrogens).')
    data_args.add_argument(
        '-mt', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel torsions (post-FF optimization).')
    # data_args.add_argument(
    #     '-pm', type=str, nargs='+', action='append',
    #     default=[], metavar='parteth',
    #     help='Tethering of parameters for FF data.')
    # data_args.add_argument(
    #     '-pr', type=str, nargs='+', action='append',
    #     default=[], metavar='parteth',
    #     help='Tethering of parameters for reference data.')
    data_args.add_argument(
        '-ja', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar angles.')
    data_args.add_argument(
        '-jb', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar bond lengths.')
    data_args.add_argument(
        '-je', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar energies.')
    data_args.add_argument(
        '-je2', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Same as -je, but uses a separate weight.')
    data_args.add_argument(
        '-jeo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Same as -je, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    # data_args.add_argument(
    #     '-jeig', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in,somename.out',
    #     help='Jaguar eigenmatrix (all elements).')
    # data_args.add_argument(
    #     '-jeigi', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in,somename.out',
    #     help='Jaguar eigenmatrix (all elements). Invert 1st eigenvalue.')
    data_args.add_argument(
        '-jeigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.in,somename.out',
        help=('Jaguar eigenmatrix. Incluldes all elements, but zeroes '
              'all elements that are off-diagonal.'))
    # data_args.add_argument(
    #     '-jeigz', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in,somename.out',
    #     help="Jaguar eigenmatrix (only diagonal elements).")
    # data_args.add_argument(
    #     '-jeigzi', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in,somename.out',
    #     help=("Jaguar eigenmatrix (only diagonal elements). "
    #           "Invert 1st eigenvalue."))
    # data_args.add_argument(
    #     '-jh', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in',
    #     help='Jaguar Hessian.')
    # data_args.add_argument(
    #     '-jhi', type=str, nargs='+', action='append',
    #     default=[], metavar='somename.in',
    #     help='Jaguar Hessian with inversion.')
    data_args.add_argument(
        '-jq', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar partial charges.')
    data_args.add_argument(
        '-jqh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar charges (excludes aliphatic hydrogens). Sums the charge '
              'of aliphatic hydrogens into the bonded sp3 carbon.'))
    data_args.add_argument(
        '-jt', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar torsions.')
    data_args.add_argument(
        '-r', type=str, nargs='+', action='append', default=[],
        metavar='somefilename',
        help=('Reads data from a simple file format. '
              'Column descriptions: 1. value 2. weight '
              '3. command for calculate 4. type (must match '
              'the MacroModel data type) 5. 1st source '
              '6. 2nd source 7. 1st index 8. 2nd index '
              '9. 1st atom 10. 2nd atom 11. 3rd atom '
              '12. 4th atom'))
    # data_args.add_argument(
    #     '-r', type=str, nargs='+', action='append',
    #     default=[], metavar='filename',
    #     help=('Read data points directly (ex. use with .cal files). '
    #           'Each row corresponds to a data point. Columns are separated '
    #           'by spaces. 1st column is the data label, 2nd column is the '
    #           'weight, and 3rd column is the value.'))
    # data_args.add_argument(
    #     '-zm', type=str, nargs='+', action='append',
    #     default=[], metavar='parteth',
    #     help='Tether parameters away from zero. FF data.')
    # data_args.add_argument(
    #     '-zr', type=str, nargs='+', action='append',
    #     default=[], metavar='parteth',
    #     help='Tether parameters away from zero. Reference data.')
    return parser

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
    
