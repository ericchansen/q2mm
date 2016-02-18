"""
Extracts QM data from reference files or calculates FF data.

Takes a sequence of keywords corresponding to various
datatypes (ex. mb = MacroModel bond lengths) followed by filenames,
and extracts that particular data type from the file. Note that the
order of filenames is important.
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

## LOCATION OF SQLITE3 DATABASE MUST BE IN MEMORY (FOR NOW AT LEAST)!
DATABASE_LOC = ':memory:'

## Commands where we need to load the force field.
COM_LOAD_FF = ['ma', 'mb', 'mt', 'ja', 'jb', 'jt']
## Commands related to Gaussian.
COM_GAUSSIAN = ['ge', 'geo', 'geigz', 'geigz2']
## Commands related to Jaguar (Schrodinger).
COM_JAGUAR = ['je', 'je2', 'jeo', 'jeigz', 'jq', 'jqh']
## Commands related to MacroModel (Schrodinger).
COM_MACROMODEL = ['ja', 'jb', 'jt', 'ma', 'mb', 'mcs', 'mcs2',
                  'mcs3', 'me', 'me2', 'meo', 'mjeig', 'mgeig',
                  'mq', 'mqh', 'mt']
## All other commands.
# COM_OTHER = ['r']
## All possible commands.
COM_ALL = COM_GAUSSIAN + COM_JAGUAR + COM_MACROMODEL
# COM_ALL = COM_GAUSSIAN + COM_JAGUAR + COM_MACROMODEL + COM_OTHER

logger = logging.getLogger(__name__)

def main(args):
    """
    Arguments
    ---------
    args : string
           Evaluated using parser returned by return_calculate_parser().
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
    if opts.fake:
        data = gather_fake_data(commands, inps, opts.directory)
        logger.log(15, '  -- Generated fake data.')
    else:
        if opts.norun:
            logger.log(15, "  -- Skipping external software calculations.")
        else:
            for filename, some_class in inps.iteritems():
                if some_class is not None:
                    some_class.run(check_tokens=opts.check)
        data = gather_data(
            commands, inps, opts.directory, opts.ffpath, opts.subnames)
    if co.SETTINGS['use_sqlite3']:
        conn = datum_to_sqlite3(data)
        if opts.doprint:
            beautiful_conn(conn)
        return conn
    else:
        if opts.doprint:
            beautiful_data(data)
        return data

def gather_data(commands, inps, directory, ff_path=None, sub_names=None):
    """
    Gathers data from files. Knows what to do based upon dictionary
    `commands`. Uses dictionary `inps` to keep track of files that may
    have been generated. Reads files and stores the data in dictionary
    `outs` to prevent having to reread files.

    Arguments
    ---------
    commands : dictionary
    inps : dictionary
    directory : string
    ff_path : string
    sub_names : list of strings or None

    Returns
    -------
    list of Datum
    """
    ff_coms = [x for x in commands if x in COM_LOAD_FF]
    if sub_names is None and ff_coms:
        logger.log(5, '  -- Must read FF for datatypes {} if substructure '
                   'name not supplied.'.format(
                ', '.join(ff_coms)))
        sub_names = get_sub_names(
            ff_path=ff_path, directory=directory)

    outs = {}
    data_list = []

    for com, groups_filenames in commands.iteritems():

        # # ----- REFERENCE DATA FILE -----
        # if com == 'r':
        #     for filename in groups_filenames:
        #         ref = filetypes.Reference(filename[0])
        #         data = ref.get_data()
        #         for datum in data:
        #             datum = co.set_data_defaults(datum)
        #             c.execute(co.STR_SQLITE3, datum)

        # ----- JAGUAR ENERGIES -----
        if com in ['je', 'je2', 'jeo']:
            if com == 'je': typ = 'energy-1'
            elif com == 'je2': typ = 'energy-2'
            elif com == 'jeo': typ = 'energy-opt'
            # Move through files. Grouping matters here. Each group (idx_1)
            # is used to separately calculate relative energies.
            for idx_1, group_filenames in enumerate(groups_filenames):
                for filename in group_filenames:
                    if filename not in outs:
                        outs[filename] = \
                            filetypes.Mae(os.path.join(directory, filename))
                    mae = outs[filename]
                    for str_num, struct in enumerate(mae.structures):
                        try:
                            data_list.append(datatypes.Datum(
                                    val=(struct.props['r_j_Gas_Phase_Energy'] * 
                                         co.HARTREE_TO_KJMOL),
                                    com=com,
                                    typ=typ,
                                    src_1=filename,
                                    idx_1=idx_1 + 1,
                                    idx_2=str_num + 1))
                        except KeyError:
                            data_list.append(datatypes.Datum(
                                    val=(struct.props['r_j_QM_Energy'] * 
                                         co.HARTREE_TO_KJMOL),
                                    com=com,
                                    typ=typ,
                                    src_1=filename,
                                    idx_1=idx_1 + 1,
                                    idx_2=str_num + 1))
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
                                    data_list.append(datatypes.Datum(
                                            val=q,
                                            com=com,
                                            typ='charge',
                                            src_1=filename,
                                            idx_1=i+1,
                                            atm_1=atom.index))

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
                                    data_list.append(datatypes.Datum(
                                            val=atom.partial_charge,
                                            com=com,
                                            typ='charge',
                                            src_1=filename,
                                            idx_1=str_num+1,
                                            atm_1=atom.index))
 
        # ----- MACROMODEL ENERGIES -----
        if com in ['me', 'me2', 'meo']:
            if com == 'me': typ = 'energy-1'
            elif com == 'me2': typ = 'energy-2'
            elif com == 'meo': typ = 'energy-opt'
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
                        data_list.append(datatypes.Datum(
                                val=struct.props['r_mmod_Potential_Energy-MM3*'],
                                com=com,
                                typ=typ,
                                src_1=inps[filename].name_mae,
                                idx_1=idx_1+1,
                                idx_2=str_num+1))

        ## ----- SCHRODINGER STRUCTURES -----
        if com in ['ja', 'jb', 'jt', 'ma', 'mb', 'mt']:
            if com in ['ja', 'jb', 'jt']: index = 'pre'
            elif com in ['ma', 'mb', 'mt']: index = 'opt'
            if com in ['ja', 'ma']: typ = 'angles'
            elif com in ['jb', 'mb']: typ = 'bonds'
            elif com in ['jt', 'mt']: typ = 'torsions'
            ## Move through files as you specified them on the command line.
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
                    for str_num, struct in selected:
                        data_list.extend(struct.select_stuff(
                                typ,
                                com=com,
                                com_match=sub_names,
                                src_1=mmo.filename,
                                idx_1=str_num + 1))

        if com in ['ge', 'geo']:
            if com == 'ge': typ = 'energy-1'
            elif com == 'geo': typ = 'energy-opt'
            for idx_1, group_filenames in enumerate(groups_filenames):
                for name_log in group_filenames:
                    if name_log not in outs:
                        outs[name_log] = filetypes.GaussLog(
                            os.path.join(directory, name_log))
                    log = outs[name_log]
                    hf = log.structures[0].props['hf']
                    zp = log.structures[0].props['zp']
                    energy = (hf + zp) * co.HARTREE_TO_KJMOL
                    data_list.append(
                        datatypes.Datum(
                            val=energy,
                            com=com,
                            typ=typ,
                            src_1=name_log,
                            idx_1=idx_1 + 1))

        ## ------ GAUSSIAN EIGENMATRIX ------
        if com == 'geigz':
            for group_filenames in groups_filenames:
                for name_log in group_filenames:
                    if name_log not in outs:
                        outs[name_log] = filetypes.GaussLog(
                            os.path.join(directory, name_log))
                    log = outs[name_log]
                    evals = log.evals * co.HESSIAN_CONVERSION
                    evals_matrix = np.diag(evals)
                    low_tri_idx = np.tril_indices_from(evals_matrix)
                    lower_tri = evals_matrix[low_tri_idx]
                    data = [datatypes.Datum(
                            val=e,
                            com=com,
                            typ='eig',
                            src_1=name_log,
                            idx_1=x+1,
                            idx_2=y+1)
                            for e, x, y in itertools.izip(
                            lower_tri, low_tri_idx[0], low_tri_idx[1])]
                    data_list.extend(data)

        if com == 'geigz2':
            for group_filenames in groups_filenames:
                for comma_filenames in group_filenames:
                    name_log, name_fchk = comma_filenames.split(',')
                    if name_log not in outs:
                        outs[name_log] = filetypes.GaussLog(
                            os.path.join(directory, name_log))
                    log = outs[name_log]
                    if name_fchk not in outs:
                        outs[name_fchk] = filetypes.GaussFormChk(
                            os.path.join(directory, name_fchk))
                    fchk = outs[name_fchk]
                    
                    hess = datatypes.Hessian()
                    hess.hess = fchk.hess
                    hess.evecs = log.evecs
                    hess.atoms = fchk.atoms
                    hess.mass_weight_hessian()
                    
                    hess.diagonalize()
                    
                    # hess.mass_weight_eigenvectors()
                    diagonal_matrix = np.diag(np.diag(hess.hess))
                    low_tri_idx = np.tril_indices_from(diagonal_matrix)
                    lower_tri = diagonal_matrix[low_tri_idx]
                    data = [datatypes.Datum(
                            val=e,
                            com=com,
                            typ='eig',
                            src_1=name_log,
                            src_2=name_fchk,
                            idx_1=x+1,
                            idx_2=y+1)
                            for e, x, y in itertools.izip(
                            lower_tri, low_tri_idx[0], low_tri_idx[1])]
                    data_list.extend(data)

        ## ------ MACROMODEL/GAUSSIAN EIGENMATRIX -----
        if com == 'mgeig':
            for group_filenames in groups_filenames:
                for comma_filenames in group_filenames:

                    name_mae, name_gau_log = comma_filenames.split(',')
                    name_macro_log = inps[name_mae].name_log
                    if name_macro_log not in outs:
                        outs[name_macro_log] = filetypes.MacroModelLog(
                            os.path.join(inps[name_mae].directory,
                                         inps[name_mae].name_log))
                    macro_log = outs[name_macro_log]
                    if name_gau_log not in outs:
                        outs[name_gau_log] = filetypes.GaussLog(
                            os.path.join(directory, name_gau_log))
                    gau_log = outs[name_gau_log]

                    hess = datatypes.Hessian()
                    hess.hess = macro_log.hessian
                    ## Eigenvectors should already be mass weighted using
                    ## pHessMan functions.
                    hess.evecs = gau_log.evecs
                    hess.diagonalize()
                    
                    low_tri_idx = np.tril_indices_from(hess.hess)
                    lower_tri = hess.hess[low_tri_idx]
                    
                    data = [datatypes.Datum(
                            val=e,
                            com=com,
                            typ='eig',
                            src_1=name_macro_log,
                            src_2=name_gau_log,
                            idx_1=x+1,
                            idx_2=y+1)
                            for e, x, y in itertools.izip(
                            lower_tri, low_tri_idx[0], low_tri_idx[1])]
                    data_list.extend(data)
        ## ------ SCHRODINGER EIGENMATRIX ------
        if com in ['jeigz', 'mjeig']:
            for group_filenames in groups_filenames:
                for comma_filenames in group_filenames:
                    if com == 'mjeig':
                        name_mae, name_out = comma_filenames.split(',')
                        name_log = inps[name_mae].name_log
                        if name_log not in outs:
                            outs[name_log] = filetypes.MacroModelLog(
                                os.path.join(inps[name_mae].directory,
                                             inps[name_mae].name_log))
                        log = outs[name_log]
                    elif com == 'jeigz':
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
                    elif com == 'mjeig':
                        hess = datatypes.Hessian(log, out)
                        hess.hess = datatypes.check_mm_dummy(
                            hess.hess,
                            out.dummy_atom_eigenvector_indices)
                    hess.mass_weight_eigenvectors()
                    hess.diagonalize()
                    if com == 'jeigz':
                        diagonal_matrix = np.diag(np.diag(hess.hess))
                    else:
                        diagonal_matrix = hess.hess
                    low_tri_idx = np.tril_indices_from(diagonal_matrix)
                    lower_tri = diagonal_matrix[low_tri_idx]
                    if com == 'jeigz':
                        src_1 = name_in
                    elif com == 'mjeig':
                        src_1 = name_mae
                    data = [datatypes.Datum(
                            val=e,
                            com=com,
                            typ='eig',
                            src_1=src_1,
                            src_2=name_out,
                            idx_1=x+1,
                            idx_2=y+1)
                            for e, x, y in itertools.izip(
                            lower_tri, low_tri_idx[0], low_tri_idx[1])]
                    data_list.extend(data)
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(len(data_list)))
    return data_list

def datum_to_sqlite3(data):
    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)
    for d in data:
        c.execute(co.STR_SQLITE3,
                  [None, ## Used for 'id' in sqlite3 table.
                   d.val, d.wht, d.com, d.typ,
                   d.src_1, d.src_2,
                   d.idx_1, d.idx_2,
                   d.atm_1, d.atm_2, d.atm_3, d.atm_4,
                   d.lbl])
    conn.commit()
    return conn

def dic_to_sqlite3(data):
    conn = sqlite3.connect(DATABASE_LOC)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.executescript(co.STR_INIT_SQLITE3)
    c.executemany(co.STR_SQLITE3, data)
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

def get_sub_names(ff_path=None, directory=None):
    """
    Some datatypes only work when we also know the names of the
    substructures containing parameters that are being optimized.
    If the substructure names aren't provided by the user, this will
    read the force field for those names.

    Arguments
    ---------
    ff_path : string
    directory : string
    """
    if ff_path is None and directory is not None:
        ff_path = os.path.join(directory, 'mm3.fld')
    elif ff_path is None and directory is None:
        ff_path = os.path.join(os.getcwd, 'mm3.fld')
    # assert ff_path is not None, \
    #     "User didn't provide required substructure names " + \
    #     "or path to force field!"
    ff = datatypes.MM3(ff_path)
    ff.import_ff()
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

def beautiful_data(data, log_level=20):
    """
    Logs data as a table.

    Arguments
    ---------
    data : list of Datum
    log_level : int
            Logging level used.
    """
    logger.log(20, '--' + ' LABEL '.center(22, '-') +
               '--' + ' VALUE '.center(22, '-') + '--')
    for d in data:
        logger.log(20, '  ' + '{:22s}'.format(d.lbl) +
                   '  ' + '{:22.4f}'.format(d.val))
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
    data_list = []
    for com, groups_filenames in commands.iteritems():
        ## ----- FAKE JAGUAR ENERGIES -----
        if com in ['je', 'je2', 'jeo']:
            if com == 'je': typ = 'energy-1'
            elif com == 'je2': typ = 'energy-2'
            elif com == 'jeo': typ = 'energy-opt'
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
                        data_list.append(energy)
        ## ----- FAKE MACROMODEL ENERGIES -----
        if com in ['me', 'me2', 'meo']:
            if com == 'me': typ = 'energy-1'
            elif com == 'me2': typ = 'energy-2'
            elif com == 'meo': typ = 'energy-opt'
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
                        data_list.append(energy)
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(len(data_list)))
    return data_list

def return_calculate_parser(add_help=True, parents=None):
    '''
    Command line argument parser for calculate.

    Arguments
    ---------
    add_help : bool
               Whether or not to add help to the parser. Default
               is True.
    parents : argparse.ArgumentParser
              Parent parser incorporated into this parser. Default
              is None.
    '''
    ## Whether or not to add help. You may not want to add help if
    ## these arguments are being used in another, higher level parser.
    if parents is None: parents = []
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    ## ----- GENERAL OPTIONS -----
    opts = parser.add_argument_group("calculate options")
    opts.add_argument(
        '--directory', '-d', type=str, metavar='somepath', default=os.getcwd(),
        help=('Directory searched for files '
              '(ex. *.mae, *.log, mm3.fld, etc.). '
              'Subshell commands (ex. MacroModel) are executed from here. '
              'Default is the current directory.'))
    opts.add_argument(
        '--doprint', '-p', action='store_true',
        help=("Logs data. Can generate extensive log files."))
    opts.add_argument(
        '--fake', action='store_true',
        help=("Generates fake data. Only works with -me, -me2, -meo, "
              "-je, -je2 and -jeo."))
    opts.add_argument(
        '--ffpath', '-f', type=str, metavar='somepath',
        help=("Path to force field. Only necessary for certain data types "
              "and if the subgroup names aren't provided."))
    opts.add_argument(
        '--nocheck', '-nc', action='store_false', dest='check', default=True,
        help=("By default, Q2MM checks whether MacroModel tokens are "
              "available before attempting a MacroModel calculation. If this "
              "option is supplied, MacroModel will not check for tokens "
              "first."))
    opts.add_argument(
        '--norun', '-n', action='store_true',
        help="Don't run 3rd party software.")
    opts.add_argument(
        '--subnames',  '-s', type=str, nargs='+',
        metavar='"Substructure Name OPT"',
        help=("Names of the substructures containing parameters to "
              "optimize in a mm3.fld file."))
    # opts.add_argument(
    #     '--usedb', action='store_true',
    #     help='Use sqlite3.')
    ## ----- DATA TYPES -----
    data_args = parser.add_argument_group("calculate data types")
    data_args.add_argument(
        '-ge', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies.'))
    data_args.add_argument(
        '-geo', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Same as -ge, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    data_args.add_argument(
        '-geigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements . Uses only the .log for '
              'the eigenvalues and eigenvectors.'))
    data_args.add_argument(
        '-geigz2', type=str, nargs='+', action='append',
        default=[], metavar='somename.log,somename.fchk',
        help=('Gaussian eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements. Uses the .log for '
              'eigenvectors and .fchk for Hessian.'))
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
              "Doesn't work as a data type for FF optimizations. "
              'Uses AUOP cutoff for number of steps.'))
    data_args.add_argument(
        '-mcs3', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Run a MacroModel conformational search. '
              "Doesn't work as a data type for FF optimizations. "
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
        '-mjeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae,somename.out',
        help='MacroModel eigenmatrix (all elements). Uses Jaguar '
        'eigenvectors.')
    data_args.add_argument(
        '-mgeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae,somename.out',
        help='MacroModel eigenmatrix (all elements). Uses Gaussian '
        'eigenvectors.')
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
    data_args.add_argument(
        '-jeigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.in,somename.out',
        help=('Jaguar eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements.'))
    data_args.add_argument(
        '-jq', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar partial charges.')
    data_args.add_argument(
        '-jqh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar partial charges (excludes aliphatic hydrogens). '
              'Sums aliphatic hydrogen charges into their bonded sp3 '
              'carbon.'))
    data_args.add_argument(
        '-jt', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar torsions.')
    # data_args.add_argument(
    #     '-r', type=str, nargs='+', action='append', default=[],
    #     metavar='somefilename',
    #     help=('Reads data from a simple file format. '
    #           'Column descriptions: 1. value 2. weight '
    #           '3. command for calculate 4. type (must match '
    #           'the MacroModel data type) 5. 1st source '
    #           '6. 2nd source 7. 1st index 8. 2nd index '
    #           '9. 1st atom 10. 2nd atom 11. 3rd atom '
    #           '12. 4th atom'))
    return parser

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
    
