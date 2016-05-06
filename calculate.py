"""
Extracts data from reference files or calculates FF data.

Takes a sequence of keywords corresponding to various
datatypes (ex. mb = MacroModel bond lengths) followed by filenames,
and extracts that particular data type from the file.

Note that the order of filenames IS IMPORTANT!

Used to manage calls to MacroModel but that is now done in the
Mae class inside filetypes. I'm still debating if that should be
there or here. Will see how this framework translates into
Amber and then decide.
"""
import argparse
import logging
import logging.config
import numpy as np
import os
import sys

# I don't really want to import all of chain if possible. I only want
# chain.from_iterable.
# chain.from_iterable flattens a list of lists similar to:
#   [child for parent in grandparent for child in parent]
# However, I think chain.from_iterable works on any number of nested lists.
from itertools import chain, izip
from textwrap import TextWrapper

import constants as co
import compare
import datatypes
import filetypes

logger = logging.getLogger(__name__)

# Commands where we need to load the force field.
COM_LOAD_FF    = ['ma', 'mb', 'mt', 'ja', 'jb', 'jt']
# Commands related to Gaussian.
COM_GAUSSIAN   = ['ge', 'geo', 'geigz', 'geigz2']
# Commands related to Jaguar (Schrodinger).
COM_JAGUAR     = ['je', 'jeo', 'jea', 'jeao', 'jeigz', 'jq', 'jqh']
# Commands related to MacroModel (Schrodinger).
COM_MACROMODEL = ['ja', 'jb', 'jt', 'ma', 'mb', 
                  'me', 'meo', 'mea', 'meao', 'mjeig', 'mgeig',
                  'mq', 'mqh', 'mt']
# All other commands.
COM_OTHER = ['r']
# All possible commands.
COM_ALL = COM_GAUSSIAN + COM_JAGUAR + COM_MACROMODEL + COM_OTHER

def main(args):
    """
    Arguments
    ---------
    args : string or list of strings
           Evaluated using parser returned by return_calculate_parser(). If
           it's a string, it will be converted into a list of strings.
    """
    # Should be a list of strings for use by argparse. Ensure that's the case.
    if isinstance(args, basestring):
        args.split()
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    # This makes a dictionary that only contains the arguments related to
    # extracting data from everything in the argparse dictionary, opts.
    # commands looks like:
    # {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'], 
    #         ['b1.01.mae', 'b2.01.mae']],
    #  'mb': [['a1.01.mae'], ['b1.01.mae']],
    #  'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
    # }
    commands = {key: value for key, value in opts.__dict__.iteritems() if key
                in COM_ALL and value}
    # Add in the empty commands. I'd rather not do this, but it makes later
    # coding when collecting data easier.
    for command in COM_ALL:
        if command not in commands:
            commands.update({command: []})
    pretty_all_commands(commands)
    # This groups all of the data type commands associated with one file.
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
    # This dictionary associates the filename that the user supplied with
    # the command file that has to be used to execute some backend software
    # calculate in order to retrieve the data that the user requested.
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
    # This generates any of the necessary command files. It uses
    # commands_for_filenames, which contains all of the data types associated
    # with the given file.
    for filename, commands_for_filename in commands_for_filenames.iteritems():
        # These next two if statements will break down what command files
        # have to be written by the backend software package.
        if any(x in COM_MACROMODEL for x in commands_for_filename):
            if os.path.splitext(filename)[1] == '.mae':
                inps[filename] = filetypes.Mae(
                    os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
                inps[filename].write_com(sometext=opts.append)
        # In this case, no command files have to be written.
        else:
            inps[filename] = None
    # Check whether or not to skip calculations.
    if opts.norun:
        logger.log(15, "  -- Skipping backend calculations.")
    else:
        for filename, some_class in inps.iteritems():
            # Works if some class is None too.
            if hasattr(some_class, 'run'):
                # Ideally this can be the same for each software backend,
                # but that means we're going to have to make some changes
                # so that this token argument is handled properly.
                some_class.run(check_tokens=opts.check)
    # This is a list comprised of datatypes.Datum objects.
    # If we remove/with sorting removed, the Datum class is less
    # useful. We may want to reduce this to a N x 3 matrix or
    # 3 vectors (labels, weights, values).
    data = collect_data(commands, inps, direc=opts.directory)
    # Adds weights to the data points in the data list.
    if opts.weight:
        compare.import_weights(data)
    # Optional printing or logging of data.
    if opts.doprint:
        pretty_data(data, log_level=None)
    return data

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
    # Whether or not to add parents parsers. Not sure if/where this may be used
    # anymore.
    if parents is None: parents = []
    # Whether or not to add help. You may not want to add help if these
    # arguments are being used in another, higher level parser.
    if add_help:
        parser = argparse.ArgumentParser(
            description=__doc__, parents=parents)
    else:
        parser = argparse.ArgumentParser(
            add_help=False, parents=parents)
    # GENERAL OPTIONS
    opts = parser.add_argument_group("calculate options")
    opts.add_argument(
        '--append', '-a', type=str, metavar='sometext',
        help='Append this text to command files generated by Q2MM.')
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
        '--ffpath', '-f', type=str, metavar='somepath',
        help=("Path to force field. Only necessary for certain data types "
              "if you don't provide the substructure name."))
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
    opts.add_argument(
        '--weight', '-w', action='store_true',
        help='Add weights to data points.')
    # DATA TYPES
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
              'all off-diagonal elements. Uses only the .log for '
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
        '-me', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (pre-FF optimization).')
    data_args.add_argument(
        '-meo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (post-FF optimization).')
    data_args.add_argument(
        '-mea', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (pre-FF optimization). Energies will be '
        'relative to the average energy.')
    data_args.add_argument(
        '-meao', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (post-FF optimization). Energies will be '
        'relative to the average energy.')
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
        '-jeo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Same as -je, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    data_args.add_argument(
        '-jea', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Everything will be relative to the average '
              'energy.'))
    data_args.add_argument(
        '-jeao', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Same as -jea, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meao.'))
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
    data_args.add_argument(
        '-r', type=str, nargs='+', action='append',
        default=[], metavar='somename.txt',
        help=('Read reference data from file. The reference file should '
              '3 space or tab separated columns. Column 1 is the labels, '
              'column 2 is the weights and column 3 is the values.'))
    return parser

def check_outs(filename, outs, classtype, direc):
    if filename not in outs:
        outs[filename] = \
            classtype(os.path.join(direc, filename))
    return outs[filename]

def collect_reference(path):
    """
    Reads the data inside a reference data text file.

    This must have 3 columns:
      1. Labels
      2. Weights
      3. Values
    """
    data = []
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            # Skip certain lines.
            if line[0] in ['-', '#']:
                continue
            # if line.startswith('-'):
            #     continue
            # Remove everything following a # in a line.
            line = line.partition('#')[0]
            cols = line.split()
            # There should always be 3 columns.
            assert len(cols) == 3, \
                'Error reading line {} from {}: {}'.format(
                i, path, line)
            lbl, wht, val = cols
            datum = datatypes.Datum(lbl=lbl, wht=float(wht), val=float(val))
            data.append(datum)
    return np.array(data)

# Must be rewritten to go in a particular order of data types every time.
def collect_data(coms, inps, direc='.', sub_names=['OPT']):
    # outs looks like:
    # {'filename1': <some class for filename1>,
    #  'filename2': <some class for filename2>,
    #  'filename3': <some class for filename3>
    # }
    outs = {}
    # List of Datum objects.
    data = []
    # REFERENCE DATA TEXT FILES
    # No grouping is necessary for this data type, so flatten the list of
    # lists.
    filenames = chain.from_iterable(coms['r'])
    for filename in filenames:
        # Unlike most datatypes, these Datum only get the attributes _lbl,
        # val and wht. This is to ensure that making and working with these
        # reference text files isn't too cumbersome.
        data.extend(collect_reference(os.path.join(direc, filename)))
    # JAGUAR ENERGIES
    filenames_s = coms['je']
    # idx_1 is the number used to group sets of relative energies.
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            mae = check_outs(filename, outs, filetypes.Mae, direc)
            # idx_2 corresponds to the structure inside the file in case the
            # .mae files contains multiple structures.
            for idx_2, structure in enumerate(mae.structures):
                try:
                    energy = structure.props['r_j_Gas_Phase_Energy']
                except KeyError:
                    energy = structure.props['r_j_QM_Energy']
                energy *= co.HARTREE_TO_KJMOL
                temp.append(datatypes.Datum(
                        val=energy,
                        com='je',
                        typ='e',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        # For this data type, we set everything relative.
        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # MACROMODEL ENERGIES
    filenames_s = coms['me']
    ind = 'pre'
    for idx_1, filenames in enumerate(filenames_s):
        for filename in filenames:
            mae_name = inps[filename].name_mae
            mae = check_outs(mae_name, outs, filetypes.Mae, direc)
            indices = inps[filename]._index_output_mae
            # This is list of sets. The 1st value in the set corresponds to the
            # number of the structure. The 2nd value is the structure class.
            selected_structures = filetypes.select_structures(
                mae.structures, indices, ind)
            print(selected_structures)
            for idx_2, structure in selected_structures:
                data.append(datatypes.Datum(
                        val=structure.props['r_mmod_Potential_Energy-MM3*'],
                        com='me',
                        typ='e',
                        src_1=inps[filename].name_mae,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(len(data)))
    return np.array(data, dtype=datatypes.Datum)

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
        for comma_separated in chain.from_iterable(groups_filenames):
            for filename in comma_separated.split(','):
                if filename in sorted_commands:
                    sorted_commands[filename].append(command)
                else:
                    sorted_commands[filename] = [command]
    return sorted_commands
            
# Will also have to be updated. Maybe the Datum class too and how it responds
# to assigning labels.
def read_reference(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Skip certain lines.
            if line.startswith('-'):
                continue
            # Remove everything following a # in a line.
            line = line.partition('#')[0]
            cols = line.split()
            # There should always be 3 columns.
            if len(cols) == 3:
                lbl, wht, val = cols
                datum = datatypes.Datum(lbl=lbl, wht=float(wht), val=float(val))
                lbl_to_data_attrs(datum, lbl)
                data.append(datum)
    data = data.sort(key=datatypes.datum_sort_key)
    return np.array(data)

# Shouldn't be necessary anymore.
# def lbl_to_data_attrs(datum, lbl):
#     parts = lbl.split('_')
#     datum.typ = parts[0]
#     if len(parts) == 3:
#         idxs = parts[-1]
#     if len(parts) == 4:
#         idxs = parts[-2]
#         atm_nums = parts[-1]
#         atm_nums = atm_nums.split('-')
#         for i, atm_num in enumerate(atm_nums):
#             setattr(datum, 'atm_{}'.format(i+1), int(atm_num))
#     idxs = idxs.split('-')
#     datum.idx_1 = int(idxs[0])
#     if len(idxs) == 2:
#         datum.idx_2 == int(idxs[1])

# Right now, this only looks good if the logger doesn't append each log
# message with something (module, date/time, etc.).
# It would be great if this output looked good regardless of the settings
# used for the logger.
# That goes for all of these pretty output functions that use TextWrapper.
def pretty_commands_for_files(commands_for_files, log_level=5):
    """
    Logs the .mae commands dictionary, or the all of the commands
    used on a particular file.

    Arguments
    ---------
    commands_for_files : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = TextWrapper(
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

def pretty_all_commands(commands, log_level=5):
    """
    Logs the arguments/commands given to calculate that are used
    to request particular datatypes from particular files.

    Arguments
    ---------
    commands : dic
    log_level : int
    """
    if logger.getEffectiveLevel() <= log_level:
        foobar = TextWrapper(width=48, subsequent_indent=' '*24)
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

def pretty_data(data, log_level=20):
    """
    Logs data as a table.

    Arguments
    ---------
    data : list of Datum
    log_level : int
    """
    if not data[0].wht:
        compare.import_weights(data)
    if log_level:
        string = ('--' + ' LABEL '.center(22, '-') +
                  '--' + ' WEIGHT '.center(22, '-') +
                  '--' + ' VALUE '.center(22, '-') +
                  '--')
        logger.log(log_level, string)
    for d in data:
        if d.wht:
            string = ('  ' + '{:22s}'.format(d.lbl) +
                      '  ' + '{:22.4f}'.format(d.wht) + 
                      '  ' + '{:22.4f}'.format(d.val))
        else:
            string = ('  ' + '{:22s}'.format(d.lbl) +
                      '  ' + '{:22.4f}'.format(d.val))
        if log_level:
            logger.log(log_level, string)
        else:
            print(string)
    if log_level:
        logger.log(log_level, '-' * 50)

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
