#!/usr/bin/env python
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
if (sys.version_info > (3, 0)):
    from itertools import chain
else:
    from itertools import chain, izip
from textwrap import TextWrapper

import constants as co
import compare
import datatypes
import filetypes
import parameters

logger = logging.getLogger(__name__)

# Commands where we need to load the force field.
COM_LOAD_FF    = ['ma', 'mb', 'mt',
                  'ja', 'jb', 'jt']
# Commands related to Gaussian.
COM_GAUSSIAN   = ['ge', 'gea', 'geo', 'geao',
                  'gh', 'geigz']
# Commands related to Jaguar (Schrodinger).
COM_JAGUAR     = ['jq', 'jqh', 'jqa',
                  'je', 'jeo', 'jea', 'jeao',
                  'jh', 'jeigz']
# Commands related to MacroModel (Schrodinger).
# Seems odd that the Jaguar geometry datatypes are in here, but we
# do a MacroModel calculation to get the data in an easy form to
# extract.
COM_MACROMODEL = ['ja', 'jb', 'jt',
                  'mq', 'mqh', 'mqa',
                  'ma', 'mb', 'mt',
                  'me', 'meo', 'mea', 'meao',
                  'mh', 'mjeig', 'mgeig',
                  'mp']
# Commands related to Tinker.
COM_TINKER     = ['ta','tao', 'tb', 'tbo',
                  'tt','tto', 'te', 'teo',
                  'tea','teao', 'th',
                  'tjeigz', 'tgeig']
# Commands related to Amber.
COM_AMBER      = ['ae']
# All other commands.
COM_OTHER = ['r']
# All possible commands.
COM_ALL = COM_GAUSSIAN + COM_JAGUAR + COM_MACROMODEL + COM_TINKER + \
          COM_AMBER + COM_OTHER

def main(args):
    """
    Arguments
    ---------
    args : string or list of strings
           Evaluated using parser returned by return_calculate_parser(). If
           it's a string, it will be converted into a list of strings.
    """
    # Should be a list of strings for use by argparse. Ensure that's the case.
    # basestring is deprecated in python3, str is probably safe to use in both
    # but should be tested, for now sys.version_info switch can handle it
    if (sys.version_info > (3, 0)):
        if isinstance(args, str):
            args = args.split()
    else:
        if isinstance(args, basestring):
            args = args.split()
    parser = return_calculate_parser()
    opts = parser.parse_args(args)
    # This makes a dictionary that only contains the arguments related to
    # extracting data from everything in the argparse dictionary, opts.
    # Given that the user supplies:
    # python calculate.py -me a1.01.mae a2.01.mae a3.01.mae -me b1.01.mae
    #    b2.01.mae -mb a1.01.mae b1.01.mae -jeig a1.01.in,a1.out
    #    b1.01.in,b1.out
    # commands looks like:
    # {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'],
    #         ['b1.01.mae', 'b2.01.mae']],
    #  'mb': [['a1.01.mae'], ['b1.01.mae']],
    #  'jeig': [['a1.01.in,a1.out', 'b1.01.in,b1.out']]
    # }
    if (sys.version_info > (3, 0)):
        commands = {key: value for key, value in iter(opts.__dict__.items()) if key
                    in COM_ALL and value}
    else:
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
    # Stuff below doesn't need both comma separated filenames simultaneously.
    if (sys.version_info > (3, 0)):
        fname_cmds = iter(commands_for_filenames.items())
    else:
        fname_cmds = commands_for_filenames.iteritems()
    for filename, commands_for_filename in fname_cmds:
        logger.log(1, '>>> filename: {}'.format(filename))
        logger.log(1, '>>> commands_for_filename: {}'.format(
            commands_for_filename))
        # These next two if statements will break down what command files
        # have to be written by the backend software package.
        if any(x in COM_MACROMODEL for x in commands_for_filename):
            if os.path.splitext(filename)[1] == '.mae':
                inps[filename] = filetypes.Mae(
                    os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
                inps[filename].write_com(sometext=opts.append)
        elif any(x in COM_TINKER for x in commands_for_filename):
            if os.path.splitext(filename)[1] == '.xyz':
                inps[filename] = filetypes.Tinker_xyz(
                    os.path.join(opts.directory, filename))
                inps[filename].commands = commands_for_filename
        elif any(x in COM_AMBER for x in commands_for_filename):
            # This doesn't work.
            # We need to know both filenames simultaneously for this Amber crap.
            # Have to add these to `inps` in some other way.
            pass
        # In this case, no command files have to be written.
        else:
            inps[filename] = None
    # Stuff below needs both comma separated filenames simultaneously.
    # Do the Amber inputs.
    # Leaving the filenames together because Taylor said this would work well.
    for comma_sep_filenames in flatten(commands['ae']):
        # Maybe make more specific later.
        inps[comma_sep_filenames] = filetypes.AmberInput(
            'DOES_PATH_EVEN_MATTER')
        split_it = comma_sep_filenames.split(',')
        inps[comma_sep_filenames].directory = opts.directory
        inps[comma_sep_filenames].inpcrd = split_it[0]
        inps[comma_sep_filenames].prmtop = split_it[1]
    logger.log(1, '>>> commands: {}'.format(commands))
    # Check whether or not to skip calculations.
    if opts.norun or opts.fake:
        logger.log(15, "  -- Skipping backend calculations.")
    else:
        if (sys.version_info > (3, 0)):
            inps_iter = iter(inps.items())
        else:
            inps_iter = inps.iteritems()
        for filename, some_class in inps_iter:
            logger.log(1, '>>> filename: {}'.format(filename))
            logger.log(1, '>>> some_class: {}'.format(some_class))
            # Works if some class is None too.
            if hasattr(some_class, 'run'):
                # Ideally this can be the same for each software backend,
                # but that means we're going to have to make some changes
                # so that this token argument is handled properly.
                some_class.run(check_tokens=opts.check)
    # `data` is a list comprised of datatypes.Datum objects.
    # If we remove/with sorting removed, the Datum class is less
    # useful. We may want to reduce this to a N x 3 matrix or
    # 3 vectors (labels, weights, values).
    sub_names = ['OPT']
    if opts.subnames:
        sub_names = opts.subnames
    if opts.fake:
        data = collect_data_fake(
            commands, inps, direc=opts.directory, invert=opts.invert,
            sub_names=sub_names)
    else:
        data = collect_data(
            commands, inps, direc=opts.directory, invert=opts.invert,
            sub_names=sub_names)
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
        '--fake', action='store_true',
        help=("Generate fake data sets. Used to expedite testing."))
    opts.add_argument(
        '--ffpath', '-f', type=str, metavar='somepath',
        help=("Path to force field. Only necessary for certain data types "
              "if you don't provide the substructure name."))
    opts.add_argument(
        '--invert', '-i', type=float, metavar='somefloat',
        help=("This option will invert the smallest eigenvalue to be whatever "
              "value is specified by this argument whenever a Hessian is "
              "read."))
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
    # GAUSSIAN OPTIONS
    gau_args = parser.add_argument_group("gaussian reference data types")
    gau_args.add_argument(
        '-ge', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies.'))
    gau_args.add_argument(
        '-gea', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Energies will be relative to the average '
              'energy within this data type.'))
    gau_args.add_argument(
        '-geo', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Same as -ge, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    gau_args.add_argument(
        '-geao', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian energies. Same as -ge, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo. Energies will be relative to the average '
              'energy within this data type.'))
    gau_args.add_argument(
        '-gh', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help='Gaussian Hessian extracted from a .log archive.')
    gau_args.add_argument(
        '-geigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.log',
        help=('Gaussian eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements. Uses only the .log for '
              'the eigenvalues and eigenvectors.'))
    # JAGUAR OPTIONS
    jag_args = parser.add_argument_group("jaguar reference data types")
    jag_args.add_argument(
        '-jq', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar partial charges.')
    jag_args.add_argument(
        '-jqh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar partial charges (excludes aliphatic hydrogens). '
              'Sums aliphatic hydrogen charges into their bonded sp3 '
              'carbon.'))
    jag_args.add_argument(
        '-jqa', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar partial charges. Sums the partial charge of all singly '
              'bonded hydrogens into its connected atom.'))
    jag_args.add_argument(
        '-je', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar energies.')
    jag_args.add_argument(
        '-jea', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Everything will be relative to the average '
              'energy.'))
    jag_args.add_argument(
        '-jeo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Same as -je, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meo.'))
    jag_args.add_argument(
        '-jeao', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('Jaguar energies. Same as -jea, except the files selected '
              'by this command will have their energies compared to those '
              'selected by -meao.'))
    jag_args.add_argument(
        '-ja', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar angles.')
    jag_args.add_argument(
        '-jb', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar bond lengths.')
    jag_args.add_argument(
        '-jt', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='Jaguar torsions.')
    jag_args.add_argument(
        '-jh', type=str, nargs='+', action='append',
        default=[], metavar='somename.in',
        help='Jaguar Hessian.')
    jag_args.add_argument(
        '-jeigz', type=str, nargs='+', action='append',
        default=[], metavar='somename.in,somename.out',
        help=('Jaguar eigenmatrix. Incluldes all elements, but zeroes '
              'all off-diagonal elements.'))
    # ADDITIONAL REFERENCE OPTIONS
    ref_args = parser.add_argument_group("other reference data types")
    ref_args.add_argument(
        '-r', type=str, nargs='+', action='append',
        default=[], metavar='somename.txt',
        help=('Read reference data from file. The reference file should '
              '3 space or tab separated columns. Column 1 is the labels, '
              'column 2 is the weights and column 3 is the values.'))
    # MACROMODEL OPTIONS
    mm_args = parser.add_argument_group("macromodel data types")
    mm_args.add_argument(
        '-mq', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel charges.')
    mm_args.add_argument(
        '-mqh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel charges (excludes aliphatic hydrogens).')
    mm_args.add_argument(
        '-mqa', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help=('MacroModel partial charges. Sums the partial charge of all '
              'singly bonded hydrogens into its connected atom.'))
    mm_args.add_argument(
        '-me', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (pre-FF optimization).')
    mm_args.add_argument(
        '-mea', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (pre-FF optimization). Energies will be '
        'relative to the average energy.')
    mm_args.add_argument(
        '-meo', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (post-FF optimization).')
    mm_args.add_argument(
        '-meao', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel energies (post-FF optimization). Energies will be '
        'relative to the average energy.')
    mm_args.add_argument(
        '-mb', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel bond lengths (post-FF optimization).')
    mm_args.add_argument(
        '-ma', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel angles (post-FF optimization).')
    mm_args.add_argument(
        '-mt', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel torsions (post-FF optimization).')
    mm_args.add_argument(
        '-mh', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae',
        help='MacroModel Hessian.')
    mm_args.add_argument(
        '-mjeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae,somename.out',
        help='MacroModel eigenmatrix (all elements). Uses Jaguar '
        'eigenvectors.')
    mm_args.add_argument(
        '-mgeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.mae,somename.log',
        help='MacroModel eigenmatrix (all elements). Uses Gaussian '
        'eigenvectors.')
    mm_args.add_argument(
        '-mp', type=str, nargs='+', action='append',
        default=[], metavar='somename.fld,somename.txt',
        help='Uses a MM3* FF file (somename.fld) and a parameter file '
        '(somename.txt) to use the current FF parameter values as data. This '
        'is used for harmonic parameter tethering.')
    # TINKER OPTIONS
    tin_args = parser.add_argument_group("tinker data types")
    tin_args.add_argument(
        '-te', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker energies (pre-FF optimization).')
    tin_args.add_argument(
        '-tea', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker energies (pre-FF optimization). Energies will be '
        'relative to the average energy.')
    tin_args.add_argument(
        '-teo', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker energies (post-FF optimization).')
    tin_args.add_argument(
        '-teao', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker energies (post-FF optimization). Energies will be '
        'relative to the average energy.')
    tin_args.add_argument(
        '-tb', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker bond lengths (pre-FF optimization).')
    tin_args.add_argument(
        '-tbo', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker bond lengths (post-FF optimization).')
    tin_args.add_argument(
        '-ta', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker angles (pre-FF optimization).')
    tin_args.add_argument(
        '-tao', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker angles (post-FF optimization).')
    tin_args.add_argument(
        '-tt', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker torsions (pre-FF optimization).')
    tin_args.add_argument(
        '-tto', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker torsions (post-FF optimization).')
    tin_args.add_argument(
        '-th', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz',
        help='Tinker Hessian.')
    tin_args.add_argument(
        '-tjeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz,somename.out',
        help='Tinker eigenmatrix (all elements). Uses Jaguar '
        'eigenvectors.')
    tin_args.add_argument(
        '-tgeig', type=str, nargs='+', action='append',
        default=[], metavar='somename.xyz,somename.log',
        help='Tinker eigenmatrix (all elements). Uses Gaussian '
        'eigenvectors.')
    # AMBER OPTIONS
    amb_args = parser.add_argument_group("amber data types")
    amb_args.add_argument(
        '-ae', type=str, nargs='+', action='append',
        default=[], metavar='somename.inpcrd,somename.prmtop',
        help='Amber energy.')
    return parser

def check_outs(filename, outs, classtype, direc):
    """
    Reads a file if necessary. Checks the output dictionary first in
    case the file has already been loaded.

    Could work on easing the use of this by somehow reducing number of
    arguments required.
    """
    logger.log(1, '>>> filename: {}'.format(filename))
    logger.log(1, '>>> outs: {}'.format(outs))
    logger.log(1, '>>> classtype: {}'.format(classtype))
    logger.log(1, '>>> direc: {}'.format(direc))
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
def collect_data(coms, inps, direc='.', sub_names=['OPT'], invert=None):
    """
    Arguments
    ---------
    invert : None or float
             If given, will modify the smallest value of the Hessian to
             this value.
    """
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
    # MACROMODEL MM3* CURRENT PARAMETER VALUES
    filenames = chain.from_iterable(coms['mp'])
    for comma_filenames in filenames:
        # FF file and parameter file.
        name_fld, name_txt = comma_filenames.split(',')
        ff = datatypes.MM3(os.path.join(direc, name_fld))
        ff.import_ff()
        ff.params = parameters.trim_params_by_file(
            ff.params, os.path.join(direc, name_txt))
        for param in ff.params:
            data.extend([datatypes.Datum(
                val=param.value,
                com='mp',
                typ='p',
                src_1=name_fld,
                src_2=name_txt,
                idx_1=param.mm3_row,
                idx_2=param.mm3_col)])
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
    # GAUSSIAN ENERGIES
    filename_s = coms['ge']
    for idx_1, filenames in enumerate(filename_s):
        temp = []
        for filename in filenames:
            log = check_outs(filename, outs, filetypes.GaussLog, direc)
            # This will be a list of lists. For example, let's say that
            # co.GAUSSIAN_ENERGIES is ['HF', 'ZeroPoint'], then
            # the 1st list in things_to_add would be the HF energies
            # and the 2nd list would be the ZP energies.
            #
            # Consider if you had ['HF', 'ZeroPoint'] as co.GAUSSIAN_ENERGIES
            # and your archive had this:
            #     HF=0.634,0.2352\ZeroPoint=0.01234,0.0164
            # The resulting things_to_add would be:
            #     things_to_add = [[0.634, 0.2352],
            #                     [0.01234, 0.0164]]
            things_to_add = []
            # Remember, thing_label is whatever you specified in
            # co.GAUSSIAN_ENERGIES.
            for thing_label in co.GAUSSIAN_ENERGIES:
                # Consider if your Gaussian log archive has the following:
                #     HF=0.234,0.1234,0.5732
                # Then, if co.GAUSSIAN_ENERGIES includes 'HF', then that
                # particular thing, or sublist that goes into things_to_add,
                # would look like:
                #     thing = ['0.234', '0.1234', '0.5732']
                # Here's another example. Consider if your archive has the
                # property "stupidproperty":
                #     stupidproperty=can,i,be,more,clear
                # Then this particular sublist, named thing, would be
                #     thing = ['can', 'i', 'be', 'more', 'clear']
                # Lastly, consider if you have this:
                #     ZeroPoint=0.12341
                # Then thing would be this:
                #     thing = ['0.12341']
                thing = log.structures[0].props[thing_label]
                # Deal with multiple structures by checking for this
                # split here.
                if ',' in thing:
                    # Note that the "stupidproperty" example would fail here
                    # because its elements can not be converted to floats.
                    thing = map(float, thing.split(','))
                    # Here, thing might look like:
                    #    thing = [0.1235235, 0.2352, 0.352345]
                else:
                    # Here it would be a list with only one element.
                    thing = [float(thing)]
                things_to_add.append(thing)
            # Initialize list of zeros. Python syntax looks funny sometimes.
            # The length of the things_to_add sublists should always be the
            # same if you're doing it right. I suppose you could add some
            # sort of assert here.
            energies = [0.] * len(things_to_add[0])
            # In this case, consider the earlier example where:
            #    things_to_add = [[0.634, 0.2352],
            #                     [0.01234, 0.0164]]
            # Here, the first thing_group would be [0.634, 0.2352] and the
            # second thing_group would be [0.01234, 0.0164].
            for thing_group in things_to_add:
                # After the loop through the 1st thing_group, we would have
                # energies = [0.634, 0.2352]. After the 2nd thing_group, we
                # would have energies = [0.634 + 0.01234, 0.2352 + 0.0164].
                for i, thing in enumerate(thing_group):
                    energies[i] += thing
            energies = [x * co.HARTREE_TO_KJMOL for x in energies]
            for i, e in enumerate(energies):
                temp.append(datatypes.Datum(
                        val=e,
                        com='ge',
                        typ='e',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=i + 1))

            # This works when HF and ZeroPoint are used. Had to make it more
            # general.
            # Revisit how structures are stored in GaussLog when you have time.
            # hf = log.structures[0].props['HF']
            # zp = log.structures[0].props['ZeroPoint']
            # if ',' in hf:
            #     hfs = map(float, hf.split(','))
            #     zps = map(float, zp.split(','))
            # else:
            #     hfs = [float(hf)]
            #     zps = [float(zp)]
            # es = []
            # for hf, zp in izip(hfs, zps):
            #     es = (hf + zp) * co.HARTREE_TO_KJMOL
            # for i, e in enumerate(es):
            #     temp.append(datatypes.Datum(
            #             val=e,
            #             com='ge',
            #             typ='e',
            #             src_1=filename,
            #             idx_1=idx_1 + 1,
            #             idx_2=i + 1))

            # Here's the old code from before we supported multiple energies.
            # I think it's helpful history for new coders trying to understand
            # how to write in new datatypes. Notice how the new code utilizes
            # idx_2.

            # hf = float(log.structures[0].props['HF'])
            # zp = float(log.structures[0].props['ZeroPoint'])
            # energy = (hf + zp) * co.HARTREE_TO_KJMOL
            # # We don't use idx_2 since we assume there is only one structure
            # # in a Gaussian .log. I think that's always the case.
            # temp.append(datatypes.Datum(
            #         val=energy,
            #         com='ge',
            #         typ='e',
            #         src_1=filename,
            #         idx_1=idx_1 + 1))

        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # MACROMODEL ENERGIES
    filenames_s = coms['me']
    ind = 'pre'
    for idx_1, filenames in enumerate(filenames_s):
        for filename in filenames:
            name_mae = inps[filename].name_mae
            mae = check_outs(name_mae, outs, filetypes.Mae, direc)
            indices = inps[filename]._index_output_mae
            # This is list of sets. The 1st value in the set corresponds to the
            # number of the structure. The 2nd value is the structure class.
            selected_structures = filetypes.select_structures(
                mae.structures, indices, ind)
            for idx_2, structure in selected_structures:
                data.append(datatypes.Datum(
                        val=structure.props['r_mmod_Potential_Energy-MM3*'],
                        com='me',
                        typ='e',
                        src_1=inps[filename].name_mae,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
    # AMBER ENERGIES
    filenames_s = coms['ae']
    for idx_1, filenames in enumerate(filenames_s):
        logger.log(1, '>>> idx_1: {}'.format(idx_1))
        logger.log(1, '>>> filenames: {}'.format(filenames))
        for idx_2, comma_sep_filenames in enumerate(filenames):
            name_1, name_2 = comma_sep_filenames.split(',')
            out = check_outs(
                comma_sep_filenames, outs, filetypes.AmberOut, direc)
            # Right now, path is a comma separated string.
            out.path = inps[comma_sep_filenames].out
            logger.log(1, '>>> out: {}'.format(out))
            energy = out.read_energy()
            data.append(datatypes.Datum(
                val=energy,
                com='ae',
                typ='e',
                src_1=name_1,
                src_2=name_2,
                idx_1=idx_1 + 1,
                idx_2=idx_2 + 1))
    # JAGUAR AVERAGE ENERGIES
    filenames_s = coms['jea']
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
                        com='jea',
                        typ='ea',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        # For this data type, we set everything relative.
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # GAUSSIAN AVERAGE ENERGIES
    filename_s = coms['gea']
    for idx_1, filenames in enumerate(filename_s):
        temp = []
        for filename in filenames:
            log = check_outs(filename, outs, filetypes.GaussLog, direc)
            things_to_add = []
            for thing_label in co.GAUSSIAN_ENERGIES:
                thing = log.structures[0].props[thing_label]
                if ',' in thing:
                    thing = map(float, thing.split(','))
                else:
                    thing = [float(thing)]
                things_to_add.append(thing)
            energies = [0.] * len(things_to_add[0])
            for thing_group in things_to_add:
                for i, thing in enumerate(thing_group):
                    energies[i] += thing
            energies = [x * co.HARTREE_TO_KJMOL for x in energies]
            for i, e in enumerate(energies):
                temp.append(datatypes.Datum(
                        val=e,
                        com='gea',
                        typ='ea',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=i + 1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # MACROMODEL AVERAGE ENERGIES
    filenames_s = coms['mea']
    ind = 'pre'
    # idx_1 is the number used to group sets of relative energies.
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            name_mae = inps[filename].name_mae
            mae = check_outs(name_mae, outs, filetypes.Mae, direc)
            indices = inps[filename]._index_output_mae
            # This is list of sets. The 1st value in the set corresponds to the
            # number of the structure. The 2nd value is the structure class.
            selected_structures = filetypes.select_structures(
                mae.structures, indices, ind)
            for idx_2, structure in selected_structures:
                temp.append(datatypes.Datum(
                        val=structure.props['r_mmod_Potential_Energy-MM3*'],
                        com='mea',
                        typ='ea',
                        src_1=inps[filename].name_mae,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # JAGUAR ENERGIES COMPARED TO OPTIMIZED MM
    filenames_s = coms['jeo']
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
                        com='jeo',
                        typ='eo',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        # For this data type, we set everything relative.
        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # GAUSSIAN ENERGIES RELATIVE TO OPTIMIZED MM
    filename_s = coms['geo']
    for idx_1, filenames in enumerate(filename_s):
        temp = []
        for filename in filenames:
            log = check_outs(filename, outs, filetypes.GaussLog, direc)
            things_to_add = []
            for thing_label in co.GAUSSIAN_ENERGIES:
                thing = log.structures[0].props[thing_label]
                if ',' in thing:
                    thing = map(float, thing.split(','))
                else:
                    thing = [float(thing)]
                things_to_add.append(thing)
            energies = [0.] * len(things_to_add[0])
            for thing_group in things_to_add:
                for i, thing in enumerate(thing_group):
                    energies[i] += thing
            energies = [x * co.HARTREE_TO_KJMOL for x in energies]
            for i, e in enumerate(energies):
                temp.append(datatypes.Datum(
                        val=e,
                        com='geo',
                        typ='eo',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=i + 1))
        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # MACROMODEL OPTIMIZED ENERGIES
    filenames_s = coms['meo']
    ind = 'opt'
    for idx_1, filenames in enumerate(filenames_s):
        for filename in filenames:
            name_mae = inps[filename].name_mae
            mae = check_outs(name_mae, outs, filetypes.Mae, direc)
            indices = inps[filename]._index_output_mae
            selected_structures = filetypes.select_structures(
                mae.structures, indices, ind)
            for idx_2, structure in selected_structures:
                data.append(datatypes.Datum(
                        val=structure.props['r_mmod_Potential_Energy-MM3*'],
                        com='meo',
                        typ='eo',
                        src_1=inps[filename].name_mae,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
    # JAGUAR ENERGIES RELATIVE TO AVERAGE COMPARED TO OPTIMIZED MM
    filenames_s = coms['jeao']
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
                        com='jeao',
                        typ='eao',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # GAUSSIAN AVERAGE ENERGIES RELATIVE TO OPTIMIZED MM
    filename_s = coms['geao']
    for idx_1, filenames in enumerate(filename_s):
        temp = []
        for filename in filenames:
            log = check_outs(filename, outs, filetypes.GaussLog, direc)
            things_to_add = []
            for thing_label in co.GAUSSIAN_ENERGIES:
                thing = log.structures[0].props[thing_label]
                if ',' in thing:
                    thing = map(float, thing.split(','))
                else:
                    thing = [float(thing)]
                things_to_add.append(thing)
            energies = [0.] * len(things_to_add[0])
            for thing_group in things_to_add:
                for i, thing in enumerate(thing_group):
                    energies[i] += thing
            energies = [x * co.HARTREE_TO_KJMOL for x in energies]
            for i, e in enumerate(energies):
                temp.append(datatypes.Datum(
                        val=e,
                        com='geao',
                        typ='eao',
                        src_1=filename,
                        idx_1=idx_1 + 1,
                        idx_2=i + 1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # MACROMODEL OPTIMIZED ENERGIES RELATIVE TO AVERAGE
    filenames_s = coms['meao']
    ind = 'opt'
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            name_mae = inps[filename].name_mae
            mae = check_outs(name_mae, outs, filetypes.Mae, direc)
            indices = inps[filename]._index_output_mae
            selected_structures = filetypes.select_structures(
                mae.structures, indices, ind)
            for idx_2, structure in selected_structures:
                temp.append(datatypes.Datum(
                        val=structure.props['r_mmod_Potential_Energy-MM3*'],
                        com='meao',
                        typ='eao',
                        src_1=inps[filename].name_mae,
                        idx_1=idx_1 + 1,
                        idx_2=idx_2 + 1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # JAGUAR BONDS
    filenames = chain.from_iterable(coms['jb'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'jb', 'pre', 'bonds'))
    # TINKER SP BONDS
    filenames = chain.from_iterable(coms['tb'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tb', 'pre', 'bonds'))
    # TINKER SP ANGLES
    filenames = chain.from_iterable(coms['ta'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'ta', 'pre', 'angles'))
    # TINKER SP TORSIONS
    filenames = chain.from_iterable(coms['tt'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tt', 'pre', 'torsions'))
    # TINKER OPTIMIZED BONDS
    filenames = chain.from_iterable(coms['tbo'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tbo', 'opt', 'bonds'))
    # TINKER OPTIMIZED ANGLE
    filenames = chain.from_iterable(coms['tao'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tao', 'opt', 'angles'))
    # TINKER OPTIMIZED ANGLE
    filenames = chain.from_iterable(coms['tto'])
    for filename in filenames:
        data.extend(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tto', 'opt', 'torsions'))
    # TINKER ENERGIES RELATIVE TO LOWEST
    filenames_s = coms['te']
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            temp.append(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'te', 'pre', 'e', idx_1 = idx_1))
        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # TINKER ENERGIES RELATIVE TO AVERAGE
    filenames_s = coms['tea']
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            temp.append(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'tea', 'pre', 'ea', idx_1 = idx_1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # TINKER OPTIMIZED ENERGIES RELATIVE LOWEST
    filenames_s = coms['teo']
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            temp.append(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'teo', 'opt', 'eo', idx_1 = idx_1))
        zero = min([x.val for x in temp])
        for datum in temp:
            datum.val -= zero
        data.extend(temp)
    # TINKER OPTIMIZED ENERGIES RELATIVE TO AVERAGE
    filenames_s = coms['teao']
    for idx_1, filenames in enumerate(filenames_s):
        temp = []
        for filename in filenames:
            temp.append(collect_structural_data_from_tinker_log(
                filename, inps, outs, direc, 'teao', 'opt', 'eao', idx_1 = idx_1))
        avg = sum([x.val for x in temp]) / len(temp)
        for datum in temp:
            datum.val -= avg
        data.extend(temp)
    # TINKER HESSIAN
    filenames = chain.from_iterable(coms['th'])
    for filename in filenames:
        xyz_struct = inps[filename].structures[0]
        num_atoms = xyz_struct.props['total atoms']
        name_hes = inps[filename].name_hes
        hes = check_outs(name_hes, outs, filetypes.TinkerHess, direc)
        hes.natoms = num_atoms
        hess = hes.hessian
        datatypes.mass_weight_hessian(hess, xyz_struct.atoms)
        # Need to figure out dummy atoms at somepoint?
        # I'm not even sure if we can use dummy atoms in TINKER.
        low_tri_idx = np.tril_indices_from(hess)
        low_tri = hess[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                val=e,
                com='th',
                typ='h',
                src_1=hes.filename,
                idx_1=x + 1,
                idx_2=y + 1)
                    for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                val=e,
                com='th',
                typ='h',
                src_1=hes.filename,
                idx_1=x + 1,
                idx_2=y + 1)
                    for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # TINKER EIGENMATRIX USING GAUSSIAN EIGENVECTORS
    filenames = chain.from_iterable(coms['tgeig'])
    for comma_filenames in filenames:
        name_xyz, name_gau_log = comma_filenames.split(',')
        name_xyz_hes = inps[name_xyz].name_hes
        xyz = check_outs(name_xyz, outs, filetypes.Tinker_xyz, direc)
        xyz_hes = check_outs(name_xyz_hes, outs, filetypes.TinkerHess, direc)
        gau_log = check_outs(name_gau_log, outs, filetypes.GaussLog, direc)
        xyz_struct = xyz.structures[0]
        num_atoms = xyz_struct.props['total atoms']
        xyz_hes.natoms = num_atoms
        hess = xyz_hes.hessian
        datatypes.mass_weight_hessian(hess, xyz_struct.atoms)
        evec = gau_log.evecs
        try:
            eigenmatrix = np.dot(np.dot(evec, hess), evec.T)
        except ValueError:
            logger.warning('Matrices not aligned!')
            logger.warning('Hessian retrieved from {}: {}'.format(
                    name_mae_log, hess.shape))
            logger.warning('Eigenvectors retrieved from {}: {}'.format(
                    name_gau_log, evec.shape))
            raise
        low_tri_idx = np.tril_indices_from(eigenmatrix)
        low_tri = eigenmatrix[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                val=e,
                com='tgeig',
                typ='eig',
                src_1=name_xyz,
                src_2=name_gau_log,
                idx_1=x + 1,
                idx_2=y + 1)
                    for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                val=e,
                com='tgeig',
                typ='eig',
                src_1=name_xyz,
                src_2=name_gau_log,
                idx_1=x + 1,
                idx_2=y + 1)
                    for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # MACROMODEL BONDS
    filenames = chain.from_iterable(coms['mb'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'mb', 'opt', 'bonds'))
    # JAGUAR ANGLES
    filenames = chain.from_iterable(coms['ja'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'ja', 'pre', 'angles'))
    # MACROMODEL BONDS
    filenames = chain.from_iterable(coms['ma'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'ma', 'opt', 'angles'))
    # JAGUAR BONDS
    filenames = chain.from_iterable(coms['jt'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'jt', 'pre', 'torsions'))
    # MACROMODEL BONDS
    filenames = chain.from_iterable(coms['mt'])
    for filename in filenames:
        data.extend(collect_structural_data_from_mae(
                filename, inps, outs, direc, sub_names, 'mt', 'opt', 'torsions'))
    # JAGUAR CHARGES
    filenames = chain.from_iterable(coms['jq'])
    for filename in filenames:
        mae = check_outs(filename, outs, filetypes.Mae, direc)
        for idx_1, structure in enumerate(mae.structures):
            for atom in structure.atoms:
                # If it doesn't have the property b_q_use_charge,
                # use it.
                # If b_q_use_charge is 1, use it. If it's 0, don't
                # use it.
                if not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']:
                    data.append(datatypes.Datum(
                            val=atom.partial_charge,
                            com='jq',
                            typ='q',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # MACROMODEL CHARGES
    filenames = chain.from_iterable(coms['mq'])
    for filename in filenames:
        name_mae = inps[filename].name_mae
        mae = check_outs(name_mae, outs, filetypes.Mae, direc)
        # Pick out the right structures. Sometimes our .com files
        # generate many structures in a .mae, not all of which
        # apply to this command.
        structures = filetypes.select_structures(
            mae.structures, inps[filename]._index_output_mae, 'pre')
        for idx_1, structure in structures:
            for atom in structure.atoms:
                if not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']:
                    data.append(datatypes.Datum(
                            val=atom.partial_charge,
                            com='mq',
                            typ='q',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # JAGUAR CHARGES EXCLUDING ALIPHATIC HYDROGENS
    filenames = chain.from_iterable(coms['jqh'])
    for filename in filenames:
        mae = check_outs(filename, outs, filetypes.Mae, direc)
        for idx_1, structure in enumerate(mae.structures):
            aliph_hyds = structure.get_aliph_hyds()
            for atom in structure.atoms:
                # If it doesn't have the property b_q_use_charge,
                # use it.
                # If b_q_use_charge is 1, use it. If it's 0, don't
                # use it.
                if (not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']) and \
                        not atom in aliph_hyds:
                    charge = atom.partial_charge
                    if atom.atom_type == 3:
                        for bonded_atom_index in atom.bonded_atom_indices:
                            bonded_atom = structure.atoms[bonded_atom_index - 1]
                            if bonded_atom in aliph_hyds:
                                charge += bonded_atom.partial_charge
                    data.append(datatypes.Datum(
                            val=charge,
                            com='jqh',
                            typ='qh',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # MACROMODEL CHARGES EXCLUDING ALIPHATIC HYDROGENS
    filenames = chain.from_iterable(coms['mqh'])
    for filename in filenames:
        name_mae = inps[filename].name_mae
        mae = check_outs(name_mae, outs, filetypes.Mae, direc)
        # Pick out the right structures. Sometimes our .com files
        # generate many structures in a .mae, not all of which
        # apply to this command.
        structures = filetypes.select_structures(
            mae.structures, inps[filename]._index_output_mae, 'pre')
        for idx_1, structure in structures:
            aliph_hyds = structure.get_aliph_hyds()
            for atom in structure.atoms:
                if (not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']) and \
                        atom not in aliph_hyds:

                    # Since the charge is always zero AS FAR AS I KNOW, this
                    # whole recalculation of the charge is totally unnecessary.
                    # However, I want users to be aware that if a situation
                    # arises that goes beyond something I experienced,
                    # uncommenting this section, thereby making it more like the
                    # code for -jqh, should solve the problem.

                    # charge = atom.partial_charge
                    # if atom.atom_type == 3:
                    #     for bonded_atom_index in atom.bonded_atom_indices:
                    #         bonded_atom = structure.atoms[bonded_atom_index - 1]
                    #         if bonded_atom in aliph_hyds:
                    #             charge += bonded_atom.partial_charge

                    data.append(datatypes.Datum(
                            # val=charge,
                            val=atom.partial_charge,
                            com='mqh',
                            typ='qh',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # JAGUAR CHARGES EXCLUDING ALL SINGLE BONDED HYDROGENS
    filenames = chain.from_iterable(coms['jqa'])
    for filename in filenames:
        mae = check_outs(filename, outs, filetypes.Mae, direc)
        for idx_1, structure in enumerate(mae.structures):
            hyds = structure.get_hyds()
            for atom in structure.atoms:
                # Check if we want to use this charge and ensure it's not a
                # hydrogen.
                if (not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']) and \
                        atom not in hyds:
                    charge = atom.partial_charge
                    # Check if it's bonded to a hydrogen.
                    for bonded_atom_index in atom.bonded_atom_indices:
                        bonded_atom = structure.atoms[bonded_atom_index - 1]
                        if bonded_atom in hyds:
                            if len(bonded_atom.bonded_atom_indices) < 2:
                                charge += bonded_atom.partial_charge
                    data.append(datatypes.Datum(
                            val=charge,
                            com='jqa',
                            typ='qa',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # MACROMODEL CHARGES EXCLUDING ALL SINGLE BONDED HYDROGENS
    filenames = chain.from_iterable(coms['mqa'])
    for filename in filenames:
        name_mae = inps[filename].name_mae
        mae = check_outs(name_mae, outs, filetypes.Mae, direc)
        # Pick out the right structures. Sometimes our .com files
        # generate many structures in a .mae, not all of which
        # apply to this command.
        structures = filetypes.select_structures(
            mae.structures, inps[filename]._index_output_mae, 'pre')
        for idx_1, structure in structures:
            hyds = structure.get_hyds()
            for atom in structure.atoms:
                if (not 'b_q_use_charge' in atom.props or \
                        atom.props['b_q_use_charge']) and \
                        atom not in hyds:
                    charge = atom.partial_charge
                    for bonded_atom_index in atom.bonded_atom_indices:
                        bonded_atom = structure.atoms[bonded_atom_index - 1]
                        if bonded_atom in hyds:
                            if len(bonded_atom.bonded_atom_indices) < 2:
                                charge += bonded_atom.partial_charge
                    data.append(datatypes.Datum(
                            val=charge,
                            com='mqa',
                            typ='qa',
                            src_1=filename,
                            idx_1=idx_1 + 1,
                            atm_1=atom.index))
    # JAGUAR HESSIAN
    filenames = chain.from_iterable(coms['jh'])
    for filename in filenames:
        jin = check_outs(filename, outs, filetypes.JaguarIn, direc)
        hess = jin.hessian
        datatypes.mass_weight_hessian(hess, jin.structures[0].atoms)
        if invert:
            evals, evecs = np.linalg.eigh(hess)
            datatypes.replace_minimum(evals, value=invert)
            hess = evecs.dot(np.diag(evals).dot(evecs.T))
        datatypes.replace_minimum(hess, value=invert)
        low_tri_idx = np.tril_indices_from(hess)
        low_tri = hess[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='jh',
                        typ='h',
                        src_1=jin.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='jh',
                        typ='h',
                        src_1=jin.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # GAUSSIAN HESSIAN
    filenames = chain.from_iterable(coms['gh'])
    for filename in filenames:
        log = check_outs(filename, outs, filetypes.GaussLog, direc)
        log.read_archive()
        # For now, the Hessian is stored on the structures inside the filetype.
        hess = log.structures[0].hess
        datatypes.mass_weight_hessian(hess, log.structures[0].atoms)
        if invert:
            # Faster to use scipy.linalg.eig or scipy.linalg.eigsh (even
            # faster).
            evals, evecs = np.linalg.eigh(hess)
            # Returns True.
            # print(np.allclose(evecs.dot(np.diag(evals).dot(evecs.T)), hess))
            datatypes.replace_minimum(evals, value=invert)
            hess = evecs.dot(np.diag(evals).dot(evecs.T))
        # Oh crap, just realized this probably needs to be mass weighted.
        # WARNING: This option may need to be mass weighted!
        low_tri_idx = np.tril_indices_from(hess)
        low_tri = hess[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='gh',
                        typ='h',
                        src_1=log.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='gh',
                        typ='h',
                        src_1=log.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # MACROMODEL HESSIAN
    filenames = chain.from_iterable(coms['mh'])
    for filename in filenames:
        # Get the .log for the .mae.
        name_log = inps[filename].name_log
        # Used to get dummy atoms.
        mae = check_outs(filename, outs, filetypes.Mae, direc)
        # Used to get the Hessian.
        log = check_outs(name_log, outs, filetypes.MacroModelLog, direc)
        hess = log.hessian
        dummies = mae.structures[0].get_dummy_atom_indices()
        hess_dummies = datatypes.get_dummy_hessian_indices(dummies)
        hess = datatypes.check_mm_dummy(hess, hess_dummies)
        low_tri_idx = np.tril_indices_from(hess)
        low_tri = hess[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='mh',
                        typ='h',
                        src_1=mae.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='mh',
                        typ='h',
                        src_1=mae.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # JAGUAR EIGENMATRIX
    filenames = chain.from_iterable(coms['jeigz'])
    for comma_sep_filenames in filenames:
        name_in, name_out = comma_sep_filenames.split(',')
        jin = check_outs(name_in, outs, filetypes.JaguarIn, direc)
        out = check_outs(name_out, outs, filetypes.JaguarOut, direc)
        hess = jin.hessian
        evec = out.eigenvectors
        datatypes.mass_weight_hessian(hess, jin.structures[0].atoms)
        datatypes.mass_weight_eigenvectors(evec, out.structures[0].atoms)
        try:
            eigenmatrix = np.dot(np.dot(evec, hess), evec.T)
        except ValueError:
            logger.warning('Matrices not aligned!')
            logger.warning('Hessian retrieved from {}: {}'.format(
                    name_in, hess.shape))
            logger.warning('Eigenvectors retrieved from {}: {}'.format(
                    name_out, evec.shape))
            raise

        # Funny way to make off-diagonal elements zero.
        # eigenmatrix = np.diag(np.diag(eigenmatrix))

        # Take diagonal into one dimensional array.
        eigenmatrix = np.diag(eigenmatrix)
        if invert:
            datatypes.replace_minimum(eigenmatrix, value=invert)
        # Turn back into a full matrix.
        eigenmatrix = np.diag(eigenmatrix)
        low_tri_idx = np.tril_indices_from(eigenmatrix)
        low_tri = eigenmatrix[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='jeigz',
                        typ='eig',
                        src_1=jin.filename,
                        src_2=out.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='jeigz',
                        typ='eig',
                        src_1=jin.filename,
                        src_2=out.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # GAUSSIAN EIGENMATRIX
    filenames = chain.from_iterable(coms['geigz'])
    for filename in filenames:
        log = check_outs(filename, outs, filetypes.GaussLog, direc)
        evals = log.evals * co.HESSIAN_CONVERSION
        if invert:
            datatypes.replace_minimum(evals, value=invert)
        eigenmatrix = np.diag(evals)
        low_tri_idx = np.tril_indices_from(eigenmatrix)
        low_tri = eigenmatrix[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='geigz',
                        typ='eig',
                        src_1=log.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='geigz',
                        typ='eig',
                        src_1=log.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # MACROMODEL EIGENMATRIX USING JAGUAR EIGENVECTORS
    filenames = chain.from_iterable(coms['mjeig'])
    for comma_sep_filenames in filenames:
        name_mae, name_out = comma_sep_filenames.split(',')
        name_log = inps[name_mae].name_log
        mae = check_outs(name_mae, outs, filetypes.Mae, direc)
        log = check_outs(name_log, outs, filetypes.MacroModelLog, direc)
        out = check_outs(name_out, outs, filetypes.JaguarOut, direc)
        hess = log.hessian
        dummies = mae.structures[0].get_dummy_atom_indices()
        hess_dummies = datatypes.get_dummy_hessian_indices(dummies)
        hess = datatypes.check_mm_dummy(hess, hess_dummies)
        evec = out.eigenvectors
        datatypes.mass_weight_eigenvectors(evec, out.structures[0].atoms)
        try:
            eigenmatrix = np.dot(np.dot(evec, hess), evec.T)
        except ValueError:
            logger.warning('Matrices not aligned!')
            logger.warning('Hessian retrieved from {}: {}'.format(
                    log.filename, hess.shape))
            logger.warning('Eigenvectors retrieved from {}: {}'.format(
                    name_out, evec.shape))
            raise
        low_tri_idx = np.tril_indices_from(eigenmatrix)
        low_tri = eigenmatrix[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='mjeig',
                        typ='eig',
                        src_1=mae.filename,
                        src_2=out.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='mjeig',
                        typ='eig',
                        src_1=mae.filename,
                        src_2=out.filename,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    # MACROMODEL EIGENMATRIX USING GAUSSIAN EIGENVECTORS
    filenames = chain.from_iterable(coms['mgeig'])
    for comma_filenames in filenames:
        name_mae, name_gau_log = comma_filenames.split(',')
        name_mae_log = inps[name_mae].name_log
        mae = check_outs(name_mae, outs, filetypes.Mae, direc)
        mae_log = check_outs(name_mae_log, outs, filetypes.MacroModelLog, direc)
        gau_log = check_outs(name_gau_log, outs, filetypes.GaussLog, direc)
        hess = mae_log.hessian
        dummies = mae.structures[0].get_dummy_atom_indices()
        hess_dummies = datatypes.get_dummy_hessian_indices(dummies)
        hess = datatypes.check_mm_dummy(hess, hess_dummies)
        evec = gau_log.evecs
        try:
            eigenmatrix = np.dot(np.dot(evec, hess), evec.T)
        except ValueError:
            logger.warning('Matrices not aligned!')
            logger.warning('Hessian retrieved from {}: {}'.format(
                    name_mae_log, hess.shape))
            logger.warning('Eigenvectors retrieved from {}: {}'.format(
                    name_gau_log, evec.shape))
            raise
        low_tri_idx = np.tril_indices_from(eigenmatrix)
        low_tri = eigenmatrix[low_tri_idx]
        if (sys.version_info > (3, 0)):
            data.extend([datatypes.Datum(
                        val=e,
                        com='mgeig',
                        typ='eig',
                        src_1=name_mae,
                        src_2=name_gau_log,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in zip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
        else:
            data.extend([datatypes.Datum(
                        val=e,
                        com='mgeig',
                        typ='eig',
                        src_1=name_mae,
                        src_2=name_gau_log,
                        idx_1=x + 1,
                        idx_2=y + 1)
                         for e, x, y in izip(
                        low_tri, low_tri_idx[0], low_tri_idx[1])])
    logger.log(15, 'TOTAL DATA POINTS: {}'.format(len(data)))
    return np.array(data, dtype=datatypes.Datum)

def collect_data_fake(coms, inps, direc='.', sub_names=['OPT']):
    """
    Generates a random data set quickly.
    """
    import random
    data = []
    filenames = flatten(coms.values())
    for idx_1, filename in enumerate(filenames):
        for idx_2 in xrange(5):
            data.append(datatypes.Datum(
                    val=random.uniform(0, 10),
                    com='rand',
                    typ='a',
                    src_1=filename,
                    idx_1=idx_1 + 1,
                    idx_2=idx_2 + 1))
    return np.array(data, dtype=datatypes.Datum)

def flatten(l):
    """
    Simple means to flatten an irregular list of lists.

    http://stackoverflow.com/questions/2158395/
        flatten-an-irregular-list-of-lists-in-python

    This goes a bit further than chain.from_iterable in that it can deal with
    an arbitrary number of nested lists.
    """
    # Move this?
    import collections
    for el in l:
        if isinstance(el, collections.Iterable) and \
          not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

def collect_structural_data_from_mae(
    name_mae, inps, outs, direc, sub_names, com, ind, typ):
    """
    Repeated code used to extract structural data from .mae files (through
    the generation of .mmo files).

    Would be nice to reduce the number of arguments. The problem here is in
    carrying through data for the generation of the Datum object.

    Not going to write a pretty __doc__ for this since I want to make so
    many changes. These changes will likely go along with modifications
    to the classes inside filetypes.
    """
    data = []
    name_mmo = inps[name_mae].name_mmo
    # The indices is jsut a list for the calculation done, 'pre' or 'opt'.
    indices = inps[name_mae]._index_output_mmo

    mmo = check_outs(name_mmo, outs, filetypes.MacroModel, direc)
    selected_structures = filetypes.select_structures(
        mmo.structures, indices, ind)
    for idx_1, structure in selected_structures:
        data.extend(structure.select_data(
                typ,
                com=com,
                com_match=sub_names,
                src_1=mmo.filename,
                idx_1=idx_1 + 1))
    return data

# Added by Tony.
# Probably want to use check_outs function at somepoint.
def collect_structural_data_from_tinker_log(
    name_xyz, inps, outs, direc, com, ind, typ, idx_1 = None):
    select_struct = {'pre':0, 'opt':1}
    data = []
    name_log = inps[name_xyz].name_log
    log = check_outs(name_log, outs, filetypes.Tinker_log, direc)
    log_structure = log.structures
    struct = log_structure[select_struct[ind]]
    # Stuff to try out hessian.
    # xyz_struct = xyz_structure[0]
    # num_atoms = xyz_struct.props['total atoms']
    # hes_structure = inps[name_xyz].hess
    # hes_structure.natoms = num_atoms
    # hessian = hes_structure.hessian()
    # Stuff to try out hessian.
    if com in ['te','teo','tea','teao']:
        energy = struct.props['energy']
        new_datum = (datatypes.Datum(
            val=energy,
            typ=typ,
            src_1=name_log,
            idx_1=idx_1 + 1))
        return(new_datum)
    else:
        data.extend(struct.select_data(
            typ,
            com=com,
            src_1=name_log))
        return(data)

def sort_commands_by_filename(commands):
    '''
    Takes a dictionary of commands like...

     {'me': [['a1.01.mae', 'a2.01.mae', 'a3.01.mae'],
             ['b1.01.mae', 'b2.01.mae']],
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
    if (sys.version_info > (3, 0)):
        for command, groups_filenames in iter(commands.items()):
            for comma_separated in chain.from_iterable(groups_filenames):
                for filename in comma_separated.split(','):
                    if filename in sorted_commands:
                        sorted_commands[filename].append(command)
                    else:
                        sorted_commands[filename] = [command]
    else:
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
        logger.log(log_level, '')
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
    # Really, this should check every data point instead of only the 1st.
    if not data[0].wht:
        compare.import_weights(data)
    if log_level:
        string = ('--' + ' LABEL '.center(22, '-') +
                  '--' + ' WEIGHT '.center(22, '-') +
                  '--' + ' VALUE '.center(22, '-') +
                  '--')
        logger.log(log_level, string)
    for d in data:
        if d.wht or d.wht == 0:
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
