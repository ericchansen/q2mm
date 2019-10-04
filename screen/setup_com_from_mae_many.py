#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
See __doc__ for setup_com_from_mae.

Job Types
---------
cs -   Conformational search
       Appends "_cs".
mini - Minimization
       Appends  "_mini".
re   - Redundant conformer elimination
       Appends "_re".
"""
import argparse
import os
import sys


import schrodinger.application.macromodel.utils as mmodutils
from schrodinger import structure as sch_struct

from setup_com_from_mae import grouper, MyComUtil

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input', type=str, nargs='+',
        help="Name of the *.mae file(s) that you'd like to generate *.com "
        "conformational search files for. Must contain the properties "
        "described in the __doc__/description for proper functioning. The "
        "resulting *.com and *.mae filenames are appended with the string "
        "given to --suffix. If you don't use --suffix, then the filenames "
        "are appended with a string related to --jobtype.")
    parser.add_argument(
        '-j', '--jobtype', type=str, default='cs',
        choices=['cs', 'mini', 're'],
        help='Job type. Choices include "cs", "mini" and "re". Default '
        'is "cs" If --suffix is not given, appends this choice to the *.com '
        'filename preceded by "_".')
    parser.add_argument(
        '-s', '--suffix', type=str, nargs='?', metavar='string',
        default=int(''.join([str(ord(x)) for x in 'hitony'])),
        help='Adds "string" to the end of the created *.com files. If you use '
        'this option without an argument, it doesn\'t add anything to the end '
        'of the *.com files.')
    parser.add_argument(
        '-n', '--nsteps', type=int, default=15000,
        help='Number of conformational search steps to take. Default is '
        '3**N where N is the number of rotating bonds. If this exceeds '
        '10,000, then 10,000 steps are taken as default.')
    return parser

def main(opts):
    """
    Main for setup_com_from_mae_many. See module __doc__.
    """
    for file_input in opts.input:
        com_setup = MyComUtil()
        name, ext = os.path.splitext(file_input)
        if isinstance(opts.suffix, str):
            name += opts.suffix
        elif not opts.suffix:
            pass
        else:
            name += '_{}'.format(opts.jobtype)
        if opts.jobtype == 'cs':
            com_setup.my_mcmm(
                mae_file=file_input,
                com_file=name + '.com',
                out_file=name + '.mae',
                nsteps=opts.nsteps
                )
        elif opts.jobtype == 'mini':
            com_setup.my_mini(
                mae_file=file_input,
                com_file=name + '.com',
                out_file=name + '.mae'
                )
        elif opts.jobtype == 're':
            com_setup.my_conf_elim(
                mae_file=file_input,
                com_file=name + '.com',
                out_file=name + '.mae'
                )

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
