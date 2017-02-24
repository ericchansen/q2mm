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
from itertools import izip_longest

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
        "resulting *.com and *.mae file a certain string to the name depending "
        " on the MacroModel job type.")
    parser.add_argument(
        '-j', '--jobtype', type=str, default='cs',
        choices=['cs', 'mini', 're'],
        help='Job type. Choices include "cs", "mini" and "re". Default '
        'is "cs".')
    return parser

def main(opts):
    """
    Main for setup_com_from_mae_many. See module __doc__.
    """
    for file_input in opts.input:
        com_setup = MyComUtil()
        name, ext = os.path.splitext(file_input)
        if opts.jobtype == 'cs':
            com_setup.my_mcmm(
                mae_file=file_input,
                com_file=name + '_cs.com',
                out_file=name + '_cs.mae'
                )
        elif opts.jobtype == 'mini':
            com_setup.my_mcmm(
                mae_file=file_input,
                com_file=name + '_mini.com',
                out_file=name + '_mini.mae'
                )
        elif opts.jobtype == 're':
            com_setup.my_mcmm(
                mae_file=file_input,
                com_file=name + '_re.com',
                out_file=name + '_re.mae'
                )

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
