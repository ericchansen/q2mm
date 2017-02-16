 #!/usr/bin/python
# -*- coding: utf-8 -*-
"""
See __doc__ for setup_com_from_mae.
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
        "resulting *.com and *.mae file will append \"_cs_\" to the name.")
    return parser

def main(opts):
    """
    Main for setup_com_from_mae_many. See module __doc__.
    """
    for file_input in opts.input:
        com_setup = MyComUtil()

        name, ext = os.path.splitext(file_input)
        file_com = name + '_cs.com'
        file_mae = name + '_cs.mae'

        com_setup.my_mcmm(
            mae_file=file_input,
            com_file=file_com,
            out_file=file_mae
            )

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
