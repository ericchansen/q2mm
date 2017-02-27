#!/usr/bin/env python
"""
See __doc__ for setup_mae_from_com.

Assumes that the *.com and *.mae file(s) have the same name.
"""
import argparse
import os
import sys
from schrodinger import structure as sch_struct

from setup_mae_from_com import add_to_mae, read_com

def return_parser():
    """
    Returns the argument parser for setup_mae_for_cs.py.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'filenames', type=str, nargs='+',
        help="Name(s) of *.com file(s) that you'd like to generate *.mae "
        "file(s) for.")
    parser.add_argument(
        '-s', '--suffix', type=str, metavar='string',
        help='Adds "string" to the end of the created *.com files.')
    return parser

def main(opts):
    """
    Main for setup_mae_from_com. See module __doc__.
    """
    for filename in opts.filenames:
        # Dealing with filenames.
        com = filename
        name, ext = os.path.splitext(filename)
        mae = name + '.mae'
        if opts.suffix:
            name += opts.suffix
        out = name + '.mae'
        # Real work.
        comp, tors, rca4, chig, torc = read_com(com)
        add_to_mae(mae, out, comp, tors, rca4, chig, torc)

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
