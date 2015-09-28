#!/usr/bin/python
'''
Prints the most converged coordinates from a Gaussian log.
'''
import argparse
import logging
import sys

from filetypes import GaussLog

logging.basicConfig(level=logging.NOTSET)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('filename', type=str, help='Filename of Gaussian log.')
parser.add_argument('--all', '-a', action='store_true',
                    help='Search for all coordinates, even those outside of the optimization output.')
parser.add_argument('--coords', '-c', type=str, choices=['input', 'standard', 'both'], default='input',
                    help='Type of coordinates to read.')
parser.add_argument('--format', '-f', type=str, choices=['gauss', 'latex'], default='gauss',
                    help='Output format.')
opts = parser.parse_args(sys.argv[1:])
    
glog = GaussLog(opts.filename)
if opts.all:
    structures = glog.read_any_coords(coords_type=opts.coords)
    for structure in structures:
        output = structure.format_coords(format=opts.format)
        for line in output:
            print line
        print
else:
    structures = glog.read_optimization(opts.coords)
    best_structure = glog.get_most_converged(structures=structures)
    output = best_structure.format_coords(format=opts.format)
    for line in output:
        print line
