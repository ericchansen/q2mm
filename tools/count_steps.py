#!/usr/bin/env python
import argparse
import glob
import os
import re
import sys

RE_INPUT = '\s+Search initialized with\s+(?P<num>\d+)\s+structures from the input structure file'
RE_TOTAL = '\s+Total number of structures processed =\s+(?P<num>\d+)'

def count_steps(direc):
    filenames = glob.glob(os.path.join(
            direc, '*.log'))
    filenames.sort()
    # print(filenames)
    mcmm_steps = 0
    lmcs_steps = 0
    lmc2_steps = 0
    for filename in filenames:
        # print(filename)
        with open(filename, 'r') as f:
            search_type = None
            num_input = None
            num_total = None
            for i, line in enumerate(f):
                if search_type is None:
                    if 'Monte Carlo Multiple Minimum Search requested' in line:
                        search_type = 'MCMM'
                    if 'Large scale Low-frequency-MODe conformational search.' in line:
                        search_type = 'LMC2'
                    if 'Low-frequency-Mode Conformational Search.' in line:
                        search_type = 'LMCS'
                if num_input is None and search_type:
                    stuff = re.match(RE_INPUT, line)
                    if stuff:
                        num_input = int(stuff.group('num'))
                if num_total is None and search_type:
                    stuff = re.match(RE_TOTAL, line)
                    if stuff:
                        num_total = int(stuff.group('num'))
                if search_type and num_input and num_total:
                    break
            if search_type is None:
                print('Skipping {}.'.format(filename))
                continue
            steps = num_total - num_input + 1
            # print(num_total, num_input)
            if search_type == 'MCMM':
                mcmm_steps += steps
            elif search_type == 'LMC2':
                lmc2_steps += steps
            elif search_type == 'LMCS':
                lmcs_steps += steps
    return mcmm_steps, lmcs_steps, lmc2_steps

def print_how_to(direc, mcmm_steps, lmcs_steps, lmc2_steps):
    print('{:15.15s} MCMM: {:10d} LMCS: {:10d} LMC2: {:10d}'.format(
            direc, mcmm_steps, lmcs_steps, lmc2_steps))
    
def main(args):
    parser = argparse.ArgumentParser(
        description='Reads MacroModel conformational search logs and tells you '
        'how many steps were taken. Works for MCMM, LMCS and LMC2.')
    parser.add_argument(
        '-a', action='store_true',
        help='Search all sub directories for conformational search log files. '
        'Still displays each diretory separately.')
    parser.add_argument(
        '-d', type=str,
        help='Search the provided directory for conformational search log '
        'files and count the number of steps.')
    opts = parser.parse_args()
    if opts.d:
        mcmm_steps, lmcs_steps, lmc2_steps = count_steps(opts.d)
        print_how_to(opts.d, mcmm_steps, lmcs_steps, lmc2_steps)
    elif opts.a:
        direcs = next(os.walk('.'))[1]
        direcs.sort()
        for direc in direcs:
            mcmm_steps, lmcs_steps, lmc2_steps = count_steps(direc)
            print_how_to(direc, mcmm_steps, lmcs_steps, lmc2_steps)

if __name__ == '__main__':
    main(sys.argv[1:])
