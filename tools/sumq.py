#!/usr/bin/python
"""
Calculates er/dr and ee/de.

Supported filetypes:
 * MacroModel conformational search .log files. NOTE: Use MacroModel's redundant
   conformer elimination before using this. For whatever reason, the energies
   reported by MacroModel's conformational search are inaccurate.
 * Schrodinger .mae files.
 * Gaussian .log files.

For now the options --max_structures and --max_energy only apply to MacroModel 
.log files.
"""
from __future__ import print_function
import argparse
import os
import re
import sys

from itertools import chain
from math import exp

K = 0.008314459848 # kJ K^-1 mol^-1
T = 300 # K
# Beta
B = 1/(K*T)
ENERGY_LABEL = 'r_mmod_Potential_Energy-MM3*'
RE_ENERGY = '(\s|\*)Conformation\s+\d+\s\(\s+(?P<energy>[\d\.\-]+)\s+kJ/mol\) was found\s+(?P<num>[\d]+)'
HARTREE_TO_KJMOL = 2625.5 # Hartree to kJ mol

def read_energy_from_macro_log(filename, 
                               max_structures=None,
                               max_energy=None):
    energies = []
    with open(filename, 'r') as f:
        energy_section = False
        for line in f:
            if '*** MC Statistics ***' in line:
                energy_section = False
            if 'Total number of structures processed =' in line:
                num_str = int(line.split()[6])
                energy_section = True
            if energy_section:
                matched = re.match(RE_ENERGY, line)
                if matched != None:
                    energy = float(matched.group('energy'))
                    num = int(matched.group('num'))
                    if max_energy and len(energies) > 0:
                        # Inefficient.
                        zero = min(energies)
                        if energy - zero > max_energy:
                            break
                    # print(energy)
                    energies.append(energy)

                    # Thought I'd try this out. Doesn't really work.

                    # Idea was that it would append the energy * the number of
                    # times that the conformation was located. Didn't really
                    # change things much.

                    # Anyway, reasoning for this was that we're really trying
                    # to explore how soft or hard the PES surface about the TS
                    # is. Many mid energy conformers may mean more than few
                    # slightly lower energy conformers.

                    # energies.extend([energy] * num)

            if max_structures:
                if len(energies) == max_structures:
                    break
    return energies

def read_energy_from_mae(filename):
    sys.path.append('~/q2mm_dev/')
    import filetypes

    mae = filetypes.Mae(filename)
    energies = [float(x.props[ENERGY_LABEL]) for x in mae.structures]
    return energies

def read_energy_from_gau_log(filename):
    """
    Also convert to kJ/mol.
    """
    sys.path.append('~/q2mm_dev/')
    import filetypes

    # This is actually a little misleading seeing as these archives only contain
    # one HF energy, and typically each file only contains one archive.
    energies = []
    file_ob = filetypes.GaussLog(filename)
    try:
        file_ob.read_archive()
        for structure in file_ob.structures:
            energies.append(float(structure.props['HF']) * HARTREE_TO_KJMOL)
        return energies
    except IndexError:
        raise

def make_relative(energies):
    """
    Makes all energies relative.
    
    Expects a list of lists, flattens it, finds the minimum, makes all energies
    in all lists relative.
    """
    zero = min(chain.from_iterable(energies))
    zero_energies = []
    for group_energies in energies:
        zero_energies.append([x - zero for x in group_energies])
    return zero_energies

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-g', '--group', metavar='filename',
        type=str, nargs='+', action='append',
        help='Group of filenames.')
    parser.add_argument(
        '-n', '--max_structures', metavar='i',
        type=int,
        help='Stop reading an individual file after reading i structures.')
    parser.add_argument(
        '-m', '--max_energy', metavar='f',
        type=float,
        help="Don't read any structures that have a relative energy above f.")
    return parser

def calc_q(energies):
    qs = []
    for group_energies in energies:
        q = sum([exp(-B*x) for x in group_energies])
        qs.append(q)
    return qs

def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)

    energies = []
    # Get all the partition function values.
    for group in opts.group:
        group_energies = []
        for filename in group:
            if filename.endswith('.log'):
                # Need system for handling both types of .log files.
                try:
                    e = read_energy_from_gau_log(filename)
                except IndexError:
                    e = read_energy_from_macro_log(
                        filename,
                        max_structures=opts.max_structures,
                        max_energy=opts.max_energy)
                group_energies.extend(e)
            elif filename.endswith('.mae'):
                e = read_energy_from_mae(filename)
                group_energies.extend(e)
        energies.append(group_energies)
        
    energies = make_relative(energies)

    # Output code.
    border = ' % CONTRIBUTION TO TOTAL '.center(50, '-')
    print(border)
    qs = calc_q(energies)
    total_q = sum(qs)
    stuff = []
    for i, q in enumerate(qs):
        ratio = q / total_q
        print('Group {}: {}'.format(i + 1, ratio))
        stuff.append(ratio)

    # Additional output for when there are only 2 isomers.
    print(' OUTPUT FOR 2 GROUPS '.center(50, '-'))
    if len(qs) == 2:
        dr12 = qs[0] / qs[1]
        dr21 = qs[1] / qs[0]
        de = (dr12 - 1) / (dr12 + 1) * 100
        print('% dr/er (Group 1 : Group 2): {}'.format(dr12))
        print('% dr/er (Group 2 : Group 1): {}'.format(dr21))
        print('% de/ee: {}'.format(abs(de)))

    print('-' * len(border))
    print('This should equal 1: {}'.format(sum(stuff)))
    
if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
