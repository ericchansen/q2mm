#!/usr/bin/python
from __future__ import print_function
import argparse
import os
import re
import sys

from math import exp

K = 0.008314459848 # kJ K^-1 mol^-1
T = 300 # K
# Beta
B = 1/(K*T)
ENERGY_LABEL = 'r_mmod_Potential_Energy-MM3*'
RE_ENERGY = '(\s|\*)Conformation\s+\d+\s\(\s+(?P<energy>[\d\.\-]+)\s+kJ/mol\) was found\s+(?P<num>[\d]+)'

def read_log(opts):
    # energy_dic = {}
    energy_dic = []
    for filename in opts.filename:
        energies = []
        with open(filename, 'r') as f:
            # print(filename)
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
                        if opts.max_energy and len(energies) > 0:
                            # Inefficient.
                            zero = min(energies)
                            if energy - zero > opts.max_energy:
                                break
                        # print(energy)
                        energies.append(energy)

                        # Thought I'd try this out. Doesn't really work.
                        # energies.extend([energy] * num)

                if opts.max_structures:
                    if len(energies) == opts.max_structures:
                        break
        # energy_dic[filename] = energies
        energy_dic.append((filename, energies))
    return energy_dic

def read_mae(opts):
    sys.path.append('~/q2mm_dev/')
    import filetypes

    # energy_dic = {}
    energy_dic = []
    for filename in opts.filename:
        energies = []
        mae = filetypes.Mae(filename)
        # energy_dic[filename] = \
            # [float(x.props[ENERGY_LABEL]) for x in mae.structures]
        energy_dic.append(
            (filename, [float(x.props[ENERGY_LABEL]) for x in mae.structures]))
    return energy_dic

def main(args):
    parser = argparse.ArgumentParser(
        description='Calculates er/dr and ee/de from MacroModel conformational '
        'search .log files or .mae files (default is to read .log). '
        "NOTE: Use MacroModel's redundant conformer elimination before using "
        'this. The direct results from a conformational search have odd '
        'energies.')
    parser.add_argument(
        'filename', nargs='+')
    parser.add_argument(
        '-n', '--max_structures', type=int, metavar='i',
        help='Stop reading the file after reading i structures.')
    parser.add_argument(
        '-m', '--max_energy', type=float, metavar='f',
        help="Don't read any structures that have a relative energy above f.")
    parser.add_argument(
        '--mae', action='store_true',
        help='Use the code for reading .mae files rather than .log files.')
    opts = parser.parse_args()

    if opts.mae:
        energy_dic = read_mae(opts)
    else:
        energy_dic = read_log(opts)
    assert len(energy_dic) == 2, 'Too much! :('
    
    # partition_dic = {}
    # for filename, energies in energy_dic.iteritems():
    partition_dic = []
    for x in energy_dic:
        filename = x[0]
        energies = x[1]
        q = 0.
        for energy in energies:
            q += exp(-B*energy)
        partition_dic.append((filename, q))
        # partition_dic[filename] = q

    # print('Ratio ({} / {}): {}'.format(
    #         opts.filename[0], opts.filename[1],
    #         partition_dic[opts.filename[0]] / partition_dic[opts.filename[1]]))
    # print('Ratio ({} / {}): {}'.format(
    #         opts.filename[1], opts.filename[0],
    #         partition_dic[opts.filename[1]] / partition_dic[opts.filename[0]]))

    # print(opts.filename)
    name = os.path.split(opts.filename[0])[-1]
    name = os.path.splitext(name)[0]
    dr = partition_dic[0][1] / partition_dic[1][1]
    de = (dr - 1) / (dr + 1) * 100
    other_dr = partition_dic[1][1] / partition_dic[0][1]
    print('{},{},{},{}'.format(name, dr, de, other_dr))

    # r = partition_dic[0][1] / (partition_dic[0][1] + partition_dic[1][1])
    # s = partition_dic[1][1] / (partition_dic[0][1] + partition_dic[1][1])
    # print( ( (s - r) / (r + s) ) * 100)


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
