#!/usr/bin/python
from __future__ import print_function
import argparse
import copy
import logging
import logging.config

import constants as co
import filetypes as ft

logger = logging.getLogger(__name__)

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'jin', type=str,
        help='Jaguar .in file to modify.')
    parser.add_argument(
        'mae', type=str,
        help='MacroModel .mae file containing the MM charges.')
    parser.add_argument(
        'mmo', type=str,
        help='MacroModel .mmo file containing the MM energies (ELST).')
    parser.add_argument(
        '--output', '-o', type=str,
        help='Name of new .in file.')
    return parser

def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    jin = ft.JaguarIn(opts.jin)
    mae = ft.Mae(opts.mae)
    mmo = ft.MacroModel(opts.mmo)
    # Select 1st structure of .mae. Sure hope your MM charges are there.
    str_mae = mae.structures[0]
    # Select 1st structure of .mmo.
    # This structure should only contain bonds that are particular to our
    # substructure.
    str_mmo = mmo.structures[0]
    substr_atoms = atom_nums_from_bonds(str_mmo.bonds)
    # Now that we have the atoms in the substructure, we need to select
    # every other atom and get those MM charges.
    non_substr_atoms = []
    for atom in str_mae.atoms:
        if atom.index not in substr_atoms:
            non_substr_atoms.append(atom)
    # Now we just need to add these to the atomic section.
    # Hopefully the position of that section doesn't matter.
    atomic_lines = gen_atomic_section(non_substr_atoms)
    new_lines = copy.deepcopy(jin.lines)
    new_lines[1:1] = atomic_lines
    # Write the new file.
    with open(opts.output, 'w') as f:
        for line in new_lines:
            f.write(line)

def gen_atomic_section(non_substr_atoms):
    lines = []
    lines.append('&atomic\n')
    lines.append('atom esp\n')
    for atom in non_substr_atoms:
        lines.append(' {}{} {}\n'.format(
                atom.element, atom.index, atom.partial_charge))
    lines.append('&\n')
    return lines

def atom_nums_from_bonds(bonds):
    substr_atoms = []
    for bond in bonds:
        logger.log(20, 'SELECTED: {} {}'.format(bond, bond.comment))
        substr_atoms.extend(bond.atom_nums)
    substr_atoms = list(set(substr_atoms))
    logger.log(
        20, 'ATOMS USED IN SUBSTRUCTURE BONDS: {}'.format(substr_atoms))
    return substr_atoms

if __name__ == '__main__':
    import sys
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
