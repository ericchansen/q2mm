#!/usr/bin/env python
"""
Used to setup ESP calculations.

Some MM partial charges can't be fit by the FF. Examples:

1. Charges that you are not optimizing. In other words, the atom may
   be present in your reference structure, but you are not optimizing
   a bond dipole that affects this atom.
2. MM3* makes all aliphatic hydrogens have a charge of zero, and sums
   their charge into the bonded carbon.

In this case, we want the ESP calculation to keep these unfittable
partial charges at their default MM value.

This script helps setup those ESP calculations for Jaguar and
Gaussian.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse
import copy
import logging
import logging.config

import constants as co
import filetypes as ft

logger = logging.getLogger(__name__)

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    inps = parser.add_argument_group('input options')
    inps.add_argument(
        'mae', type=str, metavar='somename.mae',
        help='MacroModel .mae file containing the MM charges.')
    inps.add_argument(
        'mmo', type=str, metavar='somename.mmo',
        help='MacroModel .mmo file containing the MM energies (ELST).')
    inps.add_argument(
        '--jin', type=str, metavar='somename.in',
        help='Jaguar .in file to modify. Not required, but may make '
        'your output Jaguar .in file look better, particularly if you '
        'use custom atom types.')
    outs = parser.add_argument_group('output options')
    outs.add_argument(
        '--format', '-f', type=str, choices=['jaguar', 'gaussian'],
        default='jaguar',
        help='Select format for output.')
    outs.add_argument(
        '--output', '-o', type=str,
        help='Name of output file. If not given, prints.')
    parser.add_argument(
        '--subnames', '-s', type=str, nargs='+', default=['OPT'],
        metavar='"Substructure Name OPT"',
        help=('Names of the substructures containing parameters to '
              'optimize in a mm3.fld file. Default will include any '
              'substructure that has the word "OPT" in it.'))
    parser.add_argument(
        '--charge', '-c', type=int, default=1,
        help='Set the overall charge.')
    parser.add_argument(
        '--multiplicity', '-m', type=int, default=1,
        help='Set the multiplicity.')
    return parser

def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    # ~~~ STRUCTURE AND CHARGE READING ~~~
    mae = ft.Mae(opts.mae)
    mmo = ft.MacroModel(opts.mmo)
    # Select 1st structure of .mae. Sure hope your MM charges are there.
    str_mae = mae.structures[0]
    # Select 1st structure of .mmo.
    # This structure should only contain bonds that are particular to our
    # substructure.
    str_mmo = mmo.structures[0]
    substr_bonds = str_mmo.select_stuff('bonds', com_match=opts.subnames)
    substr_atoms = atom_nums_from_bonds(substr_bonds)
    # Now that we have the atoms in the substructure, we need to select
    # every other atom and get those MM charges.
    non_substr_atoms = []
    for atom in str_mae.atoms:
        if atom.index not in substr_atoms:
            non_substr_atoms.append(atom)
    # ~~~ WORKING ON GENERATING THE OUTPUT ~~~
    if opts.format=='jaguar':
        atomic_lines = gen_atomic_section(non_substr_atoms)
        # Now we just need to add these to the atomic section.
        # Read or generate the .in file.
        if opts.jin:
            jin = ft.JaguarIn(opts.jin)
        else:
            jin=None
        new_lines = gen_jaguar_output(mae, jin, opts.charge)
        # Add the charge section.
        new_lines[0:0] = atomic_lines
    elif opts.format == 'gaussian':
        non_substr_atom_indices = [x.index for x in non_substr_atoms]
        new_lines = gen_gaussian_output(
            mae, indices_use_charge=non_substr_atom_indices,
            title=mae.filename.split('.')[0],
            charge=opts.charge, multiplicity=opts.multiplicity)
    # ~~~ WRITE OUTPUT ~~~
    if opts.output:
        with open(opts.output, 'w') as f:
            for line in new_lines:
                f.write(line + '\n')
    else:
        for line in new_lines:
            print(line)

def gen_gaussian_output(mae, indices_use_charge=None, title='Title',
                        charge='Charge', multiplicity='Multiplicity'):
    """
    Generates strings for a Gaussian .com file.

    Returns
    -------
    list of strings
    """
    # This is stuff used on the ND CRC.
    # new_lines = ['%chk={}.chk'.format(mae.filename.split('.')[0]),
    #              '%nprocs=1',
    #              '%mem=2gb',
    #              '%lindaworkers=localhost',
    #              '# Pop=ChelpG IOp(6/20=30133)',
    #              '',
    #              title,
    #              '',
    #              '{} {}'.format(charge, multiplicity)
    #              ]
    new_lines = ['# Pop=ChelpG IOp(6/20=30133)',
                 '',
                 title,
                 '',
                 '{} {}'.format(charge, multiplicity)
                 ]
    new_lines.extend(mae.structures[0].format_coords(
            format='gauss', indices_use_charge=indices_use_charge))
    # You may want to insert your atomic radii here so you don't have to
    # manually.
    # new_lines.extend(['', 'Pd 1.5', ''])
    # Similarly, you may want to add your ECP here.
    # new_lines.extend(['Pd 0',
    #                   'sdd',
    #                   '****',
    #                   'N H P C 0',
    #                   '6-31G**',
    #                   '****',
    #                   '',
    #                   'Pd 0',
    #                   'sdd'])
    new_lines.extend(['', '', '', ''])
    return new_lines


# Need to add in option for multiplicity.
def gen_jaguar_output(mae, jin=None, charge=0):
    """
    Generates strings for a Jaguar .in file.

    Returns
    -------
    list of strings
    """
    # Read the lines from the .in file if it was provided.
    if jin:
        new_lines = [line.rstrip() for line in jin.lines]
    # If there's no Jaguar .in file provided, then put in some basic
    # ESP commands and the coordinates from the .mae file.
    else:
        # Header information.
        filename = '{}.mae'.format(mae.filename.split('.')[0])
        new_lines = [
            'MAEFILE: {}'.format(filename),
            '&gen',
            'icfit=1',
            'incdip=1']
        if charge != 0:
            new_lines.append('molchg={}'.format(charge))
        new_lines.append('&')
        # Coordinates.
        new_lines.append('&zmat')
        new_lines.extend(mae.structures[0].format_coords(format='jaguar'))
        new_lines.append('&')
    return new_lines

def gen_atomic_section(non_substr_atoms):
    """
    Generates the atomic charge section for Jaguar ESP calculations.

    Returns
    -------
    list of strings
    """
    lines = []
    lines.append('&atomic')
    lines.append('atom esp')
    for atom in non_substr_atoms:
        lines.append(' {}{} {}'.format(
                atom.element, atom.index, atom.partial_charge))
    lines.append('&')
    return lines

def atom_nums_from_bonds(bonds):
    """
    From a list of selected bonds, pick out all the atoms in those
    bonds and put them all into a list.

    Returns
    -------
    list of atoms
    """
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
