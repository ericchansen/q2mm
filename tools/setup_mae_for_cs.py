#!/usr/bin/python
"""
Adds properties to Maestro atoms and bonds.

The original plan:
1. Setup a default MCMM. Not sure how to do this. Essentially I want the results
   of an AUTO command.
2. Take those results and store them in lists or dictionaries or whatever.
3. Use Schrodinger's structure to add those properties to the substrate or ligand.

However, I don't know how to get the results from AUTO from the command line.
"""
import argparse
import os
import sys
from schrodinger import structure as sch_struct

def read_com(filename):
    """
    Reads a MacroModel .com file and extracts information related to
    conformational searches.

    Returns
    -------
    comp = list of integers
           Atom numbers for COMP atoms.
    tors = list of tuples of length 2
           Each tuple is atoms assigned to a TORS command.
    """
    comp = [] # Holds the COMP atoms. This is a flat list.
    tors = [] # Holds TORS. List of tuples of length 2.
    rca4 = [] # Holds RCA4. List of tuples of length 4.
    chig = [] # Holds CHIG. Flat list.
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split()
            if cols[0] == 'COMP':
                comp.extend(map(int, cols[1:5]))
            if cols[0] == 'TORS':
                tors.append(tuple(map(int, cols[1:3])))
            if cols[0] == 'RCA4':
                rca4.append(tuple(map(int, cols[1:5])))
            if cols[0] == 'CHIG':
                chig.extend(map(int, cols[1:5]))
    # Remove all extra 0's from comp.
    # Up to 4 COMP atoms are given per line. If 3 or less are given on a line,
    # the remaining columns are 0's. Delete those meaningless 0's.
    comp = filter(lambda x: x != 0, comp)
    chig = filter(lambda x: x != 0, chig)
    return comp, tors, rca4, chig

def add_to_mae(filename, output, comp, tors, rca4, chig):
    """
    Adds properties to atoms and bonds in a .mae file. The properties it adds
    were extracted from 

    Arguments
    ---------
    filename = string
               Name of input .mae file.
    output = string
             Name of output .mae file.
    comp = list of integers
           Atom numbers for COMP atoms.
    tors = list of tuples of length 2
           Each tuple is atoms assigned to a TORS command.
    """
    structure_reader = sch_struct.StructureReader(filename)
    structure_writer = sch_struct.StructureWriter('TEMP.mae')

    tors_set = [set(x) for x in tors]

    for structure in structure_reader:

        # Copy over which atoms should go with COMP.
        for atom in structure.atom:
            if atom.index in comp:
                atom.property['b_cs_comp'] = 1
            else:
                atom.property['b_cs_comp'] = 0
            if atom.index in chig:
                atom.property['b_cs_chig'] = 1
            else:
                atom.property['b_cs_chig'] = 0

        for bond in structure.bond:
            # I'm assuming the set method is faster? Need to run tests.
            # Order doesn't matter, so sets work here.
            # if (bond.atom1.index, bond.atom2.index) in tors or \
            #         (bond.atom2.index, bond.atom1.index) in tors:
            #     bond.property['b_cs_tors'] = 1
            # else:
            #     bond.property['b_cs_tors'] = 0
            if set((bond.atom1.index, bond.atom2.index)) in tors_set:
                bond.property['b_cs_tors'] = 1
            else:
                bond.property['b_cs_tors'] = 0

        # I thought that looping over rca4 would take less time than looping
        # over the bonds. I may be wrong. Someone should test this. The downside
        # is that I have to loop over all the bonds once at the end to set all
        # the 0's. I think this still is faster?
        for rca4_list in rca4:
            for bond in structure.bond:
                if (bond.atom1.index, bond.atom2.index) == rca4_list[1:3]:
                    bond.property['i_cs_rca4_1'] = rca4_list[0]
                    bond.property['i_cs_rca4_2'] = rca4_list[3]
                    break
                elif (bond.atom2.index, bond.atom1.index) == rca4_list[1:3]:
                    bond.property['i_cs_rca4_1'] = rca4_list[3]
                    bond.property['i_cs_rca4_2'] = rca4_list[0]
                    break
        for bond in structure.bond:
            if not 'b_cs_rca4_1' in bond.property:
                bond.property['i_cs_rca4_1'] = 0
            if not 'b_cs_rca4_2' in bond.property:
                bond.property['i_cs_rca4_2'] = 0

        structure_writer.append(structure)

    structure_writer.close()
    structure_reader.close()

    os.rename('TEMP.mae', opts.out)

def return_parser():
    parser = argparse.ArgumentParser(
        description='Adds information related to conformational searches to '
        'Maestro files.')
    parser.add_argument(
        'com', type=str,
        help='MacroModel .com file from which to get the conformational search '
        'options.')
    parser.add_argument(
        'mae', type=str,
        help='MacroModel .mae file to add properties to.')
    parser.add_argument(
        'out', type=str, nargs='?', default=None,
        help='Name of MacroModel .mae output file.')
    return parser

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    if opts.out is None:
        opts.out = opts.mae
    comp, tors, rca4, chig = read_com(opts.com)
    add_to_mae(opts.mae, opts.out, comp, tors, rca4, chig)
