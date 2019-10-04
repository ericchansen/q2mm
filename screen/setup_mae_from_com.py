#!/usr/bin/env python
"""
Adds properties to Maestro atoms and bonds in *.mae files from MacroModel
conformational search *.com files.

1. Make a *.mae file for your structure, whether it be a reaction template,
   ligand, substrate or whatever.
2. Use the Maestro GUI to setup a MacroModel conformational search *.com file.
3. Use this script to add the `TORS`, `COMP`, `RCA4` and `CHIG` commands from
   the *.com file to the corresponding *.mae.
4. You can then use `vs.py` to combine partial structures, which will maintain
   all the properties needed to automatically setup conformational searches.
5. Use `setup_cs.py` or `setup_cs_many.py` to automatically generate the *.com
   files for MacroModel conformational searches from these combined *.mae
   structures.
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
    comp : list of integers
           Atom numbers for COMP atoms.
    tors : list of tuples of length 2
           Each tuple is atoms assigned to a TORS command.
    rca4 : list of tuples of length 4
           Each tuple is the indices of 4 atoms used to describe a ring break
           required for MacroModel conformational searches.
    chig : list of integers
           Atom indices for chiral centers.
    torc : list of tuples of length 6
           Each tuple is the indices of 4 atoms used to describe a torsion
           check and 2 real numbers used to describe the abs(min) and abs(max)
           torsion values.
    """
    comp = [] # Holds the COMP atoms. This is a flat list.
    tors = [] # Holds TORS. List of tuples of length 2.
    rca4 = [] # Holds RCA4. List of tuples of length 4.
    chig = [] # Holds CHIG. Flat list.
    torc = [] # Holds TORC. List of tuples of length 6.
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split()
            if cols[0] == 'COMP':
                comp.extend(map(int, cols[1:5]))
            if cols[0] == 'TORS':
                tors.append([int(x) for x in cols[1:3]])
            if cols[0] == 'RCA4':
                x = [int(x) for x in cols[1:5]]
                if x[2] < x[1]:
                    x.reverse()
                rca4.append(tuple(x))
            if cols[0] == 'TORC':
                x = [int(x) for x in cols[1:5]]
                if x[2] < x[1]:
                    x.reverse()
                x.append(float(cols[5]))
                x.append(float(cols[6]))
                torc.append(tuple(x))
            if cols[0] == 'CHIG':
                chig.extend(map(int, cols[1:5]))
    # Remove all extra 0's from comp and chig.
    # Up to 4 COMP atoms are given per line. If 3 or less are given on a line,
    # the remaining columns are 0's. Delete those meaningless 0's.
    comp = [x for x in comp if x != 0]
    chig = [x for x in chig if x != 0]
    return comp, tors, rca4, chig, torc

def add_to_mae(filename, output, comp, tors, rca4, chig, torc):
    """
    Adds properties to atoms and bonds in a *.mae file. The properties it adds
    were extracted from MacroModel conformational search *.com files.

    Arguments
    ---------
    filename : string
               Name of input .mae file.
    output   : string
               Name of output .mae file.
    comp     : list of integers
               Atom numbers for COMP atoms.
    tors     : list of tuples of length 2
               Each tuple is atoms assigned to a TORS command.
    rca4     : list of tuples of length 4
               Each tuple is the indices of 4 atoms used to describe a ring
               break required for MacroModel conformational searches.
    chig     : list of integers
               Atom indices for chiral centers.
    torc     : list of tuples of length 6
               Each tuple is the indices of 4 atoms used to describe a torsion
               check and 2 real numbers used to describe the abs(min) and
               abs(max) value of a torsion.
    """
    structure_reader = sch_struct.StructureReader(filename)
    structure_writer = sch_struct.StructureWriter('TEMP.mae')

    tors_set = [set(x) for x in tors]

    for structure in structure_reader:

        # Copy over which atoms should go with COMP.
        print('SETUP COMP:')
        for one_comp in comp:
            atom = structure.atom[one_comp]
            atom.property['b_cs_comp'] = 1
            print(' *        {:>4}/{:2}'.format(
                atom.index, atom.atom_type_name))

        # Copy over which atoms should go with CHIG.
        print('SETUP CHIG:')
        for one_chig in chig:
            atom = structure.atom[one_chig]
            atom.property['b_cs_chig'] = 1
            print(' *        {:>4}/{:2}'.format(
                atom.index, atom.atom_type_name))

        # Set all remaining b_cs_comp and b_cs_chig to 0.
        for atom in structure.atom:
            if not atom.index in comp:
                atom.property['b_cs_comp'] = 0
            if not atom.index in chig:
                atom.property['b_cs_chig'] = 0

        print('SETUP TORS:')
        for one_tors in tors:
            bond = structure.getBond(one_tors[0], one_tors[1])
            bond.property['b_cs_tors'] = 1
            print(' *        {:>4}/{:2} {:>4}/{:2}'.format(
                      bond.atom1.index,
                      bond.atom1.atom_type_name,
                      bond.atom2.index,
                      bond.atom2.atom_type_name))

        # Add RCA4 properties.
        print('SETUP RCA4:')
        for one_rca4 in rca4:
            bond = structure.getBond(one_rca4[1], one_rca4[2])
            bond.property['i_cs_rca4_1'] = one_rca4[0]
            bond.property['i_cs_rca4_2'] = one_rca4[3]
            print(' * {:>4}   {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
                      bond.property['i_cs_rca4_1'],
                      bond.atom1.index,
                      bond.atom1.atom_type_name,
                      bond.atom2.index,
                      bond.atom2.atom_type_name,
                      bond.property['i_cs_rca4_2']))

        print('SETUP TORC:')
        for one_torc in torc:
            bond = structure.getBond(one_torc[1], one_torc[2])
            if 'i_cs_torc_a1' not in bond.property or \
               not bond.property['i_cs_torc_a1']:
                bond.property['i_cs_torc_a1'] = one_torc[0]
                bond.property['i_cs_torc_a4'] = one_torc[3]
                bond.property['r_cs_torc_a5'] = one_torc[4]
                bond.property['r_cs_torc_a6'] = one_torc[5]
                # Might be nice to expand this to include the min and max torsion
                # values.
                print(' * {:>4}   {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
                    bond.property['i_cs_torc_a1'],
                    bond.atom1.index,
                    bond.atom1.atom_type_name,
                    bond.atom2.index,
                    bond.atom2.atom_type_name,
                    bond.property['i_cs_torc_a4']))
            else:
                bond.property['i_cs_torc_b1'] = one_torc[0]
                bond.property['i_cs_torc_b4'] = one_torc[3]
                bond.property['r_cs_torc_b5'] = one_torc[4]
                bond.property['r_cs_torc_b6'] = one_torc[5]
                # Might be nice to expand this to include the min and max torsion
                # values.
                print(' * {:>4}   {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
                    bond.property['i_cs_torc_b1'],
                    bond.atom1.index,
                    bond.atom1.atom_type_name,
                    bond.atom2.index,
                    bond.atom2.atom_type_name,
                    bond.property['i_cs_torc_b4']))
        # Set i_cs_rca4_1, i_cs_rca4_2 and b_cs_tors to 0 for all other bonds.
        for bond in structure.bond:
            if not 'i_cs_rca4_1' in bond.property:
                bond.property['i_cs_rca4_1'] = 0
            if not 'i_cs_rca4_2' in bond.property:
                bond.property['i_cs_rca4_2'] = 0
            if not 'b_cs_tors' in bond.property:
                bond.property['b_cs_tors'] = 0
            if not 'i_cs_torc_a1' in bond.property:
                bond.property['i_cs_torc_a1'] = 0
            if not 'i_cs_torc_a4' in bond.property:
                bond.property['i_cs_torc_a4'] = 0
            if not 'r_cs_torc_a5' in bond.property:
                bond.property['r_cs_torc_a5'] = 0
            if not 'r_cs_torc_a6' in bond.property:
                bond.property['r_cs_torc_a6'] = 0
            if not 'i_cs_torc_b1' in bond.property:
                bond.property['i_cs_torc_b1'] = 0
            if not 'i_cs_torc_b4' in bond.property:
                bond.property['i_cs_torc_b4'] = 0
            if not 'r_cs_torc_b5' in bond.property:
                bond.property['r_cs_torc_b5'] = 0
            if not 'r_cs_torc_b6' in bond.property:
                bond.property['r_cs_torc_b6'] = 0
        structure_writer.append(structure)

    structure_writer.close()
    structure_reader.close()

    os.rename('TEMP.mae', output)
    print('WROTE: {}'.format(output))

def return_parser():
    """
    Returns the argument parser for setup_mae_for_cs.py.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'com', type=str,
        help='MacroModel *.com file from which to get the conformational '
        'search options.')
    parser.add_argument(
        'mae', type=str,
        help='MacroModel *.mae file to add properties to.')
    parser.add_argument(
        'out', type=str, nargs='?', default=None,
        help='Name of MacroModel *.mae output file. If left blank, this script '
        'will overwrite the original *.mae file.')
    return parser

def main(com, mae, out=None):
    # Rewrite input *.mae if the output filename isn't provided.
    if not out:
        out = mae
    comp, tors, rca4, chig, torc = read_com(com)
    print('READ: {}'.format(com))
    print(' * COMP: {}'.format(comp))
    print(' * TORS: {}'.format(tors))
    print(' * RCA4: {}'.format(rca4))
    print(' * CHIG: {}'.format(chig))
    print(' * TORC: {}'.format(torc))
    add_to_mae(mae, out, comp, tors, rca4, chig, torc)

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts.com, opts.mae, opts.out)
