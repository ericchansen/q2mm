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
    tors = [] # Holds TORS. List of tuples.
    with open(filename, 'r') as f:
        for line in f:
            cols = line.split()
            if cols[0] == 'COMP':
                comp.extend(map(int, cols[1:5]))
            if cols[0] == 'TORS':
                tors.append(tuple(map(int, cols[1:3])))
    # Remove all extra 0's from comp.
    # Up to 4 COMP atoms are given per line. If 3 or less are given on a line,
    # the remaining columns are 0's. Delete those meaningless 0's.
    comp = filter(lambda x: x != 0, comp)
    return comp, tors

def add_to_mae(filename, output, comp, tors):
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
        structure_writer.append(structure)

        # Surprisingly, this doesn't work.
        # Schrodinger can write atom properties like this, but for whatever
        # reason, it doesn't write bonds like this.
        # Oddly, if the data exists in the .mae, it will, however, read the
        # property correctly. It's a one way street. Why? Who knows.
        # for bond in structure.bond:
        #     if set((bond.atom1.index, bond.atom2.index)) in tors_set:
        #         bond.property['b_cs_tors'] = 1
        #     else:
        #         bond.property['b_cs_tors'] = 0

        # Is it something to do with writing a list at the wrong time?
        # Well, this doesn't work.
        # new_bonds = deepcopy(structure.bond)
        # for bond in new_bonds:
        #     if set((bond.atom1.index, bond.atom2.index)) in tors_set:
        #         bond.property['b_cs_tors'] = 1
        #     else:
        #         bond.property['b_cs_tors'] = 0
        # structure.bond = new_bonds

    structure_writer.close()
    structure_reader.close()

    # It seriously upsets me that I have to do this. The method above should
    # work.
    
    # This cheap work around only works when only i_m_from, i_m_to and i_m_order
    # are defined in the .mae. If you want a more complete solution, see how
    # .mae files were read in the oldest version of filetypes.py on my GitHub.
    new_lines = []
    with open('TEMP.mae', 'r') as f:
        bond_section = False
        bond_colon_pos = None
        for i, line in enumerate(f):
            # The ordering of this crap below matters (again, I'm frustrated
            # that Schrodinger's StructureWriter doesn't write the
            # bond.property dictionary like it does for atom.property.

            # This will stop the next section from execting.
            # This means we're done.
            if bond_colon_pos and ':::' in line:
                bond_section = False

            # This will only trigger when we're past the 1st ::: inside the bond
            # section.
            if bond_section and bond_colon_pos:
                cols = line.split()
                # Let's be explicit for your sake.
                col_index, col_atom1, col_atom2, col_order = map(int, cols)
                if set((col_atom1, col_atom2)) in tors_set:
                    cols.append(1)
                else:
                    cols.append(0)
                # Now put it back into the right format.
                line = '  {}\n'.format(' '.join(map(str, cols)))

            # Adding labels/keys for the bond.property dictionary.
            if 'm_bond' in line:
                bond_section = True
            if bond_section and ':::' in line and not bond_colon_pos:
                bond_colon_pos = i

            new_lines.append(line)
    os.remove('TEMP.mae')

    new_lines.insert(bond_colon_pos, '  b_cs_tors\n')
    with open(output, 'w') as f:
        f.writelines(new_lines)

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
    comp, tors = read_com(opts.com)
    add_to_mae(opts.mae, opts.out, comp, tors)
