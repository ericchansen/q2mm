#!/usr/bin/python
"""
Helps with merging Schrodinger structures.

Still need to add superposition.
"""
import argparse
import sys

from schrodinger import structure as sch_struct

# Not specified via the command line.
# Should be commented for general use.
# Removes these properties from any structures.
# Just used for making cleaner looking Maestro files.
PROPERTIES_TO_REMOVE = \
    [
    's_m_Source_File',
    's_m_Source_Path',
    's_m_entry_id',
    'i_m_Source_File_Index',
    'i_m_ct_format',
    'i_m_ct_stereo_status',
    
    's_st_Chirality_1',
    's_st_Chirality_2'
    's_st_Chirality_3',
    
    'b_mmod_Minimization_Converged-MM3*',
    'i_mmod_Times_Found-MM3*',
    'r_mmod_RMS_Derivative-MM3*'
    ]

# In case you don't want to use the command line.
# And for testing.
# INPUT = 'collection.mae'
# OUTPUT = 'out.mae'
# MERGE = 'merge.mae'
# REMOVE_FROM_INPUT = \
#     [
#     15, 16, 56, 17, 22, # Amide backbone
#     23, 59, # Double bond
#     28, 44, 45, 46, 47, 48, # Phenyl C
#     75, 76, 77, 78, 79 # Phenyl H
#     ]
# REMOVE_FROM_MERGE = \
#     [
#     65, 69, 71, 70, 67, # Phenyl H
#     61, 62, 64, 68, 66, 63, # Phenyl C
#     52, 53, 54, 55, # Methyl
#     ]
# BONDS = \
#     [
#     '18-49', # Connect N-terminus
#     '14-56' # Connect C-terminus
#     ]

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument(
        '--input', '-i', type=str,
        help='Input filename.')
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output filename.')
    parser.add_argument(
        '--merge', '-m', type=str,
        help='File containing structure to merge.')
    parser.add_argument(
        '--remove_from_input', '-ri', nargs='+', type=int,
        help='Atom numbers to remove from the inputs.')
    parser.add_argument(
        '--remove_from_merge', '-rm', nargs='+', type=int,
        help='Atom numbers to remove from the merged structure.')
    parser.add_argument(
        '--bonds', '-b', nargs='+', type=str,
        help='Bonds to form. Ex.) 1-2 will form a bond between input atom 1 and '
        'merge atom 2.')
    return parser

# Uses Schrodinger's deleteAtoms function.
# If you try to remove them one at a time, the indices will change.
def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)

    # Remove. Only for testing. Or if you're too lazy to use the command line,
    # here's another way to do things.
    # opts.input = INPUT
    # opts.output = OUTPUT
    # opts.merge = MERGE
    # opts.remove_from_input = REMOVE_FROM_INPUT
    # opts.remove_from_merge = REMOVE_FROM_MERGE
    # opts.bonds = BONDS
    
    opts.remove_from_input.sort()
    opts.remove_from_merge.sort()

    reader_input = sch_struct.StructureReader(opts.input)
    reader_merge = sch_struct.StructureReader(opts.merge)
    writer = sch_struct.StructureWriter(opts.output)

    # Currently only designed to merge one structure.
    # In other words, no 2D combination arrays (although going that step
    # further wouldn't be challenging).
    structure_merge = list(reader_merge)[0]
    structure_merge.deleteAtoms(opts.remove_from_merge)

    # Work on input structures.
    for i, structure in enumerate(reader_input):

        # It's ugly to do this inside of the loop, but I want to access the
        # structures inside reader_input.
        if i == 0:
            # The ordering of these lists is important.
            atom_input_new = []
            atom_merge_new = []
            # Figure out what the new atom numbers should be.
            # It's kind of a pain because we have to figure out where the atom
            # number exists post-deletion and post-merge.
            for bond in opts.bonds:
                # For the input structure.
                atom_input, atom_merge = map(int, bond.split('-'))
                atoms_less_than = sum(
                    x < atom_input for x in opts.remove_from_input)
                atom_input_new.append(atom_input - atoms_less_than)
                # For the merge structure.
                atoms_less_than = sum(
                    x < atom_merge for x in opts.remove_from_merge)
                atom_merge_new.append(
                    atom_merge - atoms_less_than + len(structure.atom) \
                        - len(opts.remove_from_input))

        # May want to comment out.
        for prop in PROPERTIES_TO_REMOVE:
            structure.property.pop(prop, None)
        structure.deleteAtoms(opts.remove_from_input)
        structure_new = structure.merge(structure_merge, copy_props=True)
        for x, y in zip(atom_input_new, atom_merge_new):
            # Default is to form a single bond. Could make this more fancy
            # later by expanding the argument parsing.
            structure_new.addBond(x, y, 1)
        writer.append(structure_new)

    writer.close()
    reader_merge.close()
    reader_input.close()

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)
