#!/usr/bin/python
import argparse
import sys

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, rmsd

SYS_ARGV = ' -a LigandLibrary/binap.mae LigandLibrary/binap_b.mae -b Substrates/*.mae'.split()
SMARTS = 'C=C_[Rh+]_O'

def get_smarts_atoms(smarts, structure):
    # unique_sets should be True if the molecule is C2 symmetric.
    if structure.property['b_cs_c2']:
        something = analyze.evaluate_smarts(structure, smarts, unique_sets=True)
    else:
        something = analyze.evaluate_smarts(structure, smarts, unique_sets=False)
    print(something)
    return something

def combine(smarts, base, other):
    base_atoms = get_smarts_atoms(smarts, base)
    other_atoms = get_smarts_atoms(smarts, other)
    structure_writer = sch_struct.StructureWriter('TEST.mae')
    for base_atoms_set in base_atoms:
        for other_atoms_set in other_atoms:
            rmsd.superimpose(base, base_atoms_set, other, other_atoms_set)
            new_structure = merge_and_reform_bonds(base, base_atoms_set, other, other_atoms_set)
            structure_writer.append(new_structure)
    structure_writer.close()

def merge_and_reform_bonds(a, a_common_atoms, b, b_common_atoms):
    len_atoms = len(a.atom)
    len_common_atoms = len(a_common_atoms)
    merge_common_atoms = [x + len_atoms for x in b_common_atoms]
    merge = a.merge(b, copy_props=True)
    for merge_common_atom in merge_common_atoms:
        for b_bond in merge.atom[merge_common_atom].bond:
            identifier = [b_bond.atom1.index, b_bond.atom2.index]
            # Have to check for backwards too!
            if not any(identifier == merge_common_atoms[i:i+2] \
                       for i in xrange(len(merge_common_atoms) - 1)):
                merge_index = merge_common_atoms.index(b_bond.atom1.index)
                atom_ind_1 = a_common_atoms[merge_index]
                atom_ind_2 = b_bond.atom2.index
                merge.atom[atom_ind_1].addBond(atom_ind_2, b_bond.order)
                merge.getBond(atom_ind_1, atom_ind_2).property['b_cs_tors'] = \
                    b_bond.property['b_cs_tors']
                rca4_1 = b_bond.property['i_cs_rca4_1']
                rca4_2 = b_bond.property['i_cs_rca4_2']
                if rca4_1 == 0:
                    merge.getBond(
                        atom_ind_1, atom_ind_2).property['i_cs_rca4_1'] = 0
                    merge.getBond(
                        atom_ind_1, atom_ind_2).property['i_cs_rca4_2'] = 0
                else:
                    try:
                        rca4_1 = a_common_atoms[b_common_atoms.index(rca4_1)]
                    except ValueError:
                        rca4_1 += len_atoms - len_common_atoms
                    try:
                        rca4_2 = a_common_atoms[b_common_atoms.index(rca4_2)]
                    except ValueError:
                        rca4_2 += len_atoms - len_common_atoms
                    merge.getBond(
                        atom_ind_1, atom_ind_2).property['i_cs_rca4_1'] = \
                        rca4_1
                    merge.getBond(
                        atom_ind_1, atom_ind_2).property['i_cs_rca4_2'] = \
                        rca4_2
    merge.deleteAtoms(merge_common_atoms)
    return merge

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--group_a', nargs='+')
    parser.add_argument('-b', '--group_b', nargs='+')
    return parser

if __name__ == '__main__':
    # parser = return_parser()
    # opts = parser.parse_args(sys.argv[1:])
    # print(opts)

    # Read in reaction template.
    # Assume only one per file.
    structure_reader = sch_struct.StructureReader('ReactionTemplates/rh_hydrogenation_enamides.mae')
    base = structure_reader.next()
    structure_reader.close()

    # Assuming one ligand. Not wise.
    # structure_reader = sch_struct.StructureReader('LigandLibrary/binap_mod.mae')
    structure_reader = sch_struct.StructureReader('LigandLibrary/binap.mae')
    ligand = structure_reader.next()
    structure_reader.close()

    # Example SMARTS for matching the substrate to the base.
    get_smarts_atoms('O=C-N-C=C', base)
    # Example SMARTS for matching the ligand to the base.
    get_smarts_atoms('P-[Rh]-P', base)
    # Checking to make sure it works with the ligand too.
    get_smarts_atoms('P-[Rh]-P', ligand)
    
    combine('P-[Rh]-P', base, ligand)
    
