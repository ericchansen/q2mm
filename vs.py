#!/usr/bin/python
import argparse
import sys

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, rmsd

def get_smarts_atoms(smarts, structure):
    # Not sure if I want this here or as an argument.
    if structure.property['b_cs_c2']:
        something = analyze.evaluate_smarts(structure, smarts, unique_sets=True)
    else:
        something = analyze.evaluate_smarts(structure, smarts, unique_sets=False)
    return something

def combine(smarts, dad, mom):
    """
    When properties are passed on to the children, dad's properties are saved
    over mom's (like how we pass down last names).

    Arguments
    ---------
    smarts : str
    dad : Schrodinger structure
    mom : Schrodinger structure
    """
    dad_atoms = get_smarts_atoms(smarts, dad)
    mom_atoms = get_smarts_atoms(smarts, mom)
    combined_structures = []
    for dad_atoms_set in dad_atoms:
        for mom_atoms_set in mom_atoms:
            rmsd.superimpose(dad, dad_atoms_set, mom, mom_atoms_set)
            combined_structure = merge_and_reform_bonds(
                dad, dad_atoms_set,
                mom, mom_atoms_set)
            combined_structures.append(combined_structure)
    return combined_structures

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
    parser.add_argument('-r', '--reaction', type=str, nargs='+')
    parser.add_argument('-s', '--substrate', type=str, nargs='+')
    parser.add_argument('-l', '--ligand', type=str, nargs='+')
    parser.add_argument('-o', '--output', type=str)
    return parser

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    print(opts)

    structures = []
    for file_rxn in opts.reaction:
        print('Reading {}.'.format(file_rxn))
        reader_rxn = sch_struct.StructureReader(file_rxn)
        for struct_rxn in reader_rxn:

            for file_sub in opts.substrate:
                print('Reading {}.'.format(file_sub))
                reader_sub = sch_struct.StructureReader(file_sub)
                for struct_sub in reader_sub:
                    
                    structs_rxn_sub = combine(
                        struct_rxn.property['s_cs_smiles_substrate'],
                        struct_rxn,
                        struct_sub)
                    print('{} + {} = {}'.format(
                            file_rxn, file_sub, len(structs_rxn_sub)))

                    for file_lig in opts.ligand:
                        print('Reading {}.'.format(file_lig))
                        reader_lig = sch_struct.StructureReader(file_lig)
                        for struct_lig in reader_lig:

                            for struct_rxn_sub in structs_rxn_sub:
                                structs_rxn_sub_lig = combine(
                                    struct_rxn.property['s_cs_smiles_ligand'],
                                    struct_rxn_sub,
                                    struct_lig)
                                print('{} + {} + {} = {}'.format(
                                        file_rxn, file_sub, file_lig,
                                        len(structs_rxn_sub_lig)))
                               
                                structures.extend(structs_rxn_sub_lig)
                                
                        reader_lig.close()
                reader_sub.close()
        reader_rxn.close()
        
    for file_rxn in opts.reaction:
        print('Reading {}.'.format(file_rxn))
        reader_rxn = sch_struct.StructureReader(file_rxn)
        for struct_rxn in reader_rxn:
            
            for coords in struct_rxn.getXYZ(copy=False):
                coords[0] = -coords[0]

            for file_sub in opts.substrate:
                print('Reading {}.'.format(file_sub))
                reader_sub = sch_struct.StructureReader(file_sub)
                for struct_sub in reader_sub:
                    
                    structs_rxn_sub = combine(
                        struct_rxn.property['s_cs_smiles_substrate'],
                        struct_rxn,
                        struct_sub)
                    print('{} + {} = {}'.format(
                            file_rxn, file_sub, len(structs_rxn_sub)))

                    for file_lig in opts.ligand:
                        print('Reading {}.'.format(file_lig))
                        reader_lig = sch_struct.StructureReader(file_lig)
                        for struct_lig in reader_lig:

                            for struct_rxn_sub in structs_rxn_sub:
                                structs_rxn_sub_lig = combine(
                                    struct_rxn.property['s_cs_smiles_ligand'],
                                    struct_rxn_sub,
                                    struct_lig)
                                print('{} + {} + {} = {}'.format(
                                        file_rxn, file_sub, file_lig,
                                        len(structs_rxn_sub_lig)))
                               
                                structures.extend(structs_rxn_sub_lig)
                                
                        reader_lig.close()
                reader_sub.close()
        reader_rxn.close()

    print('Generated {} structures.'.format(len(structures)))
    structure_writer = sch_struct.StructureWriter(opts.output)
    for structure in structures:
        structure_writer.append(structure)
    structure_writer.close()

    
