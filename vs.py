#!/usr/bin/python
import argparse
import os
import sys

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, rmsd

def get_smarts_atoms(smarts, structure, unique_sets=True):
    # Not sure if I want this here or as an argument.
    if unique_sets:
        something = analyze.evaluate_smarts(
            structure, smarts, unique_sets=unique_sets)
    else:
        something = analyze.evaluate_smarts(
            structure, smarts, unique_sets=unique_sets)
    return something

def combine(smarts, dad, mom):
    """
    When properties are passed on to the children, dad's properties are saved
    over mom's (like how we pass down last names) (because we're patriarchal
    fucks).

    Arguments
    ---------
    smarts : str
    dad : Schrodinger structure
    mom : Schrodinger structure
    """
    dad_atoms = get_smarts_atoms(smarts, dad)
    mom_atoms = get_smarts_atoms(smarts, mom, unique_sets=mom.property['b_cs_c2'])
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
    # These are the same atoms as b_common_atoms, but with their new indices in
    # the merged structure.
    merge_common_atoms = [x + len_atoms for x in b_common_atoms]
    merge = a.merge(b, copy_props=True)
    for merge_common_atom in merge_common_atoms:
        for b_bond in merge.atom[merge_common_atom].bond:
            identifier = [b_bond.atom1.index, b_bond.atom2.index]
            # Have to check for backwards too!
            # Hold up. Is this already accounted for?
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
    merge.property['s_m_title'] += '-' + b.property['s_m_title']
    merge.property['s_m_entry_name'] += '-' + b.property['s_m_entry_name']
    return merge

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r', '--reaction', type=str, nargs='+',
        help='Starting structure. Contains information for connecting other '
        'structures in its Schrodinger .mae properties.')
    parser.add_argument(
        '-s', '--substrate', type=str, nargs='+',
        help='First structure(s) added to reaction.')
    parser.add_argument(
        '-l', '--ligand', type=str, nargs='+',
        help='Second structure(s) added to reaction.')
    parser.add_argument(
        '-o', '--output', type=str,
        help='Write all output structures to one file.')
    parser.add_argument(
        '-d', '--directory', type=str,
        help='Write all output structures individually to this directory.')
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

    # Probably a more efficient way to do this than to repeat the above section.
    # This sure is easy though.
    for file_rxn in opts.reaction:
        print('Reading {}.'.format(file_rxn))
        reader_rxn = sch_struct.StructureReader(file_rxn)
        for struct_rxn in reader_rxn:

            if struct_rxn.property['s_cs_stereochemistry'] == 'r':
                struct_rxn.property['s_cs_stereochemistry'] = 's'
            elif struct_rxn.property['s_cs_stereochemistry'] == 's':
                struct_rxn.property['s_cs_stereochemistry'] = 'r'
            else:
                raise ValueError('s_cs_stereochemistry must be "r" or "s".')
                
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

    if opts.output:
        structure_writer = sch_struct.StructureWriter(opts.output)
        for structure in structures:
            structure_writer.append(structure)
        structure_writer.close()
    if opts.directory:
        for structure in structures:
            structure_writer = sch_struct.StructureWriter(
                os.path.join(
                    opts.directory, 
                    structure.property['s_m_title'] +
                    '-' +
                    structure.property['s_cs_stereochemistry'] +
                    '.mae'
                    )
                )
            structure_writer.append(structure)
            structure_writer.close()
            
