#!/usr/bin/python
"""
Takes .mae structure files and merges them.

The initial file, the reaction template, contains information pertinent to
merging. This information is stored as structure properties (using Schrodinger's
structure module). The information can also be manually read and entered by
editing the .mae file.

Structure Properties
--------------------
s_cs_smiles_substrate - SMILES used to pick out the matching atoms on the
                        reaction template and substrate.
s_cs_smiles_ligand    - SMILES used to pick out the matching atoms on the
                        reaction template and ligand.
b_cs_c2               - Inappropriately named.

                        SMILES can often be matched in two directions, say atoms
                        1-2-3 or 3-2-1 (ex. P-[Rh]-P is a palindrome).

                        If this is False (0), it merges the structures using
                        both palindromes. If it's True (1), it only uses
                        whichever direction it matches first.
s_cs_stereochemistry  - Inadequate for all situations we would like to apply
                        this code to.

                        Stereochemistry of the product that results from the
                        conformation in the reaction template.
"""
import argparse
import itertools
import sys

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, rmsd

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-g', '--group',
        type=str, nargs='+', action='append',
        help='Groups of structures to merge.')
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Write all output structures to one file.')
    parser.add_argument(
        '-d', '--directory',
        type=str,
        help='Write all output structures individually to this directory.')
    parser.add_argument(
        '--substructure',
        action='store_true',
        help='By default, use schrodinger.structutils.analyze.evaluate_smarts '
        'to determine overlapping atoms. If this option is used, instead it '
         'will use schrodinger.structutils.analyze.evaluate_substructure.')
    return parser

def get_atom_numbers_from(structure,
                          pattern,
                          first_match_only=True,
                          use_substructure=False):
    if use_substructure:
        return analyze.evaluate_substructure(structure,
                                             pattern,
                                             first_match_only=first_match_only)
    else:
        return analyze.evaluate_smarts(structure,
                                       pattern,
                                       unique_sets=first_match_only)

def get_overlapping_atoms(dad, mom):
    patterns = list(search_dic_keys(mom.property, 'smiles'))
    print(' * SMARTS: {}'.format(patterns))
    for pattern in patterns:
        print('   * CHECKING: {}'.format(pattern))
        match_dad = analyze.evaluate_smarts(
            dad, pattern, unique_sets=True)
        if match_dad:
            print('     * FOUND IN: {}'.format(dad._getTitle()))
            match_mom = analyze.evaluate_smarts(
                mom, pattern, unique_sets=True)
            if match_mom:
                print('     * FOUND IN: {}'.format(mom._getTitle()))
            break
        else:
            print('     * COULDN\'T FIND IN: {}'.format(mom._getTitle()))
            continue
    # WILL HAVE TO REPEAT THIS FOR EVALUATE SUBSTRUCTURE!
    return match_dad, match_mom

def search_dic_keys(dic, lookup):
    for key, value in dic.iteritems():
        if lookup in key:
            yield value

def merge(struct_1, match_1, struct_2, match_2):
    """
    Combines two structures.

    Structures should already be superimposed.

    Arguments
    ---------
    struct_1 : Schrodinger structure
    match_1 : list of integers
              Atom indices for the superimposed atoms in struct_1
    struct_2 : Schrodinger structure
    match_2 : list of integers
              Atom indices for the superimposed atoms in struct_2
    """
    merge = struct_1.merge(struct_2, copy_props=True)

    # Number of atoms in original structure.
    num_atoms = len(struct_1.atom)
    common_atoms_2 = [x + num_atoms for x in match_2]
    common_atoms_2 = [merge.atom[x] for x in common_atoms_2]
    common_atoms_1 = [merge.atom[x] for x in match_1]

    print(' * AFTER MERGE:')
    print('   * {:<30} {} {}'.format(
        struct_2._getTitle(),
        [x.index for x in common_atoms_2],
        [x.atom_type_name for x in common_atoms_2]))
    print('ATOMS IN ORIGINAL STRUCTURE: {:>5}'.format(num_atoms))
    print('ATOMS IN MERGED STRUCTURE:   {:>5}'.format(len(merge.atom)))

    # Look at all the common atoms in struct_2.
    for i, (common_atom_1, common_atom_2) in enumerate(
            itertools.izip(common_atoms_1, common_atoms_2)):
        print('CHECKING COMMON ATOM {}:'.format(i + 1))

        print(' * ORIGINAL ATOM:      {:>4}/{}'.format(
            common_atom_1.index,
            common_atom_1.atom_type_name))
        for original_bond in common_atom_1.bond:
            print('   * BOND:             {:>4}/{} {:>4}/{}'.format(
                original_bond.atom1.index, original_bond.atom1.atom_type_name,
                original_bond.atom2.index, original_bond.atom2.atom_type_name))

        print(' * NEW ATOM:           {:>4}/{}'.format(
            common_atom_2.index,
            common_atom_2.atom_type_name))
        for merge_bond in common_atom_2.bond:
            print('   * BOND:             {:>4}/{} {:>4}/{}'.format(
                merge_bond.atom1.index, merge_bond.atom1.atom_type_name,
                merge_bond.atom2.index, merge_bond.atom2.atom_type_name))
            atom1 = common_atoms_1[
                common_atoms_2.index(merge_bond.atom1)]

            # These bonds already exist in the original structure.
            # We want to copy any new properties from the bonds in the merged
            # structure into the original bonds.
            if merge_bond.atom2 in common_atoms_2:
                atom2 = common_atoms_1[
                    common_atoms_2.index(merge_bond.atom2)]

                # Bond that we want to copy properties to.
                bond = merge.getBond(atom1, atom2)
                print('     * UPDATING:       {:>4}/{} {:>4}/{}'.format(
                    atom1.index, atom1.atom_type_name,
                    atom2.index, atom2.atom_type_name))

            else:
                atom2 = merge_bond.atom2
                print('     * ADDING:         {:>4}/{} {:>4}/{}'.format(
                    atom1.index, atom1.atom_type_name,
                    atom2.index, atom2.atom_type_name))

                atom1.addBond(atom2.index, merge_bond.order)
                bond = merge.getBond(atom1, atom2)

            # print('OLD PROPS: {}'.format(bond.property))
            for k, v in merge_bond.property.iteritems():
                if k not in bond.property or not bond.property[k]:
                    bond.property.update({k: v})
            # print('NEW PROPS: {}'.format(bond.property))

    # Delete duplicate atoms once you copied all the data.
    merge.deleteAtoms(common_atoms_2)

    merge.property['s_m_title'] += '-' + struct_2.property['s_m_title']
    merge.property['s_m_entry_name'] += \
        '-' + struct_2.property['s_m_entry_name']
    return merge

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])

    # Temporarily read the structures one at a time.
    sch_reader = sch_struct.StructureReader(opts.group[0][0])
    struct_1 = sch_reader.next()
    sch_reader.close()

    sch_reader = sch_struct.StructureReader(opts.group[1][0])
    struct_2 = sch_reader.next()
    sch_reader.close()

    # Determine the structures that overlap.
    match_1, match_2 = get_overlapping_atoms(struct_1, struct_2)
    # WILL HAVE TO EXPAND FOR MULTIPLE MATCHES?
    match_1 = match_1[0]
    match_2 = match_2[0]
    print(' * ALIGNING:')
    print('   * {:<30} {} {}'.format(
        struct_1._getTitle(),
        match_1,
        [struct_1.atom[x].atom_type_name for x in match_1]))
    print('   * {:<30} {} {}'.format(
        struct_2._getTitle(),
        match_2,
        [struct_2.atom[x].atom_type_name for x in match_2]))
    # Overlay the matching atoms.
    rmsd.superimpose(struct_1, match_1, struct_2, match_2)
    merge = merge(struct_1, match_1, struct_2, match_2)

    # Deal with RCA4. It can be done independently of all the merges. How nice!
    rca4s = []
    for bond in struct_2.bond:
        if bond.property['i_cs_rca4_1']:
            rca4 = [bond.property['i_cs_rca4_1'],
                    bond.atom1.index,
                    bond.atom2.index,
                    bond.property['i_cs_rca4_2']]
            rca4s.append(rca4)
    print('RCA4:     {}'.format(rca4s))

    # Now update the RCA4 atom indices to match the structure post merging and
    # deletion.
    # Need this for later.
    num_atoms = len(struct_1.atom)
    new_match_2 = [x + num_atoms for x in match_2]
    # Contains new RCA4 commands.
    new_rca4s = []
    for rca4 in rca4s:
        new_rca4 = []
        for x in rca4:
            if x in match_2:
                # If the RCA4 atom is one of the matching/duplicate/common
                # atoms, it's going to get deleted. This replaces the index of
                # that atom with the matching atom in the 1st structure.
                new_rca4.append(match_1[match_2.index(x)])
            else:
                # The atoms in 2nd structure will always be added after the
                # atoms in the 1st structure. This adjusts the atom indices
                # appropriately.
                new_index = x + num_atoms
                # If matching/duplicate/common atoms occur in the list before
                # this one, those atoms are going to get deleted. We need to
                # account for them disappearing.
                atoms_in_str_2_before_this_one = \
                    sum(i < new_index for i in new_match_2)
                new_index -= atoms_in_str_2_before_this_one
                new_rca4.append(new_index)
        new_rca4s.append(new_rca4)
    print('RCA4 NEW: {}'.format(new_rca4s))

    print(' * UPDATING RCA4:')
    # Now have to update the bonds RCA4 properties.
    for rca4 in new_rca4s:
        bond = merge.getBond(rca4[1], rca4[2])
        print('   * BOND:     {:>4}    {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
            bond.property['i_cs_rca4_1'],
            bond.atom1.index,
            bond.atom1.atom_type_name,
            bond.atom2.index,
            bond.atom2.atom_type_name,
            bond.property['i_cs_rca4_2']))
        bond.property['i_cs_rca4_1'] = rca4[0]
        bond.property['i_cs_rca4_2'] = rca4[3]
        print('     * UPDATE: '
              '{:>4}/{:2} {:>4}/{:2} {:>4}/{:2} {:>4}/{:2}'.format(
            merge.atom[bond.property['i_cs_rca4_1']].index,
            merge.atom[bond.property['i_cs_rca4_1']].atom_type_name,
            bond.atom1.index,
            bond.atom1.atom_type_name,
            bond.atom2.index,
            bond.atom2.atom_type_name,
            merge.atom[bond.property['i_cs_rca4_2']].index,
            merge.atom[bond.property['i_cs_rca4_2']].atom_type_name))

        # It turns out that the properties for bond 1-2 are stored to the same
        # object as 2-1. Therefore, the RCA4 will only align for the former way
        # of designating the bond.

        # bond = merge.getBond(rca4[2], rca4[1])
        # print('   * BOND:     {:>4}    {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
        #     bond.property['i_cs_rca4_1'],
        #     bond.atom1.index,
        #     bond.atom1.atom_type_name,
        #     bond.atom2.index,
        #     bond.atom2.atom_type_name,
        #     bond.property['i_cs_rca4_2']))
        # bond.property['i_cs_rca4_1'] = rca4[3]
        # bond.property['i_cs_rca4_2'] = rca4[0]
        # print('     * UPDATE: '
        #       '{:>4}/{:2} {:>4}/{:2} {:>4}/{:2} {:>4}/{:2}'.format(
        #     merge.atom[bond.property['i_cs_rca4_1']].index,
        #     merge.atom[bond.property['i_cs_rca4_1']].atom_type_name,
        #     bond.atom1.index,
        #     bond.atom1.atom_type_name,
        #     bond.atom2.index,
        #     bond.atom2.atom_type_name,
        #     merge.atom[bond.property['i_cs_rca4_2']].index,
        #     merge.atom[bond.property['i_cs_rca4_2']].atom_type_name))

    # We're done! Hooray!
    structure_writer = sch_struct.StructureWriter('cats.mae')
    structure_writer.append(merge)
    structure_writer.close()
