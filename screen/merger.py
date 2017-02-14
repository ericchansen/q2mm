#!/usr/bin/python
"""
Takes *.mae structure files and merges them.

The subsequent files contain information pertinent to merging. This information
is stored as Schrodinger structure properties and is described below. This
information is manually entered by editing the *.mae files.

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

def get_atom_numbers_from_structure_with_pattern(structure,
                                                 pattern,
                                                 first_match_only=False,
                                                 use_substructure=False):
    """
    Gets the atom indices inside a structure that match a pattern.

    Takes care of two subtle intricacies.

    1. Schrodinger has two methods to match atoms inside of a structure. The
       argument `use_substructure` selects whether to use
       `schrodinger.structutils.analyze.evaluate_substructure` or
       `schrodinger.structutils.analyze.evaluate_smarts`. One or the other may
        be more convenient for your system.

    2. The pattern may match multiple times in a structure. The argument
       `first_match_only` chooses whether to use all of the matches or just the
       first one.

    Arguments
    ---------
    structure : Schrodinger structure object
    pattern : string
    first_match_only : bool
    use_substructure : bool
    """
    if use_substructure:
        return analyze.evaluate_substructure(structure,
                                             pattern,
                                             first_match_only=first_match_only)
    else:
        return analyze.evaluate_smarts(structure,
                                       pattern,
                                       unique_sets=first_match_only)

def get_overlapping_atoms_in_both(struct_1, struct_2):
    """
    Uses properties stored inside the 2nd structure to locate a set or sets of
    matching atoms inside both structures.

    This will use all patterns that are located. As long as the structure
    property contains the string "pattern", it will attempt to locate those
    atoms. As an example, I frequently employ `s_cs_pattern`, although this
    could be extended to `s_cs_pattern_1`, `s_cs_pattern_2`, etc.

    Arguments
    ---------
    struct_1 : Schrodinger structure object
    struct_2 : Schrodinger structure object
    """
    patterns = list(search_dic_keys(struct_2.property, 'pattern'))
    print(' * PATTERN: {}'.format(patterns))
    for pattern in patterns:
        print('   * CHECKING: {}'.format(pattern))
        match_struct_1 = get_atom_numbers_from_structure_with_pattern(
            struct_1,
            pattern,
            first_match_only=struct_1.property.get(
                'b_cs_first_match_only', False),
            use_substructure=struct_1.property.get(
                'b_cs_use_substructure', False))
        if match_struct_1:
            print('     * FOUND IN: {}'.format(struct_1._getTitle()))
            match_struct_2 = get_atom_numbers_from_structure_with_pattern(
                struct_2,
                pattern,
                first_match_only=struct_2.property.get(
                    'b_cs_first_match_only', False),
                use_substructure=struct_2.property.get(
                    'b_cs_use_substructure', False))
            if match_struct_2:
                print('     * FOUND IN: {}'.format(struct_2._getTitle()))
            break
        else:
            print('     * COULDN\'T FIND IN: {}'.format(struct_2._getTitle()))
            continue
    return match_struct_1, match_struct_2

def search_dic_keys(dic, lookup):
    """
    Takes a string, looks up all the dictionary keys that contain that string
    and returns the corresponding value.

    Arguments
    ---------
    dic : dictionary
    lookup : string
    """
    for key, value in dic.iteritems():
        if lookup in key:
            yield value

def merge_structures_from_matching_atoms(struct_1, match_1, struct_2, match_2):
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

def copy_rca4(
        merge, struct_1, match_1, struct_2, match_2):
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
    return merge

def merge_in_all_ways(struct_1, struct_2):
    merges = []
    # Determine the structures that overlap.
    match_1s, match_2s = get_overlapping_atoms_in_both(struct_1, struct_2)
    print('MATCHES FROM STRUCTURE 1: {}'.format(match_1s))
    print('MATCHES FROM STRUCTURE 2: {}'.format(match_2s))
    for match_1 in match_1s:
        for match_2 in match_2s:
            print('-' * 80)
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
            merge = merge_structures_from_matching_atoms(
                struct_1, match_1, struct_2, match_2)
            merge = copy_rca4(merge, struct_1, match_1, struct_2, match_2)
            merges.append(merge)
    return merges

def merge_many_files(structures, filenames):
    new_structures = []
    for filename in filenames:
        sch_reader = sch_struct.StructureReader(filename)
        for read_structure in sch_reader:
            for old_structure in structures:
                merges = merge_in_all_ways(old_structure, read_structure)
                new_structures.extend(merges)
    return new_structures

def main(opts):
    # Load 1st group.
    filenames = opts.group[0]
    structures = []
    for filename in filenames:
        sch_reader = sch_struct.StructureReader(filename)
        structures.extend(list(sch_reader))
        sch_reader.close()
    # Now continually merge the other groups.
    for filenames in opts.group[1:]:
        structures = merge_many_files(structures, filenames)

    # Write merged structures.
    # To a single file.
    if opts.output:
        sch_writer = sch_struct.StructureWriter(opts.output)
        sch_writer.extend(structures)
        sch_writer.close()
    # To a directory.
    return structures

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
