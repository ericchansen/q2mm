#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takes *.mae structure files and merges them.

The input *.mae files contain information pertinent to merging. This information
is stored as Schrödinger structure properties and is described below. This
information is manually entered by editing the *.mae files.

Structure Properties
--------------------
s_cs_pattern           - SMILES used to pick out the matching atoms. This will
                         actually accept any property containing "pattern".
b_cs_first_match_only  - SMILES can often be matched in two directions, say
                         atoms 1-2-3 or 3-2-1 (ex. P-[Rh]-P is a palindrome).

                         If this is false (0), it merges the structures using
                         both palindromes. If it's true (1), it only uses
                         whichever direction it matches first.
b_cs_substructure      - If true, use `evaluate_substructure` to find atom
                         indices from the pattern, else use `evaluate_smarts`.
b_cs_both_enantionmers - If true, will also use the other enantiomer of this
                         structure. It does this by simply inverting all of the
                         x coordinates for the atoms.
b_cs_full_match_only   - If the pattern uses optional atoms, and some of those
                         optional atoms don't match (return atom number 0), then
                         don't include those matches.

Atom Properties
---------------
Fill in later. For now, refer to setup_com_from_mae and setup_mae_from_com.

Bond Properties
---------------
See atom properties.
"""
# This whole script could be made a lot more efficient by creating a class
# containing struct_1 and struct_2 and creating some sort of dictionary to look
# up the atom indices in the merged structure. This whole process is repeated
# too often.
import argparse
import copy
import itertools
import os
import re
import sys

from schrodinger import structure as sch_struct
from schrodinger.structutils import analyze, measure, rmsd, build

ATOMS_TO_MOVE = ['RU','IR','RH','D1']

DEBUG = False

def return_parser():
    """
    Parser for merge.
    """
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
        '-m', '--mini',
        action='store_true',
        help='Attempt to minimize merged structures using MacroModel and '
        'MM3*. This is a completely unrestrained minimization with the final, '
        'completed, merged structures. Several other optimizations will take '
        'place during the merging procedure regardless of whether this option '
        'is used. I have found that this option can sometimes be harmful. Use '
        'it at your own risk. It does have potential though. Just requires '
        'some more testing before I\'ll recommend it.')
    return parser

def main(opts):
    """
    Main for merge.

    Returns
    -------
    list of Schrodinger structures
    """
    structures = merge_many_filenames(opts.group)
    print('-' * 80)
    print('END NUMBER STRUCTURES: {}'.format(len(structures)))
    if opts.mini:
        structures = mini(structures)
    new_structures = []
    for structure in structures:
        new_structures.append(add_chirality(structure))
    structures = new_structures
    # All output below.
    # Write structures to a single file.
    if opts.output:
        print('OUTPUT FILE: {}'.format(opts.output))
        sch_writer = sch_struct.StructureWriter(opts.output)
        sch_writer.extend(structures)
        sch_writer.close()
    # Write structures to a directory.
    if opts.directory:
        print('OUTPUT DIRECTORY: {}'.format(opts.directory))
        for structure in structures:
            path = make_unique_filename(
                os.path.join(
                    opts.directory,
                    structure.property['s_m_title'] + '.mae'))
            filename = os.path.basename(path)
            print(' * WRITING : {}'.format(filename))
            sch_writer = sch_struct.StructureWriter(path)
            sch_writer.append(structure)
            sch_writer.close()
    return structures

def merge_many_filenames(list_of_lists):
    """
    Returns merged structures for a list of lists of filenames.

    Arguments
    ---------
    list_of_lists : list of lists of filenames of *.mae
    """
    # Setup list for first group of filenames/structures.
    structures = []
    for filename in list_of_lists[0]:
        structures.extend(read_filename(filename))
    print('TOTAL NUM. STRUCTURES: {}'.format(len(structures)))
    # Iterate over groups of filenames/structures.
    for filenames in list_of_lists[1:]:
        new_structures = []
        for filename in filenames:
            new_structures.extend(read_filename(filename))
        # Update existing list of structures after combining with the new
        # structures.
        structures = list(merge_many_structures(structures, new_structures))
        print('TOTAL NUM. STRUCTURES: {}'.format(len(structures)))
    return list(structures)

def read_filename(filename):
    """
    Just helps with the logging.

    Arguments
    ---------
    filename : string

    Returns
    -------
    list of Schrodinger structure objects
    """
    # print('>>> filename: {}'.format(filename))
    structures = []
    sch_reader = sch_struct.StructureReader(filename)
    for structure in sch_reader:
        for enantiomer in load_enantiomers(structure):
            structures.append(enantiomer)
    sch_reader.close()
    print('{} : {} structures (including enantiomers)'.format(
        filename, len(structures)))
    return structures

def load_enantiomers(structure):
    """
    Generator to yield both enantiomers of a structure.

    Yields both enantiomers depending on the property b_cs_both_enantiomers.

    Enantiomers are generated by inversing the x coordinate of every atom.

    b_cs_both_enantiomers
     * True  - Yield input structure and enantiomer
     * False - Yield input structure

    Arguments
    ---------
    structure : Schrödinger structure object

    Yields
    ------
    Schrödinger structure objects
    """
    # I dunno if I like this style. Seems to work though.
    yield structure
    if structure.property.get('b_cs_both_enantiomers', False):
        print(' - LOADING OTHER ENANTIOMER: {}'.format(
            structure.property['s_m_title']))
        other_enantiomer = copy.deepcopy(structure)
        for coords in other_enantiomer.getXYZ(copy=False):
            coords[0] = -coords[0]
        structures = [other_enantiomer]
        yield structures[0]

def merge_many_structures(structures_1, structures_2):
    """
    Iterator for combining two lists of structures.

    Arguments
    ---------
    structures_1 : list of Schrodinger structures
    structures_2 : list of Schrodinger structures

    Yields
    ------
    Schrodinger structure
    """
    for structure_1 in structures_1:
        for structure_2 in structures_2:
            for structure in merge(structure_1, structure_2):
                yield structure

def merge(struct_1, struct_2):
    """
    Takes two Schrödinger structures and combines them.

    Uses structure properties containing the string "pattern" to determine which
    atoms to overlap. If there are multiple pattern matches, it will try to use
    all of them.

    Arguments
    ---------
    struct_1 : Schrödinger structure object
    struct_2 : Schrödinger structure object

    Yields
    ------
    Schrödinger structure objects
    """
    # Determine the structures that overlap.
    match_1s, match_2s = get_overlapping_atoms_in_both(struct_1, struct_2)
    print('MATCHES FROM STRUCTURE 1: {}'.format(match_1s))
    print('MATCHES FROM STRUCTURE 2: {}'.format(match_2s))
    seen = set()
    for match_1 in match_1s:
        # Eliminate duplicates 1st.
        for match_2 in match_2s:
            print('-' * 80)
            # Remove the zero's if they exist.
            new_match_1, new_match_2 = remove_index_from_both_if_equals_zero(
                match_1, match_2)
            print('TRIMMED: {} {}'.format(new_match_1, new_match_2))
            tup = tuple(new_match_2)
            if tup not in seen:
                seen.add(tup)
                if struct_2.property.get('b_cs_first_match_only', False):
                    seen.add(tup[::-1])
                print(' - Unique! Continuing.')
                # Just to look good.
                print('-' * 80)
                print(' * ALIGNING:')
                print('   * {:<30} {} {}'.format(
                    struct_1._getTitle(),
                    new_match_1,
                    [struct_1.atom[x].atom_type_name for x in new_match_1]))
                print('   * {:<30} {} {}'.format(
                    struct_2._getTitle(),
                    new_match_2,
                    [struct_2.atom[x].atom_type_name for x in new_match_2]))
                # Real work below.
                rmsd.superimpose(struct_1, new_match_1, struct_2, new_match_2)
                yield merge_structures_from_matching_atoms(
                    struct_1, new_match_1, struct_2, new_match_2)
            else:
                print(' - Not unique! Skipping.')

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
    struct_1 : Schrödinger structure object
    struct_2 : Schrödinger structure object

    Returns
    -------
    match_struct_1 : list of list of integers
    match_struct_2 : list of list of integers
    """
    # Use the patterns from struct_2.
    patterns = list(search_dic_keys(struct_2.property, 'pattern'))
    # Determine whether to use analyze.evaluate_smarts or
    # analyze.evaluate_substructure from struct_2 (this needs to match the
    # pattern from struct_2).
    use_substructure = struct_2.property.get('b_cs_use_substructure', False)
    print(' * PATTERNS: {}'.format(patterns))
    for pattern in patterns:
        print('   * CHECKING: {}'.format(pattern))
        match_struct_1 = get_atom_numbers_from_structure_with_pattern(
            struct_1,
            pattern,
            first_match_only=struct_1.property.get(
                'b_cs_first_match_only', False),
            use_substructure=use_substructure)
        if match_struct_1:
            print('     * FOUND IN: {}'.format(struct_1._getTitle()))
            match_struct_2 = get_atom_numbers_from_structure_with_pattern(
                struct_2,
                pattern,
                first_match_only=struct_2.property.get(
                    'b_cs_first_match_only', False),
                use_substructure=use_substructure)
            if match_struct_2:
                print('     * FOUND IN: {}'.format(struct_2._getTitle()))
            break
        else:
            print('     * COULDN\'T FIND IN: {}'.format(struct_2._getTitle()))
            continue
    # This is an interesting way to ensure we have actually found something for
    # match_struct_1 and match_struct_2.
    try:
        match_struct_1
        match_struct_2
    except UnboundLocalError as e:
        print('ERROR: {} {}'.format(
            struct_1.property['s_m_title'],
            struct_2.property['s_m_title']))
        raise e

    # 1.) Eliminate all with zero if b_cs_full_match_only.
    print('>>> match_struct_1: {}'.format(match_struct_1))
    print('>>> match_struct_2: {}'.format(match_struct_2))
    if struct_1.property.get('b_cs_full_match_only', False):
        new_match_struct_1 = []
        for match in match_struct_1:
            if 0 not in match:
                new_match_struct_1.append(match)
    else:
        new_match_struct_1 = match_struct_1
    if struct_2.property.get('b_cs_full_match_only', False):
        new_match_struct_2 = []
        for match in match_struct_2:
            if 0 not in match:
                new_match_struct_2.append(match)
    else:
        new_match_struct_2 = match_struct_2
    print('>>> new_match_struct_1: {}'.format(new_match_struct_1))
    print('>>> new_match_struct_2: {}'.format(new_match_struct_2))

    # Sometimes a match is made that isn't what is wanted by the user and 
    # incorporates an aromatic where it should not be. This prevents aryl
    # aromatic rings that have 2 or more atoms in a match to be used as a
    # match. -TR
    new_new_match_struct_1 = []
    for match in new_match_struct_1:
        use_match = True
        for ring in struct_1.ring:
            atoms_in_ring = ring.getAtomList()
            matched_atoms_in_ring = 0
            for atom in match:
                if atom in atoms_in_ring:
                    matched_atoms_in_ring += 1
            if matched_atoms_in_ring >= 2:
                if ring.isAromatic():
                    if not struct_1.property.get('b_cs_accept_aromatic', False):
                        use_match = False
        if use_match:
            new_new_match_struct_1.append(match)

    new_new_match_struct_2 = []
    for match in new_match_struct_2:
        use_match = True
        for ring in struct_2.ring:
            atoms_in_ring = ring.getAtomList()
            matched_atoms_in_ring = 0
            for atom in match:
                if atom in atoms_in_ring:
                    matched_atoms_in_ring += 1
            if matched_atoms_in_ring >= 2:
                if ring.isAromatic():
                    if not struct_2.property.get('b_cs_accept_aromatic', False):
                        use_match = False
        if use_match:
            new_new_match_struct_2.append(match)
    print('>>> new_new_match_struct_1: {}'.format(new_new_match_struct_1))
    print('>>> new_new_match_struct_2: {}'.format(new_new_match_struct_2))

    # 2.) Eliminate zeroes indices from within lists.
    # Actually, maybe it's best to do this upon application in a case by case
    # basis.

    # This only worked if the lists were the same length.
    # print('>>> match_struct_1: {}'.format(match_struct_1))
    # print('>>> match_struct_2: {}'.format(match_struct_2))
    # new_match_struct_1 = []
    # new_match_struct_2 = []
    # for sub_match_struct_1, sub_match_struct_2 in itertools.izip(
    #         match_struct_1, match_struct_2):
    #     sub_match_struct_1, sub_match_struct_2 = \
    #         remove_index_from_both_if_equals_zero(
    #             sub_match_struct_1, sub_match_struct_2)
    #     new_match_struct_1.append(sub_match_struct_1)
    #     new_match_struct_2.append(sub_match_struct_2)
    # print('>>> new_match_struct_1: {}'.format(new_match_struct_1))
    # print('>>> new_match_struct_2: {}'.format(new_match_struct_2))
    return new_new_match_struct_1, new_new_match_struct_2

def get_atom_numbers_from_structure_with_pattern(structure,
                                                 pattern,
                                                 first_match_only=False,
                                                 use_substructure=False):
    """
    Gets the atom indices inside a structure that match a pattern.

    Takes care of two subtle intricacies.

    1. Schrödinger has two methods to match atoms inside of a structure. The
       argument `use_substructure` selects whether to use
       `schrodinger.structutils.analyze.evaluate_substructure` or
       `schrodinger.structutils.analyze.evaluate_smarts`. One or the other may
        be more convenient for your system.

    2. The pattern may match multiple times in a structure. The argument
       `first_match_only` chooses whether to use all of the matches or just the
       first one.

    Arguments
    ---------
    structure : Schrödinger structure object
    pattern : string
    first_match_only : bool
    use_substructure : bool

    Returns
    -------
    list of integers
    """
    # print(">>> structure: {}".format(structure))
    # print(">>> pattern: {}".format(pattern))
    # print(">>> first_match_only: {}".format(first_match_only))
    # print(">>> use_substructure: {}".format(use_substructure))
    if use_substructure:
        atom_numbers = analyze.evaluate_substructure(
            structure,
            pattern,
            first_match_only=first_match_only)
        # I'm not sure how the logic for Schrodinger's first match function
        # works, but in the case for evaluate_substructure() it does not
        # always returns a single match. The following logic should correct
        # this without distrubing anything else. - Tony
        # Wonder if we should move this check outside the if/else in case the
        # same behavior happens with evaluate_smarts(). - Eric
        if first_match_only and len(atom_numbers) > 1:
            del atom_numbers[1:]
    else:
        atom_numbers = analyze.evaluate_smarts(
            structure,
            pattern,
            unique_sets=first_match_only)
    return atom_numbers

def remove_index_from_both_if_equals_zero(a, b):
    """
    Scans over two lists. If any elements equal False/None, then remove that
    index from both lists.

    Arguments
    ---------
    a : list (of integers usually)
    b : list (of integers usually)

    Returns
    -------
    a : list
    b : list
    """
    assert len(a) == len(b), "Lists must be of same length!"
    # This prevents the code from breaking if the user supplies a and b as the
    # same object. It would probably be even better to check if they're the
    # exact same object than create new objects.
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    for i in range(len(a) - 1, -1, -1):
        if not a[i] or not b[i]:
            a.pop(i)
            b.pop(i)
    return a, b

def merge_structures_from_matching_atoms(struct_1, match_1, struct_2, match_2):
    """
    Combines two structures.

    Structures should already be superimposed.

    THIS IS THE HEART OF THIS PYTHON MODULE. DOES SO MUCH STUFF. SUCH WOW.

    If you're smart, it might be nice to encapsulate this a bit more. It may do
    too much right now, making it cumbersome to manipulate.

    Arguments
    ---------
    struct_1 : Schrödinger structure
    match_1 : list of integers
              Atom indices for the superimposed atoms in struct_1
    struct_2 : Schrödinger structure
    match_2 : list of integers
              Atom indices for the superimposed atoms in struct_2

    Returns
    -------
    Schrödinger structure
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
    print('ATOMS IN NEW STRUCTURE:      {:>5}'.format(
        len(merge.atom) - len(common_atoms_1)))
    print('-' * 80)
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
        print(' * MATCHING ATOM:           {:>4}/{}'.format(
            common_atom_2.index,
            common_atom_2.atom_type_name))
        # If a user is templating struct2 onto struct1, but struct1 has a wild
        # card indicated for one of the matching atoms then we need to replace
        # the atom type with that of struct2. This allows more variablity and
        # flexibility to merge. I still forsee many problems. -TR
        if common_atom_1.atom_type == 64:
            common_atom_1.atom_type = common_atom_2.atom_type
            common_atom_1.color = common_atom_2.color
        #common_atom_1.atom_type = common_atom_2.atom_type
        common_atom_1.color = common_atom_2.color
        #For somereason the rest of the properties of the atom class aren't 
        #updated when the atom_type is changed. Here we set the atom type to
        #what the atom type is, which is redundant, but it resets the atomic
        #number and weight. The AtomName isn't important, but it makes it look
        #pretty in the final mae file.
        common_atom_1._setAtomType(common_atom_1.atom_type)
        common_atom_1._setAtomName(str(common_atom_1.element) + str(common_atom_1.index))

        # Below are alternatives options for which atoms to keep. Currently, the
        # coordinates of the atoms from struct_1 are kept.

        # Okay, let's try copying the XYZ from common_atom_2 into common_atom_1.
        # This fixes some problems, but causes some other problems.
        # common_atom_1.x = common_atom_2.x
        # common_atom_1.y = common_atom_2.y
        # common_atom_1.z = common_atom_2.z

        # This keeps the average coordinates.
        # common_atom_1.x = (common_atom_1.x + common_atom_2.x) / 2
        # common_atom_1.y = (common_atom_1.y + common_atom_2.y) / 2
        # common_atom_1.z = (common_atom_1.z + common_atom_2.z) / 2

        for merge_bond in common_atom_2.bond:
            print('   * BOND:             {:>4}/{} {:>4}/{}'.format(
                merge_bond.atom1.index, merge_bond.atom1.atom_type_name,
                merge_bond.atom2.index, merge_bond.atom2.atom_type_name))

            # This is the atom in struct_1 that matches the 1st atom of the bond
            # in struct_2.
            # Actually, is this necessary? Isn't this just common_atom_1?
            atom1 = common_atoms_1[
                common_atoms_2.index(merge_bond.atom1)]

            # These bonds already exist in the original structure.

            # UPDATE: Just kidding! Jess found a case where the bond doesn't
            # exist in the original structure. See q2mm/q2mm issue #40.

            # We want to copy any new properties from the bonds in the merged
            # structure into the original bonds.
            if merge_bond.atom2 in common_atoms_2:
                atom2 = common_atoms_1[
                    common_atoms_2.index(merge_bond.atom2)]
                # Bond that we want to copy properties to.
                # Surprise! This can return None. You'd think that should raise
                # an exception.
                bond = merge.getBond(atom1, atom2)
                if bond is None:
                    atom1.addBond(atom2.index, merge_bond.order)
                    print('     * ADDED:         {:>4}/{} {:>4}/{}'.format(
                        atom1.index, atom1.atom_type_name,
                        atom2.index, atom2.atom_type_name))
                else:
                    print('     * UPDATED:       {:>4}/{} {:>4}/{}'.format(
                        atom1.index, atom1.atom_type_name,
                        atom2.index, atom2.atom_type_name))
            # If the bond doesn't exist in struct_1, we want to make a new one.
            else:
                atom2 = merge_bond.atom2
                atom1.addBond(atom2.index, merge_bond.order)
                print('     * ADDED:         {:>4}/{} {:>4}/{}'.format(
                    atom1.index, atom1.atom_type_name,
                    atom2.index, atom2.atom_type_name))

            bond = merge.getBond(atom1, atom2)
            for k, v in merge_bond.property.iteritems():
                # Here, bond is the duplicate bond in struct_1 or the new bond.
                if k not in bond.property or not bond.property[k]:
                    # For i_cs_rca4_1, i_cs_rca4_2, i_cs_torc_a1, etc.
                    # these values may not be correct. That's okay though! They
                    # get fixed soon by the function add_bond_prop. What's
                    # important is that values like i_cs_torc_a5 get copied.
                    # (because these values don't get changed by atom number).
                    bond.property.update({k: v})

    print('-' * 80)
    fix_torsions = get_torc(struct_1, struct_2, match_1, match_2)
    print('TORSION RESTRAINTS: {}'.format(fix_torsions))

    # Delete duplicate atoms once you copied all the data.
    merge.deleteAtoms(common_atoms_2)
    # This code is so dumb.
    merge = add_bond_prop(merge,
                          struct_1, match_1,
                          struct_2, match_2,
                          name='rca4')
    merge = add_bond_prop(merge,
                          struct_1, match_1,
                          struct_2, match_2,
                          name='torca')
    merge = add_bond_prop(merge,
                          struct_1, match_1,
                          struct_2, match_2,
                          name='torcb')

    merge.property['s_m_title'] += '_' + struct_2.property['s_m_title']
    merge.property['s_m_entry_name'] += \
        '_' + struct_2.property['s_m_entry_name']

    # I have encoutered problems in macromodel for metal centers with
    # multiple bond (e.g. M-Cp or M-Ph). In order for macromodel to read
    # these structures, the metal and dummy atoms have to be at the end
    # of the atom listing (this is a really silly bug). This code rearranges
    # the atom in this correct order. Interest refers to the atoms of
    # interest to reorder.
    reordered_atoms = []
    original_wo_interest = []
    interest = []
    for atom in merge.atom:
        if atom.atom_type_name in ATOMS_TO_MOVE:
            interest.append(atom.index)
        else:
            original_wo_interest.append(atom.index)
    reordered_atoms.extend(original_wo_interest)
    reordered_atoms.extend(interest)
    merge = build.reorder_atoms(merge,reordered_atoms)
    
#    print(reordered_atoms)
    for bond in merge.bond:
        for prop in bond.property:
            initial_index = bond.property[prop]
            if 'i_cs' in prop and initial_index:
                for i,old_atom_index in enumerate(reordered_atoms):
                    if old_atom_index == initial_index:
                        bond.property[prop] = i+1
    frozen_atoms = []
    for frozen_atom_index in range(1, num_atoms+1):
        initial_index = frozen_atom_index
        for i,old_atom_index in enumerate(reordered_atoms):
            if old_atom_index == initial_index:
                frozen_atoms.append(i+1)


    # Minimize the structure.
    # Freeze atoms in struct_1.
    # Also enforce the TORC commands.
    merge = mini(
        [merge],
        #frozen_atoms=range(1, num_atoms + 1),
        frozen_atoms=frozen_atoms,
        fix_torsions=fix_torsions)[0]
    # Short conformational sampling.
#    merge = mcmm(
#        [merge],
#        #frozen_atoms=range(1, num_atoms + 1))[0]
#        frozen_atoms=frozen_atoms)[0]
#    # Do another minimization, this time without frozen atoms.
    #if fix_torsions:
    #    merge = mini(
    #        [merge],
    #        fix_torsions=fix_torsions)[0]
    merge = mini(
        [merge],  
        fix_torsions=fix_torsions)[0]
#    merge = mcmm([merge])[0]
    return merge

def add_chirality(structure):
    """
    Uses Schrödinger's utilities to assign chirality to atoms, and then adds
    that information to the title and entry name of structures.
    """
    chirality_dic = analyze.get_chiral_atoms(structure)
    string = '_'
    for key, value in chirality_dic.iteritems():
        string += '{}{}'.format(key, value.lower())
    structure.property['s_m_title'] += string
    structure.property['s_m_entry_name'] += string
    return structure

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

def get_torc(struct_1, struct_2, match_1, match_2):
    """
    Generates FXTA commands for MacroModel from TORC commands.

    Very repetitive code. Whole thing should get an update when a better atom to
    atom conversion is used.
    """
    num_atoms = len(struct_1.atom)
    new_match_2 = [x + num_atoms for x in match_2]
    fix_torsions = []
    for bond in struct_2.bond:
        if bond.property['i_cs_torc_a1']:
            atoms = [
                bond.property['i_cs_torc_a1'],
                bond.atom1.index,
                bond.atom2.index,
                bond.property['i_cs_torc_a4']
                ]
            torsion = struct_2.measure(
                atoms[0], atoms[1], atoms[2], atoms[3])
            # Update atom numbers.
            new_atoms = []
            for atom in atoms:
                if atom in match_2:
                    new_atoms.append(match_1[match_2.index(atom)])
                else:
                    new_index = atom + num_atoms
                    atoms_in_str_2_before_this_one = \
                        sum(i < new_index for i in new_match_2)
                    new_index -= atoms_in_str_2_before_this_one
                    new_atoms.append(new_index)
            fix_torsions.append((
                new_atoms[0],
                new_atoms[1],
                new_atoms[2],
                new_atoms[3],
                torsion))
        # Both can happen.
        if bond.property['i_cs_torc_b1']:
            atoms = [
                bond.property['i_cs_torc_b1'],
                bond.atom1.index,
                bond.atom2.index,
                bond.property['i_cs_torc_b4']
                ]
            torsion = struct_2.measure(
                atoms[0], atoms[1], atoms[2], atoms[3])
            # Update atom numbers.
            new_atoms = []
            for atom in atoms:
                if atom in match_2:
                    new_atoms.append(match_1[match_2.index(atom)])
                else:
                    new_index = atom + num_atoms
                    atoms_in_str_2_before_this_one = \
                        sum(i < new_index for i in new_match_2)
                    new_index -= atoms_in_str_2_before_this_one
                    new_atoms.append(new_index)
            fix_torsions.append((
                new_atoms[0],
                new_atoms[1],
                new_atoms[2],
                new_atoms[3],
                torsion))
    return fix_torsions

def add_bond_prop(merge, struct_1, match_1, struct_2, match_2, name=None):
    """
    Takes the RCA4 and TORC properties from two structures and properly combines
    them into the merged structures.

    RCA4 and TORC properties are stored in Schrödinger bond properties:
     * i_cs_rca4_1
     * i_cs_rca4_2
     * i_cs_torc_a1
     * i_cs_torc_a4
     * r_cs_torc_a5
     * r_cs_torc_a6
     * i_cs_torc_b1
     * i_cs_torc_b4
     * r_cs_torc_b5
     * r_cs_torc_b6

    Arguments
    ---------
    merge : Schrödinger structure object
            This is the result of merging struct_1 and struct_2 using match_1
            and match_2 as patterns.
    struct_1 : Schrödinger structure object
    match_1 : string
    struct_2 : Schrödinger structure object
    match_2 : string
    name : string
           "rca4", "torca", or "torcb"

    Returns
    -------
    merge : Updated bonds with new RCA4 properties
    """
    if name == 'rca4':
        string = 'RCA4'
        str1 = 'i_cs_rca4_1'
        str2 = 'i_cs_rca4_2'
    elif name == 'torca':
        string = 'TORC'
        str1 = 'i_cs_torc_a1'
        str2 = 'i_cs_torc_a4'
    elif name == 'torcb':
        string = 'TORC'
        str1 = 'i_cs_torc_b1'
        str2 = 'i_cs_torc_b4'
    lists_of_atoms = []
    for bond in struct_2.bond:
        try:
            bond.property[str1]
            bond.property[str2]
        except KeyError as e:
            print('ERROR! NO {}: {}'.format(
                string, struct_2.property['s_m_title']))
            raise e
        if bond.property[str1]:
            atoms = [bond.property[str1],
                     bond.atom1.index,
                     bond.atom2.index,
                     bond.property[str2]]
            lists_of_atoms.append(atoms)
    print('{}:     {}'.format(string, lists_of_atoms))

    # Now update the RCA4 and TORC atom indices to match the structure post
    # merging and deletion.
    # Need this for later.
    num_atoms = len(struct_1.atom)
    # These are the new atom numbers for the common atoms in struct_2.
    new_match_2 = [x + num_atoms for x in match_2]
    # Contains new RCA4 commands.
    new_lists_of_atoms = []
    for atoms in lists_of_atoms:
        new_atoms = []
        for x in atoms:
            if x in match_2:
                # If the RCA4/TORC atom is one of the matching/duplicate/common
                # atoms, it's going to get deleted. This replaces the index of
                # that atom with the matching atom in the 1st structure.
                new_atoms.append(match_1[match_2.index(x)])
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
                new_atoms.append(new_index)
        new_lists_of_atoms.append(new_atoms)
    print('{} NEW: {}'.format(string, new_lists_of_atoms))

    print(' * UPDATING {}:'.format(string))
    # Now have to update the bonds RCA4 properties.
    for atoms in new_lists_of_atoms:
        bond = merge.getBond(atoms[1], atoms[2])
        print('   * BOND:     {:>4}    {:>4}/{:2} {:>4}/{:2} {:>4}'.format(
            bond.property[str1],
            bond.atom1.index,
            bond.atom1.atom_type_name,
            bond.atom2.index,
            bond.atom2.atom_type_name,
            bond.property[str2]))
        bond.property[str1] = atoms[0]
        bond.property[str2] = atoms[3]
        print('     * UPDATE: '
              '{:>4}/{:2} {:>4}/{:2} {:>4}/{:2} {:>4}/{:2}'.format(
            merge.atom[bond.property[str1]].index,
            merge.atom[bond.property[str1]].atom_type_name,
            bond.atom1.index,
            bond.atom1.atom_type_name,
            bond.atom2.index,
            bond.atom2.atom_type_name,
            merge.atom[bond.property[str2]].index,
            merge.atom[bond.property[str2]].atom_type_name))
    return merge


def mini(structures, frozen_atoms=None, fix_torsions=None):
    """
    Takes many structures, minimizes them and returns the minimized structures.
    It's faster to do multiple structures at once.

    Arguments
    ---------
    structure : list of Schrödinger structure

    Returns
    -------
    list of Schrödinger structure
    """
    import schrodinger.application.macromodel.utils as mmodutils
    import schrodinger.job.jobcontrol as jobcontrol
    from setup_com_from_mae import MyComUtil
    print(' - ATTEMPTING MINI')
    sch_writer = sch_struct.StructureWriter('TEMP.mae')
    sch_writer.extend(structures)
    sch_writer.close()
    # Setup the minimization.
    com_setup = MyComUtil()
    com_setup.my_mini(
        mae_file='TEMP.mae',
        com_file='TEMP.com',
        out_file='TEMP_OUT.mae',
        frozen_atoms=frozen_atoms,
        fix_torsions=fix_torsions)
    command = ['bmin', '-WAIT', 'TEMP']
    # Run the minimization.
    job = jobcontrol.launch_job(command)
    job.wait()
    # Read the minimized structures.
    sch_reader = sch_struct.StructureReader('TEMP_OUT.mae')
    new_structures = []
    for structure in sch_reader:
        new_structures.append(structure)
    sch_reader.close()
    if len(new_structures) > 0:
        print(' - MINI SUCCEEDED')
        structures = [new_structures[0]]
    else:
        print(' - MINI FAILED. CONTINUING W/O MINI')
    if DEBUG:
        raw_input('Press any button to continue.')
    # Remove temporary files.
    os.remove('TEMP.mae')
    os.remove('TEMP.com')
    os.remove('TEMP_OUT.mae')
    os.remove('TEMP.log')
    return structures

def mcmm(structures, frozen_atoms=None):
    """
    Takes many structures, and does a short MCMM search on them.

    Arguments
    ---------
    structure : list of Schrödinger structure

    Returns
    -------
    list of Schrödinger structure
    """
    import schrodinger.application.macromodel.utils as mmodutils
    import schrodinger.job.jobcontrol as jobcontrol
    from setup_com_from_mae import MyComUtil
    print(' - ATTEMPTING MCMM')
    sch_writer = sch_struct.StructureWriter('TEMP.mae')
    sch_writer.extend(structures)
    sch_writer.close()
    com_setup = MyComUtil()
    com_setup.my_mcmm(
        mae_file='TEMP.mae',
        com_file='TEMP.com',
        out_file='TEMP_OUT.mae',
        nsteps=50,
        frozen_atoms=frozen_atoms)
    command = ['bmin', '-WAIT', 'TEMP']
    job = jobcontrol.launch_job(command)
    job.wait()
    sch_reader = sch_struct.StructureReader('TEMP_OUT.mae')
    new_structures = []
    for structure in sch_reader:
        new_structures.append(structure)
    sch_reader.close()
    if len(new_structures) > 0:
        print(' - MCMM SUCCEEDED')
        structures = [new_structures[0]]
    else:
        print(' - MCMM FAILED. CONTINUING W/O MCMM')
    if DEBUG:
        raw_input('Press any button to continue.')
    # Remove temporary files.
    os.remove('TEMP.mae')
    os.remove('TEMP.com')
    os.remove('TEMP_OUT.mae')
    os.remove('TEMP.log')
    return structures

def make_unique_filename(path):
    """
    I'm proud of myself for this one.

    Arguments
    ---------
    path : string
           Path to some file.

    Returns
    -------
    string
    """
    # By default, append 3 digits to the end of filenames.
    len_digits = 3
    while True:
        # The path already exists.
        if os.path.isfile(path):
            name, ext = os.path.splitext(path)
            # Does the filename (without the extension) end in digits?
            m = re.search(r'\d+$', name)
            if m:
                # Digits at the end of the file.
                string = m.group()
                # Length of this string of digits.
                len_digits = len(string)
                # New number.
                number = int(string) + 1
                # Remove the existing digits.
                name = name[:-(len_digits + 1)]
                # Add on the new number with the same number of digits.
                name += '_{0:0{1}d}'.format(number, len_digits)
                path = name + ext
            else:
                name += '_{0:0{1}d}'.format(1, len_digits)
                path = name + ext
        # Hooray! It's a unique filename!
        else:
            break
    return path

def remove_uplicates(iterable):
    """
    Also checks for reversed duplicates.
    [a, b] == [b, a]

    http://stackoverflow.com/a/41166628
    """
    seen = set()
    for item in iterable:
        tup = tuple(item)
        if tup not in seen:
            seen.add(tup[::-1])
            seen.add(tup)
            yield item

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    structures = main(opts)
