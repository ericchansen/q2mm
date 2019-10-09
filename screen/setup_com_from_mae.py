#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takes *.mae structure files and generates *.com files for
conformational searches.

The *.mae files must contain several properties on the atoms and bonds. These
properties can be manually entered into the *.mae. They can also be
accessed using Schrödinger's structure module. Lastly, the script
`setup_mae_from_com.py` can assist in setting up these properties.

The properties are named following standard Schrödinger naming practices.

Atomic Properties
-----------------
b_cs_chig - True (1) if the atom is a chiral center.
b_cs_comp - True (1) if the atom should be used for comparisons to determine
            whether structures are duplicates.

Bond Properties
---------------
b_cs_tors    - True (1) if the bond should be rotated.
i_cs_rca4_1  - This and `i_cs_rca4_2` are used together and indicate where the
               conformational search method should make ring breaks.

               In MacroModel, ring breaks are specified by providing the atom
               numbers for the 4 atoms in a torsion. The two middle atoms are
               described by the existing Maestro properties `i_m_from` and
               `i_m_to`. The two ending atoms are described using `i_cs_rca4_1`
               (which is the atom connected to `i_m_from`) and `i_cs_rca4_2`
               (which is the atom connected to `i_m_to`).

               If both `i_cs_rca4_1` and `i_cs_rca4_2` are 0, then a ring break
               isn't made across this bond.

               IMPORTANT NOTE TO AVOID CONFUSION:
               By default, Schrödinger writes *.mae files with bonds listed in
               both directions, i.e. 1-2 and 2-1. However, the bond properties
               for 1-2 and 2-1 MUST BE THE SAME. Therefore, these RCA4
               properties are setup to only read properly for the bond with
               the lowest atom number first. In this case, 1-2.
i_cs_rca4_2  - See `i_cs_rca4_1`.
i_cs_torc_a1 - This and `i_cs_torc_xa` are used together to indicate where the
               conformational search method should enforce torsion checks.

               Functions similar to `i_cs_rca4_1` and `i_cs_rca4_2`.
i_cs_torc_a4 - See `i_cs_torc_a2`.
r_cs_torc_a5 - Absolute value of the minimum torsional value allowed.
r_cs_torc_a6 - Absolute value of the maximum torsional value allowed.
i_cs_torc_b1 - Same as `i_cs_torc_a1`, but allows an additional TORC per bond.
i_cs_torc_b4 - " " "
i_cs_torc_b5 - " " "
i_cs_torc_b6 - " " "

To enforce a cis bond:
 * r_cs_torc_x5 = 0
 * r_cs_torc_x6 = 90
To enforce a trans bond:
 * r_cs_torc_x5 = 90
 * r_cs_torc_x6 = 180

Watch out because MacroModel sometimes assigns these enforcing floats wrong.
"""
from __future__ import absolute_import
import argparse
import sys
from itertools import zip_longest

import schrodinger.application.macromodel.utils as mmodutils
from schrodinger import structure as sch_struct

ALL_CS_ATOM_PROPERTIES = ['b_cs_chig', 'b_cs_comp']
ALL_CS_BOND_PROPERTIES = ['b_cs_tors',
                          'i_cs_rca4_1', 'i_cs_rca4_2',
                          'i_cs_torc_a1', 'i_cs_torc_a4',
                          'r_cs_torc_a5', 'r_cs_torc_a6',
                          'i_cs_torc_b1', 'i_cs_torc_b4',
                          'r_cs_torc_b5', 'r_cs_torc_b6']

def grouper(n, iterable, fillvalue=0.):
    """
    Returns list of lists from a single list.

    Usage
    -----
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx

    Arguments
    ---------
    n : integer
        Length of sub list.
    iterable : iterable
    fillvalue : anything
                Fills up last sub list if iterable is not divisible by n
                without a remainder.

    """
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)

class MyComUtil(mmodutils.ComUtil):
    def my_mcmm(
            self,
            mae_file=None,
            com_file=None,
            out_file=None,
            nsteps=None,
            frozen_atoms=None):
        """
        Modified version of the the Schrödinger mcmm.

        Uses custom attributes inside of the mae_file to determine certain
        settings.
        """
        if not frozen_atoms:
            frozen_atoms = []
        self.FXAT.clear()
        for frozen_atom in frozen_atoms:
            self.setOpcdArgs(
                opcd='FXAT',
                arg1=frozen_atom,
                arg2=0,
                arg3=0,
                arg4=0,
                arg5=-1
            )
        # Setup the COMP, CHIG, TORS and RCA4 commands.
        self.COMP.clear()
        # self.CHIG.clear()
        # self.TORS.clear()
        # self.RCA4.clear()
        indices_comp = []
        indices_chig = []
        indices_tors = []
        indices_rca4 = []
        indices_torc = []
        # Only works with the 1st structure.
        reader = sch_struct.StructureReader(mae_file)
        structure = next(reader)
        reader.close()
        print('-' * 80)
        print('READING: {}'.format(structure.property['s_m_title']))
        # Extra step due to TORS error using automated script
        all_bond = []
        for bond in structure.bond:
            all_bond.append([bond.atom1.index,bond.atom2.index])
        for atom in structure.atom:
            for prop in ALL_CS_ATOM_PROPERTIES:
                try:
                    atom.property[prop]
                except KeyError as e:
                    print('STRUCTURE: {}'.format(
                        structure.property['s_m_title']))
                    print(' - ATOM: {}'.format(atom))
                    print('   - MISSING: {} (setting to 0)'.format(prop))
                    atom.property[prop] = 0
            if atom.property['b_cs_comp']:
                indices_comp.append(atom.index)
            if atom.property['b_cs_chig']:
                indices_chig.append(atom.index)
        for bond in structure.bond:
            for prop in ALL_CS_BOND_PROPERTIES:
                try:
                    bond.property[prop]
                except KeyError as e:
                    print('STRUCTURE: {}'.format(
                        structure.property['s_m_title']))
                    print(' - BOND: {}'.format(bond))
                    print('   - MISSING: {} (setting to 0)'.format(prop))
                    bond.property[prop] = 0
            if bond.property['b_cs_tors']:
                thing = [bond.atom1.index, bond.atom2.index]
                if set(thing).issubset(frozen_atoms):
                     print('SKIPPING TORS: {} {}'.format(
                        bond.atom1.index,
                        bond.atom2.index))
                else:
                    indices_tors.append((bond.atom1.index, bond.atom2.index))
            if bond.property['i_cs_rca4_1']:
                thing = [bond.property['i_cs_rca4_1'],
                         bond.atom1.index,
                         bond.atom2.index,
                         bond.property['i_cs_rca4_2']]
                if set(thing).issubset(frozen_atoms):
                    print('SKIPPING RCA4: {} {} {} {}'.format(
                        bond.property['i_cs_rca4_1'],
                        bond.atom1.index,
                        bond.atom2.index,
                        bond.property['i_cs_rca4_2']))
                else:
                    t_bond = [bond.property['i_cs_rca4_1'],bond.atom1.index]
                    if t_bond in all_bond or t_bond[::-1] in all_bond:
                        indices_rca4.append((
                        bond.property['i_cs_rca4_1'],
                        bond.atom1.index,
                        bond.atom2.index,
                        bond.property['i_cs_rca4_2']
                        ))
            # FIXED FOR WRONG ORDER OF RCA4
            if bond.property['i_cs_rca4_2']:
                thing = [bond.property['i_cs_rca4_2'],
                         bond.atom1.index,
                         bond.atom2.index,
                         bond.property['i_cs_rca4_1']]
                if set(thing).issubset(frozen_atoms):
                    print('SKIPPING RCA4: {} {} {} {}'.format(
                        bond.property['i_cs_rca4_2'],
                        bond.atom1.index,
                        bond.atom2.index,
                        bond.property['i_cs_rca4_1']))
                else:
                    t_bond = [bond.property['i_cs_rca4_2'], bond.atom1.index]
                    if t_bond in all_bond or t_bond[::-1] in all_bond:
                        indices_rca4.append((
                            bond.property['i_cs_rca4_2'],
                            bond.atom1.index,
                            bond.atom2.index,
                            bond.property['i_cs_rca4_1']
                        ))
            if bond.property['i_cs_torc_a1']:
                indices_torc.append((
                    bond.property['i_cs_torc_a1'],
                    bond.atom1.index,
                    bond.atom2.index,
                    bond.property['i_cs_torc_a4'],
                    bond.property['r_cs_torc_a5'],
                    bond.property['r_cs_torc_a6']
                ))
            if bond.property['i_cs_torc_b1']:
                indices_torc.append((
                    bond.property['i_cs_torc_b1'],
                    bond.atom1.index,
                    bond.atom2.index,
                    bond.property['i_cs_torc_b4'],
                    bond.property['r_cs_torc_b5'],
                    bond.property['r_cs_torc_b6']
                ))
        print('COMP: {}'.format(indices_comp))
        print('CHIG: {}'.format(indices_chig))
        print('TORS: {}'.format(indices_tors))
        print('RCA4: {}'.format(indices_rca4))
        print('TORC: {}'.format(indices_torc))
        count_comp = 0
        for count_comp, args in enumerate(grouper(4, indices_comp), 1):
            self.setOpcdArgs(
                opcd='COMP',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_chig = 0
        for count_chig, args in enumerate(grouper(4, indices_chig), 1):
            self.setOpcdArgs(
                opcd='CHIG',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_tors = 0
        for count_tors, args in enumerate(indices_tors, 1):
            self.setOpcdArgs(
                opcd='TORS',
                arg1=args[0],
                arg2=args[1],
                arg6=180.
                )
        count_rca4 = 0
        for count_rca4, args in enumerate(indices_rca4, 1):
            self.setOpcdArgs(
                opcd='RCA4',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3],
                arg5=0.5,
                arg6=2.5
                )
        count_torc = 0
        for count_torc, args in enumerate(indices_torc, 1):
            self.setOpcdArgs(
                opcd='TORC',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3],
                arg5=args[4],
                arg6=args[5]
                )
        print('NUM. COMP: {}'.format(count_comp))
        print('NUM. CHIG: {}'.format(count_chig))
        print('NUM. TORS: {}'.format(count_tors))
        print('NUM. RCA4: {}'.format(count_rca4))
        print('NUM. TORC: {}'.format(count_torc))

        self.DEBG.clear()
        self.setOpcdArgs(opcd='DEBG', arg1=55, arg2=179)
        self.SEED.clear()
        self.setOpcdArgs(opcd='SEED', arg1=40000)
        self.FFLD.clear()
        self.setOpcdArgs(opcd='FFLD', arg1=2, arg2=1, arg5=1)
        self.EXNB.clear()
        self.BDCO.clear()
        self.setOpcdArgs(opcd='BDCO', arg5=89.4427, arg6=99999)
        self.READ.clear()
        self.CRMS.clear()
        self.setOpcdArgs(opcd='CRMS', arg6=0.5)
        self.MCMM.clear()
        if not nsteps:
            # 3**N where N = number of bonds rotated
            # Maxes out at 10,000.
            nsteps = 3**len(indices_tors)
            if nsteps > 10000:
                nsteps = 10000
        # Good for testing.
        # nsteps = 50
        self.setOpcdArgs(opcd='MCMM', arg1=nsteps)
        self.NANT.clear()
        self.MCNV.clear()
        # What if we just leave this out? Sets range for allowed DOF to change.
        # Here, it says change 1 - 5 DOF. This seems low. Default is 1 - N,
        # N is the number of variable dihedral angles. At any rate, these
        # initial values are overridden by the "adaptive mechanism" as the
        # search progresses. Good luck finding any explanation of what the
        # "adaptive mechanism" is from any of the Schrodinger documentation.
        # The "adaptive mechanism" can be turned off with DEBG 103.
        # self.setOpcdArgs(opcd='MCNV', arg1=1, arg2=5)
        self.MCSS.clear()
        self.setOpcdArgs(opcd='MCSS', arg1=2, arg5=50.0)
        self.MCOP.clear()
        self.setOpcdArgs(opcd='MCOP', arg1=1, arg4=0.5)
        self.DEMX.clear()
        self.setOpcdArgs(opcd='DEMX', arg2=1000, arg5=50, arg6=100)
        self.MSYM.clear()
        self.AUOP.clear()
        self.CONV.clear()
        self.setOpcdArgs('CONV', arg1=2, arg5=0.05)
        self.MINI.clear()
        self.setOpcdArgs('MINI', arg1=1, arg3=2500)
        # Honestly, not sure why K Shawn Watts initializes this as an empty
        # list, but I suppose I will do it too just in case.
        com_args = []
        com_args = [
            com_file,
            mae_file,
            out_file,
            'MMOD',
            'DEBG',
            'SEED',
            'FFLD',
            'EXNB',
            'BDCO',
            'READ',
            'CRMS',
            'MCMM',
            'NANT',
            # 'MCNV',
            'MCSS',
            'MCOP',
            'DEMX'
            ]
        com_args.extend(['COMP'] * (count_comp))
        com_args.append('MSYM')
        com_args.extend(['FXAT'] * len(frozen_atoms))
        com_args.extend(['CHIG'] * (count_chig))
        # Used to have AUOP here.
        com_args.extend(['TORS'] * (count_tors))
        com_args.extend(['TORC'] * (count_torc))
        com_args.extend(['RCA4'] * (count_rca4))
        com_args.extend([
                'CONV',
                'MINI'
                ])
        print('WRITING: {}'.format(com_file))
        return self.writeComFile(com_args)
    def my_conf_elim(
        self,
        mae_file=None,
        com_file=None,
        out_file=None):
        """
        Modified version of the the Schrödinger redundant conformer elimination.
        """
        self.DEBG.clear()
        self.setOpcdArgs(opcd='DEBG', arg1=55, arg2=179)
        self.SEED.clear()
        self.setOpcdArgs(opcd='SEED', arg1=40000)
        self.FFLD.clear()
        self.setOpcdArgs(opcd='FFLD', arg1=2, arg2=1, arg5=1)
        self.READ.clear()
        self.MINI.clear()
        self.COMP.clear()
        self.setOpcdArgs(opcd='COMP', arg7=2)
        # 9 = Truncated Newton (TNCG)
        self.setOpcdArgs('MINI', arg1=9, arg3=2500)
        com_args = [
            com_file,
            mae_file,
            out_file,
            'MMOD',
            'DEBG',
            'SEED',
            'FFLD',
            'BGIN',
            'READ',
            'COMP',
            'MINI',
            'END']
        print('WRITING: {}'.format(com_file))
        return self.writeComFile(com_args)
    def my_mini(
        self,
        mae_file=None,
        com_file=None,
        out_file=None,
        frozen_atoms=None,
        fix_torsions=None):
        """
        Modified version of the the Schrödinger mini.
        """
        print('FROZEN ATOMS: {}'.format(frozen_atoms))
        print('FIXED TORSIONS: {}'.format(fix_torsions))
        if not frozen_atoms:
            frozen_atoms = []
        if not fix_torsions:
            fix_torsions = []
        self.FXAT.clear()
        for frozen_atom in frozen_atoms:
            self.setOpcdArgs(
                opcd='FXAT',
                arg1=frozen_atom,
                arg2=0,
                arg3=0,
                arg4=0,
                arg5=-1
                )
        self.FXTA.clear()
        for fix_torsion in fix_torsions:
            # arg5 is the force constant in kJ/mol.
            # arg6 > 360 means to keep the current torsion value.
            self.setOpcdArgs(
                opcd='FXTA',
                arg1=fix_torsion[0],
                arg2=fix_torsion[1],
                arg3=fix_torsion[2],
                arg4=fix_torsion[3],
                arg5=4000,
                arg6=fix_torsion[4])
        self.DEBG.clear()
        self.setOpcdArgs(opcd='DEBG', arg1=55, arg2=179)
        self.SEED.clear()
        self.setOpcdArgs(opcd='SEED', arg1=40000)
        self.FFLD.clear()
        self.setOpcdArgs(opcd='FFLD', arg1=2, arg2=1, arg5=1)
        self.EXNB.clear()
        self.BDCO.clear()
        self.setOpcdArgs(opcd='BDCO', arg5=89.4427, arg6=99999)
        self.READ.clear()
        self.CONV.clear()
        self.setOpcdArgs('CONV', arg1=2, arg5=0.05)
        self.MINI.clear()
        self.setOpcdArgs('MINI', arg1=1, arg3=2500)
        # Honestly, not sure why K Shawn Watts initializes this as an empty
        # list, but I suppose I will do it too just in case.
        com_args = []
        com_args = [
            com_file,
            mae_file,
            out_file,
            'MMOD',
            'DEBG',
            'SEED',
            'FFLD',
            'EXNB',
            'BDCO',
            'CRMS',
            'BGIN',
            'READ']
        com_args.extend(['FXAT'] * len(frozen_atoms))
        com_args.extend(['FXTA'] * len(fix_torsions))
        com_args.extend([
            'CONV',
            'MINI',
            'END'])
        print('WRITING: {}'.format(com_file))
        return self.writeComFile(com_args)

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input', type=str,
        help="Name of *.mae file that you'd like to generate a conformational "
        "search *.com file for. Must contain the properties as described in "
        "the __doc__ and description for proper functioning.")
    parser.add_argument(
        'com', type=str,
        help="Name for the *.com file you'd like to generate.")
    parser.add_argument(
        'output', type=str,
        help="Name for the output *.mae file generated by the conformational "
        "search.")
    parser.add_argument(
        '-j', '--jobtype', type=str, default='cs',
        choices=['cs', 'mini', 're'],
        help='Job type. Choices include "cs", "mini" and "re". Default '
        'is "cs".')
    parser.add_argument(
        '-n', '--nsteps', type=int, default=15000,
        help='Number of conformational search steps to take. Default is '
        '3**N where N is the number of rotating bonds. If this exceeds '
        '10,000, then 10,000 steps are taken as default.')
    return parser

def main(opts):
    """
    Main for setup_com_from_mae. See module __doc__.
    """
    com_setup = MyComUtil()
    if opts.jobtype == "cs":
        com_setup.my_mcmm(
            mae_file=opts.input,
            com_file=opts.com,
            out_file=opts.output,
            nsteps=opts.nsteps)
    elif opts.jobtype == "mini":
        com_setup.my_mini(
            mae_file=opts.input,
            com_file=opts.com,
            out_file=opts.output)
    elif opts.jobtype == "re":
        com_setup.my_conf_elim(
            mae_file=opts.input,
            com_file=opts.com,
            out_file=opts.output)

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
