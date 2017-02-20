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
b_cs_tors   - True (1) if the bond should be rotated.
i_cs_rca4_1 - This and `i_cs_rca4_2` are used together and indicate where the
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
i_cs_rca4_2 - See `i_cs_rca4_1`.
i_cs_torc_1 - This and `i_cs_torc_2` are used together to indicate where the
              conformational search method should enforce torsion checks.

              Functions similar to `i_cs_rca4_1` and `i_cs_rca4_2`.
i_cs_torc_2 - See `i_cs_torc_1`.
"""
import argparse
import sys
from itertools import izip_longest

import schrodinger.application.macromodel.utils as mmodutils
from schrodinger import structure as sch_struct

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
    return izip_longest(fillvalue=fillvalue, *args)

class MyComUtil(mmodutils.ComUtil):
    def my_mcmm(self, mae_file=None, com_file=None, out_file=None):
        """
        Modified version of the the Schrödinger mcmm.

        Uses custom attributes inside of the mae_file to determine certain
        settings.
        """
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
        reader = sch_struct.StructureReader(mae_file)
        for structure in reader:
            for atom in structure.atom:
                try:
                    atom.property['b_cs_comp']
                    atom.property['b_cs_chig']
                except KeyError as e:
                    print('ERROR! MISSING ATOM PROPERTIES: {}'.format(
                        structure.property['s_m_title']))
                    raise e
                if atom.property['b_cs_comp']:
                    indices_comp.append(atom.index)
                if atom.property['b_cs_chig']:
                    indices_chig.append(atom.index)
            for bond in structure.bond:
                if bond.property['b_cs_tors']:
                    indices_tors.append((bond.atom1.index, bond.atom2.index))
                if bond.property['i_cs_rca4_1']:
                    indices_rca4.append((
                            bond.property['i_cs_rca4_1'],
                            bond.atom1.index,
                            bond.atom2.index,
                            bond.property['i_cs_rca4_2']
                            ))
                if bond.property['i_cs_torc_1']:
                    indices_torc.append((
                            bond.property['i_cs_torc_1'],
                            bond.atom1.index,
                            bond.atom2.index,
                            bond.property['i_cs_torc_2']
                            ))
        reader.close()
        count_comp = 0
        for count_comp, args in enumerate(grouper(4, indices_comp)):
            self.setOpcdArgs(
                opcd='COMP',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_chig = 0
        for count_chig, args in enumerate(grouper(4, indices_chig)):
            self.setOpcdArgs(
                opcd='CHIG',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_tors = 0
        for count_tors, args in enumerate(indices_tors):
            self.setOpcdArgs(
                opcd='TORS',
                arg1=args[0],
                arg2=args[1],
                arg6=180.
                )
        count_rca4 = 0
        for count_rca4, args in enumerate(indices_rca4):
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
        for count_rca4, args in enumerate(indices_torc):
            self.setOpcdArgs(
                opcd='TORC',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3],
                arg5=90.,
                arg6=180.
                )

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
        # 3**N where N = number of bonds rotated
        # Maxes out at 50,000.
        nsteps = 3**len(indices_tors)
        # nsteps = 50
        if nsteps > 50000:
            nsteps = 50000
        self.setOpcdArgs(opcd='MCMM', arg1=nsteps)
        self.NANT.clear()
        self.MCNV.clear()
        self.setOpcdArgs(opcd='MCNV', arg1=1, arg2=5)
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
            'MCNV',
            'MCSS',
            'MCOP',
            'DEMX'
            ]
        com_args.extend(['COMP'] * (count_comp + 1))
        com_args.append('MSYM')
        com_args.extend(['CHIG'] * (count_chig + 1))
        # Used to have AUOP here.
        com_args.extend(['TORS'] * (count_tors + 1))
        com_args.extend(['TORC'] * (count_torc + 1))
        com_args.extend(['RCA4'] * (count_rca4 + 1))
        com_args.extend([
                'CONV',
                'MINI'
                ])
        return self.writeComFile(com_args)
    def my_mini(self, mae_file=None, com_file=None, out_file=None):
        """
        Modified version of the the Schrödinger mini.
        """
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
            'READ',
            'CONV',
            'MINI',
            'END']
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
    return parser

def main(opts):
    """
    Main for setup_com_from_mae. See module __doc__.
    """
    com_setup = MyComUtil()
    com_setup.my_mcmm(
        mae_file=opts.input,
        com_file=opts.com,
        out_file=opts.output)

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
