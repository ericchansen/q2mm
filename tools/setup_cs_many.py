 #!/usr/bin/python
"""
Automates conformational searching.
"""
import argparse
import os
import sys
from itertools import izip_longest

import schrodinger.application.macromodel.utils as mmodutils
import schrodinger.job.jobcontrol as jc
from schrodinger import structure as sch_struct

def grouper(n, iterable, fillvalue=0.):
    """
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

class MyComUtil(mmodutils.ComUtil):
    def my_mcmm(self, mae_file=None, com_file=None, out_file=None):
        """
        Modified version of the the mcmm.

        Uses custom attributes inside of the mae_file to determine certain
        settings.
        """
        # Setup the COMP, CHIG, TORS and RCA4 commands.
        com_setup.COMP.clear()
        # com_setup.CHIG.clear()
        # com_setup.TORS.clear()
        # com_setup.RCA4.clear()
        indices_comp = []
        indices_chig = []
        indices_tors = []
        indices_rca4 = []
        reader = sch_struct.StructureReader(mae_file)
        for structure in reader:
            for atom in structure.atom:
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
        reader.close()
        count_comp = 0
        for count_comp, args in enumerate(grouper(4, indices_comp)):
            com_setup.setOpcdArgs(
                opcd='COMP',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_chig = 0
        for count_chig, args in enumerate(grouper(4, indices_chig)):
            com_setup.setOpcdArgs(
                opcd='CHIG',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        count_tors = 0
        for count_tors, args in enumerate(indices_tors):
            com_setup.setOpcdArgs(
                opcd='TORS',
                arg1=args[0],
                arg2=args[1],
                arg6=180.
                )
        count_rca4 = 0
        for count_rca4, args in enumerate(indices_rca4):
            com_setup.setOpcdArgs(
                opcd='RCA4',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3],
                arg5=0.5,
                arg6=2.5
                )

        com_setup.DEBG.clear()
        com_setup.setOpcdArgs(opcd='DEBG', arg1=55, arg2=179)
        com_setup.SEED.clear()
        com_setup.setOpcdArgs(opcd='SEED', arg1=40000)
        com_setup.FFLD.clear()
        com_setup.setOpcdArgs(opcd='FFLD', arg1=2, arg2=1, arg5=1)
        com_setup.EXNB.clear()
        com_setup.BDCO.clear()
        com_setup.setOpcdArgs(opcd='BDCO', arg5=89.4427, arg6=99999)
        com_setup.READ.clear()
        com_setup.CRMS.clear()
        com_setup.setOpcdArgs(opcd='CRMS', arg6=0.5)
        com_setup.MCMM.clear()
        # 3**N where N = number of bonds rotated
        # Maxes out at 50,000.
        nsteps = 3**len(indices_tors)
        if nsteps > 50000:
            nsteps = 50000
        com_setup.setOpcdArgs(opcd='MCMM', arg1=nsteps)
        com_setup.NANT.clear()
        com_setup.MCNV.clear()
        com_setup.setOpcdArgs(opcd='MCNV', arg1=1, arg2=5)
        com_setup.MCSS.clear()
        com_setup.setOpcdArgs(opcd='MCSS', arg1=2, arg5=50.0)
        com_setup.MCOP.clear()
        com_setup.setOpcdArgs(opcd='MCOP', arg1=1, arg4=0.5)
        com_setup.DEMX.clear()
        com_setup.setOpcdArgs(opcd='DEMX', arg2=1000, arg5=50, arg6=100)
        com_setup.MSYM.clear()
        com_setup.AUOP.clear()
        com_setup.CONV.clear()
        com_setup.setOpcdArgs('CONV', arg1=2, arg5=0.05)
        com_setup.MINI.clear()
        com_setup.setOpcdArgs('MINI', arg1=1, arg3=2500)
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
        com_args.extend(['RCA4'] * (count_rca4 + 1))
        com_args.extend([
                'CONV',
                'MINI'
                ])
        return self.writeComFile(com_args)

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, nargs='+')
    return parser

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])

    for file_input in opts.input:
        com_setup = MyComUtil()

        name, ext = os.path.splitext(file_input)
        file_com = name + '_cs.com'
        file_mae = name + '_cs.mae'

        com_setup.my_mcmm(
            mae_file=file_input,
            com_file=file_com,
            out_file=file_mae
            )
    
