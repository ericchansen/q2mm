 #!/usr/bin/python
"""
Takes .mae structure files and generates .com files for conformational searches.

The .mae files must contain several properties on the atoms and bonds. These
properties can be manually entered and read from the .mae. They can also be
accessed using Schrodinger's structure module. The properties are named
following standard Schrodinger naming practices.

Atomic Properties
-----------------
b_cs_chig - True (1) if the atom is a chiral center.
b_cs_comp - True (1) if the atom should be used for comparisons to determine
            whether structures are duplicates.

Bond Properties
---------------
b_cs_tors   - True (1) if the bond should be rotated.
i_cs_rca4_1 - This and i_cs_rca4_2 are used together and  indicate  where the
              conformational search method should make ring breaks.

              In MacroModel, ring breaks are specified by providing the atom
              numbers for the 4 atoms in a torsion. The two middle atoms are
              described by the existing Maestro properties i_m_from and
              i_m_to. The two ending atoms are described using i_cs_rca4_1
              (which is the atom connected to i_m_from) and i_cs_rca4_2 (which
              is the atom connected to i_m_to).

              If both i_cs_rca4_1 and i_cs_rca4_2 are 0, then a ring break isn't
              made across this bond.
i_cs_rca4_2 - See i_cs_rca4_1.
"""
import argparse
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
        for count_comp, args in enumerate(grouper(4, indices_comp)):
            com_setup.setOpcdArgs(
                opcd='COMP',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        for count_chig, args in enumerate(grouper(4, indices_chig)):
            com_setup.setOpcdArgs(
                opcd='CHIG',
                arg1=args[0],
                arg2=args[1],
                arg3=args[2],
                arg4=args[3]
                )
        for count_tors, args in enumerate(indices_tors):
            com_setup.setOpcdArgs(
                opcd='TORS',
                arg1=args[0],
                arg2=args[1],
                arg6=180.
                )
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
        if 3**len(indices_tors) > 50000:
            com_setup.setOpcdArgs(opcd='MCMM', arg1=50000)
        else:
            com_setup.setOpcdArgs(opcd='MCMM', arg1=3**len(indices_tors))
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

def main_old():
    """
    Depreciated.

    Example of how I started working out this code.
    """
    com_setup = mmodutils.ComUtil()

    # This argument doesn't work. I can set it to use mm3.fld later on using the
    # operation code dictionary. Something like com_setup.FFLD[1] = 2.
    # com_setup = mmodutils.ComUtil('mm3.fld')

    # Maybe try this. Nope still doesn't work.
    # The docs say "Subclass of ComUtil with defaults that are closer to the
    # current Maestro version."
    # com_setup = mmodutils.ComUtilAppSci('mm3.fld')

    # Why did we ever use this?
    # com_setup.mmod = True
    # com_setup.MMOD.clear()
    # com_setup.setOpcdArgs(opcd='MMOD', arg2=1)

    com_setup.DEBG.clear()
    com_setup.setOpcdArgs(opcd='DEBG', arg1=55, arg2=179)
    com_setup.SEED.clear()
    com_setup.setOpcdArgs(opcd='SEED', arg1=40000)
    # Change FF.
    # com_setup.FFLD[1] = 2
    # com_setup.FFLD[2] = 1
    # com_setup.FFLD[5] = 1
    # Another way.
    com_setup.FFLD.clear()
    com_setup.setOpcdArgs(opcd='FFLD', arg1=2, arg2=1, arg5=1)
    com_setup.EXNB.clear()
    com_setup.BDCO.clear()
    com_setup.setOpcdArgs(opcd='BDCO', arg5=89.4427, arg6=99999)
    com_setup.READ.clear()
    com_setup.CRMS.clear()
    com_setup.setOpcdArgs(opcd='CRMS', arg6=0.5)
    com_setup.MCMM.clear()
    com_setup.setOpcdArgs(opcd='MCMM', arg1=10000)
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
    # Specific to the input file.
    # COMP # Comparison atoms
    # CHIG # Chirality
    # TORS # Torsions # Sets of 2
    # RCA4 # Ring closure atoms # Sets of 4

    # Example of how to get a formatted string of codes.
    # cats = com_setup.getOpcdArgs(opcd='AUTO')
    # print(cats)

    # Another way to do this. Just returns the string. Doesn't seem necessary for
    # us.
    # com_file = com_setup.mcmm(
    #     INPUT_STRUCTURE_FILE, com_file=COM_FILE, out_file=OUTPUT_STRUCTURE_FILE)

    # Turns out this isn't good enough. I want more control of the commands.
    # com_setup.mcmm(
    #     INPUT_STRUCTURE_FILE, com_file=COM_FILE, out_file=OUTPUT_STRUCTURE_FILE)
    # with open(COM_FILE, 'r') as f:
    #     print(f.read())

    com_args = [
        COM_FILE,
        INPUT_STRUCTURE_FILE,
        OUTPUT_STRUCTURE_FILE,
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
        'DEMX',
        'COMP',
        'MSYM',
        'CHIG',
        'AUOP',
        'TORS',
        'RCA4',
        'CONV',
        'MINI'
        ]

    com_setup.writeComFile(com_args)
    with open(COM_FILE, 'r') as f:
        print(f.read())

    # Returns ['bmin', '-INTERVAL', '5', COM_FILE].
    # cmd_args = com_setup.getLaunchCommand(COM_FILE)
    # job = jc.launch_job(cmd_args)
    # job.wait()
    # print('MCMM Job Status: {}'.format(job.Status))

    # output_structure_file = job.StructureOutputFile

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'input', type=str,
        help="Name of .mae file that you'd like to generate a conformational "
        "search .com file for. Must contain the properties as described in the "
        "__doc__ and description for proper functioning.")
    parser.add_argument(
        'com', type=str,
        help="Name for the .com file you'd like to generate.")
    parser.add_argument(
        'output', type=str,
        help="Name for the output .mae file generated by the conformational "
        "search.")
    return parser

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    com_setup = MyComUtil()
    com_setup.my_mcmm(
        mae_file=opts.input,
        com_file=opts.com,
        out_file=opts.output)
    
