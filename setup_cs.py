 #!/usr/bin/python
"""
Automates conformational searching.
"""
import schrodinger.application.macromodel.utils as mmodutils
import schrodinger.job.jobcontrol as jc

INPUT_STRUCTURE_FILE = 'c1r2s_004.mae'
OUTPUT_STRUCTURE_FILE = 'c1r2s_004.cs.mae'
COM_FILE = 'c1r2s_004.cs.com'


# INPUT_STRUCTURE_FILE = 'A1-R.mae'
# OUTPUT_STRUCTURE_FILE = 'A1-R.cs.mae'
# COM_FILE = 'A1-R.cs.com'

class MyComUtil(mmodutils.ComUtil):
    def my_mcmm(self, mae_file=None, com_file=None, out_file=None):
        """
        Modified version of the the mcmm.

        Uses custom attributes inside of the mae_file to determine certain
        settings.
        """
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

    # Returns ['bmin', '-INTERVAL', '5', COM_FILE] (of course COM_FILE is the string
    # that has been set).
    # cmd_args = com_setup.getLaunchCommand(COM_FILE)
    # job = jc.launch_job(cmd_args)
    # job.wait()
    # print('MCMM Job Status: {}'.format(job.Status))

    # output_structure_file = job.StructureOutputFile

if __name__ == '__main__':
    com_setup = MyComUtil()
    com_setup.my_mcmm(
        mae_file=INPUT_STRUCTURE_FILE,
        com_file=COM_FILE,
        out_file=OUTPUT_STRUCTURE_FILE)
    
