#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates ND CRC job submission files from Schrödinger *.com files.

WARNING: YOU MUST UPDATE THE VARIABLES IN ALL CAPS AT THE START OF THIS PYTHON
FILE WITH YOUR PERSONAL SETTINGS!
"""
import argparse
import os
import glob
import time
import subprocess as sp
import sys

EMAIL = "youremail@gmail.com"
MAIL = "ae"
QUEUE = "long"
RETRY = "n"
SCHRODINGER_TEMP = "~/tmp"

MACROMODEL_JOB_SCRIPT = \
"""#!/bin/csh
#$ -M {}
#$ -m {}
#$ -q {}
#$ -r {}
#$ -N T{}_{}

module load schrodinger/2016u3
setenv SCHRODINGER_TEMP_PROJECT {}
setenv SCHRODINGER_TMPDIR {}
setenv SCHRODINGER_JOBDB2 {}

"""

def return_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '-f', '--filenames', type=str, nargs='+',
        help="Name(s) of *.com file(s) you'd like to generate *.sh ND CRC job "
        "submission files for.")
    parser.add_argument(
        '-x', '--execute', action='store_true', default=False,
        help='Does csh on the *.sh files created. Overrides the --submit '
        'option.')
    parser.add_argument(
        '-s', '--submit', action='store_true', default=False,
        help='Does qsub on the *.sh files created.')
    parser.add_argument(
        '-o', '--one', type=str, nargs='?', const='job.sh', metavar='filename',
        help='Write one *.sh file instead of multiple. If no filename is '
        'provided, the default is "job.sh".')
    return parser

def main(opts):
    if opts.one:
        with open(opts.one, 'w') as f:
            f.write(
                MACROMODEL_JOB_SCRIPT.format(
                    EMAIL,
                    MAIL,
                    QUEUE,
                    RETRY,
                    time.strftime("%y%m%d%H%M"),
                    opts.one,
                    SCHRODINGER_TEMP,
                    SCHRODINGER_TEMP,
                    SCHRODINGER_TEMP
                    )
                )
            for filename in opts.filenames:
                name, ext = os.path.splitext(filename)
                f.write('bmin -WAIT {}\n'.format(name))
        print('WROTE: {}'.format(opts.one))
        if opts.execute:
            print(' - Attempting execution from {} ...'.format(os.getcwd()))
            sp.call('csh {}'.format(name + '.sh'), shell=True)
        elif opts.submit:
            sp.call('qsub {}'.format(name + '.sh'), shell=True)
    else:
        for filename in opts.filenames:
            name, ext = os.path.splitext(filename)
            with open(name + ".sh", "w") as f:
                f.write(
                    MACROMODEL_JOB_SCRIPT.format(
                        EMAIL,
                        MAIL,
                        QUEUE,
                        RETRY,
                        time.strftime("%y%m%d%H%M"),
                        name,
                        SCHRODINGER_TEMP,
                        SCHRODINGER_TEMP,
                        SCHRODINGER_TEMP
                        )
                    )
                f.write('bmin -WAIT {}\n'.format(name))
            print('WROTE: {}'.format(name + ".sh"))
            if opts.execute:
                print(' - Attempting execution from {} ...'.format(os.getcwd()))
                sp.call('csh {}'.format(name + '.sh'), shell=True)
            elif opts.submit:
                sp.call('qsub {}'.format(name + ".sh"), shell=True)

if __name__ == "__main__":
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
