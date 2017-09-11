9#!/usr/bin/env python

import argparse
import sys
import os
import subprocess as sp

DEFAULT_SUB_FILE = '''#!/bin/csh
#$ -M arosale4@nd.edu
#$ -m ae
#$ -N {}
#$ -q {}
#$ -r n
{}

module load schrodinger/2015u3
module load gaussian/09D01
module load tinker
setenv SCHRODINGER_TEMP_PROJECT "/scratch365/arosale4/schrodinger/.schrodtmp"
setenv SCHRODINGER_TMPDIR "/scratch365/arosale4/schrodinger/.schrodtmp"
setenv SCHRODINGER_JOBDB2 "/scratch365/arosale4/schrodinger/.schrodtmp"
setenv GAUSS_SCRDIR "/scratch365/arosale4/gaussian"

{}'''


def CRC_qsub(job_name,QUEUE,CPU,COMMAND):
    ## Writes the submission file with all the appropriate options: job name,
    ## queue, processors, and the job command.
    submission_file = open(job_name + '.sh', 'w')
    submission_file.write(
        DEFAULT_SUB_FILE.format(job_name,QUEUE,CPU,COMMAND.format(
            job_name + '.com')))
    submission_file.close()

def queue(opts):
    ## Queue option of the crc. I think I can only use long and debug, which
    ## long is the default.
    if opts.queue:
        QUEUE = opts.queue
    else:
        QUEUE = 'long'
    return QUEUE

def processors(opts):
    ## Sets the number of processors to request from the CRC. Default is to use
    ## 8 processors. When there is no argument to follow ("-pe") then this 
    ## section is removed to allow for only one processor. An additional 
    ## argument will just write that argument, e.g. "-pe -pe smp 16" would add
    ## #$ -pe smp 16 to the submission script.
    if opts.processors == 'default':
        CPU = '#$ -pe smp 8'
    elif opts.processors == 'none':
        CPU = ' '
    else:
        CPU = '#$ ' + opts.processors
    return CPU

def command(opts):
    ## Sets the actuall command to accomplish. By default it will do a gaussian
    ## job. Example of an alternative is "--command bmin -WAIT conf_search"
    if opts.command:
        COMMAND = opts.command
    else:
        COMMAND = 'g09 {}'
    return COMMAND

def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    QUEUE = queue(opts)
    CPU = processors(opts)
    COMMAND = command(opts)
    for filename in opts.filename:
        run_file = os.path.splitext(filename)[0]
        CRC_qsub(run_file,QUEUE,CPU,COMMAND)
        sp.call('qsub {}.sh'.format(run_file), shell=True)
#        print('This is where you would run the following command')    
#        print('>>>>> qsub {}.sh'.format(run_file))    

def return_parser():
    parser = argparse.ArgumentParser(
        description='To fill out later')
    parser.add_argument(
        'filename', type=str, nargs='+', help='Filename without extension')
    parser.add_argument(
        '-q','--queue', type=str, help='Long or short(its not short anymore)')
#    parser.add_argument(
#        '-np', '--no_MP', type=str, nargs='?', const=' ', default='none',
#        help='No multiple processing')
    parser.add_argument(
        '-pe','--processors', type=str, nargs='?', const='none',
        default='default', help='No option string = default smp 8; \n'
        'Option string but no argument = no multiple processing; and \n'
        'Option string with argument = "#$" + argument')
    parser.add_argument(
        '-c','--command', type=str, help='Command that are being ran')
    return parser
        

if __name__ == '__main__':
    main(sys.argv[1:])


##################################
