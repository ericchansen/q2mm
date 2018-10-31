#!/usr/bin/env python
import argparse
import glob
import itertools
import logging
import logging.config
import numpy as np
import os
import random
import sys

import calculate
import compare
import constants as co
import datatypes
import gradient
import opt
import parameters
import simplex

logger = logging.getLogger(__name__)

class Loop(object):
    def __init__(self):
        self.convergence = 0.01
        self.cycle_num = 0
        self.direc = '.'
        self.ff = None
        self.ff_lines = None
        self.args_ff = None
        self.args_ref = None
        self.loop_lines = None
        self.ref_data = None
    def opt_loop(self):
        """
        Iterator for cycling through optimization methods.

        Will continue to run the loop optimization methods until the convergence
        criterion has been met.

        Updates the user with logs on the optimization score changes. Backs up
        the FF after each loop cycle.
        """
        change = None
        last_score = None
        # This additional check ensures that the code won't crash if the user
        # forgets to add a COMP command in the loop input file.
        if self.ff.score is None:
            logger.warning(
                '  -- No existing FF score! Please ensure use of COMP in the '
                'input file! Calculating FF score automatically to compensate.')
            self.ff.score = compare.compare_data(
                self.ref_data, self.ff.data)
        while last_score is None \
                or change is None \
                or change > self.convergence:
            self.cycle_num += 1
            last_score = self.ff.score
            self.ff = self.run_loop_input(
                self.loop_lines, score=self.ff.score)
            logger.log(1, '>>> last_score: {}'.format(last_score))
            logger.log(1, '>>> self.ff.score: {}'.format(self.ff.score))
            change = (last_score - self.ff.score) / last_score
            pretty_loop_summary(
                self.cycle_num, self.ff.score, change)
            # MM3* specific. Will have to be changed soon to allow for expansion
            # into other FF software packages.
            mm3_files = glob.glob(os.path.join(self.direc, 'mm3_???.fld'))
            if mm3_files:
                mm3_files.sort()
                most_recent_mm3_file = mm3_files[-1]
                most_recent_mm3_file = most_recent_mm3_file.split('/')[-1]
                most_recent_num = most_recent_mm3_file[4:7]
                num = int(most_recent_num) + 1
                mm3_file = 'mm3_{:03d}.fld'.format(num)
            else:
                mm3_file = 'mm3_001.fld'
            mm3_file = os.path.join(self.direc, mm3_file)
            self.ff.export_ff(path=mm3_file)
            logger.log(20, '  -- Wrote best FF to {}'.format(mm3_file))
        return self.ff
    def run_loop_input(self, lines, score=None):
        lines_iterator = iter(lines)
        while True:
            try:
                if (sys.version_info > (3, 0)):
                    line = next(lines_iterator)
                else:
                    line = lines_iterator.next()
            except StopIteration:
                return self.ff
            cols = line.split()
            if cols[0] == 'DIR':
                self.direc = cols[1]
            if cols[0] == 'FFLD':
                # Import FF data.
                if cols[1] == 'read':
                    self.ff = datatypes.MM3(os.path.join(self.direc, cols[2]))
                    self.ff.import_ff()
                    self.ff.method = 'READ'
                    with open(os.path.join(self.direc, cols[2]), 'r') as f:
                        self.ff.lines = f.readlines()
                # Export FF data.
                if cols[1] == 'write':
                    self.ff.export_ff(os.path.join(self.direc, cols[2]))
            # Trim parameters.
            if cols[0] == 'PARM':
                logger.log(20, '~~ SELECTING PARAMETERS ~~'.rjust(79, '~'))
                self.ff.params = parameters.trim_params_by_file(
                    self.ff.params, os.path.join(self.direc, cols[1]))
            if cols[0] == 'LOOP':
                # Read lines that will be looped over.
                inner_loop_lines = []
                if (sys.version_info > (3, 0)):
                    line = next(lines_iterator)
                else:
                    line = lines_iterator.next()
                while line.split()[0] != 'END':
                    inner_loop_lines.append(line)
                    if (sys.version_info > (3, 0)):
                        line = next(lines_iterator)
                    else:
                        line = lines_iterator.next()
                # Make loop object and populate attributes.
                loop = Loop()
                loop.convergence = float(cols[1])
                loop.direc = self.direc
                loop.ff = self.ff
                loop.args_ff = self.args_ff
                loop.args_ref = self.args_ref
                loop.ref_data = self.ref_data
                loop.loop_lines = inner_loop_lines
                # Log commands.
                pretty_loop_input(
                    inner_loop_lines, name='OPTIMIZATION LOOP',
                    score=self.ff.score)
                # Run inner loop.
                self.ff = loop.opt_loop()
            # Note: Probably want to update this to append the directory given
            #       by the new DIR command.
            if cols[0] == 'RDAT':
                logger.log(
                    20, '~~ CALCULATING REFERENCE DATA ~~'.rjust(79, '~'))
                if len(cols) > 1:
                    self.args_ref = ' '.join(cols[1:]).split()
                self.ref_data = opt.return_ref_data(self.args_ref)
            if cols[0] == 'CDAT':
                logger.log(
                    20, '~~ CALCULATING FF DATA ~~'.rjust(79, '~'))
                if len(cols) > 1:
                    self.args_ff = ' '.join(cols[1:]).split()
                self.ff.data = calculate.main(self.args_ff)
            if cols[0] == 'COMP':
                self.ff.score = compare.compare_data(
                    self.ref_data, self.ff.data)
                if '-o' in cols:
                    compare.pretty_data_comp(
                        self.ref_data,
                        self.ff.data,
                        os.path.join(self.direc, cols[cols.index('-o') + 1]))
                if '-p' in cols:
                    compare.pretty_data_comp(
                        self.ref_data,
                        self.ff.data,
                        doprint=True)
            if cols[0] == 'GRAD':
                grad = gradient.Gradient(
                    direc=self.direc,
                    ff=self.ff,
                    ff_lines=self.ff.lines,
                    args_ff=self.args_ff)
                self.ff = grad.run(ref_data=self.ref_data)
            if cols[0] == 'SIMP':
                simp = simplex.Simplex(
                    direc=self.direc,
                    ff=self.ff,
                    ff_lines=self.ff.lines,
                    args_ff=self.args_ff)
                self.ff = simp.run(r_data=self.ref_data)
            if cols[0] == 'WGHT':
                data_type = cols[1]
                co.WEIGHTS[data_type] = float(cols[2])

def read_loop_input(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    lines = [x.partition('#')[0].strip('\n') for x in lines if
             x.partition('#')[0].strip('\n') != '']
    pretty_loop_input(lines)
    return lines

def pretty_loop_input(lines, name='Q2MM', score=None):
    logger.log(20, ' {} '.format(name).center(79, '='))
    logger.log(20, 'COMMANDS:')
    for line in lines:
        logger.log(20, '> ' + line)
    if score is not None:
        logger.log(20, 'SCORE: {}'.format(score))
    logger.log(20, '=' * 79)
    logger.log(20, '')
    return lines

def pretty_loop_summary(cycle_num, score, change):
    logger.log(20, ' Cycle {} Summary '.format(
            cycle_num).center(50, '-'))
    logger.log(20, '| PF Score: {:36.15f} |'.format(score))
    logger.log(20, '| % change: {:36.15f} |'.format(change * 100))
    logger.log(20, '-' * 50)

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'input', type=str, help='Filename containing loop commands.')
    opts = parser.parse_args(args)
    lines = read_loop_input(opts.input)
    loop = Loop()
    loop.run_loop_input(lines)

if __name__ == '__main__':
    # if os.path.isfile('root.log'):
    #     os.remove('root.log')
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
