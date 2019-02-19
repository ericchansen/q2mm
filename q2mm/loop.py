#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division

import argparse
import glob
import logging
import logging.config
import numpy as np
import os
import random
import sys
import re

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
                line = next(lines_iterator)
            except StopIteration:
                return self.ff
            cols = line.split()
            if cols[0] == 'DIR':
                self.direc = cols[1]
            if cols[0] == 'FFLD':
                # Import FF data.
                if cols[1] == 'read':
                    if cols[2] == 'mm3.fld':
                        self.ff = datatypes.MM3(os.path.join(self.direc, 
                                                             cols[2]))
                    if '.prm' in cols[2]:
                        self.ff = datatypes.TinkerFF(os.path.join(self.direc,
                                                                  cols[2]))
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
                line = next(lines_iterator)
                while line.split()[0] != 'END':
                    inner_loop_lines.append(line)
                    line = next(lines_iterator)
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
            # Deprecated
            #    self.ff.score = compare.compare_data(
            #        self.ref_data, self.ff.data)
            #    if '-o' in cols:
            #        compare.pretty_data_comp(
            #            self.ref_data,
            #            self.ff.data,
            #            os.path.join(self.direc, cols[cols.index('-o') + 1]))
            #    if '-p' in cols:
            #        compare.pretty_data_comp(
            #            self.ref_data,
            #            self.ff.data,
            #            doprint=True)
                output = False
                doprint = False
                r_dict = compare.data_by_type(self.ref_data)
                c_dict = compare.data_by_type(self.ff.data)
                r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                if '-o' in cols:
                    output = os.path.join(self.direc, cols[cols.index('-o') +1])
                if '-p' in cols:
                    doprint = True
                self.ff.score = compare.compare_data(
                    r_dict, c_dict, output=output, doprint=doprint)
            if cols[0] == 'GRAD':
                grad = gradient.Gradient(
                    direc=self.direc,
                    ff=self.ff,
                    ff_lines=self.ff.lines,
                    args_ff=self.args_ff)
                #### Should probably just write a function instead of looping
                #### this for every gradient method. This includes everything
                #### between the two lines of #. TR 20180112
                ##############################################################        
                for col in cols[1:]:
                    if "lstsq" in col:
                        g_args = col.split('=')[1].split(',')
                        for arg in g_args:
                            if arg == "True":
                                grad.do_lstsq=True
                            elif arg == False:
                                grad.do_lstsq=False
                            if 'radii' in arg:
                                grad.lstsq_radii = []
                                radii_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if radii_vals == "None":
                                    grad.lstsq_radii = None
                                else:
                                    for val in radii_vals:
                                        grad.lstsq_radii.append(float(val)) 
                            if 'cutoff' in arg:
                                grad.lstsq_cutoff = []
                                cutoff_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if cutoff_vals == "None":
                                    grad.lstsq_cutoff = None
                                else:
                                    if len(cutoff_vals) > 2 or \
                                        len(cutoff_vals) < 2:
                                        raise Exception("Cutoff values must " \
                                            "be between two numbers.")
                                    for val in cutoff_vals:
                                        grad.lstsq_cutoff.append(float(val))
                    elif "newton" in col:
                        g_args = col.split('=')[1].split(',')
                        for arg in g_args:
                            if arg == "True":
                                grad.do_newton=True
                            elif arg == False:
                                grad.do_newton=False
                            if 'radii' in arg:
                                grad.newton_radii = []
                                radii_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if radii_vals=='None':
                                    grad.newton_radii = None
                                else:
                                    for val in radii_vals:
                                        grad.newton_radii.append(float(val)) 
                            if 'cutoff' in arg:
                                grad.newton_cutoff = []
                                cutoff_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if cutoff_vals=='None':
                                    grad.newton_cutoff = None
                                else:
                                    if len(cutoff_vals) > 2 or \
                                        len(cutoff_vals) < 2:
                                        raise Exception("Cutoff values must " \
                                            "be between two numbers.")
                                    for val in cutoff_vals:
                                        grad.newton_cutoff.append(float(val))
                    elif "levenberg" in col:
                        g_args = col.split('=')[1].split(',')
                        for arg in g_args:
                            if arg == "True":
                                grad.do_levenberg=True
                            elif arg == False:
                                grad.do_levenberg=False
                            if 'radii' in arg:
                                grad.levenberg_radii = []
                                radii_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if radii_vals=='None':
                                    grad.levenberg_radii = None
                                else:
                                    for val in radii_vals:
                                        grad.levenberg_radii.append(float(val)) 
                            if 'cutoff' in arg:
                                grad.levenberg_cutoff = []
                                cutoff_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if cutoff_vals=='None':
                                    grad.levenberg_cutoff = None
                                else:
                                    if len(cutoff_vals) > 2 or \
                                        len(cutoff_vals) < 2:
                                        raise Exception("Cutoff values must " \
                                            "be between two numbers.")
                                    for val in cutoff_vals:
                                        grad.levenberg_cutoff.append(float(val))
                            if 'factor' in arg:
                                grad.levenberg_cutoff = []
                                factor_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if factor_vals=='None':
                                    grad.levenberg_factor = None
                                else:
                                    for val in factor_vals:
                                        grad.levenberg_factor.append(float(val))
                    elif "lagrange" in col:
                        g_args = col.split('=')[1].split(',')
                        for arg in g_args:
                            if arg == "True":
                                grad.do_lagrange=True
                            elif arg == False:
                                grad.do_lagrange=False
                            if 'radii' in arg:
                                grad.lagrange_radii = []
                                radii_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if radii_vals=='None':
                                    grad.lagrange_radii = None
                                else:
                                    for val in radii_vals:
                                        grad.lagrange_radii.append(float(val)) 
                            if 'cutoff' in arg:
                                grad.lagrange_cutoff = []
                                cutoff_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if cutoff_vals=='None':
                                    grad.lagrange_cutoff = None
                                else:
                                    if len(cutoff_vals) > 2 or \
                                        len(cutoff_vals) < 2:
                                        raise Exception("Cutoff values must " \
                                            "be between two numbers.")
                                    for val in cutoff_vals:
                                        grad.lagrange_cutoff.append(float(val))
                            if 'factor' in arg:
                                grad.lagrange_factors = []
                                factor_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if factor_vals=='None':
                                    grad.lagrange_factors = None
                                else:
                                    for val in factor_vals:
                                        grad.lagrange_factors.append(float(val))
                    elif "svd" in col:
                        g_args = col.split('=')[1].split(',')
                        for arg in g_args:
                            if arg == "True":
                                grad.do_svd=True
                            elif arg == False:
                                grad.do_svd=False
                            if 'radii' in arg:
                                grad.svd_radii = []
                                radii_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if radii_vals=='None':
                                    grad.svd_radii = None
                                else:
                                    for val in radii_vals:
                                        grad.svd_radii.append(float(val)) 
                            if 'cutoff' in arg:
                                grad.svd_cutoff = []
                                cutoff_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if cutoff_vals=='None':
                                    grad.svd_cutoff = None
                                else:
                                    if len(cutoff_vals) > 2 or \
                                        len(cutoff_vals) < 2:
                                        raise Exception("Cutoff values must " \
                                            "be between two numbers.")
                                    for val in cutoff_vals:
                                        grad.svd_cutoff.append(float(val))
                            if 'factor' in arg:
                                grad.svd_cutoff = []
                                factor_vals = re.search(
                                    r"\[(.+)\]",arg).group(1).split('/')
                                if factor_vals=='None':
                                    grad.svd_factor = None
                                else:
                                    for val in factor_vals:
                                        grad.svd_factor.append(float(val))
                    else:
                        raise Exception("'{}' : Not Recognized".format(col))
                ##############################################################
                self.ff = grad.run(ref_data=self.ref_data)
            if cols[0] == 'SIMP':
                simp = simplex.Simplex(
                    direc=self.direc,
                    ff=self.ff,
                    ff_lines=self.ff.lines,
                    args_ff=self.args_ff)
                for col in cols[1:]:
                    if "max_params" in col:
                        simp.max_params = col.split('=')[1]
                    else:
                        raise Exception("'{}' : Not Recognized".format(col))
                self.ff = simp.run(r_data=self.ref_data)
            if cols[0] == 'WGHT':
                data_type = cols[1]
                co.WEIGHTS[data_type] = float(cols[2])
            if cols[0] == 'STEP':
                param_type = cols[1]
                co.STEPS[param_type] = float(cols[2])

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
