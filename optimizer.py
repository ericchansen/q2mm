#!/usr/bin/python
'''
Contains general code related to all optimization techniques.

Instead of keeping parameters in a list, we could store them based
upon their MM3* index (row, column). Then we wouldn't need to save
entire parameter sets, significantly reducing the memory usage.
'''
import argparse
import copy
import logging
import os

from calculate import run_calculate
from compare import calc_x2, import_steps
from datatypes import FF, MM3

logger = logging.getLogger(__name__)

class Optimizer(object):
    def __init__(self):
        self.com_ref = None
        self.com_cal = None
        self.data_ref = None
        self.init_ff = None
    def calc_derivs(self, init_ff, central_ffs):
        for i in xrange(0, len(central_ffs), 2):
            init_ff.params[i / 2].der1 = (central_ffs[i].x2 - central_ffs[i + 1].x2) * 0.5 # 18
            init_ff.params[i / 2].der2 = central_ffs[i].x2 + central_ffs[i + 1].x2 - 2 * init_ff.x2 # 19
        logger.info('1st derivatives: {}'.format([x.der1 for x in init_ff.params]))
        logger.info('2nd derivatives: {}'.format([x.der2 for x in init_ff.params]))
    def calc_set_x2(self, ffs, save=False):
        for i, ff in enumerate(ffs):
            self.init_ff.export_ff(params=ff.params)
            if save:
                ff.data = run_calculate(self.com_cal.split())
                ff.x2 = calc_x2(ff.data, self.data_ref)
            else:
                ff.x2 = calc_x2(self.com_cal, self.data_ref)
            if ff.method is None:
                logger.info('{}: {}'.format(i + 1, ff.x2))
            else:
                logger.info('{}: {}'.format(ff.method, ff.x2))
    def params_diff(self, params, mode='central'):
        '''
        Perform forward or central differentiation of parameters.

        Need means of adjusting the parameter step size. When checking the parameter,
        perhaps it should account for the adjusted step size.

        I would say this function is rather memory insensitive and could
        use some improvements.
        '''
        logger.info('{} differentiation on {} parameters'.format(mode, len(params)))
        diff_ffs = []
        for i, param in enumerate(params):
            ff = FF()
            ff.params = copy.deepcopy(params)
            ff.method = 'forward {} {}'.format(param.mm3_row, param.mm3_col)
            ff.params[i].value += ff.params[i].step
            ff.params[i].check_value()
            diff_ffs.append(ff)
            if mode == 'central':
                ff = FF()
                ff.method = 'backward {} {}'.format(param.mm3_row, param.mm3_col)
                ff.params = copy.deepcopy(params)
                ff.params[i].value -= ff.params[i].step
                ff.params[i].check_value()
                diff_ffs.append(ff)
        return diff_ffs
    def return_optimizer_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--calculate', '-c', type=str,
                            metavar = '" commands for calculate.py"',
                            help=('These commands produce the calculated data. Leave one space '
                                  'after the 1st quotation mark enclosing the arguments.'))
        parser.add_argument('--directory', '-d', type=str, metavar='directory', default=os.getcwd(),
                            help='Directory where data and force field files are located.')
        parser.add_argument('--ptypes', '-pt', type=str, nargs='+')
        parser.add_argument('--pfile', '-pf', type=str)
        parser.add_argument('--reference', '-r', type=str,
                            metavar='" commands for calculate.py"',
                            help=('These commands produce the reference data. Leave one space '
                                  'after the 1st quotation mark enclosing the arguments.'))
        return parser
    def setup(self, args):
        logger.info('--- running optimization setup ---')
        opts = self.parse(args)
        self.com_ref = opts.reference
        self.com_cal = opts.calculate
        self.init_ff = MM3(os.path.join(opts.directory, 'mm3.fld'))
        self.init_ff.import_ff()
        self.init_ff.method = 'initial'
        logger.info('loaded {} ff - {} parameters'.format(
                self.init_ff.method, self.init_ff.path, len(self.init_ff.params)))

        params_to_optimize = []
        if opts.ptypes:
            params_to_optimize.extend([x for x in self.init_ff.params if x.ptype in opts.ptypes])
        if opts.pfile:
            with open(os.path.join(opts.directory, opts.pfile), 'r') as f:
                for line in f:
                    line = line.partition('#')[0]
                    cols = line.split()
                    if cols:
                        mm3_row, mm3_col = int(cols[0]), int(cols[1])
                        for param in self.init_ff.params:
                            if mm3_row == param.mm3_row and mm3_col == param.mm3_col:
                                params_to_optimize.append(param)
        for param in params_to_optimize:
            param.check_value()
        import_steps(params_to_optimize)
        self.init_ff.params = params_to_optimize
        logger.info('selected {} parameters for optimization'.format(len(self.init_ff.params)))
    def trim_params_on_2nd(self, params):
        params_sorted = sorted(params, key=lambda x: x.der2)
        params_to_keep = params_sorted[:self.max_params]
        if len(params_sorted) != len(params_to_keep):
            logger.info('reduced number of parameters from {} to {} based on 2nd derivatives'.format(
                    len(params_sorted), len(params_to_keep)))
            logger.info('2nd derivatives: {}'.format([x.der2 for x in params_to_keep]))
        return params_to_keep
