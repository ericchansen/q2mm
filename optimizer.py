#!/usr/bin/python
'''
General code related to all optimization techniques.
'''
import argparse
import copy
import logging
import os
import sys

from calculate import run_calculate
from compare import calc_x2, import_steps
from datatypes import FF, MM3, UnallowedNegative
from parameters import return_parameters_parser, select_parameters

logger = logging.getLogger(__name__)

class Optimizer(object):
    '''
    All specific optimization classes (ex. Gradient, Simplex, etc.)
    inherit from this class. This class contains methods that should
    be general to all optimization techniques.
    '''
    def __init__(self):
        self.com_ref = None
        self.com_cal = None
        self.data_ref = None
        self.init_ff = None

    def return_optimizer_parser(self, add_help=False, parents=[]):
        '''
        Return an argparse.ArgumentParser object that is general to
        all optimization techniques.
        '''
        parameters_parser = return_parameters_parser(add_help=False)
        parents.append(parameters_parser)
        parser = argparse.ArgumentParser(add_help=add_help, parents=parents)
        
        parser.add_argument(
            '--calculate', '-c', type=str, metavar='" commands for calculate.py"',
            help=('These commands produce the calculated data. Leave one space '
                  'after the first quotation mark enclosing the arguments.'))
        parser.add_argument(
            '--directory', '-d', type=str, metavar='directory', default=os.getcwd(),
            help=('Directory where data and force field files are located. '
                  'Calculations will be performed in this directory.'))
        parser.add_argument(
            '--reference', '-r', type=str, metavar='" commands for calculate.py"',
            help=('These commands produce the reference data. Leave one space '
                  'after the first quotation mark enclosing the arguments.'))
        return parser

    def central_diff_derivs(self, init_ff, central_ffs):
        for i in xrange(0, len(central_ffs), 2):
            init_ff.params[i / 2].der1 = (central_ffs[i].x2 - central_ffs[i + 1].x2) * 0.5 # 18
            init_ff.params[i / 2].der2 = central_ffs[i].x2 + \
                central_ffs[i + 1].x2 - 2 * init_ff.x2 # 19
        logger.info('1st derivatives: {}'.format([x.der1 for x in init_ff.params]))
        logger.info('2nd derivatives: {}'.format([x.der2 for x in init_ff.params]))

    def calc_x2_ff(self, ff, save_data=False):
        self.init_ff.export_ff(params=ff.params)
        if save_data:
            ff.data = run_calculate(self.com_cal.split())
            ff.x2 = calc_x2(ff.data, self.data_ref)
        else:
            ff.x2 = calc_x2(self.com_cal, self.data_ref)
        logger.info('{}: {}'.format(ff.method, ff.x2))

    def params_diff(self, params, mode='central'):
        '''
        Perform forward or central differentiation on parameters.
        '''
        logger.info('{} differentiation - {} parameters'.format(mode, len(params)))
        import_steps(params)
        diff_ffs = []
        for i, param in enumerate(params):
            while True:
                try:
                    ff_forward = FF()
                    ff_forward.params = copy.deepcopy(params)
                    ff_forward.method = 'forward {} {}'.format(param.mm3_row, param.mm3_col)
                    ff_forward.params[i].value += ff_forward.params[i].step
                    ff_forward.params[i].check_value()
                    if mode == 'central':
                        ff_backward = FF()
                        ff_backward.method = 'backward {} {}'.format(param.mm3_row, param.mm3_col)
                        ff_backward.params = copy.deepcopy(params)
                        ff_backward.params[i].value -= ff_backward.params[i].step
                        ff_backward.params[i].check_value()
                except UnallowedNegative as e:
                    logger.warning(e.message)
                    logger.warning('changing step size of {} from {} to {}'.format(
                            param, param.step, param.value * 0.05))
                    param.step = param.value * 0.05
                else:
                    ff_forward.display_params()
                    diff_ffs.append(ff_forward)
                    if mode == 'central':
                        ff_backward.display_params()
                        diff_ffs.append(ff_backward)
                    break
        logger.info('generated {} force fields for {} differentiation'.format(len(diff_ffs), mode))
        return diff_ffs

    def setup(self, args):
        '''
        General setup is applied to the start of any optimization technique.
        '''
        if self.__class__.__name__ == 'Gradient':
            gradient_parser = self.return_gradient_parser(add_help=True)
            parser = self.return_optimizer_parser(
                add_help=False, parents=[gradient_parser])
        elif self.__class__.__name__ == 'Simplex':
            simplex_parser = self.return_simplex_parser(add_help=True)
            parser = self.return_optimizer_parser(
                add_help=False, parents=[simplex_parser])

        opts = parser.parse_args(args)

        if self.__class__.__name__ == 'Gradient':
            self.setup_gradient(opts)
        elif self.__class__.__name__ == 'Simplex':
            self.setup_simplex(opts)

        logger.info('--- setup {} ---'.format(type(self).__name__))

        # not needed with parameter groups?
        self.com_ref = opts.reference
        self.com_cal = opts.calculate

        # load existing force field
        self.init_ff = MM3(os.path.join(opts.directory, 'mm3.fld'))
        self.init_ff.import_ff()
        self.init_ff.method = 'initial'
        logger.info('{} ff loaded from {} - {} parameters'.format(
                self.init_ff.method, self.init_ff.path, len(self.init_ff.params)))
        
        # select parameters from existing force field
        # reduce number of initial parameters to only those selected
        params_to_optimize = select_parameters(opts, ff=self.init_ff)
        for param in params_to_optimize:
            param.check_value()
        # import_steps(params_to_optimize)
        self.init_ff.params = params_to_optimize
        logger.info('{} parameters selected for optimization'.format(len(self.init_ff.params)))

        # evaluate the reference data
        # not needed with parameter groups?
        self.data_ref = run_calculate(self.com_ref.split())
        
        return opts

    def trim_params_on_2nd(self, params):
        '''
        Returns a new list of parameters containing the ones that have
        the lowest 2nd derivatives with respect to the objective
        function. Doesn't modify the input parameter list.
        '''
        params_sorted = sorted(params, key=lambda x: x.der2)
        params_to_keep = params_sorted[:self.max_params]
        if len(params_sorted) != len(params_to_keep):
            logger.info('reduced number of parameters from {} to {} based on 2nd derivatives'.format(
                    len(params_sorted), len(params_to_keep)))
            logger.info('2nd derivatives: {}'.format([x.der2 for x in params_to_keep]))
        return params_to_keep
