#!/usr/bin/python
'''
Loop between optimization techniques until stop criteria has been met.
'''
import argparse
import copy
import logging
import sys
import traceback

from calculate import run_calculate
from compare import calc_x2
from datatypes import MM3
from gradient import Gradient
from optimizer import Optimizer
from simplex import Simplex

logger = logging.getLogger(__name__)

class Loop(Optimizer):
    '''
    Loops between various optimization methods.

    Tries to minimize the number of duplicate calculations.
    '''
    name = 'loop'
    def __init__(self):
        self.best_ff = None
        self.convergence = 0.005
        self.current_cycle = None
        self.loop_max = None
        self.last_best = None

    def return_loop_parser(self, add_help=True, parents=[]):
        '''
        Returns an argument parser for loop.
        '''
        if add_help:
            parser = argparse.ArgumentParser(
                description=__doc__, add_help=add_help, parents=parents)
        else:
            parser = argparse.ArgumentParser(
                add_help=False, parents=parents)
        group = parser.add_argument_group('loop')
        group.add_argument(
            '--convergence', type=float, default=0.005,
            help=('Convergence criterion. If the percent change in the penalty '
                  'between rounds is less than this number, then stop.'))
        group.add_argument(
            '--loop_max', type=int, default=None,
            help='Maximum number of loop cycles.')
        group.add_argument(
            '--gradient', type=str, metavar='" commands for gradient.py"',
            help=('These commands are interpreted by the gradient based optimizer. '
                  'Leave one space after the 1st quotation mark enclosing the arguments.'))
        return parser

    def setup_loop(self, opts):
        self.convergence = opts.convergence
        self.loop_max = opts.loop_max
        return opts

    def run(self, opts=None):
        logger.info('--- {} running ---'.format(type(self).__name__))
        # setup gradient
        gradient = Gradient()
        if opts is not None:
            gradient.parse(opts.gradient.split())
        else:
            gradient.parse(['--default'])
        gradient.com_ref = self.com_ref
        gradient.com_cal = self.com_cal
        gradient.data_ref = self.data_ref
        # setup simplex
        simplex = Simplex()
        simplex.com_ref = self.com_ref
        simplex.com_cal = self.com_cal
        simplex.data_ref = self.data_ref
        # rather than copying, could overwrite to save memory
        self.best_ff = copy.deepcopy(self.init_ff)
        if self.best_ff.x2 is None:
            # data used by gradient, so save it
            self.calc_x2_ff(self.best_ff, save_data=True)
            # no need for data here
            self.init_ff.x2 = self.best_ff.x2
        self.current_cycle = 0
        self.last_best = None
        while ((self.last_best is None or
                abs(self.last_best - self.best_ff.x2) / self.last_best > self.convergence)
               and
               (self.loop_max is None or self.current_cycle < self.loop_max)):
            self.last_best = self.best_ff.x2
            self.current_cycle += 1
            logger.info('loop - start of cycle {} - {} ({})'.format(
                    self.current_cycle, self.best_ff.x2, self.best_ff.method))
            gradient.init_ff = self.best_ff
            self.best_ff = gradient.run()
            self.best_ff.export_ff(path=self.best_ff.path + '.{}.grad'.format(self.current_cycle))
            simplex.init_ff = self.best_ff
            # self.best_ff = simplex.run()
            try:
                self.best_ff = simplex.run()
            except Exception:
                logger.warning(traceback.format_exc())
                logger.warning("in case you didn't notice, simplex raised an exception")
                if simplex.trial_ffs:
                    # make a function for this...
                    self.best_ff = MM3()
                    self.best_ff.method = simplex.trial_ffs[0].method
                    self.best_ff.copy_attributes(self.init_ff)
                    if simplex.max_params is not None and len(simplex.init_ff.params) > simplex.max_params:
                        self.best_ff.params = copy.deepcopy(self.init_ff.params)
                        for param_i in self.best_ff.params:
                            for param_b in simplex.trial_ffs[0].params:
                                if param_i.mm3_row == param_b.mm3_row and param_i.mm3_col == param_b.mm3_col:
                                    param_i = copy.deepcopy(param_b)
                    else:
                        self.best_ff.params = copy.deepcopy(simplex.trial_ffs[0].params)
                    self.best_ff.x2 = simplex.trial_ffs[0].x2
                    if simplex.trial_ffs[0].data is not None:
                        self.best_ff.data = simplex.trial_ffs[0].data
                    # ... block of code
                else:
                    logger.warning('simplex never generated trial force fields')
                self.best_ff.export_ff()
            self.best_ff.export_ff(path=self.best_ff.path + '.{}.simp'.format(self.current_cycle))
            change = abs(self.last_best - self.best_ff.x2) / self.last_best
            logger.info('loop - end of cycle {} - {} ({}) - % change {}'.format(
                    self.current_cycle, self.best_ff.x2, self.best_ff.method, change))
        logger.info('--- {} complete ---'.format(type(self).__name__))
        logger.info('initial: {} ({})'.format(self.init_ff.x2, self.init_ff.method))
        logger.info('final: {} ({})'.format(self.best_ff.x2, self.best_ff.method))
        return self.best_ff
        
if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)

    loop = Loop()
    opts = loop.setup(sys.argv[1:])
    loop.run(opts)
