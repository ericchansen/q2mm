#!/usr/bin/python
'''
Simplex optimizer.
'''
import argparse
import copy
import logging
import sys

from calculate import run_calculate
from compare import calc_x2
from datatypes import FF, MM3
from optimizer import Optimizer

logger = logging.getLogger(__name__)

class Simplex(Optimizer):
    '''
    For implementation details, read:
    Norrby, Liljefors. Automated Molecular Mechanics Parameterization
    with Simultaneous Utilization of Experimental and Quantum Mechanical
    Data. J. Comp. Chem., 1998, 1146-1166.
    '''
    def __init__(self):
        super(Simplex, self).__init__()
        self.best_ff = None
        self.current_cycle = None
        self.cycles_wo_change = None
        self.ffs_central = None
        self.ffs_forward = None
        self.last_best = None
        self.massive_contraction = True
        self.max_cycles = 15
        self.max_wo_change = 3
        self.max_params = 15
        self.params = None
        self.use_weight = True

    def return_simplex_parser(self, add_help=True, parents=None):
        '''
        Return an argparse.ArgumentParser object containing options
        for simplex optimizations.
        '''
        if parents is None:
            parents = []
        if add_help:
            parser = argparse.ArgumentParser(
                description=__doc__, add_help=add_help, parents=parents)
        else:
            parser = argparse.ArgumentParser(
                add_help=False, parents=parents)

        group = parser.add_argument_group('simplex optimization')
        group.add_argument(
            '--max_cycles', type=int, default=15,
            help='Maximum number of simplex optimization cycles.')
        group.add_argument(
            '--max_wo_change', type=int, default=3,
            help=('Maximum number of consecutive simplex optimization '
                  'cycles yielding no change to the penalty function.'))
        group.add_argument(
            '--max_params', type=int, default=15,
            help=('Maximum number of parameters used in the simplex '
                  'optimization.'))
        group.add_argument(
            '--no_massive', action='store_true',
            help="Don't use massive contraction to modify parameter sets.")
        group.add_argument(
            '--no_weight', action='store_true',
            help=('Calculate the simplex inversion point without weighting '
                  'parameter sets by their value of the penalty function.'))
        return parser

    def setup_simplex(self, opts):
        '''
        Set options used in the simplex optimization.
        '''
        self.max_cycles = opts.max_cycles
        self.max_wo_change = opts.max_wo_change
        self.max_params = opts.max_params
        self.massive_contraction = not opts.no_massive
        self.use_weight = not opts.no_weight

    def run(self):
        '''
        Runs the simplex optimization. Run self.setup before this, or
        manually assign the necessary attributes that would normally
        be assigned by self.setup.

        self.init_ff must have all the required attributes for
        self.init_ff.export_ff.
        '''
        logger.info('--- running {} ---'.format(type(self).__name__))
        if self.init_ff.x2 is None:
            self.calc_x2_ff(self.init_ff)

        if self.max_params is not None and len(self.init_ff.params) > self.max_params:
            self.ffs_central = self.params_diff(self.init_ff.params, mode='central')
            for ff in self.ffs_central:
                self.calc_x2_ff(ff)
            self.central_diff_derivs(self.init_ff, self.ffs_central)
            self.params = self.trim_params_on_2nd(self.init_ff.params)
            self.ffs_forward = [x for x in self.ffs_central if
                           x.method.split()[0] =='forward' and
                           int(x.method.split()[1]) in [y.mm3_row for y in self.params] and
                           int(x.method.split()[2]) in [y.mm3_col for y in self.params]]
        else:
            self.params = copy.deepcopy(self.init_ff.params)
            self.ffs_forward = self.params_diff(self.params, mode='forward')
            for ff in self.ffs_forward:
                self.calc_x2_ff(ff)
        self.trial_ffs = sorted(self.ffs_forward + [self.init_ff], key=lambda x: x.x2)

        self.current_cycle = 0
        self.cycles_wo_change = 0
        while self.current_cycle < self.max_cycles and self.cycles_wo_change < self.max_wo_change:
            self.last_best = self.trial_ffs[0].x2 # copy necessary? copy.deepcopy?
            self.current_cycle += 1
            logger.info('simplex - start of cycle {} - {} ({})'.format(
                    self.current_cycle, self.trial_ffs[0].x2, self.trial_ffs[0].method))
            logger.info('{}'.format([x.x2 for x in self.trial_ffs]))
            inverted = FF()
            inverted.method = 'inversion'
            inverted.params = copy.deepcopy(self.params)
            reflected = FF()
            reflected.method = 'reflection'
            reflected.params = copy.deepcopy(self.params)
            for i in xrange(0, len(self.params)):
                if self.use_weight:
                    try:
                        param_inverted = \
                            sum([x.params[i].value * (x.x2 - self.trial_ffs[-1].x2)
                                 for x in self.trial_ffs[:-1]]) / \
                            sum([x.x2 - self.trial_ffs[-1].x2 for x in self.trial_ffs[:-1]])
                    except ZeroDivisionError:
                        logger.warning('zero division. all x2 are numerically equivalent')
                        raise
                else:
                    param_inverted = \
                        sum([x.params[i].value for x in self.trial_ffs[:-1]]) / \
                        len(self.trial_ffs[:-1])
                inverted.params[i].value = param_inverted
                reflected.params[i].value = 2 * param_inverted - self.trial_ffs[-1].params[i].value
            inverted.display_params()
            reflected.display_params()
            inverted.check_params()
            reflected.check_params()
            self.calc_x2_ff(reflected)
            if reflected.x2 < self.trial_ffs[0].x2:
                logger.info('attempting expansion')
                expanded = FF()
                expanded.method = 'expansion'
                expanded.params = copy.deepcopy(self.params)
                for i in xrange(0, len(self.params)):
                    # expanded_param = 3 * inverted.params[i].value - \
                    #     2 * self.trial_ffs[-1].params[i].value
                    # expanded.params[i].value = expanded_param
                    expanded.params[i].value = 3 * inverted.params[i].value - \
                        2 * self.trial_ffs[-1].params[i].value
                expanded.display_params()
                expanded.check_params()
                self.calc_x2_ff(expanded)
                if expanded.x2 < reflected.x2:
                    self.trial_ffs[-1] = expanded
                    logger.info('expansion succeeded. keeping')
                else:
                    self.trial_ffs[-1] = reflected
                    logger.info('expansion failed. keeping reflection')
            elif reflected.x2 < self.trial_ffs[-2].x2:
                logger.info('keeping reflection')
                self.trial_ffs[-1] = reflected
            else:
                logger.info('attempting contraction')
                contracted = FF()
                contracted.method = 'contraction'
                contracted.params = copy.deepcopy(self.params)
                for i in xrange(0, len(self.params)):
                    if reflected.x2 > self.trial_ffs[-1].x2:
                        contracted_param = (inverted.params[i].value + \
                                                self.trial_ffs[-1].params[i].value) / 2
                    else:
                        contracted_param = (3 * inverted.params[i].value - \
                                                self.trial_ffs[-1].params[i].value) / 2
                    contracted.params[i].value = contracted_param
                contracted.display_params()
                contracted.check_params()
                self.calc_x2_ff(contracted)
                if contracted.x2 < self.trial_ffs[-2].x2:
                    self.trial_ffs[-1] = contracted
                elif self.massive_contraction:
                    logger.info('doing massive contraction')
                    for ff_num, ff in enumerate(self.trial_ffs[1:]):
                        for i in xrange(0, len(self.params)):
                            ff.params[i].value = (ff.params[i].value + \
                                                      self.trial_ffs[0].params[i].value) / 2
                        ff.display_params()
                        ff.check_params()
                        ff.method += ' / massive contraction'
                        self.calc_x2_ff(ff)
                else:
                    logger.info('contraction failed')
            self.trial_ffs = sorted(self.trial_ffs, key=lambda x: x.x2)
            if self.trial_ffs[0].x2 < self.last_best:
                self.cycles_wo_change = 0
            else:
                self.cycles_wo_change += 1
                logger.info('{} cycles w/o change'.format(self.cycles_wo_change))
            logger.info('simplex - end of cycle {} - {} ({})'.format(
                    self.current_cycle, self.trial_ffs[0].x2, self.trial_ffs[0].method))
        if self.trial_ffs[0].x2 < self.init_ff.x2:
            self.best_ff = MM3()
            self.best_ff.method = self.trial_ffs[0].method
            self.best_ff.copy_attributes(self.init_ff)
            # add in parameters that were previously removed
            if self.max_params is not None and len(self.init_ff.params) > self.max_params:
                self.best_ff.params = copy.deepcopy(self.init_ff.params)
                for i, param_i in enumerate(self.init_ff.params):
                # for param_i in self.best_ff.params:
                    for param_b in self.trial_ffs[0].params:
                        if param_i.mm3_row == param_b.mm3_row and param_i.mm3_col == param_b.mm3_col:
                            logger.log(6, 'Updating {} to {}'.format(param_i, param_b))
                            # is deep copy necessary? im worried that if simplex is done again
                            # in the same loop, it will somehow mess this up
                            # param_i = copy.deepcopy(param_b) 
                            self.best_ff.params[i] = copy.deepcopy(param_b)
            else:
                self.best_ff.params = copy.deepcopy(self.trial_ffs[0].params)
            self.best_ff.x2 = self.trial_ffs[0].x2
            # should never happen
            if self.trial_ffs[0].data is not None:
                self.best_ff.data = self.trial_ffs[0].data
            self.best_ff.export_ff()
            logger.info('--- {} complete ---'.format(type(self).__name__))
            logger.info('initial: {} ({})'.format(self.init_ff.x2, self.init_ff.method))
            logger.info('final: {} ({})'.format(self.best_ff.x2, self.best_ff.method))
            return self.best_ff
        else:
            self.init_ff.export_ff()
            logger.info('--- {} complete ---'.format(type(self).__name__))
            logger.info('initial: {} ({})'.format(self.init_ff.x2, self.init_ff.method))
            logger.info('final: {} ({})'.format(self.trial_ffs[0].x2, self.trial_ffs[0].method))
            logger.info('no improvement from {}'.format(type(self).__name__))
            return self.init_ff

if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    
    simplex = Simplex()
    opts = simplex.parse(sys.argv[1:])
    simplex.setup(opts)
    simplex.run()
    
