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
    Norrby, Liljefors. Automated Molecular Mechanics Parameterization with
    Simultaneous Utilization of Experimental and Quantum Mechanical Data.
    J. Comp. Chem., 1998, 1146-1166.
    '''
    name = 'simplex'
    def __init__(self):
        super(Simplex, self).__init__()
        self.massive_contraction = True
        self.max_cycles = 25
        self.max_wo_change = 5
        self.max_params = 15
        self.use_weight = True
    def parse(self, args):
        parser = self.return_optimizer_parser()
        group = parser.add_argument_group('simplex')
        group.add_argument('--max_cycles', type=int, default=25)
        group.add_argument('--max_wo_change', type=int, default=5)
        group.add_argument('--max_params', type=int, default=15)
        group.add_argument('--no_massive', action='store_true')
        group.add_argument('--no_weight', action='store_true')
        opts = parser.parse_args(args)
        self.max_cycles = opts.max_cycles
        self.max_wo_change = opts.max_wo_change
        self.max_params = opts.max_params
        self.massive_contraction = not opts.no_massive
        self.use_weight = not opts.no_weight
        return opts
    def run(self, data_ref=None):
        logger.info('--- running {} ---'.format(self.name))
        if data_ref is None:
            self.data_ref = run_calculate(self.com_ref.split())
        if self.init_ff.x2 is None:
            self.init_ff.export_ff() # may be unnecessary
            self.init_ff.x2 = calc_x2(self.com_cal, self.data_ref)
        logger.info('{}: {}'.format(self.init_ff.method, self.init_ff.x2))
        if self.max_params is not None and len(self.init_ff.params) > self.max_params:
            central_ffs = self.params_diff(self.init_ff.params, mode='central')
            self.calc_set_x2(central_ffs)
            self.calc_derivs(self.init_ff, central_ffs)
            self.init_ff.params = self.trim_params_on_2nd(self.init_ff.params)
            forward_ffs = [x for x in central_ffs if
                           x.method.split()[0] =='forward' and
                           int(x.method.split()[1]) in [y.mm3_row for y in self.init_ff.params] and
                           int(x.method.split()[2]) in [y.mm3_col for y in self.init_ff.params]]
        else:
            forward_ffs = self.params_diff(self.init_ff.params, mode='forward')
            self.calc_set_x2(forward_ffs)
        self.trial_ffs = forward_ffs + [self.init_ff]
        self.trial_ffs = sorted(self.trial_ffs, key=lambda x: x.x2)
        cycle_num = 0
        cycles_wo_change = 0
        while cycle_num < self.max_cycles and cycles_wo_change < self.max_wo_change:
            old_best_x2 = self.trial_ffs[0].x2 # copy necessary? copy.deepcopy?
            cycle_num += 1
            logger.info('start simplex cycle {}: {} ({})'.format(
                    cycle_num, self.trial_ffs[0].x2, self.trial_ffs[0].method))
            logger.info('all x2: {}'.format([x.x2 for x in self.trial_ffs]))
            inverted_ff = FF()
            inverted_ff.method = 'inversion'
            inverted_ff.params = copy.deepcopy(self.init_ff.params)
            reflected_ff = FF()
            reflected_ff.method = 'reflection'
            reflected_ff.params = copy.deepcopy(self.init_ff.params)
            for i in xrange(0, len(self.init_ff.params)):
                if self.use_weight:
                    inverted_param = \
                        sum([x.params[i].value * (x.x2 - self.trial_ffs[-1].x2)
                             for x in self.trial_ffs[:-1]]) / \
                        sum([x.x2 - self.trial_ffs[-1].x2 for x in self.trial_ffs[:-1]])
                else:
                    inverted_param = \
                        sum([x.params[i].value for x in self.trial_ffs[:-1]]) / \
                        len(self.trial_ffs[:-1])
                inverted_ff.params[i].value = inverted_param
                inverted_ff.params[i].check_value()
                reflected_param = 2 * inverted_param - self.trial_ffs[-1].params[i].value
                reflected_ff.params[i].value = reflected_param
                reflected_ff.params[i].check_value()
            logger.log(7, 'inverted parameters: {}'.format(inverted_ff.params))
            logger.log(7, 'reflected parameters: {}'.format(reflected_ff.params))
            self.init_ff.export_ff(params=reflected_ff.params)
            reflected_ff.x2 = calc_x2(self.com_cal, self.data_ref)
            logger.info('{}: {}'.format(reflected_ff.method, reflected_ff.x2))
            if reflected_ff.x2 < self.trial_ffs[0].x2:
                logger.info('attempting expansion')
                expanded_ff = FF()
                expanded_ff.method = 'expansion'
                expanded_ff.params = copy.deepcopy(self.init_ff.params)
                for i in xrange(0, len(self.init_ff.params)):
                    expanded_param = 3 * inverted_ff.params[i].value - \
                        2 * self.trial_ffs[-1].params[i].value
                    expanded_ff.params[i].value = expanded_param
                    expanded_ff.params[i].check_value()
                logger.log(7, 'expanded parameters: {}'.format(expanded_ff.params))
                self.init_ff.export_ff(params=expanded_ff.params)
                expanded_ff.x2 = calc_x2(self.com_cal, self.data_ref)
                logger.info('expansion: {}'.format(expanded_ff.x2))
                if expanded_ff.x2 < reflected_ff.x2:
                    self.trial_ffs[-1] = expanded_ff
                    logger.info('expansion succeeded. keeping')
                else:
                    self.trial_ffs[-1] = reflected_ff
                    logger.info('expansion failed. keeping reflection')
            elif reflected_ff.x2 < self.trial_ffs[-2].x2:
                logger.info('keeping reflection')
                self.trial_ffs[-1] = reflected_ff
            else:
                logger.info('attempting contraction')
                contracted_ff = FF()
                contracted_ff.method = 'contraction'
                contracted_ff.params = copy.deepcopy(self.init_ff.params)
                for i in xrange(0, len(self.init_ff.params)):
                    if reflected_ff.x2 > self.trial_ffs[-1].x2:
                        contracted_param = (inverted_ff.params[i].value + self.trial_ffs[-1].params[i].value) / 2
                    else:
                        contracted_param = (3 * inverted_ff.params[i].value - self.trial_ffs[-1].params[i].value) / 2
                    contracted_ff.params[i].value = contracted_param
                    contracted_ff.params[i].check_value()
                logger.log(7, 'contracted parameters: {}'.format(contracted_ff.params))
                self.init_ff.export_ff(params=contracted_ff.params)
                contracted_ff.x2 = calc_x2(self.com_cal, self.data_ref)
                logger.info('contraction: {}'.format(contracted_ff.x2))
                if contracted_ff.x2 < self.trial_ffs[-2].x2:
                    self.trial_ffs[-1] = contracted_ff
                elif self.massive_contraction:
                    logger.info('doing massive contraction')
                    for ff_num, ff in enumerate(self.trial_ffs[1:]):
                        for i in xrange(0, len(self.init_ff.params)):
                            ff.params[i].value = (ff.params[i].value + self.trial_ffs[0].params[i].value) / 2
                            ff.params[i].check_value()
                        logger.log(7, 'massive contraction parameters {}: {}'.format(
                                ff_num + 2, ff.params))
                        self.init_ff.export_ff(params=ff.params)
                        ff.x2 = calc_x2(self.com_cal, self.data_ref)
                    logger.info('after massive contraction: {}'.format([x.x2 for x in self.trial_ffs]))
            self.trial_ffs = sorted(self.trial_ffs, key=lambda x: x.x2)
            if self.trial_ffs[0].x2 < old_best_x2:
                cycles_wo_change = 0
            else:
                cycles_wo_change += 1
                logger.info('{} cycles w/o change in best'.format(cycles_wo_change))
            # if you remove this (not necessary here), don't forget it later
            self.init_ff.export_ff(params=self.trial_ffs[0].params)
        logger.info('end simplex ({} cycles): {} ({}) vs {} ({})'.format(
                cycle_num, self.trial_ffs[0].x2, self.trial_ffs[0].method, self.init_ff.x2, self.init_ff.method))
        # would be better to copy the initial if it didn't change in case we
        # check for derivatives already existing
        best_ff = MM3()
        self.init_ff.copy_attributes_to(best_ff)
        best_ff.params = copy.deepcopy(self.trial_ffs[0].params)
        best_ff.x2 = self.trial_ffs[0].x2
        best_ff.export_ff()
        return best_ff

if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    
    simplex = Simplex()
    simplex.setup(sys.argv[1:])
    simplex.run()
    
