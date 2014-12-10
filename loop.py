#!/usr/bin/python
import copy
import logging
import sys

from calculate import run_calculate
from compare import calc_x2
from gradient import Gradient
from optimizer import Optimizer
from simplex import Simplex

logger = logging.getLogger(__name__)

class Loop(Optimizer):
    def __init__(self):
        self.best_ff = None
        self.convergence = 0.005
        self.loop_max = None
    def parse(self, args):
        parser = self.return_optimizer_parser()
        group = parser.add_argument_group('loop')
        group.add_argument('--convergence', type=float, default=0.005)
        group.add_argument('--loop_max', type=int, default=None)
        opts = parser.parse_args(args)
        self.convergence = opts.convergence
        self.loop_max = opts.loop_max
        return opts
    def run(self, data_ref=None):
        if data_ref is None:
            self.data_ref = run_calculate(self.com_ref.split())
        if self.init_ff.x2 is None:
            self.init_ff.export_ff()
            self.init_ff.x2 = calc_x2(self.com_cal, self.data_ref)
        gradient = Gradient()
        gradient.com_ref = self.com_ref
        gradient.com_cal = self.com_cal
        gradient.data_ref = self.data_ref
        simplex = Simplex()
        simplex.com_ref = self.com_ref
        simplex.com_cal = self.com_cal
        simplex.data_ref = self.data_ref
        self.best_ff = copy.deepcopy(self.init_ff)
        loop_num = 0
        last_best_x2 = None
        while ((self.loop_max is None or loop_num < self.loop_max) and
               (last_best_x2 is None or 
                abs(last_best_x2 - self.best_ff.x2) / last_best_x2 > self.convergence)):
            last_best_x2 = self.best_ff.x2
            loop_num += 1
            logger.info('loop {}: {}'.format(loop_num, last_best_x2))
            gradient.init_ff = self.best_ff
            self.best_ff = gradient.run()
            self.best_ff.export_ff()
            self.best_ff.export_ff(path=self.best_ff.path + '.{}.grad'.format(loop_num))
            simplex.init_ff = self.best_ff
            self.best_ff = simplex.run()
            self.best_ff.export_ff()
            self.best_ff.export_ff(path=self.best_ff.path + '.{}.grad'.format(loop_num))
            percent_change = abs(last_best_x2 - self.best_ff.x2) / last_best_x2
            logger.info('end loop {} - % change: {}'.format(loop_num, percent_change))
        logger.info('loop converged. initial: {} ({}) last: {} best: {} ({})'.format(
                self.init_ff.x2, self.init_ff.method, last_best_x2, self.best_ff.x2, self.best_ff.method))
        return self.best_ff
               
        
if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)

    loop = Loop()
    loop.setup(sys.argv[1:])
    best_ff = loop.run()
