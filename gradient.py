#!/usr/bin/python
'''
Gradient/Newton-Raphson based optimizer.
'''
import argparse
import copy
import logging
import numpy as np
import sys

from calculate import run_calculate
from compare import calc_x2, import_weights
from datatypes import FF, MM3, UnallowedNegative
from optimizer import Optimizer

logger = logging.getLogger(__name__)

class OptimizerException(Exception):
    pass
class RadiusException(Exception):
    pass

class Gradient(Optimizer):
    def __init__(self):
        super(Gradient, self).__init__()
        self.best_ff = None
        self.cutoffs = [0.01, 100]
        self.do_basic = True
        self.do_lagrange = True
        self.do_levenberg = True
        self.do_derivs = True
        self.do_svd = True
        self.ffs_central = None
        self.lagrange_factors = sorted([10.0, 1.0, 0.1, 0.01])
        self.levenberg_factors = sorted([10.0, 1.0, 0.1, 0.01])
        self.max_radius = 1
        self.solver_method = 'lstsq'
        self.svd_thresholds = None

    def calc_jacobian(self, ffs):
        jacobian = np.empty((len(ffs[0].data), len(ffs) / 2), dtype=float)
        for ff in ffs:
            import_weights(ff.data)
        for i, index_ff in enumerate(xrange(0, len(ffs), 2)):
            # i = 0, 1, 2, ...
            # index_ff = 0, 2, 4, ...
            for index_datum in xrange(0, len(ffs[0].data)):
                dydp = (ffs[index_ff].data[index_datum].value - ffs[index_ff + 1].data[index_datum].value) / 2
                jacobian[index_datum, i] = ffs[index_ff].data[index_datum].weight * dydp
        logger.log(8, 'created {} jacobian'.format(jacobian.shape))
        return jacobian

    def calc_lagrange(self, A, b, factor):
        A_copy = copy.deepcopy(A)
        indices = np.diag_indices_from(A_copy)
        A_copy[indices] = A_copy[indices] + factor
        changes = self.solver(A_copy, b)
        return changes

    def calc_levenberg(self, A, b, factor):
        A_copy = copy.deepcopy(A)
        indices = np.diag_indices_from(A_copy)
        A_copy[indices] = A_copy[indices] * (1 + factor)
        changes = self.solver(A_copy, b)
        return changes

    def one_dim_newton(self, central_ffs):
        changes = []
        for i, param in enumerate(self.init_ff.params):
            if param.der1 != 0.0:
                if param.der2 > 0.00000001:
                    changes.append(- param.der1 / param.der2) # ideal
                else:
                    logger.warning('2nd derivative of {} is {}'.format(param, param.der2))
                    if param.der1 > 0.0:
                        change = -1.0
                    else:
                        change = 1.0
                    logger.warning('1st derivative of {} is {}. parameter change set to {}'.format(
                            param, param.der1, change))
                    changes.append(change)
            else:
                raise OptimizerException('1st derivative of {} is {}. skipping one dimensional newton-raphson'.format(
                        param, param.der1))
        return changes

    def calc_residual_vector(self, data_cal):
        residual_vector = np.empty((len(self.data_ref), 1), dtype=float)
        for i in xrange(0, len(self.data_ref)):
            residual_vector[i, 0] = self.data_ref[i].weight * (self.data_ref[i].value - data_cal[i].value)
        logger.log(8, 'created {} residual vector'.format(residual_vector.shape))
        return residual_vector

    def calc_svd(self, A, b, svd_thresholds=None):
        param_sets = []
        U, s, V = np.linalg.svd(A)
        s_copy = copy.deepcopy(s)
        if svd_thresholds:
            svd_thresholds = sorted(svd_thresholds)
            logger.log(8, 'svd thresholds: {}'.format(svd_thresholds))
            for threshold in svd_thresholds:
                for i in xrange(0, len(s_copy)):
                    if s_copy[i] < threshold:
                        s_copy[i] = 0.
                reform = U.dot(np.diag(s_copy)).dot(V)
                changes = self.solver(reform, b)
                try:
                    self.check_radius(changes)
                except RadiusException as e:
                    logger.warning(e.message)
                else:
                    param_sets.append(changes)
        else:
            for i in xrange(0, len(s_copy) - 1):
                s_copy[- (i + 1)] = 0.
                reform = U.dot(np.diag(s_copy)).dot(V)
                changes = self.solver(reform, b)
                try:
                    self.check_radius(changes)
                except RadiusException as e:
                    logger.warning(e.message)
                else:
                    param_sets.append(changes)
        return param_sets

    def check_radius(self, changes, cutoffs=None, max_radius=None):
        if cutoffs is None and max_radius is None:
            cutoffs = self.cutoffs
        radius = np.sqrt(sum([x**2 for x in changes]))
        if cutoffs:
            if radius > max(cutoffs):
                raise RadiusException('radius {} exceeded cutoff {}. excluding'.format(radius, max(cutoffs)))
            elif radius < min(cutoffs):
                raise RadiusException('radius {} below cutoff {}. excluding'.format(radius, min(cutoffs)))
        elif max_radius:
            if radius > max_radius:
                scale_factor = max_radius / radius
                changes = [x * scale_factor for x in changes]
                logger.warning('radius {} exceeded maximum {}. scaling parameters by {}'.format(
                        radius, max_radius, scale_factor))

    def parse(self, args):
        parser = self.return_optimizer_parser()
        group = parser.add_argument_group('gradient')
        group.add_argument('--no_basic', action='store_false')
        group.add_argument('--no_lagrange', action='store_false')
        group.add_argument('--no_levenberg', action='store_false')
        group.add_argument('--no_derivs', action='store_false')
        group.add_argument('--no_svd', action='store_false')
        opts = parser.parse_args(args)
        self.do_basic = opts.no_basic
        self.do_lagrange = opts.no_lagrange
        self.do_levenberg = opts.no_levenberg
        self.do_derivs = opts.no_derivs
        self.do_svd = opts.no_svd
        return opts

    def run(self):
        logger.info('--- running {} ---'.format(type(self).__name__))
        if self.init_ff.x2 is None or self.init_ff.data is None:
            self.calc_x2_ff(self.init_ff, save_data=True)

        self.ffs_central = self.params_diff(self.init_ff.params, mode='central')
        for ff in self.ffs_central:
            self.calc_x2_ff(ff, save_data=True)

        self.trial_ffs = []

        if self.do_derivs:
            self.central_diff_derivs(self.init_ff, self.ffs_central)
            try:
                changes = self.one_dim_newton(self.ffs_central)
                ff = FF()
                ff.method = 'one dimensional newton-raphson'
                ff.params = copy.deepcopy(self.init_ff.params)
                for param, change in zip(ff.params, changes):
                    param.value += change
            except OptimizerException as e:
                logger.warning(e.message)
            else:
                ff.display_params()
                try:
                    ff.check_params()
                    self.check_radius(changes)
                except UnallowedNegative as e:
                    logger.warning(e.message)
                except RadiusException as e:
                    logger.warning(e.message)
                else:
                    self.trial_ffs.append(ff)

        if self.do_basic or self.do_lagrange or self.do_levenberg or self.do_svd:
            residual_vector = self.calc_residual_vector(self.init_ff.data)
            jacobian = self.calc_jacobian(self.ffs_central)
            A = jacobian.T.dot(jacobian) # A = J.T J
            b = jacobian.T.dot(residual_vector) # b = J.T r

            if self.do_basic:
                changes = self.solver(A, b)
                ff = FF()
                ff.params = copy.deepcopy(self.init_ff.params)
                ff.method = 'basic'
                for param, change in zip(ff.params, changes):
                    param.value += change
                ff.display_params()
                try:
                    ff.check_params()
                    self.check_radius(changes)
                except UnallowedNegative as e:
                    logger.warning(e.message)
                except RadiusException as e:
                    logger.warning(e.message)
                else:
                    self.trial_ffs.append(ff)

            if self.do_lagrange:
                logger.log(8, 'lagrange factors: {}'.format(self.lagrange_factors))
                for factor in self.lagrange_factors:
                    changes = self.calc_lagrange(A, b, factor)
                    ff = FF()
                    ff.params = copy.deepcopy(self.init_ff.params)
                    ff.method = 'lagrange {}'.format(factor)
                    for param, change in zip(ff.params, changes):
                        param.value += change
                    ff.display_params()
                    try:
                        ff.check_params()
                        self.check_radius(changes)
                    except UnallowedNegative as e:
                        logger.warning(e.message)
                    except RadiusException as e:
                        logger.warning(e.message)
                    else:
                        self.trial_ffs.append(ff)

            if self.do_levenberg:
                logger.log(8, 'levenberg factors: {}'.format(self.levenberg_factors))
                for factor in self.levenberg_factors:
                            changes = self.calc_levenberg(A, b, factor)
                            ff = FF()
                            ff.params = copy.deepcopy(self.init_ff.params)
                            ff.method = 'levenberg {}'.format(factor)
                            for param, change in zip(ff.params, changes):
                                param.value += change
                            ff.display_params()
                            try:
                                ff.check_params()
                                self.check_radius(changes)
                            except UnallowedNegative as e:
                                logger.warning(e.message)
                            except RadiusException as e:
                                logger.warning(e.message)
                            else:
                                self.trial_ffs.append(ff)

            if self.do_svd:
                param_sets = self.calc_svd(A, b)
                for changes in param_sets:
                    ff = FF()
                    ff.params = copy.deepcopy(self.init_ff.params)
                    ff.method = 'svd'
                    for param, change in zip(ff.params, changes):
                        param.value += change
                    ff.display_params()
                    try:
                        ff.check_params()
                        self.check_radius(changes)
                    except UnallowedNegative as e:
                        logger.warning(e.message)
                    except RadiusException as e:
                        logger.warning(e.message)
                    else:
                        self.trial_ffs.append(ff)

        if len(self.trial_ffs) == 0:
            self.init_ff.export_ff()
            logger.warning('zero trial force fields generated')
            logger.info('--- {} complete ---'.format(type(self).__name__))
            logger.info('initial: {} ({})'.format(self.init_ff.x2, self.init_ff.method))
            logger.info('final: {} ({})'.format(self.init_ff.x2, self.init_ff.method))
            logger.info('no change from {}'.format(type(self).__name__))
            return self.init_ff
        logger.info('{} trial force fields generated'.format(len(self.trial_ffs)))
        for ff in self.trial_ffs:
            self.calc_x2_ff(ff)
        self.trial_ffs = sorted(self.trial_ffs, key=lambda x: x.x2)
        if self.trial_ffs[0].x2 < self.init_ff.x2:
            # may be nice if this block...
            self.best_ff = MM3()
            self.best_ff.method = self.trial_ffs[0].method
            self.best_ff.copy_attributes(self.init_ff)
            self.best_ff.params = copy.deepcopy(self.trial_ffs[0].params)
            self.best_ff.x2 = self.trial_ffs[0].x2
            if self.trial_ffs[0].data is not None:
                self.best_ff.data = self.trial_ffs[0].data
            # ... was a function
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
            logger.info('no change from {}'.format(type(self).__name__))
            return self.init_ff

    def solver(self, A, b, solver_method='lstsq'):
        if solver_method == 'cholesky':
            import scipy.linalg
            cho = scipy.linalg.cholesky(A, lower=True)
            changes = scipy.linalg.cho_solve((cho, True), b)
        elif solver_method == 'lstsq':
            changes, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=10**-12)
        elif solver_method == 'solve':
            changes = np.linalg.solve(A, b)
        changes = np.concatenate(changes).tolist()
        return changes

if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    
    gradient = Gradient()
    gradient.setup(sys.argv[1:])
    gradient.run()
