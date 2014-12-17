#!/usr/bin/python
'''
Gradient/Newton-Raphson based optimizer.
'''
from itertools import izip
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
        self.extra_print = False
        self.ffs_central = None
        self.solver_method = 'lstsq'

        self.basic = False
        self.basic_cutoffs = None
        self.basic_radii = None

        self.newton = False
        self.newton_cutoffs = None
        self.newton_radii = None

        self.lagrange = False
        self.lagrange_factors = [0.01, 0.1, 1.0, 10.0]
        self.lagrange_cutoffs = None
        self.lagrange_radii = None

        self.levenberg = False
        self.levenberg_factors = [0.01, 0.1, 1.0, 10.0]
        self.levenberg_cutoffs = None
        self.levenberg_radii = None

        self.svd = False
        # self.svd_factors = [0.001, 0.01, 0.1, 1.0]
        self.svd_factors = None
        self.svd_cutoffs = None
        self.svd_radii = None

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

    def calc_lagrange(self, mat_a, vec_b, factor):
        mat_a_copy = copy.deepcopy(mat_a)
        indices = np.diag_indices_from(mat_a_copy)
        mat_a_copy[indices] = mat_a_copy[indices] + factor
        param_changes = self.solver(mat_a_copy, vec_b)
        return param_changes

    def calc_levenberg(self, mat_a, vec_b, factor):
        mat_a_copy = copy.deepcopy(mat_a)
        indices = np.diag_indices_from(mat_a_copy)
        mat_a_copy[indices] = mat_a_copy[indices] * (1 + factor)
        param_changes = self.solver(mat_a_copy, vec_b)
        return param_changes

    def calc_one_dim_newton(self, central_ffs):
        param_changes = []
        for i, param in enumerate(self.init_ff.params):
            if param.der1 != 0.0:
                if param.der2 > 0.00000001:
                    param_changes.append(- param.der1 / param.der2) # ideal
                else:
                    logger.warning('2nd derivative of {} is {}'.format(param, param.der2))
                    if param.der1 > 0.0:
                        change = -1.0
                    else:
                        change = 1.0
                    logger.warning('1st derivative of {} is {}. parameter change set to {}'.format(
                            param, param.der1, change))
                    param_changes.append(change)
            else:
                raise OptimizerException('1st derivative of {} is {}. skipping one dimensional newton-raphson'.format(
                        param, param.der1))
        return param_changes

    def calc_residual_vector(self, data_cal):
        residual_vector = np.empty((len(self.data_ref), 1), dtype=float)
        for i in xrange(0, len(self.data_ref)):
            residual_vector[i, 0] = self.data_ref[i].weight * (self.data_ref[i].value - data_cal[i].value)
        logger.log(8, 'created {} residual vector'.format(residual_vector.shape))
        return residual_vector

    def calc_svd(self, mat_a, vec_b, svd_thresholds=None):
        param_sets = []
        methods = []
        U, s, V = np.linalg.svd(mat_a)
        s_copy = copy.deepcopy(s)
        if svd_thresholds:
            svd_thresholds = sorted(svd_thresholds)
            logger.log(8, 'svd thresholds: {}'.format(svd_thresholds))
            for threshold in svd_thresholds:
                for i in xrange(0, len(s_copy)):
                    if s_copy[i] < threshold:
                        s_copy[i] = 0.
                reform = U.dot(np.diag(s_copy)).dot(V)
                param_changes = self.solver(reform, vec_b)
                param_sets.append(param_changes)
                methods.append('singular value decomposition threshold {}'.format(threshold))
        else:
            for i in xrange(0, len(s_copy) - 1):
                s_copy[- (i + 1)] = 0.
                reform = U.dot(np.diag(s_copy)).dot(V)
                param_changes = self.solver(reform, vec_b)
                param_sets.append(param_changes)
                methods.append('singular value decomposition {}'.format(i + 1))
        return param_sets, methods

    def check_radius(self, param_changes, max_radius=None, cutoffs=None):
        radius = np.sqrt(sum([x**2 for x in param_changes]))
        logger.log(8, 'radius: {}'.format(radius))
        if max_radius:
            if radius > max_radius:
                scale_factor = max_radius / radius
                new_param_changes = [x * scale_factor for x in param_changes]
                logger.warning('radius {} exceeded maximum {}. scaled parameter changes by {}'.format(
                        radius, max_radius, scale_factor))
                return new_param_changes
            else:
                return param_changes
        elif cutoffs:
            if not max(cutoffs) >= radius >= min(cutoffs):
                raise RadiusException('radius {} not in bounds {} : {}. excluding parameter changes'.format(
                        radius, min(cutoffs), max(cutoffs)))

    def do_method(self, function, method, max_radii=None, cutoffs=None):
        try:
            param_changes = function
        except OptimizerException as e:
            logger.warning(e.message)
        else:
            if max_radii:
                for max_radius in max_radii:
                    new_param_changes = self.check_radius(param_changes, max_radius=max_radius)
                    ff = FF()
                    ff.method = '{} / radius {}'.format(method, max_radius)
                    ff.params = copy.deepcopy(self.init_ff.params)
                    for param, change in izip(ff.params, new_param_changes):
                        param.value += change
                    ff.display_params()
                    try:
                        ff.check_params()
                    except UnallowedNegative as e:
                        logger.warning(e.message)
                    else:
                        self.trial_ffs.append(ff)
            elif cutoffs:
                ff = FF()
                ff.method = method
                ff.params = copy.deepcopy(self.init_ff.params)
                for param, change in izip(ff.params, param_changes):
                    param.value += change
                ff.display_params()
                try:
                    self.check_radius(param_changes, cutoffs=cutoffs)
                    ff.check_params()
                except RadiusException as e:
                    logger.warning(e.message)
                except UnallowedNegative as e:
                    logger.warning(e.message)
                else:
                    self.trial_ffs.append(ff)
            else:
                ff = FF()
                ff.method = method
                ff.params = copy.deepcopy(self.init_ff.params)
                for param, change in izip(ff.params, param_changes):
                    param.value += change
                ff.display_params()
                try:
                    ff.check_params()
                except UnallowedNegative as e:
                    logger.warning(e.message)
                else:
                    self.trial_ffs.append(ff)

    def parse(self, args):
        parser = self.return_optimizer_parser()
        group_gen = parser.add_argument_group('gradient')
        group_gen.add_argument('--default', action='store_true')
        group_gen.add_argument('--extra_print', action='store_true')
        group_com = parser.add_argument_group('gradient methods')
        group_com.add_argument('--basic', '-b', nargs='?', const=True)
        group_com.add_argument('--newton', '-n', nargs='?', const=True)
        group_com.add_argument('--lagrange', '-la', nargs='*')
        group_com.add_argument('--levenberg', '-le', nargs='*')
        group_com.add_argument('--svd', '-s', nargs='*')
        opts = parser.parse_args(args)
        if opts.extra_print:
            self.extra_print = True
        if opts.default:
            self.basic = True
            self.basic_radii = [1, 3, 10]
            self.newton = True
            self.newton_radii = [1, 3, 10]
            # lagrange and levenberg factors set in __init__
            self.lagrange = True
            self.lagrange_radii = [5]
            self.levenberg = True
            self.lagrange_radii = [5]
            self.svd = True
            self.svd_cutoffs = [0.1, 10.0]
        else:
            if opts.newton is not None:
                self.newton = True
                if not isinstance(opts.newton, bool):
                    if opts.newton.startswith('r'):
                        self.newton_radii = sorted(map(float, opts.newton.strip('r').split(',')))
                    if opts.newton.startswith('c'):
                        self.newton_cutoffs = sorted(map(float, opts.newton.strip('c').split(',')))
            if opts.basic is not None:
                self.basic = True
                if not isinstance(opts.basic, bool):
                    if opts.basic.startswith('r'):
                        self.basic_radii = sorted(map(float, opts.basic.strip('r').split(',')))
                    if opts.basic.startswith('c'):
                        self.basic_cutoffs = sorted(map(float, opts.basic.strip('c').split(',')))

            if opts.lagrange is not None:
                self.do_lagrange = True
                for arg in opts.lagrange:
                    if isinstance(arg, basestring):
                        if arg.startswith('r'):
                            self.lagrange_radii = sorted(map(float, arg.strip('r').split(',')))
                        if arg.startswith('c'):
                            self.lagrange_cutoffs = sorted(map(float, arg.strip('c').split(',')))
                        if arg.startswith('f'):
                            self.lagrange_factors = sorted(map(float, arg.strip('f').split(',')))
            if opts.levenberg is not None:
                self.levenberg = True
                for arg in opts.levenberg:
                    if isinstance(arg, basestring):
                        if arg.startswith('r'):
                            self.levenberg_radii = sorted(map(float, arg.strip('r').split(',')))
                        if arg.startswith('c'):
                            self.levenberg_cutoffs = sorted(map(float, arg.strip('c').split(',')))
                        if arg.startswith('f'):
                            self.levenberg_factors = sorted(map(float, arg.strip('f').split(',')))
            if opts.svd is not None:
                self.svd = True
                for arg in opts.svd:
                    if isinstance(arg, basestring):
                        if arg.startswith('r'):
                            self.svd_radii = sorted(map(float, arg.strip('r').split(',')))
                        if arg.startswith('c'):
                            self.svd_cutoffs = sorted(map(float, arg.strip('c').split(',')))
                        if arg.startswith('f'):
                            self.svd_factors = sorted(map(float, arg.strip('f').split(',')))
        return opts

    def run(self):
        logger.info('--- running {} ---'.format(type(self).__name__))
        if self.init_ff.x2 is None or self.init_ff.data is None:
            self.calc_x2_ff(self.init_ff, save_data=True)

        self.ffs_central = self.params_diff(self.init_ff.params, mode='central')
        for ff in self.ffs_central:
            self.calc_x2_ff(ff, save_data=True)

        # warning: i wrote extra print stuff and related options after a few beers
        if self.extra_print:
            logger.info('generating partial par.tot from central differentiation')
            lines = []
            for ff in self.ffs_central:
                for i, datum in enumerate(ff.data):
                    try:
                        lines[i] += '\t{0:>10.4f}'.format(datum.value)
                    except IndexError:
                        lines.append('{0}\t{1:>10.4f}'.format(datum.name, datum.value))
            with open('par.tot', 'w') as f:
                for line in lines:
                    f.write(line + '\n')

        self.trial_ffs = []

        if self.newton:
            self.central_diff_derivs(self.init_ff, self.ffs_central)
            self.do_method(self.calc_one_dim_newton(self.ffs_central), 'one dimensional newton-raphson',
                           max_radii=self.newton_radii, cutoffs=self.newton_cutoffs)

        if self.basic or self.lagrange or self.levenberg or self.svd:
            residual_vector = self.calc_residual_vector(self.init_ff.data)
            jacobian = self.calc_jacobian(self.ffs_central)
            mat_a = jacobian.T.dot(jacobian) # A = J.T J
            vec_b = jacobian.T.dot(residual_vector) # b = J.T r

            if self.basic:
                self.do_method(self.solver(mat_a, vec_b), 'basic', max_radii=self.basic_radii, cutoffs=self.basic_cutoffs)
            if self.lagrange:
                logger.log(8, 'lagrange factors: {}'.format(self.lagrange_factors))
                for factor in self.lagrange_factors:
                    self.do_method(self.calc_lagrange(mat_a, vec_b, factor), 'lagrange {}'.format(factor),
                                   max_radii=self.lagrange_radii, cutoffs=self.lagrange_cutoffs)
            if self.levenberg:
                logger.log(8, 'levenberg factors: {}'.format(self.levenberg_factors))
                for factor in self.levenberg_factors:
                    self.do_method(self.calc_levenberg(mat_a, vec_b, factor), 'levenberg {}'.format(factor),
                                   max_radii=self.levenberg_radii, cutoffs=self.levenberg_cutoffs)
            if self.svd:
                param_change_sets, methods = self.calc_svd(mat_a, vec_b, svd_thresholds=self.svd_factors)
                for param_changes, method in izip(param_change_sets, methods):
                    self.do_method(param_changes, method, max_radii=self.svd_radii, cutoffs=self.svd_cutoffs)

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

    def solver(self, mat_a, vec_b, solver_method='lstsq'):
        if solver_method == 'cholesky':
            import scipy.linalg
            cho = scipy.linalg.cholesky(mat_a, lower=True)
            param_changes = scipy.linalg.cho_solve((cho, True), vec_b)
        elif solver_method == 'lstsq':
            param_changes, residuals, rank, singular_values = np.linalg.lstsq(mat_a, vec_b, rcond=10**-12)
        elif solver_method == 'solve':
            param_changes = np.linalg.solve(mat_a, vec_b)
        param_changes = np.concatenate(param_changes).tolist()
        return param_changes

if __name__ == '__main__':
    import logging.config
    import yaml
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
    
    gradient = Gradient()
    gradient.setup(sys.argv[1:])
    gradient.run()
