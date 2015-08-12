#!/us/bin/python
'''
General code related to all optimization techniques.
'''
import copy
import collections
import itertools
import logging
import logging.config
import numpy as np
import sqlite3

import calculate
import compare
import constants as co
import datatypes
import parameters

logger = logging.getLogger(__name__)

class OptimizerError(Exception):
    pass

class Optimizer(object):
    def __init__(self,
                 ff=None, ff_lines=None, ff_args=None,
                 ref_args=None, ref_conn=None,
                 restore=True):
        logger.log(20, '~~ {} SETUP ~~'.format(
                self.__class__.__name__.upper()).rjust(79, '~'))
        self.ff = ff
        self.ff_lines = ff_lines
        self.ff_args = ff_args
        self.new_ffs = []
        self.ref_args = ref_args
        self.ref_conn = ref_conn
        self.restore = restore
        self._ref_vals = None
        self._ref_whts = None
        assert self.ff, \
            'Must provided initial FF!'
        assert self.ff_args, \
            'Must provide arguments for calculating FF data!'
        assert self.ref_args or self.ref_conn, \
            'Must provide either arguments to determine reference data or a ' + \
            'connection to the reference database!'
        if self.ref_conn is None:
            logger.log(20, '~~ GATHERING REFERENCE DATA ~~'.rjust(79, '~'))
            self.ref_conn = calculate.main(self.ref_args)
            compare.zero_energies(self.ref_conn)
            compare.import_weights(self.ref_conn)
        c = self.ref_conn.cursor()
        rows = c.execute('SELECT val, wht FROM data ORDER BY typ, src_1, '
                         'src_2, idx_1, idx_2, atm_1, atm_2, atm_3, atm_4')
        self._ref_vals, self._ref_whts = zip(*rows)
        if self.ff_lines is None:
            with open(ff.path, 'r') as f:
                self.ff_lines = f.readlines()
            logger.log(5,  '  -- Read {} lines from {}.'.format(
                    len(self.ff_lines), ff.path))

class Gradient(Optimizer):
    def __init__(self,
                 ff=None, ff_lines=None, ff_args=None,
                 ref_args=None, ref_conn=None,
                 restore=True):
        super(Gradient, self).__init__(
            ff, ff_lines, ff_args, ref_args, ref_conn, restore)
        if self.ff.conn is None:
            logger.log(20, '~~ GATHERING INITIAL FF DATA ~~'.rjust(79, '~'))
            self.ff.conn = calculate.main(self.ff_args)
            compare.correlate_energies(self.ref_conn, self.ff.conn)
        # Whether or not to generate parameters with these methods.
        self.do_basic = True
        self.do_lagrange = True
        self.do_levenberg = True
        self.do_newton = True
        self.do_svd = True
        # Settings for each method.
        self.basic_cutoffs = None
        self.basic_radii = [0.1, 1., 5., 10.]
        self.lagrange_cutoffs = None
        self.lagrange_factors = [0.01, 0.1, 1., 10.]
        self.lagrange_radii = [0.1, 1., 5., 10.]
        self.levenberg_cutoffs = None
        self.levenberg_factors = [0.01, 0.1, 1., 10.]
        self.levenberg_radii = [0.1, 1., 5., 10.]
        self.newton_cutoffs = None
        self.newton_radii = [0.1, 1., 5., 10.]
        self.svd_cutoffs = [0.1, 10.]
        self.svd_factors = [0.001, 0.01, 0.1, 1.]
        self.svd_radii = None
    def run(self):
        logger.log(20, '~~ GRADIENT OPTIMIZATION ~~'.rjust(79, '~'))
        logger.log(20, '~~ DIFFERENTIATING PARAMETERS ~~'.rjust(79, '~'))
        diff_values = diff_params(self.ff.params)
        logger.log(20, '~~ SCORING DIFFERENTIATED PARAMETERS ~~'.rjust(79, '~'))
        diff_scores, diff_conns = score_diff_params(
            self.ff, self.ff_args, self.ff_lines, diff_values, self.ref_conn,
            self.ff.path)
        if self.do_newton:
            logger.log(20, '~~ NEWTON-RAPHSON ~~'.rjust(79, '~'))
            if self.ff.score is None:
                logger.log(20, '  -- Calculating score for initial FF.')
                self.ff.score = compare.calculate_score(self.ref_conn, self.ff.conn)
            d1s, d2s = cent_diff_derivs(self.ff.score, diff_scores)
            changes = calc_newton(self.ff.params, d1s, d2s)
            self.new_ffs.extend(radius_tests(
                    self.ff.params, changes, self.newton_radii,
                    self.newton_cutoffs, 'Newton-Raphson'))
        if self.do_basic or self.do_lagrange or self.do_levenberg or \
                self.do_svd:
            logger.log(20,
                       '~~ JACOBIAN AND RESIDUAL VECTOR ~~ '.rjust(79, '~'))
            vals = get_from_conn('val', self.ff.conn)
            resid = calc_resid(self._ref_vals, vals, self._ref_whts)
            ordered_data_sets = []
            for conn in diff_conns:
                ordered_data_sets.append(get_from_conn('val', conn))
            jacob = calc_jacob(vals, ordered_data_sets, self._ref_whts)
            logger.log(5, 'Residual vector:\n{}'.format(resid))
            logger.log(5, 'Jacobian:\n{}'.format(jacob))
            mat_a = jacob.T.dot(jacob)
            vec_b = jacob.T.dot(resid)
            logger.log(5, 'A:\n{}'.format(mat_a))
            logger.log(5, 'b:\n{}'.format(vec_b))
            if self.do_basic:
                logger.log(20, '~~ LEAST SQUARES ~~'.rjust(79, '~'))
                changes = calc_basic(mat_a, vec_b)
                self.new_ffs.extend(radius_tests(
                        self.ff.params, changes, self.basic_radii,
                        self.basic_cutoffs, 'Least Squares'))
            if self.do_lagrange:
                logger.log(20, '~~ LAGRANGE ~~'.rjust(79, '~'))
                for factor in sorted(self.lagrange_factors):
                    logger.log(20, 'Factor: {}'.format(factor))
                    changes = calc_lagrange(mat_a, vec_b, factor)
                    self.new_ffs.extend(radius_tests(
                            self.ff.params, changes, self.lagrange_radii,
                            self.lagrange_cutoffs,
                            'Lagrange {}'.format(factor)))
            if self.do_levenberg:
                logger.log(20, '~~ LEVENBERG-MARQUARDT ~~'.rjust(79, '~'))
                for factor in sorted(self.levenberg_factors):
                    logger.log(20, 'Factor: {}'.format(factor))
                    changes = calc_levenberg(mat_a, vec_b, factor)
                    self.new_ffs.extend(radius_tests(
                            self.ff.params, changes, self.levenberg_radii,
                            self.levenberg_cutoffs,
                            'Levenberg-Marquardt {}'.format(factor)))
            if self.do_svd:
                logger.log(20,
                           '~~ SINGULAR VALUE DECOMPOSITION ~~'.rjust(79, '~'))
                mat_u, vec_s, mat_v = calc_svd(mat_a)
                if self.svd_factors:
                    for factor in sorted(self.svd_factors):
                        logger.log(20, 'Factor: {}'.format(factor))
                        changes = calc_svd_w_thresholds(mat_u, vec_s, mat_v,
                                                        factor, vec_b)
                        self.new_ffs.extend(radius_tests(
                                self.ff.params, changes, self.svd_radii,
                                self.svd_cutoffs, 'SVD {}'.format(factor)))
                else:
                    for i in xrange(0, len(vec_s) - 1):
                        logger.log(20,
                                   '  -- Zeroed {} diagonal elements.'.format(
                                i+1))
                        changes = calc_svd_wo_thresholds(mat_u, vec_s, mat_v,
                                                         i, vec_b)
                        self.new_ffs.extend(radius_tests(
                                self.ff.params, changes, self.svd_radii,
                                self.svd_cutoffs, 'SVD {}'.format(i+1)))
        logger.log(20, '  -- Generated {} trial FFs.'.format(len(self.new_ffs)))
        logger.log(20, '~~ EVALUATING TRIAL FF(S) ~~'.rjust(79, '~'))
        stored_values = [x.value for x in self.ff.params]
        for ff in self.new_ffs:
            for param, new_param in itertools.izip(self.ff.params, ff.params):
                param.value = new_param
            datatypes.export_ff(
                self.ff.path, self.ff.params, lines=self.ff_lines)
            conn = calculate.main(self.ff_args)
            ff.score = compare.compare_data(self.ref_conn, conn)
            logger.log(20, '{}: {}'.format(ff.method, ff.score))
        self.new_ffs = sorted(self.new_ffs, key=sort_ff_key)
        logger.log(20, '  -- {} performed the best.'.format(
                self.new_ffs[0].method))
        if self.new_ffs[0].score < self.ff.score:
            ff = copy.deepcopy(self.ff)
            for param, new_param in itertools.izip(
                ff.params, self.new_ffs[0].params):
                param.value = new_param
                datatypes.export_ff(
                    ff.path, ff.params, lines=self.ff_lines)
        elif self.new_ffs[0] > self.ff.score or self.restore:
            logger.log(20, ' ~~ RESTORING INITIAL FF ~~'.rjust(79, '~'))
            for param, value in itertools.izip(self.ff.params, stored_values):
                param.value = value
            datatypes.export_ff(
                self.ff.path, self.ff.params, lines=self.ff_lines)
        return ff

def score_diff_params(ff_params, ff_args, ff_lines, diff_values, ref_conn,
                      ff_path,
                      full=False,
                      central=True):
    diff_scores = []
    diff_conns = []
    if full:
        stored_values = [x.value for x in ff_params]
    for i, diff_value in enumerate(diff_values):
        # Save original parameter value because we are going to overwrite it.
        if full:
            for param, new_value in itertools.izip(ff_params, diff_value):
                param.value = new_value
        else:
            stored_value = ff_params[param_idx].value
            ff_params[param_idx].value = diff_value
        if central:
            param_idx = int(np.floor(i/2))
        else:
            param_idx = i
        datatypes.export_ff(ff_path, ff_params, ff_lines)
        conn = calculate.main(ff_args)
        diff_conns.append(conn)
        score = compare.compare_data(ref_conn, conn)
        diff_scores.append(score)
        logger.info('{}: {}'.format(
                ff_params[param_idx], score))
        if full:
            for param, stored_value in itertools.izip(ff_params, stored_values):
                param.value = stored_value
        else:
            # Restore original parameter value.
            ff_params[param_idx].value = stored_value
    return diff_scores, diff_conns

def calc_svd(mat_a):
    mat_u, vec_s, mat_v = np.linalg.svd(mat_a)
    logger.log(5, 'U:\n{}'.format(mat_u))
    logger.log(5, 's:\n{}'.format(vec_s))
    logger.log(5, 'V:\n{}'.format(mat_v))
    return mat_u, vec_s, mat_v
    
def calc_svd_wo_thresholds(mat_u, vec_s, mat_v, idx, vec_b):
    vec_s[- (idx + 1)] = 0.
    reform = mat_u.dot(np.diag(vec_s)).dot(mat_v)
    return calc_basic(reform, vec_b)

def calc_svd_w_thresholds(mat_u, vec_s, mat_v, factor, vec_b):
    for i in xrange(0, len(vec_s)):
        if vec_s[i] < factor:
            vec_s[i] = 0.
    reform = mat_u.dot(np.diag(vec_s)).dot(mat_v)
    return calc_basic(reform, vec_b)

def calc_newton(params, d1s, d2s):
    changes = []
    for param, d1, d2 in itertools.izip(params, d1s, d2s):
        if d1 != 0.:
            if d2 > 0.00000001:
                # Ideal scenario.
                changes.append(-d1/d2)
            else:
                logger.warning('  -- 2nd derivative of {} is {:.4f}.'.format(
                        param, d2))
                logger.warning('  -- 1st derivative of {} is {:.4f}.'.format(
                        param, d1))
                if d1 > 0.:
                    change = -1.
                else:
                    change = 1.
                logger.warning('  -- Change for {} set to {:.4f}.'.format(
                        param, change))
                changes.append(change)
        else:
            raise OptimizerError(
                '1st derivative of {} is {}. Skipping Newton-Raphson.'.format(
                    param, d1))
    return changes

def cent_diff_derivs(init_score, diff_scores):
    d1s = []
    d2s = []
    for i in xrange(0, len(diff_scores), 2):
        # Equation 18 from Per-Ola's 1998 paper.
        d1s.append((diff_scores[i] - diff_scores[i+1]) * 0.5)
        # Equation 19 from Per-Ola's 1998 paper.
        d2s.append(diff_scores[i] + diff_scores[i+1] - 2 * init_score)
    return d1s, d2s

def radius_tests(params, changes, max_radii, cutoffs, method=None):
    ffs = []
    radius = calc_radius(changes)
    logger.log(20, 'Radius: {}'.format(radius))
    if max_radii:
        radius_less_than = None
        for max_radius in sorted(max_radii):
            scaling = check_radius(radius, max_radius)
            if scaling is None:
                radius_less_than = max_radius
                break
            else:
                logger.log(20,
                           '  -- Max radius {} exceeded. '
                           'Scaling by {:.4f}.'.format(max_radius, scaling))
                scaled_changes = [x * scaling for x in changes]
                values = increment_params(params, scaled_changes)
                if values is not None:
                    ffs.append(datatypes.FF(
                            method='{} {}'.format(method, max_radius),
                            params=values))
                    pretty_changes(params, scaled_changes, values)
        if radius_less_than is not None:
            logger.log(20,
                       "  -- Radius didn't exceed {}.".format(
                    radius_less_than))
            values = increment_params(params, changes)
            if values is not None:
                ffs.append(datatypes.FF(
                        method='{} {:8.4}'.format(method, radius),
                        params=values))
                pretty_changes(params, changes, values)
    elif cutoffs:
        inside_cutoffs = check_cutoffs(radius, cutoffs)
        if inside_cutoffs:
            values = increment_params(params, changes)
            if values is not None:
                ffs.append(datatypes.FF(method=method, params=values))
                pretty_changes(params, changes, values)
        else:
            logger.log(20, '  -- Radius not in bounds {} : {}'.format(
                    min(cutoffs), max(cutoffs)))
    else:
        values = increment_params(params, changes)
        if values is not None:
            ffs.append(datatypes.FF(method=method, params=values))
            pretty_changes(params, changes, values)
    return ffs
                
def increment_params(params, changes):
    assert len(params) == len(changes), \
        "Somehow the number of parameters and parameter changes aren't equal!"
    try:
        values = []
        for param, change in itertools.izip(params, changes):
            value = param.value + change * param.step_size
            param.check_value(val=value)
            values.append(value)
    except datatypes.ParameterError as e:
        logger.warning(e.message)
        return None
    else:
        return values

def pretty_changes(params, changes, values, level=15):
    if logger.getEffectiveLevel() <= level:
        logger.log(level,
                   '--' + ' Parameter '.ljust(12, '-') + 
                   '--' + ' Value '.center(10, '-') +
                   '--' + ' Change '.center(10, '-') +
                   '--' + ' New --'.rjust(10, '-'))
        for param, change, value in itertools.izip(params, changes, values):
            logger.log(
                level,
                '  ' + '{:>4d} {:>1d}'.format(
                    param.mm3_row, param.mm3_col).ljust(12, ' ') +
                '  ' + '{:>7.4f}'.format(param.value).center(10, ' ') +
                '  ' + '{:>7.4f}'.format(change).center(10, ' ') +
                '  ' + '{:>7.4f}   '.format(value).rjust(10, ' '))
        logger.log(level, '-'*50)

def check_cutoffs(radius, cutoffs):
    if not max(cutoffs) >= radius >= min(cutoffs):
        return False
    else:
        return True

def check_radius(radius, max_radius):
    if radius > max_radius:
        return max_radius / radius
    else:
        return None

def calc_radius(changes):
    return np.sqrt(sum([x**2 for x in changes]))

def calc_levenberg(mat_a, vec_b, factor):
    mat_a_copy = copy.deepcopy(mat_a)
    diag_idxs = np.diag_indices_from(mat_a_copy)
    mat_a_copy[diag_idxs] = mat_a_copy[diag_idxs] * (1 + factor)
    return calc_basic(mat_a_copy, vec_b)

def calc_lagrange(mat_a, vec_b, factor):
    mat_a_copy = copy.deepcopy(mat_a)
    diag_idxs = np.diag_indices_from(mat_a_copy)
    mat_a_copy[diag_idxs] = mat_a_copy[diag_idxs] + factor
    return calc_basic(mat_a_copy, vec_b)

def calc_basic(mat_a, vec_b):
    param_changes, residuals, rank, singular_values = \
        np.linalg.lstsq(mat_a, vec_b, rcond=10**-12)
    param_changes = np.concatenate(param_changes).tolist()
    return param_changes

def calc_jacob(init_vals, sets_diff_vals, whts):
    '''
    Compute the Jacobian matrix. Dimensions are number of data points x number
    of parameters.

    Assumes central differentiation. In the future, we may support forward
    differentiation as well.
    '''
    num_data_points = len(init_vals)
    # ASSUMES CENTRAL DIFFERENTIATION!
    num_params = len(sets_diff_vals) / 2
    jacob = np.empty((num_data_points, num_params), dtype=float)
    for i, ind_ff in enumerate(xrange(0, len(sets_diff_vals), 2)):
        # i = 0, 1, 2, ...
        # These two are used to select the two related parameter sets,
        # the ones where a single parameter is moved up and down.
        # ind_ff = 0, 2, 4, ...
        # ind_ff + 1 = 1, 3, 5, ...
        for ind_data in xrange(0, num_data_points):
            dydp = (sets_diff_vals[ind_ff][ind_data] -
                    sets_diff_vals[ind_ff + 1][ind_data]) / 2
            jacob[ind_data, i] = whts[ind_data] * dydp
    return jacob

def calc_resid(r_vals, c_vals, whts):
    '''
    Compute the residual vector. Length is equal to the number of data points
    used in the optimization.
    '''
    num_rows = len(r_vals)
    resid = np.empty((num_rows, 1), dtype=float)
    for i in xrange(0, num_rows):
        resid[i, 0] = whts[i] * (r_vals[i] - c_vals[i])
    return resid

def get_from_conn(column, conn):
    '''
    Returns a list of the desired values from an sqlite3 database.
    
    column - Column label in the sqlite3 database.
    conn   - Connection object to the sqlite3 database.
    '''
    c = conn.cursor()
    rows = c.execute('SELECT {} FROM data ORDER BY typ, src_1, src_2, '
                     'idx_1, idx_2, atm_1, atm_2, atm_3, atm_4'.format(column))
    return [x[column] for x in rows]

def diff_params(params, mode='central', full=False):
    '''
    Returns new parameters from central or forward differentiation.
    '''
    logger.log(20, '  -- {} differentiation on {} parameters.'.format(
            mode.title(), len(params)))
    assert mode == 'central' or mode == 'forward', \
        ('You tried to use the mode "{}" while differentiating parameters, but '
         'the variable "mode" must be "central" or "forward".'.format(mode))
    diff_values = []
    for i in xrange(0, len(params)):
        while True:
            try:
                inc_val = params[i].value + params[i].step_size
                params[i].check_value(inc_val)
                if mode == 'central':
                    dec_val = params[i].value - params[i].step_size
                    params[i].check_value(dec_val)
            except datatypes.ParameterError as e:
                logger.warning(e.message)
                assert not isinstance(params[i]._step_size, basestring), \
                    ('Parameter reached unallowed value in differentiation, but '
                     'this should be impossible because a scaling factor was '
                     'used as the step size!')
                new_step_size = params[i].value * 0.05
                logger.warning(
                    '  -- Changing step size of {} from {} to {}.'.format(
                        params[i], params[i]._step_size, new_step_size))
                params[i]._step_size = new_step_size
            else:
                if full:
                    copied_values = [x.value for x in params]
                    copied_values[i] = inc_val
                    diff_values.append(copied_values)
                    if mode == 'central':
                        copied_values = [x.value for x in params]
                        copied_values[i] = dec_val
                        diff_values.append(copied_values)
                else:
                    diff_values.append(inc_val)
                    if mode == 'central':
                        diff_values.append(dec_val)
                break
    logger.log(20, '  -- Generated {} FFs from {} differentiation.'.format(
            len(diff_values), mode))
    return diff_values

def sort_ff_key(ff):
    return ff.score

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)

    import shutil
    # shutil.copyfile('/pscratch/ehansen3/r_sulf/b049/mm3.fld', 'b053/mm3.fld')
    # INIT_FF_PATH = 'b053/mm3.fld'
    # CAL_ARGS = " -d b053 -meig dsa.01.mae,dsa.out dsb.01.mae,dsb.out dsc.01.mae,dsc.out dsd.01.mae,dsd.out dse.01.mae,dse.out dsf.01.mae,dsf.out msa.01.mae,msa.out msb.01.mae,msb.out msc.01.mae,msc.out msd.01.mae,msd.out mse.01.mae,mse.out -meo dsa.01.mae dsb.01.mae dsd.01.mae -meo msa.01.mae msb.01.mae -me2 dsc_scan-min.mae -me2 msa_scan-min.mae -mb dsa.01.mae dsb.01.mae dsd.01.mae msa.01.mae msb.01.mae -ma dsa.01.mae dsb.01.mae dsd.01.mae msa.01.mae msb.01.mae".split()
    # REF_ARGS = " -d b053 -jeige dsa.01.in,dsa.out dsb.01.in,dsb.out dsc.01.in,dsc.out dsd.01.in,dsd.out dse.01.in,dse.out dsf.01.in,dsf.out msa.01.in,msa.out msb.01.in,msb.out msc.01.in,msc.out msd.01.in,msd.out mse.01.in,mse.out -jeo dsa.01.mae dsb.01.mae dsd.01.mae -jeo msa.01.mae msb.01.mae -je2 dsc_scan-min.mae -je2 msa_scan-min.mae -jb dsa.01.mae dsb.01.mae dsd.01.mae msa.01.mae msb.01.mae -ja dsa.01.mae dsb.01.mae dsd.01.mae msa.01.mae msb.01.mae".split()
    # PARM_FILE = 'b053/params.txt'

    # shutil.copyfile('/pscratch/ehansen3/r_sulf/b000/mm3.fld', 'b049/mm3.fld')
    # INIT_FF_PATH = 'b049/mm3.fld'
    # REF_ARGS = (' -d b049 -jb msa.01.mae -jeo msa.01.mae msb.01.mae -je '
    #             'msa.01.mae msb.01.mae msc.01.mae'.split())
    # CAL_ARGS = (' -d b049 -mb msa.01.mae -meo msa.01.mae msb.01.mae -me '
    #             'msa.01.mae msb.01.mae msc.01.mae'.split())
    # REF_ARGS = (' -d b049 -jeo msa.01.mae msb.01.mae -je msa.01.mae '
    #             'msb.01.mae msc.01.mae msd.01.mae mse.01.mae'.split())
    # CAL_ARGS = (' -d b049 -meo msa.01.mae msb.01.mae -me msa.01.mae msb.01.mae '
    #             'msc.01.mae msd.01.mae mse.01.mae'.split())
    # REF_ARGS = (' -d b049 -je msa.01.mae msb.01.mae -jb dsa.01.mae -jeige dsa.01.in,dsa.out'.split())
    # CAL_ARGS = (' -d b049 -me msa.01.mae msb.01.mae -mb dsa.01.mae -meig dsa.01.mae,dsa.out'.split())
    # REF_ARGS = (' -d b049 -je msa.01.mae msb.01.mae'.split())
    # CAL_ARGS = (' -d b049 -me msa.01.mae msb.01.mae'.split())
    # PARM_FILE = 'b049/params.txt'

    shutil.copyfile('d_sulf/mm3.fld.bup', 'd_sulf/mm3.fld')
    INIT_FF_PATH = 'd_sulf/mm3.fld'
    REF_ARGS = (' -d d_sulf -je msa.01.mae msb.01.mae'.split())
    CAL_ARGS = (' -d d_sulf -me msa.01.mae msb.01.mae'.split())
    PARM_FILE = 'd_sulf/params.txt'

    logger.log(20, '~~ IMPORTING INITIAL FF ~~'.rjust(79, '~'))
    ff = datatypes.import_ff(INIT_FF_PATH)
    # ff.params = parameters.trim_params_by_file(ff.params, PARM_FILE)
    use_these_params = ff.params[:3]
    ff.params = use_these_params

    grad = Gradient(ff=ff, ff_args=CAL_ARGS, ref_args=REF_ARGS)
    
    grad.do_newton = False
    grad.do_lagrange = False
    grad.do_levenberg = False
    grad.do_svd = False


    grad.run()
    
    # simp = Simplex(ff=ff, ff_args=CAL_ARGS, ref_args=REF_ARGS)
    # simp.run()
