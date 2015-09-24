#!/us/bin/python
"""
General code related to all optimization techniques.
"""
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
import opt
import parameters

logger = logging.getLogger(__name__)

class Gradient(opt.Optimizer):
    """
    Gradient based optimization methods (those dependent on derivatives of
    the penalty function). See `Optimizer` for repeated documentation.

    All cutoff attributes are a list of two positive floats. For new parameter
    sets, if the radius of unsigned parameter change doesn't lie between these
    floats, then that parameter set is ignored.

    All radii attributes are a list of many positive floats, as many as you'd
    like. This list is used to scale new parameter changes. For each new
    parameter set, it iterates through the list from lowest to highest values.
    If the radius of unsigned parameter change exceeds the given radius, then
    the parameter changes are scaled to match that radius. If the radius of
    unsigned parameter change is less than the given radius, the current
    parameter changes are applied without modification, and the remaining radii
    are not iterated through.

    Attributes
    ----------------
    do_basic : bool
    do_lagrange : bool
    do_levenberg : bool
    do_newton : bool
    do_svd : bool
    basic_cutoffs : list or None
    basic_radii : list or None
                  Default is [0.1, 1., 5., 10.].
    lagrange_cutoffs : list or None
    lagrange_factors : list
                       Default is [0.01, 0.1, 1., 10.].
    lagrange_radii : list or None
                     Default is [0.1, 1., 5., 10.].
    levenberg_cutoffs : list or None
    levenberg_factors : list
                        Default is [0.01, 0.1, 1., 10.].
    levenberg_radii : list or None
                      Default is [0.1, 1., 5., 10.].
    newton_cutoffs : list or None
    newton_radii : list or None
                   Default is [0.1, 1., 5., 10.].
    svd_cutoffs : list of None
                  Default is [0.1, 10.].
    svd_factors : list or None
                  Default is [0.001, 0.01, 0.1, 1.].
    svd_radii : list or None
    """
    def __init__(self,
                 ff=None, ff_lines=None, ff_args=None,
                 ref_args=None, ref_conn=None,
                 restore=False):
        super(Gradient, self).__init__(
            ff, ff_lines, ff_args, ref_args, ref_conn, restore)
        # Whether or not to generate parameters with these methods.
        self.do_lstsq = True
        self.do_lagrange = True
        self.do_levenberg = True
        self.do_newton = True
        self.do_svd = True
        # Particular settings for each method.
        self.lstsq_cutoffs = None
        self.lstsq_radii = [0.1, 1., 5., 10.]
        self.lagrange_cutoffs = None
        self.lagrange_factors = [0.01, 0.1, 1., 10.]
        # self.lagrange_factors = [0.01, 0.1, 1., 10.]
        self.lagrange_radii = [0.1, 1., 5., 10.]
        self.levenberg_cutoffs = None
        # self.levenberg_factors = [0.01, 0.1, 1., 10.]
        self.levenberg_factors = [0.01, 0.1, 1., 10.]
        self.levenberg_radii = [0.1, 1., 5., 10.]
        self.newton_cutoffs = None
        self.newton_radii = [0.1, 1., 5., 10.]
        self.svd_cutoffs = [0.1, 10.]
        # self.svd_factors = [0.001, 0.01, 0.1, 1.]
        self.svd_factors = None
        self.svd_radii = None
    def run(self):
        # Going to need this no matter what.
        if self.ff.conn is None:
            logger.log(20, '~~ GATHERING INITIAL FF DATA ~~'.rjust(79, '~'))
            # datatypes.export_ff(
            #     self.ff.path, self.ff.params, lines=self.ff.lines)
            self.ff.export_ff(self.ff.path)
            self.ff.conn = calculate.main(self.ff_args)
            compare.correlate_energies(self.ref_conn, self.ff.conn)
            self.ff.score = compare.calculate_score(
                self.ref_conn, self.ff.conn)
            opt.pretty_ff_results(self.ff)
        logger.log(20, '~~ GRADIENT OPTIMIZATION ~~'.rjust(79, '~'))
        logger.log(20, '~~ DIFFERENTIATING PARAMETERS ~~'.rjust(79, '~'))
        ffs = opt.differentiate_ff(self.ff)
        logger.log(20, '~~ SCORING DIFFERENTIATED PARAMETERS ~~'.rjust(79, '~'))
        opt.score_ffs(
            ffs, self.ff_args, self.ref_conn, parent_ff=self.ff,
            store_conn=True)
        if self.do_newton:
            logger.log(20, '~~ NEWTON-RAPHSON ~~'.rjust(79, '~'))
            opt.param_derivs(self.ff, ffs)
            try:
                changes = do_newton(self.ff.params)
            except opt.OptError as e:
                logger.warning(e)
            else:
                more_changes = do_checks(
                    changes, self.newton_radii, self.newton_cutoffs,
                    method='NR')
                for key, val in more_changes.iteritems():
                    opt.pretty_param_changes(
                        self.ff.params, val, method=key)
                self.new_ffs.extend(gen_ffs(self.ff, more_changes))
        if self.do_lstsq or self.do_lagrange or self.do_levenberg or \
                self.do_svd:
            logger.log(
                20, '~~ JACOBIAN AND RESIDUAL VECTOR ~~ '.rjust(79, '~'))
            c_vals = column_from_conn(self.ff.conn, 'val')
            r_vals, r_whts = column_from_conn(self.ref_conn, 'val', 'wht')
            resid = form_residual_vector(c_vals, r_vals, r_whts)
            diff_vals = [column_from_conn(x.conn, 'val') for x in ffs]
            jacob = form_jacobian(diff_vals, r_whts)
            ma = jacob.T.dot(jacob)
            vb = jacob.T.dot(resid)
            logger.log(5, ' MATRIX A AND VECTOR B '.center(79, '-'))
            logger.log(5, 'A:\n{}'.format(ma))
            logger.log(5, 'b:\n{}'.format(vb))
            if self.do_lstsq:
                logger.log(20, '~~ LEAST SQUARES ~~'.rjust(79, '~'))
                changes = do_lstsq(ma, vb)
                more_changes = do_checks(
                    changes, self.lstsq_radii, self.lstsq_cutoffs,
                    method='LSTSQ')
                for key, val in more_changes.iteritems():
                    opt.pretty_param_changes(
                        self.ff.params, val, method=key)
                self.new_ffs.extend(gen_ffs(self.ff, more_changes))
            if self.do_lagrange:
                logger.log(20, '~~ LAGRANGE ~~'.rjust(79, '~'))
                for factor in sorted(self.lagrange_factors):
                    logger.log(20, 'FACTOR: {}'.format(factor))
                    changes = do_lagrange(ma, vb, factor)
                    more_changes = do_checks(
                        changes, self.lagrange_radii, self.lagrange_cutoffs,
                        method='LAGRANGE F{}'.format(factor))
                    for key, val in more_changes.iteritems():
                        opt.pretty_param_changes(
                            self.ff.params, val, method=key)
                    self.new_ffs.extend(gen_ffs(self.ff, more_changes))
            if self.do_levenberg:
                logger.log(20, '~~ LEVENBERG-MARQUARDT ~~'.rjust(79, '~'))
                for factor in sorted(self.levenberg_factors):
                    logger.log(20, 'FACTOR: {}'.format(factor))
                    changes = do_levenberg(ma, vb, factor)
                    more_changes = do_checks(
                        changes, self.levenberg_radii, self.levenberg_cutoffs,
                        method='LM F{}'.format(factor))
                    for key, val in more_changes.iteritems():
                        opt.pretty_param_changes(
                            self.ff.params, val, method=key)
                    self.new_ffs.extend(gen_ffs(self.ff, more_changes))
            if self.do_svd:
                logger.log(
                    20, '~~ SINGULAR VALUE DECOMPOSITION ~~'.rjust(79, '~'))
                mu, vs, mv = do_svd(ma)
                if self.svd_factors:
                    for i, factor in enumerate(sorted(self.svd_factors)):
                        logger.log(
                            20, ' FACTOR {} '.format(factor).center(79, '-'))
                        if i != 0:
                            old_vs = new_vs
                        changes, new_vs = do_svd_thresholds(
                            mu, vs, mv, factor, vb)
                        if i != 0 and np.all(new_vs == old_vs):
                            logger.log(20,'  -- No change. Skipping.')
                            continue
                        more_changes = do_checks(
                            changes, self.svd_radii, self.svd_cutoffs,
                            method='SVD F{}'.format(factor))
                        for key, val in more_changes.iteritems():
                            opt.pretty_param_changes(
                                self.ff.params, val, method=key)
                        self.new_ffs.extend(gen_ffs(self.ff, more_changes))
                else:
                    for i in xrange(0, len(vs)):
                        logger.log(
                            20,
                            ' ZEROED {} ELEMENTS '.format(i).center(79, '-'))
                        changes = do_svd_wo_thresholds(mu, vs, mv, i, vb)
                        more_changes = do_checks(
                            changes, self.svd_radii, self.svd_cutoffs,
                            method='SVD Z{}'.format(i))
                        for key, val in more_changes.iteritems():
                            opt.pretty_param_changes(
                                self.ff.params, val, method=key)
                        self.new_ffs.extend(gen_ffs(self.ff, more_changes))
        logger.log(20, '  -- Generated {} trial force field(s).'.format(
                len(self.new_ffs)))
        if len(self.new_ffs) == 0:
            logger.log(
                20, '~~ GRADIENT FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            logger.log(20, '  -- Restoring original force field.')
            # datatypes.export_ff(
            #     self.ff.path, self.ff.params, lines=self.ff.lines)
            self.ff.export_ff(self.ff.path)
            return self.ff
        else:
            logger.log(20, '~~ EVALUATING TRIAL FF(S) ~~'.rjust(79, '~'))
            opt.score_ffs(
                self.new_ffs, self.ff_args, self.ref_conn, parent_ff=self.ff,
                restore=False)
            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            ff = self.new_ffs[0]
            if ff.score < self.ff.score:
                logger.log(20, '~~ GRADIENT FINISHED WITH IMPROVEMENTS ~~'.rjust(
                        79, '~'))
                opt.pretty_ff_results(self.ff, level=20)
                opt.pretty_ff_results(ff, level=20)
                self.ff.copy_attributes(ff)
                if self.restore:
                    logger.log(20, '  -- Restoring original force field.')
                    self.ff.export_ff(self.ff.path)
                    # datatypes.export_ff(
                    #     self.ff.path, self.ff.params, lines=self.ff.lines)
                else:
                    logger.log(20, '  -- Writing best force field from gradient.')
                    ff.export_ff(ff.path)
                    # datatypes.export_ff(
                    #     ff.path, ff.params, lines=ff.lines)
                return ff
            else:
                logger.log(20, '~~ GRADIENT FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                        79, '~'))
                opt.pretty_ff_results(self.ff, level=20)
                opt.pretty_ff_results(ff, level=20)
                logger.log(20, '  -- Restoring original force field.')
                self.ff.export_ff(self.ff.path)
                # datatypes.export_ff(
                #     self.ff.path, self.ff.params, lines=self.ff.lines)
                return self.ff

def mod_v_thresholds(v, f):
    x = np.copy(v)
    x[x < f] = 0
    return x

def do_svd_reform(u, s, v):
    a = u.dot(np.diag(s)).dot(v)
    logger.log('A:\n{}'.format(a))
    return a

def do_svd_thresholds(mu, vs, mv, factor, vb):
    """
    Reform original matrix after SVD with thresholds.

    Parameters
    ----------
    mu : NumPy matrix
    vs : NumPy vector
    mv : NumPy matrix
    factor : float
    vb : NumPy vector
    """
    for i in xrange(0, len(vs)):
        if vs[i] < factor:
            vs[i] = 0.
    logger.log(5, 's:\n{}'.format(vs))
    reform = mu.dot(np.diag(vs)).dot(mv)
    logger.log(5, 'A:\n{}'.format(reform))
    return do_lstsq(reform, vb), vs

def do_svd_wo_thresholds(mu, vs, mv, idx, vb):
    """
    Reform original matrix after SVD without thresholds.

    Parameters
    ----------
    mu : NumPy matrix
    vs : NumPy vector
    mv : NumPy matrix
    idx: int
    vb : NumPy vector
    """
    if idx:
        vs[-idx] = 0.
    logger.log(5, 's:\n{}'.format(vs))
    reform = mu.dot(np.diag(vs)).dot(mv)
    logger.log(5, 'A:\n{}'.format(reform))
    return do_lstsq(reform, vb)

def do_svd(ma):
    """
    SVD.

    Parameters
    ----------
    ma : NumPy matrix
    """
    mu, vs, mv = np.linalg.svd(ma)
    logger.log(5, ' MATRIX A DECOMPOSITION '.center(79, '-'))
    logger.log(5, 'U:\n{}'.format(mu))
    logger.log(5, 's:\n{}'.format(vs))
    logger.log(5, 'V:\n{}'.format(mv))
    return mu, vs, mv

def do_levenberg(ma, vb, factor):
    """
    Levenberg-Marquardt.

    Parameters
    ----------
    ma : NumPy matrix
    vb : NumPy vector
    factor : float
    """
    mac = copy.deepcopy(ma)
    ind = np.diag_indices_from(mac)
    mac[ind] = mac[ind] * (1 + factor)
    logger.log(5, 'A:\n{}'.format(mac))
    return do_lstsq(mac, vb)

def do_lagrange(ma, vb, factor):
    """
    Lagrange multipliers.

    Parameters
    ----------
    ma : NumPy matrix
    vb : NumPy vector
    factor : float
    """
    mac = copy.deepcopy(ma)
    ind = np.diag_indices_from(mac)
    mac[ind] = mac[ind] + factor
    logger.log(5, 'A:\n{}'.format(mac))
    return do_lstsq(mac, vb)

def do_lstsq(ma, vb):
    """
    Least-sqaures.

    Parameters
    ----------
    ma : NumPy matrix
    vb : NumPy vector
    """
    par_changes, resids, rank, singular_values = \
        np.linalg.lstsq(ma, vb, rcond=10**-12)
    par_changes = np.concatenate(par_changes).tolist()
    return par_changes

def form_jacobian(diff_vals, whts):
    """
    Forms the Jacobian. Dimensions are number of data points by number of
    parameters. Assumes central differentiation.

    Parameters
    ----------
    diff_vals : list of lists of floats
                Each inner list contains the floating point values of the
                data points for the central differentiated parameters.
                The order of the list goes first parameter incremented,
                first parameter decremented, second parameter incremented,
                etc.
    whts : list of floats
           Weights of the data points.
    """
    n_dat = len(diff_vals[0]) # Number of data points.
    n_par = len(diff_vals) / 2 # Number of parameters.
    jacob = np.empty((n_dat, n_par), dtype=float)
    for i, i_ff in enumerate(xrange(0, len(diff_vals), 2)):
        # i = 0, 1, 2, ...
        # i_ff = 0, 2, 4, ...
        # i_ff + 1 = 1, 3, 5, ...
        # Use i_ff to select the two related parameter sets, a single
        # parameter differentiated up and down.
        for i_data in xrange(0, n_dat):
            dydp = (diff_vals[i_ff][i_data] -
                    diff_vals[i_ff + 1][i_data]) / 2
            jacob[i_data, i] = whts[i_data] * dydp
    logger.log(5, 'JACOBIAN:\n{}'.format(jacob))
    logger.log(15, '  -- Formed {} Jacobian.'.format(jacob.shape))
    return jacob

def form_residual_vector(c_vals, r_vals, whts):
    """
    Forms the residual vector. Its length is the number of data points used
    in the optimization.

    Parameters
    ----------
    c_vals : list of floats
             Force field calculated data points.
    r_vals : list of floats
             Reference data points.
    whts : list of floats
           Weights for data points.
    """
    n = len(c_vals)
    r = np.empty((n, 1), dtype=float)
    for i in xrange(0, n):
        r[i, 0] = whts[i] * (r_vals[i] - c_vals[i])
    logger.log(5, 'RESIDUAL VECTOR:\n{}'.format(r))
    logger.log(15, '  -- Formed {} residual vector.'.format(r.shape))
    return r

def column_from_conn(conn, *cols):
    """
    Grabs columns of data from sqlite3 database connections.

    Parameters
    ----------
    conn : sqlite3 database connection
    cols : string
    """
    str_cols = ', '.join(cols)
    c = conn.cursor()
    c.execute('SELECT {} FROM data ORDER BY typ, src_1, src_2, idx_1, '
              'idx_2, atm_1, atm_2, atm_3, atm_4'.format(str_cols))
    rows = c.fetchall()
    if len(cols) > 1:
        return zip(*rows)
    else:
        return [x[0] for x in rows]

def gen_params(params, param_changes):
    """
    Takes the parameters and the unscaled changes, determines the properly
    scaled parameter changes, and increments the parameter values by them.

    Parameters
    ----------
    params : list of `datatypes.Param` (or subclass)
    param_changes : list of floats
                    Unscaled changes to the parameter values.
    """
    try:
        for param, param_change in itertools.izip(params, param_changes):
            param.value += param_change * param.step
    except datatypes.ParamError as e:
        logger.warning(e.message)
        raise

def gen_ffs(ff, param_changes_dictionary):
    """
    Wraps :func:`gen_params` to instead return force fields.
    """
    new_ffs = []
    for method, param_changes in param_changes_dictionary.iteritems():
        new_ff = ff.__class__()
        new_ff.method = method
        new_ff.params = copy.deepcopy(ff.params)
        try:
            gen_params(new_ff.params, param_changes)
        except datatypes.ParamError as e:
            logger.warning(e)
        else:
            new_ffs.append(new_ff)
    return new_ffs

def do_checks(par_changes, max_radii, cutoffs, method=None):
    new_changes = {}
    par_radius = opt.calculate_radius(par_changes)
    if max_radii:
        for max_radius in sorted(max_radii):
            scale_factor = check_radius(par_radius, max_radius)
            if method:
                name = '{} R{}'.format(method, max_radius)
            else:
                name = 'R{}'.format(max_radius)
            new_changes.update({name: [x * scale_factor for x in par_changes]})
            if scale_factor == 1:
                break
    elif cutoffs:
        if check_cutoffs(par_radius, cutoffs):
            if method:
                name = '{} C'.format(method)
            else:
                name = 'C'
            new_changes.update({name: par_changes})
    return new_changes

def check_cutoffs(par_rad, cutoffs):
    """
    Checks whether the radius of unscaled parameter changes lies
    within the cutoffs. If so, return True, else return False.

    Parameters
    ----------
    par_rad : float
              Radius of unscaled parameter changes.
    max_rad : float
              Maximum radius of unscaled parameter changes.
    """
    if min(cutoffs) <= par_rad <= max(cutoffs):
        return True
    else:
        logger.warning(
            "  -- Radius outside cutoffs ({} <= {} <= {}).".format(
                min(cutoffs), par_rad, max(cutoffs)))
        return False

def check_radius(par_rad, max_rad):
    """
    Checks whether the radius of unscaled parameter changes exceeds
    the maximum radius. If so, return the scaling factor.

    Parameters
    ----------
    par_rad : float
              Radius of unscaled parameter changes.
    max_rad : float
              Maximum radius of unscaled parameter changes.
    """
    if par_rad > max_rad:
        logger.warning(
            '  -- Radius of unscaled parameter changes ({}) exceeded '
            'max ({}).'.format(par_rad, max_rad))
        return max_rad / par_rad
    else:
        return 1

def do_newton(params):
    """
    Do a Newton-Raphson type parameter change prediction.

    Parameters
    ----------
    params : list of `datatypes.Param` (or subclass)
             The instances of `datatypes.Param` must already have
             their first and second derivative attributes populated.
    """
    # I would love more explanation about the logic in here that has
    # been carried down from the original code.
    par_changes = []
    for param in params:
        if param.d1 != 0.:
            if param.d2 > 0.00000001:
                par_changes.append(- param.d1 / param.d2)
            else:
                logger.warning('  -- 2nd derivative of {} is {:.4f}.'.format(
                        param, param.d2))
                logger.warning('  -- 1st derivative of {} is {:.4f}.'.format(
                        param, param.d1))
                if param.d1 > 0.:
                    par_changes.append(-1.)
                    logger.warning(
                        '  -- Change for {} set to -1.'.format(param))
                else:
                    par_changes.append(1.)
                    logger.warning(
                        '  -- Change for {} set to 1.'.format(param))
        else:
            raise opt.OptError(
                '1st derivative of {} is {}. Skipping Newton-Raphson.'.format(
                    param, param.d1))
    return par_changes

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
