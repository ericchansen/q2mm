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
    are not iterated through. If this is used with an optimization method, it
    can and probably will generate many new parameter sets for that single
    method.

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
        # Going to need this no matter what.
        if self.ff.conn is None:
            logger.log(20, '~~ GATHERING INITIAL FF DATA ~~'.rjust(79, '~'))
            datatypes.export_ff(
                self.ff.path, self.ff.params, lines=self.ff.lines)
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
                param_changes = do_newton(self.ff.params)
            except opt.OptError as e:
                logger.warning(e)
            else:
                param_changes_dictionary = do_checks(
                    param_changes, self.newton_radii, self.newton_cutoffs,
                    method='NR')
                self.new_ffs.extend(gen_ffs(self.ff, param_changes_dictionary))
        if self.do_lstsq or self.do_lagrange or self.do_levenberg or \
                self.do_svd:
            logger.log(
                20, '~~ JACOBIAN AND RESIDUAL VECTOR ~~ '.rjust(79, '~'))
            if self.do_lstsq:
                logger.log(20, '~~ LEAST SQUARES ~~'.rjust(79, '~'))
            if self.do_lagrange:
                logger.log(20, '~~ LAGRANGE ~~'.rjust(79, '~'))
            if self.do_levenberg:
                logger.log(20, '~~ LEVENBERG-MARQUARDT ~~'.rjust(79, '~'))
            if self.do_svd:
                logger.log(
                    20, '~~ SINGULAR VALUE DECOMPOSITION ~~'.rjust(79, '~'))
        logger.log(20, '  -- Generated {} trial force field(s).'.format(
                len(self.new_ffs)))
        opt.score_ffs(
            self.new_ffs, self.ff_args, self.ref_conn, parent_ff=self.ff,
            restore=False)
        logger.log(20, '~~ EVALUATING TRIAL FF(S) ~~'.rjust(79, '~'))
        self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
        ff = self.new_ffs[0]
        if ff.score < self.ff.score:
            self.ff.copy_attributes(ff)
            logger.log(20, '~~ GRADIENT FINISHED WITH IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            opt.pretty_ff_results(self.ff, level=20)
            opt.pretty_ff_results(ff, level=20)
        else:
            logger.log(20, '~~ GRADIENT FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                    79, '~'))
        if self.restore:
            logger.log(20, '  -- Restoring original force field.')
            datatypes.export_ff(
                self.ff.path, self.ff.params, lines=self.ff.lines)
        else:
            logger.log(20, '  -- Writing best force field from gradient.')
            datatypes.export_ff(
                ff.path, ff.params, lines=ff.lines)
        return ff

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
    print ff.method, param_changes_dictionary
    print ff.params
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
    par_radius = calculate_radius(par_changes)
    if max_radii:
        for max_radius in sorted(max_radii):
            scale_factor = check_radius(par_radius, max_radius)
            if method:
                name = '{} S{}'.format(method, max_radius)
            else:
                name = 'S{}'.format(max_radius)
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
    if max(cutoffs) <= par_rad <= min(cutoffs):
        return True
    else:
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
            '  -- Radius of unscaled parameter changes exceeded max.')
        return max_rad / par_rad
    else:
        return 1

def calculate_radius(par_changes):
    """
    Returns the radius of parameter changes.
    """
    return np.sqrt(sum([x**2 for x in par_changes]))

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

    import shutil
    shutil.copyfile('d_sulf/mm3.fld.bup', 'd_sulf/mm3.fld')
    INIT_FF_PATH = 'd_sulf/mm3.fld'
    REF_ARGS = (' -d d_sulf -je msa.01.mae msb.01.mae'.split())
    CAL_ARGS = (' -d d_sulf -me msa.01.mae msb.01.mae'.split())
    PARM_FILE = 'd_sulf/params.txt'

    logger.log(20, '~~ IMPORTING INITIAL FF ~~'.rjust(79, '~'))
    ff = datatypes.import_ff(INIT_FF_PATH)
    # ff.params = parameters.trim_params_by_file(ff.params, PARM_FILE)
    # use_these_params = ff.params[:3]
    use_these_params = ff.params[:2]
    ff.params = use_these_params

    grad = Gradient(ff=ff, ff_args=CAL_ARGS, ref_args=REF_ARGS)
    
    grad.do_lstsq = False
    grad.do_newton = True
    grad.do_lagrange = False
    grad.do_levenberg = False
    grad.do_svd = False

    grad.run()
    
