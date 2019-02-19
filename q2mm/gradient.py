from __future__ import absolute_import
from __future__ import division

import copy
import collections
import csv
import glob
import logging
import logging.config
import numpy as np
import os
import re
import sys

import calculate
import compare
import constants as co
import datatypes
import opt as opt
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
    lagrange_factors : list
                       Default is [0.01, 0.1, 1., 10.].
    lagrange_cutoffs : list or None
    lagrange_radii : list or None
                     Default is [0.1, 1., 5., 10.].
    levenberg_factors : list
                        Default is [0.01, 0.1, 1., 10.].
    levenberg_cutoffs : list or None
    levenberg_radii : list or None
                      Default is [0.1, 1., 5., 10.].
    newton_cutoffs : list or None
    newton_radii : list or None
                   Default is [0.1, 1., 5., 10.].
    svd_factors : list or None
                  Default is [0.001, 0.01, 0.1, 1.].
    svd_cutoffs : list of None
                  Default is [0.1, 10.].
    svd_radii : list or None
    """
    def __init__(self,
                 direc=None,
                 ff=None,
                 ff_lines=None,
                 args_ff=None,
                 args_ref=None):
        super(Gradient, self).__init__(
            direc, ff, ff_lines, args_ff, args_ref)
        # The current defaults err on the side of simplicity, so these are
        # likely more ideal for smaller parameter sets. For larger parameter
        # sets, central differentiation will take longer, and so you you will
        # likely want to try more trial FFs per iteration. This would mean
        # adding more max radii (ex. lagrange_radii) or more factors (ex.
        # svd_factors).

        # Whether or not to generate parameters with these methods.
        self.do_lstsq = False
        self.do_lagrange = True
        self.do_levenberg = False
        self.do_newton = True
        self.do_svd = False

        # Particular settings for each method.
        # LEAST SQUARES
        self.lstsq_cutoffs = None
        self.lstsq_radii = [1., 10.]
        # LAGRANGE
        self.lagrange_factors = [0.01, 0.1, 1., 10.]
        self.lagrange_cutoffs = None
        self.lagrange_radii = [0.1, 10.]
        # LEVENBERG-MARQUARDT
        self.levenberg_factors = [0.01, 0.1, 1., 10.]
        self.levenberg_cutoffs = None
        self.levenberg_radii = [0.1, 10.]
        # NEWTON-RAPHSON
        self.newton_cutoffs = None
        self.newton_radii = [1., 10.]
        # SVD
        self.svd_factors = [0.001, 0.01, 0.1, 1., 10.]
        self.svd_cutoffs = [0.1, 10.]
        self.svd_radii = None

    # Don't worry that self.ff isn't included in self.new_ffs.
    # opt.catch_run_errors will know what to do if self.new_ffs
    # is None.
    @property
    def best_ff(self):
        return sorted(self.new_ffs, key=lambda x: x.score)[0]

    @opt.catch_run_errors
    def run(self, ref_data=None, restart=None):
        """
        Runs the gradient optimization.

        Ensure that the attributes in __init__ are set as you desire before
        using this function.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """
        # We need reference data if you didn't provide it.
        if ref_data is None:
            ref_data = opt.return_ref_data(self.args_ref)

        # We need the initial FF data.
        if self.ff.data is None:
            logger.log(20, '~~ GATHERING INITIAL FF DATA ~~'.rjust(79, '~'))
            # Is opt.Optimizer.ff_lines used anymore?
            self.ff.export_ff()
            self.ff.data = calculate.main(self.args_ff)
            # Not 100% sure if this is necessary, but it certainly doesn't hurt.
            compare.correlate_energies(ref_data, self.ff.data)
        r_dict = compare.data_by_type(ref_data)
        c_dict = compare.data_by_type(self.ff.data)
        r_dict, c_dict = compare.trim_data(r_dict,c_dict)
        if self.ff.score is None:
            # Already zeroed reference and correlated the energies.
            self.ff.score = compare.compare_data(r_dict, c_dict)
        data_types = []
        for typ in r_dict:
            data_types.append(typ)
        data_types.sort()
        logger.log(20, '~~ GRADIENT OPTIMIZATION ~~'.rjust(79, '~'))
        logger.log(20, 'INIT FF SCORE: {}'.format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        logger.log(20, '~~ CENTRAL DIFFERENTIATION ~~'.rjust(79, '~'))
        if restart:
            par_file = restart
            logger.log(20, '  -- Restarting gradient from central '
                       'differentiation file {}.'.format(par_file))
        else:
            # We need a file to hold the differentiated parameter data.
            par_files = glob.glob(os.path.join(self.direc, 'par_diff_???.txt'))
            if par_files:
                par_files.sort()
                most_recent_par_file = par_files[-1]
                most_recent_par_file = most_recent_par_file.split('/')[-1]
                most_recent_num = most_recent_par_file[9:12]
                num = int(most_recent_num) + 1
                par_file = 'par_diff_{:03d}.txt'.format(num)
            else:
                par_file = 'par_diff_001.txt'
            logger.log(20, '  -- Generating central differentiation '
                       'file {}.'.format(par_file))
            f = open(os.path.join(self.direc, par_file), 'w')
            csv_writer = csv.writer(f)
            # Row 1 - Labels
            # Row 2 - Weights
            # Row 3 - Reference data values
            # Row 4 - Initial FF data values
            ## Deprecated -TR
            #csv_writer.writerow([x.lbl for x in ref_data])
            #csv_writer.writerow([x.wht for x in ref_data])
            #csv_writer.writerow([x.val for x in ref_data])
            #csv_writer.writerow([x.val for x in self.ff.data])
            writerows = [[],[],[],[]]
            for data_type in data_types:
                writerows[0].extend([x.lbl for x in r_dict[data_type]])
                writerows[1].extend([x.wht for x in r_dict[data_type]])
                writerows[2].extend([x.val for x in r_dict[data_type]])
                writerows[3].extend([x.val for x in c_dict[data_type]])
            for row in writerows:
                csv_writer.writerow(row)
            logger.log(20, '~~ DIFFERENTIATING PARAMETERS ~~'.rjust(79, '~'))
            # Save many FFs, each with their own parameter sets.
            ffs = opt.differentiate_ff(self.ff)
            logger.log(20, '~~ SCORING DIFFERENTIATED PARAMETERS ~~'.rjust(
                79, '~'))
            for ff in ffs:
                ff.export_ff(lines=self.ff.lines)
                logger.log(20, '  -- Calculating {}.'.format(ff))
                data = calculate.main(self.args_ff)
                # Deprecated
                #ff.score = compare.compare_data(ref_data, data)
                c_data = compare.data_by_type(data)
                r_dict, c_data = compare.trim_data(r_dict,c_data)
                ff.score = compare.compare_data(r_dict, c_data)
                opt.pretty_ff_results(ff)
                # Write the data rather than storing it in memory. For large
                # parameter sets, this could consume GBs of memory otherwise!
                #csv_writer.writerow([x.val for x in data])
                row = []
                for data_type in data_types:
                    row.extend([x.val for x in c_data[data_type]])
                csv_writer.writerow(row)
            f.close()

            # Make sure we have derivative information. Used for NR.
            #
            # The derivatives are useful for checking up on the progress of the
            # optimization and for deciding which parameters to use in a
            # subsequent simplex optimization.
            #
            # Still need a way to do this with the resatrt file.
            opt.param_derivs(self.ff, ffs)

        # Calculate the Jacobian, residual vector, matrix A and vector b.
        # These aren't needed if you're only doing Newton-Raphson.
        if self.do_lstsq or self.do_lagrange or self.do_levenberg or \
                self.do_svd:
            logger.log(20, '~~ JACOBIAN AND RESIDUAL VECTOR ~~'.rjust(79, '~'))
            # Setup the residual vector.
            # Deprecated - TR
            #num_d = len(ref_data)
            num_d = 0
            for datatype in r_dict:
                num_d += len(r_dict[datatype])
            resid = np.empty((num_d, 1), dtype=float)
            # Deprecated - TR
            #for i in xrange(0, num_d):
            #    resid[i, 0] = ref_data[i].wht * \
            #                  (ref_data[i].val - self.ff.data[i].val)
            count = 0
            for data_type in data_types:
                for r,c in zip(r_dict[typ],c_dict[typ]):
                    resid[count, 0] = r.wht * (r.val - c.val)
                    count += 1
            # logger.log(5, 'RESIDUAL VECTOR:\n{}'.format(resid))
            logger.log(
                20, '  -- Formed {} residual vector.'.format(resid.shape))
            # Setup the Jacobian.
            num_p = len(self.ff.params)
            # Maybe should be a part of the Jacobian function.
            jacob = np.empty((num_d, num_p), dtype=float)
            jacob = return_jacobian(jacob, os.path.join(self.direc, par_file))
            # logger.log(5, 'JACOBIAN:\n{}'.format(jacob))
            logger.log(20, '  -- Formed {} Jacobian.'.format(jacob.shape))
            ma = jacob.T.dot(jacob)
            vb = jacob.T.dot(resid)
            # We need these for most optimization methods.
            logger.log(5, ' MATRIX A AND VECTOR B '.center(79, '-'))
            # logger.log(5, 'A:\n{}'.format(ma))
            # logger.log(5, 'b:\n{}'.format(vb))
        # Start coming up with new parameter sets.
        if self.do_newton and not restart:
            logger.log(20, '~~ NEWTON-RAPHSON ~~'.rjust(79, '~'))
            # Moved the derivative section outside of here.
            changes = do_newton(self.ff.params,
                                radii=self.newton_radii,
                                cutoffs=self.newton_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)
        if self.do_lstsq:
            logger.log(20, '~~ LEAST SQUARES ~~'.rjust(79, '~'))
            changes = do_lstsq(ma, vb,
                               radii=self.lstsq_radii,
                               cutoffs=self.lstsq_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)
        if self.do_lagrange:
            logger.log(20, '~~ LAGRANGE ~~'.rjust(79, '~'))
            for factor in sorted(self.lagrange_factors):
                changes = do_lagrange(ma, vb, factor,
                                      radii=self.lagrange_radii,
                                      cutoffs=self.lagrange_cutoffs)
                cleanup(self.new_ffs, self.ff, changes)
        if self.do_levenberg:
            logger.log(20, '~~ LEVENBERG ~~'.rjust(79, '~'))
            for factor in sorted(self.levenberg_factors):
                changes = do_levenberg(ma, vb, factor,
                                       radii=self.levenberg_radii,
                                       cutoffs=self.levenberg_cutoffs)
                cleanup(self.new_ffs, self.ff, changes)
        if self.do_svd:
            logger.log(20, '~~ SINGULAR VALUE DECOMPOSITION ~~'.rjust(79, '~'))
            # J = U . s . VT
            mu, vs, mvt = return_svd(jacob)
            logger.log(1, '>>> mu.shape: {}'.format(mu.shape))
            logger.log(1, '>>> vs.shape: {}'.format(vs.shape))
            logger.log(1, '>>> mvt.shape: {}'.format(mvt.shape))
            logger.log(1, '>>> vb.shape: {}'.format(vb.shape))
            if self.svd_factors:
                changes = do_svd_w_thresholds(mu, vs, mvt, resid, self.svd_factors,
                                              radii=self.svd_radii,
                                              cutoffs=self.svd_cutoffs)
            else:
                changes = do_svd_wo_thresholds(mu, vs, mvt, resid,
                                               radii=self.svd_radii,
                                               cutoffs=self.svd_cutoffs)
            cleanup(self.new_ffs, self.ff, changes)
        # Report how many trial FFs were generated.
        logger.log(20, '  -- Generated {} trial force field(s).'.format(
                len(self.new_ffs)))
        # If there are any trials, test them.
        if self.new_ffs:
            logger.log(20, '~~ EVALUATING TRIAL FF(S) ~~'.rjust(79, '~'))
            for ff in self.new_ffs:
                data = opt.cal_ff(ff, self.args_ff, parent_ff=self.ff)
                # Shouldn't need to zero anymore.
                # Deprecated
                #ff.score = compare.compare_data(ref_data, data)
                c_data = compare.data_by_type(data)
                r_dict, c_data = compare.trim_data(r_dict,c_data)
                ff.score = compare.compare_data(r_dict, c_data)
                opt.pretty_ff_results(ff)
            self.new_ffs = sorted(
                self.new_ffs, key=lambda x: x.score)
            # Check for improvement.
            if self.new_ffs[0].score < self.ff.score:
                ff = self.new_ffs[0]
                logger.log(
                    20, '~~ GRADIENT FINISHED WITH IMPROVEMENTS ~~'.rjust(
                        79, '~'))
                opt.pretty_ff_results(self.ff, level=20)
                opt.pretty_ff_results(ff, level=20)
                # Copy parameter derivatives from original FF to save time in
                # case we move onto simplex immediately after this.
                copy_derivs(self.ff, ff)
            else:
                ff = self.ff
        else:
            ff = self.ff
        return ff

def copy_derivs(new_ff, old_ff):
    num_params = len(new_ff.params)
    assert num_params == len(old_ff.params)
    for i in range(0, num_params):
        new_ff.params[i].d1 = old_ff.params[i].d1
        new_ff.params[i].d2 = old_ff.params[i].d2
    logger.log(20, '  -- Copied parameter derivatives from {} to {}.'.format(
            old_ff, new_ff))

def check(changes, max_radii, cutoffs):
    new_changes = []
    for change in changes:
        radius = opt.calculate_radius(change[1])
        if max_radii:
            for max_radius in sorted(max_radii):
                scale_factor = check_radius(radius, max_radius)
                if scale_factor == 1:
                    new_changes.append(change)
                    break
                else:
                    new_changes.append(
                        (change[0] + ' R{}'.format(max_radius),
                         [x * scale_factor for x in change[1]]))
        elif cutoffs:
            if check_cutoffs(radius, cutoffs):
                new_changes.append(change)
        else:
            new_changes.append(change)
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

def cleanup(ffs, ff, changes):
    logger.log(1, '>>> changes: {}'.format(changes))
    if changes:
        for method, change in changes:
            opt.pretty_param_changes(
                ff.params, change, method)
            new_ff = return_ff(ff, change, method)
            if new_ff:
                ffs.append(new_ff)
    else:
        logger.warning(
            '  -- No changes generated! It may be wise to ensure this '
            'parameter has an effect on the objective function!')

def do_method(func):
    def wrapper(*args, **kwargs):
        # logger.log(1, '>>> args: {}'.format(args))
        # logger.log(1, '>>> kwargs: {}'.format(kwargs))
        try:
            changes = func(*args)
        except opt.OptError as e:
            logger.warning(e)
        else:
            changes = check(
                changes, kwargs['radii'], kwargs['cutoffs'])
            return changes
    return wrapper

@do_method
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
    logger.log(5, 'A:\n{}'.format(mac))
    mac[ind] = mac[ind] + factor
    logger.log(5, 'A:\n{}'.format(mac))
    changes = solver(mac, vb)
    return [('LAGRANGE F{}'.format(factor), changes)]

@do_method
def do_levenberg(ma, vb, factor):
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
    changes = solver(mac, vb)
    return [('LM {}'.format(factor), changes)]

@do_method
def do_lstsq(ma, vb):
    """
    Least-sqaures.

    Parameters
    ----------
    ma : NumPy matrix
    vb : NumPy vector
    """
    changes = solver(ma, vb)
    return [('LSTSQ', changes)]

@do_method
def do_newton(params):
    """
    Do a Newton-Raphson type parameter change prediction.

    Parameters
    ----------
    params : list of `datatypes.Param` (or subclass)
             The instances of `datatypes.Param` must already have
             their first and second derivative attributes populated.
    """
    changes = []
    for param in params:
        if param.d1 != 0.:
            if param.d2 > 0.00000001:
                changes.append(- param.d1 / param.d2)
            else:
                logger.warning('  -- 2nd derivative of {} is {:.4f}.'.format(
                        param, param.d2))
                logger.warning('  -- 1st derivative of {} is {:.4f}.'.format(
                        param, param.d1))
                if param.d1 > 0.:
                    changes.append(-1.)
                    logger.warning(
                        '  -- Change for {} set to -1.'.format(param))
                else:
                    changes.append(1.)
                    logger.warning(
                        '  -- Change for {} set to 1.'.format(param))
        else:
            raise opt.OptError(
                '1st derivative of {} is {}. Skipping Newton-Raphson.'.format(
                    param, param.d1))
    return [('NR', changes)]

@do_method
def do_svd_w_thresholds(mu, vs, mvt, resid, factors):
    logger.log(1, '>>> do_svd_w_thresholds <<<')
    factors.sort(reverse=True)
    all_changes = []

    # The largest values come 1st in the array vs.
    # When we invert it, the smallest values come 1st.
    vsi = invert_vector(vs)
    msi = np.diag(vsi)

    # For reduced SVD.
    # The largest values come 1st in this array.
    # ms = np.diag(vs)
    # When we invert it, the smallest values come 1st.
    # msi = np.linalg.inv(ms)

    logger.log(10, ' NO FACTORS '.center(79, '-'))
    logger.log(1, '>>> msi:\n{}'.format(msi))
    changes = mvt.T.dot(msi.dot(mu.T.dot(resid)))
    changes = changes.flatten()
    logger.log(1, '>>> changes:\n{}'.format(changes))

    for i, factor in enumerate(factors):
        logger.log(10, ' FACTOR: {} '.format(factor).center(79, '-'))
        old_msi = copy.deepcopy(msi)
        logger.log(1, '>>> msi:\n{}'.format(msi))
        logger.log(1, '>>> old_msi:\n{}'.format(old_msi))
        for i in range(0, len(vs)):
            if msi[i, i] > factor:
                msi[i, i] = 0.
        logger.log(1, '>>> msi:\n{}'.format(msi))
        logger.log(1, '>>> old_msi:\n{}'.format(old_msi))
        # Start checking after the first factor.
        # If there's no change in the vector, skip to next higher factor.
        if i != 0 and np.all(msi == old_msi):
            logger.warning('  -- No change with factor {}. Skipping.'.format(
                    factor))
            continue
        # If the vector is all zeros, quit.
        if np.all(msi == np.zeros(msi.shape)):
            logger.log(10, '  -- Vector is all zeros. Breaking.')
            break

        # Old code.
        # reform = mu.dot(np.diag(vs)).dot(mv)
        # logger.log(1, '>>> reform:\n{}'.format(reform))
        # changes = solver(reform, vb)
        # For full SVD.
        # m = mu.shape[0]
        # n = mv.shape[0]
        # ms = np.zeros((m, n), dtype=float)
        # ms[:n, :n] = np.diag(vs)
        # For reduced SVD.
        # ms = np.diag(vs)

        changes = mvt.T.dot(msi.dot(mu.T.dot(resid)))
        # We need this to be shaped like a list so transpose the parameter
        # changes.
        changes = changes.flatten()
        logger.log(1, '>>> changes:\n{}'.format(changes))
        all_changes.append(('SVD T{}'.format(factor), changes))
    logger.log(1, '>>> all_changes:\n{}'.format(all_changes))
    return all_changes

@do_method
def do_svd_wo_thresholds(mu, vs, mvt, resid):
    logger.log(1, '>>> do_svd_wo_thresholds <<<')
    all_changes = []
    logger.log(1, '>>> vs:\n{}'.format(vs))

    # The largest values come 1st in the array vs.
    # When we invert it, the smallest values come 1st.
    vsi = invert_vector(vs)
    msi = np.diag(vsi)

    # For reduced SVD.
    # The largest values come 1st in this array.
    # ms = np.diag(vs)
    # When we invert it, the smallest values come 1st.
    # msi = np.linalg.inv(ms)


    logger.log(10, ' ZEROED 0 ELEMENTS '.center(79, '-'))
    logger.log(1, '>>> msi:\n{}'.format(msi))
    changes = mvt.T.dot(msi.dot(mu.T.dot(resid)))
    changes = changes.flatten()
    logger.log(1, '>>> changes:\n{}'.format(changes))
    all_changes.append(('SVD Z0', changes))

    for i in range(0, len(vs) - 1):
        # Save a copy to check whether or not anything actually changes after
        # zeroing.
        old_msi = copy.deepcopy(msi)
        logger.log(10, ' ZEROED {} ELEMENTS '.format(i + 1).center(79, '-'))
        # We are zeroing the largest values.
        msi[-(i + 1), -(i + 1)] = 0.
        logger.log(1, '>>> msi:\n{}'.format(msi))
        if np.allclose(msi, old_msi):
            logger.warning('  -- No change with zeroing {} elements. '
                           'Skipping'.format(i + 1))
            continue

        # Old code.
        # reform = mu.dot(np.diag(vs)).dot(mv)
        # logger.log(1, '>>> reform:\n{}'.format(reform))
        # changes = solver(reform, vb)
        # For full SVD.
        # m = mu.shape[0]
        # n = mv.shape[0]
        # ms = np.zeros((m, n), dtype=float)
        # ms[:n, :n] = np.diag(vs)
        # For reduced SVD.
        # ms = np.diag(vs)

        changes = mvt.T.dot(msi.dot(mu.T.dot(resid)))
        # We need this to be shaped like a list so transpose the parameter
        # changes.
        changes = changes.flatten()
        logger.log(1, '>>> changes:\n{}'.format(changes))
        all_changes.append(('SVD Z{}'.format(i + 1), changes))

    logger.log(1, '>>> all_changes:\n{}'.format(all_changes))
    return all_changes

def invert_vector(vector, threshold=0.0001):
    """
    Inverts a vector. If the absolute value of an element in the vector is
    below the threshold, then it replaces the value with zero rather than
    inverting it.

    Arguments
    ---------
    vector : np.array
    threshold : float

    Returns
    -------
    np.array
    """
    new_vector = np.empty(vector.shape, dtype=float)
    for i, x in enumerate(vector):
        if abs(x) < threshold:
            new_x = 0.
        else:
            new_x = 1. / x
        new_vector[i] = new_x
    return new_vector

def return_ff(orig_ff, changes, method):
    """
    Returns None if ParamError is raised.
    """
    new_ff = orig_ff.__class__()
    new_ff.method = method
    new_ff.params = copy.deepcopy(orig_ff.params)
    try:
        update_params(new_ff.params, changes)
    except datatypes.ParamError as e:
        logger.warning(e)
    else:
        return new_ff

def return_jacobian(jacob, par_file):
    with open(par_file, 'r') as f:
        logger.log(15, 'READING: {}'.format(par_file))
        f.readline() # Labels.
        whts = [float(x) for x in f.readline().split(',')] # Weights.
        f.readline() # Reference values.
        f.readline() # Original values.
        # This is only for central differentiation.
        ff_ind = 0
        while True:
            l1 = f.readline()
            l2 = f.readline()
            if not l2:
                break
            inc_data = map(float, l1.split(','))
            dec_data = map(float, l2.split(','))
            for data_ind, (inc_datum, dec_datum) in \
                    enumerate(zip(inc_data, dec_data)):
                dydp = (inc_datum - dec_datum) / 2
                jacob[data_ind, ff_ind] = whts[data_ind] * dydp
            ff_ind += 1
    return jacob

def return_svd(matrix, check=False):
    """
    Parameters
    ----------
    matrix : NumPy matrix
    check : bool
    """
    # Reduced SVD.
    mu, vs, mvt = np.linalg.svd(matrix, full_matrices=False)
    if check:
        # Reform the matrix.
        reform = mu.dot(np.diag(vs).dot(mvt))
        same = np.allclose(matrix, reform)
        if not same:
            raise Exception(
                'Reformed matrix from SVD is not equivalent to original '
                'matrix!\nORIGINAL:\n{}\nREFORM:\n{}'.format(matrix, reform))
    # Full SVD.
    # mu, vs, mvt = np.linalg.svd(ma, full_matrices=True)
    # logger.log(1, ' SVD DECOMPOSITION '.center(79, '-'))
    # logger.log(1, 'U:\n{}'.format(mu))
    # logger.log(1, 's:\n{}'.format(vs))
    # logger.log(1, 'VT:\n{}'.format(mvt))
    return mu, vs, mvt

def solver(ma, vb):
    """
    Parameters
    ----------
    ma : NumPy matrix
    vb : NumPy vector
    """
    changes, resids, rank, singular_values = \
        np.linalg.lstsq(ma, vb, rcond=10**-12)
    return np.concatenate(changes).tolist()

def update_params(params, changes):
    """
    Takes the parameters and the unscaled changes, determines the properly
    scaled parameter changes, and increments the parameter values by them.

    Parameters
    ----------
    params : list of `datatypes.Param` (or subclass)
    changes : list of floats
                    Unscaled changes to the parameter values.
    """
    try:
        for param, change in zip(params, changes):
            param.value += change * param.step
    except datatypes.ParamError as e:
        logger.warning(e.message)
        raise

