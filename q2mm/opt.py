"""
General code related to all optimization techniques.
"""
from __future__ import absolute_import
from __future__ import division

import copy
import collections
import logging
import logging.config
import numpy as np
import re
import textwrap
import sys

import calculate
import compare
import constants as co
import datatypes
import parameters

logger = logging.getLogger(__name__)

class OptError(Exception):
    """
    Raised when an optimizer does something bad.
    """
    pass

def catch_run_errors(func):
    def wrapper(*args, **kwargs):
        papa_bear = args[0]
        try:
            return func(*args, **kwargs)
        except (ZeroDivisionError, OptError, datatypes.ParamError) as e:
            logger.warning('opt.catch_run_errors caught an error!')
            logger.warning(e)
            if papa_bear.best_ff is None:
                logger.warning('Exiting {} and returning initial FF.'.format(
                        papa_bear.__class__.__name__.lower()))
                papa_bear.ff.export_ff(papa_bear.ff.path)
                return papa_bear.ff
            else:
                logger.warning('Exiting {} and returning best FF.'.format(
                        papa_bear.__class__.__name__.lower()))
                papa_bear.best_ff.export_ff(papa_bear.best_ff.path)
                return papa_bear.best_ff
    return wrapper

class Optimizer(object):
    """
    Base class for all optimization methods.

    Parameters
    ----------
    direc : string
            Path to directory where intermediate files with be stored.
    ff : instance of subclass of `datatypes.FF`
         Subclass for the force field used to start the optimization.
    ff_lines : list, optional
               Obtained from using readlines() on the opened force field. Will
               read these lines from `ff.path` if not provided.
    args_ff : list
              Arguments for `calculate.main` used to calculate the force field
              data set.
    args_ref : list
               Arguments for `calculate.main` used to calculate the reference
               data set.
    ref_data : list of `datatypes.Datum`
    restore : bool, optional
              If True, will write the initial force field after the optimization
              is complete. If False, will write the best force field from the
              optimization.

    Attributes
    ----------
    new_ffs : list
              Contains the new force field subclasses and parameters generated
              during the optimization.

    Returns
    -------
    instance of subclass of `datatypes.FF`
        Same subclass as the initial force field. Contains much of the same
        information, except the parameter values are the best ones obtained
        during the optimization.
    """
    def __init__(self,
                 direc=None,
                 ff=None,
                 ff_lines=None,
                 args_ff=None,
                 args_ref=None):
        logger.log(20, '~~ {} SETUP ~~'.format(
                self.__class__.__name__.upper()).rjust(79, '~'))
        self.direc = direc
        self.ff = ff
        self.ff_lines = ff_lines
        self.args_ff = args_ff
        self.args_ref = args_ref
        self.new_ffs = []
        # We should do away with the ff_lines attribute.
        # It's specific to MM3*. This sort of stuff should be
        # encapsulated in export_ff and import_ff type functions.
        if self.ff_lines is None and self.ff.lines:
            self.ff_lines = self.ff.lines

def return_ref_data(args_ref):
    logger.log(20, '~~ GATHERING REFERENCE DATA ~~'.rjust(79, '~'))
    ref_data = calculate.main(args_ref)
    compare.import_weights(ref_data)
    return ref_data

def calculate_radius(changes):
    """
    Returns the radius of parameter change.
    """
    return float(np.sqrt(sum([x**2 for x in changes])))

def differentiate_ff(ff, central=True):
    """
    Performs central or forward differentiation of parameters.

    For more description, see `differentiate_params`, which this is more
    or less a wrapper of. This just returns FF objects instead of lists
    of `datatypes.Param`.

    Parameters
    ----------
    ff : `datatypes.FF` (or subclass)
    central : bool

    Returns
    -------
    list of `datatypes.FF` (or subclass)
    """
    param_sets = differentiate_params(ff.params, central=central)
    ffs = []
    for i, param_set in enumerate(param_sets):
        new_ff = ff.__class__()
        new_ff.params = param_set
        new_ff.path = ff.path
        if central and i % 2 == 1:
            logger.log(1, '>>> i / i % 2: {} {}'.format(i, i % 2))
            new_ff.method = 'BACKWARD {}'.format(param_set[int(np.floor(i/2.))])
        else:
            logger.log(1, '>>> i / i % 2: {} {}'.format(i, i % 2))
            if central:
                new_ff.method = 'FORWARD {}'.format(param_set[int(np.floor(i/2.))])
            else:
                new_ff.method = 'FORWARD {}'.format(param_set[i])
        ffs.append(new_ff)
    return ffs

def differentiate_params(params, central=True):
    """
    Performs central or forward differentiation of parameters.

    Parameters
    ----------
    params : list of subclasses of `datatypes.Param`
    central : bool

    Returns
    -------
    list of lists of `datatypes.Param`
    """
    if central:
        logger.log(
            20, '~~ CENTRAL DIFFERENTIATION ON {} PARAMS ~~'.format(
                len(params)).rjust(79, '~'))
    else:
        logger.log(
            20, '~~ FORWARD DIFFERENTIATION ON {} PARAMS ~~'.format(
                len(params)).rjust(79, '~'))
    param_sets = []
    for i, param in enumerate(params):
        while True:
            original_value = param.value
            forward_params = copy.deepcopy(params)
            if central:
                backward_params = copy.deepcopy(params)
            try:
                forward_params[i].value = original_value + param.step
                if central:
                    backward_params[i].value = original_value - param.step
            except datatypes.ParamError as e:
                logger.warning(e.message)
                old_step = param.step
                # New parameter step size modification.
                # Should prevent problems with parameters trying to go
                # higher than their allowed range.
                upper_allowed_range = abs(
                    param.value - max(param.allowed_range))
                lower_allowed_range = abs(
                    param.value - min(param.allowed_range))
                min_allowed_change = min(
                    upper_allowed_range, lower_allowed_range)
                param.step = min_allowed_change * 0.1
                # This was the old method. It worked for preventing parameter
                # values from going below zero, but didn't work for much else.
                # param.step = param.value * 0.1
                logger.warning(
                    '  -- Changed step size of {} from {} to {}.'.format(
                        param, old_step, param.step))
            else:
                # Each of these lists contains one changed parameter. It'd
                # be nice to just use one list, modify the necessary parameter,
                # and then change it back before making the next modification.
                param_sets.append(forward_params)
                if central:
                    param_sets.append(backward_params)
                break
    logger.log(20, '  -- Generated {} differentiated parameter sets.'.format(
            len(param_sets)))
    return param_sets

# It stinks that these two extraction methods rely upon a force field's method
# string. It would be good to have this whole organization of forward and
# backward differentiated force fields be less arbitrary.
def extract_ff_by_params(ffs, params):
    """
    From the list of provided force fields, returns only those whose
    methods include one of the parameters in the provided list of
    parameters. The MM3* row and column, along with some regex, are
    used to do this.
    Parameters
    ----------
    ffs : list of `datatypes.FF` (or subclass)
    """
    rows = [x.mm3_row for x in params]
    cols = [x.mm3_col for x in params]
    keep = []
    for ff in ffs:
        row, col = [int(x) for x in re.split('\[|\]', ff.method)[3].split(',')]
        if row in rows and col in cols:
            keep.append(ff)
    logger.log(20, 'KEEPING FFS FOR SIMPLEX:\n{}'.format(
            ' '.join([str(x) for x in keep])))
    return keep

def extract_forward(ffs):
    """
    Returns the force fields that have been forward differentiated.

    Parameters
    ----------
    ffs : list of `datatypes.FF` (or subclass)
    """
    return [x for x in ffs if 'forward' in x.method.lower()]

def param_derivs(ff, ffs):
    """
    Calculates the derivatives of parameters with respect to the penalty
    function and stores them. Note that these derivatives are stored as
    unitless quantities, as this function doesn't account for the numerical
    step size.

    Parameters
    ----------
    ff : `datatypes.FF`
         The derivatives are stored in this force field's parameter
         instances, `datatypes.FF.params`.
    ffs : list of `datatypes.FF` (or subclass)
          Central differentiated and scored force fields.
    """
    for i in range(0, len(ffs), 2):
        logger.log(1, '>>> ffs[i].score: {}'.format(ffs[i].score))
        ff.params[int(i/2)].d1 = (ffs[i].score - ffs[i+1].score) * 0.5 # 18
        ff.params[int(i/2)].d2 = ffs[i].score + ffs[i+1].score - 2 * ff.score # 19
    pretty_derivs(ff.params)

def cal_ff(ff, ff_args, parent_ff=None, store_data=False):
    if ff.path:
        path = ff.path
    else:
        path = ff.path = parent_ff.path
    if ff.lines:
        # If I understand Python, then this creates a pointer, not a
        # copy of the object. Storing the lines from mm3.fld is sort of
        # big.
        lines = parent_ff.lines
    ff.export_ff(path, lines=lines)
    data = calculate.main(ff_args)
    if store_data:
        ff.data = data
    return data

def pretty_derivs(params, level=5):
    """
    Displays the parameter derivatives in a pretty fashion.

    Parameters
    ----------
    level : int
            Minimum logging level required for this to display.
    """
    if logger.getEffectiveLevel() <= level:
        logger.log(level,
                   '--' + ' Parameter '.ljust(33, '-') +
                   '--' + ' 1st der. '.center(19, '-') +
                   '--' + ' 2nd der. '.center(19, '-') +
                   '--')
        for param in params:
            # This try/except is used to catch instances where the 1st or 2nd
            # derivative is None.
            try:
                logger.log(level,
                           '  ' + '{}'.format(param).ljust(33, ' ') +
                           '  ' + '{:15.4f}'.format(param.d1).ljust(19, ' ') +
                           '  ' + '{:15.4f}'.format(param.d2).ljust(19, ' '))
            except ValueError:
                logger.log(level,
                           '  ' + '{}'.format(param).ljust(33, ' ') +
                           '  ' + 'None'.ljust(19, ' ') +
                           '  ' + 'None'.ljust(19, ' '))
        logger.log(level, '-' * 79)

def pretty_ff_params(ffs, level=20):
    """
    Shows parameters from many force fields.

    Parameters
    ----------
    ffs : list of `datatypes.FF` (or subclass)
    level : int
    """
    if logger.getEffectiveLevel() <= level:
        wrapper = textwrap.TextWrapper(width=79, subsequent_indent=' '*29)
        logger.log(
            level,
            '--' + ' PARAMETER '.ljust(25, '-') +
            '--' + ' VALUES '.ljust(48, '-') +
            '--')
        for i in range(0, len(ffs[0].params)):
            wrapper.initial_indent = ' {:25s} '.format(repr(ffs[0].params[i]))
            all_param_values = [x.params[i].value for x in ffs]
            all_param_values = ['{:8.4f}'.format(x) for x in all_param_values]
            logger.log(level, wrapper.fill(' '.join(all_param_values)))
        logger.log(level, '-' * 79)

def pretty_ff_results(ff, level=20):
    """
    Shows a force field's method, parameters, and score.

    Parameters
    ----------
    ff : `datatypes.FF` (or subclass)
    level : int
    """
    if logger.getEffectiveLevel() <= level:
        wrapper = textwrap.TextWrapper(width=79)
        logger.log(level, ' {} '.format(ff.method).center(79, '='))
        logger.log(level, 'SCORE: {}'.format(ff.score))
        logger.log(level, 'PARAMETERS:')
        logger.log(level, wrapper.fill(' '.join(map(str, ff.params))))
        logger.log(level, '=' * 79)
        logger.log(level, '')

def pretty_param_changes(params, changes, method=None, level=20):
    """
    Shows some parameter changes.
    """
    if logger.getEffectiveLevel() <= level:
        if method:
            logger.log(level, ' {} '.format(method).center(79, '='))
        else:
            logger.log(level, '=' * 79)
        logger.log(
            level,
            '--' + ' PARAMETER '.ljust(34, '-') +
            '--' + ' UNSCALED CHANGES '.center(19, '-') +
            '--' + ' CHANGES '.center(18, '-') +
            '--')
        # This calculation of the real parameter step size is also repeated.
        # The things I do for a good looking log.
        for param, change in zip(params, changes):
            logger.log(
                level,
                '  ' + '{}'.format(param).ljust(34, ' ') +
                '  ' + '{:7.4f}'.format(change).center(19, ' ') +
                '  ' + '{:7.4f}'.format(change * param.step).center(18, ' ') +
                '  ')
        # This was probably already calculated before, but I guess it's worth
        # doing again for simplicity and a nice looking log.
        r = calculate_radius(changes)
        logger.log(level, 'RADIUS: {}'.format(r))
        logger.log(level, '=' * 79)
        logger.log(level, '')
