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
import re
import sqlite3
import textwrap

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

class Optimizer(object):
    """
    Base class for all optimization methods.

    Parameters
    ----------
    ff : instance of subclass of `datatypes.FF`
         Subclass for the force field used to start the optimization.
    ff_lines : list, optional
               Obtained from using readlines() on the opened force field. Will
               read these lines from `ff.path` if not provided.
    ff_args : list
              Arguments for `calculate.main` used to calculate the force field
              data set.
    ref_args : list
               Arguments for `calculate.main` used to calculate the reference
               data set.
    ref_conn : sqlite3 database connection, optional
               Connection to a sqlite3 database stored in memory. Will calculate
               this data on its own using `ref_args` if not provided.
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
                 ff=None, ff_lines=None, ff_args=None,
                 ref_args=None, ref_conn=None,
                 restore=False):
        logger.log(20, '~~ {} SETUP ~~'.format(
                self.__class__.__name__.upper()).rjust(79, '~'))
        self.ff = ff
        self.ff_args = ff_args
        self.new_ffs = []
        self.ref_args = ref_args
        self.ref_conn = ref_conn
        self.restore = restore
        assert self.ff, \
            'Must provided initial FF!'
        assert self.ff_args, \
            'Must provide arguments for calculating FF data!'
        assert self.ref_args or self.ref_conn, \
            'Must provide either arguments to determine reference data or ' + \
            'a connection to the reference database!'
        if self.ref_conn is None:
            logger.log(20, '~~ GATHERING REFERENCE DATA ~~'.rjust(79, '~'))
            self.ref_conn = calculate.main(self.ref_args)
            compare.zero_energies(self.ref_conn)
            compare.import_weights(self.ref_conn)
    def eval(self):
        self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
        ff = self.new_ffs[0]
        if ff.score < self.ff.score:
            logger.log(20, '~~ {} FINISHED WITH IMPROVEMENTS ~~'.format(
                    self.__class__.__name__.upper()))
            self.ff.copy_attributes(ff)
    def clean_up(self):
        if hasattr(self, 'max_params') and self.max_params is not None and \
                len(self.ff.params)> self.max_params:
            pass

def pretty_ff_results(ff, level=15):
    """
    Shows a force field's method, parameters, and score.

    Parameters
    ----------
    ff : `datatypes.FF` (or subclass)
    level : int
    """
    if logger.getEffectiveLevel() <= level:
        wrapper = textwrap.TextWrapper(width=79)
        logger.log(level, ' {} '.format(ff.method).center(79, '-'))
        logger.log(level, 'SCORE: {}'.format(ff.score))
        logger.log(level, 'PARAMETERS:')
        logger.log(level, wrapper.fill(' '.join(map(str, ff.params))))
        logger.log(level, '-' * 79)
                
def pretty_ff_params(ffs, level=15):
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
            '--' + ' Parameter '.ljust(25, '-') +
            '--' + ' Values '.ljust(48, '-') +
            '--')
        for i in xrange(0, len(ffs[0].params)):
            wrapper.initial_indent = ' {:25s} '.format(repr(ffs[0].params[i]))
            all_param_values = [x.params[i].value for x in ffs]
            all_param_values = ['{:8.4f}'.format(x) for x in all_param_values]
            logger.log(level, wrapper.fill(' '.join(all_param_values)))
        logger.log(level, '-' * 79)
        
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
            logger.log(level,
                       '  ' + '{}'.format(param).ljust(33, ' ') +
                       '  ' + '{:15.4f}'.format(param.d1).ljust(19, ' ') + 
                       '  ' + '{:15.4f}'.format(param.d2).ljust(19, ' '))
        logger.log(level, '-' * 79)

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
        row, col = map(int, re.split('\[|\]', ff.method)[3].split(','))
        if row in rows and col in cols:
            keep.append(ff)
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
    for i in xrange(0, len(ffs), 2):
        ff.params[i/2].d1 = (ffs[i].score - ffs[i+1].score) * 0.5 # 18
        ff.params[i/2].d2 = ffs[i].score + ffs[i+1].score - 2 * ff.score # 19
    pretty_derivs(ff.params)

def score_ffs(ffs, ff_args, r_conn, parent_ff=None, restore=True,
              store_conn=False):
    """
    Score many force fields. Returns nothing but modifies attributes on
    the input force fields.

    Parameters
    ----------
    ffs : list of `datatypes.FF` (or subclass)
    ff_args : string
              Arguments for `calculate.main`.
    r_conn : connection to reference database
    parent_ff : None or instance of `datatypes.FF` (or subclass)
                If the force fields in `ffs` are missing attributes (set to
                None), then it uses the corresponding attributes in
                `parent_ff`.
    restore : bool
              If True, restores the parameters in `parent_ff.params` to
              `parent_ff.path`.
    store_conn : bool
                 If True, stores the calculated data for use later in the
                 force field's conn attribute, `datatypes.FF.conn`.
    """
    logger.log(15, '  -- Scoring {} force fields.'.format(len(ffs)))
    for ff in ffs:
        print ff.method
        print ff.params
        if ff.path is None:
            path = parent_ff.path
        else:
            path = ff.path
        if not hasattr(ff, 'lines') or ff.lines is None:
            lines = parent_ff.lines
        else:
            lines = ff.lines
        datatypes.export_ff(path, ff.params, lines=lines)
        conn = calculate.main(ff_args)
        ff.score = compare.compare_data(r_conn, conn)
        logger.log(15, '{}: {}'.format(ff.method, ff.score))
        if store_conn:
            ff.conn = conn
    if restore:
        datatypes.export_ff(
            parent_ff.path, parent_ff.params, lines=parent_ff.lines)

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
        ff = ff.__class__()
        ff.params = param_set
        if central and i % 2 == 1:
            ff.method = 'BACKWARD {}'.format(param_set[int(np.floor(i/2.))])
        else:
            ff.method = 'FORWARD {}'.format(param_set[int(np.floor(i/2.))])
        ffs.append(ff)
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
            try:
                forward_val = param.value + param.step
                if central:
                    backward_val = param.value - param.step
            except datatypes.ParamError as e:
                logger.warning(e.message)
                old_step = param.step
                param.step = param.value * 0.05
                logger.warning(
                    '  -- Changed step size of {} from {} to {}.'.format(
                        param, old_step, param.step))
            else:
                # So much wasted time and memory.
                forward_params = copy.deepcopy(params)
                forward_params[i].value = forward_val
                param_sets.append(forward_params)
                if central:
                    backward_params = copy.deepcopy(params)
                    backward_params[i].value = backward_val
                    param_sets.append(backward_params)
                break
    logger.log(15, '  -- Generated {} differentiated parameter sets.'.format(
            len(param_sets)))
    return param_sets

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
    use_these_params = ff.params[:3]
    # use_these_params = ff.params[:8]
    ff.params = use_these_params

    simp = Simplex(ff=ff, ff_args=CAL_ARGS, ref_args=REF_ARGS)
    simp.max_params = 2
    # simp.max_params = 8

    simp.run()
    
