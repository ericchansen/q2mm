"""
Simplex optimization.
"""
from __future__ import absolute_import
from __future__ import division

import copy
import collections
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
import opt
import parameters

logger = logging.getLogger(__name__)

class Simplex(opt.Optimizer):
    """
    Optimizes force field parameters using an in-house version of the simplex
    method. See `Optimizer` for repeated documentation.

    Attributes
    ----------
    _max_cycles_wo_change : int
                            End the simplex optimization early if there have
                            been this many consecutive simplex steps without
                            improvement in the objective function.
    do_massive_contraction : bool
                             If True, allows massive contractions to be
                             performed, contracting all parameters at once.
    do_weighted_reflection : bool
                             If True, weights parameter sets based on their
                             objective function score when determining the
                             reflection point.
    max_cycles : int
                 Maximum number of simplex cycles.

    max_params : int
                 Maximum number of parameters used in a single simplex cycle.
    """
    def __init__(self,
                 direc=None,
                 ff=None,
                 ff_lines=None,
                 args_ff=None,
                 args_ref=None):
        super(Simplex, self).__init__(
            direc, ff, ff_lines, args_ff, args_ref)
        self._max_cycles_wo_change = None
        self.do_massive_contraction = True
        self.do_weighted_reflection = True
        self.max_cycles = 100
        self.max_params = 3
    @property
    def best_ff(self):
        # Typically, self.new_ffs would include the original FF, self.ff,
        # but this can be changed by massive contractions.
        if self.new_ffs:
            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            # I think this is necessary after massive contraction.
            # Massive contraction can potentially make eveything worse.
            # No, it can't!!! The best FF is always retained! /Per-Ola
            # Yep, he's right. /Eric
            if self.new_ffs[0].score < self.ff.score:
                best_ff = self.new_ffs[0]
                best_ff = restore_simp_ff(best_ff, self.ff)
                return best_ff
            else:
                return self.ff
        else:
            return self.ff
    @opt.catch_run_errors
    def run(self, r_data=None):
        """
        Once all attributes are setup as you so desire, run this method to
        optimize the parameters.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """
        if r_data is None:
            r_data = opt.return_ref_data(self.args_ref)

        if self.ff.score is None:
            logger.log(20, '~~ CALCULATING INITIAL FF SCORE ~~'.rjust(79, '~'))
            self.ff.export_ff()
            # Could store data on self.ff.data if we wanted. Not necessary for
            # simplex. If simplex yielded no improvements, it would return this
            # FF, and then we might want the data such taht we don't have to
            # recalculate it in gradient. Let's hope simplex generally yields
            # improvements.
            data = calculate.main(self.args_ff)
            #deprecated
            #self.ff.score = compare.compare_data(r_data, data)
            r_dict = compare.data_by_type(r_data)
            c_dict = compare.data_by_type(data)
            r_dict, c_dict = compare.trim_data(r_dict,c_dict)
            self.ff.score = compare.compare_data(r_dict, c_dict)
        else:
            logger.log(20, '  -- Reused existing score and data for initial FF.')

        logger.log(20, '~~ SIMPLEX OPTIMIZATION ~~'.rjust(79, '~'))
        logger.log(20, 'INIT FF SCORE: {}'.format(self.ff.score))
        opt.pretty_ff_results(self.ff, level=20)

        # Here's what we do if there are too many parameters.
        if self.max_params and len(self.ff.params) > self.max_params:
            logger.log(20, '  -- More parameters than the maximum allowed.')
            logger.log(5, 'CURRENT PARAMS: {}'.format(len(self.ff.params)))
            logger.log(5, 'MAX PARAMS: {}'.format(self.max_params))
            # Here we select the parameters that have the lowest 2nd
            # derivatives.

            # Could fail when simplex finds improvements but restores other
            # parameters.
            # if self.ff.params[0].d1:

            if None in [x.d1 for x in self.ff.params]:
                logger.log(15, '  -- Calculating new parameter derivatives.')
                # Do central differentiation so we can calculate derivatives.
                # Another option would be to write code to determine
                # derivatives only from forward differentiation.
                ffs = opt.differentiate_ff(self.ff, central=True)
                # We have to score to get the derivatives.
                for ff in ffs:
                    ff.export_ff(path=self.ff.path, lines=self.ff_lines)
                    logger.log(20, '  -- Calculating {}.'.format(ff))
                    data = calculate.main(self.args_ff)
                    #deprecated
                    #ff.score = compare.compare_data(r_data, data)
                    r_dict = compare.data_by_type(r_data)
                    c_dict = compare.data_by_type(data)
                    r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                    ff.score = compare.compare_data(r_dict, c_dict)
                    opt.pretty_ff_results(ff)
                # Add the derivatives to your original FF.
                opt.param_derivs(self.ff, ffs)
                # Only keep the forward differentiated FFs.
                ffs = opt.extract_forward(ffs)
                logger.log(5, '  -- Keeping {} forward differentiated '
                           'FFs.'.format(len(ffs)))
            else:
                logger.log(15, '  -- Reusing existing parameter derivatives.')
                # Differentiate all parameters forward. Yes, I know this is
                # counter-intuitive because we are going to only use subset of
                # the forward differentiated FFs. However, this is very
                # computationally inexpensive because we're not scoring them
                # now. We will remove the forward differentiated FFs we don't
                # want before scoring.
                ffs = opt.differentiate_ff(self.ff, central=False)

            # This sorts the parameters based upon their 2nd derivative.
            # It keeps the ones with lowest 2nd derivatives.

            # SCHEDULED FOR CHANGES. NOT A GOOD SORTING CRITERION.
            params = select_simp_params_on_derivs(
                self.ff.params, max_params=self.max_params)
            # From the entire list of forward differentiated FFs, pick
            # out the ones that have the lowest 2nd derivatives.
            self.new_ffs = opt.extract_ff_by_params(ffs, params)
            logger.log(1, '>>> len(self.new_ffs): {}'.format(len(self.new_ffs)))

            # Reduce number of parameters.
            # Will need an option that's not MM3* specific in the future.
            ff_rows = [x.mm3_row for x in params]
            ff_cols = [x.mm3_col for x in params]
            for ff in self.new_ffs:
                new_params = []
                for param in ff.params:
                    if param.mm3_row in ff_rows and param.mm3_col in ff_cols:
                        new_params.append(param)
                ff.params = new_params
            # Make a copy of your original FF that has less parameters.
            ff_copy = copy.deepcopy(self.ff)
            new_params = []
            for param in ff.params:
                if param.mm3_row in ff_rows and param.mm3_col in ff_cols:
                    new_params.append(param)
            ff_copy.params = new_params
        else:
            # In this case it's simple. Just forward differentiate each
            # parameter.
            self.new_ffs = opt.differentiate_ff(self.ff, central=False)
            logger.log(1, '>>> len(self.new_ffs): {}'.format(len(self.new_ffs)))
            # Still make that FF copy.
            ff_copy = copy.deepcopy(self.ff)
        # Double check and make sure they're all scored.
        for ff in self.new_ffs:
            if ff.score is None:
                ff.export_ff(path=self.ff.path, lines=self.ff_lines)
                logger.log(20, '  -- Calculating {}.'.format(ff))
                data = calculate.main(self.args_ff)
                #deprecated
                #ff.score = compare.compare_data(r_data, data)
                r_dict = compare.data_by_type(r_data)
                c_dict = compare.data_by_type(data)
                r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                ff.score = compare.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(ff)

        # Add your copy of the orignal to FF to the forward differentiated FFs.
        self.new_ffs = sorted(self.new_ffs + [ff_copy], key=lambda x: x.score)
        # Allow 3 cycles w/o change for each parameter present. Remember that
        # the initial FF was added here, hence the minus one.
        self._max_cycles_wo_change = 3 * (len(self.new_ffs) - 1)
        wrapper = textwrap.TextWrapper(width=79)
        # Shows all FFs parameters.
        opt.pretty_ff_params(self.new_ffs)

        # Start the simplex cycles.
        current_cycle = 0
        cycles_wo_change = 0
        while current_cycle < self.max_cycles \
                and cycles_wo_change < self._max_cycles_wo_change:
            current_cycle += 1

            # Save the last best in case some accidental sort goes on.
            # Plus it makes reading the code a litle easier.
            last_best_ff = copy.deepcopy(self.new_ffs[0])
            logger.log(20, '~~ START SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))
            logger.log(20, 'ORDERED FF SCORES:')
            logger.log(20, wrapper.fill('{}'.format(
                    ' '.join('{:15.4f}'.format(x.score) for x in self.new_ffs))))

            inv_ff = self.ff.__class__()
            if self.do_weighted_reflection:
                inv_ff.method = 'WEIGHTED INVERSION'
            else:
                inv_ff.method = 'INVERSION'
            inv_ff.params = copy.deepcopy(last_best_ff.params)
            ref_ff = self.ff.__class__()
            ref_ff.method = 'REFLECTION'
            ref_ff.params = copy.deepcopy(last_best_ff.params)
            # Need score difference sum for weighted inversion.
            # Calculate this value before going into loop.
            if self.do_weighted_reflection:
                # If zero, should break.
                score_diff_sum = sum([x.score - self.new_ffs[-1].score
                                      for x in self.new_ffs[:-1]])
                if score_diff_sum == 0.:
                    logger.warning(
                        'No difference between force field scores. '
                        'Exiting simplex.')
                    # We want to raise opt.OptError such that
                    # opt.catch_run_errors will write the best FF obtained thus
                    # far.
                    raise opt.OptError(
                        'No difference between force field scores. '
                        'Exiting simplex.')
            for i in range(0, len(last_best_ff.params)):
                if self.do_weighted_reflection:
                    inv_val = (
                        sum([x.params[i].value *
                             (x.score - self.new_ffs[-1].score)
                             for x in self.new_ffs[:-1]])
                        / score_diff_sum)
                else:
                    inv_val = (
                        sum([x.params[i].value for x in self.new_ffs[:-1]])
                        /
                        len(self.new_ffs[:-1]))
                inv_ff.params[i].value = inv_val
                ref_ff.params[i].value = (
                    2 * inv_val - self.new_ffs[-1].params[i].value)
            # The inversion point does not need to be scored.
            # Calculate score for reflected parameters.
            ref_ff.export_ff(path=self.ff.path, lines=self.ff.lines)
            data = calculate.main(self.args_ff)
            #deprecated
            #ref_ff.score = compare.compare_data(r_data, data)
            r_dict = compare.data_by_type(r_data)
            c_dict = compare.data_by_type(data)
            r_dict, c_dict = compare.trim_data(r_dict,c_dict)
            ref_ff.score = compare.compare_data(r_dict, c_dict)
            opt.pretty_ff_results(ref_ff)
            if ref_ff.score < last_best_ff.score:
                logger.log(20, '~~ ATTEMPTING EXPANSION ~~'.rjust(79, '~'))
                exp_ff = self.ff.__class__()
                exp_ff.method = 'EXPANSION'
                exp_ff.params = copy.deepcopy(last_best_ff.params)
                for i in range(0, len(last_best_ff.params)):
                    exp_ff.params[i].value = (
                        3 * inv_ff.params[i].value -
                        2 * self.new_ffs[-1].params[i].value)
                exp_ff.export_ff(path=self.ff.path, lines=self.ff.lines)
                data = calculate.main(self.args_ff)
                #deprecated
                #exp_ff.score = compare.compare_data(r_data, data)
                r_dict = compare.data_by_type(r_data)
                c_dict = compare.data_by_type(data)
                r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                exp_ff.score = compare.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(exp_ff)
                if exp_ff.score < ref_ff.score:
                    self.new_ffs[-1] = exp_ff
                    logger.log(
                        20, '  -- Expansion succeeded. Keeping expanded '
                        'parameters.')
                else:
                    self.new_ffs[-1] = ref_ff
                    logger.log(
                        20, '  -- Expansion failed. Keeping reflected parameters.')
            elif ref_ff.score < self.new_ffs[-2].score:
                logger.log(20, '  -- Keeping reflected parameters.')
                self.new_ffs[-1] = ref_ff
            else:
                logger.log(20, '~~ ATTEMPTING CONTRACTION ~~'.rjust(79, '~'))
                con_ff = self.ff.__class__()
                con_ff.method = 'CONTRACTION'
                con_ff.params = copy.deepcopy(last_best_ff.params)
                for i in range(0, len(last_best_ff.params)):
                    if ref_ff.score > self.new_ffs[-1].score:
                        con_val = (
                            (inv_ff.params[i].value +
                             self.new_ffs[-1].params[i].value) / 2)
                    else:
                        con_val = (
                            (3 * inv_ff.params[i].value -
                             self.new_ffs[-1].params[i].value) / 2)
                    con_ff.params[i].value = con_val
                self.ff.export_ff(params=con_ff.params)
                data = calculate.main(self.args_ff)
                #deprecated
                #con_ff.score = compare.compare_data(r_data, data)
                r_dict = compare.data_by_type(r_data)
                c_dict = compare.data_by_type(data)
                r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                con_ff.score = compare.compare_data(r_dict, c_dict)
                opt.pretty_ff_results(con_ff)
                # This change was made to reflect the 1998 Q2MM publication.
                # if con_ff.score < self.new_ffs[-1].score:
                if con_ff.score < self.new_ffs[-2].score:
                    logger.log(20, '  -- Contraction succeeded.')
                    self.new_ffs[-1] = con_ff
                elif self.do_massive_contraction:
                    logger.log(
                        20, '~~ DOING MASSIVE CONTRACTION ~~'.rjust(79, '~'))
                    for ff_num, ff in enumerate(self.new_ffs[1:]):
                        for i in range(0, len(last_best_ff.params)):
                            ff.params[i].value = (
                                (ff.params[i].value +
                                 self.new_ffs[0].params[i].value) / 2)
                        self.ff.export_ff(params=ff.params)
                        data = calculate.main(self.args_ff)
                        #deprecated
                        #ff.score = compare.compare_data(r_data, data)
                        r_dict = compare.data_by_type(r_data)
                        c_dict = compare.data_by_type(data)
                        r_dict, c_dict = compare.trim_data(r_dict,c_dict)
                        ff.score = compare.compare_data(r_dict, c_dict)
                        ff.method += ' MC'
                        opt.pretty_ff_results(ff)
                else:
                    logger.log(
                        20, '  -- Contraction failed. Keeping parmaeters '
                        'anyway.')
                    self.new_ffs[-1] = con_ff
            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            # Keep track of the number of cycles without change. If there's
            # improvement, reset the counter.
            if self.new_ffs[0].score < last_best_ff.score:
                cycles_wo_change = 0
            else:
                cycles_wo_change += 1
                logger.log(20, '  -- {} cycles without improvement out of {} '
                           'allowed.'.format(
                        cycles_wo_change, self._max_cycles_wo_change))
            logger.log(20, 'BEST:')
            opt.pretty_ff_results(self.new_ffs[0], level=20)
            logger.log(20, '~~ END SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))

        # This sort is likely unnecessary because it should be done at the end
        # of the last loop cycle, but I put it here just in case.
        self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
        best_ff = self.new_ffs[0]
        if best_ff.score < self.ff.score:
            logger.log(20, '~~ SIMPLEX FINISHED WITH IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            best_ff = restore_simp_ff(best_ff, self.ff)
        else:
            logger.log(20, '~~ SIMPLEX FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            # This restores the inital parameters, so no need to use
            # restore_simp_ff here.
            best_ff = self.ff
        opt.pretty_ff_results(self.ff, level=20)
        opt.pretty_ff_results(best_ff, level=20)
        logger.log(20, '  -- Writing best force field from simplex.')
        best_ff.export_ff(best_ff.path)
        return best_ff

def calc_simp_var(params):
    """
    Simplex variable is calculated: (2nd der.) / (1st der.)**2
    """
    logger.log(1, '>>> params: {}'.format(params))
    logger.log(1, '>>> 1st ders.: {}'.format([x.d1 for x in params]))
    logger.log(1, '>>> 2nd ders.: {}'.format([x.d2 for x in params]))
    for param in params:
        param.simp_var = param.d2 / param.d1**2.

# Sorting based upon the 2nd derivative isn't such a good criterion. This should
# be updated soon.
def select_simp_params_on_derivs(params, max_params=10):
    """
    Sorts parameter sets from lowest to highest second
    derivatives of their score in the objective function.

    Parameters
    ----------
    params : list of subclasses of `datatypes.Param`
    """
    calc_simp_var(params)
    keep = sorted(params, key=lambda x: x.simp_var)
    logger.log(1, '>>> x.simp_var: {}'.format([x.simp_var for x in keep]))

    # Eliminate all where simp_var is greater than 1. This means that the
    # correct value is bracketed by the differentiation, so gradient
    # optimization should work.
    # keep = [x for x in keep if x.simp_var < 1.]

    # Old sorting method.
    # keep = sorted(params, key=lambda x: x.d2)

    keep = keep[:max_params]
    logger.log(1, '>>> x.simp_var: {}'.format([x.simp_var for x in keep]))
    logger.log(20, 'KEEPING PARAMS FOR SIMPLEX:\n{}'.format(
            ' '.join([str(x) for x in keep])))
    return keep

def restore_simp_ff(new_ff, old_ff):
    """
    The old FF has properties that we need to copy to the new FF. We also need
    to grab all the extra parameters included in old FF and add them to the new
    FF.
    """
    old_ff.copy_attributes(new_ff)
    if len(old_ff.params) > len(new_ff.params):
        logger.log(15, '  -- Restoring {} parameters to new FF.'.format(
                len(old_ff.params) - len(new_ff.params)))

        logger.log(1, '>>> old_ff.params:')
        logger.log(1, old_ff.params)
        logger.log(1, [x.d1 for x in old_ff.params])
        logger.log(1, [x.d2 for x in old_ff.params])
        opt.pretty_derivs(old_ff.params, level=1)
        logger.log(1, '>>> new_ff.params:')
        logger.log(1, new_ff.params)
        logger.log(1, [x.d1 for x in new_ff.params])
        logger.log(1, [x.d2 for x in new_ff.params])
        opt.pretty_derivs(new_ff.params, level=1)

        # Backup new parameters.
        new_params = copy.deepcopy(new_ff.params)
        # Copy over all old parameters.
        new_ff.params = copy.deepcopy(old_ff.params)
        # Replace the old with the new.
        for i, param_o in enumerate(old_ff.params):
            for param_n in new_params:
                # Should replace this with a general index scheme.
                if param_o.mm3_row == param_n.mm3_row and \
                        param_o.mm3_col == param_n.mm3_col:
                    new_ff.params[i] = copy.deepcopy(param_n)
    return new_ff
