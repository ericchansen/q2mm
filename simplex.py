#!/usr/bin/python
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
        self.max_params = 10
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
        logger.log(20, '~~ SIMPLEX OPTIMIZATION ~~'.rjust(79, '~'))
        # Here we don't actually need the database connection/force field data.
        # We only need the score.
        if self.ff.score is None:
            logger.log(20, '~~ CALCULATING INITIAL FF SCORE ~~'.rjust(79, '~'))
            self.ff.export_ff()
            # I could store this object to prevent on self.ff to prevent garbage
            # collection. Would be nice if simplex was followed by gradient,
            # which needs that information, and if simplex yielded no
            # improvements. At most points in the optimization, this is probably
            # too infrequent for it to be worth the memory, but it might be nice
            # once the parameters are close to convergence.
            data = calculate.main(self.args_ff)
            self.ff.score = compare.compare_data(r_data, data)
            logger.log(20, 'INITIAL FF SCORE: {}'.format(self.ff.score))
        else:
            logger.log(15, '  -- Reused existing score and data for initial FF.')
        logger.log(15, 'INIT FF SCORE: {}'.format(self.ff.score))
        if self.max_params and len(self.ff.params) > self.max_params:
            if self.ff.params[0].d1:
                logger.log(15, '  -- Reusing existing parameter derivatives.')
                # Don't score so this really doesn't take much time. We're just
                # generating the forward differentiated parameter set for all
                # the parameters.
                ffs = opt.differentiate_ff(self.ff, central=False)
            else:
                logger.log(15, '  -- Calculating new parameter derivatives.')
                # Do central differentiation so we can calculate derivatives.
                # Another option would be to write code to determine
                # derivatives only from forward differentiation. The accuracy
                # vs. efficiency is probably if we did that.
                ffs = opt.differentiate_ff(self.ff, central=True)
                # We have to score to get the derivatives.
                for ff in ffs:
                    # Score the FFs.
                    ff.export_ff(lines=self.ff_lines)
                    logger.log(20, '  -- Calculating {}.'.format(ff))
                    data = calculate.main(self.args_ff)
                    ff.score = compare.compare_data(r_data, data)
                    opt.pretty_ff_results(ff)
                # Add the derivatives to your original FF.
                opt.param_derivs(self.ff, ffs)
                # Only keep the forward differentiated FFs.
                ffs = opt.extract_forward(ffs)
            # This sorts the parameters based upon their 2nd derivative.
            # It keeps the ones with lowest 2nd derivatives.
            params = select_simp_params_on_derivs(
                self.ff.params, max_params=self.max_params)
            # From the entire list of forward differentiated FFs, pick
            # out the ones that have the lowest 2nd derivatives.
            self.new_ffs = opt.extract_ff_by_params(ffs, params)
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
            # Still make that FF copy.
            ff_copy = copy.deepcopy(self.ff)
        # Double check and make sure they're all scored.
        for ff in self.new_ffs:
            if ff.score is None:
                ff.export_ff(lines=self.ff_lines)
                logger.log(20, '  -- Calculating {}.'.format(ff))
                data = calculate.main(self.args_ff)
                ff.score = compare.compare_data(r_data, data)
                opt.pretty_ff_results(ff)
        # Add your copy of the orignal to FF to the forward differentiated FFs.
        self.new_ffs = sorted(self.new_ffs + [ff_copy], key=lambda x: x.score)
        # Allow 3 cycles w/o change for each parameter present
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
            last_best = self.new_ffs[0].score
            best_ff = self.new_ffs[0]
            logger.log(20, '~~ START SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))
        logger.log(20, 'ORDERED FF SCORES:')
        logger.log(20, wrapper.fill('{}'.format(' '.join('{:15.4f}'.format(
                            x.score) for x in self.new_ffs))))
            inv_ff = self.ff.__class__()
            if self.do_weighted_reflection:
                inv_ff.method = 'WEIGHTED INVERSION'
            else:
                inv_ff.method = 'INVERSION'
            inv_ff.params = copy.deepcopy(best_ff.params)
            ref_ff = self.ff.__class__()
            ref_ff.method = 'REFLECTION'
            ref_ff.params = copy.deepcopy(best_ff.params)
            # Need score difference sum for weighted inversion.
            # If zero, should break.
            score_diff_sum = sum([x.score - self.new_ffs[-1].score
                                  for x in self.new_ffs[:-1]])
            if score_diff_sum == 0.:
                logger.warning(
                    'No difference between force field scores. '
                    'Exiting simplex.')
                # Maybe excessive to repeat message, but better safe than
                # sorry.
                # We want to raise opt.OptError such that opt.catch_run_errors
                # will write the best FF obtained thus far.
                raise opt.OptError(
                    'No difference between force field scores. '
                    'Exiting simplex.')
            for i in xrange(0, len(best_ff.params)):
                if self.do_weighted_reflection:
                    try:
                        inv_val = (
                            sum([x.params[i].value *
                                 (x.score - self.new_ffs[-1].score)
                                 for x in self.new_ffs[:-1]])
                            / 
                            sum([x.score - self.new_ffs[-1].score
                                 for x in self.new_ffs[:-1]]))
                    except ZeroDivisionError:
                        logger.warning(
                            'Attempted to divide by zero while calculating the '
                            'weighted simplex inversion point. All penalty '
                            'function scores for the trial force fields are '
                            'numerically equivalent.')
                        # This needs to be a raise. opt.catch_run_errors will
                        # catch this exception, and make sure to write the best
                        # FF obtained up until this point.
                        # Also, since this is now raise, rather than break, it
                        # should exit the simplex while loop. Really, it should
                        # exit simplex entirely.
                        raise
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
            self.ff.export_ff(self.ff.path, params=ref_ff.params)
            data = calculate.main(self.args_ff)
            ref_ff.score = compare.compare_data(r_data, data)
            opt.pretty_ff_results(ref_ff)
            if ref_ff.score < self.new_ffs[0].score:
                logger.log(20, '~~ ATTEMPTING EXPANSION ~~'.rjust(79, '~'))
                exp_ff = self.ff.__class__()
                exp_ff.method = 'EXPANSION'
                exp_ff.params = copy.deepcopy(best_ff.params)
                for i in xrange(0, len(self.new_ffs[0].params)):
                    exp_ff.params[i].value = (
                        3 * inv_ff.params[i].value -
                        2 * self.new_ffs[-1].params[i].value)
                self.ff.export_ff(self.ff.path, exp_ff.params)
                data = calculate.main(self.args_ff)
                exp_ff.score = compare.compare_data(r_data, data)
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
                con_ff.params = copy.deepcopy(best_ff.params)
                for i in xrange(0, len(best_ff.params)):
                    if ref_ff.score > self.new_ffs[-1].score:
                        con_val = (
                            (inv_ff.params[i].value +
                             self.new_ffs[-1].params[i].value) / 2)
                    else:
                        con_val = (
                            (3 * inv_ff.params[i].value -
                             self.new_ffs[-1].params[i].value) / 2)
                    con_ff.params[i].value = con_val
                self.ff.export_ff(self.ff.path, params=con_ff.params)
                data = calculate.main(self.args_ff)
                con_ff.score = compare.compare_data(r_data, data)
                opt.pretty_ff_results(con_ff)
                # This change was made to reflect the 1998 Q2MM publication.
                # if con_ff.score < self.new_ffs[-1].score:
                if con_ff.score < self.new_ffs[-2].score:
                    self.new_ffs[-1] = con_ff
                elif self.do_massive_contraction:
                    logger.log(
                        20, '~~ DOING MASSIVE CONTRACTION ~~'.rjust(79, '~'))
                    for ff_num, ff in enumerate(self.new_ffs[1:]):
                        for i in xrange(0, len(best_ff.params)):
                            ff.params[i].value = (
                                (ff.params[i].value +
                                 self.new_ffs[0].params[i].value) / 2)
                        self.ff.export_ff(self.ff.path, params=ff.params)
                        data = calculate.main(self.args_ff)
                        ff.score = compare.compare_data(r_data, data)
                        ff.method += ' MC'
                        opt.pretty_ff_results(ff)
                else:
                    logger.log(20, '  -- Contraction failed.')
            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            if self.new_ffs[0].score < last_best:
                cycles_wo_change = 0
            else:
                cycles_wo_change += 1
                logger.log(20, '  -- {} cycles without improvement, out of {} allowed.'.format(
                        cycles_wo_change, self._max_cycles_wo_change))
            best_ff = self.new_ffs[0]
            logger.log(20, 'BEST:')
            opt.pretty_ff_results(self.new_ffs[0], level=20)
            logger.log(20, '~~ END SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))
        if best_ff.score < self.ff.score:
            logger.log(20, '~~ SIMPLEX FINISHED WITH IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            best_ff = restore_simp_ff(best_ff, self.ff)
        else:
            logger.log(20, '~~ SIMPLEX FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                    79, '~'))
        opt.pretty_ff_results(self.ff, level=20)
        opt.pretty_ff_results(best_ff, level=20)
        logger.log(20, '  -- Writing best force field from simplex.')
        best_ff.export_ff(best_ff.path)
        return best_ff

def select_simp_params_on_derivs(params, max_params=10):
    """
    Sorts parameter sets from lowest to highest second
    derivatives of their score in the objective function.
    
    Parameters
    ----------
    params : list of subclasses of `datatypes.Param`
    """
    keep = sorted(params, key=lambda x: x.d2)
    keep = params[:max_params]
    return keep

def reduce_num_simp_params(ff, ffs, max_params=10):
    logger.log(
        20, '  -- Reducing number of parameters to {}'.format(max_params))
    opt.param_derivs(ff, ffs)
    simp_params = select_simp_params_on_derivs(
        ff.params, max_params=max_params)
    return simp_params
    
def reduce_num_simp_ffs(ffs, params):
    simp_ffs = opt.extract_forward(ffs)
    simp_ffs = opt.extract_ff_by_params(ffs, params)
    return simp_ffs

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

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    # Just for testing.
    ff = datatypes.MM3('b071/mm3.fld')
    ff.import_ff()
    ff.params = parameters.trim_params_by_file(ff.params, 'b071/params.txt')
    a = Simplex(
        direc='b071', ff=ff,
        args_ff=' -d b071 -me X002a.mae X002b.mae -mb X002a.mae',
        args_ref=' -d b071 -je X002a.mae X002b.mae -jb X002a.mae')
    a.run()
