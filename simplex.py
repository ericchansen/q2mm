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
import opt
import parameters

logger = logging.getLogger(__name__)

class Simplex(opt.Optimizer):
    """
    Optimizes force field parameters using an in-house version of the simplex
    method. See `Optimizer` for repeated documentation.

    Attributes
    ----------
    do_massive_contraction : bool
                             If True, allows massive contractions to be
                             performed, contracting all parameters at once.
    do_weighted_reflection : bool
                             If True, weights parameter sets based on their
                             objective function score when determining the
                             reflection point.
    max_cycles : int
                 Maximum number of simplex cycles.
    max_cycles_wo_change : int
                           End the simplex optimization early if there have
                           been this many consecutive simplex steps without
                           improvement in the objective function.
    max_params : int
                 Maximum number of parameters used in a single simplex cycle.
    """
    def __init__(self,
                 ff=None, ff_lines=None, ff_args=None,
                 ref_args=None, ref_conn=None,
                 restore=False):
        super(Simplex, self).__init__(
            ff, ff_lines, ff_args, ref_args, ref_conn, restore)
        self.do_massive_contraction = True
        self.do_weighted_reflection = True
        self.max_cycles = 10
        self.max_cycles_wo_change = 3
        self.max_params = 10
    def run(self):
        """
        Once all attributes are setup as you so desire, run this method to
        optimize the parameters.

        Returns
        -------
        `datatypes.FF` (or subclass)
            Contains the best parameters.
        """
        logger.log(20, '~~ SIMPLEX OPTIMIZATION ~~'.rjust(79, '~'))
        # Here we don't actually need the database connection/force field data.
        # We only need the score.
        if self.ff.score is None:
            logger.log(20, '~~ CALCULATING INITIAL FF SCORE ~~'.rjust(79, '~'))
            # datatypes.export_ff(
            #     self.ff.path, self.ff.params, lines=self.ff.lines)
            self.ff.export_ff(self.ff.path)
            # I could store this object to prevent on self.ff to prevent garbage
            # collection. Would be nice if simplex was followed by gradient,
            # which needs that information, and if simplex yielded no
            # improvements. At most points in the optimization, this is probably
            # too infrequent for it to be worth the memory, but it might be nice
            # once the parameters are close to convergence.
            conn = calculate.main(self.ff_args)
            self.ff.score = compare.compare_data(self.ref_conn, conn)
        else:
            logger.log(15, '  -- Reused existing score and data for initial FF.')
        logger.log(15, 'INIT FF SCORE: {}'.format(self.ff.score))
        if self.max_params and len(self.ff.params) > self.max_params:
            logger.log(
                20, '  -- Reducing number of parameters to {} for the simplex '
                'optimization.'.format(self.max_params))
            ffs = opt.differentiate_ff(self.ff)
            opt.score_ffs(
                ffs, self.ff_args, self.ref_conn, parent_ff=self.ff,
                restore=False)
            opt.param_derivs(self.ff, ffs)
            simp_params = select_simp_params(
                self.ff.params, max_params=self.max_params)
            self.new_ffs = opt.extract_forward(ffs)
            self.new_ffs = opt.extract_ff_by_params(
                self.new_ffs, simp_params)
        else:
            ffs = opt.differentiate_ff(self.ff, central=False)
            opt.score_ffs(
                ffs, self.ff_args, self.ref_conn, parent_ff=self.ff,
                restore=False)
            self.new_ffs = ffs
        self.new_ffs = sorted(self.new_ffs + [self.ff], key=lambda x: x.score)
        opt.pretty_ff_params(self.new_ffs)

        current_cycle = 0
        cycles_wo_change = 0
        while current_cycle < self.max_cycles \
                and cycles_wo_change < self.max_cycles_wo_change:
            current_cycle += 1
            last_best = self.new_ffs[0].score
            best_ff = self.new_ffs[0]
            logger.log(20, '~~ START SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))
            inv_ff = self.ff.__class__()
            if self.do_weighted_reflection:
                inv_ff.method = 'WEIGHTED INVERSION'
            else:
                inv_ff.method = 'INVERSION'
            inv_ff.params = copy.deepcopy(best_ff.params)
            ref_ff = self.ff.__class__()
            ref_ff.method = 'REFLECTION'
            ref_ff.params = copy.deepcopy(best_ff.params)
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
                        # Breaking should just exit the while loop. Should still
                        # give you the best force field determined thus far.
                        break
                else:
                    inv_val = (
                        sum([x.params[i].value for x in self.new_ffs[:-1]])
                        /
                        len(self.new_ffs[:-1]))
                inv_ff.params[i].value = inv_val
                ref_ff.params[i].value = (
                    2 * inv_val - self.new_ffs[-1].params[i].value)
            # Calculate score for inverted parameters.
            self.ff.export_ff(self.ff.path, params=inv_ff.params)
            # datatypes.export_ff(
            #     self.ff.path, inv_ff.params, lines=self.ff.lines)
            conn = calculate.main(self.ff_args)
            inv_ff.score = compare.compare_data(self.ref_conn, conn)
            opt.pretty_ff_results(inv_ff)
            
            # HERE
            sys.exit(0)
            
            # Calculate score for reflected parameters.
            self.ff.export_ff(self.ff.path, params=ref_ff.params)
            # datatypes.export_ff(
            #     self.ff.path, ref_ff.params, lines=self.ff.lines)
            conn = calculate.main(self.ff_args)
            ref_ff.score = compare.compare_data(self.ref_conn, conn)
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
                # datatypes.export_ff(
                #     self.ff.path, exp_ff.params, lines=self.ff.lines)
                conn = calculate.main(self.ff_args)
                exp_ff.score = compare.compare_data(self.ref_conn, conn)
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
                # datatypes.export_ff(
                #     self.ff.path, con_ff.params, lines=self.ff.lines)
                conn = calculate.main(self.ff_args)
                con_ff.score = compare.compare_data(self.ref_conn, conn)
                opt.pretty_ff_results(con_ff)
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
                        # datatypes.export_ff(
                        #     self.ff.path, ff.params, lines=self.ff.lines)
                        conn = calculate.main(self.ff_args)
                        ff.score = compare.compare_data(self.ref_conn, conn)
                        ff.method += ' MC'
                        opt.pretty_ff_results(ff)
                else:
                    logger.log(20, '  -- Contraction failed.')
            self.new_ffs = sorted(self.new_ffs, key=lambda x: x.score)
            if self.new_ffs[0].score < last_best:
                cycles_wo_change = 0
            else:
                cycles_wo_change += 1
                logger.log(20, '  -- {} cycles without change.'.format(
                        cycles_wo_change))
            best_ff = self.new_ffs[0]
            logger.log(20, 'BEST:')
            opt.pretty_ff_results(self.new_ffs[0], level=20)
            logger.log(20, '~~ END SIMPLEX CYCLE {} ~~'.format(
                    current_cycle).rjust(79, '~'))
        if best_ff.score < self.ff.score:
            logger.log(20, '~~ SIMPLEX FINISHED WITH IMPROVEMENTS ~~'.rjust(
                    79, '~'))
            self.ff.copy_attributes(best_ff)
            if self.max_params is not None and \
                    len(self.ff.params) > self.max_params:
                best_params = copy.deepcopy(best_ff.params)
                best_ff.params = copy.deepcopy(self.ff.params)
                for a, param_a in enumerate(self.ff.params):
                    for param_b in best_params:
                        if param_a.mm3_row == param_b.mm3_row and \
                                param_a.mm3_col == param_b.mm3_col:
                            best_ff.params[i] = copy.deepcopy(param_b)
        else:
            logger.log(20, '~~ SIMPLEX FINISHED WITHOUT IMPROVEMENTS ~~'.rjust(
                    79, '~'))
        opt.pretty_ff_results(self.ff, level=20)
        opt.pretty_ff_results(best_ff, level=20)
        # Restore original.
        if self.restore:
            logger.log(20, '  -- Restoring original force field.')
            self.ff.export_ff(self.ff.path)
            # datatypes.export_ff(
            #     self.ff.path, self.ff.params, lines=self.ff.lines)
        # The best force field should be totally okay with doing all of this
        # now. I guess it does sort suck that I have duplicate data in memory
        # now. Perhaps I should delete self.ff.
        else:
            logger.log(20, '  -- Writing best force field from simplex.')
            best_ff.export_ff(best_ff.path)
            # datatypes.export_ff(
            #     best_ff.path, best_ff.params, lines=best_ff.lines)
        return best_ff

def select_simp_params(params, max_params=10):
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

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
