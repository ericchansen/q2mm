#!/usr/bin/python
import argparse
import calculate
import copy
import evaluate
import gradient
import logging
import logging.config
import loop
import parameters
from parameters import BaseParam, MM3Param, BaseFF, MM3FF
import pickle
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

logger = logging.getLogger(__name__)

def process_args(args):
    parser = argparse.ArgumentParser(
        description='''Optimize parameters using a simplex implementation.

For more implementation details, see:
Norrby, Liljefors. Automated Molecular Mechanics Parameterization with
Simultaneous Utilization of Experimental and Quantum Mechanical Data.
J. Comp. Chem., 1998, 1146-1166.

We limit the number of parameters that we optimize with simplex at one
time, and the selection criterion here is different than in the past
code. To be honest, the old code wasn't very well documented and that
whole selection process (generating simpopt) seemed pretty random to me.
I can change the method herein if someone explains a better way.''', 
        formatter_class=argparse.RawDescriptionHelpFormatter)
    calc_opts = parser.add_argument_group('Arguments for calculate.py')
    simp_opts = parser.add_argument_group('Simplex options')
    save_opts = parser.add_argument_group('Load/save options')
    share_opts = parser.add_argument_group('Shared options')
    calc_opts.add_argument(
        '--calc', '-c', type=str, metavar='"calculate.py arguments"',
        help='Commands for calculate.py to determine the calculated FF data.')
    calc_opts.add_argument(
        '--ref', '-r', type=str, metavar='"calculate.py arguments"', 
        help='Commands for calculate.py to determine the reference data.')
    simp_opts.add_argument(
        '--nomassive', action='store_true',
        help='Disable doing massive contractions. Enabled by default.')
    simp_opts.add_argument(
        '--max', '-m', type=int, default=25, metavar=25,
        help='Maximum number of simplex cycles. Default is 25.')
    simp_opts.add_argument(
        '--nochange', '-no', type=int, default=5, metavar=5,
        help='Maximum number of simplex cycles without any improvement in ' +
        'the penalty function. Default is 5.')
    simp_opts.add_argument(
        '--num', '-n', type=int, nargs='?', const=10, metavar='10',
        help='Maximum number of parameters to use in simplex optimization. ' +
        'If no optional number is given, the default is 10 (11 apexes). ' +
        'Note that simplex will use the number of parameters + 1 if this ' +
        'is not given. The parameters with the lowest numerical 2nd ' +
        'derivatives are chosen.')
    simp_opts.add_argument(
        '--weight', action='store_true',
        help='Use weighted simplex scheme, which changes the definition of ' +
        'the inversion point.')
    save_opts.add_argument(
        '--init', type=str, metavar='relative/path/simplex.pickle',
        nargs='?', const='simplex.pickle',
        help='Do the setup calculations, save that data (by default to ' +
        "simplex.pickle), but don't actually start the simplex optimization.")
    save_opts.add_argument(
        '--load', type=str, metavar='relative/path/simplex.pickle',
        nargs='?', const='simplex.pickle',
        help='Load intermediate results from the pickle. If no argument ' +
        'is given, loads from simplex.pickle.')
    save_opts.add_argument(
        '--save', type=str, metavar='relative/path/simplex.pickle',
        nargs='?', const='simplex.pickle',
        help='Save results to the pickle. If no additional argument ' +
        'is given, save to simplex.pickle.')
    share_opts.add_argument(
        '--ffpath', '-f', type=str, metavar='relative/path/filename.fld',
        default='mm3.fld',
        help='Relative path to force field. Default is mm3.fld')
    share_opts.add_argument(
        '--params', '-p', type=str, metavar='relative/path/parameters.yaml',
        default='options/parameters.yaml',
        help='Relative path to parameter file. The parameters specified ' +
        'by this file will be optimized. Default is options/parameters.yaml. ' +
        'This file can be generated using parameters.py.')
    share_opts.add_argument(
        '--steps', type=str, metavar='relative/path/steps.yaml',
        default='options/steps.yaml',
        help='Relative path to step size YAML file. Default is ' +
        'options/steps.yaml.')
    share_opts.add_argument(
        '--substr', type=str, metavar='"Substructure Name OPT"',
        default='OPT',
        help='String used to identify the substructure to modify in the ' +
        'MM3* force field.')
    options = vars(parser.parse_args(args))
    return options

def run(args):
    options = process_args(args)
    # Load data from the pickle.
    if options['load']:
        logger.info('Loading simplex.py data from {}.'.format(
                options['load']))
        with open(options['load'], 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = {'Params.': None,
               'Ref. Data': None,
               'FF': None,
               'Trial FFs': []}
    try:
        dic = loop.setup(options, dic)
        logger.info('Starting FF X2: {}'.format(dic['FF'].x2))
        dic = execute(options, dic)
        logger.info('Evaluating trial FFs.')
        dic['Trial FFs'] = evaluate.calc_data_for_ffs(
            dic['Trial FFs'], calc_args=options['calc'],
            ref_data=dic['Ref. Data'], recalc=False, backup=True)
        # Not necessary to add in initial FF because that was
        # already used as an apex during the simplex optimization.
        dic['Trial FFs'] = sorted(
            dic['Trial FFs'], key=lambda x: x.x2)
        logger.info('Best after simplex: {}'.format(
                dic['Trial FFs'][0].x2))
        logger.info('Writing best FF ({}) to {}.'.format(
                dic['Trial FFs'][0].x2, dic['Trial FFs'][0].filename))
        parameters.export_mm3_ff(
            dic['Trial FFs'][0].params,
            in_filename=dic['Trial FFs'][0].filename,
            out_filename=dic['Trial FFs'][0].filename)
    except:
        # Backup if an error occurs.
        logger.error(
            'Error encountered. Backing up data to simplex.error.pickle.')
        with open('simplex.error.pickle', 'wb') as f:
            pickle.dump(dic, f)
        raise

def execute(options, dic):
    # Limit number of parameters.
    if options['num']:
        logger.debug('Limiting number of parameters to {}.'.format(
                options['num']))
        # Do central differentiation to determine derivatives of parameters
        # with respect to the penalty function.
        if dic['FF'].cent_ffs == []:
            dic['FF'].cent_ffs = gradient.central_differentiation(
                dic['FF'], steps_filename=options['steps'])
        # Calculate data and X2 for the stepped parameters. This function
        # checks for us if data already exists.
        logger.debug('Calculating data for central differentiated FFs.')
        dic['FF'].cent_ffs = evaluate.calc_data_for_ffs(
            dic['FF'].cent_ffs, calc_args=options['calc'],
            ref_data=dic['Ref. Data'], recalc=False, backup=True)
        dic['FF'] = gradient.calc_derivatives(
            dic['FF'], dic['FF'].cent_ffs)
        # This trims the number of parameters down to the max.
        dic['FF'].remove_other_params(sorted(
                dic['FF'].params, key=lambda x: x.der2)[:options['num']])
        logger.info('Limited number of parameters to {}.'.format(
            len(dic['FF'].params)))
    # Forward differentiation. Use to create simplex apexes. Each 
    # parameter generates one apex.
    if dic['FF'].for_ffs == []:
        dic['FF'].for_ffs = gradient.forward_differentiation(
            dic['FF'], steps_filename=options['steps'])
        logger.debug('Calculating data for forward differentiated FFs.')
        dic['FF'].for_ffs = evaluate.calc_data_for_ffs(
            dic['FF'].for_ffs, calc_args=options['calc'].split(),
            ref_data=dic['Ref. Data'], recalc=False, backup=True)
    else:
        logger.info('Checking loaded forward differentiation data.')
        dic['FF'].for_ffs = evaluate.calc_data_for_ffs(
            dic['FF'].for_ffs, calc_args=options['calc'].split(),
            ref_data=dic['Ref. Data'], recalc=False, backup=True)
    if options['init']:
        logger.info('Saving initial results to {}.'.format(options['init']))
        with open(options['init'], 'wb') as f:
            pickle.dump(dic, f)
        return dic
    # Okay, we're all setup. Time to do the simplex optimization.
    dic['Trial FFs'] = [dic['FF']] + dic['FF'].for_ffs
    dic['Trial FFs'] = sorted(dic['Trial FFs'], key=lambda x: x.x2)
    # Initialize some variables to keep track of the simplex cycles.
    cycle_num = 0
    cycles_wo_change = 0
    while cycle_num < options['max'] and \
            cycles_wo_change < options['nochange']:
        cycle_num += 1
        last_best_x2 = dic['Trial FFs'][0].x2
        logger.info('{} - Simplex Cycle - Best: {}'.format(
                cycle_num, dic['Trial FFs'][0].x2))
        # logger.debug('All X2 values: {}'.format(
        #         [x.x2 for x in dic['Trial FFs']]))
        # For simplex implementation details see:
        # Norrby; Liljefors. Automated Molecular Mechanics Parameterization
        # with Simultaneous Utilization of Experimental and Quantum
        # Mechanical Data. J. Comp. Chem., 1998, 19, 1146-1166.
        # Seriously, it's like exactly out of that paper now.
        inv_ff = dic['FF'].spawn_child_ff(gen_method='Inversion')
        ref_ff = dic['FF'].spawn_child_ff(gen_method='Reflection')
        for i in xrange(0, len(dic['FF'].params)):
            if options['weight']:
                inv_param = \
                    sum([x.params[i].value * (x.x2 - dic['Trial FFs'][-1].x2) 
                         for x in dic['Trial FFs'][:-1]]) / sum(
                        [x.x2 - dic['Trial FFs'][-1].x2
                         for x in dic['Trial FFs'][:-1]])
            else:
                inv_param = \
                    sum([x.params[i].value for x in dic['Trial FFs'][:-1]]) \
                    / len(dic['Trial FFs'][:-1])
            inv_ff.params[i].value = inv_param
            ref_param = 2 * inv_param - dic['Trial FFs'][-1].params[i].value
            ref_ff.params[i].value = ref_param
        ref_ff.calculate_data(options['calc'], backup=False)
        ref_ff.calculate_x2(dic['Ref. Data'])
        logger.info('Reflection X2: {}'.format(ref_ff.x2))
        if ref_ff.x2 < dic['Trial FFs'][0].x2:
            logger.debug('Attempting expansion.')
            exp_ff = dic['FF'].spawn_child_ff(gen_method='Expansion')
            for i in xrange(0, len(dic['FF'].params)):
                exp_param = 3 * inv_ff.params[i].value - \
                    2 * dic['Trial FFs'][-1].params[i].value
            exp_ff.calculate_data(options['calc'], backup=False)
            exp_ff.calculate_x2(dic['Ref. Data'])
            logger.info('Expansion X2: {}'.format(exp_ff.x2))
            if exp_ff.x2 < ref_ff.x2:
                dic['Trial FFs'][-1] = exp_ff
            else:
                dic['Trial FFs'][-1] = ref_ff
        elif ref_ff.x2 < dic['Trial FFs'][-2].x2:
            logger.info('Simple reflection succeeded.')
            dic['Trial FFs'][-1] = ref_ff
        else:
            logger.debug('Attempting contraction.')
            con_ff = dic['FF'].spawn_child_ff(gen_method='Contraction')
            for i in xrange(0, len(dic['FF'].params)):
                if ref_ff.x2 > dic['Trial FFs'][-1].x2:
                    con_param = (inv_ff.params[i].value + 
                                 dic['Trial FFs'][-1].params[i].value) / 2
                else:
                    con_param = (3 * inv_ff.params[i].value -
                                 dic['Trial FFs'][-1].params[i].value) / 2
            con_ff.calculate_data(options['calc'], backup=False)
            con_ff.calculate_x2(dic['Ref. Data'])
            logger.info('Contraction X2: {}'.format(con_ff.x2))
            if con_ff.x2 < dic['Trial FFs'][-2].x2:
                dic['Trial FFs'][-1] = con_ff
            elif not options['nomassive']:
                logger.info('Doing massive contraction.')
                # Best doesn't change. Skip over it.
                for ff in dic['Trial FFs'][1:]:
                    for i in xrange(0, len(dic['FF'].params)):
                        ff.params[i].value = \
                            (ff.params[i].value + \
                                 dic['Trial FFs'][0].params[i].value) / 2
                dic['Trial FFs'][1:] = evaluate.calc_data_for_ffs(
                    dic['Trial FFs'][1:], calc_args=options['calc'],
                    ref_data=dic['Ref. Data'], recalc=True, backup=False)
        dic['Trial FFs'] = sorted(dic['Trial FFs'], key=lambda x: x.x2)
        if dic['Trial FFs'][0].x2 < last_best_x2:
            cycles_wo_change = 0
        else:
            cycles_wo_change += 1
        logger.info('Cycles w/o change in best X2: {}'.format(cycles_wo_change))
    # Semi-manually restore initial FF.
    logger.debug('Restoring initial FF.')
    parameters.export_mm3_ff(params=dic['FF'].params,
                             in_filename=dic['FF'].filename,
                             out_filename=dic['FF'].filename)
    if options['save']:
        logger.info('Saving simplex.py results to {}'.format(options['save']))
        with open(options['save'], 'wb') as f:
            pickle.dump(dic, f)
    return dic

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    # Begin.
    run(sys.argv[1:])
