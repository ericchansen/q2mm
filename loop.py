#!/usr/bin/python
import argparse
import calculate
import evaluate
import gradient
import logging
import logging.config
import os
import parameters
from parameters import BaseFF, MM3FF, BaseParam, MM3Param
import pickle
from setup_logging import log_uncaught_exceptions, remove_logs
import simplex
import sys
import yaml

logger = logging.getLogger(__name__)

# Not sure if this is the "best" way to to gracefully exit, but it
# works.
class StopException(Exception):
    pass

def check_stop():
    if os.path.exists('stop'):
        os.remove('stop')
        logger.warning('Exiting loop via stop file.')
        raise StopException()

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Loops through optimization methods until the penalty ' +
        'function reaches convergence. The loop can gracefully be ended ' +
        'by creating a file "stop" in the working directory.')
    calc_opts = parser.add_argument_group('Arguments for calculate.py')
    opt_opts = parser.add_argument_group('Arguments for optimization methods')
    save_opts = parser.add_argument_group('Load/save options')
    share_opts = parser.add_argument_group('Shared options')
    parser.add_argument(
        '--conv', type=float, default=0.1, metavar='0.1',
        help='The parameters are considered to be converged and looping ' +
        'stops once the percent change in the penalty function is less than ' +
        'this variable. By default, this is set to 10%% (0.1). In later ' +
        'stages of parameter refinement, this should be decreased.')
    parser.add_argument(
        '--num', type=int, nargs='?', const=10, metavar='10',
        help='Maximum number of loop cycles. If not given, looping ' +
        'continues until the convergence criterion is met.')
    calc_opts.add_argument(
        '--calc', '-c', type=str, metavar='"calculate.py arguments"',
        help='Arguments for calculate.py to determine the calculated FF data.')
    calc_opts.add_argument(
        '--ref', '-r', type=str, metavar='"calculate.py arguments"', 
        help='Arguments for calculate.py to determine the reference data.')
    opt_opts.add_argument(
        '--custom', action='store_true',
        help='Similar to --default, but I am doing using more methods than ' +
        'before. I recommend this. Soon I will have better input methods ' +
        'available for this script so you can be more in control.')
    opt_opts.add_argument(
        '--default', action='store_true',
        help='This invokes the optimization methods used in the previous ' +
        'Python scripts. In each loop, several gradient methods based off ' + 
        'of central differentiation are used (previously defined in ' +
        'NR_Jacob.py) followed by a simplex optimization. Note that ' +
        'the maximum number of simplex cycles have been reduced from 100 ' +
        'to 25 and the parameter selection for the simplex optimization ' +
        'varies somewhat. Furthermore, the simplex allows a maximum of ' +
        '10 parameters (11 simplex apexes) by default. This option ' +
        'overrides the --simplex and --gradient arguments.')
    opt_opts.add_argument(
        '--simplex', '-s', type=str, metavar='"simplex.py arguments"',
        help='Arguments for simplex.py. Any arguments under "Shared ' +
        'options" or "Arguments for calculate.py" should\'t be included.')
    opt_opts.add_argument(
        '--gradient', '-g', type=str, metavar='"gradient.py arguments"',
        help='Arguments for gradient.py. Any arguments under "Shared ' +
        'options" or "Arguments for calculate.py" should\'t be included.')
    save_opts.add_argument(
        '--load', type=str, metavar='relative/path/loop.pickle', nargs='?',
        const='loop.pickle',
        help='Load results from the pickle. If no argument is given, loads ' +
        'from loop.pickle.')
    save_opts.add_argument(
        '--save', type=str, metavar='relative/path/loop.pickle',
        nargs='?', const='loop.pickle',
        help='Save data to the pickle. If not argument is given, saves to ' +
        'loop.pickle.')
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
    # More argument handling. This constructs the arguments that will be
    # fed to the optimization methods.
    if options['custom']:
        options['default'] = True
    # if options['default'] or options['custom']:
    if options['default']:
        options['gradient'] = \
            ['-c', options['calc'], '-r', options['ref'], '--ffpath',
             options['ffpath'], '--params', options['params'], '--steps',
             options['steps'], '--substr', options['substr']]
        options['simplex'] = \
            ['-c', options['calc'], '-r', options['ref'], '--ffpath',
             options['ffpath'], '--params', options['params'], '--steps',
             options['steps'], '--substr', options['substr']]
    else:
        if options['gradient']:
            options['gradient'] = \
                ['-c', options['calc'], '-r', options['ref'], '--ffpath',
                 options['ffpath'], '--params', options['params'], '--steps',
                 options['steps'], '--substr', options['substr']] + \
                 options['gradient'].split()
        if options['simplex']:
            options['simplex'] = \
                ['-c', options['calc'], '-r', options['ref'], '--ffpath',
                 options['ffpath'], '--params', options['params'], '--steps',
                 options['steps'], '--substr', options['substr']] + \
                 options['simplex'].split()
    return options

def run(args):
    # Determine options for loop.py.
    options = process_args(args)
    if options['load']:
        logger.info('Loading loop.py data from {}.'.format(
                options['load']))
        with open(options['load'], 'rb') as f:
            dic = pickle.load(f)
    else:
        dic = {'Params.': None,
               'Ref. Data': None,
               'FF': None,
               'Trial FFs': []}
    # Do any necessary initial calculations.
    try:
        dic = setup(options, dic)
        start = dic['FF'].x2
        loop_num = 0
        old_best = None
        while (old_best is None or abs(old_best - dic['FF'].x2) / old_best
               > options['conv']) and \
               (options['num'] is None or loop_num < options['num']):
            # Start of each loop.
            loop_num += 1
            if old_best is None:
                logger.info('{} - Loop - Best: {}'.format(
                        loop_num, dic['FF'].x2))
            else:
                logger.info('{} - Loop - Best: {} - Change: {}%'.format(
                        loop_num, dic['FF'].x2,
                        abs(old_best - dic['FF'].x2) / old_best * 100 ))
            old_best = dic['FF'].x2
            check_stop()
            # Default method (very similar to method used previously in
            # Elaine's code).
            if options['default']:
                # Get options for gradient.py.
                gradient_options = gradient.process_args(options['gradient'])
                # For the 1st time, just do the --init option. This lets us
                # combine different cutoff/max radius methods.
                dic = gradient.execute(gradient_options, dic)
                # Basic and SVD.
                gradient_options = gradient.process_args(
                    options['gradient'] + 
                    ['--cutoff', '10', '-b', '-s'])
                dic = gradient.execute(gradient_options, dic)
                # Lagrange dampening.
                gradient_options = gradient.process_args(
                    options['gradient'] + 
                    ['--radius', '5', '-l'])
                dic = gradient.execute(gradient_options, dic)
                gradient_options = gradient.process_args(
                    options['gradient'] + 
                    ['--radius', '1', '-nr'])
                dic = gradient.execute(gradient_options, dic)
                gradient_options = gradient.process_args(
                    options['gradient'] + 
                    ['--radius', '3', '-nr'])
                dic = gradient.execute(gradient_options, dic)
                gradient_options = gradient.process_args(
                    options['gradient'] + 
                    ['--radius', '10', '-nr'])
                dic = gradient.execute(gradient_options, dic)
                # Adds a few more things that we didn't use previously.
                if options['custom']:
                    # Levenberg-Marquardt. Not sure why that was skipped
                    # before. Seems to work nicely.
                    gradient_options = gradient.process_args(
                        options['gradient'] +
                        ['--radius', '5', '-lm'])
                    dic = gradient.execute(gradient_options, dic)
                # Check for no possible trial FFs.
                if len(dic['Trial FFs']) == 0:
                    logger.warning("Couldn't create trial FFs. Writing best " +
                                   "FF to {} and exiting.".format(
                            dic['FF'].filename))
                    parameters.export_mm3_ff(
                        dic['FF'].params, in_filename=dic['FF'].filename,
                        out_filename=dic['FF'].filename)
                    return dic
                check_stop()
                logger.info('Evaluating trial FFs from gradient.py.')
                dic['Trial FFs'] = evaluate.calc_data_for_ffs(
                    dic['Trial FFs'], calc_args=options['calc'],
                    ref_data=dic['Ref. Data'], recalc=False, backup=True)
                # Add in the initial FF before sorting.
                dic['Trial FFs'] = sorted(
                    dic['Trial FFs'] + [dic['FF']], key=lambda x: x.x2)
                # New FF is one with the lowest X2.
                dic['FF'] = dic['Trial FFs'][0]
                logger.info('Best after gradient: {}'.format(dic['FF'].x2))
                # Reset trial FFs.
                dic['Trial FFs'] = []
                check_stop()
                # Unlike gradient, the data and X2 values are already all
                # here. This shortens what we have to include here.
                simplex_options = simplex.process_args(options['simplex'])
                dic = simplex.execute(simplex_options, dic)
                # They're already sorted too. Hooray!
                dic['FF'] = dic['Trial FFs'][0]
                logger.info('Best after simplex: {}'.format(dic['FF'].x2))
                # Reset trial FFs.
                dic['Trial FFs'] = []
                check_stop()
            # If --default isn't given, it will currently cycle between gradient
            # and then simplex using whatever arguments you supplied. This is
            # just a temporary measure. In the future, there will be much more
            # control allowed here.
            else:
                if options['gradient']:
                    # Get options for gradient.py.
                    gradient_options = gradient.process_args(
                        options['gradient'])
                    # Do the gradient based methods specified by command line
                    # arguments.
                    dic = gradient.execute(gradient_options, dic)
                    # Check for no possible trial FFs.
                    if len(dic['Trial FFs']) == 0:
                        logger.warning(
                            "Couldn't create trial FFs. Writing best FF to " +
                            "{} and exiting.".format(dic['FF'].filename))
                        parameters.export_mm3_ff(
                            dic['FF'].params, in_filename=dic['FF'].filename,
                            out_filename=dic['FF'].filename)
                        return dic
                    check_stop()
                    logger.info('Evaluating trial FFs from gradient.py.')
                    # Evaluate results from gradient.py.
                    dic['Trial FFs'] = evaluate.calc_data_for_ffs(
                        dic['Trial FFs'], calc_args=options['calc'],
                        ref_data=dic['Ref. Data'], recalc=False, backup=True)
                    # Add in initial FF before sorting.
                    dic['Trial FFs'] = sorted(
                        dic['Trial FFs'] + [dic['FF']], key=lambda x: x.x2)
                    # Replace with best FF.
                    dic['FF'] = dic['Trial FFs'][0]
                    logger.info('Best after gradient: {}'.format(dic['FF'].x2))
                    # Reset trial FFs.
                    dic['Trial FFs'] = []
                    check_stop()
                if options['simplex']:
                    # Unlike gradient, the data and X2 values are already all
                    # here. This shortens what we have to include here.
                    simplex_options = simplex.process_args(options['simplex'])
                    dic = simplex.execute(simplex_options, dic)
                    # They're already sorted too. Hooray!
                    dic['FF'] = dic['Trial FFs'][0]
                    logger.info('Best after simplex: {}'.format(dic['FF'].x2))
                    # Reset trial FFs.
                    dic['Trial FFs'] = []
                    check_stop()
        logger.info('End {} - Last {} - Change {}% - Start {}'.format(
                dic['FF'].x2, old_best, abs(old_best - dic['FF'].x2) /
                old_best * 100, start))
        logger.info('Writing best FF to {}.'.format(
                dic['FF'].x2, dic['FF'].filename))
        parameters.export_mm3_ff(
            dic['FF'].params, in_filename=dic['FF'].filename,
            out_filename=dic['FF'].filename)
    except StopException:
        return
    except:
        logger.error(
            'Error encountered. Backing up data to loop.error.pickle.')
        with open('loop.error.pickle', 'wb') as f:
            pickle.dump(dic, f)
        raise

def setup(options, dic):
    logger.info('Performing initial calculations.')
    # Load parameters to optimize.
    if dic['Params.'] is None and options['params']:
        logger.info('Loading parameters to optimize from {}.'.format(
                options['params']))
        with open(options['params'], 'r') as f:
            dic['Params.'] = list(yaml.load_all(f))
    elif dic['Params.'] is not None:
        logger.info('Loaded parameters to optimize.')
    # Load reference data. Perhaps add an option to recalculate the
    # reference data.
    if dic['Ref. Data'] is None:
        logger.info('Calculating reference data.')
        dic['Ref. Data'] = calculate.process_args(options['ref'].split())
    else:
        logger.info('Loaded reference data.')
    # Load starting FF. Needs adjustment if we move beyond MM3*.
    if dic['FF'] is None:
        logger.info('Importing initial FF: {} ({})'.format(
                options['substr'], options['ffpath']))
        dic['FF'] = parameters.import_mm3_ff(
            filename=options['ffpath'], substr_name=options['substr'])
        # Remove parameters not currently being optimized.
        dic['FF'].remove_other_params(dic['Params.'])
    else:
        logger.info('Loaded initial FF: {} ({})'.format(
                dic['FF'].substr_name, dic['FF'].filename))
    # Calculate initial FF data if necessary.
    if dic['FF'].data == []:
        logger.info('Calculating initial FF data.')
        dic['FF'].calculate_data(options['calc'].split(), backup=True)
    assert len(dic['FF'].data) == len(dic['Ref. Data']), \
        'Num. data points for ref. ({}) '.format(len(dic['Ref. Data'])) + \
        'and FF data ({}) '.format(len(dic['FF'].data)) + \
        'is unequal.'
    # Determine X2 if necessary.
    if dic['FF'].x2 is None:
        logger.info('Calculating initial X2.')
        dic['FF'].calculate_x2(dic['Ref. Data'])
    return dic

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    # Process arguments.
    run(sys.argv[1:])
