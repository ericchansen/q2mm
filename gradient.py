#!/usr/bin/python
import argparse
import calculate
import copy
import evaluate
import logging
import logging.config
import loop
import numpy as np
import parameters
from parameters import BaseFF, MM3FF, BaseParam, MM3Param
import pickle
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

logger = logging.getLogger(__name__)

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Solve for new parameters using gradient based methods. ' +
        'Start by doing central differentiation of parameters. The ' +
        'derivatives from this can be directly used in the Newton-Raphson ' +
        'methods. Otherwise form the Jacobian J and residual vector r, ' +
        'and calculate A = J.T J and b = J.T r. Then ' +
        'solve for new parameters by solving for x in A x = b.')
    calc_opts = parser.add_argument_group('Arguments for calculate.py')
    limit_opts = parser.add_argument_group('Limits to parameter changes')
    opt_opts = parser.add_argument_group('Optimization methods')
    save_opts = parser.add_argument_group('Load/save options')
    share_opts = parser.add_argument_group('Shared options')
    parser.add_argument(
        '--results', type=str, metavar='relative/path/par.tot', nargs='?',
        const='par.tot',
        help='Generates a results file from the reference data and central ' +
        'differentiation results. Format mimics the previously used par.tot.')
    parser.add_argument(
        '--solver', type=str, default='lstsq',
        choices = ['solve', 'lstsq', 'cholesky'], help='Select the method ' +
        'used to solve A x = b for x. solve uses numpy.linalg.solve, ' +
        'lstsq uses numpy.linalg.lstsq, and cholesky uses ' +
        'scipy.linalg.cholesky followed by scipy.linalg.cho_solve. Default ' +
        'is lstsq.')
    calc_opts.add_argument(
        '--calc', '-c', type=str, metavar='"calculate.py arguments"',
        help='Commands for calculate.py to determine the calculated FF data.')
    calc_opts.add_argument(
        '--ref', '-r', type=str, metavar='"calculate.py arguments"', 
        help='Commands for calculate.py to determine the reference data.')
    limit_opts.add_argument(
        '--cutoff', '-x', type=str, metavar='0.01,100', nargs='?',
        const='0.01,100',
        help='Use a cutoff to skip output of trial solutions for which ' +
        'the total radius of parameter change falls outside the limits. ' +
        'The limits are the two values provided in the list. If --cutoff ' +
        'is specified without the following list, 0.01 and 100 are used. ' +
        'If only one value is given, the inverse of that value is added to ' +
        'this option takes precedence over --radius.')
    limit_opts.add_argument(
        '--radius', '-ra', type=str, metavar='1',
        help='When the total radius of the parameter change exceeds ' +
        'the maximum provided here, the parameter modifications are ' +
        'scaled to make the final radius equal to the maximum.')
    opt_opts.add_argument(
        '--basic', '-b', action='store_true', help='Simply solve for x in A ' +
        'x = b using the method specified by --solver.')
    opt_opts.add_argument(
        '--lagrange', '-l', type=str, metavar='10,1,0.1,0.01',
        const='10,1,0.1,0.01', nargs='?',
        help='Use the values in the list as Lagrange dampening factors to ' +
        'be added to the diagonal elements of A before solving A x = b. Each ' +
        'factor in the list generates one set of parameters. If no list of ' +
        'factors is given, uses 10, 1, 0.1, and 0.01 by default.')
    opt_opts.add_argument(
        '--levenberg', '-lm', type=str, metavar='10,1,0.1,0.01',
        const='10,1,0.1,0.01', nargs='?',
        help='Use the values in the list as Levenberg-Marquardt l factors. ' +
        'The diagonal elements of A are multiplied by l + 1 before ' +
        'solving A x = b. Each factor in the list generates one set of ' +
        'parameters. If not list of factors is given, uses 10, 1, 0.1, and ' +
        '0.01 by default.')
    opt_opts.add_argument(
        '--newton', '-nr', action='store_true', help='Solve for new ' +
        'parameters using the Newton-Raphson method and the results ' +
        'from central differentiation.')
    opt_opts.add_argument(
        '--svd', '-s', type=str, metavar='1,2,3', const='nothresh', nargs='?',
        help='Use SVD to modify A = U S V.H. By default, all singular ' +
        'values (elements of the diagonal matrix S) are dropped in turn ' +
        'before reforming A to generate many solutions by solving ' +
        'A x = b. If the optional argument list is supplied, instead of ' +
        'dropping all singular values ones at a time, remove all singular ' +
        'values below the highest threshold value. Then move on to the next ' +
        'lower threshold value and repeat. Each threshold in the list ' +
        'generates one solution. These solutions may be identical if the ' +
        'threshold values in the list are very similar.')
    save_opts.add_argument(
        '--init', type=str, metavar='relative/path/gradient.pickle',
        nargs='?', const='gradient.pickle',
        help='Do the setup calculations, save that data (by default to ' +
        "gradient.pickle), but don't actually start the simplex optimization.")
    save_opts.add_argument(
        '--load', type=str, metavar='relative/path/gradient.pickle',
        nargs='?', const='gradient.pickle',
        help='Load intermediate results from the pickle. If no argument ' +
        'is given, loads from gradient.pickle.')
    save_opts.add_argument(
        '--save', type=str, metavar='relative/path/gradient.pickle',
        nargs='?', const='gradient.pickle',
        help='Save data to the pickle. If no argument, saves to ' +
        'gradient.pickle.')
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
    # More argument handling.
    if options['cutoff']:
        cutoffs = options['cutoff'].split(',')
        cutoffs = map(float, cutoffs)
        assert len(cutoffs) == 1 or len(cutoffs) == 2, \
            '--cutoff optional argument be one or two comma separated floats.'
        if len(cutoffs) == 1:
            options['cutoff'] = [cutoffs[0], 1 / cutoffs[0]]
        else:
            options['cutoff'] = cutoffs
        logger.debug('Cutoffs: {}'.format(options['cutoff']))
    return options

def run(args):
    options = process_args(args)
    # Options to load the dictionary for gradient.py. This dictionary
    # contains the necessary pieces to generate trial force fields.
    if options['load']:
        logger.info('Loading gradient.py data from {}.'.format(
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
        # Add in initial FF before sorting.
        dic['Trial FFs'] = sorted(
            dic['Trial FFs'] + [dic['FF']], key=lambda x: x.x2)
        logger.info('Best after gradient: {}'.format(
                dic['Trial FFs'][0].x2))
        logger.info('Writing best FF ({}) to {}.'.format(
                dic['Trial FFs'][0].x2, dic['Trial FFs'][0].filename))
        parameters.export_mm3_ff(
            dic['Trial FFs'][0].params,
            in_filename=dic['Trial FFs'][0].filename,
            out_filename=dic['Trial FFs'][0].filename)
    except:
        logger.error(
            'Error encountered. Backing up data to gradient.error.pickle.')
        with open('gradient.error.pickle', 'wb') as f:
            pickle.dump(dic, f)
        raise
            
def execute(options, dic):
    # Get the residual vector. Option to load residual vector?
    resid = make_residual_vector(dic['FF'].data, dic['Ref. Data'])
    # Differentiate FF parameters.
    if dic['FF'].cent_ffs == []:
        dic['FF'].cent_ffs = central_differentiation(
            dic['FF'], steps_filename=options['steps'])
    logger.debug('Calculating data for central differentiated FFs.')
    # This won't do unnecessary calculations, so it's okay to call it
    # every time, even if we are loading up data that already has these
    # results.
    dic['FF'].cent_ffs = evaluate.calc_data_for_ffs(
        dic['FF'].cent_ffs, calc_args=options['calc'].split(),
        ref_data=dic['Ref. Data'], recalc=False, backup=True)
    # Get the Jacobian. Option to load?
    jacob = make_jacobian_center(dic['FF'].cent_ffs)
    # Make old school results file. Looks like par.tot.
    if options['results']:
        logger.info('Making optional results file.')
        make_results_file(
            dic['Ref. Data'], dic['FF'], dic['FF'].cent_ffs,
            output=options['results'])
    if options['init']:
        logger.info('Saving initial results to {}.'.format(options['init']))
        with open(options['init'], 'wb') as f:
            pickle.dump(dic, f)
        return dic
    # That was all setup. Phew. We're ready to move on.
    # Here comes the generation of new parameters.
    # NR.
    if options['newton']:
        dic['FF'] = calc_derivatives(dic['FF'], dic['FF'].cent_ffs)
        changes = newton(dic['FF'], dic['FF'].cent_ffs)
        changes = check_radius(changes, options['cutoff'], options['radius'])
        if changes is not None:
            dic['Trial FFs'].append(dic['FF'].spawn_child_ff(
                    p_changes=changes, gen_method='Newton-Raphson'))
    if any([options['basic'], options['lagrange'], options['levenberg'],
            options['svd']]):
        
        A = jacob.T.dot(jacob) # A = J.T J
        b = jacob.T.dot(resid) # b = J.T r
        logger.debug('A:\n{}'.format(A))
        logger.debug('b:\n{}'.format(b))
        # Basic.
        if options['basic']:
            changes = choose_solver(A, b, options['solver'])
            changes = check_radius(
                changes, options['cutoff'], options['radius'])
            if changes is not None:
                dic['Trial FFs'].append(dic['FF'].spawn_child_ff(
                        p_changes=changes, gen_method='Basic'))
        # Lagrange.
        if options['lagrange']:
            change_sets = lagrange(A, b, factors=options['lagrange'],
                                   solver_method=options['solver'])
            for changes in change_sets:
                changes = check_radius(
                    changes, options['cutoff'], options['radius'])
                if changes is not None:
                    dic['Trial FFs'].append(dic['FF'].spawn_child_ff(
                            p_changes=changes, gen_method='Lagrange'))
        # Levenberg-Marquardt.
        if options['levenberg']:
            change_sets = levenberg(A, b, factors=options['levenberg'],
                                    solver_method=options['solver'])
            for changes in change_sets:
                changes = check_radius(
                    changes, options['cutoff'], options['radius'])
                if changes is not None:
                    dic['Trial FFs'].append(dic['FF'].spawn_child_ff(
                            p_changes=changes, 
                            gen_method='Levenberg-Marquardt'))
        # Singular value decomposition.
        if options['svd']:
            if options['svd'] == 'nothresh':
                change_sets = svd(A, b, solver_method=options['solver'])
            else:
                change_sets = svd(A, b, thresholds=options['svd'],
                                  solver_method=options['solver'])
            for changes in change_sets:
                changes = check_radius(
                    changes, options['cutoff'], options['radius'])
                if changes is not None:
                    dic['Trial FFs'].append(dic['FF'].spawn_child_ff(
                            p_changes=changes, 
                            gen_method='Singular value decomposition'))
    logger.info('{} trial gradient solutions.'.format(
            len(dic['Trial FFs'])))
    if options['save']:
        logger.info('Saving gradient.py results to {}.'.format(
                options['save']))
        with open(options['save'], 'wb') as f:
            pickle.dump(dic, f)
    return dic

def forward_differentiation(ff, steps_filename='options/steps.yaml'):
    logger.info('Forward differentiation of {} parameters.'.format(
            len(ff.params)))
    for_ffs = []
    if not all([p.step_size for p in ff.params]):
        ff.import_step_sizes(steps_filename)
    for i in xrange(len(ff.params)):
        for_ffs.append(ff.spawn_child_ff(gen_method='Forward differentiation'))
        for_ffs[-1].params[i].value += for_ffs[-1].params[i].step_size
    return for_ffs

def central_differentiation(ff, steps_filename='options/steps.yaml'):
    logger.info('Central differentiation of {} parameters.'.format(
            len(ff.params)))
    cent_ffs = []
    if not all([p.step_size for p in ff.params]):
        ff.import_step_sizes(steps_filename)
    for i in xrange(len(ff.params)):
        cent_ffs.append(ff.spawn_child_ff(gen_method='Central differentiation'))
        cent_ffs[-1].params[i].value += cent_ffs[-1].params[i].step_size
        cent_ffs.append(ff.spawn_child_ff(gen_method='Central differentiation'))
        cent_ffs[-1].params[i].value -= cent_ffs[-1].params[i].step_size
    return cent_ffs

def make_residual_vector(calc_data, ref_data):
    logger.debug('Creating ({}, 1) residual vector.'.format(len(ref_data)))
    resid = np.empty((len(ref_data), 1), dtype=float)
    for i in xrange(0, len(ref_data)):
        resid[i, 0] = ref_data[i].weight * \
            (ref_data[i].value - calc_data[i].value)
    return resid

def make_jacobian_center(cent_ffs):
    logger.debug('Creating ({}, {}) Jacobian.'.format(len(cent_ffs[0].data),
                                                      len(cent_ffs) / 2))
    jacob = np.empty(
        (len(cent_ffs[0].data) ,len(cent_ffs) / 2), dtype=float)
    for i, ind_ff in enumerate(xrange(0, len(cent_ffs), 2)):
        # i = 0, 1, 2..
        # ind_ff = 0, 2, 4, ...
        for ind_datum in xrange(0, len(cent_ffs[0].data)):
            dydp = (cent_ffs[ind_ff].data[ind_datum].value
                    - cent_ffs[ind_ff + 1].data[ind_datum].value) / 2
            jacob[ind_datum, i] = \
                cent_ffs[ind_ff].data[ind_datum].weight * dydp
    return jacob
        
def check_radius(changes, cutoffs=None, max_radius=None):
    '''
    Sets list of parameter changes to None if the the total radius of 
    change doesn't lie within the the cutoffs. Scales the parameter
    changes if they exceed the max radius of change. Cutoffs takes
    precedence.
    '''
    if changes is None:
        return changes
    radius = np.sqrt(sum([x**2 for x in changes]))
    if cutoffs:
        if radius > max(cutoffs):
            changes = None
            logger.info('Radius {} exceeds cutoff {}. Excluding.'.format(
                    radius, max(cutoffs)))
        if radius < min(cutoffs):
            changes = None
            logger.info('Radius {} below cutoff {}. Excluding.'.format(
                    radius, min(cutoffs)))
    elif max_radius:
        if radius > max_radius:
            scale_factor = max_radius / radius
            changes = [x * scale_factor for x in changes]
            logger.info('Radius {} exceeds maximum {}. '.format(radius) +
                         'Scaling parameter changes by {}.'.format(
                    max_radius, scale_factor))
    return changes

def choose_solver(A, b, method='lstsq'):
    if method == 'cholesky':
        import scipy.linalg
        cho = scipy.linalg.cholesky(A, lower=True)
        changes = scipy.linalg.cho_solve((cho, True), b)
    if method == 'lstsq':
        changes, residuals, rank, singular_values = \
            np.linalg.lstsq(A, b, rcond=10**-12)
    if method == 'solve':
        changes = np.linalg.solve(A, b)
    changes = np.concatenate(changes).tolist()
    return changes

def calc_derivatives(orig_ff, cent_ffs):
    logger.info('Calculating parameter derivatives.')
    for i in xrange(0, len(cent_ffs), 2):
        orig_ff.params[i/2].der1 = 0.5 * (cent_ffs[i].x2 - cent_ffs[i+1].x2)
        orig_ff.params[i/2].der2 = cent_ffs[i].x2 + cent_ffs[i+1].x2 - 2 \
            * orig_ff.x2
    return orig_ff
        
def newton(orig_ff, cent_ffs):
    logger.info('Optimizing with Newton-Raphson.')
    changes = []
    for i, param in enumerate(orig_ff.params):
        if param.der1 != 0:
            if param.der2 > 0.00000001:
                changes.append(- param.der1 / param.der2) # Ideal situation.
            else: # When not ideal, this is how we handle it.
                logger.warning('2nd derivative of {} is {}.'.format(
                        param, param.der2))
                if param.der1 > 0:
                    changes.append(-1)
                    logger.warning('1st derivative of {} is {} '.format(
                            param, param.der1) + 
                                   '(greater than 0). NR step set to -1.')
                else:
                    changes.append(1)
                    logger.warning('1st derivative of {} is {}' .format(
                                param, param.der1) + 
                                   '(less than/equal to 0). NR step set to 1.')
        else:
            # Exit in this case. NR failed.
            logger.warning('1st derivative of {} is zero.'.format(param))
            changes = None
            return changes
    return changes

def svd(A, b, thresholds=None, solver_method='solve'):
    '''
    thresholds = String of comma separated threshold values.
    '''
    logger.info('Optimizing with SVD.')
    change_sets = []
    U, s, V = np.linalg.svd(A)
    logger.debug('U:\n{}'.format(U))
    logger.debug('s:\n{}'.format(s))
    logger.debug('V.H:\n{}'.format(V))
    s_copy = copy.deepcopy(s)
    if thresholds:
        thresholds = sorted(map(float, thresholds.split(',')))
        logger.info('Using thresholds: {}'.format(thresholds))
        for threshold in thresholds:
            for i in xrange(0, len(s_copy)):
                if s_copy[i] < threshold:
                    s_copy[i] = 0.
            logger.debug('Modified s for threshold ' +
                         '{}:\n{}'.format(threshold, s_copy))
            reform = U.dot(np.diag(s_copy)).dot(V)
            changes = choose_solver(reform, b, solver_method)
            change_sets.append(changes)
    else:
        for i in xrange(0, len(s_copy) - 1):
            s_copy[-(i + 1)] = 0.
            logger.debug('Modified s:\n{}'.format(s_copy))
            reform = U.dot(np.diag(s_copy)).dot(V)
            changes = choose_solver(reform, b, solver_method)
            change_sets.append(changes)
    return change_sets

def lagrange(A, b, factors, solver_method='solve'):
    '''
    factors = String of comma separated floats. Used to add
              to the diagonal of A from A x = b.
    '''
    logger.info('Optimizing with addition of Lagrange multipliers.')
    change_sets = []
    factors = sorted(map(float, factors.split(',')))
    logger.info('Factors: {}'.format(factors))
    for factor in factors:
        A_copy = copy.deepcopy(A)
        A_copy[np.diag_indices_from(A_copy)] = \
            A_copy[np.diag_indices_from(A_copy)] + factor
        logger.debug('Modified A:\n{}'.format(A_copy))
        changes = choose_solver(A_copy, b, solver_method)
        change_sets.append(changes)
    return change_sets

def levenberg(A, b, factors, solver_method='solve'):
    '''
    factors = String of comma separated floats. Used to multiply
              the diagonal elements of A by (factor + 1).
    '''
    logger.info('Optimizing using Levenberg-Marquardt method.')
    change_sets = []
    factors = sorted(map(float, factors.split(',')))
    logger.info('Factors: {}'.format(factors))
    for factor in factors:
        A_copy = copy.deepcopy(A)
        A_copy[np.diag_indices_from(A_copy)] = \
            A_copy[np.diag_indices_from(A_copy)] * (1 + factor)
        logger.debug('Modified A:\n{}'.format(A_copy))
        changes = choose_solver(A_copy, b, solver_method)
        change_sets.append(changes)
    return change_sets

def make_results_file(ref_data, init_ff, cent_ffs, output=None):
    logger.info('Making optional results file: {}'.format(output))
    sorted_data_sets = []
    for ff in cent_ffs:
        sorted_data_sets.append(sorted(ff.data, key=calculate.sort_datum))
    string = ''
    for i, (r_datum, c_datum) in enumerate(zip(
        sorted(ref_data, key=calculate.sort_datum),
        sorted(init_ff.data, key=calculate.sort_datum))):
        string += '{0:<20}{1:>10.4f}{2:>22.6f}{3:>22.6f}'.format(
            r_datum.name, r_datum.weight, r_datum.value, c_datum.value)
        for data in sorted_data_sets:
            string += '\t{0:22.6f}'.format(data[i].value)
        string += '\n'
    # Write file...
    if output:
        with open(output, 'w') as f:
            f.write(string)
    # ... or just print.
    else:
        strings = string.split('\n')
        for string in strings:
            print string
        
if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    # Begin.
    run(sys.argv[1:])
