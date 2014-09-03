#!/usr/bin/python
import argparse
import calculate
import logging
import logging.config
import os
import parameters
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

# Setup logging.
logger = logging.getLogger(__name__)

def calc_data_for_ffs(ffs, calc_args=None, ref_data=None, recalc=False,
                      backup=True):
    '''
    Need to make the importing and exporting part general beyond MM3*.

    calc_args   = Arguments for calculate.py.
    ref_data    = List of reference data. If given, also calculate X2.
    recalc      = If True, recalculate data even if data already exists.
    backup      = True to backup and restore the current FF file.
    '''
    if len(ffs) == 0:
        logger.warning('Call to evaluate FFs, but length of FFs is 0.')
        return ffs
    if backup:
        logger.debug('Backing up {} in memory.'.format(ffs[0].filename))
        # Here we shouldn't trim the list of parameters, which we
        # normally do, because there is no gaurantee that each FF
        # has the same parameters (self.params).
        orig_ff = parameters.import_mm3_ff(
            filename=ffs[0].filename, substr_name=ffs[0].substr_name)
    try:
        for ff in ffs:
            if recalc:
                ff.calculate_data(calc_args, backup=False)
            else:
                if ff.data == []:
                    ff.calculate_data(calc_args, backup=False)
            if ref_data:
                assert len(ff.data) == len(ref_data), \
                    'Num. dat points for ref. ({}) '.format(len(ref_data)) + \
                    'and FF data ({}) '.format(len(ff.data)) + \
                    'are unequal.'
                ff.calculate_x2(ref_data)
    finally:
        if backup:
            logger.debug('Restoring original {}.'.format(ffs[0].filename))
            # Because we didn't trim the parameters, this adds a bunch of
            # debug messages. Oh well.
            parameters.export_mm3_ff(params=orig_ff.params,
                                     in_filename=orig_ff.filename,
                                     out_filename=orig_ff.filename)
    return ffs

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Evaluates the penalty function for a single set of ' +
        'reference and calculated data. Not really used in the rest of the ' +
        "code, but I thought it'd be nice to be able to do this from the " +
        'command line (w/o opening Python interpreter). This just uses ' +
        'the existing force field file. Nothing fancy.')
    calc_opts = parser.add_argument_group('Arguments for calculate.py')
    share_opts = parser.add_argument_group('Arguments added to both -r and -c')
    calc_opts.add_argument(
        '--calc', '-c', type=str, metavar='"calculate.py arguments"',
        help='Commands for calculate.py to determine the calculated FF data.')
    calc_opts.add_argument(
        '--ref', '-r', type=str, metavar='"calculate.py arguments"', 
        help='Commands for calculate.py to determine the reference data.')
    parser.add_argument(
        '--output', '-o', type=str, metavar='filename', const='print',
        nargs='?',
        help='Write data to output file or print if no filename is given. ' +
        'Different from calculate.py output because it gives you the FF ' +
        'and reference data side by side.')
    share_opts.add_argument(
        '--dir', type=str, metavar='relative/path/to/data', default=os.getcwd(),
        help='Set directory where force field calculations will be ' +
        'performed. Directory should include teh necessary data and force ' +
        'field files for the calculations.')
    share_opts.add_argument(
        '--norun', action='store_true', help="Don't run FF calculations. " +
        'Assumes the output from the calculation is already present.')
    share_opts.add_argument(
        '--weights', type=str, metavar='relative/path/weights.yaml',
        default='options/weights.yaml',
        help='Relative path to weights YAML file. Uses options/weights.yaml ' +
        'by default.')
    options = vars(parser.parse_args(args))
    options['calc'] = options['calc'].split()
    options['ref'] = options['ref'].split()
    # Account for shared arguments.
    if options['dir']:
        options['calc'].extend(['--dir', options['dir']])
        options['ref'].extend(['--dir', options['dir']])
    if options['norun']:
        options['calc'].append('--norun')
        options['ref'].append('--norun')
    if options['weights']:
        options['calc'].extend(['--weights', options['weights']])
        options['ref'].extend(['--weights', options['weights']])
    # All the data we'll need. In case I add backups later.
    dic = {'Ref. Data': None,
           'Calc. Data': None}
    dic['Ref. Data'] = calculate.process_args(options['ref'])
    dic['Calc. Data'] = calculate.process_args(options['calc'])
    x2, dic['Ref. Data'], dic['Calc. Data'] = calculate.calculate_x2(
        dic['Ref. Data'], dic['Calc. Data'])
    if options['output']:
        lines = ['X2: {}'.format(x2), '']
        for r_d, c_d in zip(
                sorted(dic['Ref. Data'], key=calculate.sort_datum),
                sorted(dic['Calc. Data'], key=calculate.sort_datum)):
            lines.append('{0:<20}{1:<20}{2:>10.4f}{3:>22.6f}{4:>22.6f}'.format(
                r_d.name, c_d.name, r_d.weight, r_d.value, c_d.value))
        if options['output'] == 'print':
            for line in lines:
                print line
        else:
            with open(options['output'], 'w') as f:
                for line in lines:
                    f.write(line + '\n')

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        config = yaml.load(f)
    logging.config.dictConfig(config)
    # Execute.
    process_args(sys.argv[1:])
