#!/usr/bin/python
'''
Short script to evaluate the objective function.
'''
from collections import defaultdict
import argparse
import itertools
import logging
import sys
import yaml

from calculate import run_calculate
from datatypes import Datum, datum_sort_key
import constants as cons

logger = logging.getLogger(__name__)

def parse(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--calculate', '-c', type=str,
        metavar = '" commands for calculate.py"',
        help=('These commands produce the calculated data. Leave one space '
              'after the 1st quotation mark enclosing the arguments.'))
    parser.add_argument(
        '--reference', '-r', type=str,
        metavar='" commands for calculate.py"',
        help=('These commands produce the reference data. Leave one space '
              'after the 1st quotation mark enclosing the arguments.'))
    parser.add_argument(
        '--output', '-o', type=str, metavar='filename.txt',
        help='Write data to file.')
    parser.add_argument(
        '--print', '-p', action='store_true', dest='doprint',
        help='Print data.')
    opts = parser.parse_args(args)
    return opts

def convert_energies(data_cal, data_ref):
    # duplicating this is such a bullshit fix
    energies_ref = [x for x in data_ref if x.dtype == 'energy']
    energies_cal = [x for x in data_cal if x.dtype == 'energy']
    groups_ref = defaultdict(list)
    for datum in energies_ref:
        groups_ref[datum.group].append(datum)
    groups_cal = defaultdict(list)
    for datum in energies_cal:
        groups_cal[datum.group].append(datum)
    for gnum_ref, gnum_cal in itertools.izip(sorted(groups_ref), sorted(groups_cal)):
        group_energies_ref = groups_ref[gnum_ref]
        group_energies_cal = groups_cal[gnum_cal]
        value, index = min((datum.value, index) for index, datum in enumerate(group_energies_ref))
        minimum_cal = group_energies_cal[index].value
        for datum in group_energies_cal:
            datum.value -= minimum_cal

    energies_ref = [x for x in data_ref if x.dtype == 'energy2']
    energies_cal = [x for x in data_cal if x.dtype == 'energy2']
    groups_ref = defaultdict(list)
    for datum in energies_ref:
        groups_ref[datum.group].append(datum)
    groups_cal = defaultdict(list)
    for datum in energies_cal:
        groups_cal[datum.group].append(datum)
    for gnum_ref, gnum_cal in itertools.izip(sorted(groups_ref), sorted(groups_cal)):
        group_energies_ref = groups_ref[gnum_ref]
        group_energies_cal = groups_cal[gnum_cal]
        value, index = min((datum.value, index) for index, datum in enumerate(group_energies_ref))
        minimum_cal = group_energies_cal[index].value
        for datum in group_energies_cal:
            datum.value -= minimum_cal

def import_steps(params):
    for param in params:
        if isinstance(cons.steps[param.ptype], basestring):
            param.step = float(cons.steps[param.ptype]) * param.value
        else:
            param.step = cons.steps[param.ptype]
        if param.step  == 0.0:
            param.step = 0.1
# def import_steps(params, yamlfile='steps.yaml', **kwargs):
#     '''
#     Grabs step sizes for parameters from a yaml file. Can also take
#     arguments to override the dictionary.
#     '''
#     with open(yamlfile, 'r') as f:
#         steps = yaml.load(f)
#     for key,value in kwargs.iteritems():
#         steps[key] = value
#     for param in params:
#         param.step = steps[param.ptype]

def import_weights(data, yamlfile='weights.yaml', **kwargs):
    with open(yamlfile, 'r') as f:
        weights = yaml.load(f)
    for key, value in kwargs.iteritems():
        weights[key] = value
    for datum in data:
        if datum.dtype == 'eig' or datum.dtype == 'eigz':
            if datum.i == 0 and datum.j == 0:
                datum.weight = weights['eig_i']
            elif datum.i == datum.j:
                datum.weight = weights['eig_d']
            else:
                datum.weight = weights['eig_o']
        elif datum.dtype == 'hess':
            if datum.i == datum.j:
                datum.weight = weights['hess_11']
            else:
                datum.weight = weights['hess']
        else:
            datum.weight = weights[datum.dtype]
        
def calc_x2(data_cal, data_ref, output=None, doprint=False):
    if isinstance(data_cal, list):
        assert isinstance(data_cal[0], Datum), \
            "attempted to calculate objective function using an object that isn't Datum"
    elif isinstance(data_cal, basestring):
        data_cal = run_calculate(data_cal.split())
    else:
        raise Exception('failed to determine ff calculated data')
    if isinstance(data_ref, list):
        assert isinstance(data_ref[0], Datum), \
            "attempted to calculate objective function using an object that isn't Datum"
    elif isinstance(data_ref, basestring):
        data_ref = run_calculate(data_ref.split())
    else:
        raise Exception('failed to determine reference data')
    assert len(data_cal) == len(data_ref), "number of reference and ff calculated data points don't match"
    data_cal = sorted(data_cal, key=datum_sort_key)
    data_ref = sorted(data_ref, key=datum_sort_key)
    convert_energies(data_cal, data_ref)
    import_weights(data_ref)

    total_x2 = 0.
    if output or doprint:
        separate_x2 = defaultdict(float)
        lines = []
        header = '{0:<20} {1:>16} {2:<20} {3:>16} {4:>16} {5:>16}'.format(
            'ref', 'ref value', 'cal', 'cal value', 'weight', 'x2')
        lines.append(header)
        lines.append('-' * len(header))
    for datum_ref, datum_cal in itertools.izip(data_ref, data_cal):
        if datum_ref.dtype == 'torsion' or datum_cal.dtype == 'torsion':
            delta = abs(datum_ref.value - datum_cal.value)
            if delta > 180.:
                delta = 360. - delta
        else:
            delta = datum_ref.value - datum_cal.value
        single_x2 = datum_ref.weight**2 * (delta)**2
        total_x2 += single_x2
        if output or doprint:
            separate_x2[datum_ref.dtype] += single_x2
            lines.append(
                '{0:<20} {1:>16.6f} {2:<20} {3:>16.6f} {4:>16.6f} {5:>16.6f}'.format(
                    datum_ref.name, datum_ref.value, datum_cal.name,
                    datum_cal.value, datum_ref.weight, single_x2))
            
    if output or doprint:
        lines.append('-' * len(header))
        lines.append('')
        lines.append('total x2: {}'.format(total_x2))
        for dtype, value in separate_x2.iteritems():
            lines.append('{} x2: {}'.format(dtype, value))
        if doprint:
            for line in lines:
                print(line)
        elif output:
            lines = [x + '\n' for x in lines]
            with open(output, 'w') as f:
                f.writelines(lines)
    return total_x2

if __name__ == '__main__':
    import logging.config
    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)
   
    opts = parse(sys.argv[1:])
    calc_x2(opts.calculate, opts.reference, opts.output, opts.doprint)
