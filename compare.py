#!/usr/bin/python
# Still need to add functions to make pretty print statements and logs.
'''
compare
-------
Contains code necessary to evaluate the objective function,

.. math:: \chi^2 = w^2 (x_r^2 - x_c^2)

where :math:`w` is a weight, :math:`x_r` is the reference data point's value,
and :math:`x_c` is the calculated or force field's value for the data point.

'''
from __future__ import print_function
from itertools import izip
import argparse
import logging
import logging.config
import numpy as np
import sys

import calculate
import constants as co
import datatypes

logger = logging.getLogger(__name__)

def main(args):
    parser = return_compare_parser()
    opts = parser.parse_args(args)
    r_data = calculate.main(opts.reference.split())
    c_data = calculate.main(opts.calculate.split())
    score = compare_data(r_data, c_data)

def return_compare_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--calculate', '-c', type=str, metavar = '" commands for calculate.py"',
        help=('These commands produce the FF data. Leave one space after the '
              '1st quotation mark enclosing the arguments.'))
    parser.add_argument(
        '--reference', '-r', type=str, metavar='" commands for calculate.py"',
        help=('These commands produce the QM/reference data. Leave one space '
              'after the 1st quotation mark enclosing the arguments.'))
    parser.add_argument(
        '--output', '-o', type=str, metavar='filename', 
        help='Write pretty output filename.')
    parser.add_argument(
        '--print', '-p', action='store_true', dest='doprint',
        help='Print pretty output.')
    return parser

def compare_data(r_data, c_data):
    r_data = sorted(r_data, key=datatypes.datum_sort_key)
    c_data = sorted(c_data, key=datatypes.datum_sort_key)
    r_data = np.array(r_data)
    c_data = np.array(c_data)
    zero_energies(r_data)
    correlate_energies(r_data, c_data)
    import_weights(r_data)
    return calculate_score(r_data, c_data)

def zero_energies(data):
    # Go one data type at a time.
    # We do so because the group numbers are only unique within a given data
    # type.
    for energy_type in ['e', 'eo']:
        # Determine the unique group numbers.
        indices = np.where([x.typ == energy_type for x in data])[0]
        unique_group_nums = set([x.idx_1 for x in data[indices]])
        # Loop through the unique group numbers.
        for unique_group_num in unique_group_nums:
            # Pick out all data points that are unique to this data type
            # and group number.
            more_indices = np.where(
                [x.typ == energy_type and x.idx_1 == unique_group_num
                 for x in data])[0]
            # Find the zero for this grouping.
            zero = min([x.val for x in data[more_indices]])
            for ind in more_indices:
                data[ind].val -= zero

def correlate_energies(r_data, c_data):
    for energy_type in ['e', 'eo']:
        indices = np.where([x.typ == energy_type for x in r_data])[0]
        unique_group_nums = set([x.idx_1 for x in r_data[indices]])
        for unique_group_num in unique_group_nums:
            more_indices = np.where(
                [x.typ == energy_type and x.idx_1 == unique_group_num
                 for x in r_data])[0]
            zero, zero_ind = min((x.val, i) for i, x in enumerate(r_data[more_indices]))
            zero_ind = more_indices[zero_ind]
            # Wow, that was a lot of work to get the index of the zero.
            # Now, we need to get that same sub list, and update the calculated
            # data. As long as they are sorted the same, the indices should
            # match up.
            zero = c_data[zero_ind].val
            for ind in more_indices:
                c_data[ind].val -= zero

def import_weights(data):
    for datum in data:
        if datum.wht is None:
            if datum.typ == 'eig':
                if datum.idx_1 == datum.idx_2 == 1:
                    datum.wht = co.WEIGHTS['eig_i']
                elif datum.idx_1 == datum.idx_2:
                    datum.wht = co.WEIGHTS['eig_d']
                elif datum.idx_1 != datum.idx_2:
                    datum.wht = co.WEIGHTS['eig_o']
            else:
                datum.wht = co.WEIGHTS[datum.typ]

# Need to add some pretty print outs for this.
def calculate_score(r_data, c_data):
    score = 0.
    for r_datum, c_datum in izip(r_data, c_data):
        # Perhaps add a checking option here to ensure all the attributes
        # of each data point match up.
        # When we're talking about torsions, need to make sure that the
        # difference between -179 and 179 is 2, not 358.
        if r_datum.typ == 't':
            diff = abs(r_datum.val - c_datum.val)
            if diff > 180.:
                diff = 360. - diff
        # Simpler for other data types.
        else:
            diff = r_datum.val - c_datum.val
        individual_score = r_datum.wht**2 * diff**2
        score += individual_score
    logger.log(15, 'SCORE: {}'.format(score))
    return score
            
if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
