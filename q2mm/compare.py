#!/usr/bin/env python
'''
compare
-------
Contains code necessary to evaluate the objective function,

.. math:: \chi^2 = w^2 (x_r - x_c)^2

where :math:`w` is a weight, :math:`x_r` is the reference data point's value,
and :math:`x_c` is the calculated or force field's value for the data point.

'''
from __future__ import print_function
from collections import defaultdict
import sys
if (sys.version_info < (3, 0)):
    from itertools import izip
import argparse
from argparse import RawTextHelpFormatter
import logging
import logging.config
import numpy as np

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
    # Pretty readouts. Maybe opts.output could have 3 values:
    # True, False or None
    # Then I wouldn't need 2 if statements here.
    if opts.output or opts.print:
        pretty_data_comp(r_data, c_data, output=opts.output, doprint=opts.print)
    logger.log(1, '>>> score: {}'.format(score))

def pretty_data_comp(r_data, c_data, output=None, doprint=False):
    """
    Recalculates score along with making a pretty output.

    Can't rely upon reference Datum attributes anymore due to how reference
    data files are implemented.
    """
    strings = []
    strings.append('--' + ' Label '.ljust(30, '-') +
                   '--' + ' Weight '.center(7, '-') +
                   '--' + ' R. Value '.center(11, '-') +
                   '--' + ' C. Value '.center(11, '-') +
                   '--' + ' Score '.center(11, '-') +
                   '--' + ' Row ' + '--')
    score_typ = defaultdict(float)
    score_tot = 0.
    if (sys.version_info > (3, 0)):
        rc_zip = zip(r_data, c_data)
    else:
        rc_zip = izip(r_data, c_data)
    for r, c in rc_zip:
        # logger.log(1, '>>> {} {}'.format(r, c))
        # Double check data types.
        # Had to change to check from the FF data type because reference data
        # files may be missing this information.
        if c.typ == 't':
            diff = abs(r.val - c.val)
            if diff > 180.:
                diff = 360. - diff
        else:
            diff = r.val - c.val
        # Calculate score.
        score = r.wht**2 * diff**2
        # Update total.
        score_tot += score
        # Update dictionary.
        score_typ[c.typ] += score
        # Also calculate the score for off-diagonal vs. diagonal elements of the
        # eigenmatrix.
        if c.typ == 'eig':
            if c.idx_1 == c.idx_2:
                score_typ[c.typ + '-d'] += score
            else:
                score_typ[c.typ + '-o'] += score
        strings.append('  {:<30}  {:>7.2f}  {:>11.4f}  {:>11.4f}  {:>11.4f}  '\
                       '{!s:>5}  '.format(
                        c.lbl, r.wht, r.val, c.val, score, c.ff_row))
    strings.append('-' * 89)
    strings.append('{:<20} {:20.4f}'.format('Total score:', score_tot))
    strings.append('{:<20} {:20d}'.format('Num. data points:', len(r_data)))
    strings.append('-' * 89)
    if (sys.version_info > (3, 0)):
        score_typ_iter = iter(score_typ.items())
    else:
        score_typ_iter = score_typ.iteritems()
    for k, v in score_typ_iter:
        strings.append('{:<20} {:20.4f}'.format(k + ':', v))
    if output:
        with open(output, 'w') as f:
            for line in strings:
                f.write('{}\n'.format(line))
    if doprint:
        for line in strings:
            print(line)

def return_compare_parser():
    """
    Arguments parser for compare.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=RawTextHelpFormatter)
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
        help='Write pretty output to filename.')
    parser.add_argument(
        '--print', '-p', action='store_true', dest='print',
        help='Print pretty output.')
    return parser

def compare_data(r_data, c_data):
    """
    Calculates the objective function score after ensuring the energies are
    set properly and that the weights are all imported.
    """
    # r_data = np.array(sorted(r_data, key=datatypes.datum_sort_key))
    # c_data = np.array(sorted(c_data, key=datatypes.datum_sort_key))
    # if zero:
    #     zero_energies(r_data)
    assert len(r_data) == len(c_data), \
        "Length of reference data and FF data doesn't match!"
    correlate_energies(r_data, c_data)
    import_weights(r_data)
    return calculate_score(r_data, c_data)

# Energies should be zeroed inside calculate now.
# Save this in case that ever changes.
# def zero_energies(data):
#     logger.log(1, '>>> zero_energies <<<')
#     # Go one data type at a time.
#     # We do so because the group numbers are only unique within a given data
#     # type.
#     for energy_type in ['e', 'eo']:
#         # Determine the unique group numbers.
#         indices = np.where([x.typ == energy_type for x in data])[0]
#         # logger.log(1, '>>> indices: {}'.format(indices))
#         # logger.log(1, '>>> data[indices]: {}'.format(data[indices]))
#         # logger.log(1, '>>> [x.idx_1 for x in data[indices]]: {}'.format(
#         #         [x.idx_1 for x in data[indices]]))
#         unique_group_nums = set([x.idx_1 for x in data[indices]])
#         # Loop through the unique group numbers.
#         for unique_group_num in unique_group_nums:
#             # Pick out all data points that are unique to this data type
#             # and group number.
#             more_indices = np.where(
#                 [x.typ == energy_type and x.idx_1 == unique_group_num
#                  for x in data])[0]
#             # Find the zero for this grouping.
#             zero = min([x.val for x in data[more_indices]])
#             for ind in more_indices:
#                 data[ind].val -= zero

def correlate_energies(r_data, c_data):
    """
    Finds the indices corresponding to groups of energies from the FF data set.
    Uses those same indices to locate the matching energies in the reference
    data set.

    THIS MEANS THAT THE TWO DATA SETS MUST BE ALIGNED PROPERLY!

    Determines the minimum energy in the reference data set, and sets that to
    zero in the FF data set.
    """
    for indices in select_group_of_energies(c_data):
        # Search based on FF data because the reference data may be read from
        # a file and lack some of the necessary attributes.
        zero, zero_ind = min(
            (x.val, i) for i, x in enumerate(r_data[indices]))
        zero_ind = indices[zero_ind]
        # Now, we need to get that same sub list, and update the calculated
        # data. As long as they are sorted the same, the indices should
        # match up.
        zero = c_data[zero_ind].val
        for ind in indices:
            c_data[ind].val -= zero

# This is outdated now. Most of this is handled inside calculate.
# 6/29/16 - Actually, now this should be unnecessary simply because the new
#           method requires that reference and FF data are aligned. That being
#           said, this is probably worth saving anyway.
# def correlate_energies(r_data, c_data):
#     logger.log(1, '>>> correlate_energies <<<')
#     for indices in select_group_of_energies(r_data):
#         if r_data[indices[0]].typ in ['e', 'eo']:
#             zero, zero_ind = min(
#                 (x.val, i) for i, x in enumerate(r_data[indices]))
#             zero_ind = indices[zero_ind]
#             zero = c_data[zero_ind].val
#             for ind in indices:
#                 c_data[ind].val -= zero
#         elif r_data[indices[0]].typ in ['ea', 'eao']:
#             avg = sum([x.val for x in r_data[indices]])/len(r_data[indices])
#             for ind in indices:
#                 r_data[ind].val -= avg
#             avg = sum([x.val for x in c_data[indices]])/len(c_data[indices])
#             for ind in indices:
#                 c_data[ind].val -= avg

def select_group_of_energies(data):
    """
    Used to get the indices (numpy.array) for a single group of energies.
    """
    for energy_type in ['e', 'eo']:
        # Get all energy indices.
        indices = np.where([x.typ == energy_type for x in data])[0]
        # logger.log(1, '>>> indices: {}'.format(indices))
        # Get the unique group numbers.
        # logger.log(1, '>>> data: {}'.format(data))
        unique_group_nums = set([x.idx_1 for x in data[indices]])
        for unique_group_num in unique_group_nums:
            # Get all the indicies for the given energy type and for a single
            # group.
            more_indices = np.where(
                [x.typ == energy_type and x.idx_1 == unique_group_num
                 for x in data])[0]
            yield more_indices

def import_weights(data):
    """
    Imports weights for various data types.

    Weights can be set in constants.WEIGHTS.
    """
    # Check each data point individually for weights.
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

def calculate_score(r_data, c_data):
    """
    Calculates the objective function score.
    """
    score_tot = 0.
    if (sys.version_info > (3, 0)):
        rc_zip = zip(r_data, c_data)
    else:
        rc_zip = izip(r_data, c_data)
    for r_datum, c_datum in rc_zip:
        # Could add a check here to assure that the data points are aligned.
        # Ex.) assert r_datum.ind_1 == c_datum.ind_1, 'Oh no!'

        # For torsions, ensure the difference between -179 and 179 is 2, not
        # 358.
        if c_datum.typ == 't':
            diff = abs(r_datum.val - c_datum.val)
            if diff > 180.:
                diff = 360. - diff
        else:
            diff = r_datum.val - c_datum.val
        score_ind = r_datum.wht**2 * diff**2
        score_tot += score_ind
        # logger.log(1, '>>> {} {} {}'.format(r_datum, c_datum, score_ind))

    logger.log(5, 'SCORE: {}'.format(score_tot))
    return score_tot

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
