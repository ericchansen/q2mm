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
from __future__ import absolute_import
from collections import defaultdict
import sys
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
    r_dict = data_by_type(r_data)
    c_dict = data_by_type(c_data)
    r_dict, c_dict = trim_data(r_dict, c_dict)
    score = compare_data(r_dict, c_dict, output=opts.output, doprint=opts.print)
#    score = compare_data(r_data, c_data)
    # Pretty readouts. Maybe opts.output could have 3 values:
    # True, False or None
    # Then I wouldn't need 2 if statements here.
#    if opts.output or opts.print:
#        pretty_data_comp(r_data, c_data, output=opts.output, doprint=opts.print)
    logger.log(1, '>>> score: {}'.format(score))

def trim_data(dict1,dict2):
    """
    Within a data type of these dictionaries, data that is not present in both
    lists.
    """
    for typ in dict1:
        if typ == 't':
            to_remove = []
            # Sometimes calculations are compressed into one macromodel command
            # file. This results in the 'pre' and 'opt' structures in the same
            # file. Geometeric data from each of these would correspond to r
            # and c data for 'pre' and 'opt' structures,respectively. This would
            # mean that when trying to compare two data poitns from r and c they
            # would not have the exact same lable since they are from different
            # structure indicies in the the file. So instaed of just trying to
            # see if the labels are the same, we should try and pull out the
            # filename and the atoms that comprise that data point. This is what
            # is done with the regex below.
            for d1 in dict1[typ]:
                #if not any(x.lbl == d1.lbl for x in dict2[typ]):
                if not any(co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d1.lbl)[1] and 
                           co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d1.lbl)[2] 
                           for x in dict2[typ]):
                    to_remove.append(d1)
            for d2 in dict2[typ]:
                #if not any(x.lbl == d2.lbl for x in dict1[typ]):
                if not any(co.RE_T_LBL.split(x.lbl)[1] == co.RE_T_LBL.split(d2.lbl)[1] and 
                           co.RE_T_LBL.split(x.lbl)[2] == co.RE_T_LBL.split(d2.lbl)[2] 
                           for x in dict1[typ]):
                    to_remove.append(d2)
            for datum in to_remove:
                if datum in dict1[typ] and datum in dict2[typ]:
                    raise AssertionError("The data point that is flagged to be \
                    removed is present in both data sets.")
                # We may want to keep track of the data that is removed. 
                if datum in dict1[typ]:
                    dict1[typ].remove(datum) 
                if datum in dict2[typ]:
                    dict2[typ].remove(datum) 
            if to_remove:
                logger.log(20, '>>> Removed Data: {}'.format(len(to_remove)))
        dict1[typ] = np.array(dict1[typ], dtype=datatypes.Datum)
        dict2[typ] = np.array(dict2[typ], dtype=datatypes.Datum)
    return dict1, dict2

def data_by_type(data_iterable):
    """
    Takes a iterable of data and creates a dictionary of data types and sets
    up an array.
    """

    data_by_typ = {}
    for datum in data_iterable:
        if datum.typ not in data_by_typ:
            data_by_typ[datum.typ] = []
        data_by_typ[datum.typ].append(datum)
    # Parts of the code rely on the data to be in an array form and so we must
    # make the dictionary an array. This doesn't make sense to have here since
    # we may have to remove data in trim_data().
    #for typ in data_by_typ:
    #    data_by_typ[typ] = np.array(data_by_typ[typ], dtype=datatypes.Datum)
    return data_by_typ

def compare_data(r_dict, c_dict, output=None, doprint=False):
    """
    This function was formerly called pretty_data_comp().
    Now only one function is needed to calculate the score, with the options to
    print and direct to an output available.
    """

    strings = []
    strings.append('--' + ' Label '.ljust(30, '-') +
                   '--' + ' Weight '.center(7, '-') +
                   '--' + ' R. Value '.center(11, '-') +
                   '--' + ' C. Value '.center(11, '-') +
                   '--' + ' Score '.center(11, '-') +
                   '--' + ' Row ' + '--')
    score_typ = defaultdict(float)
    num_typ = defaultdict(int)
    score_tot = 0.
    total_num = 0
    # Do we want the datatypes always reported in the same order? This allows
    # the same order of data type to be printed the same everytime.
    data_types = []
    for typ in r_dict:
        data_types.append(typ)
    data_types.sort()
    total_num_energy = 0
    for typ in data_types:
        if typ in ['e','eo','ea','eao']:
            total_num_energy += len(r_dict[typ])
    for typ in data_types:
        total_num += int(len(r_dict[typ]))
        if typ in ['e','eo','ea','eao']:
            correlate_energies(r_dict[typ],c_dict[typ])
        import_weights(r_dict[typ])
        for r,c in zip(r_dict[typ],c_dict[typ]):
            if c.typ == 't':
                diff = abs(r.val - c.val)
                if diff > 180.:
                    diff = 360. - diff
            else:
                diff = r.val - c.val
            #score = (r.wht**2 * diff**2)
            if typ in ['e', 'eo', 'ea', 'eao']:
                score = (r.wht**2 * diff**2)/total_num_energy
            else:
                score = (r.wht**2 * diff**2)/len(r_dict[typ])
            score_tot += score
            score_typ[c.typ] += score
            num_typ[c.typ] += 1
            if c.typ == 'eig':
                if c.idx_1 == c.idx_2:
                    if r.val < 1100:
                        score_typ[c.typ + '-d-low'] += score
                        num_typ[c.typ + '-d-low'] += 1
                    else:
                        score_typ[c.typ + '-d-high'] += score
                        num_typ[c.typ + '-d-high'] += 1
                else:
                    score_typ[c.typ + '-o'] += score
                    num_typ[c.typ + '-o'] += 1
            strings.append('  {:<30}  {:>7.2f}  {:>11.4f}  {:>11.4f}  {:>11.4f}  '\
                       '{!s:>5} '.format(
                        c.lbl, r.wht, r.val, c.val, score, c.ff_row))
    strings.append('-' * 89)
    strings.append('{:<20} {:20.4f}'.format('Total score:', score_tot))
    strings.append('{:<30} {:10d}'.format('Total Num. data points:', total_num))
    for k, v in num_typ.items():
        strings.append('{:<30} {:10d}'.format(k + ':', v))
    strings.append('-' * 89)
    for k, v in score_typ.items():
        strings.append('{:<20} {:20.4f}'.format(k + ':', v))
    if output:
        with open(output, 'w') as f:
            for line in strings:
                f.write('{}\n'.format(line))
    if doprint:
        for line in strings:
            print(line)
    return score_tot

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

def compare_data_old(r_data, c_data):
    """
    *** This was the original function to compare the two tuples of data before
    *** opting in for a dictionary of data types.
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
                    if datum.val < 1100:
                        datum.wht = co.WEIGHTS['eig_d_low']
                    else:
                        datum.wht = co.WEIGHTS['eig_d_high']
                elif datum.idx_1 != datum.idx_2:
                    datum.wht = co.WEIGHTS['eig_o']
            else:
                datum.wht = co.WEIGHTS[datum.typ]

def calculate_score(r_data, c_data):
    """
    *** Depracated: All of this is in compare_data()
    Calculates the objective function score.
    """
    score_tot = 0.
    for r_datum, c_datum in zip(r_data, c_data):
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
        logger.log(1, '>>> {} {} {}'.format(r_datum, c_datum, score_ind))

    logger.log(5, 'SCORE: {}'.format(score_tot))
    return score_tot

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
