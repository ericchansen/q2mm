#!/usr/bin/python
'''
compare
-------
Contains code necessary to evaluate the objective function,

.. math:: \chi^2 = w^2 (x_r^2 - x_c^2)

where :math:`w` is a weight, :math:`x_r` is the reference data point's value,
and :math:`x_c` is the calculated or force field's value for the data point.

'''
from __future__ import print_function
from collections import defaultdict
from itertools import izip
import argparse
import logging
import logging.config
import sys

import calculate
import constants as co

logger = logging.getLogger(__name__)

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

def zero_energies(conn):
    c = conn.cursor()
    c.execute('SELECT DISTINCT typ, idx_1 FROM data WHERE typ = "energy_1" OR '
              'typ = "energy_2" OR typ = "energy_opt"')
    for set_of_energies in c.fetchall():
        c.execute("SELECT MIN(val) FROM data WHERE typ = ? AND idx_1 = ?",
                  (set_of_energies[0], set_of_energies[1]))
        zero = c.fetchone()[0]
        c.execute("UPDATE data SET val = val - ? WHERE typ = ? AND idx_1 = ?",
                  (zero, set_of_energies[0], set_of_energies[1]))
    conn.commit()

def correlate_energies(ref_conn, cal_conn):
    cr = ref_conn.cursor()
    cc = cal_conn.cursor()
    cr.execute('SELECT DISTINCT typ, idx_1 FROM data WHERE typ = "energy_1" OR '
               'typ = "energy_2" OR typ = "energy_opt"')
    for set_of_energies in cr.fetchall():
        cr.execute('SELECT * FROM data WHERE typ = ? AND idx_1 = ? ORDER BY '
                   'typ, src_1, src_2, idx_1, idx_2',
                   (set_of_energies[0], set_of_energies[1]))
        r_rows = cr.fetchall()
        r_zero, zero_idx = min((row['val'], idx) for idx, row in enumerate(r_rows))
        cc.execute('SELECT * FROM data WHERE typ = ? AND idx_1 = ? ORDER BY '
                   'typ, src_1, src_2, idx_1, idx_2',
                   (set_of_energies[0], set_of_energies[1]))
        c_rows = cc.fetchall()
        c_zero = c_rows[zero_idx]['val']
        cc.execute('UPDATE data SET val = val - ? WHERE typ = ? and idx_1 = ?',
                   (c_zero, set_of_energies[0], set_of_energies[1]))
    cal_conn.commit()

def import_weights(conn):
    c = conn.cursor()
    c.execute('SELECT DISTINCT typ FROM data')
    for typ in c.fetchall():
        # Will need to include similar code for Hessian and perhaps other
        # matrix data types.
        if typ[0] == 'eig':
            c.execute('UPDATE data set wht = ? WHERE typ = ? AND idx_1 != idx_2',
                      (co.WEIGHTS['eig_o'], typ[0]))
            c.execute(('UPDATE data set wht = ? WHERE typ = ? AND idx_1 = idx_2 '
                       'AND idx_1 != 1'),
                      (co.WEIGHTS['eig_d'], typ[0]))
            c.execute(('UPDATE data set wht = ? WHERE typ = ? AND idx_1 = 1 AND '
                      'idx_2 = 1'),
                      (co.WEIGHTS['eig_i'], typ[0]))
        else:
            c.execute('UPDATE data SET wht = ? WHERE typ = ?',
                      (co.WEIGHTS[typ[0]], typ[0]))
    conn.commit()

def compare_data(ref_conn, cal_conn, check_data=True, pretty=False):
    zero_energies(ref_conn)
    correlate_energies(ref_conn, cal_conn)
    import_weights(ref_conn)
    if pretty:
        score, pretty_str = calculate_score(
            ref_conn, cal_conn, check_data=check_data, pretty=pretty)
    else:
        score = calculate_score(ref_conn, cal_conn, check_data=check_data,
                                pretty=pretty)
    logger.log(10, 'SCORE: {}'.format(score))
    if pretty:
        return score, pretty_str
    else:
        return score

def cursor_iter(cursor, arraysize=1000):
    while True:
        results = cursor.fetchmany(arraysize)
        if not results:
            break
        for result in results:
            yield result

def calculate_score(ref_conn, cal_conn, check_data=True, pretty=False):
    cr = ref_conn.cursor()
    cc = cal_conn.cursor()
    cr.execute('SELECT * FROM data ORDER BY typ, src_1, src_2, idx_1, '
               'idx_2, atm_1, atm_2, atm_3, atm_4')
    cc.execute('SELECT * FROM data ORDER BY typ, src_1, src_2, idx_1, '
               'idx_2, atm_1, atm_2, atm_3, atm_4')
    score = 0.
    # Keeps track of contributions to objective function from particular data
    # types.
    if pretty:
        counter = defaultdict(float)
        pretty_str = pretty_data_comp(start=True)
    # Could do in smaller chunks if memory becomes an issue.
    # for r_row, c_row in izip(cr.fetchall(), cc.fetchall()):
    for r_row, c_row in izip(cursor_iter(cr), cursor_iter(cc)):
        if check_data:
            if not r_row['typ'] == c_row['typ'] and \
                    not r_row['idx_1'] == c_row['idx_1'] and \
                    not r_row['idx_2'] == c_row['idx_2'] and \
                    not r_row['atm_1'] == c_row['atm_1'] and \
                    not r_row['atm_2'] == c_row['atm_2'] and \
                    not r_row['atm_3'] == c_row['atm_3'] and \
                    not r_row['atm_4'] == c_row['atm_4']:
                raise Exception("Data isn't aligned!")
        # Need to make sure that the difference between -179 and 179 is 2, not
        # 358 when we're talking about torsions.
        if r_row['typ'] == 'Torsion':
            diff = abs(r_row['val'] - c_row['val'])
            if diff > 180.:
                diff = 360. - diff
        # Standard case. Just look at the difference.
        else:
            diff = r_row['val'] - c_row['val']
        individual_score = r_row['wht']**2 * diff**2
        score += individual_score
        if pretty:
            counter[r_row['typ']] += individual_score
            pretty_str.extend(pretty_data_comp(r_row, c_row, individual_score))
    if pretty:
        pretty_str.extend(pretty_data_comp(end=counter))
        return score, pretty_str
    else:
        return score

def pretty_data_comp(r_row=None, c_row=None, score=None, start=None, end=None):
    string = []
    if start:
        string.append('--' + ' Label '.ljust(20, '-') +
                      '--' + ' Weight '.center(8, '-') + 
                      '--' + ' R. Value '.center(13, '-') + 
                      '--' + ' C. Value '.center(13, '-') +
                      '--' + ' Score '.center(13, '-') + '--')
    elif end:
        string.append('-' * 79)
        total_score = sum((end[k] for k in end))
        string.append('{:<20}  {:20.4f}'.format('Total:', total_score))
        for k, v in end.iteritems():
            string.append('{:<20}  {:20.4f}'.format(k + ':', v))
    else:
        label = calculate.get_label(r_row)
        string.append('  {:<20}  {:>8.2f}  {:>13.4f}  {:>13.4f}  {:>13.4f}  '.format(
            label, r_row['wht'], r_row['val'], c_row['val'], score))
    return string
    
def main(args):
    parser = return_compare_parser()
    opts = parser.parse_args(args)
    if opts.doprint or opts.output:
        pretty = True
    else:
        pretty = False
    ref_conn = calculate.main(opts.reference.split())
    cal_conn = calculate.main(opts.calculate.split())
    if pretty:
        score, pretty_str = compare_data(ref_conn, cal_conn, pretty=pretty)
    else:
        score = compare_data(ref_conn, cal_conn)
    if opts.doprint:
        for line in pretty_str:
            print(line)
    if opts.output:
        with open(opts.output, 'w') as f:
            for line in pretty_str:
                f.write(line + '\n')
    return score

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])
