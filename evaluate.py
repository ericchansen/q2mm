#!/usr/bin/python
import argparse
import calculate
import logging
import logging.config
import os
import parameters
import sys

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

