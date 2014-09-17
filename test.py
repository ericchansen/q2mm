#!/usr/bin/python
import argparse
import logging
import logging.config
import numpy as np
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

# Setup logging.
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    
    from filetypes import *
    a = JagOutFile('dsc.out', directory='ref_sulf_dsc_freq')
    # print len(a.raw_data['Geometries'])
    # for i, x in enumerate(a.raw_data['Geometries']):
    #     print '--- Geo {} ---'.format(i)
    #     for y in x:
    #         print y
    # print '--- Frequencies ---'
    # print a.raw_data['Frequencies']
    # print '--- Eigenvalues ---'
    # print a.raw_data['Eigenvalues']
    # print '--- Eigenvectors --'
    # print a.raw_data['Eigenvectors']
    # b = JagInFile('dsc.01.in', directory='ref_sulf_dsc_freq')
    # print list(b.raw_data)
    # print b.raw_data['Hessian']
    # evals, evecs = np.linalg.eigh(b.raw_data['Hessian'])
    # print len(evals)
    # print evals

    h = Hessian()
    h.load_from_jaguar_out(filename='dsc.out', directory='ref_sulf_dsc_freq')
    h.load_from_jaguar_in(filename='dsc.01.in', directory='ref_sulf_dsc_freq')
    h.convert_units_for_mm()
    # print h.evals
    # print h.evecs
    h.inv_hess()
