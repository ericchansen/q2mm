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
    h = Hessian()
    h.load_from_jaguar_out(filename='dsc.out', directory='ref_sulf_dsc_freq')
    h.load_from_jaguar_in(filename='dsc.01.in', directory='ref_sulf_dsc_freq')
    h.mass_weight_hess()
    h.mass_weight_evec()
    h.convert_units_for_mm()
    h.inv_hess()
