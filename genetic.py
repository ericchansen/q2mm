#!/usr/bin/python
import argparse
import logging
import logging.config
from setup_logging import log_uncaught_exceptions, remove_logs
import sys
import yaml

# Setup logging.
logger = logging.getLogger(__name__)

def process_args(args):
    parser = argparse.ArgumentParser(
        description='Genetic algorithm for improving parameters.')
    options = vars(parser.parse_args(args))

if __name__ == '__main__':
    # Setup logs.
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        log_config = yaml.load(f)
    logging.config.dictConfig(log_config)
    # Process arguments.
    process_args(sys.argv[1:])
