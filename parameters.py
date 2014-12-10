#!/usr/bin/python
'''
Short script that prints the location of parameters in a force field
file.
'''
import argparse
import logging
import sys

from datatypes import MM3

logger = logging.getLogger(__name__)

def parse_parameters(args):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', '-a', action='store_true', help='Select all parameters.')
    parser.add_argument('--ffpath', '-f', metavar='mm3.fld', default='mm3.fld', help='Path to force field.')
    parser.add_argument('--noprint', '-n', action='store_true', help="Don't print parameter locations.")
    parser.add_argument('--ptypes', '-p', nargs='+', default=[], help='Select these parameter types.')
    opts = parser.parse_args(args)

    if opts.all:
        opts.ptypes.extend(('ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'))

    ff = MM3(opts.ffpath)
    ff.import_ff()
    logger.info('{} params in {}'.format(len(ff.params), ff.path))

    params = []
    for param in ff.params:
        if param.ptype in opts.ptypes:
            params.append(param)
    logger.info('selected {} params'.format(len(params)))            

    if not opts.noprint:
        for param in params:
            print('{} {}'.format(param.mm3_row, param.mm3_col))
    
    return params

if __name__ == '__main__':
    import logging.config
    import yaml

    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)

    parse_parameters(sys.argv[1:])

