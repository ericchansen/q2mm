#!/usr/bin/python
'''
Selects parameters from force fields.
'''
import argparse
from argparse import RawTextHelpFormatter
import logging
import sys

from datatypes import MM3

logger = logging.getLogger(__name__)

def parse_parameters(args):
    '''
    Parses command line arguments. Grabs the selected parameters.
    '''
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description=(__doc__ + 
                     '''
PTYPES:
ae   - equilibrium angles
af   - angle force constants
be   - equilibrium bond lengths
bf   - bond force constants
df   - dihedral force constants
imp1 - improper torsions (1st MM3* column)
imp2 - improper torsions (2nd MM3* column)
sb   - stretch-bend force constants
q    - bond dipoles'''))
    parser.add_argument(
        '--all', '-a', action='store_true',
        help='Select all available parameters from the force field.')
    parser.add_argument(
        '--ffpath', '-f', metavar='mm3.fld', default='mm3.fld',
        help='Path to force field.')
    parser.add_argument(
        '--doprint', '-dp', action='store_true',
        help='Print locations in force field of the selected parameters.')
    parser.add_argument(
        '--pfile', '-pf', type=str, metavar='filename',
        help='Use a file to select parameters. Allows advanced options.')
    parser.add_argument(
        '--ptypes', '-pt', nargs='+', default=[],
        help='Select these parameter types.')
    opts = parser.parse_args(args)

    if opts.all:
        opts.ptypes.extend(('ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'))

    ff = MM3(opts.ffpath)
    ff.import_ff()
    logger.info('{} params in {}'.format(len(ff.params), ff.path))

    selected_params = []
    if opts.ptypes:
        selected_params.extend([x for x in ff.params if x.ptype in opts.ptypes])

    if opts.pfile:
        temp_params = []
        with open(opts.pfile, 'r') as f:
            for line in f:
                line = line.partition('#')[0]
                cols = line.split()
                if cols:
                    mm3_row, mm3_col = int(cols[0]), int(cols[1])
                    allow_negative = None
                    group_label = None
                    for arg in cols[2:]:
                        # if any(arg == x for x in ('allowneg', 'neg', 'negative')):
                        if 'neg' in arg:
                            allow_negative = True
                            # logger.log(7, 'param {} {} allowed negative values'.format(
                            #         mm3_row, mm3_col))
                        if arg.startswith('g'):
                            group_label = arg[1:]
                    temp_params.append((mm3_row, mm3_col, allow_negative, group_label))
        for param in ff.params:
            for temp_param in temp_params:
                if param.mm3_row == temp_param[0] and param.mm3_col == temp_param[1]:
                    param._allow_negative = temp_param[2]
                    param.group = temp_param[3]
                    selected_params.append(param)
                                       
    if opts.doprint:
        for param in selected_params:
            # print('{} {}'.format(param.mm3_row, param.mm3_col))
            try:
                print('{}[{},{}]({})({})({})'.format(
                        param.ptype, param.mm3_row, param.mm3_col, param.value,
                        param.group, param.allow_negative))
            except:
                print param
            
    return selected_params

if __name__ == '__main__':
    import logging.config
    import yaml

    with open('logging.yaml', 'r') as f:
        cfg = yaml.load(f)
    logging.config.dictConfig(cfg)

    parse_parameters(sys.argv[1:])

