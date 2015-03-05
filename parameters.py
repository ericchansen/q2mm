#!/usr/bin/python
'''
Selects parameters from force fields.
'''
import argparse
from argparse import RawTextHelpFormatter
import logging
import numpy as np
import sys

from datatypes import MM3
from filetypes import MacroModel

logger = logging.getLogger(__name__)

def return_parameters_parser(add_help=True):
    '''
    Returns an argparse.ArgumentParser object for the selection of
    parameters.
    '''
    if add_help:
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
q    - bond dipoles''')
        parser = argparse.ArgumentParser(
            formatter_class=RawTextHelpFormatter,
            description=description)
    else:
        parser = argparse.ArgumentParser(add_help=False)

    arg_group = parser.add_argument_group('parameters')
    arg_group.add_argument(
        '--all', '-a', action='store_true',
        help='Select all available parameters from the force field.')
    arg_group.add_argument(
        '--average', '-av', type=str, nargs='+',
        help=('Use these MacroModel files to generate a new force field from '
              'the input force field where each equilibrium value in the '
              'optimize section is replaced by the average value listed in '
              'the MacroModel file. Use --averageff to name the output.'))
    arg_group.add_argument(
        '--averageff', '-avff', type=str, metavar='mm3.fld',
        help='Output force field filename from --average.')
    arg_group.add_argument(
        '--ffpath', '-f', metavar='mm3.fld', default='mm3.fld',
        help='Path to force field.')
    arg_group.add_argument(
        '--printparams', '-pp', action='store_true',
        help='Print information about the selected parameters.')
    arg_group.add_argument(
        '--pfile', '-pf', type=str, metavar='filename',
        help='Use a file to select parameters. Allows advanced options.')
    arg_group.add_argument(
        '--ptypes', '-pt', nargs='+', default=[],
        help='Select these parameter types.')
    return parser

def select_parameters(opts, ff=None):
    '''
    Imports a force field object, which contains a list of all the available
    parameters for optimization in the "params" attribute. Returns a list of
    only the user selected parameters.
    '''
    if opts.all:
        opts.ptypes.extend(('ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'))

    if ff is None:
        ff = MM3(opts.ffpath)
        ff.import_ff()
        logger.info('ff loaded from {} - {} parameters'.format(ff.path, len(ff.params)))

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
                                       
    if opts.average and opts.averageff:
        bond_dic = {}
        angle_dic = {}
        for filename in opts.average:
            mmo = MacroModel(filename)
            for structure in mmo.structures:
                for bond in structure.bonds:
                    if bond.ff_row in bond_dic:
                        bond_dic[bond.ff_row].append(bond.value)
                    else:
                        bond_dic[bond.ff_row] = [bond.value]
                for angle in structure.angles:
                    if angle.ff_row in angle_dic:
                        angle_dic[angle.ff_row].append(angle.value)
                    else:
                        angle_dic[angle.ff_row] = [angle.value]
        print
        print bond_dic
        print angle_dic
        print

        bond_avg = {}
        for ff_row, values in bond_dic.iteritems():
            bond_avg[ff_row] = np.mean(values)
        angle_avg = {}
        for ff_row, values in angle_dic.iteritems():
            angle_avg[ff_row] = np.mean(values)

        print
        print bond_avg
        print angle_avg
        print

        for param in selected_params:
            print param
            if param.mm3_row in bond_avg:
                param.value = bond_avg[param.mm3_row]
            if param.mm3_row in angle_avg:
                param.value = angle_avg[param.mm3_row]
                
        ff.export_ff(params=selected_params, path=opts.averageff)

    if opts.printparams:
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

    parser = return_parameters_parser()
    opts = parser.parse_args(sys.argv[1:])
    select_parameters(opts)

