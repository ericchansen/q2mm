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
        description='Used to correct mae files by removing dummy atoms.')
    parser.add_argument(
        'input', type=str, metavar='input.mae',
        help='Input mae file.')
    parser.add_argument(
        'output', type=str, metavar='output.mae',
        help='Output mae file.')
    parser.add_argument(
        '--dummy', '-d', type=str, metavar='dummy,atom,names',
        help='Comma separated list of dummy atom names.')
    options = vars(parser.parse_args(args))
    if options['dummy']:
        options['dummy'] = options['dummy'].split(',')
    with open(options['input'], 'r') as f:
        logger.info('Reading from: {}'.format(options['input']))
        lines = f.readlines()
    new_lines = []
    num_replaced = 0
    m_atom_check = False
    atom_area = False
    m_bond_check = False
    bond_area = False
    for i, line in enumerate(lines):
        if atom_area and ':::' in line:
            atom_area = False
        if bond_area and ':::' in line:
            bond_area = False
        if atom_area:
            cols = line.split()
            if cols[1] == '157':
                cols[1] = '113'
            if cols[1] == '18':
                cols[1] = '15'
            if cols[1] == '26':
                cols[1] = '25'
            if cols[1] == '62':
                cols[1]= '61'
            new_line = '  ' + ' '.join(cols) + '\n'
            new_lines.append(new_line)
            num_replaced += 1
        # if options['dummy'] and any([x in line for x in options['dummy']]):
        #     cols = line.split()
        #     cols[1] = '61'
        #     new_line = '  ' + ' '.join(cols) + '\n'
        #     new_lines.append(new_line)
        #     num_replaced += 1
        # elif bond_area:
        #     cols = line.split()
        #     if cols[1] == '1' and cols[2] == '2':
        #         cols[3] = '2'
        #     new_line = '  ' + ' '.join(cols) + '\n'
        #     new_lines.append(new_line)
        #     num_replaced += 1
        else:
            new_lines.append(line)
        if 'm_atom[' in line:
            m_atom_check = True
        if m_atom_check and ':::' in line:
            m_atom_check = False
            atom_area = True
        if 'm_bond[' in line:
            m_bond_check = True
        if m_bond_check and ':::' in line:
            m_bond_check = False
            bond_area = True
    logger.info('Replaced {} lines.'.format(num_replaced))
    with open(options['output'], 'w') as f:
        f.writelines(new_lines)

if __name__ == '__main__':
    remove_logs()
    sys.excepthook = log_uncaught_exceptions
    with open('options/logging.yaml', 'r') as f:
        config = yaml.load(f)
    logging.config.dictConfig(config)
    process_args(sys.argv[1:])
