#!/usr/bin/env python
"""
I use this to clean up FFs I find in the literature or wherever with poor
formatting.
"""
import argparse
import re
import sys

def format_ff(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    formatted_lines = []
    for line in lines:
        if line.startswith(' C'):
            print(line.strip('\n'))
        elif line.startswith(' 9'):
            print(line.strip('\n'))
        elif line.startswith('-'):
            print(line.strip('\n'))
        elif re.match('[a-z][A-Z]', line[:2]):
            # I don't think this section is general enough.
            cols = line.split()
            lbl = '{:>2}'.format(cols[0])
            a1 = '{:>2}'.format(cols[1])
            a2 = '{:>2}'.format(cols[2])
            a3 = '{:>2}'.format(cols[3])
            p1 = '{:>9.4f}'.format(float(cols[4]))
            p2 = '{:>9.4f}'.format(float(cols[5]))
            print('{}  {}  {}  {}          {} {}'.format(
                lbl, a1, a2, a3, p1, p2))
        elif re.match('[a-z\s]1', line[:2]):
            cols = line.split()
            lbl = '{:>2}'.format(cols[0])
            a1 = '{:>2}'.format(cols[1])
            a2 = '{:>2}'.format(cols[2])
            p1 = '{:>9.4f}'.format(float(cols[3]))
            p2 = '{:>9.4f}'.format(float(cols[4]))
            p3 = '{:>9.4f}'.format(float(cols[5]))
            print('{}  {}  {}              {} {} {}'.format(
                lbl, a1, a2, p1, p2, p3))
        elif re.match('[a-z\s]2', line[:2]):
            cols = line.split()
            lbl = '{:>2}'.format(cols[0])
            a1 = '{:>2}'.format(cols[1])
            a2 = '{:>2}'.format(cols[2])
            a3 = '{:>2}'.format(cols[3])
            p1 = '{:>9.4f}'.format(float(cols[4]))
            p2 = '{:>9.4f}'.format(float(cols[5]))
            print('{}  {}  {}  {}          {} {}'.format(
                lbl, a1, a2, a3, p1, p2))
        elif re.match('[a-z\s]4', line[:2]):
            cols = line.split()
            lbl = '{:>2}'.format(cols[0])
            a1 = '{:>2}'.format(cols[1])
            a2 = '{:>2}'.format(cols[2])
            a3 = '{:>2}'.format(cols[3])
            a4 = '{:>2}'.format(cols[4])
            p1 = '{:>9.4f}'.format(float(cols[5]))
            p2 = '{:>9.4f}'.format(float(cols[6]))
            p3 = '{:>9.4f}'.format(float(cols[7]))
            print('{}  {}  {}  {}  {}      {} {} {}'.format(
                lbl, a1, a2, a3, a4, p1, p2, p3))

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', '-i', type=str,
        help='File to parse and format.')
    return parser

def main(opts):
    format_ff(opts.input)

if __name__ == '__main__':
    parser = return_parser()
    opts = parser.parse_args(sys.argv[1:])
    main(opts)
