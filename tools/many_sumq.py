#!/usr/bin/env python
import argparse
import glob
import os
import sys
from itertools import izip_longest

import sumq

# Same as setup_com_from_mae.
# Would be better to import.
def grouper(n, iterable, fillvalue=0.):
    """
    Returns list of lists from a single list.

    Usage
    -----
    grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx

    Arguments
    ---------
    n : integer
        Length of sub list.
    iterable : iterable
    fillvalue : anything
                Fills up last sub list if iterable is not divisible by n
                without a remainder.

    """
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def return_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str)
    parser.add_argument('tag', type=str)
    parser.add_argument('appendfile', type=str)
    return parser

def main(args):
    parser = return_parser()
    opts = parser.parse_args(args)
    many_sumq(opts.directory, opts.tag, opts.appendfile)

def many_sumq(directory, tag, appendfile):
    filenames = glob.glob(os.path.join(directory, '*' + tag))
    # filenames = [x for x in filenames if x.endswith('re.log')]

    filenames.sort()
    assert len(filenames) % 2 == 0
    print(filenames)
    for filename1, filename2 in grouper(2, filenames):
        sumq.sumq([[filename1], [filename2]], appendfile=appendfile)

if __name__ == '__main__':
    main(sys.argv[1:])
