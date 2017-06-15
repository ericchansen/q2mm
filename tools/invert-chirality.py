#!/usr/bin/python
"""
Uses Schrodinger to invert the chirality of a molecule.
"""

import argparse
import sys
import schrodinger.structure as sch_struct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Inverts x-coords of a molecule.")
    parser.add_argument(
        '--input', '-i', type=str,
        help='Input filename.')
    parser.add_argument(
        '--output', '-o', type=str,
        help='Output filename.')
    opts = parser.parse_args(sys.argv[1:])
    writer = sch_struct.StructureWriter(opts.output)
    reader = sch_struct.StructureReader(opts.input)
    for structure in reader:
        for coords in structure.getXYZ(copy=False):
            coords[0] = -coords[0]
        writer.append(structure)
    reader.close()
    writer.close()
