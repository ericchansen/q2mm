#!/usr/bin/env python
"""
Change all of atom type ATOM_X to ATOM_Y.
"""
import os
from schrodinger import structure

ATOM_X = 1
ATOM_Y = 2

INPUT_FILE = "yourmae1.mae"
OUTPUT_FILE = "yourmae2.mae"

if INPUT_FILE == OUTPUT_FILE:
    print('Warning: INPUT_FILE ({}) == OUTPUT_FILE ({}).'.format(
            INPUT_FILE, OUTPUT_FILE))
    print('Using a file called TEMP.mae in current dir.')

reader = structure.StructureReader(INPUT_FILE)
if INPUT_FILE == OUTPUT_FILE:
    writer = structure.StructureWriter('TEMP.mae')
else:
    writer = structure.StructureWriter(OUTPUT_FILE)

for structure in reader:
    for atom in structure.atom:
        if atom.atom_type == ATOM_X:
            atom.atom_type = ATOM_Y
    writer.append(structure)

reader.close()
writer.close()

if INPUT_FILE == OUTPUT_FILE:
    os.rename('TEMP.mae', OUTPUT_FILE)
