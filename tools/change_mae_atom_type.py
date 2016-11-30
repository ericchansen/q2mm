#!/usr/bin/python
"""
Change all of atom type ATOM_X to ATOM_Y.
"""

ATOM_X = 62
ATOM_Y = 204

INPUT_FILE = "c1r2s_004.mae"
OUTPUT_FILE = "c1r2s_004.out.mae"

from schrodinger import structure

reader = structure.StructureReader(INPUT_FILE)
writer = structure.StructureWriter(OUTPUT_FILE)

for structure in reader:
    for atom in structure.atom:
        if atom.atom_type == ATOM_X:
            atom.atom_type = ATOM_Y
    writer.append(structure)

reader.close()
writer.close()
