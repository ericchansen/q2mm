#!/usr/bin/env python
"""
Cleans up *.mae files.

Accepts multiple filenames and glob as the argument(s).
"""
import os
import sys

from schrodinger import structure as sch_struct

from clean_mae import \
    PROPERTIES_TO_REMOVE, ATOM_PROPERTIES_TO_REMOVE
from clean_mae_same_update import \
    ATOM_CS_PROPERTIES, BOND_CS_PROPERTIES, CONV_DIC

PROPERTIES_TO_REMOVE.extend([
    's_cs_smiles_substrate',
    's_cs_smiles_ligand',
    'b_cs_c2',
    's_cs_stereochemistry',
    'b_cs_first_match_only',
    'b_cs_both_enantiomers'
    ])

ATOM_PROPERTIES_TO_REMOVE.extend(ATOM_CS_PROPERTIES)
BOND_PROPERTIES_TO_REMOVE = BOND_CS_PROPERTIES
BOND_PROPERTIES_TO_REMOVE.extend(CONV_DIC.keys())

if __name__ == '__main__':
    for filename in sys.argv[1:]:
        structure_reader = sch_struct.StructureReader(filename)
        structure_writer = sch_struct.StructureWriter('TEMP.mae')

        for structure in structure_reader:

            # Clean structure properties.
            for prop in PROPERTIES_TO_REMOVE:
                try:
                    del structure.property[prop]
                except KeyError:
                    pass
            # Change the name too. Why not.
            structure.property['s_m_title'] = \
                structure.property['s_m_entry_name'] = \
                    os.path.splitext(
                        os.path.basename(filename))[0]

            # Clean atom properties.
            for atom in structure.atom:
                for prop in ATOM_PROPERTIES_TO_REMOVE:
                    try:
                        del atom.property[prop]
                    except KeyError:
                        pass
                    except ValueError:
                        pass

            # Clean bond properties.
            for bond in structure.bond:
                for prop in BOND_PROPERTIES_TO_REMOVE:
                    try:
                        del bond.property[prop]
                    except KeyError:
                        pass
                    except ValueError:
                        pass

            structure_writer.append(structure)

        structure_reader.close()
        structure_writer.close()

        os.rename('TEMP.mae', filename)
