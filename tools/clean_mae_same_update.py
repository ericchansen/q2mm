#!/usr/bin/env python
"""
Cleans up *.mae files.

Accepts multiple filenames and glob as the argument(s).

This version adds in the custom conformational search properties to Maestro
structures, atoms and bonds if they don't already exist. Default for all values
is zero.
"""
import os
import sys

from schrodinger import structure as sch_struct

from clean_mae import PROPERTIES_TO_REMOVE, ATOM_PROPERTIES_TO_REMOVE

ATOM_CS_PROPERTIES = ['b_cs_chig',
                      'b_cs_comp']
BOND_CS_PROPERTIES = ['b_cs_tors',
                      'i_cs_rca4_1',
                      'i_cs_rca4_2',
                      'i_cs_torc_a1',
                      'i_cs_torc_a4',
                      'r_cs_torc_a5',
                      'r_cs_torc_a6',
                      'i_cs_torc_b1',
                      'i_cs_torc_b4',
                      'r_cs_torc_b5',
                      'r_cs_torc_b6']

# Updates for new format.
CONV_DIC = {'i_cs_torc_1': 'i_cs_torc_a1',
            'i_cs_torc_2': 'i_cs_torc_a4',
            'r_cs_torc_5': 'r_cs_torc_a5',
            'r_cs_torc_6': 'r_cs_torc_a6'}

if __name__ == "__main__":
    for filename in sys.argv[1:]:
        structure_reader = sch_struct.StructureReader(filename)
        structure_writer = sch_struct.StructureWriter('TEMP.mae')

        for structure in structure_reader:
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
            for atom in structure.atom:
                for prop in ATOM_PROPERTIES_TO_REMOVE:
                    try:
                        del atom.property[prop]
                    except KeyError:
                        pass
                    except ValueError:
                        pass
                for prop in ATOM_CS_PROPERTIES:
                    if not prop in atom.property:
                        atom.property[prop] = 0

            for bond in structure.bond:
                # Update 1st.
                for k, v in CONV_DIC.items():
                    if k in bond.property:
                        bond.property[v] = bond.property[k]
                        del bond.property[k]

                for prop in BOND_CS_PROPERTIES:
                    if not prop in bond.property:
                        bond.property[prop] = 0

            structure_writer.append(structure)

        structure_reader.close()
        structure_writer.close()

        os.rename('TEMP.mae', filename)
