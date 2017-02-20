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
                if not 'b_cs_chig' in atom.property:
                    atom.property['b_cs_chig'] = 0
                if not 'b_cs_comp' in atom.property:
                    atom.property['b_cs_comp'] = 0
            for bond in structure.bond:
                if not 'b_cs_tors' in bond.property:
                    bond.property['b_cs_tors'] = 0
                if not 'i_cs_torc_1' in bond.property:
                    bond.property['i_cs_torc_1'] = 0
                if not 'i_cs_torc_2' in bond.property:
                    bond.property['i_cs_torc_2'] = 0
                if not 'i_cs_rca4_1' in bond.property:
                    bond.property['i_cs_rca4_1'] = 0
                if not 'i_cs_rca4_2' in bond.property:
                    bond.property['i_cs_rca4_2'] = 0
            structure_writer.append(structure)

        structure_reader.close()
        structure_writer.close()

        os.rename('TEMP.mae', filename)
