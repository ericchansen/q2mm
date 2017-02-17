#!/usr/bin/python
"""
Cleans up *.mae files.

Accepts multiple filenames and glob as the argument(s).
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
            structure_writer.append(structure)

        structure_reader.close()
        structure_writer.close()

        os.rename('TEMP.mae', filename)
