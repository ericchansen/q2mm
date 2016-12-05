#!/usr/bin/python
import sys
from schrodinger import structure

PROPERTIES_TO_REMOVE = \
    [
    's_m_Source_File',
    's_m_Source_Path',
    's_m_entry_id',
    'i_m_Source_File_Index',
    'i_m_ct_format',
    'i_m_ct_stereo_status',
    
    's_st_Chirality_1',
    's_st_Chirality_2'
    's_st_Chirality_3',
    
    'b_mmod_Minimization_Converged-MM3*',
    'i_mmod_Times_Found-MM3*',
    'r_mmod_RMS_Derivative-MM3*'
    ]

ATOM_PROPERTIES_TO_REMOVE = \
    [
    's_m_mmod_res',
    'i_m_color',
    'r_m_charge1',
    'r_m_charge2',
    's_m_pdb_residue_name',
    's_m_color_rgb'
    ]

structure_reader = structure.StructureReader(sys.argv[1])
structure_writer = structure.StructureWriter(sys.argv[2])

for structure in structure_reader:
    for prop in PROPERTIES_TO_REMOVE:
        try:
            del structure.property[prop]
        except KeyError:
            pass
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
