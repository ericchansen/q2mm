import os
import unittest
import sys

import parmed

src_dir = os.path.abspath("q2mm")
sys.path.append(src_dir)
from q2mm import utilities

ethane_struct = parmed.load_file('../q2mm_example/amber/Ethane/GS.mol2', structure=True)


class MakeInput:
    def __init__(self):
        self.out = []

    def write(self, str):
        self.out.append(str)

    def __str__(self):
        return "".join(self.out)


class TestAtomTypeConversion(unittest.TestCase):
    def test_convert_atom_type(self):
        mol2_C3 = "C.3"
        mol2_metal = "Pd"

        schrod_C3 = "C3"
        schrod_metal = "PD"

        converted_C3 = utilities.convert_atom_type(mol2_C3)
        converted_metal = utilities.convert_atom_type(mol2_metal)

        self.assertEqual(
            schrod_C3,
            converted_C3,
            "Incorrect conversion from mol2 sp3 C to Schrodinger atom type.",
        )
        self.assertEqual(
            schrod_metal,
            converted_metal,
            "Incorrect conversion from mol2 metal to Schrodinger atom type.",
        )


# class TestStructureEqual(unittest.TestCase):

#     def test_is_same_bond(self):
#         self.assertEqual()

class TestIdentifyAngles(unittest.TestCase):

    def test_identify_angles_ethane(self):
        angles = utilities.identify_angles(ethane_struct.bonds)
        self.assertEqual(len(angles), 12)
