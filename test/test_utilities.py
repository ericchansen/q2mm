import os
import unittest
import sys
from pathlib import Path

from q2mm.parsers import _utils as utilities

REPO_ROOT = Path(__file__).resolve().parent.parent
ETHANE_MOL2 = REPO_ROOT / "examples" / "ethane" / "GS.mol2"


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


@unittest.skipUnless(
    Path(__file__).resolve().parent.parent.joinpath("examples", "ethane", "GS.mol2").exists(),
    "Ethane fixture not found",
)
class TestIdentifyAngles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from q2mm.parsers.mol2 import Mol2

        mol2 = Mol2(str(ETHANE_MOL2))
        cls.ethane_struct = mol2.structures[0]

    def test_identify_angles_ethane(self):
        angles = self.ethane_struct.identify_angles()
        self.assertEqual(len(angles), 12)
