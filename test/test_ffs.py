import copy
import logging
import logging.config
import os
import unittest

from q2mm import constants as co
from q2mm import datatypes

logger = logging.getLogger(__name__)

class TestMM3Import(unittest.TestCase):
    def setUp(self):
        self.ff = datatypes.import_ff('d_rhod/mm3.fld')
    def test_check_data_from_ff(self):
        substructures = ['Rh hydrogenation OPT', 'Rh-P ligand OPT']
        smiles = ['Z0(-P1)(-P2)(.O2)(.C2=C2.H7-1)(-H6)', 'Z0-PM']
        atom_types = [['Z0', 'P1', 'P2', 'O2', 'C2', 'C2', 'H7', 'Z0', 'H6'],
                      ['Z0', 'PM']]
        print(f'SUBSTRUCTURES: {self.ff.sub_names}')
        print(f'SMILES: {self.ff.smiles}')
        print(f'ATOM TYPES: {self.ff.atom_types}')
        self.assertEqual(substructures, self.ff.sub_names)
        self.assertEqual(smiles, self.ff.smiles)
        self.assertEqual(atom_types, self.ff.atom_types)

class TestMM3Export(unittest.TestCase):
    def setUp(self):
        self.ff = datatypes.import_ff('d_rhod/mm3.fld')
        with open('d_rhod/mm3.fld') as f:
            self.ff.lines = f.readlines()
        mod_params = copy.deepcopy(self.ff.params)
        mod_params[0].value = 999.
        datatypes.export_ff('test.fld', mod_params, lines=self.ff.lines)
    def tearDown(self):
        os.remove('test.fld')
    def test_export_ff(self):
        mod_ff = datatypes.import_ff('test.fld')
        self.assertEqual(mod_ff.params[0].value, 999.)

if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    unittest.main()
