#!/usr/bin/python
from datatypes import *
import random
import string
import unittest

class TestMM3(unittest.TestCase):
    def setUp(self):
        self.ff = MM3()
    def test_match_mm3_label_bond(self):
        self.assertTrue(self.ff.match_mm3_bond(' 1'))
    def test_match_mm3_label_geo_dep_bond(self):
        self.assertTrue(self.ff.match_mm3_bond('a1'))
    def test_match_mm3_label_angle(self):
        self.assertTrue(self.ff.match_mm3_angle(' 2'))
    def test_match_mm3_label_stretch_bend(self):
        self.assertTrue(self.ff.match_mm3_stretch_bend(' 3'))
    def test_match_mm3_label_lower_torsion(self):
        self.assertTrue(self.ff.match_mm3_lower_torsion(' 4'))
    def test_match_mm3_label_higher_torsion(self):
        self.assertTrue(self.ff.match_mm3_higher_torsion('54'))
    def test_match_mm3_label_improper(self):
        self.assertTrue(self.ff.match_mm3_improper(' 5'))
    def test_match_mm3_label(self):
        self.assertTrue(self.ff.match_mm3_label('{}{}'.format(
                    random.choice(string.ascii_lowercase), random.randint(1, 5))))
                        
if __name__ == '__main__':
    unittest.main()

