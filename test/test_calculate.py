from __future__ import print_function
import copy
import logging
import logging.config
import os
import unittest

import constants as co
import calculate
import compare

logger = logging.getLogger(__name__)

class TestMacroModelBonds(unittest.TestCase):
    """
    Check that the -mb command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -mb X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 8)

class TestMacroModelAngles(unittest.TestCase):
    """
    Check that the -ma command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -ma X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 34)

class TestMacroModelTorsions(unittest.TestCase):
    """
    Check that the -mt command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -mt X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 90)

class TestMacroModelEnergies(unittest.TestCase):
    """
    Check that the -me command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -me X001_E1.01.mae X001_E2.02.mae X001_Z1.02.mae '
            'X001_Z2.02.mae'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 4)

class TestMacroModelOptimizedEnergies(unittest.TestCase):
    """
    Check that the -meo command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -meo X001_E1.01.mae X001_E2.02.mae X001_Z1.02.mae '
            'X001_Z2.02.mae'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 4)

# I need to implement charges again.
class TestMacroModelCharges(unittest.TestCase):
    """
    Check that the -mq command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -mq X001_E1.01.mae'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        print(len(rows))
        for row in rows:
            print(row)

# Need to add a check on the number of data points generated.
class TestMacroModelEigenvalues(unittest.TestCase):
    """
    Check that the -meig command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -meig X001_E1.01.mae,X001_E1.out'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        print(len(rows))

class TestJaguarBonds(unittest.TestCase):
    """
    Check that the -jb command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -jb X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 8)

class TestJaguarAngles(unittest.TestCase):
    """
    Check that the -ja command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -ja X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 34)

class TestJaguarTorsions(unittest.TestCase):
    """
    Check that the -jt command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(' -d d_rhod -jt X001_E1.01.mae'.split())
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 90)

class TestJaguarEnergies(unittest.TestCase):
    """
    Check that the -je command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -je X001_E1.01.mae X001_E2.02.mae X001_Z1.02.mae '
            'X001_Z2.02.mae'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 4)

class TestJaguarEnergies(unittest.TestCase):
    """
    Check that the -jeo command for the calculate module produces the
    proper number of data points.

    Note that this command actually does nothing different than -je, but is
    instead used to indicate that -jeo energies should be matched with
    -meo energies, rather than -me energies.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -jeo X001_E1.01.mae X001_E2.02.mae X001_Z1.02.mae '
            'X001_Z2.02.mae'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 4)

# Need to add a check on the number of data points generated.
class TestJaguarEigenvalues(unittest.TestCase):
    """
    Check that the -jeige command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(
            ' -d d_rhod -jeige X001_E1.01.in,X001_E1.out'.split())
    def test_me(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))

class TestCompareBonds(unittest.TestCase):
    """
    Check that these two commands produce a reasonable data set to
    evaluate the objective function.
    """
    def setUp(self):
        self.f_conn = calculate.main(' -d d_rhod -mb X001_E1.01.mae'.split())
        self.r_conn = calculate.main(' -d d_rhod -jb X001_E1.01.mae'.split())
    def test_compare_bonds(self):
        score = compare.compare_data(self.r_conn, self.f_conn)
        print('COMPARE BONDS SCORE: {}'.format(score))
            
if __name__ == '__main__':
    logging.config.dictConfig(co.LOG_SETTINGS)
    unittest.main()
