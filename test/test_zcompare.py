import copy
import logging
import logging.config
import os
import unittest

from q2mm import constants as co
from q2mm import calculate
from q2mm import compare

logger = logging.getLogger(__name__)

class TestCompareEthaneHessian(unittest.TestCase):
    """
    Check that the -mb command for the calculate module produces the
    proper number of data points.
    """
    def setUp(self):
        self.conn = calculate.main(['-d', 'd_rhod', '-mb', 'X001_E1.01.mae'])
    def test_ma(self):
        c = self.conn.cursor()
        rows = list(c.execute('SELECT * FROM data'))
        self.assertEqual(len(rows), 8)
