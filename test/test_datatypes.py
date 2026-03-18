#!/usr/bin/env python3
import unittest

from q2mm.datatypes import (
    match_mm3_bond,
    match_mm3_angle,
    match_mm3_stretch_bend,
    match_mm3_lower_torsion,
    match_mm3_higher_torsion,
    match_mm3_improper,
    match_mm3_label,
)


class TestMM3LabelMatching(unittest.TestCase):
    def test_match_mm3_label_bond(self):
        self.assertTrue(match_mm3_bond(" 1"))

    def test_match_mm3_label_geo_dep_bond(self):
        self.assertTrue(match_mm3_bond("a1"))

    def test_match_mm3_label_angle(self):
        self.assertTrue(match_mm3_angle(" 2"))

    def test_match_mm3_label_stretch_bend(self):
        self.assertTrue(match_mm3_stretch_bend(" 3"))

    def test_match_mm3_label_lower_torsion(self):
        self.assertTrue(match_mm3_lower_torsion(" 4"))

    def test_match_mm3_label_higher_torsion(self):
        self.assertTrue(match_mm3_higher_torsion("54"))

    def test_match_mm3_label_improper(self):
        self.assertTrue(match_mm3_improper(" 5"))

    def test_match_mm3_label_various(self):
        # Verify match_mm3_label accepts any valid lowercase+digit combo
        for label in ["a1", "b2", "c3", "d4", "e5", "z1"]:
            self.assertTrue(match_mm3_label(label), f"Should match: {label!r}")

    def test_no_match_invalid_label(self):
        self.assertFalse(match_mm3_bond(" 2"))  # ' 2' is angle, not bond
        self.assertFalse(match_mm3_angle(" 1"))  # ' 1' is bond, not angle


if __name__ == "__main__":
    unittest.main()
