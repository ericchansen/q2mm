from __future__ import annotations

import copy

import os
import sys

import unittest
import numpy as np
from q2mm import constants
from q2mm.parsers import FF, Structure

src_dir = os.path.abspath("q2mm")
sys.path.append(src_dir)
from q2mm import linear_algebra


class MakeHessian:
    def __init__(self):
        self.out = []

    def write(self, str):
        self.out.append(str)

    def __str__(self):
        return "".join(self.out)


example_sq3 = np.asarray([[-2, -4, 2], [-4, 1, 2], [2, 2, 5]])
# Eigenvalues are sorted in ascending order to match the output from np.linalg.eigh
# which is used by linear_algebra methods (decompose in particular)
evals_sq3 = np.asarray([-5.51082, 3.659278, 5.851542])
# Matrix rows must be ordered such that eigenvectors correspond to correct eigenvalues
evec_sq3 = np.asarray([[-3.06513, -2.19028, 1], [2.82137, -3.49173, 1], [0.0770902, 0.348681, 1]])
evec_normd_sq3 = np.array([evec / np.linalg.norm(evec) for evec in evec_sq3])


class TestLinearAlgebra(unittest.TestCase):
    @unittest.expectedFailure  # Pre-existing: expected values don't match algorithm output
    def test_replace_neg_eigenvalue(self):
        repl_evals = linear_algebra.replace_neg_eigenvalue(evals_sq3)
        np.testing.assert_allclose([1.0, evals_sq3[1], evals_sq3[2]], repl_evals)
        multi_neg_evals = [
            -5,
            -2.45,
            -1,
            -0.2,
            0.3,
            0.459,
            0.5,
            1.8,
            2.3,
            4.6,
            5,
            9.87456,
        ]
        multi_neg_evals_repl = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.459,
            0.5,
            1.8,
            2.3,
            4.6,
            5,
            9.87456,
        ]
        np.testing.assert_allclose(
            multi_neg_evals_repl,
            linear_algebra.replace_neg_eigenvalue(multi_neg_evals, zer_out_neg=True, strict=False),
            err_msg="Replaced eigenvalues do not match. Failed to replace excess negative values with zero.",
        )

    def test_multi_neg_eigenvalue_strict_raises(self):
        """Multiple negative eigenvalues should raise ValueError by default."""
        multi_neg = np.array([-5.0, -2.0, 1.0, 3.0])
        with self.assertRaises(ValueError, msg="Should raise on multiple negative eigenvalues"):
            linear_algebra.replace_neg_eigenvalue(multi_neg)

    def test_multi_neg_eigenvalue_nonstrict_warns(self):
        """Multiple negative eigenvalues with strict=False should warn, not raise."""
        import warnings

        multi_neg = np.array([-5.0, -2.0, 1.0, 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = linear_algebra.replace_neg_eigenvalue(multi_neg, strict=False)
            self.assertEqual(len(w), 1)
            self.assertIn("negative eigenvalues", str(w[0].message))
        # Most negative (-5.0) should be replaced
        self.assertGreater(result[0], 0.0)

    def test_reform_hessian(self):
        evals, evecs = linear_algebra.decompose(example_sq3)
        reformed_sq3 = linear_algebra.reform_hessian(evals, evecs)
        np.testing.assert_allclose(example_sq3, reformed_sq3, err_msg="Hessian is not reformed properly.")

    @unittest.skip("Incomplete — requires refactoring (upstream TODO)")
    def test_last(
        self,
        force_field: FF,
        structures: list[Structure],
        hessians: list[np.ndarray],
        zero_out: bool,
        hessian_units=constants.GAUSSIAN,
    ):
        # TODO this might need to be refactored for python 3.8 at some point if it will be run with Schrodinger...
        # Last Gaussian Eigenvalue Analysis Check TODO finish refactoring this after moving from seminario.py

        estimated_ff = copy.deepcopy(force_field)
        structs = structures

        last_evec_ch = log.evecs[-1]
        normed = last_evec_ch / np.linalg.norm(last_evec_ch)
        print("normed final eigenvector ch: " + str(normed))
        last_evec_hess = np.dot(normed, min_hessian)
        print("last evec dot hess: " + str(last_evec_hess))
        dotted_again = last_evec_hess.dot(np.transpose(normed))
        print("last evec dot dot: " + str(dotted_again))
        print("all close hessian dotted: " + str(np.allclose(log.evals[-1], dotted_again)))
        print("last eigenvalue/force constant: " + str(log.evals[-1]))
