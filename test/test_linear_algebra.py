from __future__ import annotations

import unittest

import numpy as np

from q2mm import constants
from q2mm.models import hessian
from q2mm.models.structure import Structure  # noqa: F401 — used in skipped test signature
from q2mm.parsers import FF  # noqa: F401 — used in skipped test signature


class MakeHessian:
    def __init__(self) -> None:
        self.out: list[str] = []

    def write(self, str: str) -> None:
        self.out.append(str)

    def __str__(self) -> str:
        return "".join(self.out)


example_sq3 = np.asarray([[-2, -4, 2], [-4, 1, 2], [2, 2, 5]])
# Eigenvalues are sorted in ascending order to match the output from np.linalg.eigh
# which is used by linear_algebra methods (decompose in particular)
evals_sq3 = np.asarray([-5.51082, 3.659278, 5.851542])
# Matrix rows must be ordered such that eigenvectors correspond to correct eigenvalues
evec_sq3 = np.asarray([[-3.06513, -2.19028, 1], [2.82137, -3.49173, 1], [0.0770902, 0.348681, 1]])
evec_normd_sq3 = np.array([evec / np.linalg.norm(evec) for evec in evec_sq3])


class TestLinearAlgebra(unittest.TestCase):
    def test_replace_neg_eigenvalue(self) -> None:
        """Replacement uses 1.0 a.u. converted to kJ/(mol·Å²·amu).

        Per Limé & Norrby (J. Comput. Chem. 2015, 36, 1130, DOI:10.1002/jcc.23797),
        Method C forces the reaction coordinate eigenvalue to 1 Hartree·bohr⁻²·amu⁻¹
        = 9376 kJ·mol⁻¹·Å⁻²·amu⁻¹.  The default units=co.KJMOLA triggers this
        conversion via constants.HESSIAN_CONVERSION.
        """
        repl_evals = hessian.replace_neg_eigenvalue(evals_sq3)
        expected_replacement = 1.0 * constants.HESSIAN_CONVERSION  # ~9375.83
        np.testing.assert_allclose(
            [expected_replacement, evals_sq3[1], evals_sq3[2]],
            repl_evals,
            err_msg="Most negative eigenvalue should be replaced with 1.0 a.u. in kJ/(mol·Å²·amu) units.",
        )
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
            expected_replacement,  # most negative replaced with 1.0 a.u. converted
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
            hessian.replace_neg_eigenvalue(multi_neg_evals, zer_out_neg=True, strict=False),
            err_msg="Replaced eigenvalues do not match. Failed to replace excess negative values with zero.",
        )

    def test_multi_neg_eigenvalue_strict_raises(self) -> None:
        """Multiple negative eigenvalues should raise ValueError by default."""
        multi_neg = np.array([-5.0, -2.0, 1.0, 3.0])
        with self.assertRaises(ValueError, msg="Should raise on multiple negative eigenvalues"):
            hessian.replace_neg_eigenvalue(multi_neg)

    def test_multi_neg_eigenvalue_nonstrict_warns(self) -> None:
        """Multiple negative eigenvalues with strict=False should warn, not raise."""
        import warnings

        multi_neg = np.array([-5.0, -2.0, 1.0, 3.0])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = hessian.replace_neg_eigenvalue(multi_neg, strict=False)
            self.assertEqual(len(w), 1)
            self.assertIn("negative eigenvalues", str(w[0].message))
        # Most negative (-5.0) should be replaced
        self.assertGreater(result[0], 0.0)

    def test_reform_hessian(self) -> None:
        evals, evecs = hessian.decompose(example_sq3)
        reformed_sq3 = hessian.reform_hessian(evals, evecs)
        np.testing.assert_allclose(example_sq3, reformed_sq3, err_msg="Hessian is not reformed properly.")

    def test_hessian_to_frequencies_diatomic(self) -> None:
        """Deterministic H₂ stretch: k=0.5 Hartree/Bohr² → 5120.49 cm⁻¹."""
        k = 0.5  # Hartree/Bohr²
        hess = np.zeros((6, 6))
        # z-z block coupling between atom 1 (idx 2) and atom 2 (idx 5)
        hess[2, 2] = k
        hess[5, 5] = k
        hess[2, 5] = -k
        hess[5, 2] = -k
        freqs = hessian.hessian_to_frequencies(hess, ["H", "H"])
        self.assertEqual(len(freqs), 6)
        # 5 zero-frequency modes (translation/rotation), 1 stretch
        nonzero = [f for f in freqs if abs(f) > 1.0]
        self.assertEqual(len(nonzero), 1)
        self.assertAlmostEqual(nonzero[0], 5120.49, places=0)

    def test_hessian_to_frequencies_negative_eigenvalue(self) -> None:
        """Negative force constant → imaginary (negative) frequency."""
        k = -0.5
        hess = np.zeros((6, 6))
        hess[2, 2] = k
        hess[5, 5] = k
        hess[2, 5] = -k
        hess[5, 2] = -k
        freqs = hessian.hessian_to_frequencies(hess, ["H", "H"])
        negative = [f for f in freqs if f < -1.0]
        self.assertEqual(len(negative), 1)
        self.assertAlmostEqual(negative[0], -5120.49, places=0)

    def test_hessian_to_frequencies_no_mutation(self) -> None:
        """Input Hessian must not be mutated."""
        hess = np.eye(6) * 0.3
        original = hess.copy()
        hessian.hessian_to_frequencies(hess, ["H", "H"])
        np.testing.assert_array_equal(hess, original)

    @unittest.skip("Incomplete — requires refactoring (upstream TODO)")
    def test_last(
        self,
        force_field: FF,
        structures: list[Structure],
        hessians: list[np.ndarray],
        zero_out: bool,
        hessian_units: int = constants.GAUSSIAN,
    ) -> None:
        # TODO: This test is incomplete. `log` (a JaguarOut-like object with .evecs/.evals)
        # and `min_hessian` need to be derived from the test parameters before this test
        # can run. See issue #128.
        pass
