"""Tests for analytical Hessian-based gradients (frequency, hessian_element, eigenmatrix).

Unit tests using synthetic Hessians and mock engines — no JAX needed.
Validates the eigenvalue sensitivity formula and evaluator gradient chains.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from q2mm.models.hessian import (
    decompose,
    frequency_param_jacobian,
    hessian_to_frequencies,
    mass_weighted_eigenmatrix,
    mass_weighted_normal_modes,
    transform_to_eigenmatrix,
)
from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator
from q2mm.optimizers.evaluators.hessian_element import HessianElementEvaluator
from q2mm.models.observations import Observation


def _make_symmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    """Create a random symmetric positive-definite matrix."""
    A = rng.standard_normal((n, n))
    return A @ A.T + np.eye(n) * 0.1


def _make_mock_engine(hess: np.ndarray, dH_dp: np.ndarray) -> MagicMock:
    """Create a mock engine that returns given Hessian and Jacobian."""
    engine = MagicMock()
    engine.supports_analytical_hessian_gradients.return_value = True
    engine.hessian_and_param_jacobian.return_value = (hess, dH_dp)
    engine.hessian.return_value = hess
    return engine


def _make_mol(symbols: list[str], hessian: np.ndarray | None = None) -> MagicMock:
    """Create a mock molecule."""
    mol = MagicMock()
    mol.symbols = symbols
    mol.name = "test_mol"
    mol.hessian = hessian
    return mol


# ---------------------------------------------------------------------------
# frequency_param_jacobian: pure NumPy tests
# ---------------------------------------------------------------------------


class TestFrequencyParamJacobian:
    """Tests for frequency_param_jacobian in hessian.py."""

    def test_basic_shape(self) -> None:
        """Output shapes match (3N,) and (3N, n_params)."""
        rng = np.random.default_rng(42)
        n_atoms = 3
        n3 = 3 * n_atoms
        n_params = 5

        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        # Symmetrise each slice
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H", "H"]
        freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, symbols)

        assert len(freqs) == n3
        assert d_freq_dp.shape == (n3, n_params)

    def test_sorted_output(self) -> None:
        """Frequencies are returned sorted ascending by default."""
        rng = np.random.default_rng(123)
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        for j in range(2):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H"]
        freqs, _ = frequency_param_jacobian(hess, dH_dp, symbols)

        assert freqs == sorted(freqs)

    def test_unsorted_output(self) -> None:
        """With sort=False, output order matches eigenvalue order."""
        rng = np.random.default_rng(456)
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        for j in range(2):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H"]
        freqs_sorted, jac_sorted = frequency_param_jacobian(hess, dH_dp, symbols, sort=True)
        freqs_unsorted, jac_unsorted = frequency_param_jacobian(hess, dH_dp, symbols, sort=False)

        # Same values, possibly different order
        np.testing.assert_allclose(sorted(freqs_unsorted), freqs_sorted, rtol=1e-12)

    def test_frequencies_match_hessian_to_frequencies(self) -> None:
        """Frequencies from frequency_param_jacobian match hessian_to_frequencies."""
        rng = np.random.default_rng(789)
        hess = _make_symmetric(9, rng)
        dH_dp = rng.standard_normal((9, 9, 3))
        for j in range(3):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H", "O"]
        freqs_new, _ = frequency_param_jacobian(hess, dH_dp, symbols)
        freqs_ref = hessian_to_frequencies(hess, symbols)

        np.testing.assert_allclose(freqs_new, freqs_ref, rtol=1e-10)

    def test_jacobian_vs_finite_difference(self) -> None:
        """Analytical Jacobian matches finite-difference perturbation."""
        rng = np.random.default_rng(1001)
        n_atoms = 2
        n3 = 6
        n_params = 3

        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H"]
        freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, symbols)

        # Finite difference: perturb the Hessian by dH_dp * delta for each param
        delta = 1e-6
        for j in range(n_params):
            hess_plus = hess + delta * dH_dp[:, :, j]
            hess_minus = hess - delta * dH_dp[:, :, j]
            freqs_plus = hessian_to_frequencies(hess_plus, symbols)
            freqs_minus = hessian_to_frequencies(hess_minus, symbols)
            fd_deriv = (np.array(freqs_plus) - np.array(freqs_minus)) / (2 * delta)
            np.testing.assert_allclose(d_freq_dp[:, j], fd_deriv, rtol=1e-4, atol=1e-4)

    def test_zero_jacobian_gives_zero_freq_jacobian(self) -> None:
        """If dH/dp is all zeros, frequency derivatives should be zero."""
        rng = np.random.default_rng(2002)
        hess = _make_symmetric(6, rng)
        dH_dp = np.zeros((6, 6, 2))

        symbols = ["C", "H"]
        _, d_freq_dp = frequency_param_jacobian(hess, dH_dp, symbols)

        np.testing.assert_allclose(d_freq_dp, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# FrequencyEvaluator gradient tests
# ---------------------------------------------------------------------------


class TestFrequencyEvaluatorGradient:
    """Test FrequencyEvaluator.gradient() with mock engine."""

    def test_supports_analytical_gradient_true(self) -> None:
        """Returns True when engine supports Hessian param Jacobians."""
        engine = MagicMock()
        engine.supports_analytical_hessian_gradients.return_value = True
        evaluator = FrequencyEvaluator()
        assert evaluator.supports_analytical_gradient(engine) is True

    def test_supports_analytical_gradient_false(self) -> None:
        """Returns False when engine does not support Hessian param Jacobians."""
        engine = MagicMock()
        engine.supports_analytical_hessian_gradients.return_value = False
        evaluator = FrequencyEvaluator()
        assert evaluator.supports_analytical_gradient(engine) is False

    def test_gradient_shape(self) -> None:
        """Gradient vector has the right shape."""
        rng = np.random.default_rng(3003)
        n3 = 6
        n_params = 4
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"])
        ff = MagicMock()

        freqs = hessian_to_frequencies(hess, ["C", "H"])
        refs = [
            Observation(kind="frequency", value=freqs[3] + 10.0, weight=1.0, data_idx=3, case_id="0"),
        ]

        evaluator = FrequencyEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params)
        assert grad.shape == (n_params,)

    def test_gradient_vs_finite_difference(self) -> None:
        """Evaluator gradient matches finite-difference of residuals score."""
        rng = np.random.default_rng(4004)
        n3 = 6
        n_params = 3
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)

        symbols = ["C", "H"]
        freqs = hessian_to_frequencies(hess, symbols)

        refs = [
            Observation(kind="frequency", value=freqs[2] + 5.0, weight=1.5, data_idx=2, case_id="0"),
            Observation(kind="frequency", value=freqs[4] - 3.0, weight=0.8, data_idx=4, case_id="0"),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(symbols)
        ff = MagicMock()

        evaluator = FrequencyEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params)

        # FD: perturb Hessian, recompute score
        delta = 1e-6
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            for sign, coeff in [(1, 1.0), (-1, -1.0)]:
                h_pert = hess + sign * delta * dH_dp[:, :, j]
                f_pert = hessian_to_frequencies(h_pert, symbols)
                score = sum((r.weight * (r.value - f_pert[r.data_idx])) ** 2 for r in refs)
                fd_grad[j] += coeff * score
            fd_grad[j] /= 2 * delta

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# HessianElementEvaluator gradient tests
# ---------------------------------------------------------------------------


class TestHessianElementEvaluatorGradient:
    """Test HessianElementEvaluator.gradient() with mock engine."""

    def test_supports_analytical_gradient(self) -> None:
        """Returns True/False based on engine capability."""
        evaluator = HessianElementEvaluator()
        engine_yes = MagicMock()
        engine_yes.supports_analytical_hessian_gradients.return_value = True
        assert evaluator.supports_analytical_gradient(engine_yes) is True

        engine_no = MagicMock()
        engine_no.supports_analytical_hessian_gradients.return_value = False
        assert evaluator.supports_analytical_gradient(engine_no) is False

    def test_gradient_shape(self) -> None:
        """Gradient has correct shape."""
        rng = np.random.default_rng(5005)
        n3 = 6
        n_params = 3
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"])
        ff = MagicMock()

        refs = [
            Observation(
                kind="hessian_element",
                value=hess[1, 2] + 0.01,
                weight=1.0,
                data_idx=0,
                case_id="0",
                atom_indices=(1, 2),
            ),
        ]

        evaluator = HessianElementEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params)
        assert grad.shape == (n_params,)

    def test_gradient_vs_finite_difference(self) -> None:
        """Evaluator gradient matches FD of score."""
        rng = np.random.default_rng(6006)
        n3 = 6
        n_params = 2
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        row, col = 2, 3
        ref_val = hess[row, col] + 0.005
        w = 2.0
        refs = [
            Observation(
                kind="hessian_element",
                value=ref_val,
                weight=w,
                data_idx=0,
                case_id="0",
                atom_indices=(row, col),
            ),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"])
        ff = MagicMock()

        evaluator = HessianElementEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params)

        # FD
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            score_p = (w * (ref_val - h_p[row, col])) ** 2
            score_m = (w * (ref_val - h_m[row, col])) ** 2
            fd_grad[j] = (score_p - score_m) / (2 * delta)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)


# ---------------------------------------------------------------------------
# EigenmatrixEvaluator gradient tests
# ---------------------------------------------------------------------------


class TestEigenmatrixEvaluatorGradient:
    """Test EigenmatrixEvaluator.gradient() with mock engine."""

    def test_supports_analytical_gradient(self) -> None:
        """Returns True/False based on engine capability."""
        evaluator = EigenmatrixEvaluator()
        engine_yes = MagicMock()
        engine_yes.supports_analytical_hessian_gradients.return_value = True
        assert evaluator.supports_analytical_gradient(engine_yes) is True

        engine_no = MagicMock()
        engine_no.supports_analytical_hessian_gradients.return_value = False
        assert evaluator.supports_analytical_gradient(engine_no) is False

    def test_gradient_diagonal_shape(self) -> None:
        """Gradient for diagonal eigenmatrix element has correct shape."""
        rng = np.random.default_rng(7007)
        n3 = 6
        n_params = 3
        hess = _make_symmetric(n3, rng)
        qm_hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"], hessian=qm_hess)
        ff = MagicMock()

        _, qm_evecs = decompose(qm_hess)
        eigmat = transform_to_eigenmatrix(hess, qm_evecs)

        refs = [
            Observation(
                kind="eig_diagonal",
                value=eigmat[2, 2] + 0.01,
                weight=1.0,
                data_idx=2,
                case_id="0",
            ),
        ]

        evaluator = EigenmatrixEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params, mol_idx=0)
        assert grad.shape == (n_params,)

    def test_gradient_diagonal_vs_fd(self) -> None:
        """Diagonal eigenmatrix gradient matches FD."""
        rng = np.random.default_rng(8008)
        n3 = 6
        n_params = 2
        hess = _make_symmetric(n3, rng)
        qm_hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        _, qm_evecs = mass_weighted_normal_modes(qm_hess, ["C", "H"])
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, ["C", "H"])
        idx = 3
        ref_val = eigmat[idx, idx] + 0.005
        w = 1.5
        refs = [
            Observation(
                kind="eig_diagonal",
                value=ref_val,
                weight=w,
                data_idx=idx,
                case_id="0",
            ),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"], hessian=qm_hess)
        ff = MagicMock()

        evaluator = EigenmatrixEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params, mol_idx=0)

        # FD
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            em_p = mass_weighted_eigenmatrix(h_p, qm_evecs, ["C", "H"])
            em_m = mass_weighted_eigenmatrix(h_m, qm_evecs, ["C", "H"])
            score_p = (w * (ref_val - em_p[idx, idx])) ** 2
            score_m = (w * (ref_val - em_m[idx, idx])) ** 2
            fd_grad[j] = (score_p - score_m) / (2 * delta)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_gradient_offdiagonal_vs_fd(self) -> None:
        """Off-diagonal eigenmatrix gradient matches FD."""
        rng = np.random.default_rng(9009)
        n3 = 6
        n_params = 2
        hess = _make_symmetric(n3, rng)
        qm_hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        _, qm_evecs = mass_weighted_normal_modes(qm_hess, ["C", "H"])
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, ["C", "H"])
        row, col = 1, 4
        ref_val = eigmat[row, col] + 0.003
        w = 1.0
        refs = [
            Observation(
                kind="eig_offdiagonal",
                value=ref_val,
                weight=w,
                data_idx=0,
                case_id="0",
                atom_indices=(row, col),
            ),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"], hessian=qm_hess)
        ff = MagicMock()

        evaluator = EigenmatrixEvaluator()
        grad = evaluator.gradient(engine, mol, ff, refs, n_params, mol_idx=0)

        # FD
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            em_p = mass_weighted_eigenmatrix(h_p, qm_evecs, ["C", "H"])
            em_m = mass_weighted_eigenmatrix(h_m, qm_evecs, ["C", "H"])
            score_p = (w * (ref_val - em_p[row, col])) ** 2
            score_m = (w * (ref_val - em_m[row, col])) ** 2
            fd_grad[j] = (score_p - score_m) / (2 * delta)

        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_caches_qm_eigenvectors(self) -> None:
        """Eigenvectors are cached across calls."""
        rng = np.random.default_rng(1010)
        n3 = 6
        n_params = 2
        hess = _make_symmetric(n3, rng)
        qm_hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        _, qm_evecs = mass_weighted_normal_modes(qm_hess, ["C", "H"])
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, ["C", "H"])
        refs = [
            Observation(kind="eig_diagonal", value=eigmat[0, 0], weight=1.0, data_idx=0, case_id="0"),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"], hessian=qm_hess)
        ff = MagicMock()

        evaluator = EigenmatrixEvaluator()
        evaluator.gradient(engine, mol, ff, refs, n_params, mol_idx=5)

        assert 5 in evaluator._qm_eigenvectors
        np.testing.assert_allclose(evaluator._qm_eigenvectors[5], qm_evecs, atol=1e-14)

    def test_no_qm_hessian_raises(self) -> None:
        """Raises ValueError if molecule has no QM Hessian."""
        rng = np.random.default_rng(1111)
        n3 = 6
        n_params = 2
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))

        refs = [
            Observation(kind="eig_diagonal", value=0.0, weight=1.0, data_idx=0, case_id="0"),
        ]

        engine = _make_mock_engine(hess, dH_dp)
        mol = _make_mol(["C", "H"], hessian=None)
        ff = MagicMock()

        evaluator = EigenmatrixEvaluator()
        with pytest.raises(ValueError, match="no QM Hessian"):
            evaluator.gradient(engine, mol, ff, refs, n_params, mol_idx=0)


# ---------------------------------------------------------------------------
# Base engine API tests
# ---------------------------------------------------------------------------


class TestBaseEngineHessianAPI:
    """Test base engine defaults for new Hessian Jacobian API."""

    def test_default_returns_false(self) -> None:
        from q2mm.backends.base import MMEngine

        class _StubEngine(MMEngine):
            @property
            def name(self) -> str:
                return "stub"

            def energy(self, structure: Any, forcefield: Any) -> float:
                return 0.0

            def hessian(self, structure: Any, forcefield: Any) -> np.ndarray:
                return np.zeros((3, 3))

            def minimize(self, structure: Any, forcefield: Any, max_iterations: int = 200) -> tuple:
                return 0.0, [], np.zeros((1, 3))

        engine = _StubEngine()
        assert engine.supports_analytical_hessian_gradients() is False

    def test_default_raises(self) -> None:
        from q2mm.backends.base import MMEngine

        class _StubEngine(MMEngine):
            @property
            def name(self) -> str:
                return "stub"

            def energy(self, structure: Any, forcefield: Any) -> float:
                return 0.0

            def hessian(self, structure: Any, forcefield: Any) -> np.ndarray:
                return np.zeros((3, 3))

            def minimize(self, structure: Any, forcefield: Any, max_iterations: int = 200) -> tuple:
                return 0.0, [], np.zeros((1, 3))

        engine = _StubEngine()
        with pytest.raises(NotImplementedError):
            engine.hessian_and_param_jacobian(MagicMock(), MagicMock())
