"""Tests for robust eigendecomposition and bound-aware sensitivity."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from q2mm.models.hessian import PENALTY_FREQUENCY, hessian_to_frequencies


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _symmetric_hessian_au(n_atoms: int = 3, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Build a well-conditioned symmetric Hessian in Hartree/Bohr² with symbols."""
    rng = np.random.default_rng(seed)
    dim = 3 * n_atoms
    # A = random symmetric positive-definite
    raw = rng.standard_normal((dim, dim))
    hess = raw @ raw.T + np.eye(dim) * 10.0  # ensure positive-definite
    symbols = ["C"] * n_atoms
    return hess, symbols


def _slightly_asymmetric_hessian_au(n_atoms: int = 3, seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Build a Hessian with slight asymmetry (mimics jax.hessian floating-point drift)."""
    hess, symbols = _symmetric_hessian_au(n_atoms, seed)
    rng = np.random.default_rng(seed + 1)
    # Add small asymmetric noise (order 1e-8 relative to matrix scale)
    noise = rng.standard_normal(hess.shape) * 1e-8 * np.max(np.abs(hess))
    hess += noise
    return hess, symbols


# ===========================================================================
# Tests for hessian_to_frequencies robustness
# ===========================================================================


class TestSymmetrisation:
    """Verify that symmetrisation doesn't change results on well-conditioned input."""

    def test_symmetric_hessian_unchanged(self) -> None:
        """Frequencies from a perfectly symmetric Hessian are unchanged by symmetrisation."""
        hess, symbols = _symmetric_hessian_au()
        freqs_new = hessian_to_frequencies(hess, symbols)
        assert len(freqs_new) == 3 * len(symbols)
        # All should be real numbers (no NaN)
        assert all(np.isfinite(f) for f in freqs_new)

    def test_slightly_asymmetric_produces_valid_frequencies(self) -> None:
        """Slightly asymmetric Hessian (from autodiff) should produce valid frequencies."""
        hess, symbols = _slightly_asymmetric_hessian_au()
        # Confirm it is NOT perfectly symmetric
        assert not np.allclose(hess, hess.T, atol=0)
        freqs = hessian_to_frequencies(hess, symbols)
        assert len(freqs) == 3 * len(symbols)
        assert all(np.isfinite(f) for f in freqs)

    def test_symmetrisation_preserves_values(self) -> None:
        """Frequencies from a symmetric vs slightly-asymmetric Hessian should be very close."""
        hess_sym, symbols = _symmetric_hessian_au()
        hess_asym, _ = _slightly_asymmetric_hessian_au()
        freqs_sym = hessian_to_frequencies(hess_sym, symbols)
        freqs_asym = hessian_to_frequencies(hess_asym, symbols)
        np.testing.assert_allclose(freqs_sym, freqs_asym, atol=1e-3)


class TestOnErrorRaise:
    """Verify default on_error='raise' behavior is preserved."""

    def test_default_raises_on_linalg_error(self) -> None:
        """Default behavior: LinAlgError propagates."""
        hess, symbols = _symmetric_hessian_au()
        with (
            patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mock")),
            pytest.raises(np.linalg.LinAlgError, match="mock"),
        ):
            hessian_to_frequencies(hess, symbols)

    def test_explicit_raise_propagates(self) -> None:
        """on_error='raise' is equivalent to default."""
        hess, symbols = _symmetric_hessian_au()
        with (
            patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mock")),
            pytest.raises(np.linalg.LinAlgError),
        ):
            hessian_to_frequencies(hess, symbols, on_error="raise")

    def test_nonfinite_raises_by_default(self) -> None:
        """Non-finite Hessian entries raise LinAlgError by default."""
        hess, symbols = _symmetric_hessian_au()
        hess[0, 0] = np.nan
        with pytest.raises(np.linalg.LinAlgError, match="non-finite"):
            hessian_to_frequencies(hess, symbols)


class TestOnErrorPenalty:
    """Verify on_error='penalty' returns penalty frequencies."""

    def test_penalty_on_linalg_error(self) -> None:
        """Monkeypatched eigvalsh failure returns penalty frequencies."""
        hess, symbols = _symmetric_hessian_au()
        n_freqs = 3 * len(symbols)
        with patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("mock")):
            freqs = hessian_to_frequencies(hess, symbols, on_error="penalty")
        assert len(freqs) == n_freqs
        assert all(f == PENALTY_FREQUENCY for f in freqs)

    def test_penalty_on_nonfinite(self) -> None:
        """Non-finite Hessian entries return penalty frequencies when opt-in."""
        hess, symbols = _symmetric_hessian_au()
        hess[1, 1] = np.inf
        n_freqs = 3 * len(symbols)
        freqs = hessian_to_frequencies(hess, symbols, on_error="penalty")
        assert len(freqs) == n_freqs
        assert all(f == PENALTY_FREQUENCY for f in freqs)

    def test_penalty_on_nan(self) -> None:
        """NaN in Hessian returns penalty frequencies when opt-in."""
        hess, symbols = _symmetric_hessian_au()
        hess[2, 3] = np.nan
        freqs = hessian_to_frequencies(hess, symbols, on_error="penalty")
        assert all(f == PENALTY_FREQUENCY for f in freqs)

    def test_penalty_value_is_large(self) -> None:
        """Penalty frequency is large enough to dominate typical residuals."""
        assert PENALTY_FREQUENCY >= 1e4


# ===========================================================================
# Tests for bound-aware sensitivity
# ===========================================================================


class TestBoundAwareSensitivity:
    """Test compute_sensitivity with bounds parameter."""

    @pytest.fixture()
    def mock_objective(self) -> Any:
        """Create a mock objective with a quadratic landscape."""
        from unittest.mock import MagicMock

        from q2mm.models.forcefield import BondParam, ForceField

        ff = ForceField(
            name="test",
            bonds=[
                BondParam(elements=("C", "H"), force_constant=300.0, equilibrium=1.1),
            ],
        )
        obj = MagicMock()
        obj.forcefield = ff

        # Quadratic objective: f(x) = sum((x - x0)^2)
        x0 = ff.get_param_vector()

        def mock_call(pvec: np.ndarray) -> float:
            return float(np.sum((np.asarray(pvec) - x0) ** 2))

        obj.__call__ = mock_call
        obj.batched_scores = MagicMock(side_effect=lambda pm: np.array([mock_call(row) for row in pm]))
        return obj

    def test_no_bounds_matches_original(self, mock_objective: Any) -> None:
        """Without bounds, behavior is unchanged."""
        from q2mm.optimizers.cycling import compute_sensitivity

        result = compute_sensitivity(mock_objective, bounds=None)
        assert result.n_evals == 2 * mock_objective.forcefield.n_params + 1

    def test_bounds_shrink_steps(self, mock_objective: Any) -> None:
        """When param is near a bound, step is shrunk."""
        from q2mm.optimizers.cycling import compute_sensitivity

        ff = mock_objective.forcefield
        x0 = ff.get_param_vector()
        steps = ff.get_step_sizes()

        # Set tight upper bound for param 0: x0[0] + 0.01 (step is ~7.2)
        bounds = [(x0[i] - 100, x0[i] + 100) for i in range(len(x0))]
        bounds[0] = (x0[0] - 0.01, x0[0] + 0.01)

        result = compute_sensitivity(mock_objective, step_sizes=steps, bounds=bounds)
        # Should still evaluate all params (just with smaller step)
        assert result.n_evals == 2 * len(x0) + 1

    def test_param_at_bound_skipped(self, mock_objective: Any) -> None:
        """Param exactly at bound should be skipped (h_eff=0)."""
        from q2mm.optimizers.cycling import compute_sensitivity

        ff = mock_objective.forcefield
        x0 = ff.get_param_vector()
        steps = ff.get_step_sizes()

        # Set param 0 exactly at its upper bound
        bounds = [(x0[i] - 100, x0[i] + 100) for i in range(len(x0))]
        bounds[0] = (x0[0], x0[0])  # zero room

        result = compute_sensitivity(mock_objective, step_sizes=steps, bounds=bounds)
        # One fewer param evaluated (param 0 skipped)
        assert result.n_evals == 2 * (len(x0) - 1) + 1
        # Skipped param should have d1=0, simp_var=inf
        assert result.d1[0] == 0.0
        assert result.simp_var[0] == np.inf

    def test_ranking_handles_skipped_params(self, mock_objective: Any) -> None:
        """Skipped params should have inf simp_var (ranked among last)."""
        from q2mm.optimizers.cycling import compute_sensitivity

        ff = mock_objective.forcefield
        x0 = ff.get_param_vector()
        steps = ff.get_step_sizes()

        # Skip param 0
        bounds = [(x0[i] - 100, x0[i] + 100) for i in range(len(x0))]
        bounds[0] = (x0[0], x0[0])

        result = compute_sensitivity(mock_objective, step_sizes=steps, bounds=bounds)
        # Skipped param 0 should have inf simp_var and zero d1
        assert result.simp_var[0] == np.inf
        assert result.d1[0] == 0.0
