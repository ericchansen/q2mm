"""Tests for L2 regularization in ObjectiveFunction."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData


def _make_objective(
    *,
    regularization: float = 0.0,
    reference_params: np.ndarray | None = None,
    n_params: int = 3,
) -> ObjectiveFunction:
    """Build a minimal ObjectiveFunction with mock engine and no data.

    With zero reference observations the data-loss is always 0.0, which
    isolates the L2 penalty for testing.
    """
    ff = MagicMock()
    ff.get_param_vector.return_value = np.ones(n_params)
    ff.with_params.return_value = ff
    ff.n_params = n_params

    engine = MagicMock()
    engine.supports_batched_energy.return_value = False

    ref = ReferenceData()
    # No reference values → data loss is always 0
    obj = ObjectiveFunction(
        ff,
        engine,
        [],
        ref,
        regularization=regularization,
        reference_params=reference_params,
    )
    return obj


class TestL2Regularization:
    """L2 penalty on objective function."""

    def test_zero_lambda_no_penalty(self) -> None:
        """With regularization=0.0 the score is unchanged."""
        obj = _make_objective(regularization=0.0)
        params = np.array([5.0, 6.0, 7.0])
        assert obj(params) == 0.0  # pure data loss = 0

    def test_positive_lambda_adds_penalty(self) -> None:
        """Positive λ adds ||params - ref||² to the score."""
        ref = np.zeros(3)
        obj = _make_objective(regularization=1.0, reference_params=ref)
        params = np.array([1.0, 2.0, 3.0])
        # penalty = 1.0 * (1² + 2² + 3²) = 14.0
        assert obj(params) == pytest.approx(14.0)

    def test_lambda_scaling(self) -> None:
        """Penalty scales linearly with λ."""
        ref = np.zeros(2)
        params = np.array([3.0, 4.0])
        # ||params||² = 25
        obj1 = _make_objective(regularization=0.5, reference_params=ref, n_params=2)
        obj2 = _make_objective(regularization=2.0, reference_params=ref, n_params=2)
        assert obj1(params) == pytest.approx(12.5)
        assert obj2(params) == pytest.approx(50.0)

    def test_at_reference_no_penalty(self) -> None:
        """When params == reference_params the penalty is zero."""
        ref = np.array([1.0, 2.0, 3.0])
        obj = _make_objective(regularization=100.0, reference_params=ref)
        assert obj(ref) == pytest.approx(0.0)

    def test_reference_defaults_to_initial(self) -> None:
        """Without explicit reference_params, uses ff.get_param_vector()."""
        obj = _make_objective(regularization=1.0)
        # ff.get_param_vector() returns np.ones(3)
        np.testing.assert_array_equal(obj._reference_params, np.ones(3))
        # Evaluating at the reference → zero penalty
        assert obj(np.ones(3)) == pytest.approx(0.0)

    def test_custom_reference_params(self) -> None:
        """Explicit reference_params override the default."""
        custom = np.array([10.0, 20.0, 30.0])
        obj = _make_objective(regularization=1.0, reference_params=custom)
        np.testing.assert_array_equal(obj._reference_params, custom)


class TestL2Residuals:
    """L2 terms appended to the residual vector."""

    def test_zero_lambda_unchanged(self) -> None:
        """With λ=0, residuals() returns only data residuals."""
        obj = _make_objective(regularization=0.0, n_params=3)
        r = obj.residuals(np.array([1.0, 2.0, 3.0]))
        # No reference values → empty data residuals
        assert len(r) == 0

    def test_positive_lambda_appends_terms(self) -> None:
        """Positive λ appends sqrt(λ) * (p - ref) to residuals."""
        ref = np.zeros(2)
        obj = _make_objective(regularization=4.0, reference_params=ref, n_params=2)
        params = np.array([3.0, 4.0])
        r = obj.residuals(params)
        # Should be sqrt(4) * [3, 4] = [6, 8]
        np.testing.assert_array_almost_equal(r, [6.0, 8.0])
        # sum(r²) = 36 + 64 = 100 = 4 * 25 = λ * ||params||²
        assert float(np.sum(r**2)) == pytest.approx(100.0)

    def test_residuals_squared_equals_call(self) -> None:
        """sum(residuals²) should equal __call__ result."""
        ref = np.zeros(3)
        obj1 = _make_objective(regularization=2.5, reference_params=ref, n_params=3)
        obj2 = _make_objective(regularization=2.5, reference_params=ref, n_params=3)
        params = np.array([1.0, 2.0, 3.0])
        score = obj1(params)
        r = obj2.residuals(params)
        assert float(np.sum(r**2)) == pytest.approx(score)


class TestL2Gradient:
    """L2 gradient contribution: 2λ(p - p_ref)."""

    def test_zero_lambda_no_gradient(self) -> None:
        """With λ=0 the gradient has no L2 contribution."""
        obj = _make_objective(regularization=0.0)
        # gradient() goes through evaluators, but with no refs it's all zeros
        g = obj.gradient(np.array([5.0, 6.0, 7.0]))
        np.testing.assert_array_equal(g, np.zeros(3))

    def test_positive_lambda_gradient(self) -> None:
        """Positive λ adds 2λ(p - ref) to the gradient."""
        ref = np.zeros(3)
        obj = _make_objective(regularization=1.5, reference_params=ref)
        params = np.array([1.0, 2.0, 3.0])
        g = obj.gradient(params)
        # 2 * 1.5 * [1, 2, 3] = [3, 6, 9]
        np.testing.assert_array_almost_equal(g, [3.0, 6.0, 9.0])

    def test_gradient_at_reference_is_zero(self) -> None:
        """At the reference point the L2 gradient is zero."""
        ref = np.array([1.0, 2.0, 3.0])
        obj = _make_objective(regularization=10.0, reference_params=ref)
        g = obj.gradient(ref)
        np.testing.assert_array_almost_equal(g, np.zeros(3))
