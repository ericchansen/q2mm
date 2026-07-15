"""Tests for L2 regularization in ObjectiveFunction."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

import numpy as np
import pytest

from q2mm.models.observations import ObservationSet
from q2mm.optimizers.objective import ObjectiveFunction


@dataclass(frozen=True)
class MockForceField:
    """Minimal immutable force field for ObjectiveFunction tests."""

    params: tuple[float, ...]


class MockLayout:
    """Minimal layout exposing len/vector/replace over MockForceField."""

    def __init__(self, n_params: int) -> None:
        self.n_params = n_params

    def __len__(self) -> int:
        return self.n_params

    def vector(self, forcefield: MockForceField) -> np.ndarray:
        return np.asarray(forcefield.params, dtype=np.float64)

    def replace(self, forcefield: MockForceField, vector: np.ndarray) -> MockForceField:
        values = np.asarray(vector, dtype=np.float64)
        return MockForceField(tuple(values.tolist()))


def _make_objective(
    *,
    regularization: float = 0.0,
    reference_params: np.ndarray | None = None,
    n_params: int = 3,
) -> ObjectiveFunction:
    """Build a minimal ObjectiveFunction with mock engine and no data."""
    ff = MockForceField(tuple(np.ones(n_params, dtype=np.float64).tolist()))
    layout = MockLayout(n_params)

    engine = MagicMock()
    engine.supports_batched_energy.return_value = False

    ref = ObservationSet()
    return ObjectiveFunction(
        ff,
        engine,
        [],
        ref,
        layout=layout,
        regularization=regularization,
        reference_params=reference_params,
    )


class TestL2Regularization:
    """L2 penalty on objective function."""

    def test_zero_lambda_no_penalty(self) -> None:
        obj = _make_objective(regularization=0.0)
        params = np.array([5.0, 6.0, 7.0])
        assert obj(params) == 0.0

    def test_positive_lambda_adds_penalty(self) -> None:
        ref = np.zeros(3)
        obj = _make_objective(regularization=1.0, reference_params=ref)
        params = np.array([1.0, 2.0, 3.0])
        assert obj(params) == pytest.approx(14.0)

    def test_lambda_scaling(self) -> None:
        ref = np.zeros(2)
        params = np.array([3.0, 4.0])
        obj1 = _make_objective(regularization=0.5, reference_params=ref, n_params=2)
        obj2 = _make_objective(regularization=2.0, reference_params=ref, n_params=2)
        assert obj1(params) == pytest.approx(12.5)
        assert obj2(params) == pytest.approx(50.0)

    def test_at_reference_no_penalty(self) -> None:
        ref = np.array([1.0, 2.0, 3.0])
        obj = _make_objective(regularization=100.0, reference_params=ref)
        assert obj(ref) == pytest.approx(0.0)

    def test_reference_defaults_to_initial(self) -> None:
        obj = _make_objective(regularization=1.0)
        np.testing.assert_array_equal(obj._reference_params, np.ones(3))
        assert obj(np.ones(3)) == pytest.approx(0.0)

    def test_custom_reference_params(self) -> None:
        custom = np.array([10.0, 20.0, 30.0])
        obj = _make_objective(regularization=1.0, reference_params=custom)
        np.testing.assert_array_equal(obj._reference_params, custom)


class TestL2Residuals:
    """L2 terms appended to the residual vector."""

    def test_zero_lambda_unchanged(self) -> None:
        obj = _make_objective(regularization=0.0, n_params=3)
        r = obj.residuals(np.array([1.0, 2.0, 3.0]))
        assert len(r) == 0

    def test_positive_lambda_appends_terms(self) -> None:
        ref = np.zeros(2)
        obj = _make_objective(regularization=4.0, reference_params=ref, n_params=2)
        params = np.array([3.0, 4.0])
        r = obj.residuals(params)
        np.testing.assert_array_almost_equal(r, [6.0, 8.0])
        assert float(np.sum(r**2)) == pytest.approx(100.0)

    def test_residuals_squared_equals_call(self) -> None:
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
        obj = _make_objective(regularization=0.0)
        g = obj.gradient(np.array([5.0, 6.0, 7.0]))
        np.testing.assert_array_equal(g, np.zeros(3))

    def test_positive_lambda_gradient(self) -> None:
        ref = np.zeros(3)
        obj = _make_objective(regularization=1.5, reference_params=ref)
        params = np.array([1.0, 2.0, 3.0])
        g = obj.gradient(params)
        np.testing.assert_array_almost_equal(g, [3.0, 6.0, 9.0])

    def test_gradient_at_reference_is_zero(self) -> None:
        ref = np.array([1.0, 2.0, 3.0])
        obj = _make_objective(regularization=10.0, reference_params=ref)
        g = obj.gradient(ref)
        np.testing.assert_array_almost_equal(g, np.zeros(3))


class TestL2Validation:
    """Input validation for regularization parameters."""

    def test_negative_regularization_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _make_objective(regularization=-0.1)

    def test_regularization_without_forcefield_raises(self) -> None:
        ref = ObservationSet()
        with pytest.raises(ValueError, match="requires a forcefield"):
            ObjectiveFunction(None, MagicMock(), [], ref, regularization=0.5)

    def test_reference_params_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            _make_objective(reference_params=np.ones((3, 2)))

    def test_reference_params_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match"):
            _make_objective(n_params=3, reference_params=np.ones(5))
