"""Tests for ScipyOptimizer auto-detection of analytical gradients.

Unit tests that use stub engines — no backend imports needed.
"""

from __future__ import annotations

import contextlib
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

from q2mm.optimizers.scipy_opt import ScipyOptimizer


class _MockObjective:
    """Lightweight mock of ObjectiveFunction for testing jac resolution."""

    def __init__(self, *, engine_supports_grad: bool = False) -> None:
        self.engine = MagicMock()
        self.engine.supports_analytical_gradients.return_value = engine_supports_grad
        self.forcefield = MagicMock()
        self.forcefield.get_param_vector.return_value = np.array([1.0, 2.0])
        self.forcefield.get_bounds.return_value = [(0.0, 10.0), (0.0, 10.0)]
        self.history: list[float] = []
        self.n_eval = 0

    def __call__(self, x: np.ndarray) -> float:
        self.n_eval += 1
        self.history.append(1.0)
        return 1.0

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.1, 0.2])


def _run_ignoring_errors(opt: ScipyOptimizer, obj: _MockObjective) -> None:
    """Run optimizer, suppressing errors from mock returning non-standard types."""
    with contextlib.suppress(Exception):
        opt.optimize(obj)


class TestJacAutoDetection:
    """Verify the optimizer auto-detects analytical gradient support."""

    def test_lbfgsb_auto_enables_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        """L-BFGS-B with jac='auto' should auto-detect and use analytical gradients."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected analytical gradient support" in caplog.text

    def test_lbfgsb_no_analytical_when_unsupported(self, caplog: pytest.LogCaptureFixture) -> None:
        """L-BFGS-B with jac='auto' should fall back when engine doesn't support grads."""
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text

    def test_lbfgsb_default_jac_none_uses_fd(self, caplog: pytest.LogCaptureFixture) -> None:
        """L-BFGS-B with default jac=None should NOT auto-detect, even if engine supports grads."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True)

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text
        assert "analytical" not in caplog.text.lower()

    def test_nelder_mead_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        """Nelder-Mead is derivative-free — should never auto-detect even with jac='auto'."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text
        assert "analytical" not in caplog.text.lower()

    def test_powell_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        """Powell is derivative-free — should never auto-detect even with jac='auto'."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Powell", maxiter=1, verbose=True, jac="auto")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Auto-detected" not in caplog.text

    def test_explicit_analytical_overrides_auto(self, caplog: pytest.LogCaptureFixture) -> None:
        """Explicit jac='analytical' should log differently from auto-detect."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True, jac="analytical")

        with caplog.at_level(logging.INFO):
            _run_ignoring_errors(opt, obj)

        assert "Using analytical gradients (jac='analytical')" in caplog.text
        assert "Auto-detected" not in caplog.text

    def test_derivative_free_methods_set(self) -> None:
        """Verify the derivative-free method set is correct."""
        assert "Nelder-Mead" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "Powell" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "L-BFGS-B" not in ScipyOptimizer.DERIVATIVE_FREE_METHODS
