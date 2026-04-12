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
from q2mm.diagnostics.benchmark import _resolve_gradients
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData


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


class TestOptimizationResultFields:
    """Verify jac_mode and eps are set correctly on OptimizationResult."""

    def test_lbfgsb_auto_with_support_sets_eps_none(self) -> None:
        """When analytical gradients are used, eps should be None."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac="auto")
        result = opt.optimize(obj)
        assert result.jac_mode == "auto"
        assert result.eps is None

    def test_lbfgsb_fd_sets_eps(self) -> None:
        """When using FD gradients, eps should be set to the configured value."""
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac=None)
        result = opt.optimize(obj)
        assert result.jac_mode is None
        assert result.eps == 1e-3

    def test_derivative_free_sets_eps_none(self) -> None:
        """Derivative-free methods should have eps=None."""
        obj = _MockObjective(engine_supports_grad=True)
        opt = ScipyOptimizer(method="Powell", maxiter=1, jac="auto")
        result = opt.optimize(obj)
        assert result.jac_mode == "auto"
        assert result.eps is None

    def test_custom_eps_value(self) -> None:
        """Custom eps value should be recorded when FD is used."""
        obj = _MockObjective(engine_supports_grad=False)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=1, jac=None, eps=5e-4)
        result = opt.optimize(obj)
        assert result.eps == 5e-4


class TestResolveGradients:
    """Verify _resolve_gradients produces correct per-evaluator gradient maps."""

    @staticmethod
    def _make_objective(
        *, engine_supports_grad: bool, kinds: tuple[str, ...] = ("energy", "frequency")
    ) -> ObjectiveFunction:
        """Build a minimal ObjectiveFunction with real evaluators and mock engine."""
        engine = MagicMock()
        engine.supports_analytical_gradients.return_value = engine_supports_grad
        engine.supports_analytical_hessian_gradients.return_value = engine_supports_grad
        ff = MagicMock()
        ref = ReferenceData()
        for kind in kinds:
            if kind == "energy":
                ref.add_energy(0.0)
            elif kind == "frequency":
                ref.add_frequency(100.0, data_idx=0)
            elif kind == "bond_length":
                ref.add_bond_length(1.5, atom_indices=(0, 1))
            elif kind == "hessian_element":
                ref.add_hessian_element(0.1, row=0, col=0)
        return ObjectiveFunction(ff, engine, [], ref)

    def test_auto_with_analytical_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "frequency": "analytical"}

    def test_auto_without_analytical_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=False)
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "finite-diff", "frequency": "finite-diff"}

    def test_jac_none_is_fd(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients(None, obj)
        assert result == {"energy": "finite-diff", "frequency": "finite-diff"}

    def test_analytical_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("analytical", obj)
        assert result == {"energy": "analytical", "frequency": "analytical"}

    def test_derivative_free_method_overrides_jac(self) -> None:
        """Even if jac_mode='auto', a derivative-free method gets n/a."""
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj, method="Powell")
        assert result == {"energy": "n/a", "frequency": "n/a"}

    def test_nelder_mead_is_derivative_free(self) -> None:
        obj = self._make_objective(engine_supports_grad=True)
        result = _resolve_gradients("auto", obj, method="Nelder-Mead")
        assert result == {"energy": "n/a", "frequency": "n/a"}

    def test_energy_only_objective(self) -> None:
        """When objective has only energy refs, frequency is absent from output."""
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical"}

    def test_frequency_only_objective(self) -> None:
        """When objective has only frequency refs, energy is absent from output."""
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        result = _resolve_gradients("auto", obj)
        assert result == {"frequency": "analytical"}

    def test_geometry_refs_always_fd(self) -> None:
        """Geometry evaluator doesn't support analytical gradients."""
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "bond_length"))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "geometry": "finite-diff"}

    def test_hessian_refs_with_support(self) -> None:
        """Hessian element evaluator supports analytical gradients with capable engine."""
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "hessian_element"))
        result = _resolve_gradients("auto", obj)
        assert result == {"energy": "analytical", "hessian": "analytical"}


class TestPerEvaluatorGradientSupport:
    """Verify ObjectiveFunction.per_evaluator_gradient_support()."""

    @staticmethod
    def _make_objective(*, engine_supports_grad: bool, kinds: tuple[str, ...]) -> ObjectiveFunction:
        engine = MagicMock()
        engine.supports_analytical_gradients.return_value = engine_supports_grad
        engine.supports_analytical_hessian_gradients.return_value = engine_supports_grad
        ref = ReferenceData()
        for kind in kinds:
            if kind == "energy":
                ref.add_energy(0.0)
            elif kind == "frequency":
                ref.add_frequency(100.0, data_idx=0)
            elif kind == "bond_length":
                ref.add_bond_length(1.5, atom_indices=(0, 1))
            elif kind == "hessian_element":
                ref.add_hessian_element(0.1, row=0, col=0)
        return ObjectiveFunction(MagicMock(), engine, [], ref)

    def test_energy_and_frequency_with_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "frequency"))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": True, "frequency": True}

    def test_energy_and_frequency_without_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=False, kinds=("energy", "frequency"))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": False, "frequency": False}

    def test_energy_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"energy": True}

    def test_frequency_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"frequency": True}

    def test_geometry_always_false(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("bond_length",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"geometry": False}

    def test_hessian_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("hessian_element",))
        result = obj.per_evaluator_gradient_support()
        assert result == {"hessian": True}

    def test_result_is_sorted_by_category(self) -> None:
        """Categories are returned in sorted order for deterministic metadata."""
        obj = self._make_objective(
            engine_supports_grad=True,
            kinds=("frequency", "energy", "hessian_element", "bond_length"),
        )
        result = obj.per_evaluator_gradient_support()
        assert list(result.keys()) == ["energy", "frequency", "geometry", "hessian"]
