"""Tests for executor-driven SciPy gradient behavior."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("scipy")

from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from test._shared import make_diatomic


def _mock_engine(supports_grad: bool) -> MagicMock:
    """Return a MagicMock backend whose ``.info`` declares gradient capabilities."""
    from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

    caps: set[Capability] = {Capability.ENERGY, Capability.HESSIAN, Capability.FREQUENCIES}
    if supports_grad:
        caps |= {Capability.PARAMETER_GRADIENT, Capability.HESSIAN_PARAMETER_JACOBIAN}
    backend = MagicMock()
    backend.info = BackendInfo(
        name="mock",
        role=BackendRole.MM,
        capabilities=frozenset(caps),
        functional_forms=frozenset({"harmonic"}),
        provenance=BackendProvenance(backend="mock", role=BackendRole.MM),
    )
    backend.prepare.return_value.info = backend.info
    return backend


@dataclass(frozen=True)
class MockForceField:
    """Minimal immutable force field for optimizer tests."""

    params: tuple[float, ...]


class MockLayout:
    """Minimal layout exposing vector/replace over MockForceField."""

    def __init__(self, n_params: int) -> None:
        self.n_params = n_params

    def __len__(self) -> int:
        return self.n_params

    @property
    def fingerprint(self) -> str:
        return f"mock:{self.n_params}"

    def vector(self, forcefield: MockForceField) -> np.ndarray:
        return np.asarray(forcefield.params, dtype=np.float64)

    def replace(self, forcefield: MockForceField, vector: np.ndarray) -> MockForceField:
        values = np.asarray(vector, dtype=np.float64)
        return MockForceField(tuple(values.tolist()))


class MockSpace:
    """Active/full parameter projection used by optimizer tests."""

    def __init__(
        self,
        baseline: np.ndarray,
        bounds: list[tuple[float, float]],
        active_indices: np.ndarray | None = None,
    ) -> None:
        self.baseline = np.asarray(baseline, dtype=np.float64).copy()
        self.layout = MockLayout(self.baseline.size)
        self.active_indices = (
            np.arange(self.baseline.size, dtype=int)
            if active_indices is None
            else np.asarray(active_indices, dtype=int)
        )
        self._full_bounds = np.asarray(bounds, dtype=np.float64)

    @property
    def n_active(self) -> int:
        return int(self.active_indices.size)

    @property
    def n_full(self) -> int:
        return int(self.baseline.size)

    @property
    def bounds(self) -> np.ndarray:
        return self._full_bounds[self.active_indices]

    def pack(self, full_vector: np.ndarray) -> np.ndarray:
        full = np.asarray(full_vector, dtype=np.float64)
        return full[self.active_indices].copy()

    def expand(self, active_vector: np.ndarray, *, base: np.ndarray | None = None) -> np.ndarray:
        full = self.baseline.copy() if base is None else np.asarray(base, dtype=np.float64).copy()
        full[self.active_indices] = np.asarray(active_vector, dtype=np.float64)
        return full

    def with_baseline(self, vector: np.ndarray) -> MockSpace:
        return MockSpace(np.asarray(vector, dtype=float), self._full_bounds.tolist(), self.active_indices)


class _MockObjective:
    """Lightweight objective evaluator for testing executor-driven gradient modes."""

    def __init__(self, *, gradient_mode: GradientMode = GradientMode.NONE) -> None:
        baseline = np.array([1.0, 2.0], dtype=np.float64)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(2)
        self.space = MockSpace(baseline, bounds=[(0.0, 10.0), (0.0, 10.0)])
        self.plan = SimpleNamespace(categories=frozenset({"energy"}))
        self.history: list[float] = []
        self._n_eval = 0
        self._gradient_mode = gradient_mode

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    @property
    def finite_difference_step(self) -> float | None:
        return None

    @property
    def n_evaluations(self) -> int:
        return self._n_eval

    def record_evaluation(self, score: float) -> None:
        self._n_eval += 1
        self.history.append(float(score))

    def value(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        score = float(np.sum((x - np.array([0.5, 1.5])) ** 2))
        self._n_eval += 1
        self.history.append(score)
        return score

    def value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.gradient_mode is GradientMode.NONE:
            raise ObjectiveGradientError("No evaluator gradient available")
        value = self.value(x)
        return value, 2.0 * (np.asarray(x, dtype=float) - np.array([0.5, 1.5]))

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - np.array([0.5, 1.5])

    def least_squares_residuals(self, x: np.ndarray) -> np.ndarray:
        return self.residuals(x)


class _MockFrozenObjective:
    """Quadratic objective over a full parameter vector with frozen entries."""

    def __init__(self, *, gradient_mode: GradientMode = GradientMode.NONE) -> None:
        baseline = np.array([0.0, 5.0, 0.0], dtype=float)
        self.target = np.array([1.0, 4.0, 3.0], dtype=float)
        self.forcefield = MockForceField(tuple(baseline.tolist()))
        self.layout = MockLayout(3)
        self.space = MockSpace(
            baseline,
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            active_indices=np.array([0, 2]),
        )
        self.plan = SimpleNamespace(categories=frozenset({"energy"}))
        self.history: list[float] = []
        self._n_eval = 0
        self._gradient_mode = gradient_mode

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    @property
    def finite_difference_step(self) -> float | None:
        return None

    @property
    def n_evaluations(self) -> int:
        return self._n_eval

    def record_evaluation(self, score: float) -> None:
        self._n_eval += 1
        self.history.append(float(score))

    def value(self, x: np.ndarray) -> float:
        score = float(np.sum((np.asarray(x, dtype=float) - self.target) ** 2))
        self._n_eval += 1
        self.history.append(score)
        return score

    def value_and_gradient(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        if self.gradient_mode is GradientMode.NONE:
            raise ObjectiveGradientError("No evaluator gradient available")
        x = np.asarray(x, dtype=float)
        return self.value(x), 2.0 * (x - self.target)

    def residuals(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) - self.target

    def least_squares_residuals(self, x: np.ndarray) -> np.ndarray:
        return self.residuals(x)


def _run_ignoring_errors(opt: ScipyOptimizer, obj: _MockObjective) -> None:
    opt.optimize(obj, obj.space)


def _h2_ff() -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=359.7, equilibrium=0.74)],
        functional_form=FunctionalForm.HARMONIC,
    )


def _plan_for_kinds(kinds: tuple[str, ...]) -> ObjectivePlan:
    ff = _h2_ff()
    mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
    ref = ObservationSet()
    for kind in kinds:
        if kind == "energy":
            ref = ref.with_energy(0.0, case_id="0")
        elif kind == "frequency":
            ref = ref.with_frequency(100.0, data_idx=0, case_id="0")
        elif kind == "bond_length":
            ref = ref.with_bond_length(1.5, atom_indices=(0, 1), case_id="0")
        elif kind == "hessian_element":
            ref = ref.with_hessian_element(0.1, row=0, col=0, case_id="0")
    layout = ParameterLayout.from_force_field(ff)
    return ObjectivePlan(
        case_ids=("0",),
        molecules=(mol,),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ref,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, ff),
    )


class TestJacAutoDetection:
    """Verify the optimizer follows executor-declared gradient support."""

    def test_lbfgsb_auto_enables_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_lbfgsb_no_analytical_when_unsupported(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_lbfgsb_default_jac_none_uses_fd(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_nelder_mead_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Nelder-Mead", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_powell_never_uses_analytical(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Powell", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_explicit_analytical_overrides_auto(self, caplog: pytest.LogCaptureFixture) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, verbose=True).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_derivative_free_methods_set(self) -> None:
        assert "Nelder-Mead" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "Powell" in ScipyOptimizer.DERIVATIVE_FREE_METHODS
        assert "L-BFGS-B" not in ScipyOptimizer.DERIVATIVE_FREE_METHODS


class TestFrozenParameterSupport:
    """Frozen parameters are excluded from optimizer updates."""

    def test_lbfgsb_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_params[0] != pytest.approx(result.initial_params[0])
        assert result.final_params[2] != pytest.approx(result.initial_params[2])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])

    def test_least_squares_updates_only_active_params(self) -> None:
        obj = _MockFrozenObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="least_squares", maxiter=50, verbose=False).optimize(obj, obj.space)

        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_score < result.initial_score
        np.testing.assert_allclose(obj.layout.vector(obj.forcefield), [0.0, 5.0, 0.0])


class TestOptimizationResultFields:
    """Verify gradient_mode and fd_step are set correctly on OptimizationResult."""

    def test_lbfgsb_auto_with_support_sets_eps_none(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_lbfgsb_fd_sets_eps(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_derivative_free_sets_eps_none(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.ANALYTICAL)
        result = ScipyOptimizer(method="Powell", maxiter=1).optimize(obj, obj.space)
        assert result.gradient_mode == "none"
        assert result.fd_step is None

    def test_custom_eps_value(self) -> None:
        obj = _MockObjective(gradient_mode=GradientMode.NONE)
        result = ScipyOptimizer(method="L-BFGS-B", maxiter=1, eps=5e-4).optimize(obj, obj.space)
        assert result.fd_step == 5e-4


class TestPerEvaluatorGradientSupport:
    """Verify explicit Python executor analytical-gradient support checks."""

    @staticmethod
    def _make_objective(*, engine_supports_grad: bool, kinds: tuple[str, ...]) -> PythonObjectiveExecutor:
        ff = _h2_ff()
        return PythonObjectiveExecutor(
            _plan_for_kinds(kinds),
            _mock_engine(engine_supports_grad),
            ff,
            gradient_mode=GradientMode.ANALYTICAL,
        )

    def test_energy_and_frequency_with_analytical_engine(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy", "frequency"))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_energy_and_frequency_without_analytical_engine(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="PARAMETER_GRADIENT|HESSIAN_PARAMETER_JACOBIAN"):
            self._make_objective(engine_supports_grad=False, kinds=("energy", "frequency"))

    def test_energy_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("energy",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_frequency_only(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("frequency",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_geometry_always_false(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="geometry references"):
            self._make_objective(engine_supports_grad=True, kinds=("bond_length",))

    def test_hessian_with_support(self) -> None:
        obj = self._make_objective(engine_supports_grad=True, kinds=("hessian_element",))
        assert obj.gradient_mode is GradientMode.ANALYTICAL
