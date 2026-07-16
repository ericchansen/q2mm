"""Unit tests for OptaxOptimizer (no backend required)."""

from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    find_spec("optax") is None or find_spec("jax") is None,
    reason="optax and jax are required",
)


from q2mm.models.forcefield import ForceField, FunctionalForm, TorsionParam
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode
from test._shared import make_diatomic


def _synthetic_forcefield(values: np.ndarray) -> ForceField:
    return ForceField(
        name="synthetic",
        torsions=tuple(
            TorsionParam(
                elements=("C", "C", "C", "C"),
                periodicity=i + 1,
                force_constant=float(value),
                env_id=f"p{i}",
            )
            for i, value in enumerate(values)
        ),
        functional_form=FunctionalForm.HARMONIC,
    )


def _synthetic_layout(forcefield: ForceField, bounds: list[tuple[float, float]] | None) -> ParameterLayout:
    layout = ParameterLayout.from_force_field(forcefield)
    if bounds is None:
        return layout
    return ParameterLayout(
        slots=tuple(
            replace(slot, bounds=(float(lo), float(hi))) for slot, (lo, hi) in zip(layout.slots, bounds, strict=True)
        )
    )


def _synthetic_plan(
    initial: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    active_indices: np.ndarray | None = None,
) -> tuple[ForceField, ActiveParameterSpace, ObjectivePlan]:
    baseline = np.asarray(initial, dtype=np.float64)
    ff = _synthetic_forcefield(baseline)
    layout = _synthetic_layout(ff, bounds)
    space = ActiveParameterSpace(
        layout=layout,
        baseline=baseline,
        active_indices=np.arange(baseline.size, dtype=int) if active_indices is None else active_indices,
    )
    plan = ObjectivePlan(
        case_ids=("0",),
        molecules=(make_diatomic(),),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ObservationSet(),
        layout=layout,
        active_space=space,
    )
    return ff, space, plan


class QuadraticEvaluator(BaseObjectiveExecutor):
    """Synthetic quadratic evaluator with analytical full-vector gradients."""

    def __init__(
        self,
        target: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        initial: np.ndarray | None = None,
        active_indices: np.ndarray | None = None,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        baseline = (
            np.zeros_like(self.target, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
        )
        self.forcefield, self.space, plan = _synthetic_plan(baseline, bounds, active_indices)
        super().__init__(plan)
        self._gradient_mode = GradientMode.ANALYTICAL

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    def _total(self, full_vector: np.ndarray) -> float:
        return float(np.sum((np.asarray(full_vector, dtype=np.float64) - self.target) ** 2))

    def _calculated(self, full_vector: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(full_vector, dtype=np.float64) - self.target)


class DivergentEvaluator(QuadraticEvaluator):
    """Evaluator that produces increasing values to exercise divergence handling."""

    def __init__(self, n_params: int = 3) -> None:
        self._call_count = 0
        super().__init__(
            target=np.zeros(n_params, dtype=np.float64),
            initial=np.ones(n_params, dtype=np.float64),
        )

    def _total(self, full_vector: np.ndarray) -> float:
        self._call_count += 1
        return 10.0 * self._call_count

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        return np.ones_like(np.asarray(full_vector, dtype=np.float64)) * 100.0


class FrozenQuadraticEvaluator(QuadraticEvaluator):
    """Quadratic objective with one frozen full-vector coordinate."""

    def __init__(self) -> None:
        super().__init__(
            target=np.array([1.0, 4.0, 3.0], dtype=np.float64),
            bounds=[(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)],
            initial=np.array([0.0, 5.0, 0.0], dtype=np.float64),
            active_indices=np.array([0, 2], dtype=int),
        )


class TestOptaxOptimizerCreation:
    """Test optimizer instantiation and validation."""

    def test_create_adam(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adam")
        assert opt.optimizer_name == "adam"

    def test_create_adagrad(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adagrad")
        assert opt.optimizer_name == "adagrad"

    def test_create_sgd(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="sgd", momentum=0.9)
        assert opt.optimizer_name == "sgd"
        assert opt.momentum == 0.9

    def test_create_adamw(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(optimizer="adamw")
        assert opt.optimizer_name == "adamw"

    def test_invalid_optimizer(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        with pytest.raises(ValueError, match="Unknown optimizer"):
            OptaxOptimizer(optimizer="rmsprop")

    def test_invalid_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        opt = OptaxOptimizer(schedule="invalid")
        obj = QuadraticEvaluator(np.array([1.0, 2.0]))
        with pytest.raises(ValueError, match="Unknown schedule"):
            opt.optimize(obj, obj.space)


class TestOptaxConvergence:
    """Test that the optimizer converges on simple problems."""

    def test_adam_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0, 3.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert result.final_score < 0.01
        np.testing.assert_allclose(result.final_params, target, atol=0.1)

    def test_sgd_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, -1.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="sgd", learning_rate=0.1, momentum=0.9, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 0.01

    def test_adagrad_converges_quadratic(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([5.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="adagrad", learning_rate=1.0, max_steps=500, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 0.1

    def test_gradient_norm_convergence(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([0.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=1000,
            grad_norm_tol=1e-4,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.success
        assert "gradient norm" in result.message

    def test_score_plateau_convergence(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.05,
            max_steps=2000,
            ftol=1e-8,
            patience=20,
            grad_norm_tol=1e-12,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.success
        assert result.n_iterations < 2000


class TestOptaxFrozenParams:
    """Frozen parameters remain fixed throughout optimization."""

    def test_adam_updates_only_active_params(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = FrozenQuadraticEvaluator()
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=300, verbose=False)
        result = opt.optimize(obj, obj.space)

        np.testing.assert_allclose(result.initial_params, [0.0, 5.0, 0.0])
        np.testing.assert_allclose(result.final_params[[1]], [5.0])
        assert result.final_score < result.initial_score
        # REMOVED (Phase 4): objective is no longer mutated by optimizers.
        np.testing.assert_allclose(obj.space.baseline, [0.0, 5.0, 0.0])


class TestOptaxBounds:
    """Test parameter bounds enforcement."""

    def test_bounds_enforced(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([10.0, 10.0])
        bounds = [(0.0, 5.0), (0.0, 5.0)]
        obj = QuadraticEvaluator(target, bounds=bounds)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=200, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert np.all(result.final_params >= 0.0 - 1e-10)
        assert np.all(result.final_params <= 5.0 + 1e-10)

    def test_no_bounds(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([100.0])
        obj = QuadraticEvaluator(target, bounds=None)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=1.0,
            max_steps=500,
            use_bounds=False,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < 1.0


class TestOptaxDivergence:
    """Test divergence detection and early stopping."""

    def test_divergence_stops_early(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = DivergentEvaluator(n_params=3)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=1000,
            divergence_factor=3.0,
            divergence_patience=5,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert not result.success
        assert "Abandoned" in result.message
        assert result.n_iterations < 1000


class TestOptaxFinalScoreUnits:
    """The reported final_score must be in real objective-executor units."""

    def test_final_score_matches_true_objective(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0, 3.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=200, verbose=False)
        result = opt.optimize(obj, obj.space)

        true_score = float(np.sum((result.final_params - target) ** 2))
        assert result.final_score == pytest.approx(true_score, rel=1e-9, abs=1e-9)


class TestOptaxSchedules:
    """Test learning rate schedules."""

    def test_cosine_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            schedule="cosine",
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert "cosine" in result.method

    def test_exponential_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            schedule="exponential",
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        assert result.final_score < result.initial_score
        assert "exponential" in result.method


class TestOptaxResult:
    """Test OptimizationResult fields."""

    def test_result_fields(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=100, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert result.method.startswith("optax:")
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None
        assert result.n_evaluations > 0
        assert result.n_iterations > 0
        assert len(result.history) > 0
        assert result.initial_params is not None
        assert result.final_params is not None
        assert isinstance(result.improvement, float)
        assert len(result.summary()) > 0

    def test_history_tracked(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0])
        obj = QuadraticEvaluator(target)
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=50, verbose=False)
        result = opt.optimize(obj, obj.space)

        assert len(result.history) >= 2
        assert result.history[0] == result.initial_score

    def test_forcefield_is_not_mutated(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        target = np.array([1.0, 2.0])
        obj = QuadraticEvaluator(target)
        original_baseline = obj.space.baseline.copy()
        opt = OptaxOptimizer(optimizer="adam", learning_rate=0.1, max_steps=100, verbose=False)
        result = opt.optimize(obj, obj.space)

        # REMOVED (Phase 4): objective is no longer mutated by optimizers.
        np.testing.assert_array_equal(obj.space.baseline, original_baseline)
        final_ff = obj.plan.layout.replace(obj.forcefield, result.final_params)
        np.testing.assert_array_equal(obj.plan.layout.vector(final_ff), result.final_params)


class TestOptaxImport:
    """Test lazy import and registration."""

    def test_importable_from_package(self) -> None:
        from q2mm.optimizers import OptaxOptimizer

        assert OptaxOptimizer is not None

    def test_in_all(self) -> None:
        import q2mm.optimizers

        assert "OptaxOptimizer" in q2mm.optimizers.__all__


class _OvershootEvaluator(QuadraticEvaluator):
    """Analytical quadratic whose scripted gradient overshoots on the final step.

    Step 0 evaluates x0=[1.0] (score 1.0) and steps to [0.6]; step 1 evaluates
    [0.6] (score 0.36 — the best evaluated iterate) then a scripted gradient
    sends the post-update point to [2.6] (score 6.76).  The best *evaluated*
    iterate is [0.6]; the never-evaluated post-step point [2.6] must not be
    returned.
    """

    def __init__(self) -> None:
        super().__init__(target=np.array([0.0]), initial=np.array([1.0]))
        self._grad_calls = 0
        self._script = [np.array([0.4]), np.array([-2.0])]
        self.evaluated: list[np.ndarray] = []

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        self.evaluated.append(np.asarray(full_vector, dtype=np.float64).copy())
        g = self._script[min(self._grad_calls, len(self._script) - 1)]
        self._grad_calls += 1
        return g.copy()


class _FDQuadratic(QuadraticEvaluator):
    """Quadratic evaluator declaring FINITE_DIFFERENCE gradients."""

    def __init__(self, fd_step: float = 1e-3) -> None:
        super().__init__(target=np.array([0.0]), initial=np.array([1.0]))
        self._gradient_mode = GradientMode.FINITE_DIFFERENCE
        self._fd_step = fd_step


class TestOptaxBestIterateAndProvenance:
    """Regression: return the best *evaluated* iterate and correct FD provenance."""

    def test_returns_best_evaluated_iterate_not_post_step(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = _OvershootEvaluator()
        opt = OptaxOptimizer(
            optimizer="sgd",
            learning_rate=1.0,
            momentum=0.0,
            max_steps=2,
            use_bounds=False,
            divergence_factor=None,
            grad_norm_tol=0.0,
            patience=1000,
            verbose=False,
        )
        result = opt.optimize(obj, obj.space)

        scores = [float(np.sum(p**2)) for p in obj.evaluated]
        best_idx = int(np.argmin(scores))
        # Returned vector is the best EVALUATED iterate (x=0.6), not the
        # never-evaluated post-step overshoot (x=2.6).
        np.testing.assert_allclose(result.final_params, obj.evaluated[best_idx], atol=1e-9)
        assert result.final_score == pytest.approx(min(scores), rel=1e-9, abs=1e-9)
        # Score identifies the same vector it is reported against.
        assert result.final_score == pytest.approx(float(np.sum(result.final_params**2)), abs=1e-9)
        # And it is strictly better than the discarded post-step point (6.76).
        assert result.final_score < 1.0

    def test_reports_fd_step_for_fd_evaluator(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = _FDQuadratic(fd_step=1e-3)
        result = OptaxOptimizer(
            optimizer="sgd", learning_rate=0.1, max_steps=3, use_bounds=False, verbose=False
        ).optimize(obj, obj.space)
        assert result.gradient_mode == "finite_difference"
        assert result.fd_step == 1e-3

    def test_reports_none_fd_step_for_analytical_evaluator(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        obj = QuadraticEvaluator(target=np.array([0.0]), initial=np.array([1.0]))
        result = OptaxOptimizer(
            optimizer="sgd", learning_rate=0.1, max_steps=3, use_bounds=False, verbose=False
        ).optimize(obj, obj.space)
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None
