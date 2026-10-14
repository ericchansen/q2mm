"""Unit tests for parameter layouts and cycling result containers."""

from __future__ import annotations

from typing import Any

from q2mm.settings import CyclingSettings, OptimizerSettings
from q2mm.optimizers.protocols import _Optimizer

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, TorsionParam, VdwParam, FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterKind, ParameterLayout
from q2mm.models.results import OptimizationResult
from q2mm.optimizers.cycling import CyclingOptimizer
from test._shared import QuadraticEvaluator


def _full_ff() -> ForceField:
    """Build a FF with all parameter types for testing layout ordering."""
    return ForceField(
        name="test-full",
        bonds=[
            BondParam(elements=("C", "F"), force_constant=359.7, equilibrium=1.38),
            BondParam(elements=("C", "H"), force_constant=338.1, equilibrium=1.11),
        ],
        angles=[
            AngleParam(elements=("F", "C", "F"), force_constant=71.9, equilibrium=109.5),
        ],
        torsions=[
            TorsionParam(elements=("F", "C", "C", "F"), force_constant=0.5),
            TorsionParam(elements=("H", "C", "C", "H"), force_constant=0.3),
        ],
        vdws=[
            VdwParam(atom_type="C1", radius=1.7, epsilon=0.05),
        ],
        functional_form=FunctionalForm.HARMONIC,
    )


class TestParamIndicesByType:
    def test_correct_keys(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        expected_keys = {
            ParameterKind.BOND_FORCE_CONSTANT,
            ParameterKind.BOND_EQUILIBRIUM,
            ParameterKind.ANGLE_FORCE_CONSTANT,
            ParameterKind.ANGLE_EQUILIBRIUM,
            ParameterKind.TORSION_FORCE_CONSTANT,
            ParameterKind.VDW_RADIUS,
            ParameterKind.VDW_EPSILON,
        }
        assert set(indices.keys()) == expected_keys

    def test_bond_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.BOND_FORCE_CONSTANT] == (0, 2)
        assert indices[ParameterKind.BOND_EQUILIBRIUM] == (1, 3)

    def test_angle_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.ANGLE_FORCE_CONSTANT] == (4,)
        assert indices[ParameterKind.ANGLE_EQUILIBRIUM] == (5,)

    def test_torsion_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.TORSION_FORCE_CONSTANT] == (6, 7)

    def test_vdw_indices(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        assert indices[ParameterKind.VDW_RADIUS] == (8,)
        assert indices[ParameterKind.VDW_EPSILON] == (9,)

    def test_total_indices_match_layout_length(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        indices = layout.indices_by_kind
        all_indices: list[int] = []
        for idx_list in indices.values():
            all_indices.extend(idx_list)
        assert len(all_indices) == len(layout)
        assert sorted(all_indices) == list(range(len(layout)))

    def test_empty_ff(self) -> None:
        layout = ParameterLayout.from_force_field(ForceField(name="empty", functional_form=FunctionalForm.HARMONIC))
        assert layout.indices_by_kind == {}

    def test_indices_match_param_vector_values(self) -> None:
        ff = _full_ff()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        indices = layout.indices_by_kind
        assert vec[indices[ParameterKind.BOND_FORCE_CONSTANT][0]] == 359.7
        assert vec[indices[ParameterKind.BOND_EQUILIBRIUM][0]] == 1.38
        assert vec[indices[ParameterKind.ANGLE_FORCE_CONSTANT][0]] == 71.9
        assert vec[indices[ParameterKind.TORSION_FORCE_CONSTANT][0]] == 0.5
        assert vec[indices[ParameterKind.VDW_RADIUS][0]] == 1.7


class TestParamTypeLabels:
    def test_length_matches(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        labels = [kind.value for kind in layout.kinds]
        assert len(labels) == len(layout)

    def test_label_values(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        labels = [kind.value for kind in layout.kinds]
        assert labels[0] == "bond_k"
        assert labels[1] == "bond_eq"
        assert labels[2] == "bond_k"
        assert labels[3] == "bond_eq"
        assert labels[4] == "angle_k"
        assert labels[5] == "angle_eq"
        assert labels[6] == "torsion_k"
        assert labels[7] == "torsion_k"
        assert labels[8] == "vdw_radius"
        assert labels[9] == "vdw_epsilon"


class TestStepSizes:
    def test_length_matches(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert len(steps) == len(layout)

    def test_per_type_values(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert steps[0] == pytest.approx(7.2)
        assert steps[1] == pytest.approx(0.02)
        assert steps[4] == pytest.approx(7.2)
        assert steps[5] == pytest.approx(1.0)
        assert steps[6] == pytest.approx(0.1)
        assert steps[8] == pytest.approx(0.1)
        assert steps[9] == pytest.approx(0.02)

    def test_all_positive(self) -> None:
        layout = ParameterLayout.from_force_field(_full_ff())
        steps = layout.steps
        assert np.all(steps > 0)


class TestSensitivityResult:
    def test_dataclass_fields(self) -> None:
        from q2mm.optimizers.cycling import SensitivityResult

        sr = SensitivityResult(
            d1=np.array([1.0, 2.0]),
            d2=np.array([0.5, 1.0]),
            simp_var=np.array([0.5, 0.25]),
            ranking=np.array([1, 0]),
            metric="simp_var",
            n_evals=5,
        )
        assert sr.metric == "simp_var"
        assert sr.n_evals == 5
        assert len(sr.ranking) == 2


class TestCyclingOptimizerResult:
    def test_improvement(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            message="",
            initial_score=100.0,
            final_score=10.0,
            n_iterations=3,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=500,
        )
        assert lr.improvement == pytest.approx(0.9)

    def test_summary(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            initial_score=100.0,
            final_score=10.0,
            n_iterations=3,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=500,
            message="converged",
        )
        s = lr.summary()
        assert "converged" in s
        assert "90.0%" in s

    def test_zero_initial_score(self) -> None:
        from q2mm.models.results import OptimizationResult

        lr = OptimizationResult(
            success=True,
            message="",
            initial_score=0.0,
            final_score=0.0,
            n_iterations=0,
            n_params=0,
            layout_fingerprint="sha256:test",
            initial_params=np.zeros(0),
            final_params=np.zeros(0),
            n_evaluations=0,
        )
        assert lr.improvement == 0.0


def _scripted_loop(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[tuple[bool, float]],
    max_cycles: int = 3,
) -> tuple[CyclingOptimizer, QuadraticEvaluator]:
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    evaluator = QuadraticEvaluator(
        target=np.array([1.0, 5.0]),
        initial=np.array([0.0, 5.0]),
        active_indices=np.array([0]),
    )
    passes = iter(outcomes)

    def run_pass(
        self: ScipyOptimizer, evaluator: QuadraticEvaluator, space: ActiveParameterSpace
    ) -> OptimizationResult:
        success, value = next(passes)
        final_params = space.expand(np.array([value]))
        return OptimizationResult(
            success=success,
            message="inner converged" if success else "inner budget exhausted",
            initial_score=evaluator.value(space.baseline),
            final_score=evaluator.value(final_params),
            initial_params=space.baseline,
            final_params=final_params,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            n_iterations=1,
            n_evaluations=2,
            gradient_mode="analytical",
        )

    monkeypatch.setattr(ScipyOptimizer, "optimize", run_pass)
    return CyclingOptimizer(CyclingSettings(max_cycles=max_cycles, verbose=False)), evaluator


class TestCyclingOptimizerStops:
    @pytest.mark.parametrize("full_success, subspace_success", [(False, False), (False, True), (True, False)])
    @pytest.mark.parametrize("full_value, subspace_value", [(0.0, 0.0), (0.002, 0.001), (0.002, 0.004)])
    def test_stall_requires_both_passes_to_converge(
        self,
        monkeypatch: pytest.MonkeyPatch,
        full_success: bool,
        subspace_success: bool,
        full_value: float,
        subspace_value: float,
    ) -> None:
        loop, evaluator = _scripted_loop(monkeypatch, [(full_success, full_value), (subspace_success, subspace_value)])
        result = loop.optimize(evaluator, evaluator.space)

        assert not result.success
        assert result.n_iterations == 1
        assert "stalled" in result.message
        assert "inner budget exhausted" in result.message
        assert not result.stages[0].converged
        assert result.stages[0].notes["full_converged"] is full_success
        assert result.stages[0].notes["subspace_converged"] is subspace_success
        assert "inner budget exhausted" in result.stages[0].message
        np.testing.assert_array_equal(result.final_params, [max(full_value, subspace_value), 5.0])
        assert result.final_score == pytest.approx(evaluator.sample(result.final_params))

    def test_successful_small_change_converges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, evaluator = _scripted_loop(monkeypatch, [(True, 0.002), (True, 0.004)])
        result = loop.optimize(evaluator, evaluator.space)

        assert result.success
        assert result.message == "converged"
        assert result.stages[0].converged
        assert result.n_iterations == 1
        np.testing.assert_array_equal(result.final_params, [0.004, 5.0])

    def test_progress_exhausts_cycle_budget_without_convergence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, evaluator = _scripted_loop(
            monkeypatch, [(True, 0.2), (True, 0.3), (True, 0.4), (True, 0.5)], max_cycles=2
        )
        result = loop.optimize(evaluator, evaluator.space)

        assert not result.success
        assert result.message == "max cycles (2) reached"
        assert result.n_iterations == 2
        assert all(stage.converged for stage in result.stages)
        np.testing.assert_array_equal(result.final_params, [0.5, 5.0])

    def test_failed_progress_can_be_followed_by_successful_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop, evaluator = _scripted_loop(monkeypatch, [(False, 0.2), (False, 0.3), (True, 0.3), (True, 0.3)])
        result = loop.optimize(evaluator, evaluator.space)

        assert result.success
        assert result.n_iterations == 2
        assert [stage.converged for stage in result.stages] == [False, True]
        np.testing.assert_array_equal(result.final_params, [0.3, 5.0])


def test_phase_factories_are_bound_before_execution_and_fresh_each_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.optimizers import catalog

    build = catalog.build_optimizer
    constructed = []
    configured = []

    def factory(settings: OptimizerSettings) -> _Optimizer:
        configured.append(settings)
        optimizer = build(settings)
        constructed.append(optimizer)
        return optimizer

    monkeypatch.setattr(catalog, "build_optimizer", factory)
    optimizer, evaluator = _scripted_loop(
        monkeypatch, [(True, 0.2), (True, 0.3), (True, 0.4), (True, 0.5)], max_cycles=2
    )
    assert constructed == []
    assert evaluator.n_evaluations == 0

    def forbidden_resolution(_settings: OptimizerSettings) -> _Optimizer:
        raise AssertionError("Catalog resolution must not be repeated during numerical execution")

    monkeypatch.setattr(catalog, "build_optimizer", forbidden_resolution)
    result = optimizer.optimize(evaluator, evaluator.space)
    assert result.n_iterations == 2
    assert len({id(item) for item in constructed}) == 4
    assert configured == [optimizer.settings.full_optimizer, optimizer.settings.simplex_optimizer] * 2
    assert all(item.settings is settings for item, settings in zip(constructed, configured, strict=True))


@pytest.mark.parametrize("failure_phase", ["initial", "full"])
def test_error_policy_is_restored_on_failure(monkeypatch: pytest.MonkeyPatch, failure_phase: str) -> None:
    evaluator = QuadraticEvaluator(np.ones(2), initial=np.array([2.0, 3.0]))
    baseline = evaluator.space.baseline.copy()
    evaluator.on_error = "raise"

    def fail(*_args: object) -> Any:
        assert evaluator.on_error == "penalty"
        raise RuntimeError("scripted failure")

    if failure_phase == "initial":
        monkeypatch.setattr(evaluator, "value", fail)
    optimizer = CyclingOptimizer(full_optimizer_factory=fail)
    with pytest.raises(RuntimeError, match="scripted failure"):
        optimizer.optimize(evaluator, evaluator.space)
    assert evaluator.on_error == "raise"
    np.testing.assert_array_equal(evaluator.space.baseline, baseline)


def test_sensitivity_formula_steps_frozen_slots_and_counts() -> None:
    from q2mm.optimizers.cycling import compute_sensitivity

    evaluator = QuadraticEvaluator(np.array([0.0, 1.0]), initial=np.array([2.0, 3.0]))
    sensitivity = compute_sensitivity(evaluator, evaluator.space.baseline, step_sizes=np.array([0.25, 0.5]))
    np.testing.assert_allclose(sensitivity.d1, [1.0, 2.0])
    np.testing.assert_allclose(sensitivity.d2, [0.125, 0.5])
    np.testing.assert_allclose(sensitivity.simp_var, [0.125, 0.125])
    assert sensitivity.n_evals == evaluator.n_evaluations == 5
    assert len(evaluator.history) == 5

    count_before = evaluator.n_evaluations
    frozen = compute_sensitivity(evaluator, evaluator.space.baseline, step_sizes=np.array([0.0, 0.5]), metric="abs_d1")
    np.testing.assert_allclose(frozen.d1, [0.0, 2.0])
    np.testing.assert_allclose(frozen.d2, [0.0, 0.5])
    np.testing.assert_array_equal(frozen.ranking, [1, 0])
    assert frozen.n_evals == evaluator.n_evaluations - count_before == 3


def test_cycling_capture_does_not_allocate_and_rejects_unconfigured_factories() -> None:
    from q2mm.configuration import ConfigurationCaptureError

    settings = CyclingSettings()
    assert CyclingOptimizer(settings).configuration_settings() == settings.model_dump(mode="json")
    with pytest.raises(ConfigurationCaptureError, match="configuration_settings"):
        CyclingOptimizer(full_optimizer_factory=lambda: None).configuration_settings()

    class ConfiguredFactory:
        def __call__(self) -> _Optimizer:
            raise AssertionError("Configuration capture must not allocate a numerical session")

        def configuration_settings(self) -> dict[str, Any]:
            return {"custom_solver": "scripted-v1", "max_steps": 7}

    optimizer = CyclingOptimizer(full_optimizer_factory=ConfiguredFactory())
    captured = optimizer.configuration_settings()
    assert captured["full_optimizer"]["parameters"] == {"custom_solver": "scripted-v1", "max_steps": 7}
