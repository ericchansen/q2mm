"""Integration tests for parameter cycling and sensitivity selection.

Requires OpenMM.
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    FrequencyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import param_vector, prepare_case

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import make_diatomic, make_water

from q2mm.backends.mm.openmm import OpenMMBackend
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.models.results import OptimizationResult
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.objectives.protocols import GradientMode
from q2mm.optimizers.cycling import (
    OptimizationLoop,
    SensitivityResult,
    compute_sensitivity,
)


# ---- Helpers ----


def _water_ff(
    bond_k: float = 503.6,
    bond_r0: float = 0.96,
    angle_k: float = 57.6,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=(BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0),),
        angles=(AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq),),
        functional_form=FunctionalForm.HARMONIC,
    )


def _h2_ff(k: float = 359.7, r0: float = 0.74) -> ForceField:
    return ForceField(
        name="H2-test",
        bonds=(BondParam(elements=("H", "H"), force_constant=k, equilibrium=r0),),
        functional_form=FunctionalForm.HARMONIC,
    )


def _build_objective(
    ff: ForceField,
    backend: OpenMMBackend,
    molecules: list,
    reference: ObservationSet,
) -> tuple[PythonObjectiveExecutor, ParameterLayout, ActiveParameterSpace]:
    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    plan = ObjectivePlan(
        case_ids=tuple(str(i) for i in range(len(molecules))),
        molecules=tuple(molecules),
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in molecules),
        observations=reference,
        layout=layout,
        active_space=space,
    )
    objective = PythonObjectiveExecutor(plan, backend, ff, gradient_mode=GradientMode.NONE)
    return objective, layout, space


def _make_water_objective(
    true_ff: ForceField,
    guess_ff: ForceField,
) -> tuple[PythonObjectiveExecutor, ParameterLayout, ActiveParameterSpace]:
    """Build an objective that fits guess_ff toward true_ff using energy + frequencies."""
    mol = make_water()
    backend = load_backend("openmm")
    target_energy = prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
    target_freqs = [
        float(_f)
        for _f in prepare_case(backend, mol, true_ff)
        .frequencies(FrequencyRequest(parameters=param_vector(true_ff)))
        .frequencies
    ]

    ref = ObservationSet()
    ref = ref.with_energy(target_energy, weight=1.0)
    for i in range(len(target_freqs)):
        ref = ref.with_frequency(target_freqs[i], data_idx=i, weight=0.001)

    return _build_objective(guess_ff, backend, [mol], ref)


# ---- Sensitivity Analysis ----


class TestSensitivity:
    def test_basic_ranking(self) -> None:
        """Sensitivity analysis returns valid ranking."""
        mol = make_diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=503.6, r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        sens = compute_sensitivity(obj, layout.vector(guess_ff), metric="simp_var")

        assert len(sens.d1) == 2
        assert len(sens.d2) == 2
        assert len(sens.ranking) == 2
        assert set(sens.ranking.tolist()) == {0, 1}
        assert sens.n_evals == 5

    def test_abs_d1_metric(self) -> None:
        """abs_d1 metric ranks by largest normalised |d1/step| descending."""
        mol = make_diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=503.6, r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        sens = compute_sensitivity(obj, layout.vector(guess_ff), metric="abs_d1")

        step_sizes = layout.steps
        normalised = np.where(step_sizes != 0, sens.d1 / step_sizes, 0.0)
        assert np.abs(normalised[sens.ranking[0]]) >= np.abs(normalised[sens.ranking[1]])

    def test_known_insensitive_param(self) -> None:
        """A parameter at its optimal value should have near-zero d1."""
        mol = make_diatomic(0.74)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=359.7, r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        sens = compute_sensitivity(obj, layout.vector(guess_ff))

        assert np.all(np.abs(sens.d1) < 0.01)

    def test_invalid_metric_raises(self) -> None:
        mol = make_diatomic(0.74)
        ff = _h2_ff()
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(0.0)
        obj, layout, _space = _build_objective(ff, backend, [mol], ref)

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_sensitivity(obj, layout.vector(ff), metric="bad")


# ---- OptimizationLoop ----


class TestOptimizationLoop:
    def test_selected_parameter_ids_preserve_other_slots(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The simplex pass changes only its selected semantic parameter slot."""
        true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)
        guess_ff = _water_ff(bond_k=359.7, bond_r0=1.05, angle_k=36.0, angle_eq=110.0)
        obj, layout, all_active = _make_water_objective(true_ff, guess_ff)
        baseline = layout.vector(guess_ff)
        space = all_active.with_active_indices((0, 1, 3))

        class _NoOpOptimizer:
            def optimize(
                self,
                evaluator: PythonObjectiveExecutor,
                active_space: ActiveParameterSpace,
            ) -> OptimizationResult:
                score = evaluator.value(active_space.baseline)
                return OptimizationResult(
                    success=True,
                    message="no-op",
                    initial_score=score,
                    final_score=score,
                    n_iterations=0,
                    n_evaluations=1,
                    n_params=active_space.n_full,
                    layout_fingerprint=active_space.layout.fingerprint,
                    initial_params=active_space.baseline,
                    final_params=active_space.baseline,
                )

        monkeypatch.setattr(
            OptimizationLoop,
            "_build_full_optimizer",
            lambda *_args, **_kwargs: _NoOpOptimizer(),
        )
        monkeypatch.setattr(
            "q2mm.optimizers.cycling.compute_sensitivity",
            lambda *_args, **_kwargs: SensitivityResult(
                d1=np.ones(len(layout)),
                d2=np.ones(len(layout)),
                simp_var=np.arange(len(layout), dtype=float),
                ranking=np.array([0, 1, 3, 2]),
                metric="simp_var",
                n_evals=0,
            ),
        )

        result = OptimizationLoop(
            obj,
            space,
            max_params=1,
            max_cycles=1,
            convergence=0.0,
            full_method="L-BFGS-B",
            simp_method="Nelder-Mead",
            simp_maxiter=50,
            verbose=False,
        ).run()

        changed_indices = np.flatnonzero(~np.isclose(result.final_params, baseline, rtol=0.0, atol=1e-12))
        changed_ids = tuple(layout.ids[i] for i in changed_indices)
        assert changed_ids == (layout.ids[0],)
        np.testing.assert_array_equal(result.final_params[[1, 2, 3]], baseline[[1, 2, 3]])
        assert result.n_params == len(layout)
        assert result.layout_fingerprint == layout.fingerprint
        assert len(result.stages) == 1
        stage = result.stages[0]
        assert stage.n_params == len(layout)
        assert stage.layout_fingerprint == layout.fingerprint
        assert stage.notes["selected_indices"] == (0,)
        assert result.final_score < result.initial_score

    @pytest.mark.integration
    def test_loop_improves_score(self) -> None:
        """OptimizationLoop should improve over a single-shot Nelder-Mead."""
        true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)
        guess_ff = _water_ff(bond_k=359.7, bond_r0=1.05, angle_k=36.0, angle_eq=110.0)
        obj, _layout, space = _make_water_objective(true_ff, guess_ff)

        loop = OptimizationLoop(
            obj,
            space,
            max_params=2,
            max_cycles=5,
            convergence=0.001,
            full_method="L-BFGS-B",
            simp_method="Nelder-Mead",
            full_maxiter=50,
            simp_maxiter=50,
            verbose=False,
        )
        result = loop.run()

        assert isinstance(result, OptimizationResult)
        assert result.final_score < result.initial_score
        assert result.n_iterations >= 1
        assert len(result.history) == result.n_iterations + 1
        assert len([stage.notes["selected_indices"] for stage in result.stages]) == result.n_iterations
        assert result.improvement > 0

    @pytest.mark.integration
    def test_loop_tracks_sensitivity(self) -> None:
        """Each cycle should produce a sensitivity result."""
        true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)
        guess_ff = _water_ff(bond_k=359.7, bond_r0=1.05, angle_k=36.0, angle_eq=110.0)
        obj, _layout, space = _make_water_objective(true_ff, guess_ff)

        loop = OptimizationLoop(
            obj,
            space,
            max_params=2,
            max_cycles=2,
            convergence=0.0001,
            full_maxiter=20,
            simp_maxiter=20,
            verbose=False,
        )
        result = loop.run()

        assert len(result.stages) == result.n_iterations
        for stage in result.stages:
            assert len(stage.notes["sensitivity_ranking"]) == 4


class TestConvergence:
    def test_stops_at_convergence(self) -> None:
        """Loop should stop early if already converged."""
        true_ff = _h2_ff(k=359.7, r0=0.74)
        guess_ff = _h2_ff(k=359.7, r0=0.74)
        mol = make_diatomic(0.80)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
        )

        obj, _layout, space = _build_objective(guess_ff, backend, [mol], ref)
        loop = OptimizationLoop(
            obj,
            space,
            max_params=1,
            max_cycles=10,
            convergence=0.1,
            full_maxiter=10,
            simp_maxiter=10,
            verbose=False,
        )
        result = loop.run()

        assert result.success
        assert result.n_iterations <= 2

    def test_max_cycles_limit(self) -> None:
        """Loop should respect max_cycles."""
        true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)
        guess_ff = _water_ff(bond_k=215.8, bond_r0=1.2, angle_k=21.6, angle_eq=120.0)
        obj, _layout, space = _make_water_objective(true_ff, guess_ff)

        loop = OptimizationLoop(
            obj,
            space,
            max_params=1,
            max_cycles=2,
            convergence=1e-10,
            full_maxiter=5,
            simp_maxiter=5,
            verbose=False,
        )
        result = loop.run()

        assert not result.success
        assert result.n_iterations == 2
        assert "max cycles" in result.message

    @pytest.mark.integration
    def test_summary_output(self) -> None:
        """OptimizationResult.summary() produces readable output."""
        true_ff = _h2_ff(k=359.7, r0=0.74)
        guess_ff = _h2_ff(k=503.6, r0=0.78)
        mol = make_diatomic(0.80)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
        )

        obj, _layout, space = _build_objective(guess_ff, backend, [mol], ref)
        loop = OptimizationLoop(
            obj,
            space,
            max_params=1,
            max_cycles=3,
            convergence=0.01,
            full_maxiter=20,
            simp_maxiter=20,
            verbose=False,
        )
        result = loop.run()

        summary = result.summary()
        assert "Iterations" in summary
        assert "Score" in summary
        assert "improvement" in summary
