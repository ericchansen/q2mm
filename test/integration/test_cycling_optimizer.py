"""Integration tests for parameter cycling (SubspaceObjective, sensitivity, OptimizationLoop).

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
from q2mm.optimizers.cycling import (
    LoopResult,
    OptimizationLoop,
    SubspaceObjective,
    compute_sensitivity,
)
from q2mm.optimizers.objective import ObjectiveFunction


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
) -> tuple[ObjectiveFunction, ParameterLayout, ActiveParameterSpace]:
    layout = ParameterLayout.from_force_field(ff)
    objective = ObjectiveFunction(ff, backend, molecules, reference, layout=layout)
    space = ActiveParameterSpace.all_active(layout, ff)
    return objective, layout, space


def _make_water_objective(
    true_ff: ForceField,
    guess_ff: ForceField,
) -> tuple[ObjectiveFunction, ParameterLayout, ActiveParameterSpace]:
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


# ---- SubspaceObjective ----


class TestSubspaceObjective:
    def test_full_subspace_matches_full_objective(self) -> None:
        """When all indices are active, SubspaceObjective == ObjectiveFunction."""
        mol = make_diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=503.6, r0=0.78)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        full_vec = layout.vector(guess_ff)

        sub_obj = SubspaceObjective(obj, [0, 1], full_vec)
        assert sub_obj(full_vec) == pytest.approx(obj(full_vec), rel=1e-10)

    def test_single_param_subspace(self) -> None:
        """Optimising one param while holding the other fixed."""
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
        full_vec = layout.vector(guess_ff)

        sub_obj = SubspaceObjective(obj, [0], full_vec)
        score_at_7 = sub_obj(np.array([503.6]))
        score_at_5 = sub_obj(np.array([359.7]))
        assert score_at_5 < score_at_7

    def test_residuals(self) -> None:
        """residuals() returns array of correct length."""
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
        full_vec = layout.vector(guess_ff)
        sub_obj = SubspaceObjective(obj, [0], full_vec)

        residuals = sub_obj.residuals(np.array([503.6]))
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == 1

    def test_get_bounds(self) -> None:
        """Bounds are correctly subset."""
        guess_ff = _h2_ff(k=503.6, r0=0.74)
        mol = make_diatomic(0.74)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(0.0)
        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)

        full_vec = layout.vector(guess_ff)
        sub_obj = SubspaceObjective(obj, [1], full_vec)
        bounds = sub_obj.get_bounds()
        all_bounds = [tuple(row) for row in layout.bounds.tolist()]
        assert bounds == [all_bounds[1]]

    def test_empty_indices_raises(self) -> None:
        guess_ff = _h2_ff()
        mol = make_diatomic(0.74)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(0.0)
        obj, layout, _space = _build_objective(guess_ff, backend, [mol], ref)

        with pytest.raises(ValueError, match="empty"):
            SubspaceObjective(obj, [], layout.vector(guess_ff))


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

        obj, _layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        sens = compute_sensitivity(obj, metric="simp_var")

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
        sens = compute_sensitivity(obj, metric="abs_d1")

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

        obj, _layout, _space = _build_objective(guess_ff, backend, [mol], ref)
        sens = compute_sensitivity(obj)

        assert np.all(np.abs(sens.d1) < 0.01)

    def test_invalid_metric_raises(self) -> None:
        mol = make_diatomic(0.74)
        ff = _h2_ff()
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(0.0)
        obj, _layout, _space = _build_objective(ff, backend, [mol], ref)

        with pytest.raises(ValueError, match="Unknown metric"):
            compute_sensitivity(obj, metric="bad")


# ---- OptimizationLoop ----


class TestOptimizationLoop:
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

        assert isinstance(result, LoopResult)
        assert result.final_score < result.initial_score
        assert result.n_cycles >= 1
        assert len(result.cycle_scores) == result.n_cycles + 1
        assert len(result.selected_indices) == result.n_cycles
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

        assert len(result.sensitivity_results) == result.n_cycles
        for sens in result.sensitivity_results:
            assert len(sens.d1) == 4
            assert len(sens.ranking) == 4


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
        assert result.n_cycles <= 2

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
        assert result.n_cycles == 2
        assert "max cycles" in result.message

    @pytest.mark.integration
    def test_summary_output(self) -> None:
        """LoopResult.summary() produces readable output."""
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
        assert "Cycles" in summary
        assert "Score" in summary
        assert "Improvement" in summary
