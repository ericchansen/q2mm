"""Tests for q2mm.optimizers (objective, scipy_opt)."""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    FrequencyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import param_vector, prepare_case

import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import make_diatomic, make_water

from q2mm.backends.mm.openmm import OpenMMBackend
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.objectives.protocols import GradientMode
from q2mm.optimizers.scipy_opt import ScipyOptimizer


# ---- Helpers ----


def _diatomic(distance: float = 0.74) -> Molecule:
    return make_diatomic(distance=distance)


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Molecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length)


def _h2_ff(k: float = 359.7, r0: float = 0.74) -> ForceField:
    return ForceField(
        name="H2-test",
        bonds=(BondParam(elements=("H", "H"), force_constant=k, equilibrium=r0),),
        functional_form=FunctionalForm.HARMONIC,
    )


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


# ---- ObservationSet ----


class TestObservationSet:
    def test_add_energy(self) -> None:
        ref = ObservationSet()
        ref = ref.with_energy(10.0, weight=2.0, label="TS energy")
        assert ref.n_observations == 1
        assert ref.values[0].kind == "energy"
        assert ref.values[0].value == 10.0
        assert ref.values[0].weight == 2.0

    def test_add_multiple_types(self) -> None:
        ref = ObservationSet()
        ref = ref.with_energy(5.0)
        ref = ref.with_frequency(1200.0, data_idx=0)
        ref = ref.with_bond_length(1.52, data_idx=0)
        ref = ref.with_bond_angle(109.5, data_idx=0)
        assert ref.n_observations == 4

    def test_multi_molecule(self) -> None:
        ref = ObservationSet()
        ref = ref.with_energy(5.0, case_id="0")
        ref = ref.with_energy(8.0, case_id="1")
        assert ref.n_observations == 2
        assert ref.values[0].case_id == "0"
        assert ref.values[1].case_id == "1"


# ---- Objective executor ----


class TestObjectiveExecutor:
    def test_callable(self) -> None:
        """Objective is callable and returns a float."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        target_energy = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        score = obj.value(layout.vector(ff))
        assert isinstance(score, float)
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_perturbed_params_increase_score(self) -> None:
        """Perturbing parameters away from reference should increase score."""
        mol = _diatomic(0.80)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy, weight=1.0
        )

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        base_score = obj.value(layout.vector(ff))

        perturbed = layout.vector(ff).copy()
        perturbed[0] *= 1.5
        perturbed_score = obj.value(perturbed)
        assert perturbed_score > base_score

    def test_residuals_vector(self) -> None:
        """residuals() returns a weighted residual vector."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy + 1.0, weight=2.0
        )

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        r = obj.residuals(layout.vector(ff))
        assert r.shape == (1,)
        assert r[0] == pytest.approx(2.0, abs=0.1)

    def test_tracks_history(self) -> None:
        """Objective tracks evaluation count and score history."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_energy(prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy)

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        params = layout.vector(ff)
        obj.value(params)
        obj.value(params)
        obj.value(params)
        assert obj.n_evaluations == 3
        assert len(obj.history) == 3

    def test_reset(self) -> None:
        """reset() clears history."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(0.0)

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        obj.value(layout.vector(ff))
        obj.reset()
        assert obj.n_evaluations == 0
        assert len(obj.history) == 0

    def test_frequency_reference(self) -> None:
        """Objective works with frequency reference data."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        ref = ObservationSet()
        ref = ref.with_frequency(freqs[-1], data_idx=len(freqs) - 1, weight=0.01)

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        score = obj.value(layout.vector(ff))
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_out_of_range_data_idx_raises(self) -> None:
        """Out-of-range data_idx raises IndexError, not silent zero."""
        mol = _diatomic(0.74)
        ff = _h2_ff(359.7, 0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_frequency(1000.0, data_idx=999)

        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        with pytest.raises(IndexError, match="data_idx=999 out of range"):
            obj.value(layout.vector(ff))


# ---- ParameterLayout.bounds ----


class TestForceFieldBounds:
    def test_bounds_length_matches_param_vector(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        bounds = layout.bounds
        vec = layout.vector(ff)
        assert len(bounds) == len(vec)

    def test_bounds_are_tuples(self) -> None:
        ff = _h2_ff()
        layout = ParameterLayout.from_force_field(ff)
        bounds = [tuple(map(float, row)) for row in layout.bounds.tolist()]
        for lo, hi in bounds:
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            assert lo < hi

    def test_initial_params_within_bounds(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        bounds = layout.bounds.tolist()
        for val, (lo, hi) in zip(vec, bounds):
            assert lo <= val <= hi, f"{val} not in [{lo}, {hi}]"


# ---- ScipyOptimizer ----


class TestScipyOptimizer:
    def test_optimize_to_known_energy(self) -> None:
        """Optimizer can fit force constant to match target energies."""
        mol_short = _diatomic(0.70)
        mol_long = _diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol_short, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id="0",
        )
        ref = ref.with_energy(
            prepare_case(backend, mol_long, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id="1",
        )

        guess_ff = _h2_ff(k=575.5, r0=0.78)

        obj, _layout, space = _build_objective(guess_ff, backend, [mol_short, mol_long], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        assert result.final_score < 1e-3
        assert result.improvement > 0.9

    def test_nelder_mead(self) -> None:
        """Nelder-Mead can optimize without bounds."""
        mol = _diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=503.6, r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, _layout, space = _build_objective(guess_ff, backend, [mol], ref)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=200, use_bounds=False, verbose=False)
        result = opt.optimize(obj, space)

        assert result.final_score < result.initial_score

    def test_least_squares(self) -> None:
        """least_squares method uses residual vector."""
        mol = _diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )

        guess_ff = _h2_ff(k=575.5, r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)

        obj, _layout, space = _build_objective(guess_ff, backend, [mol], ref)
        opt = ScipyOptimizer(method="least_squares", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        assert result.success
        assert result.final_score < 1e-6

    def test_result_summary(self) -> None:
        """OptimizationResult.summary() returns readable string."""
        mol = _diatomic(0.74)
        ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")
        ref = ObservationSet()
        ref = ref.with_energy(prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy)

        obj, _layout, space = _build_objective(ff, backend, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=10, verbose=False)
        result = opt.optimize(obj, space)

        summary = result.summary()
        assert "L-BFGS-B" in summary
        assert "Score" in summary

    @pytest.mark.integration
    def test_water_bond_and_angle(self) -> None:
        """Optimizer can recover both bond and angle parameters."""
        mol = _water()
        true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)
        backend = load_backend("openmm")
        target_energy = (
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy
        )
        target_freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, true_ff)
            .frequencies(FrequencyRequest(parameters=param_vector(true_ff)))
            .frequencies
        ]

        guess_ff = _water_ff(bond_k=359.7, bond_r0=1.05, angle_k=36.0, angle_eq=110.0)
        ref = ObservationSet()
        ref = ref.with_energy(target_energy, weight=1.0)
        for i in range(len(target_freqs)):
            ref = ref.with_frequency(target_freqs[i], data_idx=i, weight=0.001)

        obj, _layout, space = _build_objective(guess_ff, backend, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj, space)

        assert result.final_score < result.initial_score
        assert result.improvement > 0.5

    def test_params_applied_to_ff(self) -> None:
        """After optimization, a replaced forcefield has the optimized parameters."""
        mol_short = _diatomic(0.70)
        mol_long = _diatomic(0.80)
        true_ff = _h2_ff(k=359.7, r0=0.74)
        backend = load_backend("openmm")

        ref = ObservationSet()
        ref = ref.with_energy(
            prepare_case(backend, mol_short, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id="0",
        )
        ref = ref.with_energy(
            prepare_case(backend, mol_long, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id="1",
        )

        guess_ff = _h2_ff(k=575.5, r0=0.78)

        obj, layout, space = _build_objective(guess_ff, backend, [mol_short, mol_long], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        optimized_ff = layout.replace(guess_ff, result.final_params)
        final_k = optimized_ff.bonds[0].force_constant
        assert abs(final_k - 359.7) < 143.9
        assert guess_ff.bonds[0].force_constant == 575.5
