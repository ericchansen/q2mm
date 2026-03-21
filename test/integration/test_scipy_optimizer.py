"""Tests for q2mm.optimizers (objective, scipy_opt)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import make_diatomic, make_water

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.models.forcefield import BondParam, AngleParam, ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData, ReferenceValue
from q2mm.optimizers.scipy_opt import ScipyOptimizer, OptimizationResult


# ---- Helpers ----


def _diatomic(distance: float = 0.74) -> Q2MMMolecule:
    return make_diatomic(distance=distance)


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length)


def _h2_ff(k: float = 5.0, r0: float = 0.74) -> ForceField:
    return ForceField(
        name="H2-test",
        bonds=[BondParam(elements=("H", "H"), force_constant=k, equilibrium=r0)],
    )


def _water_ff(
    bond_k: float = 7.0,
    bond_r0: float = 0.96,
    angle_k: float = 0.8,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


# ---- ReferenceData ----


class TestReferenceData:
    def test_add_energy(self):
        ref = ReferenceData()
        ref.add_energy(10.0, weight=2.0, label="TS energy")
        assert ref.n_observations == 1
        assert ref.values[0].kind == "energy"
        assert ref.values[0].value == 10.0
        assert ref.values[0].weight == 2.0

    def test_add_multiple_types(self):
        ref = ReferenceData()
        ref.add_energy(5.0)
        ref.add_frequency(1200.0, data_idx=0)
        ref.add_bond_length(1.52, data_idx=0)
        ref.add_bond_angle(109.5, data_idx=0)
        assert ref.n_observations == 4

    def test_multi_molecule(self):
        ref = ReferenceData()
        ref.add_energy(5.0, molecule_idx=0)
        ref.add_energy(8.0, molecule_idx=1)
        assert ref.n_observations == 2
        assert ref.values[0].molecule_idx == 0
        assert ref.values[1].molecule_idx == 1


# ---- ObjectiveFunction ----


class TestObjectiveFunction:
    def test_callable(self):
        """Objective is callable and returns a float."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        target_energy = engine.energy(mol, ff)
        ref.add_energy(target_energy, weight=1.0)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        score = obj(ff.get_param_vector())
        assert isinstance(score, float)
        # At the target parameters, residual should be ~0
        assert score == pytest.approx(0.0, abs=1e-10)

    def test_perturbed_params_increase_score(self):
        """Perturbing parameters away from reference should increase score."""
        # Use a displaced geometry so energy depends on force constant
        mol = _diatomic(0.80)  # displaced from r0=0.74
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_energy(engine.energy(mol, ff), weight=1.0)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        base_score = obj(ff.get_param_vector())

        perturbed = ff.get_param_vector().copy()
        perturbed[0] *= 1.5  # Change force constant by 50%
        perturbed_score = obj(perturbed)
        assert perturbed_score > base_score

    def test_residuals_vector(self):
        """residuals() returns a weighted residual vector."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_energy(engine.energy(mol, ff) + 1.0, weight=2.0)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        r = obj.residuals(ff.get_param_vector())
        assert r.shape == (1,)
        assert r[0] == pytest.approx(2.0, abs=0.1)  # weight * (ref - calc) ≈ 2*1

    def test_tracks_history(self):
        """Objective tracks evaluation count and score history."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_energy(engine.energy(mol, ff))

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        obj(ff.get_param_vector())
        obj(ff.get_param_vector())
        obj(ff.get_param_vector())
        assert obj.n_eval == 3
        assert len(obj.history) == 3

    def test_reset(self):
        """reset() clears history."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()
        ref = ReferenceData()
        ref.add_energy(0.0)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        obj(ff.get_param_vector())
        obj.reset()
        assert obj.n_eval == 0
        assert len(obj.history) == 0

    def test_frequency_reference(self):
        """Objective works with frequency reference data."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        freqs = engine.frequencies(mol, ff)
        ref = ReferenceData()
        # Use highest frequency as reference
        ref.add_frequency(freqs[-1], data_idx=len(freqs) - 1, weight=0.01)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        score = obj(ff.get_param_vector())
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_out_of_range_data_idx_raises(self):
        """Out-of-range data_idx raises IndexError, not silent zero."""
        mol = _diatomic(0.74)
        ff = _h2_ff(5.0, 0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_frequency(1000.0, data_idx=999)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        with pytest.raises(IndexError, match="data_idx=999 out of range"):
            obj(ff.get_param_vector())


# ---- ForceField.get_bounds ----


class TestForceFieldBounds:
    def test_bounds_length_matches_param_vector(self):
        ff = _water_ff()
        bounds = ff.get_bounds()
        vec = ff.get_param_vector()
        assert len(bounds) == len(vec)

    def test_bounds_are_tuples(self):
        ff = _h2_ff()
        bounds = ff.get_bounds()
        for lo, hi in bounds:
            assert isinstance(lo, float)
            assert isinstance(hi, float)
            assert lo < hi

    def test_initial_params_within_bounds(self):
        ff = _water_ff()
        vec = ff.get_param_vector()
        bounds = ff.get_bounds()
        for val, (lo, hi) in zip(vec, bounds):
            assert lo <= val <= hi, f"{val} not in [{lo}, {hi}]"


# ---- ScipyOptimizer ----


class TestScipyOptimizer:
    def test_optimize_to_known_energy(self):
        """Optimizer can fit force constant to match target energies."""
        # Use two displaced geometries so both k and r0 are identifiable
        mol_short = _diatomic(0.70)
        mol_long = _diatomic(0.80)
        true_ff = _h2_ff(k=5.0, r0=0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_energy(engine.energy(mol_short, true_ff), weight=1.0, molecule_idx=0)
        ref.add_energy(engine.energy(mol_long, true_ff), weight=1.0, molecule_idx=1)

        # Start with wrong force constant and equilibrium
        guess_ff = _h2_ff(k=8.0, r0=0.78)

        obj = ObjectiveFunction(guess_ff, engine, [mol_short, mol_long], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        assert result.final_score < 1e-4
        assert result.improvement > 0.9

    def test_nelder_mead(self):
        """Nelder-Mead can optimize without bounds."""
        mol = _diatomic(0.80)
        true_ff = _h2_ff(k=5.0, r0=0.74)
        engine = OpenMMEngine()
        target_energy = engine.energy(mol, true_ff)

        guess_ff = _h2_ff(k=7.0, r0=0.74)
        ref = ReferenceData()
        ref.add_energy(target_energy, weight=1.0)

        obj = ObjectiveFunction(guess_ff, engine, [mol], ref)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=200, use_bounds=False, verbose=False)
        result = opt.optimize(obj)

        assert result.final_score < result.initial_score

    def test_least_squares(self):
        """least_squares method uses residual vector."""
        mol = _diatomic(0.80)
        true_ff = _h2_ff(k=5.0, r0=0.74)
        engine = OpenMMEngine()
        target_energy = engine.energy(mol, true_ff)

        guess_ff = _h2_ff(k=8.0, r0=0.74)
        ref = ReferenceData()
        ref.add_energy(target_energy, weight=1.0)

        obj = ObjectiveFunction(guess_ff, engine, [mol], ref)
        opt = ScipyOptimizer(method="least_squares", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        assert result.success
        assert result.final_score < 1e-6

    def test_result_summary(self):
        """OptimizationResult.summary() returns readable string."""
        mol = _diatomic(0.74)
        ff = _h2_ff(k=5.0, r0=0.74)
        engine = OpenMMEngine()
        ref = ReferenceData()
        ref.add_energy(engine.energy(mol, ff))

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=10, verbose=False)
        result = opt.optimize(obj)

        summary = result.summary()
        assert "L-BFGS-B" in summary
        assert "Score" in summary

    @pytest.mark.medium
    def test_water_bond_and_angle(self):
        """Optimizer can recover both bond and angle parameters."""
        mol = _water()
        true_ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)
        engine = OpenMMEngine()
        target_energy = engine.energy(mol, true_ff)
        target_freqs = engine.frequencies(mol, true_ff)

        # Start with perturbed parameters
        guess_ff = _water_ff(bond_k=5.0, bond_r0=1.05, angle_k=0.5, angle_eq=110.0)
        ref = ReferenceData()
        ref.add_energy(target_energy, weight=1.0)
        # Add a few key frequencies
        for i in range(len(target_freqs)):
            ref.add_frequency(target_freqs[i], data_idx=i, weight=0.001)

        obj = ObjectiveFunction(guess_ff, engine, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj)

        assert result.final_score < result.initial_score
        assert result.improvement > 0.5

    def test_params_applied_to_ff(self):
        """After optimization, forcefield has the optimized parameters."""
        mol_short = _diatomic(0.70)
        mol_long = _diatomic(0.80)
        true_ff = _h2_ff(k=5.0, r0=0.74)
        engine = OpenMMEngine()

        ref = ReferenceData()
        ref.add_energy(engine.energy(mol_short, true_ff), weight=1.0, molecule_idx=0)
        ref.add_energy(engine.energy(mol_long, true_ff), weight=1.0, molecule_idx=1)

        guess_ff = _h2_ff(k=8.0, r0=0.78)

        obj = ObjectiveFunction(guess_ff, engine, [mol_short, mol_long], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        opt.optimize(obj)

        # k should have moved substantially toward 5.0
        final_k = guess_ff.bonds[0].force_constant
        assert abs(final_k - 5.0) < 2.0
