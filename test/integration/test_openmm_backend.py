"""Integration tests for the OpenMM MM backend."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import SN2_XYZ as TS_XYZ, SN2_HESSIAN as TS_HESS, make_diatomic, make_water, make_noble_gas_pair

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants

try:
    from q2mm.backends.mm.tinker import TinkerEngine

    _tinker_engine = TinkerEngine()
    HAS_TINKER = _tinker_engine.is_available()
    TINKER_PARAMS = _tinker_engine._params_file
except (ImportError, FileNotFoundError):
    HAS_TINKER = False
    TINKER_PARAMS = None


def _diatomic(distance: float) -> Q2MMMolecule:
    return make_diatomic(distance=distance)


def _water(angle_deg: float = 109.5, bond_length: float = 0.96) -> Q2MMMolecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length, name="water-like")


def _noble_gas_pair(distance: float, atom_type: str = "He") -> Q2MMMolecule:
    return make_noble_gas_pair(distance=distance, atom_type=atom_type)


class TestOpenMMEngine:
    def setup_method(self):
        self.engine = OpenMMEngine()

    def test_name_and_runtime_update_support(self):
        assert "OpenMM" in self.engine.name
        assert self.engine.is_available()
        assert self.engine.supports_runtime_params() is True

    def test_mm3_bond_energy_matches_reference_formula(self):
        molecule = _diatomic(0.84)
        forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        energy = self.engine.energy(molecule, forcefield)

        delta = 0.84 - 0.74
        # Canonical: E = k·Δr²·(1 − c3·Δr + c4·Δr²), k in kcal/mol/Å²
        expected_kcal = 71.9 * delta**2 * (1.0 - 2.55 * delta + (7.0 / 12.0) * 2.55**2 * delta**2)
        assert energy == pytest.approx(expected_kcal)

    def test_mm3_angle_energy_matches_reference_formula(self):
        molecule = _water(angle_deg=120.0)
        forcefield = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        energy = self.engine.energy(molecule, forcefield)

        delta_deg = 120.0 - 104.5
        delta_rad = np.deg2rad(delta_deg)
        # Canonical: E = k·Δθ²·(1 + higher-order), k in kcal/mol/rad²
        expected_kcal = (
            36.0
            * delta_rad**2
            * (1.0 - 0.014 * delta_deg + 5.6e-5 * delta_deg**2 - 7.0e-7 * delta_deg**3 + 9.0e-10 * delta_deg**4)
        )
        assert energy == pytest.approx(expected_kcal)

    def test_mm3_vdw_energy_matches_reference_formula(self):
        molecule = _noble_gas_pair(3.5)
        forcefield = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)])

        energy = self.engine.energy(molecule, forcefield)

        rv = 2.4
        expected = 0.02 * (-2.25 * (rv / 3.5) ** 6 + 184000.0 * np.exp(-12.0 * 3.5 / rv))
        assert energy == pytest.approx(expected)

    def test_energy_is_near_zero_at_equilibrium(self):
        molecule = _diatomic(0.74)
        forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        energy = self.engine.energy(molecule, forcefield)

        assert energy == pytest.approx(0.0, abs=1.0e-8)

    def test_update_forcefield_reuses_context(self):
        molecule = _diatomic(1.00)
        initial_forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])
        handle = self.engine.create_context(molecule, initial_forcefield)
        initial_energy = self.engine.energy(handle)

        updated_forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=1.00, force_constant=71.9)])
        self.engine.update_forcefield(handle, updated_forcefield)
        updated_energy = self.engine.energy(handle)

        assert updated_energy < initial_energy

    def test_update_forcefield_reuses_context_for_vdw(self):
        molecule = _noble_gas_pair(3.0)
        initial_forcefield = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.01)])
        handle = self.engine.create_context(molecule, initial_forcefield)
        initial_energy = self.engine.energy(handle)

        updated_forcefield = ForceField(vdws=[VdwParam("He", radius=1.6, epsilon=0.02)])
        self.engine.update_forcefield(handle, updated_forcefield)
        updated_energy = self.engine.energy(handle)

        assert updated_energy != pytest.approx(initial_energy)

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    def test_openmm_matches_tinker_for_mm3_bond_energy(self):
        tinker = TinkerEngine()
        forcefield = ForceField.from_tinker_prm(TINKER_PARAMS)
        molecule = Q2MMMolecule(
            symbols=["C", "H"],
            atom_types=["1", "5"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]]),
            name="CH-bond",
            bond_tolerance=1.5,
        )

        openmm_energy = self.engine.energy(molecule, forcefield)
        tinker_energy = tinker.energy(molecule)

        assert openmm_energy == pytest.approx(tinker_energy, abs=1.0e-3)

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    def test_openmm_matches_tinker_for_mm3_vdw_energy(self):
        tinker = TinkerEngine()
        forcefield = ForceField.from_tinker_prm(TINKER_PARAMS)
        molecule = Q2MMMolecule(
            symbols=["F", "F"],
            atom_types=["11", "11"],
            geometry=np.array([[0.0, 0.0, 0.0], [3.50, 0.0, 0.0]]),
            name="F2-nonbonded",
            bond_tolerance=0.5,
        )

        openmm_energy = self.engine.energy(molecule, forcefield)
        tinker_energy = tinker.energy(molecule)

        assert openmm_energy == pytest.approx(tinker_energy, abs=1.0e-3)

    def test_minimize_lowers_energy(self):
        molecule = _diatomic(1.20)
        forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=107.9)])
        initial_energy = self.engine.energy(molecule, forcefield)
        initial_distance = np.linalg.norm(molecule.geometry[0] - molecule.geometry[1])

        final_energy, atoms, coords = self.engine.minimize(molecule, forcefield, tolerance=1.0, max_iterations=200)
        final_distance = np.linalg.norm(coords[0] - coords[1])

        assert atoms == ["H", "H"]
        assert final_energy <= initial_energy + 1.0e-6
        assert abs(final_distance - 0.74) < abs(initial_distance - 0.74)

    def test_hessian_is_symmetric(self):
        molecule = _diatomic(0.80)
        forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        hessian = self.engine.hessian(molecule, forcefield)

        assert hessian.shape == (6, 6)
        np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-6)

    def test_frequencies_return_three_n_modes(self):
        molecule = _water(angle_deg=120.0)
        forcefield = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        frequencies = self.engine.frequencies(molecule, forcefield)

        assert len(frequencies) == 9
        assert all(np.isfinite(freq) for freq in frequencies)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_seminario_pipeline_energy_is_finite(self):
        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        forcefield = estimate_force_constants(molecule)

        energy = self.engine.energy(molecule, forcefield)
        hessian = self.engine.hessian(molecule, forcefield)

        assert np.isfinite(energy)
        assert hessian.shape == (18, 18)
        np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-6)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_seminario_pipeline_has_imaginary_mode(self):
        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        forcefield = estimate_force_constants(molecule)

        frequencies = self.engine.frequencies(molecule, forcefield)

        assert len(frequencies) == 18
        assert all(np.isfinite(freq) for freq in frequencies)
        assert min(frequencies) < -1.0
