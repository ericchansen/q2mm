"""OpenMM-engine-specific tests.

Contract tests (energy, hessian, frequencies, minimize, gradients) are
in test_engine_contract.py and run for every registered engine.  This
file covers only behaviour unique to the OpenMM backend:

* MM3 formula known-value checks (cubic bond, sextic angle, buffered 14-7 vdW)
* Context reuse / update_forcefield API
* Cross-backend parity with Tinker
* Seminario force-constant estimation pipeline
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import SN2_HESSIAN as TS_HESS, SN2_XYZ as TS_XYZ, make_diatomic, make_noble_gas_pair, make_water

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


class TestOpenMMEngine:
    def setup_method(self) -> None:
        self.engine = OpenMMEngine()

    def test_mm3_bond_energy_matches_reference_formula(self) -> None:
        molecule = make_diatomic(distance=0.84)
        forcefield = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])

        delta = 0.84 - 0.74
        expected_kcal = 71.9 * delta**2 * (1.0 - 2.55 * delta + (7.0 / 12.0) * 2.55**2 * delta**2)
        assert self.engine.energy(molecule, forcefield) == pytest.approx(expected_kcal)

    def test_mm3_angle_energy_matches_reference_formula(self) -> None:
        molecule = make_water(angle_deg=120.0)
        forcefield = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
        )

        delta_deg = 120.0 - 104.5
        delta_rad = np.deg2rad(delta_deg)
        expected_kcal = (
            36.0
            * delta_rad**2
            * (1.0 - 0.014 * delta_deg + 5.6e-5 * delta_deg**2 - 7.0e-7 * delta_deg**3 + 9.0e-10 * delta_deg**4)
        )
        assert self.engine.energy(molecule, forcefield) == pytest.approx(expected_kcal)

    def test_mm3_vdw_energy_matches_reference_formula(self) -> None:
        molecule = make_noble_gas_pair(distance=3.5)
        forcefield = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)])

        rv = 2.4
        expected = 0.02 * (-2.25 * (rv / 3.5) ** 6 + 184000.0 * np.exp(-12.0 * 3.5 / rv))
        assert self.engine.energy(molecule, forcefield) == pytest.approx(expected)

    def test_update_forcefield_reuses_context(self) -> None:
        molecule = make_diatomic(distance=1.00)
        initial_ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)])
        handle = self.engine.create_context(molecule, initial_ff)
        initial_energy = self.engine.energy(handle)

        updated_ff = ForceField(bonds=[BondParam(("H", "H"), equilibrium=1.00, force_constant=71.9)])
        self.engine.update_forcefield(handle, updated_ff)
        assert self.engine.energy(handle) < initial_energy

    def test_update_forcefield_reuses_context_for_vdw(self) -> None:
        molecule = make_noble_gas_pair(distance=3.0)
        initial_ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.01)])
        handle = self.engine.create_context(molecule, initial_ff)
        initial_energy = self.engine.energy(handle)

        updated_ff = ForceField(vdws=[VdwParam("He", radius=1.6, epsilon=0.02)])
        self.engine.update_forcefield(handle, updated_ff)
        assert self.engine.energy(handle) != pytest.approx(initial_energy)

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    def test_openmm_matches_tinker_for_mm3_bond_energy(self) -> None:
        tinker = TinkerEngine()
        forcefield = ForceField.from_tinker_prm(TINKER_PARAMS)
        molecule = Q2MMMolecule(
            symbols=["C", "H"],
            atom_types=["1", "5"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]]),
            name="CH-bond",
            bond_tolerance=1.5,
        )
        assert self.engine.energy(molecule, forcefield) == pytest.approx(tinker.energy(molecule), abs=1.0e-3)

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    def test_openmm_matches_tinker_for_mm3_vdw_energy(self) -> None:
        tinker = TinkerEngine()
        forcefield = ForceField.from_tinker_prm(TINKER_PARAMS)
        molecule = Q2MMMolecule(
            symbols=["F", "F"],
            atom_types=["11", "11"],
            geometry=np.array([[0.0, 0.0, 0.0], [3.50, 0.0, 0.0]]),
            name="F2-nonbonded",
            bond_tolerance=0.5,
        )
        assert self.engine.energy(molecule, forcefield) == pytest.approx(tinker.energy(molecule), abs=1.0e-3)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_seminario_pipeline_energy_is_finite(self) -> None:
        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        forcefield = estimate_force_constants(molecule)

        energy = self.engine.energy(molecule, forcefield)
        hessian = self.engine.hessian(molecule, forcefield)

        assert np.isfinite(energy)
        assert hessian.shape == (18, 18)
        np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-6)

    @pytest.mark.skipif(not TS_XYZ.exists() or not TS_HESS.exists(), reason="SN2 TS fixtures not found")
    def test_sn2_seminario_pipeline_has_imaginary_mode(self) -> None:
        molecule = Q2MMMolecule.from_xyz(TS_XYZ, bond_tolerance=1.5).with_hessian(np.load(TS_HESS))
        forcefield = estimate_force_constants(molecule)

        frequencies = self.engine.frequencies(molecule, forcefield)

        assert len(frequencies) == 18
        assert all(np.isfinite(freq) for freq in frequencies)
        assert min(frequencies) < -1.0
