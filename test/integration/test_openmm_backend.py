"""OpenMM-backend-specific tests.

Contract tests (energy, hessian, frequencies, minimize, gradients) are
in test_engine_contract.py and run for every registered backend.  This
file covers only behaviour unique to the OpenMM backend:

* MM3 formula known-value checks (cubic bond, sextic angle, buffered 14-7 vdW)
* Native-state reuse across parameter vectors in a prepared session
* Cross-backend parity with Tinker
* Seminario force-constant estimation pipeline
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    FrequencyRequest,
    HessianRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import optional_test_backend, param_vector, prepare_case

import importlib.util

import numpy as np
import pytest

pytestmark = [
    pytest.mark.openmm,
    pytest.mark.skipif(importlib.util.find_spec("openmm") is None, reason="openmm not installed"),
]

from test._shared import SN2_HESSIAN as TS_HESS, SN2_XYZ as TS_XYZ, make_diatomic, make_noble_gas_pair, make_water

from q2mm.backends.contracts import EnergyRequest, PreparationRequest
from q2mm.io.tinker import load_tinker_prm
from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam, FunctionalForm
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout
from q2mm.models.seminario import qfuerza_fresh

_tinker_backend = optional_test_backend("tinker")
HAS_TINKER = _tinker_backend is not None
TINKER_PARAMS = getattr(_tinker_backend, "_params_file", None)


class TestOpenMMBackend:
    def setup_method(self) -> None:
        self.backend = load_backend("openmm")

    @staticmethod
    def _load_sn2_ts_molecule() -> Molecule:
        molecule = load_xyz(TS_XYZ, charge=-1, bond_tolerance=1.5)
        return molecule.with_hessian(
            np.load(TS_HESS),
            HessianProvenance(
                units=HessianUnits.ATOMIC,
                source="test-fixture",
                path=str(TS_HESS),
            ),
        )

    def test_mm3_bond_energy_matches_reference_formula(self) -> None:
        molecule = make_diatomic(distance=0.84)
        forcefield = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )

        delta = 0.84 - 0.74
        expected_kcal = 71.9 * delta**2 * (1.0 - 2.55 * delta + (7.0 / 12.0) * 2.55**2 * delta**2)
        assert prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        ).energy == pytest.approx(expected_kcal)

    def test_mm3_angle_energy_matches_reference_formula(self) -> None:
        molecule = make_water(angle_deg=120.0)
        forcefield = ForceField(
            bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=71.9)],
            angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=36.0)],
            functional_form=FunctionalForm.MM3,
        )

        delta_deg = 120.0 - 104.5
        delta_rad = np.deg2rad(delta_deg)
        expected_kcal = (
            36.0
            * delta_rad**2
            * (1.0 - 0.014 * delta_deg + 5.6e-5 * delta_deg**2 - 7.0e-7 * delta_deg**3 + 9.0e-10 * delta_deg**4)
        )
        assert prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        ).energy == pytest.approx(expected_kcal)

    def test_mm3_vdw_energy_matches_reference_formula(self) -> None:
        molecule = make_noble_gas_pair(distance=3.5)
        forcefield = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.02)], functional_form=FunctionalForm.MM3)

        rv = 2.4
        expected = 0.02 * (-2.25 * (rv / 3.5) ** 6 + 184000.0 * np.exp(-12.0 * 3.5 / rv))
        assert prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        ).energy == pytest.approx(expected)

    def test_explicit_nonbonded_excluded_atom_type_has_zero_center(self) -> None:
        molecule = Molecule(
            symbols=("Ne", "He"),
            geometry=np.array([[0.0, 0.0, 0.0], [3.5, 0.0, 0.0]]),
            atom_types=("X", "He"),
            bonds=(),
            angles=(),
            torsions=(),
        )
        forcefield = ForceField(
            vdws=[VdwParam("X", radius=3.0, epsilon=1.0), VdwParam("He", radius=1.2, epsilon=0.02)],
            functional_form=FunctionalForm.MM3,
            nonbonded_excluded_atom_types=("X",),
        )

        result = prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        )
        assert result.energy == pytest.approx(0.0)

    def test_prepared_session_reuses_native_state(self) -> None:
        molecule = make_diatomic(distance=1.00)
        initial_ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=0.74, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )
        # A single prepared session reuses native state across parameter vectors.
        prepared = self.backend.prepare(PreparationRequest(case_id="0", molecule=molecule, force_field=initial_ff))
        layout = ParameterLayout.from_force_field(initial_ff)
        initial_energy = prepared.energy(EnergyRequest(parameters=layout.vector(initial_ff))).energy

        updated_ff = ForceField(
            bonds=[BondParam(("H", "H"), equilibrium=1.00, force_constant=71.9)], functional_form=FunctionalForm.MM3
        )
        updated_energy = prepared.energy(EnergyRequest(parameters=layout.vector(updated_ff))).energy
        assert updated_energy < initial_energy

    def test_prepared_session_reuses_native_state_for_vdw(self) -> None:
        molecule = make_noble_gas_pair(distance=3.0)
        initial_ff = ForceField(vdws=[VdwParam("He", radius=1.2, epsilon=0.01)], functional_form=FunctionalForm.MM3)
        prepared = self.backend.prepare(PreparationRequest(case_id="0", molecule=molecule, force_field=initial_ff))
        layout = ParameterLayout.from_force_field(initial_ff)
        initial_energy = prepared.energy(EnergyRequest(parameters=layout.vector(initial_ff))).energy

        updated_ff = ForceField(vdws=[VdwParam("He", radius=1.6, epsilon=0.02)], functional_form=FunctionalForm.MM3)
        updated_energy = prepared.energy(EnergyRequest(parameters=layout.vector(updated_ff))).energy
        assert updated_energy != pytest.approx(initial_energy)

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    @pytest.mark.cross_backend
    @pytest.mark.tinker
    def test_openmm_matches_tinker_for_mm3_bond_energy(self) -> None:
        assert _tinker_backend is not None
        forcefield = load_tinker_prm(TINKER_PARAMS)
        molecule = Molecule(
            symbols=["C", "H"],
            atom_types=["1", "5"],
            geometry=np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]]),
            name="CH-bond",
            bond_tolerance=1.5,
        )
        assert prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        ).energy == pytest.approx(
            prepare_case(_tinker_backend, molecule, forcefield)
            .energy(EnergyRequest(parameters=param_vector(forcefield)))
            .energy,
            abs=1.0e-3,
        )

    @pytest.mark.skipif(not HAS_TINKER or not TINKER_PARAMS, reason="Tinker not installed")
    @pytest.mark.cross_backend
    @pytest.mark.tinker
    def test_openmm_matches_tinker_for_mm3_vdw_energy(self) -> None:
        assert _tinker_backend is not None
        forcefield = load_tinker_prm(TINKER_PARAMS)
        molecule = Molecule(
            symbols=["F", "F"],
            atom_types=["11", "11"],
            geometry=np.array([[0.0, 0.0, 0.0], [3.50, 0.0, 0.0]]),
            name="F2-nonbonded",
            bond_tolerance=0.5,
        )
        assert prepare_case(self.backend, molecule, forcefield).energy(
            EnergyRequest(parameters=param_vector(forcefield))
        ).energy == pytest.approx(
            prepare_case(_tinker_backend, molecule, forcefield)
            .energy(EnergyRequest(parameters=param_vector(forcefield)))
            .energy,
            abs=1.0e-3,
        )

    def test_sn2_seminario_pipeline_energy_is_finite(self) -> None:
        molecule = self._load_sn2_ts_molecule()
        forcefield = qfuerza_fresh(molecule, functional_form=FunctionalForm.MM3)

        energy = (
            prepare_case(self.backend, molecule, forcefield)
            .energy(EnergyRequest(parameters=param_vector(forcefield)))
            .energy
        )
        hessian = (
            prepare_case(self.backend, molecule, forcefield)
            .hessian(HessianRequest(parameters=param_vector(forcefield)))
            .hessian
        )

        assert np.isfinite(energy)
        assert hessian.shape == (18, 18)
        np.testing.assert_allclose(hessian, hessian.T, atol=1.0e-6)

    def test_sn2_seminario_pipeline_has_imaginary_mode(self) -> None:
        molecule = self._load_sn2_ts_molecule()
        forcefield = qfuerza_fresh(molecule, functional_form=FunctionalForm.MM3)

        frequencies = [
            float(_f)
            for _f in prepare_case(self.backend, molecule, forcefield)
            .frequencies(FrequencyRequest(parameters=param_vector(forcefield)))
            .frequencies
        ]

        assert len(frequencies) == 18
        assert all(np.isfinite(freq) for freq in frequencies)
        assert min(frequencies) < -1.0
