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
from dataclasses import replace

from q2mm.backends.contracts import (
    EvaluationError,
    FrequencyRequest,
    HessianRequest,
    MinimizationRequest,
    ParameterGradientRequest,
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
from q2mm.models.molecule import Bond, Molecule
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.seminario import qfuerza_fresh

_tinker_backend = optional_test_backend("tinker")
HAS_TINKER = _tinker_backend is not None
TINKER_PARAMS = getattr(_tinker_backend, "_params_file", None)


class TestOpenMMBondBinding:
    @staticmethod
    def _case(row_order: str = "single-first", bond_order: str = "=") -> tuple[Molecule, ForceField, int]:
        molecule = Molecule(
            symbols=("C", "C"),
            geometry=np.array([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]]),
            atom_types=("C1", "C1"),
            bonds=(Bond(0, 1, ("C", "C"), 1.3, env_id="C1-C1", bond_order=bond_order),),
            angles=(),
            torsions=(),
        )
        single = BondParam(("C", "C"), equilibrium=1.5, force_constant=100.0, env_id="C1-C1", bond_order="-")
        double = BondParam(("C", "C"), equilibrium=1.2, force_constant=200.0, env_id="C1-C1", bond_order="=")
        rows = {
            "single-first": (single, double),
            "double-first": (double, single),
            "double-only": (double,),
        }[row_order]
        return molecule, ForceField(bonds=rows, functional_form=FunctionalForm.HARMONIC), rows.index(double)

    @pytest.mark.parametrize("row_order", ["single-first", "double-first", "double-only"])
    @pytest.mark.parametrize("bond_order", ["=", ""])
    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    def test_prepared_bond_binding_survives_scalar_updates(
        self, row_order: str, bond_order: str, form: FunctionalForm
    ) -> None:
        molecule, forcefield, selected = self._case(row_order, bond_order)
        forcefield = replace(forcefield, functional_form=form)
        backend = load_backend("openmm", platform_name="CPU")
        prepared = prepare_case(backend, molecule, forcefield)
        layout = ParameterLayout.from_force_field(forcefield)
        baseline = layout.vector(forcefield)
        updated = baseline.copy()
        updated[2 * selected : 2 * selected + 2] = [250.0, 1.6]
        for index in range(len(forcefield.bonds)):
            if index != selected:
                # This unused row becomes the closest length match after the update.
                updated[2 * index : 2 * index + 2] = [800.0, 1.31]

        for vector in (baseline, updated, baseline):
            k, r0 = vector[2 * selected : 2 * selected + 2]
            delta = 1.3 - r0
            expected = k * delta**2
            if form == FunctionalForm.MM3:
                expected *= 1.0 - 2.55 * delta + (7.0 / 12.0) * 2.55**2 * delta**2
            energy = prepared.energy(EnergyRequest(parameters=vector)).energy
            derivative = prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
            assert energy == pytest.approx(expected)
            assert derivative.energy == pytest.approx(energy)

            finite_difference = np.zeros(len(vector))
            step = 1e-5
            for index in range(len(vector)):
                plus, minus = vector.copy(), vector.copy()
                plus[index] += step
                minus[index] -= step
                finite_difference[index] = (
                    prepared.energy(EnergyRequest(parameters=plus)).energy
                    - prepared.energy(EnergyRequest(parameters=minus)).energy
                ) / (2 * step)
            np.testing.assert_allclose(derivative.gradient, finite_difference, rtol=1e-4, atol=1e-4)
            unused = [index for index in range(len(vector)) if index // 2 != selected]
            np.testing.assert_array_equal(derivative.gradient[unused], 0.0)

            control_ff = replace(
                forcefield,
                bonds=(replace(forcefield.bonds[selected], force_constant=k, equilibrium=r0),),
            )
            control = prepare_case(backend, molecule, control_ff)
            expected_hessian = control.hessian(HessianRequest(parameters=param_vector(control_ff))).hessian
            hessian = prepared.hessian(HessianRequest(parameters=vector)).hessian
            np.testing.assert_allclose(hessian, expected_hessian, atol=1e-10)
            assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(energy)

        minimized = prepared.minimize(MinimizationRequest(parameters=updated, tolerance=1e-8, max_iterations=100))
        distance = np.linalg.norm(minimized.coordinates[1] - minimized.coordinates[0])
        assert distance == pytest.approx(1.6, abs=1e-6)
        assert minimized.energy == pytest.approx(0.0, abs=1e-9)
        assert prepared.energy(EnergyRequest(parameters=baseline)).energy == pytest.approx(
            2.0 if form == FunctionalForm.HARMONIC else 2.0 * (1 - 0.255 + (7 / 12) * 0.255**2)
        )

    def test_frozen_bond_slots_and_vdw_gradient_keep_prepared_binding(self) -> None:
        molecule, forcefield, selected = self._case(bond_order="")
        forcefield = replace(forcefield, vdws=(VdwParam("C1", radius=1.5, epsilon=0.1),))
        backend = load_backend("openmm", platform_name="CPU")
        prepared = prepare_case(backend, molecule, forcefield)
        layout = ParameterLayout.from_force_field(forcefield)
        baseline = layout.vector(forcefield)
        space = ActiveParameterSpace(layout=layout, baseline=baseline, active_indices=(2 * selected, 2 * selected + 1))
        vector = space.expand(np.array([250.0, 1.6]))
        np.testing.assert_array_equal(vector[:2], baseline[:2])
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
        assert result.energy == pytest.approx(22.5)
        assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(result.energy)
        np.testing.assert_allclose(result.gradient, [0.0, 0.0, 0.09, 150.0, 0.0, 0.0], atol=1e-8)

    @pytest.mark.parametrize(
        "change", ["reorder", "remove", "add", "elements", "env_id", "ff_row", "bond_order", "context"]
    )
    def test_native_bond_binding_rejects_structural_reuse(self, change: str) -> None:
        from q2mm.backends.mm.openmm import OpenMMBackend

        molecule, forcefield, selected = self._case()
        backend = OpenMMBackend(platform_name="CPU")
        state = backend._build_state(molecule, forcefield)
        if change == "reorder":
            changed = replace(forcefield, bonds=forcefield.bonds[::-1])
        elif change == "remove":
            changed = replace(forcefield, bonds=forcefield.bonds[:1])
        elif change == "add":
            changed = replace(forcefield, bonds=(*forcefield.bonds, BondParam(("H", "H"), 0.74, 100.0)))
        else:
            value = {
                "elements": ("C", "N"),
                "env_id": "C2-C2",
                "ff_row": 42,
                "bond_order": "*",
                "context": "O200 0000",
            }[change]
            rows = list(forcefield.bonds)
            rows[selected] = replace(rows[selected], **{change: value})
            changed = replace(forcefield, bonds=tuple(rows))
        with pytest.raises(ValueError, match="bond parameter structure"):
            backend._update_params(state, changed)
        with pytest.raises(ValueError, match="bond parameter structure"):
            backend._build_diff_state(state, changed)
        with pytest.raises(ValueError, match="bond parameter structure"):
            backend._build_state(molecule, changed, bond_source=state)
        assert backend._evaluate_energy(state, forcefield) == pytest.approx(2.0)

    def test_repeated_bond_object_cannot_supply_a_unique_source_index(self) -> None:
        from q2mm.backends.contracts import PreparationError

        molecule, forcefield, selected = self._case()
        parameter = replace(forcefield.bonds[selected], ff_row=42)
        forcefield = replace(forcefield, bonds=(parameter, parameter))
        molecule = replace(molecule, bonds=(replace(molecule.bonds[0], ff_row=42),))
        backend = load_backend("openmm", platform_name="CPU")
        with pytest.raises(PreparationError, match="unique source index"):
            prepare_case(backend, molecule, forcefield)

    def test_native_bond_binding_rejects_a_different_molecule(self) -> None:
        from q2mm.backends.mm.openmm import OpenMMBackend

        molecule, forcefield, _selected = self._case()
        backend = OpenMMBackend(platform_name="CPU")
        state = backend._build_state(molecule, forcefield)
        with pytest.raises(ValueError, match="different molecule"):
            backend._build_state(make_diatomic(), forcefield, bond_source=state)


class TestOpenMMNonbondedAdapters:
    @staticmethod
    def _pair_control(
        form: FunctionalForm, radii: tuple[float, float], epsilons: tuple[float, float]
    ) -> tuple[float, np.ndarray]:
        """Independent derivatives of the existing native pair potential at 4 A."""
        radius = sum(radii)
        epsilon = np.sqrt(epsilons[0] * epsilons[1])
        sixth = (radius / 4.0) ** 6
        if form == FunctionalForm.HARMONIC:
            shape = sixth**2 - 2 * sixth
            radius_derivative = epsilon * 12 * (sixth**2 - sixth) / radius
        else:
            exponential = 184000.0 * np.exp(-12.0 * 4.0 / radius)
            shape = -2.25 * sixth + exponential
            radius_derivative = epsilon * (-6 * 2.25 * sixth / radius + exponential * 48.0 / radius**2)
        energy = epsilon * shape
        return energy, np.array(
            [radius_derivative, energy / (2 * epsilons[0]), radius_derivative, energy / (2 * epsilons[1])]
        )

    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    @pytest.mark.parametrize("epsilon", [1e-8, 5e-5, 0.02, 0.4])
    def test_positive_vdw_derivatives_stay_in_domain(self, form: FunctionalForm, epsilon: float) -> None:
        molecule = Molecule(symbols=("He", "Ne"), geometry=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), bonds=())
        ff = ForceField(
            functional_form=form,
            vdws=(VdwParam("He", 1.0, epsilon), VdwParam("Ne", 2.0, 0.4)),
        )
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        vector = param_vector(ff)
        expected_energy, expected_gradient = self._pair_control(form, (1.0, 2.0), (epsilon, 0.4))
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
        assert result.energy == pytest.approx(expected_energy, rel=2e-6)
        np.testing.assert_allclose(result.gradient, expected_gradient, rtol=2e-3, atol=1e-9)
        assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(result.energy, rel=2e-6)
        if form == FunctionalForm.HARMONIC and epsilon == 5e-5:
            assert result.energy == pytest.approx(-0.001450226620822632, rel=2e-6)

    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    @pytest.mark.parametrize("radius", [0.0, 5e-5, 1e-4])
    def test_small_and_zero_radius_derivatives(self, form: FunctionalForm, radius: float) -> None:
        molecule = Molecule(symbols=("He", "Ne"), geometry=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), bonds=())
        ff = ForceField(functional_form=form, vdws=(VdwParam("He", radius, 0.1), VdwParam("Ne", 2.0, 0.4)))
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        expected_energy, expected_gradient = self._pair_control(form, (radius, 2.0), (0.1, 0.4))
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        assert result.energy == pytest.approx(expected_energy, rel=2e-6)
        # The forward stencil amplifies the CPU nonbonded kernel's float32 rounding.
        np.testing.assert_allclose(result.gradient, expected_gradient, rtol=4e-3, atol=1e-9)

    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    @pytest.mark.parametrize("partner_epsilon", [0.0, 0.4])
    def test_zero_epsilon_mixed_rows_are_explicitly_singular(
        self, form: FunctionalForm, partner_epsilon: float
    ) -> None:
        molecule = Molecule(symbols=("He", "Ne"), geometry=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]), bonds=())
        ff = ForceField(functional_form=form, vdws=(VdwParam("He", 1.0, 0.0), VdwParam("Ne", 2.0, partner_epsilon)))
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        vector = param_vector(ff)
        assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(0.0)
        with pytest.raises(EvaluationError, match="epsilon derivative is singular at zero"):
            prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
        updated = vector.copy()
        updated[1] = 5e-5
        updated[3] = 0.4
        assert np.all(np.isfinite(prepared.parameter_gradient(ParameterGradientRequest(parameters=updated)).gradient))
        assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(0.0)

    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    def test_same_row_zero_epsilon_finite_one_sided_derivative(self, form: FunctionalForm) -> None:
        molecule = Molecule(
            symbols=("He", "He"),
            geometry=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            bonds=(),
        )
        ff = ForceField(functional_form=form, vdws=(VdwParam("He", 1.0, 0.0),))
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        expected = np.array([0.0, self._pair_control(form, (1.0, 1.0), (1.0, 1.0))[0]])
        assert result.energy == pytest.approx(0.0)
        np.testing.assert_allclose(result.gradient, expected, rtol=2e-5, atol=1e-10)

    @pytest.mark.parametrize("form", [FunctionalForm.HARMONIC, FunctionalForm.MM3])
    def test_zero_or_unused_vdw_rows_without_partners(self, form: FunctionalForm) -> None:
        molecule = Molecule(
            symbols=("He", "Ne"),
            geometry=np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
            bonds=(Bond(0, 1, ("He", "Ne"), 4.0),),
            angles=(),
        )
        ff = ForceField(
            functional_form=form,
            vdws=(VdwParam("He", 1.0, 0.0), VdwParam("Ne", 2.0, 0.4), VdwParam("Ar", 0.0, 0.0)),
        )
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        result = prepared.parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        assert result.energy == pytest.approx(0.0)
        np.testing.assert_array_equal(result.gradient, 0.0)

    @pytest.mark.parametrize("excluded_count", [1, 2, 3])
    @pytest.mark.parametrize("active_count", [1, 2])
    @pytest.mark.parametrize("coincident", [False, True])
    def test_absent_mm3_centers_never_enter_pair_expression(
        self, excluded_count: int, active_count: int, coincident: bool
    ) -> None:
        active_coords = [[4.0 * i, 0.0, 0.0] for i in range(active_count)]
        void_coords = [[0.0 if coincident else 3.0 * (i + 1), 0.0, 0.0] for i in range(excluded_count)]
        molecule = Molecule(
            symbols=("C",) * active_count + ("He",) * excluded_count,
            atom_types=("active",) * active_count + ("void",) * excluded_count,
            geometry=np.array(active_coords + void_coords),
            bonds=(),
            angles=(),
            torsions=(),
        )
        ff = ForceField(
            functional_form=FunctionalForm.MM3,
            vdws=(
                VdwParam("C", 5.0, 2.0),
                VdwParam("active", 1.0, 0.1, element="C"),
                VdwParam("void", 3.0, 1.0, element="He"),
            ),
            nonbonded_excluded_atom_types=("VOID",),
        )
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        baseline = param_vector(ff)
        updated = baseline.copy()
        updated[2:4] = [1.2, 0.3]
        for vector in (baseline, updated, baseline):
            radius, epsilon = vector[2:4]
            expected_energy = 0.0
            expected_gradient = np.zeros(len(vector))
            if active_count == 2:
                expected_energy, pair_gradient = self._pair_control(
                    FunctionalForm.MM3, (radius, radius), (epsilon, epsilon)
                )
                expected_gradient[2:4] = pair_gradient[:2] * 2
            assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(
                expected_energy, rel=2e-6, abs=1e-12
            )
            result = prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
            assert result.energy == pytest.approx(expected_energy, rel=2e-6, abs=1e-12)
            np.testing.assert_allclose(result.gradient, expected_gradient, rtol=2e-3, atol=1e-9)
        hessian = prepared.hessian(HessianRequest(parameters=baseline)).hessian
        assert np.all(np.isfinite(hessian))
        np.testing.assert_array_equal(hessian[3 * active_count :, :], 0.0)
        np.testing.assert_array_equal(hessian[:, 3 * active_count :], 0.0)

    @pytest.mark.parametrize(
        "n,bonds,interactions",
        [
            (3, [(0, 1), (1, 2)], []),
            (4, [(0, 1), (1, 2), (2, 3)], [(0, 3, 0.5)]),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4)], [(0, 3, 0.5), (1, 4, 0.5), (0, 4, 1.0)]),
            (3, [(0, 1), (1, 2), (2, 0)], []),
            (4, [(0, 1), (1, 2), (2, 3), (3, 0)], []),
            (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)], []),
            (6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)], [(0, 3, 0.5), (1, 4, 0.5), (2, 5, 0.5)]),
            (5, [(0, 1), (1, 2), (1, 3), (3, 4)], [(0, 4, 0.5), (2, 4, 0.5)]),
        ],
    )
    def test_harmonic_bond_graph_exclusions_and_scaled_updates(
        self, n: int, bonds: list[tuple[int, int]], interactions: list[tuple[int, int, float]]
    ) -> None:
        coords = np.array([[float(i), 0.5 * (i % 2), 0.25 * (i % 3)] for i in range(n)])
        if n == 3 and len(bonds) == 2:
            coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.0, 1.4, 0.0]])
        molecule = Molecule(
            symbols=("C",) * n,
            geometry=coords,
            bonds=tuple(Bond(i, j, ("C", "C"), float(np.linalg.norm(coords[i] - coords[j]))) for i, j in bonds),
            angles=(),
            torsions=(),
        )
        ff = ForceField(functional_form=FunctionalForm.HARMONIC, vdws=(VdwParam("C", 1.0, 0.0),))
        prepared = prepare_case(load_backend("openmm", platform_name="CPU"), molecule, ff)
        for radius, epsilon in [(1.0, 0.1), (1.0, 0.0), (1.2, 0.2), (1.0, 0.1)]:
            vector = np.array([radius, epsilon])
            expected, radius_derivative, epsilon_derivative = 0.0, 0.0, 0.0
            for i, j, scale in interactions:
                sixth = (2 * radius / np.linalg.norm(coords[i] - coords[j])) ** 6
                expected += scale * epsilon * (sixth**2 - 2 * sixth)
                radius_derivative += scale * epsilon * 12 * (sixth**2 - sixth) / radius
                epsilon_derivative += scale * (sixth**2 - 2 * sixth)
            assert prepared.energy(EnergyRequest(parameters=vector)).energy == pytest.approx(expected, abs=1e-8)
            result = prepared.parameter_gradient(ParameterGradientRequest(parameters=vector))
            assert result.energy == pytest.approx(expected, abs=1e-8)
            np.testing.assert_allclose(result.gradient, [radius_derivative, epsilon_derivative], rtol=2e-3, atol=1e-8)


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
