"""Parametrized contract tests for all MM backends.

Uses the backend registry to discover available backends at collection
time and runs the same behavioral tests on every backend.  This
guarantees that all backends satisfy the :class:`Backend` ABC contract.

Engine-specific tests (MM3 formula known-values, cross-backend parity,
internal helpers, context/handle reuse) stay in their own backend files.
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    EnergyUnit,
    FrequencyRequest,
    GeometryResult,
    HessianJacobianRequest,
    HessianRequest,
    LengthUnit,
    MinimizationRequest,
    ParameterGradientRequest,
)
from test.backend_fixtures import backend_is_usable, load_test_backend, param_vector, prepare_case

import numpy as np
import pytest

from q2mm.backends.contracts import Backend, Capability
from q2mm.backends.registry import available_mm_backends, load_backend
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, TorsionParam, VdwParam
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout
from q2mm.io.xyz import load_xyz
from test._shared import (
    SN2_XYZ,
    make_diatomic,
    make_ethane,
    make_noble_gas_pair,
    make_water,
)


_AVAILABLE = sorted(name for name in set(available_mm_backends()) | {"tinker"} if backend_is_usable(name))

if not _AVAILABLE:
    pytest.skip("no MM backends available", allow_module_level=True)


def _functional_form(backend: Backend) -> FunctionalForm:
    """Pick a FunctionalForm supported by *backend*."""
    supported = backend.info.functional_forms
    if "harmonic" in supported:
        return FunctionalForm.HARMONIC
    if "mm3" in supported:
        return FunctionalForm.MM3
    for name in sorted(supported):
        if hasattr(FunctionalForm, name.upper()):
            return getattr(FunctionalForm, name.upper())
    raise RuntimeError(f"Engine {backend.info.name} reports no mappable functional forms: {supported!r}")


def _is_harmonic(backend: Backend) -> bool:
    return "harmonic" in backend.info.functional_forms


def _h2_ff(backend: Backend, bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        functional_form=_functional_form(backend),
        bonds=(BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0),),
    )


def _water_ff(
    backend: Backend,
    bond_k: float = 553.0,
    bond_r0: float = 0.96,
    angle_k: float = 49.9,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        functional_form=_functional_form(backend),
        bonds=(BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0),),
        angles=(AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq),),
    )


def _vdw_ff(backend: Backend) -> ForceField:
    return ForceField(
        functional_form=_functional_form(backend),
        vdws=(VdwParam(atom_type="He", element="He", radius=1.40, epsilon=0.02),),
    )


def _ethane_ff(backend: Backend, torsion_k: float = 0.15) -> ForceField:
    return ForceField(
        functional_form=_functional_form(backend),
        bonds=(
            BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54),
            BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
        ),
        angles=(
            AngleParam(elements=("H", "C", "C"), force_constant=37.5, equilibrium=109.5),
            AngleParam(elements=("H", "C", "H"), force_constant=33.0, equilibrium=109.5),
        ),
        torsions=(
            # Periodicity=1 so staggered ethane has nonzero torsion energy
            TorsionParam(elements=("H", "C", "C", "H"), periodicity=1, force_constant=torsion_k, phase=0.0),
        ),
    )


@pytest.fixture(scope="module", params=_AVAILABLE, ids=_AVAILABLE)
def engine_name(request: pytest.FixtureRequest) -> str:
    """Yield each available MM backend name in turn."""
    return request.param


@pytest.fixture(scope="module")
def backend(engine_name: str) -> Backend:
    """Instantiate the backend from the registry (reused across the module)."""
    return load_test_backend(engine_name)


@pytest.fixture
def h2(backend: Backend) -> tuple[Molecule, ForceField]:
    """H₂ at equilibrium with matching force field."""
    return make_diatomic(distance=0.74, bond_tolerance=2.0), _h2_ff(backend)


@pytest.fixture
def h2_displaced(backend: Backend) -> tuple[Molecule, ForceField]:
    """H₂ stretched 20 % beyond equilibrium."""
    return make_diatomic(distance=0.74 * 1.2, bond_tolerance=2.0), _h2_ff(backend)


@pytest.fixture
def water(backend: Backend) -> tuple[Molecule, ForceField]:
    """Water at equilibrium with matching force field."""
    return make_water(), _water_ff(backend)


@pytest.fixture
def water_bent(backend: Backend) -> tuple[Molecule, ForceField]:
    """Water with angle displaced from equilibrium."""
    return make_water(angle_deg=115.0, bond_length=1.02), _water_ff(backend)


@pytest.fixture
def noble_pair(backend: Backend) -> tuple[Molecule, ForceField]:
    """He₂ at moderate distance with vdW force field."""
    return make_noble_gas_pair(distance=3.0), _vdw_ff(backend)


@pytest.fixture
def ethane(backend: Backend) -> tuple[Molecule, ForceField]:
    """Staggered ethane with bond + angle + torsion FF."""
    return make_ethane(), _ethane_ff(backend)


class TestEngineMetadata:
    """Every backend must expose basic metadata correctly."""

    def test_name_returns_string(self, backend: Backend) -> None:
        assert isinstance(backend.info.name, str)
        assert len(backend.info.name) > 0

    def test_is_available(self, backend: Backend, engine_name: str) -> None:
        from q2mm.backends.registry import available_backends

        assert engine_name in available_backends()

    def test_supported_functional_forms_nonempty(self, backend: Backend) -> None:
        forms = backend.info.functional_forms
        assert isinstance(forms, frozenset)
        assert len(forms) > 0


class TestBondEnergy:
    """Bond energy must behave like a well-behaved potential."""

    def test_returns_float(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        assert isinstance(
            prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy, float
        )

    def test_near_zero_at_equilibrium(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        assert abs(prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy) < 1.0

    def test_increases_with_stretch(
        self,
        backend: Backend,
        h2: tuple[Molecule, ForceField],
        h2_displaced: tuple[Molecule, ForceField],
    ) -> None:
        mol_eq, ff = h2
        mol_disp, _ = h2_displaced
        assert (
            prepare_case(backend, mol_disp, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
            > prepare_case(backend, mol_eq, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )

    def test_increases_with_compression(self, backend: Backend) -> None:
        mol_eq = make_diatomic(distance=0.74, bond_tolerance=2.0)
        mol_comp = make_diatomic(distance=0.64, bond_tolerance=2.0)
        ff = _h2_ff(backend)
        assert (
            prepare_case(backend, mol_comp, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
            > prepare_case(backend, mol_eq, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )

    def test_symmetric_for_harmonic(self, backend: Backend) -> None:
        """Harmonic potential is symmetric about equilibrium."""
        if not _is_harmonic(backend):
            pytest.skip("symmetry test applies to harmonic form only")
        ff = _h2_ff(backend)
        e_up = (
            prepare_case(backend, make_diatomic(distance=0.80, bond_tolerance=2.0), ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        e_down = (
            prepare_case(backend, make_diatomic(distance=0.68, bond_tolerance=2.0), ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        assert abs(e_up - e_down) < 1e-6

    def test_energy_scales_with_force_constant(self, backend: Backend) -> None:
        """Doubling k should double the energy (harmonic only)."""
        if not _is_harmonic(backend):
            pytest.skip("scaling test applies to harmonic form only")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        e1 = (
            prepare_case(backend, mol, _h2_ff(backend, bond_k=359.7))
            .energy(EnergyRequest(parameters=param_vector(_h2_ff(backend, bond_k=359.7))))
            .energy
        )
        e2 = (
            prepare_case(backend, mol, _h2_ff(backend, bond_k=719.4))
            .energy(EnergyRequest(parameters=param_vector(_h2_ff(backend, bond_k=719.4))))
            .energy
        )
        assert abs(e2 / e1 - 2.0) < 1e-6


class TestAngleEnergy:
    """Angle energy on a water molecule."""

    def test_near_zero_at_equilibrium(self, backend: Backend, water: tuple[Molecule, ForceField]) -> None:
        mol, ff = water
        assert abs(prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy) < 1.0

    def test_increases_when_bent(
        self,
        backend: Backend,
        water: tuple[Molecule, ForceField],
        water_bent: tuple[Molecule, ForceField],
    ) -> None:
        mol_eq, ff = water
        mol_bent, _ = water_bent
        assert (
            prepare_case(backend, mol_bent, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
            > prepare_case(backend, mol_eq, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        )


class TestVdwEnergy:
    """Van der Waals energy on a noble-gas pair."""

    def test_nonzero_at_typical_distance(self, backend: Backend, noble_pair: tuple[Molecule, ForceField]) -> None:
        mol, ff = noble_pair
        assert prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy != 0.0

    def test_repulsive_at_close_range(self, backend: Backend) -> None:
        ff = _vdw_ff(backend)
        e_close = (
            prepare_case(backend, make_noble_gas_pair(distance=1.5), ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        e_far = (
            prepare_case(backend, make_noble_gas_pair(distance=4.0), ff)
            .energy(EnergyRequest(parameters=param_vector(ff)))
            .energy
        )
        assert e_close > e_far


class TestAnalyticalGradients:
    """Engines reporting supports_analytical_gradients() must match FD."""

    def _skip_if_unsupported(self, backend: Backend) -> None:
        if not backend.info.supports(Capability.PARAMETER_GRADIENT):
            pytest.skip("backend does not support analytical gradients")

    @staticmethod
    def _fd_engine(backend: Backend) -> Backend:
        """Return an backend suitable for double-precision FD reference.

        CUDA/OpenCL mixed precision loses too many digits for small FD
        perturbations.  When the backend is OpenMM on a GPU, return a
        CPU OpenMM backend so the FD baseline is computed in float64.
        """
        try:
            from q2mm.backends.mm.openmm import OpenMMBackend

            if isinstance(backend, OpenMMBackend) and ("CUDA" in backend.info.name or "OpenCL" in backend.info.name):
                return load_backend("openmm", platform_name="CPU")
        except ImportError:
            pass
        return backend

    def test_gradient_has_correct_length(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        self._skip_if_unsupported(backend)
        mol, ff = h2_displaced
        layout = ParameterLayout.from_force_field(ff)
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _energy, grad = _pg.energy, _pg.gradient
        assert isinstance(grad, np.ndarray)
        assert len(grad) == len(layout)

    def test_gradient_near_zero_at_equilibrium(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        self._skip_if_unsupported(backend)
        mol, ff = h2
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _energy, grad = _pg.energy, _pg.gradient
        np.testing.assert_allclose(grad, 0.0, atol=1e-8)

    def test_gradient_nonzero_away_from_equilibrium(
        self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]
    ) -> None:
        self._skip_if_unsupported(backend)
        mol, ff = h2_displaced
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _energy, grad = _pg.energy, _pg.gradient
        assert not np.all(grad == 0.0)

    def test_gradient_vs_finite_difference_bonds(self, backend: Backend) -> None:
        """Analytical gradient must match central finite differences."""
        self._skip_if_unsupported(backend)
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff(backend)
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _energy, grad_anal = _pg.energy, _pg.gradient

        fd_engine = self._fd_engine(backend)
        layout = ParameterLayout.from_force_field(ff)
        params = layout.vector(ff).copy()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            e_plus = (
                prepare_case(fd_engine, mol, layout.replace(ff, p_plus))
                .energy(EnergyRequest(parameters=param_vector(layout.replace(ff, p_plus))))
                .energy
            )
            e_minus = (
                prepare_case(fd_engine, mol, layout.replace(ff, p_minus))
                .energy(EnergyRequest(parameters=param_vector(layout.replace(ff, p_minus))))
                .energy
            )
            grad_fd[i] = (e_plus - e_minus) / (2 * h)

        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)

    def test_gradient_vs_finite_difference_water(self, backend: Backend) -> None:
        """Multi-parameter gradient (bonds + angles) vs FD."""
        self._skip_if_unsupported(backend)
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff(backend)
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        _energy, grad_anal = _pg.energy, _pg.gradient

        fd_engine = self._fd_engine(backend)
        layout = ParameterLayout.from_force_field(ff)
        params = layout.vector(ff).copy()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            e_plus = (
                prepare_case(fd_engine, mol, layout.replace(ff, p_plus))
                .energy(EnergyRequest(parameters=param_vector(layout.replace(ff, p_plus))))
                .energy
            )
            e_minus = (
                prepare_case(fd_engine, mol, layout.replace(ff, p_minus))
                .energy(EnergyRequest(parameters=param_vector(layout.replace(ff, p_minus))))
                .energy
            )
            grad_fd[i] = (e_plus - e_minus) / (2 * h)

        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)


class TestAnalyticalHessianGradients:
    """Engines reporting supports_analytical_hessian_gradients() must match FD."""

    def _skip_if_unsupported(self, backend: Backend) -> None:
        if not backend.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN):
            pytest.skip("backend does not support analytical Hessian gradients")

    def test_hessian_jacobian_shapes(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        """H is (3N, 3N) and dH_dp is (3N, 3N, n_params)."""
        self._skip_if_unsupported(backend)
        mol, ff = h2_displaced
        layout = ParameterLayout.from_force_field(ff)
        _hj = prepare_case(backend, mol, ff).hessian_parameter_jacobian(
            HessianJacobianRequest(parameters=param_vector(ff))
        )
        hess, dH_dp = _hj.hessian, _hj.jacobian
        n3 = 3 * len(mol.symbols)
        assert hess.shape == (n3, n3)
        assert dH_dp.shape == (n3, n3, len(layout))

    def test_hessian_jacobian_symmetric(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        """Hessian and each Jacobian slice must be symmetric."""
        self._skip_if_unsupported(backend)
        mol, ff = h2_displaced
        layout = ParameterLayout.from_force_field(ff)
        _hj = prepare_case(backend, mol, ff).hessian_parameter_jacobian(
            HessianJacobianRequest(parameters=param_vector(ff))
        )
        hess, dH_dp = _hj.hessian, _hj.jacobian
        np.testing.assert_allclose(hess, hess.T, atol=1e-8)
        for j in range(len(layout)):
            np.testing.assert_allclose(
                dH_dp[:, :, j],
                dH_dp[:, :, j].T,
                atol=1e-6,
                err_msg=f"dH_dp[:,:,{j}] not symmetric",
            )

    def test_hessian_jacobian_vs_fd_bonds(self, backend: Backend) -> None:
        """dH/dp must match central finite differences of bk.hessian(backend)."""
        self._skip_if_unsupported(backend)
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff(backend)
        _hj = prepare_case(backend, mol, ff).hessian_parameter_jacobian(
            HessianJacobianRequest(parameters=param_vector(ff))
        )
        _hess, dH_dp = _hj.hessian, _hj.jacobian

        layout = ParameterLayout.from_force_field(ff)
        params = layout.vector(ff).copy()
        h = 1e-5
        n3 = 3 * len(mol.symbols)
        dH_dp_fd = np.zeros((n3, n3, len(layout)))
        for i in range(len(layout)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            h_plus = (
                prepare_case(backend, mol, layout.replace(ff, p_plus))
                .hessian(HessianRequest(parameters=param_vector(layout.replace(ff, p_plus))))
                .hessian
            )
            h_minus = (
                prepare_case(backend, mol, layout.replace(ff, p_minus))
                .hessian(HessianRequest(parameters=param_vector(layout.replace(ff, p_minus))))
                .hessian
            )
            dH_dp_fd[:, :, i] = (h_plus - h_minus) / (2 * h)

        np.testing.assert_allclose(dH_dp, dH_dp_fd, atol=1e-4, rtol=1e-4)

    def test_hessian_jacobian_vs_fd_water(self, backend: Backend) -> None:
        """Multi-parameter Hessian Jacobian (bonds + angles) vs FD."""
        self._skip_if_unsupported(backend)
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff(backend)
        _hj = prepare_case(backend, mol, ff).hessian_parameter_jacobian(
            HessianJacobianRequest(parameters=param_vector(ff))
        )
        _hess, dH_dp = _hj.hessian, _hj.jacobian

        layout = ParameterLayout.from_force_field(ff)
        params = layout.vector(ff).copy()
        h = 1e-5
        n3 = 3 * len(mol.symbols)
        dH_dp_fd = np.zeros((n3, n3, len(layout)))
        for i in range(len(layout)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            h_plus = (
                prepare_case(backend, mol, layout.replace(ff, p_plus))
                .hessian(HessianRequest(parameters=param_vector(layout.replace(ff, p_plus))))
                .hessian
            )
            h_minus = (
                prepare_case(backend, mol, layout.replace(ff, p_minus))
                .hessian(HessianRequest(parameters=param_vector(layout.replace(ff, p_minus))))
                .hessian
            )
            dH_dp_fd[:, :, i] = (h_plus - h_minus) / (2 * h)

        np.testing.assert_allclose(dH_dp, dH_dp_fd, atol=1e-4, rtol=1e-4)


class TestHessian:
    """Hessian calculations must return a valid matrix."""

    def _skip_if_unsupported(self, backend: Backend, mol: Molecule, ff: ForceField) -> np.ndarray:
        try:
            return prepare_case(backend, mol, ff).hessian(HessianRequest(parameters=param_vector(ff))).hessian
        except NotImplementedError:
            pytest.skip("backend does not implement hessian()")

    def test_shape(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        hess = self._skip_if_unsupported(backend, mol, ff)
        n = 3 * len(mol.symbols)
        assert hess.shape == (n, n)

    def test_symmetric(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        hess = self._skip_if_unsupported(backend, mol, ff)
        np.testing.assert_allclose(hess, hess.T, atol=1e-6)

    def test_water_shape(self, backend: Backend, water: tuple[Molecule, ForceField]) -> None:
        mol, ff = water
        hess = self._skip_if_unsupported(backend, mol, ff)
        assert hess.shape == (9, 9)
        np.testing.assert_allclose(hess, hess.T, atol=1e-6)


class TestFrequencies:
    """Frequency calculations must return the correct number of modes."""

    def test_returns_list_or_array(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        assert isinstance(
            [
                float(_f)
                for _f in prepare_case(backend, mol, ff)
                .frequencies(FrequencyRequest(parameters=param_vector(ff)))
                .frequencies
            ],
            (list, np.ndarray),
        )

    def test_count_equals_3n(self, backend: Backend, h2: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2
        freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        assert len(freqs) == 3 * len(mol.symbols)

    def test_all_finite(self, backend: Backend, water: tuple[Molecule, ForceField]) -> None:
        mol, ff = water
        freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        assert all(np.isfinite(f) for f in freqs)

    def test_translation_rotation_modes_near_zero(self, backend: Backend, water: tuple[Molecule, ForceField]) -> None:
        """Nonlinear molecule should have ≥5 near-zero modes."""
        mol, ff = water
        freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        sorted_abs = sorted(abs(f) for f in freqs)
        for i in range(5):
            assert sorted_abs[i] < 50.0, f"Mode {i} should be near-zero, got {sorted_abs[i]} cm⁻¹"


class TestMinimize:
    """Minimization must return valid results and lower energy."""

    def test_returns_geometry_result(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2_displaced
        result = prepare_case(backend, mol, ff).minimize(MinimizationRequest(parameters=param_vector(ff)))
        assert isinstance(result, GeometryResult)
        assert result.energy_unit is EnergyUnit.KCAL_PER_MOL
        assert result.coordinate_unit is LengthUnit.ANGSTROM
        assert len(result.symbols) >= 2

    def test_lowers_energy(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2_displaced
        e_before = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e_after = prepare_case(backend, mol, ff).minimize(MinimizationRequest(parameters=param_vector(ff))).energy
        assert e_after <= e_before + 1e-6

    def test_converges_near_equilibrium(self, backend: Backend, h2_displaced: tuple[Molecule, ForceField]) -> None:
        mol, ff = h2_displaced
        _min = prepare_case(backend, mol, ff).minimize(MinimizationRequest(parameters=param_vector(ff)))
        atoms, coords = list(_min.symbols), np.asarray(_min.coordinates)
        assert len(atoms) == 2
        dist = np.linalg.norm(coords[0] - coords[1])
        assert abs(dist - 0.74) < 0.05

    @pytest.mark.integration
    def test_minimize_water(self, backend: Backend, water_bent: tuple[Molecule, ForceField]) -> None:
        mol, ff = water_bent
        e_before = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        _min = prepare_case(backend, mol, ff).minimize(MinimizationRequest(parameters=param_vector(ff)))
        e_after, atoms, coords = _min.energy, list(_min.symbols), np.asarray(_min.coordinates)
        assert e_after <= e_before + 1e-6
        assert len(atoms) == len(mol.symbols)
        assert coords.shape == (len(mol.symbols), 3)


class TestRealMolecule:
    """Every backend should handle a realistic molecule."""

    @pytest.fixture
    def sn2(self, backend: Backend) -> tuple[Molecule, ForceField]:
        mol = load_xyz(SN2_XYZ, bond_tolerance=1.5)
        ff = ForceField.create_for_molecule(mol, functional_form=_functional_form(backend))
        return mol, ff

    def test_energy_is_finite(self, backend: Backend, sn2: tuple[Molecule, ForceField]) -> None:
        mol, ff = sn2
        assert np.isfinite(prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy)

    def test_hessian_shape_and_symmetry(self, backend: Backend, sn2: tuple[Molecule, ForceField]) -> None:
        mol, ff = sn2
        try:
            hess = prepare_case(backend, mol, ff).hessian(HessianRequest(parameters=param_vector(ff))).hessian
        except NotImplementedError:
            pytest.skip("backend does not implement hessian()")
        n = 3 * len(mol.symbols)
        assert hess.shape == (n, n)
        np.testing.assert_allclose(hess, hess.T, atol=1e-4)

    def test_frequencies_finite(self, backend: Backend, sn2: tuple[Molecule, ForceField]) -> None:
        mol, ff = sn2
        freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        assert len(freqs) == 3 * len(mol.symbols)
        assert all(np.isfinite(f) for f in freqs)

    def test_gradient_finite(self, backend: Backend, sn2: tuple[Molecule, ForceField]) -> None:
        if not backend.info.supports(Capability.PARAMETER_GRADIENT):
            pytest.skip("backend does not support analytical gradients")
        mol, ff = sn2
        _pg = prepare_case(backend, mol, ff).parameter_gradient(ParameterGradientRequest(parameters=param_vector(ff)))
        energy, grad = _pg.energy, _pg.gradient
        assert np.isfinite(energy)
        assert np.all(np.isfinite(grad))


class TestTorsionEnergy:
    """Engines must compute torsion energy contributions."""

    @staticmethod
    def _skip_if_requires_torsion_params(backend: Backend) -> None:
        """Skip backends that require torsion params when torsion topology exists.

        Tinker auto-detects torsions from bond topology and errors if
        the PRM file lacks a matching torsion line.  Tests that create
        torsion-free FFs for molecules with torsion topology must skip.
        """
        if backend.info.name == "Tinker":
            pytest.skip(f"{backend.info.name} requires torsion params when torsion topology exists")

    def test_energy_finite_with_torsions(self, backend: Backend, ethane: tuple[Molecule, ForceField]) -> None:
        mol, ff = ethane
        e = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert np.isfinite(e)

    def test_energy_changes_with_torsion_k(self, backend: Backend) -> None:
        """Changing torsion k should change total energy."""
        mol = make_ethane()
        ff_low = _ethane_ff(backend, torsion_k=0.05)
        ff_high = _ethane_ff(backend, torsion_k=1.00)
        e_low = prepare_case(backend, mol, ff_low).energy(EnergyRequest(parameters=param_vector(ff_low))).energy
        e_high = prepare_case(backend, mol, ff_high).energy(EnergyRequest(parameters=param_vector(ff_high))).energy
        assert np.isfinite(e_low)
        assert np.isfinite(e_high)
        assert e_low != e_high

    def test_torsion_energy_nonzero_for_nonzero_k(self, backend: Backend) -> None:
        """With torsion k > 0, total energy should differ from torsion-free."""
        self._skip_if_requires_torsion_params(backend)
        mol = make_ethane()
        ff_with = _ethane_ff(backend, torsion_k=0.50)
        ff_without = ForceField(
            functional_form=_functional_form(backend),
            bonds=ff_with.bonds,
            angles=ff_with.angles,
            torsions=(),
        )
        e_with = prepare_case(backend, mol, ff_with).energy(EnergyRequest(parameters=param_vector(ff_with))).energy
        e_without = (
            prepare_case(backend, mol, ff_without).energy(EnergyRequest(parameters=param_vector(ff_without))).energy
        )
        assert abs(e_with - e_without) > 1e-6

    @pytest.mark.cross_backend
    @pytest.mark.openmm
    def test_torsion_energy_matches_openmm(self, backend: Backend, ethane: tuple[Molecule, ForceField]) -> None:
        """Cross-backend parity: torsion energy must agree with OpenMM reference."""
        if backend.info.name.startswith("OpenMM"):
            pytest.skip("Reference backend")
        if "openmm" not in available_mm_backends():
            pytest.skip("OpenMM not available for reference comparison")
        openmm = load_backend("openmm")
        mol, ff = ethane
        e_backend = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e_openmm = prepare_case(openmm, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert abs(e_backend - e_openmm) < 1e-4, (
            f"{backend.info.name} torsion energy {e_backend:.6f} != OpenMM {e_openmm:.6f} "
            f"(diff={abs(e_backend - e_openmm):.2e})"
        )
