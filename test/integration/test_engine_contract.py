"""Parametrized contract tests for all MM engine backends.

Uses the engine registry to discover available backends at collection
time and runs the same behavioral tests on every engine.  This
guarantees that all engines satisfy the :class:`MMEngine` ABC contract.

Engine-specific tests (MM3 formula known-values, cross-backend parity,
internal helpers, context/handle reuse) stay in their own backend files.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.backends.base import MMEngine
from q2mm.backends.registry import available_mm_engines, get_mm_engine
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, TorsionParam, VdwParam
from q2mm.models.molecule import Q2MMMolecule
from test._shared import (
    SN2_HESSIAN,
    SN2_XYZ,
    make_diatomic,
    make_ethane,
    make_noble_gas_pair,
    make_water,
)


_AVAILABLE = available_mm_engines()

if not _AVAILABLE:
    pytest.skip("no MM engines available", allow_module_level=True)

_SN2_DATA_AVAILABLE = SN2_XYZ.exists() and SN2_HESSIAN.exists()


def _functional_form(engine: MMEngine) -> FunctionalForm:
    """Pick a FunctionalForm supported by *engine*."""
    supported = engine.supported_functional_forms()
    if "harmonic" in supported:
        return FunctionalForm.HARMONIC
    if "mm3" in supported:
        return FunctionalForm.MM3
    for name in sorted(supported):
        if hasattr(FunctionalForm, name.upper()):
            return getattr(FunctionalForm, name.upper())
    raise RuntimeError(f"Engine {engine.name} reports no mappable functional forms: {supported!r}")


def _is_harmonic(engine: MMEngine) -> bool:
    return "harmonic" in engine.supported_functional_forms()


def _h2_ff(engine: MMEngine, bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        functional_form=_functional_form(engine),
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    engine: MMEngine,
    bond_k: float = 553.0,
    bond_r0: float = 0.96,
    angle_k: float = 49.9,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        functional_form=_functional_form(engine),
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _vdw_ff(engine: MMEngine) -> ForceField:
    return ForceField(
        functional_form=_functional_form(engine),
        vdws=[VdwParam(atom_type="He", element="He", radius=1.40, epsilon=0.02)],
    )


def _ethane_ff(engine: MMEngine, torsion_k: float = 0.15) -> ForceField:
    return ForceField(
        functional_form=_functional_form(engine),
        bonds=[
            BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54),
            BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
        ],
        angles=[
            AngleParam(elements=("H", "C", "C"), force_constant=37.5, equilibrium=109.5),
            AngleParam(elements=("H", "C", "H"), force_constant=33.0, equilibrium=109.5),
        ],
        torsions=[
            # Periodicity=1 so staggered ethane has nonzero torsion energy
            TorsionParam(elements=("H", "C", "C", "H"), periodicity=1, force_constant=torsion_k, phase=0.0),
        ],
    )


@pytest.fixture(scope="module", params=_AVAILABLE, ids=_AVAILABLE)
def engine_name(request: pytest.FixtureRequest) -> str:
    """Yield each available MM engine name in turn."""
    return request.param


@pytest.fixture(scope="module")
def engine(engine_name: str) -> MMEngine:
    """Instantiate the engine from the registry (reused across the module)."""
    return get_mm_engine(engine_name)


@pytest.fixture
def h2(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """H₂ at equilibrium with matching force field."""
    return make_diatomic(distance=0.74, bond_tolerance=2.0), _h2_ff(engine)


@pytest.fixture
def h2_displaced(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """H₂ stretched 20 % beyond equilibrium."""
    return make_diatomic(distance=0.74 * 1.2, bond_tolerance=2.0), _h2_ff(engine)


@pytest.fixture
def water(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """Water at equilibrium with matching force field."""
    return make_water(), _water_ff(engine)


@pytest.fixture
def water_bent(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """Water with angle displaced from equilibrium."""
    return make_water(angle_deg=115.0, bond_length=1.02), _water_ff(engine)


@pytest.fixture
def noble_pair(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """He₂ at moderate distance with vdW force field."""
    return make_noble_gas_pair(distance=3.0), _vdw_ff(engine)


@pytest.fixture
def ethane(engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
    """Staggered ethane with bond + angle + torsion FF."""
    return make_ethane(), _ethane_ff(engine)


class TestEngineMetadata:
    """Every engine must expose basic metadata correctly."""

    def test_name_returns_string(self, engine: MMEngine) -> None:
        assert isinstance(engine.name, str)
        assert len(engine.name) > 0

    def test_is_available(self, engine: MMEngine) -> None:
        assert engine.is_available() is True

    def test_supported_functional_forms_nonempty(self, engine: MMEngine) -> None:
        forms = engine.supported_functional_forms()
        assert isinstance(forms, frozenset)
        assert len(forms) > 0


class TestBondEnergy:
    """Bond energy must behave like a well-behaved potential."""

    def test_returns_float(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        assert isinstance(engine.energy(mol, ff), float)

    def test_near_zero_at_equilibrium(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        assert abs(engine.energy(mol, ff)) < 1.0

    def test_increases_with_stretch(
        self,
        engine: MMEngine,
        h2: tuple[Q2MMMolecule, ForceField],
        h2_displaced: tuple[Q2MMMolecule, ForceField],
    ) -> None:
        mol_eq, ff = h2
        mol_disp, _ = h2_displaced
        assert engine.energy(mol_disp, ff) > engine.energy(mol_eq, ff)

    def test_increases_with_compression(self, engine: MMEngine) -> None:
        mol_eq = make_diatomic(distance=0.74, bond_tolerance=2.0)
        mol_comp = make_diatomic(distance=0.64, bond_tolerance=2.0)
        ff = _h2_ff(engine)
        assert engine.energy(mol_comp, ff) > engine.energy(mol_eq, ff)

    def test_symmetric_for_harmonic(self, engine: MMEngine) -> None:
        """Harmonic potential is symmetric about equilibrium."""
        if not _is_harmonic(engine):
            pytest.skip("symmetry test applies to harmonic form only")
        ff = _h2_ff(engine)
        e_up = engine.energy(make_diatomic(distance=0.80, bond_tolerance=2.0), ff)
        e_down = engine.energy(make_diatomic(distance=0.68, bond_tolerance=2.0), ff)
        assert abs(e_up - e_down) < 1e-6

    def test_energy_scales_with_force_constant(self, engine: MMEngine) -> None:
        """Doubling k should double the energy (harmonic only)."""
        if not _is_harmonic(engine):
            pytest.skip("scaling test applies to harmonic form only")
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        e1 = engine.energy(mol, _h2_ff(engine, bond_k=359.7))
        e2 = engine.energy(mol, _h2_ff(engine, bond_k=719.4))
        assert abs(e2 / e1 - 2.0) < 1e-6


class TestAngleEnergy:
    """Angle energy on a water molecule."""

    def test_near_zero_at_equilibrium(self, engine: MMEngine, water: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = water
        assert abs(engine.energy(mol, ff)) < 1.0

    def test_increases_when_bent(
        self,
        engine: MMEngine,
        water: tuple[Q2MMMolecule, ForceField],
        water_bent: tuple[Q2MMMolecule, ForceField],
    ) -> None:
        mol_eq, ff = water
        mol_bent, _ = water_bent
        assert engine.energy(mol_bent, ff) > engine.energy(mol_eq, ff)


class TestVdwEnergy:
    """Van der Waals energy on a noble-gas pair."""

    def test_nonzero_at_typical_distance(self, engine: MMEngine, noble_pair: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = noble_pair
        assert engine.energy(mol, ff) != 0.0

    def test_repulsive_at_close_range(self, engine: MMEngine) -> None:
        ff = _vdw_ff(engine)
        e_close = engine.energy(make_noble_gas_pair(distance=1.5), ff)
        e_far = engine.energy(make_noble_gas_pair(distance=4.0), ff)
        assert e_close > e_far


class TestAnalyticalGradients:
    """Engines reporting supports_analytical_gradients() must match FD."""

    def _skip_if_unsupported(self, engine: MMEngine) -> None:
        if not engine.supports_analytical_gradients():
            pytest.skip("engine does not support analytical gradients")

    def test_gradient_has_correct_length(self, engine: MMEngine, h2_displaced: tuple[Q2MMMolecule, ForceField]) -> None:
        self._skip_if_unsupported(engine)
        mol, ff = h2_displaced
        _energy, grad = engine.energy_and_param_grad(mol, ff)
        assert isinstance(grad, np.ndarray)
        assert len(grad) == ff.n_params

    def test_gradient_near_zero_at_equilibrium(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        self._skip_if_unsupported(engine)
        mol, ff = h2
        _energy, grad = engine.energy_and_param_grad(mol, ff)
        np.testing.assert_allclose(grad, 0.0, atol=1e-8)

    def test_gradient_nonzero_away_from_equilibrium(
        self, engine: MMEngine, h2_displaced: tuple[Q2MMMolecule, ForceField]
    ) -> None:
        self._skip_if_unsupported(engine)
        mol, ff = h2_displaced
        _energy, grad = engine.energy_and_param_grad(mol, ff)
        assert not np.all(grad == 0.0)

    def test_gradient_vs_finite_difference_bonds(self, engine: MMEngine) -> None:
        """Analytical gradient must match central finite differences."""
        self._skip_if_unsupported(engine)
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff(engine)
        _energy, grad_anal = engine.energy_and_param_grad(mol, ff)

        params = ff.get_param_vector().copy()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            ff.set_param_vector(p_plus)
            e_plus = engine.energy(mol, ff)
            ff.set_param_vector(p_minus)
            e_minus = engine.energy(mol, ff)
            grad_fd[i] = (e_plus - e_minus) / (2 * h)
        ff.set_param_vector(params)

        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)

    def test_gradient_vs_finite_difference_water(self, engine: MMEngine) -> None:
        """Multi-parameter gradient (bonds + angles) vs FD."""
        self._skip_if_unsupported(engine)
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff(engine)
        _energy, grad_anal = engine.energy_and_param_grad(mol, ff)

        params = ff.get_param_vector().copy()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            ff.set_param_vector(p_plus)
            e_plus = engine.energy(mol, ff)
            ff.set_param_vector(p_minus)
            e_minus = engine.energy(mol, ff)
            grad_fd[i] = (e_plus - e_minus) / (2 * h)
        ff.set_param_vector(params)

        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)


class TestHessian:
    """Hessian calculations must return a valid matrix."""

    def _skip_if_unsupported(self, engine: MMEngine, mol: Q2MMMolecule, ff: ForceField) -> np.ndarray:
        try:
            return engine.hessian(mol, ff)
        except NotImplementedError:
            pytest.skip("engine does not implement hessian()")

    def test_shape(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        hess = self._skip_if_unsupported(engine, mol, ff)
        n = 3 * len(mol.symbols)
        assert hess.shape == (n, n)

    def test_symmetric(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        hess = self._skip_if_unsupported(engine, mol, ff)
        np.testing.assert_allclose(hess, hess.T, atol=1e-6)

    def test_water_shape(self, engine: MMEngine, water: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = water
        hess = self._skip_if_unsupported(engine, mol, ff)
        assert hess.shape == (9, 9)
        np.testing.assert_allclose(hess, hess.T, atol=1e-6)


class TestFrequencies:
    """Frequency calculations must return the correct number of modes."""

    def test_returns_list_or_array(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        assert isinstance(engine.frequencies(mol, ff), (list, np.ndarray))

    def test_count_equals_3n(self, engine: MMEngine, h2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2
        freqs = engine.frequencies(mol, ff)
        assert len(freqs) == 3 * len(mol.symbols)

    def test_all_finite(self, engine: MMEngine, water: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = water
        freqs = engine.frequencies(mol, ff)
        assert all(np.isfinite(f) for f in freqs)

    def test_translation_rotation_modes_near_zero(
        self, engine: MMEngine, water: tuple[Q2MMMolecule, ForceField]
    ) -> None:
        """Nonlinear molecule should have ≥5 near-zero modes."""
        mol, ff = water
        freqs = engine.frequencies(mol, ff)
        sorted_abs = sorted(abs(f) for f in freqs)
        for i in range(5):
            assert sorted_abs[i] < 50.0, f"Mode {i} should be near-zero, got {sorted_abs[i]} cm⁻¹"


class TestMinimize:
    """Minimization must return valid results and lower energy."""

    def test_returns_tuple(self, engine: MMEngine, h2_displaced: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2_displaced
        result = engine.minimize(mol, ff)
        assert isinstance(result, tuple)
        assert len(result) >= 3

    def test_lowers_energy(self, engine: MMEngine, h2_displaced: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2_displaced
        e_before = engine.energy(mol, ff)
        e_after, _atoms, _coords, *_ = engine.minimize(mol, ff)
        assert e_after <= e_before + 1e-6

    def test_converges_near_equilibrium(self, engine: MMEngine, h2_displaced: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = h2_displaced
        _energy, atoms, coords, *_ = engine.minimize(mol, ff)
        assert len(atoms) == 2
        dist = np.linalg.norm(coords[0] - coords[1])
        assert abs(dist - 0.74) < 0.05

    @pytest.mark.medium
    def test_minimize_water(self, engine: MMEngine, water_bent: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = water_bent
        e_before = engine.energy(mol, ff)
        e_after, atoms, coords, *_ = engine.minimize(mol, ff)
        assert e_after <= e_before + 1e-6
        assert len(atoms) == len(mol.symbols)
        assert coords.shape == (len(mol.symbols), 3)


@pytest.mark.skipif(not _SN2_DATA_AVAILABLE, reason="SN2 fixtures not found")
class TestRealMolecule:
    """Every engine should handle a realistic molecule."""

    @pytest.fixture
    def sn2(self, engine: MMEngine) -> tuple[Q2MMMolecule, ForceField]:
        mol = Q2MMMolecule.from_xyz(SN2_XYZ, bond_tolerance=1.5)
        ff = ForceField.create_for_molecule(mol)
        ff.functional_form = _functional_form(engine)
        return mol, ff

    def test_energy_is_finite(self, engine: MMEngine, sn2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = sn2
        assert np.isfinite(engine.energy(mol, ff))

    def test_hessian_shape_and_symmetry(self, engine: MMEngine, sn2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = sn2
        try:
            hess = engine.hessian(mol, ff)
        except NotImplementedError:
            pytest.skip("engine does not implement hessian()")
        n = 3 * len(mol.symbols)
        assert hess.shape == (n, n)
        np.testing.assert_allclose(hess, hess.T, atol=1e-4)

    def test_frequencies_finite(self, engine: MMEngine, sn2: tuple[Q2MMMolecule, ForceField]) -> None:
        mol, ff = sn2
        freqs = engine.frequencies(mol, ff)
        assert len(freqs) == 3 * len(mol.symbols)
        assert all(np.isfinite(f) for f in freqs)

    def test_gradient_finite(self, engine: MMEngine, sn2: tuple[Q2MMMolecule, ForceField]) -> None:
        if not engine.supports_analytical_gradients():
            pytest.skip("engine does not support analytical gradients")
        mol, ff = sn2
        energy, grad = engine.energy_and_param_grad(mol, ff)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(grad))


class TestTorsionEnergy:
    """Engines must compute torsion energy contributions."""

    @staticmethod
    def _skip_if_no_torsion_support(engine: MMEngine) -> None:
        """Skip engines that don't yet support torsion energy."""
        unsupported = {"Tinker"}
        if engine.name in unsupported:
            pytest.skip(f"{engine.name} does not yet support torsion energy evaluation")

    def test_energy_finite_with_torsions(self, engine: MMEngine, ethane: tuple[Q2MMMolecule, ForceField]) -> None:
        self._skip_if_no_torsion_support(engine)
        mol, ff = ethane
        e = engine.energy(mol, ff)
        assert np.isfinite(e)

    def test_energy_changes_with_torsion_k(self, engine: MMEngine) -> None:
        """Changing torsion k should change total energy."""
        self._skip_if_no_torsion_support(engine)
        mol = make_ethane()
        ff_low = _ethane_ff(engine, torsion_k=0.05)
        ff_high = _ethane_ff(engine, torsion_k=1.00)
        e_low = engine.energy(mol, ff_low)
        e_high = engine.energy(mol, ff_high)
        assert np.isfinite(e_low)
        assert np.isfinite(e_high)
        assert e_low != e_high

    def test_torsion_energy_nonzero_for_nonzero_k(self, engine: MMEngine) -> None:
        """With torsion k > 0, total energy should differ from torsion-free."""
        self._skip_if_no_torsion_support(engine)
        mol = make_ethane()
        ff_with = _ethane_ff(engine, torsion_k=0.50)
        ff_without = ForceField(
            functional_form=_functional_form(engine),
            bonds=ff_with.bonds[:],
            angles=ff_with.angles[:],
            torsions=[],
        )
        e_with = engine.energy(mol, ff_with)
        e_without = engine.energy(mol, ff_without)
        assert abs(e_with - e_without) > 1e-6

    def test_torsion_energy_matches_openmm(self, engine: MMEngine, ethane: tuple[Q2MMMolecule, ForceField]) -> None:
        """Cross-engine parity: torsion energy must agree with OpenMM reference."""
        self._skip_if_no_torsion_support(engine)
        if engine.name == "OpenMM":
            pytest.skip("Reference engine")
        openmm = get_mm_engine("openmm")
        mol, ff = ethane
        e_engine = engine.energy(mol, ff)
        e_openmm = openmm.energy(mol, ff)
        assert abs(e_engine - e_openmm) < 1e-4, (
            f"{engine.name} torsion energy {e_engine:.6f} != OpenMM {e_openmm:.6f} "
            f"(diff={abs(e_engine - e_openmm):.2e})"
        )
