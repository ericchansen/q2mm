"""Integration tests for the JAX-MD OPLSAA backend."""

import numpy as np
import pytest

try:
    from q2mm.backends.mm.jax_md_engine import JaxMDEngine, _HAS_JAX_MD
except ImportError:
    _HAS_JAX_MD = False

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX_MD, reason="jax-md not installed"),
    pytest.mark.jax_md,
]

from test._shared import make_diatomic, make_water, make_noble_gas_pair

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam
from q2mm.models.molecule import Q2MMMolecule


# ---------------------------------------------------------------------------
# Molecule & FF factories
# ---------------------------------------------------------------------------


def _diatomic(distance: float = 0.74) -> Q2MMMolecule:
    return make_diatomic(distance=distance, bond_tolerance=1.5)


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length)


def _noble_gas_pair(distance: float = 3.0) -> Q2MMMolecule:
    return make_noble_gas_pair(distance=distance)


def _h2_ff(bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        name="H2-test",
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    bond_k: float = 503.6,
    bond_r0: float = 0.96,
    angle_k: float = 57.6,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _vdw_ff() -> ForceField:
    return ForceField(
        name="vdw-test",
        vdws=[VdwParam(atom_type="He", element="He", radius=1.40, epsilon=0.02)],
    )


def _engine(**kwargs):
    """Create JaxMDEngine with test defaults."""
    kwargs.setdefault("box", (50.0, 50.0, 50.0))
    return JaxMDEngine(**kwargs)


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineBasic:
    """Basic JaxMDEngine properties."""

    def setup_method(self):
        self.engine = _engine()

    def test_name(self):
        assert "JAX-MD" in self.engine.name

    def test_is_available(self):
        assert self.engine.is_available()

    def test_supports_runtime_params(self):
        assert self.engine.supports_runtime_params()

    def test_supports_analytical_gradients(self):
        assert self.engine.supports_analytical_gradients()


# ---------------------------------------------------------------------------
# Bond energy tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineBondEnergy:
    """Bond stretch energy tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_energy_at_equilibrium(self):
        mol = _diatomic(distance=0.74)
        ff = _h2_ff(bond_r0=0.74)
        assert self.engine.energy(mol, ff) == pytest.approx(0.0, abs=1e-10)

    def test_energy_stretched(self):
        mol = _diatomic(distance=0.78)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        expected = 359.7 * (0.78 - 0.74) ** 2
        assert self.engine.energy(mol, ff) == pytest.approx(expected, rel=1e-6)

    def test_energy_compressed(self):
        mol = _diatomic(distance=0.70)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        expected = 359.7 * (0.70 - 0.74) ** 2
        assert self.engine.energy(mol, ff) == pytest.approx(expected, rel=1e-6)

    def test_symmetry(self):
        """Harmonic energy is symmetric about equilibrium."""
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        e_up = self.engine.energy(_diatomic(0.78), ff)
        e_down = self.engine.energy(_diatomic(0.70), ff)
        assert e_up == pytest.approx(e_down, rel=1e-6)


# ---------------------------------------------------------------------------
# Angle energy tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineAngleEnergy:
    """Angle bending energy tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_energy_at_equilibrium(self):
        mol = _water(angle_deg=104.5)
        ff = _water_ff(angle_eq=104.5)
        assert self.engine.energy(mol, ff) == pytest.approx(0.0, abs=1e-8)

    def test_energy_bent(self):
        mol = _water(angle_deg=110.0)
        ff = _water_ff(angle_eq=104.5)
        e = self.engine.energy(mol, ff)
        assert e > 0.1, f"Expected positive angle energy, got {e}"


# ---------------------------------------------------------------------------
# vdW energy tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineVdW:
    """Lennard-Jones vdW energy tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_vdw_nonzero(self):
        mol = _noble_gas_pair(distance=3.0)
        ff = _vdw_ff()
        e = self.engine.energy(mol, ff)
        assert e != 0.0

    def test_vdw_repulsive_close(self):
        mol = _noble_gas_pair(distance=1.5)
        ff = _vdw_ff()
        e = self.engine.energy(mol, ff)
        assert e > 0.0, "Expected repulsive energy at close range"

    def test_vdw_lj_numeric(self):
        """Deterministic LJ check: two He atoms at known distance."""
        distance = 4.0
        mol = _noble_gas_pair(distance=distance)
        ff = _vdw_ff()  # radius=1.40 (Rmin/2), epsilon=0.02

        sigma = 1.40 * 2.0 / 2 ** (1.0 / 6.0)
        sr6 = (sigma / distance) ** 6
        expected = 4.0 * 0.02 * (sr6**2 - sr6)

        e = self.engine.energy(mol, ff)
        assert e == pytest.approx(expected, abs=1e-10)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineGradients:
    """Analytical gradient tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_gradient_matches_fd_bonds(self):
        """Analytical gradient matches finite differences for bonds."""
        mol = _diatomic(distance=0.78)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        e_anal, grad_anal = self.engine.energy_and_param_grad(mol, ff)

        # Finite difference gradient
        params = ff.get_param_vector()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += h
            ff.set_param_vector(p_plus)
            e_plus = self.engine.energy(mol, ff)

            p_minus = params.copy()
            p_minus[i] -= h
            ff.set_param_vector(p_minus)
            e_minus = self.engine.energy(mol, ff)

            grad_fd[i] = (e_plus - e_minus) / (2 * h)

        # Restore
        ff.set_param_vector(params)
        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)

    def test_gradient_matches_fd_angles(self):
        """Analytical gradient matches finite differences for angles."""
        mol = _water(angle_deg=110.0)
        ff = _water_ff(angle_eq=104.5)
        e_anal, grad_anal = self.engine.energy_and_param_grad(mol, ff)

        params = ff.get_param_vector()
        grad_fd = np.zeros_like(params)
        h = 1e-5
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += h
            ff.set_param_vector(p_plus)
            e_plus = self.engine.energy(mol, ff)

            p_minus = params.copy()
            p_minus[i] -= h
            ff.set_param_vector(p_minus)
            e_minus = self.engine.energy(mol, ff)

            grad_fd[i] = (e_plus - e_minus) / (2 * h)

        ff.set_param_vector(params)
        np.testing.assert_allclose(grad_anal, grad_fd, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Hessian & frequency tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineHessian:
    """Hessian and frequency tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_hessian_shape(self):
        mol = _water()
        ff = _water_ff()
        hess = self.engine.hessian(mol, ff)
        assert hess.shape == (9, 9)

    def test_hessian_symmetric(self):
        mol = _water()
        ff = _water_ff()
        hess = self.engine.hessian(mol, ff)
        np.testing.assert_allclose(hess, hess.T, atol=1e-12)

    def test_frequencies_count(self):
        mol = _water()
        ff = _water_ff()
        freqs = self.engine.frequencies(mol, ff)
        assert len(freqs) == 9
        # 6 near-zero (trans/rot), 3 vibrational
        near_zero = sum(1 for f in freqs if abs(f) < 10.0)
        assert near_zero == 6, f"Expected 6 near-zero modes, got {near_zero}"


# ---------------------------------------------------------------------------
# Cross-engine parity tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineParity:
    """Compare JaxMDEngine vs JaxEngine for consistency."""

    def setup_method(self):
        from q2mm.backends.mm.jax_engine import JaxEngine

        self.jaxmd = _engine()
        self.jax = JaxEngine()

    def test_bond_energy_matches(self):
        mol = _diatomic(distance=0.78)
        ff = _h2_ff()
        e1 = self.jaxmd.energy(mol, ff)
        e2 = self.jax.energy(mol, ff)
        assert e1 == pytest.approx(e2, abs=1e-10)

    def test_angle_energy_matches(self):
        mol = _water(angle_deg=110.0)
        ff = _water_ff()
        e1 = self.jaxmd.energy(mol, ff)
        e2 = self.jax.energy(mol, ff)
        assert e1 == pytest.approx(e2, abs=1e-10)

    def test_gradient_matches(self):
        mol = _water(angle_deg=110.0)
        ff = _water_ff()
        e1, g1 = self.jaxmd.energy_and_param_grad(mol, ff)
        e2, g2 = self.jax.energy_and_param_grad(mol, ff)
        assert e1 == pytest.approx(e2, abs=1e-10)
        np.testing.assert_allclose(g1, g2, atol=1e-10)

    def test_hessian_matches(self):
        mol = _water()
        ff = _water_ff()
        h1 = self.jaxmd.hessian(mol, ff)
        h2 = self.jax.hessian(mol, ff)
        np.testing.assert_allclose(h1, h2, atol=1e-12)

    def test_frequencies_match(self):
        mol = _water()
        ff = _water_ff()
        f1 = self.jaxmd.frequencies(mol, ff)
        f2 = self.jax.frequencies(mol, ff)
        np.testing.assert_allclose(f1, f2, atol=1e-2)


# ---------------------------------------------------------------------------
# Energy breakdown tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineBreakdown:
    """Per-term energy breakdown tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_breakdown_keys(self):
        mol = _water()
        ff = _water_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        assert set(bd.keys()) == {"bond", "angle", "torsion", "lj", "coulomb", "total"}

    def test_breakdown_sums_to_total(self):
        mol = _water()
        ff = _water_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        expected_total = bd["bond"] + bd["angle"] + bd["torsion"] + bd["lj"] + bd["coulomb"]
        assert bd["total"] == pytest.approx(expected_total, abs=1e-12)

    def test_only_bond_contributes(self):
        """For a stretched diatomic with no vdW, only bond should contribute."""
        mol = _diatomic(distance=0.78)
        ff = _h2_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        assert bd["bond"] > 0
        assert bd["angle"] == pytest.approx(0.0, abs=1e-12)
        assert bd["torsion"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Minimize tests
# ---------------------------------------------------------------------------


class TestJaxMDEngineMinimize:
    """Energy minimization tests."""

    def setup_method(self):
        self.engine = _engine()

    def test_minimize_converges(self):
        mol = _diatomic(distance=0.78)
        ff = _h2_ff(bond_r0=0.74)
        energy, atoms, coords = self.engine.minimize(mol, ff)
        assert energy < 0.01, f"Minimized energy too high: {energy}"
        assert len(atoms) == 2

    def test_minimize_water(self):
        mol = _water(angle_deg=115.0)
        ff = _water_ff(angle_eq=104.5)
        energy, atoms, coords = self.engine.minimize(mol, ff)
        assert energy < 0.01, f"Minimized energy too high: {energy}"


# ---------------------------------------------------------------------------
# Optimizer integration tests
# ---------------------------------------------------------------------------


class TestJaxMDOptimizerIntegration:
    """Full optimization loop with analytical gradients."""

    def setup_method(self):
        self.engine = _engine()

    def test_optimize_bond_k(self):
        """Optimize bond force constant to match a target energy."""
        from scipy.optimize import minimize as scipy_minimize

        mol = _diatomic(distance=0.78)
        target_energy = 1.0  # kcal/mol

        def objective(params):
            ff = _h2_ff(bond_k=params[0], bond_r0=0.74)
            e, _grad = self.engine.energy_and_param_grad(mol, ff)
            return (e - target_energy) ** 2

        def objective_grad(params):
            ff = _h2_ff(bond_k=params[0], bond_r0=0.74)
            e, grad = self.engine.energy_and_param_grad(mol, ff)
            # Chain rule: d/dk (e - target)^2 = 2*(e - target)*de/dk
            dloss_dk = 2.0 * (e - target_energy) * grad[0]
            return np.array([dloss_dk], dtype=float)

        result = scipy_minimize(
            objective,
            x0=[300.0],
            method="L-BFGS-B",
            bounds=[(1.0, 1000.0)],
            jac=objective_grad,
        )
        assert result.success or result.fun < 0.01
