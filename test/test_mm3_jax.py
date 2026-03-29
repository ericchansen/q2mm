"""Tests for MM3 functional forms in JaxEngine (issue #91).

Verifies:
- MM3 cubic bond, sextic angle, and buffered 14-7 vdW energy functions
- JaxEngine(functional_form="mm3") produces correct energies
- jax.grad works correctly through all MM3 forms
- Parity with OpenMM MM3 implementation (when OpenMM available)
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

try:
    import openmm  # noqa: F401

    _HAS_OPENMM = True
except ImportError:
    _HAS_OPENMM = False

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import make_diatomic, make_noble_gas_pair, make_water

from q2mm.backends.mm.jax_engine import (
    JaxEngine,
    _mm3_angle_energy,
    _mm3_bond_energy,
    _mm3_vdw_energy,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, VdwParam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2_ff_mm3() -> ForceField:
    """H₂ force field with MM3 functional form."""
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74)],
        functional_form=FunctionalForm.MM3,
    )


def _water_ff_mm3() -> ForceField:
    """Water force field with MM3 functional form."""
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
        functional_form=FunctionalForm.MM3,
    )


def _he2_ff_mm3() -> ForceField:
    """He₂ force field with MM3 vdW (no bonds)."""
    return ForceField(
        vdws=[VdwParam(atom_type="He", element="He", radius=1.4, epsilon=0.056)],
        functional_form=FunctionalForm.MM3,
    )


# ---------------------------------------------------------------------------
# Unit-level tests for individual energy kernels
# ---------------------------------------------------------------------------


class TestMM3BondEnergy:
    """Test _mm3_bond_energy against known values."""

    def test_at_equilibrium_zero(self) -> None:
        """Energy is zero at equilibrium distance."""
        k = jnp.array([5.0])
        r0 = jnp.array([0.74])
        coords = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        e = _mm3_bond_energy(k, r0, coords, bond_idx)
        assert float(e) == pytest.approx(0.0, abs=1e-10)

    def test_small_stretch(self) -> None:
        """Verify cubic correction for small displacement."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        dr = 0.05  # Å
        coords = jnp.array([[0.0, 0.0, 0.0], [1.0 + dr, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        e = float(_mm3_bond_energy(k, r0, coords, bond_idx))

        # Expected: k*dr²*(1 - 2.55*dr + c4*dr²)
        c4 = (7.0 / 12.0) * 2.55**2
        expected = 5.0 * dr**2 * (1.0 - 2.55 * dr + c4 * dr**2)
        assert e == pytest.approx(expected, rel=1e-8)

    def test_differs_from_harmonic(self) -> None:
        """MM3 bond energy differs from harmonic for non-zero displacement."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        coords = jnp.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
        bond_idx = jnp.array([[0, 1]])
        mm3_e = float(_mm3_bond_energy(k, r0, coords, bond_idx))
        harmonic_e = float(jnp.sum(k * (1.1 - 1.0) ** 2))
        assert mm3_e != pytest.approx(harmonic_e, rel=1e-4)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 bond energy."""
        k = jnp.array([5.0])
        r0 = jnp.array([1.0])
        bond_idx = jnp.array([[0, 1]])

        def energy_of_r(r: float) -> float:
            coords = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
            return _mm3_bond_energy(k, r0, coords, bond_idx)

        grad_fn = jax.grad(energy_of_r)
        # Gradient should be non-zero at displaced distance
        g = float(grad_fn(1.05))
        assert abs(g) > 0.01

        # Verify against finite differences
        h = 1e-6
        fd = (float(energy_of_r(1.05 + h)) - float(energy_of_r(1.05 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-4)


class TestMM3AngleEnergy:
    """Test _mm3_angle_energy against known values."""

    def test_at_equilibrium_zero(self) -> None:
        """Energy is zero at equilibrium angle."""
        k = jnp.array([0.7])
        theta0 = jnp.array([np.deg2rad(109.5)])
        # Build a perfectly angled geometry
        theta = np.deg2rad(109.5)
        coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.cos(theta), np.sin(theta), 0.0]])
        angle_idx = jnp.array([[0, 1, 2]])
        e = _mm3_angle_energy(k, theta0, coords, angle_idx)
        assert float(e) == pytest.approx(0.0, abs=1e-10)

    def test_small_bend(self) -> None:
        """Verify sextic correction for small angular displacement."""
        k = jnp.array([0.7])
        theta0_deg = 109.5
        theta0_rad = np.deg2rad(theta0_deg)
        theta0 = jnp.array([theta0_rad])

        # Bent by 5 degrees
        theta_actual = np.deg2rad(114.5)
        coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [np.cos(theta_actual), np.sin(theta_actual), 0.0]])
        angle_idx = jnp.array([[0, 1, 2]])
        e = float(_mm3_angle_energy(k, theta0, coords, angle_idx))

        dtheta = theta_actual - theta0_rad
        dtheta_deg = dtheta * (180.0 / np.pi)
        anharmonic = (
            1.0 + (-0.014) * dtheta_deg + 5.6e-5 * dtheta_deg**2 + (-7.0e-7) * dtheta_deg**3 + 9.0e-10 * dtheta_deg**4
        )
        expected = 0.7 * dtheta**2 * anharmonic
        assert e == pytest.approx(expected, rel=1e-6)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 angle energy."""
        k = jnp.array([0.7])
        theta0 = jnp.array([np.deg2rad(109.5)])
        angle_idx = jnp.array([[0, 1, 2]])

        def energy_of_angle(theta_rad: jnp.ndarray) -> jnp.ndarray:
            coords = jnp.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [jnp.cos(theta_rad), jnp.sin(theta_rad), 0.0]])
            return _mm3_angle_energy(k, theta0, coords, angle_idx)

        grad_fn = jax.grad(energy_of_angle)
        g = float(grad_fn(jnp.deg2rad(115.0)))
        assert abs(g) > 0.0


class TestMM3VdwEnergy:
    """Test _mm3_vdw_energy against known values."""

    def test_zero_pairs_returns_zero(self) -> None:
        """No pairs → zero energy."""
        radius = jnp.array([1.4])
        epsilon = jnp.array([0.056])
        coords = jnp.array([[0.0, 0.0, 0.0]])
        pairs = jnp.empty((0, 2), dtype=jnp.int32)
        assert float(_mm3_vdw_energy(radius, epsilon, coords, pairs)) == 0.0

    def test_at_equilibrium_negative(self) -> None:
        """At r = rv, energy should be negative (attractive well)."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        rv = 2.8  # radius1 + radius2
        coords = jnp.array([[0.0, 0.0, 0.0], [rv, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        # At r=rv: ε*(184000*exp(-12) - 2.25*(1)^6) ≈ ε*(1.129 - 2.25) < 0
        assert e < 0

    def test_short_range_wall(self) -> None:
        """Below rc=0.34*rv, repulsive wall kicks in."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        rv = 2.8
        rc = 0.34 * rv
        coords = jnp.array([[0.0, 0.0, 0.0], [rc * 0.5, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        # Should be very repulsive
        assert e > 100.0

    def test_known_value_at_2rv(self) -> None:
        """Verify energy at r=2*rv matches analytical expression."""
        radius = jnp.array([1.5, 1.5])
        epsilon = jnp.array([0.1, 0.1])
        rv = 3.0
        r = 2.0 * rv  # r = 6.0 Å
        coords = jnp.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        pairs = jnp.array([[0, 1]])
        e = float(_mm3_vdw_energy(radius, epsilon, coords, pairs))
        expected = 0.1 * (184000.0 * np.exp(-12.0 * r / rv) - 2.25 * (rv / r) ** 6)
        assert e == pytest.approx(expected, rel=1e-6)

    def test_differentiable(self) -> None:
        """jax.grad works through MM3 vdW energy."""
        radius = jnp.array([1.4, 1.4])
        epsilon = jnp.array([0.056, 0.056])
        pairs = jnp.array([[0, 1]])

        def energy_of_dist(d: float) -> float:
            coords = jnp.array([[0.0, 0.0, 0.0], [d, 0.0, 0.0]])
            return _mm3_vdw_energy(radius, epsilon, coords, pairs)

        grad_fn = jax.grad(energy_of_dist)
        g = float(grad_fn(3.0))
        h = 1e-6
        fd = (float(energy_of_dist(3.0 + h)) - float(energy_of_dist(3.0 - h))) / (2 * h)
        assert g == pytest.approx(fd, rel=1e-3)


# ---------------------------------------------------------------------------
# Integration tests: JaxEngine with functional_form="mm3"
# ---------------------------------------------------------------------------


class TestJaxEngineMM3:
    """Test JaxEngine with MM3 functional form end-to-end."""

    def test_supported_forms_includes_mm3(self) -> None:
        """JaxEngine now supports both harmonic and mm3."""
        engine = JaxEngine()
        forms = engine.supported_functional_forms()
        assert "harmonic" in forms
        assert "mm3" in forms

    def test_mm3_energy_diatomic(self) -> None:
        """MM3 energy computation for H₂."""
        engine = JaxEngine()
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        e = engine.energy(mol, ff)
        assert isinstance(e, float)
        assert np.isfinite(e)
        assert e > 0  # displaced from equilibrium

    def test_mm3_energy_at_equilibrium(self) -> None:
        """MM3 energy is zero at equilibrium geometry."""
        engine = JaxEngine()
        mol = make_diatomic(distance=0.74, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        e = engine.energy(mol, ff)
        assert e == pytest.approx(0.0, abs=1e-8)

    def test_mm3_energy_water(self) -> None:
        """MM3 energy computation for water (bonds + angles)."""
        engine = JaxEngine()
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff_mm3()
        e = engine.energy(mol, ff)
        assert isinstance(e, float)
        assert np.isfinite(e)

    def test_mm3_hessian(self) -> None:
        """MM3 Hessian computation produces valid matrix."""
        engine = JaxEngine()
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        hess = engine.hessian(mol, ff)
        assert hess.shape == (6, 6)
        # Hessian should be symmetric
        np.testing.assert_allclose(hess, hess.T, atol=1e-10)

    def test_mm3_differs_from_harmonic(self) -> None:
        """MM3 energy differs from harmonic for same parameters."""
        engine = JaxEngine()
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff_mm3 = _h2_ff_mm3()
        ff_harm = ForceField(
            bonds=[BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74)],
            functional_form=FunctionalForm.HARMONIC,
        )
        e_mm3 = engine.energy(mol, ff_mm3)
        e_harm = engine.energy(mol, ff_harm)
        # Both should be positive (displaced from equilibrium) but different
        assert e_mm3 > 0
        assert e_harm > 0
        assert e_mm3 != pytest.approx(e_harm, rel=1e-4)

    def test_mm3_analytical_gradient(self) -> None:
        """jax.grad through full MM3 energy matches finite differences."""
        engine = JaxEngine()
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()
        _e, grad_anal = engine.energy_and_param_grad(mol, ff)

        # Finite difference gradient
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

    def test_mm3_vdw_energy(self) -> None:
        """MM3 vdW energy computation for He pair."""
        engine = JaxEngine()
        mol = make_noble_gas_pair(distance=3.0, bond_tolerance=0.5)
        ff = _he2_ff_mm3()
        e = engine.energy(mol, ff)
        assert isinstance(e, float)
        assert np.isfinite(e)


# ---------------------------------------------------------------------------
# Parity test: JAX MM3 vs OpenMM MM3
# ---------------------------------------------------------------------------


@pytest.mark.openmm
@pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed")
class TestMM3ParityJaxVsOpenMM:
    """Verify JAX MM3 produces identical energies to OpenMM MM3."""

    def test_bond_parity(self) -> None:
        """MM3 bond energy: JAX vs OpenMM within 1e-6 kcal/mol."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff_mm3()

        jax_e = JaxEngine().energy(mol, ff)
        omm_e = OpenMMEngine().energy(mol, ff)
        assert jax_e == pytest.approx(omm_e, abs=1e-6)

    def test_water_parity(self) -> None:
        """MM3 bond+angle energy: JAX vs OpenMM within 1e-5 kcal/mol."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff_mm3()

        jax_e = JaxEngine().energy(mol, ff)
        omm_e = OpenMMEngine().energy(mol, ff)
        assert jax_e == pytest.approx(omm_e, abs=1e-5)

    def test_vdw_parity(self) -> None:
        """MM3 vdW energy: JAX vs OpenMM within 1e-6 kcal/mol."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = make_noble_gas_pair(distance=3.5, bond_tolerance=0.5)
        ff = _he2_ff_mm3()

        jax_e = JaxEngine().energy(mol, ff)
        omm_e = OpenMMEngine().energy(mol, ff)
        assert jax_e == pytest.approx(omm_e, abs=1e-6)
