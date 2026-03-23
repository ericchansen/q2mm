"""JAX-MD-engine-specific tests.

Contract tests (energy, hessian, frequencies, minimize, gradients) are
in test_engine_contract.py and run for every registered engine.  This
file covers only behaviour unique to the JAX-MD backend:

* Known-value LJ numeric check using sigma/epsilon conversion
* Cross-engine parity (JAX-MD vs JAX)
* Per-term energy breakdown API (JaxMDEngine-only)
* Optimizer integration with analytical gradients
"""

from __future__ import annotations

from typing import Any

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

from test._shared import make_diatomic, make_noble_gas_pair, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam


def _h2_ff(bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    bond_k: float = 503.6, bond_r0: float = 0.96, angle_k: float = 57.6, angle_eq: float = 104.5
) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _vdw_ff() -> ForceField:
    return ForceField(
        vdws=[VdwParam(atom_type="He", element="He", radius=1.40, epsilon=0.02)],
    )


def _engine(**kwargs: Any) -> JaxMDEngine:
    kwargs.setdefault("box", (50.0, 50.0, 50.0))
    return JaxMDEngine(**kwargs)


class TestJaxMDEngineLJNumeric:
    """Verify LJ energy against analytical sigma/epsilon formula."""

    def setup_method(self) -> None:
        self.engine = _engine()

    def test_vdw_lj_numeric(self) -> None:
        distance = 4.0
        mol = make_noble_gas_pair(distance=distance)
        ff = _vdw_ff()

        sigma = 1.40 * 2.0 / 2 ** (1.0 / 6.0)
        sr6 = (sigma / distance) ** 6
        expected = 4.0 * 0.02 * (sr6**2 - sr6)

        assert self.engine.energy(mol, ff) == pytest.approx(expected, abs=1e-10)


class TestJaxMDEngineParity:
    """Compare JaxMDEngine vs JaxEngine for consistency."""

    def setup_method(self) -> None:
        from q2mm.backends.mm.jax_engine import JaxEngine

        self.jaxmd = _engine()
        self.jax = JaxEngine()

    def test_bond_energy_matches(self) -> None:
        mol = make_diatomic(distance=0.78, bond_tolerance=1.5)
        ff = _h2_ff()
        assert self.jaxmd.energy(mol, ff) == pytest.approx(self.jax.energy(mol, ff), abs=1e-10)

    def test_angle_energy_matches(self) -> None:
        mol = make_water(angle_deg=110.0)
        ff = _water_ff()
        assert self.jaxmd.energy(mol, ff) == pytest.approx(self.jax.energy(mol, ff), abs=1e-10)

    def test_gradient_matches(self) -> None:
        mol = make_water(angle_deg=110.0)
        ff = _water_ff()
        e1, g1 = self.jaxmd.energy_and_param_grad(mol, ff)
        e2, g2 = self.jax.energy_and_param_grad(mol, ff)
        assert e1 == pytest.approx(e2, abs=1e-10)
        np.testing.assert_allclose(g1, g2, atol=1e-10)

    def test_hessian_matches(self) -> None:
        mol = make_water()
        ff = _water_ff()
        np.testing.assert_allclose(self.jaxmd.hessian(mol, ff), self.jax.hessian(mol, ff), atol=1e-12)

    def test_frequencies_match(self) -> None:
        mol = make_water()
        ff = _water_ff()
        np.testing.assert_allclose(self.jaxmd.frequencies(mol, ff), self.jax.frequencies(mol, ff), atol=1e-2)


class TestJaxMDEngineBreakdown:
    """Per-term energy breakdown tests."""

    def setup_method(self) -> None:
        self.engine = _engine()

    def test_breakdown_keys(self) -> None:
        mol = make_water()
        ff = _water_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        assert set(bd.keys()) == {"bond", "angle", "torsion", "lj", "coulomb", "total"}

    def test_breakdown_sums_to_total(self) -> None:
        mol = make_water()
        ff = _water_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        expected_total = bd["bond"] + bd["angle"] + bd["torsion"] + bd["lj"] + bd["coulomb"]
        assert bd["total"] == pytest.approx(expected_total, abs=1e-12)

    def test_only_bond_contributes(self) -> None:
        mol = make_diatomic(distance=0.78, bond_tolerance=1.5)
        ff = _h2_ff()
        bd = self.engine.energy_breakdown(mol, ff)
        assert bd["bond"] > 0
        assert bd["angle"] == pytest.approx(0.0, abs=1e-12)
        assert bd["torsion"] == pytest.approx(0.0, abs=1e-12)


class TestJaxMDOptimizerIntegration:
    """Full optimization loop with analytical gradients."""

    def setup_method(self) -> None:
        self.engine = _engine()

    def test_optimize_bond_k(self) -> None:
        from scipy.optimize import minimize as scipy_minimize

        mol = make_diatomic(distance=0.78, bond_tolerance=1.5)
        target_energy = 1.0

        def objective(params: np.ndarray) -> float:
            ff = _h2_ff(bond_k=params[0], bond_r0=0.74)
            e, _grad = self.engine.energy_and_param_grad(mol, ff)
            return (e - target_energy) ** 2

        def objective_grad(params: np.ndarray) -> np.ndarray:
            ff = _h2_ff(bond_k=params[0], bond_r0=0.74)
            e, grad = self.engine.energy_and_param_grad(mol, ff)
            return np.array([2.0 * (e - target_energy) * grad[0]], dtype=float)

        result = scipy_minimize(objective, x0=[300.0], method="L-BFGS-B", bounds=[(1.0, 1000.0)], jac=objective_grad)
        assert result.success or result.fun < 0.01
