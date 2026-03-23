"""JAX-engine-specific tests.

Contract tests (energy, hessian, frequencies, minimize, gradients) are
in test_engine_contract.py and run for every registered engine.  This
file covers only behaviour unique to the JAX backend:

* Known-value energy check using internal ``_BOND_K_CONV`` constant
* Context / handle reuse API
* Internal ``_build_vdw_pairs`` helper
* Optimizer integration with analytical JAX gradients
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import make_diatomic

from q2mm.backends.mm.jax_engine import JaxEngine, _build_vdw_pairs
from q2mm.models.forcefield import BondParam, ForceField


def _h2_ff(bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


class TestJaxEngineKnownValue:
    """Verify energy against hand calculation using internal constant."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_energy_known_value(self) -> None:
        from q2mm.backends.mm.jax_engine import _BOND_K_CONV

        mol = make_diatomic(distance=0.84, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        energy = self.engine.energy(mol, ff)
        expected = _BOND_K_CONV * 359.7 * 0.1**2
        assert abs(energy - expected) < 1e-8


class TestJaxEngineHandle:
    """Context/handle reuse tests."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_create_context_and_reuse(self) -> None:
        mol = make_diatomic(distance=0.84, bond_tolerance=1.5)
        ff = _h2_ff()
        handle = self.engine.create_context(mol, ff)
        e1 = self.engine.energy(handle, ff)
        e2 = self.engine.energy(handle, ff)
        assert e1 == e2

    def test_handle_with_different_params(self) -> None:
        mol = make_diatomic(distance=0.84, bond_tolerance=1.5)
        ff1 = _h2_ff(bond_k=359.7)
        ff2 = _h2_ff(bond_k=719.4)
        handle = self.engine.create_context(mol, ff1)
        e1 = self.engine.energy(handle, ff1)
        e2 = self.engine.energy(handle, ff2)
        assert abs(e2 / e1 - 2.0) < 1e-10


class TestBuildVdwPairs:
    """Unit tests for vdW pair list construction."""

    def test_no_bonds_all_pairs(self) -> None:
        pairs = _build_vdw_pairs(3, [])
        expected = np.array([[0, 1], [0, 2], [1, 2]])
        np.testing.assert_array_equal(pairs, expected)

    def test_12_exclusion(self) -> None:
        pairs = _build_vdw_pairs(3, [(0, 1), (1, 2)])
        assert len(pairs) == 0

    def test_4_atom_chain(self) -> None:
        pairs = _build_vdw_pairs(4, [(0, 1), (1, 2), (2, 3)])
        assert len(pairs) == 1
        assert tuple(pairs[0]) == (0, 3)

    def test_single_atom(self) -> None:
        pairs = _build_vdw_pairs(1, [])
        assert len(pairs) == 0


class TestJaxOptimizerIntegration:
    """Test JaxEngine + ScipyOptimizer with analytical gradients."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_analytical_gradient_optimization(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=100, jac="analytical", verbose=False)
        result = optimizer.optimize(objective)
        assert result.final_score < 1e-10

    def test_analytical_vs_fd_optimization_convergence(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        target_energy = 10.0
        ff_analytical = _h2_ff(bond_k=215.8, bond_r0=0.74)
        ff_fd = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=target_energy, molecule_idx=0, weight=1.0)

        obj_a = ObjectiveFunction(forcefield=ff_analytical, engine=self.engine, molecules=[mol], reference=ref)
        opt_a = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_a = opt_a.optimize(obj_a)

        obj_fd = ObjectiveFunction(forcefield=ff_fd, engine=self.engine, molecules=[mol], reference=ref)
        opt_fd = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac=None, verbose=False)
        res_fd = opt_fd.optimize(obj_fd)

        assert res_a.final_score < 1e-4
        assert res_fd.final_score < 0.01

    def test_objective_gradient_method_exists(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        grad = obj.gradient(ff.get_param_vector())
        assert len(grad) == ff.n_params
        assert isinstance(grad, np.ndarray)

    def test_objective_gradient_matches_fd(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=2.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = ff.get_param_vector().copy()
        analytical_grad = obj.gradient(vec)

        eps = 1e-5
        fd_grad = np.zeros_like(vec)
        for i in range(len(vec)):
            v_plus, v_minus = vec.copy(), vec.copy()
            v_plus[i] += eps
            v_minus[i] -= eps
            obj.reset()
            f_plus = obj(v_plus)
            obj.reset()
            f_minus = obj(v_minus)
            fd_grad[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(analytical_grad, fd_grad, atol=1e-3, rtol=1e-3)
