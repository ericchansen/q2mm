"""Integration tests for the JAX differentiable MM backend."""

from __future__ import annotations

import math
import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import SN2_QM_REF as QM_REF, make_diatomic, make_water, make_noble_gas_pair

from q2mm.backends.mm.jax_engine import JaxEngine, _build_vdw_pairs
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, VdwParam
from q2mm.models.molecule import Q2MMMolecule


# ---------------------------------------------------------------------------
# Molecule factories (thin wrappers preserving original defaults)
# ---------------------------------------------------------------------------


def _diatomic(distance: float = 0.74) -> Q2MMMolecule:
    """H2 molecule at specified bond distance."""
    return make_diatomic(distance=distance, bond_tolerance=1.5)


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
    """Water molecule at specified geometry."""
    return make_water(angle_deg=angle_deg, bond_length=bond_length)


def _noble_gas_pair(distance: float = 3.0) -> Q2MMMolecule:
    """Two noble gas atoms for vdW testing (no bonds)."""
    return make_noble_gas_pair(distance=distance)


def _h2_ff(bond_k: float = 5.0, bond_r0: float = 0.74) -> ForceField:
    """H2 force field with one bond parameter."""
    return ForceField(
        name="H2-test",
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    bond_k: float = 7.0,
    bond_r0: float = 0.96,
    angle_k: float = 0.8,
    angle_eq: float = 104.5,
) -> ForceField:
    """Water force field with bond and angle parameters."""
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _vdw_ff() -> ForceField:
    """Force field with only vdW parameters for He."""
    return ForceField(
        name="vdw-test",
        vdws=[VdwParam(atom_type="He", element="He", radius=1.40, epsilon=0.02)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestJaxEngineBasic:
    """Basic JaxEngine functionality."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_name(self):
        assert "JAX" in self.engine.name

    def test_is_available(self):
        assert self.engine.is_available()

    def test_supports_runtime_params(self):
        assert self.engine.supports_runtime_params()


class TestJaxEngineBondEnergy:
    """Bond stretch energy tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_energy_at_equilibrium_is_zero(self):
        mol = _diatomic(distance=0.74)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        energy = self.engine.energy(mol, ff)
        assert abs(energy) < 1e-10, f"Energy at equilibrium should be ~0, got {energy}"

    def test_energy_increases_with_stretch(self):
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        mol_eq = _diatomic(distance=0.74)
        mol_stretch = _diatomic(distance=0.84)
        e_eq = self.engine.energy(mol_eq, ff)
        e_stretch = self.engine.energy(mol_stretch, ff)
        assert e_stretch > e_eq + 1e-6, "Stretched bond should have higher energy"

    def test_energy_increases_with_compression(self):
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        mol_eq = _diatomic(distance=0.74)
        mol_compress = _diatomic(distance=0.64)
        e_eq = self.engine.energy(mol_eq, ff)
        e_compress = self.engine.energy(mol_compress, ff)
        assert e_compress > e_eq + 1e-6, "Compressed bond should have higher energy"

    def test_energy_symmetric_stretch_compress(self):
        """Harmonic potential is symmetric about equilibrium."""
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        mol_stretch = _diatomic(distance=0.80)
        mol_compress = _diatomic(distance=0.68)
        e_stretch = self.engine.energy(mol_stretch, ff)
        e_compress = self.engine.energy(mol_compress, ff)
        assert abs(e_stretch - e_compress) < 1e-10, "Harmonic energy should be symmetric"

    def test_energy_scales_with_force_constant(self):
        """Doubling k should double the energy."""
        mol = _diatomic(distance=0.84)
        e1 = self.engine.energy(mol, _h2_ff(bond_k=5.0, bond_r0=0.74))
        e2 = self.engine.energy(mol, _h2_ff(bond_k=10.0, bond_r0=0.74))
        assert abs(e2 / e1 - 2.0) < 1e-10, f"Energy ratio should be 2.0, got {e2 / e1}"

    def test_energy_known_value(self):
        """Verify energy against hand calculation.

        E = k_conv * k_mdyna * (r - r0)^2
        k_conv ≈ 71.94 kcal/mol/Å² per mdyn/Å
        k = 5.0 mdyn/Å, dr = 0.1 Å
        E = 71.94 * 5.0 * 0.01 = 3.597 kcal/mol
        """
        from q2mm.backends.mm.jax_engine import _BOND_K_CONV

        mol = _diatomic(distance=0.84)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        energy = self.engine.energy(mol, ff)
        expected = _BOND_K_CONV * 5.0 * 0.1**2
        assert abs(energy - expected) < 1e-8, f"Expected {expected:.6f}, got {energy:.6f}"


class TestJaxEngineAngleEnergy:
    """Angle bend energy tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_energy_at_equilibrium_is_near_zero(self):
        mol = _water(angle_deg=104.5)
        ff = _water_ff(angle_eq=104.5)
        energy = self.engine.energy(mol, ff)
        # Bond contribution is ~0 (at eq), angle contribution is ~0
        assert abs(energy) < 1e-6, f"Energy at equilibrium should be ~0, got {energy}"

    def test_energy_increases_away_from_angle_eq(self):
        ff = _water_ff(angle_k=0.8, angle_eq=104.5)
        mol_eq = _water(angle_deg=104.5)
        mol_bent = _water(angle_deg=120.0)
        e_eq = self.engine.energy(mol_eq, ff)
        e_bent = self.engine.energy(mol_bent, ff)
        assert e_bent > e_eq + 1e-6, "Bent angle should have higher energy"


class TestJaxEngineVdW:
    """Van der Waals energy tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_vdw_energy_nonzero(self):
        mol = _noble_gas_pair(distance=3.0)
        ff = _vdw_ff()
        energy = self.engine.energy(mol, ff)
        # At typical distances, LJ energy should be small but nonzero
        assert energy != 0.0, "vdW energy should be nonzero"

    def test_vdw_repulsive_at_close_range(self):
        """At very close distance, repulsive wall dominates."""
        ff = _vdw_ff()
        mol_close = _noble_gas_pair(distance=1.5)
        mol_far = _noble_gas_pair(distance=4.0)
        e_close = self.engine.energy(mol_close, ff)
        e_far = self.engine.energy(mol_far, ff)
        assert e_close > e_far, "Close-range vdW should be more repulsive"


class TestJaxEngineGradients:
    """Analytical gradient tests via jax.grad."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_param_gradient_exists(self):
        mol = _diatomic(distance=0.84)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        energy, grad = self.engine.energy_and_param_grad(mol, ff)
        assert isinstance(grad, np.ndarray)
        assert len(grad) == ff.n_params
        assert not np.all(grad == 0.0), "Gradient should be nonzero away from equilibrium"

    def test_gradient_zero_at_equilibrium(self):
        """At equilibrium geometry, dE/dk = 0 (energy minimum w.r.t. geometry)."""
        mol = _diatomic(distance=0.74)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        _, grad = self.engine.energy_and_param_grad(mol, ff)
        # dE/dk = k_conv * (r-r0)^2 = 0 at r=r0
        assert abs(grad[0]) < 1e-10, f"dE/dk should be 0 at equilibrium, got {grad[0]}"

    def test_gradient_vs_finite_difference(self):
        """Analytical gradient should match finite differences."""
        mol = _diatomic(distance=0.84)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        _, grad_analytical = self.engine.energy_and_param_grad(mol, ff)

        eps = 1e-5
        vec = ff.get_param_vector().copy()
        grad_fd = np.zeros_like(vec)
        for i in range(len(vec)):
            vec_plus = vec.copy()
            vec_plus[i] += eps
            ff_plus = ForceField(
                name="fd+",
                bonds=[BondParam(elements=("H", "H"), force_constant=vec_plus[0], equilibrium=vec_plus[1])],
            )
            vec_minus = vec.copy()
            vec_minus[i] -= eps
            ff_minus = ForceField(
                name="fd-",
                bonds=[BondParam(elements=("H", "H"), force_constant=vec_minus[0], equilibrium=vec_minus[1])],
            )
            e_plus = self.engine.energy(mol, ff_plus)
            e_minus = self.engine.energy(mol, ff_minus)
            grad_fd[i] = (e_plus - e_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_analytical,
            grad_fd,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Analytical gradient does not match finite differences",
        )

    def test_water_gradient_vs_finite_difference(self):
        """Multi-parameter gradient test with bonds + angles."""
        mol = _water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)
        _, grad_analytical = self.engine.energy_and_param_grad(mol, ff)

        eps = 1e-5
        vec = ff.get_param_vector().copy()
        grad_fd = np.zeros_like(vec)

        for i in range(len(vec)):
            vec_plus = vec.copy()
            vec_plus[i] += eps
            ff_plus = ForceField(
                name="fd+",
                bonds=[BondParam(elements=("H", "O"), force_constant=vec_plus[0], equilibrium=vec_plus[1])],
                angles=[AngleParam(elements=("H", "O", "H"), force_constant=vec_plus[2], equilibrium=vec_plus[3])],
            )
            vec_minus = vec.copy()
            vec_minus[i] -= eps
            ff_minus = ForceField(
                name="fd-",
                bonds=[BondParam(elements=("H", "O"), force_constant=vec_minus[0], equilibrium=vec_minus[1])],
                angles=[AngleParam(elements=("H", "O", "H"), force_constant=vec_minus[2], equilibrium=vec_minus[3])],
            )
            e_plus = self.engine.energy(mol, ff_plus)
            e_minus = self.engine.energy(mol, ff_minus)
            grad_fd[i] = (e_plus - e_minus) / (2 * eps)

        np.testing.assert_allclose(
            grad_analytical,
            grad_fd,
            atol=1e-4,
            rtol=1e-4,
            err_msg="Water gradient (bonds+angles) does not match finite differences",
        )


class TestJaxEngineHessian:
    """Hessian and frequency tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_hessian_shape(self):
        mol = _diatomic(distance=0.74)
        ff = _h2_ff()
        hess = self.engine.hessian(mol, ff)
        n = len(mol.symbols)
        assert hess.shape == (3 * n, 3 * n), f"Expected ({3 * n}, {3 * n}), got {hess.shape}"

    def test_hessian_symmetric(self):
        mol = _water()
        ff = _water_ff()
        hess = self.engine.hessian(mol, ff)
        np.testing.assert_allclose(hess, hess.T, atol=1e-12, err_msg="Hessian should be symmetric")

    def test_frequencies_count(self):
        """3N modes for N atoms."""
        mol = _water()
        ff = _water_ff()
        freqs = self.engine.frequencies(mol, ff)
        assert len(freqs) == 3 * len(mol.symbols), f"Expected {3 * len(mol.symbols)} modes, got {len(freqs)}"

    def test_frequencies_have_near_zero_translations(self):
        """First 5-6 modes should be near zero (translations + rotations)."""
        mol = _water()
        ff = _water_ff()
        freqs = self.engine.frequencies(mol, ff)
        sorted_abs = sorted(abs(f) for f in freqs)
        # For a nonlinear molecule, 6 modes should be near-zero
        for i in range(5):
            assert sorted_abs[i] < 50.0, f"Mode {i} should be near-zero, got {sorted_abs[i]} cm⁻¹"


class TestJaxEngineMinimize:
    """Coordinate minimization tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_minimize_relaxes_stretched_bond(self):
        mol = _diatomic(distance=0.84)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        e_before = self.engine.energy(mol, ff)
        opt_energy, symbols, opt_coords = self.engine.minimize(mol, ff)
        assert opt_energy < e_before - 1e-6, "Minimization should lower energy"
        # Optimized distance should be near r0
        dist = np.linalg.norm(opt_coords[0] - opt_coords[1])
        assert abs(dist - 0.74) < 0.01, f"Optimized distance {dist:.4f} should be near r0=0.74"

    def test_minimize_water(self):
        mol = _water(angle_deg=120.0, bond_length=1.05)
        ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)
        e_before = self.engine.energy(mol, ff)
        opt_energy, symbols, opt_coords = self.engine.minimize(mol, ff)
        assert opt_energy < e_before - 1e-6, "Minimization should lower energy"


class TestJaxEngineHandle:
    """Context/handle reuse tests."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_create_context_and_reuse(self):
        mol = _diatomic(distance=0.84)
        ff = _h2_ff()
        handle = self.engine.create_context(mol, ff)
        e1 = self.engine.energy(handle, ff)
        e2 = self.engine.energy(handle, ff)
        assert e1 == e2, "Same handle should give same energy"

    def test_handle_with_different_params(self):
        mol = _diatomic(distance=0.84)
        ff1 = _h2_ff(bond_k=5.0, bond_r0=0.74)
        ff2 = _h2_ff(bond_k=10.0, bond_r0=0.74)
        handle = self.engine.create_context(mol, ff1)
        e1 = self.engine.energy(handle, ff1)
        e2 = self.engine.energy(handle, ff2)
        assert abs(e2 / e1 - 2.0) < 1e-10, "Doubling k should double energy"


class TestBuildVdwPairs:
    """Unit tests for vdW pair list construction."""

    def test_no_bonds_all_pairs(self):
        pairs = _build_vdw_pairs(3, [])
        expected = np.array([[0, 1], [0, 2], [1, 2]])
        np.testing.assert_array_equal(pairs, expected)

    def test_12_exclusion(self):
        # Three atoms in a chain: 0-1-2
        pairs = _build_vdw_pairs(3, [(0, 1), (1, 2)])
        # 0-1 excluded (1-2 bond), 1-2 excluded (1-2 bond), 0-2 excluded (1-3)
        assert len(pairs) == 0, "All pairs should be excluded for 3-atom chain"

    def test_4_atom_chain(self):
        # 0-1-2-3 chain: 1-2 exclusions: (0,1), (1,2), (2,3)
        # 1-3 exclusions: (0,2), (1,3)
        # Remaining: (0,3)
        pairs = _build_vdw_pairs(4, [(0, 1), (1, 2), (2, 3)])
        assert len(pairs) == 1
        assert tuple(pairs[0]) == (0, 3)

    def test_single_atom(self):
        pairs = _build_vdw_pairs(1, [])
        assert len(pairs) == 0


class TestJaxEngineSN2:
    """Integration test with real SN2 test case data."""

    def setup_method(self):
        self.engine = JaxEngine()

    @pytest.mark.skipif(
        not (QM_REF / "sn2-ts-optimized.xyz").exists(),
        reason="SN2 test fixtures not found",
    )
    def test_sn2_energy_is_finite(self):
        """Verify JaxEngine can process a real molecule (SN2 TS)."""
        mol = Q2MMMolecule.from_xyz(QM_REF / "sn2-ts-optimized.xyz", bond_tolerance=1.5)
        ff = ForceField.create_for_molecule(mol)
        energy = self.engine.energy(mol, ff)
        assert np.isfinite(energy), f"Energy should be finite, got {energy}"

    @pytest.mark.skipif(
        not (QM_REF / "sn2-ts-optimized.xyz").exists(),
        reason="SN2 test fixtures not found",
    )
    def test_sn2_gradient_is_finite(self):
        """Verify analytical gradient works on a real molecule."""
        mol = Q2MMMolecule.from_xyz(QM_REF / "sn2-ts-optimized.xyz", bond_tolerance=1.5)
        ff = ForceField.create_for_molecule(mol)
        energy, grad = self.engine.energy_and_param_grad(mol, ff)
        assert np.isfinite(energy)
        assert np.all(np.isfinite(grad)), "All gradient components should be finite"


class TestJaxOptimizerIntegration:
    """Test JaxEngine + ScipyOptimizer with analytical gradients."""

    def setup_method(self):
        self.engine = JaxEngine()

    def test_analytical_gradient_optimization(self):
        """Optimize H2 bond params with analytical JAX gradients."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = _diatomic(distance=0.74)
        # Start with wrong k; target energy at eq is 0, so aim for that
        ff = _h2_ff(bond_k=3.0, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)

        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=100, jac="analytical", verbose=False)
        result = optimizer.optimize(objective)
        # Energy at equilibrium is 0 regardless of k, so score should be ~0
        assert result.final_score < 1e-10

    def test_analytical_vs_fd_optimization_convergence(self):
        """Analytical and FD optimizations should reach similar final scores."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = _diatomic(distance=0.80)
        target_energy = 10.0  # arbitrary target
        ff_analytical = _h2_ff(bond_k=3.0, bond_r0=0.74)
        ff_fd = _h2_ff(bond_k=3.0, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=target_energy, molecule_idx=0, weight=1.0)

        # Analytical gradient optimization
        obj_a = ObjectiveFunction(forcefield=ff_analytical, engine=self.engine, molecules=[mol], reference=ref)
        opt_a = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_a = opt_a.optimize(obj_a)

        # Finite-difference optimization
        obj_fd = ObjectiveFunction(forcefield=ff_fd, engine=self.engine, molecules=[mol], reference=ref)
        opt_fd = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac=None, verbose=False)
        res_fd = opt_fd.optimize(obj_fd)

        # Both should converge to similar scores
        assert res_a.final_score < 1e-4, f"Analytical didn't converge: {res_a.final_score}"
        assert res_fd.final_score < 1e-4, f"FD didn't converge: {res_fd.final_score}"

    def test_objective_gradient_method_exists(self):
        """ObjectiveFunction.gradient() should work with JaxEngine."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = _diatomic(distance=0.80)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        grad = obj.gradient(ff.get_param_vector())
        assert len(grad) == ff.n_params
        assert isinstance(grad, np.ndarray)

    def test_objective_gradient_matches_fd(self):
        """ObjectiveFunction.gradient() should match finite-diff of objective."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = _diatomic(distance=0.80)
        ff = _h2_ff(bond_k=5.0, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=2.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = ff.get_param_vector().copy()
        analytical_grad = obj.gradient(vec)

        # Finite differences of the objective
        eps = 1e-5
        fd_grad = np.zeros_like(vec)
        for i in range(len(vec)):
            v_plus = vec.copy()
            v_plus[i] += eps
            obj.reset()
            f_plus = obj(v_plus)
            v_minus = vec.copy()
            v_minus[i] -= eps
            obj.reset()
            f_minus = obj(v_minus)
            fd_grad[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(
            analytical_grad,
            fd_grad,
            atol=1e-3,
            rtol=1e-3,
            err_msg="ObjectiveFunction.gradient() does not match FD of objective",
        )
