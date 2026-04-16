"""Unit tests for JaxOptOptimizer.

Verifies constructor validation, method dispatch, and basic convergence
on simple energy objectives.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_JAXOPT = importlib.util.find_spec("jaxopt") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.skipif(not _HAS_JAXOPT, reason="jaxopt not installed"),
    pytest.mark.jax,
]

from test._shared import make_diatomic, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField

_HAS_CH3F = False
try:
    from test._shared import CH3F_DATA_AVAILABLE

    _HAS_CH3F = CH3F_DATA_AVAILABLE
except ImportError:
    pass

# Module-level globals populated by autouse fixture
JaxEngine = None


def _h2_ff(bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    bond_k: float = 553.0,
    bond_r0: float = 0.96,
    angle_k: float = 49.9,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


@pytest.fixture(autouse=True)
def _init_jax() -> None:
    """Import JAX lazily so module collection is CUDA-free."""
    from q2mm.backends.mm._jax_common import ensure_jax, ensure_jaxopt

    ensure_jax()
    ensure_jaxopt()
    global JaxEngine  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxEngine as _JE

    JaxEngine = _JE


class TestJaxOptOptimizerValidation:
    """Constructor and input validation tests."""

    def test_invalid_method(self) -> None:
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        with pytest.raises(ValueError, match="Unknown method"):
            JaxOptOptimizer(method="not_a_method")

    def test_valid_methods(self) -> None:
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        for method in ("lbfgs", "lbfgsb", "gradient_descent"):
            opt = JaxOptOptimizer(method=method)
            assert opt.method == method

    def test_engine_type_check(self) -> None:
        from unittest.mock import MagicMock

        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        fake_engine = MagicMock()
        fake_engine.__class__.__name__ = "FakeEngine"
        obj = ObjectiveFunction(forcefield=ff, engine=fake_engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=10, verbose=False)
        with pytest.raises(TypeError, match="JaxOptOptimizer requires a JaxEngine"):
            optimizer.optimize(obj)


class TestJaxOptOptimizerConvergence:
    """Convergence tests on simple systems."""

    def test_lbfgs_h2_energy(self) -> None:
        """L-BFGS converges on H2 energy optimization."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        # Perturbed r0 so energy at geometry != 0 (gives non-zero initial loss)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj)

        assert result.final_score <= result.initial_score
        assert result.method == "jaxopt:lbfgs"
        assert result.jac_mode == "jit"
        assert result.eps is None

    def test_lbfgsb_h2_energy(self) -> None:
        """L-BFGS-B converges with box constraints."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgsb", maxiter=200, verbose=False)
        result = optimizer.optimize(obj)

        assert result.final_score <= result.initial_score
        assert result.method == "jaxopt:lbfgsb"

    def test_result_format(self) -> None:
        """OptimizationResult has all expected fields."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import OptimizationResult

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=10, verbose=False)
        result = optimizer.optimize(obj)

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.initial_score, float)
        assert isinstance(result.final_score, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.initial_params, np.ndarray)
        assert isinstance(result.final_params, np.ndarray)
        assert isinstance(result.history, list)

    def test_water_energy_convergence(self) -> None:
        """Water (bond + angle) energy converges with L-BFGS."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=104.5)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj)

        assert result.final_score <= result.initial_score

    def test_forcefield_updated(self) -> None:
        """After optimization, the forcefield params are updated."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        initial_params = ff.get_param_vector().copy()

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj)

        final_params = ff.get_param_vector()
        np.testing.assert_array_equal(final_params, result.final_params)
        # If the optimizer improved, params should differ from initial
        if result.final_score < result.initial_score:
            assert not np.allclose(final_params, initial_params)


class TestJaxOptFrequencyConvergence:
    """Frequency-based optimization convergence."""

    @pytest.mark.skipif(not _HAS_CH3F, reason="CH3F data not available")
    def test_ch3f_frequency_convergence(self) -> None:
        """L-BFGS converges on CH3F frequency optimization."""
        from q2mm.models.hessian import hessian_to_frequencies
        from q2mm.models.molecule import Q2MMMolecule
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        from test._shared import CH3F_HESS, CH3F_XYZ

        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        hess_qm = np.load(CH3F_HESS)
        mol = mol.with_hessian(hess_qm)

        # Build a real FF and perturb it
        ff = estimate_force_constants(mol)
        freqs_qm = hessian_to_frequencies(hess_qm, list(mol.symbols))

        # Perturb bond force constants by 20%
        params = ff.get_param_vector()
        n_bonds = len(ff.bonds)
        for i in range(n_bonds):
            params[2 * i] *= 0.8  # force constant
        ff.set_param_vector(params)

        engine = JaxEngine()

        # Add only real vibrational frequencies (skip first 6 trans/rot)
        n3 = 3 * mol.n_atoms
        ref = ReferenceData()
        for i in range(6, n3):
            if abs(freqs_qm[i]) > 10.0:
                ref.add_frequency(freqs_qm[i], data_idx=i, weight=1.0, molecule_idx=0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj)

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.2f} → {result.final_score:.2f}"
        )

        # Final sorted frequencies should be closer to QM
        final_ff = ff.with_params(result.final_params)
        final_freqs = engine.frequencies(mol, final_ff)
        initial_freqs = engine.frequencies(mol, ff.with_params(result.initial_params))

        qm_real = [freqs_qm[i] for i in range(6, n3) if abs(freqs_qm[i]) > 10.0]
        final_real = [final_freqs[i] for i in range(6, n3) if abs(freqs_qm[i]) > 10.0]
        initial_real = [initial_freqs[i] for i in range(6, n3) if abs(freqs_qm[i]) > 10.0]

        final_rmse = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(qm_real, final_real)]))
        initial_rmse = np.sqrt(np.mean([(a - b) ** 2 for a, b in zip(qm_real, initial_real)]))
        assert final_rmse < initial_rmse, (
            f"Final RMSE ({final_rmse:.1f} cm⁻¹) should be less than initial ({initial_rmse:.1f} cm⁻¹)"
        )


class TestJaxOptBoundsActive:
    """L-BFGS-B active constraint enforcement."""

    def test_bounds_active_constraint(self) -> None:
        """When unconstrained optimum is outside bounds, final params land on bound."""
        from dataclasses import replace

        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        # Water at bond_length=0.96. Energy minimum wants bond_r0≈0.96.
        # We start at bond_r0=0.88 and constrain to [0.85, 0.90].
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.88, angle_k=49.9, angle_eq=104.5)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()

        # Override bounds: constrain bond_r0 (index 1) to [0.85, 0.90]
        lower = spec.lower_bounds.copy()
        upper = spec.upper_bounds.copy()
        lower[1] = 0.85
        upper[1] = 0.90
        spec = replace(spec, lower_bounds=lower, upper_bounds=upper)

        # Build JaxLoss + optimizer manually with custom spec
        from q2mm.optimizers.jaxloss import JaxLoss

        jax_loss = JaxLoss(spec, engine, [mol], ff)

        from q2mm.backends.mm._jax_common import ensure_jaxopt, jnp

        ensure_jaxopt()
        import jaxopt

        solver = jaxopt.LBFGSB(fun=jax_loss._loss_fn, maxiter=200, tol=1e-6)
        params = jnp.array(ff.get_param_vector(), dtype=jnp.float64)
        lower_jnp = jnp.array(lower, dtype=jnp.float64)
        upper_jnp = jnp.array(upper, dtype=jnp.float64)
        result_params, _state = solver.run(params, bounds=(lower_jnp, upper_jnp))

        final_params = np.asarray(result_params, dtype=float)
        bond_r0_final = final_params[1]

        # The unconstrained optimum (0.96) is above the upper bound (0.90),
        # so the optimizer should push bond_r0 to the upper bound.
        np.testing.assert_allclose(
            bond_r0_final, 0.90, atol=0.01, err_msg=(f"bond_r0 ({bond_r0_final:.4f}) should be near upper bound 0.90")
        )
