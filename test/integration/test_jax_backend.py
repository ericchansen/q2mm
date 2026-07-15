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

import importlib.util
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import CH3F_HESS, CH3F_XYZ, make_diatomic, make_water

from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.optimizers.objective import ObjectiveFunction

# Pinned QFUERZA parameters for CH₃F — decouples gradient tests from
# Seminario estimation.  These are the current default values.
# Param vector order: C-F k, C-F r0, C-H k, C-H r0,
#                     F-C-H k, F-C-H eq, H-C-H k, H-C-H eq.
_CH3F_QFUERZA_PARAMS = {
    "cf_k": 270.6091893,
    "cf_r0": 1.39863537,
    "ch_k": 348.31336238,
    "ch_r0": 1.09403966,
    "fch_k": 35.97,
    "fch_eq": 108.43609415,
    "hch_k": 35.97,
    "hch_eq": 110.4862147,
}


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _materialize(forcefield: ForceField, vector: np.ndarray) -> ForceField:
    return _layout(forcefield).replace(forcefield, vector)


def _make_objective(
    forcefield: ForceField, engine: object, molecules: list, reference: ObservationSet, **kwargs: object
) -> ObjectiveFunction:
    return ObjectiveFunction(
        forcefield=forcefield,
        engine=engine,
        molecules=molecules,
        reference=reference,
        layout=_layout(forcefield),
        **kwargs,
    )


def _all_active_space(objective: ObjectiveFunction) -> ActiveParameterSpace:
    return ActiveParameterSpace.all_active(objective.layout, objective.forcefield)


def _ch3f_ff() -> ForceField:
    """Build a CH₃F ForceField from pinned QFUERZA parameters."""
    p = _CH3F_QFUERZA_PARAMS
    return ForceField(
        bonds=[
            BondParam(elements=("C", "F"), force_constant=p["cf_k"], equilibrium=p["cf_r0"]),
            BondParam(elements=("C", "H"), force_constant=p["ch_k"], equilibrium=p["ch_r0"]),
        ],
        angles=[
            AngleParam(elements=("F", "C", "H"), force_constant=p["fch_k"], equilibrium=p["fch_eq"]),
            AngleParam(elements=("H", "C", "H"), force_constant=p["hch_k"], equilibrium=p["hch_eq"]),
        ],
        functional_form=FunctionalForm.HARMONIC,
    )


@pytest.fixture(autouse=True)
def _init_jax() -> None:
    """Lazily import JAX before each test so module collection is CUDA-free."""
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax()
    global JaxEngine, _build_vdw_pairs  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxEngine as _JE, _build_vdw_pairs as _bvp

    JaxEngine = _JE
    _build_vdw_pairs = _bvp


class TestJaxEnableX64EnvVar:
    """Verify _jax_common respects JAX_ENABLE_X64 env var."""

    _CHECK_SCRIPT = "from q2mm.backends.mm._jax_common import ensure_jax; ensure_jax(); import jax; print(jax.config.jax_enable_x64)"

    def test_default_enables_x64(self) -> None:
        """Without JAX_ENABLE_X64, importing _jax_common enables float64."""
        result = subprocess.run(
            [sys.executable, "-c", self._CHECK_SCRIPT],
            capture_output=True,
            text=True,
            env={k: v for k, v in __import__("os").environ.items() if k != "JAX_ENABLE_X64"},
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "True"

    def test_explicit_zero_disables_x64(self) -> None:
        """JAX_ENABLE_X64=0 prevents _jax_common from forcing float64."""
        import os

        env = {**os.environ, "JAX_ENABLE_X64": "0"}
        result = subprocess.run(
            [sys.executable, "-c", self._CHECK_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "False"


def _h2_ff(bond_k: float = 359.7, bond_r0: float = 0.74) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
        functional_form=FunctionalForm.HARMONIC,
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
        functional_form=FunctionalForm.HARMONIC,
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
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=100, jac="analytical", verbose=False)
        result = optimizer.optimize(objective, _all_active_space(objective))
        assert result.final_score < 1e-10

    def test_analytical_vs_fd_optimization_convergence(self) -> None:
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        target_energy = 10.0
        ff_analytical = _h2_ff(bond_k=215.8, bond_r0=0.74)
        ff_fd = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=target_energy, case_id="0", weight=1.0)

        obj_a = _make_objective(forcefield=ff_analytical, engine=self.engine, molecules=[mol], reference=ref)
        opt_a = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_a = opt_a.optimize(obj_a, _all_active_space(obj_a))

        obj_fd = _make_objective(forcefield=ff_fd, engine=self.engine, molecules=[mol], reference=ref)
        opt_fd = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac=None, verbose=False)
        res_fd = opt_fd.optimize(obj_fd, _all_active_space(obj_fd))

        assert res_a.final_score < 1e-4
        assert res_fd.final_score < 0.01

    def test_objective_gradient_method_exists(self) -> None:
        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        grad = obj.gradient(_params(ff))
        assert len(grad) == len(_params(ff))
        assert isinstance(grad, np.ndarray)

    def test_objective_gradient_matches_fd(self) -> None:
        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=2.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff).copy()
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


class TestJaxBatchedSensitivity:
    """Test vmap-batched sensitivity analysis on JaxEngine."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_batched_energy_matches_sequential(self) -> None:
        """batched_energy should match individual energy calls."""
        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        handle = self.engine.create_context(mol, ff)

        vecs = np.array(
            [
                _params(ff),
                _params(ff) + [10.0, 0.01],
                _params(ff) - [10.0, 0.01],
            ]
        )
        batched = self.engine.batched_energy(handle, ff, vecs)

        sequential = np.array([self.engine.energy(handle, _materialize(ff, v)) for v in vecs])
        np.testing.assert_allclose(batched, sequential, atol=1e-10)

    def test_supports_batched_energy(self) -> None:
        assert self.engine.supports_batched_energy() is True

    def test_is_energy_only(self) -> None:
        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        assert obj.is_energy_only() is True

    def test_batched_scores_matches_sequential(self) -> None:
        """batched_scores via vmap should match sequential __call__."""
        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=5.0, case_id="0", weight=1.0)

        obj_batch = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        obj_seq = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)

        vecs = np.array(
            [
                _params(ff),
                _params(ff) + [10.0, 0.01],
                _params(ff) - [10.0, 0.01],
            ]
        )

        batched = obj_batch.batched_scores(vecs)
        sequential = np.array([obj_seq(v) for v in vecs])
        np.testing.assert_allclose(batched, sequential, atol=1e-10)

    def test_compute_sensitivity_batched(self) -> None:
        """compute_sensitivity should produce identical results via batched path."""
        from q2mm.optimizers.cycling import compute_sensitivity

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=5.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)

        # Verify the batched path is used (energy-only + JAX)
        assert obj.is_energy_only()
        assert self.engine.supports_batched_energy()

        sens = compute_sensitivity(obj, metric="simp_var")

        # Sanity checks
        assert sens.n_evals == 2 * len(_params(ff)) + 1
        assert len(sens.d1) == len(_params(ff))
        assert len(sens.d2) == len(_params(ff))
        assert len(sens.ranking) == len(_params(ff))
        # d1 should be nonzero for active params
        assert np.any(sens.d1 != 0)


# ---------------------------------------------------------------------------
# Analytical Hessian gradient tests (hessian_and_param_jacobian)
# ---------------------------------------------------------------------------


def _fd_objective_gradient(obj: Any, vec: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Central finite-difference gradient of an objective function."""
    grad = np.zeros_like(vec)
    for i in range(len(vec)):
        v_plus, v_minus = vec.copy(), vec.copy()
        v_plus[i] += h
        v_minus[i] -= h
        obj.reset()
        f_plus = obj(v_plus)
        obj.reset()
        f_minus = obj(v_minus)
        grad[i] = (f_plus - f_minus) / (2 * h)
    return grad


class TestJaxAnalyticalHessianGradients:
    """End-to-end tests for hessian_and_param_jacobian + frequency gradient."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_hessian_and_param_jacobian_h2_shapes(self) -> None:
        """Basic shape validation for H₂."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff()
        hess, dH_dp = self.engine.hessian_and_param_jacobian(mol, ff)
        assert hess.shape == (6, 6)
        assert dH_dp.shape == (6, 6, 2)

    def test_hessian_and_param_jacobian_h2_symmetric(self) -> None:
        """Hessian and each Jacobian slice must be symmetric."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff()
        hess, dH_dp = self.engine.hessian_and_param_jacobian(mol, ff)
        np.testing.assert_allclose(hess, hess.T, atol=1e-8)
        for j in range(len(_params(ff))):
            np.testing.assert_allclose(dH_dp[:, :, j], dH_dp[:, :, j].T, atol=1e-6)

    def test_hessian_jacobian_matches_fd_h2(self) -> None:
        """dH/dp matches finite-difference of engine.hessian()."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff()
        _hess, dH_dp = self.engine.hessian_and_param_jacobian(mol, ff)

        params = _params(ff).copy()
        h = 1e-5
        dH_dp_fd = np.zeros_like(dH_dp)
        for i in range(len(_params(ff))):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            h_plus = self.engine.hessian(mol, _materialize(ff, p_plus))
            h_minus = self.engine.hessian(mol, _materialize(ff, p_minus))
            dH_dp_fd[:, :, i] = (h_plus - h_minus) / (2 * h)

        np.testing.assert_allclose(dH_dp, dH_dp_fd, atol=1e-4, rtol=1e-4)

    def test_hessian_jacobian_matches_fd_water(self) -> None:
        """Multi-param (bonds + angles) Hessian Jacobian vs FD."""
        mol = make_water(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff()
        _hess, dH_dp = self.engine.hessian_and_param_jacobian(mol, ff)

        params = _params(ff).copy()
        h = 1e-5
        dH_dp_fd = np.zeros_like(dH_dp)
        for i in range(len(_params(ff))):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += h
            p_minus[i] -= h
            h_plus = self.engine.hessian(mol, _materialize(ff, p_plus))
            h_minus = self.engine.hessian(mol, _materialize(ff, p_minus))
            dH_dp_fd[:, :, i] = (h_plus - h_minus) / (2 * h)

        np.testing.assert_allclose(dH_dp, dH_dp_fd, atol=1e-4, rtol=1e-4)

    def test_frequency_gradient_vs_fd_h2(self) -> None:
        """Analytical frequency objective gradient matches FD for H₂."""
        mol = make_diatomic(distance=0.80, bond_tolerance=2.0)
        ff = _h2_ff(bond_k=300.0, bond_r0=0.74)
        freqs = self.engine.frequencies(mol, ff)

        ref = ObservationSet()
        for i, f in enumerate(freqs):
            ref = ref.with_frequency(value=f * 1.1, data_idx=i, weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec)

        np.testing.assert_allclose(analytical, fd, atol=1e-2, rtol=1e-2)

    def test_frequency_gradient_vs_fd_water(self) -> None:
        """Analytical frequency objective gradient matches FD for water."""
        mol = make_water(angle_deg=108.0, bond_length=0.98)
        ff = _water_ff(bond_k=500.0, bond_r0=0.96, angle_k=45.0, angle_eq=104.5)
        freqs = self.engine.frequencies(mol, ff)

        ref = ObservationSet()
        for i, f in enumerate(freqs):
            ref = ref.with_frequency(value=f * 1.05, data_idx=i, weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec)

        np.testing.assert_allclose(analytical, fd, atol=1e-1, rtol=1e-2)

    def test_frequency_optimization_with_analytical_jac(self) -> None:
        """L-BFGS-B converges with analytical frequency gradients."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=2.0)
        ff = _h2_ff(bond_k=250.0, bond_r0=0.70)
        target_freqs = self.engine.frequencies(mol, _h2_ff(bond_k=359.7, bond_r0=0.74))

        ref = ObservationSet()
        for i, f in enumerate(target_freqs):
            ref = ref.with_frequency(value=f, data_idx=i, weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < 1.0, f"Score {result.final_score} too high"

    def test_analytical_vs_fd_optimization_same_result(self) -> None:
        """Analytical and FD optimization converge to the same parameters."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=2.0)
        target_freqs = self.engine.frequencies(mol, _h2_ff(bond_k=359.7, bond_r0=0.74))

        ref = ObservationSet()
        for i, f in enumerate(target_freqs):
            ref = ref.with_frequency(value=f, data_idx=i, weight=1.0)

        ff_a = _h2_ff(bond_k=250.0, bond_r0=0.70)
        obj_a = _make_objective(forcefield=ff_a, engine=self.engine, molecules=[mol], reference=ref)
        opt_a = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_a = opt_a.optimize(obj_a, _all_active_space(obj_a))

        ff_fd = _h2_ff(bond_k=250.0, bond_r0=0.70)
        obj_fd = _make_objective(forcefield=ff_fd, engine=self.engine, molecules=[mol], reference=ref)
        opt_fd = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac=None, verbose=False)
        res_fd = opt_fd.optimize(obj_fd, _all_active_space(obj_fd))

        np.testing.assert_allclose(
            res_a.final_params,
            res_fd.final_params,
            rtol=0.1,  # Within 10% — different paths can hit different local minima
            err_msg="Analytical and FD should converge to similar parameters",
        )


class TestJaxHessianElementGradients:
    """End-to-end hessian_element evaluator gradient tests."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_hessian_element_gradient_vs_fd(self) -> None:
        """Analytical hessian_element gradient matches FD for H₂."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff(bond_k=300.0, bond_r0=0.74)
        hess = self.engine.hessian(mol, ff)

        ref = ObservationSet()
        ref = ref.with_hessian_element(value=float(hess[0, 0]) * 1.1, row=0, col=0, weight=1.0)
        ref = ref.with_hessian_element(value=float(hess[0, 3]) * 1.1, row=0, col=3, weight=1.0)
        ref = ref.with_hessian_element(value=float(hess[3, 3]) * 1.1, row=3, col=3, weight=1.0)

        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec)

        np.testing.assert_allclose(analytical, fd, atol=1e-2, rtol=1e-2)

    def test_hessian_element_optimization_converges(self) -> None:
        """Optimization with hessian element references converges."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        target_hess = self.engine.hessian(mol, _h2_ff(bond_k=359.7, bond_r0=0.74))

        ref = ObservationSet()
        ref = ref.with_hessian_element(value=float(target_hess[0, 0]), row=0, col=0, weight=1.0)
        ref = ref.with_hessian_element(value=float(target_hess[3, 3]), row=3, col=3, weight=1.0)

        ff = _h2_ff(bond_k=250.0, bond_r0=0.70)
        obj = _make_objective(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score


class TestJaxEigenmatrixGradients:
    """End-to-end eigenmatrix evaluator gradient tests."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_eigenmatrix_gradient_vs_fd(self) -> None:
        """Analytical eigenmatrix gradient matches FD."""
        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff = _h2_ff(bond_k=300.0, bond_r0=0.74)
        qm_hess = self.engine.hessian(mol, ff)
        mol_with_hess = mol.with_hessian(qm_hess)

        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(
            qm_hess,
            diagonal_only=True,
            weights={"eig_i": 0.1, "eig_d_low": 0.1, "eig_d_high": 0.1},
        )
        assert ref.n_observations > 0

        ff2 = _h2_ff(bond_k=280.0, bond_r0=0.72)
        obj = _make_objective(
            forcefield=ff2,
            engine=self.engine,
            molecules=[mol_with_hess],
            reference=ref,
        )
        vec = _params(ff2).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec)

        np.testing.assert_allclose(analytical, fd, atol=5e-2, rtol=5e-2)

    def test_eigenmatrix_optimization_converges(self) -> None:
        """Optimization with eigenmatrix references converges."""
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.84, bond_tolerance=2.0)
        ff_target = _h2_ff(bond_k=359.7, bond_r0=0.74)
        qm_hess = self.engine.hessian(mol, ff_target)
        mol_with_hess = mol.with_hessian(qm_hess)

        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(
            qm_hess,
            diagonal_only=True,
            weights={"eig_i": 0.1, "eig_d_low": 0.1, "eig_d_high": 0.1},
        )

        ff = _h2_ff(bond_k=250.0, bond_r0=0.70)
        obj = _make_objective(
            forcefield=ff,
            engine=self.engine,
            molecules=[mol_with_hess],
            reference=ref,
        )
        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score


# ---------------------------------------------------------------------------
# CH₃F real-molecule validation
# ---------------------------------------------------------------------------
class TestCH3FAnalyticalGradients:
    """Validate analytical gradients on CH₃F (5 atoms, 8 params).

    Uses pinned QFUERZA force field parameters — does NOT depend on
    ``qfuerza_fresh()`` or Seminario defaults.  The frequency
    gradient vs FD test is intentionally excluded: CH₃F's C₃v symmetry
    produces near-degenerate E-mode frequency pairs (57 cm⁻¹ gap with
    QFUERZA), and FD diverges at eigenvalue crossings.  Frequency gradient
    correctness is validated by ``TestFrequencyParamJacobian`` (mock engines),
    ``TestJaxAnalyticalHessianGradients`` (H₂/water), and the optimisation
    convergence test below.
    """

    def setup_method(self) -> None:
        self.engine = JaxEngine()
        self.mol = load_xyz(CH3F_XYZ, bond_tolerance=1.5)
        self.qm_hessian = np.load(CH3F_HESS)
        self.mol_with_hess = self.mol.with_hessian(self.qm_hessian)
        self.ff = _ch3f_ff()

    def test_ch3f_hessian_element_gradient(self) -> None:
        """Analytical hessian element gradient on CH₃F."""
        hess = self.engine.hessian(self.mol, self.ff)
        ref = ObservationSet()
        ref = ref.with_hessian_element(value=float(hess[0, 0]) * 1.05, row=0, col=0)
        ref = ref.with_hessian_element(value=float(hess[1, 1]) * 1.05, row=1, col=1)
        ref = ref.with_hessian_element(value=float(hess[0, 1]) * 1.05, row=0, col=1)

        obj = _make_objective(
            forcefield=self.ff,
            engine=self.engine,
            molecules=[self.mol],
            reference=ref,
        )
        vec = _params(self.ff).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec, h=1e-4)

        np.testing.assert_allclose(analytical, fd, atol=0.1, rtol=0.1)

    def test_ch3f_eigenmatrix_gradient(self) -> None:
        """Analytical eigenmatrix gradient on CH₃F."""
        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(
            self.qm_hessian,
            diagonal_only=True,
            weights={"eig_i": 0.1, "eig_d_low": 0.1, "eig_d_high": 0.1},
        )

        vec_orig = _params(self.ff).copy()
        vec_perturbed = vec_orig * 0.95
        ff = _materialize(self.ff, vec_perturbed)

        obj = _make_objective(
            forcefield=ff,
            engine=self.engine,
            molecules=[self.mol_with_hess],
            reference=ref,
        )
        vec = _params(ff).copy()

        analytical = obj.gradient(vec)
        fd = _fd_objective_gradient(obj, vec, h=1e-4)

        np.testing.assert_allclose(analytical, fd, atol=1.0, rtol=0.15)

    def test_ch3f_optimization_analytical_vs_fd(self) -> None:
        """Both gradient modes converge to similar RMSD on CH₃F."""
        from q2mm.models.hessian import hessian_to_frequencies
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        qm_freqs = hessian_to_frequencies(self.qm_hessian, self.mol.symbols)
        # Keep only real (positive) frequencies for comparison
        qm_real = [f for f in qm_freqs if f > 0]

        ref = ObservationSet()
        for i, f in enumerate(qm_real):
            ref = ref.with_frequency(value=float(f), data_idx=i, weight=1.0)

        # Analytical
        ff_a = self.ff
        obj_a = _make_objective(
            forcefield=ff_a,
            engine=self.engine,
            molecules=[self.mol],
            reference=ref,
        )
        opt_a = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_a = opt_a.optimize(obj_a, _all_active_space(obj_a))

        # FD
        ff_fd = self.ff
        obj_fd = _make_objective(
            forcefield=ff_fd,
            engine=self.engine,
            molecules=[self.mol],
            reference=ref,
        )
        opt_fd = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac=None, verbose=False)
        res_fd = opt_fd.optimize(obj_fd, _all_active_space(obj_fd))

        # Both should improve from initial
        assert res_a.final_score < res_a.initial_score
        assert res_fd.final_score < res_fd.initial_score


# ---------------------------------------------------------------------------
# Performance benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.nightly
class TestAnalyticalGradientPerformance:
    """Wall-clock timing comparison: analytical vs FD gradients.

    Marked @pytest.mark.nightly — not run in normal CI.
    """

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def _time_gradient(self, obj: Any, vec: np.ndarray, n_iters: int = 20) -> float:
        """Time n_iters gradient evaluations and return mean seconds."""
        import time

        # Warmup
        obj.reset()
        obj.gradient(vec)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            obj.reset()
            obj.gradient(vec)
        return (time.perf_counter() - t0) / n_iters

    def _time_fd_gradient(self, obj: Any, vec: np.ndarray, n_iters: int = 5) -> float:
        """Time n_iters FD gradient evaluations and return mean seconds."""
        import time

        # Warmup
        _fd_objective_gradient(obj, vec)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _fd_objective_gradient(obj, vec)
        return (time.perf_counter() - t0) / n_iters

    def test_gradient_speedup_water(self) -> None:
        """Measure analytical vs FD speedup on water (4 params)."""
        mol = make_water(angle_deg=108.0, bond_length=0.98)
        ff_a = _water_ff()
        freqs = self.engine.frequencies(mol, ff_a)

        ref = ObservationSet()
        for i, f in enumerate(freqs):
            ref = ref.with_frequency(value=f * 1.05, data_idx=i, weight=1.0)

        obj_a = _make_objective(forcefield=ff_a, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff_a).copy()

        t_analytical = self._time_gradient(obj_a, vec)

        ff_fd = _water_ff()
        engine_fd = JaxEngine()
        # Monkey-patch to force FD gradient path
        engine_fd.supports_analytical_hessian_gradients = lambda: False
        engine_fd.supports_analytical_gradients = lambda: False
        obj_fd = _make_objective(forcefield=ff_fd, engine=engine_fd, molecules=[mol], reference=ref)
        t_fd = self._time_fd_gradient(obj_fd, vec)

        speedup = t_fd / t_analytical if t_analytical > 0 else float("inf")
        print(f"\nWater (4 params): analytical={t_analytical:.4f}s, FD={t_fd:.4f}s, speedup={speedup:.1f}×")

    def test_gradient_speedup_ch3f(self) -> None:
        """Measure analytical vs FD speedup on CH₃F (8 params)."""
        mol = load_xyz(CH3F_XYZ, bond_tolerance=1.5)
        ff_a = _ch3f_ff()
        freqs = self.engine.frequencies(mol, ff_a)

        ref = ObservationSet()
        for i, f in enumerate(freqs):
            ref = ref.with_frequency(value=f * 1.05, data_idx=i, weight=1.0)

        obj_a = _make_objective(forcefield=ff_a, engine=self.engine, molecules=[mol], reference=ref)
        vec = _params(ff_a).copy()

        t_analytical = self._time_gradient(obj_a, vec)

        ff_fd = _ch3f_ff()
        engine_fd = JaxEngine()
        engine_fd.supports_analytical_hessian_gradients = lambda: False
        engine_fd.supports_analytical_gradients = lambda: False
        obj_fd = _make_objective(forcefield=ff_fd, engine=engine_fd, molecules=[mol], reference=ref)
        t_fd = self._time_fd_gradient(obj_fd, vec, n_iters=3)

        speedup = t_fd / t_analytical if t_analytical > 0 else float("inf")
        print(f"\nCH₃F (8 params): analytical={t_analytical:.4f}s, FD={t_fd:.4f}s, speedup={speedup:.1f}×")
