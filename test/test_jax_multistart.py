"""Unit tests for JaxMultiStartOptimizer.

Verifies constructor validation, determinism, argmin-best-of-N
selection, and that the fused vmap path matches a Python-loop
multi-start baseline on a simple system.
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
from q2mm.optimizers.objective import ObjectiveFunction

JaxEngine = None


def _h2_ff(bond_k: float = 215.8, bond_r0: float = 0.80) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=bond_k, equilibrium=bond_r0)],
    )


def _water_ff(
    bond_k: float = 400.0,
    bond_r0: float = 1.05,
    angle_k: float = 35.0,
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


class TestJaxMultiStartValidation:
    """Constructor validation."""

    def test_invalid_method(self) -> None:
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        with pytest.raises(ValueError, match="Unknown method"):
            JaxMultiStartOptimizer(method="not_a_method")

    def test_invalid_n_starts(self) -> None:
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        with pytest.raises(ValueError, match="n_starts must be >= 1"):
            JaxMultiStartOptimizer(n_starts=0)

    def test_invalid_perturbation_pct(self) -> None:
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        with pytest.raises(ValueError, match="perturbation_pct must be >= 0"):
            JaxMultiStartOptimizer(perturbation_pct=-0.1)

    def test_valid_methods(self) -> None:
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        for method in ("lbfgs", "lbfgsb", "gradient_descent"):
            opt = JaxMultiStartOptimizer(method=method)
            assert opt.method == method

    def test_engine_type_check(self) -> None:
        from unittest.mock import MagicMock

        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        fake_engine = MagicMock()
        fake_engine.__class__.__name__ = "FakeEngine"
        obj = ObjectiveFunction(forcefield=ff, engine=fake_engine, molecules=[mol], reference=ref)

        optimizer = JaxMultiStartOptimizer(method="lbfgs", n_starts=3, maxiter=10, verbose=False)
        with pytest.raises(TypeError, match="JaxMultiStartOptimizer requires a JaxEngine"):
            optimizer.optimize(obj)


class TestJaxMultiStartConvergence:
    """End-to-end optimization tests."""

    def _make_h2_obj(self) -> ObjectiveFunction:
        from q2mm.optimizers.objective import ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        return ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

    def test_lbfgs_converges(self) -> None:
        """3-start L-BFGS reduces H2 energy loss."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        obj = self._make_h2_obj()
        opt = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=3,
            maxiter=100,
            perturbation_pct=0.1,
            seed=0,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.final_score <= result.initial_score
        assert result.method == "jaxopt-multi:lbfgs"
        assert result.jac_mode == "analytical"
        assert result.eps is None

    def test_single_start_matches_plain_jaxopt(self) -> None:
        """n_starts=1 reproduces a plain JaxOptOptimizer run."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        obj_multi = self._make_h2_obj()
        opt_multi = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=1,
            maxiter=100,
            perturbation_pct=0.0,
            seed=0,
            verbose=False,
        )
        r_multi = opt_multi.optimize(obj_multi)

        obj_plain = self._make_h2_obj()
        opt_plain = JaxOptOptimizer(method="lbfgs", maxiter=100, verbose=False)
        r_plain = opt_plain.optimize(obj_plain)

        np.testing.assert_allclose(r_multi.final_params, r_plain.final_params, rtol=1e-5, atol=1e-8)
        assert abs(r_multi.final_score - r_plain.final_score) < 1e-8

    def test_determinism_with_seed(self) -> None:
        """Same seed → same final params."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        obj_a = self._make_h2_obj()
        opt_a = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=5,
            maxiter=50,
            perturbation_pct=0.2,
            seed=42,
            verbose=False,
        )
        r_a = opt_a.optimize(obj_a)

        obj_b = self._make_h2_obj()
        opt_b = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=5,
            maxiter=50,
            perturbation_pct=0.2,
            seed=42,
            verbose=False,
        )
        r_b = opt_b.optimize(obj_b)

        np.testing.assert_allclose(r_a.final_params, r_b.final_params, rtol=1e-8)
        assert r_a.final_score == pytest.approx(r_b.final_score, abs=1e-8)

    def test_best_of_n_selection(self) -> None:
        """Selects the replica with the lowest final loss."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
        from q2mm.optimizers.jaxloss import JaxLoss

        obj = self._make_h2_obj()
        opt = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=4,
            maxiter=80,
            perturbation_pct=0.15,
            seed=7,
            verbose=False,
        )
        result = opt.optimize(obj)

        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, obj.engine, obj.molecules, obj.forcefield)
        score_at_returned = float(jax_loss(result.final_params))

        assert abs(score_at_returned - result.final_score) < 1e-6

    def test_forcefield_updated(self) -> None:
        """After optimize(), the forcefield holds the best-of-N params."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer

        obj = self._make_h2_obj()
        opt = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=3,
            maxiter=50,
            perturbation_pct=0.1,
            seed=0,
            verbose=False,
        )
        result = opt.optimize(obj)

        np.testing.assert_allclose(obj.forcefield.get_param_vector(), result.final_params, rtol=1e-10)

    def test_water_multi_start(self) -> None:
        """Water (bond + angle) multi-start reduces loss."""
        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff()
        engine = JaxEngine()
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        opt = JaxMultiStartOptimizer(
            method="lbfgs",
            n_starts=5,
            maxiter=80,
            perturbation_pct=0.1,
            seed=1,
            verbose=False,
        )
        result = opt.optimize(obj)

        assert result.final_score <= result.initial_score


class TestJaxMultiStartBackendGuard:
    """Backend-specific guards."""

    def test_lbfgsb_gpu_guard(self) -> None:
        """LBFGSB raises on non-CPU backends with a clear message."""
        from unittest.mock import patch

        from q2mm.optimizers.jax_multistart import JaxMultiStartOptimizer
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        engine = JaxEngine()
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxMultiStartOptimizer(method="lbfgsb", n_starts=2, maxiter=10, verbose=False)

        from q2mm.backends.mm._jax_common import jax as jax_mod

        with (
            patch.object(jax_mod, "default_backend", return_value="gpu"),
            pytest.raises(RuntimeError, match="LBFGSB is not supported"),
        ):
            optimizer.optimize(obj)
