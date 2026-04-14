"""Integration tests for OptaxOptimizer with JAX backend.

Verifies that OptaxOptimizer converges on real force field problems
using the JAX engine's analytical gradients.
"""

from __future__ import annotations

import importlib.util

import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_OPTAX = importlib.util.find_spec("optax") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.skipif(not _HAS_OPTAX, reason="optax not installed"),
    pytest.mark.jax,
    pytest.mark.integration,
]

from test._shared import make_diatomic, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField

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
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax()
    global JaxEngine  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxEngine as _JE

    JaxEngine = _JE


class TestOptaxJaxDiatomic:
    """OptaxOptimizer on a simple H₂ diatomic with JAX engine."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_adam_converges(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=200,
            verbose=False,
        )
        result = optimizer.optimize(objective)

        assert result.final_score < 1e-4
        assert result.method == "optax:adam"

    def test_sgd_converges(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="sgd",
            learning_rate=0.05,
            momentum=0.9,
            max_steps=300,
            verbose=False,
        )
        result = optimizer.optimize(objective)

        assert result.final_score < 0.1

    def test_cosine_schedule(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=5.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=200,
            schedule="cosine",
            verbose=False,
        )
        result = optimizer.optimize(objective)

        assert result.final_score < result.initial_score
        assert "cosine" in result.method


class TestOptaxJaxWater:
    """OptaxOptimizer on H₂O (bond + angle params) with JAX engine."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_adam_improves_water(self) -> None:
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_water()
        ff = _water_ff(bond_k=400.0, bond_r0=1.0, angle_k=40.0, angle_eq=109.5)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.05,
            max_steps=300,
            verbose=False,
        )
        result = optimizer.optimize(objective)

        assert result.final_score < result.initial_score
        assert result.improvement > 0.0


class TestOptaxVsScipyBaseline:
    """Compare OptaxOptimizer against ScipyOptimizer L-BFGS-B."""

    def setup_method(self) -> None:
        self.engine = JaxEngine()

    def test_adam_comparable_to_lbfgsb(self) -> None:
        """Adam should achieve comparable (if not better) results to L-BFGS-B."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        target_energy = 5.0

        # Run L-BFGS-B
        ff_scipy = _h2_ff(bond_k=215.8, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=target_energy, molecule_idx=0, weight=1.0)

        obj_scipy = ObjectiveFunction(forcefield=ff_scipy, engine=self.engine, molecules=[mol], reference=ref)
        opt_scipy = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_scipy = opt_scipy.optimize(obj_scipy)

        # Run Adam
        ff_optax = _h2_ff(bond_k=215.8, bond_r0=0.74)
        obj_optax = ObjectiveFunction(forcefield=ff_optax, engine=self.engine, molecules=[mol], reference=ref)
        opt_optax = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            verbose=False,
        )
        res_optax = opt_optax.optimize(obj_optax)

        # Both should converge well on this simple problem
        assert res_scipy.final_score < 1.0
        assert res_optax.final_score < 1.0

    def test_gradient_support_detected(self) -> None:
        """OptaxOptimizer should detect analytical gradient support."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        objective = ObjectiveFunction(forcefield=ff, engine=self.engine, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=10,
            verbose=False,
        )
        result = optimizer.optimize(objective)

        # JAX engine → full analytical support
        assert result.jac_mode == "analytical"
