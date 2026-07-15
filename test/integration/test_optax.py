"""Integration tests for OptaxOptimizer with JAX backend.

Verifies that OptaxOptimizer converges on real force field problems
using the JAX backend's analytical gradients.
"""

from __future__ import annotations
from q2mm.backends.registry import load_backend

import importlib.util

import numpy as np
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

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.optimizers.objective import ObjectiveFunction

# Module-level globals populated by autouse fixture
JaxBackend = None


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _make_objective(
    forcefield: ForceField, backend: object, molecules: list, reference: ObservationSet, **kwargs: object
) -> ObjectiveFunction:
    return ObjectiveFunction(
        forcefield=forcefield,
        backend=backend,
        molecules=molecules,
        reference=reference,
        layout=_layout(forcefield),
        **kwargs,
    )


def _all_active_space(objective: ObjectiveFunction) -> ActiveParameterSpace:
    return ActiveParameterSpace.all_active(objective.layout, objective.forcefield)


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


@pytest.fixture(autouse=True)
def _init_jax() -> None:
    """Import JAX lazily so module collection is CUDA-free."""
    from q2mm.backends.mm._jax_common import ensure_jax

    ensure_jax()
    global JaxBackend  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxBackend as _JE

    JaxBackend = _JE


class TestOptaxJaxDiatomic:
    """OptaxOptimizer on a simple H₂ diatomic with JAX backend."""

    def setup_method(self) -> None:
        self.backend = load_backend("jax")

    def test_adam_converges(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, backend=self.backend, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=200,
            verbose=False,
        )
        result = optimizer.optimize(objective, _all_active_space(objective))

        assert result.final_score < 1e-4
        assert result.method == "optax:adam"

    def test_sgd_converges(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, backend=self.backend, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="sgd",
            learning_rate=0.05,
            momentum=0.9,
            max_steps=300,
            verbose=False,
        )
        result = optimizer.optimize(objective, _all_active_space(objective))

        assert result.final_score < 0.1

    def test_cosine_schedule(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=5.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, backend=self.backend, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=200,
            schedule="cosine",
            verbose=False,
        )
        result = optimizer.optimize(objective, _all_active_space(objective))

        assert result.final_score < result.initial_score
        assert "cosine" in result.method


class TestOptaxJaxWater:
    """OptaxOptimizer on H₂O (bond + angle params) with JAX backend."""

    def setup_method(self) -> None:
        self.backend = load_backend("jax")

    def test_adam_improves_water(self) -> None:
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_water()
        ff = _water_ff(bond_k=400.0, bond_r0=1.0, angle_k=40.0, angle_eq=109.5)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, backend=self.backend, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.05,
            max_steps=300,
            verbose=False,
        )
        result = optimizer.optimize(objective, _all_active_space(objective))

        assert result.final_score < result.initial_score
        assert result.improvement > 0.0


class TestOptaxVsScipyBaseline:
    """Compare OptaxOptimizer against ScipyOptimizer L-BFGS-B."""

    def setup_method(self) -> None:
        self.backend = load_backend("jax")

    def test_adam_comparable_to_lbfgsb(self) -> None:
        """Adam should achieve comparable (if not better) results to L-BFGS-B."""
        from q2mm.optimizers.optax import OptaxOptimizer
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.80, bond_tolerance=1.5)
        target_energy = 5.0

        # Run L-BFGS-B
        ff_scipy = _h2_ff(bond_k=215.8, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=target_energy, case_id="0", weight=1.0)

        obj_scipy = _make_objective(forcefield=ff_scipy, backend=self.backend, molecules=[mol], reference=ref)
        opt_scipy = ScipyOptimizer(method="L-BFGS-B", maxiter=200, jac="analytical", verbose=False)
        res_scipy = opt_scipy.optimize(obj_scipy, _all_active_space(obj_scipy))

        # Run Adam
        ff_optax = _h2_ff(bond_k=215.8, bond_r0=0.74)
        obj_optax = _make_objective(forcefield=ff_optax, backend=self.backend, molecules=[mol], reference=ref)
        opt_optax = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.1,
            max_steps=500,
            verbose=False,
        )
        res_optax = opt_optax.optimize(obj_optax, _all_active_space(obj_optax))

        # Both should converge well on this simple problem
        assert res_scipy.final_score < 1.0
        assert res_optax.final_score < 1.0

    def test_gradient_support_detected(self) -> None:
        """OptaxOptimizer should detect analytical gradient support."""
        from q2mm.optimizers.optax import OptaxOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        objective = _make_objective(forcefield=ff, backend=self.backend, molecules=[mol], reference=ref)
        optimizer = OptaxOptimizer(
            optimizer="adam",
            learning_rate=0.01,
            max_steps=10,
            verbose=False,
        )
        result = optimizer.optimize(objective, _all_active_space(objective))

        # JAX backend → uses JaxLoss gradient path (memory-efficient)
        assert result.jac_mode == "jax_loss"
