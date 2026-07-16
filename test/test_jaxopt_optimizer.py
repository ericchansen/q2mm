"""Unit tests for JaxOptOptimizer.

Verifies constructor validation, method dispatch, and basic convergence
on simple energy objectives.
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    FrequencyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import param_vector, prepare_case

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

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor

# Module-level globals populated by autouse fixture
JaxBackend = None


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _materialize(forcefield: ForceField, vector: np.ndarray) -> ForceField:
    return _layout(forcefield).replace(forcefield, vector)


def _make_plan(forcefield: ForceField, molecules: list, reference: object, **kwargs: object) -> ObjectivePlan:
    layout = _layout(forcefield)
    mols = tuple(molecules)
    plan = ObjectivePlan(
        case_ids=tuple(str(i) for i in range(len(mols))),
        molecules=mols,
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in mols),
        observations=reference,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, forcefield),
        regularization=float(kwargs.pop("regularization", 0.0)),
        reference_params=kwargs.pop("reference_params", None),
    )
    if kwargs:
        raise TypeError(f"Unsupported objective kwargs: {sorted(kwargs)}")
    return plan


def _make_objective(
    forcefield: ForceField, backend: object, molecules: list, reference: object, **kwargs: object
) -> JaxObjectiveExecutor:
    plan = _make_plan(forcefield, molecules, reference, **kwargs)
    return JaxObjectiveExecutor(plan, backend, forcefield)


def _all_active_space(objective: JaxObjectiveExecutor) -> ActiveParameterSpace:
    return objective.plan.active_space


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
    from q2mm.backends.mm._jax_common import ensure_jax, ensure_jaxopt

    ensure_jax()
    ensure_jaxopt()
    global JaxBackend  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxBackend as _JE

    JaxBackend = _JE


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

    def test_backend_type_check(self) -> None:
        from unittest.mock import MagicMock

        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        fake_backend = MagicMock()
        fake_backend.__class__.__name__ = "FakeBackend"
        plan = _make_plan(ff, [mol], ref)
        obj = PythonObjectiveExecutor(plan, fake_backend, ff)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=10, verbose=False)
        with pytest.raises(TypeError, match="JaxObjectiveExecutor"):
            optimizer.optimize(obj, _all_active_space(obj))


class TestJaxOptOptimizerConvergence:
    """Convergence tests on simple systems."""

    def test_lbfgs_h2_energy(self) -> None:
        """L-BFGS converges on H2 energy optimization."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        # Perturbed r0 so energy at geometry != 0 (gives non-zero initial loss)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score <= result.initial_score
        assert result.method == "jaxopt:lbfgs"
        assert result.gradient_mode == "analytical"
        assert result.fd_step is None

    def test_lbfgsb_h2_energy(self) -> None:
        """L-BFGS-B converges with box constraints."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgsb", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score <= result.initial_score
        assert result.method == "jaxopt:lbfgsb"

    @pytest.mark.nightly
    def test_reported_scores_in_objective_units(self) -> None:
        """F6: reported initial/final scores are in PythonObjectiveExecutor units.

        The internal revert guard uses JaxObjectiveExecutor-unit surrogate scores, but the
        returned ``OptimizationResult`` must report true PythonObjectiveExecutor
        units so cross-stage comparisons in cycling.py compare like-for-like.
        Regression: ``final_score``/``initial_score`` used to leak the
        surrogate scale.
        """
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        true_initial = float(obj.value(np.asarray(result.initial_params)))
        true_final = float(obj.value(np.asarray(result.final_params)))
        assert result.initial_score == pytest.approx(true_initial, rel=1e-6, abs=1e-9)
        assert result.final_score == pytest.approx(true_final, rel=1e-6, abs=1e-9)

    def test_result_format(self) -> None:
        """OptimizationResult has all expected fields."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet
        from q2mm.models.results import OptimizationResult

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=10, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert isinstance(result, OptimizationResult)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.initial_score, float)
        assert isinstance(result.final_score, float)
        assert isinstance(result.n_iterations, int)
        assert isinstance(result.initial_params, np.ndarray)
        assert isinstance(result.final_params, np.ndarray)
        assert isinstance(result.history, tuple)

    def test_water_energy_convergence(self) -> None:
        """Water (bond + angle) energy converges with L-BFGS."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=104.5)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score <= result.initial_score

    def test_forcefield_updated(self) -> None:
        """After optimization, final_params can be materialized into a new force field."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        initial_params = _params(ff).copy()

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        np.testing.assert_array_equal(_params(ff), initial_params)
        optimized_ff = _materialize(ff, result.final_params)
        final_params = _params(optimized_ff)
        np.testing.assert_array_equal(final_params, result.final_params)
        # If the optimizer improved, params should differ from initial
        if result.final_score < result.initial_score:
            assert not np.allclose(final_params, initial_params)

    def test_inactive_param_stays_fixed(self) -> None:
        """Inactive parameters are held constant while active ones optimize."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=110.0)
        initial_params = _params(ff).copy()
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        full_space = _all_active_space(obj)
        active_indices = [slot.index for slot in obj.plan.layout if slot.owner != "bonds"]
        constrained_space = full_space.with_active_indices(active_indices)
        result = optimizer.optimize(obj, constrained_space)

        bond_indices = [slot.index for slot in obj.plan.layout if slot.owner == "bonds"]
        np.testing.assert_allclose(result.final_params[bond_indices], initial_params[bond_indices])
        np.testing.assert_allclose(_params(ff), initial_params)
        assert result.final_score <= result.initial_score


class TestJaxOptFrequencyConvergence:
    """Frequency-based optimization convergence."""

    def test_ch3f_frequency_convergence(self) -> None:
        """L-BFGS converges on CH3F frequency optimization."""
        from q2mm.models.hessian import hessian_to_frequencies
        from q2mm.io.xyz import load_xyz
        from q2mm.models.forcefield import FunctionalForm
        from q2mm.models.seminario import qfuerza_fresh
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer
        from q2mm.models.observations import ObservationSet
        from test._shared import CH3F_HESS, CH3F_XYZ

        mol = load_xyz(CH3F_XYZ)
        hess_qm = np.load(CH3F_HESS)
        mol = mol.with_hessian(hess_qm)

        # Build a real FF and perturb it
        ff = qfuerza_fresh(mol, functional_form=FunctionalForm.HARMONIC)
        freqs_qm = hessian_to_frequencies(hess_qm, list(mol.symbols))

        # Perturb bond force constants by 20%
        params = _params(ff).copy()
        n_bonds = len(ff.bonds)
        for i in range(n_bonds):
            params[2 * i] *= 0.8  # force constant
        ff = _materialize(ff, params)

        backend = load_backend("jax")

        # Add only real vibrational frequencies (skip first 6 trans/rot)
        n3 = 3 * mol.n_atoms
        ref = ObservationSet()
        for i in range(6, n3):
            if abs(freqs_qm[i]) > 10.0:
                ref = ref.with_frequency(freqs_qm[i], data_idx=i, weight=1.0, case_id="0")

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.2f} → {result.final_score:.2f}"
        )

        # Final sorted frequencies should be closer to QM
        final_ff = _materialize(ff, result.final_params)
        final_freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, final_ff)
            .frequencies(FrequencyRequest(parameters=param_vector(final_ff)))
            .frequencies
        ]
        initial_freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, _materialize(ff, result.initial_params))
            .frequencies(FrequencyRequest(parameters=param_vector(_materialize(ff, result.initial_params))))
            .frequencies
        ]

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

        from q2mm.models.observations import ObservationSet

        # Water at bond_length=0.96. Energy minimum wants bond_r0≈0.96.
        # We start at bond_r0=0.88 and constrain to [0.85, 0.90].
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.88, angle_k=49.9, angle_eq=104.5)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        layout = _layout(ff)
        slots = list(layout.slots)
        slots[1] = replace(slots[1], bounds=(0.85, 0.90))
        layout = replace(layout, slots=tuple(slots))
        space = ActiveParameterSpace.all_active(layout, ff)
        plan = ObjectivePlan(
            case_ids=("0",),
            molecules=(mol,),
            stationary_points=(StationaryPointKind.GROUND_STATE,),
            observations=ref,
            layout=layout,
            active_space=space,
        )
        obj = JaxObjectiveExecutor(plan, backend, ff)

        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        result = JaxOptOptimizer(method="lbfgsb", maxiter=200, verbose=False).optimize(obj, space)
        final_params = result.final_params
        bond_r0_final = final_params[1]

        # The unconstrained optimum (0.96) is above the upper bound (0.90),
        # so the optimizer should push bond_r0 to the upper bound.
        np.testing.assert_allclose(
            bond_r0_final, 0.90, atol=0.01, err_msg=(f"bond_r0 ({bond_r0_final:.4f}) should be near upper bound 0.90")
        )


@pytest.mark.nightly
class TestScipyJaxObjectiveExecutorTelemetry:
    """F5: JaxObjectiveExecutor-path scipy runs must report real evaluation counts.

    On the JaxObjectiveExecutor analytical path scipy is driven by an
    internal loss/grad function and never calls ``evaluator.value``, so
    ``evaluator.n_evaluations`` stays frozen.  The optimizer now tracks the surrogate
    call count via telemetry and reports it as ``n_evaluations``.  Regression:
    ``n_evaluations`` used to be stuck near zero on this path.
    """

    def test_n_evaluations_reflects_executor_calls(self) -> None:
        from q2mm.models.observations import ObservationSet
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.80)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)

        optimizer = ScipyOptimizer(method="L-BFGS-B", maxiter=50, verbose=False)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.gradient_mode == "analytical"
        # Telemetry counts JAX executor value/gradient calls.
        assert result.n_evaluations > 2
