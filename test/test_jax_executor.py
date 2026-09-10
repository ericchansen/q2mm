"""Focused tests for the JAX objective executor.

Verifies that JAX-compiled per-case objective evaluation produces identical
results to the Python objective executor for supported references.
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    HessianJacobianRequest,
)
from q2mm.backends.contracts import (
    FrequencyRequest,
    HessianRequest,
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
    pytest.mark.jax,
]

from test._shared import make_diatomic, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.jax import JaxObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveConvergenceError
from q2mm.objectives.python import PythonObjectiveExecutor

# Module-level globals populated by autouse fixture
JaxBackend = None


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _make_objective(
    forcefield: ForceField, backend: object, molecules: list, reference: object, **kwargs: object
) -> PythonObjectiveExecutor:
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
    return PythonObjectiveExecutor(plan, backend, forcefield)


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


class TestObjectivePlan:
    """Tests for ObjectivePlan construction."""

    def test_energy_spec_roundtrip(self) -> None:
        """ObjectivePlan captures energy observations."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=load_backend("jax"), molecules=[mol], reference=ref)
        spec = obj.plan

        assert spec.n_params == 2
        assert len(spec.molecules) == 1
        assert spec.categories == frozenset({"energy"})
        obs = spec.observations.values
        assert len(obs) == 1
        assert obs[0].kind == "energy"
        assert obs[0].value == pytest.approx(0.0)
        assert obs[0].weight == pytest.approx(1.0)

    def test_geometry_included(self) -> None:
        """Geometry references are included via implicit-diff path."""
        from q2mm.models.observations import ObservationSet

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff()

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        ref = ref.with_bond_length(value=0.96, case_id="0", atom_indices=(0, 1), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=load_backend("jax"), molecules=[mol], reference=ref)
        spec = obj.plan

        assert spec.categories == frozenset({"energy", "geometry"})
        obs = spec.observations.values
        assert [o.kind for o in obs] == ["energy", "bond_length"]
        assert obs[1].atom_indices == (0, 1)
        assert obs[1].value == pytest.approx(0.96)

    def test_bounds_roundtrip(self) -> None:
        """Parameter bounds transferred to spec."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=load_backend("jax"), molecules=[mol], reference=ref)
        spec = obj.plan

        assert spec.active_space.bounds.shape == (2, 2)
        np.testing.assert_allclose(spec.active_space.bounds, spec.layout.bounds)

    @pytest.mark.parametrize("kind", ["frequency", "eig_diagonal", "hessian_element", "eig_offdiagonal"])
    @pytest.mark.parametrize("index", [6, 99])
    def test_invalid_indices_rejected_before_executor_preparation(
        self, monkeypatch: pytest.MonkeyPatch, kind: str, index: int
    ) -> None:
        from q2mm.models.observations import Observation, ObservationSet

        mol = make_diatomic(distance=0.9, bond_tolerance=1.5)
        ff = _h2_ff()
        backend = load_backend("jax")
        prepare_calls = []

        def forbidden_prepare(*args: object, **kwargs: object) -> None:
            prepare_calls.append((args, kwargs))
            pytest.fail("Invalid references reached backend preparation")

        monkeypatch.setattr(backend, "prepare", forbidden_prepare)
        if kind in ("frequency", "eig_diagonal"):
            obs = Observation(kind=kind, value=0.0, data_idx=index)
        else:
            obs = Observation(kind=kind, value=0.0, atom_indices=(0, index))
        with pytest.raises(ValueError, match="out of range"):
            py = _make_objective(ff, backend, [mol], ObservationSet((obs,)))
            JaxObjectiveExecutor(py.plan, backend, ff)
        assert prepare_calls == []


class TestJaxObjectiveExecutor:
    """Tests for JaxObjectiveExecutor compiled loss function."""

    def test_loss_matches_objective(self) -> None:
        """JaxObjectiveExecutor(params) ≈ _make_objective(params) for energy."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        np.testing.assert_allclose(jax_score, python_score, rtol=1e-6)

    def test_loss_and_grad_returns_gradient(self) -> None:
        """loss_and_grad returns a scalar loss and param-shaped gradient."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        loss, grad = jax_loss.loss_and_grad(params)

        assert isinstance(loss, float)
        assert grad.shape == params.shape

    def test_loss_with_regularization(self) -> None:
        """L2 regularization adds penalty to loss."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj_noreg = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref, regularization=0.0)
        obj_reg = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref, regularization=0.1)

        spec_noreg = obj_noreg.plan
        spec_reg = obj_reg.plan

        loss_noreg = JaxObjectiveExecutor(spec_noreg, backend, ff)
        loss_reg = JaxObjectiveExecutor(spec_reg, backend, ff)

        params = _params(ff)
        score_noreg = loss_noreg.value(params)
        score_reg = loss_reg.value(params)

        # At reference params, L2 penalty should be zero
        np.testing.assert_allclose(score_noreg, score_reg, rtol=1e-10)

        # With perturbed params, regularized loss should be higher
        perturbed = params * 1.1
        assert loss_reg.value(perturbed) > loss_noreg.value(perturbed)

    def test_backend_type_check(self) -> None:
        """JaxObjectiveExecutor raises TypeError for non-JAX backends."""
        from unittest.mock import MagicMock

        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff()
        ref = ObservationSet().with_energy(value=0.0, case_id="0", weight=1.0)
        spec = _make_objective(forcefield=ff, backend=load_backend("jax"), molecules=[mol], reference=ref).plan
        fake_backend = MagicMock()
        fake_backend.__class__.__name__ = "FakeBackend"

        with pytest.raises(TypeError, match="JaxObjectiveExecutor requires a JaxBackend"):
            JaxObjectiveExecutor(spec, fake_backend, ff)

    def test_water_energy_loss(self) -> None:
        """JaxObjectiveExecutor works for water (bond + angle params)."""
        from q2mm.models.observations import ObservationSet

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        # Both scores are effectively zero at equilibrium; use absolute tolerance
        np.testing.assert_allclose(jax_score, python_score, atol=1e-10)

    def test_valid_boundary_scalar_residual_and_gradient_parity(self) -> None:
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.9, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=100.0)
        backend = load_backend("jax")
        hessian = prepare_case(backend, mol, ff).hessian(HessianRequest(parameters=_params(ff))).hessian
        mol = mol.with_hessian(hessian)
        refs = (
            ObservationSet()
            .with_energy(1.0, weight=2.0)
            .with_hessian_eigenvalue(0.01, mode_idx=5, weight=3.0)
            .with_hessian_offdiagonal(0.01, row=0, col=5, weight=2.0)
            .with_hessian_offdiagonal(0.01, row=5, col=0, weight=2.0)
            .with_hessian_element(0.01, row=5, col=5, weight=3.0)
            .with_hessian_element(0.01, row=0, col=5, weight=2.0)
        )
        scalar_py = _make_objective(ff, backend, [mol], refs, regularization=0.1)
        py = PythonObjectiveExecutor(scalar_py.plan, backend, ff, gradient_mode=GradientMode.ANALYTICAL)
        jx = JaxObjectiveExecutor(py.plan, backend, ff)
        params = _params(ff) + np.array([0.5, 0.01])
        value, gradient = jx.value_and_gradient(params)
        py_value, py_gradient = py.value_and_gradient(params)
        assert value > 0.0
        assert value == pytest.approx(py_value, rel=1e-10)
        assert value == pytest.approx(jx.value(params), rel=1e-10)
        for executor in (py, jx):
            evaluation = executor.evaluate(params)
            assert value == pytest.approx(evaluation.total, rel=1e-10)
            assert value == pytest.approx(np.sum(executor.least_squares_residuals(params) ** 2), rel=1e-10)
        np.testing.assert_allclose(py.evaluate(params).calculated, jx.evaluate(params).calculated, atol=1e-8)
        np.testing.assert_allclose(
            py.evaluate(params).weighted_residuals, jx.evaluate(params).weighted_residuals, atol=1e-8
        )
        np.testing.assert_allclose(gradient, py_gradient, atol=1e-7)
        fd = np.zeros_like(params)
        for index in range(params.size):
            step = np.zeros_like(params)
            step[index] = 1e-5
            fd[index] = (py.value(params + step) - py.value(params - step)) / (2e-5)
        np.testing.assert_allclose(gradient, fd, atol=1e-5, rtol=1e-5)

    def test_last_cartesian_frequency_scalar_and_residual_parity(self) -> None:
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.9, bond_tolerance=1.5)
        ff = _h2_ff()
        backend = load_backend("jax")
        refs = ObservationSet().with_frequency(100.0, data_idx=5, weight=1e-3)
        py = _make_objective(ff, backend, [mol], refs)
        jx = JaxObjectiveExecutor(py.plan, backend, ff)
        params = _params(ff)
        frequencies = prepare_case(backend, mol, ff).frequencies(FrequencyRequest(parameters=params)).frequencies
        assert frequencies.shape == (6,)
        expected = (1e-3 * (100.0 - frequencies[5])) ** 2
        for executor in (py, jx):
            assert executor.value(params) == pytest.approx(expected, rel=1e-10)
            evaluation = executor.evaluate(params)
            assert evaluation.calculated[0] == pytest.approx(frequencies[5], rel=1e-10)
            assert evaluation.total == pytest.approx(expected, rel=1e-10)
            assert evaluation.weighted_residuals[0] ** 2 == pytest.approx(expected, rel=1e-10)


def _water_with_qm_refs(backend: object) -> tuple:
    """Build water molecule with computed 'QM' reference data.

    Computes MM hessian/frequencies at equilibrium params to serve
    as reference data, then returns a perturbed FF for non-trivial loss.
    """
    mol = make_water(bond_length=0.96, angle_deg=104.5)
    ff_ref = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)

    # Compute 'QM' hessian at equilibrium
    hess_qm = prepare_case(backend, mol, ff_ref).hessian(HessianRequest(parameters=param_vector(ff_ref))).hessian
    # Attach as QM hessian (needed for eigenmatrix)
    mol = mol.with_hessian(hess_qm)

    # Compute 'QM' frequencies
    from q2mm.models.hessian import hessian_to_frequencies

    freqs_qm = hessian_to_frequencies(hess_qm, list(mol.symbols))

    # Perturbed FF for non-trivial loss
    ff_pert = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=110.0)

    return mol, ff_ref, ff_pert, hess_qm, freqs_qm


class TestJaxObjectiveExecutorFrequencyParity:
    """Frequency path: JaxObjectiveExecutor vs PythonObjectiveExecutor parity."""

    def test_frequency_loss_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree on frequency loss."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, _hess, freqs_qm = _water_with_qm_refs(backend)

        # Add only the 3 real vibrational modes (indices 6, 7, 8)
        ref = ObservationSet()
        for i in range(6, 9):
            ref = ref.with_frequency(freqs_qm[i], data_idx=i, weight=1.0, case_id="0")

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-4)

    def test_frequency_gradient_shape_and_direction(self) -> None:
        """JaxObjectiveExecutor gradient has correct shape and points downhill."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, _hess, freqs_qm = _water_with_qm_refs(backend)

        ref = ObservationSet()
        for i in range(6, 9):
            ref = ref.with_frequency(freqs_qm[i], data_idx=i, weight=1.0, case_id="0")

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        loss, grad = jax_loss.loss_and_grad(params)

        assert grad.shape == params.shape
        assert np.linalg.norm(grad) > 0, "Gradient should be nonzero"

        # Small step in negative gradient direction should decrease loss.
        # Frequencies are large (cm⁻¹), so gradients are huge — use tiny step.
        lr = 1e-6 / np.linalg.norm(grad)
        step = params - lr * grad
        loss_after = jax_loss.value(step)
        assert loss_after < loss, "One step in -grad direction should lower loss"


class TestJaxObjectiveExecutorHessianParity:
    """Hessian-element path: JaxObjectiveExecutor vs PythonObjectiveExecutor parity."""

    def test_hessian_element_loss_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree on hessian-element loss."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, hess_qm, _freqs = _water_with_qm_refs(backend)

        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            case_id="0",
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-6)


class TestJaxObjectiveExecutorEigenmatrixParity:
    """Eigenmatrix diagonal path: JaxObjectiveExecutor vs PythonObjectiveExecutor parity."""

    def test_eigenmatrix_diagonal_loss_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree on eigenmatrix diagonal loss."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, hess_qm, _freqs = _water_with_qm_refs(backend)

        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(
            hess_qm,
            symbols=list(mol.symbols),
            diagonal_only=True,
            case_id="0",
            weights={"eig_i": 0.0, "eig_d_low": 0.1, "eig_d_high": 0.1, "eig_o": 0.0},
        )

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-6)

    def test_eigenmatrix_offdiagonal_loss_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree on the full (off-diagonal) eigenmatrix."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, hess_qm, _freqs = _water_with_qm_refs(backend)

        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(
            hess_qm,
            symbols=list(mol.symbols),
            diagonal_only=False,
            case_id="0",
            weights={"eig_i": 0.0, "eig_d_low": 0.1, "eig_d_high": 0.1, "eig_o": 0.05},
        )

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-6)


class TestJaxObjectiveExecutorMixedObjective:
    """Multi-category loss: energy + frequency + hessian combined."""

    def test_mixed_loss_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree on combined energy+freq+hessian."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, hess_qm, freqs_qm = _water_with_qm_refs(backend)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        for i in range(6, 9):
            ref = ref.with_frequency(freqs_qm[i], data_idx=i, weight=1.0, case_id="0")
        ref = ref.with_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            case_id="0",
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Combined loss should be nonzero"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-4)

    def test_mixed_gradient_lowers_loss(self) -> None:
        """One gradient step on combined loss reduces the score."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol, _ff_ref, ff_pert, hess_qm, freqs_qm = _water_with_qm_refs(backend)

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        for i in range(6, 9):
            ref = ref.with_frequency(freqs_qm[i], data_idx=i, weight=1.0, case_id="0")
        ref = ref.with_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            case_id="0",
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = _make_objective(forcefield=ff_pert, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff_pert)

        params = _params(ff_pert)
        loss0, grad = jax_loss.loss_and_grad(params)
        lr = 1e-6 / np.linalg.norm(grad)
        step = params - lr * grad
        loss1 = jax_loss.value(step)

        assert loss1 < loss0, "One gradient step should lower combined loss"


class TestJaxObjectiveExecutorMultiMolecule:
    """Multi-molecule routing: H2 + water with separate energy refs."""

    def test_multi_molecule_energy_parity(self) -> None:
        """JaxObjectiveExecutor and PythonObjectiveExecutor agree across 2 molecules."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")

        mol_h2 = make_diatomic(distance=0.74, bond_tolerance=1.5)
        mol_water = make_water(bond_length=0.96, angle_deg=104.5)

        ff = ForceField(
            bonds=[
                BondParam(elements=("H", "H"), force_constant=215.8, equilibrium=0.80),
                BondParam(elements=("H", "O"), force_constant=400.0, equilibrium=1.05),
            ],
            angles=[
                AngleParam(elements=("H", "O", "H"), force_constant=35.0, equilibrium=110.0),
            ],
            functional_form=FunctionalForm.HARMONIC,
        )

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        ref = ref.with_energy(value=0.0, case_id="1", weight=1.0)

        molecules = [mol_h2, mol_water]
        obj = _make_objective(forcefield=ff, backend=backend, molecules=molecules, reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert python_score > 0, "Multi-molecule loss should be nonzero"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-10)


class TestFrequencySensitivityParity:
    """_jax_frequency_param_jacobian vs NumPy frequency_param_jacobian."""

    def test_water_freq_jacobian_parity(self) -> None:
        """JAX and NumPy frequency Jacobians agree on water."""
        from q2mm.backends.mm._jax_common import jnp
        from q2mm.models.hessian import (
            _jax_frequency_param_jacobian,
            frequency_param_jacobian,
            symbols_to_masses_3n,
        )
        from q2mm.models.units import KCALMOLA2_TO_HESSIAN_AU

        backend = load_backend("jax")
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)

        # Compute Hessian + Jacobian via JaxBackend
        _hj = prepare_case(backend, mol, ff).hessian_parameter_jacobian(
            HessianJacobianRequest(parameters=param_vector(ff))
        )
        hess_au, dH_dp_kcal = _hj.hessian, _hj.jacobian
        dH_dp_au = dH_dp_kcal * KCALMOLA2_TO_HESSIAN_AU

        # NumPy path
        freqs_np, jac_np = frequency_param_jacobian(hess_au, dH_dp_au, list(mol.symbols))

        # JAX path
        masses_3n = jnp.array(symbols_to_masses_3n(list(mol.symbols)), dtype=jnp.float64)
        hess_jax = jnp.array(hess_au, dtype=jnp.float64)
        dH_dp_jax = jnp.array(dH_dp_au, dtype=jnp.float64)
        freqs_jax, jac_jax = _jax_frequency_param_jacobian(hess_jax, dH_dp_jax, masses_3n)

        freqs_jax_np = np.asarray(freqs_jax)
        jac_jax_np = np.asarray(jac_jax)

        # Eigenvalue parity — real modes (last 3) should agree closely
        np.testing.assert_allclose(
            freqs_jax_np[6:], freqs_np[6:], atol=1e-4, err_msg="Real vibrational frequencies should match"
        )

        # Jacobian direction parity (cosine similarity per mode)
        for i in range(6, 9):
            j_np = jac_np[i, :]
            j_jax = jac_jax_np[i, :]
            norm_np = np.linalg.norm(j_np)
            norm_jax = np.linalg.norm(j_jax)
            if norm_np > 1e-10 and norm_jax > 1e-10:
                cos_sim = np.dot(j_np, j_jax) / (norm_np * norm_jax)
                assert cos_sim > 0.99, f"Mode {i}: cosine similarity {cos_sim:.4f} too low"


class TestJaxObjectiveExecutorGeometryParity:
    """Tests for the geometry-references implicit-diff loss path."""

    pytestmark = pytest.mark.skipif(not _HAS_JAXOPT, reason="jaxopt not installed (required for geometry refs)")

    def test_bond_length_relaxes_to_eq(self) -> None:
        """Relaxed H2 bond length matches the FF equilibrium parameter."""
        from q2mm.models.observations import ObservationSet

        # Start with a stretched bond; equilibrium r0 = 0.74 Å.
        mol = make_diatomic(distance=0.90, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        backend = load_backend("jax")

        # Reference equals the eq value → relaxed loss should be ~0.
        ref = ObservationSet()
        ref = ref.with_bond_length(value=0.74, case_id="0", atom_indices=(0, 1), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        loss = jax_loss.value(params)
        # With harmonic geometry restraint (K=100), relaxation from 0.90 to
        # 0.74 Å is approximate — the restraint anchors the geometry near
        # the starting point.  Loss should be small but not exactly zero.
        assert loss < 0.01

    def test_bond_length_grad_matches_fd(self) -> None:
        """∇_p loss for a bond-length ref matches finite differences."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.90, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        backend = load_backend("jax")

        # Reference offset from current r0 so the gradient is non-trivial.
        ref = ObservationSet()
        ref = ref.with_bond_length(value=0.80, case_id="0", atom_indices=(0, 1), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        _, grad_jax = jax_loss.loss_and_grad(params)

        # Central-difference reference gradient (full-pipeline FD: each
        # eval relaxes the geometry at the perturbed parameters).
        eps = 1e-4
        grad_fd = np.zeros_like(params)
        for i in range(len(params)):
            dp = np.zeros_like(params)
            dp[i] = eps
            grad_fd[i] = (jax_loss.value(params + dp) - jax_loss.value(params - dp)) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, atol=1e-5, rtol=1e-3)

    def test_bond_angle_relaxes_to_eq(self) -> None:
        """Relaxed H2O bond angle matches the FF equilibrium parameter."""
        from q2mm.models.observations import ObservationSet

        # Start with a perturbed angle; eq is 104.5°.
        mol = make_water(bond_length=0.96, angle_deg=110.0)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)
        backend = load_backend("jax")

        ref = ObservationSet()
        # Atom indices: H(1)–O(0)–H(2) with O as vertex.
        ref = ref.with_bond_angle(value=104.5, case_id="0", atom_indices=(1, 0, 2), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        loss = jax_loss.value(params)
        # With harmonic geometry restraint (K=100), the angle doesn't
        # fully relax from 110° to the 104.5° equilibrium — the
        # restraint anchors geometry near the starting coordinates.
        # Loss should be moderate (not huge, not zero).
        assert loss < 10.0

    def test_geometry_grad_jit_callable(self) -> None:
        """JaxObjectiveExecutor.loss_and_grad with geometry refs is jit-compilable."""
        from q2mm.models.observations import ObservationSet

        mol = make_water(bond_length=0.97, angle_deg=104.5)
        ff = _water_ff()
        backend = load_backend("jax")

        ref = ObservationSet()
        ref = ref.with_bond_length(value=0.96, case_id="0", atom_indices=(0, 1), weight=1.0)
        ref = ref.with_bond_angle(value=104.5, case_id="0", atom_indices=(1, 0, 2), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        loss, grad = jax_loss.loss_and_grad(params)
        assert np.isfinite(loss)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))

    def test_torsion_observable_math(self) -> None:
        """_torsion_angles_deg produces correct dihedrals for known geometries."""
        import jax.numpy as jnp

        from q2mm.objectives.jax import _torsion_angles_deg

        # Planar cis (0°): all four atoms in xy-plane, zig-zag along y.
        coords_cis = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        # Planar trans (180°): atom 3 on opposite side.
        coords_trans = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        # Perpendicular (+90°): atom 3 rotated about b2 axis.
        coords_90 = jnp.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        atoms = jnp.array([[0, 1, 2, 3]])

        np.testing.assert_allclose(float(_torsion_angles_deg(coords_cis, atoms)[0]), 0.0, atol=1e-6)
        np.testing.assert_allclose(abs(float(_torsion_angles_deg(coords_trans, atoms)[0])), 180.0, atol=1e-6)
        np.testing.assert_allclose(abs(float(_torsion_angles_deg(coords_90, atoms)[0])), 90.0, atol=1e-6)

    def test_torsion_residual_wraps_across_180(self) -> None:
        """Torsion loss at observed=+179°, ref=-179° gives ~2° residual, not 358°."""
        import jax.numpy as jnp

        # Mimic the wrap logic used inside _loss_fn.
        observed = jnp.array([179.0])
        ref = jnp.array([-179.0])
        weights = jnp.array([1.0])
        diff = observed - ref
        diff = (diff + 180.0) % 360.0 - 180.0
        loss = float(jnp.sum(weights * diff * diff))
        # Un-wrapped diff would be 358° → loss ≈ 128164.  Wrapped: −2° → 4.
        assert loss < 10.0
        np.testing.assert_allclose(loss, 4.0, atol=1e-6)

    def test_nonconvergence_penalizes_loss_and_blocks_observables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A failed trial retreats toward a valid anchor but is not a valid endpoint."""
        import q2mm.objectives.jax as jax_objective
        from q2mm.models.observations import ObservationSet

        monkeypatch.setattr(jax_objective, "_GEOM_INNER_MAXITER", 1)
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff()
        backend = load_backend("jax")
        ref = (
            ObservationSet().with_bond_length(0.96, atom_indices=(0, 1)).with_bond_angle(104.5, atom_indices=(1, 0, 2))
        )
        obj = _make_objective(ff, backend, [mol], ref)
        jax_loss = JaxObjectiveExecutor(obj.plan, backend, ff)
        baseline = obj.plan.active_space.baseline
        assert jax_loss.evaluate(baseline).total == pytest.approx(0.0, abs=1e-15)
        assert jax_loss.value(baseline) == pytest.approx(0.0, abs=1e-15)

        params = _params(_water_ff(bond_r0=1.2, angle_eq=80.0))
        loss, grad = jax_loss.value_and_gradient(params)
        assert loss > jax_objective._GEOM_NONCONVERGENCE_PENALTY
        assert jax_loss.value(params) == pytest.approx(loss)
        native_loss, native_grad = jax_loss.value_and_grad_jax(params)
        assert float(native_loss) == pytest.approx(loss)
        np.testing.assert_allclose(native_grad, grad)
        scales = np.maximum(np.abs(baseline), 1.0)
        expected_grad = 2 * jax_objective._GEOM_NONCONVERGENCE_BARRIER_SCALE * (params - baseline) / scales**2
        np.testing.assert_allclose(grad, expected_grad, atol=1e-10)
        assert np.dot(grad, params - baseline) > 0.0
        with pytest.raises(ObjectiveConvergenceError, match="did not converge"):
            jax_loss.evaluate(params)
        with pytest.raises(ObjectiveConvergenceError, match="did not converge"):
            jax_loss.least_squares_residuals(params)

    @pytest.mark.parametrize("weight", [1.0, 0.0])
    def test_capped_baseline_rejected_before_building_barrier(
        self, monkeypatch: pytest.MonkeyPatch, weight: float
    ) -> None:
        """A later case's failed anchor must reject the entire executor."""
        import q2mm.objectives.jax as jax_objective
        from q2mm.models.observations import ObservationSet

        monkeypatch.setattr(jax_objective, "_GEOM_INNER_MAXITER", 1)
        ff = _water_ff()
        backend = load_backend("jax")
        mols = [
            make_water(bond_length=0.96, angle_deg=104.5),
            make_water(bond_length=1.2, angle_deg=80.0),
        ]
        refs = (
            ObservationSet()
            .with_bond_length(0.96, case_id="0", atom_indices=(0, 1))
            .with_bond_length(0.96, case_id="1", atom_indices=(0, 1), weight=weight)
            .with_bond_angle(104.5, case_id="1", atom_indices=(1, 0, 2), weight=weight)
        )
        obj = _make_objective(ff, backend, mols, refs)
        with pytest.raises(ObjectiveConvergenceError, match="baseline.*case '1'.*did not converge") as exc:
            JaxObjectiveExecutor(obj.plan, backend, ff)
        assert "maximum force" in str(exc.value)
        assert f"tolerance {jax_objective._GEOM_INNER_TOL:.6g}" in str(exc.value)
        assert "iteration cap 1" in str(exc.value)

    def test_baseline_validation_uses_active_space_not_preparation_values(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The supplied FF provides structure, not the barrier's parameter anchor."""
        import q2mm.objectives.jax as jax_objective
        from q2mm.models.observations import ObservationSet

        monkeypatch.setattr(jax_objective, "_GEOM_INNER_MAXITER", 1)
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        structural_ff = _water_ff()
        baseline_ff = _water_ff(bond_r0=1.2, angle_eq=80.0)
        backend = load_backend("jax")
        refs = ObservationSet().with_bond_angle(104.5, atom_indices=(1, 0, 2))
        obj = _make_objective(baseline_ff, backend, [mol], refs)
        with pytest.raises(ObjectiveConvergenceError, match="baseline.*case '0'.*did not converge"):
            JaxObjectiveExecutor(obj.plan, backend, structural_ff)

    def test_valid_baselines_checked_once_with_reused_preparations(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Preparation and anchor validation are per case, not per evaluation."""
        from unittest.mock import patch

        import jax
        import q2mm.objectives.jax as jax_objective
        from q2mm.models.observations import ObservationSet

        ff = _h2_ff()
        backend = load_backend("jax")
        mols = [make_diatomic(distance=d, bond_tolerance=5.0) for d in (0.9, 1.0, 3.0)]
        refs = (
            ObservationSet()
            .with_bond_length(0.8, case_id="0", atom_indices=(0, 1), weight=2.0)
            .with_bond_length(0.85, case_id="1", atom_indices=(0, 1), weight=3.0)
            .with_energy(0.0, case_id="2")
        )
        obj = _make_objective(ff, backend, mols, refs, regularization=0.1)
        relax_coords = jax_objective._relax_coords
        baseline_calls = []

        def track_relaxation(energy_fn: object, params: object, coords: object) -> object:
            if not isinstance(params, jax.core.Tracer) and np.array_equal(params, obj.plan.active_space.baseline):
                baseline_calls.append((energy_fn, np.asarray(coords)))
            return relax_coords(energy_fn, params, coords)

        monkeypatch.setattr(jax_objective, "_relax_coords", track_relaxation)
        with patch.object(backend, "prepare", wraps=backend.prepare) as prepare:
            jax_loss = JaxObjectiveExecutor(obj.plan, backend, ff)
            assert prepare.call_count == 3
            assert len(baseline_calls) == 2
            for index, (energy_fn, coords) in enumerate(baseline_calls):
                assert energy_fn is jax_loss._sessions[str(index)]._energy_kernel()
                np.testing.assert_array_equal(coords, mols[index].geometry)
            assert len(jax_loss._compiled_value_fns) == 3
            assert len(jax_loss._compiled_vag_fns) == 3
            assert jax_loss.n_evaluations == 0
            assert jax_loss.history == ()

            params = _params(_h2_ff(bond_k=350.0, bond_r0=0.76))
            for _ in range(2):
                scalar = jax_loss.value(params)
                value, grad = jax_loss.value_and_gradient(params)
                assert value == pytest.approx(scalar)
                assert jax_loss.evaluate(params).total == pytest.approx(scalar)
                assert np.all(np.isfinite(grad))
                jax_loss.reset()
            assert prepare.call_count == 3
            assert len(baseline_calls) == 2

    def test_valid_geometry_scalar_and_gradient_preserved(self) -> None:
        """Baseline validation leaves weighted geometry and regularization unchanged."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.9, bond_tolerance=1.5)
        ff = _h2_ff()
        backend = load_backend("jax")
        refs = ObservationSet().with_bond_length(0.8, atom_indices=(0, 1), weight=2.0)
        obj = _make_objective(ff, backend, [mol], refs, regularization=0.1)
        jax_loss = JaxObjectiveExecutor(obj.plan, backend, ff)
        baseline = obj.plan.active_space.baseline.copy()
        params = _params(_h2_ff(bond_k=350.0, bond_r0=0.76))
        expected = (2 * (0.8 - params[1])) ** 2 + 0.1 * np.sum((params - baseline) ** 2)
        expected_grad = 0.2 * (params - baseline)
        expected_grad[1] += 8 * (params[1] - 0.8)
        assert jax_loss.value(params) == pytest.approx(expected, abs=1e-10)
        assert jax_loss.evaluate(params).total == pytest.approx(expected, abs=1e-10)
        assert np.sum(jax_loss.least_squares_residuals(params) ** 2) == pytest.approx(expected, abs=1e-10)
        for value, gradient in (jax_loss.value_and_gradient(params), jax_loss.value_and_grad_jax(params)):
            assert float(value) == pytest.approx(expected, abs=1e-10)
            np.testing.assert_allclose(gradient, expected_grad, atol=1e-8)
        np.testing.assert_array_equal(obj.plan.active_space.baseline, baseline)
        np.testing.assert_array_equal(_params(ff), baseline)


def test_nongeometry_plan_never_relaxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """An invalid geometry anchor is irrelevant to fixed-coordinate objectives."""
    import q2mm.objectives.jax as jax_objective
    from q2mm.models.observations import ObservationSet

    def forbidden_relaxation(*args: object, **kwargs: object) -> None:
        pytest.fail("Non-geometry objective attempted a geometry relaxation")

    monkeypatch.setattr(jax_objective, "_relax_coords", forbidden_relaxation)
    mol = make_water(bond_length=1.2, angle_deg=80.0).with_hessian(np.eye(9))
    ff = _water_ff()
    backend = load_backend("jax")
    refs = (
        ObservationSet()
        .with_energy(1.0, weight=0.1)
        .with_frequency(100.0, data_idx=8, weight=0.001)
        .with_hessian_element(0.0, row=8, col=8)
        .with_hessian_eigenvalue(0.0, mode_idx=8)
        .with_hessian_offdiagonal(0.0, row=7, col=8)
    )
    obj = _make_objective(ff, backend, [mol], refs, regularization=0.1)
    jax_loss = JaxObjectiveExecutor(obj.plan, backend, ff)
    params = _params(ff)
    value, gradient = jax_loss.value_and_gradient(params)
    assert value == pytest.approx(obj.value(params), rel=1e-8)
    assert jax_loss.value(params) == pytest.approx(value, rel=1e-10)
    assert jax_loss.evaluate(params).total == pytest.approx(value, rel=1e-10)
    assert np.all(np.isfinite(gradient))


class TestJaxObjectiveExecutorTopologyBatching:
    """Verify per-case JIT dispatch produces identical results.

    Both tests add frequency references so ``needs_hessian_computation``
    is True and the per-case JIT Hessian path is exercised.
    """

    def test_two_water_freq_batching_parity(self) -> None:
        """Two water molecules (same topology) — per-case JIT Hessian matches Python executor."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol1 = make_water(bond_length=0.96, angle_deg=104.5)
        mol2 = make_water(bond_length=0.97, angle_deg=105.0)

        ff = ForceField(
            bonds=[BondParam(elements=("H", "O"), force_constant=553.0, equilibrium=0.96)],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=49.9, equilibrium=104.5)],
            functional_form=FunctionalForm.HARMONIC,
        )

        # Get MM frequencies to build realistic reference indices
        mm_freqs1 = [
            float(_f)
            for _f in prepare_case(backend, mol1, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        mm_freqs2 = [
            float(_f)
            for _f in prepare_case(backend, mol2, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        real_indices1 = [i for i, f in enumerate(mm_freqs1) if f > 50.0]
        real_indices2 = [i for i, f in enumerate(mm_freqs2) if f > 50.0]

        ref = ObservationSet()
        for idx in real_indices1[:3]:
            ref = ref.with_frequency(mm_freqs1[idx] * 1.05, data_idx=idx, weight=1.0, case_id="0")
        for idx in real_indices2[:3]:
            ref = ref.with_frequency(mm_freqs2[idx] * 1.05, data_idx=idx, weight=1.0, case_id="1")

        molecules = [mol1, mol2]
        obj = _make_objective(
            forcefield=ff,
            backend=backend,
            molecules=molecules,
            reference=ref,
        )

        # Per-case JaxObjectiveExecutor (same topology, 2 molecules)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert jax_score > 0, "Frequency loss should be nonzero"
        np.testing.assert_allclose(jax_score, python_score, rtol=1e-6)

        # Gradient should propagate through per-case Hessian path
        _, grad = jax_loss.loss_and_grad(params)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))

    def test_mixed_topology_freq_parity(self) -> None:
        """H₂ + 2× water: mixed topologies use per-case compiled fragments."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        mol_h2 = make_diatomic(distance=0.74, bond_tolerance=1.5)
        mol_w1 = make_water(bond_length=0.96, angle_deg=104.5)
        mol_w2 = make_water(bond_length=0.97, angle_deg=105.0)

        ff = ForceField(
            bonds=[
                BondParam(elements=("H", "H"), force_constant=359.7, equilibrium=0.74),
                BondParam(elements=("H", "O"), force_constant=553.0, equilibrium=0.96),
            ],
            angles=[
                AngleParam(elements=("H", "O", "H"), force_constant=49.9, equilibrium=104.5),
            ],
            functional_form=FunctionalForm.HARMONIC,
        )

        # Frequency refs for all three molecules (exercises Hessian path)
        mm_h2 = [
            float(_f)
            for _f in prepare_case(backend, mol_h2, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        mm_w1 = [
            float(_f)
            for _f in prepare_case(backend, mol_w1, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        mm_w2 = [
            float(_f)
            for _f in prepare_case(backend, mol_w2, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]

        ref = ObservationSet()
        # H₂ — singleton topology group
        real_h2 = [i for i, f in enumerate(mm_h2) if f > 50.0]
        for idx in real_h2[:1]:
            ref = ref.with_frequency(mm_h2[idx] * 1.05, data_idx=idx, weight=1.0, case_id="0")
        # Water 1 + 2 — same topology, separate per-case fragments
        real_w = [i for i, f in enumerate(mm_w1) if f > 50.0]
        for idx in real_w[:3]:
            ref = ref.with_frequency(mm_w1[idx] * 1.05, data_idx=idx, weight=1.0, case_id="1")
            ref = ref.with_frequency(mm_w2[idx] * 1.05, data_idx=idx, weight=1.0, case_id="2")

        molecules = [mol_h2, mol_w1, mol_w2]
        obj = _make_objective(
            forcefield=ff,
            backend=backend,
            molecules=molecules,
            reference=ref,
        )
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)
        python_score = obj.value(params)
        jax_score = jax_loss.value(params)

        assert jax_score > 0, "Frequency loss should be nonzero"
        np.testing.assert_allclose(jax_score, python_score, rtol=1e-6)


class TestJaxObjectiveExecutorNaNProtection:
    """Tests for NaN/Inf protection in loss_and_grad."""

    def test_nan_returns_sentinel(self) -> None:
        """When a per-molecule function returns NaN, loss_and_grad returns (1e30, zeros)."""
        from q2mm.models.observations import ObservationSet

        mol = make_diatomic(distance=0.74)
        ff = _h2_ff()
        backend = load_backend("jax")
        ref = ObservationSet()
        ref = ref.with_bond_length(value=0.74, case_id="0", atom_indices=(0, 1), weight=1.0)

        obj = _make_objective(forcefield=ff, backend=backend, molecules=[mol], reference=ref)
        spec = obj.plan
        jax_loss = JaxObjectiveExecutor(spec, backend, ff)

        params = _params(ff)

        # Patch the compiled vag fns to return NaN
        import jax.numpy as jnp

        nan_val = jnp.float64(float("nan"))
        nan_grad = jnp.full_like(jnp.array(params), float("nan"))

        orig_nongeom = jax_loss._compiled_vag_fns
        jax_loss._compiled_vag_fns = [lambda p: (nan_val, nan_grad)]

        loss, grad = jax_loss.loss_and_grad(params)

        assert loss == pytest.approx(1e30)
        assert np.all(grad == 0.0), "Gradient should be zeros on NaN"

        # Restore
        jax_loss._compiled_vag_fns = orig_nongeom
