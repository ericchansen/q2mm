"""Unit tests for ObjectiveSpec and JaxLoss.

Verifies that the JIT-compiled loss function produces identical
results to the Python ObjectiveFunction for energy references.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.jax,
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


class TestObjectiveSpec:
    """Tests for ObjectiveSpec construction."""

    def test_energy_spec_roundtrip(self) -> None:
        """ObjectiveFunction.to_jax_spec() captures energy references."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=JaxEngine(), molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()

        assert spec.n_params == 2
        assert spec.n_molecules == 1
        assert "energy" in spec.supported_categories
        assert spec.molecules[0].has_energy
        np.testing.assert_array_equal(spec.molecules[0].energy_refs, [0.0])
        np.testing.assert_array_equal(spec.molecules[0].energy_weights, [1.0])

    def test_geometry_included(self) -> None:
        """Geometry references are included via implicit-diff path."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        ref.add_bond_length(value=0.96, molecule_idx=0, atom_indices=(0, 1), weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=JaxEngine(), molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()

        assert spec.molecules[0].has_energy
        assert spec.molecules[0].has_bond_length
        assert spec.molecules[0].has_geometry
        assert "geometry" in spec.supported_categories
        assert spec.has_geometry_refs() is True
        np.testing.assert_array_equal(spec.molecules[0].bond_atoms, [[0, 1]])
        np.testing.assert_array_equal(spec.molecules[0].bond_refs, [0.96])

    def test_bounds_roundtrip(self) -> None:
        """Parameter bounds transferred to spec."""
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=JaxEngine(), molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()

        assert len(spec.lower_bounds) == 2
        assert len(spec.upper_bounds) == 2


class TestJaxLoss:
    """Tests for JaxLoss compiled loss function."""

    def test_loss_matches_objective(self) -> None:
        """JaxLoss(params) ≈ ObjectiveFunction(params) for energy."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        np.testing.assert_allclose(jax_score, python_score, rtol=1e-6)

    def test_loss_and_grad_returns_gradient(self) -> None:
        """loss_and_grad returns a scalar loss and param-shaped gradient."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        loss, grad = jax_loss.loss_and_grad(params)

        assert isinstance(loss, float)
        assert grad.shape == params.shape

    def test_loss_with_regularization(self) -> None:
        """L2 regularization adds penalty to loss."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.74, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=215.8, bond_r0=0.74)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj_noreg = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref, regularization=0.0)
        obj_reg = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref, regularization=0.1)

        spec_noreg = obj_noreg.to_jax_spec()
        spec_reg = obj_reg.to_jax_spec()

        loss_noreg = JaxLoss(spec_noreg, engine, [mol], ff)
        loss_reg = JaxLoss(spec_reg, engine, [mol], ff)

        params = ff.get_param_vector()
        score_noreg = loss_noreg(params)
        score_reg = loss_reg(params)

        # At reference params, L2 penalty should be zero
        np.testing.assert_allclose(score_noreg, score_reg, rtol=1e-10)

        # With perturbed params, regularized loss should be higher
        perturbed = params * 1.1
        assert loss_reg(perturbed) > loss_noreg(perturbed)

    def test_engine_type_check(self) -> None:
        """JaxLoss raises TypeError for non-JAX engines."""
        from unittest.mock import MagicMock

        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.spec import ObjectiveSpec

        spec = ObjectiveSpec(molecules=(), n_params=2)
        fake_engine = MagicMock()
        fake_engine.__class__.__name__ = "FakeEngine"

        with pytest.raises(TypeError, match="JaxLoss requires a JaxEngine"):
            JaxLoss(spec, fake_engine, [], _h2_ff())

    def test_water_energy_loss(self) -> None:
        """JaxLoss works for water (bond + angle params)."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        # Both scores are effectively zero at equilibrium; use absolute tolerance
        np.testing.assert_allclose(jax_score, python_score, atol=1e-10)


def _water_with_qm_refs(engine: object) -> tuple:
    """Build water molecule with computed 'QM' reference data.

    Computes MM hessian/frequencies at equilibrium params to serve
    as reference data, then returns a perturbed FF for non-trivial loss.
    """
    mol = make_water(bond_length=0.96, angle_deg=104.5)
    ff_ref = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)

    # Compute 'QM' hessian at equilibrium
    hess_qm = engine.hessian(mol, ff_ref)
    # Attach as QM hessian (needed for eigenmatrix)
    mol = mol.with_hessian(hess_qm)

    # Compute 'QM' frequencies
    from q2mm.models.hessian import hessian_to_frequencies

    freqs_qm = hessian_to_frequencies(hess_qm, list(mol.symbols))

    # Perturbed FF for non-trivial loss
    ff_pert = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=110.0)

    return mol, ff_ref, ff_pert, hess_qm, freqs_qm


class TestJaxLossFrequencyParity:
    """Frequency path: JaxLoss vs ObjectiveFunction parity."""

    def test_frequency_loss_parity(self) -> None:
        """JaxLoss and ObjectiveFunction agree on frequency loss."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, _hess, freqs_qm = _water_with_qm_refs(engine)

        # Add only the 3 real vibrational modes (indices 6, 7, 8)
        ref = ReferenceData()
        for i in range(6, 9):
            ref.add_frequency(freqs_qm[i], data_idx=i, weight=1.0, molecule_idx=0)

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-4)

    def test_frequency_gradient_shape_and_direction(self) -> None:
        """JaxLoss gradient has correct shape and points downhill."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, _hess, freqs_qm = _water_with_qm_refs(engine)

        ref = ReferenceData()
        for i in range(6, 9):
            ref.add_frequency(freqs_qm[i], data_idx=i, weight=1.0, molecule_idx=0)

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        loss, grad = jax_loss.loss_and_grad(params)

        assert grad.shape == params.shape
        assert np.linalg.norm(grad) > 0, "Gradient should be nonzero"

        # Small step in negative gradient direction should decrease loss.
        # Frequencies are large (cm⁻¹), so gradients are huge — use tiny step.
        lr = 1e-6 / np.linalg.norm(grad)
        step = params - lr * grad
        loss_after = jax_loss(step)
        assert loss_after < loss, "One step in -grad direction should lower loss"


class TestJaxLossHessianParity:
    """Hessian-element path: JaxLoss vs ObjectiveFunction parity."""

    def test_hessian_element_loss_parity(self) -> None:
        """JaxLoss and ObjectiveFunction agree on hessian-element loss."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, hess_qm, _freqs = _water_with_qm_refs(engine)

        ref = ReferenceData()
        ref.add_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            molecule_idx=0,
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-6)


class TestJaxLossEigenmatrixParity:
    """Eigenmatrix diagonal path: JaxLoss vs ObjectiveFunction parity."""

    def test_eigenmatrix_diagonal_loss_parity(self) -> None:
        """JaxLoss and ObjectiveFunction agree on eigenmatrix diagonal loss."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, hess_qm, _freqs = _water_with_qm_refs(engine)

        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(
            hess_qm,
            diagonal_only=True,
            molecule_idx=0,
            weights={"eig_i": 0.0, "eig_d_low": 0.1, "eig_d_high": 0.1, "eig_o": 0.0},
        )

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        assert python_score > 0, "Loss should be nonzero with perturbed params"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-6)


class TestJaxLossMixedObjective:
    """Multi-category loss: energy + frequency + hessian combined."""

    def test_mixed_loss_parity(self) -> None:
        """JaxLoss and ObjectiveFunction agree on combined energy+freq+hessian."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, hess_qm, freqs_qm = _water_with_qm_refs(engine)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        for i in range(6, 9):
            ref.add_frequency(freqs_qm[i], data_idx=i, weight=1.0, molecule_idx=0)
        ref.add_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            molecule_idx=0,
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

        assert python_score > 0, "Combined loss should be nonzero"
        np.testing.assert_allclose(jax_score, python_score, atol=1e-4)

    def test_mixed_gradient_lowers_loss(self) -> None:
        """One gradient step on combined loss reduces the score."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        mol, _ff_ref, ff_pert, hess_qm, freqs_qm = _water_with_qm_refs(engine)

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        for i in range(6, 9):
            ref.add_frequency(freqs_qm[i], data_idx=i, weight=1.0, molecule_idx=0)
        ref.add_hessian_from_matrix(
            hess_qm,
            diagonal_only=True,
            molecule_idx=0,
            diagonal_weight=0.1,
            offdiagonal_weight=0.0,
        )

        obj = ObjectiveFunction(forcefield=ff_pert.copy(), engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff_pert.copy())

        params = ff_pert.get_param_vector()
        loss0, grad = jax_loss.loss_and_grad(params)
        lr = 1e-6 / np.linalg.norm(grad)
        step = params - lr * grad
        loss1 = jax_loss(step)

        assert loss1 < loss0, "One gradient step should lower combined loss"


class TestJaxLossMultiMolecule:
    """Multi-molecule routing: H2 + water with separate energy refs."""

    def test_multi_molecule_energy_parity(self) -> None:
        """JaxLoss and ObjectiveFunction agree across 2 molecules."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()

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
        )

        ref = ReferenceData()
        ref.add_energy(value=0.0, molecule_idx=0, weight=1.0)
        ref.add_energy(value=0.0, molecule_idx=1, weight=1.0)

        molecules = [mol_h2, mol_water]
        obj = ObjectiveFunction(forcefield=ff.copy(), engine=engine, molecules=molecules, reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, molecules, ff.copy())

        params = ff.get_param_vector()
        python_score = obj(params)
        jax_score = jax_loss(params)

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

        engine = JaxEngine()
        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)

        # Compute Hessian + Jacobian via JaxEngine
        hess_au, dH_dp_kcal = engine.hessian_and_param_jacobian(mol, ff)
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


class TestJaxLossGeometryParity:
    """Tests for the geometry-references implicit-diff loss path."""

    def test_bond_length_relaxes_to_eq(self) -> None:
        """Relaxed H2 bond length matches the FF equilibrium parameter."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        # Start with a stretched bond; equilibrium r0 = 0.74 Å.
        mol = make_diatomic(distance=0.90, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        engine = JaxEngine()

        # Reference equals the eq value → relaxed loss should be ~0.
        ref = ReferenceData()
        ref.add_bond_length(value=0.74, molecule_idx=0, atom_indices=(0, 1), weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        loss = jax_loss(params)
        # relaxed bond ≈ 0.74, so (0.74 − 0.74)^2 should be ~0.
        assert loss < 1e-10

    def test_bond_length_grad_matches_fd(self) -> None:
        """∇_p loss for a bond-length ref matches finite differences."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_diatomic(distance=0.90, bond_tolerance=1.5)
        ff = _h2_ff(bond_k=359.7, bond_r0=0.74)
        engine = JaxEngine()

        # Reference offset from current r0 so the gradient is non-trivial.
        ref = ReferenceData()
        ref.add_bond_length(value=0.80, molecule_idx=0, atom_indices=(0, 1), weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        _, grad_jax = jax_loss.loss_and_grad(params)

        # Central-difference reference gradient (full-pipeline FD: each
        # eval relaxes the geometry at the perturbed parameters).
        eps = 1e-4
        grad_fd = np.zeros_like(params)
        for i in range(len(params)):
            dp = np.zeros_like(params)
            dp[i] = eps
            grad_fd[i] = (jax_loss(params + dp) - jax_loss(params - dp)) / (2 * eps)

        np.testing.assert_allclose(grad_jax, grad_fd, atol=1e-5, rtol=1e-3)

    def test_bond_angle_relaxes_to_eq(self) -> None:
        """Relaxed H2O bond angle matches the FF equilibrium parameter."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        # Start with a perturbed angle; eq is 104.5°.
        mol = make_water(bond_length=0.96, angle_deg=110.0)
        ff = _water_ff(bond_k=553.0, bond_r0=0.96, angle_k=49.9, angle_eq=104.5)
        engine = JaxEngine()

        ref = ReferenceData()
        # Atom indices: H(1)–O(0)–H(2) with O as vertex.
        ref.add_bond_angle(value=104.5, molecule_idx=0, atom_indices=(1, 0, 2), weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        loss = jax_loss(params)
        assert loss < 1e-6

    def test_geometry_grad_jit_callable(self) -> None:
        """JaxLoss.loss_and_grad with geometry refs is jit-compilable."""
        from q2mm.optimizers.jaxloss import JaxLoss
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mol = make_water(bond_length=0.97, angle_deg=104.5)
        ff = _water_ff()
        engine = JaxEngine()

        ref = ReferenceData()
        ref.add_bond_length(value=0.96, molecule_idx=0, atom_indices=(0, 1), weight=1.0)
        ref.add_bond_angle(value=104.5, molecule_idx=0, atom_indices=(1, 0, 2), weight=1.0)

        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)
        spec = obj.to_jax_spec()
        jax_loss = JaxLoss(spec, engine, [mol], ff)

        params = ff.get_param_vector()
        loss, grad = jax_loss.loss_and_grad(params)
        assert np.isfinite(loss)
        assert grad.shape == params.shape
        assert np.all(np.isfinite(grad))
