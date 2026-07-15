"""Tests for per-data-type evaluators.

Tests each evaluator independently using stub backends and known data.
"""

from __future__ import annotations

import numpy as np
import pytest

from test._shared import GS_FCHK, make_water, make_ethane

from q2mm.backends.contracts import (
    UnsupportedCapabilityError,
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyResult,
    EnergyUnit,
    FrequencyResult,
    FrequencyUnit,
    GeometryResult,
    HessianJacobianResult,
    HessianResult,
    HessianUnit,
    LengthUnit,
    ParameterGradientResult,
    readonly_array,
)
from q2mm.io.fchk import load_fchk_reference
from q2mm.models.observations import Observation, ObservationSet
from test.backend_fixtures import MockLayout

# ---- Prepared-session doubles ----

_PROV = BackendProvenance(backend="stub", role=BackendRole.MM)

#: Dummy full parameter vector — the fakes ignore it (they return fixed data).
P = np.zeros(2)


def _prep(backend: StubPrepared, mol: object) -> StubPrepared:
    """Attach *mol* to a stub prepared session and return it."""
    backend._molecule = mol
    return backend


class StubPrepared(AbstractPreparedBackend):
    """Minimal prepared-session double for testing evaluators in isolation."""

    def __init__(
        self,
        *,
        energy: float = 0.0,
        frequencies: list[float] | None = None,
        hessian: np.ndarray | None = None,
        minimize_coords: np.ndarray | None = None,
        minimize_symbols: list[str] | None = None,
    ) -> None:
        info = BackendInfo(
            name="stub",
            role=BackendRole.MM,
            capabilities=frozenset(
                {Capability.ENERGY, Capability.HESSIAN, Capability.FREQUENCIES, Capability.MINIMIZE}
            ),
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        super().__init__(info=info, case_id="0", molecule=None, force_field=None, layout=None)
        self._energy_value = energy
        self._freqs = frequencies or []
        self._hess = hessian
        self._minimize_coords = minimize_coords
        self._minimize_symbols = minimize_symbols or []

    def _energy(self, request: object) -> EnergyResult:
        return EnergyResult(energy=self._energy_value, unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROV)

    def _frequencies(self, request: object) -> FrequencyResult:
        return FrequencyResult(frequencies=readonly_array(self._freqs), unit=FrequencyUnit.INVERSE_CM, provenance=_PROV)

    def _hessian(self, request: object) -> HessianResult:
        if self._hess is None:
            raise ValueError("No hessian configured")
        return HessianResult(hessian=readonly_array(self._hess), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=_PROV)

    def _minimize(self, request: object) -> GeometryResult:
        if self._minimize_coords is None:
            raise ValueError("No minimize_coords configured")
        return GeometryResult(
            energy=0.0,
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=tuple(self._minimize_symbols),
            coordinates=readonly_array(self._minimize_coords),
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=_PROV,
        )


class GradStubPrepared(StubPrepared):
    """Stub prepared session that supports analytical parameter gradients."""

    def __init__(
        self,
        *,
        energy: float = 0.0,
        param_grad: np.ndarray | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(energy=energy, **kwargs)
        self._param_grad = param_grad if param_grad is not None else np.zeros(0)
        self._layout = MockLayout(len(self._param_grad))
        self._info = BackendInfo(
            name="grad_stub",
            role=BackendRole.MM,
            capabilities=self._info.capabilities | {Capability.PARAMETER_GRADIENT},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )

    def _parameter_gradient(self, request: object) -> ParameterGradientResult:
        return ParameterGradientResult(
            energy=self._energy_value,
            gradient=readonly_array(self._param_grad),
            unit=EnergyUnit.KCAL_PER_MOL,
            provenance=_PROV,
        )


class HessJacStubPrepared(StubPrepared):
    """Stub prepared session that supports Hessian parameter Jacobians."""

    def __init__(self, *, hessian: np.ndarray, hess_jac: np.ndarray, **kwargs: object) -> None:
        super().__init__(hessian=hessian, **kwargs)
        self._hess_jac = hess_jac
        self._layout = MockLayout(np.asarray(hess_jac).shape[2])
        self._info = BackendInfo(
            name="hessjac_stub",
            role=BackendRole.MM,
            capabilities=self._info.capabilities | {Capability.HESSIAN_PARAMETER_JACOBIAN},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )

    def _hessian_parameter_jacobian(self, request: object) -> HessianJacobianResult:
        return HessianJacobianResult(
            hessian=readonly_array(self._hess),
            jacobian=readonly_array(self._hess_jac),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROV,
        )


class StubBackend:
    """Minimal backend double whose ``prepare`` returns a stub prepared session."""

    def __init__(self, prepared: StubPrepared) -> None:
        self._prepared = prepared
        self.info = prepared.info

    def prepare(self, request: object) -> StubPrepared:
        self._prepared._molecule = request.molecule  # type: ignore[attr-defined]
        return self._prepared


# ---- EnergyEvaluator tests ----


class TestEnergyEvaluator:
    def test_compute(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        backend = StubPrepared(energy=42.5)
        mol = make_water()

        result = evaluator.compute(_prep(backend, mol), P)
        assert result.energy == 42.5

    def test_residuals_single(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator, EnergyResult

        evaluator = EnergyEvaluator()
        computed = EnergyResult(energy=10.0)
        refs = [Observation(kind="energy", value=12.0, weight=2.0)]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(2.0 * (12.0 - 10.0))

    def test_residuals_multiple(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator, EnergyResult

        evaluator = EnergyEvaluator()
        computed = EnergyResult(energy=5.0)
        refs = [
            Observation(kind="energy", value=5.0, weight=1.0),
            Observation(kind="energy", value=7.0, weight=0.5),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(0.0)
        assert residuals[1] == pytest.approx(0.5 * 2.0)

    def test_extract_value(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        calc = {"energy": 99.9}
        ref = Observation(kind="energy", value=0.0)
        assert EnergyEvaluator.extract_value(calc, ref) == 99.9

    def test_compute_with_structure(self) -> None:
        """When structure is provided, it should be passed to backend."""
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        backend = StubPrepared(energy=1.0)
        mol = make_water()

        result = evaluator.compute(_prep(backend, mol), P)
        assert result.energy == 1.0


# ---- FrequencyEvaluator tests ----


class TestFrequencyEvaluator:
    def test_compute(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        evaluator = FrequencyEvaluator()
        backend = StubPrepared(frequencies=[100.0, 200.0, 300.0])
        mol = make_water()

        result = evaluator.compute(_prep(backend, mol), P)
        assert result.frequencies == [100.0, 200.0, 300.0]

    def test_residuals(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator, FrequencyResult

        evaluator = FrequencyEvaluator()
        computed = FrequencyResult(frequencies=[100.0, 200.0, 300.0])
        refs = [
            Observation(kind="frequency", value=105.0, weight=1.0, data_idx=0),
            Observation(kind="frequency", value=195.0, weight=2.0, data_idx=1),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(1.0 * (105.0 - 100.0))
        assert residuals[1] == pytest.approx(2.0 * (195.0 - 200.0))

    def test_residuals_out_of_range(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator, FrequencyResult

        evaluator = FrequencyEvaluator()
        computed = FrequencyResult(frequencies=[100.0])
        refs = [Observation(kind="frequency", value=200.0, data_idx=5)]

        with pytest.raises(IndexError, match="data_idx=5"):
            evaluator.residuals(computed, refs)

    def test_extract_value(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        calc = {"frequencies": [100.0, 200.0, 300.0]}
        ref = Observation(kind="frequency", value=0.0, data_idx=2)
        assert FrequencyEvaluator.extract_value(calc, ref) == 300.0


# ---- GeometryEvaluator tests ----


class TestGeometryEvaluator:
    def test_dihedral_angle_staggered(self) -> None:
        """Staggered ethane dihedral ≈ 60°."""
        from q2mm.optimizers.evaluators.geometry import dihedral_angle

        ethane = make_ethane()
        coords = ethane.geometry

        # H2-C0-C1-H5 (indices 2, 0, 1, 5) — should be ~60° or ~180°
        angle = dihedral_angle(coords[2], coords[0], coords[1], coords[5])
        # Accept any 60° multiple (staggered)
        abs_angle = abs(angle)
        assert abs_angle > 50 or abs(abs_angle - 180) < 10

    def test_dihedral_angle_collinear(self) -> None:
        """Collinear atoms should return 0.0 (degenerate)."""
        from q2mm.optimizers.evaluators.geometry import dihedral_angle

        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        p3 = np.array([3.0, 0.0, 0.0])
        assert dihedral_angle(p0, p1, p2, p3) == 0.0

    def test_residuals_bond_length(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator, GeometryResult

        evaluator = GeometryEvaluator()
        computed = GeometryResult(
            bond_lengths=[1.0, 1.5],
            bond_lengths_by_atoms={(0, 1): 1.0, (1, 2): 1.5},
        )
        refs = [
            Observation(kind="bond_length", value=1.1, weight=10.0, atom_indices=(0, 1)),
        ]
        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(10.0 * (1.1 - 1.0))

    def test_residuals_bond_angle(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator, GeometryResult

        evaluator = GeometryEvaluator()
        computed = GeometryResult(
            bond_angles=[109.5],
            bond_angles_by_atoms={(0, 1, 2): 109.5},
        )
        refs = [
            Observation(kind="bond_angle", value=110.0, weight=5.0, atom_indices=(0, 1, 2)),
        ]
        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(5.0 * 0.5)

    def test_residuals_torsion_wrapping(self) -> None:
        """Torsion residuals should wrap around 360°."""
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator, GeometryResult

        evaluator = GeometryEvaluator()
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [3.0, 1.0, 1.0],
            ]
        )
        computed = GeometryResult(torsion_coords=coords)
        refs = [
            Observation(
                kind="torsion_angle",
                value=170.0,
                weight=1.0,
                atom_indices=(0, 1, 2, 3),
            ),
        ]
        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        # Diff should be wrapped to [-180, 180]
        assert -180.0 <= residuals[0] <= 180.0

    def test_extract_value_bond_by_atoms(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        calc = {
            "bond_lengths": [1.0, 1.5],
            "bond_lengths_by_atoms": {(0, 1): 1.0, (1, 2): 1.5},
        }
        ref = Observation(kind="bond_length", value=0.0, atom_indices=(1, 2))
        assert GeometryEvaluator.extract_value(calc, ref) == 1.5

    def test_extract_value_bond_by_idx(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        calc = {"bond_lengths": [1.0, 1.5], "bond_lengths_by_atoms": {}}
        ref = Observation(kind="bond_length", value=0.0, data_idx=1)
        assert GeometryEvaluator.extract_value(calc, ref) == 1.5

    def test_extract_value_angle_reverse_order(self) -> None:
        """Should try both (i,j,k) and (k,j,i) orderings."""
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        calc = {
            "bond_angles": [109.5],
            "bond_angles_by_atoms": {(2, 1, 0): 109.5},
        }
        ref = Observation(kind="bond_angle", value=0.0, atom_indices=(0, 1, 2))
        assert GeometryEvaluator.extract_value(calc, ref) == 109.5


# ---- EigenmatrixEvaluator tests ----


class TestEigenmatrixEvaluator:
    def test_compute_self_projection(self) -> None:
        """Self-projection (MM == QM) should produce a diagonal eigenmatrix."""
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.models.molecule import Molecule

        hess = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mol = Molecule(
            symbols=["H"],
            geometry=np.array([[0.0, 0.0, 0.0]]),
            name="stub",
            hessian=hess,
        )
        backend = StubPrepared(hessian=hess)
        evaluator = EigenmatrixEvaluator()

        result = evaluator.compute(_prep(backend, mol), P, mol_idx=0)
        eigmat = result.eigenmatrix

        # Self-projection → off-diagonal should be ~0
        off_diag = eigmat - np.diag(np.diag(eigmat))
        assert np.allclose(off_diag, 0, atol=1e-12)

    def test_residuals_diagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator, EigenmatrixResult

        evaluator = EigenmatrixEvaluator()
        eigmat = np.diag([1.0, 2.0, 3.0])
        computed = EigenmatrixResult(eigenmatrix=eigmat)
        refs = [
            Observation(kind="eig_diagonal", value=1.5, weight=0.1, data_idx=0),
            Observation(kind="eig_diagonal", value=2.0, weight=0.1, data_idx=1),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(0.1 * (1.5 - 1.0))
        assert residuals[1] == pytest.approx(0.1 * (2.0 - 2.0))

    def test_residuals_offdiagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator, EigenmatrixResult

        evaluator = EigenmatrixEvaluator()
        eigmat = np.array([[1.0, 0.3], [0.3, 2.0]])
        computed = EigenmatrixResult(eigenmatrix=eigmat)
        refs = [
            Observation(kind="eig_offdiagonal", value=0.0, weight=0.05, atom_indices=(1, 0)),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(0.05 * (0.0 - 0.3))

    def test_eigenvector_caching(self) -> None:
        """QM eigenvectors should be computed once and cached."""
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.models.molecule import Molecule

        _rng = np.random.default_rng(42)
        _a = _rng.standard_normal((6, 6))
        hess = _a @ _a.T + np.eye(6)
        mol = Molecule(
            symbols=["H", "H"],
            geometry=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
            name="stub",
            hessian=hess,
        )
        backend = StubPrepared(hessian=hess)
        evaluator = EigenmatrixEvaluator()

        evaluator.compute(_prep(backend, mol), P, mol_idx=0)
        assert 0 in evaluator._qm_eigenvectors

        evaluator.compute(_prep(backend, mol), P, mol_idx=0)
        # Same key, same cached value
        assert 0 in evaluator._qm_eigenvectors

    def test_reset_clears_cache(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        evaluator = EigenmatrixEvaluator()
        evaluator._qm_eigenvectors[0] = np.eye(3)
        evaluator.reset()
        assert len(evaluator._qm_eigenvectors) == 0

    def test_no_qm_hessian_raises(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.models.molecule import Molecule

        mol = Molecule(
            symbols=["H"],
            geometry=np.array([[0.0, 0.0, 0.0]]),
            name="stub",
        )
        backend = StubPrepared(hessian=np.eye(3))
        evaluator = EigenmatrixEvaluator()

        with pytest.raises(ValueError, match="no QM Hessian"):
            evaluator.compute(_prep(backend, mol), P, mol_idx=0)

    def test_extract_value_diagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        calc = {"eigenmatrix": np.diag([1.0, 2.0, 3.0])}
        ref = Observation(kind="eig_diagonal", value=0.0, data_idx=2)
        assert EigenmatrixEvaluator.extract_value(calc, ref) == 3.0

    def test_extract_value_offdiagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        mat = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 3.0]])
        calc = {"eigenmatrix": mat}
        ref = Observation(kind="eig_offdiagonal", value=0.0, atom_indices=(2, 1))
        assert EigenmatrixEvaluator.extract_value(calc, ref) == 0.3


# ---- Evaluator Protocol compliance ----


class TestProtocolCompliance:
    def test_energy_evaluator_is_evaluator(self) -> None:
        from q2mm.optimizers.evaluators import Evaluator
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        assert isinstance(EnergyEvaluator(), Evaluator)

    def test_frequency_evaluator_is_evaluator(self) -> None:
        from q2mm.optimizers.evaluators import Evaluator
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        assert isinstance(FrequencyEvaluator(), Evaluator)

    def test_geometry_evaluator_is_evaluator(self) -> None:
        from q2mm.optimizers.evaluators import Evaluator
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        assert isinstance(GeometryEvaluator(), Evaluator)

    def test_eigenmatrix_evaluator_is_evaluator(self) -> None:
        from q2mm.optimizers.evaluators import Evaluator
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        assert isinstance(EigenmatrixEvaluator(), Evaluator)


# ---- Evaluator gradient tests ----


class TestEnergyEvaluatorGradient:
    def test_supports_analytical_gradient_true(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        backend = GradStubPrepared(energy=1.0, param_grad=np.array([1.0, 2.0]))
        assert evaluator.supports_analytical_gradient(backend) is True

    def test_supports_analytical_gradient_false(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        backend = StubPrepared(energy=1.0)
        assert evaluator.supports_analytical_gradient(backend) is False

    def test_gradient_single_ref(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        de_dp = np.array([3.0, -1.0])
        backend = GradStubPrepared(energy=10.0, param_grad=de_dp)
        mol = make_water()
        refs = [Observation(kind="energy", value=12.0, weight=2.0)]

        grad = evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=2)
        # d(score)/d(p) = -2 * w^2 * (ref - calc) * dE/dp
        # = -2 * 4.0 * 2.0 * [3.0, -1.0] = [-48.0, 16.0]
        expected = -2.0 * 2.0**2 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(grad, expected)

    def test_gradient_multiple_refs(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        de_dp = np.array([1.0, 0.5])
        backend = GradStubPrepared(energy=5.0, param_grad=de_dp)
        mol = make_water()
        refs = [
            Observation(kind="energy", value=5.0, weight=1.0),
            Observation(kind="energy", value=7.0, weight=0.5),
        ]

        grad = evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=2)
        expected = -2.0 * 0.5**2 * (7.0 - 5.0) * de_dp
        np.testing.assert_allclose(grad, expected)

    def test_gradient_raises_on_unsupported_engine(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        backend = StubPrepared(energy=1.0)
        mol = make_water()
        refs = [Observation(kind="energy", value=2.0, weight=1.0)]

        with pytest.raises(UnsupportedCapabilityError):
            evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=1)

    def test_gradient_validates_de_dp_shape(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        # Engine returns 2 derivatives but caller expects 3
        de_dp = np.array([1.0, 2.0])
        backend = GradStubPrepared(energy=5.0, param_grad=de_dp)
        mol = make_water()
        refs = [Observation(kind="energy", value=6.0, weight=1.0)]

        with pytest.raises(ValueError, match="returned 2 derivatives but expected 3"):
            evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=3)


class TestFrequencyEvaluatorGradient:
    def test_supports_analytical_gradient_false(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        evaluator = FrequencyEvaluator()
        backend = GradStubPrepared(energy=0.0, param_grad=np.array([1.0]))
        assert evaluator.supports_analytical_gradient(backend) is False

    def test_gradient_raises_on_unsupported_engine(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        evaluator = FrequencyEvaluator()
        backend = StubPrepared(frequencies=[100.0])
        mol = make_water()
        refs = [Observation(kind="frequency", value=105.0, data_idx=0)]

        with pytest.raises(UnsupportedCapabilityError):
            evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=1)


class TestGeometryEvaluatorGradient:
    def test_supports_analytical_gradient_false(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        evaluator = GeometryEvaluator()
        backend = GradStubPrepared(energy=0.0, param_grad=np.array([1.0]))
        assert evaluator.supports_analytical_gradient(backend) is False

    def test_gradient_returns_none(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        evaluator = GeometryEvaluator()
        backend = StubPrepared()
        mol = make_water()
        refs = [Observation(kind="bond_length", value=1.0, atom_indices=(0, 1))]

        result = evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=1)
        assert result is None


class TestEigenmatrixEvaluatorGradient:
    def test_supports_analytical_gradient_false(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        evaluator = EigenmatrixEvaluator()
        backend = GradStubPrepared(energy=0.0, param_grad=np.array([1.0]))
        assert evaluator.supports_analytical_gradient(backend) is False

    def test_gradient_raises_on_unsupported_engine(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        evaluator = EigenmatrixEvaluator()
        backend = StubPrepared()
        mol = make_water()
        refs = [Observation(kind="eig_diagonal", value=1.0, data_idx=0)]

        with pytest.raises(UnsupportedCapabilityError):
            evaluator.gradient(_prep(backend, mol), P, references=refs, n_params=1)


# ---- ObjectiveFunction.gradient() delegation tests ----


class TestObjectiveFunctionGradient:
    def test_energy_only_gradient_delegates_to_evaluator(self) -> None:
        """Energy-only gradient should use analytical evaluator gradient."""
        from q2mm.optimizers.objective import ObjectiveFunction

        de_dp = np.array([3.0, -1.0])
        backend = GradStubPrepared(energy=10.0, param_grad=de_dp)
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_energy(12.0, weight=2.0)

        # Need a mock forcefield with with_params
        ff = _StubForceField(n_params=2)
        layout = _StubLayout(ff.n_params)
        obj = ObjectiveFunction(
            forcefield=ff,
            backend=StubBackend(backend),
            molecules=[mol],
            reference=ref,
            layout=layout,
        )

        grad = obj.gradient(np.array([0.0, 0.0]))
        expected = -2.0 * 2.0**2 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(grad, expected)

    def test_mixed_refs_uses_fd_fallback_for_frequency(self) -> None:
        """Mixed energy+frequency refs should use FD for frequency part."""
        from q2mm.optimizers.objective import ObjectiveFunction

        # Engine supports analytical gradients (for energy)
        # but frequency evaluator always falls back to FD
        de_dp = np.array([1.0])
        backend = GradStubPrepared(
            energy=10.0,
            param_grad=de_dp,
            frequencies=[100.0],
        )
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_energy(10.0, weight=1.0)  # zero diff → zero energy grad
        ref = ref.with_frequency(100.0, data_idx=0, weight=1.0)  # zero diff → zero freq grad

        ff = _StubForceField(n_params=1)
        layout = _StubLayout(ff.n_params)
        obj = ObjectiveFunction(
            forcefield=ff,
            backend=StubBackend(backend),
            molecules=[mol],
            reference=ref,
            layout=layout,
        )

        # Should NOT raise — it should fall back to FD for frequency
        grad = obj.gradient(np.array([0.0]))
        # Both diffs are zero, so gradient should be ~zero
        np.testing.assert_allclose(grad, [0.0], atol=1e-6)

    def test_mixed_refs_nonzero_gradient(self) -> None:
        """Mixed energy+frequency with non-zero residuals produces correct gradient."""
        from q2mm.optimizers.objective import ObjectiveFunction

        de_dp = np.array([2.0])
        backend = GradStubPrepared(
            energy=10.0,
            param_grad=de_dp,
            frequencies=[100.0],
        )
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_energy(15.0, weight=1.0)
        ref = ref.with_frequency(100.0, data_idx=0, weight=1.0)  # zero diff → zero FD contribution

        ff = _StubForceField(n_params=1)
        layout = _StubLayout(ff.n_params)
        obj = ObjectiveFunction(
            forcefield=ff,
            backend=StubBackend(backend),
            molecules=[mol],
            reference=ref,
            layout=layout,
        )

        grad = obj.gradient(np.array([0.0]))
        # Energy part (analytical): -2 * 1^2 * (15 - 10) * 2.0 = -20.0
        # Frequency part (FD): stub returns constant → FD ≈ 0
        expected_energy_grad = -2.0 * 1.0**2 * (15.0 - 10.0) * de_dp
        np.testing.assert_allclose(grad, expected_energy_grad, atol=1e-6)
        assert grad[0] != 0.0

    def test_objective_gradient_uses_prepared_parameter_gradient(self) -> None:
        """The objective's analytical gradient routes through prepared.parameter_gradient."""
        from q2mm.optimizers.objective import ObjectiveFunction

        de_dp = np.array([1.0])
        backend = GradStubPrepared(energy=10.0, param_grad=de_dp)
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_energy(12.0, weight=1.0)

        ff = _StubForceField(n_params=1)
        layout = _StubLayout(ff.n_params)
        obj = ObjectiveFunction(
            forcefield=ff,
            backend=StubBackend(backend),
            molecules=[mol],
            reference=ref,
            layout=layout,
        )

        grad = obj.gradient(np.array([0.0]))
        expected = -2.0 * 1.0**2 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(grad, expected)


class _StubForceField:
    """Minimal stub for ForceField used in gradient tests."""

    def __init__(self, n_params: int = 1) -> None:
        self.n_params = n_params

    def with_params(self, param_vector: np.ndarray) -> _StubForceField:
        return self


class _StubLayout:
    """Minimal ParameterLayout-like stub for ObjectiveFunction tests."""

    def __init__(self, n_params: int) -> None:
        self._n_params = n_params
        self.bounds = np.zeros((n_params, 2), dtype=float)
        self.steps = np.ones(n_params, dtype=float)

    def __len__(self) -> int:
        return self._n_params

    def vector(self, force_field: _StubForceField) -> np.ndarray:
        return np.zeros(force_field.n_params, dtype=float)

    def replace(self, force_field: _StubForceField, vector: np.ndarray) -> _StubForceField:
        return force_field


# ---- Parsers ----


class TestFchkParser:
    """Tests that FCHK loading emits the canonical molecule."""

    def test_load_fchk_from_io_boundary(self) -> None:
        from q2mm.io.fchk import load_fchk

        molecule = load_fchk(GS_FCHK)
        assert len(molecule.symbols) == 8
        assert molecule.symbols.count("H") == 6
        assert molecule.symbols.count("C") == 2
        assert molecule.geometry.shape == (8, 3)
        assert molecule.charge == 0
        assert molecule.multiplicity == 1
        assert molecule.hessian is not None
        assert molecule.hessian.shape == (24, 24)
        assert molecule.hessian_provenance.source == "fchk"
        np.testing.assert_allclose(molecule.hessian, molecule.hessian.T, atol=1e-15)


# ---- Integration: evaluators produce same results as old ObjectiveFunction ----


class TestEvaluatorObjectiveParity:
    """Verify that evaluator delegation in ObjectiveFunction produces identical results."""

    def test_energy_parity(self) -> None:
        """Energy evaluation via evaluator matches direct backend call."""
        from q2mm.optimizers.objective import ObjectiveFunction

        backend = StubPrepared(energy=42.0)
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_energy(40.0, weight=2.0)

        obj = ObjectiveFunction(forcefield=None, backend=StubBackend(backend), molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0, P)

        assert result["energy"] == 42.0
        calc_value = obj._extract_value(result, ref.values[0])
        assert calc_value == 42.0

    def test_frequency_parity(self) -> None:
        """Frequency evaluation via evaluator matches direct backend call."""
        from q2mm.optimizers.objective import ObjectiveFunction

        backend = StubPrepared(frequencies=[100.0, 200.0, 300.0])
        mol = make_water()
        ref = ObservationSet()
        ref = ref.with_frequency(105.0, data_idx=0, weight=1.0)
        ref = ref.with_frequency(195.0, data_idx=1, weight=1.0)

        obj = ObjectiveFunction(forcefield=None, backend=StubBackend(backend), molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0, P)

        assert result["frequencies"] == [100.0, 200.0, 300.0]
        assert obj._extract_value(result, ref.values[0]) == 100.0
        assert obj._extract_value(result, ref.values[1]) == 200.0

    def test_eigenmatrix_parity(self) -> None:
        """Eigenmatrix evaluation via evaluator matches old implementation."""
        from q2mm.optimizers.objective import ObjectiveFunction

        _ref_data, mol = load_fchk_reference(str(GS_FCHK))
        assert mol.hessian is not None
        qm_hessian = np.array(mol.hessian, dtype=float)

        ref = ObservationSet()
        ref = ref.with_eigenmatrix_from_hessian(qm_hessian, diagonal_only=True)

        backend = StubPrepared(hessian=qm_hessian)
        obj = ObjectiveFunction(forcefield=None, backend=StubBackend(backend), molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0, P)

        assert "eigenmatrix" in result
        eigmat = np.array(result["eigenmatrix"], dtype=float)
        # Self-projection should be diagonal
        diag_only = np.diag(np.diag(eigmat))
        assert np.allclose(eigmat, diag_only, atol=1e-8)
