"""Tests for analytical Hessian-based gradients.

Uses synthetic Hessians and mock prepared backends to validate the pure Hessian
sensitivity helpers plus PythonObjectiveExecutor gradients for frequency,
hessian_element, and eigenmatrix observations.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    FrequencyResult,
    FrequencyUnit,
    HessianJacobianResult,
    HessianResult,
    HessianUnit,
    readonly_array,
)
from q2mm.models.hessian import (
    frequency_param_jacobian,
    hessian_to_frequencies,
    mass_weighted_eigenmatrix,
    mass_weighted_normal_modes,
)
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor
from test.backend_fixtures import MockLayout

P = np.zeros(2)
_PROV = BackendProvenance(backend="mock", role=BackendRole.MM)


class _FakePrepared(AbstractPreparedBackend):
    """Prepared-session double returning fixed Hessian/Jacobian data."""

    def __init__(
        self, *, hess: object = None, dH_dp: object = None, supports_jac: bool = True, molecule: object = None
    ) -> None:
        caps = {Capability.HESSIAN, Capability.FREQUENCIES}
        if supports_jac:
            caps.add(Capability.HESSIAN_PARAMETER_JACOBIAN)
        info = BackendInfo(
            name="mock",
            role=BackendRole.MM,
            capabilities=frozenset(caps),
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        n_params = np.asarray(dH_dp).shape[2] if dH_dp is not None else 2
        super().__init__(info=info, case_id="0", molecule=molecule, force_field=None, layout=MockLayout(n_params))
        self._h = np.asarray(hess, dtype=float) if hess is not None else None
        self._j = np.asarray(dH_dp, dtype=float) if dH_dp is not None else None

    def _hessian(self, request: object) -> HessianResult:
        return HessianResult(hessian=readonly_array(self._h), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=_PROV)

    def _frequencies(self, request: object) -> FrequencyResult:
        return FrequencyResult(
            frequencies=readonly_array(hessian_to_frequencies(self._h, self._molecule.symbols)),
            unit=FrequencyUnit.INVERSE_CM,
            provenance=_PROV,
        )

    def _hessian_parameter_jacobian(self, request: object) -> HessianJacobianResult:
        return HessianJacobianResult(
            hessian=readonly_array(self._h),
            jacobian=readonly_array(self._j),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROV,
        )


class _FakeBackend:
    def __init__(self, prepared: _FakePrepared) -> None:
        self._prepared = prepared
        self.info = prepared.info

    def prepare(self, request: object) -> _FakePrepared:
        self._prepared._molecule = request.molecule  # type: ignore[attr-defined]
        self._prepared._case_id = request.case_id  # type: ignore[attr-defined]
        return self._prepared


class _StubForceField:
    def __init__(self, n_params: int) -> None:
        self.n_params = n_params


class _StubLayout:
    def __init__(self, n_params: int) -> None:
        self._n = n_params
        self.bounds = np.zeros((n_params, 2), dtype=float)
        self.steps = np.ones(n_params, dtype=float)

    def __len__(self) -> int:
        return self._n

    def vector(self, force_field: _StubForceField) -> np.ndarray:
        return np.zeros(force_field.n_params, dtype=float)

    def replace(self, force_field: _StubForceField, vector: np.ndarray) -> _StubForceField:
        return force_field


def _make_symmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((n, n))
    return a @ a.T + np.eye(n) * 0.1


def _make_mock_engine(hess: np.ndarray, dH_dp: np.ndarray) -> _FakePrepared:
    return _FakePrepared(hess=hess, dH_dp=dH_dp, supports_jac=True)


def _make_mol(symbols: list[str], hessian: np.ndarray | None = None) -> object:
    """Return a real Molecule (ObjectivePlan validates Molecule instances)."""
    from q2mm.models.molecule import Molecule

    geometry = np.zeros((len(symbols), 3), dtype=float)
    geometry[:, 0] = np.arange(len(symbols), dtype=float) * 10.0
    return Molecule(symbols=tuple(symbols), geometry=geometry, hessian=hessian, name="test_mol")


def _objective(
    prepared: _FakePrepared, mol: object, refs: list[Observation]
) -> tuple[PythonObjectiveExecutor, np.ndarray]:
    n = len(prepared.layout)
    ff = _StubForceField(n)
    layout = _StubLayout(n)
    space = ActiveParameterSpace(layout=layout, baseline=np.zeros(n), active_indices=np.arange(n))
    plan = ObjectivePlan(
        case_ids=("0",),
        molecules=(mol,),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ObservationSet(values=tuple(refs)),
        layout=layout,
        active_space=space,
    )
    return PythonObjectiveExecutor(plan, _FakeBackend(prepared), ff, gradient_mode=GradientMode.ANALYTICAL), np.zeros(n)


def _grad(backend: _FakePrepared, mol: object, refs: list[Observation]) -> np.ndarray:
    obj, x = _objective(backend, mol, refs)
    return obj.gradient(x)


class TestFrequencyParamJacobian:
    def test_basic_shape(self) -> None:
        rng = np.random.default_rng(42)
        n_atoms = 3
        n3 = 3 * n_atoms
        n_params = 5
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        freqs, d_freq_dp = frequency_param_jacobian(hess, dH_dp, ["C", "H", "H"])
        assert len(freqs) == n3
        assert d_freq_dp.shape == (n3, n_params)

    def test_sorted_output(self) -> None:
        rng = np.random.default_rng(123)
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        for j in range(2):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        freqs, _ = frequency_param_jacobian(hess, dH_dp, ["C", "H"])
        assert freqs == sorted(freqs)

    def test_unsorted_output(self) -> None:
        rng = np.random.default_rng(456)
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        for j in range(2):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        freqs_sorted, _ = frequency_param_jacobian(hess, dH_dp, ["C", "H"], sort=True)
        freqs_unsorted, _ = frequency_param_jacobian(hess, dH_dp, ["C", "H"], sort=False)
        np.testing.assert_allclose(sorted(freqs_unsorted), freqs_sorted, rtol=1e-12)

    def test_frequencies_match_hessian_to_frequencies(self) -> None:
        rng = np.random.default_rng(789)
        hess = _make_symmetric(9, rng)
        dH_dp = rng.standard_normal((9, 9, 3))
        for j in range(3):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        freqs_new, _ = frequency_param_jacobian(hess, dH_dp, ["C", "H", "O"])
        freqs_ref = hessian_to_frequencies(hess, ["C", "H", "O"])
        np.testing.assert_allclose(freqs_new, freqs_ref, rtol=1e-10)

    def test_jacobian_vs_finite_difference(self) -> None:
        rng = np.random.default_rng(1001)
        n3 = 6
        n_params = 3
        hess = _make_symmetric(n3, rng)
        dH_dp = rng.standard_normal((n3, n3, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        symbols = ["C", "H"]
        _, d_freq_dp = frequency_param_jacobian(hess, dH_dp, symbols)
        delta = 1e-6
        for j in range(n_params):
            hess_plus = hess + delta * dH_dp[:, :, j]
            hess_minus = hess - delta * dH_dp[:, :, j]
            fd_deriv = (
                np.array(hessian_to_frequencies(hess_plus, symbols))
                - np.array(hessian_to_frequencies(hess_minus, symbols))
            ) / (2 * delta)
            np.testing.assert_allclose(d_freq_dp[:, j], fd_deriv, rtol=1e-4, atol=1e-4)

    def test_zero_jacobian_gives_zero_freq_jacobian(self) -> None:
        rng = np.random.default_rng(2002)
        hess = _make_symmetric(6, rng)
        _, d_freq_dp = frequency_param_jacobian(hess, np.zeros((6, 6, 2)), ["C", "H"])
        np.testing.assert_allclose(d_freq_dp, 0.0, atol=1e-15)


class TestFrequencyExecutorGradient:
    def test_supports_analytical_gradient_true(self) -> None:
        hess = np.eye(6)
        dH_dp = np.zeros((6, 6, 2))
        obj, _ = _objective(
            _FakePrepared(hess=hess, dH_dp=dH_dp, supports_jac=True),
            _make_mol(["C", "H"]),
            [Observation(kind="frequency", value=0.0, data_idx=0)],
        )
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_supports_analytical_gradient_false(self) -> None:
        hess = np.eye(6)
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _objective(
                _FakePrepared(hess=hess, supports_jac=False),
                _make_mol(["C", "H"]),
                [Observation(kind="frequency", value=0.0, data_idx=0)],
            )

    def test_gradient_shape(self) -> None:
        rng = np.random.default_rng(3003)
        n_params = 4
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        freqs = hessian_to_frequencies(hess, ["C", "H"])
        refs = [Observation(kind="frequency", value=freqs[3] + 10.0, weight=1.0, data_idx=3, case_id="0")]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"]), refs)
        assert grad.shape == (n_params,)

    def test_gradient_vs_finite_difference(self) -> None:
        rng = np.random.default_rng(4004)
        n_params = 3
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        for j in range(n_params):
            dH_dp[:, :, j] = 0.5 * (dH_dp[:, :, j] + dH_dp[:, :, j].T)
        symbols = ["C", "H"]
        freqs = hessian_to_frequencies(hess, symbols)
        refs = [
            Observation(kind="frequency", value=freqs[2] + 5.0, weight=1.5, data_idx=2, case_id="0"),
            Observation(kind="frequency", value=freqs[4] - 3.0, weight=0.8, data_idx=4, case_id="0"),
        ]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(symbols), refs)
        delta = 1e-6
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            for sign, coeff in [(1, 1.0), (-1, -1.0)]:
                h_pert = hess + sign * delta * dH_dp[:, :, j]
                f_pert = hessian_to_frequencies(h_pert, symbols)
                score = sum((r.weight * (r.value - f_pert[r.data_idx])) ** 2 for r in refs)
                fd_grad[j] += coeff * score
            fd_grad[j] /= 2 * delta
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-3, atol=1e-6)


class TestHessianElementExecutorGradient:
    def test_supports_analytical_gradient(self) -> None:
        hess = np.eye(6)
        dH_dp = np.zeros((6, 6, 2))
        obj, _ = _objective(
            _FakePrepared(hess=hess, dH_dp=dH_dp, supports_jac=True),
            _make_mol(["C", "H"]),
            [Observation(kind="hessian_element", value=0.0, atom_indices=(0, 0))],
        )
        assert obj.gradient_mode is GradientMode.ANALYTICAL
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _objective(
                _FakePrepared(hess=hess, supports_jac=False),
                _make_mol(["C", "H"]),
                [Observation(kind="hessian_element", value=0.0, atom_indices=(0, 0))],
            )

    def test_gradient_shape(self) -> None:
        rng = np.random.default_rng(5005)
        n_params = 3
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        refs = [
            Observation(kind="hessian_element", value=hess[1, 2] + 0.01, weight=1.0, case_id="0", atom_indices=(1, 2))
        ]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"]), refs)
        assert grad.shape == (n_params,)

    def test_gradient_vs_finite_difference(self) -> None:
        rng = np.random.default_rng(6006)
        n_params = 2
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        row, col = 2, 3
        ref_val = hess[row, col] + 0.005
        w = 2.0
        refs = [Observation(kind="hessian_element", value=ref_val, weight=w, case_id="0", atom_indices=(row, col))]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"]), refs)
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            score_p = (w * (ref_val - h_p[row, col])) ** 2
            score_m = (w * (ref_val - h_m[row, col])) ** 2
            fd_grad[j] = (score_p - score_m) / (2 * delta)
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)


class TestEigenmatrixExecutorGradient:
    def test_supports_analytical_gradient(self) -> None:
        hess = np.eye(6)
        dH_dp = np.zeros((6, 6, 2))
        mol = _make_mol(["C", "H"], hessian=hess)
        obj, _ = _objective(
            _FakePrepared(hess=hess, dH_dp=dH_dp, supports_jac=True),
            mol,
            [Observation(kind="eig_diagonal", value=0.0, data_idx=0)],
        )
        assert obj.gradient_mode is GradientMode.ANALYTICAL
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _objective(
                _FakePrepared(hess=hess, supports_jac=False),
                mol,
                [Observation(kind="eig_diagonal", value=0.0, data_idx=0)],
            )

    def test_gradient_diagonal_shape(self) -> None:
        rng = np.random.default_rng(7007)
        n_params = 3
        hess = _make_symmetric(6, rng)
        qm_hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        _, qm_evecs = mass_weighted_normal_modes(qm_hess, ["C", "H"])
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, ["C", "H"])
        refs = [Observation(kind="eig_diagonal", value=eigmat[2, 2] + 0.01, weight=1.0, data_idx=2, case_id="0")]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"], hessian=qm_hess), refs)
        assert grad.shape == (n_params,)

    def test_gradient_diagonal_vs_fd(self) -> None:
        rng = np.random.default_rng(8008)
        n_params = 2
        hess = _make_symmetric(6, rng)
        qm_hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        symbols = ["C", "H"]
        _, qm_evecs = mass_weighted_normal_modes(qm_hess, symbols)
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, symbols)
        idx = 3
        ref_val = eigmat[idx, idx] + 0.005
        w = 1.5
        refs = [Observation(kind="eig_diagonal", value=ref_val, weight=w, data_idx=idx, case_id="0")]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(symbols, hessian=qm_hess), refs)
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            em_p = mass_weighted_eigenmatrix(h_p, qm_evecs, symbols)
            em_m = mass_weighted_eigenmatrix(h_m, qm_evecs, symbols)
            fd_grad[j] = ((w * (ref_val - em_p[idx, idx])) ** 2 - (w * (ref_val - em_m[idx, idx])) ** 2) / (2 * delta)
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_gradient_offdiagonal_vs_fd(self) -> None:
        rng = np.random.default_rng(9009)
        n_params = 2
        hess = _make_symmetric(6, rng)
        qm_hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, n_params))
        symbols = ["C", "H"]
        _, qm_evecs = mass_weighted_normal_modes(qm_hess, symbols)
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, symbols)
        row, col = 1, 4
        ref_val = eigmat[row, col] + 0.003
        w = 1.0
        refs = [Observation(kind="eig_offdiagonal", value=ref_val, weight=w, case_id="0", atom_indices=(row, col))]
        grad = _grad(_make_mock_engine(hess, dH_dp), _make_mol(symbols, hessian=qm_hess), refs)
        delta = 1e-7
        fd_grad = np.zeros(n_params)
        for j in range(n_params):
            h_p = hess + delta * dH_dp[:, :, j]
            h_m = hess - delta * dH_dp[:, :, j]
            em_p = mass_weighted_eigenmatrix(h_p, qm_evecs, symbols)
            em_m = mass_weighted_eigenmatrix(h_m, qm_evecs, symbols)
            fd_grad[j] = ((w * (ref_val - em_p[row, col])) ** 2 - (w * (ref_val - em_m[row, col])) ** 2) / (2 * delta)
        np.testing.assert_allclose(grad, fd_grad, rtol=1e-5)

    def test_caches_qm_eigenvectors(self) -> None:
        rng = np.random.default_rng(1010)
        hess = _make_symmetric(6, rng)
        qm_hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        _, qm_evecs = mass_weighted_normal_modes(qm_hess, ["C", "H"])
        eigmat = mass_weighted_eigenmatrix(hess, qm_evecs, ["C", "H"])
        refs = [Observation(kind="eig_diagonal", value=eigmat[0, 0], weight=1.0, data_idx=0, case_id="0")]
        obj, x = _objective(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"], hessian=qm_hess), refs)
        obj.gradient(x)
        assert "0" in obj._qm_eigenvectors
        np.testing.assert_allclose(obj._qm_eigenvectors["0"], qm_evecs, atol=1e-14)

    def test_no_qm_hessian_raises(self) -> None:
        rng = np.random.default_rng(1111)
        hess = _make_symmetric(6, rng)
        dH_dp = rng.standard_normal((6, 6, 2))
        refs = [Observation(kind="eig_diagonal", value=0.0, weight=1.0, data_idx=0, case_id="0")]
        obj, x = _objective(_make_mock_engine(hess, dH_dp), _make_mol(["C", "H"], hessian=None), refs)
        with pytest.raises(ValueError, match="no QM Hessian"):
            obj.gradient(x)


class TestBaseEngineHessianAPI:
    def test_default_does_not_declare_jacobian(self) -> None:
        prepared = _FakePrepared(supports_jac=False)
        assert prepared.info.supports(Capability.HESSIAN_PARAMETER_JACOBIAN) is False

    def test_unsupported_call_raises(self) -> None:
        from q2mm.backends.contracts import HessianJacobianRequest, UnsupportedCapabilityError

        prepared = _FakePrepared(hess=np.zeros((3, 3)), supports_jac=False)
        with pytest.raises(UnsupportedCapabilityError):
            prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=P))
