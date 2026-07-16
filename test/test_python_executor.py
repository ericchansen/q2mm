"""Tests for Python objective executor per-observable behavior.

These tests cover the executor paths that replaced the deleted per-data-type
optimizer evaluators: calculated values, residuals, extraction errors, cache
reset, and analytical gradients.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from test._shared import GS_FCHK, make_ethane, make_water
from test.backend_fixtures import MockLayout

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
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
from q2mm.geometry import dihedral_angle
from q2mm.io.fchk import load_fchk_reference
from q2mm.models.molecule import Angle, Bond, Molecule
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.problem import StationaryPointKind
from q2mm.models.hessian import mass_weighted_eigenmatrix, mass_weighted_normal_modes
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor

_PROV = BackendProvenance(backend="stub", role=BackendRole.MM)
P = np.zeros(2)


class StubPrepared(AbstractPreparedBackend):
    """Prepared-session double returning fixed data through the typed API."""

    def __init__(
        self,
        *,
        energy: float = 0.0,
        frequencies: list[float] | None = None,
        hessian: np.ndarray | None = None,
        minimize_coords: np.ndarray | None = None,
        minimize_symbols: list[str] | None = None,
        n_params: int = 2,
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
        super().__init__(info=info, case_id="0", molecule=None, force_field=None, layout=MockLayout(n_params))
        self._energy_value = energy
        self._freqs = frequencies or []
        self._hess = hessian
        self._minimize_coords = minimize_coords
        self._minimize_symbols = minimize_symbols

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
        symbols = self._minimize_symbols or list(self._molecule.symbols)
        return GeometryResult(
            energy=0.0,
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=tuple(symbols),
            coordinates=readonly_array(self._minimize_coords),
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=_PROV,
        )


class GradStubPrepared(StubPrepared):
    """Prepared-session double that supports analytical energy gradients."""

    def __init__(self, *, energy: float = 0.0, param_grad: np.ndarray | None = None, **kwargs: object) -> None:
        self._param_grad = np.asarray(param_grad if param_grad is not None else np.zeros(0), dtype=float)
        super().__init__(energy=energy, n_params=len(self._param_grad), **kwargs)
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
    """Prepared-session double that supports Hessian parameter Jacobians."""

    def __init__(self, *, hessian: np.ndarray, hess_jac: np.ndarray, **kwargs: object) -> None:
        self._hess_jac = np.asarray(hess_jac, dtype=float)
        super().__init__(hessian=hessian, n_params=self._hess_jac.shape[2], **kwargs)
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
    """Backend double whose ``prepare`` returns a fixed prepared session."""

    def __init__(self, prepared: StubPrepared, *, info: BackendInfo | None = None) -> None:
        self._prepared = prepared
        self.info = info or prepared.info

    def prepare(self, request: object) -> StubPrepared:
        self._prepared._molecule = request.molecule  # type: ignore[attr-defined]
        self._prepared._case_id = request.case_id  # type: ignore[attr-defined]
        return self._prepared


class _StubForceField:
    def __init__(self, n_params: int = 2) -> None:
        self.n_params = n_params

    def with_params(self, param_vector: np.ndarray) -> _StubForceField:
        return self


class _StubLayout:
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


def _obs(*values: Observation) -> ObservationSet:
    return ObservationSet(values=values)


class _CountingPrepared(StubPrepared):
    """Stub prepared session that counts Hessian and frequency calls."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.hessian_calls = 0
        self.frequency_calls = 0

    def _hessian(self, request: object) -> HessianResult:
        self.hessian_calls += 1
        return super()._hessian(request)

    def _frequencies(self, request: object) -> FrequencyResult:
        self.frequency_calls += 1
        return super()._frequencies(request)


class TestSingleHessianPerCase:
    """When frequency + eigenmatrix + hessian refs coexist, one Hessian is computed."""

    def test_one_hessian_and_no_frequency_call(self) -> None:
        rng = np.random.default_rng(0)
        a = rng.standard_normal((9, 9))
        hess = a @ a.T + 9.0 * np.eye(9)  # SPD 9x9 (water, 3 atoms)
        mol = make_water().with_hessian(hess)
        prepared = _CountingPrepared(hessian=hess, frequencies=[100.0] * 9, n_params=2)
        refs = _obs(
            Observation(kind="frequency", value=0.0, data_idx=6, case_id="0"),
            Observation(kind="eig_diagonal", value=0.0, data_idx=6, case_id="0"),
            Observation(kind="hessian_element", value=0.0, atom_indices=(0, 0), case_id="0"),
        )
        obj, x = _executor(prepared, mol, refs)
        obj.value(x)
        # Frequencies are derived from the single MM Hessian, not a second call.
        assert prepared.hessian_calls == 1
        assert prepared.frequency_calls == 0

    def test_prepare_count_survives_reset_and_sampling(self) -> None:
        """A case is prepared exactly once across value(), reset(), and sample()."""
        prepared = _CountingPrepared(energy=1.0, n_params=2)
        obj, x = _executor(prepared, make_water(), _obs(Observation(kind="energy", value=0.0, case_id="0")))
        prepare_calls = {"n": 0}
        real_prepare = obj._backend.prepare

        def counting_prepare(request: object) -> object:
            prepare_calls["n"] += 1
            return real_prepare(request)

        obj._backend.prepare = counting_prepare  # type: ignore[attr-defined]
        obj.value(x)
        obj.value(x)
        obj.reset()
        obj.sample(x)
        obj.value(x)
        # Session prepared once for the case; reset/sampling never re-prepare.
        assert prepare_calls["n"] == 1
        assert set(obj._prepared.keys()) == {"0"}


def _executor(
    prepared: StubPrepared,
    mol: object,
    ref: ObservationSet,
    *,
    n_params: int | None = None,
    gradient_mode: GradientMode = GradientMode.NONE,
    backend_info: BackendInfo | None = None,
) -> tuple[PythonObjectiveExecutor, np.ndarray]:
    n = len(prepared.layout) if n_params is None else n_params
    ff = _StubForceField(n)
    layout = _StubLayout(n)
    space = ActiveParameterSpace(layout=layout, baseline=np.zeros(n), active_indices=np.arange(n))
    plan = ObjectivePlan(
        case_ids=("0",),
        molecules=(mol,),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ref,
        layout=layout,
        active_space=space,
    )
    return PythonObjectiveExecutor(
        plan, StubBackend(prepared, info=backend_info), ff, gradient_mode=gradient_mode
    ), np.zeros(n)


class TestEnergyExecutor:
    def test_compute(self) -> None:
        obj, x = _executor(StubPrepared(energy=42.5), make_water(), _obs(Observation(kind="energy", value=0.0)))
        assert obj.evaluate(x).calculated[0] == 42.5

    def test_residuals_single(self) -> None:
        ref = _obs(Observation(kind="energy", value=12.0, weight=2.0))
        obj, x = _executor(StubPrepared(energy=10.0), make_water(), ref)
        result = obj.evaluate(x)
        assert len(result.weighted_residuals) == 1
        assert result.weighted_residuals[0] == pytest.approx(2.0 * (12.0 - 10.0))
        assert result.raw_residuals[0] == pytest.approx(2.0)

    def test_residuals_multiple(self) -> None:
        refs = _obs(
            Observation(kind="energy", value=5.0, weight=1.0),
            Observation(kind="energy", value=7.0, weight=0.5),
        )
        obj, x = _executor(StubPrepared(energy=5.0), make_water(), refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(0.0)
        assert residuals[1] == pytest.approx(0.5 * 2.0)

    def test_extract_value(self) -> None:
        obj, x = _executor(StubPrepared(energy=99.9), make_water(), _obs(Observation(kind="energy", value=0.0)))
        assert obj.evaluate(x).calculated[0] == 99.9

    def test_compute_with_structure(self) -> None:
        mol = make_water()
        backend = StubPrepared(energy=1.0)
        obj, x = _executor(backend, mol, _obs(Observation(kind="energy", value=0.0)))
        assert obj.evaluate(x).calculated[0] == 1.0
        assert backend.molecule is mol


class TestFrequencyExecutor:
    def test_compute(self) -> None:
        refs = _obs(Observation(kind="frequency", value=0.0, data_idx=0))
        obj, x = _executor(StubPrepared(frequencies=[100.0, 200.0, 300.0]), make_water(), refs)
        assert obj.evaluate(x).calculated[0] == 100.0

    def test_residuals(self) -> None:
        refs = _obs(
            Observation(kind="frequency", value=105.0, weight=1.0, data_idx=0),
            Observation(kind="frequency", value=195.0, weight=2.0, data_idx=1),
        )
        obj, x = _executor(StubPrepared(frequencies=[100.0, 200.0, 300.0]), make_water(), refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(1.0 * (105.0 - 100.0))
        assert residuals[1] == pytest.approx(2.0 * (195.0 - 200.0))

    def test_residuals_out_of_range(self) -> None:
        refs = _obs(Observation(kind="frequency", value=200.0, data_idx=5))
        obj, x = _executor(StubPrepared(frequencies=[100.0]), make_water(), refs)
        with pytest.raises(IndexError, match="data_idx=5"):
            obj.evaluate(x)

    def test_extract_value(self) -> None:
        refs = _obs(Observation(kind="frequency", value=0.0, data_idx=2))
        obj, x = _executor(StubPrepared(frequencies=[100.0, 200.0, 300.0]), make_water(), refs)
        assert obj.evaluate(x).calculated[0] == 300.0

    def test_penalty_error_mode_is_forwarded(self) -> None:
        refs = _obs(Observation(kind="frequency", value=0.0, data_idx=0))
        obj, _ = _executor(StubPrepared(frequencies=[100.0]), make_water(), refs)
        obj.on_error = "penalty"
        assert obj.on_error == "penalty"


class TestGeometryExecutor:
    def test_dihedral_angle_staggered(self) -> None:
        ethane = make_ethane()
        coords = ethane.geometry
        angle = dihedral_angle(coords[2], coords[0], coords[1], coords[5])
        abs_angle = abs(angle)
        assert abs_angle > 50 or abs(abs_angle - 180) < 10

    def test_dihedral_angle_collinear(self) -> None:
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([1.0, 0.0, 0.0])
        p2 = np.array([2.0, 0.0, 0.0])
        p3 = np.array([3.0, 0.0, 0.0])
        assert dihedral_angle(p0, p1, p2, p3) == 0.0

    def test_residuals_bond_length(self) -> None:
        mol = Molecule(
            symbols=("X", "X", "X"),
            geometry=np.zeros((3, 3)),
            bonds=(Bond(0, 1, ("X", "X"), 1.0), Bond(1, 2, ("X", "X"), 1.5)),
            angles=(),
            torsions=(),
        )
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        refs = _obs(Observation(kind="bond_length", value=1.1, weight=10.0, atom_indices=(0, 1)))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(10.0 * (1.1 - 1.0))

    def test_residuals_bond_angle(self) -> None:
        mol = Molecule(
            symbols=("X", "X", "X"),
            geometry=np.zeros((3, 3)),
            bonds=(),
            angles=(Angle(0, 1, 2, ("X", "X", "X"), 109.5),),
            torsions=(),
        )
        theta = np.deg2rad(109.5)
        coords = np.array([[np.cos(theta), np.sin(theta), 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        refs = _obs(Observation(kind="bond_angle", value=110.0, weight=5.0, atom_indices=(0, 1, 2)))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(5.0 * 0.5)

    def test_residuals_torsion_wrapping(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 1.0]])
        mol = Molecule(symbols=("X", "X", "X", "X"), geometry=np.zeros((4, 3)), bonds=(), angles=(), torsions=())
        refs = _obs(Observation(kind="torsion_angle", value=170.0, weight=1.0, atom_indices=(0, 1, 2, 3)))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 1
        assert -180.0 <= residuals[0] <= 180.0

    def test_extract_value_bond_by_atoms(self) -> None:
        mol = Molecule(
            symbols=("X", "X", "X"),
            geometry=np.zeros((3, 3)),
            bonds=(Bond(0, 1, ("X", "X"), 1.0), Bond(1, 2, ("X", "X"), 1.5)),
            angles=(),
            torsions=(),
        )
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        refs = _obs(Observation(kind="bond_length", value=0.0, atom_indices=(1, 2)))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        assert obj.evaluate(x).calculated[0] == 1.5

    def test_extract_value_bond_by_idx(self) -> None:
        mol = Molecule(
            symbols=("X", "X", "X"),
            geometry=np.zeros((3, 3)),
            bonds=(Bond(0, 1, ("X", "X"), 1.0), Bond(1, 2, ("X", "X"), 1.5)),
            angles=(),
            torsions=(),
        )
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        refs = _obs(Observation(kind="bond_length", value=0.0, data_idx=1))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        assert obj.evaluate(x).calculated[0] == 1.5

    def test_extract_value_angle_reverse_order(self) -> None:
        mol = Molecule(
            symbols=("X", "X", "X"),
            geometry=np.zeros((3, 3)),
            bonds=(),
            angles=(Angle(2, 1, 0, ("X", "X", "X"), 109.5),),
            torsions=(),
        )
        theta = np.deg2rad(109.5)
        coords = np.array([[np.cos(theta), np.sin(theta), 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        refs = _obs(Observation(kind="bond_angle", value=0.0, atom_indices=(0, 1, 2)))
        obj, x = _executor(StubPrepared(minimize_coords=coords), mol, refs)
        assert obj.evaluate(x).calculated[0] == pytest.approx(109.5)


class TestEigenmatrixExecutor:
    def test_compute_self_projection(self) -> None:
        hess = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), name="stub", hessian=hess)
        refs = _obs(Observation(kind="eig_diagonal", value=0.0, data_idx=0))
        obj, x = _executor(StubPrepared(hessian=hess), mol, refs)
        obj.evaluate(x)
        eigmat = obj._compute_case("0", x, {"eig_diagonal"})["eigenmatrix"]
        off_diag = eigmat - np.diag(np.diag(eigmat))
        assert np.allclose(off_diag, 0, atol=1e-12)

    def test_residuals_diagonal(self) -> None:
        hess = np.diag([1.0, 2.0, 3.0])
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        refs = _obs(
            Observation(kind="eig_diagonal", value=1.5, weight=0.1, data_idx=0),
            Observation(kind="eig_diagonal", value=2.0, weight=0.1, data_idx=1),
        )
        obj, x = _executor(StubPrepared(hessian=hess), mol, refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 2
        evals, _ = mass_weighted_normal_modes(hess, mol.symbols)
        assert residuals[0] == pytest.approx(0.1 * (1.5 - evals[0]))
        assert residuals[1] == pytest.approx(0.1 * (2.0 - evals[1]))

    def test_residuals_offdiagonal(self) -> None:
        hess = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.0], [0.0, 0.0, 3.0]])
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=np.eye(3))
        refs = _obs(Observation(kind="eig_offdiagonal", value=0.0, weight=0.05, atom_indices=(1, 0)))
        obj, x = _executor(StubPrepared(hessian=hess), mol, refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 1
        _, modes = mass_weighted_normal_modes(mol.hessian, mol.symbols)
        expected = mass_weighted_eigenmatrix(hess, modes, mol.symbols)[1, 0]
        assert residuals[0] == pytest.approx(0.05 * (0.0 - expected))

    def test_eigenvector_caching(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal((6, 6))
        hess = a @ a.T + np.eye(6)
        mol = Molecule(symbols=("H", "H"), geometry=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]), hessian=hess)
        refs = _obs(Observation(kind="eig_diagonal", value=0.0, data_idx=0))
        obj, x = _executor(StubPrepared(hessian=hess), mol, refs)
        obj.evaluate(x)
        assert "0" in obj._qm_eigenvectors
        cached = obj._qm_eigenvectors["0"]
        obj.evaluate(x)
        assert obj._qm_eigenvectors["0"] is cached

    def test_reset_retains_cache(self) -> None:
        """reset() clears counters/history but RETAINS the QM eigenvector cache.

        Phase 4 (r4): prepared sessions and derived QM eigenvectors survive a
        reset so repeated optimizer/workflow stages never re-prepare a case.
        """
        hess = np.eye(3)
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        obj, x = _executor(StubPrepared(hessian=hess), mol, _obs(Observation(kind="eig_diagonal", value=0.0)))
        obj.evaluate(x)
        assert obj._qm_eigenvectors
        cached = dict(obj._qm_eigenvectors)
        obj.value(x)
        assert obj.n_evaluations > 0
        obj.reset()
        # Counters/history cleared; QM eigenvector cache retained.
        assert obj.n_evaluations == 0
        assert obj.history == ()
        assert obj._qm_eigenvectors.keys() == cached.keys()

    def test_no_qm_hessian_raises(self) -> None:
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), name="stub")
        refs = _obs(Observation(kind="eig_diagonal", value=1.0, data_idx=0))
        obj, x = _executor(StubPrepared(hessian=np.eye(3)), mol, refs)
        with pytest.raises(ValueError, match="no QM Hessian"):
            obj.evaluate(x)

    def test_extract_value_diagonal(self) -> None:
        hess = np.diag([1.0, 2.0, 3.0])
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        obj, x = _executor(
            StubPrepared(hessian=hess), mol, _obs(Observation(kind="eig_diagonal", value=0.0, data_idx=2))
        )
        evals, _ = mass_weighted_normal_modes(hess, mol.symbols)
        assert obj.evaluate(x).calculated[0] == pytest.approx(evals[2])

    def test_extract_value_offdiagonal(self) -> None:
        mat = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 3.0]])
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=np.eye(3))
        obj, x = _executor(
            StubPrepared(hessian=mat), mol, _obs(Observation(kind="eig_offdiagonal", value=0.0, atom_indices=(2, 1)))
        )
        _, modes = mass_weighted_normal_modes(mol.hessian, mol.symbols)
        expected = mass_weighted_eigenmatrix(mat, modes, mol.symbols)[2, 1]
        assert obj.evaluate(x).calculated[0] == pytest.approx(expected)


class TestExecutorProtocolCompliance:
    def test_energy_executor_is_objective_evaluator(self) -> None:
        obj, _ = _executor(StubPrepared(energy=1.0), make_water(), _obs(Observation(kind="energy", value=0.0)))
        assert isinstance(obj, ObjectiveEvaluator)

    def test_frequency_executor_is_objective_evaluator(self) -> None:
        obj, _ = _executor(
            StubPrepared(frequencies=[1.0]), make_water(), _obs(Observation(kind="frequency", value=0.0))
        )
        assert isinstance(obj, ObjectiveEvaluator)

    def test_geometry_executor_is_objective_evaluator(self) -> None:
        mol = make_water()
        ref = _obs(Observation(kind="bond_length", value=1.0, atom_indices=(0, 1)))
        obj, _ = _executor(StubPrepared(minimize_coords=mol.geometry), mol, ref)
        assert isinstance(obj, ObjectiveEvaluator)

    def test_eigenmatrix_executor_is_objective_evaluator(self) -> None:
        hess = np.eye(3)
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        obj, _ = _executor(StubPrepared(hessian=hess), mol, _obs(Observation(kind="eig_diagonal", value=0.0)))
        assert isinstance(obj, ObjectiveEvaluator)


class TestEnergyExecutorGradient:
    def test_supports_analytical_gradient_true(self) -> None:
        obj, _ = _executor(
            GradStubPrepared(energy=1.0, param_grad=np.array([1.0, 2.0])),
            make_water(),
            _obs(Observation(kind="energy", value=0.0)),
            gradient_mode=GradientMode.ANALYTICAL,
        )
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_supports_analytical_gradient_false(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="PARAMETER_GRADIENT"):
            _executor(
                StubPrepared(energy=1.0),
                make_water(),
                _obs(Observation(kind="energy", value=0.0)),
                gradient_mode=GradientMode.ANALYTICAL,
            )

    def test_gradient_single_ref(self) -> None:
        de_dp = np.array([3.0, -1.0])
        refs = _obs(Observation(kind="energy", value=12.0, weight=2.0))
        obj, x = _executor(
            GradStubPrepared(energy=10.0, param_grad=de_dp), make_water(), refs, gradient_mode=GradientMode.ANALYTICAL
        )
        grad = obj.gradient(x)
        expected = -2.0 * 2.0**2 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(grad, expected)

    def test_gradient_multiple_refs(self) -> None:
        de_dp = np.array([1.0, 0.5])
        refs = _obs(
            Observation(kind="energy", value=5.0, weight=1.0),
            Observation(kind="energy", value=7.0, weight=0.5),
        )
        obj, x = _executor(
            GradStubPrepared(energy=5.0, param_grad=de_dp), make_water(), refs, gradient_mode=GradientMode.ANALYTICAL
        )
        expected = -2.0 * 0.5**2 * (7.0 - 5.0) * de_dp
        np.testing.assert_allclose(obj.gradient(x), expected)

    def test_gradient_raises_on_unsupported_prepared_session(self) -> None:
        backend_info = BackendInfo(
            name="declares_grad",
            role=BackendRole.MM,
            capabilities=StubPrepared(energy=1.0).info.capabilities | {Capability.PARAMETER_GRADIENT},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        refs = _obs(Observation(kind="energy", value=2.0, weight=1.0))
        obj, x = _executor(
            StubPrepared(energy=1.0),
            make_water(),
            refs,
            gradient_mode=GradientMode.ANALYTICAL,
            backend_info=backend_info,
        )
        with pytest.raises(Exception, match="parameter_gradient"):
            obj.gradient(x)

    def test_gradient_validates_de_dp_shape(self) -> None:
        de_dp = np.array([1.0, 2.0])
        backend = GradStubPrepared(energy=5.0, param_grad=de_dp)
        backend._layout = MockLayout(3)
        refs = _obs(Observation(kind="energy", value=6.0, weight=1.0))
        obj, x = _executor(backend, make_water(), refs, n_params=3, gradient_mode=GradientMode.ANALYTICAL)
        with pytest.raises(EvaluationError, match="gradient shape"):
            obj.gradient(x)


class TestFrequencyExecutorGradient:
    def test_supports_analytical_gradient_false_without_hessian_jacobian(self) -> None:
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _executor(
                GradStubPrepared(energy=0.0, param_grad=np.array([1.0]), frequencies=[100.0]),
                make_water(),
                _obs(Observation(kind="frequency", value=105.0, data_idx=0)),
                gradient_mode=GradientMode.ANALYTICAL,
            )

    def test_gradient_raises_on_unsupported_prepared_session(self) -> None:
        prepared = StubPrepared(frequencies=[100.0])
        backend_info = BackendInfo(
            name="declares_hess_jac",
            role=BackendRole.MM,
            capabilities=prepared.info.capabilities | {Capability.HESSIAN_PARAMETER_JACOBIAN},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        refs = _obs(Observation(kind="frequency", value=105.0, data_idx=0))
        obj, x = _executor(
            prepared, make_water(), refs, gradient_mode=GradientMode.ANALYTICAL, backend_info=backend_info
        )
        with pytest.raises(Exception, match="hessian_parameter_jacobian"):
            obj.gradient(x)


class TestGeometryExecutorGradient:
    def test_supports_analytical_gradient_false(self) -> None:
        mol = make_water()
        refs = _obs(Observation(kind="bond_length", value=1.0, atom_indices=(0, 1)))
        with pytest.raises(ObjectiveGradientError, match="geometry references"):
            _executor(
                GradStubPrepared(energy=0.0, param_grad=np.array([1.0]), minimize_coords=mol.geometry),
                mol,
                refs,
                gradient_mode=GradientMode.ANALYTICAL,
            )

    def test_gradient_mode_none_raises_when_gradient_requested(self) -> None:
        mol = make_water()
        refs = _obs(Observation(kind="bond_length", value=1.0, atom_indices=(0, 1)))
        obj, x = _executor(StubPrepared(minimize_coords=mol.geometry), mol, refs)
        with pytest.raises(ObjectiveGradientError, match="gradient_mode=none"):
            obj.gradient(x)


class TestEigenmatrixExecutorGradient:
    def test_supports_analytical_gradient_false_without_hessian_jacobian(self) -> None:
        hess = np.eye(3)
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _executor(
                StubPrepared(hessian=hess),
                mol,
                _obs(Observation(kind="eig_diagonal", value=1.0)),
                gradient_mode=GradientMode.ANALYTICAL,
            )

    def test_gradient_raises_on_unsupported_prepared_session(self) -> None:
        prepared = StubPrepared(hessian=np.eye(3))
        backend_info = BackendInfo(
            name="declares_hess_jac",
            role=BackendRole.MM,
            capabilities=prepared.info.capabilities | {Capability.HESSIAN_PARAMETER_JACOBIAN},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=np.eye(3))
        obj, x = _executor(
            prepared,
            mol,
            _obs(Observation(kind="eig_diagonal", value=1.0)),
            gradient_mode=GradientMode.ANALYTICAL,
            backend_info=backend_info,
        )
        with pytest.raises(Exception, match="hessian_parameter_jacobian"):
            obj.gradient(x)


class TestObjectiveExecutorGradient:
    def test_energy_only_gradient_uses_prepared_parameter_gradient(self) -> None:
        de_dp = np.array([3.0, -1.0])
        refs = ObservationSet().with_energy(12.0, weight=2.0)
        obj, x = _executor(
            GradStubPrepared(energy=10.0, param_grad=de_dp), make_water(), refs, gradient_mode=GradientMode.ANALYTICAL
        )
        expected = -2.0 * 2.0**2 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(obj.gradient(x), expected)

    def test_mixed_refs_require_hessian_jacobian_for_analytical_frequency(self) -> None:
        ref = ObservationSet().with_energy(10.0, weight=1.0).with_frequency(100.0, data_idx=0, weight=1.0)
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _executor(
                GradStubPrepared(energy=10.0, param_grad=np.array([1.0]), frequencies=[100.0]),
                make_water(),
                ref,
                gradient_mode=GradientMode.ANALYTICAL,
            )

    def test_mixed_energy_frequency_gradient(self) -> None:
        de_dp = np.array([2.0])
        hess = np.eye(3)
        dH_dp = np.zeros((3, 3, 1))
        refs = ObservationSet().with_energy(15.0, weight=1.0).with_frequency(0.0, data_idx=0, weight=1.0)
        prepared = HessJacStubPrepared(hessian=hess, hess_jac=dH_dp, energy=10.0, frequencies=[0.0])
        prepared._info = BackendInfo(
            name="energy_hessjac_stub",
            role=BackendRole.MM,
            capabilities=prepared.info.capabilities | {Capability.PARAMETER_GRADIENT},
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        prepared._param_grad = de_dp
        prepared._parameter_gradient = lambda request: ParameterGradientResult(
            energy=10.0, gradient=readonly_array(de_dp), unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROV
        )  # type: ignore[method-assign]
        mol = Molecule(symbols=("H",), geometry=np.array([[0.0, 0.0, 0.0]]), hessian=hess)
        obj, x = _executor(prepared, mol, refs, n_params=1, gradient_mode=GradientMode.ANALYTICAL)
        expected_energy_grad = -2.0 * (15.0 - 10.0) * de_dp
        np.testing.assert_allclose(obj.gradient(x), expected_energy_grad, atol=1e-6)

    def test_objective_gradient_uses_prepared_parameter_gradient(self) -> None:
        de_dp = np.array([1.0])
        refs = ObservationSet().with_energy(12.0, weight=1.0)
        obj, x = _executor(
            GradStubPrepared(energy=10.0, param_grad=de_dp), make_water(), refs, gradient_mode=GradientMode.ANALYTICAL
        )
        expected = -2.0 * (12.0 - 10.0) * de_dp
        np.testing.assert_allclose(obj.gradient(x), expected)


class TestFchkParser:
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


class TestExecutorObjectiveParity:
    def test_energy_parity(self) -> None:
        ref = ObservationSet().with_energy(40.0, weight=2.0)
        obj, x = _executor(StubPrepared(energy=42.0), make_water(), ref)
        result = obj.evaluate(x)
        assert result.calculated[0] == 42.0
        assert result.weighted_residuals[0] == pytest.approx(2.0 * (40.0 - 42.0))

    def test_frequency_parity(self) -> None:
        ref = ObservationSet().with_frequency(105.0, data_idx=0).with_frequency(195.0, data_idx=1)
        obj, x = _executor(StubPrepared(frequencies=[100.0, 200.0, 300.0]), make_water(), ref)
        result = obj.evaluate(x)
        np.testing.assert_allclose(result.calculated, [100.0, 200.0])

    def test_eigenmatrix_parity(self) -> None:
        _ref_data, mol = load_fchk_reference(str(GS_FCHK))
        assert mol.hessian is not None
        qm_hessian = np.array(mol.hessian, dtype=float)
        ref = ObservationSet().with_eigenmatrix_from_hessian(qm_hessian, diagonal_only=True)
        obj, x = _executor(StubPrepared(hessian=qm_hessian), mol, ref)
        obj.evaluate(x)
        eigmat = obj._compute_case("0", x, {"eig_diagonal"})["eigenmatrix"]
        diag_only = np.diag(np.diag(eigmat))
        assert np.allclose(eigmat, diag_only, atol=1e-8)
