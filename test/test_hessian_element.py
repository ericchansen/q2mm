"""Tests for raw Hessian element training data pipeline.

Covers hessian_element ObservationSet/YAML loading and the
PythonObjectiveExecutor behavior for hessian_element extraction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    HessianJacobianResult,
    HessianResult,
    HessianUnit,
    readonly_array,
)
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveGradientError
from q2mm.objectives.python import PythonObjectiveExecutor
from test.backend_fixtures import MockLayout, mock_molecule

P = np.zeros(2)
_PROV = BackendProvenance(backend="mock", role=BackendRole.MM)


class _FakePrepared(AbstractPreparedBackend):
    """Prepared-session double returning fixed Hessian/Jacobian data."""

    def __init__(
        self,
        *,
        hessian: object = None,
        hess_jac: object = None,
        supports_jac: bool = False,
        molecule: object = None,
        n_params: int = 2,
    ) -> None:
        caps = {Capability.HESSIAN}
        if supports_jac:
            caps.add(Capability.HESSIAN_PARAMETER_JACOBIAN)
        info = BackendInfo(
            name="mock",
            role=BackendRole.MM,
            capabilities=frozenset(caps),
            functional_forms=frozenset({"harmonic"}),
            provenance=_PROV,
        )
        if hess_jac is not None:
            n_params = np.asarray(hess_jac).shape[2]
        if molecule is None and hessian is not None:
            n3 = np.asarray(hessian).shape[0]
            molecule = mock_molecule(["X"] * (n3 // 3))
        super().__init__(info=info, case_id="0", molecule=molecule, force_field=None, layout=MockLayout(n_params))
        self._h = np.asarray(hessian, dtype=float) if hessian is not None else None
        self._j = np.asarray(hess_jac, dtype=float) if hess_jac is not None else None

    def _hessian(self, request: object) -> HessianResult:
        return HessianResult(hessian=readonly_array(self._h), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=_PROV)

    def _hessian_parameter_jacobian(self, request: object) -> HessianJacobianResult:
        return HessianJacobianResult(
            hessian=readonly_array(self._h),
            jacobian=readonly_array(self._j),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=_PROV,
        )


class _FakeBackend:
    def __init__(self, prepared: _FakePrepared, *, info: BackendInfo | None = None) -> None:
        self._p = prepared
        self.info = info or prepared.info

    def prepare(self, request: object) -> _FakePrepared:
        self._p._molecule = request.molecule  # type: ignore[attr-defined]
        self._p._case_id = request.case_id  # type: ignore[attr-defined]
        return self._p


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


def _executor(
    prepared: _FakePrepared,
    ref: ObservationSet,
    *,
    molecule: object | None = None,
    gradient_mode: GradientMode = GradientMode.NONE,
    backend_info: BackendInfo | None = None,
) -> tuple[PythonObjectiveExecutor, np.ndarray]:
    mol = molecule if molecule is not None else prepared.molecule
    n = len(prepared.layout)
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
        plan, _FakeBackend(prepared, info=backend_info), ff, gradient_mode=gradient_mode
    ), np.zeros(n)


@pytest.fixture
def small_hessian() -> np.ndarray:
    return np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])


@pytest.fixture
def hessian_6x6() -> np.ndarray:
    rng = np.random.default_rng(42)
    h = rng.standard_normal((6, 6))
    return (h + h.T) / 2


class TestHessianElementCompute:
    def test_compute_returns_raw_hessian(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_element(4.0, row=0, col=0)
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        computed = obj._compute_case("0", x, {"hessian_element"})
        np.testing.assert_array_equal(computed["raw_hessian"], small_hessian)

    def test_evaluate_reuses_prepared_session(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_element(4.0, row=0, col=0)
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        assert obj.evaluate(x).calculated[0] == 4.0
        assert len(obj._prepared) == 1


class TestHessianElementExtract:
    def test_extract_diagonal(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_element(0.0, row=1, col=1)
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        assert obj.evaluate(x).calculated[0] == 3.0

    def test_extract_offdiagonal(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_element(0.0, row=2, col=0)
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        assert obj.evaluate(x).calculated[0] == 0.5

    def test_extract_corner(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_element(0.0, row=0, col=0)
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        assert obj.evaluate(x).calculated[0] == 4.0

    def test_extract_out_of_range_raises(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet(values=(Observation(kind="hessian_element", value=0.0, atom_indices=(5, 0), label="bad"),))
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        with pytest.raises(IndexError, match="out of range"):
            obj.evaluate(x)

    def test_extract_missing_atom_indices_raises(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet(values=(Observation(kind="hessian_element", value=0.0),))
        obj, x = _executor(_FakePrepared(hessian=small_hessian), ref)
        with pytest.raises(ValueError, match="requires atom_indices"):
            obj.evaluate(x)


class TestHessianElementResiduals:
    def test_residuals_correct(self, small_hessian: np.ndarray) -> None:
        refs = ObservationSet(
            values=(
                Observation(kind="hessian_element", value=5.0, weight=0.1, atom_indices=(0, 0)),
                Observation(kind="hessian_element", value=3.0, weight=0.2, atom_indices=(1, 1)),
            )
        )
        obj, x = _executor(_FakePrepared(hessian=small_hessian), refs)
        residuals = obj.evaluate(x).weighted_residuals
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(0.1)
        assert residuals[1] == pytest.approx(0.0)


class TestObservationSetAddHessianElement:
    def test_add_hessian_element(self) -> None:
        ref = ObservationSet().with_hessian_element(1.5, row=2, col=1, weight=0.1, label="test")
        rv = ref.values[0]
        assert ref.n_observations == 1
        assert rv.kind == "hessian_element"
        assert rv.value == 1.5
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.1
        assert rv.label == "test"

    def test_add_hessian_element_default_label(self) -> None:
        ref = ObservationSet().with_hessian_element(0.5, row=3, col=4)
        assert ref.values[0].label == "hess[3,4]"


class TestObservationSetAddHessianFromMatrix:
    def test_full_lower_triangle(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_from_matrix(small_hessian)
        assert ref.n_observations == 6
        assert all(rv.kind == "hessian_element" for rv in ref.values)
        diag_entries = [rv for rv in ref.values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag_entries = [rv for rv in ref.values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert len(diag_entries) == 3
        assert len(offdiag_entries) == 3
        assert all(rv.weight == 0.1 for rv in diag_entries)
        assert all(rv.weight == 0.05 for rv in offdiag_entries)

    def test_diagonal_only(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_from_matrix(small_hessian, diagonal_only=True)
        assert ref.n_observations == 3
        for rv in ref.values:
            assert rv.atom_indices[0] == rv.atom_indices[1]
        assert [rv.value for rv in ref.values] == [4.0, 3.0, 2.0]

    def test_skip_translational(self, hessian_6x6: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_from_matrix(hessian_6x6, skip_translational=3, diagonal_only=True)
        assert ref.n_observations == 3
        assert [rv.atom_indices[0] for rv in ref.values] == [3, 4, 5]

    def test_skip_translational_full(self, hessian_6x6: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_from_matrix(hessian_6x6, skip_translational=3)
        assert ref.n_observations == 6

    def test_non_square_raises(self) -> None:
        with pytest.raises(ValueError, match="square"):
            ObservationSet().with_hessian_from_matrix(np.ones((3, 4)))

    def test_custom_weights(self, small_hessian: np.ndarray) -> None:
        ref = ObservationSet().with_hessian_from_matrix(small_hessian, diagonal_weight=0.5, offdiagonal_weight=0.2)
        diag_entries = [rv for rv in ref.values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag_entries = [rv for rv in ref.values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert all(rv.weight == 0.5 for rv in diag_entries)
        assert all(rv.weight == 0.2 for rv in offdiag_entries)


class TestYAMLRoundTrip:
    def test_hessian_element_parse_and_serialize(self) -> None:
        from q2mm.io.reference import _parse_datum, _reference_value_to_dict

        datum = {"kind": "hessian_element", "value": 1.23, "row": 2, "col": 1, "weight": 0.1}
        refs = _parse_datum(datum, case_id="0", context="test")
        rv = refs[0]
        assert len(refs) == 1
        assert rv.kind == "hessian_element"
        assert rv.value == 1.23
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.1
        assert rv.label == "hess[2,1]"
        d = _reference_value_to_dict(rv)
        assert d["kind"] == "hessian_element"
        assert d["value"] == 1.23
        assert d["row"] == 2
        assert d["col"] == 1
        assert d["weight"] == 0.1

    def test_hessian_element_with_label(self) -> None:
        from q2mm.io.reference import _parse_datum

        refs = _parse_datum(
            {"kind": "hessian_element", "value": 0.5, "row": 0, "col": 0, "label": "H(0,0)"},
            case_id="0",
            context="test",
        )
        assert refs[0].label == "H(0,0)"

    def test_hessian_element_negative_indices_rejected(self) -> None:
        from q2mm.io.reference import ReferenceYAMLError, _parse_datum

        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            _parse_datum({"kind": "hessian_element", "value": 0.5, "row": -1, "col": 0}, case_id="0", context="test")


class TestYAMLBulkHessian:
    def test_bulk_hessian_directive(self, tmp_path: Path) -> None:
        from q2mm.io.reference import _load_molecule

        hess = np.diag([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        hess_path = tmp_path / "test_hessian.npy"
        np.save(str(hess_path), hess)
        mol_dict = {
            "name": "test_mol",
            "geometry": {"symbols": ["H", "H"], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
            "hessian": str(hess_path),
            "data": [{"kind": "hessian", "diagonal_only": True}],
        }
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)
        assert mol.hessian is not None
        assert len(ref_values) == 6
        assert all(rv.kind == "hessian_element" for rv in ref_values)
        assert all(rv.atom_indices[0] == rv.atom_indices[1] for rv in ref_values)

    def test_bulk_hessian_full(self, tmp_path: Path) -> None:
        from q2mm.io.reference import _load_molecule

        hess = np.diag([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        hess_path = tmp_path / "test_hessian_full.npy"
        np.save(str(hess_path), hess)
        mol_dict = {
            "name": "test_mol",
            "geometry": {"symbols": ["H", "H"], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
            "hessian": str(hess_path),
            "data": [{"kind": "hessian"}],
        }
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)
        assert len(ref_values) == 21
        diag = [rv for rv in ref_values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag = [rv for rv in ref_values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert len(diag) == 6
        assert len(offdiag) == 15

    def test_bulk_hessian_skip_translational(self, tmp_path: Path) -> None:
        from q2mm.io.reference import _load_molecule

        hess = np.eye(6)
        hess_path = tmp_path / "test_hessian_skip.npy"
        np.save(str(hess_path), hess)
        mol_dict = {
            "name": "test_mol",
            "geometry": {"symbols": ["H", "H"], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
            "hessian": str(hess_path),
            "data": [{"kind": "hessian", "skip_translational": 2, "diagonal_only": True}],
        }
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)
        assert len(ref_values) == 4
        assert [rv.atom_indices[0] for rv in ref_values] == [2, 3, 4, 5]

    def test_bulk_hessian_no_hessian_raises(self, tmp_path: Path) -> None:
        from q2mm.io.reference import ReferenceYAMLError, _load_molecule

        mol_dict = {
            "name": "test_mol",
            "geometry": {"symbols": ["H", "H"], "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]},
            "data": [{"kind": "hessian"}],
        }
        with pytest.raises(ReferenceYAMLError, match="requires a molecule with a hessian"):
            _load_molecule(mol_dict, tmp_path, 0)


class TestObjectiveExecutorHessianElement:
    def test_extract_hessian_element(self) -> None:
        hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        ref = ObservationSet().with_hessian_element(0.0, row=2, col=1)
        obj, x = _executor(_FakePrepared(hessian=hessian), ref)
        assert obj.evaluate(x).calculated[0] == 0.2

    def test_extract_hessian_element_diagonal(self) -> None:
        hessian = np.diag([1.0, 2.0, 3.0])
        ref = ObservationSet().with_hessian_element(0.0, row=2, col=2)
        obj, x = _executor(_FakePrepared(hessian=hessian), ref)
        assert obj.evaluate(x).calculated[0] == 3.0

    def test_evaluate_molecule_hessian_element(self) -> None:
        hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        ref = (
            ObservationSet()
            .with_hessian_element(4.0, row=0, col=0, weight=0.1)
            .with_hessian_element(1.0, row=1, col=0, weight=0.05)
        )
        obj, x = _executor(_FakePrepared(hessian=hessian), ref)
        result = obj._compute_case("0", x, {"hessian_element"})
        np.testing.assert_array_equal(result["raw_hessian"], hessian)

    def test_full_objective_with_hessian_elements(self) -> None:
        qm_hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mm_hessian = np.array([[4.1, 1.1, 0.5], [1.1, 2.9, 0.2], [0.5, 0.2, 2.0]])
        ref = ObservationSet().with_hessian_from_matrix(qm_hessian, diagonal_only=True)
        obj, x = _executor(_FakePrepared(hessian=mm_hessian), ref)
        score = obj.value(x)
        assert score > 0
        assert isinstance(score, float)

    def test_supports_analytical_gradient_false(self) -> None:
        ref = ObservationSet().with_hessian_element(0.0, row=0, col=0)
        with pytest.raises(ObjectiveGradientError, match="HESSIAN_PARAMETER_JACOBIAN"):
            _executor(_FakePrepared(hessian=np.eye(3), supports_jac=False), ref, gradient_mode=GradientMode.ANALYTICAL)

    def test_supports_analytical_gradient_true(self) -> None:
        ref = ObservationSet().with_hessian_element(0.0, row=0, col=0)
        hess = np.eye(3)
        dH_dp = np.zeros((3, 3, 2))
        obj, _ = _executor(
            _FakePrepared(hessian=hess, hess_jac=dH_dp, supports_jac=True), ref, gradient_mode=GradientMode.ANALYTICAL
        )
        assert obj.gradient_mode is GradientMode.ANALYTICAL

    def test_gradient_computes(self) -> None:
        hess = np.eye(3)
        dH_dp = np.zeros((3, 3, 2))
        dH_dp[0, 1, 0] = 1.0
        ref = ObservationSet(values=(Observation(kind="hessian_element", value=0.5, weight=1.0, atom_indices=(0, 1)),))
        obj, x = _executor(
            _FakePrepared(hessian=hess, hess_jac=dH_dp, supports_jac=True), ref, gradient_mode=GradientMode.ANALYTICAL
        )
        result = obj.gradient(x)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
        np.testing.assert_allclose(result, [-1.0, 0.0])

    def test_reset_runs_without_error(self) -> None:
        ref = ObservationSet().with_hessian_element(1.0, row=0, col=0)
        obj, x = _executor(_FakePrepared(hessian=np.eye(3)), ref)
        obj.evaluate(x)
        obj.reset()
        assert obj.n_evaluations == 0
