"""Tests for raw Hessian element training data pipeline.

Tests the HessianElementEvaluator and the ObservationSet/ObjectiveFunction
integration for raw Hessian matrix element training.
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
    HessianResult as _CHessianResult,
    HessianUnit,
    readonly_array,
)
from q2mm.optimizers.evaluators.hessian_element import (
    HessianElementEvaluator,
    HessianResult,
)
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ParameterLayout
from test.backend_fixtures import MockLayout, mock_molecule

#: Dummy full parameter vector (the fakes ignore it).
P = np.zeros(2)


class _FakePrepared(AbstractPreparedBackend):
    """Prepared-session double returning fixed Hessian/Jacobian data."""

    def __init__(
        self, *, hessian: object = None, hess_jac: object = None, supports_jac: bool = False, molecule: object = None
    ) -> None:
        caps = {Capability.HESSIAN}
        if supports_jac:
            caps.add(Capability.HESSIAN_PARAMETER_JACOBIAN)
        info = BackendInfo(
            name="mock",
            role=BackendRole.MM,
            capabilities=frozenset(caps),
            functional_forms=frozenset({"harmonic"}),
            provenance=BackendProvenance(backend="mock", role=BackendRole.MM),
        )
        # Derive a physically-consistent molecule (N = 3N/3 atoms) from the
        # Hessian so central (3N, 3N) result validation can run.
        if molecule is None and hessian is not None:
            n3 = np.asarray(hessian).shape[0]
            molecule = mock_molecule(["X"] * (n3 // 3))
        super().__init__(
            info=info,
            case_id="0",
            molecule=molecule,
            force_field=None,
            layout=MockLayout(np.asarray(hess_jac).shape[2]) if hess_jac is not None else None,
        )
        self._h = hessian
        self._j = hess_jac

    def _hessian(self, request: object) -> _CHessianResult:
        return _CHessianResult(
            hessian=readonly_array(self._h), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=self._info.provenance
        )

    def _hessian_parameter_jacobian(self, request: object) -> HessianJacobianResult:
        return HessianJacobianResult(
            hessian=readonly_array(self._h),
            jacobian=readonly_array(self._j),
            unit=HessianUnit.HARTREE_PER_BOHR2,
            provenance=self._info.provenance,
        )


class _FakeBackend:
    """Backend double whose ``prepare`` returns a fixed prepared session."""

    def __init__(self, prepared: _FakePrepared) -> None:
        self._p = prepared
        self.info = prepared.info

    def prepare(self, request: object) -> _FakePrepared:
        self._p._molecule = request.molecule  # type: ignore[attr-defined]
        return self._p


# ---- Fixtures ----


@pytest.fixture
def small_hessian() -> np.ndarray:
    """Small 3×3 symmetric Hessian for testing."""
    return np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])


@pytest.fixture
def hessian_6x6() -> np.ndarray:
    """6×6 symmetric Hessian for skip_translational testing."""
    rng = np.random.default_rng(42)
    h = rng.standard_normal((6, 6))
    return (h + h.T) / 2  # Symmetrise


@pytest.fixture
def mock_engine(small_hessian: np.ndarray) -> _FakePrepared:
    """Prepared-session double returning the small_hessian."""
    return _FakePrepared(hessian=small_hessian)


@pytest.fixture
def evaluator() -> HessianElementEvaluator:
    return HessianElementEvaluator()


# ---- HessianElementEvaluator.compute ----


class TestHessianElementCompute:
    def test_compute_returns_hessian(
        self,
        evaluator: HessianElementEvaluator,
        mock_engine: _FakePrepared,
        small_hessian: np.ndarray,
    ) -> None:
        """compute() calls prepared.hessian and wraps result."""
        result = evaluator.compute(mock_engine, P)

        assert isinstance(result, HessianResult)
        np.testing.assert_array_equal(result.hessian, small_hessian)

    def test_compute_reuses_prepared_session(
        self,
        evaluator: HessianElementEvaluator,
        mock_engine: _FakePrepared,
        small_hessian: np.ndarray,
    ) -> None:
        """compute() reads the Hessian from the prepared session's typed result."""
        result = evaluator.compute(mock_engine, P)
        np.testing.assert_array_equal(result.hessian, small_hessian)


# ---- HessianElementEvaluator._extract ----


class TestHessianElementExtract:
    def test_extract_diagonal(self, small_hessian: np.ndarray) -> None:
        """Diagonal element extraction at (1,1)."""
        computed = HessianResult(hessian=small_hessian)
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(1, 1))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 3.0

    def test_extract_offdiagonal(self, small_hessian: np.ndarray) -> None:
        """Off-diagonal element extraction at (2,0)."""
        computed = HessianResult(hessian=small_hessian)
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(2, 0))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 0.5

    def test_extract_corner(self, small_hessian: np.ndarray) -> None:
        """Element at (0,0)."""
        computed = HessianResult(hessian=small_hessian)
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(0, 0))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 4.0

    def test_extract_out_of_range_raises(self, small_hessian: np.ndarray) -> None:
        """Out-of-range indices raise IndexError."""
        computed = HessianResult(hessian=small_hessian)
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(5, 0), label="bad")

        with pytest.raises(IndexError, match="out of range"):
            HessianElementEvaluator._extract(computed, ref)

    def test_extract_missing_atom_indices_raises(self, small_hessian: np.ndarray) -> None:
        """Missing atom_indices raises ValueError."""
        computed = HessianResult(hessian=small_hessian)
        ref = Observation(kind="hessian_element", value=0.0)

        with pytest.raises(ValueError, match="requires atom_indices"):
            HessianElementEvaluator._extract(computed, ref)


# ---- HessianElementEvaluator.residuals ----


class TestHessianElementResiduals:
    def test_residuals_correct(
        self,
        evaluator: HessianElementEvaluator,
        small_hessian: np.ndarray,
    ) -> None:
        """Residuals are weight * (ref - calc)."""
        computed = HessianResult(hessian=small_hessian)
        refs = [
            Observation(kind="hessian_element", value=5.0, weight=0.1, atom_indices=(0, 0)),
            Observation(kind="hessian_element", value=3.0, weight=0.2, atom_indices=(1, 1)),
        ]

        residuals = evaluator.residuals(computed, refs)

        assert len(residuals) == 2
        # ref=5.0, calc=4.0, diff=1.0, w=0.1 → 0.1
        assert residuals[0] == pytest.approx(0.1)
        # ref=3.0, calc=3.0, diff=0.0, w=0.2 → 0.0
        assert residuals[1] == pytest.approx(0.0)


# ---- ObservationSet.add_hessian_element ----


class TestObservationSetAddHessianElement:
    def test_add_hessian_element(self) -> None:
        """with_hessian_element creates a hessian_element Observation."""
        ref = ObservationSet()
        ref = ref.with_hessian_element(1.5, row=2, col=1, weight=0.1, label="test")

        assert ref.n_observations == 1
        rv = ref.values[0]
        assert rv.kind == "hessian_element"
        assert rv.value == 1.5
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.1
        assert rv.label == "test"

    def test_add_hessian_element_default_label(self) -> None:
        """Default label is generated from row/col."""
        ref = ObservationSet()
        ref = ref.with_hessian_element(0.5, row=3, col=4)

        assert ref.values[0].label == "hess[3,4]"


# ---- ObservationSet.with_hessian_from_matrix ----


class TestObservationSetAddHessianFromMatrix:
    def test_full_lower_triangle(self, small_hessian: np.ndarray) -> None:
        """Full loading adds n*(n+1)/2 elements."""
        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(small_hessian)

        # 3×3 lower triangle: 3*(3+1)/2 = 6
        assert ref.n_observations == 6

        # Check all are hessian_element kind
        assert all(rv.kind == "hessian_element" for rv in ref.values)

        # Check diagonal vs off-diagonal weights
        diag_entries = [rv for rv in ref.values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag_entries = [rv for rv in ref.values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert len(diag_entries) == 3
        assert len(offdiag_entries) == 3
        assert all(rv.weight == 0.1 for rv in diag_entries)
        assert all(rv.weight == 0.05 for rv in offdiag_entries)

    def test_diagonal_only(self, small_hessian: np.ndarray) -> None:
        """diagonal_only=True adds only N elements."""
        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(small_hessian, diagonal_only=True)

        assert ref.n_observations == 3
        # All entries should be on diagonal
        for rv in ref.values:
            assert rv.atom_indices[0] == rv.atom_indices[1]
        # Values should match diagonal
        expected_values = [4.0, 3.0, 2.0]
        actual_values = [rv.value for rv in ref.values]
        assert actual_values == expected_values

    def test_skip_translational(self, hessian_6x6: np.ndarray) -> None:
        """skip_translational skips leading rows/cols."""
        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(
            hessian_6x6,
            skip_translational=3,
            diagonal_only=True,
        )

        # Only indices 3, 4, 5 → 3 diagonal entries
        assert ref.n_observations == 3
        rows = [rv.atom_indices[0] for rv in ref.values]
        assert rows == [3, 4, 5]

    def test_skip_translational_full(self, hessian_6x6: np.ndarray) -> None:
        """skip_translational with full lower triangle."""
        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(
            hessian_6x6,
            skip_translational=3,
        )

        # Remaining 3×3 block: 3*(3+1)/2 = 6
        assert ref.n_observations == 6

    def test_non_square_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        hess = np.ones((3, 4))
        ref = ObservationSet()

        with pytest.raises(ValueError, match="square"):
            ref.with_hessian_from_matrix(hess)

    def test_custom_weights(self, small_hessian: np.ndarray) -> None:
        """Custom diagonal/offdiagonal weights are applied."""
        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(
            small_hessian,
            diagonal_weight=0.5,
            offdiagonal_weight=0.2,
        )

        diag_entries = [rv for rv in ref.values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag_entries = [rv for rv in ref.values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert all(rv.weight == 0.5 for rv in diag_entries)
        assert all(rv.weight == 0.2 for rv in offdiag_entries)


# ---- YAML round-trip ----


class TestYAMLRoundTrip:
    def test_hessian_element_parse_and_serialize(self) -> None:
        """hessian_element kind parses from YAML dict and serializes back."""
        from q2mm.io.reference import _parse_datum, _reference_value_to_dict

        datum = {
            "kind": "hessian_element",
            "value": 1.23,
            "row": 2,
            "col": 1,
            "weight": 0.1,
        }

        refs = _parse_datum(datum, case_id="0", context="test")
        assert len(refs) == 1
        rv = refs[0]
        assert rv.kind == "hessian_element"
        assert rv.value == 1.23
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.1
        assert rv.label == "hess[2,1]"

        # Serialize back
        d = _reference_value_to_dict(rv)
        assert d["kind"] == "hessian_element"
        assert d["value"] == 1.23
        assert d["row"] == 2
        assert d["col"] == 1
        assert d["weight"] == 0.1

    def test_hessian_element_with_label(self) -> None:
        """Custom label is preserved."""
        from q2mm.io.reference import _parse_datum

        datum = {
            "kind": "hessian_element",
            "value": 0.5,
            "row": 0,
            "col": 0,
            "label": "H(0,0)",
        }

        refs = _parse_datum(datum, case_id="0", context="test")
        assert refs[0].label == "H(0,0)"

    def test_hessian_element_negative_indices_rejected(self) -> None:
        """Negative row/col indices are rejected."""
        from q2mm.io.reference import ReferenceYAMLError, _parse_datum

        datum = {
            "kind": "hessian_element",
            "value": 0.5,
            "row": -1,
            "col": 0,
        }

        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            _parse_datum(datum, case_id="0", context="test")


# ---- YAML bulk hessian directive ----


class TestYAMLBulkHessian:
    def test_bulk_hessian_directive(self, tmp_path: Path) -> None:
        """kind: hessian bulk directive creates hessian_element entries."""
        from q2mm.io.reference import _load_molecule

        hess = np.diag([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        mol_dict = {
            "name": "test_mol",
            "geometry": {
                "symbols": ["H", "H"],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "data": [{"kind": "hessian", "diagonal_only": True}],
        }

        hess_path = tmp_path / "test_hessian.npy"
        np.save(str(hess_path), hess)
        mol_dict["hessian"] = str(hess_path)
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)

        assert mol.hessian is not None
        assert len(ref_values) == 6  # diagonal_only, 6×6 → 6
        assert all(rv.kind == "hessian_element" for rv in ref_values)
        assert all(rv.atom_indices[0] == rv.atom_indices[1] for rv in ref_values)

    def test_bulk_hessian_full(self, tmp_path: Path) -> None:
        """kind: hessian without diagonal_only creates full lower triangle."""
        from q2mm.io.reference import _load_molecule

        hess = np.diag([4.0, 3.0, 2.0, 1.0, 0.5, 0.25])
        hess_path = tmp_path / "test_hessian_full.npy"
        np.save(str(hess_path), hess)
        mol_dict = {
            "name": "test_mol",
            "geometry": {
                "symbols": ["H", "H"],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "hessian": str(hess_path),
            "data": [{"kind": "hessian"}],
        }
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)

        # 6×6 lower triangle: 6*7/2 = 21 elements
        assert len(ref_values) == 21
        diag = [rv for rv in ref_values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag = [rv for rv in ref_values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert len(diag) == 6
        assert len(offdiag) == 15

    def test_bulk_hessian_skip_translational(self, tmp_path: Path) -> None:
        """skip_translational parameter works in bulk directive."""
        from q2mm.io.reference import _load_molecule

        hess = np.eye(6)
        hess_path = tmp_path / "test_hessian_skip.npy"
        np.save(str(hess_path), hess)
        mol_dict = {
            "name": "test_mol",
            "geometry": {
                "symbols": ["H", "H"],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "hessian": str(hess_path),
            "data": [{"kind": "hessian", "skip_translational": 2, "diagonal_only": True}],
        }
        mol, ref_values = _load_molecule(mol_dict, tmp_path, 0)

        # skip 2, diagonal only → indices 2, 3, 4, 5
        assert len(ref_values) == 4
        rows = [rv.atom_indices[0] for rv in ref_values]
        assert rows == [2, 3, 4, 5]

    def test_bulk_hessian_no_hessian_raises(self, tmp_path: Path) -> None:
        """kind: hessian raises when molecule has no hessian."""
        from q2mm.io.reference import ReferenceYAMLError, _load_molecule

        mol_dict = {
            "name": "test_mol",
            "geometry": {
                "symbols": ["H", "H"],
                "coordinates": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            },
            "data": [{"kind": "hessian"}],
        }

        with pytest.raises(ReferenceYAMLError, match="requires a molecule with a hessian"):
            _load_molecule(mol_dict, tmp_path, 0)


# ---- ObjectiveFunction integration ----


class TestObjectiveFunctionHessianElement:
    def test_extract_hessian_element(self) -> None:
        """_extract_value handles hessian_element kind."""
        from q2mm.optimizers.objective import ObjectiveFunction

        hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        calc = {"raw_hessian": hessian}
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(2, 1))

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 0.2

    def test_extract_hessian_element_diagonal(self) -> None:
        """_extract_value extracts diagonal hessian element."""
        from q2mm.optimizers.objective import ObjectiveFunction

        hessian = np.diag([1.0, 2.0, 3.0])
        calc = {"raw_hessian": hessian}
        ref = Observation(kind="hessian_element", value=0.0, atom_indices=(2, 2))

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 3.0

    def test_evaluate_molecule_hessian_element(self) -> None:
        """_evaluate_molecule computes raw_hessian for hessian_element refs."""
        from q2mm.models.forcefield import ForceField, FunctionalForm
        from q2mm.optimizers.objective import ObjectiveFunction

        hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mol = mock_molecule(["X"])
        mol.name = "test"
        mol.hessian = None  # Not needed for raw Hessian (no eigendecomposition)

        backend = _FakeBackend(_FakePrepared(hessian=hessian))

        ref = ObservationSet()
        ref = ref.with_hessian_element(4.0, row=0, col=0, weight=0.1)
        ref = ref.with_hessian_element(1.0, row=1, col=0, weight=0.05)

        ff = ForceField(functional_form=FunctionalForm.HARMONIC)
        layout = ParameterLayout.from_force_field(ff)
        obj = ObjectiveFunction(forcefield=ff, backend=backend, molecules=[mol], reference=ref, layout=layout)

        result = obj._evaluate_molecule(0, layout.vector(ff))
        assert "raw_hessian" in result
        np.testing.assert_array_equal(result["raw_hessian"], hessian)

    def test_full_objective_with_hessian_elements(self) -> None:
        """Full objective evaluation with hessian_element references."""
        from q2mm.models.forcefield import ForceField, FunctionalForm
        from q2mm.optimizers.objective import ObjectiveFunction

        qm_hessian = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mm_hessian = np.array([[4.1, 1.1, 0.5], [1.1, 2.9, 0.2], [0.5, 0.2, 2.0]])

        mol = mock_molecule(["X"])
        mol.name = "test"
        mol.hessian = None

        backend = _FakeBackend(_FakePrepared(hessian=mm_hessian))

        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(qm_hessian, diagonal_only=True)

        ff = ForceField(functional_form=FunctionalForm.HARMONIC)
        layout = ParameterLayout.from_force_field(ff)
        obj = ObjectiveFunction(forcefield=ff, backend=backend, molecules=[mol], reference=ref, layout=layout)

        score = obj(layout.vector(ff))
        assert score > 0  # Non-zero since MM != QM
        assert isinstance(score, float)

    def test_supports_analytical_gradient_false(self) -> None:
        """HessianElementEvaluator returns False when backend has no Hessian Jacobians."""
        ev = HessianElementEvaluator()
        backend = _FakePrepared(supports_jac=False)
        assert ev.supports_analytical_gradient(backend) is False

    def test_supports_analytical_gradient_true(self) -> None:
        """HessianElementEvaluator returns True when backend supports Hessian Jacobians."""
        ev = HessianElementEvaluator()
        backend = _FakePrepared(supports_jac=True)
        assert ev.supports_analytical_gradient(backend) is True

    def test_gradient_computes(self) -> None:
        """gradient() computes an actual gradient array."""
        ev = HessianElementEvaluator()
        hess = np.eye(3)
        dH_dp = np.zeros((3, 3, 2))
        dH_dp[0, 1, 0] = 1.0
        backend = _FakePrepared(hessian=hess, hess_jac=dH_dp, supports_jac=True, molecule=mock_molecule(["X"]))
        refs = [Observation(kind="hessian_element", value=0.5, weight=1.0, data_idx=0, atom_indices=(0, 1))]
        result = ev.gradient(backend, P, refs, 2)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_reset_does_nothing(self) -> None:
        """reset() runs without error."""
        ev = HessianElementEvaluator()
        ev.reset()  # Should not raise
