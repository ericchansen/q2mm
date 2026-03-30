"""Tests for raw Hessian element training data pipeline.

Tests the HessianElementEvaluator and the ReferenceData/ObjectiveFunction
integration for raw Hessian matrix element training.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from pathlib import Path

import numpy as np
import pytest

from q2mm.optimizers.evaluators.hessian_element import (
    HessianElementEvaluator,
    HessianResult,
)
from q2mm.optimizers.objective import ReferenceData, ReferenceValue

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
def mock_engine(small_hessian: np.ndarray) -> MagicMock:
    """Mock MM engine returning the small_hessian."""
    engine = MagicMock()
    engine.hessian.return_value = small_hessian
    engine.name = "mock"
    return engine


@pytest.fixture
def evaluator() -> HessianElementEvaluator:
    return HessianElementEvaluator()


# ---- HessianElementEvaluator.compute ----


class TestHessianElementCompute:
    def test_compute_returns_hessian(
        self,
        evaluator: HessianElementEvaluator,
        mock_engine: MagicMock,
        small_hessian: np.ndarray,
    ) -> None:
        """compute() calls engine.hessian and wraps result."""
        mol = MagicMock()
        ff = MagicMock()

        result = evaluator.compute(mock_engine, mol, ff)

        assert isinstance(result, HessianResult)
        np.testing.assert_array_equal(result.hessian, small_hessian)
        mock_engine.hessian.assert_called_once_with(mol, ff)

    def test_compute_uses_structure_when_provided(
        self,
        evaluator: HessianElementEvaluator,
        mock_engine: MagicMock,
    ) -> None:
        """compute() passes structure instead of mol when given."""
        mol = MagicMock()
        ff = MagicMock()
        structure = MagicMock()

        evaluator.compute(mock_engine, mol, ff, structure=structure)

        mock_engine.hessian.assert_called_once_with(structure, ff)


# ---- HessianElementEvaluator._extract ----


class TestHessianElementExtract:
    def test_extract_diagonal(self, small_hessian: np.ndarray) -> None:
        """Diagonal element extraction at (1,1)."""
        computed = HessianResult(hessian=small_hessian)
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(1, 1))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 3.0

    def test_extract_offdiagonal(self, small_hessian: np.ndarray) -> None:
        """Off-diagonal element extraction at (2,0)."""
        computed = HessianResult(hessian=small_hessian)
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(2, 0))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 0.5

    def test_extract_corner(self, small_hessian: np.ndarray) -> None:
        """Element at (0,0)."""
        computed = HessianResult(hessian=small_hessian)
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(0, 0))

        result = HessianElementEvaluator._extract(computed, ref)
        assert result == 4.0

    def test_extract_out_of_range_raises(self, small_hessian: np.ndarray) -> None:
        """Out-of-range indices raise IndexError."""
        computed = HessianResult(hessian=small_hessian)
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(5, 0), label="bad")

        with pytest.raises(IndexError, match="out of range"):
            HessianElementEvaluator._extract(computed, ref)

    def test_extract_missing_atom_indices_raises(self, small_hessian: np.ndarray) -> None:
        """Missing atom_indices raises ValueError."""
        computed = HessianResult(hessian=small_hessian)
        ref = ReferenceValue(kind="hessian_element", value=0.0)

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
            ReferenceValue(kind="hessian_element", value=5.0, weight=0.1, atom_indices=(0, 0)),
            ReferenceValue(kind="hessian_element", value=3.0, weight=0.2, atom_indices=(1, 1)),
        ]

        residuals = evaluator.residuals(computed, refs)

        assert len(residuals) == 2
        # ref=5.0, calc=4.0, diff=1.0, w=0.1 → 0.1
        assert residuals[0] == pytest.approx(0.1)
        # ref=3.0, calc=3.0, diff=0.0, w=0.2 → 0.0
        assert residuals[1] == pytest.approx(0.0)


# ---- ReferenceData.add_hessian_element ----


class TestReferenceDataAddHessianElement:
    def test_add_hessian_element(self) -> None:
        """add_hessian_element creates a hessian_element ReferenceValue."""
        ref = ReferenceData()
        ref.add_hessian_element(1.5, row=2, col=1, weight=0.1, label="test")

        assert ref.n_observations == 1
        rv = ref.values[0]
        assert rv.kind == "hessian_element"
        assert rv.value == 1.5
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.1
        assert rv.label == "test"

    def test_add_hessian_element_default_label(self) -> None:
        """Default label is generated from row/col."""
        ref = ReferenceData()
        ref.add_hessian_element(0.5, row=3, col=4)

        assert ref.values[0].label == "hess[3,4]"


# ---- ReferenceData.add_hessian_from_matrix ----


class TestReferenceDataAddHessianFromMatrix:
    def test_full_lower_triangle(self, small_hessian: np.ndarray) -> None:
        """Full loading adds n*(n+1)/2 elements."""
        ref = ReferenceData()
        n_added = ref.add_hessian_from_matrix(small_hessian)

        # 3×3 lower triangle: 3*(3+1)/2 = 6
        assert n_added == 6
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
        ref = ReferenceData()
        n_added = ref.add_hessian_from_matrix(small_hessian, diagonal_only=True)

        assert n_added == 3
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
        ref = ReferenceData()
        n_added = ref.add_hessian_from_matrix(
            hessian_6x6,
            skip_translational=3,
            diagonal_only=True,
        )

        # Only indices 3, 4, 5 → 3 diagonal entries
        assert n_added == 3
        rows = [rv.atom_indices[0] for rv in ref.values]
        assert rows == [3, 4, 5]

    def test_skip_translational_full(self, hessian_6x6: np.ndarray) -> None:
        """skip_translational with full lower triangle."""
        ref = ReferenceData()
        n_added = ref.add_hessian_from_matrix(
            hessian_6x6,
            skip_translational=3,
        )

        # Remaining 3×3 block: 3*(3+1)/2 = 6
        assert n_added == 6

    def test_non_square_raises(self) -> None:
        """Non-square matrix raises ValueError."""
        hess = np.ones((3, 4))
        ref = ReferenceData()

        with pytest.raises(ValueError, match="square"):
            ref.add_hessian_from_matrix(hess)

    def test_custom_weights(self, small_hessian: np.ndarray) -> None:
        """Custom diagonal/offdiagonal weights are applied."""
        ref = ReferenceData()
        ref.add_hessian_from_matrix(
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
        from q2mm.parsers.reference_yaml import _parse_datum, _reference_value_to_dict

        datum = {
            "kind": "hessian_element",
            "value": 1.23,
            "row": 2,
            "col": 1,
            "weight": 0.1,
        }

        refs = _parse_datum(datum, molecule_idx=0, context="test")
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
        from q2mm.parsers.reference_yaml import _parse_datum

        datum = {
            "kind": "hessian_element",
            "value": 0.5,
            "row": 0,
            "col": 0,
            "label": "H(0,0)",
        }

        refs = _parse_datum(datum, molecule_idx=0, context="test")
        assert refs[0].label == "H(0,0)"

    def test_hessian_element_negative_indices_rejected(self) -> None:
        """Negative row/col indices are rejected."""
        from q2mm.parsers.reference_yaml import ReferenceYAMLError, _parse_datum

        datum = {
            "kind": "hessian_element",
            "value": 0.5,
            "row": -1,
            "col": 0,
        }

        with pytest.raises(ReferenceYAMLError, match="non-negative"):
            _parse_datum(datum, molecule_idx=0, context="test")


# ---- YAML bulk hessian directive ----


class TestYAMLBulkHessian:
    def test_bulk_hessian_directive(self, tmp_path: Path) -> None:
        """kind: hessian bulk directive creates hessian_element entries."""
        from q2mm.parsers.reference_yaml import _load_molecule

        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
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
        assert len(ref_values) == 2  # diagonal_only, 2×2 → 2
        assert all(rv.kind == "hessian_element" for rv in ref_values)
        assert all(rv.atom_indices[0] == rv.atom_indices[1] for rv in ref_values)

    def test_bulk_hessian_full(self, tmp_path: Path) -> None:
        """kind: hessian without diagonal_only creates full lower triangle."""
        from q2mm.parsers.reference_yaml import _load_molecule

        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
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

        # 2×2 lower triangle: 3 elements
        assert len(ref_values) == 3
        diag = [rv for rv in ref_values if rv.atom_indices[0] == rv.atom_indices[1]]
        offdiag = [rv for rv in ref_values if rv.atom_indices[0] != rv.atom_indices[1]]
        assert len(diag) == 2
        assert len(offdiag) == 1

    def test_bulk_hessian_skip_translational(self, tmp_path: Path) -> None:
        """skip_translational parameter works in bulk directive."""
        from q2mm.parsers.reference_yaml import _load_molecule

        hess = np.eye(4)
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

        # skip 2, diagonal only → indices 2, 3
        assert len(ref_values) == 2
        rows = [rv.atom_indices[0] for rv in ref_values]
        assert rows == [2, 3]

    def test_bulk_hessian_no_hessian_raises(self, tmp_path: Path) -> None:
        """kind: hessian raises when molecule has no hessian."""
        from q2mm.parsers.reference_yaml import ReferenceYAMLError, _load_molecule

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
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(2, 1))

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 0.2

    def test_extract_hessian_element_diagonal(self) -> None:
        """_extract_value extracts diagonal hessian element."""
        from q2mm.optimizers.objective import ObjectiveFunction

        hessian = np.diag([1.0, 2.0, 3.0])
        calc = {"raw_hessian": hessian}
        ref = ReferenceValue(kind="hessian_element", value=0.0, atom_indices=(2, 2))

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 3.0

    def test_evaluate_molecule_hessian_element(self) -> None:
        """_evaluate_molecule computes raw_hessian for hessian_element refs."""
        from q2mm.models.forcefield import ForceField
        from q2mm.optimizers.objective import ObjectiveFunction

        hessian = np.array([[4.0, 1.0], [1.0, 3.0]])
        mol = MagicMock()
        mol.name = "test"
        mol.hessian = None  # Not needed for raw Hessian (no eigendecomposition)

        engine = MagicMock()
        engine.hessian.return_value = hessian
        engine.supports_runtime_params.return_value = False

        ref = ReferenceData()
        ref.add_hessian_element(4.0, row=0, col=0, weight=0.1)
        ref.add_hessian_element(1.0, row=1, col=0, weight=0.05)

        ff = ForceField()
        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        result = obj._evaluate_molecule(0, ff)
        assert "raw_hessian" in result
        np.testing.assert_array_equal(result["raw_hessian"], hessian)
        engine.hessian.assert_called_once()

    def test_full_objective_with_hessian_elements(self) -> None:
        """Full objective evaluation with hessian_element references."""
        from q2mm.models.forcefield import ForceField
        from q2mm.optimizers.objective import ObjectiveFunction

        qm_hessian = np.array([[4.0, 1.0], [1.0, 3.0]])
        mm_hessian = np.array([[4.1, 1.1], [1.1, 2.9]])

        mol = MagicMock()
        mol.name = "test"
        mol.hessian = None

        engine = MagicMock()
        engine.hessian.return_value = mm_hessian
        engine.supports_runtime_params.return_value = False

        ref = ReferenceData()
        ref.add_hessian_from_matrix(qm_hessian, diagonal_only=True)

        ff = ForceField()
        obj = ObjectiveFunction(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        score = obj(ff.get_param_vector())
        assert score > 0  # Non-zero since MM != QM
        assert isinstance(score, float)

    def test_supports_analytical_gradient_false(self) -> None:
        """HessianElementEvaluator does not support analytical gradients."""
        ev = HessianElementEvaluator()
        engine = MagicMock()
        assert ev.supports_analytical_gradient(engine) is False

    def test_gradient_returns_none(self) -> None:
        """gradient() returns None."""
        ev = HessianElementEvaluator()
        result = ev.gradient(MagicMock(), MagicMock(), MagicMock(), [], 5)
        assert result is None

    def test_reset_does_nothing(self) -> None:
        """reset() runs without error."""
        ev = HessianElementEvaluator()
        ev.reset()  # Should not raise
