"""Tests for eigenmatrix training data pipeline.

Tests the building blocks in hessian.py (transform_to_eigenmatrix,
extract_eigenmatrix_data) and the ReferenceData/ObjectiveFunction
integration in objective.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from test._shared import GS_FCHK, SN2_DATA_AVAILABLE, SN2_QM_REF

from q2mm.models.hessian import (
    decompose,
    extract_eigenmatrix_data,
    reform_hessian,
    transform_to_eigenmatrix,
)
from q2mm.optimizers.objective import ReferenceData, ReferenceValue

# ---- Fixtures ----

_ETHANE_FCHK = GS_FCHK
_ETHANE_DATA_AVAILABLE = _ETHANE_FCHK.exists()


@pytest.fixture
def symmetric_matrix() -> np.ndarray:
    """Small symmetric test matrix with known eigenvalues."""
    return np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])


@pytest.fixture
def sn2_hessian() -> np.ndarray:
    if not SN2_DATA_AVAILABLE:
        pytest.skip("SN2 data not available")
    return np.load(str(SN2_QM_REF / "sn2-ts-hessian.npy"))


# ---- transform_to_eigenmatrix ----


class TestTransformToEigenmatrix:
    def test_self_projection_is_diagonal(self, symmetric_matrix: np.ndarray) -> None:
        """Projecting a Hessian onto its own eigenvectors gives a diagonal matrix."""
        evals, evecs = decompose(symmetric_matrix)
        eigenmatrix = transform_to_eigenmatrix(symmetric_matrix, evecs)

        # Off-diagonal elements should be zero
        off_diag = eigenmatrix - np.diag(np.diag(eigenmatrix))
        assert np.allclose(off_diag, 0, atol=1e-12)

        # Diagonal should match eigenvalues (sorted by eigh)
        np.testing.assert_allclose(np.diag(eigenmatrix), evals, atol=1e-12)

    def test_different_hessian_not_diagonal(self, symmetric_matrix: np.ndarray) -> None:
        """Projecting a different matrix onto the eigenvectors is generally NOT diagonal."""
        _, evecs = decompose(symmetric_matrix)
        other = np.array([[2.0, 0.5, 0.1], [0.5, 1.0, 0.3], [0.1, 0.3, 3.0]])
        eigenmatrix = transform_to_eigenmatrix(other, evecs)

        off_diag = eigenmatrix - np.diag(np.diag(eigenmatrix))
        assert not np.allclose(off_diag, 0, atol=1e-4)

    def test_roundtrip_consistency(self, symmetric_matrix: np.ndarray) -> None:
        """Eigenmatrix of self-projection should round-trip via reform_hessian."""
        evals, evecs = decompose(symmetric_matrix)
        eigenmatrix = transform_to_eigenmatrix(symmetric_matrix, evecs)
        diagonal_evals = np.diag(eigenmatrix)

        reformed = reform_hessian(diagonal_evals, evecs)
        np.testing.assert_allclose(reformed, symmetric_matrix, atol=1e-12)

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_self_projection(self, sn2_hessian: np.ndarray) -> None:
        """SN2 TS Hessian self-projection should be diagonal with eigenvalues."""
        evals, evecs = decompose(sn2_hessian)
        eigenmatrix = transform_to_eigenmatrix(sn2_hessian, evecs)

        np.testing.assert_allclose(np.diag(eigenmatrix), evals, atol=1e-10)

        off_diag_norm = np.linalg.norm(eigenmatrix - np.diag(np.diag(eigenmatrix)))
        assert off_diag_norm < 1e-10

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_has_negative_eigenvalue(self, sn2_hessian: np.ndarray) -> None:
        """SN2 TS should have a negative eigenvalue (reaction coordinate)."""
        evals, evecs = decompose(sn2_hessian)
        eigenmatrix = transform_to_eigenmatrix(sn2_hessian, evecs)

        diagonal = np.diag(eigenmatrix)
        assert np.any(diagonal < -0.01), "Expected negative eigenvalue for TS"


# ---- extract_eigenmatrix_data ----


class TestExtractEigenmatrixData:
    def test_diagonal_only(self) -> None:
        """diagonal_only=True returns only diagonal elements."""
        mat = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 3.0]])
        data = extract_eigenmatrix_data(mat, diagonal_only=True)

        assert len(data) == 3
        for row, col, val in data:
            assert row == col
        assert data[0] == (0, 0, 1.0)
        assert data[1] == (1, 1, 2.0)
        assert data[2] == (2, 2, 3.0)

    def test_lower_triangular(self) -> None:
        """Default returns all lower-triangular elements."""
        mat = np.array([[1.0, 0.5], [0.5, 2.0]])
        data = extract_eigenmatrix_data(mat)

        # Lower triangle of 2x2: (0,0), (1,0), (1,1) = 3 elements
        assert len(data) == 3
        assert data[0] == (0, 0, 1.0)
        assert data[1] == (1, 0, 0.5)
        assert data[2] == (1, 1, 2.0)

    def test_count_for_3x3(self) -> None:
        """3x3 lower triangle has 6 elements."""
        mat = np.eye(3)
        data = extract_eigenmatrix_data(mat)
        assert len(data) == 6  # n*(n+1)/2

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_diagonal_count(self, sn2_hessian: np.ndarray) -> None:
        """SN2 has 6 atoms → 18x18 Hessian → 18 diagonal eigenvalues."""
        evals, evecs = decompose(sn2_hessian)
        eigenmatrix = transform_to_eigenmatrix(sn2_hessian, evecs)
        data = extract_eigenmatrix_data(eigenmatrix, diagonal_only=True)
        assert len(data) == 18

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_full_count(self, sn2_hessian: np.ndarray) -> None:
        """18x18 lower triangle = 18*19/2 = 171 elements."""
        evals, evecs = decompose(sn2_hessian)
        eigenmatrix = transform_to_eigenmatrix(sn2_hessian, evecs)
        data = extract_eigenmatrix_data(eigenmatrix)
        assert len(data) == 18 * 19 // 2  # 171


# ---- ReferenceData eigenvalue support ----


class TestReferenceDataEigenvalues:
    def test_add_hessian_eigenvalue(self) -> None:
        """add_hessian_eigenvalue creates eig_diagonal entries."""
        ref = ReferenceData()
        ref.add_hessian_eigenvalue(1.5, mode_idx=3, weight=0.1, label="mode 3")

        assert ref.n_observations == 1
        rv = ref.values[0]
        assert rv.kind == "eig_diagonal"
        assert rv.value == 1.5
        assert rv.data_idx == 3
        assert rv.weight == 0.1

    def test_add_hessian_offdiagonal(self) -> None:
        """add_hessian_offdiagonal creates eig_offdiagonal entries."""
        ref = ReferenceData()
        ref.add_hessian_offdiagonal(0.001, row=2, col=1, weight=0.05)

        assert ref.n_observations == 1
        rv = ref.values[0]
        assert rv.kind == "eig_offdiagonal"
        assert rv.value == 0.001
        assert rv.atom_indices == (2, 1)
        assert rv.weight == 0.05

    def test_add_eigenmatrix_from_hessian_diagonal_only(self) -> None:
        """Bulk loader with diagonal_only adds N eigenvalues."""
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
        ref = ReferenceData()
        n_added = ref.add_eigenmatrix_from_hessian(hess, diagonal_only=True)

        assert n_added == 2
        assert ref.n_observations == 2
        assert all(rv.kind == "eig_diagonal" for rv in ref.values)

    def test_add_eigenmatrix_from_hessian_full(self) -> None:
        """Bulk loader without diagonal_only adds N*(N+1)/2 elements."""
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
        ref = ReferenceData()
        n_added = ref.add_eigenmatrix_from_hessian(hess, diagonal_only=False)

        # 2x2 lower triangle: 3 elements
        assert n_added == 3
        kinds = [rv.kind for rv in ref.values]
        assert kinds.count("eig_diagonal") == 2
        assert kinds.count("eig_offdiagonal") == 1

    def test_weight_scheme_skip_first(self) -> None:
        """First eigenvalue gets eig_i weight (default 0.0) when skip_first=True."""
        hess = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(hess, diagonal_only=True, skip_first=True)

        # First entry should have weight 0.0 (eig_i)
        assert ref.values[0].weight == 0.0
        # Others should have non-zero weight
        assert all(rv.weight > 0 for rv in ref.values[1:])

    def test_weight_scheme_custom(self) -> None:
        """Custom weights override defaults."""
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(
            hess,
            diagonal_only=True,
            skip_first=False,
            weights={"eig_d_low": 0.5, "eig_d_high": 0.8},
        )

        # Both eigenvalues (~2.38, ~4.62) are above default threshold (0.1173)
        assert all(rv.weight == 0.8 for rv in ref.values)

    def test_eigenvalue_threshold_separates_weights(self) -> None:
        """eigenvalue_threshold correctly splits diagonal weights."""
        # eigenvalues of [[4, 1], [1, 3]] are ~2.38 and ~4.62
        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(
            hess,
            diagonal_only=True,
            skip_first=False,
            eigenvalue_threshold=3.5,  # splits: 2.38 < 3.5, 4.62 ≥ 3.5
            weights={"eig_d_low": 0.2, "eig_d_high": 0.9},
        )

        weights = sorted(rv.weight for rv in ref.values)
        assert weights == [0.2, 0.9]

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_eigenmatrix_reference_data(self, sn2_hessian: np.ndarray) -> None:
        """SN2 bulk loader produces 18 diagonal + 153 off-diagonal = 171 entries."""
        ref = ReferenceData()
        n = ref.add_eigenmatrix_from_hessian(sn2_hessian, diagonal_only=False)
        assert n == 171  # 18*19/2

        diag_count = sum(1 for rv in ref.values if rv.kind == "eig_diagonal")
        offdiag_count = sum(1 for rv in ref.values if rv.kind == "eig_offdiagonal")
        assert diag_count == 18
        assert offdiag_count == 153

    @pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
    def test_sn2_first_eigenvalue_weight_zero(self, sn2_hessian: np.ndarray) -> None:
        """For the SN2 TS, the first eigenvalue (imaginary mode) gets weight 0."""
        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(sn2_hessian, diagonal_only=True)

        first_eig = next(rv for rv in ref.values if rv.kind == "eig_diagonal" and rv.data_idx == 0)
        assert first_eig.weight == 0.0

    @pytest.mark.skipif(not _ETHANE_DATA_AVAILABLE, reason="Ethane fchk not found")
    def test_from_molecule_with_eigenmatrix(self) -> None:
        """from_molecule with include_eigenmatrix adds eigenvalue data."""
        # Use ReferenceData.from_fchk to get a molecule with Hessian, then test from_molecule
        ref_data, mol = ReferenceData.from_fchk(str(_ETHANE_FCHK))
        assert mol.hessian is not None

        ref = ReferenceData.from_molecule(mol, include_eigenmatrix=True, eigenmatrix_diagonal_only=True)

        # Should have bond_length + bond_angle + eig_diagonal entries
        kinds = {rv.kind for rv in ref.values}
        assert "bond_length" in kinds
        assert "bond_angle" in kinds
        assert "eig_diagonal" in kinds

        n_eig = sum(1 for rv in ref.values if rv.kind == "eig_diagonal")
        assert n_eig == 3 * mol.n_atoms  # 24 for ethane


# ---- ObjectiveFunction eigenvalue extraction ----


class TestObjectiveFunctionEigenmatrix:
    def test_extract_eig_diagonal(self) -> None:
        """_extract_value handles eig_diagonal kind."""
        from q2mm.optimizers.objective import ObjectiveFunction

        eigenmatrix = np.diag([1.0, 2.0, 3.0])
        calc = {"eigenmatrix": eigenmatrix}
        ref = ReferenceValue(kind="eig_diagonal", value=1.0, data_idx=1)

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 2.0  # eigenmatrix[1, 1]

    def test_extract_eig_offdiagonal(self) -> None:
        """_extract_value handles eig_offdiagonal kind."""
        from q2mm.optimizers.objective import ObjectiveFunction

        eigenmatrix = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 3.0]])
        calc = {"eigenmatrix": eigenmatrix}
        ref = ReferenceValue(kind="eig_offdiagonal", value=0.0, atom_indices=(2, 1))

        result = ObjectiveFunction._extract_value(calc, ref)
        assert result == 0.3  # eigenmatrix[2, 1]

    @pytest.mark.skipif(not _ETHANE_DATA_AVAILABLE, reason="Ethane fchk not found")
    def test_evaluate_molecule_eigenmatrix_projection_and_caching(self) -> None:
        """_evaluate_molecule computes eigenmatrix from engine.hessian using cached QM eigenvectors."""
        from q2mm.optimizers.objective import ObjectiveFunction

        ref_data, mol = ReferenceData.from_fchk(str(_ETHANE_FCHK))
        assert mol.hessian is not None

        qm_hessian = np.array(mol.hessian, dtype=float)

        class StubMMEngine:
            """Stub MM engine returning a fixed Hessian in canonical units."""

            name = "stub"

            def __init__(self, hessian: np.ndarray) -> None:
                self._hessian = np.array(hessian, dtype=float)
                self.hessian_calls = 0

            def hessian(self, structure: object, forcefield: object = None) -> np.ndarray:
                self.hessian_calls += 1
                return self._hessian

            def supports_runtime_params(self) -> bool:
                return False

        # Build reference data with eigenmatrix entries
        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(qm_hessian, diagonal_only=True)

        # Use MM Hessian == QM Hessian → self-projection should be diagonal
        engine = StubMMEngine(qm_hessian)
        obj = ObjectiveFunction(forcefield=None, engine=engine, molecules=[mol], reference=ref)

        result = obj._evaluate_molecule(0, obj.forcefield)
        assert "eigenmatrix" in result

        eigmat = np.array(result["eigenmatrix"], dtype=float)
        assert eigmat.shape == qm_hessian.shape

        # Self-projection → eigenmatrix should be diagonal
        diag_only = np.diag(np.diag(eigmat))
        assert np.allclose(eigmat, diag_only, atol=1e-8)
        assert engine.hessian_calls == 1

        # Second call: swap in a scaled engine, eigenvectors stay cached
        engine2 = StubMMEngine(2.0 * qm_hessian)
        obj.engine = engine2
        result2 = obj._evaluate_molecule(0, obj.forcefield)
        eigmat2 = np.array(result2["eigenmatrix"], dtype=float)

        # Diagonal should scale by 2
        assert np.allclose(np.diag(eigmat2), 2.0 * np.diag(eigmat), atol=1e-8)
        assert engine2.hessian_calls == 1
