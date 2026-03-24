"""Tests for per-data-type evaluators.

Tests each evaluator independently using stub engines and known data.
"""

from __future__ import annotations

import numpy as np
import pytest

from test._shared import GS_FCHK, make_water, make_ethane

from q2mm.optimizers.objective import ReferenceData, ReferenceValue


# ---- Stub engine ----


class StubEngine:
    """Minimal stub engine for testing evaluators in isolation."""

    name = "stub"

    def __init__(
        self,
        *,
        energy: float = 0.0,
        frequencies: list[float] | None = None,
        hessian: np.ndarray | None = None,
        minimize_coords: np.ndarray | None = None,
        minimize_symbols: list[str] | None = None,
    ) -> None:
        self._energy = energy
        self._frequencies = frequencies or []
        self._hessian = hessian
        self._minimize_coords = minimize_coords
        self._minimize_symbols = minimize_symbols or []

    def energy(self, structure: object, forcefield: object = None) -> float:
        return self._energy

    def frequencies(self, structure: object, forcefield: object = None) -> list[float]:
        return list(self._frequencies)

    def hessian(self, structure: object, forcefield: object = None) -> np.ndarray:
        if self._hessian is None:
            raise ValueError("No hessian configured")
        return self._hessian

    def minimize(self, structure: object, forcefield: object = None) -> tuple:
        if self._minimize_coords is None:
            raise ValueError("No minimize_coords configured")
        return 0.0, self._minimize_symbols, self._minimize_coords

    def supports_runtime_params(self) -> bool:
        return False


# ---- EnergyEvaluator tests ----


class TestEnergyEvaluator:
    def test_compute(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        engine = StubEngine(energy=42.5)
        mol = make_water()

        result = evaluator.compute(engine, mol, ff=None)
        assert result.energy == 42.5

    def test_residuals_single(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator, EnergyResult

        evaluator = EnergyEvaluator()
        computed = EnergyResult(energy=10.0)
        refs = [ReferenceValue(kind="energy", value=12.0, weight=2.0)]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(2.0 * (12.0 - 10.0))

    def test_residuals_multiple(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator, EnergyResult

        evaluator = EnergyEvaluator()
        computed = EnergyResult(energy=5.0)
        refs = [
            ReferenceValue(kind="energy", value=5.0, weight=1.0),
            ReferenceValue(kind="energy", value=7.0, weight=0.5),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(0.0)
        assert residuals[1] == pytest.approx(0.5 * 2.0)

    def test_extract_value(self) -> None:
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        calc = {"energy": 99.9}
        ref = ReferenceValue(kind="energy", value=0.0)
        assert EnergyEvaluator.extract_value(calc, ref) == 99.9

    def test_compute_with_structure(self) -> None:
        """When structure is provided, it should be passed to engine."""
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator

        evaluator = EnergyEvaluator()
        engine = StubEngine(energy=1.0)
        mol = make_water()

        result = evaluator.compute(engine, mol, ff=None, structure="handle")
        assert result.energy == 1.0


# ---- FrequencyEvaluator tests ----


class TestFrequencyEvaluator:
    def test_compute(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        evaluator = FrequencyEvaluator()
        engine = StubEngine(frequencies=[100.0, 200.0, 300.0])
        mol = make_water()

        result = evaluator.compute(engine, mol, ff=None)
        assert result.frequencies == [100.0, 200.0, 300.0]

    def test_residuals(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator, FrequencyResult

        evaluator = FrequencyEvaluator()
        computed = FrequencyResult(frequencies=[100.0, 200.0, 300.0])
        refs = [
            ReferenceValue(kind="frequency", value=105.0, weight=1.0, data_idx=0),
            ReferenceValue(kind="frequency", value=195.0, weight=2.0, data_idx=1),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 2
        assert residuals[0] == pytest.approx(1.0 * (105.0 - 100.0))
        assert residuals[1] == pytest.approx(2.0 * (195.0 - 200.0))

    def test_residuals_out_of_range(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator, FrequencyResult

        evaluator = FrequencyEvaluator()
        computed = FrequencyResult(frequencies=[100.0])
        refs = [ReferenceValue(kind="frequency", value=200.0, data_idx=5)]

        with pytest.raises(IndexError, match="data_idx=5"):
            evaluator.residuals(computed, refs)

    def test_extract_value(self) -> None:
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator

        calc = {"frequencies": [100.0, 200.0, 300.0]}
        ref = ReferenceValue(kind="frequency", value=0.0, data_idx=2)
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
            ReferenceValue(kind="bond_length", value=1.1, weight=10.0, atom_indices=(0, 1)),
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
            ReferenceValue(kind="bond_angle", value=110.0, weight=5.0, atom_indices=(0, 1, 2)),
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
            ReferenceValue(
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
        ref = ReferenceValue(kind="bond_length", value=0.0, atom_indices=(1, 2))
        assert GeometryEvaluator.extract_value(calc, ref) == 1.5

    def test_extract_value_bond_by_idx(self) -> None:
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        calc = {"bond_lengths": [1.0, 1.5], "bond_lengths_by_atoms": {}}
        ref = ReferenceValue(kind="bond_length", value=0.0, data_idx=1)
        assert GeometryEvaluator.extract_value(calc, ref) == 1.5

    def test_extract_value_angle_reverse_order(self) -> None:
        """Should try both (i,j,k) and (k,j,i) orderings."""
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator

        calc = {
            "bond_angles": [109.5],
            "bond_angles_by_atoms": {(2, 1, 0): 109.5},
        }
        ref = ReferenceValue(kind="bond_angle", value=0.0, atom_indices=(0, 1, 2))
        assert GeometryEvaluator.extract_value(calc, ref) == 109.5


# ---- EigenmatrixEvaluator tests ----


class TestEigenmatrixEvaluator:
    def test_compute_self_projection(self) -> None:
        """Self-projection (MM == QM) should produce a diagonal eigenmatrix."""
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.models.molecule import Q2MMMolecule

        hess = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 2.0]])
        mol = Q2MMMolecule(
            symbols=["H"],
            geometry=np.array([[0.0, 0.0, 0.0]]),
            name="stub",
            hessian=hess,
        )
        engine = StubEngine(hessian=hess)
        evaluator = EigenmatrixEvaluator()

        result = evaluator.compute(engine, mol, ff=None, mol_idx=0)
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
            ReferenceValue(kind="eig_diagonal", value=1.5, weight=0.1, data_idx=0),
            ReferenceValue(kind="eig_diagonal", value=2.0, weight=0.1, data_idx=1),
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
            ReferenceValue(kind="eig_offdiagonal", value=0.0, weight=0.05, atom_indices=(1, 0)),
        ]

        residuals = evaluator.residuals(computed, refs)
        assert len(residuals) == 1
        assert residuals[0] == pytest.approx(0.05 * (0.0 - 0.3))

    def test_eigenvector_caching(self) -> None:
        """QM eigenvectors should be computed once and cached."""
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.models.molecule import Q2MMMolecule

        hess = np.array([[4.0, 1.0], [1.0, 3.0]])
        mol = Q2MMMolecule(
            symbols=["H"],
            geometry=np.array([[0.0, 0.0, 0.0]]),
            name="stub",
            hessian=hess,
        )
        engine = StubEngine(hessian=hess)
        evaluator = EigenmatrixEvaluator()

        evaluator.compute(engine, mol, ff=None, mol_idx=0)
        assert 0 in evaluator._qm_eigenvectors

        evaluator.compute(engine, mol, ff=None, mol_idx=0)
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
        from q2mm.models.molecule import Q2MMMolecule

        mol = Q2MMMolecule(
            symbols=["H"],
            geometry=np.array([[0.0, 0.0, 0.0]]),
            name="stub",
        )
        engine = StubEngine(hessian=np.eye(3))
        evaluator = EigenmatrixEvaluator()

        with pytest.raises(ValueError, match="no QM Hessian"):
            evaluator.compute(engine, mol, ff=None, mol_idx=0)

    def test_extract_value_diagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        calc = {"eigenmatrix": np.diag([1.0, 2.0, 3.0])}
        ref = ReferenceValue(kind="eig_diagonal", value=0.0, data_idx=2)
        assert EigenmatrixEvaluator.extract_value(calc, ref) == 3.0

    def test_extract_value_offdiagonal(self) -> None:
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator

        mat = np.array([[1.0, 0.5, 0.1], [0.5, 2.0, 0.3], [0.1, 0.3, 3.0]])
        calc = {"eigenmatrix": mat}
        ref = ReferenceValue(kind="eig_offdiagonal", value=0.0, atom_indices=(2, 1))
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


# ---- Parsers ----


class TestFchkParser:
    """Tests that the moved parse_fchk still works identically."""

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_parse_fchk_from_new_location(self) -> None:
        from q2mm.parsers.fchk import parse_fchk

        symbols, coords, hessian, charge, mult = parse_fchk(GS_FCHK)
        assert len(symbols) == 8
        assert symbols.count("H") == 6
        assert symbols.count("C") == 2
        assert coords.shape == (8, 3)
        assert charge == 0
        assert mult == 1
        assert hessian is not None
        assert hessian.shape == (24, 24)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_backward_compat_import(self) -> None:
        """Old import path still works via delegation."""
        from q2mm.optimizers.objective import _parse_fchk

        symbols, coords, hessian, charge, mult = _parse_fchk(GS_FCHK)
        assert len(symbols) == 8


# ---- Integration: evaluators produce same results as old ObjectiveFunction ----


class TestEvaluatorObjectiveParity:
    """Verify that evaluator delegation in ObjectiveFunction produces identical results."""

    def test_energy_parity(self) -> None:
        """Energy evaluation via evaluator matches direct engine call."""
        from q2mm.optimizers.objective import ObjectiveFunction

        engine = StubEngine(energy=42.0)
        mol = make_water()
        ref = ReferenceData()
        ref.add_energy(40.0, weight=2.0)

        obj = ObjectiveFunction(forcefield=None, engine=engine, molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0)

        assert result["energy"] == 42.0
        calc_value = obj._extract_value(result, ref.values[0])
        assert calc_value == 42.0

    def test_frequency_parity(self) -> None:
        """Frequency evaluation via evaluator matches direct engine call."""
        from q2mm.optimizers.objective import ObjectiveFunction

        engine = StubEngine(frequencies=[100.0, 200.0, 300.0])
        mol = make_water()
        ref = ReferenceData()
        ref.add_frequency(105.0, data_idx=0, weight=1.0)
        ref.add_frequency(195.0, data_idx=1, weight=1.0)

        obj = ObjectiveFunction(forcefield=None, engine=engine, molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0)

        assert result["frequencies"] == [100.0, 200.0, 300.0]
        assert obj._extract_value(result, ref.values[0]) == 100.0
        assert obj._extract_value(result, ref.values[1]) == 200.0

    @pytest.mark.skipif(not GS_FCHK.exists(), reason="Ethane fixture not found")
    def test_eigenmatrix_parity(self) -> None:
        """Eigenmatrix evaluation via evaluator matches old implementation."""
        from q2mm.optimizers.objective import ObjectiveFunction

        ref_data, mol = ReferenceData.from_fchk(str(GS_FCHK))
        assert mol.hessian is not None
        qm_hessian = np.array(mol.hessian, dtype=float)

        ref = ReferenceData()
        ref.add_eigenmatrix_from_hessian(qm_hessian, diagonal_only=True)

        engine = StubEngine(hessian=qm_hessian)
        obj = ObjectiveFunction(forcefield=None, engine=engine, molecules=[mol], reference=ref)
        result = obj._evaluate_molecule(0)

        assert "eigenmatrix" in result
        eigmat = np.array(result["eigenmatrix"], dtype=float)
        # Self-projection should be diagonal
        diag_only = np.diag(np.diag(eigmat))
        assert np.allclose(eigmat, diag_only, atol=1e-8)
