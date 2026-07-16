"""Tests for batched Hessian evaluation via jax.vmap.

Covers:
- Topology signature and grouping logic
- Batched Hessian parity with sequential evaluation
- Batched frequency parity
- Objective executor integration (multi-case vs sequential)
- Graceful fallback for non-JAX backends
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    FrequencyRequest,
    HessianRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import mock_backend_info, param_vector, prepare_case

import importlib.util

import numpy as np
import pytest


_HAS_JAX = importlib.util.find_spec("jax") is not None

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import make_diatomic, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _make_objective(
    forcefield: ForceField, backend: object, molecules: list, reference: object, **kwargs: object
) -> PythonObjectiveExecutor:
    layout = _layout(forcefield)
    case_ids = tuple(str(i) for i in range(len(molecules)))
    plan = ObjectivePlan(
        case_ids=case_ids,
        molecules=tuple(molecules),
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in molecules),
        observations=reference,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, forcefield),
        **kwargs,
    )
    return PythonObjectiveExecutor(plan, backend, forcefield)


def _h2_ff() -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74)],
        functional_form=FunctionalForm.MM3,
    )


def _water_ff() -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
        functional_form=FunctionalForm.MM3,
    )


# ---------------------------------------------------------------------------
# Topology signature tests
# ---------------------------------------------------------------------------


class TestTopologyGrouping:
    """Test topology-compatibility grouping into typed batches."""

    def test_same_topology_grouped_into_one_batch(self) -> None:
        """Two conformations of the same molecule land in one batch."""
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _h2_ff()
        s_a = prepare_case(backend, make_diatomic(distance=0.74), ff, "a")
        s_b = prepare_case(backend, make_diatomic(distance=0.84), ff, "b")

        batches = group_by_topology([s_a, s_b])
        assert len(batches) == 1
        assert batches[0].case_ids == ("a", "b")

    def test_different_topology_separate_batches(self) -> None:
        """Different molecules land in different batches."""
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        s_h2 = prepare_case(backend, make_diatomic(), _h2_ff(), "h2")
        s_water = prepare_case(backend, make_water(), _water_ff(), "water")

        batches = group_by_topology([s_h2, s_water])
        assert len(batches) == 2
        assert {cid for b in batches for cid in b.case_ids} == {"h2", "water"}


# ---------------------------------------------------------------------------
# group_by_topology tests
# ---------------------------------------------------------------------------


class TestGroupByTopology:
    """Test prepared-session grouping logic."""

    def test_same_molecules_grouped(self) -> None:
        """Two conformations of the same molecule land in one batch."""
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _h2_ff()
        mols = [make_diatomic(distance=0.74), make_diatomic(distance=0.84)]
        sessions = [prepare_case(backend, m, ff, str(i)) for i, m in enumerate(mols)]

        batches = group_by_topology(sessions)
        assert len(batches) == 1
        assert batches[0].case_ids == ("0", "1")

    def test_different_molecules_separate_groups(self) -> None:
        """Molecules with different topologies get separate batches."""
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        # Use a combined FF that supports both molecules
        ff = ForceField(
            bonds=[
                BondParam(elements=("H", "H"), force_constant=5.0, equilibrium=0.74),
                BondParam(elements=("H", "O"), force_constant=8.0, equilibrium=0.96),
            ],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.7, equilibrium=104.5)],
            functional_form=FunctionalForm.MM3,
        )
        mols = [make_diatomic(distance=0.74), make_water()]
        sessions = [prepare_case(backend, m, ff, str(i)) for i, m in enumerate(mols)]

        batches = group_by_topology(sessions)
        assert len(batches) == 2

    def test_batch_preserves_case_ids_and_isolation(self) -> None:
        """The batch tracks each session's stable case ID; cases stay isolated.

        Isolation is proven by distinct case IDs plus different-coordinate
        numerical parity: the batched Hessian of each case equals that case's
        own independent Hessian, so no case's state leaks into another's.
        """
        from q2mm.backends.contracts import BatchedHessianRequest, HessianRequest
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _h2_ff()
        # Two conformers with clearly different coordinates.
        s0 = prepare_case(backend, make_diatomic(distance=0.72), ff, "case-0")
        s1 = prepare_case(backend, make_diatomic(distance=0.95), ff, "case-1")
        vec = param_vector(ff)

        batches = group_by_topology([s0, s1])
        assert len(batches) == 1
        batch = batches[0]
        assert batch.case_ids == ("case-0", "case-1")

        result = batch.hessians(BatchedHessianRequest(parameters=vec))
        assert result.case_ids == ("case-0", "case-1")
        # Each batched row equals the corresponding session's own Hessian at its
        # own coordinates (distinct coords -> distinct Hessians, no cross-leak).
        h0 = s0.hessian(HessianRequest(parameters=vec)).hessian
        h1 = s1.hessian(HessianRequest(parameters=vec)).hessian
        np.testing.assert_allclose(result.hessians[0], h0, rtol=1e-10)
        np.testing.assert_allclose(result.hessians[1], h1, rtol=1e-10)
        assert not np.allclose(result.hessians[0], result.hessians[1])


class TestIncompatibleBatch:
    """PreparedJaxBatch construction rejects incompatible/ill-formed inputs."""

    def test_duplicate_case_ids_rejected(self) -> None:
        from q2mm.backends.contracts import EvaluationError
        from q2mm.backends.mm.batched import PreparedJaxBatch

        backend = load_backend("jax")
        ff = _h2_ff()
        s0 = prepare_case(backend, make_diatomic(distance=0.74), ff, "dup")
        s1 = prepare_case(backend, make_diatomic(distance=0.84), ff, "dup")
        with pytest.raises(EvaluationError):
            PreparedJaxBatch([s0, s1])

    def test_topology_mismatch_rejected(self) -> None:
        from q2mm.backends.contracts import EvaluationError
        from q2mm.backends.mm.batched import PreparedJaxBatch

        backend = load_backend("jax")
        s_h2 = prepare_case(backend, make_diatomic(), _h2_ff(), "h2")
        s_water = prepare_case(backend, make_water(), _water_ff(), "water")
        # Different atom count / topology signature -> rejected.
        with pytest.raises(EvaluationError):
            PreparedJaxBatch([s_h2, s_water])

    def test_non_prepared_jax_rejected(self) -> None:
        from q2mm.backends.contracts import EvaluationError
        from q2mm.backends.mm.batched import PreparedJaxBatch

        with pytest.raises(EvaluationError):
            PreparedJaxBatch([object()])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Batched Hessian tests
# ---------------------------------------------------------------------------


class TestBatchedHessians:
    """Test PreparedJaxBatch.hessians produces correct results."""

    def test_single_molecule_matches_standard(self) -> None:
        """Single-case batch matches the prepared-session Hessian."""
        from q2mm.backends.contracts import BatchedHessianRequest, HessianRequest, HessianUnit
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _h2_ff()
        mol = make_diatomic(distance=0.80)
        session = prepare_case(backend, mol, ff, "only")
        vec = param_vector(ff)

        # Standard path (same prepared session, reused)
        hess_std = session.hessian(HessianRequest(parameters=vec)).hessian

        # Batched path (single case)
        batches = group_by_topology([session])
        assert len(batches) == 1
        result = batches[0].hessians(BatchedHessianRequest(parameters=vec))

        assert result.unit is HessianUnit.HARTREE_PER_BOHR2
        assert result.case_ids == ("only",)
        assert result.hessians.shape[0] == 1
        np.testing.assert_allclose(result.hessians[0], hess_std, rtol=1e-10)

    def test_multi_molecule_matches_sequential(self) -> None:
        """Multi-case vmap path matches independent per-session Hessians."""
        from q2mm.backends.contracts import BatchedHessianRequest, HessianRequest
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _h2_ff()
        distances = [0.70, 0.74, 0.80, 0.90]
        sessions = [prepare_case(backend, make_diatomic(distance=d), ff, str(i)) for i, d in enumerate(distances)]
        vec = param_vector(ff)

        # Independent per-session Hessians
        sequential = [s.hessian(HessianRequest(parameters=vec)).hessian for s in sessions]

        # Batched Hessians (one topology batch)
        batches = group_by_topology(sessions)
        assert len(batches) == 1
        result = batches[0].hessians(BatchedHessianRequest(parameters=vec))

        assert result.hessians.shape[0] == len(sequential)
        for i, s in enumerate(sequential):
            np.testing.assert_allclose(result.hessians[i], s, rtol=1e-10)

    def test_water_multi_conformation(self) -> None:
        """Batched Hessians work for water at different angles."""
        from q2mm.backends.contracts import BatchedHessianRequest, HessianRequest
        from q2mm.backends.mm.batched import group_by_topology

        backend = load_backend("jax")
        ff = _water_ff()
        angles = [100.0, 104.5, 110.0]
        sessions = [prepare_case(backend, make_water(angle_deg=a), ff, str(i)) for i, a in enumerate(angles)]
        vec = param_vector(ff)

        sequential = [s.hessian(HessianRequest(parameters=vec)).hessian for s in sessions]

        batches = group_by_topology(sessions)
        assert len(batches) == 1
        result = batches[0].hessians(BatchedHessianRequest(parameters=vec))

        for i, s in enumerate(sequential):
            np.testing.assert_allclose(result.hessians[i], s, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# Batched frequencies (derived from batched Hessians)
# ---------------------------------------------------------------------------


class TestBatchedFrequencies:
    """Frequencies derived from batched Hessians match per-session frequencies."""

    def test_frequencies_match_sequential(self) -> None:
        """Batched-Hessian frequencies match independent per-session frequencies."""
        from q2mm.backends.contracts import BatchedHessianRequest, FrequencyRequest
        from q2mm.backends.mm.batched import group_by_topology
        from q2mm.models.hessian import hessian_to_frequencies

        backend = load_backend("jax")
        ff = _water_ff()
        angles = [100.0, 104.5, 110.0]
        mols = [make_water(angle_deg=a) for a in angles]
        sessions = [prepare_case(backend, m, ff, str(i)) for i, m in enumerate(mols)]
        vec = param_vector(ff)

        # Independent per-session frequencies
        sequential = [
            [float(_f) for _f in s.frequencies(FrequencyRequest(parameters=vec)).frequencies] for s in sessions
        ]

        batches = group_by_topology(sessions)
        assert len(batches) == 1
        result = batches[0].hessians(BatchedHessianRequest(parameters=vec))
        batched = [hessian_to_frequencies(result.hessians[i], list(m.symbols)) for i, m in enumerate(mols)]

        assert len(batched) == len(sequential)
        for b_freqs, s_freqs in zip(batched, sequential):
            np.testing.assert_allclose(b_freqs, s_freqs, rtol=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# Objective executor integration tests
# ---------------------------------------------------------------------------


class TestObjectiveExecutorIntegration:
    """Test multi-case objective evaluation parity."""

    def test_multi_case_frequency_objective_evaluates(self) -> None:
        """Multi-case frequency references evaluate through the executor."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ObservationSet()
        ref = ref.with_frequency(1000.0, data_idx=0, case_id="0")
        ref = ref.with_frequency(1000.0, data_idx=0, case_id="1")

        obj = _make_objective(ff, backend, mols, ref)
        score = obj.value(_params(ff))
        assert np.isfinite(score)
        assert obj.plan.case_ids == ("0", "1")

    def test_single_case_frequency_objective_evaluates(self) -> None:
        """Single-case frequency references evaluate through the same executor."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        ff = _water_ff()
        mols = [make_water()]

        ref = ObservationSet()
        ref = ref.with_frequency(1000.0, data_idx=0, case_id="0")

        obj = _make_objective(ff, backend, mols, ref)
        score = obj.value(_params(ff))
        assert np.isfinite(score)
        assert obj.plan.case_ids == ("0",)

    def test_energy_only_multi_case_objective_evaluates(self) -> None:
        """Energy-only multi-case references evaluate without Hessian work."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ObservationSet()
        ref = ref.with_energy(1.0, case_id="0")
        ref = ref.with_energy(2.0, case_id="1")

        obj = _make_objective(ff, backend, mols, ref)
        score = obj.value(_params(ff))
        assert np.isfinite(score)

    def test_batched_vs_sequential_parity(self) -> None:
        """Objective score is identical whether batched or sequential."""
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        # Compute reference frequencies using the backend directly
        ref_freqs = []
        for mol in mols:
            freqs = [
                float(_f)
                for _f in prepare_case(backend, mol, ff)
                .frequencies(FrequencyRequest(parameters=param_vector(ff)))
                .frequencies
            ]
            ref_freqs.append(freqs)

        ref = ObservationSet()
        for mol_idx, freqs in enumerate(ref_freqs):
            for i, f in enumerate(freqs):
                # Add slightly perturbed frequencies so residuals aren't zero
                ref = ref.with_frequency(f * 1.05, data_idx=i, case_id=str(mol_idx))

        # Compute score with batching enabled (the default for 2+ mols)
        obj_batched = _make_objective(ff, backend, mols, ref)
        params = _params(ff)
        score_batched = obj_batched.value(params)

        # Compute score with batching forcibly disabled by using single-mol
        # calls (force sequential by evaluating each molecule separately)
        score_sequential = 0.0
        for mol_idx, mol in enumerate(mols):
            ref_single = ObservationSet()
            for i, f in enumerate(ref_freqs[mol_idx]):
                ref_single = ref_single.with_frequency(f * 1.05, data_idx=i, case_id="0")
            obj_single = _make_objective(ff, backend, [mol], ref_single)
            score_sequential += obj_single.value(params)

        assert score_batched == pytest.approx(score_sequential, rel=1e-10)

    def test_batched_vs_sequential_eigenmatrix_parity(self) -> None:
        """Eigenmatrix score matches whether batched or sequential.

        Regression guard for the batched ``precomputed_hessian`` eigenmatrix
        path: it must use the same mass-weighted normal-mode basis as the
        per-molecule ``EigenmatrixEvaluator`` (sequential) path.
        """
        from q2mm.models.observations import ObservationSet

        backend = load_backend("jax")
        ff_ref = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        # Use each molecule's MM Hessian at ff_ref as its 'QM' Hessian.
        mols = [
            mol.with_hessian(
                prepare_case(backend, mol, ff_ref).hessian(HessianRequest(parameters=param_vector(ff_ref))).hessian
            )
            for mol in mols
        ]

        # Perturbed FF so the eigenmatrix residuals are non-zero.
        ff = ForceField(
            bonds=[BondParam(elements=("H", "O"), force_constant=6.0, equilibrium=1.02)],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=0.5, equilibrium=110.0)],
            functional_form=FunctionalForm.MM3,
        )
        params = _params(ff)

        ref = ObservationSet()
        for mol_idx, mol in enumerate(mols):
            ref = ref.with_eigenmatrix_from_hessian(
                mol.hessian,
                symbols=list(mol.symbols),
                diagonal_only=False,
                case_id=str(mol_idx),
            )
        obj_batched = _make_objective(ff, backend, mols, ref)
        score_batched = obj_batched.value(params)

        score_sequential = 0.0
        for mol in mols:
            ref_single = ObservationSet()
            ref_single = ref_single.with_eigenmatrix_from_hessian(
                mol.hessian, symbols=list(mol.symbols), diagonal_only=False, case_id="0"
            )
            obj_single = _make_objective(ff, backend, [mol], ref_single)
            score_sequential += obj_single.value(params)

        assert score_batched > 0.0
        assert score_batched == pytest.approx(score_sequential, rel=1e-8)


class TestFallback:
    """Test graceful fallback for non-JAX backends."""

    def test_empty_reference_non_jax_does_not_prepare_backend(self) -> None:
        """No-reference objectives do not require JAX-specific batch support."""
        from unittest.mock import MagicMock

        from q2mm.models.observations import ObservationSet

        mock_engine = MagicMock()
        mock_engine.info = mock_backend_info(batched=False)
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ObservationSet()

        obj = _make_objective(ff, mock_engine, mols, ref)
        assert obj.value(_params(ff)) == pytest.approx(0.0)
        mock_engine.prepare.assert_not_called()
