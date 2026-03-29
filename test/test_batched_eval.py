"""Tests for batched Hessian evaluation via jax.vmap.

Covers:
- Topology signature and grouping logic
- Batched Hessian parity with sequential evaluation
- Batched frequency parity
- ObjectiveFunction integration (batched vs sequential)
- Graceful fallback for non-JAX engines
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

pytestmark = [pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"), pytest.mark.jax]

from test._shared import make_diatomic, make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


class TestTopologySignature:
    """Test _topology_signature produces correct groupings."""

    def test_same_topology_different_coords(self) -> None:
        """Two conformations of the same molecule produce the same signature."""
        from q2mm.backends.mm.batched import _topology_signature
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _h2_ff()
        mol_a = make_diatomic(distance=0.74)
        mol_b = make_diatomic(distance=0.84)

        handle_a = engine.create_context(mol_a, ff)
        handle_b = engine.create_context(mol_b, ff)

        assert _topology_signature(handle_a) == _topology_signature(handle_b)

    def test_different_topology_different_signature(self) -> None:
        """Different molecules produce different signatures."""
        from q2mm.backends.mm.batched import _topology_signature
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        h2_ff = _h2_ff()
        water_ff = _water_ff()

        handle_h2 = engine.create_context(make_diatomic(), h2_ff)
        handle_water = engine.create_context(make_water(), water_ff)

        assert _topology_signature(handle_h2) != _topology_signature(handle_water)


# ---------------------------------------------------------------------------
# group_by_topology tests
# ---------------------------------------------------------------------------


class TestGroupByTopology:
    """Test molecule grouping logic."""

    def test_same_molecules_grouped(self) -> None:
        """Two conformations of the same molecule land in one group."""
        from q2mm.backends.mm.batched import group_by_topology
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _h2_ff()
        mols = [make_diatomic(distance=0.74), make_diatomic(distance=0.84)]

        groups = group_by_topology(mols, ff, engine)
        assert len(groups) == 1
        assert len(groups[0].mol_indices) == 2
        assert len(groups[0].geometries) == 2

    def test_different_molecules_separate_groups(self) -> None:
        """Molecules with different topologies get separate groups."""
        from q2mm.backends.mm.batched import group_by_topology
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
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

        groups = group_by_topology(mols, ff, engine)
        assert len(groups) == 2

    def test_uses_prebuilt_handles(self) -> None:
        """Pre-built handles are reused instead of creating new ones."""
        from q2mm.backends.mm.batched import group_by_topology
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _h2_ff()
        mols = [make_diatomic(distance=0.74), make_diatomic(distance=0.84)]

        # Pre-build handle for first molecule
        prebuilt = {0: engine.create_context(mols[0], ff)}
        groups = group_by_topology(mols, ff, engine, handles=prebuilt)

        assert len(groups) == 1
        # The handle in the group should be the pre-built one
        assert groups[0].handle is prebuilt[0]


# ---------------------------------------------------------------------------
# Batched Hessian tests
# ---------------------------------------------------------------------------


class TestBatchedHessians:
    """Test batched_hessians produces correct results."""

    def test_single_molecule_matches_standard(self) -> None:
        """Single-molecule batched path matches JaxEngine.hessian()."""
        from q2mm.backends.mm.batched import TopologyGroup, batched_hessians
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _h2_ff()
        mol = make_diatomic(distance=0.80)
        handle = engine.create_context(mol, ff)

        # Standard path
        hess_std = engine.hessian(handle, ff)

        # Batched path (single molecule)
        group = TopologyGroup(
            handle=handle,
            mol_indices=[0],
            geometries=[np.asarray(mol.geometry, dtype=np.float64)],
        )
        hess_batch = batched_hessians(group, ff)

        assert len(hess_batch) == 1
        np.testing.assert_allclose(hess_batch[0], hess_std, rtol=1e-10)

    def test_multi_molecule_matches_sequential(self) -> None:
        """Multi-molecule vmap path matches sequential Hessians."""
        from q2mm.backends.mm.batched import TopologyGroup, batched_hessians
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _h2_ff()
        distances = [0.70, 0.74, 0.80, 0.90]
        mols = [make_diatomic(distance=d) for d in distances]

        # Build shared handle from first molecule
        handle = engine.create_context(mols[0], ff)

        # Sequential Hessians
        sequential = []
        for mol in mols:
            h = engine.create_context(mol, ff)
            sequential.append(engine.hessian(h, ff))

        # Batched Hessians
        group = TopologyGroup(
            handle=handle,
            mol_indices=list(range(len(mols))),
            geometries=[np.asarray(m.geometry, dtype=np.float64) for m in mols],
        )
        batched = batched_hessians(group, ff)

        assert len(batched) == len(sequential)
        for b, s in zip(batched, sequential):
            np.testing.assert_allclose(b, s, rtol=1e-10)

    def test_water_multi_conformation(self) -> None:
        """Batched Hessians work for water at different angles."""
        from q2mm.backends.mm.batched import TopologyGroup, batched_hessians
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _water_ff()
        angles = [100.0, 104.5, 110.0]
        mols = [make_water(angle_deg=a) for a in angles]

        handle = engine.create_context(mols[0], ff)

        # Sequential
        sequential = []
        for mol in mols:
            h = engine.create_context(mol, ff)
            sequential.append(engine.hessian(h, ff))

        # Batched
        group = TopologyGroup(
            handle=handle,
            mol_indices=list(range(len(mols))),
            geometries=[np.asarray(m.geometry, dtype=np.float64) for m in mols],
        )
        batched = batched_hessians(group, ff)

        for b, s in zip(batched, sequential):
            np.testing.assert_allclose(b, s, rtol=1e-10, atol=1e-15)


# ---------------------------------------------------------------------------
# Batched frequencies tests
# ---------------------------------------------------------------------------


class TestBatchedFrequencies:
    """Test batched_frequencies produces correct results."""

    def test_frequencies_match_sequential(self) -> None:
        """Batched frequencies match sequential engine.frequencies()."""
        from q2mm.backends.mm.batched import TopologyGroup, batched_frequencies
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        ff = _water_ff()
        angles = [100.0, 104.5, 110.0]
        mols = [make_water(angle_deg=a) for a in angles]

        handle = engine.create_context(mols[0], ff)

        # Sequential
        sequential = []
        for mol in mols:
            h = engine.create_context(mol, ff)
            sequential.append(engine.frequencies(h, ff))

        # Batched
        group = TopologyGroup(
            handle=handle,
            mol_indices=list(range(len(mols))),
            geometries=[np.asarray(m.geometry, dtype=np.float64) for m in mols],
        )
        symbols = [list(m.symbols) for m in mols]
        batched = batched_frequencies(group, ff, symbols)

        assert len(batched) == len(sequential)
        for b_freqs, s_freqs in zip(batched, sequential):
            np.testing.assert_allclose(b_freqs, s_freqs, rtol=1e-6, atol=1e-4)


# ---------------------------------------------------------------------------
# ObjectiveFunction integration tests
# ---------------------------------------------------------------------------


class TestObjectiveFunctionIntegration:
    """Test batched path through ObjectiveFunction."""

    def test_can_batch_hessians_true(self) -> None:
        """_can_batch_hessians returns True for JaxEngine with freq refs."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ReferenceData()
        ref.add_frequency(1000.0, data_idx=0, molecule_idx=0)
        ref.add_frequency(1000.0, data_idx=0, molecule_idx=1)

        obj = ObjectiveFunction(ff, engine, mols, ref)
        assert obj._can_batch_hessians() is True

    def test_can_batch_hessians_false_single_mol(self) -> None:
        """_can_batch_hessians returns False with only one molecule."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        ff = _water_ff()
        mols = [make_water()]

        ref = ReferenceData()
        ref.add_frequency(1000.0, data_idx=0, molecule_idx=0)

        obj = ObjectiveFunction(ff, engine, mols, ref)
        assert obj._can_batch_hessians() is False

    def test_can_batch_hessians_false_energy_only(self) -> None:
        """_can_batch_hessians returns False for energy-only refs."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ReferenceData()
        ref.add_energy(1.0, molecule_idx=0)
        ref.add_energy(2.0, molecule_idx=1)

        obj = ObjectiveFunction(ff, engine, mols, ref)
        assert obj._can_batch_hessians() is False

    def test_batched_vs_sequential_parity(self) -> None:
        """Objective score is identical whether batched or sequential."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        engine = JaxEngine()
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        # Compute reference frequencies using the engine directly
        ref_freqs = []
        for mol in mols:
            h = engine.create_context(mol, ff)
            freqs = engine.frequencies(h, ff)
            ref_freqs.append(freqs)

        ref = ReferenceData()
        for mol_idx, freqs in enumerate(ref_freqs):
            for i, f in enumerate(freqs):
                # Add slightly perturbed frequencies so residuals aren't zero
                ref.add_frequency(f * 1.05, data_idx=i, molecule_idx=mol_idx)

        # Compute score with batching enabled (the default for 2+ mols)
        obj_batched = ObjectiveFunction(ff, engine, mols, ref)
        params = ff.get_param_vector()
        score_batched = obj_batched(params)

        # Compute score with batching forcibly disabled by using single-mol
        # calls (force sequential by evaluating each molecule separately)
        score_sequential = 0.0
        for mol_idx, mol in enumerate(mols):
            ref_single = ReferenceData()
            for i, f in enumerate(ref_freqs[mol_idx]):
                ref_single.add_frequency(f * 1.05, data_idx=i, molecule_idx=0)
            obj_single = ObjectiveFunction(ff, engine, [mol], ref_single)
            score_sequential += obj_single(params)

        assert score_batched == pytest.approx(score_sequential, rel=1e-10)


# ---------------------------------------------------------------------------
# Fallback tests
# ---------------------------------------------------------------------------


class TestFallback:
    """Test graceful fallback for non-JAX engines."""

    def test_can_batch_false_for_non_jax(self) -> None:
        """_can_batch_hessians returns False for a non-JAX engine."""
        from unittest.mock import MagicMock

        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData

        mock_engine = MagicMock()
        mock_engine.supports_runtime_params.return_value = False
        mock_engine.supports_batched_energy.return_value = False
        ff = _water_ff()
        mols = [make_water(angle_deg=100.0), make_water(angle_deg=110.0)]

        ref = ReferenceData()
        ref.add_frequency(1000.0, data_idx=0, molecule_idx=0)
        ref.add_frequency(1000.0, data_idx=0, molecule_idx=1)

        obj = ObjectiveFunction(ff, mock_engine, mols, ref)
        assert obj._can_batch_hessians() is False
