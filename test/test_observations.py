"""Tests for :mod:`q2mm.models.observations` multi-target builders.

Split out of the old ``test/test_systems.py`` — these tests exercise
:meth:`ObservationSet.from_molecules`/``from_molecule`` directly and have
nothing to do with the benchmark-system registry (they were only
co-located with ``q2mm.systems`` before the phase-2 package split; see
``test/test_benchmarks_systems.py`` for the registry's own tests).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from test._shared import make_water

from q2mm.models.molecule import Molecule
from q2mm.models.observations import Observation, ObservationSet


def _make_water_with_hessian(diag_start: float = 0.01, diag_stop: float = 0.09) -> Molecule:
    mol = make_water()
    return mol.with_hessian(np.diag(np.linspace(diag_start, diag_stop, 3 * mol.n_atoms)))


class TestObservationSetFromMolecules:
    def test_from_molecules_adds_eigenmatrix_and_geometry_but_not_frequencies(self) -> None:
        mol = _make_water_with_hessian()

        ref = ObservationSet.from_molecules([mol], case_ids=["0"])

        counts = Counter(value.kind for value in ref.values)
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36
        assert counts["bond_length"] == len(mol.bonds)
        assert counts["bond_angle"] == len(mol.angles)
        assert counts["frequency"] == 0

    def test_from_molecules_accepts_custom_eigenvalue_weights(self) -> None:
        mol = _make_water_with_hessian(diag_stop=0.2)

        ref = ObservationSet.from_molecules(
            [mol],
            case_ids=["0"],
            weights={
                "eig_i": 0.1,
                "eig_d_low": 0.2,
                "eig_d_high": 0.3,
                "eig_o": 0.4,
            },
        )

        # Custom weights propagate. With rigid-body exclusion, the first mode
        # (skip_first) and the six smallest-|eigenvalue| rigid modes get eig_i
        # (0.1); surviving diagonal modes get eig_d_low/eig_d_high (0.2/0.3).
        diag_weights = [value.weight for value in ref.values if value.kind == "eig_diagonal"]
        assert diag_weights[0] == 0.1
        assert set(diag_weights) <= {0.1, 0.2, 0.3}
        assert 0.1 in diag_weights  # excluded (rigid-body / reaction-coordinate) modes
        assert any(w in (0.2, 0.3) for w in diag_weights)  # at least one real mode kept

        # Off-diagonal couplings use eig_o (0.4), except those touching an
        # excluded mode, which are zero-weighted.
        offdiag_weights = {value.weight for value in ref.values if value.kind == "eig_offdiagonal"}
        assert offdiag_weights <= {0.0, 0.4}
        assert 0.4 in offdiag_weights

    def test_from_molecules_include_geometry_false(self) -> None:
        """include_geometry=False omits bond and angle references."""
        mol = _make_water_with_hessian()
        ref = ObservationSet.from_molecules([mol], case_ids=["0"], include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        # Eigenmatrix data should still be present
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36

    def test_from_molecule_include_geometry_false(self) -> None:
        """Single-molecule factory also respects include_geometry=False."""
        mol = _make_water_with_hessian()
        ref = ObservationSet.from_molecule(mol, include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        assert counts["eig_diagonal"] > 0

    def test_include_geometry_true_is_default(self) -> None:
        """Default behavior includes geometry references."""
        mol = _make_water_with_hessian()
        ref = ObservationSet.from_molecules([mol], case_ids=["0"])

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] > 0
        assert counts["bond_angle"] > 0


# ---------------------------------------------------------------------------
# Deep immutability (domain-review Finding #1)
# ---------------------------------------------------------------------------


class TestObservationIsFrozen:
    """``Observation`` must be a frozen dataclass — no attribute may be reassigned."""

    def test_cannot_assign_value(self) -> None:
        obs = Observation(kind="energy", value=1.0)
        with pytest.raises(FrozenInstanceError):
            obs.value = 2.0  # type: ignore[misc]

    def test_cannot_assign_case_id(self) -> None:
        obs = Observation(kind="energy", value=1.0, case_id="0")
        with pytest.raises(FrozenInstanceError):
            obs.case_id = "1"  # type: ignore[misc]

    def test_cannot_assign_kind(self) -> None:
        obs = Observation(kind="energy", value=1.0)
        with pytest.raises(FrozenInstanceError):
            obs.kind = "frequency"  # type: ignore[misc]

    def test_cannot_assign_weight(self) -> None:
        obs = Observation(kind="energy", value=1.0, weight=1.0)
        with pytest.raises(FrozenInstanceError):
            obs.weight = 2.0  # type: ignore[misc]

    def test_is_hashable(self) -> None:
        """Frozen dataclasses are hashable by default — usable in sets/dict keys."""
        obs = Observation(kind="energy", value=1.0, case_id="0")
        assert hash(obs) == hash(Observation(kind="energy", value=1.0, case_id="0"))


class TestObservationSetIsImmutable:
    """``ObservationSet`` must be deeply immutable: frozen dataclass, tuple-backed."""

    def test_values_is_a_tuple(self) -> None:
        ref = ObservationSet().with_energy(1.0, case_id="0")
        assert isinstance(ref.values, tuple)

    def test_empty_values_is_a_tuple(self) -> None:
        assert isinstance(ObservationSet().values, tuple)

    def test_cannot_assign_values(self) -> None:
        ref = ObservationSet()
        with pytest.raises(FrozenInstanceError):
            ref.values = (Observation(kind="energy", value=1.0),)  # type: ignore[misc]

    def test_with_energy_does_not_mutate_original(self) -> None:
        """``with_*`` methods are pure: the receiver is left untouched."""
        original = ObservationSet()
        updated = original.with_energy(5.0, case_id="0")

        assert original.n_observations == 0
        assert updated.n_observations == 1
        assert original is not updated

    def test_with_frequency_does_not_mutate_original(self) -> None:
        original = ObservationSet().with_energy(1.0, case_id="0")
        updated = original.with_frequency(100.0, data_idx=0, case_id="0")

        assert original.n_observations == 1
        assert updated.n_observations == 2
        assert original.values[0] == updated.values[0]

    def test_with_bond_length_does_not_mutate_original(self) -> None:
        original = ObservationSet()
        updated = original.with_bond_length(0.96, atom_indices=(0, 1), case_id="0")

        assert original.n_observations == 0
        assert updated.n_observations == 1

    def test_chained_with_calls_accumulate_without_mutating_intermediates(self) -> None:
        """Each ``with_*`` call in a chain returns an independent snapshot."""
        step0 = ObservationSet()
        step1 = step0.with_energy(1.0, case_id="0")
        step2 = step1.with_frequency(100.0, data_idx=0, case_id="0")

        assert step0.n_observations == 0
        assert step1.n_observations == 1
        assert step2.n_observations == 2

    def test_no_add_methods_remain(self) -> None:
        """The old in-place ``add_*`` mutators must not exist on the new API."""
        ref = ObservationSet()
        assert not hasattr(ref, "add_energy")
        assert not hasattr(ref, "add_frequency")
        assert not hasattr(ref, "add_bond_length")
        assert not hasattr(ref, "add_bond_angle")
        assert not hasattr(ref, "add_torsion_angle")
        assert not hasattr(ref, "add_hessian_eigenvalue")
        assert not hasattr(ref, "add_hessian_offdiagonal")
        assert not hasattr(ref, "add_hessian_element")
        assert not hasattr(ref, "add_eigenmatrix_from_hessian")
        assert not hasattr(ref, "add_hessian_from_matrix")
        assert not hasattr(ref, "add_frequencies_from_array")
