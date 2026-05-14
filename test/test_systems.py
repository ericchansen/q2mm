"""Tests for benchmark system reference-data helpers."""

from __future__ import annotations

from collections import Counter

import numpy as np

from test._shared import make_water

from q2mm.diagnostics import systems
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceData


def _make_water_with_hessian(diag_start: float = 0.01, diag_stop: float = 0.09) -> Q2MMMolecule:
    mol = make_water()
    mol.hessian = np.diag(np.linspace(diag_start, diag_stop, 3 * mol.n_atoms))
    return mol


class TestSystemReferenceConstruction:
    def test_from_molecules_adds_eigenmatrix_and_geometry_but_not_frequencies(self) -> None:
        mol = _make_water_with_hessian()
        qm_freqs = systems._qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        ref = ReferenceData.from_molecules([mol])

        counts = Counter(value.kind for value in ref.values)
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36
        assert counts["bond_length"] == len(mol.bonds)
        assert counts["bond_angle"] == len(mol.angles)
        assert counts["frequency"] == 0
        assert sum(freq > 50.0 for freq in qm_freqs) > 0

    def test_from_molecules_accepts_custom_eigenvalue_weights(self) -> None:
        mol = _make_water_with_hessian(diag_stop=0.2)

        ref = ReferenceData.from_molecules(
            [mol],
            weights={
                "eig_i": 0.1,
                "eig_d_low": 0.2,
                "eig_d_high": 0.3,
                "eig_o": 0.4,
            },
        )

        diag_weights = [value.weight for value in ref.values if value.kind == "eig_diagonal"]
        assert diag_weights[0] == 0.1
        assert 0.2 in diag_weights
        assert 0.3 in diag_weights
        assert {value.weight for value in ref.values if value.kind == "eig_offdiagonal"} == {0.4}

    def test_from_molecules_include_geometry_false(self) -> None:
        """include_geometry=False omits bond and angle references."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecules([mol], include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        # Eigenmatrix data should still be present
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36

    def test_from_molecule_include_geometry_false(self) -> None:
        """Single-molecule factory also respects include_geometry=False."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecule(mol, include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        assert counts["eig_diagonal"] > 0

    def test_include_geometry_true_is_default(self) -> None:
        """Default behavior includes geometry references."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecules([mol])

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] > 0
        assert counts["bond_angle"] > 0
