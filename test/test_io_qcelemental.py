"""QCElemental boundary tests for canonical molecule topology."""

from __future__ import annotations

import numpy as np
import pytest

qcel = pytest.importorskip("qcelemental")

from q2mm.io.qcelemental import molecule_from_qcel, molecule_to_qcel
from q2mm.models.molecule import Bond, Molecule


def test_qcelemental_roundtrip_preserves_explicit_bond_order() -> None:
    source = qcel.models.Molecule(
        symbols=["C", "O"],
        geometry=[0.0, 0.0, 0.0, 2.2, 0.0, 0.0],
        molecular_charge=0,
        molecular_multiplicity=1,
        connectivity=[(0, 1, 2.0)],
        fix_com=True,
        fix_orientation=True,
    )

    molecule = molecule_from_qcel(source, name="carbonyl")
    roundtripped = molecule_from_qcel(molecule_to_qcel(molecule))

    assert molecule.bonds_explicit
    assert molecule.bonds[0].bond_order == "="
    assert float(molecule.bonds[0].source_bond_order) == 2.0
    assert roundtripped.bonds_explicit
    assert roundtripped.bonds[0].bond_order == "="
    np.testing.assert_allclose(roundtripped.geometry, molecule.geometry)


def test_qcelemental_roundtrip_preserves_explicit_empty_topology() -> None:
    molecule = Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        bonds=(),
    )

    roundtripped = molecule_from_qcel(molecule_to_qcel(molecule))

    assert roundtripped.bonds_explicit
    assert roundtripped.bonds == ()


def test_qcelemental_export_preserves_aromatic_order() -> None:
    molecule = Molecule(
        symbols=("C", "C"),
        geometry=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        bonds=(
            Bond(
                atom_i=0,
                atom_j=1,
                elements=("C", "C"),
                length=1.4,
                bond_order="*",
                source_bond_order="1.5",
            ),
        ),
    )

    converted = molecule_to_qcel(molecule)

    assert float(converted.connectivity[0][2]) == 1.5
