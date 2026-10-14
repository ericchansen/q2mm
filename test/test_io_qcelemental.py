"""QCElemental boundary tests for canonical molecule topology."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("qcelemental")
from qcelemental.models.v2 import Molecule as QCSchemaMolecule

from q2mm.io.qcelemental import molecule_from_qcel, molecule_to_qcel
from q2mm.models.molecule import Bond, Molecule


@pytest.mark.parametrize(
    ("source_order", "canonical_order"),
    [(1.0, "-"), (1.5, "*"), (2.0, "="), (3.0, "%"), (1.25, "")],
)
def test_qcelemental_roundtrip_preserves_explicit_bond_order(source_order: float, canonical_order: str) -> None:
    source = QCSchemaMolecule(
        symbols=["C", "O"],
        geometry=[1.0, 2.0, 3.0, 8.0, 2.0, 3.0],
        molecular_charge=0,
        molecular_multiplicity=1,
        connectivity=[(0, 1, source_order)],
        fix_com=True,
        fix_orientation=True,
    )

    molecule = molecule_from_qcel(source, name="carbonyl")
    converted = molecule_to_qcel(molecule)
    roundtripped = molecule_from_qcel(converted)

    assert molecule.name == "carbonyl"
    assert molecule.symbols == ("C", "O")
    assert molecule.bonds_explicit
    assert len(molecule.bonds) == 1
    assert molecule.bonds[0].bond_order == canonical_order
    assert float(molecule.bonds[0].source_bond_order) == source_order
    assert molecule.bonds[0].length == pytest.approx(7.0 * 0.52917721067)
    assert converted.connectivity == [(0, 1, source_order)]
    assert roundtripped.bonds_explicit
    assert roundtripped.bonds[0].bond_order == canonical_order
    assert float(roundtripped.bonds[0].source_bond_order) == source_order
    np.testing.assert_allclose(roundtripped.geometry, molecule.geometry, rtol=0, atol=5.3e-9)


@pytest.mark.parametrize(("charge", "multiplicity"), [(0, 1), (1, 2), (-1, 2)])
def test_qcelemental_export_preserves_units_frame_and_electronic_state(charge: int, multiplicity: int) -> None:
    molecule = Molecule(
        symbols=("O", "H", "H"),
        geometry=np.array([[2.123456789123, -3.2, 4.1], [2.8, -2.6, 4.5], [1.6, -3.5, 4.8]]),
        charge=charge,
        multiplicity=multiplicity,
    )

    converted = molecule_to_qcel(molecule)
    roundtripped = molecule_from_qcel(converted)

    assert isinstance(converted, QCSchemaMolecule)
    assert tuple(converted.symbols) == molecule.symbols
    assert converted.fix_com
    assert converted.fix_orientation
    assert converted.molecular_charge == roundtripped.charge == charge
    assert converted.molecular_multiplicity == roundtripped.multiplicity == multiplicity
    # QCSchema rounds geometry to eight decimal places in Bohr.
    np.testing.assert_allclose(converted.geometry, molecule.geometry / 0.52917721067, rtol=0, atol=5e-9)
    np.testing.assert_allclose(roundtripped.geometry, molecule.geometry, rtol=0, atol=5.3e-9)


@pytest.mark.parametrize("orient", [False, True])
def test_qcelemental_import_preserves_qcschema_frame_and_bohr_units(orient: bool) -> None:
    source = QCSchemaMolecule(
        symbols=["O", "H", "H"],
        geometry=[[2.1, -3.2, 4.1], [3.4, -2.0, 4.9], [1.1, -3.8, 5.4]],
        molecular_charge=1,
        molecular_multiplicity=2,
        fix_com=not orient,
        fix_orientation=not orient,
        orient=orient,
    )

    molecule = molecule_from_qcel(source)
    converted = molecule_to_qcel(molecule)

    np.testing.assert_allclose(molecule.geometry, source.geometry * 0.52917721067, rtol=0, atol=1e-14)
    np.testing.assert_allclose(converted.geometry, source.geometry, rtol=0, atol=5e-9)
    assert molecule.charge == 1
    assert molecule.multiplicity == 2
    if orient:
        np.testing.assert_allclose(np.average(source.geometry, axis=0, weights=source.masses), 0, atol=1e-8)


def test_qcelemental_roundtrip_keeps_inferred_connectivity_unspecified() -> None:
    source = QCSchemaMolecule(symbols=["H", "H"], geometry=[0.0, 0.0, 0.0, 1.4, 0.0, 0.0])

    molecule = molecule_from_qcel(source)
    converted = molecule_to_qcel(molecule)
    roundtripped = molecule_from_qcel(converted)

    assert not molecule.bonds_explicit
    assert len(molecule.bonds) == 1
    assert converted.connectivity is None
    assert not roundtripped.bonds_explicit
    assert len(roundtripped.bonds) == 1


def test_qcelemental_roundtrip_preserves_explicit_empty_topology() -> None:
    molecule = Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        bonds=(),
    )

    converted = molecule_to_qcel(molecule)
    serialized = QCSchemaMolecule.model_validate_json(converted.model_dump_json())
    roundtripped = molecule_from_qcel(serialized)

    assert converted.connectivity is None
    assert roundtripped.bonds_explicit
    assert roundtripped.bonds == ()
    assert roundtripped.angles == ()
    assert roundtripped.torsions == ()


@pytest.mark.parametrize(
    ("canonical_order", "source_order", "expected_order"),
    [("*", "1.5", 1.5), ("*", None, 1.5), ("*", "ar", 1.5), ("=", "1.25", 1.25)],
)
def test_qcelemental_export_preserves_source_or_canonical_order(
    canonical_order: str, source_order: str | None, expected_order: float
) -> None:
    molecule = Molecule(
        symbols=("C", "C"),
        geometry=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        bonds=(
            Bond(
                atom_i=0,
                atom_j=1,
                elements=("C", "C"),
                length=1.4,
                bond_order=canonical_order,
                source_bond_order=source_order,
            ),
        ),
    )

    converted = molecule_to_qcel(molecule)

    roundtripped = molecule_from_qcel(converted)

    assert float(converted.connectivity[0][2]) == expected_order
    assert float(roundtripped.bonds[0].source_bond_order) == expected_order
