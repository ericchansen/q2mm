"""Reference-YAML representability, without adding formats or scientific policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q2mm.application.models import molecule_fingerprint_payload
from q2mm.io.reference import ReferenceYAMLError, _parse_datum, load_reference_yaml, save_reference_yaml
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import (
    Observation,
    ObservationEnergyUnit,
    ObservationSet,
    ScanCoordinate,
    ScanCoordinateKind,
    ThermodynamicQuantity,
    observation_payload,
)
from q2mm.models.parameters import ParameterLayout, ParameterUnit

_RICH_KINDS = (
    "atomic_partial_charge",
    "direct_electrostatic_potential",
    "relative_energy",
    "scan_energy",
    "parameter_tether",
)


def _molecule(name: str, partial_charges: tuple[float | None, ...] | None = None) -> Molecule:
    return Molecule(
        symbols=("H", "C", "C", "H"),
        geometry=((0, 1, 0), (0, 0, 0), (1.5, 0, 0), (1.5, 0.5, np.sqrt(0.75))),
        atom_types=("H1", "C2", "C3", "H4"),
        name=name,
        charge=1,
        multiplicity=2,
        bond_tolerance=1.25,
        partial_charges=partial_charges,
    )


def _rich_references(kind: str) -> ObservationSet:
    ref = ObservationSet().with_energy(2.0, case_id="a", label="supported first")
    if kind == "atomic_partial_charge":
        return ref.with_atomic_partial_charge(0.25, atom_index=1, case_id="b", label="charge")
    if kind == "direct_electrostatic_potential":
        return ref.with_direct_electrostatic_potential(0.25, point=(1.0, 2.0, 3.0), case_id="b", label="grid")
    if kind == "relative_energy":
        return ref.with_relative_energy_group(
            [("a", 0.0), ("b", 2.0)],
            group_id="relative-group",
            reference_case_id="a",
            unit=ObservationEnergyUnit.HARTREE,
            quantity=ThermodynamicQuantity.ENTHALPY,
            weight=0.25,
            label="relative",
        )
    if kind == "scan_energy":
        return ref.with_scan_energy_group(
            [
                ("a", 0.0, ScanCoordinate(ScanCoordinateKind.DISTANCE, (0, 1), 1.0, "angstrom")),
                ("b", 1.0, ScanCoordinate(ScanCoordinateKind.DISTANCE, (0, 1), 1.1, "angstrom")),
            ],
            group_id="scan-group",
            reference_case_id="a",
            unit=ObservationEnergyUnit.KCAL_PER_MOL,
            weight=3.0,
            label="scan",
        )
    if kind == "parameter_tether":
        layout = ParameterLayout.from_force_field(
            ForceField(functional_form=FunctionalForm.HARMONIC, bonds=(BondParam(("C", "C"), 1.5, 10.0),))
        )
        return ref.with_parameter_tether(
            10.0,
            parameter_id=layout.ids[0],
            unit=ParameterUnit.KCAL_PER_MOL_PER_ANGSTROM2,
            case_id="b",
            label="tether",
        )
    raise AssertionError(f"Unknown fixture kind {kind}")


def _destination(tmp_path: Path, existing: bool) -> tuple[Path, dict[str, bytes]]:
    path = tmp_path / "reference.yaml"
    if existing:
        path.write_bytes(b"preserve destination bytes")
    return path, {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def _assert_unchanged(tmp_path: Path, before: dict[str, bytes]) -> None:
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


@pytest.mark.parametrize("kind", _RICH_KINDS)
@pytest.mark.parametrize("existing", [False, True])
def test_rich_and_grouped_observations_fail_before_output(tmp_path: Path, kind: str, existing: bool) -> None:
    path, before = _destination(tmp_path, existing)
    with pytest.raises(ReferenceYAMLError, match=kind) as raised:
        save_reference_yaml(path, _rich_references(kind), [_molecule("a"), _molecule("b")])
    assert "case" in str(raised.value)
    _assert_unchanged(tmp_path, before)


@pytest.mark.parametrize(
    "charges",
    [(0.1, 0.0, 0.0, -0.1), (0.0, 0.0, 0.0, 0.0), (None, 0.25, None, -0.25), (None, None, None, None)],
    ids=["nonzero", "zero", "partial", "explicit-none-values"],
)
@pytest.mark.parametrize("existing", [False, True])
def test_partial_charges_fail_before_output(tmp_path: Path, charges: tuple[float | None, ...], existing: bool) -> None:
    path, before = _destination(tmp_path, existing)
    with pytest.raises(ReferenceYAMLError, match="partial_charges") as raised:
        save_reference_yaml(path, ObservationSet(), [_molecule("a"), _molecule("b", charges)])
    assert "'b'" in str(raised.value)
    _assert_unchanged(tmp_path, before)


_LOSSY_SCALARS = [
    Observation("energy", 1.0, case_id="b", data_idx=3),
    Observation("energy", 1.0, case_id="b", atom_indices=(0, 1)),
    Observation("frequency", 1.0, case_id="b", atom_indices=()),
    Observation("frequency", 1.0, case_id="b", atom_indices=(0, 1)),
    Observation("eig_diagonal", 1.0, case_id="b", atom_indices=(0, 1)),
    Observation("bond_length", 1.0, case_id="b", atom_indices=(0, 1), data_idx=3),
    Observation("bond_angle", 1.0, case_id="b", atom_indices=(0, 1, 2), data_idx=3),
    Observation("torsion_angle", 1.0, case_id="b", atom_indices=(0, 1, 2, 3), data_idx=3),
    Observation("eig_offdiagonal", 1.0, case_id="b", atom_indices=(0, 1), data_idx=3),
    Observation("hessian_element", 1.0, case_id="b", atom_indices=(0, 1), data_idx=3),
    Observation("bond_length", 1.0, case_id="b", atom_indices=(0, 1, 2)),
    Observation("bond_angle", 1.0, case_id="b", atom_indices=(0, 1)),
    Observation("torsion_angle", 1.0, case_id="b", atom_indices=(0, 1, 2)),
    *[
        Observation(kind, 1.0, case_id="b", atom_indices=indices)
        for kind in ("eig_offdiagonal", "hessian_element")
        for indices in (None, (), (0,), (0, 1, 2))
    ],
]


@pytest.mark.parametrize("observation", _LOSSY_SCALARS)
@pytest.mark.parametrize("existing", [False, True])
def test_unrepresentable_scalar_index_metadata_fails_before_output(
    tmp_path: Path, observation: Observation, existing: bool
) -> None:
    path, before = _destination(tmp_path, existing)
    ref = ObservationSet(values=(Observation("energy", 0.0, case_id="a"), observation))
    with pytest.raises(ReferenceYAMLError) as raised:
        save_reference_yaml(path, ref, [_molecule("a"), _molecule("b")])
    message = str(raised.value)
    assert observation.kind in message
    assert "'b'" in message
    _assert_unchanged(tmp_path, before)


@pytest.mark.parametrize("label", ["", "explicit label"])
def test_supported_scalar_fields_and_molecular_metadata_roundtrip(tmp_path: Path, label: str) -> None:
    values = (
        Observation("energy", -1.0, weight=2.0, label=label, case_id="a"),
        Observation("frequency", 123.0, data_idx=2, weight=0.5, label=label, case_id="a"),
        Observation("eig_diagonal", 1.0, data_idx=2, label=label, case_id="a"),
        Observation("eig_offdiagonal", 0.1, atom_indices=(0, 1), label=label, case_id="a"),
        Observation("hessian_element", 0.2, atom_indices=(1, 2), label=label, case_id="a"),
        Observation("bond_length", 1.0, atom_indices=(0, 1), label=label, case_id="b"),
        Observation("bond_angle", 90.0, atom_indices=(0, 1, 2), label=label, case_id="b"),
        Observation("torsion_angle", 60.0, atom_indices=(0, 1, 2, 3), label=label, case_id="b"),
        Observation("bond_length", 1.5, data_idx=1, label=label, case_id="b"),
        Observation("bond_angle", 90.0, data_idx=1, label=label, case_id="b"),
        Observation("torsion_angle", 60.0, data_idx=0, label=label, case_id="b"),
    )
    ref = ObservationSet(values=values)
    molecules = [_molecule("a"), _molecule("b")]
    path = tmp_path / "supported.yaml"
    for _ in range(2):
        save_reference_yaml(path, ref, molecules)
        loaded_ref, loaded_molecules = load_reference_yaml(path)
        assert [observation_payload(value) for value in loaded_ref.values] == [
            observation_payload(value) for value in values
        ]
        assert [molecule_fingerprint_payload(mol) for mol in loaded_molecules] == [
            molecule_fingerprint_payload(mol) for mol in molecules
        ]
        ref, molecules = loaded_ref, loaded_molecules


@pytest.mark.parametrize(
    "datum,implicit_labels",
    [
        ({"kind": "frequency", "value": 1.0, "data_idx": 2}, ["mode 2"]),
        ({"kind": "frequency", "values": [1.0, 2.0]}, ["mode 0", "mode 1"]),
        ({"kind": "eig_diagonal", "value": 1.0, "mode_idx": 2}, ["eig[2]"]),
        ({"kind": "eig_offdiagonal", "value": 1.0, "row": 0, "col": 1}, ["eig[0,1]"]),
        ({"kind": "hessian_element", "value": 1.0, "row": 1, "col": 2}, ["hess[1,2]"]),
    ],
)
def test_existing_label_field_preserves_explicit_empty_without_changing_missing_defaults(
    datum: dict[str, object], implicit_labels: list[str]
) -> None:
    assert [value.label for value in _parse_datum(datum, "a", "label control")] == implicit_labels
    assert [value.label for value in _parse_datum({**datum, "label": ""}, "a", "label control")] == [""] * len(
        implicit_labels
    )
    assert [value.label for value in _parse_datum({**datum, "label": "custom"}, "a", "label control")] == [
        "custom"
    ] * len(implicit_labels)
