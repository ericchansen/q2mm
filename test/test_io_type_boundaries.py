"""Focused checks for the now-type-checked reference-YAML and Mol2 boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import get_args, get_type_hints
from unittest.mock import patch

import pytest
import yaml

from q2mm.io import reference
from q2mm.io.mol2 import Mol2
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet, ObservationValue, _ObservationKind
from test._shared import ETHANE_DIR


@pytest.mark.parametrize("kind", get_args(_ObservationKind))
def test_reference_kind_guard_preserves_the_existing_scalar_vocabulary(kind: str) -> None:
    assert reference._is_reference_kind(kind)
    assert reference._validate_kind(kind, "typed boundary") is kind


@pytest.mark.parametrize("kind", ["Energy", "energy ", "", "unknown", "eigenmatrix", "atomic_partial_charge"])
def test_reference_kind_guard_does_not_expand_or_normalize_the_schema(kind: str) -> None:
    assert not reference._is_reference_kind(kind)
    with pytest.raises(reference.ReferenceYAMLError, match="Unknown kind") as raised:
        reference._validate_kind(kind, "typed boundary")
    assert str(raised.value) == (
        f"Unknown kind '{kind}' in typed boundary. Must be one of: {sorted(reference._VALID_KINDS)}"
    )


def test_reference_boundary_annotations_reuse_canonical_observation_aliases() -> None:
    assert get_type_hints(reference._validate_kind)["return"] == _ObservationKind
    assert get_type_hints(reference._reference_value_to_dict)["rv"] == ObservationValue


@pytest.mark.parametrize("atom_types", [None, [1, "H1"]])
def test_inline_geometry_and_observation_set_receive_normalized_tuples(
    tmp_path: Path, atom_types: list[object] | None
) -> None:
    geometry = {"symbols": ["H", "H"], "coordinates": [[0, 0, 0], [0.74, 0, 0]], "atom_types": atom_types}
    path = tmp_path / "reference.yaml"
    path.write_text(
        yaml.safe_dump(
            {"molecules": [{"name": "h2", "geometry": geometry, "data": [{"kind": "frequency", "values": [1, 2.0]}]}]}
        ),
        encoding="utf-8",
    )
    with (
        patch.object(reference, "Molecule", wraps=Molecule) as molecule_factory,
        patch.object(reference, "ObservationSet", wraps=ObservationSet) as observation_factory,
    ):
        observations, molecules = reference.load_reference_yaml(path)
    molecule_factory.assert_called_once()
    observation_factory.assert_called_once()
    assert molecule_factory.call_args.kwargs["symbols"] == ("H", "H")
    assert molecule_factory.call_args.kwargs["atom_types"] == (None if atom_types is None else ("1", "H1"))
    assert isinstance(observation_factory.call_args.kwargs["values"], tuple)
    assert observation_factory.call_args.kwargs["values"] == observations.values
    assert molecules[0].atom_types == (("H", "H") if atom_types is None else ("1", "H1"))
    assert [observation.value for observation in observations.values] == [1.0, 2.0]
    assert [observation.case_id for observation in observations.values] == ["h2", "h2"]


def test_mol2_caches_remain_lazy_and_parse_lines_keeps_its_return_contract() -> None:
    parser = Mol2(str(ETHANE_DIR / "GS.mol2"))
    assert parser._lines is None
    assert parser._records is None
    lines = parser.lines
    assert isinstance(lines, list)
    assert parser.lines is lines
    assert parser._records is None
    records = parser._records_parsed
    assert isinstance(records, list)
    assert parser._records_parsed is records
    assert parser.parse_lines() is None
    assert parser._records is not None
    assert parser._records_parsed == records
    assert parser.lines is lines
