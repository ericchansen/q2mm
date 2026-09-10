"""Tinker header, scalar-domain, and reversible-template boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.io import tinker
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, VdwParam


def _standalone() -> ForceField:
    return ForceField(
        functional_form=FunctionalForm.MM3,
        bonds=(BondParam(("C", "H"), 1.1, 100.0, env_id="C1-H1"),),
        angles=(AngleParam(("H", "C", "H"), 109.5, 50.0, env_id="H1-C1-H1"),),
        vdws=(VdwParam("H1", 1.2, 0.1),),
    )


@pytest.mark.parametrize("header", ["Q2MMish", "Q2MM_guide", "Q2MM2"])
def test_only_complete_q2mm_header_selects_blocks(tmp_path: Path, header: str) -> None:
    source = tmp_path / "source.prm"
    source.write_text(f"# {header}\nbond C1 H1 2.0 1.1\n", encoding="utf-8")
    force_field = tinker.load_tinker_prm(source)
    assert len(force_field.bonds) == 1
    output = tmp_path / "copy.prm"
    tinker.save_tinker_prm(force_field, output)
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("header", ["NOT OPT", "OPTIMIZED", "ABOUT OPT"])
def test_only_complete_opt_header_selects_parameters(tmp_path: Path, header: str) -> None:
    source = tmp_path / "source.prm"
    source.write_text(
        f"# Q2MM\n# {header}\nbond C1 H1 2.0 1.1\n# OPT Actual\nbond C1 H1 3.0 1.2\n",
        encoding="utf-8",
    )
    force_field = tinker.load_tinker_prm(source)
    assert len(force_field.bonds) == 1
    assert force_field.bonds[0].ff_row == 5
    output = tmp_path / "edited.prm"
    edited = replace(force_field, bonds=(replace(force_field.bonds[0], equilibrium=1.3),))
    tinker.save_tinker_prm(edited, output)
    assert output.read_text().splitlines()[2] == "bond C1 H1 2.0 1.1"
    assert tinker.load_tinker_prm(output).bonds[0].equilibrium == 1.3


@pytest.mark.parametrize(
    "family,field",
    [
        ("bonds", "force_constant"),
        ("bonds", "equilibrium"),
        ("angles", "force_constant"),
        ("angles", "equilibrium"),
        ("vdws", "radius"),
        ("vdws", "epsilon"),
        ("vdws", "reduction"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("existing", [False, True])
def test_standalone_rejects_nonfinite_scalars_before_output(
    tmp_path: Path, family: str, field: str, value: float, existing: bool
) -> None:
    force_field = _standalone()
    parameter = replace(getattr(force_field, family)[0], **{field: value})
    force_field = replace(force_field, **{family: (parameter,)})
    output = tmp_path / "output.prm"
    if existing:
        output.write_bytes(b"preserve output")
    with pytest.raises(ValueError, match="finite"):
        tinker.save_tinker_prm(force_field, output)
    if existing:
        assert output.read_bytes() == b"preserve output"
    else:
        assert not output.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_standalone_validates_converted_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool) -> None:
    output = tmp_path / "output.prm"
    if existing:
        output.write_bytes(b"preserve output")
    monkeypatch.setattr(tinker, "canonical_to_mm3_angle_k", lambda _value: float("inf"))
    with pytest.raises(ValueError, match="finite"):
        tinker.save_tinker_prm(_standalone(), output)
    if existing:
        assert output.read_bytes() == b"preserve output"
    else:
        assert not output.exists()


@pytest.mark.parametrize("source_reversed", [False, True])
@pytest.mark.parametrize("canonical_hint", [False, True])
@pytest.mark.parametrize("source_hint", [False, True])
def test_reversed_heterogeneous_torsion_binds_without_rewriting_types(
    tmp_path: Path, source_reversed: bool, canonical_hint: bool, source_hint: bool
) -> None:
    labels = ("C1", "N1", "O1", "H1")
    source_labels = tuple(reversed(labels)) if source_reversed else labels
    source = tmp_path / "source.prm"
    source.write_text(f"torsion {' '.join(source_labels)} 1.0 37 1 # preserve types\n", encoding="utf-8")
    force_field = tinker.load_tinker_prm(source)
    original = force_field.torsions[0]
    edited = replace(
        original,
        elements=tuple(reversed(original.elements)),
        env_id="-".join(labels if canonical_hint else tuple(reversed(labels))),
        ff_row=original.ff_row if source_hint else None,
        force_constant=1.25,
        phase=52.0,
    )
    output = tmp_path / "edited.prm"
    tinker.save_tinker_prm(replace(force_field, torsions=(edited,)), output)
    loaded = tinker.load_tinker_prm(output).torsions[0]
    assert loaded.force_constant == 1.25
    assert loaded.phase == 52.0
    assert loaded.periodicity == original.periodicity
    assert loaded.elements in (edited.elements, tuple(reversed(edited.elements)))
    assert output.read_text().split()[:5] == ["torsion", *source_labels]
    assert output.read_text().endswith("# preserve types\n")


@pytest.mark.parametrize("elements", [("C", "O", "N", "H"), ("F", "N", "O", "H")])
def test_nonreversible_torsion_element_changes_remain_rejected(
    tmp_path: Path, elements: tuple[str, str, str, str]
) -> None:
    source = tmp_path / "source.prm"
    source.write_text("torsion C1 N1 O1 H1 1.0 37 1\n", encoding="utf-8")
    force_field = tinker.load_tinker_prm(source)
    edited = replace(force_field, torsions=(replace(force_field.torsions[0], elements=elements),))
    output = tmp_path / "output.prm"
    output.write_bytes(b"preserve output")
    with pytest.raises(ValueError, match="non-scalar"):
        tinker.save_tinker_prm(edited, output)
    assert output.read_bytes() == b"preserve output"
