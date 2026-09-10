"""Dependency-light coverage for standalone OpenMM ForceField XML."""

from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from q2mm.io.openmm import save_openmm_xml
from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    CmapGrid,
    ForceField,
    FunctionalForm,
    StretchBendParam,
    TorsionParam,
    VdwParam,
)


@pytest.mark.parametrize(
    ("ff", "message"),
    [
        (
            ForceField(
                angles=[AngleParam(("H", "O", "H"), 104.5, 40.0, ub_force_constant=3.0, ub_equilibrium=1.5)],
                functional_form=FunctionalForm.MM3,
            ),
            "Urey-Bradley",
        ),
        (
            ForceField(
                stretch_bends=[StretchBendParam(("H", "O", "H"), 1.0)],
                functional_form=FunctionalForm.MM3,
            ),
            "stretch-bend",
        ),
        (
            ForceField(
                cmaps=[CmapGrid(("C",) * 4, ("C",) * 4, 2, (1.0, 2.0, 3.0, 4.0))],
                functional_form=FunctionalForm.MM3,
            ),
            "CMAP",
        ),
        (
            ForceField(
                torsions=[TorsionParam(("C",) * 4, force_constant=2.0, is_improper=True)],
                functional_form=FunctionalForm.MM3,
            ),
            "improper",
        ),
        (
            ForceField(
                torsions=[
                    TorsionParam(("H", "C", "C", "F"), periodicity=1, force_constant=1.0),
                    TorsionParam(("F", "C", "C", "H"), periodicity=2, force_constant=2.0),
                ],
                functional_form=FunctionalForm.MM3,
            ),
            "multiple.*torsion",
        ),
        (
            ForceField(
                bonds=[BondParam(("C", "C"), 1.5, 100.0, dipole_moment=0.5)],
                functional_form=FunctionalForm.MM3,
            ),
            "dipole",
        ),
        (
            ForceField(
                vdws=[VdwParam("H", 1.2, 0.02, reduction=0.9)],
                functional_form=FunctionalForm.MM3,
            ),
            "reduc",
        ),
        (
            ForceField(
                bonds=[BondParam(("C", "C"), 1.5, 100.0, env_id="00-C")],
                functional_form=FunctionalForm.MM3,
            ),
            "wildcard",
        ),
        (
            ForceField(
                angles=[AngleParam(("C", "C", "C"), 109.5, 40.0, env_id="00-C-00")],
                functional_form=FunctionalForm.MM3,
            ),
            "wildcard",
        ),
        (
            ForceField(
                bonds=[BondParam(("C", "C"), 1.5, 100.0)],
                torsions=[TorsionParam(("C",) * 4, force_constant=2.0, env_id="00-C-C-00")],
                functional_form=FunctionalForm.MM3,
            ),
            "wildcard",
        ),
        (
            ForceField(
                torsions=[TorsionParam(("C",) * 4, force_constant=2.0, env_id="C-C-C-C-extra")],
                functional_form=FunctionalForm.MM3,
            ),
            "complete atom classes",
        ),
    ],
    ids=[
        "urey-bradley",
        "stretch-bend",
        "cmap",
        "improper",
        "multiple-propers",
        "dipole",
        "reduction",
        "wildcard-bond",
        "wildcard-angle",
        "partial-wildcard-torsion",
        "malformed-torsion-classes",
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_rejects_unrepresented_physics_before_writing(
    tmp_path: Path, ff: ForceField, message: str, existing: bool
) -> None:
    destination = tmp_path / "forcefield.xml"
    if existing:
        destination.write_bytes(b"existing force field")

    with pytest.raises(ValueError, match=message):
        save_openmm_xml(ff, destination)

    if existing:
        assert destination.read_bytes() == b"existing force field"
    else:
        assert not destination.exists()


def test_single_proper_uses_openmm_schema(tmp_path: Path) -> None:
    ff = ForceField(
        torsions=[TorsionParam(("H", "C", "C", "H"), periodicity=3, force_constant=1.75, phase=180.0)],
        functional_form=FunctionalForm.MM3,
    )

    destination = save_openmm_xml(ff, tmp_path / "forcefield.xml")
    root = ET.parse(destination).getroot()

    assert root.find("CustomTorsionForce/Proper") is not None
    assert root.find("CustomTorsionForce/Torsion") is None


@pytest.mark.parametrize("atom_type", ["00", "0", "000", "+0", "-00"])
@pytest.mark.parametrize("existing", [False, True])
def test_wildcard_vdw_rejected_before_writing(tmp_path: Path, atom_type: str, existing: bool) -> None:
    ff = ForceField(vdws=(VdwParam(atom_type, 1.2, 0.02),), functional_form=FunctionalForm.MM3)
    path = tmp_path / "vdw.xml"
    if existing:
        path.write_bytes(b"existing XML")

    with pytest.raises(ValueError, match="wildcard"):
        save_openmm_xml(ff, path)

    assert path.read_bytes() == b"existing XML" if existing else not path.exists()


@pytest.mark.parametrize("atom_type", ["0", "000", "+0"])
def test_other_native_zero_classes_are_rejected(tmp_path: Path, atom_type: str) -> None:
    ff = ForceField(
        bonds=(BondParam(("C", "C"), 1.5, 100.0, env_id=f"{atom_type}-C"),),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "bond.xml"
    with pytest.raises(ValueError, match="wildcard"):
        save_openmm_xml(ff, path)
    assert not path.exists()


@pytest.mark.parametrize("atom_type", ["H", "He", "C00", "00C", "0001"])
def test_vdw_wildcard_lookalikes_remain_literal_classes(tmp_path: Path, atom_type: str) -> None:
    ff = ForceField(vdws=(VdwParam(atom_type, 1.2, 0.02),), functional_form=FunctionalForm.MM3)
    path = save_openmm_xml(ff, tmp_path / "vdw.xml")
    atom = ET.parse(path).getroot().find("CustomNonbondedForce/Atom")
    assert atom is not None
    assert atom.get("class") == atom_type


@pytest.mark.parametrize("existing", [False, True])
def test_nonbonded_excluded_types_are_not_silently_lost(tmp_path: Path, existing: bool) -> None:
    ff = ForceField(
        vdws=(VdwParam("H", 1.2, 0.02),),
        nonbonded_excluded_atom_types=("H",),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "excluded.xml"
    if existing:
        path.write_bytes(b"existing XML")

    with pytest.raises(ValueError, match="nonbonded-excluded"):
        save_openmm_xml(ff, path)

    assert path.read_bytes() == b"existing XML" if existing else not path.exists()


def test_nonzero_unit_reduction_is_not_silently_discarded(tmp_path: Path) -> None:
    ff = ForceField(vdws=(VdwParam("H", 1.2, 0.02, reduction=1.0),), functional_form=FunctionalForm.MM3)
    path = tmp_path / "reduction.xml"
    with pytest.raises(ValueError, match="reduc"):
        save_openmm_xml(ff, path)
    assert not path.exists()


@pytest.mark.parametrize("family", ["bonds", "angles", "torsions"])
@pytest.mark.parametrize("length_delta", [-1, 1])
@pytest.mark.parametrize("typed", [False, True])
def test_fixed_element_and_class_arities_are_required(
    tmp_path: Path, family: str, length_delta: int, typed: bool
) -> None:
    arity = {"bonds": 2, "angles": 3, "torsions": 4}[family]
    elements = ("C",) * (arity + length_delta)
    env_id = "-".join(("C",) * arity) if typed else ""
    if family == "bonds":
        parameters = {"bonds": (BondParam(elements, 1.5, 100.0, env_id=env_id),)}
    elif family == "angles":
        parameters = {"angles": (AngleParam(elements, 109.5, 30.0, env_id=env_id),)}
    else:
        parameters = {"torsions": (TorsionParam(elements, force_constant=1.0, env_id=env_id),)}
    ff = ForceField(functional_form=FunctionalForm.MM3, **parameters)
    path = tmp_path / "arity.xml"
    path.write_bytes(b"existing XML")
    with pytest.raises(ValueError, match="elements|atom classes"):
        save_openmm_xml(ff, path)
    assert path.read_bytes() == b"existing XML"


def test_whitespace_does_not_disguise_a_native_zero_class(tmp_path: Path) -> None:
    ff = ForceField(
        bonds=(BondParam(("C", "C"), 1.5, 100.0, env_id="00 -C"),),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "wildcard.xml"
    with pytest.raises(ValueError, match="atom classes"):
        save_openmm_xml(ff, path)
    assert not path.exists()


def test_valid_list_elements_can_be_exported(tmp_path: Path) -> None:
    elements = ["H", "C", "C", "H"]
    ff = ForceField(
        torsions=(TorsionParam(elements, periodicity=3, force_constant=1.0),),
        functional_form=FunctionalForm.MM3,
    )
    path = save_openmm_xml(ff, tmp_path / "list.xml")
    proper = ET.parse(path).getroot().find("CustomTorsionForce/Proper")
    assert proper is not None
    assert [proper.get(f"class{i}") for i in range(1, 5)] == elements
    assert ff.torsions[0].elements is elements


def test_mixed_element_containers_still_detect_duplicate_torsions(tmp_path: Path) -> None:
    ff = ForceField(
        torsions=(
            TorsionParam(["H", "C", "C", "H"], periodicity=1, force_constant=1.0),
            TorsionParam(("H", "C", "C", "H"), periodicity=2, force_constant=1.0),
        ),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "mixed.xml"
    with pytest.raises(ValueError, match="multiple.*torsion"):
        save_openmm_xml(ff, path)
    assert not path.exists()


@pytest.mark.parametrize(
    "selector",
    [
        {"bond_order": "-"},
        {"bond_order": "="},
        {"bond_order": "*"},
        {"bond_order": "%"},
        {"context": "O200 0000"},
        {"context": "special"},
    ],
)
def test_unrepresentable_bond_selectors_are_rejected(tmp_path: Path, selector: dict[str, str]) -> None:
    ff = ForceField(
        bonds=(BondParam(("C", "C"), 1.5, 100.0, **selector),),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "selector.xml"
    path.write_bytes(b"original XML")
    with pytest.raises(ValueError, match="selector"):
        save_openmm_xml(ff, path)
    assert path.read_bytes() == b"original XML"


@pytest.mark.parametrize("family", ["bond", "angle"])
def test_duplicate_bonded_classes_cannot_select_an_arbitrary_row(tmp_path: Path, family: str) -> None:
    if family == "bond":
        ff = ForceField(
            bonds=(BondParam(("C", "H"), 1.1, 100.0), BondParam(("H", "C"), 1.2, 200.0)),
            functional_form=FunctionalForm.MM3,
        )
    else:
        ff = ForceField(
            angles=(AngleParam(("C", "O", "H"), 100.0, 20.0), AngleParam(("H", "O", "C"), 110.0, 30.0)),
            functional_form=FunctionalForm.MM3,
        )
    path = tmp_path / "duplicate.xml"
    with pytest.raises(ValueError, match=f"multiple.*{family}"):
        save_openmm_xml(ff, path)
    assert not path.exists()


@pytest.mark.parametrize("env_id", ["C -H", "C- H", "\tC-H", "C-H "])
def test_padded_atom_classes_are_rejected(tmp_path: Path, env_id: str) -> None:
    ff = ForceField(
        bonds=(BondParam(("C", "H"), 1.1, 100.0, env_id=env_id),),
        functional_form=FunctionalForm.MM3,
    )
    path = tmp_path / "padded.xml"
    with pytest.raises(ValueError, match="atom classes"):
        save_openmm_xml(ff, path)
    assert not path.exists()


@pytest.mark.parametrize("context", ["", "0000 0000"])
def test_generic_bond_context_remains_supported(tmp_path: Path, context: str) -> None:
    ff = ForceField(
        bonds=(BondParam(("C", "H"), 1.1, 100.0, context=context),),
        functional_form=FunctionalForm.MM3,
    )
    path = save_openmm_xml(ff, tmp_path / "generic.xml")
    assert ET.parse(path).getroot().find("CustomBondForce/Bond") is not None
