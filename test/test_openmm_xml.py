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
