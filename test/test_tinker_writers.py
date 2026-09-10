"""CPU-only writer coverage; no Tinker executable is invoked."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.backends.contracts import PreparationError, PreparationRequest
from q2mm.backends.mm import tinker as native_tinker
from q2mm.backends.mm.tinker import TinkerBackend
from q2mm.io import tinker
from q2mm.io.tinker import load_tinker_prm, save_tinker_prm
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
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout

_LOSSES = [
    *[
        pytest.param(
            lambda ff, order=order: replace(ff, bonds=(replace(ff.bonds[0], bond_order=order),)),
            "bond order",
            id=f"bond-order-{name}",
        )
        for name, order in (("single", "-"), ("double", "="), ("aromatic", "*"), ("triple", "%"))
    ],
    pytest.param(
        lambda ff: replace(ff, bonds=(replace(ff.bonds[0], context="O200 0000"),)),
        "bond context",
        id="bond-context",
    ),
    pytest.param(
        lambda ff: replace(ff, stretch_bends=(StretchBendParam(("H", "C", "H"), 2.0),)),
        "stretch-bend",
        id="stretch-bend",
    ),
    pytest.param(
        lambda ff: replace(ff, stretch_bends=(StretchBendParam(("H", "C", "H"), 0.0),)),
        "stretch-bend",
        id="zero-stretch-bend",
    ),
    pytest.param(
        lambda ff: replace(ff, angles=(replace(ff.angles[0], ub_force_constant=3.0, ub_equilibrium=2.0),)),
        "Urey-Bradley",
        id="urey-bradley",
    ),
    pytest.param(
        lambda ff: replace(ff, angles=(replace(ff.angles[0], ub_force_constant=0.0, ub_equilibrium=0.0),)),
        "Urey-Bradley",
        id="zero-urey-bradley",
    ),
    pytest.param(
        lambda ff: replace(ff, angles=(replace(ff.angles[0], ub_force_constant=3.0),)),
        "Urey-Bradley",
        id="urey-bradley-force-only",
    ),
    pytest.param(
        lambda ff: replace(ff, angles=(replace(ff.angles[0], ub_equilibrium=2.0),)),
        "Urey-Bradley",
        id="urey-bradley-equilibrium-only",
    ),
    pytest.param(
        lambda ff: replace(ff, bonds=(replace(ff.bonds[0], dipole_moment=0.4),)),
        "dipole",
        id="dipole",
    ),
    pytest.param(
        lambda ff: replace(ff, bonds=(replace(ff.bonds[0], dipole_moment=-0.4),)),
        "dipole",
        id="negative-dipole",
    ),
    pytest.param(
        lambda ff: replace(ff, torsions=(TorsionParam(("H", "C", "C", "H"), force_constant=2.0, is_improper=True),)),
        "improper",
        id="improper",
    ),
    pytest.param(
        lambda ff: replace(ff, torsions=(TorsionParam(("H", "C", "C", "H"), force_constant=0.0, is_improper=True),)),
        "improper",
        id="zero-improper",
    ),
    pytest.param(
        lambda ff: replace(ff, cmaps=(CmapGrid(("C",) * 4, ("C",) * 4, 2, (1.0, 2.0, 3.0, 4.0)),)),
        "CMAP",
        id="cmap",
    ),
    pytest.param(
        lambda ff: replace(ff, nonbonded_excluded_atom_types=("H1",)),
        "nonbonded_excluded_atom_types",
        id="nonbonded-exclusion",
    ),
]
_NATIVE_LOSSES = [
    *_LOSSES,
    pytest.param(
        lambda ff: replace(ff, vdws=(VdwParam("", 1.5, 0.02),)),
        "vdW atom type",
        id="missing-vdw-type-and-element",
    ),
    *[
        pytest.param(
            lambda ff, atom_type=atom_type: replace(ff, vdws=(replace(ff.vdws[0], atom_type=atom_type, element="H"),)),
            "placeholder",
            id=f"zero-vdw-type-{index}",
        )
        for index, atom_type in enumerate(("0", "00", "+00", "-00", " 0 ", "0_0", "\u0660"))
    ],
    pytest.param(
        lambda ff: replace(ff, bonds=(replace(ff.bonds[0], elements=("00", "H"), force_constant=0.0),)),
        "placeholder",
        id="placeholder-bond",
    ),
    pytest.param(
        lambda ff: replace(ff, angles=(replace(ff.angles[0], elements=("00", "C", "H"), force_constant=0.0),)),
        "placeholder",
        id="placeholder-angle",
    ),
    pytest.param(
        lambda ff: replace(ff, torsions=(TorsionParam(("00", "C", "C", "H"), force_constant=0.0),)),
        "placeholder",
        id="placeholder-torsion",
    ),
    pytest.param(
        lambda ff: replace(ff, vdws=(replace(ff.vdws[0], element="00", epsilon=0.0),)),
        "placeholder",
        id="placeholder-vdw",
    ),
    pytest.param(
        lambda ff: replace(ff, vdws=(replace(ff.vdws[0], reduction=0.923),)),
        "reduction",
        id="vdw-reduction",
    ),
    pytest.param(
        lambda ff: replace(ff, vdws=(replace(ff.vdws[0], reduction=1.0),)),
        "reduction",
        id="unit-vdw-reduction",
    ),
]


@pytest.fixture
def simple_ff() -> ForceField:
    return ForceField(
        bonds=(BondParam(("C", "H"), 1.1, 100.0, env_id="C1-H1"),),
        angles=(AngleParam(("H", "C", "H"), 109.0, 50.0, env_id="H1-C1-H1"),),
        vdws=(VdwParam("H1", 1.5, 0.02),),
        functional_form=FunctionalForm.MM3,
    )


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.prm"
    path.write_bytes(
        b'atom 1 C "carbon # ! description" 6 12.0 4\r\n'
        b'atom 5 H "hydrogen" 1 1.0 1\r\n'
        b"torsionunit 0.5\r\n"
        b"# Q2MM\r\n# OPT Synthetic\r\n"
        b"bond 1 5 5.0 1.1 ! bond\r\n"
        b"angle 5 1 5 0.5 109.0 111.0 112.0 # extra equilibria\r\n"
        b"vdw 5 1.5 0.02 ! reduction omitted\r\n"
        b"torsion 5 1 1 5 1.0 30 4 -2.0 180 2 # proper\r\n"
        b"strbnd 5 1 5 2.0 3.0 ! native cross-term\r\n"
        b"ureybrad 5 1 5 3.0 2.0\r\n"
        b"dipole 1 5 0.4 0.5\r\n"
        b"imptors 5 1 1 5 2.0 180 2\r\n"
        b"opbend 5 1 1 5 0.5\r\n"
    )
    return path


@pytest.fixture
def backend(source: Path, tmp_path: Path) -> TinkerBackend:
    return TinkerBackend(tinker_dir=str(tmp_path), params_file=str(source))


@pytest.fixture
def molecule() -> Molecule:
    return Molecule(
        symbols=("C", "H"),
        atom_types=("1", "5"),
        geometry=((0.0, 0.0, 0.0), (1.1, 0.0, 0.0)),
    )


def _assert_unchanged(path: Path, before: bytes | None) -> None:
    if before is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == before


@pytest.mark.parametrize("edit,message", _LOSSES)
@pytest.mark.parametrize("mode", ["standalone", "implicit-template", "explicit-template"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_rejects_each_loss_before_writing(
    source: Path,
    simple_ff: ForceField,
    tmp_path: Path,
    edit: Callable[[ForceField], ForceField],
    message: str,
    mode: str,
    destination: str,
) -> None:
    ff = simple_ff if mode == "standalone" else load_tinker_prm(source)
    if mode == "explicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve existing destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match=message):
        save_tinker_prm(edit(ff), output, template_path=source if mode == "explicit-template" else None)
    _assert_unchanged(output, before)


@pytest.mark.parametrize("edit,message", _NATIVE_LOSSES)
@pytest.mark.parametrize("entry", ["standalone", "xyz"])
@pytest.mark.parametrize("existing", [False, True])
def test_native_rejects_each_loss_before_writing(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    edit: Callable[[ForceField], ForceField],
    message: str,
    entry: str,
    existing: bool,
) -> None:
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    if existing:
        for path in paths:
            path.write_bytes(b"preserve " + path.suffix.encode())
    before = [path.read_bytes() if path.exists() else None for path in paths]
    with pytest.raises(PreparationError, match=message):
        if entry == "standalone":
            backend._write_standalone_prm(edit(simple_ff), str(paths[0]), ["C", "H"], [1, 5])
        else:
            backend._write_tinker_xyz(molecule, edit(simple_ff), str(tmp_path))
    for path, content in zip(paths, before, strict=True):
        _assert_unchanged(path, content)


@pytest.mark.parametrize("edit,message", _LOSSES)
@pytest.mark.parametrize("fallback_template", [False, True])
@pytest.mark.parametrize("existing", [False, True])
def test_backend_template_rejects_canonical_losses_before_writing(
    backend: TinkerBackend,
    molecule: Molecule,
    source: Path,
    tmp_path: Path,
    edit: Callable[[ForceField], ForceField],
    message: str,
    fallback_template: bool,
    existing: bool,
) -> None:
    ff = load_tinker_prm(source)
    if fallback_template:
        ff = replace(ff, source_path=None)
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    if existing:
        for path in paths:
            path.write_bytes(b"preserve " + path.suffix.encode())
    before = [path.read_bytes() if path.exists() else None for path in paths]
    with pytest.raises(PreparationError, match=message):
        backend._write_tinker_xyz(molecule, edit(ff), str(tmp_path))
    for path, content in zip(paths, before, strict=True):
        _assert_unchanged(path, content)


@pytest.mark.parametrize("edit,message", _NATIVE_LOSSES)
def test_backend_prepare_rejects_standalone_losses(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    edit: Callable[[ForceField], ForceField],
    message: str,
) -> None:
    with pytest.raises(PreparationError, match=message):
        backend.prepare(PreparationRequest(case_id="loss", molecule=molecule, force_field=edit(simple_ff)))


@pytest.mark.parametrize("edit,message", _LOSSES)
def test_backend_prepare_rejects_canonical_template_losses(
    backend: TinkerBackend,
    molecule: Molecule,
    source: Path,
    edit: Callable[[ForceField], ForceField],
    message: str,
) -> None:
    with pytest.raises(PreparationError, match=message):
        backend.prepare(
            PreparationRequest(case_id="loss", molecule=molecule, force_field=edit(load_tinker_prm(source)))
        )


@pytest.mark.parametrize("existing", [False, True])
def test_backend_template_unrepresentable_edit_preserves_all_destinations(
    backend: TinkerBackend, molecule: Molecule, source: Path, tmp_path: Path, existing: bool
) -> None:
    ff = load_tinker_prm(source)
    ff = replace(ff, torsions=(replace(ff.torsions[0], periodicity=5), *ff.torsions[1:]))
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    if existing:
        for path in paths:
            path.write_bytes(b"preserve output")
    with pytest.raises(ValueError, match="Tinker"):
        backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    for path in paths:
        _assert_unchanged(path, b"preserve output" if existing else None)


def test_public_standalone_preserves_supported_reduction(simple_ff: ForceField, tmp_path: Path) -> None:
    ff = replace(simple_ff, vdws=(replace(simple_ff.vdws[0], reduction=0.923),))
    output = save_tinker_prm(ff, tmp_path / "supported.prm")
    actual = load_tinker_prm(output)
    assert actual.bonds[0].force_constant == pytest.approx(ff.bonds[0].force_constant, abs=0.01)
    assert actual.angles[0].force_constant == pytest.approx(ff.angles[0].force_constant, abs=0.01)
    assert (actual.vdws[0].radius, actual.vdws[0].epsilon, actual.vdws[0].reduction) == (1.5, 0.02, 0.923)


@pytest.mark.parametrize("context", ["", "0000 0000"])
@pytest.mark.parametrize("entry", ["public", "backend"])
def test_standalone_accepts_generic_bond_context(
    backend: TinkerBackend, molecule: Molecule, simple_ff: ForceField, tmp_path: Path, context: str, entry: str
) -> None:
    ff = replace(simple_ff, bonds=(replace(simple_ff.bonds[0], context=context),))
    output = tmp_path / "molecule.prm"
    if entry == "public":
        save_tinker_prm(ff, output)
    else:
        backend.prepare(PreparationRequest(case_id="generic", molecule=molecule, force_field=ff))
        backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    actual = load_tinker_prm(output)
    assert len(actual.bonds) == 1
    assert actual.bonds[0].force_constant == pytest.approx(ff.bonds[0].force_constant, abs=0.01)
    assert actual.bonds[0].equilibrium == ff.bonds[0].equilibrium


@pytest.mark.parametrize("entry", ["standalone", "xyz"])
@pytest.mark.parametrize("existing", [False, True])
def test_native_missing_torsion_element_preserves_all_destinations(
    backend: TinkerBackend, molecule: Molecule, simple_ff: ForceField, tmp_path: Path, entry: str, existing: bool
) -> None:
    ff = replace(simple_ff, torsions=(TorsionParam(("O", "C", "C", "H"), force_constant=2.0),))
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    if existing:
        for path in paths:
            path.write_bytes(b"preserve " + path.suffix.encode())
    before = [path.read_bytes() if path.exists() else None for path in paths]
    with pytest.raises(ValueError, match="torsion.*'O'.*not present"):
        if entry == "standalone":
            backend._write_standalone_prm(ff, str(paths[0]), ["C", "H"], [1, 5])
        else:
            backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    for path, content in zip(paths, before, strict=True):
        _assert_unchanged(path, content)


@pytest.mark.parametrize("entry", ["public", "backend", "backend-fallback"])
def test_templates_preserve_native_wildcard_records(
    backend: TinkerBackend, molecule: Molecule, source: Path, tmp_path: Path, entry: str
) -> None:
    source.write_bytes(
        source.read_bytes()
        + b"bond 0 5 5.0 1.1\r\n"
        + b"angle 0 1 5 0.5 109.0\r\n"
        + b"torsion 0 1 1 0 2.0 180 2\r\n"
        + b"vdw 0 1.5 0.02\r\n"
    )
    ff = load_tinker_prm(source)
    if entry == "backend-fallback":
        ff = replace(ff, source_path=None)
    output = tmp_path / "molecule.prm"
    if entry == "public":
        save_tinker_prm(ff, output)
    else:
        backend.prepare(PreparationRequest(case_id="wildcard-template", molecule=molecule, force_field=ff))
        backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    assert output.read_bytes() == source.read_bytes()


@pytest.mark.parametrize("marked", [False, True])
@pytest.mark.parametrize("entry", ["public", "backend", "backend-fallback"])
def test_templates_preserve_opaque_native_terms_and_supported_scalar_edits(
    backend: TinkerBackend, molecule: Molecule, source: Path, tmp_path: Path, marked: bool, entry: str
) -> None:
    if not marked:
        source.write_bytes(source.read_bytes().replace(b"# Q2MM\r\n# OPT Synthetic\r\n", b""))
    expected = source.read_bytes()
    ff = load_tinker_prm(source)
    assert not ff.stretch_bends and not ff.has_urey_bradley
    assert all(b.dipole_moment == 0.0 for b in ff.bonds)
    assert all(not t.is_improper for t in ff.torsions)
    if entry == "backend-fallback":
        ff = replace(ff, source_path=None)
    output = tmp_path / "molecule.prm"
    for edited in (False, True):
        if edited:
            ff = replace(
                ff,
                bonds=(replace(ff.bonds[0], equilibrium=1.25),),
                torsions=(replace(ff.torsions[0], force_constant=1.25, phase=45.0), *ff.torsions[1:]),
                vdws=(replace(ff.vdws[0], reduction=0.923),),
            )
            expected = (
                expected.replace(b"5.0 1.1 !", b"5.0 1.25 !")
                .replace(b"1.0 30 4", b"2.5 45.0 4")
                .replace(b"vdw 5 1.5 0.02 !", b"vdw 5 1.5 0.02 0.923 !")
            )
        if entry == "public":
            save_tinker_prm(ff, output)
        else:
            backend.prepare(PreparationRequest(case_id="supported", molecule=molecule, force_field=ff))
            backend._write_tinker_xyz(molecule, ff, str(tmp_path))
        assert output.read_bytes() == expected
        actual = load_tinker_prm(output)
        for category in ("bonds", "angles", "torsions", "vdws"):
            assert getattr(actual, category) == getattr(ff, category)


@pytest.mark.parametrize("vdw_type", ["", "H1", "5", "005", "+5"])
def test_native_standalone_preserves_its_supported_model(
    backend: TinkerBackend, molecule: Molecule, simple_ff: ForceField, tmp_path: Path, vdw_type: str
) -> None:
    ff = replace(
        simple_ff,
        vdws=(replace(simple_ff.vdws[0], atom_type=vdw_type, element="H"),),
        torsions=(TorsionParam(("H", "C", "C", "H"), periodicity=4, force_constant=2.25, phase=30.0),),
    )
    backend.prepare(PreparationRequest(case_id="supported", molecule=molecule, force_field=ff))
    xyz = backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    assert Path(xyz).is_file()
    output = tmp_path / "molecule.prm"
    assert (tmp_path / "molecule.key").read_text() == f"parameters {output}\n"
    assert "angle-sextic            0.000000022" in output.read_text()
    actual = load_tinker_prm(output)
    assert actual.bonds[0].force_constant == pytest.approx(100.0, abs=0.01)
    assert actual.angles[0].force_constant == pytest.approx(50.0, abs=0.01)
    assert (actual.torsions[0].force_constant, actual.torsions[0].periodicity, actual.torsions[0].phase) == (
        2.25,
        4,
        30.0,
    )
    assert not actual.torsions[0].is_improper
    assert actual.vdws[0].reduction == 0.0


def _native_failure(
    backend: TinkerBackend,
    molecule: Molecule,
    ff: ForceField,
    tmp_path: Path,
    entry: str,
    destination: str,
    message: str,
) -> None:
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    if destination != "absent":
        for path in paths:
            path.write_bytes(b"preserve " + path.suffix.encode())
    if destination == "source":
        paths[0].write_bytes(Path(backend._params_file).read_bytes())
        ff = replace(ff, source_path=paths[0])
    before = [path.read_bytes() if path.exists() else None for path in paths]
    error = PreparationError if entry == "prepare" else ValueError
    with pytest.raises(error, match=message):
        if entry == "prepare":
            backend.prepare(PreparationRequest(case_id="native-preflight", molecule=molecule, force_field=ff))
        elif entry == "standalone":
            backend._write_standalone_prm(
                ff, str(paths[0]), list(molecule.symbols), [int(t) for t in molecule.atom_types]
            )
        else:
            backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    for path, content in zip(paths, before, strict=True):
        _assert_unchanged(path, content)


@pytest.mark.parametrize(
    "family,field",
    [
        ("bonds", "force_constant"),
        ("bonds", "equilibrium"),
        ("angles", "force_constant"),
        ("angles", "equilibrium"),
        ("torsions", "force_constant"),
        ("torsions", "phase"),
        ("vdws", "radius"),
        ("vdws", "epsilon"),
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_scalar_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    family: str,
    field: str,
    value: float,
    entry: str,
    destination: str,
) -> None:
    ff = replace(simple_ff, torsions=(TorsionParam(("H", "C", "C", "H"), force_constant=2.0),))
    ff = replace(ff, **{family: (replace(getattr(ff, family)[0], **{field: value}),)})
    _native_failure(backend, molecule, ff, tmp_path, entry, destination, "finite")


@pytest.mark.parametrize(
    "family,converter", [("bonds", "canonical_to_mm3_bond_k"), ("angles", "canonical_to_mm3_angle_k")]
)
@pytest.mark.parametrize("boundary", ["converted-overflow", "raw-masked"])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_conversion_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    family: str,
    converter: str,
    boundary: str,
    entry: str,
    destination: str,
) -> None:
    if boundary == "converted-overflow":
        # Current MM3 factors reduce floats; inject an overflowing conversion
        # to exercise the separate post-conversion gate with finite input.
        monkeypatch.setattr(native_tinker, converter, lambda value: value * 1e308)
        ff = simple_ff
    else:
        monkeypatch.setattr(native_tinker, converter, lambda _value: 1.0)
        ff = replace(simple_ff, **{family: (replace(getattr(simple_ff, family)[0], force_constant=float("inf")),)})
    _native_failure(backend, molecule, ff, tmp_path, entry, destination, "finite")


@pytest.mark.parametrize("atom_type", ["0", "00", "+00", "-00", " 0 ", "0_0", "\u0660", "-5"])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_resolved_type_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    atom_type: str,
    entry: str,
    destination: str,
) -> None:
    mol = replace(molecule, atom_types=("1", atom_type))
    _native_failure(backend, mol, simple_ff, tmp_path, entry, destination, "positive")


@pytest.mark.parametrize(
    "vdw",
    [
        VdwParam("5", 1.5, 0.02, element="H"),
        VdwParam("005", 1.5, 0.02),
        VdwParam("-5", 1.5, 0.02, element="H"),
        VdwParam("42", 1.5, 0.02, element="C"),
        VdwParam("1", 1.5, 0.02, element="H"),
        VdwParam("42", 1.5, 0.02, element="005"),
    ],
    ids=[
        "unused-explicit",
        "unused-inferred",
        "negative",
        "wrong-element",
        "other-assigned-class",
        "wrong-numeric-element",
    ],
)
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_vdw_binding_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    vdw: VdwParam,
    entry: str,
    destination: str,
) -> None:
    mol = replace(molecule, atom_types=("1", "42"))
    _native_failure(backend, mol, replace(simple_ff, vdws=(vdw,)), tmp_path, entry, destination, "vdW.*class")


@pytest.mark.parametrize(
    "vdw_type,element,actual_type",
    [
        ("5", "", "5"),
        ("005", "", "005"),
        ("+5", "", "+5"),
        ("0_5", "", "0_5"),
        ("\u0665", "", "\u0665"),
        ("42", "", "42"),
        ("042", "", "+42"),
        ("42", "H", "42"),
        ("H1", "H", "42"),
        ("", "H", "42"),
        ("H1", "H", ""),
        ("H1", "H", "H1"),
    ],
)
def test_native_numeric_and_element_binding_controls(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    vdw_type: str,
    element: str,
    actual_type: str,
) -> None:
    ff = replace(simple_ff, vdws=(VdwParam(vdw_type, 1.5, 0.02, element=element),))
    mol = replace(molecule, atom_types=("1", actual_type))
    backend.prepare(PreparationRequest(case_id="numeric-control", molecule=mol, force_field=ff))
    xyz = Path(backend._write_tinker_xyz(mol, ff, str(tmp_path)))
    actual = load_tinker_prm(tmp_path / "molecule.prm")
    assigned = actual_type if actual_type not in ("", "H1") else "5"
    assert xyz.read_text().splitlines()[2].split()[5] == str(int(assigned))
    assert actual.vdws[0].atom_type == str(int(assigned))
    assert actual.vdws[0].element == "H"
    assert (actual.vdws[0].radius, actual.vdws[0].epsilon) == (1.5, 0.02)


@pytest.mark.parametrize(
    "folds",
    [
        (0,),
        (7,),
        (-1,),
        (1, 1),
        (6, 6),
        (1, 2, 3, 4, 5, 6, 7),
        pytest.param((1.5,), id="fractional"),
        pytest.param((True,), id="boolean"),
        pytest.param(("1 2.0 0 2",), id="malformed-triplet"),
    ],
)
@pytest.mark.parametrize("reversed_group", [False, True])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_torsion_group_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    folds: tuple[int, ...],
    reversed_group: bool,
    entry: str,
    destination: str,
) -> None:
    elements = ("C", "C", "C", "H")
    torsions = tuple(
        TorsionParam(
            elements[::-1] if reversed_group and i % 2 else elements,
            periodicity=fold,
            force_constant=i + 1.0,
            phase=30.0,
            env_id=f"environment-{i}",
        )
        for i, fold in enumerate(folds)
    )
    _native_failure(backend, molecule, replace(simple_ff, torsions=torsions), tmp_path, entry, destination, "torsion")


@pytest.mark.parametrize("reversed_group", [False, True])
def test_native_all_six_folds_roundtrip(
    backend: TinkerBackend, molecule: Molecule, simple_ff: ForceField, tmp_path: Path, reversed_group: bool
) -> None:
    elements = ("C", "C", "C", "H")
    torsions = tuple(
        TorsionParam(
            elements[::-1] if reversed_group and fold % 2 else elements,
            periodicity=fold,
            force_constant=fold * -0.25,
            phase=fold * 15.0,
        )
        for fold in (6, 2, 5, 1, 3, 4)
    )
    ff = replace(simple_ff, torsions=torsions)
    backend.prepare(PreparationRequest(case_id="fold-control", molecule=molecule, force_field=ff))
    backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    output = tmp_path / "molecule.prm"
    assert sum(line.startswith("torsion ") for line in output.read_text().splitlines()) == 1
    actual = load_tinker_prm(output)
    assert [(t.periodicity, t.force_constant, t.phase) for t in actual.torsions] == [
        (fold, fold * -0.25, fold * 15.0) for fold in range(1, 7)
    ]


@pytest.mark.parametrize("atom_type", ["", " ", "H1 H2", "#", "H1!comment", '"H1"', "H1\nvdw 5"])
@pytest.mark.parametrize("element", ["", "H"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_vdw_token_preflight(
    simple_ff: ForceField, source: Path, tmp_path: Path, atom_type: str, element: str, destination: str
) -> None:
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve public destination")
    before = output.read_bytes() if output.exists() else None
    ff = replace(simple_ff, vdws=(VdwParam(atom_type, 1.5, 0.02, element=element),))
    with pytest.raises(ValueError, match="vdW atom type"):
        save_tinker_prm(ff, output)
    _assert_unchanged(output, before)


@pytest.mark.parametrize("atom_type", ["0", "5", "H1"])
def test_public_vdw_token_controls(simple_ff: ForceField, tmp_path: Path, atom_type: str) -> None:
    ff = replace(simple_ff, vdws=(VdwParam(atom_type, 1.5, 0.02, reduction=0.923),))
    actual = load_tinker_prm(save_tinker_prm(ff, tmp_path / "output.prm"))
    assert (actual.vdws[0].atom_type, actual.vdws[0].radius, actual.vdws[0].epsilon, actual.vdws[0].reduction) == (
        atom_type,
        1.5,
        0.02,
        0.923,
    )


@pytest.mark.parametrize("family", ["bonds", "angles", "torsions", "vdws"])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_record_length_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    family: str,
    entry: str,
    destination: str,
) -> None:
    ff = replace(simple_ff, torsions=(TorsionParam(("H", "C", "C", "H"), force_constant=2.0),))
    field = "radius" if family == "vdws" else "force_constant"
    ff = replace(ff, **{family: (replace(getattr(ff, family)[0], **{field: 1e250}),)})
    _native_failure(backend, molecule, ff, tmp_path, entry, destination, "240-byte")


@pytest.mark.parametrize("family", ["bonds", "angles", "vdws"])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_duplicate_binding_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    family: str,
    entry: str,
    destination: str,
) -> None:
    original = getattr(simple_ff, family)[0]
    if family == "vdws":
        duplicate = replace(original, atom_type="5", element="H", epsilon=0.05)
    else:
        duplicate = replace(original, elements=original.elements[::-1], env_id="different-environment", equilibrium=2.0)
    ff = replace(simple_ff, **{family: (original, duplicate)})
    _native_failure(backend, molecule, ff, tmp_path, entry, destination, "duplicate native")


@pytest.mark.parametrize(
    "symbols,types,message",
    [
        (("C", "H"), ("1", "1"), "Inconsistent.*type"),
        (("H", "H"), ("5", "42"), "Inconsistent.*element"),
        (("C", "Xe"), ("1", "42"), "atomic data"),
    ],
)
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_atom_definition_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    tmp_path: Path,
    symbols: tuple[str, str],
    types: tuple[str, str],
    message: str,
    entry: str,
    destination: str,
) -> None:
    mol = replace(molecule, symbols=symbols, atom_types=types)
    _native_failure(backend, mol, ForceField(functional_form=FunctionalForm.MM3), tmp_path, entry, destination, message)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e250])
@pytest.mark.parametrize("entry", ["prepare", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_xyz_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    value: float,
    entry: str,
    destination: str,
) -> None:
    mol = replace(molecule, geometry=((value, 0.0, 0.0), (1.1, 0.0, 0.0)))
    _native_failure(backend, mol, simple_ff, tmp_path, entry, destination, "finite|240-byte")


@pytest.mark.parametrize("count", [3, 7])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_native_torsion_arity_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    count: int,
    entry: str,
    destination: str,
) -> None:
    torsion = replace(TorsionParam(("H", "C", "C", "H")), elements=(("C",) * 5 + ("H", "H"))[:count])
    _native_failure(
        backend, molecule, replace(simple_ff, torsions=(torsion,)), tmp_path, entry, destination, "torsion.*4 elements"
    )


@pytest.mark.parametrize("types", [[1], [1, 5, 42], [1, 1.5], [1, True]])
@pytest.mark.parametrize("existing", [False, True])
def test_native_direct_assignment_shape(
    backend: TinkerBackend, simple_ff: ForceField, tmp_path: Path, types: list[int], existing: bool
) -> None:
    output = tmp_path / "molecule.prm"
    if existing:
        output.write_bytes(b"preserve direct output")
    with pytest.raises(ValueError, match="positive|zip"):
        backend._write_standalone_prm(simple_ff, str(output), ["C", "H"], types)
    _assert_unchanged(output, b"preserve direct output" if existing else None)


@pytest.mark.parametrize("family", ["bonds", "angles"])
@pytest.mark.parametrize("entry", ["prepare", "standalone", "xyz"])
def test_native_missing_bonded_element_preflight(
    backend: TinkerBackend,
    molecule: Molecule,
    simple_ff: ForceField,
    tmp_path: Path,
    family: str,
    entry: str,
) -> None:
    param = getattr(simple_ff, family)[0]
    ff = replace(simple_ff, **{family: (replace(param, elements=("O", *param.elements[1:])),)})
    _native_failure(backend, molecule, ff, tmp_path, entry, "existing", "'O'.*not present")


@pytest.mark.parametrize(
    "family,converter", [("bonds", "canonical_to_mm3_bond_k"), ("angles", "canonical_to_mm3_angle_k")]
)
def test_public_raw_scalar_preflight(
    simple_ff: ForceField, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, family: str, converter: str
) -> None:
    from q2mm.io import tinker

    monkeypatch.setattr(tinker, converter, lambda _value: 1.0)
    ff = replace(simple_ff, **{family: (replace(getattr(simple_ff, family)[0], force_constant=float("inf")),)})
    output = tmp_path / "output.prm"
    with pytest.raises(ValueError, match="finite"):
        save_tinker_prm(ff, output)
    assert not output.exists()


@pytest.mark.parametrize("context", ["", "0000 0000"])
@pytest.mark.parametrize("entry", ["implicit-template", "explicit-template", "backend", "backend-fallback"])
@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("source_hint", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_template_generic_context_alias(
    backend: TinkerBackend,
    molecule: Molecule,
    source: Path,
    tmp_path: Path,
    context: str,
    entry: str,
    edited: bool,
    source_hint: bool,
    destination: str,
) -> None:
    output = tmp_path / "molecule.prm"
    expected = source.read_bytes()
    if destination == "source":
        output.write_bytes(expected)
        source = output
        backend._params_file = str(source)
    elif destination == "existing":
        output.write_bytes(b"preserve until export")
    ff = load_tinker_prm(source)
    bond = replace(
        ff.bonds[0],
        context=context,
        equilibrium=1.25 if edited else ff.bonds[0].equilibrium,
        ff_row=ff.bonds[0].ff_row if source_hint else None,
    )
    ff = replace(ff, bonds=(bond,))
    if entry == "explicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    elif entry == "backend-fallback":
        ff = replace(ff, source_path=None)
    before = replace(ff)
    fingerprint = ParameterLayout.from_force_field(ff).fingerprint
    if entry.startswith("backend"):
        session = backend.prepare(PreparationRequest(case_id="generic-template", molecule=molecule, force_field=ff))
        assert session.force_field is ff
        assert session.layout.fingerprint == fingerprint
        backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    else:
        save_tinker_prm(ff, output, template_path=source if entry == "explicit-template" else None)
    if edited:
        expected = expected.replace(b"5.0 1.1 !", b"5.0 1.25 !")
    assert output.read_bytes() == expected
    assert ff == before
    assert ff.bonds[0] is bond and bond.context == context
    assert ParameterLayout.from_force_field(ff).fingerprint == fingerprint
    actual = load_tinker_prm(output)
    assert actual.bonds[0].context == ""
    assert actual.bonds[0].equilibrium == bond.equilibrium
    assert actual.bonds[0].force_constant == bond.force_constant


@pytest.mark.parametrize("original_context", ["", "0000 0000"])
@pytest.mark.parametrize("updated_context", ["", "0000 0000"])
def test_template_generic_context_comparison_only(original_context: str, updated_context: str) -> None:
    original = BondParam(("C", "H"), 1.1, 100.0, context=original_context, env_id="C1-H1", ff_row=1)
    update = replace(original, context=updated_context, equilibrium=1.25)
    pairs = tinker._tinker_template_pairs((original,), (update,), lambda p: p.env_id, ("equilibrium", "force_constant"))
    assert pairs[0][0] is original
    assert pairs[0][1] is update
    assert (original.context, update.context) == (original_context, updated_context)


@pytest.mark.parametrize("context", [" ", "0000", "0000  0000", "0000\t0000", "O200 0000"])
@pytest.mark.parametrize("entry", ["implicit-template", "explicit-template", "backend", "backend-fallback", "prepare"])
def test_template_generic_context_rejects_other_spellings(
    backend: TinkerBackend, molecule: Molecule, source: Path, tmp_path: Path, context: str, entry: str
) -> None:
    ff = load_tinker_prm(source)
    ff = replace(ff, bonds=(replace(ff.bonds[0], context=context),))
    if entry == "explicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    elif entry == "backend-fallback":
        ff = replace(ff, source_path=None)
    paths = [tmp_path / f"molecule.{ext}" for ext in ("prm", "xyz", "key")]
    for path in paths:
        path.write_bytes(b"preserve output")
    if entry.startswith("backend") or entry == "prepare":
        with pytest.raises(PreparationError, match="bond context"):
            if entry == "prepare":
                backend.prepare(PreparationRequest(case_id="not-generic", molecule=molecule, force_field=ff))
            else:
                backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    else:
        with pytest.raises(ValueError, match="bond context"):
            save_tinker_prm(ff, paths[0], template_path=source if entry == "explicit-template" else None)
    for path in paths:
        _assert_unchanged(path, b"preserve output")


@pytest.mark.parametrize("field,value", [("elements", ("H", "C")), ("env_id", "5-1"), ("ff_row", 999)])
def test_template_generic_context_preserves_other_identity_checks(
    source: Path, tmp_path: Path, field: str, value: object
) -> None:
    ff = load_tinker_prm(source)
    ff = replace(ff, bonds=(replace(ff.bonds[0], context="0000 0000", **{field: value}),))
    output = tmp_path / "output.prm"
    with pytest.raises(ValueError, match="non-scalar|identity"):
        save_tinker_prm(ff, output)
    assert not output.exists()


@pytest.mark.parametrize("family,position", [("bonds", 0), ("bonds", 1), ("angles", 0), ("angles", 1), ("angles", 2)])
@pytest.mark.parametrize("type_source", ["environment", "element"])
@pytest.mark.parametrize(
    "bad_token",
    ["C extra", "C\textra", "C#comment", "C!comment", "C\nvdw 5", "C\rvdw 5", '"C"', "'C'"],
)
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_generated_atom_type_preflight(
    simple_ff: ForceField,
    source: Path,
    tmp_path: Path,
    family: str,
    position: int,
    type_source: str,
    bad_token: str,
    destination: str,
) -> None:
    parameter = getattr(simple_ff, family)[0]
    if type_source == "environment":
        atom_types = parameter.env_id.split("-")
        atom_types[position] = bad_token
        parameter = replace(parameter, env_id="-".join(atom_types))
    else:
        elements = list(parameter.elements)
        elements[position] = bad_token
        parameter = replace(parameter, env_id="", elements=tuple(elements))
    generated = tinker._tinker_atom_types(parameter.env_id, parameter.elements)
    assert bad_token in generated[position]
    ff = replace(simple_ff, **{family: (parameter,)})
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve public output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="Tinker.*atom type"):
        save_tinker_prm(ff, output)
    _assert_unchanged(output, before)


@pytest.mark.parametrize("family", ["bonds", "angles"])
@pytest.mark.parametrize("arity", ["short", "long"])
@pytest.mark.parametrize("type_source", ["environment", "element"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_generated_atom_type_arity(
    simple_ff: ForceField, source: Path, tmp_path: Path, family: str, arity: str, type_source: str, destination: str
) -> None:
    parameter = getattr(simple_ff, family)[0]
    elements = parameter.elements[:-1] if arity == "short" else (*parameter.elements, "H")
    parameter = replace(
        parameter,
        elements=elements,
        env_id="-".join(f"T{i}" for i in range(len(elements))) if type_source == "environment" else "",
    )
    ff = replace(simple_ff, **{family: (parameter,)})
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve public output")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="Tinker.*atom types"):
        save_tinker_prm(ff, output)
    _assert_unchanged(output, before)


@pytest.mark.parametrize("family", ["bonds", "angles"])
@pytest.mark.parametrize("type_source", ["explicit", "default", "partial-fallback", "empty-separator", "wildcard"])
def test_public_generated_atom_type_controls(
    simple_ff: ForceField, tmp_path: Path, family: str, type_source: str
) -> None:
    parameter = getattr(simple_ff, family)[0]
    expected = ["C1", "H1"] if family == "bonds" else ["H1", "C1", "H2"]
    if type_source == "explicit":
        expected = ["7", "42"] if family == "bonds" else ["42", "7", "42"]
        env_id = " - ".join(expected)
    elif type_source == "wildcard":
        expected = ["0", "5"] if family == "bonds" else ["0", "1", "5"]
        env_id = "-".join(expected)
    elif type_source == "empty-separator":
        env_id = "--".join(expected)
    else:
        env_id = "partial\nunused" if type_source == "partial-fallback" else ""
    ff = replace(simple_ff, **{family: (replace(parameter, env_id=env_id),)})
    output = save_tinker_prm(ff, tmp_path / "output.prm")
    record = "bond" if family == "bonds" else "angle"
    line = next(line for line in output.read_text().splitlines() if line.startswith(record + " "))
    assert line.split()[1 : 1 + len(expected)] == expected
    actual = getattr(load_tinker_prm(output), family)[0]
    assert actual.force_constant == pytest.approx(parameter.force_constant, abs=0.01)
    assert actual.equilibrium == parameter.equilibrium


def _with_public_types(
    parameter: BondParam | AngleParam | VdwParam, atom_types: tuple[str, ...]
) -> BondParam | AngleParam | VdwParam:
    if isinstance(parameter, VdwParam):
        return replace(parameter, atom_type=atom_types[0])
    return replace(parameter, env_id="-".join(atom_types))


@pytest.mark.parametrize("family", ["bonds", "angles", "vdws"])
@pytest.mark.parametrize(
    "alias", ["same", "reversed", "padded", "signed", "underscored", "unicode", "zero", "symbolic"]
)
@pytest.mark.parametrize("variation", ["identical", "metadata-and-values", "zero-values"])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_duplicate_native_identity(
    simple_ff: ForceField,
    source: Path,
    tmp_path: Path,
    family: str,
    alias: str,
    variation: str,
    destination: str,
) -> None:
    types = {"bonds": ("1", "5"), "angles": ("5", "1", "6"), "vdws": ("5",)}[family]
    if alias == "zero":
        types = ("0", *types[1:])
    elif alias == "symbolic":
        types = {"bonds": ("C1", "H1"), "angles": ("H1", "C1", "H2"), "vdws": ("H1",)}[family]
    other_types = types
    if alias == "reversed":
        other_types = types[::-1]
    elif alias in ("padded", "zero"):
        other_types = tuple(f"00{t}" for t in types)
    elif alias == "signed":
        other_types = tuple(f"+{t}" for t in types)
    elif alias == "underscored":
        other_types = tuple(f"0_{t}" for t in types)
    elif alias == "unicode":
        other_types = tuple(t.translate(str.maketrans("156", "\u0661\u0665\u0666")) for t in types)
    original = _with_public_types(getattr(simple_ff, family)[0], types)
    if variation == "zero-values":
        original = replace(original, **{"epsilon" if family == "vdws" else "force_constant": 0.0})
    duplicate = _with_public_types(original, other_types)
    if variation == "metadata-and-values":
        duplicate = replace(
            duplicate,
            label="different canonical record",
            ff_row=999,
            **{"epsilon" if family == "vdws" else "force_constant": 0.25},
        )
        if isinstance(duplicate, BondParam):
            duplicate = replace(duplicate, context="0000 0000")
    ff = replace(simple_ff, **{family: (original, duplicate)})
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve public duplicate destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="duplicate native"):
        save_tinker_prm(ff, output)
    _assert_unchanged(output, before)
    assert getattr(ff, family) == (original, duplicate)


@pytest.mark.parametrize("family", ["bonds", "angles"])
@pytest.mark.parametrize("reversed_types", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_public_duplicate_generated_identity(
    simple_ff: ForceField, source: Path, tmp_path: Path, family: str, reversed_types: bool, destination: str
) -> None:
    original = replace(getattr(simple_ff, family)[0], env_id="")
    types = tinker._tinker_atom_types(original.env_id, original.elements)
    duplicate = replace(
        original,
        env_id="-".join(types[::-1]) if reversed_types else "incomplete",
        label="same emitted types from different inference",
        ff_row=123,
    )
    assert tinker._tinker_atom_types(duplicate.env_id, duplicate.elements) == (types[::-1] if reversed_types else types)
    ff = replace(simple_ff, **{family: (original, duplicate)})
    output = source if destination == "source" else tmp_path / "output.prm"
    if destination == "existing":
        output.write_bytes(b"preserve generated duplicate destination")
    before = output.read_bytes() if output.exists() else None
    with pytest.raises(ValueError, match="duplicate native"):
        save_tinker_prm(ff, output)
    _assert_unchanged(output, before)


@pytest.mark.parametrize(
    "family,first_types,second_types",
    [
        ("bonds", ("1", "5"), ("1", "42")),
        ("bonds", ("C1", "H1"), ("c1", "H1")),
        ("bonds", ("005", "+1"), ("5", "42")),
        ("angles", ("5", "1", "6"), ("5", "42", "6")),
        ("angles", ("5", "1", "6"), ("1", "5", "6")),
        ("angles", ("H1", "C1", "H2"), ("h1", "C1", "H2")),
        ("angles", ("005", "+1", "006"), ("5", "1", "42")),
        ("vdws", ("5",), ("42",)),
        ("vdws", ("H1",), ("h1",)),
        ("vdws", ("005",), ("+42",)),
    ],
)
def test_public_unique_native_identity_preserves_tokens(
    simple_ff: ForceField, tmp_path: Path, family: str, first_types: tuple[str, ...], second_types: tuple[str, ...]
) -> None:
    parameter = getattr(simple_ff, family)[0]
    ff = replace(
        simple_ff, **{family: (_with_public_types(parameter, first_types), _with_public_types(parameter, second_types))}
    )
    output = save_tinker_prm(ff, tmp_path / "unique.prm")
    keyword = {"bonds": "bond", "angles": "angle", "vdws": "vdw"}[family]
    rows = [line.split() for line in output.read_text().splitlines() if line.startswith(keyword + " ")]
    assert [tuple(row[1 : 1 + len(first_types)]) for row in rows] == [first_types, second_types]
    actual = getattr(load_tinker_prm(output), family)
    assert len(actual) == 2
    for parameter in actual:
        if isinstance(parameter, VdwParam):
            assert (parameter.radius, parameter.epsilon) == (1.5, 0.02)
        else:
            assert parameter.force_constant == pytest.approx(getattr(ff, family)[0].force_constant, abs=0.01)


@pytest.mark.parametrize("alias", ["-05", "-0_5", "-\u0665"])
def test_public_negative_vdw_alias_is_duplicate(simple_ff: ForceField, tmp_path: Path, alias: str) -> None:
    ff = replace(simple_ff, vdws=(VdwParam("-5", 1.5, 0.02), VdwParam(alias, 1.5, 0.02)))
    output = tmp_path / "negative-alias.prm"
    with pytest.raises(ValueError, match="duplicate native vdW"):
        save_tinker_prm(ff, output)
    assert not output.exists()


@pytest.mark.parametrize("entry", ["implicit-template", "explicit-template", "backend", "backend-fallback"])
@pytest.mark.parametrize("family", ["bonds", "angles", "vdws"])
@pytest.mark.parametrize("edited", [False, True])
@pytest.mark.parametrize("destination", ["absent", "existing", "source"])
def test_templates_preserve_ordered_duplicate_native_records(
    backend: TinkerBackend,
    molecule: Molecule,
    source: Path,
    tmp_path: Path,
    entry: str,
    family: str,
    edited: bool,
    destination: str,
) -> None:
    extra, modified = {
        "bonds": (b"bond 1 5 6.0 1.2 ! second bond\r\n", b"bond 1 5 6.0 1.25 ! second bond\r\n"),
        "angles": (b"angle 5 1 5 0.75 108.0 ! second angle\r\n", b"angle 5 1 5 0.75 107.5 ! second angle\r\n"),
        "vdws": (b"vdw 5 1.75 0.04 ! second vdw\r\n", b"vdw 5 1.75 0.05 ! second vdw\r\n"),
    }[family]
    source.write_bytes(source.read_bytes() + extra)
    expected = source.read_bytes()
    output = tmp_path / "molecule.prm"
    if destination == "source":
        output.write_bytes(expected)
        source = output
        backend._params_file = str(source)
    elif destination == "existing":
        output.write_bytes(b"replace on successful template export")
    ff = load_tinker_prm(source)
    if edited:
        parameters = getattr(ff, family)
        changes = {"epsilon": 0.05} if family == "vdws" else {"equilibrium": 1.25 if family == "bonds" else 107.5}
        ff = replace(ff, **{family: (*parameters[:-1], replace(parameters[-1], **changes))})
        expected = expected.replace(extra, modified)
    if entry == "explicit-template":
        ff = replace(ff, source_path=None, source_format=None)
    elif entry == "backend-fallback":
        ff = replace(ff, source_path=None)
    if entry.startswith("backend"):
        backend.prepare(PreparationRequest(case_id="ordered-native-rows", molecule=molecule, force_field=ff))
        backend._write_tinker_xyz(molecule, ff, str(tmp_path))
    else:
        save_tinker_prm(ff, output, template_path=source if entry == "explicit-template" else None)
    assert output.read_bytes() == expected
    assert getattr(load_tinker_prm(output), family) == getattr(ff, family)
