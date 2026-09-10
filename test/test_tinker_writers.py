"""CPU-only writer coverage; no Tinker executable is invoked."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.backends.contracts import PreparationError, PreparationRequest
from q2mm.backends.mm.tinker import TinkerBackend
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

_LOSSES = [
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


def test_native_standalone_preserves_its_supported_model(
    backend: TinkerBackend, molecule: Molecule, simple_ff: ForceField, tmp_path: Path
) -> None:
    ff = replace(
        simple_ff,
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
