"""Known native wildcard tokens must not disappear in JAX/OpenMM preparation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from q2mm.backends.contracts import PreparationError, PreparationRequest
from q2mm.io.amber import load_amber_frcmod
from q2mm.io.mm3 import load_mm3_fld, save_mm3_fld
from q2mm.io.tinker import load_tinker_prm
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm, TorsionParam
from q2mm.models.molecule import Bond, Molecule
from q2mm.models.parameters import ParameterLayout
from test.test_backend_term_coverage import _energy, _native_backend, _unit_backend

_TARGETS = [(key, form) for key in ("jax", "openmm") for form in FunctionalForm]
_NATIVE_TARGETS = [pytest.param(key, marks=getattr(pytest.mark, key)) for key in ("jax", "openmm")]


def _molecule(types: tuple[str, str, str, str] = ("h1", "c3", "c3", "h1")) -> Molecule:
    return Molecule(
        symbols=("H", "C", "C", "H"),
        atom_types=types,
        geometry=np.array([[0, 1, 0], [0, 0, 0], [1.5, 0, 0], [1.5, 0.5, np.sqrt(0.75)]]),
        bonds=(Bond(0, 1, ("H", "C"), 1.0), Bond(1, 2, ("C", "C"), 1.5), Bond(2, 3, ("C", "H"), 1.0)),
    )


def _base_ff(form: FunctionalForm) -> ForceField:
    return ForceField(
        functional_form=form,
        bonds=(BondParam(("H", "C"), 1.0, 50.0), BondParam(("C", "C"), 1.4, 50.0)),
    )


@pytest.mark.parametrize("key,form", _TARGETS)
@pytest.mark.parametrize("token", ["X", "00", "0"])
@pytest.mark.parametrize(
    "typed,improper,k",
    [(True, False, 2.0), (True, True, 0.0), (False, False, 0.0), (False, True, 2.0)],
    ids=["typed-proper", "typed-zero-improper", "element-zero-proper", "element-improper"],
)
def test_wildcards_rejected_before_layout_and_native_state(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    form: FunctionalForm,
    token: str,
    typed: bool,
    improper: bool,
    k: float,
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    tor = TorsionParam(
        ("H", "C", "C", "H") if typed else (token, "C", "C", token),
        force_constant=k,
        env_id=f" {token} - c3 - c3 - {token} " if typed else "",
        is_improper=improper,
        ff_row=42,
    )
    ff = replace(_base_ff(form), torsions=(tor,))
    with (
        patch.object(ParameterLayout, "from_force_field", side_effect=AssertionError("layout constructed")) as layout,
        pytest.raises(PreparationError, match="wildcard torsions"),
    ):
        backend.prepare(PreparationRequest(case_id="wildcard", molecule=_molecule(), force_field=ff))
    layout.assert_not_called()
    builder.assert_not_called()


@pytest.mark.parametrize("key,form", _TARGETS)
@pytest.mark.parametrize("token", ["000", "0000", "+0", "-0"])
def test_native_integer_zero_spellings_are_not_exact_atom_classes(
    monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm, token: str
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    tor = TorsionParam((token, "C", "C", token), force_constant=2.0)
    ff = replace(_base_ff(form), torsions=(tor,))
    with pytest.raises(PreparationError, match="wildcard torsions"):
        backend.prepare(PreparationRequest(case_id="zero-class", molecule=_molecule(), force_field=ff))
    builder.assert_not_called()


@pytest.mark.parametrize("key,form", _TARGETS)
@pytest.mark.parametrize("token", ["X1", "Xe", "CX", "00A", "100", "x", "*", "c*", "unknown"])
def test_explicit_nonwild_types_are_not_reclassified_by_inferred_elements(
    monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm, token: str
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    # An inferred X element does not turn an authoritative X1/Xe/... type
    # into the literal native X wildcard.
    tor = TorsionParam(("X", "C", "C", "X"), force_constant=2.0, env_id=f"{token}-c3-c3-{token}")
    ff = replace(_base_ff(form), torsions=(tor,))
    backend.prepare(PreparationRequest(case_id="ordinary-types", molecule=_molecule(), force_field=ff))
    builder.assert_called_once()


@pytest.mark.parametrize("key,form", _TARGETS)
def test_generic_elements_and_descriptive_metadata_are_preserved(
    monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    tor = TorsionParam(("H", "C", "C", "H"), force_constant=2.0, label="X 00 0 * are only comment text", ff_row=0)
    ff = replace(
        _base_ff(form),
        name="X-00-0-X",
        source_format="amber_frcmod",
        source_path=Path("X-00-0-X.frcmod"),
        torsions=(tor,),
    )
    mol = replace(_molecule(), partial_charges=(0.1, -0.1, -0.1, 0.1))
    backend.prepare(PreparationRequest(case_id="metadata", molecule=mol, force_field=ff))
    builder.assert_called_once_with(mol, ff)


def _loaded_wildcard(kind: str, tmp_path: Path) -> tuple[ForceField, Molecule, str]:
    if kind == "amber":
        path = tmp_path / "wildcard.frcmod"
        path.write_text("Synthetic\nDIHE\nX -c3-c3-X    1 2.0 0.0 1.0\n\n", encoding="ascii")
        ff = load_amber_frcmod(path)
        mol, token = _molecule(), "X"
    elif kind == "mm3":
        path = tmp_path / "wildcard.fld"
        save_mm3_fld(
            ForceField(
                functional_form=FunctionalForm.MM3,
                torsions=(TorsionParam(("00", "C", "C", "00"), force_constant=2.0, env_id="00-C3-C3-00"),),
            ),
            path,
        )
        ff = load_mm3_fld(path)
        mol, token = _molecule(("H1", "C3", "C3", "H1")), "00"
    elif kind == "tinker":
        path = tmp_path / "wildcard.prm"
        path.write_text(
            'atom 1 C "carbon" 6 12.0 4\natom 2 H "hydrogen" 1 1.0 1\n'
            "# Q2MM\n# OPT Synthetic\ntorsion 0 1 1 0 2.0 0 1 0.0 180 2 0.0 0 3\n",
            encoding="ascii",
        )
        ff = load_tinker_prm(path)
        mol, token = _molecule(("2", "1", "1", "2")), "0"
    else:
        raise AssertionError(f"Unknown fixture format {kind}")
    assert ff.torsions and ff.torsions[0].force_constant == 2.0
    assert ff.torsions[0].env_id.split("-")[0] == token
    assert ff.torsions[0].elements[0] == token
    return replace(ff, bonds=_base_ff(ff.functional_form).bonds), mol, token


@pytest.mark.parametrize("key", _NATIVE_TARGETS)
@pytest.mark.parametrize("kind", ["amber", "mm3", "tinker"])
def test_native_partial_omission_rejects_imported_wildcards(tmp_path: Path, key: str, kind: str) -> None:
    ff, mol, token = _loaded_wildcard(kind, tmp_path)
    backend = _native_backend(key)
    baseline = _energy(backend, mol, replace(ff, torsions=()))
    assert np.isfinite(baseline) and baseline > 0.0
    exact = replace(
        ff,
        torsions=tuple(replace(t, elements=("H", "C", "C", "H"), env_id="-".join(mol.atom_types)) for t in ff.torsions),
    )
    bound = exact.match_torsion(("H", "C", "C", "H"), env_id="-".join(mol.atom_types), is_improper=False)
    assert bound == list(exact.torsions)
    control = _energy(backend, mol, exact)
    assert control - baseline == pytest.approx(3.0, abs=1e-10)
    with pytest.raises(PreparationError, match="wildcard torsions"):
        observed = _energy(backend, mol, ff)
        pytest.fail(
            f"{key} accepted {kind} wildcard {token!r}: energy={observed}, "
            f"non-torsion baseline={baseline}, exact control={control}"
        )


@pytest.mark.parametrize("key", _NATIVE_TARGETS)
@pytest.mark.parametrize("token", ["X1", "Xe", "100", "x", "*", "c*"])
def test_native_exact_types_still_bind_without_glob_interpretation(key: str, token: str) -> None:
    mol = _molecule((token, "c3", "c3", token))
    ff = _base_ff(FunctionalForm.HARMONIC)
    tor = TorsionParam(("H", "C", "C", "H"), force_constant=2.0, env_id="-".join(mol.atom_types))
    backend = _native_backend(key)
    baseline = _energy(backend, mol, ff)
    ff = replace(ff, torsions=(tor,))
    assert ff.match_torsion(tor.elements, env_id=tor.env_id, is_improper=False) == [tor]
    assert _energy(backend, mol, ff) - baseline == pytest.approx(3.0, abs=1e-10)


@pytest.mark.parametrize("key", _NATIVE_TARGETS)
def test_imported_unknown_types_do_not_become_wildcards_from_element_inference(tmp_path: Path, key: str) -> None:
    path = tmp_path / "ordinary.frcmod"
    path.write_text(
        "Comments about X, 00, 0 and *\nMASS\nX1 1.008\nc3 12.011\n\nDIHE\nX1-c3-c3-X1 1 2.0 0.0 1.0\n\n",
        encoding="ascii",
    )
    ff = load_amber_frcmod(path)
    assert ff.torsions[0].elements == ("H", "C", "C", "H")
    assert ff.torsions[0].env_id == "X1-c3-c3-X1"
    ff = replace(ff, bonds=_base_ff(ff.functional_form).bonds)
    mol = replace(_molecule(("X1", "c3", "c3", "X1")), partial_charges=(0.1, -0.1, -0.1, 0.1))
    backend = _native_backend(key)
    baseline = _energy(backend, mol, replace(ff, torsions=()))
    assert _energy(backend, mol, ff) - baseline == pytest.approx(3.0, abs=1e-10)
