"""Preparation loss gates, with separate dependency-light and native evidence."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from q2mm.backends.contracts import Backend, EnergyRequest, PreparationError, PreparationRequest
from q2mm.backends.mm.jax_engine import JaxBackend
from q2mm.backends.mm.jax_md_engine import JaxMdBackend
from q2mm.backends.mm.openmm import OpenMMBackend
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
from q2mm.models.molecule import Bond, Molecule
from q2mm.models.parameters import ParameterLayout
from test.backend_fixtures import load_test_backend, mock_backend_info

_BACKENDS = {"jax": JaxBackend, "openmm": OpenMMBackend, "jax-md": JaxMdBackend}
_FORMS = [
    ("jax", FunctionalForm.HARMONIC),
    ("jax", FunctionalForm.MM3),
    ("openmm", FunctionalForm.HARMONIC),
    ("openmm", FunctionalForm.MM3),
    ("jax-md", FunctionalForm.HARMONIC),
]
_REJECTIONS = [
    ("jax", FunctionalForm.HARMONIC, "CMAP"),
    ("jax", FunctionalForm.MM3, "CMAP"),
    ("jax", FunctionalForm.HARMONIC, "stretch-bend"),
    ("jax", FunctionalForm.HARMONIC, "vdW reduction"),
    ("jax", FunctionalForm.MM3, "vdW reduction"),
    ("openmm", FunctionalForm.HARMONIC, "bond dipoles"),
    ("openmm", FunctionalForm.MM3, "bond dipoles"),
    ("openmm", FunctionalForm.HARMONIC, "vdW reduction"),
    ("openmm", FunctionalForm.MM3, "vdW reduction"),
    ("jax-md", FunctionalForm.HARMONIC, "Urey-Bradley"),
    ("jax-md", FunctionalForm.HARMONIC, "CMAP"),
    ("jax-md", FunctionalForm.HARMONIC, "improper torsions"),
]
_SUPPORTED = [
    ("jax", FunctionalForm.MM3, "stretch-bend"),
    ("jax", FunctionalForm.MM3, "bond dipoles"),
    ("jax", FunctionalForm.HARMONIC, "Urey-Bradley"),
    ("jax", FunctionalForm.MM3, "Urey-Bradley"),
    ("jax", FunctionalForm.HARMONIC, "improper torsions"),
    ("jax", FunctionalForm.MM3, "improper torsions"),
    ("openmm", FunctionalForm.HARMONIC, "Urey-Bradley"),
    ("openmm", FunctionalForm.MM3, "Urey-Bradley"),
    ("openmm", FunctionalForm.HARMONIC, "CMAP"),
    ("openmm", FunctionalForm.HARMONIC, "improper torsions"),
    ("openmm", FunctionalForm.MM3, "improper torsions"),
]


@pytest.fixture
def molecule() -> Molecule:
    coords = np.array([[0.0, 1.5, 0.0], [0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 0.75, 1.299], [3.0, 0.75, 1.299]])
    return Molecule(
        symbols=("C",) * 5,
        atom_types=("A", "B", "C", "D", "E"),
        geometry=coords,
        bonds=tuple(Bond(i, i + 1, ("C", "C"), float(np.linalg.norm(coords[i + 1] - coords[i]))) for i in range(4)),
    )


def _base_ff(form: FunctionalForm) -> ForceField:
    return ForceField(
        functional_form=form,
        bonds=(BondParam(("C", "C"), 1.4, 10.0),),
        angles=(AngleParam(("C", "C", "C"), 100.0, 5.0),),
        vdws=(VdwParam("C", 1.0, 0.01),),
    )


def _with_term(ff: ForceField, term: str, edge: bool = False) -> ForceField:
    if term == "CMAP":
        return replace(
            ff, cmaps=(CmapGrid(("A", "B", "C", "D"), ("B", "C", "D", "E"), 2, (0.0 if edge else 5.0,) * 4),)
        )
    if term == "stretch-bend":
        return replace(ff, stretch_bends=(StretchBendParam(("C", "C", "C"), 0.0 if edge else 2.0),))
    if term == "Urey-Bradley":
        return replace(ff, angles=(replace(ff.angles[0], ub_force_constant=0.0 if edge else 3.0, ub_equilibrium=2.0),))
    if term == "improper torsions":
        return replace(ff, torsions=(TorsionParam(("C",) * 4, force_constant=0.0 if edge else 2.0, is_improper=True),))
    if term == "bond dipoles":
        return replace(ff, bonds=(replace(ff.bonds[0], dipole_moment=-0.4 if edge else 0.4),))
    if term == "vdW reduction":
        return replace(ff, vdws=(replace(ff.vdws[0], reduction=1.0 if edge else 0.5, epsilon=0.0 if edge else 0.01),))
    raise AssertionError(f"Unknown test term {term}")


def _unit_backend(key: str, monkeypatch: pytest.MonkeyPatch) -> tuple[Backend, Mock]:
    """Exercise real prepare methods, replacing only runtime metadata/state construction."""
    cls = _BACKENDS[key]
    backend = object.__new__(cls)
    forms = ("harmonic",) if key == "jax-md" else ("harmonic", "mm3")
    monkeypatch.setattr(cls, "info", property(lambda self: mock_backend_info(forms=forms)))
    builder = Mock(return_value=object())
    monkeypatch.setattr(backend, "_build_state", builder)
    return backend, builder


@pytest.mark.parametrize("key,form,term", _REJECTIONS)
@pytest.mark.parametrize("edge", [False, True], ids=["nonzero", "zero-or-nondefault-edge"])
def test_preparation_rejects_before_layout_or_native_state(
    molecule: Molecule, monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm, term: str, edge: bool
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    ff = _with_term(_base_ff(form), term, edge)
    with (
        patch.object(ParameterLayout, "from_force_field", side_effect=AssertionError("layout constructed")) as layout,
        pytest.raises(PreparationError, match=term),
    ):
        backend.prepare(PreparationRequest(case_id="partial-omission", molecule=molecule, force_field=ff))
    layout.assert_not_called()
    builder.assert_not_called()


@pytest.mark.parametrize("field", ["ub_force_constant", "ub_equilibrium"])
@pytest.mark.parametrize("value", [0.0, 2.0])
def test_jax_md_rejects_individually_populated_ub_fields(
    molecule: Molecule, monkeypatch: pytest.MonkeyPatch, field: str, value: float
) -> None:
    backend, builder = _unit_backend("jax-md", monkeypatch)
    ff = _base_ff(FunctionalForm.HARMONIC)
    ff = replace(ff, angles=(replace(ff.angles[0], **{field: value}),))
    with pytest.raises(PreparationError, match="Urey-Bradley"):
        backend.prepare(PreparationRequest(case_id="partial-ub", molecule=molecule, force_field=ff))
    builder.assert_not_called()


@pytest.mark.parametrize("key,form,term", _SUPPORTED)
def test_preparation_keeps_supported_terms(
    molecule: Molecule, monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm, term: str
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    ff = _with_term(_base_ff(form), term)
    session = backend.prepare(PreparationRequest(case_id="supported", molecule=molecule, force_field=ff))
    assert session.case_id == "supported"
    builder.assert_called_once_with(molecule, ff)


@pytest.mark.parametrize("key,form", _FORMS)
def test_reference_charges_and_source_metadata_do_not_request_energy_terms(
    molecule: Molecule, monkeypatch: pytest.MonkeyPatch, key: str, form: FunctionalForm
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    molecule = replace(molecule, partial_charges=(0.2, -0.2, 0.0, -0.1, 0.1))
    ff = replace(
        _base_ff(form),
        name="CMAP, dipoles, and reduction in a descriptive label",
        source_format="tinker_prm",
        source_path=Path("opaque-template.prm"),
    )
    backend.prepare(PreparationRequest(case_id="metadata", molecule=molecule, force_field=ff))
    builder.assert_called_once_with(molecule, ff)


@pytest.mark.parametrize("key", ["jax", "openmm", "jax-md"])
def test_missing_force_field_still_has_typed_preparation_error(
    molecule: Molecule, monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    backend, builder = _unit_backend(key, monkeypatch)
    with pytest.raises(PreparationError, match="base ForceField"):
        backend.prepare(PreparationRequest(case_id="missing", molecule=molecule))
    builder.assert_not_called()


def test_jax_md_still_rejects_mm3(molecule: Molecule, monkeypatch: pytest.MonkeyPatch) -> None:
    backend, builder = _unit_backend("jax-md", monkeypatch)
    with pytest.raises(PreparationError, match="functional form"):
        backend.prepare(PreparationRequest(case_id="form", molecule=molecule, force_field=_base_ff(FunctionalForm.MM3)))
    builder.assert_not_called()


def _native_backend(key: str) -> Backend:
    if key == "openmm":
        return load_test_backend(key, platform_name="CPU")
    if key == "jax-md":
        return load_test_backend(key, box=(50.0, 50.0, 50.0))
    return load_test_backend(key)


def _energy(backend: Backend, molecule: Molecule, ff: ForceField) -> float:
    session = backend.prepare(PreparationRequest(case_id="native", molecule=molecule, force_field=ff))
    return session.energy(EnergyRequest(parameters=ParameterLayout.from_force_field(ff).vector(ff))).energy


@pytest.mark.parametrize(
    "key,form,term",
    [pytest.param(*case, marks=getattr(pytest.mark, case[0].replace("-", "_"))) for case in _REJECTIONS],
)
def test_native_partial_omission_is_rejected(molecule: Molecule, key: str, form: FunctionalForm, term: str) -> None:
    backend = _native_backend(key)
    ff = _base_ff(form)
    baseline = _energy(backend, molecule, ff)
    assert np.isfinite(baseline) and abs(baseline) > 1e-6
    with pytest.raises(PreparationError, match=term):
        _energy(backend, molecule, _with_term(ff, term))


@pytest.mark.parametrize(
    "key,form,term",
    [
        pytest.param(*case, marks=getattr(pytest.mark, case[0].replace("-", "_")))
        for case in _SUPPORTED
        if case[2] != "improper torsions"
    ],
)
def test_native_supported_terms_remain_active(molecule: Molecule, key: str, form: FunctionalForm, term: str) -> None:
    backend = _native_backend(key)
    ff = _base_ff(form)
    baseline = _energy(backend, molecule, ff)
    populated = _energy(backend, molecule, _with_term(ff, term))
    assert np.isfinite(populated)
    assert abs(populated - baseline) > 1e-6
    if term == "CMAP":
        assert populated - baseline == pytest.approx(5.0, abs=1e-6)


@pytest.mark.parametrize(
    "key,form",
    [pytest.param(*case, marks=getattr(pytest.mark, case[0].replace("-", "_"))) for case in _FORMS],
)
def test_native_reference_charges_do_not_change_energy(molecule: Molecule, key: str, form: FunctionalForm) -> None:
    backend = _native_backend(key)
    ff = _base_ff(form)
    charged_reference = replace(molecule, partial_charges=(0.2, -0.2, 0.0, -0.1, 0.1))
    assert _energy(backend, charged_reference, ff) == pytest.approx(_energy(backend, molecule, ff), abs=1e-10)
