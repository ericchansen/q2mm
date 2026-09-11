"""Source-scoped subset rows must not select unrelated or ambiguous entries."""

from dataclasses import replace
from pathlib import Path

import pytest

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    FunctionalForm,
    StretchBendParam,
    TorsionParam,
    VdwParam,
)
from q2mm.models.parameters import ParameterLayout, opt_substructure_membership

_FAMILIES = ["bonds", "angles", "stretch_bends", "torsions", "vdws", "urey_bradley"]


def _parameter(family: str) -> BondParam | AngleParam | StretchBendParam | TorsionParam | VdwParam:
    if family == "bonds":
        return BondParam(("C", "H"), 1.1, 100.0, ff_row=40)
    if family in ("angles", "urey_bradley"):
        return AngleParam(("H", "C", "H"), 109.5, 30.0, ff_row=40, ub_force_constant=2.0, ub_equilibrium=1.8)
    if family == "stretch_bends":
        return StretchBendParam(("H", "C", "H"), 1.0, ff_row=40)
    if family == "torsions":
        return TorsionParam(("H", "C", "C", "H"), force_constant=1.0, ff_row=40)
    return VdwParam("C", 1.5, 0.1, ff_row=40)


@pytest.mark.parametrize("family", _FAMILIES)
def test_source_row_match_consumes_the_subset_entry_once(tmp_path: Path, family: str) -> None:
    parameter = _parameter(family)
    attr = "angles" if family == "urey_bradley" else family
    source = tmp_path / "same.fld"
    full = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=source,
        **{attr: (parameter, replace(parameter, ff_row=None))},
    )
    subset = ForceField(functional_form=FunctionalForm.MM3, source_path=source, **{attr: (parameter,)})
    before = ParameterLayout.from_force_field(full).fingerprint

    membership = opt_substructure_membership(full, subset)

    assert getattr(membership, family) == frozenset({0})
    assert ParameterLayout.from_force_field(full).fingerprint == before


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("known_sources", [False, True])
def test_partial_ambiguous_fallback_is_not_occurrence_order_selection(family: str, known_sources: bool) -> None:
    parameter = _parameter(family)
    attr = "angles" if family == "urey_bradley" else family
    full = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=Path("full.fld") if known_sources else None,
        **{attr: (parameter, replace(parameter, ff_row=80))},
    )
    subset = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=Path("other.fld") if known_sources else None,
        **{attr: (parameter,)},
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        opt_substructure_membership(full, subset)


@pytest.mark.parametrize("family", _FAMILIES)
def test_empty_and_complete_duplicate_groups_remain_unambiguous(family: str) -> None:
    parameter = _parameter(family)
    attr = "angles" if family == "urey_bradley" else family
    full = ForceField(functional_form=FunctionalForm.MM3, **{attr: (parameter, replace(parameter, ff_row=80))})

    assert getattr(opt_substructure_membership(full, full), family) == frozenset({0, 1})
    empty = ForceField(functional_form=FunctionalForm.MM3)
    assert getattr(opt_substructure_membership(full, empty), family) == frozenset()


@pytest.mark.parametrize("periodicity", [1, 2, 3])
@pytest.mark.parametrize("reverse", [False, True])
def test_source_row_torsion_components_are_distinguished(periodicity: int, reverse: bool) -> None:
    source = Path("same.fld")
    terms = tuple(TorsionParam(("H", "C", "C", "H"), periodicity=p, ff_row=40) for p in (1, 2, 3))
    if reverse:
        terms = terms[::-1]
    full = ForceField(functional_form=FunctionalForm.MM3, source_path=source, torsions=terms)
    selected = next(term for term in terms if term.periodicity == periodicity)
    subset = ForceField(functional_form=FunctionalForm.MM3, source_path=source, torsions=(selected,))

    assert opt_substructure_membership(full, subset).torsions == frozenset({terms.index(selected)})


def test_source_row_does_not_conflate_proper_and_improper_components() -> None:
    proper = TorsionParam(("H", "C", "C", "H"), ff_row=40)
    improper = replace(proper, is_improper=True)
    full = ForceField(functional_form=FunctionalForm.MM3, source_path=Path("same.fld"), torsions=(proper, improper))
    subset = replace(full, torsions=(improper,))
    assert opt_substructure_membership(full, subset).torsions == frozenset({1})


@pytest.mark.parametrize("family", _FAMILIES)
def test_explicit_rowless_subset_entry_is_not_consumed_by_a_different_row(family: str) -> None:
    parameter = _parameter(family)
    attr = "angles" if family == "urey_bradley" else family
    full = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=Path("same.fld"),
        **{attr: (parameter, replace(parameter, ff_row=None))},
    )
    assert getattr(opt_substructure_membership(full, full), family) == frozenset({0, 1})


@pytest.mark.parametrize("family", _FAMILIES)
def test_partial_duplicate_source_rows_are_explicitly_ambiguous(family: str) -> None:
    parameter = _parameter(family)
    attr = "angles" if family == "urey_bradley" else family
    full = ForceField(
        functional_form=FunctionalForm.MM3, source_path=Path("same.fld"), **{attr: (parameter, parameter)}
    )
    subset = ForceField(functional_form=FunctionalForm.MM3, source_path=Path("same.fld"), **{attr: (parameter,)})
    with pytest.raises(ValueError, match="Ambiguous"):
        opt_substructure_membership(full, subset)


def test_urey_bradley_ambiguity_is_checked_independently_of_bending() -> None:
    angle = AngleParam(("H", "C", "H"), 109.5, 30.0, ub_force_constant=2.0, ub_equilibrium=1.8)
    full = ForceField(functional_form=FunctionalForm.HARMONIC, angles=(angle, angle))
    subset = replace(full, angles=(angle, replace(angle, ub_force_constant=None, ub_equilibrium=None)))
    with pytest.raises(ValueError, match="Ambiguous Urey-Bradley"):
        opt_substructure_membership(full, subset)


def test_changed_source_identity_does_not_leave_fallback_credit() -> None:
    generic = BondParam(("C", "C"), 1.5, 100.0, ff_row=40)
    selected = replace(generic, bond_order="=", context="O200 0000")
    full = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=Path("same.fld"),
        bonds=(selected, replace(generic, ff_row=None)),
    )
    subset = replace(full, bonds=(generic,))
    assert opt_substructure_membership(full, subset).bonds == frozenset({0})


@pytest.mark.parametrize("reverse", [False, True])
def test_repeated_references_to_a_source_row_cannot_leak_fallback_credit(reverse: bool) -> None:
    first = BondParam(("C", "H"), 1.1, 100.0, env_id="C1-H1", ff_row=40)
    second = replace(first, env_id="C2-H2")
    full = ForceField(
        functional_form=FunctionalForm.MM3,
        source_path=Path("same.fld"),
        bonds=(first, replace(first, ff_row=None), replace(second, ff_row=None)),
    )
    requested = (second, first) if reverse else (first, second)
    subset = replace(full, bonds=requested)
    assert opt_substructure_membership(full, subset).bonds == frozenset({0})
