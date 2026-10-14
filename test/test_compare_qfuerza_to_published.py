"""Tests for the QFUERZA-to-published parameter comparison script."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from scripts import compare_qfuerza_to_published as comparison


def test_published_projection_includes_ub_only_active_angle(monkeypatch: pytest.MonkeyPatch) -> None:
    force_field = ForceField(
        bonds=(BondParam(("C", "H"), 1.1, 100.0), BondParam(("C", "C"), 1.5, 200.0)),
        angles=(AngleParam(("H", "C", "H"), 109.5, 30.0, ub_force_constant=2.0, ub_equilibrium=1.8),),
        functional_form=FunctionalForm.MM3,
    )
    layout = ParameterLayout.from_force_field(force_field)
    space = ActiveParameterSpace.all_active(layout, force_field).with_active_indices([2, len(layout) - 2])
    problem = SimpleNamespace(starting_force_field=force_field, layout=layout, active_space=space)

    def load(system: str, *, starting_point: str) -> SimpleNamespace:
        assert (system, starting_point) == ("pd-allyl", "published")
        return SimpleNamespace(problem=problem)

    monkeypatch.setattr(comparison, "load_system", load)
    assert comparison._load_published("pd-allyl") == (force_field, frozenset({1}), frozenset({0}))


def test_main_loads_optimized_force_field_with_full_topology(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    force_field = ForceField(
        bonds=(BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.54),),
        functional_form=FunctionalForm.MM3,
    )
    optimized_path = tmp_path / "optimized.fld"
    optimized_path.write_text("", encoding="utf-8")
    seen_include_standard: list[bool] = []

    monkeypatch.setattr(
        comparison,
        "_load_published",
        lambda _system: (force_field, frozenset({0}), frozenset()),
    )

    def _load(path: Path, *, include_standard: bool) -> ForceField:
        assert path == optimized_path
        seen_include_standard.append(include_standard)
        return force_field

    monkeypatch.setattr(comparison, "load_mm3_fld", _load)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_qfuerza_to_published.py",
            "--system",
            "rh-enamide",
            "--optimized",
            str(optimized_path),
        ],
    )

    assert comparison.main() == 0
    assert seen_include_standard == [True]
