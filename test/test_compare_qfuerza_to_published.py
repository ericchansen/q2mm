"""Tests for the QFUERZA-to-published parameter comparison script."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from scripts import compare_qfuerza_to_published as comparison


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
