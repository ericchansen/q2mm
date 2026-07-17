"""Minimal harmonic MM backend implementation for the plugin fixture.

Declares exactly one capability — :attr:`~q2mm.backends.contracts.Capability.ENERGY`
— and computes a genuine harmonic bond-stretch energy
``E = sum_bonds k * (r - r0)^2`` (kcal/mol, no ½ factor, matching Q2MM's
canonical harmonic convention) from the prepared molecule geometry and the
force field's matched bond parameters.

It is intentionally tiny: its purpose is to prove capability-gated discovery
and conformance, not to add a new scientific engine.  Every other prepared
capability is left to the base class, which raises
:class:`~q2mm.backends.contracts.UnsupportedCapabilityError`.

This module is imported only when the descriptor's ``factory`` import string is
resolved by an explicit backend load — never during descriptor enumeration or
cataloging.
"""

from __future__ import annotations

import numpy as np

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    PreparationRequest,
)
from q2mm.models.parameters import ParameterLayout

_NAME = "harmonic-fixture"
_PROVENANCE = BackendProvenance(
    backend=_NAME,
    role=BackendRole.MM,
    details={
        "implementation": {"name": "q2mm fixture backend"},
        "model": {"functional_form": "harmonic", "terms": ["bond_stretch"]},
    },
)
_INFO = BackendInfo(
    name=_NAME,
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_PROVENANCE,
)


class _HarmonicFixturePrepared(AbstractPreparedBackend):
    """Prepared session computing a harmonic bond-stretch energy."""

    def _energy(self, request: EnergyRequest) -> EnergyResult:
        force_field = self.layout.replace(self.force_field, request.parameters)
        coordinates = np.asarray(self.molecule.geometry, dtype=float)
        symbols = self.molecule.symbols
        total = 0.0
        for bond in self.molecule.bonds or ():
            elements = tuple(sorted((symbols[bond.atom_i], symbols[bond.atom_j])))
            param = force_field.match_bond(elements)
            if param is None:
                continue
            distance = float(np.linalg.norm(coordinates[bond.atom_i] - coordinates[bond.atom_j]))
            total += param.force_constant * (distance - param.equilibrium) ** 2
        return EnergyResult(energy=total, unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROVENANCE)


class HarmonicFixtureBackend:
    """A minimal, valid MM backend declaring only ENERGY (harmonic form)."""

    @property
    def info(self) -> BackendInfo:
        """Static capability declaration (role MM, ENERGY, harmonic)."""
        return _INFO

    def prepare(self, request: PreparationRequest) -> _HarmonicFixturePrepared:
        """Prepare a harmonic-energy session for one training case."""
        force_field = request.force_field
        if force_field is None:
            raise ValueError("harmonic-fixture requires a force field to prepare an MM session.")
        return _HarmonicFixturePrepared(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=force_field,
            layout=ParameterLayout.from_force_field(force_field),
        )
