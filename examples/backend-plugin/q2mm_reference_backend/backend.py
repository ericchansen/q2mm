"""Minimal harmonic ENERGY implementation for the reference plugin."""

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

_NAME = "harmonic-reference"
_PROVENANCE = BackendProvenance(
    backend=_NAME,
    role=BackendRole.MM,
    details={
        "implementation": {"name": "Q2MM harmonic reference plugin"},
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


class _HarmonicReferencePrepared(AbstractPreparedBackend):
    def _energy(self, request: EnergyRequest) -> EnergyResult:
        force_field = self.layout.replace(self.force_field, request.parameters)
        coordinates = np.asarray(self.molecule.geometry, dtype=float)
        symbols = self.molecule.symbols
        total = 0.0
        for bond in self.molecule.bonds or ():
            elements = tuple(sorted((symbols[bond.atom_i], symbols[bond.atom_j])))
            parameter = force_field.match_bond(elements)
            if parameter is None:
                continue
            distance = float(np.linalg.norm(coordinates[bond.atom_i] - coordinates[bond.atom_j]))
            total += parameter.force_constant * (distance - parameter.equilibrium) ** 2
        return EnergyResult(energy=total, unit=EnergyUnit.KCAL_PER_MOL, provenance=_PROVENANCE)


class HarmonicReferenceBackend:
    """Small, valid MM backend implementing harmonic bond energy."""

    @property
    def info(self) -> BackendInfo:
        """Return the runtime's exact capability and form subsets."""
        return _INFO

    def prepare(self, request: PreparationRequest) -> _HarmonicReferencePrepared:
        """Prepare one immutable harmonic-energy session."""
        if request.force_field is None:
            raise ValueError("harmonic-reference requires a force field.")
        return _HarmonicReferencePrepared(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )
