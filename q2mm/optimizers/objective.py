"""Objective function for force field optimization.

Wraps the ForceField ↔ MM-engine ↔ reference-data loop into a single
callable that :func:`scipy.optimize.minimize` can drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule


# ---- Reference data containers ----


@dataclass
class ReferenceValue:
    """A single reference observation (QM or experimental)."""

    kind: Literal["energy", "frequency", "bond_length", "bond_angle"]
    value: float
    weight: float = 1.0
    label: str = ""
    # Indices for matching to calculated data
    molecule_idx: int = 0
    data_idx: int = 0


@dataclass
class ReferenceData:
    """Complete set of reference data for an optimization.

    Each entry describes one observable: an energy, a frequency, or a
    geometric parameter that the force field should reproduce.
    """

    values: list[ReferenceValue] = field(default_factory=list)

    def add_energy(
        self,
        value: float,
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        self.values.append(
            ReferenceValue(
                kind="energy",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                label=label,
            )
        )

    def add_frequency(
        self,
        value: float,
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        data_idx: int = 0,
        label: str = "",
    ):
        self.values.append(
            ReferenceValue(
                kind="frequency",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=data_idx,
                label=label,
            )
        )

    def add_bond_length(
        self,
        value: float,
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        data_idx: int = 0,
        label: str = "",
    ):
        self.values.append(
            ReferenceValue(
                kind="bond_length",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=data_idx,
                label=label,
            )
        )

    def add_bond_angle(
        self,
        value: float,
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        data_idx: int = 0,
        label: str = "",
    ):
        self.values.append(
            ReferenceValue(
                kind="bond_angle",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=data_idx,
                label=label,
            )
        )

    @property
    def n_observations(self) -> int:
        return len(self.values)


# ---- Objective function ----


class ObjectiveFunction:
    """Objective function for scipy-based force field optimization.

    Evaluates the weighted sum-of-squares between MM-calculated and
    reference data for one or more molecules.

    Parameters
    ----------
    forcefield : ForceField
        The force field whose parameters are being optimized.
    engine : MMEngine
        The MM backend (OpenMM, Tinker, etc.).
    molecules : list[Q2MMMolecule]
        Training set molecules.
    reference : ReferenceData
        QM/experimental reference observations.
    """

    def __init__(
        self,
        forcefield: ForceField,
        engine: MMEngine,
        molecules: list[Q2MMMolecule],
        reference: ReferenceData,
    ):
        self.forcefield = forcefield
        self.engine = engine
        self.molecules = molecules
        self.reference = reference
        self.n_eval = 0
        self.history: list[float] = []

    def __call__(self, param_vector: np.ndarray) -> float:
        """Evaluate objective for a given parameter vector.

        This is the function signature that :func:`scipy.optimize.minimize`
        expects: ``f(x) -> float``.
        """
        self.forcefield.set_param_vector(param_vector)

        residuals = self._compute_residuals()
        score = float(np.sum(residuals**2))

        self.n_eval += 1
        self.history.append(score)
        return score

    def residuals(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute weighted residual vector (for least-squares methods)."""
        self.forcefield.set_param_vector(param_vector)
        return self._compute_residuals()

    def _compute_residuals(self) -> np.ndarray:
        """Compute weighted residuals for all reference observations."""
        calc_cache: dict[int, dict] = {}

        residuals = []
        for ref in self.reference.values:
            mol_idx = ref.molecule_idx
            if mol_idx not in calc_cache:
                calc_cache[mol_idx] = self._evaluate_molecule(mol_idx)

            calc = calc_cache[mol_idx]
            calc_value = self._extract_value(calc, ref)
            residual = ref.weight * (ref.value - calc_value)
            residuals.append(residual)

        return np.array(residuals)

    def _evaluate_molecule(self, mol_idx: int) -> dict:
        """Run MM calculations for a single molecule."""
        mol = self.molecules[mol_idx]
        result: dict = {}

        # Determine what data types are needed for this molecule
        needed = {ref.kind for ref in self.reference.values if ref.molecule_idx == mol_idx}

        if "energy" in needed:
            result["energy"] = self.engine.energy(mol, self.forcefield)

        if "frequency" in needed:
            result["frequencies"] = self.engine.frequencies(mol, self.forcefield)

        if "bond_length" in needed:
            result["bond_lengths"] = [b.length for b in mol.bonds]

        if "bond_angle" in needed:
            result["bond_angles"] = [a.angle for a in mol.angles]

        return result

    @staticmethod
    def _extract_value(calc: dict, ref: ReferenceValue) -> float:
        """Extract a calculated value matching a reference observation."""
        if ref.kind == "energy":
            return calc["energy"]
        elif ref.kind == "frequency":
            freqs = calc["frequencies"]
            if ref.data_idx >= len(freqs):
                return 0.0
            return freqs[ref.data_idx]
        elif ref.kind == "bond_length":
            lengths = calc["bond_lengths"]
            if ref.data_idx >= len(lengths):
                return 0.0
            return lengths[ref.data_idx]
        elif ref.kind == "bond_angle":
            angles = calc["bond_angles"]
            if ref.data_idx >= len(angles):
                return 0.0
            return angles[ref.data_idx]
        else:
            raise ValueError(f"Unknown reference kind: {ref.kind}")

    def reset(self):
        """Reset evaluation counter and history."""
        self.n_eval = 0
        self.history.clear()
