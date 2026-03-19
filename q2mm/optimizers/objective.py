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
    # Atom-identity matching (preferred over positional data_idx for geometry)
    atom_indices: tuple[int, ...] | None = None


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
        data_idx: int,
        weight: float = 1.0,
        molecule_idx: int = 0,
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
        data_idx: int = -1,
        atom_indices: tuple[int, int] | None = None,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        if atom_indices is None and data_idx < 0:
            raise ValueError("Either atom_indices or data_idx must be provided for bond_length.")
        self.values.append(
            ReferenceValue(
                kind="bond_length",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=max(data_idx, 0),
                atom_indices=atom_indices,
                label=label,
            )
        )

    def add_bond_angle(
        self,
        value: float,
        *,
        data_idx: int = -1,
        atom_indices: tuple[int, int, int] | None = None,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        if atom_indices is None and data_idx < 0:
            raise ValueError("Either atom_indices or data_idx must be provided for bond_angle.")
        self.values.append(
            ReferenceValue(
                kind="bond_angle",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=max(data_idx, 0),
                atom_indices=atom_indices,
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
        # Reusable engine handles for backends that support runtime parameter
        # updates (e.g., OpenMM). Avoids rebuilding simulation contexts each
        # evaluation — critical for optimization performance.
        self._handles: dict[int, object] = {}

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
        r = self._compute_residuals()
        self.n_eval += 1
        self.history.append(float(np.sum(r**2)))
        return r

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

    def _get_structure(self, mol_idx: int):
        """Get the structure handle for a molecule, reusing if possible.

        For backends that support runtime parameter updates (OpenMM), this
        creates the simulation context once and reuses it across evaluations.
        For stateless backends (Tinker), this returns the raw molecule.
        """
        mol = self.molecules[mol_idx]
        if not self.engine.supports_runtime_params():
            return mol
        if mol_idx not in self._handles:
            # First call — let the engine create its context/handle
            self._handles[mol_idx] = self.engine.create_context(mol, self.forcefield)
        return self._handles[mol_idx]

    def _evaluate_molecule(self, mol_idx: int) -> dict:
        """Run MM calculations for a single molecule."""
        mol = self.molecules[mol_idx]
        structure = self._get_structure(mol_idx)
        result: dict = {}

        # Determine what data types are needed for this molecule
        needed = {ref.kind for ref in self.reference.values if ref.molecule_idx == mol_idx}

        if "energy" in needed:
            result["energy"] = self.engine.energy(structure, self.forcefield)

        if "frequency" in needed:
            result["frequencies"] = self.engine.frequencies(structure, self.forcefield)

        if "bond_length" in needed or "bond_angle" in needed:
            # Geometry observables require MM-minimized structures to be
            # meaningful (the input geometry is fixed). Minimize first.
            _energy, _atoms, opt_coords = self.engine.minimize(structure, self.forcefield)
            opt_mol = Q2MMMolecule(
                symbols=mol.symbols,
                geometry=opt_coords,
                name=mol.name,
                atom_types=list(mol.atom_types),
                bond_tolerance=mol.bond_tolerance,
            )
            if "bond_length" in needed:
                # Store both positional list and atom-keyed dict for matching
                result["bond_lengths"] = [b.length for b in opt_mol.bonds]
                result["bond_lengths_by_atoms"] = {
                    tuple(sorted((b.atom_i, b.atom_j))): b.length
                    for b in opt_mol.bonds
                }
            if "bond_angle" in needed:
                result["bond_angles"] = [a.value for a in opt_mol.angles]
                result["bond_angles_by_atoms"] = {
                    (a.atom_i, a.atom_j, a.atom_k): a.value
                    for a in opt_mol.angles
                }

        return result

    @staticmethod
    def _extract_value(calc: dict, ref: ReferenceValue) -> float:
        """Extract a calculated value matching a reference observation.

        For bond_length and bond_angle, prefers atom-identity matching via
        ``ref.atom_indices`` when available, falling back to positional
        ``ref.data_idx`` for backwards compatibility.
        """
        if ref.kind == "energy":
            return calc["energy"]
        elif ref.kind == "frequency":
            freqs = calc["frequencies"]
            if ref.data_idx >= len(freqs):
                raise IndexError(
                    f"Frequency data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(freqs)} modes). Label: {ref.label!r}"
                )
            return freqs[ref.data_idx]
        elif ref.kind == "bond_length":
            if ref.atom_indices is not None:
                key = tuple(sorted(ref.atom_indices[:2]))
                by_atoms = calc.get("bond_lengths_by_atoms", {})
                if key not in by_atoms:
                    raise KeyError(
                        f"No bond found for atoms {key}. "
                        f"Available bonds: {list(by_atoms.keys())}. Label: {ref.label!r}"
                    )
                return by_atoms[key]
            lengths = calc["bond_lengths"]
            if ref.data_idx >= len(lengths):
                raise IndexError(
                    f"Bond data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(lengths)} bonds). Label: {ref.label!r}"
                )
            return lengths[ref.data_idx]
        elif ref.kind == "bond_angle":
            if ref.atom_indices is not None:
                by_atoms = calc.get("bond_angles_by_atoms", {})
                key = tuple(ref.atom_indices[:3])
                # Try both orderings: (i, j, k) and (k, j, i)
                if key not in by_atoms:
                    key = (key[2], key[1], key[0])
                if key not in by_atoms:
                    raise KeyError(
                        f"No angle found for atoms {ref.atom_indices[:3]}. "
                        f"Available angles: {list(by_atoms.keys())}. Label: {ref.label!r}"
                    )
                return by_atoms[key]
            angles = calc["bond_angles"]
            if ref.data_idx >= len(angles):
                raise IndexError(
                    f"Angle data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(angles)} angles). Label: {ref.label!r}"
                )
            return angles[ref.data_idx]
        else:
            raise ValueError(f"Unknown reference kind: {ref.kind}")

    def reset(self):
        """Reset evaluation counter, history, and cached engine handles."""
        self.n_eval = 0
        self.history.clear()
        self._handles.clear()
