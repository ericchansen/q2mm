"""Geometry evaluator — computes optimized geometry observables and residuals.

Handles bond lengths, bond angles, and torsion (dihedral) angles.
Geometry observables require an MM minimization before extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceValue


def dihedral_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Compute dihedral angle (degrees) for four points.

    Delegates to :func:`q2mm.geometry.dihedral_angle`.

    Args:
        p0: Coordinates of the first atom.
        p1: Coordinates of the second atom.
        p2: Coordinates of the third atom.
        p3: Coordinates of the fourth atom.

    Returns:
        Dihedral angle in degrees, in the range [-180, 180].

    """
    from q2mm.geometry import dihedral_angle as _dihedral

    return _dihedral(p0, p1, p2, p3)


@dataclass
class GeometryResult:
    """Container for computed MM geometry observables.

    Attributes:
        bond_lengths: Positional list of bond lengths (Å).
        bond_lengths_by_atoms: Bond lengths keyed by sorted atom-index
            pairs for identity-based matching.
        bond_angles: Positional list of bond angles (degrees).
        bond_angles_by_atoms: Angles keyed by ``(i, j, k)`` atom triples.
        torsion_coords: Optimized Cartesian coordinates for dihedral
            computation. ``None`` if not needed.

    """

    bond_lengths: list[float] = field(default_factory=list)
    bond_lengths_by_atoms: dict[tuple[int, ...], float] = field(default_factory=dict)
    bond_angles: list[float] = field(default_factory=list)
    bond_angles_by_atoms: dict[tuple[int, ...], float] = field(default_factory=dict)
    torsion_coords: np.ndarray | None = None


class GeometryEvaluator:
    """Evaluates MM-optimized geometry against QM reference geometry.

    Runs ``engine.minimize()`` to get an optimized structure, then
    extracts bond lengths, bond angles, and torsion angles for comparison
    with reference data.

    Note:
        Minimization uses the *raw molecule* (not a cached engine handle)
        because ``minimize()`` mutates context positions — reusing a
        cached handle would corrupt subsequent energy/frequency evaluations.

    """

    GEOMETRY_KINDS = frozenset({"bond_length", "bond_angle", "torsion_angle"})

    def compute(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        *,
        structure: Any | None = None,
        needed_kinds: frozenset[str] | None = None,
    ) -> GeometryResult:
        """Minimize the molecule and extract geometry observables.

        Args:
            engine: The MM backend.
            mol: The molecule being evaluated.
            ff: The current force field.
            structure: Ignored (minimization always uses raw molecule).
            needed_kinds: Which geometry kinds are needed. Defaults to
                all geometry kinds.

        Returns:
            GeometryResult with computed bond lengths, angles, and/or
            torsion coordinates.

        """
        if needed_kinds is None:
            needed_kinds = self.GEOMETRY_KINDS

        _energy, _atoms, opt_coords = engine.minimize(mol, ff)
        opt_mol = Q2MMMolecule(
            symbols=mol.symbols,
            geometry=opt_coords,
            name=mol.name,
            atom_types=list(mol.atom_types),
            bond_tolerance=mol.bond_tolerance,
        )

        result = GeometryResult()

        if "bond_length" in needed_kinds:
            result.bond_lengths = [b.length for b in opt_mol.bonds]
            result.bond_lengths_by_atoms = {tuple(sorted((b.atom_i, b.atom_j))): b.length for b in opt_mol.bonds}

        if "bond_angle" in needed_kinds:
            result.bond_angles = [a.value for a in opt_mol.angles]
            result.bond_angles_by_atoms = {(a.atom_i, a.atom_j, a.atom_k): a.value for a in opt_mol.angles}

        if "torsion_angle" in needed_kinds:
            result.torsion_coords = opt_coords

        return result

    def residuals(
        self,
        computed: GeometryResult,
        references: list[ReferenceValue],
    ) -> list[float]:
        """Compute weighted residuals for geometry references.

        Args:
            computed: Output from :meth:`compute`.
            references: Reference geometry values (bond_length,
                bond_angle, torsion_angle).

        Returns:
            List of ``w * (ref - calc)`` residuals.

        """
        result: list[float] = []
        for ref in references:
            calc_value = self._extract(computed, ref)
            diff = ref.value - calc_value
            if ref.kind == "torsion_angle":
                diff = (diff + 180.0) % 360.0 - 180.0
            result.append(ref.weight * diff)
        return result

    @staticmethod
    def _extract(computed: GeometryResult, ref: ReferenceValue) -> float:
        """Extract a single calculated value from a GeometryResult.

        Args:
            computed: Geometry result.
            ref: Reference value to match.

        Returns:
            The calculated geometry value.

        Raises:
            IndexError: If positional index is out of range.
            KeyError: If atom-identity match fails.
            ValueError: If torsion is missing atom indices.

        """
        if ref.kind == "bond_length":
            if ref.atom_indices is not None:
                key = tuple(sorted(ref.atom_indices[:2]))
                if key not in computed.bond_lengths_by_atoms:
                    raise KeyError(
                        f"No bond found for atoms {key}. "
                        f"Available bonds: {list(computed.bond_lengths_by_atoms.keys())}. "
                        f"Label: {ref.label!r}"
                    )
                return computed.bond_lengths_by_atoms[key]
            if ref.data_idx < 0 or ref.data_idx >= len(computed.bond_lengths):
                raise IndexError(
                    f"Bond data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(computed.bond_lengths)} bonds). "
                    f"Label: {ref.label!r}"
                )
            return computed.bond_lengths[ref.data_idx]

        elif ref.kind == "bond_angle":
            if ref.atom_indices is not None:
                key = tuple(ref.atom_indices[:3])
                by_atoms = computed.bond_angles_by_atoms
                if key not in by_atoms:
                    key = (key[2], key[1], key[0])
                if key not in by_atoms:
                    raise KeyError(
                        f"No angle found for atoms {ref.atom_indices[:3]}. "
                        f"Available angles: {list(by_atoms.keys())}. "
                        f"Label: {ref.label!r}"
                    )
                return by_atoms[key]
            if ref.data_idx < 0 or ref.data_idx >= len(computed.bond_angles):
                raise IndexError(
                    f"Angle data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(computed.bond_angles)} angles). "
                    f"Label: {ref.label!r}"
                )
            return computed.bond_angles[ref.data_idx]

        elif ref.kind == "torsion_angle":
            if ref.atom_indices is None or len(ref.atom_indices) < 4:
                raise ValueError(f"torsion_angle requires atom_indices with 4 atoms. Label: {ref.label!r}")
            if computed.torsion_coords is None:
                raise ValueError("GeometryResult has no torsion_coords")
            coords = computed.torsion_coords
            return dihedral_angle(
                coords[ref.atom_indices[0]],
                coords[ref.atom_indices[1]],
                coords[ref.atom_indices[2]],
                coords[ref.atom_indices[3]],
            )

        raise ValueError(f"GeometryEvaluator cannot handle kind: {ref.kind}")

    def supports_analytical_gradient(self, engine: MMEngine) -> bool:
        """Geometry gradients require differentiating through the minimizer.

        Args:
            engine: The MM backend to check.

        Returns:
            Always ``False`` — not yet implemented.

        """
        return False

    def gradient(
        self,
        engine: MMEngine,
        mol: Q2MMMolecule,
        ff: ForceField,
        references: list[ReferenceValue],
        n_params: int,
        *,
        structure: Any | None = None,
    ) -> np.ndarray | None:
        """Not yet implemented — geometry analytical gradients.

        Differentiating through the MM geometry optimizer is planned
        for a future release.

        Returns:
            ``None`` — analytical gradients are not yet supported.

        """
        return None

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: ReferenceValue) -> float:
        """Extract a calculated geometry value from a results dict.

        Backward-compatible bridge for ObjectiveFunction._extract_value.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated geometry value.

        """
        if ref.kind == "bond_length":
            if ref.atom_indices is not None:
                key = tuple(sorted(ref.atom_indices[:2]))
                by_atoms = calc.get("bond_lengths_by_atoms", {})
                if key not in by_atoms:
                    raise KeyError(
                        f"No bond found for atoms {key}. Available bonds: {list(by_atoms.keys())}. Label: {ref.label!r}"
                    )
                return by_atoms[key]
            lengths = calc["bond_lengths"]
            if ref.data_idx < 0 or ref.data_idx >= len(lengths):
                raise IndexError(
                    f"Bond data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(lengths)} bonds). "
                    f"Label: {ref.label!r}"
                )
            return lengths[ref.data_idx]
        elif ref.kind == "bond_angle":
            if ref.atom_indices is not None:
                by_atoms = calc.get("bond_angles_by_atoms", {})
                key = tuple(ref.atom_indices[:3])
                if key not in by_atoms:
                    key = (key[2], key[1], key[0])
                if key not in by_atoms:
                    raise KeyError(
                        f"No angle found for atoms {ref.atom_indices[:3]}. "
                        f"Available angles: {list(by_atoms.keys())}. "
                        f"Label: {ref.label!r}"
                    )
                return by_atoms[key]
            angles = calc["bond_angles"]
            if ref.data_idx < 0 or ref.data_idx >= len(angles):
                raise IndexError(
                    f"Angle data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(angles)} angles). "
                    f"Label: {ref.label!r}"
                )
            return angles[ref.data_idx]
        elif ref.kind == "torsion_angle":
            if ref.atom_indices is None or len(ref.atom_indices) < 4:
                raise ValueError(f"torsion_angle requires atom_indices with 4 atoms. Label: {ref.label!r}")
            coords = calc["torsion_coords"]
            return dihedral_angle(
                coords[ref.atom_indices[0]],
                coords[ref.atom_indices[1]],
                coords[ref.atom_indices[2]],
                coords[ref.atom_indices[3]],
            )
        raise ValueError(f"GeometryEvaluator cannot handle kind: {ref.kind}")
