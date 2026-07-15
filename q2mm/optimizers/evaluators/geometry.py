"""Geometry evaluator — computes optimized geometry observables and residuals.

Handles bond lengths, bond angles, and torsion (dihedral) angles.
Geometry observables require an MM minimization before extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from q2mm.backends.contracts import MinimizationRequest, PreparedBackend
from q2mm.models.observations import Observation


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

    Runs ``prepared.minimize()`` to get an optimized structure, then
    extracts bond lengths, bond angles, and torsion angles for comparison
    with reference data.

    Note:
        Minimization uses the *raw molecule* (not a cached engine handle)
        because ``minimize()`` mutates context positions — reusing a
        cached handle would corrupt subsequent energy/frequency evaluations.

    """

    GEOMETRY_KINDS = frozenset({"bond_length", "bond_angle", "torsion_angle"})
    HANDLED_KINDS = GEOMETRY_KINDS

    def compute(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        *,
        needed_kinds: frozenset[str] | None = None,
    ) -> GeometryResult:
        """Minimize the molecule and extract geometry observables.

        Args:
            prepared: The prepared per-case backend session.
            parameters: Full parameter vector.
            needed_kinds: Which geometry kinds are needed. Defaults to
                all geometry kinds.

        Returns:
            GeometryResult with computed bond lengths, angles, and/or
            torsion coordinates.

        """
        if needed_kinds is None:
            needed_kinds = self.GEOMETRY_KINDS

        mol = prepared.molecule
        min_result = prepared.minimize(MinimizationRequest(parameters=parameters))
        opt_coords = np.asarray(min_result.coordinates)

        # Use the ORIGINAL molecule's bond/angle/torsion topology —
        # bonding is a user-level decision from the QM input, not
        # something to re-detect from MM-optimized coordinates.
        result = GeometryResult()

        if "bond_length" in needed_kinds:
            for bond in mol.bonds or ():
                length = float(np.linalg.norm(opt_coords[bond.atom_j] - opt_coords[bond.atom_i]))
                result.bond_lengths.append(length)
                key = tuple(sorted((bond.atom_i, bond.atom_j)))
                result.bond_lengths_by_atoms[key] = length

        if "bond_angle" in needed_kinds:
            for angle in mol.angles or ():
                v1 = opt_coords[angle.atom_i] - opt_coords[angle.atom_j]
                v2 = opt_coords[angle.atom_k] - opt_coords[angle.atom_j]
                cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                value = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
                result.bond_angles.append(value)
                result.bond_angles_by_atoms[(angle.atom_i, angle.atom_j, angle.atom_k)] = value

        if "torsion_angle" in needed_kinds:
            result.torsion_coords = opt_coords

        return result

    def residuals(
        self,
        computed: GeometryResult,
        references: list[Observation],
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
    def _extract(computed: GeometryResult, ref: Observation) -> float:
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

    def supports_analytical_gradient(self, prepared: PreparedBackend) -> bool:
        """Whether the geometry evaluator supports analytical gradients.

        Returns ``False``: geometry gradients require differentiating
        through a geometry minimization.  The JAX backend *does* support a
        fully-differentiable geometry loss via
        :class:`~q2mm.optimizers.jaxloss.JaxLoss`, but that is a separate
        end-to-end code path that does not flow through
        :meth:`ObjectiveFunction.gradient`.

        Args:
            prepared: The prepared backend session to check.

        Returns:
            Always ``False`` — not supported through this path.

        """
        return False

    def gradient(
        self,
        prepared: PreparedBackend,
        parameters: np.ndarray,
        references: list[Observation],
        n_params: int,
        *,
        mol_idx: int = 0,
    ) -> np.ndarray | None:
        """Not supported — geometry analytical gradients.

        Differentiating through the MM geometry optimizer flows through the
        JAX loss path, not this evaluator.

        Returns:
            ``None`` — analytical gradients are not supported here.

        """
        return None

    @staticmethod
    def extract_value(calc: dict[str, Any], ref: Observation) -> float:
        """Extract a calculated geometry value from a results dict.

        Backward-compatible bridge for ObjectiveFunction._extract_value.
        Delegates to :meth:`_extract` via a temporary :class:`GeometryResult`.

        Args:
            calc: Results dict from ``_evaluate_molecule``.
            ref: The reference value to match.

        Returns:
            The calculated geometry value.

        """
        result = GeometryResult(
            bond_lengths=calc.get("bond_lengths", []),
            bond_lengths_by_atoms=calc.get("bond_lengths_by_atoms", {}),
            bond_angles=calc.get("bond_angles", []),
            bond_angles_by_atoms=calc.get("bond_angles_by_atoms", {}),
            torsion_coords=calc.get("torsion_coords"),
        )
        return GeometryEvaluator._extract(result, ref)
