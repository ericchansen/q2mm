"""Backend-neutral observable extraction shared by both executors.

Pure NumPy helpers that turn a per-case ``computed`` dict (energy,
frequencies, relaxed geometry, MM Hessian, eigenmatrix) into the calculated
value for a single :class:`~q2mm.models.observations.Observation`, and that
build the geometry portion of that dict from relaxed coordinates using the
molecule's own bond/angle topology.

These functions contain no backend imports, so the Python and JAX executors
are independent consumers of one shared observable-extraction implementation
(neither executor depends on the other).
"""

from __future__ import annotations

import numpy as np

from q2mm.models.observations import Observation

__all__ = ["geometry_computed", "extract_calc_value"]


def geometry_computed(mol: object, coords: np.ndarray, needed: set[str]) -> dict:
    """Build the geometry portion of a per-case ``computed`` dict from *coords*.

    Uses the molecule's own bond/angle topology (a user-level decision from
    the QM input, not re-detected from optimized coordinates), so both
    executors extract identical per-observation geometry values from a
    relaxed structure.
    """
    coords = np.asarray(coords)
    computed: dict = {}
    if "bond_length" in needed:
        by_atoms: dict[tuple[int, ...], float] = {}
        ordered: list[float] = []
        for bond in getattr(mol, "bonds", None) or ():
            length = float(np.linalg.norm(coords[bond.atom_j] - coords[bond.atom_i]))
            by_atoms[tuple(sorted((bond.atom_i, bond.atom_j)))] = length
            ordered.append(length)
        computed["bond_lengths"] = ordered
        computed["bond_lengths_by_atoms"] = by_atoms
        computed["_bond_lengths_ordered"] = ordered
    if "bond_angle" in needed:
        angles_by_atoms: dict[tuple[int, ...], float] = {}
        ordered_angles: list[float] = []
        for angle in getattr(mol, "angles", None) or ():
            v1 = coords[angle.atom_i] - coords[angle.atom_j]
            v2 = coords[angle.atom_k] - coords[angle.atom_j]
            cos_val = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            value = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
            angles_by_atoms[(angle.atom_i, angle.atom_j, angle.atom_k)] = value
            ordered_angles.append(value)
        computed["bond_angles"] = ordered_angles
        computed["bond_angles_by_atoms"] = angles_by_atoms
    if "torsion_angle" in needed:
        computed["torsion_coords"] = coords
    return computed


def extract_calc_value(computed: dict, ref: Observation) -> float:
    """Extract the calculated value for *ref* from a per-case ``computed`` dict."""
    kind = ref.kind
    if kind == "energy":
        return float(computed["energy"])
    if kind == "frequency":
        freqs = computed["frequencies"]
        if ref.data_idx < 0 or ref.data_idx >= len(freqs):
            raise IndexError(
                f"Frequency data_idx={ref.data_idx} out of range ({len(freqs)} modes). Label: {ref.label!r}"
            )
        return float(freqs[ref.data_idx])
    if kind == "bond_length":
        if ref.atom_indices is not None:
            key = tuple(sorted(ref.atom_indices[:2]))
            by_atoms = computed["bond_lengths_by_atoms"]
            if key not in by_atoms:
                raise KeyError(
                    f"No bond found for atoms {key}. Available: {list(by_atoms.keys())}. Label: {ref.label!r}"
                )
            return float(by_atoms[key])
        ordered = computed["_bond_lengths_ordered"]
        if ref.data_idx < 0 or ref.data_idx >= len(ordered):
            raise IndexError(f"Bond data_idx={ref.data_idx} out of range. Label: {ref.label!r}")
        return float(ordered[ref.data_idx])
    if kind == "bond_angle":
        if ref.atom_indices is not None:
            key = tuple(ref.atom_indices[:3])
            by_atoms = computed["bond_angles_by_atoms"]
            if key not in by_atoms:
                key = (key[2], key[1], key[0])
            if key not in by_atoms:
                raise KeyError(
                    f"No angle found for atoms {ref.atom_indices[:3]}. "
                    f"Available: {list(by_atoms.keys())}. Label: {ref.label!r}"
                )
            return float(by_atoms[key])
        ordered = computed["bond_angles"]
        if ref.data_idx < 0 or ref.data_idx >= len(ordered):
            raise IndexError(f"Angle data_idx={ref.data_idx} out of range. Label: {ref.label!r}")
        return float(ordered[ref.data_idx])
    if kind == "torsion_angle":
        if ref.atom_indices is None or len(ref.atom_indices) < 4:
            raise ValueError(f"torsion_angle requires 4 atom_indices. Label: {ref.label!r}")
        from q2mm.geometry import dihedral_angle

        coords = computed["torsion_coords"]
        return float(
            dihedral_angle(
                coords[ref.atom_indices[0]],
                coords[ref.atom_indices[1]],
                coords[ref.atom_indices[2]],
                coords[ref.atom_indices[3]],
            )
        )
    if kind == "eig_diagonal":
        eigmat = computed["eigenmatrix"]
        n = eigmat.shape[0]
        if ref.data_idx < 0 or ref.data_idx >= n:
            raise IndexError(f"Eigenmatrix data_idx={ref.data_idx} out of range ({n} modes). Label: {ref.label!r}")
        return float(eigmat[ref.data_idx, ref.data_idx])
    if kind == "eig_offdiagonal":
        eigmat = computed["eigenmatrix"]
        if ref.atom_indices is None or len(ref.atom_indices) < 2:
            raise ValueError(f"eig_offdiagonal requires atom_indices=(row, col). Label: {ref.label!r}")
        row, col = ref.atom_indices[:2]
        return float(eigmat[row, col])
    if kind == "hessian_element":
        hess = computed["raw_hessian"]
        if ref.atom_indices is None or len(ref.atom_indices) < 2:
            raise ValueError(f"hessian_element requires atom_indices=(row, col). Label: {ref.label!r}")
        row, col = ref.atom_indices[:2]
        n = hess.shape[0]
        if row < 0 or row >= n or col < 0 or col >= n:
            raise IndexError(f"Hessian indices ({row}, {col}) out of range for {n}×{n}. Label: {ref.label!r}")
        return float(hess[row, col])
    raise ValueError(f"Unknown reference kind: {kind!r}")
