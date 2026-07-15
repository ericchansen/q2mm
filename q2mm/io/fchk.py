"""Minimal self-contained parser for Gaussian formatted checkpoint (.fchk) files.

Extracts geometry, atomic numbers, and (optionally) the Cartesian Force
Constants (Hessian) from a ``.fchk`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q2mm import constants
from q2mm.elements import ATOMIC_SYMBOLS as _ATOMIC_SYMBOLS
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet

_BOHR_TO_ANG = constants.BOHR_TO_ANG


def load_fchk(
    path: Path,
    *,
    bond_tolerance: float = 1.3,
    charge: int = 0,
    multiplicity: int = 1,
    name: str = "",
) -> Molecule:
    """Load a canonical molecule from a Gaussian ``.fchk`` file.

    Args:
        path: Path to the ``.fchk`` file.
        bond_tolerance: Multiplier on covalent radii for inferred bonds.
        charge: Charge used when the file omits it.
        multiplicity: Multiplicity used when the file omits it.
        name: Molecule name; defaults to the file stem.

    Returns:
        Molecule with geometry in Angstrom and an optional canonical
        Hartree/Bohr² Hessian carrying FCHK provenance.

    Raises:
        ValueError: If atomic numbers or coordinates cannot be parsed.

    """
    with open(path) as f:
        lines = f.readlines()

    n_atoms = None
    file_charge = None
    file_multiplicity = None
    atomic_numbers: list[int] = []
    coords_bohr: list[float] = []
    hessian_flat: list[float] = []
    reading = None  # tracks which array section we're in
    expected = 0

    for line in lines:
        # Scalar integer fields
        if line.startswith("Number of atoms"):
            n_atoms = int(line.split()[-1])
            continue
        if line.startswith("Charge "):
            file_charge = int(line.split()[-1])
            continue
        if line.startswith("Multiplicity"):
            file_multiplicity = int(line.split()[-1])
            continue

        # Array section headers
        if line.startswith("Atomic numbers") and "N=" in line:
            reading = "atomic_numbers"
            expected = int(line.split("N=")[1].strip())
            continue
        if line.startswith("Current cartesian coordinates") and "N=" in line:
            reading = "coords"
            expected = int(line.split("N=")[1].strip())
            continue
        if line.startswith("Cartesian Force Constants") and "N=" in line:
            reading = "hessian"
            expected = int(line.split("N=")[1].strip())
            continue

        # Other array headers end the current section
        if len(line) > 40 and ("N=" in line[40:] or ("I" in line[40:50] and line[40:50].strip() in ("I", "R"))):
            if reading:
                reading = None
            continue

        # Read array data
        if reading == "atomic_numbers" and len(atomic_numbers) < expected:
            atomic_numbers.extend(int(x) for x in line.split())
            if len(atomic_numbers) >= expected:
                reading = None
        elif reading == "coords" and len(coords_bohr) < expected:
            coords_bohr.extend(float(x) for x in line.split())
            if len(coords_bohr) >= expected:
                reading = None
        elif reading == "hessian" and len(hessian_flat) < expected:
            hessian_flat.extend(float(x) for x in line.split())
            if len(hessian_flat) >= expected:
                reading = None

    if not atomic_numbers or not coords_bohr:
        raise ValueError(f"Could not parse atomic numbers or coordinates from {path}")

    symbols = []
    for z in atomic_numbers:
        sym = _ATOMIC_SYMBOLS.get(z)
        if sym is None:
            raise ValueError(f"Unsupported atomic number {z} in {path}")
        symbols.append(sym)
    coords_ang = np.array(coords_bohr).reshape(-1, 3) * _BOHR_TO_ANG

    hessian = None
    if hessian_flat:
        n = len(symbols)
        dim = 3 * n
        # .fchk stores lower triangle in row-major order
        hessian = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i + 1):
                hessian[i, j] = hessian_flat[idx]
                hessian[j, i] = hessian_flat[idx]
                idx += 1

    provenance = (
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source="fchk",
            path=str(path.resolve()),
        )
        if hessian is not None
        else None
    )
    return Molecule(
        symbols=tuple(symbols),
        geometry=coords_ang,
        charge=file_charge if file_charge is not None else charge,
        multiplicity=file_multiplicity if file_multiplicity is not None else multiplicity,
        name=name or path.stem,
        bond_tolerance=bond_tolerance,
        hessian=hessian,
        hessian_provenance=provenance,
    )


def load_fchk_reference(
    path: str | Path,
    *,
    weights: dict[str, float] | None = None,
    bond_tolerance: float = constants.DEFAULT_BOND_TOLERANCE,
    charge: int = 0,
    multiplicity: int = 1,
) -> tuple[ObservationSet, Molecule]:
    """Load a molecule from a Gaussian ``.fchk`` file and build its reference data.

    Parses the ``.fchk`` file for geometry, Cartesian Force Constants
    (Hessian), and atom data, then auto-populates bond lengths, angles,
    and (when a Hessian is available) eigenmatrix training data via
    :meth:`~q2mm.models.observations.ObservationSet.from_molecule`.

    Args:
        path (str | Path): Path to the Gaussian ``.fchk`` file.
        weights (dict[str, float] | None): Weight overrides (same keys
            as :meth:`~q2mm.models.observations.ObservationSet.from_molecule`).
        bond_tolerance (float): Multiplier for covalent-radii bond
            detection.
        charge (int): Molecular charge (overridden by file values if
            present).
        multiplicity (int): Spin multiplicity (overridden by file
            values if present).

    Returns:
        tuple[ObservationSet, Molecule]: Populated reference data and
            the parsed molecule with Hessian.

    """
    mol = load_fchk(
        Path(path),
        bond_tolerance=bond_tolerance,
        charge=charge,
        multiplicity=multiplicity,
    )
    ref = ObservationSet.from_molecule(mol, weights=weights)
    return ref, mol
