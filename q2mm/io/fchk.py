"""Minimal self-contained parser for Gaussian formatted checkpoint (.fchk) files.

Extracts geometry, atomic numbers, and (optionally) the Cartesian Force
Constants (Hessian) from a ``.fchk`` file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from q2mm import constants
from q2mm.elements import ATOMIC_SYMBOLS as _ATOMIC_SYMBOLS

_BOHR_TO_ANG = constants.BOHR_TO_ANG


def parse_fchk(
    path: Path,
) -> tuple[list[str], np.ndarray, np.ndarray | None, int | None, int | None]:
    """Parse a Gaussian .fchk file for geometry and Hessian.

    Args:
        path: Path to the ``.fchk`` file.

    Returns:
        ``(symbols, coords_angstrom, hessian_au_or_None, charge,
        multiplicity)``. The Hessian is in Hartree/Bohr² (atomic
        units) — the native .fchk format.

    Raises:
        ValueError: If atomic numbers or coordinates cannot be parsed.

    """
    with open(path) as f:
        lines = f.readlines()

    n_atoms = None
    charge = None
    multiplicity = None
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
            charge = int(line.split()[-1])
            continue
        if line.startswith("Multiplicity"):
            multiplicity = int(line.split()[-1])
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

    return symbols, coords_ang, hessian, charge, multiplicity
