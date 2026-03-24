"""Psi4 quantum mechanics engine backend.

Wraps the Psi4 Python API for QM calculations: energy, Hessian, geometry
optimization, and vibrational frequencies.

Requires: ``conda install psi4 -c conda-forge``
"""

from __future__ import annotations


import os
import shutil
import tempfile
import numpy as np

from q2mm.backends.base import QMEngine
from q2mm.backends.registry import register_qm
from q2mm.constants import BOHR_TO_ANG

try:
    import psi4 as _psi4

    _HAS_PSI4 = True
except ImportError:
    _psi4 = None
    _HAS_PSI4 = False


def _read_xyz(path: str) -> tuple[list[str], np.ndarray]:
    """Read an XYZ file.

    Args:
        path: Path to the XYZ file.

    Returns:
        tuple[list[str], np.ndarray]: ``(atom_labels, coordinates)`` where
            coordinates are in Å with shape ``(N, 3)``.

    """
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)


def _make_psi4_geometry(atoms: list[str], coords: np.ndarray, charge: int = 0, multiplicity: int = 1) -> object:
    """Create a Psi4 molecule object from atoms and coordinates.

    Args:
        atoms: Element symbols.
        coords: Cartesian coordinates, shape ``(N, 3)``, in Å.
        charge: Molecular charge.
        multiplicity: Spin multiplicity.

    Returns:
        A Psi4 ``Molecule`` object.

    """
    geom_str = f"    {charge} {multiplicity}\n"
    for atom, (x, y, z) in zip(atoms, coords):
        geom_str += f"    {atom} {x:.10f} {y:.10f} {z:.10f}\n"
    return _psi4.geometry(geom_str)


@register_qm("psi4")
class Psi4Engine(QMEngine):
    """Quantum mechanics engine using Psi4.

    Args:
        method: DFT functional or method (default: "b3lyp")
        basis: Basis set (default: "6-31+G(d)")
        memory: Memory allocation (default: "2 GB")
        n_threads: Number of threads (default: 4)
        charge: Molecular charge (default: 0)
        multiplicity: Spin multiplicity (default: 1)

    """

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "6-31+G(d)",
        memory: str = "2 GB",
        n_threads: int = 4,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        """Initialize the Psi4 engine.

        Args:
            method: DFT functional or method (e.g. ``"b3lyp"``, ``"mp2"``).
            basis: Basis set (e.g. ``"6-31+G(d)"``).
            memory: Memory allocation string (e.g. ``"2 GB"``).
            n_threads: Number of threads for parallel computation.
            charge: Molecular charge.
            multiplicity: Spin multiplicity.

        Raises:
            ImportError: If Psi4 is not installed.

        """
        if not _HAS_PSI4:
            raise ImportError("Psi4 is not installed. Install via: conda install psi4 -c conda-forge")
        self._method = method
        self._basis = basis
        self._charge = charge
        self._multiplicity = multiplicity
        _psi4.set_memory(memory)
        _psi4.set_num_threads(n_threads)
        # Suppress output by default
        self._tmpdir = tempfile.mkdtemp(prefix="q2mm_psi4_")
        _psi4.core.set_output_file(os.path.join(self._tmpdir, "psi4_output.dat"), False)

    @property
    def name(self) -> str:
        """Human-readable engine name.

        Returns:
            str: Engine name including method and basis set.

        """
        return f"Psi4 ({self._method}/{self._basis})"

    def is_available(self) -> bool:
        """Check if Psi4 is installed.

        Returns:
            bool: ``True`` if the ``psi4`` package is importable.

        """
        return _HAS_PSI4

    def _load_molecule(self, structure: str | tuple[list[str], np.ndarray]) -> object:
        """Load a molecule from an XYZ file path or ``(atoms, coords)`` tuple.

        Args:
            structure (str | tuple): Path to an XYZ file (``str``) or a tuple of
                ``(atoms, coords)``.

        Returns:
            object: A Psi4 ``Molecule`` object with basis and reference set.

        """
        if isinstance(structure, str):
            atoms, coords = _read_xyz(structure)
        else:
            atoms, coords = structure
        mol = _make_psi4_geometry(atoms, coords, self._charge, self._multiplicity)
        ref = "rhf" if self._multiplicity == 1 else "uhf"
        _psi4.set_options({"basis": self._basis, "reference": ref})
        return mol

    def energy(
        self, structure: str | tuple[list[str], np.ndarray], method: str | None = None, basis: str | None = None
    ) -> float:
        """Calculate single-point energy in Hartrees.

        Args:
            structure (str | tuple): XYZ file path or ``(atoms, coords)`` tuple.
            method: Override the default QM method.
            basis: Override the default basis set.

        Returns:
            float: Electronic energy in Hartrees.

        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        return _psi4.energy(m, molecule=mol)

    def hessian(
        self, structure: str | tuple[list[str], np.ndarray], method: str | None = None, basis: str | None = None
    ) -> np.ndarray:
        """Calculate Hessian matrix (second derivatives of energy).

        Args:
            structure (str | tuple): XYZ file path or ``(atoms, coords)`` tuple.
            method: Override the default QM method.
            basis: Override the default basis set.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian in Hartree/Bohr².

        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _, wfn = _psi4.frequency(m, molecule=mol, return_wfn=True)
        return np.array(wfn.hessian())

    def optimize(
        self,
        structure: str | tuple[list[str], np.ndarray],
        method: str | None = None,
        basis: str | None = None,
        opt_type: str = "min",
    ) -> tuple[float, list[str], np.ndarray]:
        """Optimize geometry.

        Args:
            structure (str | tuple): XYZ file path or ``(atoms, coords)`` tuple.
            method: Override the default QM method.
            basis: Override the default basis set.
            opt_type: ``"min"`` for minimization, ``"ts"`` for transition
                state search.

        Returns:
            tuple[float, list[str], np.ndarray]: ``(energy, atoms, coords_angstrom)`` with energy in Hartrees.

        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _psi4.set_options({"opt_type": opt_type, "geom_maxiter": 100})
        energy = _psi4.optimize(m, molecule=mol)
        coords_bohr = mol.geometry().np
        coords_ang = coords_bohr * BOHR_TO_ANG
        atoms = [mol.symbol(i) for i in range(mol.natom())]
        return energy, atoms, coords_ang

    def frequencies(
        self, structure: str | tuple[list[str], np.ndarray], method: str | None = None, basis: str | None = None
    ) -> list[float]:
        """Calculate vibrational frequencies in cm⁻¹.

        Args:
            structure (str | tuple): XYZ file path or ``(atoms, coords)`` tuple.
            method: Override the default QM method.
            basis: Override the default basis set.

        Returns:
            list[float]: Vibrational frequencies in cm⁻¹.

        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _, wfn = _psi4.frequency(m, molecule=mol, return_wfn=True)
        return list(np.array(wfn.frequencies()))

    def close(self) -> None:
        """Clean up temporary files created by Psi4."""
        if hasattr(self, "_tmpdir") and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def __enter__(self) -> Psi4Engine:
        """Enter context manager.

        Returns:
            Psi4Engine: This engine instance.

        """
        return self

    def __exit__(self, *args: object) -> None:
        """Exit context manager and clean up temporary files."""
        self.close()

    def __del__(self) -> None:
        """Destructor — clean up temporary files."""
        self.close()
