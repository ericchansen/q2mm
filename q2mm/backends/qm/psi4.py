"""Psi4 quantum mechanics engine backend.

Wraps the Psi4 Python API for QM calculations:
energy, hessian, geometry optimization, and vibrational frequencies.

Requires: conda install psi4 -c conda-forge
"""
import os
import shutil
import tempfile
import numpy as np

from q2mm.backends.base import QMEngine

try:
    import psi4 as _psi4
    _HAS_PSI4 = True
except ImportError:
    _psi4 = None
    _HAS_PSI4 = False


def _read_xyz(path: str) -> tuple[list[str], np.ndarray]:
    """Read an XYZ file, return (atom_labels, coordinates_angstrom)."""
    with open(path) as f:
        lines = f.readlines()
    n_atoms = int(lines[0].strip())
    atoms = []
    coords = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append([float(x) for x in parts[1:4]])
    return atoms, np.array(coords)


def _make_psi4_geometry(atoms: list[str], coords: np.ndarray,
                        charge: int = 0, multiplicity: int = 1):
    """Create a Psi4 molecule object from atoms and coordinates."""
    geom_str = f"    {charge} {multiplicity}\n"
    for atom, (x, y, z) in zip(atoms, coords):
        geom_str += f"    {atom} {x:.10f} {y:.10f} {z:.10f}\n"
    return _psi4.geometry(geom_str)


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

    def __init__(self, method: str = "b3lyp", basis: str = "6-31+G(d)",
                 memory: str = "2 GB", n_threads: int = 4,
                 charge: int = 0, multiplicity: int = 1):
        if not _HAS_PSI4:
            raise ImportError(
                "Psi4 is not installed. Install via: conda install psi4 -c conda-forge"
            )
        self._method = method
        self._basis = basis
        self._charge = charge
        self._multiplicity = multiplicity
        _psi4.set_memory(memory)
        _psi4.set_num_threads(n_threads)
        # Suppress output by default
        self._tmpdir = tempfile.mkdtemp(prefix="q2mm_psi4_")
        _psi4.core.set_output_file(
            os.path.join(self._tmpdir, "psi4_output.dat"), False
        )

    @property
    def name(self) -> str:
        return f"Psi4 ({self._method}/{self._basis})"

    def is_available(self) -> bool:
        return _HAS_PSI4

    def _load_molecule(self, structure):
        """Load a molecule from an XYZ file path or (atoms, coords) tuple."""
        if isinstance(structure, str):
            atoms, coords = _read_xyz(structure)
        else:
            atoms, coords = structure
        mol = _make_psi4_geometry(atoms, coords, self._charge, self._multiplicity)
        _psi4.set_options({"basis": self._basis, "reference": "rhf"})
        return mol

    def energy(self, structure, method: str = None, basis: str = None) -> float:
        """Calculate single-point energy in Hartrees."""
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        return _psi4.energy(m, molecule=mol)

    def hessian(self, structure, method: str = None, basis: str = None) -> np.ndarray:
        """Calculate Hessian matrix (second derivatives of energy).

        Returns:
            np.ndarray of shape (3*N, 3*N) in Hartree/Bohr^2
        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _, wfn = _psi4.frequency(m, molecule=mol, return_wfn=True)
        return np.array(wfn.hessian())

    def optimize(self, structure, method: str = None, basis: str = None,
                 opt_type: str = "min"):
        """Optimize geometry. Returns (energy, atoms, coords_angstrom).

        Args:
            opt_type: "min" for minimization, "ts" for transition state
        """
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _psi4.set_options({"opt_type": opt_type, "geom_maxiter": 100})
        energy = _psi4.optimize(m, molecule=mol)
        coords_bohr = mol.geometry().np
        coords_ang = coords_bohr * 0.529177  # Bohr to Angstrom
        atoms = [mol.symbol(i) for i in range(mol.natom())]
        return energy, atoms, coords_ang

    def frequencies(self, structure, method: str = None, basis: str = None) -> list[float]:
        """Calculate vibrational frequencies in cm^-1."""
        mol = self._load_molecule(structure)
        m = method or self._method
        if basis:
            _psi4.set_options({"basis": basis})
        _, wfn = _psi4.frequency(m, molecule=mol, return_wfn=True)
        return list(np.array(wfn.frequencies()))

    def __del__(self):
        if hasattr(self, '_tmpdir') and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
