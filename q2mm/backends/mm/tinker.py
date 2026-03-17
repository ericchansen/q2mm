"""Tinker molecular mechanics engine backend.

Wraps Tinker executables (analyze, minimize, vibrate) for MM calculations
with MM3 and other force fields.

Requires: Tinker binaries on PATH or configured via tinker_dir parameter.
Download from: https://dasher.wustl.edu/tinker/
"""
import os
import re
import subprocess
import tempfile
import shutil
import numpy as np

from q2mm.backends.base import MMEngine


def _find_tinker_dir() -> str | None:
    """Auto-detect Tinker installation directory."""
    # Check common locations
    candidates = [
        os.path.join(os.path.expanduser("~"), "tinker", "bin-windows"),
        os.path.join(os.path.expanduser("~"), "tinker", "bin"),
        r"C:\Tinker\bin",
        r"/usr/local/bin",
        r"/opt/tinker/bin",
    ]
    for d in candidates:
        if os.path.isfile(os.path.join(d, "analyze.exe")) or \
           os.path.isfile(os.path.join(d, "analyze")):
            return d
    # Check PATH
    for name in ["analyze.exe", "analyze"]:
        path = shutil.which(name)
        if path:
            return os.path.dirname(path)
    return None


def _exe(tinker_dir: str, name: str) -> str:
    """Get full path to a Tinker executable."""
    for ext in [".exe", ""]:
        path = os.path.join(tinker_dir, name + ext)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Tinker executable '{name}' not found in {tinker_dir}")


class TinkerEngine(MMEngine):
    """Molecular mechanics engine using Tinker.

    Args:
        tinker_dir: Path to Tinker bin directory (auto-detected if None)
        params_file: Path to MM3 parameter file (auto-detected if None)
    """

    def __init__(self, tinker_dir: str = None, params_file: str = None):
        self._tinker_dir = tinker_dir or _find_tinker_dir()
        if self._tinker_dir is None:
            raise FileNotFoundError(
                "Tinker not found. Install from https://dasher.wustl.edu/tinker/ "
                "or pass tinker_dir parameter."
            )

        if params_file is None:
            # Try common locations for MM3 params
            candidates = [
                os.path.join(os.path.dirname(self._tinker_dir), "params", "mm3.prm"),
                os.path.join(self._tinker_dir, "mm3.prm"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    params_file = c
                    break
        self._params_file = params_file
        if self._params_file is None:
            raise FileNotFoundError(
                "MM3 parameter file not found. Provide params_file parameter "
                "or place mm3.prm alongside the Tinker bin directory."
            )

    @property
    def name(self) -> str:
        return "Tinker (MM3)"

    def is_available(self) -> bool:
        try:
            _exe(self._tinker_dir, "analyze")
            return True
        except FileNotFoundError:
            return False

    def _write_tinker_xyz(self, structure, forcefield, workdir: str) -> str:
        """Write a Tinker-format XYZ file from a standard XYZ file.

        Args:
            structure: Path to standard XYZ file
            forcefield: Force field object or dict mapping element -> atom type
            workdir: Directory to write the file

        Returns:
            Path to the Tinker XYZ file
        """
        # Default MM3 atom type mapping
        if forcefield is None or isinstance(forcefield, str):
            type_map = {"C": 1, "H": 5, "F": 11, "Cl": 12, "Br": 13,
                        "N": 8, "O": 6, "S": 15, "P": 25}
        elif isinstance(forcefield, dict):
            type_map = forcefield
        else:
            type_map = getattr(forcefield, 'atom_type_map', {"C": 1, "H": 5, "F": 11})

        # Read standard XYZ
        with open(structure) as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        atoms = []
        coords = []
        for line in lines[2:2 + n_atoms]:
            parts = line.split()
            atoms.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])

        # Build connectivity (simple distance-based)
        coords_arr = np.array(coords)
        bonds = self._detect_bonds(atoms, coords_arr)

        # Write Tinker XYZ
        txyz_path = os.path.join(workdir, "molecule.xyz")
        with open(txyz_path, "w") as f:
            f.write(f"     {n_atoms}  Q2MM Tinker input\n")
            for i, (atom, (x, y, z)) in enumerate(zip(atoms, coords)):
                atype = type_map.get(atom, 1)
                bonded = [str(j + 1) for j in bonds.get(i, [])]
                bond_str = "     ".join(bonded)
                f.write(f"     {i+1}  {atom:2s}  {x:12.6f} {y:12.6f} {z:12.6f}"
                        f"    {atype:2d}     {bond_str}\n")

        # Write key file
        key_path = os.path.join(workdir, "molecule.key")
        params = self._params_file or ""
        with open(key_path, "w") as f:
            f.write(f"parameters {params}\n")

        return txyz_path

    @staticmethod
    def _detect_bonds(atoms: list[str], coords: np.ndarray) -> dict:
        """Simple distance-based bond detection."""
        cov_radii = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66,
                     "F": 0.57, "Cl": 0.99, "Br": 1.14, "S": 1.05, "P": 1.07}
        bonds = {i: [] for i in range(len(atoms))}
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                ri = cov_radii.get(atoms[i], 0.76)
                rj = cov_radii.get(atoms[j], 0.76)
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < 1.3 * (ri + rj):
                    bonds[i].append(j)
                    bonds[j].append(i)
        return bonds

    def _run_tinker(self, exe_name: str, xyz_path: str, args: list = None,
                    stdin: str = None) -> subprocess.CompletedProcess:
        """Run a Tinker executable."""
        exe = _exe(self._tinker_dir, exe_name)
        key_path = xyz_path.replace(".xyz", ".key")
        cmd = [exe, xyz_path, "-k", key_path] + (args or [])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            input=stdin, cwd=os.path.dirname(xyz_path)
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Tinker {exe_name} failed (exit {result.returncode}):\n{result.stderr}"
            )
        return result

    def energy(self, structure, forcefield=None) -> float:
        """Calculate MM energy in kcal/mol."""
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("analyze", txyz, ["E"])
            for line in result.stdout.split("\n"):
                if "Total Potential Energy" in line:
                    return float(line.split(":")[1].split()[0])
        raise RuntimeError(f"Could not parse energy from Tinker output:\n{result.stdout}")

    def minimize(self, structure, forcefield=None, rms_grad: float = 0.01):
        """Energy-minimize structure. Returns (energy, atoms, coords)."""
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("minimize", txyz, [str(rms_grad)])

            # Parse final energy
            energy = None
            for line in result.stdout.split("\n"):
                if "Final Function Value" in line:
                    energy = float(line.split(":")[1].strip().split()[0])

            if energy is None:
                raise RuntimeError(
                    f"Could not parse energy from Tinker minimize output:\n{result.stdout}"
                )

            # Read minimized coordinates from .xyz_2 output
            min_xyz = txyz + "_2"
            if not os.path.exists(min_xyz):
                raise RuntimeError(f"Minimized file not found: {min_xyz}")

            atoms = []
            coords = []
            with open(min_xyz) as f:
                lines = f.readlines()
            for line in lines[1:]:
                parts = line.split()
                if len(parts) >= 6:
                    atoms.append(parts[1])
                    coords.append([float(parts[2]), float(parts[3]), float(parts[4])])

            return energy, atoms, np.array(coords)

    def hessian(self, structure, forcefield=None) -> np.ndarray:
        """Calculate MM Hessian matrix.

        Note: Full Hessian extraction from Tinker requires the testhess program.
        Use frequencies() for vibrational analysis instead.
        """
        raise NotImplementedError(
            "Full Hessian extraction not yet implemented. Use frequencies() instead."
        )

    def frequencies(self, structure, forcefield=None) -> list[float]:
        """Calculate vibrational frequencies in cm^-1."""
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("vibrate", txyz, stdin="A\n")
            freqs = []
            for m in re.finditer(r"Frequency\s+([-\d.]+)\s+cm-1", result.stdout):
                freqs.append(float(m.group(1)))
            return freqs
