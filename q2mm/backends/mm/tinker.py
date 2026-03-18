"""Tinker molecular mechanics engine backend.

Wraps Tinker executables (analyze, minimize, vibrate) for MM calculations
with MM3 and other force fields.

Requires: Tinker binaries on PATH or configured via tinker_dir parameter.
Download from: https://dasher.wustl.edu/tinker/
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import shutil
import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.models.molecule import Q2MMMolecule


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
        if os.path.isfile(os.path.join(d, "analyze.exe")) or os.path.isfile(os.path.join(d, "analyze")):
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
                "Tinker not found. Install from https://dasher.wustl.edu/tinker/ or pass tinker_dir parameter."
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
        _default_type_map = {"C": 1, "H": 5, "F": 11, "Cl": 12, "Br": 13, "N": 8, "O": 6, "S": 15, "P": 25}
        if forcefield is None or isinstance(forcefield, str):
            type_map = _default_type_map
        elif isinstance(forcefield, dict):
            type_map = forcefield
        else:
            type_map = getattr(forcefield, "atom_type_map", _default_type_map)

        if isinstance(structure, Q2MMMolecule):
            atoms = list(structure.symbols)
            coords = np.asarray(structure.geometry, dtype=float).tolist()
            n_atoms = len(atoms)
            bonds = {i: [] for i in range(n_atoms)}
            for bond in structure.bonds:
                bonds[bond.atom_i].append(bond.atom_j)
                bonds[bond.atom_j].append(bond.atom_i)
            atom_type_numbers = []
            for atom, atom_type in zip(atoms, structure.atom_types, strict=False):
                try:
                    atom_type_numbers.append(int(atom_type))
                except (TypeError, ValueError):
                    atom_type_numbers.append(type_map.get(atom, 1))
        else:
            # Read standard XYZ
            with open(structure) as f:
                lines = f.readlines()
            n_atoms = int(lines[0].strip())
            atoms = []
            coords = []
            for line in lines[2 : 2 + n_atoms]:
                parts = line.split()
                atoms.append(parts[0])
                coords.append([float(x) for x in parts[1:4]])

            # Build connectivity (simple distance-based)
            coords_arr = np.array(coords)
            bonds = self._detect_bonds(atoms, coords_arr)
            atom_type_numbers = [type_map.get(atom, 1) for atom in atoms]

        # Write Tinker XYZ
        txyz_path = os.path.join(workdir, "molecule.xyz")
        with open(txyz_path, "w") as f:
            f.write(f"     {n_atoms}  Q2MM Tinker input\n")
            for i, (atom, (x, y, z), atype) in enumerate(zip(atoms, coords, atom_type_numbers, strict=False)):
                bonded = [str(j + 1) for j in bonds.get(i, [])]
                bond_str = "     ".join(bonded)
                f.write(f"     {i + 1}  {atom:2s}  {x:12.6f} {y:12.6f} {z:12.6f}    {atype:2d}     {bond_str}\n")

        # Determine which parameter file to use
        params_path = self._params_file or ""

        # If a ForceField model was supplied, export its (possibly modified)
        # parameters to a workdir .prm so Tinker evaluates the updated values.
        from q2mm.models.forcefield import ForceField

        if isinstance(forcefield, ForceField):
            exported_prm = os.path.join(workdir, "molecule.prm")
            if forcefield.source_format == "tinker_prm" and (forcefield.source_path or self._params_file):
                # FF came from a .prm file — use template-based export
                forcefield.to_tinker_prm(
                    exported_prm,
                    template_path=forcefield.source_path or self._params_file,
                )
            else:
                # Programmatic FF — write standalone .prm with atom defs
                self._write_standalone_prm(forcefield, exported_prm, atom_type_numbers)
            params_path = exported_prm

        # Write key file
        key_path = os.path.join(workdir, "molecule.key")
        with open(key_path, "w") as f:
            f.write(f"parameters {params_path}\n")

        return txyz_path

    @staticmethod
    def _detect_bonds(atoms: list[str], coords: np.ndarray) -> dict:
        """Simple distance-based bond detection."""
        cov_radii = {
            "H": 0.31,
            "C": 0.76,
            "N": 0.71,
            "O": 0.66,
            "F": 0.57,
            "Cl": 0.99,
            "Br": 1.14,
            "S": 1.05,
            "P": 1.07,
        }
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

    # Atomic numbers and masses for standalone .prm generation
    _ATOMIC_DATA: dict[str, tuple[int, float, int]] = {
        # element: (atomic_number, mass, default_valence)
        "H": (1, 1.008, 1),
        "He": (2, 4.003, 0),
        "C": (6, 12.011, 4),
        "N": (7, 14.007, 3),
        "O": (8, 15.999, 2),
        "F": (9, 18.998, 1),
        "P": (15, 30.974, 3),
        "S": (16, 32.060, 2),
        "Cl": (17, 35.453, 1),
        "Br": (35, 79.904, 1),
    }

    def _write_standalone_prm(self, ff, prm_path: str, atom_type_numbers: list[int]):
        """Write a complete standalone Tinker .prm for a programmatic ForceField.

        Generates a self-contained parameter file with atom definitions,
        MM3 functional form headers, and only the bond/angle/vdW terms
        defined in the ForceField. This ensures Tinker evaluates exactly
        the same terms as OpenMM for cross-backend parity.
        """
        from q2mm.models.forcefield import ForceField

        # Build element → type_number map from the atom_type_numbers used in .xyz
        # Must use the same default type_map as _write_tinker_xyz
        elem_to_type: dict[str, int] = {}
        type_map = getattr(ff, "atom_type_map", None) or {
            "C": 1, "H": 5, "F": 11, "Cl": 12, "Br": 13, "N": 8, "O": 6, "S": 15, "P": 25
        }
        # Collect all unique types referenced by the FF
        for b in ff.bonds:
            for e in b.elements:
                if e not in elem_to_type:
                    elem_to_type[e] = type_map.get(e, len(elem_to_type) + 1)
        for a in ff.angles:
            for e in a.elements:
                if e not in elem_to_type:
                    elem_to_type[e] = type_map.get(e, len(elem_to_type) + 1)

        with open(prm_path, "w") as f:
            # MM3 functional form header (matches mm3.prm conventions)
            f.write("forcefield          Q2MM-Custom\n\n")
            f.write("bondunit                71.94\n")
            f.write("bond-cubic              -2.55\n")
            f.write("bond-quartic            3.793125\n")
            f.write("angleunit               0.02191418\n")
            f.write("angle-cubic             -0.014\n")
            f.write("angle-quartic           0.000056\n")
            f.write("angle-pentic            -0.0000007\n")
            f.write("angle-sextic            0.000000022\n\n")

            # Atom definitions (MM3 format: type, symbol, "description", anum, mass, valence)
            for elem, tnum in sorted(elem_to_type.items(), key=lambda x: x[1]):
                anum, mass, valence = self._ATOMIC_DATA.get(elem, (0, 0.0, 1))
                f.write(f'atom   {tnum:5d}    {elem:2s}    "{elem:<20s}"'
                        f'{anum:7d}   {mass:8.3f}    {valence}\n')
            f.write("\n")

            # Bond parameters
            for bond in ff.bonds:
                t1 = elem_to_type.get(bond.elements[0], 1)
                t2 = elem_to_type.get(bond.elements[1], 1)
                f.write(f"bond   {t1:5d} {t2:5d}         {bond.force_constant:8.4f}   "
                        f"{bond.equilibrium:8.4f}\n")

            # Angle parameters
            for angle in ff.angles:
                t1 = elem_to_type.get(angle.elements[0], 1)
                t2 = elem_to_type.get(angle.elements[1], 1)
                t3 = elem_to_type.get(angle.elements[2], 1)
                f.write(f"angle  {t1:5d} {t2:5d} {t3:5d}         {angle.force_constant:8.4f}   "
                        f"{angle.equilibrium:8.4f}\n")

            # vdW parameters
            for vdw in ff.vdws:
                t = vdw.atom_type or elem_to_type.get(vdw.elements[0] if vdw.elements else "", 1)
                f.write(f"vdw    {t:5d}         {vdw.radius:8.4f}   {vdw.epsilon:8.4f}\n")

    def _run_tinker(
        self, exe_name: str, xyz_path: str, args: list = None, stdin: str = None
    ) -> subprocess.CompletedProcess:
        """Run a Tinker executable."""
        exe = _exe(self._tinker_dir, exe_name)
        key_path = xyz_path.replace(".xyz", ".key")
        cmd = [exe, xyz_path, "-k", key_path] + (args or [])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, input=stdin, cwd=os.path.dirname(xyz_path)
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tinker {exe_name} failed (exit {result.returncode}):\n{result.stderr}")
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
                raise RuntimeError(f"Could not parse energy from Tinker minimize output:\n{result.stdout}")

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
        raise NotImplementedError("Full Hessian extraction not yet implemented. Use frequencies() instead.")

    def frequencies(self, structure, forcefield=None) -> list[float]:
        """Calculate vibrational frequencies in cm^-1."""
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("vibrate", txyz, stdin="A\n")
            freqs = []
            for m in re.finditer(r"Frequency\s+([-\d.]+)\s+cm-1", result.stdout):
                freqs.append(float(m.group(1)))
            return freqs
