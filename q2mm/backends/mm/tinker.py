"""Tinker molecular mechanics engine backend.

Wraps Tinker executables (``analyze``, ``minimize``, ``vibrate``) for MM
calculations with MM3 and other force fields.

Requires: Tinker binaries on ``PATH`` or configured via *tinker_dir*
parameter.  Download from: https://dasher.wustl.edu/tinker/
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import shutil
import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.backends.registry import register_mm
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.units import canonical_to_mm3_bond_k, canonical_to_mm3_angle_k

logger = logging.getLogger(__name__)


def _find_tinker_dir() -> str | None:
    """Auto-detect Tinker installation directory.

    Searches common installation paths and the system ``PATH`` for
    the ``analyze`` executable.

    Returns:
        str | None: Path to the Tinker bin directory, or ``None`` if not
            found.

    """
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
    """Get full path to a Tinker executable.

    Args:
        tinker_dir: Directory containing Tinker binaries.
        name: Base name of the executable (without extension).

    Returns:
        str: Full path to the executable.

    Raises:
        FileNotFoundError: If the executable is not found in *tinker_dir*.

    """
    for ext in [".exe", ""]:
        path = os.path.join(tinker_dir, name + ext)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Tinker executable '{name}' not found in {tinker_dir}")


@register_mm("tinker")
class TinkerEngine(MMEngine):
    """Molecular mechanics engine using Tinker.

    Args:
        tinker_dir: Path to Tinker bin directory (auto-detected if None)
        params_file: Path to MM3 parameter file (auto-detected if None)
        bond_tolerance: Distance multiplier for bond detection. Two atoms
            are bonded when their distance is within
            ``bond_tolerance * (r_cov_A + r_cov_B)``. Default 1.3.

    """

    def __init__(
        self, tinker_dir: str | None = None, params_file: str | None = None, bond_tolerance: float = 1.3
    ) -> None:
        """Initialize the Tinker engine.

        Args:
            tinker_dir: Path to Tinker bin directory. Auto-detected if
                ``None``.
            params_file: Path to MM3 parameter file. Auto-detected if
                ``None``.
            bond_tolerance: Distance multiplier for bond detection. Two
                atoms are bonded when their distance is within
                ``bond_tolerance * (r_cov_A + r_cov_B)``.

        Raises:
            FileNotFoundError: If Tinker binaries or the MM3 parameter file
                cannot be found.

        """
        self._tinker_dir = tinker_dir or _find_tinker_dir()
        self._bond_tolerance = bond_tolerance
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
        """Human-readable engine name.

        Returns:
            str: ``"Tinker"``.

        """
        return "Tinker"

    def supported_functional_forms(self) -> frozenset[str]:
        """Tinker with MM3 params supports MM3 functional forms only.

        Returns:
            frozenset[str]: ``{"mm3"}``.

        """
        return frozenset({"mm3"})

    def is_available(self) -> bool:
        """Check if Tinker ``analyze`` executable is accessible.

        Returns:
            bool: ``True`` if the executable can be located.

        """
        try:
            _exe(self._tinker_dir, "analyze")
            return True
        except FileNotFoundError:
            return False

    def _write_tinker_xyz(
        self, structure: str | Q2MMMolecule, forcefield: ForceField | dict[str, int] | str | None, workdir: str
    ) -> str:
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
            bonds = self._detect_bonds(atoms, coords_arr, self._bond_tolerance)
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
            self._validate_forcefield(forcefield)
            exported_prm = os.path.join(workdir, "molecule.prm")
            if forcefield.source_format == "tinker_prm" and (forcefield.source_path or self._params_file):
                # FF came from a .prm file — use template-based export
                forcefield.to_tinker_prm(
                    exported_prm,
                    template_path=forcefield.source_path or self._params_file,
                )
            else:
                # Programmatic FF — write standalone .prm with atom defs
                self._write_standalone_prm(forcefield, exported_prm, atoms, atom_type_numbers)
            params_path = exported_prm

        # Write key file
        key_path = os.path.join(workdir, "molecule.key")
        with open(key_path, "w") as f:
            f.write(f"parameters {params_path}\n")

        return txyz_path

    @staticmethod
    def _detect_bonds(atoms: list[str], coords: np.ndarray, bond_tolerance: float = 1.3) -> dict:
        """Detect bonds using distance-based shared covalent radii.

        Args:
            atoms: Element symbols for each atom.
            coords: Cartesian coordinates, shape ``(N, 3)``, in Å.
            bond_tolerance: Multiplier applied to the sum of covalent radii.

        Returns:
            dict: Mapping of atom index to list of bonded atom indices.

        """
        from q2mm.models.molecule import COVALENT_RADII

        bonds = {i: [] for i in range(len(atoms))}
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                ri = COVALENT_RADII.get(atoms[i], 0.76)
                rj = COVALENT_RADII.get(atoms[j], 0.76)
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < bond_tolerance * (ri + rj):
                    bonds[i].append(j)
                    bonds[j].append(i)
        return bonds

    # Atomic numbers and masses for standalone .prm generation
    _ATOMIC_DATA: dict[str, tuple[int, float, int]] = {
        # Fields: atomic number, mass, default valence
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

    def _write_standalone_prm(
        self, ff: ForceField, prm_path: str, atoms: list[str], atom_type_numbers: list[int]
    ) -> None:
        """Write a complete standalone Tinker .prm for a programmatic ForceField.

        Generates a self-contained parameter file with atom definitions,
        MM3 functional form headers, and only the bond/angle/vdW terms
        defined in the ForceField. This ensures Tinker evaluates exactly
        the same terms as OpenMM for cross-backend parity.

        Args:
            ff: ForceField model with bonds, angles, vdws.
            prm_path: Output path for the .prm file.
            atoms: Element symbols for each atom (same order as .xyz).
            atom_type_numbers: Tinker type numbers assigned in _write_tinker_xyz
                (guarantees XYZ ↔ PRM consistency).

        Note: This approach maps one Tinker type per element. Force fields
        that distinguish same-element params by env_id should use the
        template-based export path (source_format="tinker_prm").

        """
        # Build element → type_number map from the actual atoms + type numbers
        # used in the .xyz file (guarantees XYZ ↔ PRM consistency).
        elem_to_type: dict[str, int] = {}
        for elem, tnum in zip(atoms, atom_type_numbers, strict=False):
            if elem in elem_to_type and elem_to_type[elem] != tnum:
                raise ValueError(
                    f"Inconsistent type assignment for element {elem}: "
                    f"got {tnum} but previously assigned {elem_to_type[elem]}"
                )
            elem_to_type[elem] = tnum

        # Only skip FF parameters whose elements are known MM3 placeholder
        # labels (e.g. "00" for wildcard atom types).  Real elements that
        # aren't in the molecule are genuine errors — raise immediately so
        # they aren't silently hidden.
        _KNOWN_PLACEHOLDERS = {"00"}

        def _check_elements(elements: tuple[str, ...], label: str) -> bool:
            """Return True if all *elements* are in elem_to_type.

            Skips (with a debug log) if a placeholder is detected.
            Raises ValueError for real elements missing from the molecule.
            """
            for el in elements:
                if el in elem_to_type:
                    continue
                if el in _KNOWN_PLACEHOLDERS:
                    logger.debug("Skipping %s with placeholder element '%s'", label, el)
                    return False
                raise ValueError(
                    f"FF {label} references element '{el}' not present in molecule atoms (and not a known placeholder)"
                )
            return True

        bonds = [b for b in ff.bonds if _check_elements(b.elements, "bond")]
        angles = [a for a in ff.angles if _check_elements(a.elements, "angle")]
        vdws = []
        for v in ff.vdws:
            if not v.element or v.element in elem_to_type:
                vdws.append(v)
            elif v.element in _KNOWN_PLACEHOLDERS:
                logger.debug("Skipping vdW with placeholder element '%s'", v.element)
            else:
                raise ValueError(
                    f"FF vdW references element '{v.element}' not present "
                    f"in molecule atoms (and not a known placeholder)"
                )

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
                f.write(f'atom   {tnum:5d}    {elem:2s}    "{elem:<20s}"{anum:7d}   {mass:8.3f}    {valence}\n')
            f.write("\n")

            # Bond parameters
            for bond in bonds:
                t1 = elem_to_type[bond.elements[0]]
                t2 = elem_to_type[bond.elements[1]]
                f.write(
                    f"bond   {t1:5d} {t2:5d}         {canonical_to_mm3_bond_k(bond.force_constant):8.4f}   {bond.equilibrium:8.4f}\n"
                )

            # Angle parameters
            for angle in angles:
                t1 = elem_to_type[angle.elements[0]]
                t2 = elem_to_type[angle.elements[1]]
                t3 = elem_to_type[angle.elements[2]]
                f.write(
                    f"angle  {t1:5d} {t2:5d} {t3:5d}         {canonical_to_mm3_angle_k(angle.force_constant):8.4f}   {angle.equilibrium:8.4f}\n"
                )

            # vdW parameters
            for vdw in vdws:
                # atom_type is a string (element or Tinker type label);
                # convert to numeric Tinker type via elem_to_type
                try:
                    t = int(vdw.atom_type)
                except (ValueError, TypeError):
                    t = elem_to_type[vdw.element]
                f.write(f"vdw    {t:5d}         {vdw.radius:8.4f}   {vdw.epsilon:8.4f}\n")

    def _run_tinker(
        self, exe_name: str, xyz_path: str, args: list | None = None, stdin: str | None = None
    ) -> subprocess.CompletedProcess:
        """Run a Tinker executable.

        Args:
            exe_name: Base name of the Tinker executable (e.g. ``"analyze"``).
            xyz_path: Path to the Tinker XYZ input file.
            args: Additional command-line arguments.
            stdin: Text to pipe to the process's standard input.

        Returns:
            subprocess.CompletedProcess: Completed process result.

        Raises:
            RuntimeError: If the Tinker executable exits with a non-zero
                return code.

        """
        exe = _exe(self._tinker_dir, exe_name)
        key_path = xyz_path.replace(".xyz", ".key")
        cmd = [exe, xyz_path, "-k", key_path] + (args or [])
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, input=stdin, cwd=os.path.dirname(xyz_path)
        )
        if result.returncode != 0:
            raise RuntimeError(f"Tinker {exe_name} failed (exit {result.returncode}):\n{result.stderr}")
        return result

    def energy(self, structure: str | Q2MMMolecule, forcefield: ForceField | dict[str, int] | None = None) -> float:
        """Calculate MM energy in kcal/mol.

        Args:
            structure (str | Q2MMMolecule): Path to XYZ file or :class:`Q2MMMolecule`.
            forcefield (ForceField | dict | None): Force field or atom-type mapping. Uses default MM3
                types if ``None``.

        Returns:
            float: Total potential energy in kcal/mol.

        Raises:
            RuntimeError: If the energy cannot be parsed from Tinker output.

        """
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("analyze", txyz, ["E"])
            for line in result.stdout.split("\n"):
                if "Total Potential Energy" in line:
                    return float(line.split(":")[1].split()[0])
        raise RuntimeError(f"Could not parse energy from Tinker output:\n{result.stdout}")

    def minimize(
        self,
        structure: str | Q2MMMolecule,
        forcefield: ForceField | dict[str, int] | None = None,
        rms_grad: float = 0.01,
    ) -> tuple[float, list[str], np.ndarray]:
        """Energy-minimize structure.

        Args:
            structure (str | Q2MMMolecule): Path to XYZ file or :class:`Q2MMMolecule`.
            forcefield (ForceField | dict | None): Force field or atom-type mapping. Uses default MM3
                types if ``None``.
            rms_grad: RMS gradient convergence criterion in kcal/mol/Å.

        Returns:
            tuple[float, list[str], np.ndarray]: ``(energy, atoms, coords)``
                where energy is in kcal/mol and coords are in Å.

        Raises:
            RuntimeError: If the energy cannot be parsed from output or the
                minimized coordinate file is not found.

        """
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

    def hessian(
        self, structure: str | Q2MMMolecule, forcefield: ForceField | dict[str, int] | None = None
    ) -> np.ndarray:
        """Calculate MM Hessian matrix via Tinker ``testhess``.

        Calls ``testhess`` to compute the analytical Cartesian Hessian,
        parses the ``.hes`` output file (diagonal + upper-triangle
        off-diagonal blocks), symmetrizes, and converts to the canonical
        unit contract (Hartree/Bohr²).

        Args:
            structure (str | Q2MMMolecule): Path to XYZ file or :class:`Q2MMMolecule`.
            forcefield (ForceField | dict | None): Force field or atom-type mapping.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian in Hartree/Bohr².

        Raises:
            RuntimeError: If ``testhess`` fails or the ``.hes`` file cannot
                be parsed.

        """
        from q2mm.constants import KCALMOLA2_TO_HESSIAN_AU

        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            # Y = compute analytical Hessian, N = skip numerical comparison
            self._run_tinker("testhess", txyz, stdin="Y\nN\n")

            # Parse the .hes file written by testhess
            hes_path = txyz.replace(".xyz", ".hes")
            if not os.path.exists(hes_path):
                raise RuntimeError(f"testhess did not produce {hes_path}")

            with open(hes_path) as f:
                content = f.read()

            # Split into sections by "Diagonal" and "Off-diagonal" headers
            sections = re.split(r"\n\s*(?:Diagonal|Off-diagonal)\s+Hessian\s+Elements.*\n", content)
            # sections[0] is empty/header, sections[1] is diagonal data,
            # sections[2..] are off-diagonal blocks for each (atom, coord)

            if len(sections) < 2:
                raise RuntimeError("Could not parse .hes file: no diagonal section found")

            try:
                # Parse diagonal elements
                diag_vals = [float(v) for v in sections[1].split()]
                n3 = len(diag_vals)
                hessian = np.zeros((n3, n3))
                for i, val in enumerate(diag_vals):
                    hessian[i, i] = val

                # Parse off-diagonal blocks: one block per (row_index),
                # containing elements H[row, row+1], H[row, row+2], ..., H[row, n3-1]
                expected_blocks = n3 - 1
                row = 0
                for block_idx, block in enumerate(sections[2:]):
                    vals = [float(v) for v in block.split()]
                    if not vals:
                        continue
                    expected_vals = n3 - row - 1
                    if len(vals) != expected_vals:
                        raise ValueError(
                            f"Off-diagonal block {block_idx} (row {row}): "
                            f"expected {expected_vals} values, got {len(vals)}"
                        )
                    col_start = row + 1
                    for j, val in enumerate(vals):
                        col = col_start + j
                        hessian[row, col] = val
                        hessian[col, row] = val
                    row += 1
            except (ValueError, IndexError) as exc:
                n3_str = str(n3) if "n3" in locals() else "?"
                raise RuntimeError(
                    f"Failed to parse .hes file: {exc}. "
                    f"File had {len(sections)} sections, "
                    f"expected diagonal size {n3_str}."
                ) from exc

            # Tinker outputs Hessian in kcal/(mol·Å²); convert to Hartree/Bohr²
            return hessian * KCALMOLA2_TO_HESSIAN_AU

    def frequencies(
        self, structure: str | Q2MMMolecule, forcefield: ForceField | dict[str, int] | None = None
    ) -> list[float]:
        """Calculate vibrational frequencies in cm⁻¹.

        Args:
            structure (str | Q2MMMolecule): Path to XYZ file or :class:`Q2MMMolecule`.
            forcefield (ForceField | dict | None): Force field or atom-type mapping. Uses default MM3
                types if ``None``.

        Returns:
            list[float]: Vibrational frequencies in cm⁻¹.

        """
        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            result = self._run_tinker("vibrate", txyz, stdin="A\n")
            freqs = []
            for m in re.finditer(r"Frequency\s+([-\d.]+)\s+cm-1", result.stdout):
                freqs.append(float(m.group(1)))
            return freqs
