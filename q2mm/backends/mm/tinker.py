"""Tinker molecular mechanics backend.

Wraps Tinker executables (``analyze``, ``minimize``, ``testhess``) for MM
calculations with the MM3 functional form.

Requires: Tinker binaries on ``PATH`` or configured via *tinker_dir*
parameter.  Download from: https://dasher.wustl.edu/tinker/

Tinker is a subprocess backend: each evaluation writes a fresh Tinker XYZ and
parameter file and shells out.  There is no reusable native state, so the
backend does **not** declare :attr:`~q2mm.backends.contracts.Capability.REUSABLE_STATE`.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import shutil
from numbers import Integral

import numpy as np

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    EvaluationError,
    FrequencyRequest,
    FrequencyResult,
    FrequencyUnit,
    GeometryResult,
    HessianRequest,
    HessianResult,
    HessianUnit,
    LengthUnit,
    MinimizationRequest,
    PreparationError,
    PreparationRequest,
    readonly_array,
)
from q2mm.constants import (
    DEFAULT_BOND_TOLERANCE,
    MM3_BOND_C3,
    MM3_BOND_C4,
    MM3_ANGLE_C3,
    MM3_ANGLE_C4,
    MM3_ANGLE_C5,
    TINKER_BONDUNIT,
    TINKER_ANGLEUNIT,
)
from q2mm.io.tinker import (
    _require_finite_tinker_values,
    _require_tinker_token,
    _require_unique_tinker_types,
    _tinker_atom_type_number,
    _tinker_torsion_terms,
    _validate_tinker_export_terms,
    _validate_tinker_record_lengths,
)
from q2mm.models.forcefield import ForceField, TorsionParam
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout
from q2mm.models.units import canonical_to_mm3_bond_k, canonical_to_mm3_angle_k

logger = logging.getLogger(__name__)

_TINKER_PROVENANCE = BackendProvenance(
    backend="tinker",
    role=BackendRole.MM,
    details={"implementation": {"name": "Tinker"}, "model": {"functional_form": "MM3"}},
)
_TINKER_INFO = BackendInfo(
    name="Tinker",
    role=BackendRole.MM,
    capabilities=frozenset(
        {
            Capability.ENERGY,
            Capability.MINIMIZE,
            Capability.HESSIAN,
            Capability.FREQUENCIES,
        }
    ),
    functional_forms=frozenset({"mm3"}),
    provenance=_TINKER_PROVENANCE,
)


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


def _validate_form(forcefield: ForceField, info: BackendInfo) -> None:
    """Raise ``PreparationError`` if the FF functional form is unsupported.

    Args:
        forcefield: Force field whose functional form is checked.
        info: Backend info declaring supported functional forms.

    Raises:
        PreparationError: If the form is not declared in *info*.

    """
    form = forcefield.functional_form.value
    if not info.supports_form(form):
        raise PreparationError(
            f"{info.name} does not support functional form {form!r}. Supported: {sorted(info.functional_forms)}"
        )


def _require_positive_tinker_type(number: int) -> None:
    if isinstance(number, bool) or not isinstance(number, Integral) or number <= 0:
        raise ValueError(f"Tinker molecule atom types must be positive integers, got {number!r}")


class TinkerBackend:
    """Molecular mechanics backend using the Tinker executables.

    Args:
        tinker_dir: Path to Tinker bin directory (auto-detected if None)
        params_file: Path to MM3 parameter file (auto-detected if None)
        bond_tolerance: Distance multiplier for bond detection. Two atoms
            are bonded when their distance is within
            ``bond_tolerance * (r_cov_A + r_cov_B)``. Default 1.3.

    """

    def __init__(
        self,
        tinker_dir: str | None = None,
        params_file: str | None = None,
        bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
    ) -> None:
        """Initialize the Tinker backend.

        Args:
            tinker_dir: Path to Tinker bin directory. Auto-detected if
                ``None``.
            params_file: Path to MM3 parameter file. Auto-detected if
                ``None``.
            bond_tolerance: Distance multiplier for bond detection. Two
                atoms are bonded when their distance is within
                ``bond_tolerance * (r_cov_A + r_cov_B)``.

        Raises:
            BackendUnavailableError: If Tinker binaries or the MM3 parameter
                file cannot be found.

        """
        from q2mm.backends.contracts import BackendUnavailableError

        self._tinker_dir = tinker_dir or _find_tinker_dir()
        self._bond_tolerance = bond_tolerance
        if self._tinker_dir is None:
            raise BackendUnavailableError(
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
            raise BackendUnavailableError(
                "MM3 parameter file not found. Provide params_file parameter "
                "or place mm3.prm alongside the Tinker bin directory."
            )

    @property
    def info(self) -> BackendInfo:
        """Immutable capability declaration for this backend."""
        return _TINKER_INFO

    def prepare(self, request: PreparationRequest) -> PreparedTinker:
        """Build a prepared session for one training case.

        Args:
            request: Preparation request carrying the molecule and base MM3
                force field.

        Returns:
            PreparedTinker: A per-case session.

        Raises:
            PreparationError: If no force field is supplied or its functional
                form or populated terms are unsupported.

        """
        if request.force_field is None:
            raise PreparationError("Tinker requires a base ForceField in the PreparationRequest.")
        _validate_form(request.force_field, _TINKER_INFO)
        use_template = request.force_field.source_format == "tinker_prm" and bool(
            request.force_field.source_path or self._params_file
        )
        _validate_tinker_export_terms(
            request.force_field,
            supports_reduction=use_template,
            supports_placeholder_elements=use_template,
            error_type=PreparationError,
        )
        try:
            atom_types = self._molecule_type_numbers(request.molecule)
            self._tinker_xyz_lines(request.molecule, atom_types)
            if not use_template:
                self._standalone_prm_lines(request.force_field, list(request.molecule.symbols), atom_types)
        except ValueError as exc:
            raise PreparationError(f"Tinker preparation failed: {exc}") from exc
        layout = ParameterLayout.from_force_field(request.force_field)
        return PreparedTinker(
            backend=self,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=layout,
        )

    def _write_tinker_xyz(self, molecule: Molecule, forcefield: ForceField, workdir: str) -> str:
        """Write a Tinker-format XYZ + key file for a Molecule and MM3 force field.

        Args:
            molecule: The molecule to write (Tinker types come from its
                ``atom_types``, falling back to the MM3 element map).
            forcefield: The MM3 force field whose (possibly modified) parameters
                are exported to a workdir ``.prm``.
            workdir: Directory to write the files.

        Returns:
            str: Path to the Tinker XYZ file.

        """
        _validate_form(forcefield, _TINKER_INFO)
        use_template = forcefield.source_format == "tinker_prm" and bool(forcefield.source_path or self._params_file)
        _validate_tinker_export_terms(
            forcefield,
            supports_reduction=use_template,
            supports_placeholder_elements=use_template,
            error_type=PreparationError,
        )

        atoms = list(molecule.symbols)
        atom_type_numbers = self._molecule_type_numbers(molecule)
        xyz_lines = self._tinker_xyz_lines(molecule, atom_type_numbers)

        exported_prm = os.path.join(workdir, "molecule.prm")
        key_lines = [f"parameters {exported_prm}\n"]
        _validate_tinker_record_lengths(key_lines)
        # All XYZ/key formatting precedes parameter export; both parameter
        # writers stage and validate their records before opening output.
        if use_template:
            # FF came from a .prm file — use template-based export
            from q2mm.io.tinker import save_tinker_prm

            save_tinker_prm(
                forcefield,
                exported_prm,
                template_path=forcefield.source_path or self._params_file,
            )
        else:
            # Programmatic FF — write standalone .prm with atom defs
            self._write_standalone_prm(forcefield, exported_prm, atoms, atom_type_numbers)

        # Write Tinker XYZ
        txyz_path = os.path.join(workdir, "molecule.xyz")
        with open(txyz_path, "w", encoding="utf-8") as f:
            f.writelines(xyz_lines)

        # Write key file
        key_path = os.path.join(workdir, "molecule.key")
        with open(key_path, "w", encoding="utf-8") as f:
            f.writelines(key_lines)

        return txyz_path

    @staticmethod
    def _molecule_type_numbers(molecule: Molecule) -> list[int]:
        """Resolve actual XYZ types without guessing an unknown element's class."""
        default_types = {"C": 1, "H": 5, "F": 11, "Cl": 12, "Br": 13, "N": 8, "O": 6, "S": 15, "P": 25}
        numbers = []
        for atom, atom_type in zip(molecule.symbols, molecule.atom_types, strict=True):
            number = _tinker_atom_type_number(atom_type)
            if number is None:
                if atom not in default_types:
                    raise ValueError(f"Tinker has no default atom type for element {atom!r}")
                number = default_types[atom]
            _require_positive_tinker_type(number)
            numbers.append(number)
        return numbers

    @staticmethod
    def _tinker_xyz_lines(molecule: Molecule, atom_type_numbers: list[int]) -> list[str]:
        """Stage XYZ records so invalid coordinates cannot replace any inputs."""
        bonds: dict[int, list[int]] = {i: [] for i in range(molecule.n_atoms)}
        for bond in molecule.bonds:
            bonds[bond.atom_i].append(bond.atom_j)
            bonds[bond.atom_j].append(bond.atom_i)
        lines = [f"     {molecule.n_atoms}  Q2MM Tinker input\n"]
        for i, (atom, (x, y, z), atype) in enumerate(
            zip(molecule.symbols, molecule.geometry, atom_type_numbers, strict=True)
        ):
            _require_tinker_token(atom, "XYZ atom symbol")
            _require_finite_tinker_values(x, y, z)
            bond_str = "     ".join(str(j + 1) for j in bonds[i])
            lines.append(f"     {i + 1}  {atom:2s}  {x:12.6f} {y:12.6f} {z:12.6f}    {atype:2d}     {bond_str}\n")
        _validate_tinker_record_lengths(lines)
        return lines

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
        """Write the supported standalone Tinker model for a programmatic ForceField.

        Generates a self-contained parameter file with atom definitions,
        native MM3 functional form headers, and bond/angle/proper-torsion/vdW
        terms. Unsupported populated terms raise ``PreparationError``.
        Invalid native bindings, scalars and records raise ``ValueError``.
        All validation and formatting completes before output is opened.

        Args:
            ff: ForceField model with bonds, angles, torsions, and vdws.
            prm_path: Output path for the .prm file.
            atoms: Element symbols for each atom (same order as .xyz).
            atom_type_numbers: Tinker type numbers assigned in _write_tinker_xyz
                (guarantees XYZ ↔ PRM consistency).

        Note: This approach maps one Tinker type per element. Force fields
        that distinguish same-element params by env_id should use the
        template-based export path (source_format="tinker_prm").

        """
        lines = self._standalone_prm_lines(ff, atoms, atom_type_numbers)
        with open(prm_path, "w", encoding="utf-8") as f:
            f.writelines(lines)

    def _standalone_prm_lines(self, ff: ForceField, atoms: list[str], atom_type_numbers: list[int]) -> list[str]:
        """Stage the complete native model with molecule-authoritative classes."""
        _validate_tinker_export_terms(
            ff, supports_reduction=False, supports_placeholder_elements=False, error_type=PreparationError
        )
        elem_to_type: dict[str, int] = {}
        type_to_elem: dict[int, str] = {}
        for elem, tnum in zip(atoms, atom_type_numbers, strict=True):
            _require_positive_tinker_type(tnum)
            if elem not in self._ATOMIC_DATA:
                raise ValueError(f"Tinker standalone has no atomic data for element {elem!r}")
            if elem in elem_to_type and elem_to_type[elem] != tnum:
                raise ValueError(
                    f"Inconsistent type assignment for element {elem}: "
                    f"got {tnum} but previously assigned {elem_to_type[elem]}"
                )
            if tnum in type_to_elem and type_to_elem[tnum] != elem:
                raise ValueError(
                    f"Inconsistent element assignment for type {tnum}: got {elem} but previously assigned {type_to_elem[tnum]}"
                )
            elem_to_type[elem] = tnum
            type_to_elem[tnum] = elem

        def _check_elements(elements: tuple[str, ...], label: str) -> None:
            for el in elements:
                if el not in elem_to_type:
                    raise ValueError(f"FF {label} references element '{el}' not present in molecule atoms")

        for label, terms, size in (("bond", ff.bonds, 2), ("angle", ff.angles, 3), ("torsion", ff.torsions, 4)):
            for param in terms:
                if len(param.elements) != size:
                    raise ValueError(f"FF {label} requires {size} elements")
                _check_elements(param.elements, label)
        vdw_type_numbers = []
        for v in ff.vdws:
            number = _tinker_atom_type_number(v.atom_type)
            if number is None:
                _check_elements((v.element,), "vdW")
                number = elem_to_type[v.element]
            else:
                if number not in type_to_elem:
                    raise ValueError(f"FF vdW class {number} is not assigned to any molecule atom")
                # Numeric-only labels infer a numeric element in VdwParam.
                # The model does not retain explicit-vs-inferred provenance;
                # an equivalent numeric element is a class label, not chemistry.
                if v.element and _tinker_atom_type_number(v.element) != number and v.element != type_to_elem[number]:
                    raise ValueError(
                        f"FF vdW class {number} belongs to element {type_to_elem[number]!r}, not {v.element!r}"
                    )
            vdw_type_numbers.append(number)

        # Each string is one physical record for the native 240-byte guard.
        lines = [
            "forcefield          Q2MM-Custom\n",
            "\n",
            f"bondunit                {TINKER_BONDUNIT}\n",
            f"bond-cubic              {-MM3_BOND_C3}\n",
            f"bond-quartic            {MM3_BOND_C4}\n",
            f"angleunit               {TINKER_ANGLEUNIT}\n",
            f"angle-cubic             {MM3_ANGLE_C3}\n",
            f"angle-quartic           {MM3_ANGLE_C4:.6f}\n",
            f"angle-pentic            {MM3_ANGLE_C5:.7f}\n",
            # Tinker's angle-sextic differs from the canonical MM3 coefficient.
            "angle-sextic            0.000000022\n",
            "\n",
        ]
        for elem, tnum in sorted(elem_to_type.items(), key=lambda x: x[1]):
            anum, mass, valence = self._ATOMIC_DATA[elem]
            lines.append(f'atom   {tnum:5d}    {elem:2s}    "{elem:<20s}"{anum:7d}   {mass:8.3f}    {valence}\n')
        lines.append("\n")

        identities: set[tuple[str, tuple[str, ...]]] = set()

        for bond in ff.bonds:
            types = tuple(elem_to_type[e] for e in bond.elements)
            _require_unique_tinker_types(identities, "bond", types)
            t1, t2 = types
            _require_finite_tinker_values(bond.force_constant, bond.equilibrium)
            native_k = canonical_to_mm3_bond_k(bond.force_constant)
            _require_finite_tinker_values(native_k)
            lines.append(f"bond   {t1:5d} {t2:5d}         {native_k:8.4f}   {bond.equilibrium:8.4f}\n")

        for angle in ff.angles:
            types = tuple(elem_to_type[e] for e in angle.elements)
            _require_unique_tinker_types(identities, "angle", types)
            t1, t2, t3 = types
            _require_finite_tinker_values(angle.force_constant, angle.equilibrium)
            native_k = canonical_to_mm3_angle_k(angle.force_constant)
            _require_finite_tinker_values(native_k)
            lines.append(f"angle  {t1:5d} {t2:5d} {t3:5d}         {native_k:8.4f}   {angle.equilibrium:8.4f}\n")

        torsion_groups: dict[tuple[int, ...], list[TorsionParam]] = {}
        for tp in ff.torsions:
            types = tuple(elem_to_type[e] for e in tp.elements)
            torsion_groups.setdefault(min(types, types[::-1]), []).append(tp)
        for types, tps in torsion_groups.items():
            raw_parts = ["torsion", *(str(t) for t in types)]
            for tp in tps:
                _require_finite_tinker_values(tp.force_constant, tp.phase)
                raw_parts.extend((str(tp.force_constant), str(tp.phase), str(tp.periodicity)))
            terms = _tinker_torsion_terms(raw_parts, len(lines) + 1)
            parts = []
            for amplitude, phase, fold in sorted(terms, key=lambda term: term[2]):
                parts.extend([f"{amplitude:8.4f}", f"{phase:6.1f}", f" {fold}"])
            lines.append(f"torsion {types[0]:4d} {types[1]:4d} {types[2]:4d} {types[3]:4d}  {'  '.join(parts)}\n")

        for vdw, t in zip(ff.vdws, vdw_type_numbers, strict=True):
            _require_unique_tinker_types(identities, "vdW", (t,))
            _require_finite_tinker_values(vdw.radius, vdw.epsilon)
            lines.append(f"vdw    {t:5d}         {vdw.radius:8.4f}   {vdw.epsilon:8.4f}\n")
        _validate_tinker_record_lengths(lines)
        return lines

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

    def _evaluate_energy(self, structure: Molecule, forcefield: ForceField) -> float:
        """Calculate MM energy in kcal/mol.

        Args:
            structure (Molecule): Molecule to evaluate.
            forcefield (ForceField): Force field with the parameter values.

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

    def _evaluate_minimize(
        self,
        structure: Molecule,
        forcefield: ForceField,
        rms_grad: float = 0.01,
        *,
        max_iterations: int | None = None,
    ) -> tuple[float, list[str], np.ndarray]:
        """Energy-minimize structure.

        Args:
            structure (Molecule): Molecule to minimize.
            forcefield (ForceField): Force field with the parameter values.
            rms_grad: RMS gradient convergence criterion in kcal/mol/Å.
            max_iterations: Native ``MAXITER`` limit, or ``None`` to leave
                the native default unchanged.

        Returns:
            tuple[float, list[str], np.ndarray]: ``(energy, atoms, coords)``
                where energy is in kcal/mol and coords are in Å.

        Raises:
            EvaluationError: If the iteration limit is not representable
                as a positive native integer.
            RuntimeError: If the energy cannot be parsed from output or the
                minimized coordinate file is not found.

        References:
            Tinker's ``minimize`` calls ``lbfgs``, which reads ``MAXITER``
            from the key file:
            https://github.com/TinkerTools/tinker/blob/87050685eff8840d312e2a332cc82c33f63c7c3d/source/lbfgs.f

        """
        if max_iterations is not None and (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, (int, np.integer))
            or not 1 <= max_iterations < 2**31
        ):
            raise EvaluationError("Tinker max_iterations must be an integer between 1 and 2147483647.")

        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            if max_iterations is not None:
                with open(os.path.splitext(txyz)[0] + ".key", "a", encoding="utf-8") as key:
                    key.write(f"MAXITER {max_iterations}\n")
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

    def _evaluate_hessian(self, structure: Molecule, forcefield: ForceField) -> np.ndarray:
        """Calculate MM Hessian matrix via Tinker ``testhess``.

        Calls ``testhess`` to compute the analytical Cartesian Hessian,
        parses the ``.hes`` output file (diagonal + upper-triangle
        off-diagonal blocks), symmetrizes, and converts to the canonical
        unit contract (Hartree/Bohr²).

        Args:
            structure (Molecule): Molecule to evaluate.
            forcefield (ForceField): Force field with the parameter values.

        Returns:
            np.ndarray: Shape ``(3N, 3N)`` Hessian in Hartree/Bohr².

        Raises:
            RuntimeError: If ``testhess`` fails.
            EvaluationError: If the ``.hes`` file is missing, incomplete,
                or cannot be parsed.

        """
        from q2mm.constants import KCALMOLA2_TO_HESSIAN_AU

        with tempfile.TemporaryDirectory(prefix="q2mm_tinker_") as workdir:
            txyz = self._write_tinker_xyz(structure, forcefield, workdir)
            # Y = compute analytical Hessian, N = skip numerical comparison
            self._run_tinker("testhess", txyz, stdin="Y\nN\n")

            # Parse the .hes file written by testhess
            hes_path = txyz.replace(".xyz", ".hes")
            if not os.path.exists(hes_path):
                raise EvaluationError(f"Tinker testhess did not produce {hes_path}")

            with open(hes_path) as f:
                content = f.read()

            # Keep native atom/axis labels; line wrapping does not start a new block.
            headers = list(
                re.finditer(
                    r"^[ \t]*(Diagonal|Off-diagonal)[ \t]+Hessian[ \t]+Elements([^\r\n]*)",
                    content,
                    re.MULTILINE,
                )
            )
            n3 = 3 * structure.n_atoms
            try:
                if not headers or headers[0].group(1) != "Diagonal":
                    raise ValueError("no diagonal section found before off-diagonal blocks")
                if any(header.group(1) == "Diagonal" for header in headers[1:]):
                    raise ValueError("unexpected diagonal section after the first section")
                sections = [
                    content[header.end() : headers[i + 1].start() if i + 1 < len(headers) else len(content)]
                    for i, header in enumerate(headers)
                ]
                diag_vals = [float(v.replace("D", "E").replace("d", "e")) for v in sections[0].split()]
                if len(diag_vals) != n3:
                    raise ValueError(f"expected {n3} diagonal values, got {len(diag_vals)}")
                expected_blocks = n3 - 1
                if len(sections) - 1 != expected_blocks:
                    raise ValueError(f"expected {expected_blocks} off-diagonal blocks, got {len(sections) - 1}")

                hessian = np.empty((n3, n3))
                np.fill_diagonal(hessian, diag_vals)
                for row, (header, block) in enumerate(zip(headers[1:], sections[1:], strict=True)):
                    atom, axis = row // 3 + 1, "XYZ"[row % 3]
                    label = header.group(2).strip()
                    if label:
                        identity = re.fullmatch(r"for\s+Atom\s+(\d+)\s+([XYZ])", label)
                        if identity is None or (int(identity.group(1)), identity.group(2)) != (atom, axis):
                            raise ValueError(f"Off-diagonal block {row}: expected Atom {atom} {axis}, got {label!r}")
                    vals = [float(v.replace("D", "E").replace("d", "e")) for v in block.split()]
                    expected_vals = n3 - row - 1
                    if len(vals) != expected_vals:
                        raise ValueError(
                            f"Off-diagonal block {row} (Atom {atom} {axis}): "
                            f"expected {expected_vals} values, got {len(vals)}"
                        )
                    hessian[row, row + 1 :] = vals
                    hessian[row + 1 :, row] = vals
            except ValueError as exc:
                raise EvaluationError(f"Failed to parse Tinker Hessian file {hes_path}: {exc}") from exc

            # Tinker outputs Hessian in kcal/(mol·Å²); convert to Hartree/Bohr²
            return hessian * KCALMOLA2_TO_HESSIAN_AU

    def _evaluate_frequencies(
        self, structure: Molecule, forcefield: ForceField, on_error: str = "raise"
    ) -> list[float]:
        """Calculate vibrational frequencies in cm⁻¹.

        Args:
            structure (Molecule): Molecule to evaluate.
            forcefield (ForceField): Force field with the parameter values.
            on_error: Forwarded to
                :func:`~q2mm.models.hessian.hessian_to_frequencies`.

        Returns:
            list[float]: Vibrational frequencies in cm⁻¹.

        """
        from q2mm.models.hessian import hessian_to_frequencies

        hessian_au = self._evaluate_hessian(structure, forcefield)
        symbols = list(structure.symbols)
        return hessian_to_frequencies(hessian_au, symbols, on_error=on_error)


class PreparedTinker(AbstractPreparedBackend):
    """Prepared Tinker session for a single training case.

    Owns the molecule, base force field, and parameter layout.  Tinker has no
    reusable native state, so each evaluation reconstructs the force field from
    the incoming full parameter vector and shells out.
    """

    def __init__(
        self,
        *,
        backend: TinkerBackend,
        case_id: str,
        molecule: Molecule,
        force_field: ForceField,
        layout: ParameterLayout,
    ) -> None:
        super().__init__(
            info=_TINKER_INFO,
            case_id=case_id,
            molecule=molecule,
            force_field=force_field,
            layout=layout,
        )
        self._backend = backend

    def _ff_for(self, parameters: np.ndarray) -> ForceField:
        vec = self._validate_vector(parameters)
        return self.layout.replace(self.force_field, vec)

    def _energy(self, request: EnergyRequest) -> EnergyResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        value = self._backend._evaluate_energy(self.molecule, ff)
        return EnergyResult(energy=float(value), unit=EnergyUnit.KCAL_PER_MOL, provenance=_TINKER_PROVENANCE)

    def _minimize(self, request: MinimizationRequest) -> GeometryResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        rms_grad = request.tolerance if request.tolerance is not None else 0.01
        energy, atoms, coords = self._backend._evaluate_minimize(
            self.molecule, ff, rms_grad=rms_grad, max_iterations=request.max_iterations
        )
        return GeometryResult(
            energy=float(energy),
            energy_unit=EnergyUnit.KCAL_PER_MOL,
            symbols=tuple(atoms),
            coordinates=readonly_array(coords),
            coordinate_unit=LengthUnit.ANGSTROM,
            provenance=_TINKER_PROVENANCE,
        )

    def _hessian(self, request: HessianRequest) -> HessianResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        hess = self._backend._evaluate_hessian(self.molecule, ff)
        return HessianResult(
            hessian=readonly_array(hess), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=_TINKER_PROVENANCE
        )

    def _frequencies(self, request: FrequencyRequest) -> FrequencyResult:  # type: ignore[override]
        ff = self._ff_for(request.parameters)
        try:
            freqs = self._backend._evaluate_frequencies(self.molecule, ff, on_error=request.on_error)
        except Exception as exc:  # noqa: BLE001
            raise EvaluationError(f"Tinker frequency evaluation failed: {exc}") from exc
        return FrequencyResult(
            frequencies=readonly_array(freqs), unit=FrequencyUnit.INVERSE_CM, provenance=_TINKER_PROVENANCE
        )
