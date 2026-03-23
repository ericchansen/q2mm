"""Objective function for force field optimization.

Wraps the ForceField ↔ MM-engine ↔ reference-data loop into a single
callable that :func:`scipy.optimize.minimize` can drive.

Scoring approach
----------------
This module uses **raw weighted residuals** (modern approach):

.. math:: r_i = w_i (x_{ref,i} - x_{calc,i})

The objective value is ``sum(r_i**2)``.  This is the standard form expected
by ``scipy.optimize.least_squares`` and gradient-based minimizers.

The legacy code in ``q2mm.optimizers.scoring`` uses a different normalisation
inherited from upstream ``compare.py``:

- Energies are zero-referenced via ``correlate_energies()`` before scoring
- A denominator based on the *total* count of energy-type data points is
  applied (see ``compare_data()``).

For migration validation, use :func:`q2mm.optimizers.scoring.compare_data`
directly — it is importable and usable standalone to cross-check scores
against the upstream code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from q2mm import constants
from q2mm.backends.base import MMEngine
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

if TYPE_CHECKING:
    from q2mm.parsers.gaussian import GaussLog


# ---- Reference data containers ----


@dataclass
class ReferenceValue:
    """A single reference observation (QM or experimental)."""

    kind: Literal[
        "energy",
        "frequency",
        "bond_length",
        "bond_angle",
        "torsion_angle",
        "eig_diagonal",
        "eig_offdiagonal",
    ]
    value: float
    weight: float = 1.0
    label: str = ""
    # Indices for matching to calculated data
    molecule_idx: int = 0
    data_idx: int = 0
    # Atom-identity matching (preferred over positional data_idx for geometry)
    atom_indices: tuple[int, ...] | None = None


@dataclass
class ReferenceData:
    """Complete set of reference data for an optimization.

    Each entry describes one observable: an energy, a frequency, or a
    geometric parameter that the force field should reproduce.
    """

    values: list[ReferenceValue] = field(default_factory=list)

    def add_energy(
        self,
        value: float,
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a single energy reference value.

        Args:
            value (float): Reference energy value.
            weight (float): Weight for this entry.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.
        """
        self.values.append(
            ReferenceValue(
                kind="energy",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                label=label,
            )
        )

    def add_frequency(
        self,
        value: float,
        *,
        data_idx: int,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a single vibrational frequency reference value.

        Args:
            value (float): Reference frequency in cm⁻¹.
            data_idx (int): 0-based index of this frequency mode.
            weight (float): Weight for this entry.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.
        """
        self.values.append(
            ReferenceValue(
                kind="frequency",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=data_idx,
                label=label,
            )
        )

    def add_bond_length(
        self,
        value: float,
        *,
        data_idx: int = -1,
        atom_indices: tuple[int, int] | None = None,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a single bond length reference value.

        Args:
            value (float): Reference bond length in Ångströms.
            data_idx (int): 0-based positional index (fallback if
                ``atom_indices`` is not provided).
            atom_indices (tuple[int, int] | None): Atom pair for
                identity-based matching.
            weight (float): Weight for this entry.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.

        Raises:
            ValueError: If neither ``atom_indices`` nor a non-negative
                ``data_idx`` is provided.
        """
        if atom_indices is None and data_idx < 0:
            raise ValueError("Either atom_indices or data_idx must be provided for bond_length.")
        self.values.append(
            ReferenceValue(
                kind="bond_length",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=max(data_idx, 0),
                atom_indices=atom_indices,
                label=label,
            )
        )

    def add_bond_angle(
        self,
        value: float,
        *,
        data_idx: int = -1,
        atom_indices: tuple[int, int, int] | None = None,
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a single bond angle reference value.

        Args:
            value (float): Reference bond angle in degrees.
            data_idx (int): 0-based positional index (fallback if
                ``atom_indices`` is not provided).
            atom_indices (tuple[int, int, int] | None): Atom triple
                (i, j, k) for identity-based matching.
            weight (float): Weight for this entry.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.

        Raises:
            ValueError: If neither ``atom_indices`` nor a non-negative
                ``data_idx`` is provided.
        """
        if atom_indices is None and data_idx < 0:
            raise ValueError("Either atom_indices or data_idx must be provided for bond_angle.")
        self.values.append(
            ReferenceValue(
                kind="bond_angle",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=max(data_idx, 0),
                atom_indices=atom_indices,
                label=label,
            )
        )

    def add_torsion_angle(
        self,
        value: float,
        *,
        atom_indices: tuple[int, int, int, int],
        weight: float = 1.0,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a single torsion (dihedral) angle reference value.

        Args:
            value (float): Reference torsion angle in degrees.
            atom_indices (tuple[int, int, int, int]): Four atom indices
                defining the dihedral.
            weight (float): Weight for this entry.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.
        """
        self.values.append(
            ReferenceValue(
                kind="torsion_angle",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                atom_indices=atom_indices,
                label=label,
            )
        )

    def add_hessian_eigenvalue(
        self,
        value: float,
        *,
        mode_idx: int,
        weight: float = 0.1,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add a diagonal element (eigenvalue) of the eigenmatrix.

        Args:
            value (float): QM eigenvalue for this mode.
            mode_idx (int): 0-based index of the vibrational mode.
            weight (float): Weight for this entry. Legacy defaults: 0.10
                for both low- and high-frequency modes, 0.00 for the
                first (imaginary) mode.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.
        """
        self.values.append(
            ReferenceValue(
                kind="eig_diagonal",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                data_idx=mode_idx,
                label=label,
            )
        )

    def add_hessian_offdiagonal(
        self,
        value: float,
        *,
        row: int,
        col: int,
        weight: float = 0.05,
        molecule_idx: int = 0,
        label: str = "",
    ):
        """Add an off-diagonal element of the eigenmatrix.

        Off-diagonal elements measure cross-coupling between modes.
        They should be close to zero when the MM Hessian closely
        reproduces the QM eigenvector structure.

        Args:
            value (float): QM eigenmatrix element (typically 0.0 for
                the QM self-projection).
            row (int): 0-based row index into the eigenmatrix.
            col (int): 0-based column index into the eigenmatrix.
            weight (float): Weight for this entry. Legacy default: 0.05.
            molecule_idx (int): Index into the molecules list.
            label (str): Human-readable label.
        """
        self.values.append(
            ReferenceValue(
                kind="eig_offdiagonal",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                atom_indices=(row, col),
                label=label,
            )
        )

    def add_eigenmatrix_from_hessian(
        self,
        hessian: np.ndarray,
        *,
        diagonal_only: bool = False,
        molecule_idx: int = 0,
        weights: dict[str, float] | None = None,
        skip_first: bool = True,
        eigenvalue_threshold: float = 0.1173,
    ) -> int:
        """Bulk-load eigenmatrix training data from a QM Hessian.

        Decomposes the Hessian, computes the eigenmatrix, and adds all
        elements as reference values with the legacy weight scheme.

        The Hessian should be in canonical units (Hartree/Bohr²).

        Args:
            hessian (np.ndarray): QM Hessian matrix ``(3N, 3N)`` in
                Hartree/Bohr².
            diagonal_only (bool): If ``True``, add only diagonal elements
                (eigenvalues). If ``False``, add all lower-triangular
                elements.
            molecule_idx (int): Index into the molecules list.
            weights (dict[str, float] | None): Weight overrides. Keys:
                ``"eig_i"`` (1st eigenvalue), ``"eig_d_low"`` (diagonal,
                value < threshold), ``"eig_d_high"`` (diagonal, value ≥
                threshold), ``"eig_o"`` (off-diagonal). Defaults match
                legacy: ``{eig_i: 0.0, eig_d_low: 0.1, eig_d_high: 0.1,
                eig_o: 0.05}``.
            skip_first (bool): If ``True``, the first eigenvalue gets
                weight ``eig_i`` (default 0.0, effectively skipping it).
                This is standard for TS fitting where the first mode is
                imaginary.
            eigenvalue_threshold (float): Eigenvalue threshold
                (Hartree/Bohr²) separating low/high frequency modes for
                weight assignment. The default 0.1173 corresponds to the
                legacy threshold of 1100 kJ/(mol·Å²).

        Returns:
            int: Number of entries added.
        """
        from q2mm.models.hessian import decompose, transform_to_eigenmatrix, extract_eigenmatrix_data

        w = {"eig_i": 0.0, "eig_d_low": 0.1, "eig_d_high": 0.1, "eig_o": 0.05}
        if weights:
            w.update(weights)

        eigenvalues, eigenvectors = decompose(hessian)
        eigenmatrix = transform_to_eigenmatrix(hessian, eigenvectors)
        elements = extract_eigenmatrix_data(eigenmatrix, diagonal_only=diagonal_only)

        added = 0
        for row, col, value in elements:
            if row == col:
                # Diagonal element
                if row == 0 and skip_first:
                    weight = w["eig_i"]
                elif value < eigenvalue_threshold:
                    weight = w["eig_d_low"]
                else:
                    weight = w["eig_d_high"]
                self.add_hessian_eigenvalue(
                    value,
                    mode_idx=row,
                    weight=weight,
                    molecule_idx=molecule_idx,
                    label=f"eig[{row}]",
                )
            else:
                # Off-diagonal element
                self.add_hessian_offdiagonal(
                    value,
                    row=row,
                    col=col,
                    weight=w["eig_o"],
                    molecule_idx=molecule_idx,
                    label=f"eig[{row},{col}]",
                )
            added += 1
        return added

    @property
    def n_observations(self) -> int:
        """Total number of reference observations.

        Returns:
            int: Length of the ``values`` list.
        """
        return len(self.values)

    # ---- Bulk loaders ----

    def add_frequencies_from_array(
        self,
        frequencies: np.ndarray | list[float],
        *,
        weight: float = 1.0,
        molecule_idx: int = 0,
        skip_imaginary: bool = False,
    ) -> int:
        """Add all frequencies from a 1-D array.

        Args:
            frequencies (np.ndarray | list[float]): Vibrational frequencies
                (cm⁻¹). Imaginary modes should be negative values.
            weight (float): Weight applied to every frequency entry.
            molecule_idx (int): Index into the molecules list for
                multi-structure fits.
            skip_imaginary (bool): If ``True``, negative frequencies
                (imaginary modes) are skipped.

        Returns:
            int: Number of frequency entries added.
        """
        freqs = np.asarray(frequencies, dtype=float).ravel()
        added = 0
        for i, freq in enumerate(freqs):
            if skip_imaginary and freq < 0:
                continue
            self.add_frequency(
                float(freq),
                data_idx=i,
                weight=weight,
                molecule_idx=molecule_idx,
                label=f"mode {i}",
            )
            added += 1
        return added

    # ---- Factory methods ----

    @classmethod
    def from_molecule(
        cls,
        mol: Q2MMMolecule,
        *,
        weights: dict[str, float] | None = None,
        molecule_idx: int = 0,
        frequencies: np.ndarray | list[float] | None = None,
        skip_imaginary: bool = False,
        include_eigenmatrix: bool = False,
        eigenmatrix_diagonal_only: bool = False,
    ) -> ReferenceData:
        """Auto-populate reference data from a molecule's detected geometry.

        Extracts all auto-detected bond lengths and bond angles from the
        molecule. Optionally adds vibrational frequencies and/or Hessian
        eigenmatrix training data.

        Args:
            mol (Q2MMMolecule): Molecule with geometry (bonds/angles
                auto-detected).
            weights (dict[str, float] | None): Weight overrides keyed by
                data type. Supported keys: ``"bond_length"``,
                ``"bond_angle"``, ``"frequency"``, and the eigenmatrix
                keys ``"eig_i"``, ``"eig_d_low"``, ``"eig_d_high"``,
                ``"eig_o"``. Defaults: ``{"bond_length": 10.0,
                "bond_angle": 5.0, "frequency": 1.0}``.
            molecule_idx (int): Index for multi-molecule fits.
            frequencies (np.ndarray | list[float] | None): Vibrational
                frequencies (cm⁻¹) to include.
            skip_imaginary (bool): If ``True``, negative frequencies are
                skipped.
            include_eigenmatrix (bool): If ``True`` and the molecule has
                a Hessian, add eigenmatrix training data (diagonal and
                optionally off-diagonal elements).
            eigenmatrix_diagonal_only (bool): If ``True``, only diagonal
                eigenmatrix elements are added.

        Returns:
            ReferenceData: Populated with bond lengths, angles, and
                (optionally) frequencies and eigenmatrix data.
        """
        w = {"bond_length": 10.0, "bond_angle": 5.0, "frequency": 1.0}
        if weights:
            w.update(weights)

        ref = cls()

        for bond in mol.bonds:
            ref.add_bond_length(
                bond.length,
                atom_indices=(bond.atom_i, bond.atom_j),
                weight=w["bond_length"],
                molecule_idx=molecule_idx,
                label=f"{bond.element_pair} bond",
            )

        for angle in mol.angles:
            ref.add_bond_angle(
                angle.value,
                atom_indices=(angle.atom_i, angle.atom_j, angle.atom_k),
                weight=w["bond_angle"],
                molecule_idx=molecule_idx,
                label=f"{angle.elements} angle",
            )

        if frequencies is not None:
            ref.add_frequencies_from_array(
                frequencies,
                weight=w["frequency"],
                molecule_idx=molecule_idx,
                skip_imaginary=skip_imaginary,
            )

        if include_eigenmatrix and mol.hessian is not None:
            eig_weights = {k: w[k] for k in ("eig_i", "eig_d_low", "eig_d_high", "eig_o") if k in w}
            ref.add_eigenmatrix_from_hessian(
                mol.hessian,
                diagonal_only=eigenmatrix_diagonal_only,
                molecule_idx=molecule_idx,
                weights=eig_weights or None,
            )

        return ref

    @classmethod
    def from_molecules(
        cls,
        molecules: list[Q2MMMolecule],
        *,
        weights: dict[str, float] | None = None,
        frequencies_list: list[np.ndarray | list[float]] | None = None,
        skip_imaginary: bool = False,
        include_eigenmatrix: bool = False,
        eigenmatrix_diagonal_only: bool = False,
    ) -> ReferenceData:
        """Auto-populate reference data from multiple molecules.

        Each molecule is assigned a sequential ``molecule_idx`` starting
        from 0.  Delegates to :meth:`from_molecule` per molecule.

        Args:
            molecules (list[Q2MMMolecule]): Training set molecules.
            weights (dict[str, float] | None): Weight overrides (same
                keys as :meth:`from_molecule`).
            frequencies_list (list[np.ndarray | list[float]] | None):
                Per-molecule frequencies. Must have the same length as
                *molecules* if provided.
            skip_imaginary (bool): If ``True``, negative frequencies are
                skipped.
            include_eigenmatrix (bool): If ``True`` and a molecule has a
                Hessian, add eigenmatrix data.
            eigenmatrix_diagonal_only (bool): If ``True``, only diagonal
                eigenmatrix elements are added.

        Returns:
            ReferenceData: Combined reference data for all molecules.

        Raises:
            ValueError: If ``frequencies_list`` length does not match
                ``molecules`` length.
        """
        if frequencies_list is not None and len(frequencies_list) != len(molecules):
            raise ValueError(
                f"frequencies_list length ({len(frequencies_list)}) must match molecules length ({len(molecules)})."
            )

        ref = cls()
        for idx, mol in enumerate(molecules):
            single = cls.from_molecule(
                mol,
                weights=weights,
                molecule_idx=idx,
                frequencies=frequencies_list[idx] if frequencies_list is not None else None,
                skip_imaginary=skip_imaginary,
                include_eigenmatrix=include_eigenmatrix,
                eigenmatrix_diagonal_only=eigenmatrix_diagonal_only,
            )
            ref.values.extend(single.values)

        return ref

    @classmethod
    def from_gaussian(
        cls,
        path: str | Path,
        *,
        weights: dict[str, float] | None = None,
        bond_tolerance: float = 1.3,
        charge: int = 0,
        multiplicity: int = 1,
        include_frequencies: bool = True,
        skip_imaginary: bool = False,
        au_hessian: bool = True,
    ) -> tuple[ReferenceData, Q2MMMolecule]:
        """Build reference data from a Gaussian log file.

        Parses the log file for the optimised geometry and vibrational
        frequencies, then auto-populates bond lengths, angles, and
        (optionally) frequencies.

        Args:
            path (str | Path): Path to the Gaussian ``.log`` file
                (from an ``opt freq`` job).
            weights (dict[str, float] | None): Weight overrides (same
                keys as :meth:`from_molecule`).
            bond_tolerance (float): Multiplier for covalent-radii bond
                detection. Use 1.4+ for TS.
            charge (int): Molecular charge.
            multiplicity (int): Spin multiplicity.
            include_frequencies (bool): Whether to add frequency data
                from the log file.
            skip_imaginary (bool): If ``True``, negative frequencies are
                skipped.
            au_hessian (bool): Keep Hessian in atomic units
                (Hartree/Bohr²).

        Returns:
            tuple[ReferenceData, Q2MMMolecule]: Populated reference data
                and the parsed molecule (with Hessian attached if
                available).
        """
        from q2mm.parsers.gaussian import GaussLog
        from q2mm.models.hessian import reform_hessian

        log = GaussLog(str(path), au_hessian=au_hessian)

        # Build molecule from the last (optimised) structure
        structure = log.structures[-1]
        hessian = None
        if log.evals is not None and log.evecs is not None and log.evals.size and log.evecs.size:
            hessian = reform_hessian(log.evals, log.evecs)

        mol = Q2MMMolecule.from_structure(
            structure,
            charge=charge,
            multiplicity=multiplicity,
            bond_tolerance=bond_tolerance,
            hessian=hessian,
        )

        # Frequencies in cm⁻¹ from the Gaussian log
        # Note: log.evals are eigenvalues (mass-weighted force constants in
        # atomic units), NOT frequencies.  Use log.frequencies for cm⁻¹ values.
        frequencies = None
        if include_frequencies and log.frequencies is not None and len(log.frequencies):
            frequencies = np.array(log.frequencies)

        ref = cls.from_molecule(
            mol,
            weights=weights,
            frequencies=frequencies,
            skip_imaginary=skip_imaginary,
        )

        return ref, mol

    @classmethod
    def from_fchk(
        cls,
        path: str | Path,
        *,
        weights: dict[str, float] | None = None,
        bond_tolerance: float = 1.3,
        charge: int = 0,
        multiplicity: int = 1,
    ) -> tuple[ReferenceData, Q2MMMolecule]:
        """Build reference data from a Gaussian formatted checkpoint file.

        Parses the ``.fchk`` file for geometry, Cartesian Force Constants
        (Hessian), and atom data. Auto-populates bond lengths and angles.

        Args:
            path (str | Path): Path to the Gaussian ``.fchk`` file.
            weights (dict[str, float] | None): Weight overrides (same
                keys as :meth:`from_molecule`).
            bond_tolerance (float): Multiplier for covalent-radii bond
                detection.
            charge (int): Molecular charge (overridden by file values
                if present).
            multiplicity (int): Spin multiplicity (overridden by file
                values if present).

        Returns:
            tuple[ReferenceData, Q2MMMolecule]: Populated reference data
                and the parsed molecule with Hessian.
        """
        path = Path(path)
        symbols, coords_ang, hessian, file_charge, file_mult = _parse_fchk(path)

        mol = Q2MMMolecule(
            symbols=symbols,
            geometry=coords_ang,
            charge=file_charge if file_charge is not None else charge,
            multiplicity=file_mult if file_mult is not None else multiplicity,
            name=path.stem,
            bond_tolerance=bond_tolerance,
            hessian=hessian,
        )

        ref = cls.from_molecule(mol, weights=weights)

        return ref, mol


# ---- .fchk parser (minimal, self-contained) ----

# Atomic numbers → element symbols, from centralized element data.
from q2mm.elements import ATOMIC_SYMBOLS as _ATOMIC_SYMBOLS

_BOHR_TO_ANG = constants.BOHR_TO_ANG


def _parse_fchk(path: Path) -> tuple[list[str], np.ndarray, np.ndarray | None, int | None, int | None]:
    """Parse a Gaussian .fchk file for geometry and Hessian.

    Args:
        path (Path): Path to the ``.fchk`` file.

    Returns:
        tuple[list[str], np.ndarray, np.ndarray | None, int | None, int | None]:
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


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute dihedral angle (degrees) for four points.

    Uses the atan2 formulation that returns values in [-180, 180].

    Args:
        p0 (np.ndarray): Coordinates of the first atom.
        p1 (np.ndarray): Coordinates of the second atom.
        p2 (np.ndarray): Coordinates of the third atom.
        p3 (np.ndarray): Coordinates of the fourth atom.

    Returns:
        float: Dihedral angle in degrees, in the range [-180, 180].
    """
    b1 = np.asarray(p1) - np.asarray(p0)
    b2 = np.asarray(p2) - np.asarray(p1)
    b3 = np.asarray(p3) - np.asarray(p2)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    if n1_norm < 1e-10 or n2_norm < 1e-10:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


# ---- Objective function ----


class ObjectiveFunction:
    """Objective function for scipy-based force field optimization.

    Evaluates the weighted sum-of-squares between MM-calculated and
    reference data for one or more molecules.

    Args:
        forcefield (ForceField): The force field whose parameters are
            being optimized.
        engine (MMEngine): The MM backend (OpenMM, Tinker, etc.).
        molecules (list[Q2MMMolecule]): Training set molecules.
        reference (ReferenceData): QM/experimental reference observations.
    """

    def __init__(
        self,
        forcefield: ForceField,
        engine: MMEngine,
        molecules: list[Q2MMMolecule],
        reference: ReferenceData,
    ):
        """Initialize the objective function.

        Args:
            forcefield (ForceField): The force field whose parameters
                are being optimized.
            engine (MMEngine): The MM backend (OpenMM, Tinker, etc.).
            molecules (list[Q2MMMolecule]): Training set molecules.
            reference (ReferenceData): QM/experimental reference
                observations.
        """
        self.forcefield = forcefield
        self.engine = engine
        self.molecules = molecules
        self.reference = reference
        self.n_eval = 0
        self.history: list[float] = []
        # Reusable engine handles for backends that support runtime parameter
        # updates (e.g., OpenMM). Avoids rebuilding simulation contexts each
        # evaluation — critical for optimization performance.
        self._handles: dict[int, object] = {}
        # Cached QM eigenvectors per molecule (precomputed once for
        # eigenmatrix comparisons — the QM basis is fixed).
        self._qm_eigenvectors: dict[int, np.ndarray] = {}

    def __call__(self, param_vector: np.ndarray) -> float:
        """Evaluate objective for a given parameter vector.

        This is the function signature that :func:`scipy.optimize.minimize`
        expects: ``f(x) -> float``.

        Args:
            param_vector (np.ndarray): Flat parameter vector.

        Returns:
            float: Sum-of-squared weighted residuals.
        """
        self.forcefield.set_param_vector(param_vector)

        residuals = self._compute_residuals()
        score = float(np.sum(residuals**2))

        self.n_eval += 1
        self.history.append(score)
        return score

    def residuals(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute weighted residual vector (for least-squares methods).

        Args:
            param_vector (np.ndarray): Flat parameter vector.

        Returns:
            np.ndarray: Weighted residuals for each reference observation.
        """
        self.forcefield.set_param_vector(param_vector)
        r = self._compute_residuals()
        self.n_eval += 1
        self.history.append(float(np.sum(r**2)))
        return r

    def _compute_residuals(self) -> np.ndarray:
        """Compute weighted residuals for all reference observations.

        Returns:
            np.ndarray: Array of ``w_i * (ref_i - calc_i)`` residuals.
        """
        calc_cache: dict[int, dict] = {}

        residuals = []
        for ref in self.reference.values:
            mol_idx = ref.molecule_idx
            if mol_idx not in calc_cache:
                calc_cache[mol_idx] = self._evaluate_molecule(mol_idx)

            calc = calc_cache[mol_idx]
            calc_value = self._extract_value(calc, ref)
            diff = ref.value - calc_value
            # Torsion angles wrap around 360°
            if ref.kind == "torsion_angle":
                diff = (diff + 180.0) % 360.0 - 180.0
            residual = ref.weight * diff
            residuals.append(residual)

        return np.array(residuals)

    def gradient(self, param_vector: np.ndarray) -> np.ndarray:
        """Compute analytical gradient of the score w.r.t. parameters.

        Uses the engine's ``energy_and_param_grad()`` method (available on
        :class:`~q2mm.backends.mm.jax_engine.JaxEngine`) to compute exact
        derivatives for energy reference data.  Raises ``NotImplementedError``
        for reference data types that require Hessians or minimized geometries;
        use ``jac=None`` (finite differences) for mixed reference data.

        The score is ``sum_i (w_i * (ref_i - calc_i))**2``, so:

        ``d(score)/d(p) = -2 * sum_i [w_i^2 * (ref_i - calc_i) * d(calc_i)/d(p)]``

        Note:
            This method does **not** increment ``n_eval`` or append to
            ``history``.  SciPy's ``minimize`` calls ``fun(x)`` and ``jac(x)``
            separately, so tracking state here would double-count evaluations.
            Evaluation counting is handled exclusively in ``__call__``.

        Args:
            param_vector (np.ndarray): Flat parameter vector (same as
                :meth:`__call__`).

        Returns:
            np.ndarray: Gradient of the score with respect to each parameter.

        Raises:
            TypeError: If the engine does not support
                ``energy_and_param_grad()``.
            NotImplementedError: If the reference data contains types
                other than ``energy`` (Hessian/frequency/geometry gradient
                support is planned).
        """
        if not self.engine.supports_analytical_gradients():
            raise TypeError(
                f"{self.engine.name} does not support analytical gradients. "
                "Use a JaxEngine or another differentiable engine."
            )

        self.forcefield.set_param_vector(param_vector)

        # Check which data types are needed
        all_kinds = {ref.kind for ref in self.reference.values}
        unsupported = all_kinds - {"energy"}
        if unsupported:
            raise NotImplementedError(
                f"Analytical gradients not yet supported for data types: {unsupported}. "
                "Only 'energy' references are supported. Use finite differences (jac=None) "
                "for mixed reference data, or contribute Hessian/geometry gradient support."
            )

        # Compute energy + gradient for each molecule
        n_params = len(param_vector)
        total_grad = np.zeros(n_params)
        energy_cache: dict[int, tuple[float, np.ndarray]] = {}

        for ref in self.reference.values:
            mol_idx = ref.molecule_idx
            if mol_idx not in energy_cache:
                structure = self._get_structure(mol_idx)
                energy_cache[mol_idx] = self.engine.energy_and_param_grad(structure, self.forcefield)

            calc_value, calc_grad = energy_cache[mol_idx]
            diff = ref.value - calc_value
            # d(score)/d(p) = -2 * w^2 * (ref - calc) * d(calc)/d(p)
            total_grad += -2.0 * ref.weight**2 * diff * calc_grad

        return total_grad

    def _get_structure(self, mol_idx: int):
        """Get the structure handle for a molecule, reusing if possible.

        For backends that support runtime parameter updates (OpenMM), this
        creates the simulation context once and reuses it across evaluations.
        For stateless backends (Tinker), this returns the raw molecule.

        Args:
            mol_idx (int): Index into the molecules list.

        Returns:
            object: Engine-specific structure handle or raw molecule.
        """
        mol = self.molecules[mol_idx]
        if not self.engine.supports_runtime_params():
            return mol
        if mol_idx not in self._handles:
            # First call — let the engine create its context/handle
            self._handles[mol_idx] = self.engine.create_context(mol, self.forcefield)
        return self._handles[mol_idx]

    def _evaluate_molecule(self, mol_idx: int) -> dict:
        """Run MM calculations for a single molecule.

        Args:
            mol_idx (int): Index into the molecules list.

        Returns:
            dict: Calculated results keyed by data type (e.g.
                ``"energy"``, ``"frequencies"``, ``"bond_lengths"``).
        """
        mol = self.molecules[mol_idx]
        structure = self._get_structure(mol_idx)
        result: dict = {}

        # Determine what data types are needed for this molecule
        needed = {ref.kind for ref in self.reference.values if ref.molecule_idx == mol_idx}

        if "energy" in needed:
            result["energy"] = self.engine.energy(structure, self.forcefield)

        if "frequency" in needed:
            result["frequencies"] = self.engine.frequencies(structure, self.forcefield)

        if "bond_length" in needed or "bond_angle" in needed or "torsion_angle" in needed:
            # Geometry observables require MM-minimized structures to be
            # meaningful (the input geometry is fixed). Minimize first.
            # Pass the raw molecule (not cached handle) because minimize()
            # mutates context positions — reusing the cached handle would
            # corrupt subsequent energy/frequency evaluations.
            _energy, _atoms, opt_coords = self.engine.minimize(mol, self.forcefield)
            opt_mol = Q2MMMolecule(
                symbols=mol.symbols,
                geometry=opt_coords,
                name=mol.name,
                atom_types=list(mol.atom_types),
                bond_tolerance=mol.bond_tolerance,
            )
            if "bond_length" in needed:
                # Store both positional list and atom-keyed dict for matching
                result["bond_lengths"] = [b.length for b in opt_mol.bonds]
                result["bond_lengths_by_atoms"] = {tuple(sorted((b.atom_i, b.atom_j))): b.length for b in opt_mol.bonds}
            if "bond_angle" in needed:
                result["bond_angles"] = [a.value for a in opt_mol.angles]
                result["bond_angles_by_atoms"] = {(a.atom_i, a.atom_j, a.atom_k): a.value for a in opt_mol.angles}
            if "torsion_angle" in needed:
                result["torsion_coords"] = opt_coords

        if "eig_diagonal" in needed or "eig_offdiagonal" in needed:
            from q2mm.models.hessian import decompose, transform_to_eigenmatrix

            # Compute MM Hessian (engine returns canonical Hartree/Bohr²)
            # and project onto QM eigenvectors for eigenmatrix comparison.
            mm_hess = self.engine.hessian(structure, self.forcefield)

            # Cache QM eigenvectors (fixed across evaluations)
            if mol_idx not in self._qm_eigenvectors:
                if mol.hessian is None:
                    raise ValueError(
                        f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                        "Eigenmatrix training requires a QM Hessian for the eigenvector basis."
                    )
                _, qm_evecs = decompose(mol.hessian)
                self._qm_eigenvectors[mol_idx] = qm_evecs

            qm_evecs = self._qm_eigenvectors[mol_idx]
            eigenmatrix = transform_to_eigenmatrix(mm_hess, qm_evecs)
            result["eigenmatrix"] = eigenmatrix

        return result

    @staticmethod
    def _extract_value(calc: dict, ref: ReferenceValue) -> float:
        """Extract a calculated value matching a reference observation.

        For bond_length and bond_angle, prefers atom-identity matching via
        ``ref.atom_indices`` when available, falling back to positional
        ``ref.data_idx`` for backwards compatibility.

        Args:
            calc (dict): Calculated results from :meth:`_evaluate_molecule`.
            ref (ReferenceValue): Reference observation to match.

        Returns:
            float: The calculated value corresponding to the reference.

        Raises:
            IndexError: If ``data_idx`` is out of range.
            KeyError: If atom-identity match fails.
            ValueError: If ``ref.kind`` is unknown or torsion is missing
                atom indices.
        """
        if ref.kind == "energy":
            return calc["energy"]
        elif ref.kind == "frequency":
            freqs = calc["frequencies"]
            if ref.data_idx >= len(freqs):
                raise IndexError(
                    f"Frequency data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(freqs)} modes). Label: {ref.label!r}"
                )
            return freqs[ref.data_idx]
        elif ref.kind == "bond_length":
            if ref.atom_indices is not None:
                key = tuple(sorted(ref.atom_indices[:2]))
                by_atoms = calc.get("bond_lengths_by_atoms", {})
                if key not in by_atoms:
                    raise KeyError(
                        f"No bond found for atoms {key}. Available bonds: {list(by_atoms.keys())}. Label: {ref.label!r}"
                    )
                return by_atoms[key]
            lengths = calc["bond_lengths"]
            if ref.data_idx >= len(lengths):
                raise IndexError(
                    f"Bond data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(lengths)} bonds). Label: {ref.label!r}"
                )
            return lengths[ref.data_idx]
        elif ref.kind == "bond_angle":
            if ref.atom_indices is not None:
                by_atoms = calc.get("bond_angles_by_atoms", {})
                key = tuple(ref.atom_indices[:3])
                # Try both orderings: (i, j, k) and (k, j, i)
                if key not in by_atoms:
                    key = (key[2], key[1], key[0])
                if key not in by_atoms:
                    raise KeyError(
                        f"No angle found for atoms {ref.atom_indices[:3]}. "
                        f"Available angles: {list(by_atoms.keys())}. Label: {ref.label!r}"
                    )
                return by_atoms[key]
            angles = calc["bond_angles"]
            if ref.data_idx >= len(angles):
                raise IndexError(
                    f"Angle data_idx={ref.data_idx} out of range "
                    f"(molecule has {len(angles)} angles). Label: {ref.label!r}"
                )
            return angles[ref.data_idx]
        elif ref.kind == "torsion_angle":
            if ref.atom_indices is None or len(ref.atom_indices) < 4:
                raise ValueError(f"torsion_angle requires atom_indices with 4 atoms. Label: {ref.label!r}")
            coords = calc["torsion_coords"]
            return _dihedral_angle(
                coords[ref.atom_indices[0]],
                coords[ref.atom_indices[1]],
                coords[ref.atom_indices[2]],
                coords[ref.atom_indices[3]],
            )
        elif ref.kind == "eig_diagonal":
            eigenmatrix = calc["eigenmatrix"]
            return float(eigenmatrix[ref.data_idx, ref.data_idx])
        elif ref.kind == "eig_offdiagonal":
            eigenmatrix = calc["eigenmatrix"]
            row, col = ref.atom_indices[:2]
            return float(eigenmatrix[row, col])
        else:
            raise ValueError(f"Unknown reference kind: {ref.kind}")

    def reset(self):
        """Reset evaluation counter, history, and cached engine handles."""
        self.n_eval = 0
        self.history.clear()
        self._handles.clear()
        self._qm_eigenvectors.clear()
