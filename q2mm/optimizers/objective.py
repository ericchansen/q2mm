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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from q2mm.backends.base import MMEngine
from q2mm.constants import DEFAULT_BOND_TOLERANCE
from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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
        "hessian_element",
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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
    ) -> None:
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

    def add_hessian_element(
        self,
        value: float,
        *,
        row: int,
        col: int,
        weight: float = 0.1,
        molecule_idx: int = 0,
        label: str = "",
    ) -> None:
        """Add a single raw Hessian matrix element as reference data.

        Args:
            value: QM Hessian element in Hartree/Bohr².
            row: 0-based row index.
            col: 0-based column index.
            weight: Weight for this entry.
            molecule_idx: Index into the molecules list.
            label: Human-readable label.

        Raises:
            ValueError: If *row* or *col* is negative.

        """
        if row < 0 or col < 0:
            raise ValueError(f"row and col must be non-negative, got row={row}, col={col}")
        self.values.append(
            ReferenceValue(
                kind="hessian_element",
                value=value,
                weight=weight,
                molecule_idx=molecule_idx,
                atom_indices=(row, col),
                label=label or f"hess[{row},{col}]",
            )
        )

    def add_hessian_from_matrix(
        self,
        hessian: np.ndarray,
        *,
        diagonal_only: bool = False,
        molecule_idx: int = 0,
        diagonal_weight: float = 0.1,
        offdiagonal_weight: float = 0.05,
        skip_translational: int = 0,
    ) -> int:
        """Bulk-load raw Hessian elements as reference data.

        Unlike :meth:`add_eigenmatrix_from_hessian`, this uses the raw
        Cartesian Hessian directly without eigendecomposition.

        Args:
            hessian: QM Hessian (3N, 3N) in Hartree/Bohr².
            diagonal_only: If ``True``, only add diagonal elements.
            molecule_idx: Index into molecules list.
            diagonal_weight: Weight for diagonal elements.
            offdiagonal_weight: Weight for off-diagonal elements.
            skip_translational: Number of leading rows/cols to skip
                (e.g. 6 for trans+rot modes in Cartesian basis).

        Returns:
            Number of entries added.

        """
        n = hessian.shape[0]
        if hessian.shape != (n, n):
            raise ValueError(f"Hessian must be square, got shape {hessian.shape}")
        if skip_translational < 0:
            raise ValueError(f"skip_translational must be non-negative, got {skip_translational}")
        if skip_translational >= n:
            raise ValueError(f"skip_translational ({skip_translational}) must be less than matrix size ({n})")

        added = 0
        start = skip_translational
        for i in range(start, n):
            for j in range(start, i + 1) if not diagonal_only else [i]:
                weight = diagonal_weight if i == j else offdiagonal_weight
                self.add_hessian_element(
                    float(hessian[i, j]),
                    row=i,
                    col=j,
                    weight=weight,
                    molecule_idx=molecule_idx,
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
        bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
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
        mol = log.molecules[-1]
        mol.charge = charge
        mol.multiplicity = multiplicity
        mol.bond_tolerance = bond_tolerance

        # Override hessian with eigenvalue-reconstructed version if available
        if log.evals is not None and log.evecs is not None and log.evals.size and log.evecs.size:
            mol.hessian = reform_hessian(log.evals, log.evecs)

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
        bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
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
        from q2mm.parsers.fchk import parse_fchk as _parse_fchk  # noqa: E402

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

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
    ) -> tuple[ReferenceData, list[Q2MMMolecule]]:
        """Load reference data and molecules from a YAML file.

        Delegates to :func:`q2mm.parsers.reference_yaml.load_reference_yaml`.

        Args:
            path (str | Path): Path to the YAML reference file.

        Returns:
            tuple[ReferenceData, list[Q2MMMolecule]]: Loaded reference
                data and parsed molecules.

        """
        from q2mm.parsers.reference_yaml import load_reference_yaml

        return load_reference_yaml(path)

    def to_yaml(
        self,
        path: str | Path,
        molecules: list[Q2MMMolecule],
    ) -> None:
        """Save this reference data and molecules to a YAML file.

        Delegates to :func:`q2mm.parsers.reference_yaml.save_reference_yaml`.

        Args:
            path (str | Path): Output file path.
            molecules (list[Q2MMMolecule]): Molecules corresponding to
                the reference values.

        """
        from q2mm.parsers.reference_yaml import save_reference_yaml

        save_reference_yaml(path, self, molecules)


# ---- Lazy re-exports (avoid circular imports at module load) ----


def __getattr__(name: str) -> Any:
    """Lazy re-exports to preserve backward-compatible import paths."""
    if name == "_parse_fchk":
        from q2mm.parsers.fchk import parse_fchk

        return parse_fchk
    if name == "_dihedral_angle":
        from q2mm.optimizers.evaluators.geometry import dihedral_angle

        return dihedral_angle
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---- Per-data-type evaluators (lazy-imported in ObjectiveFunction.__init__) ----


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
    ) -> None:
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
        self.fd_step = 1e-4
        self.history: list[float] = []
        # Reusable engine handles for backends that support runtime parameter
        # updates (e.g., OpenMM). Avoids rebuilding simulation contexts each
        # evaluation — critical for optimization performance.
        self._handles: dict[int, object] = {}
        # Per-data-type evaluator instances (created once, reused).
        # Lazy imports to break circular dependency (evaluators import
        # ReferenceValue from this module).
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator
        from q2mm.optimizers.evaluators.hessian_element import HessianElementEvaluator

        self._evaluators = [
            EnergyEvaluator(),
            FrequencyEvaluator(),
            GeometryEvaluator(),
            EigenmatrixEvaluator(),
            HessianElementEvaluator(),
        ]
        # Build kind → evaluator lookup from HANDLED_KINDS
        self._kind_to_evaluator: dict[str, Any] = {}
        for ev in self._evaluators:
            for kind in ev.HANDLED_KINDS:
                self._kind_to_evaluator[kind] = ev

    def __call__(self, param_vector: np.ndarray) -> float:
        """Evaluate objective for a given parameter vector.

        This is the function signature that :func:`scipy.optimize.minimize`
        expects: ``f(x) -> float``.

        Args:
            param_vector (np.ndarray): Flat parameter vector.

        Returns:
            float: Sum-of-squared weighted residuals.

        """
        ff = self.forcefield.with_params(param_vector)

        residuals = self._compute_residuals(ff)
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
        ff = self.forcefield.with_params(param_vector)
        r = self._compute_residuals(ff)
        self.n_eval += 1
        self.history.append(float(np.sum(r**2)))
        return r

    def is_energy_only(self) -> bool:
        """Return ``True`` if every reference is an energy value.

        When true, :meth:`batched_scores` can use the engine's vectorised
        ``batched_energy`` path (e.g. ``jax.vmap``) for GPU-parallel
        sensitivity analysis.
        """
        return all(ref.kind == "energy" for ref in self.reference.values)

    def batched_scores(self, param_matrix: np.ndarray) -> np.ndarray:
        """Evaluate objective for multiple parameter vectors in one call.

        When the engine supports :meth:`~MMEngine.batched_energy` and all
        references are energy-only, energy evaluations are vectorised
        (e.g. via ``jax.vmap``) for GPU-parallel evaluation.

        Otherwise falls back to sequential ``__call__`` per vector.

        Args:
            param_matrix: Shape ``(batch, n_params)`` parameter vectors.

        Returns:
            np.ndarray: Shape ``(batch,)`` objective scores.

        Raises:
            ValueError: If *param_matrix* has wrong number of columns.

        """
        param_matrix = np.atleast_2d(np.asarray(param_matrix, dtype=float))
        if param_matrix.ndim != 2 or param_matrix.shape[1] != self.forcefield.n_params:
            raise ValueError(
                f"param_matrix must have shape (batch, {self.forcefield.n_params}), got {param_matrix.shape}"
            )

        use_batched = self.engine.supports_batched_energy() and self.is_energy_only()
        if not use_batched:
            return np.array([self(pvec) for pvec in param_matrix])

        return self._batched_energy_scores(param_matrix)

    def _batched_energy_scores(self, param_matrix: np.ndarray) -> np.ndarray:
        """Compute scores for a batch of param vectors (energy-only fast path).

        Uses :meth:`MMEngine.batched_energy` to evaluate all parameter
        vectors in a single vectorised call per molecule.
        """
        batch_size = len(param_matrix)
        # Group energy references by molecule index
        mol_refs: dict[int, list] = {}
        for ref in self.reference.values:
            mol_refs.setdefault(ref.molecule_idx, []).append(ref)

        # For each molecule, batch-evaluate energies
        mol_energies: dict[int, np.ndarray] = {}
        for mol_idx in mol_refs:
            structure = self._get_structure(mol_idx)
            mol_energies[mol_idx] = self.engine.batched_energy(
                structure,
                self.forcefield,
                param_matrix,
            )

        # Compute residuals and scores
        scores = np.zeros(batch_size)
        for ref in self.reference.values:
            energies = mol_energies[ref.molecule_idx]
            residuals = ref.weight * (ref.value - energies)
            scores += residuals**2

        n = len(param_matrix)
        self.n_eval += n
        for s in scores:
            self.history.append(float(s))
        return scores

    def _can_batch_hessians(self) -> bool:
        """Check whether batched Hessian evaluation can be used.

        Batching is possible when the engine is a
        :class:`~q2mm.backends.mm.jax_engine.JaxEngine`, there are at
        least two molecules, and some references require Hessian-derived
        data (frequencies or eigenmatrix).
        """
        try:
            from q2mm.backends.mm.jax_engine import JaxEngine
        except ImportError:
            return False

        if not isinstance(self.engine, JaxEngine):
            return False
        if len(self.molecules) < 2:
            return False

        hessian_kinds = {"frequency", "eig_diagonal", "eig_offdiagonal"}
        return any(ref.kind in hessian_kinds for ref in self.reference.values)

    def _precompute_batched_hessians(
        self,
        forcefield: ForceField,
    ) -> dict[int, np.ndarray]:
        """Pre-compute Hessians for topology-compatible molecule groups.

        Groups molecules that share the same topology and uses
        ``jax.vmap`` via :func:`~q2mm.backends.mm.batched.batched_hessians`
        to compute all Hessians in a single vectorised call per group.

        Returns a mapping from molecule index to its ``(3N, 3N)`` Hessian
        in Hartree/Bohr².  Only molecules that need Hessian-derived data
        are included.
        """
        from q2mm.backends.mm.batched import batched_hessians, group_by_topology

        hessian_kinds = {"frequency", "eig_diagonal", "eig_offdiagonal"}
        mol_indices_needing_hess: set[int] = set()
        for ref in self.reference.values:
            if ref.kind in hessian_kinds:
                mol_indices_needing_hess.add(ref.molecule_idx)

        if not mol_indices_needing_hess:
            return {}

        mols_subset = [self.molecules[i] for i in sorted(mol_indices_needing_hess)]
        idx_list = sorted(mol_indices_needing_hess)

        # Build handle cache for molecules that already have handles
        handle_cache: dict[int, object] = {}
        for pos, mol_idx in enumerate(idx_list):
            if mol_idx in self._handles:
                handle_cache[pos] = self._handles[mol_idx]

        groups = group_by_topology(
            mols_subset,
            forcefield,
            self.engine,  # type: ignore[arg-type]
            handles=handle_cache,  # type: ignore[arg-type]
        )

        hess_map: dict[int, np.ndarray] = {}
        for group in groups:
            # Cache handles created during grouping to avoid duplicate
            # compilation in subsequent _evaluate_molecule calls.
            for mol_local_idx in group.mol_indices:
                original_idx = idx_list[mol_local_idx]
                if original_idx not in self._handles:
                    self._handles[original_idx] = group.handle

            hessians = batched_hessians(group, forcefield)
            for local_i, hess in enumerate(hessians):
                original_idx = idx_list[group.mol_indices[local_i]]
                hess_map[original_idx] = hess

        return hess_map

    def _compute_residuals(self, forcefield: ForceField) -> np.ndarray:
        """Compute weighted residuals for all reference observations.

        When the engine is a :class:`~q2mm.backends.mm.jax_engine.JaxEngine`
        and multiple molecules share the same topology, Hessians are
        computed in a single ``jax.vmap`` call per topology group
        (batched path).  Otherwise, falls back to per-molecule sequential
        evaluation.

        Args:
            forcefield: The force field to evaluate (typically a temporary
                instance from :meth:`~ForceField.with_params`).

        Returns:
            np.ndarray: Array of ``w_i * (ref_i - calc_i)`` residuals.

        """
        # Attempt batched Hessian pre-computation
        precomputed_hessians: dict[int, np.ndarray] = {}
        if self._can_batch_hessians():
            try:
                precomputed_hessians = self._precompute_batched_hessians(forcefield)
            except Exception:
                logger.warning(
                    "Batched Hessian pre-computation failed; falling back to sequential evaluation.",
                    exc_info=True,
                )
                precomputed_hessians = {}

        calc_cache: dict[int, dict] = {}

        residuals = []
        for ref in self.reference.values:
            mol_idx = ref.molecule_idx
            if mol_idx not in calc_cache:
                calc_cache[mol_idx] = self._evaluate_molecule(
                    mol_idx,
                    forcefield,
                    precomputed_hessian=precomputed_hessians.get(mol_idx),
                )

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

        Delegates to each evaluator's ``gradient()`` method where
        analytical gradients are available.  For evaluators that do not
        yet support analytical gradients, falls back to finite-difference
        approximation of that evaluator's score contribution.

        The score is ``sum_i (w_i * (ref_i - calc_i))**2``, so each
        evaluator computes:

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

        Note:
            Evaluators that support analytical gradients (e.g. energy via
            ``energy_and_param_grad``) are used directly.  Evaluators that
            do not support them are handled transparently via central
            finite-difference fallback — no error is raised.

        Note:
            ``n_eval`` and ``history`` track only objective-function
            evaluations made through ``__call__``.  The finite-difference
            gradient evaluations performed here are internal to the
            gradient computation and are intentionally excluded from
            those counters.

        """
        ff = self.forcefield.with_params(param_vector)
        n_params = len(param_vector)
        total_grad = np.zeros(n_params)

        # Group references by molecule and evaluator kind
        refs_by_mol: dict[int, dict[str, list[ReferenceValue]]] = {}
        for ref in self.reference.values:
            mol_refs = refs_by_mol.setdefault(ref.molecule_idx, {})
            # Map kinds to evaluator categories
            category = self._kind_to_category(ref.kind)
            mol_refs.setdefault(category, []).append(ref)

        # Process each molecule's evaluator contributions
        for mol_idx, category_refs in refs_by_mol.items():
            mol = self.molecules[mol_idx]

            for category, refs in category_refs.items():
                evaluator = self._get_evaluator(category)
                if evaluator.supports_analytical_gradient(self.engine):
                    # _get_structure returns a cached handle for engines
                    # that support runtime params, or the raw molecule
                    # for stateless backends — no exception-based dispatch.
                    structure = self._get_structure(mol_idx)
                    grad = evaluator.gradient(
                        self.engine,
                        mol,
                        ff,
                        refs,
                        n_params,
                        structure=structure,
                    )
                    total_grad += grad
                else:
                    # Finite-difference fallback for this evaluator's contribution
                    grad = self._finite_difference_gradient(
                        param_vector,
                        mol_idx,
                        category,
                        refs,
                    )
                    total_grad += grad

        # Warn about zero-gradient slots which may indicate incomplete
        # analytical gradient support (e.g. missing improper torsions).
        n_zero = int(np.sum(total_grad == 0))
        if n_zero > 0:
            logger.debug(
                "gradient: %d/%d parameter slots have zero gradient",
                n_zero,
                len(total_grad),
            )

        return total_grad

    def _finite_difference_gradient(
        self,
        param_vector: np.ndarray,
        mol_idx: int,
        category: str,
        refs: list[ReferenceValue],
        step: float | None = None,
    ) -> np.ndarray:
        """Compute finite-difference gradient for one evaluator's contribution.

        Uses central differences: ``(f(x+h) - f(x-h)) / (2h)`` for each
        parameter, where ``f`` is the sum-of-squared weighted residuals
        from *refs* only.

        .. warning::

            For ``frequency`` and ``eigenmatrix`` categories, the FD
            perturbation evaluates at the *original* (unperturbed)
            geometry rather than re-optimizing at each perturbed
            parameter set.  This is an approximation — the true
            derivative includes an implicit geometry-relaxation term.
            For small parameter perturbations this is usually
            acceptable, but the resulting gradient may be inaccurate
            when the potential energy surface is highly anharmonic.

        Args:
            param_vector: Current parameter vector.
            mol_idx: Molecule index.
            category: Evaluator category (``"energy"``, ``"frequency"``,
                ``"geometry"``, or ``"eigenmatrix"``).
            refs: Reference values for this evaluator and molecule.
            step: Finite-difference step size. Defaults to
                :attr:`fd_step` (configurable, initially ``1e-4``).

        Returns:
            Gradient vector of shape ``(n_params,)``.

        """
        if step is None:
            step = self.fd_step
        n_params = len(param_vector)
        grad = np.zeros(n_params)

        for j in range(n_params):
            params_plus = param_vector.copy()
            params_plus[j] += step
            score_plus = self._partial_score(params_plus, mol_idx, category, refs)

            params_minus = param_vector.copy()
            params_minus[j] -= step
            score_minus = self._partial_score(params_minus, mol_idx, category, refs)

            grad[j] = (score_plus - score_minus) / (2.0 * step)

        return grad

    def _partial_score(
        self,
        param_vector: np.ndarray,
        mol_idx: int,
        category: str,
        refs: list[ReferenceValue],
    ) -> float:
        """Evaluate score contribution from a subset of references.

        .. warning::

            For ``frequency`` and ``eigenmatrix`` categories this
            evaluates at the *unperturbed* geometry.  Strictly, the
            Hessian (and therefore frequencies / eigenmatrix) should
            be computed at the minimum-energy geometry for the
            perturbed parameters.  See the note on
            :meth:`_finite_difference_gradient`.

        Args:
            param_vector: Parameter vector to evaluate.
            mol_idx: Molecule index.
            category: Evaluator category.
            refs: Reference values to score.

        Returns:
            Sum-of-squared weighted residuals for the given references.

        """
        ff = self.forcefield.with_params(param_vector)
        mol = self.molecules[mol_idx]
        evaluator = self._get_evaluator(category)

        if category == "geometry":
            needed_kinds = frozenset(r.kind for r in refs)
            computed = evaluator.compute(self.engine, mol, ff, needed_kinds=needed_kinds)
        elif category == "eigenmatrix":
            # TODO(#149): Re-optimize geometry at perturbed parameters before
            # computing the eigenmatrix.  Currently evaluates the Hessian at the
            # original geometry, which is an approximation.
            structure = self._get_structure(mol_idx)
            computed = evaluator.compute(
                self.engine,
                mol,
                ff,
                structure=structure,
                mol_idx=mol_idx,
            )
        elif category == "frequency":
            # TODO(#149): Re-optimize geometry at perturbed parameters before
            # computing frequencies.  Currently evaluates the Hessian at the
            # original geometry, which is an approximation.
            structure = self._get_structure(mol_idx)
            computed = evaluator.compute(self.engine, mol, ff, structure=structure)
        else:
            structure = self._get_structure(mol_idx)
            computed = evaluator.compute(self.engine, mol, ff, structure=structure)

        residuals = evaluator.residuals(computed, refs)
        return float(np.sum(np.array(residuals) ** 2))

    @staticmethod
    def _kind_to_category(kind: str) -> str:
        """Map a reference value kind to its evaluator category.

        Args:
            kind: Reference value kind string.

        Returns:
            Evaluator category: ``"energy"``, ``"frequency"``,
            ``"geometry"``, ``"eigenmatrix"``, or ``"hessian"``.

        Raises:
            ValueError: If the kind is unknown.

        """
        _KIND_CATEGORIES = {
            "energy": "energy",
            "frequency": "frequency",
            "bond_length": "geometry",
            "bond_angle": "geometry",
            "torsion_angle": "geometry",
            "eig_diagonal": "eigenmatrix",
            "eig_offdiagonal": "eigenmatrix",
            "hessian_element": "hessian",
        }
        if kind not in _KIND_CATEGORIES:
            raise ValueError(f"Unknown reference kind: {kind}")
        return _KIND_CATEGORIES[kind]

    def _get_evaluator(self, category: str) -> Any:
        """Get the evaluator instance for a category.

        Args:
            category: Evaluator category string.

        Returns:
            The evaluator instance.

        Raises:
            ValueError: If the category is unknown.

        """
        # Find the first evaluator whose HANDLED_KINDS intersects with
        # the kinds in this category.
        for ev in self._evaluators:
            # Check if this evaluator handles any kind in the category
            for kind in ev.HANDLED_KINDS:
                if self._kind_to_category(kind) == category:
                    return ev
        raise ValueError(f"Unknown evaluator category: {category}")

    def _get_structure(self, mol_idx: int) -> Any:
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

    def _evaluate_molecule(
        self,
        mol_idx: int,
        forcefield: ForceField,
        *,
        precomputed_hessian: np.ndarray | None = None,
    ) -> dict:
        """Run MM calculations for a single molecule.

        Delegates to per-data-type evaluators where available, populating
        the results dict with the same keys for backward compatibility.

        When *precomputed_hessian* is provided (from batched vmap
        evaluation), it is used directly for frequency and eigenmatrix
        calculations, avoiding a redundant ``engine.hessian()`` call.

        Args:
            mol_idx (int): Index into the molecules list.
            forcefield: The force field to evaluate (typically a temporary
                instance from :meth:`~ForceField.with_params`).
            precomputed_hessian: Optional ``(3N, 3N)`` Hessian in
                Hartree/Bohr² from batched evaluation.

        Returns:
            dict: Calculated results keyed by data type (e.g.
                ``"energy"``, ``"frequencies"``, ``"bond_lengths"``).

        """
        mol = self.molecules[mol_idx]
        structure = self._get_structure(mol_idx)
        result: dict = {}

        # Determine what data types are needed for this molecule
        needed = {ref.kind for ref in self.reference.values if ref.molecule_idx == mol_idx}

        energy_ev = self._kind_to_evaluator.get("energy")
        freq_ev = self._kind_to_evaluator.get("frequency")
        geom_ev = self._kind_to_evaluator.get("bond_length")
        eigm_ev = self._kind_to_evaluator.get("eig_diagonal")

        if "energy" in needed and energy_ev is not None:
            er = energy_ev.compute(self.engine, mol, forcefield, structure=structure)
            result["energy"] = er.energy

        if "frequency" in needed and freq_ev is not None:
            if precomputed_hessian is not None:
                from q2mm.models.hessian import hessian_to_frequencies

                result["frequencies"] = hessian_to_frequencies(
                    precomputed_hessian,
                    list(mol.symbols),
                )
            else:
                fr = freq_ev.compute(self.engine, mol, forcefield, structure=structure)
                result["frequencies"] = fr.frequencies

        if geom_ev is not None and needed & geom_ev.HANDLED_KINDS:
            geo_needed = frozenset(needed & geom_ev.HANDLED_KINDS)
            # Geometry evaluator runs minimize internally on the raw molecule.
            gr = geom_ev.compute(
                self.engine,
                mol,
                forcefield,
                needed_kinds=geo_needed,
            )
            if "bond_length" in geo_needed:
                result["bond_lengths"] = gr.bond_lengths
                result["bond_lengths_by_atoms"] = gr.bond_lengths_by_atoms
            if "bond_angle" in geo_needed:
                result["bond_angles"] = gr.bond_angles
                result["bond_angles_by_atoms"] = gr.bond_angles_by_atoms
            if "torsion_angle" in geo_needed:
                result["torsion_coords"] = gr.torsion_coords

        if eigm_ev is not None and needed & eigm_ev.HANDLED_KINDS:
            if precomputed_hessian is not None:
                from q2mm.models.hessian import decompose, transform_to_eigenmatrix

                # Use precomputed Hessian to build eigenmatrix directly
                eigm_evaluator = eigm_ev
                if mol_idx not in eigm_evaluator._qm_eigenvectors:
                    if mol.hessian is None:
                        raise ValueError(
                            f"Molecule {mol_idx} ({mol.name}) has no QM Hessian. "
                            "Eigenmatrix training requires a QM Hessian for the "
                            "eigenvector basis."
                        )
                    _, qm_evecs = decompose(mol.hessian)
                    eigm_evaluator._qm_eigenvectors[mol_idx] = qm_evecs

                qm_evecs = eigm_evaluator._qm_eigenvectors[mol_idx]
                result["eigenmatrix"] = transform_to_eigenmatrix(
                    precomputed_hessian,
                    qm_evecs,
                )
            else:
                emr = eigm_ev.compute(
                    self.engine,
                    mol,
                    forcefield,
                    structure=structure,
                    mol_idx=mol_idx,
                )
                result["eigenmatrix"] = emr.eigenmatrix

        hess_ev = self._kind_to_evaluator.get("hessian_element")
        if "hessian_element" in needed and hess_ev is not None:
            hr = hess_ev.compute(self.engine, mol, forcefield, structure=structure)
            result["raw_hessian"] = hr.hessian

        return result

    @staticmethod
    def _extract_value(calc: dict, ref: ReferenceValue) -> float:
        """Extract a calculated value matching a reference observation.

        Delegates to per-data-type evaluators for extraction logic.
        Uses each evaluator's ``HANDLED_KINDS`` to find the right handler.

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
        from q2mm.optimizers.evaluators.eigenmatrix import EigenmatrixEvaluator
        from q2mm.optimizers.evaluators.energy import EnergyEvaluator
        from q2mm.optimizers.evaluators.frequency import FrequencyEvaluator
        from q2mm.optimizers.evaluators.geometry import GeometryEvaluator
        from q2mm.optimizers.evaluators.hessian_element import HessianElementEvaluator

        _EVALUATOR_CLASSES = [
            EnergyEvaluator,
            FrequencyEvaluator,
            GeometryEvaluator,
            EigenmatrixEvaluator,
            HessianElementEvaluator,
        ]
        for cls in _EVALUATOR_CLASSES:
            if ref.kind in cls.HANDLED_KINDS:
                return cls.extract_value(calc, ref)
        raise ValueError(f"Unknown reference kind: {ref.kind}")

    def reset(self) -> None:
        """Reset evaluation counter, history, and cached engine handles."""
        self.n_eval = 0
        self.history.clear()
        self._handles.clear()
        for ev in self._evaluators:
            if hasattr(ev, "reset"):
                ev.reset()
