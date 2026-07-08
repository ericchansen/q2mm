"""Reference data containers for force field optimization.

Defines :class:`ReferenceValue` (a single QM or experimental observation) and
:class:`ReferenceData` (a collection of observations plus builder
constructors).  Split out of :mod:`q2mm.optimizers.objective` so that code
which only needs to *build* reference data does not pull in the heavier
:class:`~q2mm.optimizers.objective.ObjectiveFunction` machinery.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from q2mm.constants import DEFAULT_BOND_TOLERANCE
from q2mm.models.molecule import Q2MMMolecule


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
        symbols: Sequence[str] | None = None,
        diagonal_only: bool = False,
        molecule_idx: int = 0,
        weights: dict[str, float] | None = None,
        skip_first: bool = True,
        eigenvalue_threshold: float = 0.1173,
        n_rigid_modes: int = 6,
    ) -> int:
        """Bulk-load eigenmatrix training data from a QM Hessian.

        Builds the reference eigenmatrix by projecting the Hessian onto the
        molecule's **normal modes** (the eigenvectors of the mass-weighted
        Hessian) in the mass-weighted metric, following the Q2MM eigenmatrix
        protocol (Farrugia, Helquist, Norrby & Wiest 2025, *J. Chem. Theory
        Comput.* **22**, 469).  The diagonal elements are the mass-weighted
        reference eigenvalues; the off-diagonal elements are zero for a
        perfectly fit force field.

        The Hessian should be in canonical units (Hartree/Bohr²).

        Args:
            hessian (np.ndarray): QM Hessian matrix ``(3N, 3N)`` in
                Hartree/Bohr².
            symbols (Sequence[str] | None): Element symbols (length *N*)
                used to mass-weight the Hessian.  Required for the
                normal-mode projection; if ``None``, the raw (non
                mass-weighted) Cartesian eigenbasis is used as a fallback
                (legacy behaviour).
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
            eigenvalue_threshold (float): Mass-weighted eigenvalue
                threshold separating low/high frequency modes for weight
                assignment.  (Numerically inert while ``eig_d_low`` equals
                ``eig_d_high``.)
            n_rigid_modes (int): Number of rigid-body (translation +
                rotation) modes to exclude from the fit, **in addition to**
                the ``skip_first`` reaction-coordinate mode.  The
                ``n_rigid_modes`` smallest mass-weighted eigenvalue-magnitude
                modes (chosen after the reaction-coordinate mode is set aside)
                are given weight ``eig_i`` (0.0), and any off-diagonal element
                touching one of them (or the skipped reaction-coordinate mode)
                is likewise zero-weighted.  Defaults to 6 (non-linear
                molecules; use 5 for linear).  A count-based criterion is used
                rather than a magnitude tolerance because genuine
                low-frequency vibrations (tens of cm⁻¹) can have eigenvalue
                magnitudes comparable to a loose tolerance, whereas the true
                rigid-body modes are always the few smallest.

        Returns:
            int: Number of entries added.

        """
        from q2mm.models.hessian import (
            decompose,
            extract_eigenmatrix_data,
            mass_weighted_eigenmatrix,
            mass_weighted_normal_modes,
            transform_to_eigenmatrix,
        )

        w = {"eig_i": 0.0, "eig_d_low": 0.1, "eig_d_high": 0.1, "eig_o": 0.05}
        if weights:
            w.update(weights)

        if symbols is not None:
            eigenvalues, eigenvectors = mass_weighted_normal_modes(hessian, symbols)
            eigenmatrix = mass_weighted_eigenmatrix(hessian, eigenvectors, symbols)
        else:
            eigenvalues, eigenvectors = decompose(hessian)
            eigenmatrix = transform_to_eigenmatrix(hessian, eigenvectors)
        elements = extract_eigenmatrix_data(eigenmatrix, diagonal_only=diagonal_only)

        # Modes carrying no force-constant information, excluded (weight
        # ``eig_i``) from both the diagonal and any off-diagonal target:
        #   * the reaction-coordinate mode (``skip_first`` — the most-negative
        #     eigenvalue, i.e. index 0 after the ascending eigh sort), and
        #   * the ``n_rigid_modes`` rigid-body (translation/rotation) modes,
        #     the smallest mass-weighted eigenvalue magnitude.
        # The reaction-coordinate mode is removed from the rigid-mode candidate
        # pool first, so a *soft* imaginary mode (whose |eigenvalue| may itself
        # be among the smallest) cannot displace a genuine rigid-body mode.
        # The number of excluded modes is therefore a consistent
        # ``skip_first + n_rigid_modes`` regardless of the imaginary-mode
        # magnitude.
        excluded_modes: set[int] = set()
        rigid_candidates = list(np.argsort(np.abs(eigenvalues)))
        if skip_first and len(eigenvalues):
            excluded_modes.add(0)
            rigid_candidates = [m for m in rigid_candidates if m != 0]
        if n_rigid_modes > 0 and rigid_candidates:
            n_rigid = min(n_rigid_modes, len(rigid_candidates))
            excluded_modes.update(int(m) for m in rigid_candidates[:n_rigid])

        added = 0
        for row, col, value in elements:
            if row == col:
                # Diagonal element
                if row in excluded_modes:
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
                # Off-diagonal element — zero-weighted if it couples an
                # excluded (rigid-body / reaction-coordinate) mode.
                weight = 0.0 if (row in excluded_modes or col in excluded_modes) else w["eig_o"]
                self.add_hessian_offdiagonal(
                    value,
                    row=row,
                    col=col,
                    weight=weight,
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
        include_geometry: bool = True,
        include_eigenmatrix: bool = True,
        eigenmatrix_diagonal_only: bool = False,
        eigenmatrix_hessian: np.ndarray | None = None,
    ) -> ReferenceData:
        """Auto-populate reference data from a molecule's detected geometry.

        Extracts all auto-detected bond lengths and bond angles from the
        molecule. By default, also adds Hessian eigenmatrix training data
        when a Hessian is available (this matches the standard Q2MM
        workflow from the literature). Vibrational frequencies and raw
        Hessian elements are **not** included by default.

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
            include_geometry (bool): If ``True`` (the default), add bond
                length and bond angle reference data.  Set to ``False``
                to build an eigenmatrix-only objective (useful for
                QFUERZA validation where geometry terms dominate).
            include_eigenmatrix (bool): If ``True`` (the default) and the
                molecule has a Hessian, add eigenmatrix training data
                (diagonal and optionally off-diagonal elements).
            eigenmatrix_diagonal_only (bool): If ``True``, only diagonal
                eigenmatrix elements are added.
            eigenmatrix_hessian (np.ndarray | None): Optional override for
                the Hessian used to build eigenmatrix references.  When
                ``None`` (default), uses ``mol.hessian`` directly (Limé
                & Norrby Method D — unmodified Hessian).  When provided,
                this Hessian is used instead — a true override, so it
                takes effect even when ``mol.hessian is None`` (Limé &
                Norrby Method C — pass an inverted Hessian here for
                Round 2 of the Method E2 protocol; geometry references
                and any explicit ``frequencies`` still use the
                unmodified ``mol.hessian`` / passed values).

        Returns:
            ReferenceData: Populated with bond lengths, angles, and
                (by default) eigenmatrix data when a Hessian is present.

        """
        w = {"bond_length": 10.0, "bond_angle": 5.0, "frequency": 1.0}
        if weights:
            w.update(weights)

        ref = cls()

        if include_geometry:
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

        if include_eigenmatrix:
            # ``eigenmatrix_hessian`` is a *true* override — it stands in
            # for ``mol.hessian`` for eigenmatrix construction even when
            # the base ``mol.hessian`` is ``None``.  If neither is
            # available, no eigenmatrix block is built.
            hess_for_eigenmatrix = eigenmatrix_hessian if eigenmatrix_hessian is not None else mol.hessian
            if hess_for_eigenmatrix is not None:
                eig_weights = {k: w[k] for k in ("eig_i", "eig_d_low", "eig_d_high", "eig_o") if k in w}
                ref.add_eigenmatrix_from_hessian(
                    hess_for_eigenmatrix,
                    symbols=list(mol.symbols),
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
        include_geometry: bool = True,
        include_eigenmatrix: bool = True,
        eigenmatrix_diagonal_only: bool = False,
        eigenmatrix_hessians: list[np.ndarray] | None = None,
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
            include_geometry (bool): If ``True`` (the default), add bond
                length and bond angle reference data.  Set to ``False``
                for eigenmatrix-only objectives.
            include_eigenmatrix (bool): If ``True`` (the default) and a
                molecule has a Hessian, add eigenmatrix data.
            eigenmatrix_diagonal_only (bool): If ``True``, only diagonal
                eigenmatrix elements are added.
            eigenmatrix_hessians (list[np.ndarray] | None): Optional
                per-molecule Hessian override for eigenmatrix
                construction.  When ``None`` (default), each molecule's
                ``mol.hessian`` is used (Limé & Norrby Method D).  When
                provided, must have the same length as *molecules*; the
                override Hessian is used per molecule (Limé & Norrby
                Method C — pass inverted Hessians here for Round 2 of
                the Method E2 protocol).

        Returns:
            ReferenceData: Combined reference data for all molecules.

        Raises:
            ValueError: If ``frequencies_list`` or ``eigenmatrix_hessians``
                length does not match ``molecules`` length.

        """
        if frequencies_list is not None and len(frequencies_list) != len(molecules):
            raise ValueError(
                f"frequencies_list length ({len(frequencies_list)}) must match molecules length ({len(molecules)})."
            )
        if eigenmatrix_hessians is not None and len(eigenmatrix_hessians) != len(molecules):
            raise ValueError(
                f"eigenmatrix_hessians length ({len(eigenmatrix_hessians)}) must match "
                f"molecules length ({len(molecules)})."
            )

        ref = cls()
        for idx, mol in enumerate(molecules):
            single = cls.from_molecule(
                mol,
                weights=weights,
                molecule_idx=idx,
                frequencies=frequencies_list[idx] if frequencies_list is not None else None,
                skip_imaginary=skip_imaginary,
                include_geometry=include_geometry,
                include_eigenmatrix=include_eigenmatrix,
                eigenmatrix_diagonal_only=eigenmatrix_diagonal_only,
                eigenmatrix_hessian=eigenmatrix_hessians[idx] if eigenmatrix_hessians is not None else None,
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
        include_frequencies: bool = False,
        skip_imaginary: bool = False,
        au_hessian: bool = True,
        include_eigenmatrix: bool = True,
    ) -> tuple[ReferenceData, Q2MMMolecule]:
        """Build reference data from a Gaussian log file.

        Parses the log file for the optimised geometry and vibrational
        frequencies, then auto-populates bond lengths, angles, and
        (when a Hessian is available) eigenmatrix data.

        By default, **frequencies are not included** — eigenmatrix
        training from the Hessian is the standard Q2MM approach per
        Norrby & Liljefors (1998). Set ``include_frequencies=True``
        to add them.

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
                from the log file. Default is ``False``.
            skip_imaginary (bool): If ``True``, negative frequencies are
                skipped.
            au_hessian (bool): Keep Hessian in atomic units
                (Hartree/Bohr²).
            include_eigenmatrix (bool): If ``True`` (the default) and a
                Hessian is available, add eigenmatrix training data.

        Returns:
            tuple[ReferenceData, Q2MMMolecule]: Populated reference data
                and the parsed molecule (with Hessian attached if
                available).

        """
        from q2mm.io.gaussian import GaussLog

        log = GaussLog(str(path), au_hessian=au_hessian)

        # Build molecule from the last (optimised) structure
        mol = log.molecules[-1]
        mol.charge = charge
        mol.multiplicity = multiplicity
        mol.bond_tolerance = bond_tolerance

        # ``mol.hessian`` carries the archive Cartesian Hessian (Hartree/Bohr²,
        # full rank 3N, imaginary mode intact) in a frame consistent with the
        # geometry.  Do NOT override it with a reconstruction from
        # ``log.evals``/``log.evecs`` — those come from Gaussian's mass-weighted
        # frequency analysis and would reintroduce a ~√(mᵢmⱼ) error into every
        # heavy-atom force constant.

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
            include_eigenmatrix=include_eigenmatrix,
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
        from q2mm.io.fchk import parse_fchk as _parse_fchk  # noqa: E402

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

        Delegates to :func:`q2mm.io.reference.load_reference_yaml`.

        Args:
            path (str | Path): Path to the YAML reference file.

        Returns:
            tuple[ReferenceData, list[Q2MMMolecule]]: Loaded reference
                data and parsed molecules.

        """
        from q2mm.io.reference import load_reference_yaml

        return load_reference_yaml(path)

    def to_yaml(
        self,
        path: str | Path,
        molecules: list[Q2MMMolecule],
    ) -> None:
        """Save this reference data and molecules to a YAML file.

        Delegates to :func:`q2mm.io.reference.save_reference_yaml`.

        Args:
            path (str | Path): Output file path.
            molecules (list[Q2MMMolecule]): Molecules corresponding to
                the reference values.

        """
        from q2mm.io.reference import save_reference_yaml

        save_reference_yaml(path, self, molecules)
