"""Canonical observation vocabulary for Q2MM force field optimization.

Defines :class:`Observation` (a single QM or experimental observation) and
:class:`ObservationSet` (an immutable collection of observations plus the
pure builder methods that populate it from a
:class:`~q2mm.models.molecule.Molecule`). This is the *one* observation
model in Q2MM: :mod:`q2mm.io.reference` (YAML), :mod:`q2mm.io.fchk`, and
:mod:`q2mm.io.gaussian` construct :class:`ObservationSet` from parsed
files, and :mod:`q2mm.objectives` consumes it — but this module
itself never imports I/O, backends, objectives, optimizers, or workflows.

Both :class:`Observation` and :class:`ObservationSet` are deeply
immutable value objects: ``Observation`` is a frozen dataclass and
``ObservationSet.values`` is a tuple. There is no in-place mutation
anywhere in the public API — every ``with_*`` method returns a *new*
:class:`ObservationSet` built from ``self.values`` plus the added
entry/entries, leaving ``self`` untouched. Bulk loaders
(:meth:`ObservationSet.with_eigenmatrix_from_hessian`,
:meth:`ObservationSet.with_hessian_from_matrix`,
:meth:`ObservationSet.with_frequencies_from_array`) build their batch of
new :class:`Observation` instances in a private, transient list — an
implementation detail invisible to callers — and freeze it into the
returned :class:`ObservationSet` in one step.

Each :class:`Observation` binds to a training case via the stable
:attr:`Observation.case_id` string (matching
:attr:`q2mm.models.problem.TrainingCase.case_id`), never a positional
index — case order is free to change (reordering, insertion, filtering)
without silently re-binding any observation to the wrong molecule.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from q2mm.models.molecule import Molecule

_ObservationKind = Literal[
    "energy",
    "frequency",
    "bond_length",
    "bond_angle",
    "torsion_angle",
    "eig_diagonal",
    "eig_offdiagonal",
    "hessian_element",
]


@dataclass(frozen=True)
class Observation:
    """A single, immutable reference observation (QM or experimental)."""

    kind: _ObservationKind
    value: float
    weight: float = 1.0
    label: str = ""
    # Stable identity of the training case this observation belongs to —
    # matches q2mm.models.problem.TrainingCase.case_id. Never a positional
    # index: case order must be free to change without silently re-binding
    # observations to the wrong molecule.
    case_id: str = "0"
    # Indices for matching to calculated data
    data_idx: int = 0
    # Atom-identity matching (preferred over positional data_idx for geometry)
    atom_indices: tuple[int, ...] | None = None


def _energy_observation(value: float, *, weight: float, case_id: str, label: str) -> Observation:
    return Observation(kind="energy", value=value, weight=weight, case_id=case_id, label=label)


def _frequency_observation(value: float, *, data_idx: int, weight: float, case_id: str, label: str) -> Observation:
    return Observation(kind="frequency", value=value, weight=weight, case_id=case_id, data_idx=data_idx, label=label)


def _bond_length_observation(
    value: float,
    *,
    data_idx: int,
    atom_indices: tuple[int, int] | None,
    weight: float,
    case_id: str,
    label: str,
) -> Observation:
    if atom_indices is None and data_idx < 0:
        raise ValueError("Either atom_indices or data_idx must be provided for bond_length.")
    return Observation(
        kind="bond_length",
        value=value,
        weight=weight,
        case_id=case_id,
        data_idx=max(data_idx, 0),
        atom_indices=atom_indices,
        label=label,
    )


def _bond_angle_observation(
    value: float,
    *,
    data_idx: int,
    atom_indices: tuple[int, int, int] | None,
    weight: float,
    case_id: str,
    label: str,
) -> Observation:
    if atom_indices is None and data_idx < 0:
        raise ValueError("Either atom_indices or data_idx must be provided for bond_angle.")
    return Observation(
        kind="bond_angle",
        value=value,
        weight=weight,
        case_id=case_id,
        data_idx=max(data_idx, 0),
        atom_indices=atom_indices,
        label=label,
    )


def _torsion_angle_observation(
    value: float, *, atom_indices: tuple[int, int, int, int], weight: float, case_id: str, label: str
) -> Observation:
    return Observation(
        kind="torsion_angle", value=value, weight=weight, case_id=case_id, atom_indices=atom_indices, label=label
    )


def _hessian_eigenvalue_observation(
    value: float, *, mode_idx: int, weight: float, case_id: str, label: str
) -> Observation:
    return Observation(kind="eig_diagonal", value=value, weight=weight, case_id=case_id, data_idx=mode_idx, label=label)


def _hessian_offdiagonal_observation(
    value: float, *, row: int, col: int, weight: float, case_id: str, label: str
) -> Observation:
    return Observation(
        kind="eig_offdiagonal", value=value, weight=weight, case_id=case_id, atom_indices=(row, col), label=label
    )


def _hessian_element_observation(
    value: float, *, row: int, col: int, weight: float, case_id: str, label: str
) -> Observation:
    if row < 0 or col < 0:
        raise ValueError(f"row and col must be non-negative, got row={row}, col={col}")
    return Observation(
        kind="hessian_element",
        value=value,
        weight=weight,
        case_id=case_id,
        atom_indices=(row, col),
        label=label or f"hess[{row},{col}]",
    )


@dataclass(frozen=True)
class ObservationSet:
    """Complete, immutable set of reference observations for an optimization.

    Each entry describes one observable: an energy, a frequency, or a
    geometric parameter that the force field should reproduce. Every
    ``with_*`` method is pure — it returns a new :class:`ObservationSet`
    and never mutates ``self`` or any argument.
    """

    values: tuple[Observation, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", tuple(self.values))

    def _with(self, *new_observations: Observation) -> ObservationSet:
        """Return a new :class:`ObservationSet` with *new_observations* appended."""
        return ObservationSet(values=(*self.values, *new_observations))

    def with_energy(
        self,
        value: float,
        *,
        weight: float = 1.0,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single energy reference value added.

        Args:
            value (float): Reference energy value.
            weight (float): Weight for this entry.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        """
        return self._with(_energy_observation(value, weight=weight, case_id=case_id, label=label))

    def with_frequency(
        self,
        value: float,
        *,
        data_idx: int,
        weight: float = 1.0,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single vibrational frequency reference value added.

        Args:
            value (float): Reference frequency in cm⁻¹.
            data_idx (int): 0-based index of this frequency mode.
            weight (float): Weight for this entry.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        """
        return self._with(_frequency_observation(value, data_idx=data_idx, weight=weight, case_id=case_id, label=label))

    def with_bond_length(
        self,
        value: float,
        *,
        data_idx: int = -1,
        atom_indices: tuple[int, int] | None = None,
        weight: float = 1.0,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single bond length reference value added.

        Args:
            value (float): Reference bond length in Ångströms.
            data_idx (int): 0-based positional index (fallback if
                ``atom_indices`` is not provided).
            atom_indices (tuple[int, int] | None): Atom pair for
                identity-based matching.
            weight (float): Weight for this entry.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        Raises:
            ValueError: If neither ``atom_indices`` nor a non-negative
                ``data_idx`` is provided.

        """
        return self._with(
            _bond_length_observation(
                value, data_idx=data_idx, atom_indices=atom_indices, weight=weight, case_id=case_id, label=label
            )
        )

    def with_bond_angle(
        self,
        value: float,
        *,
        data_idx: int = -1,
        atom_indices: tuple[int, int, int] | None = None,
        weight: float = 1.0,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single bond angle reference value added.

        Args:
            value (float): Reference bond angle in degrees.
            data_idx (int): 0-based positional index (fallback if
                ``atom_indices`` is not provided).
            atom_indices (tuple[int, int, int] | None): Atom triple
                (i, j, k) for identity-based matching.
            weight (float): Weight for this entry.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        Raises:
            ValueError: If neither ``atom_indices`` nor a non-negative
                ``data_idx`` is provided.

        """
        return self._with(
            _bond_angle_observation(
                value, data_idx=data_idx, atom_indices=atom_indices, weight=weight, case_id=case_id, label=label
            )
        )

    def with_torsion_angle(
        self,
        value: float,
        *,
        atom_indices: tuple[int, int, int, int],
        weight: float = 1.0,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single torsion (dihedral) angle reference value added.

        Args:
            value (float): Reference torsion angle in degrees.
            atom_indices (tuple[int, int, int, int]): Four atom indices
                defining the dihedral.
            weight (float): Weight for this entry.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        """
        return self._with(
            _torsion_angle_observation(value, atom_indices=atom_indices, weight=weight, case_id=case_id, label=label)
        )

    def with_hessian_eigenvalue(
        self,
        value: float,
        *,
        mode_idx: int,
        weight: float = 0.1,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a diagonal eigenmatrix element (eigenvalue) added.

        Args:
            value (float): QM eigenvalue for this mode.
            mode_idx (int): 0-based index of the vibrational mode.
            weight (float): Weight for this entry. Legacy defaults: 0.10
                for both low- and high-frequency modes, 0.00 for the
                first (imaginary) mode.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        """
        return self._with(
            _hessian_eigenvalue_observation(value, mode_idx=mode_idx, weight=weight, case_id=case_id, label=label)
        )

    def with_hessian_offdiagonal(
        self,
        value: float,
        *,
        row: int,
        col: int,
        weight: float = 0.05,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with an off-diagonal eigenmatrix element added.

        Off-diagonal elements measure cross-coupling between modes.
        They should be close to zero when the MM Hessian closely
        reproduces the QM eigenvector structure.

        Args:
            value (float): QM eigenmatrix element (typically 0.0 for
                the QM self-projection).
            row (int): 0-based row index into the eigenmatrix.
            col (int): 0-based column index into the eigenmatrix.
            weight (float): Weight for this entry. Legacy default: 0.05.
            case_id (str): Stable ID of the training case this observation
                belongs to.
            label (str): Human-readable label.

        """
        return self._with(
            _hessian_offdiagonal_observation(value, row=row, col=col, weight=weight, case_id=case_id, label=label)
        )

    def with_eigenmatrix_from_hessian(
        self,
        hessian: np.ndarray,
        *,
        symbols: Sequence[str] | None = None,
        diagonal_only: bool = False,
        case_id: str = "0",
        weights: dict[str, float] | None = None,
        skip_first: bool = True,
        eigenvalue_threshold: float = 0.1173,
        n_rigid_modes: int = 6,
    ) -> ObservationSet:
        """Return a new set with bulk eigenmatrix training data from a QM Hessian.

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
            case_id (str): Stable ID of the training case these
                observations belong to.
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
            ObservationSet: A new set with the eigenmatrix entries appended.

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

        new_observations: list[Observation] = []
        for row, col, value in elements:
            if row == col:
                # Diagonal element
                if row in excluded_modes:
                    weight = w["eig_i"]
                elif value < eigenvalue_threshold:
                    weight = w["eig_d_low"]
                else:
                    weight = w["eig_d_high"]
                new_observations.append(
                    _hessian_eigenvalue_observation(
                        value, mode_idx=row, weight=weight, case_id=case_id, label=f"eig[{row}]"
                    )
                )
            else:
                # Off-diagonal element — zero-weighted if it couples an
                # excluded (rigid-body / reaction-coordinate) mode.
                weight = 0.0 if (row in excluded_modes or col in excluded_modes) else w["eig_o"]
                new_observations.append(
                    _hessian_offdiagonal_observation(
                        value, row=row, col=col, weight=weight, case_id=case_id, label=f"eig[{row},{col}]"
                    )
                )
        return self._with(*new_observations)

    def with_hessian_element(
        self,
        value: float,
        *,
        row: int,
        col: int,
        weight: float = 0.1,
        case_id: str = "0",
        label: str = "",
    ) -> ObservationSet:
        """Return a new set with a single raw Hessian matrix element added.

        Args:
            value: QM Hessian element in Hartree/Bohr².
            row: 0-based row index.
            col: 0-based column index.
            weight: Weight for this entry.
            case_id: Stable ID of the training case this observation
                belongs to.
            label: Human-readable label.

        Raises:
            ValueError: If *row* or *col* is negative.

        """
        return self._with(
            _hessian_element_observation(value, row=row, col=col, weight=weight, case_id=case_id, label=label)
        )

    def with_hessian_from_matrix(
        self,
        hessian: np.ndarray,
        *,
        diagonal_only: bool = False,
        case_id: str = "0",
        diagonal_weight: float = 0.1,
        offdiagonal_weight: float = 0.05,
        skip_translational: int = 0,
    ) -> ObservationSet:
        """Return a new set with bulk raw Hessian elements added.

        Unlike :meth:`with_eigenmatrix_from_hessian`, this uses the raw
        Cartesian Hessian directly without eigendecomposition.

        Args:
            hessian: QM Hessian (3N, 3N) in Hartree/Bohr².
            diagonal_only: If ``True``, only add diagonal elements.
            case_id: Stable ID of the training case these observations
                belong to.
            diagonal_weight: Weight for diagonal elements.
            offdiagonal_weight: Weight for off-diagonal elements.
            skip_translational: Number of leading rows/cols to skip
                (e.g. 6 for trans+rot modes in Cartesian basis).

        Returns:
            ObservationSet: A new set with the Hessian elements appended.

        """
        n = hessian.shape[0]
        if hessian.shape != (n, n):
            raise ValueError(f"Hessian must be square, got shape {hessian.shape}")
        if skip_translational < 0:
            raise ValueError(f"skip_translational must be non-negative, got {skip_translational}")
        if skip_translational >= n:
            raise ValueError(f"skip_translational ({skip_translational}) must be less than matrix size ({n})")

        new_observations: list[Observation] = []
        start = skip_translational
        for i in range(start, n):
            for j in range(start, i + 1) if not diagonal_only else [i]:
                weight = diagonal_weight if i == j else offdiagonal_weight
                new_observations.append(
                    _hessian_element_observation(
                        float(hessian[i, j]), row=i, col=j, weight=weight, case_id=case_id, label=""
                    )
                )
        return self._with(*new_observations)

    @property
    def n_observations(self) -> int:
        """Total number of reference observations.

        Returns:
            int: Length of the ``values`` tuple.

        """
        return len(self.values)

    # ---- Bulk loaders ----

    def with_frequencies_from_array(
        self,
        frequencies: np.ndarray | list[float],
        *,
        weight: float = 1.0,
        case_id: str = "0",
        skip_imaginary: bool = False,
    ) -> ObservationSet:
        """Return a new set with all frequencies from a 1-D array added.

        Args:
            frequencies (np.ndarray | list[float]): Vibrational frequencies
                (cm⁻¹). Imaginary modes should be negative values.
            weight (float): Weight applied to every frequency entry.
            case_id (str): Stable ID of the training case these
                observations belong to.
            skip_imaginary (bool): If ``True``, negative frequencies
                (imaginary modes) are skipped.

        Returns:
            ObservationSet: A new set with the frequency entries appended.

        """
        freqs = np.asarray(frequencies, dtype=float).ravel()
        new_observations: list[Observation] = []
        for i, freq in enumerate(freqs):
            if skip_imaginary and freq < 0:
                continue
            new_observations.append(
                _frequency_observation(float(freq), data_idx=i, weight=weight, case_id=case_id, label=f"mode {i}")
            )
        return self._with(*new_observations)

    # ---- Factory methods ----

    @classmethod
    def from_molecule(
        cls,
        mol: Molecule,
        *,
        case_id: str = "0",
        weights: dict[str, float] | None = None,
        frequencies: np.ndarray | list[float] | None = None,
        skip_imaginary: bool = False,
        include_geometry: bool = True,
        include_eigenmatrix: bool = True,
        eigenmatrix_diagonal_only: bool = False,
        eigenmatrix_hessian: np.ndarray | None = None,
    ) -> ObservationSet:
        """Auto-populate reference data from a molecule's detected geometry.

        Extracts all auto-detected bond lengths and bond angles from the
        molecule. By default, also adds Hessian eigenmatrix training data
        when a Hessian is available (this matches the standard Q2MM
        workflow from the literature). Vibrational frequencies and raw
        Hessian elements are **not** included by default.

        Args:
            mol (Molecule): Molecule with geometry (bonds/angles
                auto-detected).
            case_id (str): Stable ID of the training case (matches
                :attr:`q2mm.models.problem.TrainingCase.case_id`) every
                observation built from *mol* is bound to.
            weights (dict[str, float] | None): Weight overrides keyed by
                data type. Supported keys: ``"bond_length"``,
                ``"bond_angle"``, ``"frequency"``, and the eigenmatrix
                keys ``"eig_i"``, ``"eig_d_low"``, ``"eig_d_high"``,
                ``"eig_o"``. Defaults: ``{"bond_length": 10.0,
                "bond_angle": 5.0, "frequency": 1.0}``.
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
            ObservationSet: Populated with bond lengths, angles, and
                (by default) eigenmatrix data when a Hessian is present.

        """
        w = {"bond_length": 10.0, "bond_angle": 5.0, "frequency": 1.0}
        if weights:
            w.update(weights)

        new_observations: list[Observation] = []

        if include_geometry:
            for bond in mol.bonds or ():
                new_observations.append(
                    _bond_length_observation(
                        bond.length,
                        data_idx=-1,
                        atom_indices=(bond.atom_i, bond.atom_j),
                        weight=w["bond_length"],
                        case_id=case_id,
                        label=f"{bond.element_pair} bond",
                    )
                )

            for angle in mol.angles or ():
                new_observations.append(
                    _bond_angle_observation(
                        angle.value,
                        data_idx=-1,
                        atom_indices=(angle.atom_i, angle.atom_j, angle.atom_k),
                        weight=w["bond_angle"],
                        case_id=case_id,
                        label=f"{angle.elements} angle",
                    )
                )

        ref = cls(values=tuple(new_observations))

        if frequencies is not None:
            ref = ref.with_frequencies_from_array(
                frequencies,
                weight=w["frequency"],
                case_id=case_id,
                skip_imaginary=skip_imaginary,
            )

        if include_eigenmatrix:
            # ``eigenmatrix_hessian`` is a *true* override — it stands in
            # for ``mol.hessian`` for eigenmatrix construction even when
            # the base ``mol.hessian`` is ``None``.  If neither is
            # available, no eigenmatrix block is built.
            hess_for_eigenmatrix = eigenmatrix_hessian if eigenmatrix_hessian is not None else mol.hessian
            if hess_for_eigenmatrix is not None:
                eig_weights: dict[str, float] = {
                    k: w[k] for k in ("eig_i", "eig_d_low", "eig_d_high", "eig_o") if k in w
                }
                ref = ref.with_eigenmatrix_from_hessian(
                    hess_for_eigenmatrix,
                    symbols=list(mol.symbols),
                    diagonal_only=eigenmatrix_diagonal_only,
                    case_id=case_id,
                    weights=eig_weights or None,
                )

        return ref

    @classmethod
    def from_molecules(
        cls,
        molecules: Sequence[Molecule],
        case_ids: Sequence[str],
        *,
        weights: dict[str, float] | None = None,
        frequencies_list: list[np.ndarray | list[float]] | None = None,
        skip_imaginary: bool = False,
        include_geometry: bool = True,
        include_eigenmatrix: bool = True,
        eigenmatrix_diagonal_only: bool = False,
        eigenmatrix_hessians: list[np.ndarray] | None = None,
    ) -> ObservationSet:
        """Auto-populate reference data from multiple molecules.

        Each molecule is bound to the corresponding entry of *case_ids*
        (matching :attr:`q2mm.models.problem.TrainingCase.case_id`).
        Delegates to :meth:`from_molecule` per molecule.

        Args:
            molecules (Sequence[Molecule]): Training set molecules.
            case_ids (Sequence[str]): Stable case ID for each molecule, in
                the same order as *molecules*. Must have the same length
                as *molecules*.
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
            ObservationSet: Combined reference data for all molecules.

        Raises:
            ValueError: If ``case_ids``, ``frequencies_list``, or
                ``eigenmatrix_hessians`` length does not match
                ``molecules`` length.

        """
        if len(case_ids) != len(molecules):
            raise ValueError(f"case_ids length ({len(case_ids)}) must match molecules length ({len(molecules)}).")
        if frequencies_list is not None and len(frequencies_list) != len(molecules):
            raise ValueError(
                f"frequencies_list length ({len(frequencies_list)}) must match molecules length ({len(molecules)})."
            )
        if eigenmatrix_hessians is not None and len(eigenmatrix_hessians) != len(molecules):
            raise ValueError(
                f"eigenmatrix_hessians length ({len(eigenmatrix_hessians)}) must match "
                f"molecules length ({len(molecules)})."
            )

        combined: list[Observation] = []
        for idx, mol in enumerate(molecules):
            single = cls.from_molecule(
                mol,
                case_id=case_ids[idx],
                weights=weights,
                frequencies=frequencies_list[idx] if frequencies_list is not None else None,
                skip_imaginary=skip_imaginary,
                include_geometry=include_geometry,
                include_eigenmatrix=include_eigenmatrix,
                eigenmatrix_diagonal_only=eigenmatrix_diagonal_only,
                eigenmatrix_hessian=eigenmatrix_hessians[idx] if eigenmatrix_hessians is not None else None,
            )
            combined.extend(single.values)
        return cls(values=tuple(combined))
