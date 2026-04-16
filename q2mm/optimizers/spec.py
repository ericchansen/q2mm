"""Compiled objective specification for JAX-native loss functions.

An :class:`ObjectiveSpec` encodes all the reference data, evaluator
configuration, and regularization settings needed by :mod:`jaxloss`
into JAX-compatible arrays and static configuration.  This is the
shared contract between the Python-side :class:`ObjectiveFunction`
and the JIT-compiled JAX loss.

The spec is built via :meth:`ObjectiveFunction.to_jax_spec()` and
consumed by :class:`~q2mm.optimizers.jaxloss.JaxLoss`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class MoleculeSpec:
    """Reference data for a single molecule, grouped by evaluator category.

    All arrays use NumPy (converted to JAX arrays at JIT-compile time
    by :class:`~q2mm.optimizers.jaxloss.JaxLoss`).

    Attributes:
        mol_idx: Index of this molecule in the training set.
        symbols: Element symbols (length *N*).
        n_atoms: Number of atoms.
        energy_refs: ``(n_energy,)`` reference energy values.
        energy_weights: ``(n_energy,)`` weights for energy residuals.
        freq_indices: ``(n_freq,)`` 0-based mode indices into the
            ``3N``-length frequency array.
        freq_refs: ``(n_freq,)`` reference frequencies (cm⁻¹).
        freq_weights: ``(n_freq,)`` weights for frequency residuals.
        hess_indices: ``(n_hess,)`` packed Hessian indices
            (``row * 3N + col``) into the flat ``(3N * 3N,)`` Hessian
            for hessian-element references.
        hess_refs: ``(n_hess,)`` reference Hessian element values.
        hess_weights: ``(n_hess,)`` weights for Hessian residuals.
        eig_diag_indices: ``(n_ediag,)`` indices into the eigenvalue
            vector for diagonal eigenmatrix references.
        eig_diag_refs: ``(n_ediag,)`` reference eigenvalue values.
        eig_diag_weights: ``(n_ediag,)`` weights for diagonal residuals.
        eig_offdiag_indices: ``(n_eoff,)`` packed indices
            (``row * 3N + col``) for off-diagonal eigenmatrix references.
        eig_offdiag_refs: ``(n_eoff,)`` reference off-diagonal values.
        eig_offdiag_weights: ``(n_eoff,)`` weights for off-diagonal
            residuals.

    """

    mol_idx: int
    symbols: tuple[str, ...]
    n_atoms: int
    # Energy
    energy_refs: np.ndarray
    energy_weights: np.ndarray
    # Frequency
    freq_indices: np.ndarray
    freq_refs: np.ndarray
    freq_weights: np.ndarray
    # Hessian element
    hess_indices: np.ndarray
    hess_refs: np.ndarray
    hess_weights: np.ndarray
    # Eigenmatrix diagonal
    eig_diag_indices: np.ndarray
    eig_diag_refs: np.ndarray
    eig_diag_weights: np.ndarray
    # Eigenmatrix off-diagonal
    eig_offdiag_indices: np.ndarray
    eig_offdiag_refs: np.ndarray
    eig_offdiag_weights: np.ndarray

    @property
    def has_energy(self) -> bool:
        """Whether this molecule has energy references."""
        return len(self.energy_refs) > 0

    @property
    def has_frequency(self) -> bool:
        """Whether this molecule has frequency references."""
        return len(self.freq_refs) > 0

    @property
    def has_hessian(self) -> bool:
        """Whether this molecule has Hessian element references."""
        return len(self.hess_refs) > 0

    @property
    def has_eigenmatrix(self) -> bool:
        """Whether this molecule has eigenmatrix references."""
        return len(self.eig_diag_refs) > 0 or len(self.eig_offdiag_refs) > 0

    @property
    def needs_hessian_computation(self) -> bool:
        """Whether this molecule requires a Hessian from the engine."""
        return self.has_frequency or self.has_hessian or self.has_eigenmatrix


@dataclass(frozen=True)
class ObjectiveSpec:
    """JAX-compatible specification for the objective function.

    Encodes all information needed to compile a JIT loss function:
    reference data arrays, molecule metadata, regularization config,
    and parameter bounds.

    Built by :meth:`ObjectiveFunction.to_jax_spec` and consumed by
    :class:`~q2mm.optimizers.jaxloss.JaxLoss`.

    Attributes:
        molecules: Per-molecule reference data specifications.
        n_params: Length of the flat parameter vector.
        regularization: L2 penalty strength (λ).
        reference_params: ``(n_params,)`` parameter anchor for L2.
        lower_bounds: ``(n_params,)`` lower bounds (``-inf`` = unbounded).
        upper_bounds: ``(n_params,)`` upper bounds (``+inf`` = unbounded).
        supported_categories: Frozenset of evaluator categories present
            in the spec (e.g. ``{"energy", "frequency"}``).  Geometry
            is excluded — see module docstring.

    """

    molecules: tuple[MoleculeSpec, ...]
    n_params: int
    regularization: float = 0.0
    reference_params: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    lower_bounds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    upper_bounds: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    supported_categories: frozenset[str] = field(default_factory=frozenset)

    def has_geometry_refs(self) -> bool:
        """Return True if any molecule has geometry references.

        Geometry references (bond_length, bond_angle, torsion_angle)
        are NOT supported by the JIT loss — they require differentiable
        energy minimization (implicit differentiation).  This method
        is provided for diagnostic checks.
        """
        return False  # ObjectiveSpec intentionally excludes geometry

    @property
    def n_molecules(self) -> int:
        """Number of molecules in the spec."""
        return len(self.molecules)


def _build_molecule_spec(
    mol_idx: int,
    symbols: tuple[str, ...],
    refs: list,
) -> MoleculeSpec:
    """Build a MoleculeSpec from a list of ReferenceValue objects.

    Args:
        mol_idx: Molecule index in the training set.
        symbols: Element symbols for this molecule.
        refs: List of ReferenceValue objects for this molecule.

    Returns:
        MoleculeSpec with arrays populated from the references.

    """
    energy_vals, energy_wts = [], []
    freq_idx, freq_vals, freq_wts = [], [], []
    hess_idx, hess_vals, hess_wts = [], [], []
    ediag_idx, ediag_vals, ediag_wts = [], [], []
    eoff_idx, eoff_vals, eoff_wts = [], [], []

    for ref in refs:
        if ref.kind == "energy":
            energy_vals.append(ref.value)
            energy_wts.append(ref.weight)
        elif ref.kind == "frequency":
            freq_idx.append(ref.data_idx)
            freq_vals.append(ref.value)
            freq_wts.append(ref.weight)
        elif ref.kind == "hessian_element":
            # Pack (row, col) as row * 3N + col for JaxLoss indexing.
            # add_hessian_element stores indices in atom_indices=(row, col).
            if ref.atom_indices is not None and len(ref.atom_indices) >= 2:
                row, col = ref.atom_indices[:2]
                n3 = 3 * len(symbols)
                hess_idx.append(row * n3 + col)
            else:
                hess_idx.append(ref.data_idx)
            hess_vals.append(ref.value)
            hess_wts.append(ref.weight)
        elif ref.kind == "eig_diagonal":
            ediag_idx.append(ref.data_idx)
            ediag_vals.append(ref.value)
            ediag_wts.append(ref.weight)
        elif ref.kind == "eig_offdiagonal":
            # Pack (row, col) as row * 3N + col for JaxLoss indexing.
            # add_hessian_offdiagonal stores indices in atom_indices=(row, col).
            if ref.atom_indices is not None and len(ref.atom_indices) >= 2:
                row, col = ref.atom_indices[:2]
                n3 = 3 * len(symbols)
                eoff_idx.append(row * n3 + col)
            else:
                eoff_idx.append(ref.data_idx)
            eoff_vals.append(ref.value)
            eoff_wts.append(ref.weight)
        elif ref.kind in ("bond_length", "bond_angle", "torsion_angle"):
            pass  # Geometry excluded from JIT loss
        else:
            raise ValueError(f"Unknown reference kind: {ref.kind!r}")

    return MoleculeSpec(
        mol_idx=mol_idx,
        symbols=symbols,
        n_atoms=len(symbols),
        energy_refs=np.array(energy_vals, dtype=float),
        energy_weights=np.array(energy_wts, dtype=float),
        freq_indices=np.array(freq_idx, dtype=int),
        freq_refs=np.array(freq_vals, dtype=float),
        freq_weights=np.array(freq_wts, dtype=float),
        hess_indices=np.array(hess_idx, dtype=int),
        hess_refs=np.array(hess_vals, dtype=float),
        hess_weights=np.array(hess_wts, dtype=float),
        eig_diag_indices=np.array(ediag_idx, dtype=int),
        eig_diag_refs=np.array(ediag_vals, dtype=float),
        eig_diag_weights=np.array(ediag_wts, dtype=float),
        eig_offdiag_indices=np.array(eoff_idx, dtype=int),
        eig_offdiag_refs=np.array(eoff_vals, dtype=float),
        eig_offdiag_weights=np.array(eoff_wts, dtype=float),
    )
