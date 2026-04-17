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
        bond_atoms: ``(n_bond, 2)`` integer atom-index pairs for
            bond-length geometry references.
        bond_refs: ``(n_bond,)`` reference bond lengths (Å).
        bond_weights: ``(n_bond,)`` weights for bond-length residuals.
        angle_atoms: ``(n_angle, 3)`` integer atom-index triples
            ``(outer, vertex, outer)`` for bond-angle references.
        angle_refs: ``(n_angle,)`` reference bond angles (degrees).
        angle_weights: ``(n_angle,)`` weights for bond-angle residuals.
        torsion_atoms: ``(n_tors, 4)`` integer atom-index quadruples
            for dihedral references.
        torsion_refs: ``(n_tors,)`` reference dihedrals in ``[-180, 180]``
            degrees.
        torsion_weights: ``(n_tors,)`` weights for torsion residuals.

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
    # Geometry — bond length (atom pairs, Å)
    bond_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    bond_refs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    bond_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Geometry — bond angle (atom triples, degrees; middle atom is vertex)
    angle_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=int))
    angle_refs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    angle_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Geometry — torsion angle (atom quadruples, degrees, range [-180, 180])
    torsion_atoms: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), dtype=int))
    torsion_refs: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    torsion_weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    # Transition-state curvature inversion (QFUERZA). When True, the JIT
    # loss applies :func:`~q2mm.models.hessian.invert_ts_curvature_jax` to
    # the MM Hessian before computing frequency / Hessian-element /
    # eigenmatrix residuals for this molecule.
    invert_ts_curvature: bool = False

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
    def has_bond_length(self) -> bool:
        """Whether this molecule has bond-length geometry references."""
        return len(self.bond_refs) > 0

    @property
    def has_bond_angle(self) -> bool:
        """Whether this molecule has bond-angle geometry references."""
        return len(self.angle_refs) > 0

    @property
    def has_torsion(self) -> bool:
        """Whether this molecule has torsion-angle geometry references."""
        return len(self.torsion_refs) > 0

    @property
    def has_geometry(self) -> bool:
        """Whether this molecule has any geometry references.

        Geometry references require relaxing the molecular geometry at
        each set of parameters; :class:`~q2mm.optimizers.jaxloss.JaxLoss`
        handles them via implicit differentiation through
        ``jaxopt.LBFGS``.
        """
        return self.has_bond_length or self.has_bond_angle or self.has_torsion

    @property
    def needs_hessian_computation(self) -> bool:
        """Whether this molecule requires a Hessian from the engine."""
        return self.has_frequency or self.has_hessian or self.has_eigenmatrix

    @property
    def needs_geometry_relaxation(self) -> bool:
        """Whether this molecule requires a relaxed geometry from the engine.

        True when any bond/angle/torsion geometry reference is present.
        :class:`~q2mm.optimizers.jaxloss.JaxLoss` will run an inner
        ``jaxopt.LBFGS`` geometry minimization per loss call and
        differentiate through it via the implicit function theorem.
        """
        return self.has_geometry


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
            in the spec (e.g. ``{"energy", "frequency", "geometry"}``).
            Geometry references are handled via implicit differentiation
            through an inner ``jaxopt.LBFGS`` geometry minimization; see
            :class:`~q2mm.optimizers.jaxloss.JaxLoss`.

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
        are supported by the JIT loss via implicit differentiation
        through ``jaxopt.LBFGS``. See
        :class:`~q2mm.optimizers.jaxloss.JaxLoss` for details.
        """
        return any(m.has_geometry for m in self.molecules)

    @property
    def n_molecules(self) -> int:
        """Number of molecules in the spec."""
        return len(self.molecules)


def _build_molecule_spec(
    mol_idx: int,
    symbols: tuple[str, ...],
    refs: list,
    *,
    topology: object | None = None,
    invert_ts_curvature: bool = False,
) -> MoleculeSpec:
    """Build a MoleculeSpec from a list of ReferenceValue objects.

    Args:
        mol_idx: Molecule index in the training set.
        symbols: Element symbols for this molecule.
        refs: List of ReferenceValue objects for this molecule.
        topology: Optional molecule object providing ``bonds``,
            ``angles``, and ``torsions`` lists. Used to resolve
            positional ``data_idx`` for geometry references (e.g.
            ``ref.kind == "bond_length"`` with no ``atom_indices``) to
            explicit atom-index tuples required by the JIT loss.
        invert_ts_curvature: If True, mark this molecule for
            transition-state curvature inversion (QFUERZA) inside the
            JIT loss.  See :func:`~q2mm.models.hessian.invert_ts_curvature_jax`.

    Returns:
        MoleculeSpec with arrays populated from the references.

    Raises:
        ValueError: If a geometry reference uses ``data_idx`` but no
            ``topology`` is provided, or if ``data_idx`` is out of range.

    """
    energy_vals, energy_wts = [], []
    freq_idx, freq_vals, freq_wts = [], [], []
    hess_idx, hess_vals, hess_wts = [], [], []
    ediag_idx, ediag_vals, ediag_wts = [], [], []
    eoff_idx, eoff_vals, eoff_wts = [], [], []
    bond_at, bond_v, bond_w = [], [], []
    ang_at, ang_v, ang_w = [], [], []
    tor_at, tor_v, tor_w = [], [], []

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
        elif ref.kind == "bond_length":
            atoms = _resolve_geom_atoms(ref, "bonds", 2, topology)
            bond_at.append(atoms)
            bond_v.append(ref.value)
            bond_w.append(ref.weight)
        elif ref.kind == "bond_angle":
            atoms = _resolve_geom_atoms(ref, "angles", 3, topology)
            ang_at.append(atoms)
            ang_v.append(ref.value)
            ang_w.append(ref.weight)
        elif ref.kind == "torsion_angle":
            atoms = _resolve_geom_atoms(ref, "torsions", 4, topology)
            tor_at.append(atoms)
            tor_v.append(ref.value)
            tor_w.append(ref.weight)
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
        bond_atoms=np.array(bond_at, dtype=int).reshape(-1, 2),
        bond_refs=np.array(bond_v, dtype=float),
        bond_weights=np.array(bond_w, dtype=float),
        angle_atoms=np.array(ang_at, dtype=int).reshape(-1, 3),
        angle_refs=np.array(ang_v, dtype=float),
        angle_weights=np.array(ang_w, dtype=float),
        torsion_atoms=np.array(tor_at, dtype=int).reshape(-1, 4),
        torsion_refs=np.array(tor_v, dtype=float),
        torsion_weights=np.array(tor_w, dtype=float),
        invert_ts_curvature=bool(invert_ts_curvature),
    )


def _resolve_geom_atoms(
    ref: object,
    attr: str,
    arity: int,
    topology: object | None,
) -> tuple[int, ...]:
    """Resolve a geometry reference to an explicit atom-index tuple.

    Prefers ``ref.atom_indices`` when present. Falls back to looking up
    ``topology.<attr>[ref.data_idx]`` (e.g. ``topology.bonds[0]``) and
    pulling the atom indices off that record (``atom_i``, ``atom_j``,
    ...).

    Args:
        ref: The :class:`~q2mm.optimizers.objective.ReferenceValue`.
        attr: The topology attribute to fall back to
            (``"bonds"`` / ``"angles"`` / ``"torsions"``).
        arity: Number of atom indices expected (2, 3, or 4).
        topology: Molecule object; required when falling back to
            ``data_idx``.

    Returns:
        Tuple of atom indices with length ``arity``.

    Raises:
        ValueError: If atoms cannot be resolved.

    """
    if ref.atom_indices is not None and len(ref.atom_indices) >= arity:
        return tuple(int(i) for i in ref.atom_indices[:arity])
    if topology is None:
        raise ValueError(
            f"{ref.kind} reference {ref.label!r} has no atom_indices and no "
            "molecule topology was provided to resolve data_idx."
        )
    records = getattr(topology, attr, None)
    if records is None:
        raise ValueError(
            f"{ref.kind} reference {ref.label!r} requires molecule.{attr}, but molecule has no such attribute."
        )
    if ref.data_idx < 0 or ref.data_idx >= len(records):
        raise ValueError(
            f"{ref.kind} reference {ref.label!r} has data_idx={ref.data_idx} "
            f"out of range (molecule.{attr} has {len(records)} entries)."
        )
    record = records[ref.data_idx]
    index_attrs = ("atom_i", "atom_j", "atom_k", "atom_l")[:arity]
    return tuple(int(getattr(record, a)) for a in index_attrs)
