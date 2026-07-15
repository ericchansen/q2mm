"""Canonical, immutable molecular structure representation for Q2MM.

:class:`Molecule` is Q2MM's *only* public molecular structure type. It is an
immutable value object: symbols, atom types, and partial charges are stored
as tuples; geometry and Hessian arrays are defensive, read-only NumPy copies;
bonds/angles/torsions are tuples of frozen :class:`Bond` / :class:`Angle` /
:class:`Torsion` records resolved once at construction time. There is no
mutation API — callers that need a modified molecule use one of the
``with_*`` methods, each of which returns a new :class:`Molecule`.

File-format parsers (``q2mm.io.gaussian``, ``q2mm.io.jaguar``,
``q2mm.io.macromodel``, ``q2mm.io.mol2``) construct :class:`Molecule`
directly; :mod:`q2mm.io.xyz` loads XYZ files, and :mod:`q2mm.io.qcelemental`
converts to/from QCElemental ``Molecule`` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.models.identifiers import (
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
    canonicalize_torsion_env_id,
)

if TYPE_CHECKING:
    # Deferred to a function-level import at runtime (see with_hessian):
    # q2mm.models.hessian imports q2mm.constants, which (transitively, via
    # q2mm.optimizers) imports q2mm.models.molecule back — importing it
    # eagerly here would create an import cycle during package init.
    from q2mm.models.hessian import HessianProvenance

# Covalent radii — imported from the single-source-of-truth element table.
from q2mm.elements import COVALENT_RADII


def _dihedral_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Compute signed dihedral angle (degrees) for four points using atan2.

    Delegates to :func:`q2mm.geometry.dihedral_angle`.
    Returns a value in [-180, 180].
    """
    from q2mm.geometry import dihedral_angle

    return dihedral_angle(p0, p1, p2, p3)


def _immutable_array(values: Any, *, dtype: type = float) -> np.ndarray:
    """Return a defensive, read-only copy of *values* as a NumPy array."""
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Bond:
    """An immutable bond between two atoms in a :class:`Molecule`.

    Populated either by geometry-based detection (covalent-radii distance
    check) or, when a parser supplies explicit connectivity, directly from
    that source data — see :class:`Molecule` for the inference rules.
    """

    atom_i: int  # 0-based index
    atom_j: int  # 0-based index
    elements: tuple[str, str]
    length: float  # Angstrom
    env_id: str = ""
    ff_row: int | None = None
    bond_order: str = ""  # "-" single, "=" double, "*" aromatic, "%" triple
    source_bond_order: str | None = None  # Original parser token, if supplied.

    @property
    def element_pair(self) -> tuple[str, str]:
        """Sorted element pair for matching (e.g., ('C', 'F'))."""
        return tuple(sorted(self.elements))


@dataclass(frozen=True)
class Angle:
    """An immutable angle between three atoms in a :class:`Molecule`."""

    atom_i: int  # 0-based (outer)
    atom_j: int  # 0-based (center)
    atom_k: int  # 0-based (outer)
    elements: tuple[str, str, str]
    value: float  # degrees
    env_id: str = ""
    ff_row: int | None = None

    @property
    def element_triple(self) -> tuple[str, str, str]:
        """Canonical element triple: (outer, center, outer) sorted by outer elements."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass(frozen=True)
class Torsion:
    """An immutable torsion/dihedral between four atoms in a :class:`Molecule`."""

    atom_i: int  # 0-based (end)
    atom_j: int  # 0-based (central)
    atom_k: int  # 0-based (central)
    atom_l: int  # 0-based (end)
    elements: tuple[str, str, str, str]
    value: float  # dihedral angle in degrees, [-180, 180]
    env_id: str = ""
    ff_row: int | None = None

    @property
    def element_quad(self) -> tuple[str, str, str, str]:
        """Canonical element quad: forward or reversed, whichever is lexically smaller."""
        fwd = self.elements
        rev = (fwd[3], fwd[2], fwd[1], fwd[0])
        return min(fwd, rev)


@dataclass(frozen=True, eq=False)
class Molecule:
    """Q2MM's single canonical, immutable molecular structure representation.

    Wraps atomic symbols, coordinates, charge, and multiplicity together
    with bonds/angles/torsions and an optional canonical (Hartree/Bohr²)
    Hessian.

    Topology resolution (PR #309 semantics, preserved exactly):

    - ``bonds=None`` (the default) infers bonds from geometry using
      covalent-radii distances scaled by *bond_tolerance*.
    - ``angles=None``/``torsions=None`` infer from the *resolved* bonds
      (whether those bonds were themselves inferred or explicitly supplied).
    - Passing an explicit sequence — including an empty one — for *bonds*,
      *angles*, or *torsions* is authoritative for that degree-of-freedom
      category independently of the others; it is never re-inferred.

    Topology is resolved exactly once, at construction time. There is no
    mutable cache-invalidation API: callers that need a molecule with
    different geometry, atom types, or scalar fields use one of the
    ``with_*`` methods, which construct a new, independently-resolved
    :class:`Molecule`.

    Equality is identity-based (``eq=False``): geometry/Hessian arrays are
    NumPy arrays, which do not support the elementwise ``==`` that a
    generated dataclass ``__eq__`` would attempt.
    """

    symbols: tuple[str, ...]
    geometry: np.ndarray  # Shape (N, 3), Angstrom
    atom_types: tuple[str, ...] | None = None
    charge: int = 0
    multiplicity: int = 1
    name: str = ""
    bond_tolerance: float = 1.3  # See constants.DEFAULT_BOND_TOLERANCE. 1.4+ for TS.
    hessian: np.ndarray | None = None  # Shape (3N, 3N), Hartree/Bohr^2
    hessian_provenance: HessianProvenance | None = None
    partial_charges: tuple[float | None, ...] | None = None
    bonds: tuple[Bond, ...] | None = None
    angles: tuple[Angle, ...] | None = None
    torsions: tuple[Torsion, ...] | None = None
    bonds_explicit: bool = field(default=False, init=False)
    angles_explicit: bool = field(default=False, init=False)
    torsions_explicit: bool = field(default=False, init=False)
    improper_torsions: tuple[Torsion, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        """Normalize inputs to immutable form and resolve topology once."""
        symbols = tuple(str(symbol) for symbol in self.symbols)
        object.__setattr__(self, "symbols", symbols)

        if self.atom_types is None:
            atom_types = symbols
        else:
            atom_types = tuple(str(atom_type) for atom_type in self.atom_types)
        if len(atom_types) != len(symbols):
            raise ValueError("atom_types must have the same length as symbols.")
        object.__setattr__(self, "atom_types", atom_types)

        if self.partial_charges is not None:
            if len(self.partial_charges) != len(symbols):
                raise ValueError("partial_charges must have the same length as symbols.")
            object.__setattr__(
                self,
                "partial_charges",
                tuple(None if charge is None else float(charge) for charge in self.partial_charges),
            )

        geometry = _immutable_array(self.geometry)
        if geometry.shape != (len(symbols), 3):
            raise ValueError(f"geometry must have shape ({len(symbols)}, 3), got {geometry.shape}.")
        object.__setattr__(self, "geometry", geometry)

        if self.hessian is not None:
            hessian = _immutable_array(self.hessian)
            expected_hessian_shape = (3 * len(symbols), 3 * len(symbols))
            if hessian.shape != expected_hessian_shape:
                raise ValueError(f"hessian must have shape {expected_hessian_shape}, got {hessian.shape}.")
            object.__setattr__(self, "hessian", hessian)
            if self.hessian_provenance is None:
                from q2mm.models.hessian import HessianProvenance, HessianUnits

                object.__setattr__(
                    self,
                    "hessian_provenance",
                    HessianProvenance(units=HessianUnits.ATOMIC, source="programmatic"),
                )
        elif self.hessian_provenance is not None:
            raise ValueError("hessian_provenance requires a Hessian.")

        bonds_explicit = self.bonds is not None
        bonds = tuple(self.bonds) if bonds_explicit else self._detect_bonds()
        object.__setattr__(self, "bonds", bonds)
        object.__setattr__(self, "bonds_explicit", bonds_explicit)

        angles_explicit = self.angles is not None
        angles = tuple(self.angles) if angles_explicit else self._detect_angles(bonds)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "angles_explicit", angles_explicit)

        torsions_explicit = self.torsions is not None
        torsions = tuple(self.torsions) if torsions_explicit else self._detect_torsions(bonds)
        object.__setattr__(self, "torsions", torsions)
        object.__setattr__(self, "torsions_explicit", torsions_explicit)

        object.__setattr__(self, "improper_torsions", self._detect_improper_torsions(bonds))

    @property
    def n_atoms(self) -> int:
        """Number of atoms in the molecule."""
        return len(self.symbols)

    # ---- Topology detection (geometry-derived) ----

    def _detect_bonds(self) -> tuple[Bond, ...]:
        """Detect bonds based on covalent radii with tolerance factor."""
        tolerance = self.bond_tolerance
        bonds = []
        for i in range(self.n_atoms):
            for j in range(i + 1, self.n_atoms):
                ri = COVALENT_RADII.get(self.symbols[i], 0.76)
                rj = COVALENT_RADII.get(self.symbols[j], 0.76)
                dist = float(np.linalg.norm(self.geometry[i] - self.geometry[j]))
                if dist < tolerance * (ri + rj):
                    bonds.append(
                        Bond(
                            atom_i=i,
                            atom_j=j,
                            elements=(self.symbols[i], self.symbols[j]),
                            length=dist,
                            env_id=canonicalize_bond_env_id([self.atom_types[i], self.atom_types[j]]),
                        )
                    )
        return tuple(bonds)

    def _detect_angles(self, bonds: tuple[Bond, ...]) -> tuple[Angle, ...]:
        """Detect angles from resolved bonds."""
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        angles = []
        for center in range(self.n_atoms):
            neighbors = adj[center]
            for ii in range(len(neighbors)):
                for jj in range(ii + 1, len(neighbors)):
                    a, b = neighbors[ii], neighbors[jj]
                    v1 = self.geometry[a] - self.geometry[center]
                    v2 = self.geometry[b] - self.geometry[center]
                    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle_val = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
                    angles.append(
                        Angle(
                            atom_i=a,
                            atom_j=center,
                            atom_k=b,
                            elements=(self.symbols[a], self.symbols[center], self.symbols[b]),
                            value=float(angle_val),
                            env_id=canonicalize_angle_env_id(
                                [self.atom_types[a], self.atom_types[center], self.atom_types[b]]
                            ),
                        )
                    )
        return tuple(angles)

    def _detect_torsions(self, bonds: tuple[Bond, ...]) -> tuple[Torsion, ...]:
        """Detect torsion/dihedral angles from resolved bonds.

        For each bond B-C, finds all atoms A bonded to B (A != C) and all
        atoms D bonded to C (D != B) to form torsions A-B-C-D.  Deduplicates
        so that A-B-C-D and D-C-B-A are not both stored.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        seen: set[tuple[int, int, int, int]] = set()
        torsions: list[Torsion] = []
        for bond in bonds:
            b, c = bond.atom_i, bond.atom_j
            for a in adj[b]:
                if a == c:
                    continue
                for d in adj[c]:
                    if d in (b, a):
                        continue
                    key = (a, b, c, d)
                    key_rev = (d, c, b, a)
                    if key in seen or key_rev in seen:
                        continue
                    seen.add(key)
                    value = _dihedral_angle(self.geometry[a], self.geometry[b], self.geometry[c], self.geometry[d])
                    torsions.append(
                        Torsion(
                            atom_i=a,
                            atom_j=b,
                            atom_k=c,
                            atom_l=d,
                            elements=(
                                self.symbols[a],
                                self.symbols[b],
                                self.symbols[c],
                                self.symbols[d],
                            ),
                            value=value,
                            env_id=canonicalize_torsion_env_id(
                                [self.atom_types[a], self.atom_types[b], self.atom_types[c], self.atom_types[d]]
                            ),
                        )
                    )
        return tuple(torsions)

    def _detect_improper_torsions(self, bonds: tuple[Bond, ...]) -> tuple[Torsion, ...]:
        """Detect improper torsions at trigonal (sp2) centres.

        For each atom with exactly 3 bonded neighbours, generates an
        improper torsion quad.  The centre atom goes in position j
        (second slot) following the MM3 out-of-plane convention:
        ``(neighbour_0, centre, neighbour_1, neighbour_2)``.

        Neighbours are sorted by index for deterministic ordering.
        """
        adj: dict[int, list[int]] = {i: [] for i in range(self.n_atoms)}
        for bond in bonds:
            adj[bond.atom_i].append(bond.atom_j)
            adj[bond.atom_j].append(bond.atom_i)

        impropers: list[Torsion] = []
        for centre in range(self.n_atoms):
            nbrs = sorted(adj[centre])
            if len(nbrs) != 3:
                continue
            a, b, d = nbrs
            value = _dihedral_angle(
                self.geometry[a],
                self.geometry[centre],
                self.geometry[b],
                self.geometry[d],
            )
            impropers.append(
                Torsion(
                    atom_i=a,
                    atom_j=centre,
                    atom_k=b,
                    atom_l=d,
                    elements=(
                        self.symbols[a],
                        self.symbols[centre],
                        self.symbols[b],
                        self.symbols[d],
                    ),
                    value=value,
                    env_id=canonicalize_torsion_env_id(
                        [self.atom_types[a], self.atom_types[centre], self.atom_types[b], self.atom_types[d]]
                    ),
                )
            )
        return tuple(impropers)

    # ---- Pure "with_*" replacements ----
    #
    # Molecule is immutable; these return a new Molecule rather than
    # mutating self. Each preserves whichever bonds/angles/torsions
    # category was explicit (re-supplying that exact tuple, still
    # authoritative) and re-infers whichever category was not (from the
    # possibly-updated geometry/atom_types/bond_tolerance), matching PR #309
    # semantics for every derived Molecule, not just the first one built.

    def _replace(self, **overrides: Any) -> Molecule:
        kwargs: dict[str, Any] = {
            "symbols": self.symbols,
            "geometry": self.geometry,
            "atom_types": self.atom_types,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "name": self.name,
            "bond_tolerance": self.bond_tolerance,
            "hessian": self.hessian,
            "hessian_provenance": self.hessian_provenance,
            "partial_charges": self.partial_charges,
            "bonds": self.bonds if self.bonds_explicit else None,
            "angles": self.angles if self.angles_explicit else None,
            "torsions": self.torsions if self.torsions_explicit else None,
        }
        kwargs.update(overrides)
        return Molecule(**kwargs)

    def _replace_preserving_topology(self, **overrides: Any) -> Molecule:
        """Replace fields while preserving connectivity and refreshing geometry metadata."""
        geometry = np.asarray(overrides.get("geometry", self.geometry), dtype=float)
        atom_types = tuple(overrides.get("atom_types", self.atom_types))
        if geometry.shape != (self.n_atoms, 3):
            raise ValueError(f"geometry must have shape ({self.n_atoms}, 3), got {geometry.shape}.")
        if len(atom_types) != self.n_atoms:
            raise ValueError("atom_types must have the same length as symbols.")

        bonds = tuple(
            replace(
                bond,
                length=float(np.linalg.norm(geometry[bond.atom_i] - geometry[bond.atom_j])),
                env_id=canonicalize_bond_env_id([atom_types[bond.atom_i], atom_types[bond.atom_j]]),
            )
            for bond in self.bonds
        )

        angles = []
        for angle in self.angles:
            v1 = geometry[angle.atom_i] - geometry[angle.atom_j]
            v2 = geometry[angle.atom_k] - geometry[angle.atom_j]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angles.append(
                replace(
                    angle,
                    value=float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))),
                    env_id=canonicalize_angle_env_id(
                        [atom_types[angle.atom_i], atom_types[angle.atom_j], atom_types[angle.atom_k]]
                    ),
                )
            )

        torsions = tuple(
            replace(
                torsion,
                value=_dihedral_angle(
                    geometry[torsion.atom_i],
                    geometry[torsion.atom_j],
                    geometry[torsion.atom_k],
                    geometry[torsion.atom_l],
                ),
                env_id=canonicalize_torsion_env_id(
                    [
                        atom_types[torsion.atom_i],
                        atom_types[torsion.atom_j],
                        atom_types[torsion.atom_k],
                        atom_types[torsion.atom_l],
                    ]
                ),
            )
            for torsion in self.torsions
        )

        kwargs: dict[str, Any] = {
            "symbols": self.symbols,
            "geometry": geometry,
            "atom_types": atom_types,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "name": self.name,
            "bond_tolerance": self.bond_tolerance,
            "hessian": self.hessian,
            "hessian_provenance": self.hessian_provenance,
            "partial_charges": self.partial_charges,
            "bonds": bonds,
            "angles": tuple(angles),
            "torsions": torsions,
        }
        kwargs.update(overrides)
        molecule = Molecule(**kwargs)
        object.__setattr__(molecule, "bonds_explicit", self.bonds_explicit)
        object.__setattr__(molecule, "angles_explicit", self.angles_explicit)
        object.__setattr__(molecule, "torsions_explicit", self.torsions_explicit)
        return molecule

    def __deepcopy__(self, memo: dict[int, object]) -> Molecule:
        """Return this immutable value unchanged while preserving read-only arrays."""
        memo[id(self)] = self
        return self

    def with_hessian(self, hessian: np.ndarray | None, provenance: HessianProvenance | None = None) -> Molecule:
        """Return a copy with a (canonical-AU) Hessian attached.

        Args:
            hessian: Cartesian Hessian matrix of shape ``(3N, 3N)``. Accepts
                a bare array (assumed already Hartree/Bohr²) or a
                ``pint.Quantity`` (converted via its own units). Pass
                ``None`` to remove an existing Hessian.
            provenance: Optional :class:`~q2mm.models.hessian.HessianProvenance`
                recording where *hessian* came from. Defaults to a
                programmatic-origin provenance when *hessian* is not
                ``None``.

        Returns:
            A new :class:`Molecule` identical to this one but with the
            given *hessian* (and provenance) attached.

        """
        from q2mm.models.hessian import HessianProvenance, HessianUnits, hessian_to_atomic_units

        resolved = hessian_to_atomic_units(hessian, HessianUnits.ATOMIC)
        resolved_provenance = provenance
        if resolved is not None and resolved_provenance is None:
            resolved_provenance = HessianProvenance(units=HessianUnits.ATOMIC, source="programmatic")
        if resolved is None:
            resolved_provenance = None
        return self._replace(hessian=resolved, hessian_provenance=resolved_provenance)

    def with_geometry(self, geometry: np.ndarray) -> Molecule:
        """Return a copy with new *geometry*.

        Connectivity and source metadata are preserved, while bond lengths,
        angle values, and torsion values are recomputed from the new geometry.
        The old Hessian is removed because it belongs to the old coordinates.
        """
        return self._replace_preserving_topology(
            geometry=geometry,
            hessian=None,
            hessian_provenance=None,
        )

    def with_atom_types(self, atom_types: list[str]) -> Molecule:
        """Return a copy with new *atom_types*.

        Connectivity and source metadata are preserved, while every topology
        record's ``env_id`` is recomputed from the new atom types.
        """
        return self._replace_preserving_topology(atom_types=tuple(str(atom_type) for atom_type in atom_types))

    def with_overrides(
        self,
        *,
        charge: int | None = None,
        multiplicity: int | None = None,
        bond_tolerance: float | None = None,
        name: str | None = None,
    ) -> Molecule:
        """Return a copy with the given scalar fields replaced.

        Only the fields passed (non-``None``) are changed. Inferred
        bonds/angles/torsions are recomputed when *bond_tolerance* changes;
        any explicitly-supplied categories are preserved unchanged.
        """
        overrides: dict[str, Any] = {}
        if charge is not None:
            overrides["charge"] = charge
        if multiplicity is not None:
            overrides["multiplicity"] = multiplicity
        if bond_tolerance is not None:
            overrides["bond_tolerance"] = bond_tolerance
        if name is not None:
            overrides["name"] = name
        return self._replace(**overrides)

    def __repr__(self) -> str:
        formula = "".join(f"{s}{self.symbols.count(s)}" for s in dict.fromkeys(self.symbols))
        hess_str = f", hessian={self.hessian.shape}" if self.hessian is not None else ""
        return f"Molecule({formula}, {self.n_atoms} atoms, {len(self.bonds)} bonds{hess_str})"
