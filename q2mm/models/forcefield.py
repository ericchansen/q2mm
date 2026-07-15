"""Clean, format-agnostic force field representation for Q2MM.

Decouples Q2MM's optimization from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod). Parameters are identified by element
pairs/triples, not format-specific atom type strings or line numbers.

Every parameter record and :class:`ForceField` itself is a value-like
immutable object: frozen dataclasses with tuple collections. There is no
optimizer state (no ``frozen`` flag, no active/frozen partition) on these
records — that lives entirely in
:class:`~q2mm.models.parameters.ActiveParameterSpace`. Pure replacement
via :meth:`q2mm.models.parameters.ParameterLayout.replace` is the only
way to change parameter values; construct a new force field explicitly
for anything else (file I/O lives in ``q2mm.io``, not here).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_torsion_env_id,
)

if TYPE_CHECKING:
    from q2mm.models.molecule import Molecule


class FunctionalForm(str, Enum):
    """Physical functional form used by a force field.

    Determines which energy expressions an engine should build:

    - ``HARMONIC``: Standard harmonic bonds/angles, periodic torsions,
      Lennard-Jones 12-6 vdW.  Used by AMBER, OPLS, GAFF, etc.
    - ``MM3``: Allinger's MM3 cubic bond stretch, sextic angle bend,
      Buckingham exp-6 vdW.

    The enum is orthogonal to ``source_format`` (file format) — an MM3
    force field can be loaded from ``.fld`` or ``.prm`` files, while a
    HARMONIC force field comes from ``.frcmod`` or programmatic
    construction.
    """

    HARMONIC = "harmonic"
    MM3 = "mm3"


@dataclass(frozen=True)
class BondParam:
    """A bond force field parameter.

    Units (canonical): ``force_constant`` in kcal/(mol·Å²),
    ``equilibrium`` in Å.  Energy convention: ``E = k·(r − r₀)²``.
    """

    elements: tuple[str, str]  # Sorted element pair, e.g., ('C', 'F')
    equilibrium: float  # Å
    force_constant: float  # kcal/(mol·Å²)
    label: str = ""  # Human-readable label
    env_id: str = ""  # Environment ID for disambiguating same-element params
    # (e.g., MM3 ff_row, atom type codes 'C1-F1' vs 'C2-F1')
    ff_row: int | None = None  # Source force-field row for exact legacy parity
    bond_order: str = ""  # Bond order from .fld: "-" single, "=" double,
    # "*" aromatic, "%" triple.  Empty = unknown.
    context: str = ""  # MM3 context flags (e.g., "O200 0000"). Empty or
    # "0000 0000" = generic (any context).
    dipole_moment: float = 0.0  # Bond dipole in Debye (MM3 .fld P3 column)

    @property
    def key(self) -> tuple[str, str]:
        """Sorted element pair for canonical matching (e.g., ``('C', 'F')``)."""
        return tuple(sorted(self.elements))


@dataclass(frozen=True)
class AngleParam:
    """An angle force field parameter.

    Units (canonical): ``force_constant`` in kcal/(mol·rad²),
    ``equilibrium`` in degrees.  Energy convention: ``E = k·(θ − θ₀)²``.

    Optional Urey-Bradley 1-3 distance term (CHARMM):
    ``E_UB = ub_force_constant · (r_13 − ub_equilibrium)²``
    where ``r_13`` is the distance between the first and third atoms.
    """

    elements: tuple[str, str, str]  # (outer, center, outer)
    equilibrium: float  # degrees
    force_constant: float  # kcal/(mol·rad²)
    label: str = ""
    env_id: str = ""  # Environment ID for disambiguating same-element params
    ff_row: int | None = None  # Source force-field row for exact legacy parity
    ub_force_constant: float | None = None  # kcal/(mol·Å²), None = no UB term
    ub_equilibrium: float | None = None  # Å, None = no UB term

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical key: center fixed, outers sorted."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass(frozen=True)
class StretchBendParam:
    """A stretch-bend cross-term parameter (MM3).

    Couples bond stretching with angle bending:
    ``E_sb = k_sb · (r_ij − r₀) · (θ − θ₀) + k_sb · (r_jk − r₀') · (θ − θ₀)``

    Each stretch-bend parameter is associated with an angle triple
    (i, j, k).  The force constant ``k_sb`` has units of
    mdyn/rad (canonical: kcal/(mol·Å·rad)).
    """

    elements: tuple[str, str, str]  # Same triple as the parent angle
    force_constant: float = 0.0  # kcal/(mol·Å·rad)
    label: str = ""
    env_id: str = ""
    ff_row: int | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical key: center fixed, outers sorted."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass(frozen=True)
class TorsionParam:
    """A torsion/dihedral force field parameter.

    Each object represents a single Fourier component (V_n).  An MM3
    torsion line with V1, V2, V3 produces three ``TorsionParam``
    objects with ``periodicity`` 1, 2, 3 respectively.

    Improper torsions (out-of-plane bending) are distinguished by
    ``is_improper=True``.  They originate from the AMBER IMPROPER
    section or equivalent force field blocks rather than molecular
    geometry detection.
    """

    elements: tuple[str, str, str, str]
    periodicity: int = 1
    force_constant: float = 0.0  # kcal/mol
    phase: float = 0.0  # degrees
    label: str = ""
    env_id: str = ""  # Environment ID for disambiguating same-element params
    ff_row: int | None = None  # Source force-field row for legacy parity
    is_improper: bool = False


@dataclass(frozen=True)
class VdwParam:
    """An atom-type van der Waals parameter."""

    atom_type: str
    radius: float  # Angstrom
    epsilon: float  # kcal/mol
    element: str = ""
    reduction: float = 0.0
    label: str = ""
    ff_row: int | None = None

    def __post_init__(self) -> None:
        """Normalize atom_type and auto-extract element if not provided."""
        object.__setattr__(self, "atom_type", str(self.atom_type).strip())
        if not self.element:
            object.__setattr__(self, "element", _extract_element(self.atom_type))


@dataclass(frozen=True)
class CmapGrid:
    """A CMAP (correction map) energy grid for backbone φ/ψ dihedrals.

    CMAP is a 2D spline correction used in CHARMM force fields to improve
    backbone conformational energetics.  The grid stores tabulated energy
    corrections as a function of two dihedral angles (typically φ and ψ).

    CMAP grids are **read-only** — they are not included in the optimizable
    parameter vector.  During force field fitting, the CMAP correction is
    applied as a fixed energy contribution.

    Attributes:
        atom_types_phi: Atom types defining the φ dihedral (4 types).
        atom_types_psi: Atom types defining the ψ dihedral (4 types).
        resolution: Number of grid points along each axis (e.g., 24 for
            15° spacing over 360°).
        energy: Flat tuple of energy corrections in kcal/mol, length
            ``resolution * resolution``.  Entry ``energy[i * resolution + j]``
            corresponds to φ = -180 + i * 360/resolution and
            ψ = -180 + j * 360/resolution.
        label: Optional human-readable label for this CMAP grid.

    """

    atom_types_phi: tuple[str, str, str, str]
    atom_types_psi: tuple[str, str, str, str]
    resolution: int
    energy: tuple[float, ...]
    label: str = ""

    def __post_init__(self) -> None:
        energy = tuple(float(v) for v in self.energy)
        object.__setattr__(self, "energy", energy)
        expected = self.resolution * self.resolution
        if len(energy) != expected:
            raise ValueError(
                f"CMAP grid energy has {len(energy)} values, expected {expected} ({self.resolution}×{self.resolution})."
            )
        if self.resolution < 2:
            raise ValueError(f"CMAP resolution must be ≥ 2, got {self.resolution}.")


@dataclass(frozen=True)
class ForceField:
    """Format-agnostic, immutable force field representation.

    Parameters are identified by element tuples, not format-specific
    atom types or line numbers. This eliminates matching bugs between
    different I/O backends.

    Every collection is a tuple; there is no in-place mutation API.
    Building a force field with different parameter values is done via
    :meth:`q2mm.models.parameters.ParameterLayout.replace`, never by
    assigning to an existing instance's fields.

    :attr:`functional_form` is required — there is no implicit default
    (e.g. no silent fallback to ``HARMONIC`` or ``MM3``). Callers must
    always state which physical functional form the parameter values
    are meant to be evaluated under.

    Usage::

        from q2mm.io.mm3 import load_mm3_fld
        ff = load_mm3_fld("mm3.fld")  # functional_form is set by the loader
        ff = ForceField(
            bonds=(BondParam(('C', 'F'), 1.38, 5.0),),
            functional_form=FunctionalForm.MM3,
        )

    """

    name: str = "Q2MM Force Field"
    bonds: tuple[BondParam, ...] = field(default_factory=tuple)
    angles: tuple[AngleParam, ...] = field(default_factory=tuple)
    stretch_bends: tuple[StretchBendParam, ...] = field(default_factory=tuple)
    torsions: tuple[TorsionParam, ...] = field(default_factory=tuple)
    vdws: tuple[VdwParam, ...] = field(default_factory=tuple)
    cmaps: tuple[CmapGrid, ...] = field(default_factory=tuple)
    source_path: Path | None = field(default=None, repr=False)
    source_format: Literal["mm3_fld", "tinker_prm", "openmm_xml", "amber_frcmod", "charmm_prm"] | None = field(
        default=None, repr=False
    )
    functional_form: FunctionalForm = field(kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "bonds", tuple(self.bonds))
        object.__setattr__(self, "angles", tuple(self.angles))
        object.__setattr__(self, "stretch_bends", tuple(self.stretch_bends))
        object.__setattr__(self, "torsions", tuple(self.torsions))
        object.__setattr__(self, "vdws", tuple(self.vdws))
        object.__setattr__(self, "cmaps", tuple(self.cmaps))

    @property
    def _ub_angles(self) -> tuple[AngleParam, ...]:
        """Angles that have Urey-Bradley parameters set."""
        return tuple(a for a in self.angles if a.ub_force_constant is not None and a.ub_equilibrium is not None)

    @property
    def has_urey_bradley(self) -> bool:
        """Whether any angle has Urey-Bradley parameters."""
        return len(self._ub_angles) > 0

    @property
    def has_cmap(self) -> bool:
        """Whether the force field includes CMAP correction grids."""
        return len(self.cmaps) > 0

    def get_bond(
        self,
        elem1: str,
        elem2: str,
        env_id: str = "",
        *,
        bond_order: str = "",
        prefer_generic_context: bool = False,
    ) -> BondParam | None:
        """Find bond parameter by element pair, env_id, and bond order.

        When *prefer_generic_context* is True and multiple candidates
        match, prefer the one with empty context (generic ``0000 0000``
        fallback) over context-specific entries.
        """
        key = tuple(sorted([elem1, elem2]))
        candidates: list[BondParam] = []
        for b in self.bonds:
            if b.key != key:
                continue
            if env_id and b.env_id and b.env_id != env_id:
                continue
            if bond_order and b.bond_order and b.bond_order != bond_order:
                continue
            candidates.append(b)
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # Multiple matches — prefer generic context if requested
        if prefer_generic_context:
            generic = [c for c in candidates if not c.context]
            if generic:
                return generic[0]
        return candidates[0]

    def get_bonds(self, elem1: str, elem2: str) -> list[BondParam]:
        """Find ALL bond parameters matching an element pair."""
        key = tuple(sorted([elem1, elem2]))
        return [b for b in self.bonds if b.key == key]

    def get_angle(self, elem1: str, elem_center: str, elem2: str, env_id: str = "") -> AngleParam | None:
        """Find angle parameter by element triple and optional environment ID."""
        outer = tuple(sorted([elem1, elem2]))
        key = (outer[0], elem_center, outer[1])
        for a in self.angles:
            if a.key == key:
                if env_id and a.env_id and a.env_id != env_id:
                    continue
                return a
        return None

    def get_vdw(self, atom_type: str = "", element: str = "") -> VdwParam | None:
        """Find vdW parameter by atom type or element.

        Args:
            atom_type (str): Exact atom type string to match.
            element (str): Element symbol to match (returns only if unique).

        Returns:
            VdwParam | None: Matching parameter, or None if not found.

        """
        if atom_type:
            normalized = atom_type.strip()
            for vdw in self.vdws:
                if vdw.atom_type == normalized:
                    return vdw
        if element:
            normalized = _extract_element(element)
            matches = [vdw for vdw in self.vdws if vdw.element == normalized]
            if len(matches) == 1:
                return matches[0]
        return None

    def get_torsion(
        self, elem1: str, elem2: str, elem3: str, elem4: str, periodicity: int | None = None, env_id: str = ""
    ) -> TorsionParam | None:
        """Find torsion parameter by element quad and optional periodicity/env_id."""
        target = (elem1, elem2, elem3, elem4)
        target_rev = (elem4, elem3, elem2, elem1)
        for t in self.torsions:
            if t.elements not in (target, target_rev):
                continue
            if periodicity is not None and t.periodicity != periodicity:
                continue
            if env_id and t.env_id and t.env_id != env_id:
                continue
            return t
        return None

    # --- Parameter matching with ff_row → env_id → element fallback ---

    def match_bond(
        self,
        elements: tuple[str, str],
        env_id: str = "",
        ff_row: int | None = None,
        *,
        bond_order: str = "",
        bond_length: float | None = None,
    ) -> BondParam | None:
        """Match a bond parameter using a priority chain.

        Priority:
        1. Exact ``ff_row`` match (highest — used by MacroModel path).
        2. ``env_id`` + ``bond_order`` (typed atom pair + order).
        3. ``env_id`` + closest ``equilibrium`` to ``bond_length``
           (when bond_order is unknown but length is available).
        4. ``env_id`` only, prefer generic context.
        5. Element-only, prefer generic context (lowest).
        """
        # Tier 1: exact ff_row
        if ff_row is not None:
            for bond in self.bonds:
                if bond.ff_row == ff_row:
                    return bond

        e0, e1 = elements[0], elements[1]

        # Tier 2: env_id + bond_order
        if env_id and bond_order:
            matched = self.get_bond(e0, e1, env_id=env_id, bond_order=bond_order, prefer_generic_context=True)
            if matched is not None:
                return matched

        # Tier 3: env_id + closest r₀ to bond_length
        if env_id and bond_length is not None:
            key = tuple(sorted([e0, e1]))
            candidates = [b for b in self.bonds if b.key == key and (not b.env_id or b.env_id == env_id)]
            if candidates:
                best = min(candidates, key=lambda b: abs(b.equilibrium - bond_length))
                return best

        # Tier 4: env_id only, prefer generic context
        if env_id:
            matched = self.get_bond(e0, e1, env_id=env_id, prefer_generic_context=True)
            if matched is not None:
                return matched

        # Tier 5: element-only, prefer generic context
        return self.get_bond(e0, e1, prefer_generic_context=True)

    def match_angle(
        self, elements: tuple[str, str, str], env_id: str = "", ff_row: int | None = None
    ) -> AngleParam | None:
        """Match an angle parameter using ff_row, then env_id, then elements."""
        if ff_row is not None:
            for angle in self.angles:
                if angle.ff_row == ff_row:
                    return angle
        if env_id:
            matched = self.get_angle(elements[0], elements[1], elements[2], env_id=env_id)
            if matched is not None:
                return matched
        return self.get_angle(elements[0], elements[1], elements[2])

    def match_stretch_bend(
        self,
        elements: tuple[str, str, str],
        env_id: str = "",
        ff_row: int | None = None,
    ) -> StretchBendParam | None:
        """Match stretch-bend parameter using ff_row, then env_id, then elements."""
        if ff_row is not None:
            for sb in self.stretch_bends:
                if sb.ff_row == ff_row:
                    return sb
        outer = tuple(sorted([elements[0], elements[2]]))
        target_key = (outer[0], elements[1], outer[1])
        if env_id:
            for sb in self.stretch_bends:
                if sb.key == target_key and sb.env_id == env_id:
                    return sb
        for sb in self.stretch_bends:
            if sb.key == target_key:
                return sb
        return None

    def match_torsion(
        self,
        elements: tuple[str, str, str, str],
        periodicity: int | None = None,
        env_id: str = "",
        ff_row: int | None = None,
        is_improper: bool | None = None,
    ) -> list[TorsionParam]:
        """Match torsion parameters using ff_row, then env_id, then elements.

        Returns all matching ``TorsionParam`` entries (one per periodicity
        component).  Returns an empty list if no match is found.

        Args:
            elements: Element symbols of the four torsion atoms.
            periodicity: If set, only match this periodicity component.
            env_id: Chemical environment identifier.
            ff_row: Optional row index hint for matching.
            is_improper: If set, only match proper (False) or improper (True)
                torsions.  ``None`` matches both.

        """
        if ff_row is not None:
            matches = [t for t in self.torsions if t.ff_row == ff_row]
            if is_improper is not None:
                matches = [t for t in matches if t.is_improper == is_improper]
            if matches:
                if periodicity is not None:
                    matches = [t for t in matches if t.periodicity == periodicity]
                return matches
        target = elements
        target_rev = (elements[3], elements[2], elements[1], elements[0])
        results: list[TorsionParam] = []
        for t in self.torsions:
            if t.elements not in (target, target_rev):
                continue
            if is_improper is not None and t.is_improper != is_improper:
                continue
            if env_id and t.env_id:
                canon_env = canonicalize_torsion_env_id(env_id.split("-"))
                canon_t = canonicalize_torsion_env_id(t.env_id.split("-"))
                if canon_t != canon_env:
                    continue
            if periodicity is not None and t.periodicity != periodicity:
                continue
            results.append(t)
        if not results and env_id:
            return self.match_torsion(
                elements, periodicity=periodicity, env_id="", ff_row=None, is_improper=is_improper
            )
        return results

    @property
    def proper_torsions(self) -> list[TorsionParam]:
        """Proper torsion parameters only (not improper)."""
        return [t for t in self.torsions if not t.is_improper]

    @property
    def improper_torsions(self) -> list[TorsionParam]:
        """Improper torsion parameters only."""
        return [t for t in self.torsions if t.is_improper]

    def match_vdw(self, atom_type: str = "", element: str = "", ff_row: int | None = None) -> VdwParam | None:
        """Match a vdW parameter using ff_row, then atom_type/element lookup (with fallback)."""
        if ff_row is not None:
            for vdw in self.vdws:
                if vdw.ff_row == ff_row:
                    return vdw
        return self.get_vdw(atom_type=atom_type, element=element)

    @classmethod
    def create_for_molecule(
        cls,
        molecule: Molecule,
        *,
        functional_form: FunctionalForm,
        default_bond_k: float = 5.0,
        default_angle_k: float = 0.5,
        name: str = "",
    ) -> ForceField:
        """Create a force field with default parameters for a molecule.

        Auto-detects unique bond and angle types from the molecule's
        geometry and creates parameters with sensible defaults.

        Args:
            molecule: Molecule to auto-detect bond/angle types from.
            functional_form: Required — every :class:`ForceField` must
                carry an explicit functional form (there is no implicit
                default). Callers must decide the scientifically
                correct form for their use case (e.g. ``MM3`` for an
                MM3/Tinker-evaluated force field, ``HARMONIC`` for a
                JAX/JAX-MD/AMBER-evaluated one).
            default_bond_k: Default bond force constant for every
                auto-detected bond type.
            default_angle_k: Default angle force constant for every
                auto-detected angle type.
            name: Force field name; defaults to ``"Auto FF for
                {molecule.name}"`` when empty.

        """
        # Unique bond types
        bond_types: dict[tuple[str, str], list[float]] = {}
        for bond in molecule.bonds:
            key = bond.element_pair
            if key not in bond_types:
                bond_types[key] = []
            bond_types[key].append(bond.length)

        bonds = []
        for key, lengths in bond_types.items():
            avg_len = np.mean(lengths)
            bonds.append(
                BondParam(
                    elements=key,
                    equilibrium=avg_len,
                    force_constant=default_bond_k,
                    label=f"{key[0]}-{key[1]} (auto)",
                )
            )

        # Unique angle types
        angle_types: dict[tuple[str, str, str], list[float]] = {}
        for angle in molecule.angles:
            key = angle.element_triple
            if key not in angle_types:
                angle_types[key] = []
            angle_types[key].append(angle.value)

        angles = []
        for key, values in angle_types.items():
            avg_val = np.mean(values)
            angles.append(
                AngleParam(
                    elements=key,
                    equilibrium=avg_val,
                    force_constant=default_angle_k,
                    label=f"{key[0]}-{key[1]}-{key[2]} (auto)",
                )
            )

        return cls(
            name=name or f"Auto FF for {molecule.name}",
            bonds=tuple(bonds),
            angles=tuple(angles),
            functional_form=functional_form,
        )

    def __repr__(self) -> str:
        return (
            f"ForceField('{self.name}', "
            f"{len(self.bonds)} bonds, {len(self.angles)} angles, "
            f"{len(self.torsions)} torsions, {len(self.vdws)} vdW)"
        )
