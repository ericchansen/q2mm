"""Clean, format-agnostic force field representation for Q2MM.

Decouples Q2MM's optimization from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod). Parameters are identified by element
pairs/triples, not format-specific atom type strings or line numbers.
"""

from __future__ import annotations


import copy
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_torsion_env_id,
)

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule


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


@dataclass
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
    frozen: bool = False

    @property
    def key(self) -> tuple[str, str]:
        """Sorted element pair for canonical matching (e.g., ``('C', 'F')``)."""
        return tuple(sorted(self.elements))


@dataclass
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
    frozen: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical key: center fixed, outers sorted."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass
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
    frozen: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """Canonical key: center fixed, outers sorted."""
        outer = tuple(sorted([self.elements[0], self.elements[2]]))
        return (outer[0], self.elements[1], outer[1])


@dataclass
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
    frozen: bool = False


@dataclass
class VdwParam:
    """An atom-type van der Waals parameter."""

    atom_type: str
    radius: float  # Angstrom
    epsilon: float  # kcal/mol
    element: str = ""
    reduction: float = 0.0
    label: str = ""
    ff_row: int | None = None
    frozen: bool = False

    def __post_init__(self) -> None:
        """Normalize atom_type and auto-extract element if not provided."""
        self.atom_type = str(self.atom_type).strip()
        if not self.element:
            self.element = _extract_element(self.atom_type)


@dataclass
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
        energy: Flat list of energy corrections in kcal/mol, length
            ``resolution * resolution``.  Entry ``energy[i * resolution + j]``
            corresponds to φ = -180 + i * 360/resolution and
            ψ = -180 + j * 360/resolution.
        label: Optional human-readable label for this CMAP grid.

    """

    atom_types_phi: tuple[str, str, str, str]
    atom_types_psi: tuple[str, str, str, str]
    resolution: int
    energy: list[float]
    label: str = ""

    def __post_init__(self) -> None:
        expected = self.resolution * self.resolution
        if len(self.energy) != expected:
            raise ValueError(
                f"CMAP grid energy has {len(self.energy)} values, "
                f"expected {expected} ({self.resolution}×{self.resolution})."
            )
        if self.resolution < 2:
            raise ValueError(f"CMAP resolution must be ≥ 2, got {self.resolution}.")


@dataclass
class ForceField:
    """Format-agnostic force field representation.

    Parameters are identified by element tuples, not format-specific
    atom types or line numbers. This eliminates matching bugs between
    different I/O backends.

    Usage:
        ff = ForceField.from_mm3_fld("mm3.fld")
        ff = ForceField(bonds=[BondParam(('C', 'F'), 1.38, 5.0)])
        ff.to_mm3_fld("output.fld", template_path="mm3.fld")
    """

    name: str = "Q2MM Force Field"
    bonds: list[BondParam] = field(default_factory=list)
    angles: list[AngleParam] = field(default_factory=list)
    stretch_bends: list[StretchBendParam] = field(default_factory=list)
    torsions: list[TorsionParam] = field(default_factory=list)
    vdws: list[VdwParam] = field(default_factory=list)
    cmaps: list[CmapGrid] = field(default_factory=list)
    source_path: Path | None = field(default=None, repr=False)
    source_format: Literal["mm3_fld", "tinker_prm", "openmm_xml", "amber_frcmod", "charmm_prm"] | None = field(
        default=None, repr=False
    )
    functional_form: FunctionalForm | None = field(default=None, repr=True)

    # Schema for the flat parameter vector layout.  Each entry is
    # (collection_attribute, [param_attribute_names...]).  This is the
    # single source of truth consumed by n_params, get_param_vector,
    # set_param_vector, and with_params.
    _PARAM_SLOTS: ClassVar[list[tuple[str, list[str]]]] = [
        ("bonds", ["force_constant", "equilibrium"]),
        ("angles", ["force_constant", "equilibrium"]),
        ("torsions", ["force_constant"]),
        ("stretch_bends", ["force_constant"]),
        ("vdws", ["radius", "epsilon"]),
    ]

    @property
    def _ub_angles(self) -> list[AngleParam]:
        """Angles that have Urey-Bradley parameters set."""
        return [a for a in self.angles if a.ub_force_constant is not None and a.ub_equilibrium is not None]

    @property
    def has_urey_bradley(self) -> bool:
        """Whether any angle has Urey-Bradley parameters."""
        return len(self._ub_angles) > 0

    @property
    def has_cmap(self) -> bool:
        """Whether the force field includes CMAP correction grids."""
        return len(self.cmaps) > 0

    @property
    def n_params(self) -> int:
        """Number of adjustable scalar parameters in get_param_vector().

        Layout: 2 per bond (k, r0) + 2 per angle (k, theta0)
        + 1 per torsion (k) + 2 per vdw (radius, epsilon)
        + 2 per UB angle (ub_k, ub_eq).
        """
        base = sum(len(slots) * len(getattr(self, attr)) for attr, slots in self._PARAM_SLOTS)
        return base + 2 * len(self._ub_angles)

    @property
    def active_mask(self) -> np.ndarray:
        """Boolean mask over get_param_vector() — True for active (non-frozen) params."""
        mask: list[bool] = []
        for attr, slots in self._PARAM_SLOTS:
            for param in getattr(self, attr):
                frozen = getattr(param, "frozen", False)
                mask.extend([not frozen] * len(slots))
        for angle in self._ub_angles:
            frozen = getattr(angle, "frozen", False)
            mask.extend([not frozen, not frozen])
        return np.array(mask, dtype=bool)

    @property
    def n_active_params(self) -> int:
        """Number of active (non-frozen) scalar parameters."""
        return int(self.active_mask.sum())

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

    def get_param_vector(self) -> np.ndarray:
        """Get all adjustable parameters as a flat vector.

        Order: bond (k, r0), angle (k, theta0), torsion (k), vdw (radius, epsilon),
        UB (ub_k, ub_eq) for angles with Urey-Bradley terms.
        """
        values: list[float] = []
        for attr, slots in self._PARAM_SLOTS:
            for param in getattr(self, attr):
                values.extend(getattr(param, s) for s in slots)
        for angle in self._ub_angles:
            values.append(angle.ub_force_constant)
            values.append(angle.ub_equilibrium)
        return np.array(values)

    def get_param_names(self) -> list[str]:
        """Build human-readable names for each parameter in get_param_vector() order."""
        names: list[str] = []
        for bond in self.bonds:
            label = "-".join(bond.key) + (f"[{bond.env_id}]" if bond.env_id else "")
            names.append(f"kb_{label}")
            names.append(f"r0_{label}")
        for angle in self.angles:
            label = "-".join(angle.key) + (f"[{angle.env_id}]" if angle.env_id else "")
            names.append(f"ka_{label}")
            names.append(f"th0_{label}")
        for torsion in self.torsions:
            label = "-".join(torsion.elements) + f"_n{torsion.periodicity}"
            if torsion.is_improper:
                label += "_imp"
            names.append(f"kt_{label}")
        for stretch_bend in self.stretch_bends:
            label = "-".join(stretch_bend.key) + (f"[{stretch_bend.env_id}]" if stretch_bend.env_id else "")
            names.append(f"ksb_{label}")
        for vdw in self.vdws:
            label = vdw.atom_type or vdw.element
            names.append(f"rvdw_{label}")
            names.append(f"evdw_{label}")
        for angle in self._ub_angles:
            label = "-".join(angle.key) + (f"[{angle.env_id}]" if angle.env_id else "")
            names.append(f"kub_{label}")
            names.append(f"r13_{label}")
        return names

    def get_active_param_vector(self) -> np.ndarray:
        """Get only the active (non-frozen) parameters as a flat vector."""
        return self.get_param_vector()[self.active_mask]

    def set_active_param_vector(self, vec: np.ndarray) -> None:
        """Set only the active (non-frozen) parameters from a flat vector.

        Frozen parameters are left unchanged.
        """
        if len(vec) != self.n_active_params:
            raise ValueError(
                f"Active parameter vector length {len(vec)} does not match "
                f"expected {self.n_active_params} active parameters."
            )
        full = self.get_param_vector()
        full[self.active_mask] = vec
        self.set_param_vector(full)

    def with_active_params(self, vec: np.ndarray) -> ForceField:
        """Return a new ForceField with active parameters set from *vec*.

        Frozen parameters retain their current values.
        """
        if len(vec) != self.n_active_params:
            raise ValueError(
                f"Active parameter vector length {len(vec)} does not match "
                f"expected {self.n_active_params} active parameters."
            )
        full = self.get_param_vector()
        full[self.active_mask] = vec
        return self.with_params(full)

    def get_active_param_names(self) -> list[str]:
        """Get parameter names for active (non-frozen) parameters only."""
        all_names = self.get_param_names()
        mask = self.active_mask
        return [name for name, is_active in zip(all_names, mask, strict=True) if is_active]

    def get_active_step_sizes(self) -> np.ndarray:
        """Get step sizes for active (non-frozen) parameters only."""
        return self.get_step_sizes()[self.active_mask]

    def get_active_bounds(self) -> np.ndarray:
        """Get bounds for active (non-frozen) parameters only.

        Returns array of shape (n_active_params, 2).
        """
        bounds = np.asarray(self.get_bounds(), dtype=float)
        if bounds.size == 0:
            return bounds.reshape(0, 2)
        return bounds[self.active_mask]

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

    def set_param_vector(self, vec: np.ndarray) -> None:
        """Set parameters from a flat vector (inverse of get_param_vector)."""
        if len(vec) != self.n_params:
            raise ValueError(f"Parameter vector length {len(vec)} does not match expected {self.n_params} parameters.")
        idx = 0
        for attr, slots in self._PARAM_SLOTS:
            for param in getattr(self, attr):
                for s in slots:
                    setattr(param, s, vec[idx])
                    idx += 1
        for angle in self._ub_angles:
            angle.ub_force_constant = vec[idx]
            angle.ub_equilibrium = vec[idx + 1]
            idx += 2

    def with_params(self, vec: np.ndarray) -> ForceField:
        """Return a new ForceField with parameters set from *vec*.

        Unlike :meth:`set_param_vector`, this does **not** mutate the
        current instance.  The returned object shares metadata (labels,
        env_ids, source_path, …) but has independent parameter values.

        Args:
            vec: Flat parameter vector (same layout as
                :meth:`get_param_vector`).

        Returns:
            A new :class:`ForceField` with updated parameter values.

        Raises:
            ValueError: If *vec* length does not match :attr:`n_params`.

        """
        if len(vec) != self.n_params:
            raise ValueError(f"Parameter vector length {len(vec)} does not match expected {self.n_params} parameters.")
        idx = 0
        new_collections: dict[str, list] = {}
        for attr, slots in self._PARAM_SLOTS:
            new_list = []
            for param in getattr(self, attr):
                updates = {}
                for s in slots:
                    updates[s] = vec[idx]
                    idx += 1
                new_list.append(replace(param, **updates))
            new_collections[attr] = new_list
        # Update UB params on the new angle list
        ub_angles = [
            a for a in new_collections["angles"] if a.ub_force_constant is not None and a.ub_equilibrium is not None
        ]
        for angle in ub_angles:
            angle.ub_force_constant = float(vec[idx])
            angle.ub_equilibrium = float(vec[idx + 1])
            idx += 2
        return replace(self, **new_collections)

    @staticmethod
    def _param_identity(
        attr: str,
        param: BondParam | AngleParam | StretchBendParam | TorsionParam | VdwParam,
    ) -> tuple:
        """Build a stable identity for matching parameters across FF variants."""
        if attr == "vdws":
            return (attr, param.atom_type, param.element)
        if attr == "torsions":
            elements = min(param.elements, tuple(reversed(param.elements)))
            env_id = ""
            if param.env_id:
                env_id = canonicalize_torsion_env_id(param.env_id.split("-"))
            return (attr, elements, param.periodicity, param.is_improper, env_id)
        return (attr, param.key, getattr(param, "env_id", ""))

    def freeze_all(self) -> None:
        """Mark all parameters as frozen (not optimizable)."""
        for attr, _ in self._PARAM_SLOTS:
            for param in getattr(self, attr):
                param.frozen = True
        for angle in self._ub_angles:
            angle.frozen = True

    def freeze_standard_params(self, opt_ff: ForceField) -> None:
        """Mark params as frozen unless they match an OPT-substructure param."""
        self.freeze_all()

        same_source = (
            self.source_path is not None
            and opt_ff.source_path is not None
            and self.source_path.resolve() == opt_ff.source_path.resolve()
        )
        opt_rows = {
            attr: Counter(param.ff_row for param in getattr(opt_ff, attr) if param.ff_row is not None)
            for attr, _ in self._PARAM_SLOTS
        }
        opt_ids = {
            attr: Counter(self._param_identity(attr, param) for param in getattr(opt_ff, attr))
            for attr, _ in self._PARAM_SLOTS
        }

        for attr, _ in self._PARAM_SLOTS:
            for param in getattr(self, attr):
                if same_source and param.ff_row is not None:
                    if opt_rows[attr][param.ff_row] > 0:
                        param.frozen = False
                        opt_rows[attr][param.ff_row] -= 1
                    continue
                ident = self._param_identity(attr, param)
                if opt_ids[attr][ident] > 0:
                    param.frozen = False
                    opt_ids[attr][ident] -= 1

        opt_ub_rows = Counter(angle.ff_row for angle in opt_ff._ub_angles if angle.ff_row is not None)
        opt_ub_ids = Counter(self._param_identity("angles", angle) for angle in opt_ff._ub_angles)
        for angle in self._ub_angles:
            if same_source and angle.ff_row is not None:
                if opt_ub_rows[angle.ff_row] > 0:
                    angle.frozen = False
                    opt_ub_rows[angle.ff_row] -= 1
                continue
            ident = self._param_identity("angles", angle)
            if opt_ub_ids[ident] > 0:
                angle.frozen = False
                opt_ub_ids[ident] -= 1

    # Default bounds per parameter type (min, max) in canonical units.
    # bond_k allows negative values for transition-state force fields (TSFF),
    # where reaction-coordinate bonds have negative force constants.
    # Bond/angle k in kcal/(mol·Å²) and kcal/(mol·rad²) respectively.
    DEFAULT_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "bond_k": (-3600.0, 3600.0),
        "bond_eq": (0.5, 3.0),
        "angle_k": (-720.0, 720.0),
        "angle_eq": (30.0, 180.0),
        "torsion_k": (-20.0, 20.0),
        "sb_k": (-50.0, 50.0),
        "vdw_radius": (0.5, 5.0),
        "vdw_epsilon": (0.001, 2.0),
        "ub_k": (0.0, 500.0),
        "ub_eq": (1.0, 4.0),
    }

    # Maps ForceField param-vector slot types to legacy STEPS keys for
    # per-type differentiation step sizes (upstream constants.py).
    _PARAM_TYPE_TO_STEP_KEY: ClassVar[dict[str, str]] = {
        "bond_k": "bf",
        "bond_eq": "be",
        "angle_k": "af",
        "angle_eq": "ae",
        "torsion_k": "df",
        "sb_k": "sb",
        "vdw_radius": "vdwr",
        "vdw_epsilon": "vdwfc",
        "ub_k": "bf",
        "ub_eq": "be",
    }

    def get_param_indices_by_type(self) -> dict[str, list[int]]:
        """Map parameter type names to their indices in the param vector.

        Returns a dict with keys ``bond_k``, ``bond_eq``, ``angle_k``,
        ``angle_eq``, ``torsion_k``, ``vdw_radius``, ``vdw_epsilon``,
        ``ub_k``, ``ub_eq`` and values that are lists of integer indices
        into :meth:`get_param_vector`.
        """
        idx = 0
        result: dict[str, list[int]] = {
            "bond_k": [],
            "bond_eq": [],
            "angle_k": [],
            "angle_eq": [],
            "torsion_k": [],
            "sb_k": [],
            "vdw_radius": [],
            "vdw_epsilon": [],
            "ub_k": [],
            "ub_eq": [],
        }
        for _ in self.bonds:
            result["bond_k"].append(idx)
            result["bond_eq"].append(idx + 1)
            idx += 2
        for _ in self.angles:
            result["angle_k"].append(idx)
            result["angle_eq"].append(idx + 1)
            idx += 2
        for _ in self.torsions:
            result["torsion_k"].append(idx)
            idx += 1
        for _ in self.stretch_bends:
            result["sb_k"].append(idx)
            idx += 1
        for _ in self.vdws:
            result["vdw_radius"].append(idx)
            result["vdw_epsilon"].append(idx + 1)
            idx += 2
        for _ in self._ub_angles:
            result["ub_k"].append(idx)
            result["ub_eq"].append(idx + 1)
            idx += 2
        return result

    def get_param_type_labels(self) -> list[str]:
        """Return the type label for each element of the param vector.

        Same length as :meth:`get_param_vector`, useful for mapping each
        scalar to its per-type step size or bounds category.
        """
        labels: list[str] = []
        for _ in self.bonds:
            labels.extend(["bond_k", "bond_eq"])
        for _ in self.angles:
            labels.extend(["angle_k", "angle_eq"])
        for _ in self.torsions:
            labels.append("torsion_k")
        for _ in self.stretch_bends:
            labels.append("sb_k")
        for _ in self.vdws:
            labels.extend(["vdw_radius", "vdw_epsilon"])
        for _ in self._ub_angles:
            labels.extend(["ub_k", "ub_eq"])
        return labels

    def get_step_sizes(self) -> np.ndarray:
        """Per-element differentiation step sizes for the param vector.

        Uses the legacy ``STEPS`` dictionary values from
        :mod:`q2mm.optimizers.defaults`, mapped via
        :attr:`_PARAM_TYPE_TO_STEP_KEY`.

        Returns
        -------
        np.ndarray
            Array of step sizes, same length as :meth:`get_param_vector`.

        """
        from q2mm.optimizers.defaults import STEPS

        labels = self.get_param_type_labels()
        return np.array([STEPS[self._PARAM_TYPE_TO_STEP_KEY[lbl]] for lbl in labels])

    def get_bounds(self, overrides: dict[str, tuple[float, float]] | None = None) -> list[tuple[float, float]]:
        """Get (min, max) bounds for each element of the param vector.

        Matches the layout of :meth:`get_param_vector`:
        bond (k, r0), angle (k, theta0), torsion (k), stretch-bend (k),
        vdw (radius, epsilon), UB (ub_k, ub_eq).

        Parameters
        ----------
        overrides : dict, optional
            Override default bounds per type. Keys: ``bond_k``,
            ``bond_eq``, ``angle_k``, ``angle_eq``, ``torsion_k``,
            ``sb_k``, ``vdw_radius``, ``vdw_epsilon``, ``ub_k``,
            ``ub_eq``.

        """
        b = {**self.DEFAULT_BOUNDS, **(overrides or {})}
        bounds: list[tuple[float, float]] = []
        for _bond in self.bonds:
            bounds.append(b["bond_k"])
            bounds.append(b["bond_eq"])
        for _angle in self.angles:
            bounds.append(b["angle_k"])
            bounds.append(b["angle_eq"])
        for _torsion in self.torsions:
            bounds.append(b["torsion_k"])
        for _sb in self.stretch_bends:
            bounds.append(b["sb_k"])
        for _vdw in self.vdws:
            bounds.append(b["vdw_radius"])
            bounds.append(b["vdw_epsilon"])
        for _ub in self._ub_angles:
            bounds.append(b["ub_k"])
            bounds.append(b["ub_eq"])
        return bounds

    def copy(self) -> ForceField:
        """Deep copy."""
        return copy.deepcopy(self)

    # ---- Format converters ----

    @classmethod
    def from_mm3_fld(cls, path: str | Path, *, include_standard: bool = True) -> ForceField:
        """Load from Schrödinger MM3 .fld file.

        Args:
            path: Path to the mm3.fld file.
            include_standard: When ``True`` (the default), also load
                standard MM3 parameters from the main body of the file.

        """
        from q2mm.io.mm3 import load_mm3_fld

        return load_mm3_fld(path, include_standard=include_standard)

    @classmethod
    def from_tinker_prm(cls, path: str | Path) -> ForceField:
        """Load bond and angle parameters from a Tinker .prm file."""
        from q2mm.io.tinker import load_tinker_prm

        return load_tinker_prm(path)

    def to_mm3_fld(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        substructure_name: str = "OPT Generated",
        smiles: str = "AUTO",
    ) -> Path:
        """Export to MM3 .fld format."""
        from q2mm.io.mm3 import save_mm3_fld

        return save_mm3_fld(self, path, template_path, substructure_name=substructure_name, smiles=smiles)

    def to_tinker_prm(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        section_name: str = "OPT Generated",
    ) -> Path:
        """Export to Tinker .prm format."""
        from q2mm.io.tinker import save_tinker_prm

        return save_tinker_prm(self, path, template_path, section_name=section_name)

    def to_openmm_xml(
        self,
        path: str | Path,
        molecule: Q2MMMolecule | list[Q2MMMolecule] | None = None,
    ) -> Path:
        """Export to OpenMM ForceField XML format.

        Produces a standalone ``<ForceField>`` XML file loadable by
        ``openmm.app.ForceField(path)``.  Uses custom force definitions
        with MM3 functional forms (cubic bond, sextic angle, buffered
        14-7 vdW).

        Args:
            path (str | Path): Output file path.
            molecule (Q2MMMolecule | list[Q2MMMolecule] | None): Optional molecule(s) for generating
                ``<AtomTypes>`` and ``<Residues>`` sections.

        Returns:
            The resolved output path.

        """
        from q2mm.io.openmm import save_openmm_xml

        return save_openmm_xml(self, path, molecule=molecule)

    @classmethod
    def from_amber_frcmod(cls, path: str | Path) -> ForceField:
        """Load from an AMBER .frcmod file."""
        from q2mm.io.amber import load_amber_frcmod

        return load_amber_frcmod(path)

    def to_amber_frcmod(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        remark: str = "Q2MM generated frcmod",
    ) -> Path:
        """Export to AMBER .frcmod format."""
        from q2mm.io.amber import save_amber_frcmod

        return save_amber_frcmod(self, path, template_path, remark=remark)

    @classmethod
    def create_for_molecule(
        cls, molecule: Q2MMMolecule, default_bond_k: float = 5.0, default_angle_k: float = 0.5, name: str = ""
    ) -> ForceField:
        """Create a force field with default parameters for a molecule.

        Auto-detects unique bond and angle types from the molecule's
        geometry and creates parameters with sensible defaults.
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
            bonds=bonds,
            angles=angles,
        )

    def __repr__(self) -> str:
        return (
            f"ForceField('{self.name}', "
            f"{len(self.bonds)} bonds, {len(self.angles)} angles, "
            f"{len(self.torsions)} torsions, {len(self.vdws)} vdW)"
        )
