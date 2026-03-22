"""Clean, format-agnostic force field representation for Q2MM.

Decouples Q2MM's optimization from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod). Parameters are identified by element
pairs/triples, not format-specific atom type strings or line numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule


def _split_env_id(env_id: str, expected_len: int) -> list[str]:
    parts = [part.strip() for part in env_id.split("-") if part.strip()]
    if len(parts) == expected_len:
        return parts
    return []


def _default_tinker_atom_types(elements: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    atom_types = []
    for element in elements:
        count = counts.get(element, 0) + 1
        counts[element] = count
        atom_types.append(f"{element}{count}")
    return atom_types


def _default_mm3_atom_types(elements: tuple[str, ...]) -> list[str]:
    counts: dict[str, int] = {}
    atom_types = []
    for element in elements:
        normalized = _extract_element(element)
        count = counts.get(normalized, 0) + 1
        counts[normalized] = count
        if len(normalized) == 1:
            atom_types.append(f"{normalized}{count}")
        else:
            atom_types.append(normalized[:2].upper())
    return atom_types


def _tinker_atom_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    return _split_env_id(env_id, len(elements)) or _default_tinker_atom_types(elements)


def _mm3_atom_types(env_id: str, elements: tuple[str, ...]) -> list[str]:
    parts = _split_env_id(env_id, len(elements))
    if parts and all(len(part) <= 2 for part in parts):
        return parts
    return _default_mm3_atom_types(elements)


def _format_mm3_bond_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 1{atom_types[0]:>4}{atom_types[1]:>4}{'':13}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_angle_line(atom_types: list[str], equilibrium: float, force_constant: float) -> str:
    prefix = f" 2{atom_types[0]:>4}{atom_types[1]:>4}{atom_types[2]:>4}{'':9}"
    return f"{prefix}{equilibrium:10.4f} {force_constant:10.4f}\n"


def _format_mm3_torsion_line(atom_types: list[str], v1: float, v2: float, v3: float) -> str:
    prefix = f" 4{atom_types[0]:>4}{atom_types[1]:>4}{atom_types[2]:>4}{atom_types[3]:>4}{'':5}"
    return f"{prefix}{v1:10.4f} {v2:10.4f} {v3:10.4f}\n"


def _format_tinker_bond_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return f"bond   {atom_types[0]:>4} {atom_types[1]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"


def _format_tinker_angle_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return (
        f"angle  {atom_types[0]:>4} {atom_types[1]:>4} {atom_types[2]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"
    )


def _format_tinker_vdw_line(atom_type: str, radius: float, epsilon: float, reduction: float = 0.0) -> str:
    return f"vdw    {atom_type:>4} {radius:10.4f} {epsilon:10.4f} {reduction:10.4f}\n"


def _clean_atom_types(atom_types: list[str] | tuple[str, ...] | None, expected_len: int) -> list[str]:
    if atom_types is None:
        return []
    if isinstance(atom_types, str):
        atom_types = [atom_types]
    cleaned = [
        str(atom_type).strip() for atom_type in atom_types if str(atom_type).strip() and str(atom_type).strip() != "-"
    ]
    return cleaned[:expected_len]


def _clean_atom_type(atom_types: list[str] | tuple[str, ...] | str | None) -> str:
    cleaned = _clean_atom_types(atom_types, 1)
    return cleaned[0] if cleaned else ""


def _build_param_maps(params, secondary_key: str) -> tuple[dict, dict]:
    """Build ff_row and secondary-key lookup dicts for a list of parameters."""
    by_row = {p.ff_row: p for p in params if p.ff_row is not None}
    by_key = {getattr(p, secondary_key): p for p in params if getattr(p, secondary_key, None)}
    return by_row, by_key


def _build_bond_maps(bonds: list[BondParam]) -> tuple[dict[int, BondParam], dict[str, BondParam]]:
    return _build_param_maps(bonds, "env_id")


def _build_angle_maps(angles: list[AngleParam]) -> tuple[dict[int, AngleParam], dict[str, AngleParam]]:
    return _build_param_maps(angles, "env_id")


def _build_vdw_maps(vdws: list[VdwParam]) -> tuple[dict[int, VdwParam], dict[str, VdwParam]]:
    return _build_param_maps(vdws, "atom_type")


def _match_for_export(param, by_row: dict, by_env: dict, expected_len: int, canonicalize_fn):
    """Match a parsed parameter to an internal param by ff_row or env_id."""
    if param.ff_row is not None and param.ff_row in by_row:
        return by_row[param.ff_row]
    atom_types = _clean_atom_types(getattr(param, "atom_types", None), expected_len)
    if len(atom_types) == expected_len:
        return by_env.get(canonicalize_fn(atom_types))
    return None


def _match_bond_for_export(
    param, bond_by_row: dict[int, BondParam], bond_by_env: dict[str, BondParam]
) -> BondParam | None:
    return _match_for_export(param, bond_by_row, bond_by_env, 2, canonicalize_bond_env_id)


def _match_angle_for_export(
    param,
    angle_by_row: dict[int, AngleParam],
    angle_by_env: dict[str, AngleParam],
) -> AngleParam | None:
    return _match_for_export(param, angle_by_row, angle_by_env, 3, canonicalize_angle_env_id)


@dataclass
class BondParam:
    """A bond force field parameter."""

    elements: tuple[str, str]  # Sorted element pair, e.g., ('C', 'F')
    equilibrium: float  # Angstrom
    force_constant: float  # mdyn/A (MM3 units)
    label: str = ""  # Human-readable label
    env_id: str = ""  # Environment ID for disambiguating same-element params
    # (e.g., MM3 ff_row, atom type codes 'C1-F1' vs 'C2-F1')
    ff_row: int | None = None  # Source force-field row for exact legacy parity

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted(self.elements))


@dataclass
class AngleParam:
    """An angle force field parameter."""

    elements: tuple[str, str, str]  # (outer, center, outer)
    equilibrium: float  # degrees
    force_constant: float  # mdyn*A/rad^2
    label: str = ""
    env_id: str = ""  # Environment ID for disambiguating same-element params
    ff_row: int | None = None  # Source force-field row for exact legacy parity

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
    """

    elements: tuple[str, str, str, str]
    periodicity: int = 1
    force_constant: float = 0.0  # kcal/mol
    phase: float = 0.0  # degrees
    label: str = ""
    env_id: str = ""  # Environment ID for disambiguating same-element params
    ff_row: int | None = None  # Source force-field row for legacy parity


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

    def __post_init__(self):
        self.atom_type = str(self.atom_type).strip()
        if not self.element:
            self.element = _extract_element(self.atom_type)


@dataclass
class ForceField:
    """Format-agnostic force field representation.

    Parameters are identified by element tuples, not format-specific
    atom types or line numbers. This eliminates matching bugs between
    different parsers.

    Usage:
        ff = ForceField.from_mm3_fld("mm3.fld")
        ff = ForceField(bonds=[BondParam(('C', 'F'), 1.38, 5.0)])
        # Export to MM3 .fld is planned but not yet implemented.
    """

    name: str = "Q2MM Force Field"
    bonds: list[BondParam] = field(default_factory=list)
    angles: list[AngleParam] = field(default_factory=list)
    torsions: list[TorsionParam] = field(default_factory=list)
    vdws: list[VdwParam] = field(default_factory=list)
    source_path: Path | None = field(default=None, repr=False)
    source_format: Literal["mm3_fld", "tinker_prm", "openmm_xml", "amber_frcmod"] | None = field(
        default=None, repr=False
    )

    @property
    def n_params(self) -> int:
        """Number of adjustable scalar parameters in get_param_vector().

        Layout: 2 per bond (k, r0) + 2 per angle (k, theta0)
        + 1 per torsion (k) + 2 per vdw (radius, epsilon).
        """
        return 2 * len(self.bonds) + 2 * len(self.angles) + len(self.torsions) + 2 * len(self.vdws)

    def get_bond(self, elem1: str, elem2: str, env_id: str = "") -> BondParam | None:
        """Find bond parameter by element pair and optional environment ID."""
        key = tuple(sorted([elem1, elem2]))
        for b in self.bonds:
            if b.key == key:
                if env_id and b.env_id and b.env_id != env_id:
                    continue
                return b
        return None

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

        Order: bond (k, r0), angle (k, theta0), torsion (k), vdw (radius, epsilon).
        """
        values = []
        for b in self.bonds:
            values.extend([b.force_constant, b.equilibrium])
        for a in self.angles:
            values.extend([a.force_constant, a.equilibrium])
        for t in self.torsions:
            values.append(t.force_constant)
        for vdw in self.vdws:
            values.extend([vdw.radius, vdw.epsilon])
        return np.array(values)

    # --- Parameter matching with ff_row → env_id → element fallback ---

    def match_bond(self, elements: tuple[str, str], env_id: str = "", ff_row: int | None = None) -> BondParam | None:
        """Match a bond parameter using ff_row, then env_id, then elements."""
        if ff_row is not None:
            for bond in self.bonds:
                if bond.ff_row == ff_row:
                    return bond
        if env_id:
            matched = self.get_bond(elements[0], elements[1], env_id=env_id)
            if matched is not None:
                return matched
        return self.get_bond(elements[0], elements[1])

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

    def match_vdw(self, atom_type: str = "", element: str = "", ff_row: int | None = None) -> VdwParam | None:
        """Match a vdW parameter using ff_row, then atom_type/element lookup (with fallback)."""
        if ff_row is not None:
            for vdw in self.vdws:
                if vdw.ff_row == ff_row:
                    return vdw
        return self.get_vdw(atom_type=atom_type, element=element)

    def set_param_vector(self, vec: np.ndarray):
        """Set parameters from a flat vector (inverse of get_param_vector)."""
        if len(vec) != self.n_params:
            raise ValueError(f"Parameter vector length {len(vec)} does not match expected {self.n_params} parameters.")
        idx = 0
        for b in self.bonds:
            b.force_constant = vec[idx]
            b.equilibrium = vec[idx + 1]
            idx += 2
        for a in self.angles:
            a.force_constant = vec[idx]
            a.equilibrium = vec[idx + 1]
            idx += 2
        for t in self.torsions:
            t.force_constant = vec[idx]
            idx += 1
        for vdw in self.vdws:
            vdw.radius = vec[idx]
            vdw.epsilon = vec[idx + 1]
            idx += 2

    # Default bounds per parameter type (min, max).
    # bond_k allows negative values for transition-state force fields (TSFF),
    # where reaction-coordinate bonds have negative force constants.
    DEFAULT_BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        "bond_k": (-50.0, 50.0),
        "bond_eq": (0.5, 3.0),
        "angle_k": (-10.0, 10.0),
        "angle_eq": (30.0, 180.0),
        "torsion_k": (-20.0, 20.0),
        "vdw_radius": (0.5, 5.0),
        "vdw_epsilon": (0.001, 2.0),
    }

    def get_bounds(self, overrides: dict[str, tuple[float, float]] | None = None) -> list[tuple[float, float]]:
        """Get (min, max) bounds for each element of the param vector.

        Matches the layout of :meth:`get_param_vector`:
        bond (k, r0), angle (k, theta0), torsion (k), vdw (radius, epsilon).

        Parameters
        ----------
        overrides : dict, optional
            Override default bounds per type. Keys: ``bond_k``,
            ``bond_eq``, ``angle_k``, ``angle_eq``, ``torsion_k``,
            ``vdw_radius``, ``vdw_epsilon``.
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
        for _vdw in self.vdws:
            bounds.append(b["vdw_radius"])
            bounds.append(b["vdw_epsilon"])
        return bounds

    def copy(self) -> ForceField:
        """Deep copy."""
        return copy.deepcopy(self)

    # ---- Format converters ----

    @classmethod
    def from_mm3_fld(cls, path: str | Path) -> ForceField:
        """Load from Schrödinger MM3 .fld file."""
        from q2mm.models.ff_io import load_mm3_fld

        return load_mm3_fld(path)

    @classmethod
    def from_tinker_prm(cls, path: str | Path) -> ForceField:
        """Load bond and angle parameters from a Tinker .prm file."""
        from q2mm.models.ff_io import load_tinker_prm

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
        from q2mm.models.ff_io import save_mm3_fld

        return save_mm3_fld(self, path, template_path, substructure_name=substructure_name, smiles=smiles)

    def to_tinker_prm(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        section_name: str = "OPT Generated",
    ) -> Path:
        """Export to Tinker .prm format."""
        from q2mm.models.ff_io import save_tinker_prm

        return save_tinker_prm(self, path, template_path, section_name=section_name)

    def to_openmm_xml(
        self,
        path: str | Path,
        molecule=None,
    ) -> Path:
        """Export to OpenMM ForceField XML format.

        Produces a standalone ``<ForceField>`` XML file loadable by
        ``openmm.app.ForceField(path)``.  Uses custom force definitions
        with MM3 functional forms (cubic bond, sextic angle, buffered
        14-7 vdW).

        Args:
            path: Output file path.
            molecule: Optional molecule(s) for generating
                ``<AtomTypes>`` and ``<Residues>`` sections.

        Returns:
            The resolved output path.
        """
        from q2mm.models.ff_io import save_openmm_xml

        return save_openmm_xml(self, path, molecule=molecule)

    @classmethod
    def from_amber_frcmod(cls, path: str | Path) -> ForceField:
        """Load from an AMBER .frcmod file."""
        from q2mm.models.ff_io import load_amber_frcmod

        return load_amber_frcmod(path)

    def to_amber_frcmod(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        remark: str = "Q2MM generated frcmod",
    ) -> Path:
        """Export to AMBER .frcmod format."""
        from q2mm.models.ff_io import save_amber_frcmod

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


def _format_mm3_vdw_line(vdw: VdwParam) -> str:
    return f"  {vdw.atom_type:<3} {vdw.radius:10.4f} {vdw.epsilon:10.4f} {vdw.reduction:10.4f}                                   0000    O 1\n"


def _parse_mm3_vdw_params(path: Path) -> list[VdwParam]:
    vdws: list[VdwParam] = []
    in_vdw_section = False
    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if stripped == "-6":
            in_vdw_section = True
            continue
        if not in_vdw_section:
            continue
        if stripped.startswith("-2") or "END OF NONBONDED INTERACTIONS" in stripped:
            break
        parts = raw_line.split()
        if len(parts) < 3:
            continue
        try:
            radius = float(parts[1])
            epsilon = float(parts[2])
        except ValueError:
            continue
        atom_type = parts[0]
        vdws.append(
            VdwParam(
                atom_type=atom_type,
                radius=radius,
                epsilon=epsilon,
                reduction=float(parts[3]) if len(parts) > 3 else 0.0,
                label=f"MM3 row {row}",
                ff_row=row,
            )
        )
    return vdws


def _parse_tinker_vdw_params(path: Path) -> list[VdwParam]:
    vdws: list[VdwParam] = []
    q2mm_sec = False
    gather_data = False
    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        if not q2mm_sec and "# Q2MM" in raw_line:
            q2mm_sec = True
            continue
        if q2mm_sec and raw_line.startswith("#"):
            gather_data = "OPT" in raw_line
            continue
        if not gather_data:
            continue
        parts = raw_line.split()
        if not parts or parts[0] != "vdw" or len(parts) < 4:
            continue
        vdws.append(
            VdwParam(
                atom_type=parts[1],
                radius=float(parts[2]),
                epsilon=float(parts[3]),
                reduction=float(parts[4]) if len(parts) > 4 else 0.0,
                label=f"Tinker row {row}",
                ff_row=row,
            )
        )
    return vdws


def _parse_generic_tinker_prm(path: Path) -> tuple[list[BondParam], list[AngleParam], list[VdwParam]]:
    bonds: list[BondParam] = []
    angles: list[AngleParam] = []
    vdws: list[VdwParam] = []
    atom_elements: dict[str, str] = {}

    for row, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = raw_line.split()
        record = parts[0].lower()

        if record == "atom" and len(parts) >= 3:
            # Standard Tinker: atom <type> <symbol> "desc" <anum> <mass> <val>
            # AMOEBA-style:    atom <type> <class> <symbol> "desc" ...
            # Distinguish: if parts[2] is purely numeric, it's a class field.
            symbol_col = 2
            if parts[2].isdigit() and len(parts) >= 4:
                symbol_col = 3
            atom_elements[parts[1]] = _extract_element(parts[symbol_col])
            continue

        if record.startswith("bond") and len(parts) >= 5:
            atom_types = parts[1:3]
            elements = tuple(atom_elements.get(atom_type, _extract_element(atom_type)) for atom_type in atom_types)
            bonds.append(
                BondParam(
                    elements=elements,
                    equilibrium=float(parts[4]),
                    force_constant=float(parts[3]),
                    label=f"Tinker row {row}",
                    env_id=canonicalize_bond_env_id(atom_types),
                    ff_row=row,
                )
            )
            continue

        if record.startswith("angle") and len(parts) >= 6:
            atom_types = parts[1:4]
            elements = tuple(atom_elements.get(atom_type, _extract_element(atom_type)) for atom_type in atom_types)
            angles.append(
                AngleParam(
                    elements=elements,
                    equilibrium=float(parts[5]),
                    force_constant=float(parts[4]),
                    label=f"Tinker row {row}",
                    env_id=canonicalize_angle_env_id(atom_types),
                    ff_row=row,
                )
            )
            continue

        if record == "vdw" and len(parts) >= 4:
            atom_type = parts[1]
            vdws.append(
                VdwParam(
                    atom_type=atom_type,
                    radius=float(parts[2]),
                    epsilon=float(parts[3]),
                    reduction=float(parts[4]) if len(parts) > 4 else 0.0,
                    label=f"Tinker row {row}",
                    ff_row=row,
                    element=atom_elements.get(atom_type, _extract_element(atom_type)),
                )
            )

    return bonds, angles, vdws


def _update_mm3_vdw_lines(path: Path, vdws: list[VdwParam]):
    lines = path.read_text(encoding="utf-8").splitlines()
    by_row, by_type = _build_vdw_maps(vdws)
    for index, line in enumerate(lines):
        row = index + 1
        match = by_row.get(row)
        parts = line.split()
        if match is None and parts:
            match = by_type.get(parts[0].strip())
        if match is None:
            continue
        tail = " ".join(parts[4:]) if len(parts) > 4 else ""
        lines[index] = f"  {match.atom_type:<3} {match.radius:10.4f} {match.epsilon:10.4f} {match.reduction:10.4f}" + (
            f" {tail}" if tail else ""
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _update_tinker_vdw_lines(path: Path, vdws: list[VdwParam]):
    lines = path.read_text(encoding="utf-8").splitlines()
    by_row, by_type = _build_vdw_maps(vdws)
    for index, line in enumerate(lines):
        row = index + 1
        parts = line.split()
        if not parts or parts[0] != "vdw":
            continue
        match = by_row.get(row)
        if match is None and len(parts) > 1:
            match = by_type.get(parts[1].strip())
        if match is None:
            continue
        base = f"vdw    {match.atom_type:>4} {match.radius:10.4f} {match.epsilon:10.4f}"
        if match.reduction != 0.0:
            base += f" {match.reduction:10.4f}"
        lines[index] = base
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
