"""Clean, format-agnostic force field representation for Q2MM.

Decouples Q2MM's optimization from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod). Parameters are identified by element
pairs/triples, not format-specific atom type strings or line numbers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from q2mm.models.identifiers import (
    _extract_element,
    canonicalize_angle_env_id,
    canonicalize_bond_env_id,
)


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


def _format_tinker_bond_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return f"bond   {atom_types[0]:>4} {atom_types[1]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"


def _format_tinker_angle_line(atom_types: list[str], force_constant: float, equilibrium: float) -> str:
    return (
        f"angle  {atom_types[0]:>4} {atom_types[1]:>4} {atom_types[2]:>4} {force_constant:10.4f} {equilibrium:10.4f}\n"
    )


def _clean_atom_types(atom_types: list[str] | tuple[str, ...] | None, expected_len: int) -> list[str]:
    if atom_types is None:
        return []
    cleaned = [
        str(atom_type).strip() for atom_type in atom_types if str(atom_type).strip() and str(atom_type).strip() != "-"
    ]
    return cleaned[:expected_len]


def _build_bond_maps(bonds: list[BondParam]) -> tuple[dict[int, BondParam], dict[str, BondParam]]:
    by_row = {bond.ff_row: bond for bond in bonds if bond.ff_row is not None}
    by_env = {bond.env_id: bond for bond in bonds if bond.env_id}
    return by_row, by_env


def _build_angle_maps(angles: list[AngleParam]) -> tuple[dict[int, AngleParam], dict[str, AngleParam]]:
    by_row = {angle.ff_row: angle for angle in angles if angle.ff_row is not None}
    by_env = {angle.env_id: angle for angle in angles if angle.env_id}
    return by_row, by_env


def _match_bond_for_export(
    param, bond_by_row: dict[int, BondParam], bond_by_env: dict[str, BondParam]
) -> BondParam | None:
    if param.ff_row is not None and param.ff_row in bond_by_row:
        return bond_by_row[param.ff_row]
    atom_types = _clean_atom_types(getattr(param, "atom_types", None), 2)
    if len(atom_types) == 2:
        return bond_by_env.get(canonicalize_bond_env_id(atom_types))
    return None


def _match_angle_for_export(
    param,
    angle_by_row: dict[int, AngleParam],
    angle_by_env: dict[str, AngleParam],
) -> AngleParam | None:
    if param.ff_row is not None and param.ff_row in angle_by_row:
        return angle_by_row[param.ff_row]
    atom_types = _clean_atom_types(getattr(param, "atom_types", None), 3)
    if len(atom_types) == 3:
        return angle_by_env.get(canonicalize_angle_env_id(atom_types))
    return None


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
    """A torsion/dihedral force field parameter."""

    elements: tuple[str, str, str, str]
    periodicity: int = 1
    force_constant: float = 0.0  # kcal/mol
    phase: float = 0.0  # degrees
    label: str = ""


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
    source_path: Path | None = field(default=None, repr=False)
    source_format: Literal["mm3_fld", "tinker_prm"] | None = field(default=None, repr=False)

    @property
    def n_params(self) -> int:
        """Number of adjustable scalar parameters in get_param_vector().

        Currently: 2 per bond (k, r0) + 2 per angle (k, theta0).
        Torsions not yet included in the parameter vector.
        """
        return 2 * len(self.bonds) + 2 * len(self.angles)

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

    def get_param_vector(self) -> np.ndarray:
        """Get all adjustable parameters as a flat vector.

        Order: bond force constants, bond equilibria,
               angle force constants, angle equilibria.
        """
        values = []
        for b in self.bonds:
            values.extend([b.force_constant, b.equilibrium])
        for a in self.angles:
            values.extend([a.force_constant, a.equilibrium])
        return np.array(values)

    def set_param_vector(self, vec: np.ndarray):
        """Set parameters from a flat vector (inverse of get_param_vector)."""
        idx = 0
        for b in self.bonds:
            b.force_constant = vec[idx]
            b.equilibrium = vec[idx + 1]
            idx += 2
        for a in self.angles:
            a.force_constant = vec[idx]
            a.equilibrium = vec[idx + 1]
            idx += 2

    def copy(self) -> ForceField:
        """Deep copy."""
        return copy.deepcopy(self)

    # ---- Format converters ----

    @classmethod
    def from_mm3_fld(cls, path: str | Path) -> ForceField:
        """Load from Schrödinger MM3 .fld file.

        Parses bond and angle parameters from the substructure sections,
        extracting element letters from MM3 atom type codes.
        """
        from q2mm.schrod_indep_filetypes import MM3 as MM3Parser

        parser = MM3Parser(str(path))
        parser.import_ff()

        bonds = []
        angles = []

        # Pre-build lookup for equilibrium values by (ptype, ff_row)
        eq_lookup = {}
        for p in parser.params:
            if p.ptype in ("be", "ae"):
                eq_lookup[(p.ptype, p.ff_row)] = p.value

        for param in parser.params:
            # Extract element letters from atom type (e.g., 'C1' -> 'C', ' F' -> 'F')
            atom_types = [t.strip() for t in param.atom_types if t.strip() and t.strip() != "-"]

            if param.ptype == "bf" and len(atom_types) >= 2:
                elems = tuple(_extract_element(t) for t in atom_types[:2])
                env_id = canonicalize_bond_env_id(atom_types[:2])
                eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
                bonds.append(
                    BondParam(
                        elements=elems,
                        equilibrium=eq_val,
                        force_constant=param.value,
                        label=f"MM3 row {param.ff_row}",
                        env_id=env_id,
                        ff_row=param.ff_row,
                    )
                )

            elif param.ptype == "af" and len(atom_types) >= 2:
                # Angle: extract center and outer elements
                if len(atom_types) >= 3:
                    elems = tuple(_extract_element(t) for t in atom_types[:3])
                    env_id = canonicalize_angle_env_id(atom_types[:3])
                else:
                    elems = (_extract_element(atom_types[0]), _extract_element(atom_types[1]), "?")
                    env_id = canonicalize_angle_env_id(atom_types[:2])
                eq_val = eq_lookup.get(("ae", param.ff_row), 0.0)
                angles.append(
                    AngleParam(
                        elements=elems,
                        equilibrium=eq_val,
                        force_constant=param.value,
                        label=f"MM3 row {param.ff_row}",
                        env_id=env_id,
                        ff_row=param.ff_row,
                    )
                )

        return cls(
            name=f"MM3 from {Path(path).name}",
            bonds=bonds,
            angles=angles,
            source_path=Path(path),
            source_format="mm3_fld",
        )

    @classmethod
    def from_tinker_prm(cls, path: str | Path) -> ForceField:
        """Load bond and angle parameters from a Tinker .prm file."""
        from q2mm.datatypes import TinkerFF

        parser = TinkerFF(str(path))
        parser.import_ff()

        bonds = []
        angles = []

        eq_lookup: dict[tuple[str, int], float] = {}
        for param in parser.params:
            if param.ptype == "be" or (param.ptype == "ae" and getattr(param, "ff_col", None) == 2):
                eq_lookup[(param.ptype, param.ff_row)] = param.value

        for param in parser.params:
            atom_types = _clean_atom_types(getattr(param, "atom_types", None), 4)

            if param.ptype == "bf" and len(atom_types) >= 2:
                elems = tuple(_extract_element(t) for t in atom_types[:2])
                env_id = canonicalize_bond_env_id(atom_types[:2])
                eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
                bonds.append(
                    BondParam(
                        elements=elems,
                        equilibrium=eq_val,
                        force_constant=param.value,
                        label=f"Tinker row {param.ff_row}",
                        env_id=env_id,
                        ff_row=param.ff_row,
                    )
                )
            elif param.ptype == "af" and len(atom_types) >= 3:
                elems = tuple(_extract_element(t) for t in atom_types[:3])
                env_id = canonicalize_angle_env_id(atom_types[:3])
                eq_val = eq_lookup.get(("ae", param.ff_row), 0.0)
                angles.append(
                    AngleParam(
                        elements=elems,
                        equilibrium=eq_val,
                        force_constant=param.value,
                        label=f"Tinker row {param.ff_row}",
                        env_id=env_id,
                        ff_row=param.ff_row,
                    )
                )

        return cls(
            name=f"Tinker from {Path(path).name}",
            bonds=bonds,
            angles=angles,
            source_path=Path(path),
            source_format="tinker_prm",
        )

    def to_mm3_fld(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        substructure_name: str = "OPT Generated",
        smiles: str = "AUTO",
    ) -> Path:
        """Write the force field to MM3 .fld format.

        If a template path is provided, or this force field came from
        :meth:`from_mm3_fld`, the existing file is updated in-place via the
        legacy MM3 exporter so comments and unrelated parameters are preserved.

        Otherwise, a minimal bond/angle-only MM3 substructure is generated.
        """
        output_path = Path(path)
        template = Path(template_path) if template_path is not None else None
        if template is None and self.source_format == "mm3_fld" and self.source_path is not None:
            template = self.source_path

        if template is not None:
            from q2mm.datatypes import MM3

            parser = MM3(str(template))
            parser.import_ff()
            updated_params = copy.deepcopy(parser.params)
            bond_by_row, bond_by_env = _build_bond_maps(self.bonds)
            angle_by_row, angle_by_env = _build_angle_maps(self.angles)

            for param in updated_params:
                if param.ptype in ("bf", "be"):
                    bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                    if bond is not None:
                        param.value = bond.force_constant if param.ptype == "bf" else bond.equilibrium
                elif param.ptype in ("af", "ae"):
                    angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                    if angle is not None:
                        param.value = angle.force_constant if param.ptype == "af" else angle.equilibrium

            parser.export_ff(path=str(output_path), params=updated_params, lines=list(parser.lines))
            return output_path

        lines = [f" C  {substructure_name}\n", f" 9  {smiles}\n"]
        for bond in self.bonds:
            lines.append(
                _format_mm3_bond_line(
                    _mm3_atom_types(bond.env_id, bond.elements), bond.equilibrium, bond.force_constant
                )
            )
        for angle in self.angles:
            lines.append(
                _format_mm3_angle_line(
                    _mm3_atom_types(angle.env_id, angle.elements), angle.equilibrium, angle.force_constant
                )
            )
        lines.append("-3\n")
        output_path.write_text("".join(lines), encoding="utf-8")
        return output_path

    def to_tinker_prm(
        self,
        path: str | Path,
        template_path: str | Path | None = None,
        *,
        section_name: str = "OPT Generated",
    ) -> Path:
        """Write the force field to Tinker .prm format.

        If a template path is provided, or this force field came from
        :meth:`from_tinker_prm`, the existing file is updated via the legacy
        exporter. Otherwise, a minimal Q2MM bond/angle section is written.
        """
        output_path = Path(path)
        template = Path(template_path) if template_path is not None else None
        if template is None and self.source_format == "tinker_prm" and self.source_path is not None:
            template = self.source_path

        if template is not None:
            from q2mm.datatypes import TinkerFF

            parser = TinkerFF(str(template))
            parser.import_ff()
            updated_params = copy.deepcopy(parser.params)
            bond_by_row, bond_by_env = _build_bond_maps(self.bonds)
            angle_by_row, angle_by_env = _build_angle_maps(self.angles)

            for param in updated_params:
                if param.ptype in ("bf", "be"):
                    bond = _match_bond_for_export(param, bond_by_row, bond_by_env)
                    if bond is not None:
                        param.value = bond.force_constant if param.ptype == "bf" else bond.equilibrium
                elif param.ptype == "af":
                    angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                    if angle is not None:
                        param.value = angle.force_constant
                elif param.ptype == "ae" and getattr(param, "ff_col", None) == 2:
                    angle = _match_angle_for_export(param, angle_by_row, angle_by_env)
                    if angle is not None:
                        param.value = angle.equilibrium

            parser.export_ff(path=str(output_path), params=updated_params, lines=list(parser.lines))
            return output_path

        lines = ["# Q2MM\n", f"# {section_name}\n"]
        for bond in self.bonds:
            lines.append(
                _format_tinker_bond_line(
                    _tinker_atom_types(bond.env_id, bond.elements), bond.force_constant, bond.equilibrium
                )
            )
        for angle in self.angles:
            lines.append(
                _format_tinker_angle_line(
                    _tinker_atom_types(angle.env_id, angle.elements), angle.force_constant, angle.equilibrium
                )
            )
        output_path.write_text("".join(lines), encoding="utf-8")
        return output_path

    @classmethod
    def create_for_molecule(
        cls, molecule: Q2MMMolecule, default_bond_k: float = 5.0, default_angle_k: float = 0.5, name: str = ""
    ) -> ForceField:
        """Create a force field with default parameters for a molecule.

        Auto-detects unique bond and angle types from the molecule's
        geometry and creates parameters with sensible defaults.
        """
        from q2mm.models.molecule import Q2MMMolecule

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
            f"{len(self.torsions)} torsions)"
        )
