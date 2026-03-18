"""Clean, format-agnostic force field representation for Q2MM.

Decouples Q2MM's optimization from specific file formats (MM3 .fld,
Tinker .prm, AMBER .frcmod). Parameters are identified by element
pairs/triples, not format-specific atom type strings or line numbers.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np


def _extract_element(atom_type: str) -> str:
    """Extract element symbol from MM3 atom type (e.g., 'Cl1' -> 'Cl', 'C3' -> 'C')."""
    s = atom_type.strip()
    if len(s) >= 2 and s[0].isupper() and s[1].islower():
        return s[:2]
    return s[0] if s and s[0].isalpha() else s


@dataclass
class BondParam:
    """A bond force field parameter."""
    elements: tuple[str, str]    # Sorted element pair, e.g., ('C', 'F')
    equilibrium: float           # Angstrom
    force_constant: float        # mdyn/A (MM3 units)
    label: str = ""              # Human-readable label
    env_id: str = ""             # Environment ID for disambiguating same-element params
                                 # (e.g., MM3 ff_row, atom type codes 'C1-F1' vs 'C2-F1')

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted(self.elements))


@dataclass
class AngleParam:
    """An angle force field parameter."""
    elements: tuple[str, str, str]  # (outer, center, outer)
    equilibrium: float              # degrees
    force_constant: float           # mdyn*A/rad^2
    label: str = ""
    env_id: str = ""             # Environment ID for disambiguating same-element params

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
    phase: float = 0.0           # degrees
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
            atom_types = [t.strip() for t in param.atom_types if t.strip() and t.strip() != '-']

            if param.ptype == "bf" and len(atom_types) >= 2:
                elems = tuple(_extract_element(t) for t in atom_types[:2])
                env_id = "-".join(atom_types[:2])
                eq_val = eq_lookup.get(("be", param.ff_row), 0.0)
                bonds.append(BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                ))

            elif param.ptype == "af" and len(atom_types) >= 2:
                # Angle: extract center and outer elements
                if len(atom_types) >= 3:
                    elems = tuple(_extract_element(t) for t in atom_types[:3])
                    env_id = "-".join(atom_types[:3])
                else:
                    elems = (_extract_element(atom_types[0]), _extract_element(atom_types[1]), "?")
                    env_id = "-".join(atom_types[:2])
                eq_val = eq_lookup.get(("ae", param.ff_row), 0.0)
                angles.append(AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
                    env_id=env_id,
                ))

        return cls(
            name=f"MM3 from {Path(path).name}",
            bonds=bonds,
            angles=angles,
        )

    @classmethod
    def create_for_molecule(cls, molecule: Q2MMMolecule,
                            default_bond_k: float = 5.0,
                            default_angle_k: float = 0.5,
                            name: str = "") -> ForceField:
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
            bonds.append(BondParam(
                elements=key,
                equilibrium=avg_len,
                force_constant=default_bond_k,
                label=f"{key[0]}-{key[1]} (auto)",
            ))

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
            angles.append(AngleParam(
                elements=key,
                equilibrium=avg_val,
                force_constant=default_angle_k,
                label=f"{key[0]}-{key[1]}-{key[2]} (auto)",
            ))

        return cls(
            name=name or f"Auto FF for {molecule.name}",
            bonds=bonds,
            angles=angles,
        )

    def __repr__(self) -> str:
        return (f"ForceField('{self.name}', "
                f"{len(self.bonds)} bonds, {len(self.angles)} angles, "
                f"{len(self.torsions)} torsions)")
