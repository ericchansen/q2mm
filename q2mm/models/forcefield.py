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


@dataclass
class BondParam:
    """A bond force field parameter."""
    elements: tuple[str, str]    # Sorted element pair, e.g., ('C', 'F')
    equilibrium: float           # Angstrom
    force_constant: float        # mdyn/A (MM3 units)
    label: str = ""              # Human-readable label

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
        ff.to_mm3_fld("output.fld")
    """
    name: str = "Q2MM Force Field"
    bonds: list[BondParam] = field(default_factory=list)
    angles: list[AngleParam] = field(default_factory=list)
    torsions: list[TorsionParam] = field(default_factory=list)

    @property
    def n_params(self) -> int:
        return len(self.bonds) + len(self.angles) + len(self.torsions)

    def get_bond(self, elem1: str, elem2: str) -> BondParam | None:
        """Find bond parameter by element pair."""
        key = tuple(sorted([elem1, elem2]))
        for b in self.bonds:
            if b.key == key:
                return b
        return None

    def get_angle(self, elem1: str, elem_center: str, elem2: str) -> AngleParam | None:
        """Find angle parameter by element triple."""
        outer = tuple(sorted([elem1, elem2]))
        key = (outer[0], elem_center, outer[1])
        for a in self.angles:
            if a.key == key:
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

        for param in parser.params:
            # Extract element letters from atom type (e.g., 'C1' -> 'C', ' F' -> 'F')
            atom_types = [t.strip() for t in param.atom_types if t.strip() and t.strip() != '-']

            if param.ptype == "bf" and len(atom_types) >= 2:
                elems = tuple(t[0] if t[0].isalpha() else t for t in atom_types[:2])
                # Find matching be param for equilibrium
                eq_val = 0.0
                for p2 in parser.params:
                    if p2.ptype == "be" and p2.ff_row == param.ff_row:
                        eq_val = p2.value
                        break
                bonds.append(BondParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
                ))

            elif param.ptype == "af" and len(atom_types) >= 2:
                # Angle: extract center and outer elements
                if len(atom_types) >= 3:
                    elems = tuple(t[0] for t in atom_types[:3])
                else:
                    elems = (atom_types[0][0], atom_types[1][0], "?")
                eq_val = 0.0
                for p2 in parser.params:
                    if p2.ptype == "ae" and p2.ff_row == param.ff_row:
                        eq_val = p2.value
                        break
                angles.append(AngleParam(
                    elements=elems,
                    equilibrium=eq_val,
                    force_constant=param.value,
                    label=f"MM3 row {param.ff_row}",
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
