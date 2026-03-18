"""Shared regex patterns and atom type conversion utilities for Q2MM parsers."""

from __future__ import annotations

import re

# Match any float in a string.
RE_FLOAT = r"[+-]?\s*(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# MM3.FLD RELATED
# Match SMARTS notation used by MM3* substructures.
RE_SMILES = r"[\w\-\=\(\)\.\+\[\]\*]+"
# Possible symbols used to split atoms in SMARTS notation.
RE_SPLIT_ATOMS = r"[\s\-\(\)\=\.\[\]\*]+"
# Name of MM3* substructures.
RE_SUB = r"[\w\s\-\.\*\(\)\%\=\,]+"

# .MMO RELATED
# Match bonds in lines of a .mmo file.
RE_BOND = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+({RE_FLOAT})\s+{RE_FLOAT}\s+\w+"
    rf"\s+\d+\s+({RE_SUB})\s+(\d+)"
)
# Match angles in lines of a .mmo file.
RE_ANGLE = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+{RE_FLOAT}\s+"
    rf"({RE_FLOAT})\s+{RE_FLOAT}\s+{RE_FLOAT}\s+\w+\s+\d+\s+({RE_SUB})\s+(\d+)"
)
# Match torsions in lines of a .mmo file.
RE_TORSION = re.compile(
    rf"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+{RE_FLOAT}\s+{RE_FLOAT}\s+{RE_FLOAT}\s+"
    rf"({RE_FLOAT})\s+{RE_FLOAT}\s+\w+\s+\d+({RE_SUB})\s+(\d+)"
)

# Match the filename and atoms for a torsion label.
RE_T_LBL = re.compile(r"\At_(\S+)_\d+_(\d+-\d+-\d+-\d+)")


def convert_atom_type(atom_type: str) -> str:
    """Normalize atom type string to uppercase alphanumeric."""
    return "".join(c for c in atom_type if c.isalnum()).upper()


def convert_atom_type_pair(atom_type_pair):
    """Convert a pair of atom types."""
    return tuple(convert_atom_type(at) for at in atom_type_pair)


def convert_atom_types(atom_type_pairs: list) -> list:
    """Convert multiple atom type pairs."""
    return [convert_atom_type_pair(pair) for pair in atom_type_pairs]
