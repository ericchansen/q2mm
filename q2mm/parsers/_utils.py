#!/usr/bin/env python3
"""Basic utility methods for use in Q2MM."""

import numpy as np


# region Atom Type Handling


def convert_atom_type(atom_type: str) -> str:
    """Convert a raw atom type string to a normalized Q2MM atom type.

    Strips non-alphanumeric characters and converts to uppercase.

    Args:
        atom_type (str): Raw atom type string from a force field file.

    Returns:
        (str): Normalized uppercase alphanumeric atom type.

    """
    q2mm_atom_type = "".join(filter(str.isalnum, atom_type))
    q2mm_atom_type = q2mm_atom_type.upper()
    # TODO: MF Add a check to verify it is included in atom.typ here,
    # exception should be caught, propagated, and handled here to avoid
    # silent failure within MacroModel upon FF export (or other silent or loud failures).
    return q2mm_atom_type


def convert_atom_type_pair(atom_type_pair: list[str]) -> list[str]:
    """Convert a pair of raw atom type strings to normalized Q2MM atom types.

    Args:
        atom_type_pair (list[str]): A pair of raw atom type strings.

    Returns:
        (list[str]): A list of normalized uppercase alphanumeric atom types.

    """
    q2mm_atom_type_pair = [convert_atom_type(atom_type) for atom_type in atom_type_pair]
    return q2mm_atom_type_pair


def measure_bond(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Return bond length between 2 sets of coordinates.

    Delegates to :func:`q2mm.geometry.bond_length`.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]

    Returns:
        (float): measured bond length

    """
    from q2mm.geometry import bond_length

    return bond_length(coords1, coords2)


def measure_angle(coords1: np.ndarray, coords2: np.ndarray, coords3: np.ndarray) -> float:
    """Return angle between 3 sets of coordinates in degrees.

    Delegates to :func:`q2mm.geometry.bond_angle`.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]
        coords3 (np.ndarray): atom3 coordinates [x, y, z]

    Returns:
        (float): Angle between coords1, coords2, coords3 in degrees

    """
    from q2mm.geometry import bond_angle

    return bond_angle(coords1, coords2, coords3)


# endregion Atom Type Handling
