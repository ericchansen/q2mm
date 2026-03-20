#!/usr/bin/env python3
"""
Contains basic utility methods for use in Q2MM.
"""

import copy
import numpy as np

try:
    import parmed
except ImportError:
    parmed = None


# region Atom Type Handling


def convert_atom_type(atom_type: str) -> str:
    """_summary_

    Args:
        atom_type (str): _description_

    Returns:
        str: _description_
    """
    q2mm_atom_type = "".join(filter(str.isalnum, atom_type))
    q2mm_atom_type = q2mm_atom_type.upper()
    # TODO: MF Add a check to verify it is included in atom.typ here,
    # exception should be caught, propagated, and handled here to avoid
    # silent failure within MacroModel upon FF export (or other silent or loud failures).
    return q2mm_atom_type


def convert_atom_type_pair(atom_type_pair):
    q2mm_atom_type_pair = [convert_atom_type(atom_type) for atom_type in atom_type_pair]
    return q2mm_atom_type_pair


def convert_atom_types(atom_type_pairs: list) -> list:
    q2mm_atom_type_pairs = [convert_atom_type_pair(atom_type_pair) for atom_type_pair in atom_type_pairs]
    return q2mm_atom_type_pairs


def measure_bond(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Returns bond length between 2 sets of coordinates.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]

    Returns:
        float: measured bond length
    """
    vector = coords2 - coords1
    return np.sqrt(vector.dot(vector))  # Used over np.linalg.norm due to speed advantage


def measure_angle(coords1: np.ndarray, coords2: np.ndarray, coords3: np.ndarray) -> float:
    """Returns angle between 3 sets of coordinates in degrees.

    Args:
        coords1 (np.ndarray): atom1 coordinates [x, y, z]
        coords2 (np.ndarray): atom2 coordinates [x, y, z]
        coords3 (np.ndarray): atom3 coordinates [x, y, z]

    Returns:
        float: Angle between coords1, coords2, coords3 in degrees
    """
    vector21 = coords1 - coords2
    vector23 = coords3 - coords2
    cos_angle = np.dot(vector21, vector23) / (np.sqrt(vector21.dot(vector21)) * np.sqrt(vector23.dot(vector23)))
    angle = np.arccos(cos_angle)
    return np.degrees(angle)


def is_same_type_DOF(atom_types1: list, atom_types2: list) -> bool:
    reverse_1 = copy.deepcopy(atom_types1)
    reverse_1.reverse()
    return atom_types1 == atom_types2 or reverse_1 == atom_types2


# endregion Atom Type Handling
