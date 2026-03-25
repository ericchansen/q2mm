"""Canonical geometry calculations for bond lengths, angles, and dihedrals.

All geometry math in q2mm should delegate to this module to avoid
duplicating formulas across parsers, models, and evaluators.
"""

from __future__ import annotations

import numpy as np


def bond_length(p0: np.ndarray, p1: np.ndarray) -> float:
    """Compute bond length between two points.

    Args:
        p0: Coordinates of the first atom ``[x, y, z]``.
        p1: Coordinates of the second atom ``[x, y, z]``.

    Returns:
        Distance between *p0* and *p1* in the same units as the input.

    """
    v = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    return float(np.sqrt(v.dot(v)))


def bond_angle(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float:
    """Compute bond angle in degrees for three points.

    The angle is measured at the central atom *p1*.

    Args:
        p0: Coordinates of the first atom ``[x, y, z]``.
        p1: Coordinates of the central atom ``[x, y, z]``.
        p2: Coordinates of the third atom ``[x, y, z]``.

    Returns:
        Angle in degrees in the range [0, 180].

    """
    v21 = np.asarray(p0, dtype=float) - np.asarray(p1, dtype=float)
    v23 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    cos_angle = np.dot(v21, v23) / (np.sqrt(v21.dot(v21)) * np.sqrt(v23.dot(v23)))
    # Clamp to [-1, 1] to guard against floating-point overshoot.
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def dihedral_angle(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """Compute signed dihedral angle in degrees for four points.

    Uses the ``atan2`` formulation that returns values in [-180, 180].

    Args:
        p0: Coordinates of the first atom.
        p1: Coordinates of the second atom.
        p2: Coordinates of the third atom.
        p3: Coordinates of the fourth atom.

    Returns:
        Dihedral angle in degrees, in the range [-180, 180].

    """
    b1 = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    b2 = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    b3 = np.asarray(p3, dtype=float) - np.asarray(p2, dtype=float)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    b2_norm = np.linalg.norm(b2)
    if n1_norm < 1e-10 or n2_norm < 1e-10 or b2_norm < 1e-10:
        return 0.0
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    m1 = np.cross(n1, b2 / b2_norm)
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))
