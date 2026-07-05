"""Scalar unit converters for the OpenMM backend.

Thin wrappers over :mod:`q2mm.models.units` that translate canonical Q2MM
parameter values into the units OpenMM expects.  These are implementation
details of :mod:`q2mm.backends.mm.openmm` and are not part of the public API.
"""

from __future__ import annotations

from q2mm.models.units import (
    ang_to_nm,
    canonical_to_openmm_angle_k,
    canonical_to_openmm_bond_k,
    canonical_to_openmm_epsilon,
    canonical_to_openmm_harmonic_angle_k,
    canonical_to_openmm_harmonic_bond_k,
    rmin_half_to_sigma_nm,
)


def _bond_k_to_openmm(force_constant: float) -> float:
    """Convert canonical bond force constant (kcal/mol/Å²) to kJ/mol/Å².

    Args:
        force_constant: Bond force constant in kcal/mol/Å².

    Returns:
        float: Bond force constant in kJ/mol/Å².

    """
    return canonical_to_openmm_bond_k(force_constant)


def _angle_k_to_openmm(force_constant: float) -> float:
    """Convert canonical angle force constant (kcal/mol/rad²) to kJ/mol/rad².

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad².

    """
    return canonical_to_openmm_angle_k(force_constant)


def _bond_k_to_harmonic(force_constant: float) -> float:
    """Canonical bond k (kcal/mol/Å²) → HarmonicBondForce k (kJ/mol/nm²).

    OpenMM's HarmonicBondForce uses E = ½·k·(r−r₀)² while the canonical
    convention is E = k·(r−r₀)², so k_openmm = 2·k.  Additionally convert
    kcal→kJ (×4.184) and Å⁻²→nm⁻² (×100).

    Args:
        force_constant: Bond force constant in kcal/mol/Å².

    Returns:
        float: Bond force constant in kJ/mol/nm² with the ½ convention.

    """
    return canonical_to_openmm_harmonic_bond_k(force_constant)


def _angle_k_to_harmonic(force_constant: float) -> float:
    """Canonical angle k (kcal/mol/rad²) → HarmonicAngleForce k (kJ/mol/rad²).

    OpenMM's HarmonicAngleForce uses E = ½·k·(θ−θ₀)² while the canonical
    convention is E = k·(θ−θ₀)², so k_openmm = 2·k.  Convert kcal→kJ.

    Args:
        force_constant: Angle force constant in kcal/mol/rad².

    Returns:
        float: Angle force constant in kJ/mol/rad² with the ½ convention.

    """
    return canonical_to_openmm_harmonic_angle_k(force_constant)


def _vdw_sigma_nm(radius: float) -> float:
    """Convert Rmin/2 (Å) to LJ sigma (nm) for standard 12-6 NonbondedForce.

    Args:
        radius: Van der Waals radius (Rmin/2) in Å.

    Returns:
        float: LJ sigma in nm.

    """
    return rmin_half_to_sigma_nm(radius)


def _vdw_radius_to_openmm(radius: float) -> float:
    """Convert vdW radius from Å to nm for CustomNonbondedForce.

    Args:
        radius: Van der Waals radius in Å.

    Returns:
        float: Van der Waals radius in nm.

    """
    return ang_to_nm(radius)


def _vdw_epsilon_to_openmm(epsilon: float) -> float:
    """Convert vdW epsilon from kcal/mol to kJ/mol.

    Args:
        epsilon: Well depth in kcal/mol.

    Returns:
        float: Well depth in kJ/mol.

    """
    return canonical_to_openmm_epsilon(epsilon)
