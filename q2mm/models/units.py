"""Canonical unit conversions for force field parameters.

ForceField stores all parameters in a canonical unit system:

    ============  ========================
    Parameter     Canonical unit
    ============  ========================
    bond_k        kcal/(mol·Å²)
    bond_eq       Å
    angle_k       kcal/(mol·rad²)
    angle_eq      degrees
    torsion_k     kcal/mol
    vdw_radius    Å
    vdw_epsilon   kcal/mol
    ============  ========================

Energy convention: ``E = k·(x − x₀)²`` (no ½ factor), matching AMBER.

Loaders convert from format-native units to canonical on load.
Savers convert from canonical back to format-native on save.
Engines convert from canonical to engine-native at the boundary.

NewType wrappers provide zero-runtime-cost type safety: each unit quantity
is a ``NewType`` alias of ``float``, so mypy/pyright can catch mismatched
conversions while CPython sees plain floats with no overhead.
"""

from __future__ import annotations

import math
from typing import NewType

from q2mm.constants import (
    AU_TO_MDYNA,
    AU_TO_MDYN_ANGLE,
    BOHR_TO_ANG,
    HARTREE_TO_KCALMOL,
    HESSIAN_AU_TO_KJMOLA2,
    KCAL_TO_KJ,
    KJMOLA2_TO_HESSIAN_AU,
    KJMOLNM2_TO_HESSIAN_AU,
    MDYNA_TO_KJMOLA2,
    MM3_STR,
    RAD_TO_DEG,  # noqa: F401
)

# ---------------------------------------------------------------------------
# NewType wrappers — zero runtime cost, type-checked by mypy / pyright
# ---------------------------------------------------------------------------

# Energy
KcalPerMol = NewType("KcalPerMol", float)
KJPerMol = NewType("KJPerMol", float)
Hartree = NewType("Hartree", float)

# Force constants
KcalPerMolAngSq = NewType("KcalPerMolAngSq", float)
KcalPerMolRadSq = NewType("KcalPerMolRadSq", float)
KJPerMolAngSq = NewType("KJPerMolAngSq", float)
KJPerMolRadSq = NewType("KJPerMolRadSq", float)
KJPerMolNmSq = NewType("KJPerMolNmSq", float)
HartreePerBohrSq = NewType("HartreePerBohrSq", float)
MdynPerAng = NewType("MdynPerAng", float)
MdynAngPerRadSq = NewType("MdynAngPerRadSq", float)

# Length
Angstrom = NewType("Angstrom", float)
Nanometer = NewType("Nanometer", float)
Bohr = NewType("Bohr", float)

# Angle
Degrees = NewType("Degrees", float)
Radians = NewType("Radians", float)

# ---------------------------------------------------------------------------
# Fundamental conversion factors (module-level, derived from constants.py)
# ---------------------------------------------------------------------------

# --- MM3 bond: mdyn/Å  ↔  kcal/(mol·Å²) ---
# From Allinger's MM3: E_stretch = 71.94·k_s·Δr²·(1 + higher terms)
# where k_s is in mdyn/Å and energy is in kcal/mol.
# Factor = 0.5 * N_A * 1e-1 / 4.184 (accounts for MM3's 2× convention).
MDYNA_TO_KCALMOLA2: float = 0.5 * MDYNA_TO_KJMOLA2 / KCAL_TO_KJ

# --- MM3 angle: mdyn·Å/rad²  ↔  kcal/(mol·rad²) ---
# Uses MM3_STR (601.99) rather than MDYNA_TO_KJMOLA2 (602.21)
# to preserve exact MM3 parity.
MDYNA_RAD2_TO_KCALMOLRAD2: float = 0.5 * MM3_STR / KCAL_TO_KJ

# Inverses
KCALMOLA2_TO_MDYNA: float = 1.0 / MDYNA_TO_KCALMOLA2
KCALMOLRAD2_TO_MDYNA_RAD2: float = 1.0 / MDYNA_RAD2_TO_KCALMOLRAD2

# Hessian: kcal/(mol·Å²) → Hartree/Bohr² (two-step via kJ intermediary)
_KCALMOLA2_TO_HESSIAN_AU: float = KCAL_TO_KJ * KJMOLA2_TO_HESSIAN_AU

# QM → canonical (Seminario path): AU → mdyn → kcal
_AU_BOND_K_TO_CANONICAL: float = AU_TO_MDYNA * MDYNA_TO_KCALMOLA2
_AU_ANGLE_K_TO_CANONICAL: float = AU_TO_MDYN_ANGLE * MDYNA_RAD2_TO_KCALMOLRAD2

# =====================================================================
# MM3 boundary conversions
# =====================================================================


def mm3_bond_k_to_canonical(k: float) -> KcalPerMolAngSq:
    """Convert MM3 bond force constant (mdyn/Å) to canonical (kcal/mol/Å²)."""
    return KcalPerMolAngSq(k * MDYNA_TO_KCALMOLA2)


def canonical_to_mm3_bond_k(k: float) -> MdynPerAng:
    """Convert canonical bond force constant (kcal/mol/Å²) to MM3 (mdyn/Å)."""
    return MdynPerAng(k * KCALMOLA2_TO_MDYNA)


def mm3_angle_k_to_canonical(k: float) -> KcalPerMolRadSq:
    """Convert MM3 angle force constant (mdyn·Å/rad²) to canonical (kcal/mol/rad²)."""
    return KcalPerMolRadSq(k * MDYNA_RAD2_TO_KCALMOLRAD2)


def canonical_to_mm3_angle_k(k: float) -> MdynAngPerRadSq:
    """Convert canonical angle force constant (kcal/mol/rad²) to MM3 (mdyn·Å/rad²)."""
    return MdynAngPerRadSq(k * KCALMOLRAD2_TO_MDYNA_RAD2)


# =====================================================================
# OpenMM boundary conversions
# =====================================================================
# OpenMM uses kJ/mol with nm and radians.
# HarmonicBondForce / HarmonicAngleForce use E = ½·k·(x−x₀)², so
# k_openmm = 2·k_canonical.


def canonical_to_openmm_bond_k(k: float) -> KJPerMolAngSq:
    """Convert canonical bond k (kcal/mol/Å²) → OpenMM custom-force k (kJ/mol/Å²)."""
    return KJPerMolAngSq(float(k) * KCAL_TO_KJ)


def openmm_to_canonical_bond_k(k: float) -> KcalPerMolAngSq:
    """Convert OpenMM custom-force bond k (kJ/mol/Å²) → canonical (kcal/mol/Å²)."""
    return KcalPerMolAngSq(float(k) / KCAL_TO_KJ)


def canonical_to_openmm_angle_k(k: float) -> KJPerMolRadSq:
    """Convert canonical angle k (kcal/mol/rad²) → OpenMM custom-force k (kJ/mol/rad²)."""
    return KJPerMolRadSq(float(k) * KCAL_TO_KJ)


def openmm_to_canonical_angle_k(k: float) -> KcalPerMolRadSq:
    """Convert OpenMM custom-force angle k (kJ/mol/rad²) → canonical (kcal/mol/rad²)."""
    return KcalPerMolRadSq(float(k) / KCAL_TO_KJ)


def canonical_to_openmm_harmonic_bond_k(k: float) -> KJPerMolNmSq:
    """Convert canonical bond k (kcal/mol/Å²) → HarmonicBondForce k (kJ/mol/nm²).

    Accounts for E=k(x−x₀)² → E=½k(x−x₀)² convention and Å→nm.
    """
    return KJPerMolNmSq(2.0 * float(k) * KCAL_TO_KJ * 100.0)


def openmm_to_canonical_harmonic_bond_k(k: float) -> KcalPerMolAngSq:
    """Convert HarmonicBondForce k (kJ/mol/nm²) → canonical bond k (kcal/mol/Å²)."""
    return KcalPerMolAngSq(float(k) / (2.0 * KCAL_TO_KJ * 100.0))


def canonical_to_openmm_harmonic_angle_k(k: float) -> KJPerMolRadSq:
    """Convert canonical angle k (kcal/mol/rad²) → HarmonicAngleForce k (kJ/mol/rad²).

    Accounts for E=k(x−x₀)² → E=½k(x−x₀)² convention.
    """
    return KJPerMolRadSq(2.0 * float(k) * KCAL_TO_KJ)


def openmm_to_canonical_harmonic_angle_k(k: float) -> KcalPerMolRadSq:
    """Convert HarmonicAngleForce k (kJ/mol/rad²) → canonical angle k (kcal/mol/rad²)."""
    return KcalPerMolRadSq(float(k) / (2.0 * KCAL_TO_KJ))


def canonical_to_openmm_torsion_k(k: float) -> KJPerMol:
    """Convert canonical torsion k (kcal/mol) → OpenMM torsion k (kJ/mol)."""
    return KJPerMol(float(k) * KCAL_TO_KJ)


def openmm_to_canonical_torsion_k(k: float) -> KcalPerMol:
    """Convert OpenMM torsion k (kJ/mol) → canonical torsion k (kcal/mol)."""
    return KcalPerMol(float(k) / KCAL_TO_KJ)


def canonical_to_openmm_epsilon(eps: float) -> KJPerMol:
    """Convert canonical vdW epsilon (kcal/mol) → OpenMM epsilon (kJ/mol)."""
    return KJPerMol(float(eps) * KCAL_TO_KJ)


def openmm_to_canonical_epsilon(eps: float) -> KcalPerMol:
    """Convert OpenMM vdW epsilon (kJ/mol) → canonical epsilon (kcal/mol)."""
    return KcalPerMol(float(eps) / KCAL_TO_KJ)


def ang_to_nm(length: float) -> Nanometer:
    """Convert Angstroms to nanometers."""
    return Nanometer(float(length) * 0.1)


def nm_to_ang(length: float) -> Angstrom:
    """Convert nanometers to Angstroms."""
    return Angstrom(float(length) * 10.0)


def rmin_half_to_sigma_nm(radius: float) -> Nanometer:
    """Convert Rmin/2 (Å) to LJ sigma (nm) for standard 12-6 NonbondedForce."""
    return Nanometer(float(radius) * 2.0 / (2.0 ** (1.0 / 6.0)) * 0.1)


def rmin_half_to_sigma(radius: float) -> Angstrom:
    """Convert Rmin/2 (Å) to LJ sigma (Å) for 12-6 LJ potential."""
    return Angstrom(float(radius) * 2.0 / (2.0 ** (1.0 / 6.0)))


def deg_to_rad(angle: float) -> Radians:
    """Convert degrees to radians."""
    return Radians(math.radians(float(angle)))


def rad_to_deg(angle: float) -> Degrees:
    """Convert radians to degrees."""
    return Degrees(math.degrees(float(angle)))


# =====================================================================
# Tinker / MM3 boundary (aliases for clarity at call sites)
# =====================================================================
# Tinker uses MM3 parameter units, so these are the same as the MM3
# conversions above.  Explicit aliases improve readability at call sites.

canonical_to_tinker_bond_k = canonical_to_mm3_bond_k
tinker_to_canonical_bond_k = mm3_bond_k_to_canonical
canonical_to_tinker_angle_k = canonical_to_mm3_angle_k
tinker_to_canonical_angle_k = mm3_angle_k_to_canonical


# =====================================================================
# QM boundary conversions (Hartree/Bohr → canonical)
# =====================================================================


def qm_to_canonical_bond_k(k: float) -> KcalPerMolAngSq:
    """Convert QM bond force constant (Hartree/Bohr²) → canonical (kcal/mol/Å²).

    Uses the Seminario two-step path: AU → mdyn/Å → kcal/mol/Å².
    """
    return KcalPerMolAngSq(float(k) * _AU_BOND_K_TO_CANONICAL)


def canonical_to_qm_bond_k(k: float) -> HartreePerBohrSq:
    """Convert canonical bond k (kcal/mol/Å²) → QM (Hartree/Bohr²)."""
    return HartreePerBohrSq(float(k) / _AU_BOND_K_TO_CANONICAL)


def qm_to_canonical_angle_k(k: float) -> KcalPerMolRadSq:
    """Convert QM angle force constant (Hartree/rad²) → canonical (kcal/mol/rad²).

    Uses the Seminario two-step path: AU → mdyn·Å/rad² → kcal/mol/rad².
    """
    return KcalPerMolRadSq(float(k) * _AU_ANGLE_K_TO_CANONICAL)


def canonical_to_qm_angle_k(k: float) -> float:
    """Convert canonical angle k (kcal/mol/rad²) → QM (Hartree/rad²)."""
    return float(k) / _AU_ANGLE_K_TO_CANONICAL


def qm_to_canonical_energy(e: float) -> KcalPerMol:
    """Convert QM energy (Hartree) → canonical energy (kcal/mol)."""
    return KcalPerMol(float(e) * HARTREE_TO_KCALMOL)


def canonical_to_qm_energy(e: float) -> Hartree:
    """Convert canonical energy (kcal/mol) → QM energy (Hartree)."""
    return Hartree(float(e) / HARTREE_TO_KCALMOL)


def bohr_to_ang(length: float) -> Angstrom:
    """Convert Bohr to Angstrom."""
    return Angstrom(float(length) * BOHR_TO_ANG)


def ang_to_bohr(length: float) -> Bohr:
    """Convert Angstrom to Bohr."""
    return Bohr(float(length) / BOHR_TO_ANG)


# =====================================================================
# Hessian boundary conversions
# =====================================================================


def hessian_kcalmola2_to_au(k: float) -> HartreePerBohrSq:
    """Convert Hessian element kcal/(mol·Å²) → Hartree/Bohr²."""
    return HartreePerBohrSq(float(k) * _KCALMOLA2_TO_HESSIAN_AU)


def hessian_au_to_kcalmola2(k: float) -> KcalPerMolAngSq:
    """Convert Hessian element Hartree/Bohr² → kcal/(mol·Å²)."""
    return KcalPerMolAngSq(float(k) / _KCALMOLA2_TO_HESSIAN_AU)


def hessian_kjmolnm2_to_au(k: float) -> HartreePerBohrSq:
    """Convert Hessian element kJ/(mol·nm²) → Hartree/Bohr²."""
    return HartreePerBohrSq(float(k) * KJMOLNM2_TO_HESSIAN_AU)


def hessian_au_to_kjmolnm2(k: float) -> KJPerMolNmSq:
    """Convert Hessian element Hartree/Bohr² → kJ/(mol·nm²)."""
    return KJPerMolNmSq(float(k) / KJMOLNM2_TO_HESSIAN_AU)


def hessian_au_to_kjmola2(k: float) -> KJPerMolAngSq:
    """Convert Hessian element Hartree/Bohr² → kJ/(mol·Å²)."""
    return KJPerMolAngSq(float(k) * HESSIAN_AU_TO_KJMOLA2)


def hessian_kjmola2_to_au(k: float) -> HartreePerBohrSq:
    """Convert Hessian element kJ/(mol·Å²) → Hartree/Bohr²."""
    return HartreePerBohrSq(float(k) / HESSIAN_AU_TO_KJMOLA2)
