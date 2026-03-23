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
"""

from q2mm.constants import KCAL_TO_KJ, MDYNA_TO_KJMOLA2, MM3_STR

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


def mm3_bond_k_to_canonical(k: float) -> float:
    """Convert MM3 bond force constant (mdyn/Å) to canonical (kcal/mol/Å²)."""
    return k * MDYNA_TO_KCALMOLA2


def canonical_to_mm3_bond_k(k: float) -> float:
    """Convert canonical bond force constant (kcal/mol/Å²) to MM3 (mdyn/Å)."""
    return k * KCALMOLA2_TO_MDYNA


def mm3_angle_k_to_canonical(k: float) -> float:
    """Convert MM3 angle force constant (mdyn·Å/rad²) to canonical (kcal/mol/rad²)."""
    return k * MDYNA_RAD2_TO_KCALMOLRAD2


def canonical_to_mm3_angle_k(k: float) -> float:
    """Convert canonical angle force constant (kcal/mol/rad²) to MM3 (mdyn·Å/rad²)."""
    return k * KCALMOLRAD2_TO_MDYNA_RAD2
