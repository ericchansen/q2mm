"""Shared test constants and molecule factories.

Centralises path definitions and molecule helpers so every test module
can ``from test._shared import …`` instead of redefining them locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SN2_QM_REF = EXAMPLES_DIR / "sn2-test" / "qm-reference"
ETHANE_DIR = EXAMPLES_DIR / "ethane"

# SN2 test data paths
SN2_XYZ = SN2_QM_REF / "sn2-ts-optimized.xyz"
SN2_HESSIAN = SN2_QM_REF / "sn2-ts-hessian.npy"
SN2_FREQS = SN2_QM_REF / "sn2-ts-frequencies.txt"
SN2_ENERGY = SN2_QM_REF / "sn2-ts-energy.txt"

# CH3F test data paths
CH3F_XYZ = SN2_QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = SN2_QM_REF / "ch3f-hessian.npy"
CH3F_FREQS = SN2_QM_REF / "ch3f-frequencies.txt"
CH3F_ENERGY = SN2_QM_REF / "ch3f-energy.txt"
CH3F_MODES = SN2_QM_REF / "ch3f-normal-modes.npz"

# Complex
COMPLEX_XYZ = SN2_QM_REF / "complex-optimized.xyz"

# Ethane
GS_FCHK = ETHANE_DIR / "GS.fchk"
TS_FCHK = ETHANE_DIR / "TS.fchk"

# Data availability flags
SN2_DATA_AVAILABLE = SN2_XYZ.exists() and SN2_HESSIAN.exists()
CH3F_DATA_AVAILABLE = CH3F_XYZ.exists() and CH3F_HESS.exists()


# ---------------------------------------------------------------------------
# Molecule factories
# ---------------------------------------------------------------------------


def make_diatomic(
    distance: float = 0.74,
    bond_tolerance: float = 2.0,
) -> Q2MMMolecule:
    """H2 molecule at specified bond distance."""
    from q2mm.models.molecule import Q2MMMolecule

    return Q2MMMolecule(
        symbols=["H", "H"],
        geometry=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        name="H2",
        bond_tolerance=bond_tolerance,
    )


def make_water(
    angle_deg: float = 104.5,
    bond_length: float = 0.96,
    bond_tolerance: float = 1.5,
    name: str = "water",
) -> Q2MMMolecule:
    """Water molecule at specified geometry."""
    from q2mm.models.molecule import Q2MMMolecule

    theta = np.deg2rad(angle_deg)
    return Q2MMMolecule(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [bond_length, 0.0, 0.0],
                [bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0],
            ]
        ),
        name=name,
        bond_tolerance=bond_tolerance,
    )


def make_noble_gas_pair(
    distance: float = 3.0,
    atom_type: str = "He",
    bond_tolerance: float = 0.5,
) -> Q2MMMolecule:
    """Two noble gas atoms for vdW testing (no bonds)."""
    from q2mm.models.molecule import Q2MMMolecule

    return Q2MMMolecule(
        symbols=["He", "He"],
        atom_types=[atom_type, atom_type],
        geometry=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        name="He2",
        bond_tolerance=bond_tolerance,
    )
