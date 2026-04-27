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

# External validation data (gitignored, ~1.9 GB).
# Set Q2MM_SUPPORTING_INFO to override the default location.
import os as _os

SUPPORTING_INFO_DIR: Path | None = None
_si_env = _os.environ.get("Q2MM_SUPPORTING_INFO")
if _si_env:
    _si_candidate = Path(_si_env)
else:
    _si_candidate = REPO_ROOT / "validation" / "supporting-info"
if _si_candidate.is_dir():
    SUPPORTING_INFO_DIR = _si_candidate

# Ethane
GS_FCHK = ETHANE_DIR / "GS.fchk"
TS_FCHK = ETHANE_DIR / "TS.fchk"

# In-repo fixture availability — assert at import time per AGENTS.md §2.
# These files are tracked in the repo; if any is missing, the working copy is
# corrupt and tests should fail loudly at collection rather than silently skip.
_REQUIRED_FIXTURES = {
    "SN2_XYZ": SN2_XYZ,
    "SN2_HESSIAN": SN2_HESSIAN,
    "SN2_FREQS": SN2_FREQS,
    "SN2_ENERGY": SN2_ENERGY,
    "CH3F_XYZ": CH3F_XYZ,
    "CH3F_HESS": CH3F_HESS,
    "CH3F_FREQS": CH3F_FREQS,
    "CH3F_ENERGY": CH3F_ENERGY,
    "CH3F_MODES": CH3F_MODES,
    "COMPLEX_XYZ": COMPLEX_XYZ,
    "GS_FCHK": GS_FCHK,
    "TS_FCHK": TS_FCHK,
}
_missing = sorted(name for name, p in _REQUIRED_FIXTURES.items() if not p.exists())
if _missing:
    raise RuntimeError(
        "Missing in-repo test fixtures (working copy corrupt?): "
        + ", ".join(f"{n}={_REQUIRED_FIXTURES[n]}" for n in _missing)
    )


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


def make_ethane() -> Q2MMMolecule:
    """Staggered ethane (C₂H₆) for torsion testing.

    C-C along x-axis, tetrahedral H arrangement, staggered by 60°.
    Has 9 H-C-C-H torsions with ≈60° or 180° dihedral angles.
    """
    from q2mm.models.molecule import Q2MMMolecule

    r_cc = 1.54
    r_ch = 1.09
    theta = np.radians(109.5)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([r_cc, 0.0, 0.0])
    h1 = c1 + r_ch * np.array([cos_t, sin_t, 0.0])
    h2 = c1 + r_ch * np.array([cos_t, sin_t * np.cos(2 * np.pi / 3), sin_t * np.sin(2 * np.pi / 3)])
    h3 = c1 + r_ch * np.array([cos_t, sin_t * np.cos(4 * np.pi / 3), sin_t * np.sin(4 * np.pi / 3)])
    h4 = c2 + r_ch * np.array([-cos_t, sin_t * np.cos(np.pi / 3), sin_t * np.sin(np.pi / 3)])
    h5 = c2 + r_ch * np.array([-cos_t, sin_t * np.cos(np.pi), sin_t * np.sin(np.pi)])
    h6 = c2 + r_ch * np.array([-cos_t, sin_t * np.cos(5 * np.pi / 3), sin_t * np.sin(5 * np.pi / 3)])
    return Q2MMMolecule(
        symbols=["C", "C", "H", "H", "H", "H", "H", "H"],
        geometry=np.array([c1, c2, h1, h2, h3, h4, h5, h6]),
        name="ethane",
    )
