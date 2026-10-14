"""Shared test constants, molecule factories, and synthetic evaluator support.

Centralises path definitions and molecule helpers so every test module
can ``from test._shared import …`` instead of redefining them locally.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from q2mm.resources import sn2_reference_dir
from q2mm.models.forcefield import ForceField, FunctionalForm, TorsionParam
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode

if TYPE_CHECKING:
    from q2mm.models.molecule import Molecule

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
SN2_QM_REF = sn2_reference_dir()
ETHANE_DIR = REPO_ROOT / "test" / "fixtures" / "ethane"

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
) -> Molecule:
    """H2 molecule at specified bond distance."""
    from q2mm.models.molecule import Molecule

    return Molecule(
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
) -> Molecule:
    """Water molecule at specified geometry."""
    from q2mm.models.molecule import Molecule

    theta = np.deg2rad(angle_deg)
    return Molecule(
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


def make_harmonic_diatomic() -> Molecule:
    """H2 with an analytic bond-stretch Hessian and five rigid modes."""
    molecule = make_diatomic()
    stretch = np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    return molecule.with_hessian(0.5 * np.outer(stretch, stretch))


def make_harmonic_water(*, angle_stiffness: float = 0.1) -> Molecule:
    """Water geometry with analytic bond/angle curvature, not ab-initio data."""
    from q2mm.constants import BOHR_TO_ANG

    molecule = make_water()
    xyz = molecule.geometry / BOHR_TO_ANG
    rows = []
    for terminal in (1, 2):
        displacement = xyz[terminal] - xyz[0]
        direction = displacement / np.linalg.norm(displacement)
        row = np.zeros_like(xyz)
        row[terminal] = direction
        row[0] = -direction
        rows.append(row.ravel())
    u, v = xyz[1] - xyz[0], xyz[2] - xyz[0]
    ru, rv = np.linalg.norm(u), np.linalg.norm(v)
    u, v = u / ru, v / rv
    cosine = np.dot(u, v)
    sine = np.sqrt(1.0 - cosine**2)
    angle = np.zeros_like(xyz)
    angle[1] = (cosine * u - v) / (ru * sine)
    angle[2] = (cosine * v - u) / (rv * sine)
    angle[0] = -angle[1] - angle[2]
    rows.append(angle.ravel())
    jacobian = np.asarray(rows)
    return molecule.with_hessian(jacobian.T @ np.diag([0.5, 0.5, angle_stiffness]) @ jacobian)


def make_noble_gas_pair(
    distance: float = 3.0,
    atom_type: str = "He",
    bond_tolerance: float = 0.5,
) -> Molecule:
    """Two noble gas atoms for vdW testing (no bonds)."""
    from q2mm.models.molecule import Molecule

    return Molecule(
        symbols=["He", "He"],
        atom_types=[atom_type, atom_type],
        geometry=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        name="He2",
        bond_tolerance=bond_tolerance,
    )


def make_ethane() -> Molecule:
    """Staggered ethane (C₂H₆) for torsion testing.

    C-C along x-axis, tetrahedral H arrangement, staggered by 60°.
    Has 9 H-C-C-H torsions with ≈60° or 180° dihedral angles.
    """
    from q2mm.models.molecule import Molecule

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
    return Molecule(
        symbols=["C", "C", "H", "H", "H", "H", "H", "H"],
        geometry=np.array([c1, c2, h1, h2, h3, h4, h5, h6]),
        name="ethane",
    )


# ---------------------------------------------------------------------------
# Engine -> functional_form mapping
# ---------------------------------------------------------------------------


def backend_functional_form(backend: object) -> FunctionalForm:
    """Pick the :class:`~q2mm.models.forcefield.FunctionalForm` *backend* actually supports.

    Several benchmark/optimizer tests parametrize the exact same helper
    function or fixture across multiple backend types (OpenMM/Tinker are
    MM3-only-or-MM3-capable; JAX/JAX-MD are harmonic-only-or-harmonic-
    preferred).  Prefers harmonic when a backend supports both.
    """
    from q2mm.models.forcefield import FunctionalForm

    supported = backend.info.functional_forms  # type: ignore[attr-defined]
    if "harmonic" in supported:
        return FunctionalForm.HARMONIC
    if "mm3" in supported:
        return FunctionalForm.MM3
    for name in sorted(supported):
        if hasattr(FunctionalForm, name.upper()):
            return getattr(FunctionalForm, name.upper())  # type: ignore[no-any-return]
    raise RuntimeError(f"Backend {backend!r} reports no mappable functional forms: {supported!r}")


def _synthetic_plan(
    initial: np.ndarray,
    bounds: list[tuple[float, float]] | None = None,
    active_indices: np.ndarray | None = None,
) -> tuple[ForceField, ActiveParameterSpace, ObjectivePlan]:
    """Build a core-only synthetic torsion space and empty observation plan."""
    baseline = np.asarray(initial, dtype=np.float64)
    ff = ForceField(
        name="synthetic",
        torsions=tuple(
            TorsionParam(
                elements=("C", "C", "C", "C"),
                periodicity=i + 1,
                force_constant=float(value),
                env_id=f"p{i}",
            )
            for i, value in enumerate(baseline)
        ),
        functional_form=FunctionalForm.HARMONIC,
    )
    layout = ParameterLayout.from_force_field(ff)
    if bounds is not None:
        layout = ParameterLayout(
            slots=tuple(
                replace(slot, bounds=(float(lo), float(hi)))
                for slot, (lo, hi) in zip(layout.slots, bounds, strict=True)
            )
        )
    space = ActiveParameterSpace(
        layout=layout,
        baseline=baseline,
        active_indices=np.arange(baseline.size, dtype=int) if active_indices is None else active_indices,
    )
    plan = ObjectivePlan(
        case_ids=("0",),
        molecules=(make_diatomic(),),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=ObservationSet(),
        layout=layout,
        active_space=space,
    )
    return ff, space, plan


class QuadraticEvaluator(BaseObjectiveExecutor):
    """Synthetic quadratic evaluator with analytical full-vector gradients."""

    def __init__(
        self,
        target: np.ndarray,
        bounds: list[tuple[float, float]] | None = None,
        initial: np.ndarray | None = None,
        active_indices: np.ndarray | None = None,
    ) -> None:
        self.target = np.asarray(target, dtype=np.float64)
        baseline = (
            np.zeros_like(self.target, dtype=np.float64) if initial is None else np.asarray(initial, dtype=np.float64)
        )
        self.forcefield, self.space, plan = _synthetic_plan(baseline, bounds, active_indices)
        super().__init__(plan)
        self._gradient_mode = GradientMode.ANALYTICAL

    @property
    def gradient_mode(self) -> GradientMode:
        return self._gradient_mode

    def _total(self, full_vector: np.ndarray) -> float:
        return float(np.sum((np.asarray(full_vector, dtype=np.float64) - self.target) ** 2))

    def _calculated(self, full_vector: np.ndarray) -> np.ndarray:
        return np.zeros(0, dtype=np.float64)

    def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
        return 2.0 * (np.asarray(full_vector, dtype=np.float64) - self.target)
