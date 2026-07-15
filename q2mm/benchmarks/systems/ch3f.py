"""CH3F: single ground-state molecule, QFUERZA-fresh force field.

Frequency-only benchmark against a fresh QFUERZA-derived force field —
no published OPT block to start from (see
:func:`~q2mm.benchmarks.systems._forcefield.load_qfuerza_fresh`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.systems._assembly import assemble_qfuerza_fresh_case
from q2mm.benchmarks.systems._paths import ExternalDataRoots, StartingPoint
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule
from q2mm.models.problem import StationaryPointKind

KEY = "ch3f"
NAME = "CH3F"
DESCRIPTION = "Single CH3F molecule (SN2 test, B3LYP/6-31+G(d))"
DEFAULT_FORMS: tuple[str, ...] = ("harmonic", "mm3")
METADATA = {"level_of_theory": "B3LYP/6-31+G(d)"}


def _find_data_dir() -> Path:
    """Return the installed CH3F/SN2 package-resource directory."""
    from q2mm.resources import sn2_reference_dir

    data_dir = sn2_reference_dir()
    if not (data_dir / "ch3f-optimized.xyz").is_file():
        raise FileNotFoundError(f"Packaged CH3F reference data is incomplete: {data_dir}")
    return data_dir


def load_molecule(*, data_dir: Path | None = None) -> Molecule:
    """Load the single CH3F molecule + its QM Hessian (B3LYP/6-31+G(d))."""
    from q2mm.io.xyz import load_xyz

    qm_dir = data_dir or _find_data_dir()
    xyz = qm_dir / "ch3f-optimized.xyz"
    hess_path = qm_dir / "ch3f-hessian.npy"
    molecule = load_xyz(xyz, bond_tolerance=1.5)
    return molecule.with_hessian(
        np.load(hess_path),
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source="q2mm-sn2-resource",
            path=str(hess_path.resolve()),
        ),
    )


def _normal_modes_path(data_dir_override: Path | None) -> Path:
    """Resolve the CH3F normal-modes ``.npz`` path, honouring a CLI override."""
    base = data_dir_override or _find_data_dir()
    return base / "ch3f-normal-modes.npz"


def load(
    *,
    backend: Any,
    data_dir: Path | None = None,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str,
) -> BenchmarkCase:
    """Build the CH3F :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Args:
        backend: MM backend used to compute frequencies at the starting
            force field (required — CH3F's reference is a frequency-only
            fit; see :func:`~q2mm.benchmarks.systems._assembly.assemble_qfuerza_fresh_case`).
        data_dir: Optional override for the packaged CH3F/SN2 resource
            directory.
        data_roots: Unused by CH3F (no external, non-distributed data);
            accepted for registry-call symmetry with other systems.
        starting_point: Accepted for interface symmetry; a no-op for
            this ``qfuerza_fresh``-strategy system (see
            :func:`~q2mm.benchmarks.systems._assembly.assemble_qfuerza_fresh_case`).
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative eigenvalue during QFUERZA projection — a
            no-op here since CH3F is a genuine ground-state minimum
            (no negative eigenvalue exists to replace).
        functional_form: Required (``"harmonic"`` or ``"mm3"``) — CH3F
            genuinely supports both forms (JAX/JAX-MD use harmonic,
            OpenMM/Tinker use MM3); there is no scientifically correct
            single default across engines, so the caller must decide.

    Returns:
        A fully-populated :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    """
    del data_roots  # CH3F ships in-package; no external roots needed.
    molecule = load_molecule(data_dir=data_dir)
    return assemble_qfuerza_fresh_case(
        key=KEY,
        name=NAME,
        molecule=molecule,
        stationary_point=StationaryPointKind.GROUND_STATE,
        backend=backend,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
        functional_form=functional_form,
        metadata=METADATA,
        normal_modes_path=_normal_modes_path,
        data_dir=data_dir,
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
    )
