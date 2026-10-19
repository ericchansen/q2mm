"""CH3F-SN2: F⁻ + CH3F identity SN2 transition state.

The D3h-symmetric TS of the identity SN2 reaction F⁻ + CH3F -> FCH3 + F⁻
at B3LYP/6-31+G(d) (one imaginary mode at approx -462 cm⁻¹, the
asymmetric C-F stretch along the reaction coordinate). This is the test
case Limé & Norrby 2015 (J. Comput. Chem. 36, 244) used to demonstrate
the FACAF bend force constant going negative under naive Method C
fitting and to motivate the Method E2 hybrid protocol.

Kept as a separate module from :mod:`q2mm.benchmarks.systems.ch3f`
(rather than a shared parametrized loader) so its ground-state/
transition-state distinction and TS-specific bond tolerance/charge stay
explicit at the call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from q2mm.benchmarks.cases import BenchmarkCase
from q2mm.benchmarks.systems._assembly import assemble_qfuerza_fresh_case
from q2mm.benchmarks.systems._paths import ExternalDataRoots, StartingPoint
from q2mm.benchmarks.systems.ch3f import _find_data_dir, _load_sn2_molecule
from q2mm.models.molecule import Molecule
from q2mm.models.problem import StationaryPointKind

KEY = "ch3f-sn2"
NAME = "F⁻ + CH3F SN2 TS"
DESCRIPTION = (
    "F⁻ + CH3F → FCH3 + F⁻ identity SN2 transition state "
    "(D3h, B3LYP/6-31+G(d)). Limé & Norrby 2015's canonical "
    "test case for Method E2 (FACAF bend force constant goes "
    "to zero/negative under naive Method C fitting). One "
    "imaginary mode ≈ −462 cm⁻¹ along the asymmetric C-F stretch."
)
DEFAULT_FORMS: tuple[str, ...] = ("harmonic", "mm3")
METADATA = {
    "level_of_theory": "B3LYP/6-31+G(d)",
    "publication": "Limé & Norrby, J. Comput. Chem. 2015, 36, 244",
    "doi": "10.1002/jcc.23797",
    "is_transition_state": True,
    "imaginary_mode_freq_cm": -461.7,
}


def load_molecule(*, data_dir: Path | None = None) -> Molecule:
    """Load the F⁻ + CH3F SN2 transition state + its QM Hessian.

    Bond tolerance is set to ``1.5`` (a unitless multiplier on the sum
    of covalent radii — see :func:`~q2mm.io.xyz.load_xyz`) to include
    the partially-formed C-F bonds at the TS geometry (~1.85 Å each);
    the default ``1.3`` misses them.  Charge is set to −1 to match the
    anionic complex on which the QM Hessian was computed (see
    ``scripts/generate_sn2_reference.py``).
    """
    qm_dir = data_dir or _find_data_dir()
    missing = [name for name in ("sn2-ts-optimized.xyz", "sn2-ts-hessian.npy") if not (qm_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"SN2 TS reference data missing in {qm_dir}: {missing}. "
            "Run ``scripts/generate_sn2_reference.py`` to regenerate the packaged TS "
            "Hessian + frequencies, or pass ``data_dir=`` pointing at a "
            "complete reference directory."
        )
    return _load_sn2_molecule("sn2-ts-optimized.xyz", "sn2-ts-hessian.npy", data_dir=qm_dir, charge=-1)


def load(
    *,
    backend: Any,
    data_dir: Path | None = None,
    data_roots: ExternalDataRoots | None = None,
    starting_point: StartingPoint = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    functional_form: str,
) -> BenchmarkCase:
    """Build the CH3F-SN2 :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    Args:
        backend: MM backend used to compute frequencies at the starting
            force field (required — the reference is a frequency-only
            fit; see :func:`~q2mm.benchmarks.systems._assembly.assemble_qfuerza_fresh_case`).
        data_dir: Optional override for the packaged CH3F/SN2 resource
            directory.
        data_roots: Unused (no external, non-distributed data); accepted
            for registry-call symmetry with other systems.
        starting_point: Accepted for interface symmetry; a no-op for
            this ``qfuerza_fresh``-strategy system.
        qfuerza_replace_with: Replacement value (Hartree/Bohr²) for the
            most-negative TS-Hessian eigenvalue during QFUERZA
            projection (Limé & Norrby Method C; default ``1.0``).
        functional_form: Required (``"harmonic"`` or ``"mm3"``) — CH3F-SN2
            genuinely supports both forms (JAX/JAX-MD use harmonic,
            OpenMM/Tinker use MM3); there is no scientifically correct
            single default across engines, so the caller must decide.

    Returns:
        A fully-populated :class:`~q2mm.benchmarks.cases.BenchmarkCase`.

    """
    del data_roots  # CH3F-SN2 ships in-package; no external roots needed.
    molecule = load_molecule(data_dir=data_dir)
    return assemble_qfuerza_fresh_case(
        key=KEY,
        name=NAME,
        molecule=molecule,
        stationary_point=StationaryPointKind.TRANSITION_STATE,
        backend=backend,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
        functional_form=functional_form,
        metadata=METADATA,
        normal_modes_path=(data_dir or _find_data_dir()) / "sn2-ts-normal-modes.npz",
        default_forms=DEFAULT_FORMS,
        description=DESCRIPTION,
    )
