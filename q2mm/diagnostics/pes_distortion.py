"""PES distortion analysis along QM normal modes.

Displaces a molecule along its QM-derived normal mode eigenvectors
and compares the MM energy change to the QM harmonic prediction.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from q2mm.constants import AMU_TO_KG, BOHR_TO_ANG, HARTREE_TO_J, SPEED_OF_LIGHT_MS

if TYPE_CHECKING:
    from q2mm.backends.base import MMEngine
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule

HA_TO_KCAL = 627.5094740631


def load_normal_modes(path: Path) -> dict:
    """Load pre-computed normal mode decomposition from ``.npz`` file.

    The file must contain arrays ``eigenvalues``, ``eigenvectors``, and
    ``masses_amu`` (from a mass-weighted Hessian eigendecomposition).

    Args:
        path (Path): Path to the ``.npz`` file.

    Returns:
        dict: Dictionary with keys ``'eigenvalues'``, ``'eigenvectors'``,
            and ``'masses_amu'``, each mapping to a ``numpy.ndarray``.

    """
    data = np.load(path, allow_pickle=False)
    return {
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": data["eigenvectors"],
        "masses_amu": data["masses_amu"],
    }


def compute_distortions(
    mol: Q2MMMolecule,
    ff: ForceField,
    engine: MMEngine,
    modes: dict,
    target_norms_ang: list[float] | None = None,
) -> tuple[list[dict], float, float]:
    """Displace molecule along QM normal modes and compare energies.

    Args:
        mol (Q2MMMolecule): Equilibrium geometry molecule.
        ff (ForceField): Force field to evaluate MM energies with.
        engine (MMEngine): Any backend engine with an ``energy()`` method.
        modes (dict): Output from ``load_normal_modes()``.
        target_norms_ang (list[float] | None): Cartesian displacement
            magnitudes in Angstrom. Defaults to ``[0.05, 0.10, 0.15]``.

    Returns:
        tuple[list[dict], float, float]: A 3-tuple of:

            - **results** (*list[dict]*) — Per-mode results with keys
              ``mode_idx``, ``freq_cm1``, and ``displacements``.
            - **e_eq** (*float*) — Equilibrium MM energy in kcal/mol.
            - **elapsed** (*float*) — Wall-clock time in seconds.

    """
    from q2mm.models.molecule import Q2MMMolecule as _Mol

    if target_norms_ang is None:
        target_norms_ang = [0.05, 0.10, 0.15]

    eigenvalues = modes["eigenvalues"]
    eigenvectors = modes["eigenvectors"]
    masses_amu = modes["masses_amu"]

    bohr_to_m = BOHR_TO_ANG * 1e-10
    sqrt_m = np.sqrt(np.repeat(masses_amu, 3))

    # Identify real vibrational modes (skip 6 trans/rot near zero)
    real_mode_indices = [i for i, ev in enumerate(eigenvalues) if ev > 1e-3]

    e_eq = engine.energy(mol, ff)

    t0 = time.perf_counter()
    results = []

    for mi in real_mode_indices:
        ev = eigenvalues[mi]
        evec_mw = eigenvectors[:, mi]

        # Eigenvalue -> frequency for labeling
        ev_si = ev * HARTREE_TO_J / (bohr_to_m**2 * AMU_TO_KG)
        freq_cm1 = np.sqrt(ev_si) / (2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0)

        # Un-mass-weight eigenvector to get Cartesian direction
        v_cart = evec_mw / sqrt_m  # Bohr (per unit q)
        v_cart_ang = v_cart * BOHR_TO_ANG  # Angstrom
        v_norm = np.linalg.norm(v_cart_ang)

        displacements = []
        for d_ang in target_norms_ang:
            # Scale q so Cartesian displacement norm = d_ang
            q = d_ang / v_norm  # Bohr * sqrt(amu)

            # QM harmonic energy: E = 0.5 * eigenvalue * q^2
            e_qm = 0.5 * ev * q**2 * HA_TO_KCAL  # kcal/mol

            # MM energy at displaced geometry
            delta_xyz = (q * v_cart * BOHR_TO_ANG).reshape(-1, 3)
            disp_mol = _Mol(
                symbols=mol.symbols,
                geometry=mol.geometry + delta_xyz,
                atom_types=mol.atom_types,
                charge=mol.charge,
                multiplicity=mol.multiplicity,
                bond_tolerance=mol.bond_tolerance,
            )
            e_mm = engine.energy(disp_mol, ff) - e_eq

            pct_err = ((e_mm - e_qm) / e_qm * 100.0) if abs(e_qm) > 1e-8 else 0.0
            displacements.append({"d_ang": d_ang, "e_qm": e_qm, "e_mm": e_mm, "pct_err": pct_err})

        results.append({"mode_idx": mi, "freq_cm1": freq_cm1, "displacements": displacements})

    elapsed = time.perf_counter() - t0
    return results, e_eq, elapsed
