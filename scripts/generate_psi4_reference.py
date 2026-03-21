#!/usr/bin/env python
"""Generate Psi4 QM reference data for ethane GS/TS.

Computes Hessians and frequencies using Psi4 at B3LYP/6-31G* and saves
results as NumPy fixtures for cross-validation against Gaussian .fchk.

Usage::

    python scripts/generate_psi4_reference.py

Outputs are saved to ``test/fixtures/full_loop/psi4_ethane/``.

Requirements: ``pip install q2mm[psi4]`` (needs psi4 from conda-forge).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import psi4  # noqa: F401
except ImportError:
    raise SystemExit("Psi4 not installed — run: conda install -c conda-forge psi4")

from q2mm.backends.qm.psi4 import Psi4Engine
from q2mm.constants import (
    AMU_TO_KG,
    BOHR_TO_ANG,
    HARTREE_TO_J,
    MASSES,
    SPEED_OF_LIGHT_MS,
)
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants
from q2mm.optimizers.objective import ReferenceData

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "test" / "fixtures" / "full_loop" / "psi4_ethane"
GS_FCHK = REPO_ROOT / "examples" / "ethane" / "GS.fchk"
TS_FCHK = REPO_ROOT / "examples" / "ethane" / "TS.fchk"


def qm_frequencies_from_hessian(hessian_au: np.ndarray, symbols: list[str]) -> np.ndarray:
    """Compute harmonic frequencies (cm⁻¹) from a Cartesian Hessian in AU."""
    bohr_to_m = BOHR_TO_ANG * 1e-10
    hessian_si = hessian_au * HARTREE_TO_J / (bohr_to_m**2)
    masses = np.array([MASSES[s] * AMU_TO_KG for s in symbols], dtype=float)
    mass_vec = np.repeat(masses, 3)
    mw = hessian_si / np.sqrt(np.outer(mass_vec, mass_vec))
    eigenvalues = np.linalg.eigvalsh(mw)
    freqs = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    freqs /= 2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0
    return freqs


def generate_for_system(fchk_path: Path, label: str) -> dict:
    """Run Psi4 on a system and return comparison data."""
    print(f"\n{'=' * 60}")
    print(f"Generating Psi4 reference for {label}")
    print(f"{'=' * 60}")

    ref, mol = ReferenceData.from_fchk(str(fchk_path), bond_tolerance=1.4)
    print(f"  Atoms: {len(mol.symbols)}, Bonds: {len(mol.bonds)}")
    print(f"  Gaussian Hessian shape: {mol.hessian.shape}")

    # Compute Psi4 Hessian
    with Psi4Engine(method="b3lyp", basis="6-31g*") as engine:
        psi4_hessian = engine.hessian(mol)
        psi4_energy = engine.energy(mol)

    # Compare Hessians
    gauss_hessian = mol.hessian
    hess_diff = np.abs(psi4_hessian - gauss_hessian)
    print(f"  Hessian max abs diff: {hess_diff.max():.6e}")
    print(f"  Hessian mean abs diff: {hess_diff.mean():.6e}")

    # Compare frequencies
    psi4_freqs = qm_frequencies_from_hessian(psi4_hessian, mol.symbols)
    gauss_freqs = qm_frequencies_from_hessian(gauss_hessian, mol.symbols)
    psi4_real = sorted(f for f in psi4_freqs if f > 50.0)
    gauss_real = sorted(f for f in gauss_freqs if f > 50.0)
    freq_mae = np.mean(np.abs(np.array(psi4_real) - np.array(gauss_real)))
    print(f"  Frequency MAE: {freq_mae:.2f} cm⁻¹")

    # Seminario comparison
    mol_psi4 = Q2MMMolecule(
        symbols=mol.symbols,
        geometry=mol.geometry,
        name=f"{label}-psi4",
        bond_tolerance=1.4,
        hessian=psi4_hessian,
    )
    ff_psi4 = estimate_force_constants(mol_psi4, au_hessian=True)
    ff_gauss = estimate_force_constants(mol, au_hessian=True)

    param_diff = np.abs(ff_psi4.get_param_vector() - ff_gauss.get_param_vector())
    print(f"  Seminario param max diff: {param_diff.max():.6e}")

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{label}-psi4-hessian.npy", psi4_hessian)

    result = {
        "system": label,
        "method": "b3lyp",
        "basis": "6-31g*",
        "psi4_energy_hartree": float(psi4_energy),
        "hessian_max_abs_diff": float(hess_diff.max()),
        "hessian_mean_abs_diff": float(hess_diff.mean()),
        "frequency_mae_cm1": float(freq_mae),
        "psi4_real_freqs": [round(f, 4) for f in psi4_real],
        "gauss_real_freqs": [round(f, 4) for f in gauss_real],
        "psi4_seminario_params": ff_psi4.get_param_vector().tolist(),
        "gauss_seminario_params": ff_gauss.get_param_vector().tolist(),
        "seminario_param_max_diff": float(param_diff.max()),
    }
    return result


def main():
    results = {}

    if GS_FCHK.exists():
        results["ethane_gs"] = generate_for_system(GS_FCHK, "ethane-gs")
    else:
        print(f"Skipping GS: {GS_FCHK} not found")

    if TS_FCHK.exists():
        results["ethane_ts"] = generate_for_system(TS_FCHK, "ethane-ts")
    else:
        print(f"Skipping TS: {TS_FCHK} not found")

    # Save summary
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "cross_validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
