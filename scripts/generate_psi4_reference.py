#!/usr/bin/env python3
"""Generate Psi4 QM reference data for rh-enamide (issue #74, D2).

Computes Hessians for the 9 rh-enamide training-set structures using Psi4
at B3LYP/def2-SVP (comparable to Jaguar's B3LYP/LACVP**). def2-SVP includes
built-in ECPs for heavy elements like Rh, matching the Hay-Wadt ECP approach
used by LACVP**.

Usage (inside Psi4 Docker container)::

    docker run --rm -v C:\\Users\\ericc\\repos\\q2mm:/q2mm -w /q2mm \\
        ghcr.io/ericchansen/q2mm/ci-psi4:latest \\
        python scripts/generate_psi4_reference.py

Outputs saved to ``examples/rh-enamide/psi4_reference/``.
"""

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import psi4  # noqa: F401
except ImportError:
    raise SystemExit("Psi4 not installed — run in the ci-psi4 Docker container")

from q2mm.backends.qm.psi4 import Psi4Engine
from q2mm.constants import (
    AMU_TO_KG,
    BOHR_TO_ANG,
    HARTREE_TO_J,
    MASSES,
    SPEED_OF_LIGHT_MS,
)
from q2mm.models.molecule import Q2MMMolecule
from q2mm.parsers import JaguarIn, MacroModel

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING = RH_DIR / "rh_enamide_training_set"
MMO = TRAINING / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING / "jaguar_spe_freq_in_out"
OUT_DIR = RH_DIR / "psi4_reference"


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


def main():
    print("Loading rh-enamide structures...")
    mm = MacroModel(str(MMO))
    jag_files = sorted(
        JAG_DIR.glob("*.in"),
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)],
    )
    n_mm_structures = len(mm.structures)
    n_jag_files = len(jag_files)
    expected_n = 9
    if not (n_mm_structures == n_jag_files == expected_n):
        raise SystemExit(
            "Input mismatch for rh-enamide training set:\n"
            f"  MacroModel structures: {n_mm_structures}\n"
            f"  Jaguar input files:     {n_jag_files}\n"
            f"  Expected count:         {expected_n}\n"
            f"  MMO file:               {MMO}\n"
            f"  Jaguar input dir:       {JAG_DIR}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # Psi4 memory is configured via Psi4Engine constructor; threads via n_threads=16.

    for i, (struct, jag_path) in enumerate(zip(mm.structures, jag_files)):
        label = jag_path.stem
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/9] {label} ({len(struct.atoms)} atoms)")
        print(f"{'=' * 60}")

        # Load Jaguar reference
        jag = JaguarIn(str(jag_path))
        jag_hessian = jag.get_hessian(len(struct.atoms))
        mol = Q2MMMolecule.from_structure(struct, hessian=jag_hessian)

        # Compute Psi4 Hessian at B3LYP/def2-SVP (charge=+1 for cationic Rh complex)
        structure = (list(mol.symbols), mol.geometry)
        t0 = time.perf_counter()
        with Psi4Engine(method="b3lyp", basis="def2-svp", charge=1, n_threads=16, memory="8 GB") as engine:
            psi4_hessian = engine.hessian(structure)
            psi4_energy = engine.energy(structure)
        elapsed = time.perf_counter() - t0
        print(f"  Psi4 Hessian computed in {elapsed:.1f}s")

        # Save Hessian as .npy
        np.save(OUT_DIR / f"{label}_hessian.npy", psi4_hessian)

        # Compare to Jaguar
        hess_diff = np.abs(psi4_hessian - jag_hessian)
        psi4_freqs = qm_frequencies_from_hessian(psi4_hessian, mol.symbols)
        jag_freqs = qm_frequencies_from_hessian(jag_hessian, mol.symbols)
        psi4_real = sorted(f for f in psi4_freqs if f > 50.0)
        jag_real = sorted(f for f in jag_freqs if f > 50.0)
        n = min(len(psi4_real), len(jag_real))
        freq_mae = float(np.mean(np.abs(np.array(psi4_real[:n]) - np.array(jag_real[:n]))))

        print(f"  Hessian max diff: {hess_diff.max():.6e}")
        print(f"  Freq MAE: {freq_mae:.1f} cm⁻¹ ({n} modes)")

        results.append(
            {
                "structure": label,
                "n_atoms": len(mol.symbols),
                "method": "b3lyp",
                "basis": "def2-svp",
                "psi4_energy_hartree": float(psi4_energy),
                "elapsed_s": round(elapsed, 1),
                "hessian_max_abs_diff": float(hess_diff.max()),
                "hessian_mean_abs_diff": float(hess_diff.mean()),
                "n_psi4_real_freqs": len(psi4_real),
                "n_jaguar_real_freqs": len(jag_real),
                "frequency_mae_cm1": round(freq_mae, 2),
                "psi4_real_freqs_cm1": [round(f, 2) for f in psi4_real],
                "jaguar_real_freqs_cm1": [round(f, 2) for f in jag_real],
            }
        )

    # Save summary
    summary = {
        "description": "Psi4 B3LYP/def2-SVP Hessians for rh-enamide training set",
        "jaguar_level": "B3LYP/LACVP**",
        "psi4_level": "B3LYP/def2-SVP",
        "n_structures": len(results),
        "structures": results,
    }
    summary_path = OUT_DIR / "psi4_reference_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Done! {len(results)} structures processed.")
    print(f"Hessians: {OUT_DIR}/*_hessian.npy")
    print(f"Summary:  {summary_path}")
    total_time = sum(r["elapsed_s"] for r in results)
    print(f"Total Psi4 time: {total_time:.0f}s ({total_time / 60:.1f} min)")


if __name__ == "__main__":
    sys.exit(main())
