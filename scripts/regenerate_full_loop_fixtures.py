#!/usr/bin/env python3
"""Regenerate golden fixtures for full-loop parity tests.

Run this script after any intentional change to the optimization pipeline
to update the golden fixtures that ``test_full_loop_parity.py`` validates
against.

Usage::

    python scripts/regenerate_full_loop_fixtures.py

Outputs are saved to ``test/fixtures/full_loop/``.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from q2mm.backends.mm import OpenMMEngine
from q2mm.constants import (
    AMU_TO_KG,
    BOHR_TO_ANG,
    HARTREE_TO_J,
    MASSES,
    SPEED_OF_LIGHT_MS,
)
from q2mm.models.seminario import estimate_force_constants
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
from q2mm.optimizers.scipy_opt import ScipyOptimizer

OUT_DIR = REPO_ROOT / "test" / "fixtures" / "full_loop"
GS_FCHK = REPO_ROOT / "examples" / "ethane" / "GS.fchk"


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


def main() -> None:
    """Regenerate ethane GS golden fixture for full-loop parity tests."""
    if not GS_FCHK.exists():
        raise FileNotFoundError(f"Ethane GS.fchk not found at {GS_FCHK}")

    print("Regenerating ethane GS golden fixture...")

    ref, mol = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)
    qm_freqs = qm_frequencies_from_hessian(mol.hessian, mol.symbols)
    qm_real = sorted(f for f in qm_freqs if f > 50.0)

    # Seminario
    t0 = time.perf_counter()
    ff = estimate_force_constants(mol, au_hessian=True)
    t_sem = time.perf_counter() - t0
    sem_vec = ff.get_param_vector().copy()

    # MM frequencies + reference
    engine = OpenMMEngine()
    mm_all = engine.frequencies(mol, ff)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all) if f > 50.0)
    n = min(len(qm_real), len(mm_real_idx))

    freq_ref = ReferenceData()
    for k in range(n):
        freq_ref.add_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=0.001, molecule_idx=0)

    # Score + optimize
    obj = ObjectiveFunction(ff, engine, [mol], freq_ref)
    sem_score = obj(sem_vec)

    t0 = time.perf_counter()
    opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
    result = opt.optimize(obj)
    t_opt = time.perf_counter() - t0
    final_vec = ff.get_param_vector()

    # Frequency MAE
    mm_opt = engine.frequencies(mol, ff)
    mm_opt_real = sorted(f for f in mm_opt if f > 50.0)
    mae_sem = np.mean(np.abs(np.array(sorted([mm_all[i] for i in mm_real_idx])[:n]) - np.array(qm_real[:n])))
    mae_opt = np.mean(np.abs(np.array(sorted(mm_opt_real)[:n]) - np.array(qm_real[:n])))

    golden = {
        "system": "ethane-gs",
        "ref_type": "frequency",
        "n_atoms": len(mol.symbols),
        "n_real_freqs": len(qm_real),
        "n_params": ff.n_params,
        "qm_frequencies_cm1": [round(f, 4) for f in qm_real],
        "seminario": {
            "params": sem_vec.tolist(),
            "score": float(sem_score),
            "freq_mae_cm1": round(float(mae_sem), 4),
        },
        "optimized": {
            "method": "L-BFGS-B",
            "maxiter": 200,
            "params": final_vec.tolist(),
            "score": float(result.final_score),
            "improvement_pct": round(float(result.improvement * 100), 4),
            "converged": result.success,
            "freq_mae_cm1": round(float(mae_opt), 4),
        },
        "timing": {
            "seminario_s": round(t_sem, 4),
            "optimize_s": round(t_opt, 4),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fp = OUT_DIR / "ethane_gs_golden.json"
    with open(fp, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"  Saved: {fp}")
    print(f"  Seminario score: {sem_score:.10f}")
    print(f"  Optimized score: {result.final_score:.10f}")
    print(f"  Improvement: {result.improvement * 100:.2f}%")
    print(f"  Converged: {result.success}")


if __name__ == "__main__":
    sys.exit(main())
