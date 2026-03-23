#!/usr/bin/env python3
"""Generate performance benchmark data for docs/performance.md.

Runs each optimizer method from the SAME initial (perturbed) parameters
to produce comparable results.

Usage:
    python scripts/generate_benchmarks.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from q2mm.backends.mm.openmm import OpenMMEngine

    HAS_OPENMM = True
except ImportError:
    HAS_OPENMM = False

try:
    from q2mm.backends.mm.tinker import TinkerEngine

    HAS_TINKER = True
except Exception:
    HAS_TINKER = False

from q2mm.models.forcefield import AngleParam, BondParam, ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
from q2mm.optimizers.scipy_opt import ScipyOptimizer


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
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
        name="water",
        bond_tolerance=1.5,
    )


def _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def build_problem(engine):
    """Build reference data from true parameters, return (guess_ff, mols, ref)."""
    true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)

    mol_eq = _water(104.5, 0.96)
    mol_wide = _water(115.0, 0.96)
    mol_long = _water(104.5, 1.05)
    mols = [mol_eq, mol_wide, mol_long]

    ref = ReferenceData()
    for i, mol in enumerate(mols):
        ref.add_energy(engine.energy(mol, true_ff), weight=1.0, molecule_idx=i)

    freqs = engine.frequencies(mol_eq, true_ff)
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:
            ref.add_frequency(f, data_idx=j, weight=0.001, molecule_idx=0)

    # Perturbed starting point — same for all methods
    guess_ff = _water_ff(bond_k=611.5, bond_r0=1.01, angle_k=79.1, angle_eq=109.5)

    return guess_ff, mols, ref


def run_benchmark(engine, method: str, guess_ff: ForceField, mols, ref, use_bounds=True):
    """Run a single optimizer benchmark from a FRESH copy of guess_ff."""
    # Deep copy the guess FF so each method starts from identical params
    fresh_ff = _water_ff(
        bond_k=guess_ff.bonds[0].force_constant,
        bond_r0=guess_ff.bonds[0].equilibrium,
        angle_k=guess_ff.angles[0].force_constant,
        angle_eq=guess_ff.angles[0].equilibrium,
    )

    obj = ObjectiveFunction(fresh_ff, engine, mols, ref)
    opt = ScipyOptimizer(method=method, maxiter=500, use_bounds=use_bounds, verbose=False)

    t0 = time.perf_counter()
    result = opt.optimize(obj)
    elapsed = time.perf_counter() - t0

    return {
        "method": method,
        "time_s": elapsed,
        "n_eval": result.n_evaluations,
        "evals_per_s": result.n_evaluations / elapsed if elapsed > 0 else 0,
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "improvement": result.improvement,
        "success": result.success,
    }


def format_score(score: float) -> str:
    if score < 0.01:
        return f"{score:.3f}"
    elif score < 10:
        return f"{score:.2f}"
    else:
        return f"{score:.1f}"


def print_table(backend_name: str, results: list[dict]):
    print(f"\n### {backend_name} Backend\n")
    print("| Method | Time | Evaluations | Evals/s | Initial → Final Score |")
    print("|--------|------|-------------|---------|----------------------|")
    for r in results:
        init = format_score(r["initial_score"])
        final = format_score(r["final_score"])
        print(
            f"| **{r['method']}** | {r['time_s']:.1f} s | {r['n_eval']} | {r['evals_per_s']:.1f} | {init} → {final} |"
        )


def main() -> int:
    methods_bounded = ["L-BFGS-B", "Powell"]
    methods_unbounded = ["Nelder-Mead"]

    if HAS_OPENMM:
        print("=" * 60)
        print("OpenMM Backend Benchmarks")
        print("=" * 60)
        engine = OpenMMEngine()
        guess_ff, mols, ref = build_problem(engine)

        # Verify all methods get the same initial score
        test_obj = ObjectiveFunction(
            _water_ff(
                bond_k=guess_ff.bonds[0].force_constant,
                bond_r0=guess_ff.bonds[0].equilibrium,
                angle_k=guess_ff.angles[0].force_constant,
                angle_eq=guess_ff.angles[0].equilibrium,
            ),
            engine,
            mols,
            ref,
        )
        init_score = test_obj(test_obj.forcefield.get_param_vector())
        print(f"\nCommon initial score: {init_score:.4f}")
        print(f"Initial params: {guess_ff.get_param_vector().tolist()}\n")

        results = []
        for method in methods_unbounded:
            print(f"  Running {method}...", end=" ", flush=True)
            r = run_benchmark(engine, method, guess_ff, mols, ref, use_bounds=False)
            print(f"{r['time_s']:.1f}s, {r['n_eval']} evals, {r['initial_score']:.4f} → {r['final_score']:.6f}")
            results.append(r)

        for method in methods_bounded:
            print(f"  Running {method}...", end=" ", flush=True)
            r = run_benchmark(engine, method, guess_ff, mols, ref, use_bounds=True)
            print(f"{r['time_s']:.1f}s, {r['n_eval']} evals, {r['initial_score']:.4f} → {r['final_score']:.6f}")
            results.append(r)

        print_table("OpenMM", results)
    else:
        print("OpenMM not available — skipping.")

    if HAS_TINKER:
        print(f"\n{'=' * 60}")
        print("Tinker Backend Benchmarks")
        print("=" * 60)
        engine = TinkerEngine()
        guess_ff, mols, ref = build_problem(engine)

        results = []
        for method in methods_unbounded:
            print(f"  Running {method}...", end=" ", flush=True)
            r = run_benchmark(engine, method, guess_ff, mols, ref, use_bounds=False)
            print(f"{r['time_s']:.1f}s, {r['n_eval']} evals, {r['initial_score']:.4f} → {r['final_score']:.6f}")
            results.append(r)

        print_table("Tinker", results)
    else:
        print("\nTinker not available — skipping.")

    print("\n--- Copy the tables above into docs/performance.md ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
