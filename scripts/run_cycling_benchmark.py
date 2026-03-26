#!/usr/bin/env python3
"""GRAD→SIMP cycling benchmark for GPU vs CPU comparison.

Runs OptimizationLoop on CH3F (small) or rh-enamide (large) with
frequency-based reference data and reports timing, convergence, and
score progression.

Usage:
    python scripts/run_cycling_benchmark.py --molecule ch3f --engine jax
    python scripts/run_cycling_benchmark.py --molecule rh-enamide --engine jax \
        --max-cycles 100 --max-params 5 --output benchmarks/rh-enamide/results-cycling
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REAL_FREQUENCY_THRESHOLD = 10.0  # cm⁻¹


def _real_indices(freqs: list[float]) -> list[int]:
    """Return sorted indices of real (positive) vibrational modes."""
    return sorted(i for i, f in enumerate(freqs) if f > REAL_FREQUENCY_THRESHOLD)


def _load_ch3f(engine: object, data_dir: Path) -> tuple:
    """Load CH3F molecule, force field, and frequency reference data."""
    from q2mm.diagnostics.benchmark import real_frequencies
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.optimizers.objective import ReferenceData
    from q2mm.models.seminario import estimate_force_constants

    qm_dir = data_dir / "examples" / "sn2-test" / "qm-reference"
    mol = Q2MMMolecule.from_xyz(str(qm_dir / "ch3f-optimized.xyz"), bond_tolerance=1.5)
    qm_freqs = np.loadtxt(str(qm_dir / "ch3f-frequencies.txt"))
    qm_hessian = np.load(str(qm_dir / "ch3f-hessian.npy"))

    qm_real = real_frequencies(qm_freqs)

    # Build Seminario FF as starting point
    mol_h = mol.with_hessian(qm_hessian)
    ff = estimate_force_constants(mol_h)

    # Map QM frequencies to MM mode indices
    mm_all = engine.frequencies(mol, ff)
    mm_real_idx = _real_indices(mm_all)

    ref = ReferenceData()
    n = min(len(qm_real), len(mm_real_idx))
    for k in range(n):
        ref.add_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=0.001, molecule_idx=0)

    return [mol], ff, ref, {"qm_real": qm_real.tolist(), "n_modes": n}


def _load_rh_enamide(engine: object, data_dir: Path) -> tuple:
    """Load 9 rh-enamide molecules, force field, and frequency reference data.

    If the engine doesn't support MM3 functional form, a harmonic force field
    is auto-generated from the first molecule's topology.
    """
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.optimizers.objective import ReferenceData
    from q2mm.parsers.jaguar import JaguarOut

    rh_dir = data_dir / "examples" / "rh-enamide"
    xyz_dir = rh_dir / "rh_enamide_training_set" / "raw_xyz"
    jag_dir = rh_dir / "rh_enamide_training_set" / "jaguar_spe_freq_in_out"

    # Map xyz files to jaguar output files by leading digit
    xyz_files = sorted(xyz_dir.glob("*.xyz"))
    jag_files = sorted(jag_dir.glob("*.out"))

    # Load molecules first
    molecules = []
    for xyz_path in xyz_files:
        mol = Q2MMMolecule.from_xyz(str(xyz_path), bond_tolerance=1.5)
        molecules.append(mol)

    # Choose force field based on engine capability
    mm3_ff = ForceField.from_mm3_fld(str(rh_dir / "mm3.fld"))
    engine_forms = getattr(engine, "supported_functional_forms", lambda: frozenset())()
    ff_source = mm3_ff.source_format or "mm3_fld"

    if ff_source == "mm3_fld" and "mm3" not in engine_forms and "harmonic" in engine_forms:
        print("  Engine doesn't support MM3 — auto-generating harmonic FF from topology")
        ff = ForceField.create_for_molecule(molecules[0], name="rh-enamide auto-harmonic")
        ff_type = "harmonic (auto)"
    else:
        ff = mm3_ff
        ff_type = "mm3"

    ref = ReferenceData()
    total_modes = 0

    for mol_idx, (mol, jag_path) in enumerate(zip(molecules, jag_files)):
        jag = JaguarOut(str(jag_path))
        qm_freqs = np.array(jag.frequencies)
        qm_real = qm_freqs[qm_freqs > REAL_FREQUENCY_THRESHOLD]
        qm_real = np.sort(qm_real)

        # Map to MM mode indices
        mm_all = engine.frequencies(mol, ff)
        mm_real_idx = _real_indices(mm_all)

        n = min(len(qm_real), len(mm_real_idx))
        for k in range(n):
            ref.add_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=0.001, molecule_idx=mol_idx)
        total_modes += n
        print(
            f"  Loaded mol {mol_idx + 1}/{len(xyz_files)}: {xyz_files[mol_idx].name} "
            f"({mol.n_atoms} atoms, {n} freq modes)"
        )

    meta = {
        "n_molecules": len(molecules),
        "total_modes": total_modes,
        "n_params": ff.n_params,
        "ff_type": ff_type,
    }
    return molecules, ff, ref, meta


def _get_engine(engine_name: str) -> object:
    """Instantiate the requested MM engine."""
    if engine_name == "jax":
        from q2mm.backends.mm.jax_engine import JaxEngine

        return JaxEngine()
    elif engine_name == "jax-md":
        from q2mm.backends.mm.jax_md_engine import JaxMDEngine

        return JaxMDEngine()
    elif engine_name == "openmm":
        from q2mm.backends.mm.openmm import OpenMMEngine

        return OpenMMEngine()
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


def main() -> None:
    """Run GRAD→SIMP cycling benchmark."""
    parser = argparse.ArgumentParser(description="GRAD→SIMP cycling benchmark")
    parser.add_argument(
        "--molecule", required=True, choices=["ch3f", "rh-enamide"], help="Which molecule set to benchmark"
    )
    parser.add_argument("--engine", required=True, choices=["jax", "jax-md", "openmm"], help="MM engine backend")
    parser.add_argument("--max-cycles", type=int, default=50, help="Max optimization cycles")
    parser.add_argument("--convergence", type=float, default=0.001, help="Convergence threshold")
    parser.add_argument("--max-params", type=int, default=3, help="Max params for simplex subspace")
    parser.add_argument("--full-method", default="L-BFGS-B", help="Full-space optimizer")
    parser.add_argument("--simp-method", default="Nelder-Mead", help="Simplex optimizer")
    parser.add_argument("--full-maxiter", type=int, default=200, help="Full-space max iterations")
    parser.add_argument("--simp-maxiter", type=int, default=200, help="Simplex max iterations")
    parser.add_argument(
        "--jac",
        choices=["analytical", "finite-diff"],
        default="finite-diff",
        help="Jacobian strategy for GRAD phase (default: finite-diff)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default=None, help="Root directory of q2mm repo (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parent.parent
    if not (data_dir / "examples").is_dir():
        print(f"ERROR: Cannot find examples/ in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Detect compute device
    try:
        import jax

        device = jax.default_backend()
    except ImportError:
        device = "cpu"

    print("╔══════════════════════════════════════════════════╗")
    print("║  GRAD→SIMP Cycling Benchmark                    ║")
    print("╠══════════════════════════════════════════════════╣")
    print(f"║  Molecule:    {args.molecule:<35s}║")
    print(f"║  Engine:      {args.engine:<35s}║")
    print(f"║  Device:      {device:<35s}║")
    print(f"║  Max cycles:  {args.max_cycles:<35d}║")
    print(f"║  Max params:  {args.max_params:<35d}║")
    print(f"║  Convergence: {args.convergence:<35g}║")
    jac_value = "analytical" if args.jac == "analytical" else None
    print(f"║  Jacobian:    {args.jac:<35s}║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    engine = _get_engine(args.engine)

    print(f"Loading {args.molecule} data...")
    t_load = time.perf_counter()
    if args.molecule == "ch3f":
        molecules, ff, ref, meta = _load_ch3f(engine, data_dir)
    else:
        molecules, ff, ref, meta = _load_rh_enamide(engine, data_dir)
    load_elapsed = time.perf_counter() - t_load
    print(f"  Loaded in {load_elapsed:.1f}s")
    print(f"  FF params: {ff.n_params}")
    print(f"  References: {len(ref.values)}")
    print()

    # Build objective + cycling loop
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.cycling import OptimizationLoop

    obj = ObjectiveFunction(ff, engine, molecules, ref)

    loop = OptimizationLoop(
        obj,
        max_params=args.max_params,
        convergence=args.convergence,
        max_cycles=args.max_cycles,
        full_method=args.full_method,
        simp_method=args.simp_method,
        full_maxiter=args.full_maxiter,
        simp_maxiter=args.simp_maxiter,
        full_jac=jac_value,
        verbose=True,
    )

    print("Starting optimization loop...")
    print("=" * 60)
    t_opt = time.perf_counter()
    result = loop.run()
    opt_elapsed = time.perf_counter() - t_opt
    print("=" * 60)
    print()

    print(result.summary())
    print(f"\nTotal optimization time: {opt_elapsed:.1f}s")
    print(f"Device: {device}")

    # Save results
    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        result_data = {
            "metadata": {
                "molecule": args.molecule,
                "engine": args.engine,
                "device": device,
                "max_cycles": args.max_cycles,
                "max_params": args.max_params,
                "convergence": args.convergence,
                "full_method": args.full_method,
                "simp_method": args.simp_method,
                "full_maxiter": args.full_maxiter,
                "simp_maxiter": args.simp_maxiter,
                "jac": args.jac,
                **meta,
            },
            "results": {
                "success": result.success,
                "initial_score": result.initial_score,
                "final_score": result.final_score,
                "n_cycles": result.n_cycles,
                "cycle_scores": result.cycle_scores,
                "improvement": result.improvement,
                "message": result.message,
                "total_elapsed_s": opt_elapsed,
                "load_elapsed_s": load_elapsed,
                "n_eval": result.n_eval,
            },
        }

        fname = f"{args.molecule}_{args.engine}_{device}_cycling.json"
        out_path = out_dir / fname
        with open(out_path, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
