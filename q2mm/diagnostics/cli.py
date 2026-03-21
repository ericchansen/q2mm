"""Command-line interface for Q2MM benchmarking.

Usage::

    q2mm-benchmark                      # Run all available backends x optimizers
    q2mm-benchmark --backend openmm     # Only OpenMM backend
    q2mm-benchmark --optimizer L-BFGS-B # Only L-BFGS-B optimizer
    q2mm-benchmark --output results/    # Save JSON results to directory
    q2mm-benchmark --load results/      # Load saved results and print report
    q2mm-benchmark --list               # Show available backends and optimizers
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np


def _discover_backends() -> list[tuple[str, type, str]]:
    """Discover available MM backends at runtime."""
    backends: list[tuple[str, type, str]] = []

    try:
        import openmm  # noqa: F401

        from q2mm.backends.mm.openmm import OpenMMEngine

        backends.append(("OpenMM", OpenMMEngine, "openmm"))
    except ImportError:
        pass

    try:
        from q2mm.backends.mm.tinker import TinkerEngine

        if TinkerEngine().is_available():
            backends.append(("Tinker", TinkerEngine, "tinker"))
    except (ImportError, FileNotFoundError, OSError):
        pass

    try:
        import jax  # noqa: F401

        from q2mm.backends.mm.jax_engine import JaxEngine

        backends.append(("JAX", JaxEngine, "jax"))
    except ImportError:
        pass

    return backends


def _optimizer_configs() -> list[tuple[str, dict]]:
    """Build the optimizer configuration list."""
    configs: list[tuple[str, dict]] = [
        ("L-BFGS-B", {"method": "L-BFGS-B"}),
        ("Nelder-Mead", {"method": "Nelder-Mead"}),
        ("Powell", {"method": "Powell"}),
    ]

    try:
        import jax  # noqa: F401

        configs.append(("L-BFGS-B+analytical", {"method": "L-BFGS-B", "jac": "analytical"}))
    except ImportError:
        pass

    return configs


def _find_reference_data() -> tuple[Path, Path, Path, Path | None]:
    """Locate CH3F reference data files.

    Returns (xyz, hessian, freqs, normal_modes_or_None).
    """
    # Try relative to package, then common locations
    candidates = [
        Path(__file__).resolve().parent.parent.parent / "examples" / "sn2-test" / "qm-reference",
        Path.cwd() / "examples" / "sn2-test" / "qm-reference",
    ]

    for qm_ref in candidates:
        xyz = qm_ref / "ch3f-optimized.xyz"
        hess = qm_ref / "ch3f-hessian.npy"
        freqs = qm_ref / "ch3f-frequencies.txt"
        modes = qm_ref / "ch3f-normal-modes.npz"

        if xyz.exists() and hess.exists() and freqs.exists():
            return xyz, hess, freqs, modes if modes.exists() else None

    raise FileNotFoundError(
        "Cannot find CH3F reference data. Expected in examples/sn2-test/qm-reference/. "
        "Run from the q2mm repository root."
    )


def _run_matrix(
    backends: list[tuple[str, type, str]],
    optimizers: list[tuple[str, dict]],
    output_dir: Path | None = None,
    *,
    leaderboard_only: bool = False,
) -> list:
    """Run the full backend × optimizer matrix.

    Returns list of BenchmarkResult.
    """
    from q2mm.diagnostics.benchmark import BenchmarkResult, run_benchmark
    from q2mm.diagnostics.pes_distortion import load_normal_modes
    from q2mm.diagnostics.report import detailed_report
    from q2mm.models.molecule import Q2MMMolecule

    xyz, hess_path, freqs_path, modes_path = _find_reference_data()

    molecule = Q2MMMolecule.from_xyz(xyz, bond_tolerance=1.5)
    qm_freqs = np.loadtxt(freqs_path)
    qm_hessian = np.load(hess_path)
    normal_modes = load_normal_modes(modes_path) if modes_path else None

    results: list[BenchmarkResult] = []
    total = len(backends) * len(optimizers)
    idx = 0

    for backend_name, engine_cls, _ in backends:
        try:
            engine = engine_cls()
        except Exception as e:
            print(f"  Skipping {backend_name}: {e}", file=sys.stderr)
            continue

        for opt_label, opt_config in optimizers:
            idx += 1
            combo = f"{backend_name} + {opt_label}"
            print(f"  [{idx}/{total}] {combo} ...", end=" ", flush=True)

            try:
                method = opt_config["method"]
                extra_kwargs = {k: v for k, v in opt_config.items() if k != "method"}

                t0 = time.perf_counter()
                r = run_benchmark(
                    engine,
                    molecule,
                    qm_freqs,
                    qm_hessian=qm_hessian,
                    normal_modes=normal_modes,
                    optimizer_method=method,
                    optimizer_kwargs=extra_kwargs,
                    maxiter=10_000,
                    backend_name=backend_name,
                    molecule_name="CH3F",
                    level_of_theory="B3LYP/6-31+G(d)",
                )
                elapsed = time.perf_counter() - t0
                results.append(r)

                opt = r.optimized or {}
                rmsd = opt.get("rmsd", float("nan"))

                # Show starting RMSD → final RMSD on progress line
                start_rmsd = None
                if r.seminario and r.seminario.get("rmsd") is not None:
                    start_rmsd = r.seminario["rmsd"]
                elif r.default_ff and r.default_ff.get("rmsd") is not None:
                    start_rmsd = r.default_ff["rmsd"]

                if start_rmsd is not None:
                    print(f"RMSD {start_rmsd:.0f}→{rmsd:.0f}  ({elapsed:.1f}s)")
                else:
                    print(f"RMSD={rmsd:.1f}  ({elapsed:.1f}s)")

                # Stream detailed tables immediately
                if not leaderboard_only:
                    for table in detailed_report(r, combo_label=combo):
                        table.flush()

            except Exception as e:
                print(f"FAILED: {e}", file=sys.stderr)
                results.append(
                    BenchmarkResult(
                        metadata={
                            "backend": backend_name,
                            "optimizer": opt_label,
                            "molecule": "CH3F",
                            "source": "q2mm",
                            "error": str(e),
                        },
                    )
                )

    # Save results if output directory specified
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(results):
            meta = r.metadata
            fname = f"{meta.get('backend', 'unk')}_{meta.get('optimizer', 'unk')}_{i:02d}.json"
            fname = fname.replace("+", "_").replace(" ", "_")
            r.to_json(output_dir / fname)
        print(f"\n  Results saved to: {output_dir}")

    return results


def _load_results(directory: Path) -> list:
    """Load all BenchmarkResult JSON files from a directory."""
    from q2mm.diagnostics.benchmark import BenchmarkResult

    results = []
    for jf in sorted(directory.glob("*.json")):
        try:
            results.append(BenchmarkResult.from_json(jf))
        except Exception as e:
            print(f"  Warning: could not load {jf.name}: {e}", file=sys.stderr)
    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``q2mm-benchmark`` CLI."""
    parser = argparse.ArgumentParser(
        prog="q2mm-benchmark",
        description="Run Q2MM benchmark matrix across backends and optimizers.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available backends and optimizers, then exit.",
    )
    parser.add_argument(
        "--backend",
        nargs="*",
        metavar="NAME",
        help="Run only these backends (e.g. openmm tinker jax). Default: all available.",
    )
    parser.add_argument(
        "--optimizer",
        nargs="*",
        metavar="NAME",
        help="Run only these optimizers (e.g. L-BFGS-B Powell). Default: all.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="DIR",
        help="Save JSON results to this directory.",
    )
    parser.add_argument(
        "--load",
        type=Path,
        metavar="DIR",
        help="Load results from a directory instead of running benchmarks.",
    )
    parser.add_argument(
        "--leaderboard-only",
        action="store_true",
        help="Print only the summary leaderboard, not detailed SI tables.",
    )

    args = parser.parse_args(argv)

    all_backends = _discover_backends()
    all_optimizers = _optimizer_configs()

    # --list: show what's available
    if args.list:
        print("\nAvailable backends:")
        if all_backends:
            for name, _, marker in all_backends:
                print(f"  {name:<12} (marker: {marker})")
        else:
            print("  (none detected)")

        print("\nAvailable optimizers:")
        for label, config in all_optimizers:
            print(f"  {label:<24} {config}")

        print()
        return 0

    # --load: load from directory
    if args.load:
        if not args.load.is_dir():
            print(f"Error: {args.load} is not a directory", file=sys.stderr)
            return 1
        results = _load_results(args.load)
        if not results:
            print(f"No results found in {args.load}", file=sys.stderr)
            return 1
        print(f"Loaded {len(results)} results from {args.load}\n")
    else:
        # Filter backends
        backends = all_backends
        if args.backend:
            filter_names = {b.lower() for b in args.backend}
            backends = [(n, c, m) for n, c, m in all_backends if n.lower() in filter_names]
            if not backends:
                print(f"Error: no matching backends for {args.backend}", file=sys.stderr)
                print(f"Available: {[n for n, _, _ in all_backends]}", file=sys.stderr)
                return 1

        # Filter optimizers
        optimizers = all_optimizers
        if args.optimizer:
            filter_names = {o.lower() for o in args.optimizer}
            optimizers = [(l, c) for l, c in all_optimizers if l.lower() in filter_names]
            if not optimizers:
                print(f"Error: no matching optimizers for {args.optimizer}", file=sys.stderr)
                print(f"Available: {[l for l, _ in all_optimizers]}", file=sys.stderr)
                return 1

        print("\nQ2MM Benchmark Matrix")
        print(f"  Backends:   {', '.join(n for n, _, _ in backends)}")
        print(f"  Optimizers: {', '.join(l for l, _ in optimizers)}")
        print(f"  Combos:     {len(backends) * len(optimizers)}\n")

        results = _run_matrix(backends, optimizers, output_dir=args.output, leaderboard_only=args.leaderboard_only)

    if not results:
        print("No results to report.", file=sys.stderr)
        return 1

    # Print leaderboard summary at the end
    print()
    from q2mm.diagnostics.tables import leaderboard_table

    rows = []
    for r in results:
        meta = r.metadata
        opt = r.optimized or {}
        # Starting RMSD: Seminario if available, else default FF
        initial_rmsd = float("nan")
        if r.seminario and r.seminario.get("rmsd") is not None:
            initial_rmsd = r.seminario["rmsd"]
        elif r.default_ff and r.default_ff.get("rmsd") is not None:
            initial_rmsd = r.default_ff["rmsd"]
        rows.append(
            {
                "backend": meta.get("backend", "?"),
                "optimizer": meta.get("optimizer", "?"),
                "rmsd": opt.get("rmsd", float("nan")),
                "mae": opt.get("mae", float("nan")),
                "time_s": opt.get("elapsed_s", 0.0) or 0.0,
                "n_eval": opt.get("n_eval", 0) or 0,
                "final_score": opt.get("final_score", float("nan")) or float("nan"),
                "converged": opt.get("converged", False),
                "message": opt.get("message", ""),
                "error": meta.get("error", ""),
                "initial_rmsd": initial_rmsd,
            }
        )
    if rows:
        leaderboard_table(rows).flush()

    # For --load, detailed tables weren't streamed, so print them now
    if args.load and not args.leaderboard_only:
        from q2mm.diagnostics.report import full_report

        full_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
