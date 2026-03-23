"""Command-line interface for Q2MM benchmarking.

Usage::

    q2mm-benchmark                      # Run all available backends x optimizers
    q2mm-benchmark --backend openmm     # Only OpenMM backend
    q2mm-benchmark --optimizer L-BFGS-B # Only L-BFGS-B optimizer
    q2mm-benchmark --output results/    # Save JSON results to directory
    q2mm-benchmark --load results/      # Load saved results and print report
    q2mm-benchmark --list               # Show available backends and optimizers
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def _discover_backends() -> list[tuple[str, type, str]]:
    """Discover available MM backends at runtime.

    Returns:
        list[tuple[str, type, str]]: List of ``(display_name, engine_class,
            marker)`` tuples for each importable backend.
    """
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
    """Build the optimizer configuration list.

    Returns:
        list[tuple[str, dict]]: List of ``(label, config_dict)`` tuples.
            Each ``config_dict`` contains at minimum a ``'method'`` key.
    """
    configs: list[tuple[str, dict]] = [
        ("L-BFGS-B", {"method": "L-BFGS-B"}),
        ("Nelder-Mead", {"method": "Nelder-Mead"}),
        ("Powell", {"method": "Powell"}),
    ]
    # Note: L-BFGS-B+analytical is omitted because the benchmark uses
    # frequency reference data and ObjectiveFunction.gradient() only
    # supports energy-type references. Once frequency gradients are
    # implemented, this can be re-enabled.
    return configs


def _find_reference_data(data_dir: Path | None = None) -> tuple[Path, Path, Path, Path | None]:
    """Locate CH3F reference data files.

    Args:
        data_dir (Path | None): Explicit directory to search. Falls back
            to the repo ``examples/`` directory and CWD if ``None``.

    Returns:
        tuple[Path, Path, Path, Path | None]: A 4-tuple of
            ``(xyz_path, hessian_path, frequencies_path, normal_modes_path)``.
            The normal-modes path is ``None`` if the ``.npz`` file is absent.

    Raises:
        FileNotFoundError: If the required reference files cannot be found
            in any candidate directory.
    """
    candidates = []
    if data_dir is not None:
        candidates.append(data_dir)
    # Fallback: try relative to repo checkout, then CWD
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent.parent / "examples" / "sn2-test" / "qm-reference",
            Path.cwd() / "examples" / "sn2-test" / "qm-reference",
        ]
    )

    for qm_ref in candidates:
        xyz = qm_ref / "ch3f-optimized.xyz"
        hess = qm_ref / "ch3f-hessian.npy"
        freqs = qm_ref / "ch3f-frequencies.txt"
        modes = qm_ref / "ch3f-normal-modes.npz"

        if xyz.exists() and hess.exists() and freqs.exists():
            return xyz, hess, freqs, modes if modes.exists() else None

    raise FileNotFoundError(
        "Cannot find CH3F reference data. Use --data-dir to specify the "
        "directory containing ch3f-optimized.xyz, ch3f-hessian.npy, etc. "
        "For a git checkout, run from the repo root."
    )


def _run_matrix(
    backends: list[tuple[str, type, str]],
    optimizers: list[tuple[str, dict]],
    output_dir: Path | None = None,
    *,
    leaderboard_only: bool = False,
    data_dir: Path | None = None,
) -> list:
    """Run the full backend × optimizer matrix.

    Args:
        backends (list[tuple[str, type, str]]): Backend entries from
            ``_discover_backends()``.
        optimizers (list[tuple[str, dict]]): Optimizer entries from
            ``_optimizer_configs()``.
        output_dir (Path | None): Directory to save JSON result files.
            Created if it does not exist.
        leaderboard_only (bool): If ``True``, skip streaming detailed
            SI tables during the run.
        data_dir (Path | None): Override for QM reference data directory.

    Returns:
        list[BenchmarkResult]: One result per (backend, optimizer) combination.
    """
    from q2mm.diagnostics.benchmark import BenchmarkResult, run_benchmark
    from q2mm.diagnostics.pes_distortion import load_normal_modes
    from q2mm.diagnostics.report import detailed_report
    from q2mm.models.molecule import Q2MMMolecule

    xyz, hess_path, freqs_path, modes_path = _find_reference_data(data_dir)

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
    """Load all BenchmarkResult JSON files from a directory.

    Args:
        directory (Path): Directory containing ``*.json`` result files.

    Returns:
        list[BenchmarkResult]: Successfully loaded results (files that
            fail to parse are skipped with a warning).
    """
    from q2mm.diagnostics.benchmark import BenchmarkResult

    results = []
    for jf in sorted(directory.glob("*.json")):
        try:
            results.append(BenchmarkResult.from_json(jf))
        except Exception as e:
            print(f"  Warning: could not load {jf.name}: {e}", file=sys.stderr)
    return results


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``q2mm-benchmark`` CLI.

    Args:
        argv (list[str] | None): Command-line arguments. If ``None``,
            ``sys.argv[1:]`` is used (via ``argparse``).

    Returns:
        int: Exit code — ``0`` on success, ``1`` on error.
    """
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
    parser.add_argument(
        "--data-dir",
        type=Path,
        metavar="DIR",
        help="Path to QM reference data directory (containing ch3f-*.xyz, etc.). "
        "Required for pip-installed q2mm; optional for git checkouts.",
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

        results = _run_matrix(
            backends, optimizers, output_dir=args.output, leaderboard_only=args.leaderboard_only, data_dir=args.data_dir
        )

    if not results:
        print("No results to report.", file=sys.stderr)
        return 1

    # Print leaderboard summary at the end
    print()
    from q2mm.diagnostics.report import build_leaderboard_rows
    from q2mm.diagnostics.tables import leaderboard_table

    rows = build_leaderboard_rows(results)
    if rows:
        leaderboard_table(rows).flush()

    # For --load, detailed tables weren't streamed, so print them now
    if args.load and not args.leaderboard_only:
        from q2mm.diagnostics.report import full_report

        full_report(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
