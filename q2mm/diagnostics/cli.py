"""Command-line interface for Q2MM benchmarking.

Usage::

    q2mm-benchmark                              # Run CH3F (default) across all backends
    q2mm-benchmark --system rh-enamide          # Run Rh-enamide (9 molecules)
    q2mm-benchmark --backend openmm             # Only OpenMM backend
    q2mm-benchmark --optimizer L-BFGS-B         # Only L-BFGS-B optimizer
    q2mm-benchmark --output results/            # Save to custom directory
    q2mm-benchmark --no-save                    # Run without saving results
    q2mm-benchmark --load results/              # Load saved results and print report
    q2mm-benchmark --list                       # Show available backends, optimizers, systems
    q2mm-benchmark --platform CUDA              # Force OpenMM CUDA platform

By default, results (JSON + force field files) are saved to
``./benchmark_results/``.  Use ``--no-save`` to disable.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from q2mm.backends.base import MMEngine
    from q2mm.diagnostics.benchmark import BenchmarkResult
    from q2mm.diagnostics.systems import BenchmarkSystem, SystemData

import numpy as np


def _discover_backends() -> list[tuple[str, type, str]]:
    """Discover available MM backends at runtime via the engine registry.

    Returns:
        list[tuple[str, type, str]]: List of ``(display_name, engine_class,
            registry_key)`` tuples for each available backend.

    """
    from q2mm.backends.registry import available_mm_engines, registered_mm_engines

    engines = registered_mm_engines()
    backends = []
    for key in available_mm_engines():
        cls = engines[key]
        # Use the engine's display name (from the ``name`` property).
        # Fall back to the registry key if instantiation fails.
        try:
            display_name = cls().name
        except Exception:
            display_name = key
        backends.append((display_name, cls, key))
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


def _resolve_system(system_key: str) -> BenchmarkSystem:
    """Look up a benchmark system by key.

    Args:
        system_key: System name (e.g. ``"ch3f"``, ``"rh-enamide"``).

    Returns:
        BenchmarkSystem: The system configuration.

    Raises:
        SystemExit: If the system is not registered.

    """
    from q2mm.diagnostics.systems import SYSTEMS

    if system_key not in SYSTEMS:
        available = ", ".join(sorted(SYSTEMS))
        raise SystemExit(f"Error: unknown system {system_key!r}. Available: {available}")
    return SYSTEMS[system_key]


def _run_matrix(
    backends: list[tuple[str, type, str]],
    optimizers: list[tuple[str, dict]],
    output_dir: Path | None = None,
    *,
    leaderboard_only: bool = False,
    data_dir: Path | None = None,
    platform: str | None = None,
    system_key: str = "ch3f",
    max_iter: int = 10_000,
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
        data_dir (Path | None): Override for QM reference data directory
            (only used for CH3F system).
        platform (str | None): OpenMM platform override (e.g. ``"CUDA"``).
            Passed to :class:`OpenMMEngine` when instantiating OpenMM
            backends.  ``None`` triggers auto-detection.
        system_key (str): Benchmark system to run (e.g. ``"ch3f"``,
            ``"rh-enamide"``).
        max_iter (int): Maximum optimizer iterations.

    Returns:
        list[BenchmarkResult]: One result per (backend, optimizer) combination.

    """
    from q2mm.diagnostics.benchmark import BenchmarkResult, run_benchmark
    from q2mm.diagnostics.pes_distortion import load_normal_modes
    from q2mm.diagnostics.report import detailed_report

    system_cfg = _resolve_system(system_key)

    results: list[BenchmarkResult] = []
    total = len(backends) * len(optimizers)
    idx = 0

    for backend_name, engine_cls, registry_key in backends:
        try:
            # Pass platform to OpenMM when specified
            if registry_key == "openmm" and platform is not None:
                engine = engine_cls(platform_name=platform)
            else:
                engine = engine_cls()
            # Update display name to reflect actual engine config
            backend_name = engine.name
        except Exception as e:
            print(f"  Skipping {backend_name}: {e}", file=sys.stderr)
            continue

        # Load system data with this engine (engine computes MM frequencies)
        try:
            if system_key == "ch3f" and data_dir is not None:
                # CH3F supports data_dir override
                from q2mm.diagnostics.systems import load_ch3f

                sys_data = load_ch3f(engine, data_dir=data_dir)
            else:
                sys_data = system_cfg.loader(engine)
        except Exception as e:
            print(f"  Skipping {backend_name}: cannot load {system_key} data: {e}", file=sys.stderr)
            continue

        molecule_name = sys_data.metadata.get("molecule_name", system_key)
        level_of_theory = sys_data.metadata.get("level_of_theory", "unknown")

        # For single-molecule systems, use run_benchmark directly
        # For multi-molecule systems, use the system's freq_ref + objective
        is_multi = len(sys_data.molecules) > 1

        for opt_label, opt_config in optimizers:
            idx += 1
            combo = f"{backend_name} + {opt_label}"
            print(f"  [{idx}/{total}] {combo} ...", end=" ", flush=True)

            try:
                method = opt_config["method"]
                extra_kwargs = {k: v for k, v in opt_config.items() if k != "method"}

                t0 = time.perf_counter()

                if is_multi:
                    r = _run_multi_molecule_benchmark(
                        engine=engine,
                        sys_data=sys_data,
                        optimizer_method=method,
                        optimizer_kwargs=extra_kwargs,
                        maxiter=max_iter,
                        backend_name=backend_name,
                        molecule_name=molecule_name,
                        level_of_theory=level_of_theory,
                    )
                else:
                    # Load CH3F-specific data for PES distortion support
                    normal_modes = None
                    qm_hessian = None
                    if system_key == "ch3f":
                        try:
                            qm_dir = data_dir or (
                                Path(__file__).resolve().parent.parent.parent / "examples" / "sn2-test" / "qm-reference"
                            )
                            hess_path = qm_dir / "ch3f-hessian.npy"
                            modes_path = qm_dir / "ch3f-normal-modes.npz"
                            if hess_path.exists():
                                qm_hessian = np.load(hess_path)
                            if modes_path.exists():
                                normal_modes = load_normal_modes(modes_path)
                        except Exception:
                            pass

                    r = run_benchmark(
                        engine,
                        sys_data.molecules[0],
                        sys_data.qm_freqs_per_mol[0],
                        qm_hessian=qm_hessian,
                        normal_modes=normal_modes,
                        optimizer_method=method,
                        optimizer_kwargs=extra_kwargs,
                        maxiter=max_iter,
                        backend_name=backend_name,
                        molecule_name=molecule_name,
                        level_of_theory=level_of_theory,
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
                            "molecule": molecule_name,
                            "source": "q2mm",
                            "error": str(e),
                        },
                    )
                )

    # Save results if output directory specified
    if output_dir is not None:
        from q2mm.diagnostics.benchmark import benchmark_stem

        # Use system-specific subdirectory
        system_output = output_dir
        results_dir = system_output / "results"
        ff_dir = system_output / "forcefields"
        results_dir.mkdir(parents=True, exist_ok=True)
        ff_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            stem = benchmark_stem(r.metadata)
            r.to_json(results_dir / f"{stem}.json")
            # For multi-molecule systems, save with first molecule
            mol_for_save = sys_data.molecules[0] if sys_data else None
            if mol_for_save is not None:
                saved_ffs = r.save_forcefields(ff_dir, stem=stem, molecule=mol_for_save)
                if saved_ffs:
                    exts = ", ".join(p.suffix for p in saved_ffs)
                    print(f"    FF saved: {stem} ({exts})")

        print(f"\n  Results saved to: {system_output}/")

    return results


def _run_multi_molecule_benchmark(
    engine: MMEngine,
    sys_data: SystemData,
    optimizer_method: str,
    optimizer_kwargs: dict,
    maxiter: int,
    backend_name: str,
    molecule_name: str,
    level_of_theory: str,
) -> BenchmarkResult:
    """Run a benchmark on a multi-molecule system (e.g. Rh-enamide).

    Uses the pre-built frequency reference from SystemData.
    """
    from q2mm.diagnostics.benchmark import BenchmarkResult, frequency_rmsd, real_frequencies
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    ff = sys_data.forcefield.copy()
    seminario_params = ff.get_param_vector().copy()

    # Compute aggregate QM frequencies for RMSD reporting
    all_qm_real = np.concatenate(sys_data.qm_freqs_per_mol)

    # Compute aggregate MM frequencies for initial RMSD
    all_mm_real = []
    for mol in sys_data.molecules:
        mm_freqs = engine.frequencies(mol, ff)
        mm_real = real_frequencies(mm_freqs)
        all_mm_real.extend(mm_real.tolist())
    all_mm_real = np.array(sorted(all_mm_real))

    n = min(len(all_qm_real), len(all_mm_real))
    initial_rmsd = frequency_rmsd(np.sort(all_qm_real)[:n], all_mm_real[:n])

    result = BenchmarkResult(
        metadata={
            "backend": backend_name,
            "optimizer": optimizer_method,
            "molecule": molecule_name,
            "n_molecules": len(sys_data.molecules),
            "source": "q2mm",
            "level_of_theory": level_of_theory,
        },
    )

    # Optimize
    obj = ObjectiveFunction(ff, engine, sys_data.molecules, sys_data.freq_ref)
    initial_score = obj(seminario_params)

    result.seminario = {
        "rmsd": initial_rmsd,
        "param_values": seminario_params.tolist(),
        "score": initial_score,
    }

    opt_kwargs = {"method": optimizer_method, "maxiter": maxiter, "verbose": False, "jac": "auto"}
    opt_kwargs.update(optimizer_kwargs)
    opt = ScipyOptimizer(**opt_kwargs)

    t0 = time.perf_counter()
    opt_result = opt.optimize(obj)
    opt_elapsed = time.perf_counter() - t0

    # Final aggregate RMSD
    all_mm_real_final = []
    for mol in sys_data.molecules:
        mm_freqs = engine.frequencies(mol, ff)
        mm_real = real_frequencies(mm_freqs)
        all_mm_real_final.extend(mm_real.tolist())
    all_mm_real_final = np.array(sorted(all_mm_real_final))
    n_final = min(len(all_qm_real), len(all_mm_real_final))
    final_rmsd = frequency_rmsd(np.sort(all_qm_real)[:n_final], all_mm_real_final[:n_final])

    result.optimized = {
        "rmsd": final_rmsd,
        "elapsed_s": opt_elapsed,
        "n_eval": opt_result.n_evaluations,
        "converged": opt_result.success,
        "initial_score": opt_result.initial_score,
        "final_score": opt_result.final_score,
        "message": opt_result.message,
        "param_final": ff.get_param_vector().tolist(),
    }
    result.optimized_ff = ff.copy()

    return result


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
    # Ensure UTF-8 output on Windows (cp1252 can't encode →, ⁻¹, etc.)
    if sys.platform == "win32":
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            encoding = (sys.stdout.encoding or "").lower()
            if encoding != "utf-8":
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            encoding = (sys.stderr.encoding or "").lower()
            if encoding != "utf-8":
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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
        default=Path("benchmark_results"),
        help="Save results to this directory (default: ./benchmark_results/). Use --no-save to disable.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save results to disk (overrides --output).",
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
    parser.add_argument(
        "--platform",
        metavar="NAME",
        default=None,
        help="OpenMM platform to use (e.g. CPU, CUDA, OpenCL). Default: auto-detect fastest available.",
    )
    parser.add_argument(
        "--system",
        metavar="NAME",
        default="ch3f",
        help="Benchmark system to run (e.g. ch3f, rh-enamide). Default: ch3f. Use --list to see available systems.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        metavar="N",
        default=10_000,
        help="Maximum optimizer iterations (default: 10000). Use a small value for quick benchmarks.",
    )

    args = parser.parse_args(argv)

    # --no-save suppresses all output saving
    output_dir: Path | None = None if args.no_save else args.output

    all_backends = _discover_backends()
    all_optimizers = _optimizer_configs()

    # --list: show what's available
    if args.list:
        from q2mm.diagnostics.systems import SYSTEMS

        print("\nAvailable systems:")
        for key, sys_cfg in SYSTEMS.items():
            print(f"  {key:<14} {sys_cfg.description}")

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
        # Support both new layout (results/ subdir) and flat layout
        load_dir = args.load / "results" if (args.load / "results").is_dir() else args.load
        results = _load_results(load_dir)
        if not results:
            print(f"No results found in {args.load}", file=sys.stderr)
            return 1
        print(f"Loaded {len(results)} results from {load_dir}\n")
    else:
        # Filter backends
        backends = all_backends
        if args.backend:
            filter_names = {b.lower() for b in args.backend}
            backends = [(n, c, m) for n, c, m in all_backends if m.lower() in filter_names]
            if not backends:
                print(f"Error: no matching backends for {args.backend}", file=sys.stderr)
                print(f"Available: {[m for _, _, m in all_backends]}", file=sys.stderr)
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
        print(f"  System:     {args.system}")
        print(f"  Backends:   {', '.join(n for n, _, _ in backends)}")
        print(f"  Optimizers: {', '.join(l for l, _ in optimizers)}")
        print(f"  Combos:     {len(backends) * len(optimizers)}\n")

        results = _run_matrix(
            backends,
            optimizers,
            output_dir=output_dir,
            leaderboard_only=args.leaderboard_only,
            data_dir=args.data_dir,
            platform=args.platform,
            system_key=args.system,
            max_iter=args.max_iter,
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
