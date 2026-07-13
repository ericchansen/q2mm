"""Command-line interface for Q2MM benchmarking.

Usage::

    q2mm-benchmark                              # Run CH3F (default) across all backends
    q2mm-benchmark --system rh-enamide          # Run Rh-enamide (9 molecules)
    q2mm-benchmark --backend openmm             # Only OpenMM backend
    q2mm-benchmark --optimizer scipy-lbfgsb      # Only SciPy L-BFGS-B optimizer
    q2mm-benchmark --output results/            # Save to custom directory
    q2mm-benchmark --no-save                    # Run without saving results
    q2mm-benchmark --load results/              # Load saved results and print report
    q2mm-benchmark --list                       # Show available backends, optimizers, systems
    q2mm-benchmark --platform CUDA              # Force OpenMM CUDA platform
    q2mm-benchmark --preflight                  # Check GPU/platform environment

By default, results (JSON + force field files) are saved to
``./results/``.  Use ``--no-save`` to disable.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.diagnostics.benchmark import BenchmarkResult
    from q2mm.systems import SystemSpec


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


def _optimizer_configs() -> list[tuple[str, str, dict]]:
    """Build the optimizer configuration list.

    Returns:
        list[tuple[str, str, dict]]: ``(key, label, config_dict)`` tuples.
            *key* is a unique kebab-case slug used for ``--optimizer``
            matching.  *label* is a human-readable display name for
            leaderboard output.  *config_dict* contains at minimum a
            ``'method'`` key.

    """
    configs: list[tuple[str, str, dict]] = [
        # SciPy optimizers
        ("scipy-lbfgsb", "SciPy L-BFGS-B", {"method": "L-BFGS-B"}),
        ("scipy-lbfgsb-jax", "SciPy L-BFGS-B (JAX direct)", {"method": "L-BFGS-B", "jac": "auto"}),
        ("scipy-lbfgsb-fd", "SciPy L-BFGS-B (FD)", {"method": "L-BFGS-B", "jac": None}),
        ("scipy-nm", "Nelder-Mead", {"method": "Nelder-Mead"}),
        ("scipy-powell", "Powell", {"method": "Powell"}),
        # --- Cycling ---
        ("grad-simp", "Grad-Simp", {"method": "cycling"}),
        ("grad-simp-auto", "Grad-Simp (auto jac)", {"method": "cycling", "full_jac": "auto"}),
        # --- Optax ---
        ("optax-adam", "Optax Adam", {"method": "optax:adam"}),
        ("optax-adam-cosine", "Optax Adam+cosine", {"method": "optax:adam", "schedule": "cosine"}),
        ("optax-adagrad", "Optax AdaGrad", {"method": "optax:adagrad"}),
        ("optax-sgd", "Optax SGD", {"method": "optax:sgd"}),
        # Global / multi-start
        ("basinhopping", "Basin-hopping (T=1.0)", {"method": "basinhopping", "niter": 25}),
        ("basinhopping-cold", "Basin-hopping (T=0.5)", {"method": "basinhopping", "niter": 25, "T": 0.5}),
        ("multi-lbfgsb-5", "Multi-start n=5", {"method": "multi:L-BFGS-B", "n_starts": 5}),
        ("multi-lbfgsb-10", "Multi-start n=10", {"method": "multi:L-BFGS-B", "n_starts": 10}),
        # Regularized variants
        ("scipy-lbfgsb-l2", "SciPy L-BFGS-B+L2", {"method": "L-BFGS-B", "regularization": 0.01}),
        ("optax-adam-l2", "Optax Adam+L2", {"method": "optax:adam", "regularization": 0.01}),
        # JaxOpt (end-to-end differentiable pipeline)
        # Note: run_combo auto-applies regularization=0.01 for jaxopt:lbfgs
        # when not explicitly set, preventing unbounded parameter drift.
        ("jaxopt-lbfgs", "JaxOpt L-BFGS", {"method": "jaxopt:lbfgs"}),
        ("jaxopt-lbfgsb", "JaxOpt L-BFGS-B", {"method": "jaxopt:lbfgsb"}),
        # Composed workflows
        ("grad-simp-multi", "Grad-Simp (multi inner)", {"method": "cycling", "full_method": "multi:L-BFGS-B"}),
    ]
    return configs


def _functional_form_configs() -> list[tuple[str, str]]:
    """Build the list of functional forms to benchmark.

    Returns:
        list[tuple[str, str]]: ``(display_label, form_value)`` tuples.
            ``form_value`` matches :class:`FunctionalForm` enum values.

    """
    from q2mm.models.forcefield import FunctionalForm

    return [
        ("Harmonic", FunctionalForm.HARMONIC.value),
        ("MM3", FunctionalForm.MM3.value),
    ]


def _resolve_system(system_key: str) -> SystemSpec:
    """Look up a benchmark system by key.

    Args:
        system_key: System name (e.g. ``"ch3f"``, ``"rh-enamide"``).

    Returns:
        SystemSpec: The system configuration.

    Raises:
        SystemExit: If the system is not registered.

    """
    from q2mm.systems import SYSTEMS

    if system_key not in SYSTEMS:
        available = ", ".join(sorted(SYSTEMS))
        raise SystemExit(f"Error: unknown system {system_key!r}. Available: {available}")
    return SYSTEMS[system_key]


def _run_matrix(
    backends: list[tuple[str, type, str]],
    optimizers: list[tuple[str, str, dict]],
    forms: list[tuple[str, str]],
    output_dir: Path | None = None,
    *,
    leaderboard_only: bool = False,
    data_dir: Path | None = None,
    platform: str | None = None,
    system_key: str = "ch3f",
    max_iter: int | None = None,
    starting_point: str = "qfuerza",
) -> list:
    """Run the full backend × form × optimizer matrix.

    Args:
        backends (list[tuple[str, type, str]]): Backend entries from
            ``_discover_backends()``.
        optimizers (list[tuple[str, str, dict]]): Optimizer entries from
            ``_optimizer_configs()``.  Each tuple is
            ``(key, label, config_dict)``.
        forms (list[tuple[str, str]]): Form entries from
            ``_functional_form_configs()``, filtered by ``--form``.
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
        max_iter (int | None): Maximum optimizer iterations.  ``None``
            lets each optimizer use its own default.
        starting_point (str): Starting FF parameter values; one of
            ``"qfuerza"`` (canonical default, Farrugia 2025) or
            ``"published"`` (literature OPT values verbatim).  No-op for
            ``ch3f`` (qfuerza_fresh strategy).

    Returns:
        list[BenchmarkResult]: One result per (backend, form, optimizer)
            combination.

    """
    from q2mm.diagnostics.benchmark import BenchmarkResult, run_combo
    from q2mm.diagnostics.report import detailed_report

    system_cfg = _resolve_system(system_key)

    results: list[BenchmarkResult] = []
    n_success = 0
    n_failed = 0
    n_skipped = 0
    idx = 0

    # Create output directories up front so results can be saved
    # incrementally — if the process is killed, completed combos
    # are already on disk.
    results_dir: Path | None = None
    ff_dir: Path | None = None
    if output_dir is not None:
        from q2mm.diagnostics.benchmark import benchmark_stem

        results_dir = output_dir / "results"
        ff_dir = output_dir / "forcefields"
        results_dir.mkdir(parents=True, exist_ok=True)
        ff_dir.mkdir(parents=True, exist_ok=True)

    for backend_name, engine_cls, registry_key in backends:
        try:
            if registry_key == "openmm" and platform is not None:
                engine = engine_cls(platform_name=platform)
            else:
                engine = engine_cls()
            backend_name = engine.name
        except Exception as e:
            print(f"  Skipping {backend_name}: {e}")
            print(f"  Skipping {backend_name}: {e}", file=sys.stderr)
            n_skipped += 1
            continue

        # Determine which forms this engine supports
        supported = getattr(engine, "supported_functional_forms", lambda: frozenset())()

        for form_label, form_value in forms:
            if supported and form_value not in supported:
                continue

            # Reload system data per-form (reference data depends on form)
            try:
                from q2mm.systems import load_system

                molecule_loader_kwargs: dict[str, Any] | None = None
                if system_key in {"ch3f", "ch3f-sn2"} and data_dir is not None:
                    molecule_loader_kwargs = {"data_dir": data_dir}
                sys_data = load_system(
                    system_key,
                    engine=engine,
                    functional_form=form_value,
                    molecule_loader_kwargs=molecule_loader_kwargs,
                    starting_point=starting_point,
                )
            except Exception as e:
                print(
                    f"  Skipping {backend_name}/{form_label}: cannot load {system_key} data: {e}",
                )
                print(
                    f"  Skipping {backend_name}/{form_label}: cannot load {system_key} data: {e}",
                    file=sys.stderr,
                )
                n_skipped += 1
                continue

            molecule_name = sys_data.metadata.get("molecule_name", system_key)

            for opt_key, opt_label, opt_config in optimizers:
                idx += 1
                combo = f"{backend_name} + {form_label} + {opt_label}"
                print(f"  [{idx}] {combo} ...", end=" ", flush=True)

                try:
                    method = opt_config["method"]
                    extra_kwargs = {k: v for k, v in opt_config.items() if k != "method"}

                    t0 = time.perf_counter()

                    r = run_combo(
                        engine,
                        sys_data,
                        optimizer_method=method,
                        optimizer_kwargs=extra_kwargs,
                        maxiter=max_iter,
                        backend_name=backend_name,
                    )
                    # Tag the result with display-facing labels
                    r.metadata["functional_form"] = form_value
                    r.metadata["optimizer"] = opt_label
                    r.metadata["optimizer_key"] = opt_key

                    elapsed = time.perf_counter() - t0
                    results.append(r)
                    n_success += 1
                    # Save immediately so kills don't lose completed work
                    if results_dir is not None and ff_dir is not None:
                        try:
                            stem = benchmark_stem(r.metadata)
                            r.to_json(results_dir / f"{stem}.json")
                            saved_ffs = r.save_forcefields(
                                ff_dir,
                                stem=stem,
                                molecule=sys_data.molecules[0],
                            )
                            if saved_ffs:
                                exts = ", ".join(p.suffix for p in saved_ffs)
                                print(f"    FF saved: {stem} ({exts})")
                        except OSError as save_err:
                            print(f"    Warning: save failed: {save_err}", file=sys.stderr)

                    opt = r.optimized or {}
                    rmsd = opt.get("rmsd", float("nan"))

                    start_rmsd = None
                    if r.seminario and r.seminario.get("rmsd") is not None:
                        start_rmsd = r.seminario["rmsd"]

                    if start_rmsd is not None:
                        print(f"RMSD {start_rmsd:.0f}→{rmsd:.0f}  ({elapsed:.1f}s)")
                    else:
                        print(f"RMSD={rmsd:.1f}  ({elapsed:.1f}s)")

                    if not leaderboard_only:
                        detail_tables = detailed_report(r, combo_label=combo)
                        # Write detailed tables (per-frequency, per-parameter)
                        # to a log file to keep console output clean.
                        if results_dir is not None:
                            log_path = results_dir / f"{stem}_detail.txt"
                            with open(log_path, "w", encoding="utf-8") as f:
                                for table in detail_tables:
                                    f.write(table.to_string() + "\n")
                        else:
                            for table in detail_tables:
                                table.flush()

                except Exception as e:
                    print(f"FAILED: {e}")
                    print(f"FAILED: {e}", file=sys.stderr)
                    n_failed += 1

                    # Best-effort GPU memory recovery after JAX OOM
                    if "RESOURCE_EXHAUSTED" in str(e):
                        try:
                            import jax

                            jax.clear_caches()
                            print("  (cleared JAX caches — consider using --optimizer jaxopt-lbfgs for GPU)")
                        except ImportError:
                            pass

                    err_result = BenchmarkResult(
                        metadata={
                            "backend": backend_name,
                            "optimizer": opt_label,
                            "optimizer_key": opt_key,
                            "functional_form": form_value,
                            "molecule": molecule_name,
                            "source": "q2mm",
                            "error": str(e),
                        },
                    )
                    results.append(err_result)

                    # Save error results incrementally too
                    if results_dir is not None:
                        stem = benchmark_stem(err_result.metadata)
                        err_result.to_json(results_dir / f"{stem}.json")

    # Summary: report output location
    if output_dir is not None:
        print(f"\n  Results saved to: {output_dir}/")

    print(f"\nMatrix complete: {n_success} succeeded, {n_failed} failed, {n_skipped} skipped")
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


def _run_preflight() -> None:
    """Print GPU/platform environment diagnostics and return."""
    from q2mm import __version__

    print("=== Q2MM Benchmark Pre-flight Check ===\n")

    # --- System info ---
    print(f"Python:   {sys.version.split()[0]}")
    print(f"OS:       {platform.platform()}")
    print(f"q2mm:     {__version__}\n")

    # --- OpenMM platforms ---
    has_openmm_cuda = False
    has_opencl = False
    has_openmm = False
    try:
        import openmm as mm

        has_openmm = True
        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        print(f"OpenMM platforms: {', '.join(names)}")
        has_openmm_cuda = "CUDA" in names
        has_opencl = "OpenCL" in names

        if has_openmm_cuda:
            print("\u2705 OpenMM CUDA: available")
        elif has_opencl:
            print("\u26a0\ufe0f  OpenMM CUDA: NOT available (only OpenCL \u2014 very slow, ~14% GPU utilization)")
        else:
            print("\u274c OpenMM GPU: NOT available")
    except ImportError:
        print("\u2139\ufe0f  OpenMM: not installed")
    except Exception as exc:
        has_openmm = True
        print(f"\u26a0\ufe0f  OpenMM: installed but platform probe failed: {exc}")
    print()

    # --- JAX devices ---
    has_jax_cuda = False
    jax_installed = False
    try:
        import jax

        jax_installed = True
        devices = jax.devices()
        device_strs = [str(d) for d in devices]
        print(f"JAX devices: {', '.join(device_strs)}")
        has_jax_cuda = any("cuda" in s.lower() for s in device_strs)
        if has_jax_cuda:
            print("\u2705 JAX CUDA: available")
        else:
            print("\u26a0\ufe0f  JAX CUDA: NOT available (CPU only)")
    except ImportError:
        print("\u2139\ufe0f  JAX: not installed")
    except Exception as exc:
        jax_installed = True
        print(f"\u26a0\ufe0f  JAX: installed but device probe failed: {exc}")
    print()

    # --- GPU info via nvidia-smi ---
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    print(f"GPU:      {parts[0]}")
                    print(f"Driver:   {parts[1]}")
                    print(f"GPU util: {parts[2]}%")
                else:
                    print(f"GPU:      {line.strip()}")
        else:
            print("GPU: not detected")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("GPU: not detected (nvidia-smi not found)")
    print()

    # --- Recommendation ---
    if has_openmm_cuda and has_jax_cuda:
        print("\u2705 Ready for GPU benchmarks")
    elif not has_openmm_cuda or not has_jax_cuda:
        missing: list[str] = []
        if not has_openmm_cuda and has_openmm:
            missing.append("OpenMM CUDA")
        if not has_jax_cuda and jax_installed:
            missing.append("JAX CUDA")
        if missing:
            print(f"\u26a0\ufe0f  GPU benchmarks will be slow or unavailable ({', '.join(missing)} missing).")
            print("  Consider using WSL2.")
        else:
            print("\u26a0\ufe0f  GPU benchmarks will be slow or unavailable. Consider using WSL2.")
        if sys.platform == "win32" and not has_jax_cuda and jax_installed:
            print("\u2139\ufe0f  JAX CUDA is not available on Windows. Use WSL2 for full GPU stack.")
    print()


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
        metavar="KEY",
        help="Run only these optimizers by key (e.g. jaxopt-lbfgs optax-adam). "
        "Use --list to see available keys. Default: all.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        metavar="DIR",
        default=Path("results"),
        help="Save results to this directory (default: ./results/). Use --no-save to disable.",
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
        default=None,
        help="Override optimizer iteration limit. If not specified, each optimizer "
        "uses its own default: JaxOpt 200, Optax 2000, SciPy 500, cycling 200/pass.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check GPU/platform environment and exit without running benchmarks.",
    )
    parser.add_argument(
        "--form",
        nargs="*",
        metavar="NAME",
        help="Run only these functional forms (e.g. harmonic mm3). Default: system-specific.",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        metavar="N",
        default=3,
        help="Cycling optimizer: parameters per simplex pass (default: 3).",
    )
    parser.add_argument(
        "--max-cycles",
        type=int,
        metavar="N",
        default=10,
        help="Cycling optimizer: maximum grad-simp cycles (default: 10).",
    )
    parser.add_argument(
        "--convergence",
        type=float,
        metavar="FLOAT",
        default=0.01,
        help="Cycling optimizer: fractional improvement threshold (default: 0.01).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        metavar="FLOAT",
        default=0.001,
        help="Optax optimizer: learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--optax-max-steps",
        type=int,
        metavar="N",
        default=2000,
        help="Optax optimizer: maximum gradient steps (default: 2000).",
    )
    parser.add_argument(
        "--starting-point",
        choices=("published", "qfuerza"),
        default="qfuerza",
        help="Starting force-field parameters for TS systems. 'qfuerza' "
        "(canonical default) overwrites OPT bond/angle scalars with QFUERZA "
        "(Farrugia 2025) Hessian-derived values; 'published' retains the "
        "literature OPT values verbatim — pass this to reproduce historical "
        "publication-baseline benchmarks. No-op for ch3f (qfuerza_fresh "
        "strategy, which is already QFUERZA-derived).",
    )

    args = parser.parse_args(argv)

    # --preflight: run environment check and exit
    if args.preflight:
        _run_preflight()
        return 0

    # --no-save suppresses all output saving
    output_dir: Path | None = None if args.no_save else args.output

    all_backends = _discover_backends()
    all_optimizers = _optimizer_configs()
    all_forms = _functional_form_configs()

    # --list: show what's available
    if args.list:
        from q2mm.systems import SYSTEMS

        print("\nAvailable systems:")
        for key, sys_cfg in SYSTEMS.items():
            forms_str = ", ".join(sys_cfg.default_forms)
            print(f"  {key:<14} {sys_cfg.description}  [forms: {forms_str}]")

        print("\nAvailable backends:")
        if all_backends:
            for name, engine_cls, marker in all_backends:
                try:
                    eng = engine_cls()
                    supported = eng.supported_functional_forms()
                    forms_str = ", ".join(sorted(supported))
                except Exception:
                    forms_str = "unknown"
                print(f"  {name:<12} (marker: {marker}, forms: {forms_str})")
        else:
            print("  (none detected)")

        print("\nAvailable functional forms:")
        for label, value in all_forms:
            print(f"  {label:<12} ({value})")

        print("\nAvailable optimizers:")
        print(f"  {'KEY':<22} {'LABEL':<28} METHOD")
        print(f"  {'---':<22} {'-----':<28} ------")
        for key, label, config in all_optimizers:
            print(f"  {key:<22} {label:<28} {config.get('method', '?')}")

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
        # Resolve system config for default_forms
        system_cfg = _resolve_system(args.system)

        # Filter backends
        backends = all_backends
        if args.backend:
            filter_names = {b.lower() for b in args.backend}
            backends = [(n, c, m) for n, c, m in all_backends if m.lower() in filter_names]
            if not backends:
                print(f"Error: no matching backends for {args.backend}", file=sys.stderr)
                print(f"Available: {[m for _, _, m in all_backends]}", file=sys.stderr)
                return 1

        # Filter optimizers by unique key (exact match only)
        optimizers = all_optimizers
        if args.optimizer:
            filter_keys = {o.lower() for o in args.optimizer}
            optimizers = [(k, l, c) for k, l, c in all_optimizers if k.lower() in filter_keys]
            if not optimizers:
                print(f"Error: no matching optimizers for {args.optimizer}", file=sys.stderr)
                print(f"Available keys: {', '.join(k for k, _, _ in all_optimizers)}", file=sys.stderr)
                return 1

        # Inject cycling-specific CLI args into the cycling optimizer config
        cycling_kwargs = {
            "max_params": args.max_params,
            "max_cycles": args.max_cycles,
            "convergence": args.convergence,
        }
        optimizers = [
            (k, l, {**c, **cycling_kwargs}) if c.get("method") == "cycling" else (k, l, c) for k, l, c in optimizers
        ]

        # Inject optax-specific CLI args into optax optimizer configs.
        # --max-iter takes precedence over --optax-max-steps when provided.
        optax_max = args.max_iter if args.max_iter is not None else args.optax_max_steps
        optax_kwargs = {
            "learning_rate": args.learning_rate,
            "max_steps": optax_max,
        }
        optimizers = [
            (k, l, {**c, **optax_kwargs})
            if isinstance(c.get("method"), str) and c["method"].startswith("optax:")
            else (k, l, c)
            for k, l, c in optimizers
        ]

        # Filter forms: --form overrides system defaults
        if args.form:
            filter_names = {f.lower() for f in args.form}
            forms = [(l, v) for l, v in all_forms if v.lower() in filter_names]
            if not forms:
                print(f"Error: no matching forms for {args.form}", file=sys.stderr)
                print(f"Available: {[v for _, v in all_forms]}", file=sys.stderr)
                return 1
        else:
            # Use system's default forms
            forms = [(l, v) for l, v in all_forms if v in system_cfg.default_forms]

        print("\nQ2MM Benchmark Matrix")
        print(f"  System:     {args.system}")
        print(f"  Backends:   {', '.join(n for n, _, _ in backends)}")
        print(f"  Forms:      {', '.join(l for l, _ in forms)}")
        print(f"  Optimizers: {', '.join(l for _, l, _ in optimizers)}")
        print(f"  Max combos: {len(backends) * len(forms) * len(optimizers)}")
        print("  (combos filtered by engine support)\n")

        results = _run_matrix(
            backends,
            optimizers,
            forms,
            output_dir=output_dir,
            leaderboard_only=args.leaderboard_only,
            data_dir=args.data_dir,
            platform=args.platform,
            system_key=args.system,
            max_iter=args.max_iter,
            starting_point=args.starting_point,
        )

    if not results:
        print("No results to report.", file=sys.stderr)
        return 1

    # Exit non-zero if every result was an error
    has_success = any(r.optimized is not None for r in results)
    if not has_success:
        print("All benchmark combos failed.", file=sys.stderr)
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
