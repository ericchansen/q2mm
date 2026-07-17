"""The one ``q2mm-benchmark`` command-line entry point.

Explicit, current-only subcommands compose
:class:`~q2mm.benchmarks.profiles.RunProfile` objects and hand them to the
single :func:`~q2mm.benchmarks.runner.run_profiles` execution path:

- ``list`` — enumerate systems, backends, functional forms, and optimizers
  using only side-effect-free catalog probes (no device initialization).
- ``preflight`` — explicitly probe GPU/JAX/OpenMM device availability.
- ``single`` — run one (system, backend, form, optimizer) profile.
- ``batch`` — run the convergence workflow across one or more systems.
- ``matrix`` — run a backend x form x optimizer matrix for a system.
- ``load`` — load and summarise previously persisted candidate records.

Every run persists one immutable record per candidate (accepted, rejected,
skipped, or errored) and promotes only accepted candidates to canonical
result and force-field names.
"""

from __future__ import annotations

import argparse
import logging
import platform
import sys
from pathlib import Path

from q2mm.benchmarks.acceptance import AcceptancePolicy, CandidateStatus
from q2mm.benchmarks.profiles import FUNCTIONAL_FORMS, OPTIMIZER_CATALOG, RunProfile
from q2mm.benchmarks.publications import KNOWN_OBJECTIVE_PROFILES
from q2mm.benchmarks.runner import RunOutcome, load_candidates, run_profiles

__all__ = ["main"]

logger = logging.getLogger("q2mm.benchmarks.cli")


# ---------------------------------------------------------------------------
# list (side-effect free)
# ---------------------------------------------------------------------------


def _cmd_list(_args: argparse.Namespace) -> int:
    from q2mm.backends.contracts import BackendRole
    from q2mm.backends.registry import catalog, discovery_report
    from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata

    print("\nSystems:")
    for key in SYSTEM_KEYS:
        meta = system_metadata(key)
        print(f"  {key:<14} {meta.description}  [forms: {', '.join(meta.default_forms)}]")

    print("\nBackends (cheap probe, no device init):")
    for status in catalog(role=BackendRole.MM):
        desc = status.descriptor
        forms = ", ".join(sorted(desc.functional_form_ceiling)) or "n/a"
        health = "available" if status.healthy else f"unavailable ({status.reason})"
        print(f"  {desc.name:<14} {health}  [forms: {forms}]")

    # Surface isolated plugin-discovery issues (rejected/unavailable plugins)
    # without letting them break the listing above.  Cheap: probes only.
    issues = [record for record in discovery_report().issues if not record.registered]
    if issues:
        print("\nDiscovery issues (isolated; healthy backends unaffected):")
        for record in issues:
            source = record.entry_point or record.distribution or record.source.value
            label = record.name or source
            print(f"  {label:<14} {record.issue.value if record.issue else 'unknown'}: {record.message}")

    print("\nFunctional forms:")
    for form in FUNCTIONAL_FORMS:
        print(f"  {form}")

    print("\nPublication objective profiles:")
    for profile in sorted(KNOWN_OBJECTIVE_PROFILES):
        print(f"  {profile}")

    print("\nOptimizers:")
    print(f"  {'KEY':<20} {'LABEL':<28} METHOD (evaluator)")
    for spec in OPTIMIZER_CATALOG.values():
        print(f"  {spec.key:<20} {spec.label:<28} {spec.method} ({spec.evaluator})")
    print()
    return 0


# ---------------------------------------------------------------------------
# preflight (explicit device probe)
# ---------------------------------------------------------------------------


def _cmd_preflight(_args: argparse.Namespace) -> int:
    from q2mm import __version__

    print("=== Q2MM Benchmark Pre-flight Check ===\n")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"OS:       {platform.platform()}")
    print(f"q2mm:     {__version__}\n")

    has_openmm_cuda = has_openmm = jax_installed = has_jax_cuda = False
    try:
        import openmm as mm

        names = [mm.Platform.getPlatform(i).getName() for i in range(mm.Platform.getNumPlatforms())]
        has_openmm = True
        has_openmm_cuda = "CUDA" in names
        print(f"OpenMM platforms: {', '.join(names)}")
        if has_openmm_cuda:
            print("OpenMM CUDA: available")
        elif "OpenCL" in names:
            print("OpenMM CUDA: NOT available (only OpenCL - slow, ~14% GPU utilization)")
        else:
            print("OpenMM GPU: NOT available")
    except ImportError:
        print("OpenMM: not installed")
    except Exception as exc:  # pragma: no cover - environment dependent
        has_openmm = True
        print(f"OpenMM: installed but platform probe failed: {exc}")
    print()

    try:
        import jax

        jax_installed = True
        device_strs = [str(d) for d in jax.devices()]
        has_jax_cuda = any("cuda" in s.lower() for s in device_strs)
        print(f"JAX devices: {', '.join(device_strs)}")
        print("JAX CUDA: available" if has_jax_cuda else "JAX CUDA: NOT available (CPU only)")
    except ImportError:
        print("JAX: not installed")
    except Exception as exc:  # pragma: no cover - environment dependent
        jax_installed = True
        print(f"JAX: installed but device probe failed: {exc}")
    print()

    if has_openmm_cuda and has_jax_cuda:
        print("Ready for GPU benchmarks.")
    else:
        missing = []
        if has_openmm and not has_openmm_cuda:
            missing.append("OpenMM CUDA")
        if jax_installed and not has_jax_cuda:
            missing.append("JAX CUDA")
        note = f" ({', '.join(missing)} missing)" if missing else ""
        print(f"GPU benchmarks will be slow or unavailable{note}. Consider WSL2.")
        if sys.platform == "win32" and jax_installed and not has_jax_cuda:
            print("Note: JAX CUDA is not available on Windows; use WSL2 for the full GPU stack.")
    print()
    return 0


# ---------------------------------------------------------------------------
# Shared profile construction
# ---------------------------------------------------------------------------


def _data_root_pair(value: str) -> tuple[str, str]:
    """Parse a ``KEY=PATH`` external-data-root argument."""
    from q2mm.benchmarks.profiles import DATA_ROOT_KEYS

    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--data-root must be KEY=PATH, got {value!r}")
    key, path = value.split("=", 1)
    key = key.strip()
    if key not in DATA_ROOT_KEYS:
        raise argparse.ArgumentTypeError(f"--data-root key must be one of {sorted(DATA_ROOT_KEYS)}, got {key!r}")
    if not path.strip():
        raise argparse.ArgumentTypeError(f"--data-root {key!r} needs a non-empty path")
    return key, path.strip()


def _data_roots(args: argparse.Namespace) -> dict[str, str]:
    return dict(getattr(args, "data_root", None) or [])


def _optimizer_knobs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "maxiter": args.maxiter,
        "ftol": args.ftol,
        "fc_fraction": args.fc_fraction,
        "eq_fraction": args.eq_fraction,
        "regularization": args.regularization,
        "learning_rate": args.learning_rate,
        "max_params": args.max_params,
        "max_cycles": args.max_cycles,
        "convergence": args.convergence,
        "n_evals": args.n_evals,
        "executor_ratio_tol": args.executor_ratio_tol,
        "qfuerza_replace_with": args.qfuerza_replace_with,
        "seed": args.seed,
        "data_roots": _data_roots(args),
    }


def _summarise(outcome: RunOutcome) -> None:
    counts = {status: len(outcome.by_status(status)) for status in CandidateStatus}
    print("\nCandidate outcomes:")
    for status in CandidateStatus:
        print(f"  {status.value:<9} {counts[status]}")
    for candidate in outcome.candidates:
        summary = candidate.summary
        line = f"  [{candidate.status.value:<8}] {candidate.candidate_id}"
        if "improvement_pct" in summary:
            line += f"  improvement={summary['improvement_pct']:.2f}%"
        if candidate.status is not CandidateStatus.ACCEPTED:
            line += f"  ({candidate.reason})"
        print(line)


def _cmd_single(args: argparse.Namespace) -> int:
    profile = RunProfile(
        system=args.system,
        backend=args.backend,
        functional_form=args.form,
        starting_point=args.starting_point,
        objective_profile=args.objective_profile,
        workflow=args.workflow,
        optimizer=args.optimizer,
        skip_optimization=args.skip_optimization,
        platform=args.platform,
        label=args.label or "",
        **_optimizer_knobs(args),  # type: ignore[arg-type]
    )
    outcome = run_profiles(
        [profile],
        output_dir=args.output,
        generator="q2mm-benchmark single",
        policy=AcceptancePolicy(),
        analyze=not args.no_analyze,
        promote=not args.no_promote,
    )
    _summarise(outcome)
    print(f"\nResults saved under: {args.output}")
    return 0 if outcome.ok else 1


def _cmd_batch(args: argparse.Namespace) -> int:
    from q2mm.benchmarks.systems import SYSTEM_KEYS

    systems = args.system or list(SYSTEM_KEYS)
    unknown = [s for s in systems if s not in SYSTEM_KEYS]
    if unknown:
        print(f"Error: unknown system(s) {unknown}. Available: {sorted(SYSTEM_KEYS)}", file=sys.stderr)
        return 1
    profiles = [
        RunProfile(
            system=system,
            backend=args.backend,
            functional_form=args.form,
            starting_point=args.starting_point,
            objective_profile=args.objective_profile,
            workflow=args.workflow,
            optimizer=args.optimizer,
            skip_optimization=args.skip_optimization,
            platform=args.platform,
            **_optimizer_knobs(args),  # type: ignore[arg-type]
        )
        for system in systems
    ]
    outcome = run_profiles(
        profiles,
        output_dir=args.output,
        generator="q2mm-benchmark batch",
        analyze=args.analyze,
        promote=not args.no_promote,
    )
    _summarise(outcome)
    print(f"\nResults saved under: {args.output}")
    return 0 if outcome.ok else 1


def _cmd_matrix(args: argparse.Namespace) -> int:
    from q2mm.backends.registry import available_mm_backends
    from q2mm.benchmarks.systems import SYSTEM_KEYS, system_metadata

    if args.system not in SYSTEM_KEYS:
        print(f"Error: unknown system {args.system!r}. Available: {sorted(SYSTEM_KEYS)}", file=sys.stderr)
        return 1

    backends = args.backend or available_mm_backends()
    if not backends:
        print("Error: no MM backends available.", file=sys.stderr)
        return 1
    forms = args.form or list(system_metadata(args.system).default_forms)
    optimizers = args.optimizer or list(OPTIMIZER_CATALOG)
    unknown_opt = [o for o in optimizers if o not in OPTIMIZER_CATALOG]
    if unknown_opt:
        print(f"Error: unknown optimizer(s) {unknown_opt}. Available: {sorted(OPTIMIZER_CATALOG)}", file=sys.stderr)
        return 1

    profiles = [
        RunProfile(
            system=args.system,
            backend=backend,
            functional_form=form,
            starting_point=args.starting_point,
            objective_profile=args.objective_profile,
            workflow="single-stage",
            optimizer=optimizer,
            skip_optimization=args.skip_optimization,
            platform=args.platform,
            **_optimizer_knobs(args),  # type: ignore[arg-type]
        )
        for backend in backends
        for form in forms
        for optimizer in optimizers
    ]
    print(
        f"Matrix: system={args.system} backends={backends} forms={forms} "
        f"optimizers={len(optimizers)} -> {len(profiles)} candidates"
    )
    outcome = run_profiles(
        profiles,
        output_dir=args.output,
        generator="q2mm-benchmark matrix",
        analyze=not args.no_analyze,
        promote=not args.no_promote,
    )
    _summarise(outcome)
    print(f"\nResults saved under: {args.output}")
    return 0 if outcome.ok else 1


def _cmd_load(args: argparse.Namespace) -> int:
    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        return 1
    candidates = load_candidates(directory)
    if not candidates:
        print(f"No candidate records found under {directory}", file=sys.stderr)
        return 1
    print(f"Loaded {len(candidates)} candidate record(s) from {directory}\n")
    for cand in candidates:
        summary = cand.summary
        line = f"  [{cand.status.value:<8}] {cand.candidate_id}"
        if "improvement_pct" in summary:
            line += f"  improvement={summary['improvement_pct']}"
        if cand.reason:
            line += f"  ({cand.reason})"
        print(line)
    return 0


# ---------------------------------------------------------------------------
# Argument wiring
# ---------------------------------------------------------------------------


def _add_optimizer_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--maxiter", type=int, default=None, help="Optimizer iteration cap (default: optimizer-specific)."
    )
    parser.add_argument("--ftol", type=float, default=1e-8, help="L-BFGS-B function-value tolerance.")
    parser.add_argument("--fc-fraction", type=float, default=None, help="Fractional bounds for force-constant params.")
    parser.add_argument("--eq-fraction", type=float, default=None, help="Fractional bounds for equilibrium params.")
    parser.add_argument(
        "--regularization",
        type=float,
        default=None,
        help="L2 penalty on the objective plan (overrides the optimizer catalog default; "
        "pass 0 to disable an optimizer's L2 preset). Default: catalog value.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Optax learning rate.")
    parser.add_argument("--max-params", type=int, default=3, help="Cycling optimizer: params per simplex pass.")
    parser.add_argument("--max-cycles", type=int, default=10, help="Cycling optimizer: maximum cycles.")
    parser.add_argument(
        "--convergence", type=float, default=0.01, help="Cycling optimizer: fractional-improvement threshold."
    )
    parser.add_argument("--n-evals", type=int, default=1, help="Post-hoc real-objective samples at each endpoint.")
    parser.add_argument(
        "--executor-ratio-tol",
        type=_ratio_tol,
        default=None,
        help="JAX/Python executor score-ratio gate tolerance. Use 'none' to disable (default).",
    )
    parser.add_argument(
        "--qfuerza-replace-with",
        type=float,
        default=1.0,
        help="Replacement value (Hartree/Bohr^2) for the negative TS-Hessian eigenvalue during QFUERZA setup.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Deterministic seed for stochastic optimizers.")
    parser.add_argument(
        "--data-root",
        type=_data_root_pair,
        action="append",
        metavar="KEY=PATH",
        help="External data root (repeatable). KEY in {ch3f, rh_enamide, supporting_info, mm3_base}.",
    )


def _ratio_tol(value: str) -> float | None:
    if value.lower() in {"none", "off", "disabled"}:
        return None
    parsed = float(value)
    return None if parsed < 0 else parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="q2mm-benchmark", description=__doc__.splitlines()[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List systems, backends, forms, and optimizers.").set_defaults(func=_cmd_list)
    sub.add_parser("preflight", help="Probe GPU/JAX/OpenMM device availability.").set_defaults(func=_cmd_preflight)

    single = sub.add_parser("single", help="Run one (system, backend, form, optimizer) profile.")
    single.add_argument("--system", required=True, help="Benchmark system key (e.g. ch3f, rh-enamide).")
    single.add_argument("--backend", default="jax", help="MM backend key (default: jax).")
    single.add_argument(
        "--form", default=None, choices=FUNCTIONAL_FORMS, help="Functional form (default: system default)."
    )
    single.add_argument(
        "--optimizer", default="scipy-lbfgsb-jax", choices=sorted(OPTIMIZER_CATALOG), help="Optimizer key."
    )
    single.add_argument("--workflow", default="single-stage", choices=("single-stage", "method-e2"), help="Workflow.")
    single.add_argument("--starting-point", default="qfuerza", choices=("qfuerza", "published"), help="Starting FF.")
    single.add_argument(
        "--objective-profile",
        default=None,
        choices=sorted(KNOWN_OBJECTIVE_PROFILES),
        help="Publication objective/completeness profile (default: repository compatibility for publication systems).",
    )
    single.add_argument("--skip-optimization", action="store_true", help="Compute baseline metrics only.")
    single.add_argument("--platform", default=None, help="OpenMM platform override.")
    single.add_argument("--output", type=Path, default=Path("results"), help="Output directory (default: ./results).")
    single.add_argument("--label", default="", help="Optional human label (not part of run identity).")
    single.add_argument("--no-analyze", action="store_true", help="Skip frequency/PES benchmark analysis.")
    single.add_argument(
        "--no-promote", action="store_true", help="Persist candidates but do not promote accepted ones."
    )
    _add_optimizer_knobs(single)
    single.set_defaults(func=_cmd_single)

    batch = sub.add_parser("batch", help="Run the convergence workflow across one or more systems.")
    batch.add_argument("--system", action="append", help="System key (repeatable; default: all systems).")
    batch.add_argument("--backend", default="jax", help="MM backend key (default: jax).")
    batch.add_argument(
        "--form", default=None, choices=FUNCTIONAL_FORMS, help="Functional form (default: system default)."
    )
    batch.add_argument(
        "--optimizer", default="scipy-lbfgsb-jax", choices=sorted(OPTIMIZER_CATALOG), help="Optimizer key."
    )
    batch.add_argument("--workflow", default="method-e2", choices=("single-stage", "method-e2"), help="Workflow.")
    batch.add_argument("--starting-point", default="qfuerza", choices=("qfuerza", "published"), help="Starting FF.")
    batch.add_argument(
        "--objective-profile",
        default=None,
        choices=sorted(KNOWN_OBJECTIVE_PROFILES),
        help="Publication objective/completeness profile (default: repository compatibility for publication systems).",
    )
    batch.add_argument("--skip-optimization", action="store_true", help="Compute baseline metrics only.")
    batch.add_argument("--platform", default=None, help="OpenMM platform override.")
    batch.add_argument("--output", type=Path, default=Path("results"), help="Output directory (default: ./results).")
    batch.add_argument("--analyze", action="store_true", help="Also run frequency/PES benchmark analysis (slower).")
    batch.add_argument("--no-promote", action="store_true", help="Persist candidates but do not promote accepted ones.")
    _add_optimizer_knobs(batch)
    batch.set_defaults(func=_cmd_batch)

    matrix = sub.add_parser("matrix", help="Run a backend x form x optimizer matrix for a system.")
    matrix.add_argument("--system", required=True, help="Benchmark system key.")
    matrix.add_argument("--backend", action="append", help="MM backend key (repeatable; default: all available).")
    matrix.add_argument("--form", action="append", choices=FUNCTIONAL_FORMS, help="Functional form (repeatable).")
    matrix.add_argument(
        "--optimizer", action="append", choices=sorted(OPTIMIZER_CATALOG), help="Optimizer key (repeatable)."
    )
    matrix.add_argument("--starting-point", default="qfuerza", choices=("qfuerza", "published"), help="Starting FF.")
    matrix.add_argument(
        "--objective-profile",
        default=None,
        choices=sorted(KNOWN_OBJECTIVE_PROFILES),
        help="Publication objective/completeness profile (default: repository compatibility for publication systems).",
    )
    matrix.add_argument("--skip-optimization", action="store_true", help="Compute baseline metrics only.")
    matrix.add_argument("--platform", default=None, help="OpenMM platform override.")
    matrix.add_argument("--output", type=Path, default=Path("results"), help="Output directory (default: ./results).")
    matrix.add_argument("--no-analyze", action="store_true", help="Skip frequency/PES benchmark analysis.")
    matrix.add_argument(
        "--no-promote", action="store_true", help="Persist candidates but do not promote accepted ones."
    )
    _add_optimizer_knobs(matrix)
    matrix.set_defaults(func=_cmd_matrix)

    load = sub.add_parser("load", help="Load and summarise persisted candidate records.")
    load.add_argument("directory", type=Path, help="Run directory (or a candidates/ directory) to load.")
    load.set_defaults(func=_cmd_load)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``q2mm-benchmark`` console script."""
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if stream is not None and hasattr(stream, "reconfigure") and (stream.encoding or "").lower() != "utf-8":
                stream.reconfigure(encoding="utf-8", errors="replace")

    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
    )
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
