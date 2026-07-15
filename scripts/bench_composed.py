#!/usr/bin/env python3
"""Run composed-workflow benchmarks on CH₃F MM3.

Workflow B: multi-start → optax Adam refinement
  Phase 1: multi-start n=10 (L-BFGS-B inner) finds best basin
  Phase 2: optax Adam (100 steps) refines from phase-1 winner

Results are saved as separate JSON files for each phase plus a composed
summary.  All results go to results/ch3f/results/ with force fields
to results/ch3f/forcefields/.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_info() -> dict[str, Any]:
    """Gather git metadata."""
    from q2mm.diagnostics.benchmark import _git_info

    return _git_info()


def _collect_environment() -> dict[str, Any]:
    """Gather environment metadata."""
    from q2mm.diagnostics.benchmark import _collect_environment

    return _collect_environment()


def _param_names(layout: Any) -> list[str]:
    """Extract parameter names from a :class:`~q2mm.models.parameters.ParameterLayout`."""
    from q2mm.diagnostics.benchmark import _param_names

    return _param_names(layout)


def run_workflow_b(
    engine: Any,
    sys_data: Any,
    backend_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run Workflow B: multi-start n=10 → optax Adam refinement.

    Args:
        engine: MM backend engine.
        sys_data: Loaded system data.
        backend_name: Display name for the backend.
        output_dir: Directory for result and force field files.

    Returns:
        Summary dict with phase1, phase2, and composed metrics.

    """
    from q2mm.diagnostics.benchmark import run_combo

    results_dir = output_dir / "results"
    ff_dir = output_dir / "forcefields"

    starting_ff = sys_data.problem.starting_force_field
    form = starting_ff.functional_form.value
    molecule = sys_data.metadata.get("molecule_name", "unknown")

    print(f"\n{'=' * 60}")
    print(f"Workflow B: {backend_name} / {form}")
    print(f"{'=' * 60}")

    # --- Phase 1: multi-start n=10 ---
    print("\n  Phase 1: multi-start n=10 (L-BFGS-B inner) ...")
    t0_total = time.perf_counter()

    p1_result = run_combo(
        engine,
        sys_data,
        optimizer_method="multi:L-BFGS-B",
        optimizer_kwargs={"n_starts": 10},
        backend_name=backend_name,
    )
    p1_result.metadata["optimizer"] = "multi:L-BFGS-B (n=10)"
    p1_result.metadata["functional_form"] = form
    p1_elapsed = p1_result.optimized["elapsed_s"] if p1_result.optimized else 0

    p1_rmsd = p1_result.optimized.get("rmsd", float("nan")) if p1_result.optimized else float("nan")
    p1_score = p1_result.optimized.get("final_score", float("nan")) if p1_result.optimized else float("nan")
    print(f"    → RMSD {p1_rmsd:.1f}, score {p1_score:.4f} ({p1_elapsed:.1f}s)")

    # Save phase 1
    p1_label = f"ch3f_{form}_composed_p1_multi10_{backend_name.lower().replace(' ', '_')}"
    p1_result.to_json(results_dir / f"{p1_label}.json")
    if p1_result.optimized_ff:
        p1_result.save_forcefields(ff_dir / f"{p1_label}")

    # --- Phase 2: optax Adam from phase-1 force field ---
    if p1_result.optimized_ff is None:
        print("    ✗ Phase 1 produced no optimized FF — skipping phase 2")
        return {"error": "no optimized FF from phase 1"}

    print("  Phase 2: optax Adam (100 steps) from phase-1 winner ...")

    # Build a new BenchmarkCase/OptimizationProblem with the phase-1
    # optimized FF as the new starting point.  Both are immutable, so
    # dataclasses.replace() rebuilds them (re-validating structural
    # consistency, e.g. layout/vector-length match) instead of mutating.
    # The active space's baseline must move too — with_baseline() keeps
    # its active/frozen partition but re-snapshots inactive values from
    # the phase-1 FF, so phase 2 never silently reverts an inactive slot
    # to the phase-1 *starting* value via a stale baseline.
    import dataclasses

    layout = sys_data.problem.layout
    active_space_p2 = sys_data.problem.active_space.with_baseline(layout.vector(p1_result.optimized_ff))
    problem_p2 = dataclasses.replace(
        sys_data.problem,
        starting_force_field=p1_result.optimized_ff,
        active_space=active_space_p2,
    )
    sys_data_p2 = dataclasses.replace(sys_data, problem=problem_p2)

    p2_result = run_combo(
        engine,
        sys_data_p2,
        optimizer_method="optax:adam",
        optimizer_kwargs={"max_steps": 100},
        backend_name=backend_name,
    )
    p2_result.metadata["optimizer"] = "composed:adam-refine"
    p2_result.metadata["functional_form"] = form
    p2_result.metadata["composed_phase"] = "phase2"
    p2_result.metadata["phase1_optimizer"] = "multi:L-BFGS-B (n=10)"
    p2_elapsed = p2_result.optimized["elapsed_s"] if p2_result.optimized else 0

    p2_rmsd = p2_result.optimized.get("rmsd", float("nan")) if p2_result.optimized else float("nan")
    p2_score = p2_result.optimized.get("final_score", float("nan")) if p2_result.optimized else float("nan")
    print(f"    → RMSD {p2_rmsd:.1f}, score {p2_score:.4f} ({p2_elapsed:.1f}s)")

    total_elapsed = time.perf_counter() - t0_total

    # Save phase 2
    p2_label = f"ch3f_{form}_composed_p2_adam_{backend_name.lower().replace(' ', '_')}"
    p2_result.to_json(results_dir / f"{p2_label}.json")
    if p2_result.optimized_ff:
        p2_result.save_forcefields(ff_dir / f"{p2_label}")

    # --- Composed summary ---
    seminario_rmsd = p1_result.seminario.get("rmsd", float("nan")) if p1_result.seminario else float("nan")
    p1_n_eval = p1_result.optimized.get("n_eval", 0) if p1_result.optimized else 0
    p2_n_eval = p2_result.optimized.get("n_eval", 0) if p2_result.optimized else 0

    composed = {
        "metadata": {
            "backend": backend_name,
            "optimizer": "composed:multi→adam",
            "molecule": molecule,
            "functional_form": form,
            "source": "q2mm",
            "composed": True,
            "phases": ["multi:L-BFGS-B (n=10)", "optax:adam (100 steps)"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_git_info(),
        },
        "environment": _collect_environment(),
        "seminario_rmsd": seminario_rmsd,
        "phase1": {
            "optimizer": "multi:L-BFGS-B (n=10)",
            "rmsd": p1_rmsd,
            "score": p1_score,
            "n_eval": p1_n_eval,
            "elapsed_s": p1_elapsed,
        },
        "phase2": {
            "optimizer": "optax:adam (100 steps)",
            "rmsd": p2_rmsd,
            "score": p2_score,
            "n_eval": p2_n_eval,
            "elapsed_s": p2_elapsed,
        },
        "composed": {
            "final_rmsd": p2_rmsd,
            "total_elapsed_s": total_elapsed,
            "total_n_eval": p1_n_eval + p2_n_eval,
            "improvement_over_phase1_rmsd": p1_rmsd - p2_rmsd,
        },
    }

    label = f"ch3f_{form}_composed_multi_adam_{backend_name.lower().replace(' ', '_')}"
    path = results_dir / f"{label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(composed, f, indent=2)

    print(f"\n  Composed: Seminario {seminario_rmsd:.0f} → Phase 1 {p1_rmsd:.1f} → Phase 2 {p2_rmsd:.1f}")
    print(f"  Total: {total_elapsed:.1f}s, {p1_n_eval + p2_n_eval} evals")

    return composed


def main() -> None:
    """Run all composed workflow benchmarks."""
    import argparse

    parser = argparse.ArgumentParser(description="Run composed workflow benchmarks")
    parser.add_argument(
        "--backend",
        choices=["openmm", "jax", "all"],
        default="all",
        help="Backend to use (default: all available)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ch3f"),
        help="Output directory (default: results/ch3f)",
    )
    parser.add_argument(
        "--platform",
        default=None,
        help="OpenMM platform override (e.g., CUDA)",
    )
    args = parser.parse_args()

    from q2mm.benchmarks.systems import load_system

    output_dir = args.output
    all_results: list[dict] = []

    # Only MM3 — composed workflows target rugged landscapes
    form = "mm3"

    # Discover backends
    backends_to_run: list[tuple[str, Any]] = []

    if args.backend in ("openmm", "all"):
        try:
            from q2mm.backends.registry import registered_mm_engines

            engines = registered_mm_engines()
            if "openmm" in engines:
                cls = engines["openmm"]
                if args.platform:
                    engine = cls(platform_name=args.platform)
                else:
                    engine = cls()
                backends_to_run.append((engine.name, engine))
        except Exception as e:
            print(f"  Skipping OpenMM: {e}")

    if args.backend in ("jax", "all"):
        try:
            from q2mm.backends.registry import registered_mm_engines

            engines = registered_mm_engines()
            if "jax" in engines:
                cls = engines["jax"]
                engine = cls()
                backends_to_run.append((engine.name, engine))
        except Exception as e:
            print(f"  Skipping JAX: {e}")

    if not backends_to_run:
        print("No backends available!")
        return

    for backend_name, engine in backends_to_run:
        try:
            sys_data = load_system("ch3f", engine=engine, functional_form=form)
        except Exception as e:
            print(f"  Cannot load CH3F {form} for {backend_name}: {e}")
            continue

        result = run_workflow_b(engine, sys_data, backend_name, output_dir)
        all_results.append(result)

    # Print summary
    print(f"\n{'=' * 60}")
    print("COMPOSED BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        if "error" in r:
            print(f"  {r.get('metadata', {}).get('backend', '?')}: ERROR — {r['error']}")
            continue
        meta = r["metadata"]
        comp = r["composed"]
        print(
            f"  {meta['backend']} / {meta['functional_form']}: "
            f"Seminario {r['seminario_rmsd']:.0f} → "
            f"multi {r['phase1']['rmsd']:.1f} → "
            f"adam {comp['final_rmsd']:.1f}  "
            f"({comp['total_elapsed_s']:.0f}s, {comp['total_n_eval']} evals)"
        )


if __name__ == "__main__":
    main()
