"""Report generation from benchmark results.

Combines one or more :class:`~q2mm.diagnostics.benchmark.BenchmarkResult`
objects into detailed per-result SI tables and a summary leaderboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from q2mm.diagnostics.tables import (
    TablePrinter,
    convergence_table,
    frequency_progression_table,
    leaderboard_table,
    parameter_table,
    pes_distortion_table,
    timing_table,
)

if TYPE_CHECKING:
    from q2mm.diagnostics.benchmark import BenchmarkResult


def build_leaderboard_rows(results: list[BenchmarkResult]) -> list[dict]:
    """Build leaderboard row dicts from benchmark results."""
    rows = []
    for r in results:
        meta = r.metadata
        opt = r.optimized or {}
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
                "final_score": float("nan") if opt.get("final_score") is None else opt["final_score"],
                "converged": opt.get("converged", False),
                "message": opt.get("message", ""),
                "error": meta.get("error", ""),
                "initial_rmsd": initial_rmsd,
            }
        )
    return rows


def detailed_report(result: BenchmarkResult, *, combo_label: str | None = None) -> list[TablePrinter]:
    """Generate all SI tables for a single benchmark result.

    Returns a list of ``TablePrinter`` objects (one per table).
    Call ``.flush()`` on each to print, or ``.to_string()`` to capture.
    """
    meta = result.metadata
    if combo_label is None:
        combo_label = f"{meta.get('backend', '?')} + {meta.get('optimizer', '?')}"

    tables: list[TablePrinter] = []

    qm_freqs = result.qm_reference.get("frequencies_cm1", [])

    # --- Summary first: Convergence status ---
    if result.optimized:
        opt = result.optimized
        if opt.get("initial_score") is not None and opt.get("final_score") is not None:
            # Get starting RMSD (Seminario if available, else default)
            initial_rmsd = None
            if result.seminario and result.seminario.get("rmsd") is not None:
                initial_rmsd = result.seminario["rmsd"]
            elif result.default_ff and result.default_ff.get("rmsd") is not None:
                initial_rmsd = result.default_ff["rmsd"]

            tables.append(
                convergence_table(
                    opt["initial_score"],
                    opt["final_score"],
                    opt.get("n_eval", 0),
                    opt.get("converged", False),
                    opt.get("message", ""),
                    title=f"SUMMARY [{combo_label}]",
                    initial_rmsd=initial_rmsd,
                    final_rmsd=opt.get("rmsd"),
                    elapsed_s=opt.get("elapsed_s"),
                )
            )

    # --- SI Table 1: Frequency progression ---
    stages = []
    if result.default_ff and result.default_ff.get("frequencies_cm1"):
        stages.append(("Default FF", result.default_ff["frequencies_cm1"]))
    if result.seminario and result.seminario.get("frequencies_cm1"):
        stages.append(("Seminario", result.seminario["frequencies_cm1"]))
    if result.optimized and result.optimized.get("frequencies_cm1"):
        stages.append(("Optimized", result.optimized["frequencies_cm1"]))

    if stages and qm_freqs:
        tables.append(
            frequency_progression_table(
                qm_freqs,
                stages,
                title=f"FREQUENCY PROGRESSION [{combo_label}]",
            )
        )

    # --- SI Table 2: PES distortion ---
    if result.pes_distortion and result.pes_distortion.get("modes"):
        tables.append(
            pes_distortion_table(
                result.pes_distortion["modes"],
                title=f"PES DISTORTION [{combo_label}]",
                elapsed_s=result.pes_distortion.get("elapsed_s"),
            )
        )

    # --- SI Table 3: Timing ---
    if result.optimized:
        timings = {}
        if result.seminario and result.seminario.get("elapsed_s") is not None:
            timings["seminario_estimation_s"] = result.seminario["elapsed_s"]
        if result.optimized.get("elapsed_s") is not None:
            timings["optimization_s"] = result.optimized["elapsed_s"]
        if result.optimized.get("n_eval") is not None:
            timings["function_evaluations"] = result.optimized["n_eval"]
            if result.optimized.get("elapsed_s") and result.optimized["n_eval"] > 0:
                per_eval = result.optimized["elapsed_s"] / result.optimized["n_eval"]
                timings["per_evaluation_s"] = per_eval
        if result.pes_distortion and result.pes_distortion.get("elapsed_s") is not None:
            timings["pes_distortion_s"] = result.pes_distortion["elapsed_s"]

        if timings:
            tables.append(timing_table(timings, title=f"TIMING [{combo_label}]"))

    # --- SI Table 4: Parameters ---
    if result.optimized:
        names = result.optimized.get("param_names", [])
        final = result.optimized.get("param_final", [])
        default_params = result.default_ff.get("param_values") if result.default_ff else None
        seminario_params = result.seminario.get("param_values") if result.seminario else None
        if names and final and len(names) == len(final):
            tables.append(
                parameter_table(
                    names,
                    default_params,
                    seminario_params,
                    final,
                    title=f"OPTIMIZED PARAMETERS [{combo_label}]",
                )
            )

    return tables


def full_report(results: list[BenchmarkResult]) -> None:
    """Print the complete report: leaderboard + all SI tables.

    Parameters
    ----------
    results : list[BenchmarkResult]
        One result per (backend, optimizer) combination.
    """
    # --- Leaderboard ---
    rows = build_leaderboard_rows(results)

    if rows:
        lb = leaderboard_table(rows)
        lb.flush()

    # --- Detailed SI tables per result ---
    for i, r in enumerate(results, 1):
        meta = r.metadata
        combo = f"{meta.get('backend', '?')} + {meta.get('optimizer', '?')}"

        header = TablePrinter()
        header.blank()
        header.bar()
        header.title(f"=== DETAILED RESULTS #{i}: {combo} ===")
        header.bar()
        header.flush()

        for table in detailed_report(r, combo_label=combo):
            table.flush()
