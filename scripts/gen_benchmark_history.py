"""Generate the benchmark history comparison page at build time.

Called by mkdocs-gen-files during ``mkdocs build``.  Reads every JSON
file in ``benchmarks/history/`` and renders a Markdown page that shows
how RMSD results have changed across commits.

The generated page is a comparison supplement — it does *not* replace
the curated per-system benchmark pages.
"""

from __future__ import annotations

import json
from pathlib import Path

import mkdocs_gen_files

HISTORY_DIR = Path(__file__).parent.parent / "benchmarks" / "history"
OUTPUT_PATH = "benchmarks/history.md"

# Display labels for backends
BACKEND_LABELS = {
    "jax": "JAX",
    "jax_md": "JAX-MD",
    "openmm": "OpenMM",
    "tinker": "Tinker",
}


def _load_runs() -> list[dict]:
    """Load and sort all history JSONs by timestamp."""
    runs = []
    if not HISTORY_DIR.is_dir():
        return runs
    for path in sorted(HISTORY_DIR.glob("*.json")):
        try:
            with open(path) as fh:
                runs.append(json.load(fh))
        except Exception:
            continue
    runs.sort(key=lambda r: r.get("timestamp", ""))
    return runs


def _short_sha(sha: str | None) -> str:
    if not sha:
        return "unknown"
    return sha[:8]


def _format_delta(current: float | None, previous: float | None) -> str:
    """Format a delta value for display."""
    if current is None or previous is None:
        return ""
    delta = current - previous
    if abs(delta) < 0.05:
        return "="
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"


def _generate_page(runs: list[dict]) -> str:
    """Build the Markdown content for the history page."""
    lines: list[str] = []

    lines.append("# Benchmark History\n")
    lines.append(
        "This page tracks how benchmark results change across commits. "
        "It is auto-generated from `benchmarks/history/*.json` at build "
        "time. For detailed analysis of any single run, see the "
        "[Small Molecules](small-molecules.md) or "
        "[Rh-Enamide](rh-enamide.md) pages.\n"
    )

    if not runs:
        lines.append("!!! info\n")
        lines.append("    No benchmark history data found.\n")
        return "\n".join(lines)

    # --- Runs overview table ---
    lines.append("## Runs\n")
    lines.append("| # | Commit | Date | System | Combos | GPU | q2mm |")
    lines.append("|---|--------|------|--------|--------|-----|------|")

    for i, run in enumerate(runs, 1):
        sha = _short_sha(run.get("git_sha"))
        dirty = " (dirty)" if run.get("git_dirty") else ""
        date = run.get("timestamp", "")[:10]
        system = run.get("system", "?")
        n_combos = len(run.get("combos", {}))
        env = run.get("environment", {})
        gpu = env.get("gpu", "?")
        # Truncate GPU name for table
        if gpu and len(gpu) > 30:
            gpu = gpu[:27] + "..."
        q2mm_ver = env.get("q2mm", "?")
        lines.append(f"| {i} | `{sha}`{dirty} | {date} | {system} | {n_combos} | {gpu} | {q2mm_ver} |")

    lines.append("")

    # --- RMSD comparison grid ---
    if len(runs) >= 2:
        lines.append("## RMSD Comparison\n")
        lines.append(
            "Each cell shows the optimized RMSD for that combo. "
            "The **Δ** column shows the change from the previous run. "
            "Lower RMSD is better.\n"
        )

        # Collect all combo stems across all runs
        all_stems: list[str] = []
        seen: set[str] = set()
        for run in runs:
            for stem in run.get("combos", {}):
                if stem not in seen:
                    all_stems.append(stem)
                    seen.add(stem)

        # Group by backend for readability
        by_backend: dict[str, list[str]] = {}
        for stem in all_stems:
            # Extract backend from stem: ch3f_{backend}_{form}_{device}_{opt}[_{jac}]
            parts = stem.split("_")
            if len(parts) >= 3:
                backend = parts[1]
                if parts[1] == "jax" and len(parts) >= 4 and parts[2] == "md":
                    backend = "jax_md"
            else:
                backend = "other"
            by_backend.setdefault(backend, []).append(stem)

        for backend_key in ["jax", "jax_md", "openmm", "tinker"]:
            stems = by_backend.get(backend_key, [])
            if not stems:
                continue

            label = BACKEND_LABELS.get(backend_key, backend_key)
            lines.append(f"### {label}\n")

            # Build header: Combo | Run 1 | Run 2 | Δ(1→2) | ...
            header = "| Combo |"
            sep = "|-------|"
            for i, run in enumerate(runs, 1):
                sha = _short_sha(run.get("git_sha"))
                header += f" `{sha}` |"
                sep += "--------:|"
                if i > 1:
                    header += " Δ |"
                    sep += "---:|"

            lines.append(header)
            lines.append(sep)

            for stem in sorted(stems):
                # Build a short display name from the stem
                # Remove the system prefix (e.g. "ch3f_")
                parts = stem.split("_")
                if len(parts) >= 2:
                    display = "_".join(parts[1:])
                else:
                    display = stem

                row = f"| `{display}` |"
                prev_rmsd: float | None = None

                for i, run in enumerate(runs):
                    combo = run.get("combos", {}).get(stem, {})
                    rmsd = combo.get("rmsd")

                    if rmsd is not None:
                        row += f" {rmsd:.1f} |"
                    elif combo.get("status") == "failed":
                        row += " ❌ |"
                    else:
                        row += " — |"

                    if i > 0:
                        delta = _format_delta(rmsd, prev_rmsd)
                        if delta and delta != "=":
                            # Highlight improvements (negative) vs regressions (positive)
                            if delta.startswith("-"):
                                row += f" **{delta}** |"
                            elif delta.startswith("+"):
                                row += f" {delta} |"
                            else:
                                row += f" {delta} |"
                        else:
                            row += f" {delta} |"

                    prev_rmsd = rmsd

                lines.append(row)

            lines.append("")

    # --- Timing comparison ---
    if len(runs) >= 2:
        lines.append("## Timing Comparison\n")
        lines.append(
            "Wall-clock time (seconds) per combo. Timing depends on "
            "hardware and system load, so small variations are expected.\n"
        )

        # Just show a summary: total time per run
        lines.append("| Run | Commit | Total Time (s) | Avg Time/Combo (s) |")
        lines.append("|-----|--------|---------------:|-------------------:|")

        for i, run in enumerate(runs, 1):
            sha = _short_sha(run.get("git_sha"))
            combos = run.get("combos", {})
            times = [c.get("time_s") for c in combos.values() if c.get("time_s") is not None]
            total = sum(times)
            avg = total / len(times) if times else 0
            lines.append(f"| {i} | `{sha}` | {total:.0f} | {avg:.1f} |")

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point for mkdocs-gen-files."""
    runs = _load_runs()
    content = _generate_page(runs)
    with mkdocs_gen_files.open(OUTPUT_PATH, "w") as fh:
        fh.write(content)


main()
