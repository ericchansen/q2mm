"""Benchmark run history tracking.

Stores one JSON file per benchmark run in ``benchmarks/history/``,
enabling cross-commit comparison of optimization results.  Each file
captures the run configuration, environment, git provenance, and
per-combo summary metrics.

The CLI appends to this directory automatically after each matrix run.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from q2mm.diagnostics.benchmark import BenchmarkResult


@dataclass
class RunSummary:
    """Summary of a single benchmark matrix run.

    Attributes:
        run_id: Unique identifier for this run.
        system: Benchmark system name (e.g. ``"ch3f"``).
        git_sha: Full git commit SHA at runtime, or ``None``.
        git_dirty: Whether the working tree had uncommitted changes.
        timestamp: ISO 8601 timestamp when the run started.
        environment: Reproducibility metadata (Python, GPU, packages).
        config: Run configuration (requested backends, forms, etc.).
        combos: Per-combo summary metrics keyed by filename stem.

    """

    run_id: str
    system: str
    git_sha: str | None
    git_dirty: bool | None
    timestamp: str
    environment: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    combos: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self, path: str | Path) -> None:
        """Save this run summary to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(asdict(self), fh, indent=2)
            fh.write("\n")

    @classmethod
    def from_json(cls, path: str | Path) -> RunSummary:
        """Load a run summary from a JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        return cls(**data)


def _combo_summary(result: BenchmarkResult) -> dict[str, Any]:
    """Extract summary metrics from a single combo result."""
    from q2mm.diagnostics.benchmark import benchmark_stem

    stem = benchmark_stem(result.metadata)
    opt = result.optimized or {}
    sem = result.seminario or {}
    error = result.metadata.get("error")

    if error:
        return {
            "stem": stem,
            "status": "failed",
            "error": error,
        }

    if not opt:
        return {
            "stem": stem,
            "status": "no_result",
        }

    return {
        "stem": stem,
        "status": "converged" if opt.get("converged") else "not_converged",
        "rmsd": opt.get("rmsd"),
        "mae": opt.get("mae"),
        "time_s": opt.get("elapsed_s"),
        "n_eval": opt.get("n_eval"),
        "seminario_rmsd": sem.get("rmsd"),
        "backend": result.metadata.get("backend"),
        "optimizer": result.metadata.get("optimizer"),
        "form": result.metadata.get("functional_form"),
        "jac_mode": result.metadata.get("jac_mode"),
        "gradients": result.metadata.get("gradients"),
    }


def build_run_summary(
    results: list[BenchmarkResult],
    *,
    system: str,
    run_id: str,
    config: dict[str, Any] | None = None,
) -> RunSummary:
    """Build a :class:`RunSummary` from a list of benchmark results.

    Args:
        results: Completed benchmark results from ``_run_matrix``.
        system: Benchmark system key (e.g. ``"ch3f"``).
        run_id: Unique run identifier.
        config: Run configuration metadata (backends, forms, etc.).

    Returns:
        A populated :class:`RunSummary`.

    """
    from q2mm.diagnostics.benchmark import _collect_environment, _git_info, benchmark_stem

    git = _git_info()
    env_raw = _collect_environment()

    # Flatten environment for the summary
    packages = env_raw.get("packages", {})
    environment = {
        "python": env_raw.get("python_version"),
        "platform": env_raw.get("platform"),
        "gpu": env_raw.get("gpu"),
    }
    environment.update(packages)

    # Determine timestamp from first result, or now
    timestamp = None
    for r in results:
        ts = r.metadata.get("timestamp")
        if ts:
            timestamp = ts
            break
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    # Build combo summaries
    combos: dict[str, dict[str, Any]] = {}
    for r in results:
        summary = _combo_summary(r)
        stem = benchmark_stem(r.metadata)
        combos[stem] = summary

    return RunSummary(
        run_id=run_id,
        system=system,
        git_sha=git.get("git_sha"),
        git_dirty=git.get("git_dirty"),
        timestamp=timestamp,
        environment=environment,
        config=config or {},
        combos=combos,
    )


def generate_run_id(system: str) -> str:
    """Generate a unique run identifier.

    Format: ``{system}_{sha_short}_{timestamp}``

    Falls back to ``nosha`` if git info is unavailable.
    """
    from q2mm.diagnostics.benchmark import _git_info

    git = _git_info()
    sha = git.get("git_sha")
    sha_short = sha[:8] if sha else "nosha"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{system}_{sha_short}_{ts}"


def load_history(history_dir: str | Path) -> list[RunSummary]:
    """Load all run summaries from a history directory.

    Args:
        history_dir: Path to the ``benchmarks/history/`` directory.

    Returns:
        List of :class:`RunSummary` objects, sorted by timestamp.

    """
    history_dir = Path(history_dir)
    if not history_dir.is_dir():
        return []

    summaries: list[RunSummary] = []
    for path in sorted(history_dir.glob("*.json")):
        try:
            summaries.append(RunSummary.from_json(path))
        except Exception:
            continue

    summaries.sort(key=lambda s: s.timestamp)
    return summaries
