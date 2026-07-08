"""Workflow-driven benchmark runner shared by ``q2mm.benchmark()`` and ``scripts/benchmark.py``.

This module is the single source of truth for the convergence-style
benchmark pipeline: load a registered system, build a JAX engine,
choose a :class:`~q2mm.workflows.Workflow` (single-stage or Method E2),
run it with a :class:`~q2mm.optimizers.scipy_opt.ScipyOptimizer`, and
report the result.

It Replaces the ad-hoc per-script duplication that existed in earlier alpha
revisions of ``scripts/regenerate_convergence_results.py`` (now
``scripts/benchmark.py``): ratio gating, per-category metrics, post-hoc objective sampling, strict-JSON
output, and provenance embedding all live here.

Two callers:

- :func:`q2mm.benchmark` — top-level convenience facade for notebook /
  REPL use.  One line: ``q2mm.benchmark("rh-enamide")``.
- :mod:`scripts.benchmark` — batch CLI that iterates :func:`run_benchmark`
  across many systems and writes the canonical artifacts under
  ``q2mm-data/benchmarks/<system>/convergence/`` (QFUERZA-start) or
  ``q2mm-data/benchmarks/<system>/from-published/`` (publication baseline).

The legacy ``q2mm-benchmark`` matrix CLI in
:mod:`q2mm.diagnostics.cli` is a different shape of tool (multi-backend
leaderboard exploration); it is not refactored through this module.
"""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.workflows.base import Workflow

logger = logging.getLogger("q2mm.benchmark_runner")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT.parent / "q2mm-data" / "benchmarks"

# Map q2mm SYSTEMS key → the directory name in q2mm-data/benchmarks/.
DATA_DIR_FOR_SYSTEM: dict[str, str] = {
    "ch3f": "ch3f",
    "ch3f-sn2": "ch3f-sn2",
    "rh-enamide": "rh-enamide",
    "heck-relay": "heck-relay",
    "pd-allyl": "pd-allyl-amination",
    "pd-conjugate": "pd-1,4-conjugate-addition",
    "rh-conjugate": "rh-1,4-conjugate-addition",
}


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkRunResult:
    """Result of a single :func:`run_benchmark` call.

    Attributes:
        system_key: The registered system identifier (e.g. ``"rh-enamide"``).
        workflow_name: Identifier of the workflow that produced the result
            (e.g. ``"method-e2"`` or ``"single-stage"``).
        initial_ff: Force field before optimization.
        final_ff: Force field after the workflow completes.  Same as
            *initial_ff* when ``skip_optimization=True``.
        skipped: ``True`` when no optimization ran (user opt-out, ratio
            check, or JaxLoss divergence).
        skip_reason: Free-form string describing why optimization was
            skipped; ``None`` when an optimization ran.
        summary: Strict-JSON-safe dict matching the
            ``validation_results.json`` schema documented in
            :mod:`scripts.benchmark`.  Keys include
            ``initial_obj_score``, ``initial_jaxloss``, ``ratio*``,
            ``seminario`` (per-category metrics), and — when
            optimization ran — ``final_obj_score``, ``optimized``,
            ``improvement_pct``, ``n_iterations``, etc.
        paper: Strict-JSON-safe dict matching the ``paper_metrics.json``
            schema — Seminario and (when run) optimized per-category
            stats with embedded objective scores and ref counts.

    """

    system_key: str
    workflow_name: str
    initial_ff: ForceField
    final_ff: ForceField
    skipped: bool
    skip_reason: str | None
    summary: dict[str, Any]
    paper: dict[str, Any]


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_info(repo: Path) -> dict[str, Any]:
    """Return ``{git_sha, git_dirty}`` for a git repo, or empty dict on failure."""
    if not (repo / ".git").exists():
        return {}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return {"git_sha": sha, "git_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _device_info() -> dict[str, Any]:
    """Probe JAX and OpenMM device info — best effort, never raises."""
    info: dict[str, Any] = {}
    try:
        import jax

        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        info["jax_devices_error"] = repr(exc)
    try:
        import openmm

        info["openmm_platforms"] = [
            openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())
        ]
    except Exception as exc:
        info["openmm_platforms_error"] = repr(exc)
    return info


def build_provenance(
    *,
    output_dir: Path,
    generator: str,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Build the standard provenance block embedded in every output file.

    Args:
        output_dir: User-supplied root for system convergence outputs
            (e.g. ``../q2mm-data/benchmarks``); its parent is the
            q2mm-data repo root we record git info for.
        generator: Identifier of the caller (e.g. ``"scripts/benchmark.py"``
            or ``"q2mm.benchmark()"``).
        settings: Caller-specific knobs to embed alongside git/device
            info (workflow name, optimizer config, ratio_tol, etc.).

    """
    return {
        "generator": generator,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": shlex.join(sys.argv),
        "q2mm": _git_info(REPO_ROOT),
        "q2mm_data": _git_info(output_dir.parent),
        "settings": settings,
        "devices": _device_info(),
    }


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------


def _r2(ref: np.ndarray, calc: np.ndarray) -> float:
    """Q2MM-convention coefficient of determination ``1 - SS_res/SS_tot``."""
    if ref.size < 2:
        return float("nan")
    ss_res = float(np.sum((ref - calc) ** 2))
    mean = float(np.mean(ref))
    ss_tot = float(np.sum((ref - mean) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _category_stats(ref_values: np.ndarray, calc_values: np.ndarray) -> dict[str, float]:
    """Compute n_refs, R², RMSD, MAE for a single category."""
    n = int(ref_values.size)
    if n == 0:
        return {"n_refs": 0, "r2": float("nan"), "rmsd": float("nan"), "mae": float("nan")}
    residuals = ref_values - calc_values
    rmsd = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    r2 = _r2(ref_values, calc_values)
    return {"n_refs": n, "r2": r2, "rmsd": rmsd, "mae": mae}


def per_category_metrics(obj: Any, ff: Any) -> dict[str, dict[str, float]]:
    """Compute per-category R²/RMSD/MAE by inverting residuals through weights.

    ``ObjectiveFunction._compute_residuals(ff)`` returns ``w_i * (ref - calc)``
    for every reference value in order.  We bucket by ``ref.kind``, undo the
    weight to recover ``calc``, and compute statistics on the raw values.
    """
    residuals = obj._compute_residuals(ff)  # noqa: SLF001 — direct API for diagnostics
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ref, weighted in zip(obj.reference.values, residuals, strict=True):
        if ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        buckets[ref.kind].append((float(ref.value), calc_value))

    out: dict[str, dict[str, float]] = {}
    for kind, pairs in buckets.items():
        ref_arr = np.array([p[0] for p in pairs])
        calc_arr = np.array([p[1] for p in pairs])
        out[kind] = _category_stats(ref_arr, calc_arr)
    return out


# ---------------------------------------------------------------------------
# Strict-JSON helpers
# ---------------------------------------------------------------------------


def sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats with structured strings."""
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.floating):
        return sanitize_for_json(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return sanitize_for_json(value.tolist())
    return value


def write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* with strict JSON (no NaN/Infinity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = sanitize_for_json(payload)
    with path.open("w") as fh:
        json.dump(sanitized, fh, indent=2, allow_nan=False, sort_keys=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Ratio gate
# ---------------------------------------------------------------------------


def classify_ratio(ratio: float, tol: float | None) -> dict[str, Any]:
    """Encode the JaxLoss/ObjFun ratio state as ``{ratio, ratio_status, ratio_passes}``."""
    if not math.isfinite(ratio):
        return {
            "ratio": None,
            "ratio_status": "diverged" if math.isinf(ratio) else "nan",
            "ratio_passes": False,
        }
    if tol is None:
        return {"ratio": ratio, "ratio_status": "ok_bypassed", "ratio_passes": True}
    passes = (1.0 - tol) <= ratio <= (1.0 + tol)
    return {"ratio": ratio, "ratio_status": "ok" if passes else "out_of_band", "ratio_passes": passes}


def _initial_jaxloss(obj: Any) -> float:
    """Compute the JaxLoss surrogate at the current params, ``inf`` on failure."""
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.optimizers.jaxloss import JaxLoss

    if not isinstance(obj.engine, JaxEngine):
        return float("inf")
    spec = obj.to_jax_spec()
    jl = JaxLoss(spec, obj.engine, obj.molecules, obj.forcefield)
    x = obj.forcefield.get_param_vector()
    val_jax, _ = jl.value_and_grad_jax(x)
    val = float(val_jax)
    if not math.isfinite(val):
        return float("inf")
    return val


# ---------------------------------------------------------------------------
# Mean / CI helpers (multi-eval reporting)
# ---------------------------------------------------------------------------


def _mean_ci95(samples: list[float]) -> tuple[float, float]:
    """Sample mean and t-distribution 95% CI half-width."""
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        raise ValueError("at least one objective sample is required")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    if not math.isfinite(std) or std == 0.0:
        return mean, 0.0

    from scipy.stats import t

    ci95 = float(t.ppf(0.975, arr.size - 1) * std / math.sqrt(arr.size))
    return mean, ci95


def _score_interval_summary(initial_samples: list[float], final_samples: list[float]) -> dict[str, Any]:
    """Build mean/CI improvement fields from repeated objective samples."""
    initial_mean, initial_ci95 = _mean_ci95(initial_samples)
    final_mean, final_ci95 = _mean_ci95(final_samples)
    improvement_pct_mean = 100.0 * (1.0 - final_mean / initial_mean) if initial_mean > 0 else 0.0
    return {
        "initial_obj_score_mean": initial_mean,
        "initial_obj_score_ci95": initial_ci95,
        "final_obj_score_mean": final_mean,
        "final_obj_score_ci95": final_ci95,
        "improvement_pct_mean": improvement_pct_mean,
        "improvement_significant": bool(abs(final_mean - initial_mean) > (initial_ci95 + final_ci95)),
    }


# ---------------------------------------------------------------------------
# Workflow resolution
# ---------------------------------------------------------------------------


def resolve_workflow(workflow: str | Workflow) -> Workflow:
    """Resolve a workflow identifier or instance.

    Accepts:

    - ``"method-e2"`` (default) — :class:`q2mm.workflows.MethodE2Workflow`
      with all-defaults configuration.
    - ``"single-stage"`` — :class:`q2mm.workflows.SingleStageWorkflow`,
      the historical pre-Phase-9 behaviour.
    - A :class:`~q2mm.workflows.base.Workflow` instance — returned as-is
      (use for non-default workflow knobs).
    """
    from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow
    from q2mm.workflows.base import Workflow as WorkflowProto

    if isinstance(workflow, str):
        key = workflow.lower()
        if key in {"method-e2", "method_e2", "e2"}:
            return MethodE2Workflow()
        if key in {"single-stage", "single", "single_stage"}:
            return SingleStageWorkflow()
        raise ValueError(
            f"Unknown workflow {workflow!r}.  Use 'method-e2' (default), 'single-stage', or pass a Workflow instance."
        )
    if isinstance(workflow, WorkflowProto):
        return workflow
    raise TypeError(f"workflow must be a string identifier or Workflow instance; got {type(workflow).__name__}")


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def run_benchmark(
    system_key: str,
    *,
    workflow: str | Workflow = "method-e2",
    starting_point: str = "qfuerza",
    qfuerza_replace_with: float = 1.0,
    ratio_tol: float | None = None,
    maxiter: int = 500,
    ftol: float = 1e-8,
    fc_fraction: float | None = None,
    eq_fraction: float | None = None,
    n_evals: int = 1,
    skip_optimization: bool = False,
) -> BenchmarkRunResult:
    """Run one workflow-driven benchmark and return the result in memory.

    This is the workhorse used by :func:`q2mm.benchmark` (the
    notebook-style facade) and by :func:`run_benchmark_batch` (the
    batch wrapper that ``scripts/benchmark.py`` calls).  See
    :ref:`docs/workflows.md` for usage patterns.

    Args:
        system_key: Registered system identifier.  Must be present in
            ``q2mm.systems.SYSTEMS``.
        workflow: ``"method-e2"`` (default), ``"single-stage"``, or a
            pre-configured :class:`~q2mm.workflows.base.Workflow`
            instance.
        starting_point: ``"qfuerza"`` (canonical, Farrugia 2025) or
            ``"published"`` (literature OPT values verbatim).
        qfuerza_replace_with: Forwarded to ``load_system`` —
            replacement value for the negative TS-Hessian eigenvalue
            during QFUERZA starting-FF construction.
        ratio_tol: JaxLoss/ObjFun ratio tolerance for the safety gate.
            ``None`` disables the gate (treat as always-passing).
            Use ``None`` for all 5 publication TS systems where the
            ratio sits in ``[0.1, 0.4]``.
        maxiter: ``scipy.optimize`` max iterations.
        ftol: ``L-BFGS-B`` function-value convergence tolerance.
        fc_fraction: Fractional bound width for force-constant params
            (``None`` = sanity bounds).  Recommended ``0.20`` for
            from-poor-start runs to keep L-BFGS-B in the starting
            basin.
        eq_fraction: Fractional bound width for equilibrium params.
            Recommended ``0.05`` for from-poor-start runs.
        n_evals: Post-hoc real-objective evaluations at the initial
            and final parameter vectors (for mean/CI reporting).
            ``1`` is the default; raise to ``5``+ for noise analysis.
        skip_optimization: When ``True``, compute baseline (Seminario)
            metrics only — no workflow executes.

    Returns:
        :class:`BenchmarkRunResult` with the in-memory force fields
        and JSON-ready summary/paper dicts.  No disk I/O; the caller
        is responsible for persistence (see :func:`run_benchmark_batch`
        for the canonical write path).

    """
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.systems import SYSTEMS, load_system
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    if system_key not in SYSTEMS:
        raise ValueError(f"Unknown system {system_key!r}.  Available: {sorted(SYSTEMS)}")

    workflow_obj = resolve_workflow(workflow)

    logger.info("[%s] loading (starting_point=%s)", system_key, starting_point)
    engine = JaxEngine()
    sys_data = load_system(
        system_key,
        engine=engine,
        starting_point=starting_point,
        qfuerza_replace_with=qfuerza_replace_with,
    )
    initial_ff = sys_data.forcefield.copy()

    # ---- Seminario fit quality at the starting FF -----------------------
    obj_initial = ObjectiveFunction(initial_ff, engine, sys_data.molecules, sys_data.reference)
    initial_score = float(obj_initial(initial_ff.get_param_vector()))
    seminario_categories = per_category_metrics(obj_initial, initial_ff)

    # ---- JaxLoss ratio gate ---------------------------------------------
    initial_jaxloss = _initial_jaxloss(obj_initial)
    ratio = initial_jaxloss / initial_score if initial_score > 0 else float("nan")
    ratio_info = classify_ratio(ratio, ratio_tol)

    n_active = int(np.sum(initial_ff.active_mask))
    summary: dict[str, Any] = {
        "system": system_key,
        "workflow": workflow_obj.name,
        "n_molecules": len(sys_data.molecules),
        "n_active_params": n_active,
        "starting_point": starting_point,
        "starting_point_audit": sys_data.metadata.get("starting_point_audit"),
        "initial_obj_score": initial_score,
        "initial_jaxloss": initial_jaxloss,
        **ratio_info,
        "seminario": seminario_categories,
    }
    paper: dict[str, Any] = {
        "seminario": {
            **seminario_categories,
            "_objective_score": initial_score,
            "_total_refs": sum(cat["n_refs"] for cat in seminario_categories.values()),
        },
    }

    # ---- Decide whether to skip optimization ----------------------------
    skip_reason: str | None = None
    if skip_optimization:
        skip_reason = "user_requested"
    elif not ratio_info["ratio_passes"] and ratio_tol is not None:
        skip_reason = "ratio_check_failed" if ratio_info["ratio_status"] == "out_of_band" else "jaxloss_diverged"

    if skip_reason is not None:
        summary["skipped"] = True
        summary["skip_reason"] = skip_reason
        logger.info("[%s] skipping optimization (%s)", system_key, skip_reason)
        return BenchmarkRunResult(
            system_key=system_key,
            workflow_name=workflow_obj.name,
            initial_ff=initial_ff,
            final_ff=initial_ff,
            skipped=True,
            skip_reason=skip_reason,
            summary=summary,
            paper=paper,
        )

    # ---- Run the workflow ------------------------------------------------
    logger.info(
        "[%s] running workflow=%s (ratio=%.3f, n_active=%d)",
        system_key,
        workflow_obj.name,
        ratio,
        n_active,
    )
    optimizer = ScipyOptimizer(
        method="L-BFGS-B",
        maxiter=maxiter,
        ftol=ftol,
        verbose=True,
        jac="auto",
        ratio_tol=ratio_tol,
        fc_fraction=fc_fraction,
        eq_fraction=eq_fraction,
    )
    t0 = time.perf_counter()
    wf_result = workflow_obj.run(sys_data, engine, optimizer, n_evals=n_evals)
    elapsed = time.perf_counter() - t0

    final_ff = wf_result.final_ff

    # Real ObjectiveFunction score at the final FF — measured against
    # the ORIGINAL reference data (not any modified-Hessian Round-2
    # reference), so the improvement number is comparable across
    # workflows and to historical baselines.
    obj_real_at_final = ObjectiveFunction(final_ff, engine, sys_data.molecules, sys_data.reference)
    final_obj_score = float(obj_real_at_final(final_ff.get_param_vector()))
    optimized_categories = per_category_metrics(obj_real_at_final, final_ff)

    last_stage = wf_result.stages[-1]
    score_summary = (
        _score_interval_summary(wf_result.initial_obj_samples, wf_result.final_obj_samples)
        if (wf_result.initial_obj_samples and wf_result.final_obj_samples)
        else {}
    )

    improvement_pct = 100.0 * (1.0 - final_obj_score / initial_score) if initial_score > 0 else 0.0
    summary.update(
        {
            "final_obj_score": final_obj_score,
            "final_optimizer_score": float(last_stage.final_score),
            "initial_optimizer_score": float(wf_result.stages[0].initial_score),
            "n_iterations": int(sum(stg.n_iterations for stg in wf_result.stages)),
            "n_evaluations": int(sum(stg.n_evaluations for stg in wf_result.stages)),
            "converged": bool(last_stage.converged),
            "message": str(last_stage.message),
            "jac_mode": str(last_stage.jac_mode),
            "optimized_categories": optimized_categories,
            "opt_time_s": elapsed,
            "improvement_pct": improvement_pct,
            "stages": [_stage_to_dict(stg) for stg in wf_result.stages],
            **score_summary,
        }
    )
    summary["optimized"] = optimized_categories
    paper["optimized"] = {
        **optimized_categories,
        "_objective_score": final_obj_score,
        "_total_refs": sum(cat["n_refs"] for cat in optimized_categories.values()),
    }

    return BenchmarkRunResult(
        system_key=system_key,
        workflow_name=workflow_obj.name,
        initial_ff=initial_ff,
        final_ff=final_ff,
        skipped=False,
        skip_reason=None,
        summary=summary,
        paper=paper,
    )


def _stage_to_dict(stage: Any) -> dict[str, Any]:
    """JSON-safe StageResult serialisation (matches WorkflowResult attrs)."""
    return {
        "name": stage.name,
        "initial_score": float(stage.initial_score),
        "final_score": float(stage.final_score),
        "n_iterations": int(stage.n_iterations),
        "n_evaluations": int(stage.n_evaluations),
        "converged": bool(stage.converged),
        "message": str(stage.message),
        "jac_mode": str(stage.jac_mode),
        "elapsed_s": float(stage.elapsed_s),
        "locked_param_indices": list(stage.locked_param_indices),
        "notes": dict(stage.notes),
    }


# ---------------------------------------------------------------------------
# Batch runner (scripts/benchmark.py)
# ---------------------------------------------------------------------------


@dataclass
class BatchOutcome:
    """Outcome of a :func:`run_benchmark_batch` call."""

    results: dict[str, BenchmarkRunResult] = field(default_factory=dict)
    failed_systems: list[str] = field(default_factory=list)
    no_progress_systems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no system failed and at least one system optimized progress."""
        if self.failed_systems:
            return False
        optimized = [r for r in self.results.values() if not r.skipped]
        return not (optimized and len(self.no_progress_systems) == len(optimized))


def run_benchmark_batch(
    system_keys: list[str],
    *,
    output_dir: Path | None = None,
    generator: str = "q2mm.benchmark_runner",
    **kwargs: Any,
) -> BatchOutcome:
    """Run :func:`run_benchmark` for many systems and persist artifacts.

    Per-system files written to ``<output_dir>/<data_dir>/convergence/``
    where ``<data_dir>`` is resolved via :data:`DATA_DIR_FOR_SYSTEM`:

    - ``validation_results.json`` — summary numbers
    - ``paper_metrics.json`` — Seminario + optimized per-category stats
    - ``<system>_optimized.fld`` — optimized FF (only when optimization ran)

    Args:
        system_keys: List of registered system identifiers.
        output_dir: Root for system convergence outputs.  Defaults to
            ``q2mm-data/benchmarks`` relative to the q2mm repo root.
        generator: Caller identifier embedded in the provenance block
            of each written file.
        **kwargs: Forwarded to :func:`run_benchmark` per system.

    Returns:
        :class:`BatchOutcome` with per-system results, failure list,
        and a batch-level no-progress watchdog flag.

    """
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(
        output_dir=output_dir,
        generator=generator,
        settings={k: _provenance_safe(v) for k, v in kwargs.items()},
    )

    outcome = BatchOutcome()
    for system_key in system_keys:
        try:
            result = run_benchmark(system_key, **kwargs)
            outcome.results[system_key] = result
            _write_artifacts(
                output_dir,
                result,
                provenance,
                starting_point=kwargs.get("starting_point", "qfuerza"),
            )
        except Exception:
            logger.exception("[%s] FAILED", system_key)
            outcome.failed_systems.append(system_key)

    # --- No-progress watchdog (AGENTS.md §11) ----------------------------
    if not kwargs.get("skip_optimization"):
        optimized = [r for r in outcome.results.values() if not r.skipped]
        for r in optimized:
            n_iters = int(r.summary.get("n_iterations", 0))
            impr = abs(float(r.summary.get("improvement_pct", 0.0)))
            if n_iters <= 2 and impr < 1.0:
                outcome.no_progress_systems.append(r.system_key)
        if optimized and len(outcome.no_progress_systems) == len(optimized):
            logger.error(
                "BATCH FAILURE: all %d optimized system(s) exited at n_iterations<=2 "
                "with |improvement_pct|<1%%. The optimizer did NOT optimize. "
                "Inspect ratio_tol, ftol, bounds, and starting force field. Systems: %s",
                len(optimized),
                outcome.no_progress_systems,
            )

    return outcome


def _provenance_safe(value: Any) -> Any:
    """Serialise non-trivial knob values for the provenance block."""
    if hasattr(value, "name") and not isinstance(value, type):
        return getattr(value, "name", repr(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def _write_artifacts(
    output_dir: Path,
    result: BenchmarkRunResult,
    provenance: dict[str, Any],
    *,
    starting_point: str = "qfuerza",
) -> None:
    """Persist one :class:`BenchmarkRunResult` to disk."""
    data_dir = DATA_DIR_FOR_SYSTEM.get(result.system_key, result.system_key)
    subdir = "from-published" if starting_point == "published" else "convergence"
    sys_out = output_dir / data_dir / subdir
    sys_out.mkdir(parents=True, exist_ok=True)

    write_strict_json(
        sys_out / "validation_results.json",
        {"provenance": provenance, "result": result.summary},
    )
    write_strict_json(
        sys_out / "paper_metrics.json",
        {"provenance": provenance, "metrics": result.paper},
    )
    if not result.skipped:
        ff_path = sys_out / f"{result.system_key}_optimized.fld"
        result.final_ff.to_mm3_fld(str(ff_path))
        logger.info("[%s] wrote optimized FF: %s", result.system_key, ff_path)
