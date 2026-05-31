"""Regenerate convergence baseline artifacts for the published-FF systems.

For each benchmark system this script:

1. Loads the system (Seminario-estimated force field + reference data).
2. Computes per-category Seminario fit quality (bond_length, bond_angle,
   eig_diagonal): R², RMSD, MAE, n_refs.
3. Computes the initial ObjectiveFunction score and the JaxLoss surrogate
   score; reports their ratio.
4. If the ratio is within the configured tolerance (or the tolerance has
   been disabled with ``--ratio-tol -1``) it runs a scipy L-BFGS-B
   optimization using JaxLoss analytical gradients and writes the optimized
   force field as a ``.fld`` file.
5. With ``--n-evals N``, repeats post-hoc ObjectiveFunction evaluations at
   both the initial and optimized parameter vectors to report sample-mean
   scores, t-distribution 95% confidence-interval half-widths, mean
   improvement percentage, and whether the mean change exceeds the summed
   confidence intervals.

Outputs (per system) live under
``<output-dir>/<system-data-dir>/<subdir>/`` where ``<subdir>`` is:

- ``convergence`` (default, for ``--starting-point published``)
- ``from-qfuerza`` (for ``--starting-point qfuerza``)

Per-system files:

- ``validation_results.json`` — summary numbers for the system (strict JSON,
  no ``Infinity`` or ``NaN``).  Ratio state is encoded across three keys:
  ``ratio`` (the numeric value, or ``null`` when JaxLoss returned non-finite
  values), ``ratio_status`` (one of ``"ok"``, ``"ok_bypassed"``,
  ``"out_of_band"``, ``"diverged"``, ``"nan"``), and ``ratio_passes`` (bool).
  The legacy single-call ``initial_obj_score``, ``final_obj_score``, and
  ``improvement_pct`` fields are preserved; optimized runs also include
  ``initial_obj_score_mean``, ``initial_obj_score_ci95``,
  ``final_obj_score_mean``, ``final_obj_score_ci95``,
  ``improvement_pct_mean``, and ``improvement_significant`` (the mean
  and t-distribution 95% CI half-width over ``--n-evals`` samples).
  When optimization was not attempted, ``skipped`` is ``true`` and
  ``skip_reason`` describes why (e.g. ``"ratio_check_failed"``,
  ``"jaxloss_diverged"``, ``"user_requested"``).
- ``paper_metrics.json`` — per-category Seminario + optimized stats.
- ``<system>_optimized.fld`` — optimized force field (only when optimization
  ran and succeeded).

Every output embeds a ``provenance`` block: git SHA + dirty flag for q2mm
and q2mm-data, full command line, ratio_tol setting, JAX/OpenMM device
names, ISO-8601 timestamp.

This is the single committed producer for the convergence artifacts —
satisfies AGENTS.md Rule 8 ("every claim grounded in evidence").
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("regen-convergence")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = REPO_ROOT.parent / "q2mm-data" / "benchmarks"

# Map q2mm SYSTEMS key → the directory name in q2mm-data/benchmarks/.
DATA_DIR_FOR_SYSTEM: dict[str, str] = {
    "ch3f": "ch3f",
    "rh-enamide": "rh-enamide",
    "heck-relay": "heck-relay",
    "pd-allyl": "pd-allyl-amination",
    "pd-conjugate": "pd-1,4-conjugate-addition",
    "rh-conjugate": "rh-1,4-conjugate-addition",
}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_info(repo: Path) -> dict[str, Any]:
    """Return ``{git_sha, git_dirty}`` for a git repo, or empty dict on failure."""
    # ``.git`` can be either a directory (regular repo) or a file (worktrees /
    # submodules), so a single ``exists()`` check covers both.
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


def _build_provenance(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Construct the provenance block embedded in every output file.

    *output_dir* is the user-supplied root for system convergence outputs
    (e.g. ``../q2mm-data/benchmarks``); its parent is the q2mm-data repo
    root we want to record git info for.
    """
    q2mm_git = _git_info(REPO_ROOT)
    data_git = _git_info(output_dir.parent)
    return {
        "generator": "scripts/regenerate_convergence_results.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": shlex.join(sys.argv),
        "q2mm": q2mm_git,
        "q2mm_data": data_git,
        "ratio_tol": args.ratio_tol,
        "maxiter": args.maxiter,
        "ftol": args.ftol,
        "fc_fraction": args.fc_fraction,
        "eq_fraction": args.eq_fraction,
        "n_evals": args.n_evals,
        "skip_optimization": args.skip_optimization,
        "starting_point": args.starting_point,
        "devices": _device_info(),
    }


# ---------------------------------------------------------------------------
# Per-category R² helpers
# ---------------------------------------------------------------------------


def _r2(ref: np.ndarray, calc: np.ndarray) -> float:
    """Coefficient of determination, defined only for n>=2.

    Returns NaN when n<2 (caller handles); returns the Q2MM convention
    ``1 - SS_res/SS_tot`` otherwise.  Can be negative for very poor fits.
    """
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


def _per_category_metrics(obj: Any, ff: Any) -> dict[str, dict[str, float]]:
    """Compute per-category R²/RMSD/MAE by inverting residuals through weights.

    ``ObjectiveFunction._compute_residuals(ff)`` returns ``w_i * (ref - calc)``
    for every reference value in order.  We bucket by ``ref.kind``, undo the
    weight to recover ``calc``, and compute statistics on the raw values.
    """
    residuals = obj._compute_residuals(ff)  # noqa: SLF001 — direct API for diagnostics
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for ref, weighted in zip(obj.reference.values, residuals, strict=True):
        if ref.weight == 0.0:
            continue  # excluded refs (e.g. weight-0 imaginary modes)
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


def _sanitize_for_json(value: Any) -> Any:
    """Recursively replace non-finite floats with structured strings.

    JSON's strict mode (``allow_nan=False``) rejects NaN/±Infinity.  We
    encode them as the literal strings ``"NaN"``, ``"Infinity"``,
    ``"-Infinity"`` so consumers can detect and handle them, while the
    output remains valid JSON.  Numeric ratios for diverged systems use
    a structured ``status`` field at the call site rather than this
    fallback.
    """
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.floating):
        return _sanitize_for_json(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _sanitize_for_json(value.tolist())
    return value


def _write_strict_json(path: Path, payload: dict[str, Any]) -> None:
    """Write *payload* to *path* with strict JSON (no NaN/Infinity)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_for_json(payload)
    with path.open("w") as fh:
        json.dump(sanitized, fh, indent=2, allow_nan=False, sort_keys=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Per-system pipeline
# ---------------------------------------------------------------------------


def _classify_ratio(ratio: float, tol: float | None) -> dict[str, Any]:
    """Encode the ratio state as a structured dict.

    Returns ``{ratio, ratio_status, ratio_passes}``.  ``ratio`` is set to
    ``None`` for non-finite values; the status string describes the case.
    """
    if not math.isfinite(ratio):
        return {
            "ratio": None,
            "ratio_status": "diverged" if math.isinf(ratio) else "nan",
            "ratio_passes": False,
        }
    if tol is None:
        # Tolerance disabled — always treat as passing for gate purposes.
        return {"ratio": ratio, "ratio_status": "ok_bypassed", "ratio_passes": True}
    passes = (1.0 - tol) <= ratio <= (1.0 + tol)
    return {"ratio": ratio, "ratio_status": "ok" if passes else "out_of_band", "ratio_passes": passes}


def _mean_ci95(samples: list[float]) -> tuple[float, float]:
    """Return sample mean and t-distribution 95% CI half-width.

    Uses the standard Student-t CI for the mean: ``t_{0.975, n-1} * s / sqrt(n)``.
    A previous revision of this code reported the sample *median* with this
    same CI half-width; that was statistically inconsistent (the t-CI
    describes the sampling distribution of the mean, not the median).
    For the per-call engine noise we measure here (n ≤ 10, distributions
    are not pathologically skewed), the sample mean and median differ
    negligibly, and the mean is the right center to pair with a t-CI.
    """
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


def _evaluate_objective_samples(obj: Any, params: np.ndarray, n_evals: int) -> list[float]:
    """Evaluate objective repeatedly without polluting its counters/history."""
    n_eval_before = obj.n_eval
    history_len_before = len(obj.history)
    scores: list[float] = []
    # Repeat real ObjectiveFunction calls to quantify per-call engine noise;
    # see q2mm#284 §2.  Keep cached handles intact so this measures the same
    # evaluator instance used by the optimizer.  Truncate history (which
    # ObjectiveFunction.__call__ only appends to) rather than copying it —
    # O(1) vs O(len(history)) per sample, which matters when the optimizer
    # has accumulated thousands of evaluations.
    for _ in range(n_evals):
        try:
            score = float(obj(params))
        finally:
            obj.n_eval = n_eval_before
            del obj.history[history_len_before:]
        scores.append(score)
    return scores


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


def _initial_jaxloss(obj: Any) -> float:
    """Compute the JaxLoss surrogate value at the current parameters.

    Returns +inf when JaxLoss is not applicable (non-JAX engine) or when
    the underlying JaxLoss evaluation returns a non-finite value (NaN/Inf
    from out-of-range parameters).  Uses :meth:`JaxLoss.value_and_grad_jax`
    rather than :meth:`JaxLoss.loss_and_grad` to avoid the latter's
    silent ``1e30`` penalty substitution, which would otherwise mask a
    real divergence as a merely "out of band" finite ratio.
    """
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


def _run_optimization(
    sys_data: Any,
    engine: Any,
    *,
    ratio_tol: float | None,
    maxiter: int,
    n_evals: int,
    ftol: float = 1e-8,
    fc_fraction: float | None = None,
    eq_fraction: float | None = None,
) -> dict[str, Any]:
    """Run scipy L-BFGS-B with JaxLoss gradients; return summary dict."""
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    obj = ObjectiveFunction(sys_data.forcefield, engine, sys_data.molecules, sys_data.reference)
    opt = ScipyOptimizer(
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
    result = opt.optimize(obj)
    elapsed = time.perf_counter() - t0

    # Re-evaluate optimized force field for category metrics and the
    # *real* ObjectiveFunction at the final parameters.  ``result.final_score``
    # is whatever the optimizer was internally minimizing — for
    # ``jac="auto"`` on the JaxEngine this is JaxLoss (surrogate), not
    # the real ObjectiveFunction.  Storing JaxLoss as ``final_obj_score``
    # was misleading and produced bogus ``improvement_pct`` values when
    # the surrogate disagreed with the real objective (a key failure
    # mode for the Wahlers/Rosales metal-TS systems where JaxLoss can
    # diverge while the real objective is well-behaved).
    optimized_ff = sys_data.forcefield.with_params(result.final_params)
    initial_samples = _evaluate_objective_samples(obj, result.initial_params, n_evals)
    final_samples = _evaluate_objective_samples(obj, result.final_params, n_evals)
    score_summary = _score_interval_summary(initial_samples, final_samples)
    final_obj_score = float(final_samples[0])
    optimized_categories = _per_category_metrics(obj, optimized_ff)

    return {
        "final_obj_score": final_obj_score,
        **score_summary,
        "final_optimizer_score": float(result.final_score),
        "initial_optimizer_score": float(result.initial_score),
        "n_iterations": int(result.n_iterations),
        "n_evaluations": int(result.n_evaluations),
        "converged": bool(result.success),
        "message": str(result.message),
        "jac_mode": str(result.jac_mode) if result.jac_mode is not None else "unknown",
        "optimized_categories": optimized_categories,
        "opt_time_s": elapsed,
        "optimized_ff": optimized_ff,
    }


def process_system(
    system_key: str,
    *,
    output_dir: Path,
    ratio_tol: float | None,
    maxiter: int,
    n_evals: int,
    skip_optimization: bool,
    starting_point: str,
    provenance: dict[str, Any],
    ftol: float = 1e-8,
    fc_fraction: float | None = None,
    eq_fraction: float | None = None,
) -> dict[str, Any]:
    """Process one system end-to-end and write its artifacts."""
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.diagnostics.systems import SYSTEMS, load_system
    from q2mm.optimizers.objective import ObjectiveFunction

    if system_key not in SYSTEMS:
        raise ValueError(f"Unknown system: {system_key}")

    logger.info("[%s] loading (starting_point=%s)", system_key, starting_point)
    engine = JaxEngine()
    sys_data = load_system(system_key, engine=engine, starting_point=starting_point)
    ff = sys_data.forcefield

    # ---- Seminario fit quality ------------------------------------------
    obj = ObjectiveFunction(ff, engine, sys_data.molecules, sys_data.reference)
    initial_score = float(obj(ff.get_param_vector()))
    seminario_categories = _per_category_metrics(obj, ff)

    # ---- JaxLoss surrogate and ratio gate --------------------------------
    initial_jaxloss = _initial_jaxloss(obj)
    ratio = initial_jaxloss / initial_score if initial_score > 0 else float("nan")
    ratio_info = _classify_ratio(ratio, ratio_tol)

    n_active = int(np.sum(ff.active_mask))
    summary: dict[str, Any] = {
        "system": system_key,
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

    # ---- Optimization (when allowed) -------------------------------------
    optimized_ff = None
    skip_reason: str | None = None
    if skip_optimization:
        skip_reason = "user_requested"
    elif not ratio_info["ratio_passes"] and ratio_tol is not None:
        skip_reason = "ratio_check_failed" if ratio_info["ratio_status"] == "out_of_band" else "jaxloss_diverged"

    if skip_reason is not None:
        summary["skipped"] = True
        summary["skip_reason"] = skip_reason
        logger.info("[%s] skipping optimization (%s)", system_key, skip_reason)
    else:
        logger.info("[%s] optimizing (ratio=%.3f, n_active=%d)", system_key, ratio, n_active)
        opt_result = _run_optimization(
            sys_data,
            engine,
            ratio_tol=ratio_tol,
            maxiter=maxiter,
            n_evals=n_evals,
            ftol=ftol,
            fc_fraction=fc_fraction,
            eq_fraction=eq_fraction,
        )
        optimized_ff = opt_result.pop("optimized_ff")
        optimized_categories = opt_result.pop("optimized_categories")
        summary.update(opt_result)
        # Real ObjectiveFunction improvement (initial_score and
        # opt_result["final_obj_score"] are both real OF, even when the
        # optimizer was internally driven by a JaxLoss surrogate).
        final_obj = float(opt_result["final_obj_score"])
        summary["improvement_pct"] = 100.0 * (1.0 - final_obj / initial_score) if initial_score > 0 else 0.0
        # Surrogate-only improvement, for diagnosing JaxLoss vs OF
        # mismatch.  Only meaningful when jac_mode == "jax_loss".
        if opt_result.get("jac_mode") == "jax_loss":
            init_surr = float(opt_result["initial_optimizer_score"])
            final_surr = float(opt_result["final_optimizer_score"])
            summary["surrogate_improvement_pct"] = 100.0 * (1.0 - final_surr / init_surr) if init_surr > 0 else 0.0
        summary["optimized"] = optimized_categories
        paper["optimized"] = {
            **optimized_categories,
            "_objective_score": final_obj,
            "_total_refs": sum(cat["n_refs"] for cat in optimized_categories.values()),
        }
        if n_evals > 1:
            init_mean = float(summary["initial_obj_score_mean"])
            init_ci = float(summary["initial_obj_score_ci95"])
            final_mean = float(summary["final_obj_score_mean"])
            final_ci = float(summary["final_obj_score_ci95"])
            ci_pct = 100.0 * (init_ci + final_ci) / init_mean if init_mean > 0 else 0.0
            significance = "SIGNIFICANT" if summary["improvement_significant"] else "NOT SIGNIFICANT"
            logger.info(
                "[%s] optimized: %.3g → %.3g (%.2f%% mean ± %.2f%% CI, %s, %.1fs)",
                system_key,
                init_mean,
                final_mean,
                summary["improvement_pct_mean"],
                ci_pct,
                significance,
                opt_result["opt_time_s"],
            )
        else:
            logger.info(
                "[%s] optimized: %.3f → %.3f (%.2f%% real OF improvement, %.1fs)",
                system_key,
                initial_score,
                final_obj,
                summary["improvement_pct"],
                opt_result["opt_time_s"],
            )

    # ---- Write artifacts -------------------------------------------------
    data_dir_name = DATA_DIR_FOR_SYSTEM.get(system_key, system_key)
    subdir = "convergence" if starting_point == "published" else f"from-{starting_point}"
    sys_out = output_dir / data_dir_name / subdir
    sys_out.mkdir(parents=True, exist_ok=True)

    _write_strict_json(
        sys_out / "validation_results.json",
        {"provenance": provenance, "result": summary},
    )
    _write_strict_json(
        sys_out / "paper_metrics.json",
        {"provenance": provenance, "metrics": paper},
    )
    if optimized_ff is not None:
        ff_path = sys_out / f"{system_key}_optimized.fld"
        optimized_ff.to_mm3_fld(str(ff_path))
        logger.info("[%s] wrote optimized FF: %s", system_key, ff_path)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_ratio_tol(value: str) -> float | None:
    """Parse ratio_tol from a string; ``"none"`` or negative → ``None``."""
    if value.lower() in {"none", "off", "disabled", "-1"}:
        return None
    parsed = float(value)
    if parsed < 0:
        return None
    return parsed


def _parse_positive_int(value: str) -> int:
    """Parse a positive integer CLI value."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--system",
        action="append",
        help="System key to process (repeatable). Defaults to all systems in SYSTEMS.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Root output directory. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--ratio-tol",
        type=_parse_ratio_tol,
        default=0.15,
        help="JaxLoss/ObjFun ratio tolerance (e.g. 0.15 → [0.85, 1.15]). "
        "Use 'none' or negative value to disable the gate.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=500,
        help="Maximum L-BFGS-B iterations per optimization.",
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=1e-8,
        help="L-BFGS-B function-value tolerance. Tighten (e.g. 1e-12) "
        "for from-poor-start runs where the default exits too soon.",
    )
    parser.add_argument(
        "--fc-fraction",
        type=float,
        default=None,
        help="Fractional bounds for force-constant parameters: each FC is "
        "clamped to (val ± fc_fraction*|val|). Use for from-poor-start runs "
        "(e.g. --starting-point qfuerza) to keep the optimizer in the "
        "starting basin. Recommended: 0.20 (i.e. ±20%%). Omit for sanity bounds.",
    )
    parser.add_argument(
        "--eq-fraction",
        type=float,
        default=None,
        help="Fractional bounds for equilibrium parameters (bond_eq, angle_eq, "
        "vdw_radius, ub_eq): each is clamped to (val ± eq_fraction*|val|). "
        "Recommended: 0.05 (i.e. ±5%%). Omit for sanity bounds.",
    )
    parser.add_argument(
        "--n-evals",
        type=_parse_positive_int,
        default=1,
        help="Number of post-hoc ObjectiveFunction evaluations at x0 and x_final for median/CI reporting.",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Compute baseline metrics only; do not optimize any system.",
    )
    parser.add_argument(
        "--starting-point",
        choices=("published", "qfuerza"),
        default="published",
        help="Starting force-field parameters. 'published' uses the literature OPT values "
        "(default, backward compatible). 'qfuerza' overwrites the OPT bond/angle scalars "
        "with QFUERZA (Farrugia 2025) Hessian-derived values while keeping the published "
        "topology and frozen MM3 backbone — used for QFUERZA-recovery validation runs. "
        "Output subdirectory becomes 'from-qfuerza' instead of 'convergence' to avoid "
        "overwriting baselines.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="If set, also write a combined JSON aggregating all systems to this "
        "path. Schema is {provenance: ..., results: {<system>: summary}} — useful "
        "for cross-system review, not a drop-in replacement for any historical "
        "single-file schema.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from q2mm.diagnostics.systems import SYSTEMS

    systems = args.system or list(SYSTEMS.keys())
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"Unknown system(s): {unknown}. Available: {sorted(SYSTEMS.keys())}")

    output_dir = args.output_dir.resolve()
    provenance = _build_provenance(args, output_dir)
    logger.info("Output directory: %s", output_dir)
    logger.info("Systems: %s", systems)
    logger.info(
        "ratio_tol=%s, maxiter=%d, ftol=%.2e, fc_fraction=%s, eq_fraction=%s, n_evals=%d, starting_point=%s",
        args.ratio_tol,
        args.maxiter,
        args.ftol,
        args.fc_fraction,
        args.eq_fraction,
        args.n_evals,
        args.starting_point,
    )

    combined: dict[str, Any] = {}
    failures: list[str] = []
    for sys_key in systems:
        try:
            summary = process_system(
                sys_key,
                output_dir=output_dir,
                ratio_tol=args.ratio_tol,
                maxiter=args.maxiter,
                n_evals=args.n_evals,
                skip_optimization=args.skip_optimization,
                starting_point=args.starting_point,
                provenance=provenance,
                ftol=args.ftol,
                fc_fraction=args.fc_fraction,
                eq_fraction=args.eq_fraction,
            )
            combined[sys_key] = summary
        except Exception:
            logger.exception("[%s] FAILED", sys_key)
            failures.append(sys_key)

    if args.combined_output is not None:
        _write_strict_json(
            args.combined_output,
            {"provenance": provenance, "results": combined},
        )
        logger.info("Wrote combined output: %s", args.combined_output)

    # Batch-level silent-failure detection.  If we optimized any system
    # but every one of them exited at n_iterations <= 2 with negligible
    # change in the real ObjectiveFunction, the batch did not optimize.
    # See AGENTS.md §11 (Benchmark Pre-Flight Checklist).
    optimized = [s for s in combined.values() if not s.get("skipped") and s.get("n_iterations") is not None]
    if optimized and not args.skip_optimization:
        no_progress = [
            s
            for s in optimized
            if int(s.get("n_iterations", 0)) <= 2 and abs(float(s.get("improvement_pct", 0.0))) < 1.0
        ]
        if len(no_progress) == len(optimized):
            logger.error(
                "BATCH FAILURE: all %d optimized system(s) exited at "
                "n_iterations<=2 with |improvement_pct|<1%%. The optimizer "
                "did NOT optimize. Inspect ratio_tol, ftol, bounds, and "
                "starting force field. Systems: %s",
                len(optimized),
                [s.get("system") for s in no_progress],
            )
            if not failures:
                failures.append("batch_no_progress")

    if failures:
        logger.error("Failed systems: %s", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
