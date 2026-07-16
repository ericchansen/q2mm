"""Diagnose Heck-relay Seminario divergence (ericchansen/q2mm#277).

Builds three no-optimization baselines for the Heck-relay system and
evaluates each against the QM training set:

A. Untouched Rosales FF — ``load_mm3_fld(include_standard=True)``, no
   active/frozen partition applied, no Seminario re-estimation.
B. Current (fixed) loader pattern — ``load_mm3_fld(include_standard=True)``
   composed with the OPT-only block via ``opt_substructure_membership`` +
   ``ActiveParameterSpace.from_membership``, then ``qfuerza_into(ff,
   molecules, active_bonds=..., active_angles=..., active_torsions=...,
   invert_ts_curvature=True)`` re-estimates only the OPT-substructure
   parameters.  Reconstructed inline (NOT by calling
   ``q2mm.benchmarks.systems.heck_relay.load``), so this diagnostic
   stays independent of the production loader.  Note: the #277 bug
   itself — ``ForceField.freeze_standard_params()`` silently mutating
   rows in place and interacting badly with a mutating ``qfuerza_into``
   — is now structurally impossible: parameter values are immutable,
   ``qfuerza_into`` is pure and takes explicit ``active_bonds``/
   ``active_angles``/``active_torsions`` index sets, and frozen/active
   state lives only in :class:`~q2mm.models.parameters.ActiveParameterSpace`,
   never on the force field itself.  Baseline B therefore now shows the
   *fixed* loader pattern rather than the pre-fix bug — kept for its
   original diagnostic value (comparing "backbone kept as literature
   values" against A and C).
C. Seminario-only — ``load_mm3_fld(include_standard=False)`` (Rosales
   OPT block as the template), then Seminario re-estimates every active
   param (no freeze step; every parameter in the OPT-only block is
   active by construction).

For each baseline this script reports:

- Per-category R² / RMSD / MAE (bond_length, bond_angle, eig_diagonal)
- Objective-executor score
- Whether the JAX executor returns a finite value
- Worst-10 bond-length residuals (with atom-pair labels)

Output: strict JSON at
``../q2mm-data/benchmarks/heck-relay/diagnostic/three_baseline_comparison.json``
with embedded provenance, plus a human-readable summary on stdout.

Time-boxed per #277; not a benchmark — won't replace
``scripts/benchmark.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("diag-heck")
REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Provenance + JSON helpers (deliberately small re-implementations so the
# script has no dependency on scripts/benchmark.py internals).
# ---------------------------------------------------------------------------


def _git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"git_sha": sha, "git_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def _find_git_repo_root(start: Path) -> Path | None:
    """Walk parents from *start* until a ``.git`` entry is found.

    Returns the directory containing ``.git``, or ``None`` if no git
    repo is found before reaching the filesystem root.  ``.git`` may be
    a directory (regular repo) or a file (worktrees / submodules).
    """
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None


def _device_info() -> dict[str, Any]:
    """Probe JAX device info — best effort, never raises."""
    info: dict[str, Any] = {}
    try:
        import jax

        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:
        info["jax_devices_error"] = repr(exc)
    return info


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, np.floating):
        return _sanitize(float(value))
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(_sanitize(payload), fh, indent=2, allow_nan=False, sort_keys=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Per-category metrics
# ---------------------------------------------------------------------------


def _category_stats(ref_values: np.ndarray, calc_values: np.ndarray) -> dict[str, float]:
    n = int(ref_values.size)
    if n == 0:
        return {"n_refs": 0, "r2": float("nan"), "rmsd": float("nan"), "mae": float("nan")}
    residuals = ref_values - calc_values
    rmsd = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))
    mean_ref = float(np.mean(ref_values))
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((ref_values - mean_ref) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"n_refs": n, "r2": r2, "rmsd": rmsd, "mae": mae}


def _per_category_metrics(evaluator: Any, x: Any) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Compute per-category fit metrics and per-bond residuals."""
    evaluation = evaluator.evaluate(x)
    residuals = evaluation.weighted_residuals
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    bond_records: list[dict[str, Any]] = []
    for ref, weighted in zip(evaluator.plan.observations.values, residuals, strict=True):
        if ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        buckets[ref.kind].append((float(ref.value), calc_value))
        if ref.kind == "bond_length":
            bond_records.append(
                {
                    "molecule_idx": int(ref.case_id),
                    "atom_indices": list(ref.atom_indices) if ref.atom_indices is not None else None,
                    "ref_value": float(ref.value),
                    "calc_value": calc_value,
                    "abs_residual": abs(raw_residual),
                }
            )

    category_stats = {
        kind: _category_stats(
            np.array([p[0] for p in pairs]),
            np.array([p[1] for p in pairs]),
        )
        for kind, pairs in buckets.items()
    }
    return category_stats, bond_records


def _per_molecule_r2(evaluator: Any, x: Any) -> list[dict[str, float]]:
    """Compute per-molecule eig_diagonal R² (the system-level loss target)."""
    residuals = evaluator.evaluate(x).weighted_residuals
    per_mol: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for ref, weighted in zip(evaluator.plan.observations.values, residuals, strict=True):
        if ref.kind != "eig_diagonal" or ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        per_mol[int(ref.case_id)].append((float(ref.value), calc_value))
    out: list[dict[str, float]] = []
    for idx in sorted(per_mol):
        pairs = per_mol[idx]
        ref_arr = np.array([p[0] for p in pairs])
        calc_arr = np.array([p[1] for p in pairs])
        out.append({"molecule_idx": idx, **_category_stats(ref_arr, calc_arr)})
    return out


# ---------------------------------------------------------------------------
# Force-field builders for the three baselines
# ---------------------------------------------------------------------------


def _ff_path() -> Path:
    """Resolve the Rosales Heck-relay FF file path."""
    from q2mm.benchmarks.systems._paths import resolve_supporting_info_dir

    si = resolve_supporting_info_dir()
    return si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"


def build_ff_a_untouched(molecules: list[Any]) -> tuple[Any, int]:
    """Build baseline A — untouched Rosales FF (no Seminario, no freeze).

    Returns:
        ``(ff, n_active)`` — every parameter counts as "active" since no
        frozen/active partition is applied.

    """
    from q2mm.io.mm3 import load_mm3_fld
    from q2mm.models.parameters import ParameterLayout

    del molecules  # unused — baseline A applies no Seminario re-estimation
    ff = load_mm3_fld(str(_ff_path()), include_standard=True)
    n_active = len(ParameterLayout.from_force_field(ff))
    return ff, n_active


def build_ff_b_prefix_loader_pattern(molecules: list[Any]) -> tuple[Any, int]:
    """Build baseline B — the current (fixed) loader pattern, reconstructed inline.

    Composes the full Rosales FF with the OPT-only block via
    :func:`~q2mm.models.parameters.opt_substructure_membership` and
    :meth:`~q2mm.models.parameters.ActiveParameterSpace.from_membership`,
    then ``qfuerza_into`` re-estimates only the OPT-substructure
    bonds/angles/torsions.  This mirrors
    ``q2mm.benchmarks.systems.heck_relay.load(starting_point="qfuerza")``
    without calling it directly, so this diagnostic stays independent
    of the production loader.  See the module docstring for why this is
    no longer "the pre-fix bug" — that bug's root cause (mutable
    ``.frozen`` row state) no longer exists.

    Returns:
        ``(ff, n_active)`` where *n_active* is the OPT-substructure
        parameter count from :class:`~q2mm.models.parameters.ActiveParameterSpace`.

    """
    from q2mm.io.mm3 import load_mm3_fld
    from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout, opt_substructure_membership
    from q2mm.models.seminario import qfuerza_into

    ff = load_mm3_fld(str(_ff_path()), include_standard=True)
    opt_ff = load_mm3_fld(str(_ff_path()), include_standard=False)
    membership = opt_substructure_membership(ff, opt_ff)
    layout = ParameterLayout.from_force_field(ff)
    ff = qfuerza_into(
        ff,
        molecules,
        active_bonds=membership.bonds,
        active_angles=membership.angles,
        active_torsions=membership.torsions,
        invert_ts_curvature=True,
    )
    space = ActiveParameterSpace.from_membership(layout, ff, membership)
    return ff, space.n_active


def build_ff_c_seminario_only(molecules: list[Any]) -> tuple[Any, int]:
    """Build baseline C — Seminario over the OPT block, no published values.

    Returns:
        ``(ff, n_active)`` — every parameter in the OPT-only block is
        active by construction (no frozen backbone).

    """
    from q2mm.io.mm3 import load_mm3_fld
    from q2mm.models.parameters import ParameterLayout
    from q2mm.models.seminario import qfuerza_into

    ff_template = load_mm3_fld(str(_ff_path()), include_standard=False)
    ff = qfuerza_into(ff_template, molecules, invert_ts_curvature=True)
    n_active = len(ParameterLayout.from_force_field(ff))
    return ff, n_active


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(
    label: str,
    ff: Any,
    n_active: int,
    molecules: list[Any],
    reference: Any,
    backend: Any,
) -> dict[str, Any]:
    """Compute the full diagnostic block for one baseline."""
    from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
    from q2mm.models.problem import StationaryPointKind
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor

    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    plan = ObjectivePlan(
        case_ids=tuple(str(i) for i in range(len(molecules))),
        molecules=tuple(molecules),
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in molecules),
        observations=reference,
        layout=layout,
        active_space=space,
    )
    evaluator = PythonObjectiveExecutor(plan, backend, ff)
    x = layout.vector(ff)
    obj_score = float(evaluator.value(x))

    categories, bond_records = _per_category_metrics(evaluator, x)
    per_molecule = _per_molecule_r2(evaluator, x)

    bond_records.sort(key=lambda r: -r["abs_residual"])
    worst_10 = bond_records[:10]

    jax_score_val: float | None = None
    jax_score_finite: bool | None = None
    try:
        from q2mm.backends.mm.jax_engine import JaxBackend
        from q2mm.objectives.jax import JaxObjectiveExecutor

        if isinstance(backend, JaxBackend):
            jax_eval = JaxObjectiveExecutor(plan, backend, ff)
            raw = float(jax_eval.value(x))
            jax_score_finite = math.isfinite(raw)
            jax_score_val = raw if jax_score_finite else None
    except Exception as exc:
        logger.warning("[%s] JAX-executor evaluation failed: %s", label, exc)
        jax_score_finite = False

    n_total = len(layout)

    return {
        "label": label,
        "n_total_params": n_total,
        "n_active_params": n_active,
        "objective_score": obj_score,
        "jax_score": jax_score_val,
        "jax_score_finite": jax_score_finite,
        "executor_ratio": (jax_score_val / obj_score) if (jax_score_val is not None and obj_score > 0) else None,
        "categories": categories,
        "per_molecule_eig_diagonal_r2": per_molecule,
        "worst_10_bond_length_residuals": worst_10,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the three-baseline Heck-relay diagnostic and write the JSON report."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT.parent
        / "q2mm-data"
        / "benchmarks"
        / "heck-relay"
        / "diagnostic"
        / "three_baseline_comparison.json",
        help="Output JSON path (default: q2mm-data/benchmarks/heck-relay/diagnostic/three_baseline_comparison.json).",
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

    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems.heck_relay import load_molecules as load_heck_relay_molecules
    from q2mm.models.observations import ObservationSet

    logger.info("Loading 23 Heck-relay molecules + QM Hessians")
    backend = JaxBackend()
    molecules = load_heck_relay_molecules()
    reference = ObservationSet.from_molecules(
        molecules,
        case_ids=[str(i) for i in range(len(molecules))],
        eigenmatrix_diagonal_only=True,
    )
    logger.info(
        "Loaded %d molecules; %d reference values across categories",
        len(molecules),
        len(reference.values),
    )

    baselines: list[tuple[str, Any, int]] = [
        ("A_untouched_rosales", *build_ff_a_untouched(molecules)),
        ("B_prefix_loader_pattern", *build_ff_b_prefix_loader_pattern(molecules)),
        ("C_seminario_only", *build_ff_c_seminario_only(molecules)),
    ]

    results: dict[str, Any] = {}
    for label, ff, n_active in baselines:
        logger.info("Evaluating baseline %s", label)
        results[label] = evaluate_baseline(label, ff, n_active, molecules, reference, backend)
        r = results[label]
        cats = r["categories"]
        logger.info(
            "[%s] obj=%.3e ratio=%s bond_len R²=%.3f bond_ang R²=%.3f eig_diag R²=%.3f",
            label,
            r["objective_score"],
            f"{r['executor_ratio']:.3e}" if r["executor_ratio"] is not None else "n/a",
            cats.get("bond_length", {}).get("r2", float("nan")),
            cats.get("bond_angle", {}).get("r2", float("nan")),
            cats.get("eig_diagonal", {}).get("r2", float("nan")),
        )

    data_repo = _find_git_repo_root(args.output.resolve()) or args.output.parent
    provenance = {
        "generator": "scripts/diagnose_heck_relay.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": shlex.join(sys.argv),
        "q2mm": _git_info(REPO_ROOT),
        "q2mm_data": _git_info(data_repo),
        "devices": _device_info(),
    }

    payload = {
        "provenance": provenance,
        "issue": "ericchansen/q2mm#277",
        "results": results,
    }
    _write_json(args.output, payload)
    logger.info("Wrote %s", args.output)

    # ---- Human-readable decision summary ---------------------------------
    print()
    print("=" * 70)
    print("Three-baseline summary (Heck relay)")
    print("=" * 70)
    print(f"{'Baseline':<22} {'n_active':>9} {'obj':>12} {'ratio':>12} {'R²(bond_len)':>14}")
    for label, _, _ in baselines:
        r = results[label]
        ratio_s = f"{r['executor_ratio']:.2e}" if r["executor_ratio"] is not None else "n/a"
        r2_bond = r["categories"].get("bond_length", {}).get("r2", float("nan"))
        print(f"  {label:<20} {r['n_active_params']:>9d} {r['objective_score']:>12.3e} {ratio_s:>12} {r2_bond:>14.3f}")

    print()
    print("Decision rule (per issue #277, historical — the bug's root cause")
    print("no longer exists post-phase-2; kept for the 3-baseline comparison):")
    print("  A's bond_length R² >> B's   → backbone-vs-OPT-only estimation gap")
    print("  A ≈ B ≈ C                   → bug upstream (MM3 evaluator / atom types)")
    print("  A fine, B explodes          → narrow on Seminario/active-space interaction")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
