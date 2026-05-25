"""Diagnose Heck-relay Seminario divergence (ericchansen/q2mm#277).

Builds three no-optimization baselines for the Heck-relay system and
evaluates each against the QM training set:

A. Untouched Rosales FF — ForceField.from_mm3_fld(include_standard=True),
   no freeze_standard_params, no Seminario re-estimation.
B. Pre-fix loader pattern — the BUGGY combination that load_heck_relay
   used before #277 was fixed: ForceField.from_mm3_fld +
   freeze_standard_params + qfuerza_into(ff, molecules,
   invert_ts_curvature=True).  Reconstructed inline (NOT by calling
   the current loader), so this diagnostic stays valid as a
   regression-detection tool after the fix lands.
C. Seminario-only — ForceField.from_mm3_fld(include_standard=False)
   (Rosales OPT block as the template), then Seminario re-estimates
   every active param (no freeze step).

For each baseline this script reports:

- Per-category R² / RMSD / MAE (bond_length, bond_angle, eig_diagonal)
- ObjectiveFunction score
- Whether JaxLoss returns a finite value
- Worst-10 bond-length residuals (with atom-pair labels)

Output: strict JSON at
``../q2mm-data/benchmarks/heck-relay/diagnostic/three_baseline_comparison.json``
with embedded provenance, plus a human-readable summary on stdout.

Time-boxed per #277; not a benchmark — won't replace
``regenerate_convergence_results.py``.
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
# script has no dependency on regenerate_convergence_results.py internals).
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


def _per_category_metrics(obj: Any, ff: Any) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Compute per-category fit metrics and per-bond residuals.

    Returns ``(category_stats, bond_residuals)`` where *bond_residuals*
    contains one dict per bond_length reference: molecule_idx,
    atom_indices, ref_value, calc_value, abs_residual.
    """
    residuals = obj._compute_residuals(ff)  # noqa: SLF001
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    bond_records: list[dict[str, Any]] = []
    for ref, weighted in zip(obj.reference.values, residuals, strict=True):
        if ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        buckets[ref.kind].append((float(ref.value), calc_value))
        if ref.kind == "bond_length":
            bond_records.append(
                {
                    "molecule_idx": int(ref.molecule_idx),
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


def _per_molecule_r2(obj: Any, ff: Any) -> list[dict[str, float]]:
    """Compute per-molecule eig_diagonal R² (the system-level loss target)."""
    residuals = obj._compute_residuals(ff)  # noqa: SLF001
    per_mol: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for ref, weighted in zip(obj.reference.values, residuals, strict=True):
        if ref.kind != "eig_diagonal" or ref.weight == 0.0:
            continue
        raw_residual = float(weighted) / float(ref.weight)
        calc_value = float(ref.value) - raw_residual
        per_mol[ref.molecule_idx].append((float(ref.value), calc_value))
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
    from q2mm.diagnostics.systems import _resolve_supporting_info_dir

    si = _resolve_supporting_info_dir()
    return si / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"


def build_ff_a_untouched(molecules: list[Any]) -> Any:
    """Build baseline A — untouched Rosales FF (no Seminario, no freeze)."""
    from q2mm.models.forcefield import ForceField, FunctionalForm

    ff = ForceField.from_mm3_fld(str(_ff_path()), include_standard=True)
    ff.functional_form = FunctionalForm.MM3
    return ff


def build_ff_b_prefix_loader_pattern(molecules: list[Any]) -> Any:
    """Build baseline B — explicit reconstruction of the PRE-FIX loader bug.

    Reproduces the pre-#277 ``load_heck_relay()`` exactly:
    ``from_mm3_fld(include_standard=True)`` + ``freeze_standard_params``
    + ``qfuerza_into(ff, molecules, invert_ts_curvature=True)``.  This
    is the bug we are diagnosing — we do **not** call the current
    loader, because the current loader was fixed in #280 and no longer
    reproduces the bug.  Keeping the buggy pattern inline here means
    the diagnostic stays valid as a regression-detection tool even
    after the fix lands.
    """
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import qfuerza_into

    ff = ForceField.from_mm3_fld(str(_ff_path()), include_standard=True)
    opt_ff = ForceField.from_mm3_fld(str(_ff_path()), include_standard=False)
    ff.freeze_standard_params(opt_ff)
    qfuerza_into(ff, molecules, invert_ts_curvature=True)
    ff.functional_form = FunctionalForm.MM3
    return ff


def build_ff_c_seminario_only(molecules: list[Any]) -> Any:
    """Build baseline C — Seminario over the OPT block, no published values."""
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import qfuerza_into

    ff_template = ForceField.from_mm3_fld(str(_ff_path()), include_standard=False)
    ff = ff_template.copy()
    qfuerza_into(ff, molecules, invert_ts_curvature=True)
    ff.functional_form = FunctionalForm.MM3
    return ff


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(
    label: str,
    ff: Any,
    molecules: list[Any],
    reference: Any,
    engine: Any,
) -> dict[str, Any]:
    """Compute the full diagnostic block for one baseline."""
    from q2mm.optimizers.objective import ObjectiveFunction

    obj = ObjectiveFunction(ff, engine, molecules, reference)
    x = ff.get_param_vector()
    obj_score = float(obj(x))

    categories, bond_records = _per_category_metrics(obj, ff)
    per_molecule = _per_molecule_r2(obj, ff)

    bond_records.sort(key=lambda r: -r["abs_residual"])
    worst_10 = bond_records[:10]

    jaxloss_val: float | None = None
    jaxloss_finite: bool | None = None
    try:
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.optimizers.jaxloss import JaxLoss

        if isinstance(engine, JaxEngine):
            spec = obj.to_jax_spec()
            jl = JaxLoss(spec, engine, molecules, ff)
            val_jax, _ = jl.value_and_grad_jax(x)
            raw = float(val_jax)
            jaxloss_finite = math.isfinite(raw)
            jaxloss_val = raw if jaxloss_finite else None
    except Exception as exc:
        logger.warning("[%s] JaxLoss evaluation failed: %s", label, exc)
        jaxloss_finite = False

    n_active = int(np.sum(ff.active_mask))
    n_total = int(ff.n_params)

    return {
        "label": label,
        "n_total_params": n_total,
        "n_active_params": n_active,
        "objective_score": obj_score,
        "jaxloss": jaxloss_val,
        "jaxloss_finite": jaxloss_finite,
        "ratio": (jaxloss_val / obj_score) if (jaxloss_val is not None and obj_score > 0) else None,
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

    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.diagnostics.systems import load_heck_relay_molecules
    from q2mm.optimizers.objective import ReferenceData

    logger.info("Loading 23 Heck-relay molecules + QM Hessians")
    engine = JaxEngine()
    molecules = load_heck_relay_molecules()
    reference = ReferenceData.from_molecules(molecules, eigenmatrix_diagonal_only=True)
    logger.info(
        "Loaded %d molecules; %d reference values across categories",
        len(molecules),
        len(reference.values),
    )

    baselines: list[tuple[str, Any]] = [
        ("A_untouched_rosales", build_ff_a_untouched(molecules)),
        ("B_prefix_loader_pattern", build_ff_b_prefix_loader_pattern(molecules)),
        ("C_seminario_only", build_ff_c_seminario_only(molecules)),
    ]

    results: dict[str, Any] = {}
    for label, ff in baselines:
        logger.info("Evaluating baseline %s", label)
        results[label] = evaluate_baseline(label, ff, molecules, reference, engine)
        r = results[label]
        cats = r["categories"]
        logger.info(
            "[%s] obj=%.3e ratio=%s bond_len R²=%.3f bond_ang R²=%.3f eig_diag R²=%.3f",
            label,
            r["objective_score"],
            f"{r['ratio']:.3e}" if r["ratio"] is not None else "n/a",
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
    for label, _ in baselines:
        r = results[label]
        ratio_s = f"{r['ratio']:.2e}" if r["ratio"] is not None else "n/a"
        r2_bond = r["categories"].get("bond_length", {}).get("r2", float("nan"))
        print(f"  {label:<20} {r['n_active_params']:>9d} {r['objective_score']:>12.3e} {ratio_s:>12} {r2_bond:>14.3f}")

    print()
    print("Decision rule (per issue #277):")
    print("  A's bond_length R² >> B's   → bug in freeze_standard_params + Seminario interaction")
    print("  A ≈ B ≈ C                   → bug upstream (MM3 evaluator / atom types)")
    print("  A fine, B explodes          → narrow on Seminario behavior with freeze_standard_params")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
