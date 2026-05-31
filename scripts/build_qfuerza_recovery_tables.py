r"""Generate the R²/RMSD comparison tables for docs/benchmarks/qfuerza-recovery.md.

For each TS system, emits a markdown table comparing:

- q2mm starting from QFUERZA (`from-qfuerza/`)
- q2mm starting from published OPT (`convergence/`)
- published-paper goodness-of-fit (from `paper_r2.json`, optional)

Per-system tables cover three metrics: bond_length, bond_angle, eig_diagonal.

Usage
-----
    python scripts/build_qfuerza_recovery_tables.py \\
        --data-dir /home/eric/repos/q2mm-data/benchmarks \\
        --paper-r2 /tmp/qfuerza-rerun/paper_r2.json \\
        --out /tmp/qfuerza-recovery-tables.md

If ``--paper-r2`` is omitted, the table's "Published paper" column is filled
with ``—`` (literature-not-fetched placeholder).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SYSTEM_ORDER = [
    ("rh-enamide", "rh-enamide"),
    ("pd-allyl-amination", "pd-allyl"),
    ("pd-1,4-conjugate-addition", "pd-conjugate"),
    ("rh-1,4-conjugate-addition", "rh-conjugate"),
    ("heck-relay", "heck-relay"),
]

CATEGORIES = [
    ("bond_length", "Bond length", "Å"),
    ("bond_angle", "Bond angle", "deg"),
    ("eig_diagonal", "Hessian eig (diag)", "mdyn/Å"),
]


def _load_run(data_dir: Path, sys_dir: str, sub: str) -> dict[str, Any] | None:
    p = data_dir / sys_dir / sub / "validation_results.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["result"]


def _fmt(v: Any, prec: int = 4) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.{prec}g}"
    except (TypeError, ValueError):
        return str(v)


def _system_table(
    label: str,
    pub_run: dict[str, Any] | None,
    qf_run: dict[str, Any] | None,
    paper_r2: dict[str, Any] | None,
) -> str:
    lines = [
        f"### {label}",
        "",
        "| Metric | Published paper R²/RMSD | q2mm @ published start | q2mm @ QFUERZA start |",
        "|---|---:|---:|---:|",
    ]
    for key, name, units in CATEGORIES:
        # Map q2mm category name to paper category name.
        paper_key = {
            "bond_length": "bond_length",
            "bond_angle": "bond_angle",
            "eig_diagonal": "eigenvalue",
        }[key]
        paper_cell = "—"
        if paper_r2 is not None:
            p = paper_r2.get(paper_key)
            if p is not None:
                r2 = _fmt(p.get("r2"))
                rmsd = _fmt(p.get("rmsd"))
                paper_cell = f"R²={r2} / RMSD={rmsd} {p.get('units', units)}"

        def cell(run: dict[str, Any] | None, sub: str) -> str:
            if run is None:
                return "—"
            d = run.get(sub, {}).get(key)
            if not d:
                return "—"
            return f"R²={_fmt(d.get('r2'))} / RMSD={_fmt(d.get('rmsd'))} {units}"

        pub_opt = cell(pub_run, "optimized")
        qf_opt = cell(qf_run, "optimized")
        lines.append(f"| {name} | {paper_cell} | {pub_opt} | {qf_opt} |")

    # Add objective-function row.
    def obj_cell(run: dict[str, Any] | None) -> str:
        if run is None:
            return "—"
        v = run.get("final_obj_score")
        return f"{v:.3e}" if v is not None else "—"

    lines.append(f"| Final OF | — | {obj_cell(pub_run)} | {obj_cell(qf_run)} |")
    # n_iterations row for transparency.
    lines.append(
        "| Optimizer L-BFGS-B iters | — | "
        f"{pub_run['n_iterations'] if pub_run else '—'} | "
        f"{qf_run['n_iterations'] if qf_run else '—'} |"
    )

    if paper_r2 is None or paper_r2.get("notes"):
        notes = (paper_r2 or {}).get("notes", "")
        if notes:
            lines += ["", f"_Paper note: {notes}_"]

    return "\n".join(lines)


def main() -> int:
    """Entry point: parse CLI args and emit the comparison tables."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--paper-r2", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    paper_r2_all: dict[str, Any] = {}
    if args.paper_r2 and args.paper_r2.exists():
        paper_r2_all = json.loads(args.paper_r2.read_text())

    parts: list[str] = [
        "## R² / RMSD comparison: published paper vs q2mm @ published vs q2mm @ QFUERZA",
        "",
        "Per-system, per-category goodness of fit between MM predictions and the QM training data. "
        "_Same reference data is used for both q2mm columns_ (the published TSFF papers use the same training set, "
        "evaluated through MacroModel/MM3* instead of q2mm/JaxEngine).",
        "",
    ]
    for sys_dir, sys_short in SYSTEM_ORDER:
        pub_run = _load_run(args.data_dir, sys_dir, "convergence")
        qf_run = _load_run(args.data_dir, sys_dir, "from-qfuerza")
        paper_r2 = paper_r2_all.get(sys_short)
        parts.append(_system_table(sys_short, pub_run, qf_run, paper_r2))
        parts.append("")

    md = "\n".join(parts)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
