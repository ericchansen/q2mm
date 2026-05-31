r"""Compare a QFUERZA-optimized FF against its published counterpart.

Emits a per-parameter markdown table grouped by chemical motif
(metal-involving vs ligand-only) so the QFUERZA-recovery doc can answer
the user's literal question: "do our params end up near the published
params, and where they don't, why?".

Usage
-----
    python scripts/compare_qfuerza_to_published.py \\
        --system rh-enamide \\
        --optimized /path/to/rh-enamide_optimized.fld \\
        --out /tmp/compare-rh-enamide.md

Outputs markdown to ``--out``.  If ``--out`` is omitted, prints to stdout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

# Make `q2mm` importable when run from the worktree.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from q2mm.diagnostics.systems import SYSTEMS  # noqa: E402
from q2mm.models.forcefield import ForceField  # noqa: E402


def _resolve_published_ff_paths(system_key: str) -> list[Path]:
    """Return the .fld file paths the system's loader points at."""
    spec = SYSTEMS[system_key]
    paths: list[Path] = []
    for _, fn in (spec.ff_paths or {}).items():
        with contextlib.suppress(Exception):  # pragma: no cover - missing data dir
            paths.append(Path(fn()))
    return paths


def _load_published(system_key: str) -> ForceField:
    """Load the published FF for a system, composing if necessary."""
    spec = SYSTEMS[system_key]
    paths = _resolve_published_ff_paths(system_key)
    if not paths:
        raise FileNotFoundError(f"No FF paths configured for {system_key}")

    if spec.ff_strategy == "published_opt":
        return ForceField.from_mm3_fld(paths[0], include_standard=False)

    if spec.ff_strategy == "published_opt_composed":
        # Wahlers: base MM3 + OPT substructure file. For per-parameter
        # comparison we only need the OPT block — the backbone MM3 is
        # not optimized. Both the published and optimized .fld files
        # contain only the OPT block, so load them the same way.
        spec_paths = spec.ff_paths or {}
        opt_path = spec_paths["opt_path"]()
        return ForceField.from_mm3_fld(Path(opt_path), include_standard=False)

    raise ValueError(f"Unsupported ff_strategy for {system_key}: {spec.ff_strategy}")


def _bond_motif(elements: tuple[str, ...]) -> str:
    metals = {"Rh", "Pd", "Os", "Ru"}
    if any(e in metals for e in elements):
        return "M-L bond"
    if all(e == "C" for e in elements):
        return "C-C bond"
    if "H" in elements:
        return "X-H bond"
    return "ligand bond"


def _angle_motif(elements: tuple[str, ...]) -> str:
    metals = {"Rh", "Pd", "Os", "Ru"}
    if elements[1] in metals:
        return "L-M-L angle"
    if any(e in metals for e in elements):
        return "M-L-X angle"
    if all(e == "C" for e in elements):
        return "C-C-C angle"
    return "ligand angle"


def _diff_row(
    cat: str,
    motif: str,
    label: str,
    pub: float,
    opt: float,
    units: str,
) -> dict:
    delta = opt - pub
    rel = (abs(delta) / abs(pub) * 100.0) if pub != 0 else float("nan")
    return {
        "category": cat,
        "motif": motif,
        "label": label,
        "pub": pub,
        "opt": opt,
        "abs_dev": delta,
        "rel_dev_pct": rel,
        "units": units,
    }


def _iter_diffs(pub: ForceField, opt: ForceField) -> Iterable[dict]:
    # Comparison tooling — fail fast on topology mismatches instead of
    # silently producing an incomplete diff.  zip() truncating to the
    # shorter list or skipping mismatched keys would let parameters
    # vanish from the rollup tables without warning.
    if len(pub.bonds) != len(opt.bonds):
        raise ValueError(
            f"Bond-row count mismatch: published has {len(pub.bonds)} bonds, "
            f"optimized has {len(opt.bonds)}.  The two force fields must "
            "share topology for a per-parameter diff to be meaningful."
        )
    if len(pub.angles) != len(opt.angles):
        raise ValueError(
            f"Angle-row count mismatch: published has {len(pub.angles)} angles, "
            f"optimized has {len(opt.angles)}.  The two force fields must "
            "share topology for a per-parameter diff to be meaningful."
        )

    # Bonds
    for i, (bp, bo) in enumerate(zip(pub.bonds, opt.bonds)):
        if bp.frozen and bo.frozen:
            continue
        if bp.key != bo.key:
            raise ValueError(
                f"Bond-row key mismatch at index {i}: published key={bp.key!r}, "
                f"optimized key={bo.key!r}.  Cannot diff parameters between "
                "non-aligned force fields."
            )
        motif = _bond_motif(bp.elements)
        label = "-".join(bp.elements)
        yield _diff_row("bond_eq", motif, label, bp.equilibrium, bo.equilibrium, "Å")
        yield _diff_row("bond_fc", motif, label, bp.force_constant, bo.force_constant, "kcal/mol/Å²")

    # Angles
    for i, (ap, ao) in enumerate(zip(pub.angles, opt.angles)):
        if ap.frozen and ao.frozen:
            continue
        if ap.key != ao.key:
            raise ValueError(
                f"Angle-row key mismatch at index {i}: published key={ap.key!r}, "
                f"optimized key={ao.key!r}.  Cannot diff parameters between "
                "non-aligned force fields."
            )
        motif = _angle_motif(ap.elements)
        label = "-".join(ap.elements)
        yield _diff_row("angle_eq", motif, label, ap.equilibrium, ao.equilibrium, "deg")
        yield _diff_row("angle_fc", motif, label, ap.force_constant, ao.force_constant, "kcal/mol/rad²")


def _summary_table(diffs: list[dict]) -> str:
    by_cat = defaultdict(list)
    for d in diffs:
        by_cat[d["category"]].append(d)

    lines = [
        "| Category | n | mean |Δ| | max |Δ| | mean rel% | max rel% | units |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cat in ["bond_eq", "bond_fc", "angle_eq", "angle_fc"]:
        rows = by_cat.get(cat, [])
        if not rows:
            continue
        absdevs = [abs(r["abs_dev"]) for r in rows]
        reldevs = [r["rel_dev_pct"] for r in rows if r["rel_dev_pct"] == r["rel_dev_pct"]]
        units = rows[0]["units"]
        lines.append(
            f"| `{cat}` | {len(rows)} | {sum(absdevs) / len(absdevs):.3g} | "
            f"{max(absdevs):.3g} | "
            f"{(sum(reldevs) / len(reldevs)) if reldevs else float('nan'):.2f} | "
            f"{(max(reldevs)) if reldevs else float('nan'):.2f} | {units} |"
        )
    return "\n".join(lines)


def _top_deviations(diffs: list[dict], n: int = 10) -> str:
    ranked = sorted(
        diffs,
        key=lambda d: d["rel_dev_pct"] if d["rel_dev_pct"] == d["rel_dev_pct"] else 0,
        reverse=True,
    )
    lines = [
        "| # | Category | Motif | Atoms | Published | QF-Optimized | Abs Δ | Rel Δ% |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, d in enumerate(ranked[:n], 1):
        lines.append(
            f"| {i} | `{d['category']}` | {d['motif']} | `{d['label']}` | "
            f"{d['pub']:.4g} | {d['opt']:.4g} | {d['abs_dev']:+.4g} | "
            f"{d['rel_dev_pct']:.1f}% |"
        )
    return "\n".join(lines)


def _by_motif(diffs: list[dict]) -> str:
    """Per-motif rollup table — useful for spotting chemical patterns."""
    by_motif: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in diffs:
        by_motif[(d["category"], d["motif"])].append(d)

    lines = [
        "| Category | Motif | n | mean |Δ| | max |Δ| | mean rel% |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (cat, motif), rows in sorted(by_motif.items()):
        absdevs = [abs(r["abs_dev"]) for r in rows]
        reldevs = [r["rel_dev_pct"] for r in rows if r["rel_dev_pct"] == r["rel_dev_pct"]]
        mean_rel = (sum(reldevs) / len(reldevs)) if reldevs else float("nan")
        lines.append(
            f"| `{cat}` | {motif} | {len(rows)} | {sum(absdevs) / len(absdevs):.3g} | "
            f"{max(absdevs):.3g} | {mean_rel:.2f} |"
        )
    return "\n".join(lines)


def main() -> int:
    """Entry point: parse CLI args and emit the per-parameter comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--optimized", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the raw diff list as JSON.",
    )
    args = ap.parse_args()

    pub = _load_published(args.system)
    opt = ForceField.from_mm3_fld(args.optimized, include_standard=False)
    diffs = list(_iter_diffs(pub, opt))

    md_parts = [
        f"## Per-parameter comparison: {args.system}",
        "",
        f"_Source: published FF (loader-resolved) vs `{args.optimized.name}`._",
        "",
        f"Total comparable active parameter rows: **{len(diffs) // 2}** (each row contributes one eq and one fc cell)",
        "",
        "### Summary by category",
        "",
        _summary_table(diffs),
        "",
        "### Summary by chemical motif",
        "",
        _by_motif(diffs),
        "",
        "### Top 15 largest relative deviations",
        "",
        _top_deviations(diffs, n=15),
        "",
    ]
    md = "\n".join(md_parts)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(md)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(diffs, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {args.json_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
