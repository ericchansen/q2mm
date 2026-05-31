"""Row-by-row OPT-parameter comparison between two .fld files.

Compares OPT-substructure parameters between two ``.fld`` files
without going through the q2mm ForceField loader.

Bypasses the loader's atom-type-table lookup so it can handle:
- Wahlers OPT-only files (numeric atom-type indices, no own atom table)
- q2mm-saved optimized files (atom-type-name tokens, "AUTO" header)

Strategy: extract OPT rows in source order from each file, match them
by position (same q2mm run produces same row order), and report per-row
deviations.  Rows are classified as bond (col 0 = "1") or angle ("2").

Usage
-----
    python scripts/compare_opt_rows.py \
        --published /path/to/published.fld \
        --optimized /path/to/_optimized.fld \
        --system rh-enamide \
        --out /tmp/compare-rh-enamide.md
"""

from __future__ import annotations

import argparse
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple


_BOND_RE = re.compile(r"^\s*1\s+(\S+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")
_ANGLE_RE = re.compile(r"^\s*2\s+(\S+)\s+(\S+)\s+(\S+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)")


class Row(NamedTuple):
    """One parsed OPT row (bond or angle) from an MM3 .fld file."""

    kind: str  # "bond" or "angle"
    atoms: tuple[str, ...]
    eq: float
    fc: float
    line_no: int


def _is_metal(token: str) -> bool:
    metals = {"Pd", "PD", "Rh", "RH", "Os", "OS", "Ru", "RU", "55", "56", "57"}
    return token.upper() in {m.upper() for m in metals}


def _bond_motif(atoms: tuple[str, ...]) -> str:
    if any(_is_metal(a) for a in atoms):
        return "M-L bond"
    if "H" in atoms or "H1" in atoms or "H2" in atoms:
        return "X-H bond"
    return "ligand bond"


def _angle_motif(atoms: tuple[str, ...]) -> str:
    if _is_metal(atoms[1]):
        return "L-M-L angle"
    if any(_is_metal(a) for a in atoms):
        return "M-L-X angle"
    if "H" in atoms or "H1" in atoms or "H2" in atoms:
        return "X-H-Y angle"
    return "ligand angle"


def _parse_opt_block(path: Path) -> list[Row]:
    """Extract bond and angle rows from the OPT block of an MM3 .fld file.

    The OPT block starts after a header of the form ``9  AUTO`` or
    ``9  <SMILES>`` and continues to the next section or EOF.  Within
    the block, bond rows start with ``1`` and angle rows start with
    ``2`` (column 1 of the file).
    """
    text = path.read_text().splitlines()
    rows: list[Row] = []
    in_opt = False
    for i, line in enumerate(text, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        # OPT block header: starts with " 9" followed by AUTO or SMILES
        if re.match(r"^\s*9\s+\S+", line):
            in_opt = True
            continue
        if not in_opt:
            continue
        # End of OPT block: any "C  " comment-block line or a new "9 " header
        if stripped.startswith("C  ") or re.match(r"^[A-Z]\s", line):
            in_opt = False
            continue
        m = _BOND_RE.match(line)
        if m:
            rows.append(Row("bond", (m.group(1), m.group(2)), float(m.group(3)), float(m.group(4)), i))
            continue
        m = _ANGLE_RE.match(line)
        if m:
            rows.append(Row("angle", (m.group(1), m.group(2), m.group(3)), float(m.group(4)), float(m.group(5)), i))
            continue
    return rows


def _diff_rows(pub: list[Row], opt: list[Row]) -> tuple[list[dict], list[str]]:
    """Match pub and opt rows by position; return per-row diff and warnings."""
    warnings: list[str] = []
    if len(pub) != len(opt):
        warnings.append(
            f"Row count mismatch: published has {len(pub)} OPT rows, "
            f"optimized has {len(opt)} — matching the first "
            f"{min(len(pub), len(opt))} positions. Manual inspection "
            f"recommended."
        )
    diffs = []
    for p, o in zip(pub, opt):
        if p.kind != o.kind:
            warnings.append(
                f"Kind mismatch at line pub:{p.line_no}/opt:{o.line_no}: {p.kind} vs {o.kind}; stopping comparison."
            )
            break
        if len(p.atoms) != len(o.atoms):
            warnings.append(f"Atom-arity mismatch at pub:{p.line_no}/opt:{o.line_no}; stopping.")
            break
        for param in ("eq", "fc"):
            pv = getattr(p, param)
            ov = getattr(o, param)
            abs_dev = ov - pv
            rel_dev = abs_dev / pv if pv != 0 else None
            motif = _bond_motif(p.atoms) if p.kind == "bond" else _angle_motif(p.atoms)
            diffs.append(
                {
                    "kind": p.kind,
                    "param": param,
                    "atoms_pub": p.atoms,
                    "atoms_opt": o.atoms,
                    "value_pub": pv,
                    "value_opt": ov,
                    "abs_dev": abs_dev,
                    "rel_dev": rel_dev,
                    "motif": motif,
                }
            )
    return diffs, warnings


def _fmt_value(v: float, kind: str, param: str) -> str:
    if kind == "bond" and param == "eq":
        return f"{v:.4f} Å"
    if kind == "bond" and param == "fc":
        return f"{v:.4f}"
    if kind == "angle" and param == "eq":
        return f"{v:.2f}°"
    if kind == "angle" and param == "fc":
        return f"{v:.4f}"
    return f"{v:.4f}"


def _summary_by_category(diffs: list[dict]) -> str:
    """Rollup table: per (kind, param) — count, mean |abs|, max |abs|, mean |rel|."""
    by_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in diffs:
        by_cat[(d["kind"], d["param"])].append(d)
    rows = []
    rows.append("| Category | N | Mean abs dev | Max abs dev | Median rel dev | Max rel dev |")
    rows.append("|---|---:|---:|---:|---:|---:|")
    order = [("bond", "eq"), ("bond", "fc"), ("angle", "eq"), ("angle", "fc")]
    for key in order:
        items = by_cat.get(key, [])
        if not items:
            continue
        abs_devs = [abs(d["abs_dev"]) for d in items]
        rel_devs = [abs(d["rel_dev"]) for d in items if d["rel_dev"] is not None]
        kind, param = key
        label = {
            ("bond", "eq"): "bond eq (Å)",
            ("bond", "fc"): "bond fc (mdyn/Å)",
            ("angle", "eq"): "angle eq (°)",
            ("angle", "fc"): "angle fc (mdyn·Å/rad²)",
        }[key]
        mean_abs = statistics.mean(abs_devs) if abs_devs else 0
        max_abs = max(abs_devs) if abs_devs else 0
        median_rel = statistics.median(rel_devs) * 100 if rel_devs else 0
        max_rel = max(rel_devs) * 100 if rel_devs else 0
        rows.append(f"| {label} | {len(items)} | {mean_abs:.4f} | {max_abs:.4f} | {median_rel:.2f}% | {max_rel:.2f}% |")
    return "\n".join(rows)


def _summary_by_motif(diffs: list[dict]) -> str:
    """Rollup by chemical motif."""
    by_motif: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for d in diffs:
        by_motif[(d["motif"], d["param"])].append(d)
    rows = []
    rows.append("| Motif | Param | N | Mean abs dev | Max abs dev | Median rel dev |")
    rows.append("|---|---|---:|---:|---:|---:|")
    for (motif, param), items in sorted(by_motif.items()):
        abs_devs = [abs(d["abs_dev"]) for d in items]
        rel_devs = [abs(d["rel_dev"]) for d in items if d["rel_dev"] is not None]
        mean_abs = statistics.mean(abs_devs) if abs_devs else 0
        max_abs = max(abs_devs) if abs_devs else 0
        median_rel = statistics.median(rel_devs) * 100 if rel_devs else 0
        rows.append(f"| {motif} | {param} | {len(items)} | {mean_abs:.4f} | {max_abs:.4f} | {median_rel:.2f}% |")
    return "\n".join(rows)


def _top_deviations(diffs: list[dict], n: int = 15) -> str:
    """Show the top-N largest relative deviations with chemical context."""
    rated = [d for d in diffs if d["rel_dev"] is not None]
    rated.sort(key=lambda d: abs(d["rel_dev"]), reverse=True)
    rows = [
        "| Rank | Kind | Param | Atoms | Pub | Optimized | Abs Δ | Rel Δ | Motif |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for i, d in enumerate(rated[:n], start=1):
        atoms_str = "–".join(d["atoms_pub"])
        rows.append(
            f"| {i} | {d['kind']} | {d['param']} | {atoms_str} | "
            f"{_fmt_value(d['value_pub'], d['kind'], d['param'])} | "
            f"{_fmt_value(d['value_opt'], d['kind'], d['param'])} | "
            f"{d['abs_dev']:+.4f} | {d['rel_dev'] * 100:+.2f}% | {d['motif']} |"
        )
    return "\n".join(rows)


def main() -> int:
    """Entry point: parse CLI args and emit the row-by-row comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--published",
        type=Path,
        default=None,
        help="Path to published .fld. If --system is given and "
        "this is omitted, the system loader is used to compose "
        "the published FF and round-trip-save it to a temp file "
        "(so atom-type tokens match the optimizer's save format).",
    )
    ap.add_argument("--optimized", required=True, type=Path)
    ap.add_argument("--system", required=True)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    if args.published is None:
        # Compose the published FF via the system loader and round-trip
        # it to a temp .fld so its OPT-row ordering and atom-type tokens
        # match the optimizer-saved file's format.
        import os as _os
        import sys as _sys
        import tempfile as _tempfile

        _sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from q2mm.diagnostics.systems import load_system

        sd = load_system(args.system)
        fd, published_path_str = _tempfile.mkstemp(prefix=f"pub-{args.system}-", suffix=".fld")
        _os.close(fd)
        published_path = Path(published_path_str)
        sd.forcefield.to_mm3_fld(str(published_path))
    else:
        published_path = args.published

    pub_rows = _parse_opt_block(published_path)
    opt_rows = _parse_opt_block(args.optimized)
    print(f"Parsed {len(pub_rows)} published OPT rows from {published_path.name}", flush=True)
    print(f"Parsed {len(opt_rows)} optimized OPT rows from {args.optimized.name}", flush=True)

    diffs, warnings = _diff_rows(pub_rows, opt_rows)

    parts = [
        f"## Per-parameter comparison: {args.system}",
        "",
        f"Published FF: `{published_path.name}`  ",
        f"Optimized FF: `{args.optimized.name}`",
        "",
    ]
    if warnings:
        parts.append("### ⚠ Warnings")
        parts.append("")
        for w in warnings:
            parts.append(f"- {w}")
        parts.append("")
    parts.append(f"Matched **{len(diffs) // 2}** OPT rows ({len(diffs)} parameter cells: eq + fc per row).")
    parts.append("")
    parts.append("### Summary by category")
    parts.append("")
    parts.append(_summary_by_category(diffs))
    parts.append("")
    parts.append("### Summary by chemical motif")
    parts.append("")
    parts.append(_summary_by_motif(diffs))
    parts.append("")
    parts.append("### Top 15 largest relative deviations")
    parts.append("")
    parts.append(_top_deviations(diffs, n=15))
    parts.append("")

    args.out.write_text("\n".join(parts))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
