"""Microbenchmark: Pint vs bare-multiply unit conversions.

Compares three approaches to converting a bond force constant from
kcal/(mol·Å²) to kJ/(mol·nm²):

1. **Bare multiply** — plain ``float * float``, the current approach in
   ``q2mm/models/units.py``.
2. **Pint full parse** — construct a ``pint.Quantity`` from string unit
   names on every call (worst-case; simulates naive Pint usage).
3. **Pint prebuilt units** — reuse pre-parsed ``pint.Unit`` objects
   (realistic Pint usage with caching).
4. **Pint factor-only** — precompute the conversion factor once with Pint,
   then use a plain multiply (equivalent to the current NewType approach).

Run with::

    python scripts/bench_pint.py

Pint is an **optional** dependency — install it with::

    pip install 'q2mm[qm]'

Results are documented in GitHub issue #161 ("Evaluate Pint for unit
handling: performance impact assessment") and inform the two-tier
design in ``q2mm/models/units.py``: NewType for hot-path scalars,
Pint at cold-path I/O boundaries (see ``q2mm/io/jaguar.py``).

Expected output (reference machine: Intel Core i9-13980HX, CPython 3.12)::

    ┌──────────────────────────────┬──────────────┬────────────────┐
    │ Approach                     │   µs / call  │  vs bare ×     │
    ├──────────────────────────────┼──────────────┼────────────────┤
    │ Bare multiply                │         0.05 │            1×  │
    │ Pint full parse              │       111.24 │        2,388×  │
    │ Pint prebuilt units          │        10.12 │          217×  │
    │ Pint factor-only             │         0.04 │            1×  │
    └──────────────────────────────┴──────────────┴────────────────┘

    VERDICT: Pint per-call overhead (217–2,388×) exceeds the 5× threshold
    for hot-path scalar conversions.  Hot loops retain NewType (zero cost).
    Pint IS used at cold-path I/O boundaries (parsers) where the overhead
    is immeasurable (once per file load).  See architecture docs.
"""

from __future__ import annotations

import sys
import timeit
from typing import Any

# ---------------------------------------------------------------------------
# Conversion factor used by the bare-multiply baseline
# (kcal/mol/Å² → kJ/mol/nm²: multiply by 4.184 * 100 = 418.4)
# ---------------------------------------------------------------------------
_FACTOR = 4.184 * 100.0  # kcal→kJ (×4.184) and Å²→nm² (1 Å = 0.1 nm, so 1 Å² = 0.01 nm², multiply by 100)

_N = 100_000  # number of repetitions per timing measurement
_REPS = 5  # number of independent timing rounds (min is reported)

_SAMPLE_K = 523.6  # representative force constant in kcal/(mol·Å²)


def _time_us(stmt: str, setup: str = "pass", n: int = _N, reps: int = _REPS) -> float:
    """Return the minimum wall-clock time in µs per call."""
    times = timeit.repeat(stmt, setup=setup, number=n, repeat=reps)
    return min(times) / n * 1e6


def _run_bare() -> float:
    """Baseline: plain float multiplication."""
    return _time_us(
        stmt=f"k * {_FACTOR!r}",
        setup=f"k = {_SAMPLE_K!r}",
    )


def _run_pint_full_parse() -> float:
    """Pint with full string parsing on every call (naive usage)."""
    setup = f"import pint; ureg = pint.UnitRegistry(); k = {_SAMPLE_K!r}"
    stmt = "ureg.Quantity(k, 'kcal/mol/angstrom**2').to('kJ/mol/nm**2').magnitude"
    return _time_us(stmt=stmt, setup=setup)


def _run_pint_prebuilt() -> float:
    """Pint with pre-parsed unit objects (realistic cached usage)."""
    setup_lines = [
        "import pint",
        "ureg = pint.UnitRegistry()",
        "src_unit = ureg.Unit('kcal/mol/angstrom**2')",
        "tgt_unit = ureg.Unit('kJ/mol/nm**2')",
        f"k = {_SAMPLE_K!r}",
    ]
    stmt = "ureg.Quantity(k, src_unit).to(tgt_unit).magnitude"
    return _time_us(stmt=stmt, setup="\n".join(setup_lines))


def _run_pint_factor_only(ureg: Any) -> float:
    """Pint used once to compute factor, then bare multiply (equivalent to NewType)."""
    factor = ureg.Quantity(1.0, "kcal/mol/angstrom**2").to("kJ/mol/nm**2").magnitude
    return _time_us(
        stmt=f"k * {factor!r}",
        setup=f"k = {_SAMPLE_K!r}",
    )


def main() -> None:
    """Run all benchmarks and print a summary table."""
    # ------------------------------------------------------------------
    # Check Pint is available
    # ------------------------------------------------------------------
    try:
        import pint  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: pint is not installed.  Run:  pip install 'q2mm[qm]'",
            file=sys.stderr,
        )
        sys.exit(1)

    ureg = pint.UnitRegistry()

    print("\nBenchmark: unit conversion (kcal/mol/Å² → kJ/mol/nm²)")
    print(f"  Sample value  : {_SAMPLE_K} kcal/(mol·Å²)")
    print(f"  Repetitions   : {_N:,} per timing round, {_REPS} rounds (min reported)\n")

    results: list[tuple[str, float]] = []

    print("  Running bare multiply …", flush=True)
    t_bare = _run_bare()
    results.append(("Bare multiply", t_bare))

    print("  Running Pint full parse …", flush=True)
    t_full = _run_pint_full_parse()
    results.append(("Pint full parse", t_full))

    print("  Running Pint prebuilt units …", flush=True)
    t_pre = _run_pint_prebuilt()
    results.append(("Pint prebuilt units", t_pre))

    print("  Running Pint factor-only …", flush=True)
    t_factor = _run_pint_factor_only(ureg)
    results.append(("Pint factor-only", t_factor))

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    col_name = 30
    col_us = 14
    col_ratio = 14
    sep = f"  {'─' * col_name}  {'─' * col_us}  {'─' * col_ratio}"
    hdr = f"  {'Approach':<{col_name}}  {'µs / call':>{col_us}}  {'vs bare':>{col_ratio}}"

    print()
    print(sep)
    print(hdr)
    print(sep)
    for name, t in results:
        ratio = t / t_bare
        ratio_str = f"{ratio:,.0f}×" if ratio >= 10 else f"{ratio:.1f}×"
        print(f"  {name:<{col_name}}  {t:{col_us}.2f}  {ratio_str:>{col_ratio}}")
    print(sep)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    threshold = 5.0
    # The two meaningful Pint approaches are full-parse and prebuilt;
    # factor-only is equivalent to the current approach, so exclude it.
    pint_overhead = max(
        results[1][1] / t_bare,  # full parse
        results[2][1] / t_bare,  # prebuilt
    )
    if pint_overhead <= threshold:
        verdict = (
            f"\nVERDICT: Overhead {pint_overhead:,.0f}× ≤ {threshold:.0f}× threshold.\n"
            "         Consider prototyping Pint-backed units.py (see issue #161)."
        )
    else:
        verdict = (
            f"\nVERDICT: Overhead {pint_overhead:,.0f}× >> {threshold:.0f}× threshold.\n"
            "         Pint excluded from hot-path scalar conversions (NewType retained).\n"
            "         Pint IS used at cold-path I/O boundaries — see architecture docs."
        )
    print(verdict)


if __name__ == "__main__":
    main()
