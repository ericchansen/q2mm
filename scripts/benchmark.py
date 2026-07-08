"""Batch convergence-benchmark runner for the published-FF systems.

Thin CLI wrapper over :func:`q2mm.benchmark_runner.run_benchmark_batch`.
For each requested system this script:

1. Loads the system (Seminario-estimated starting FF + reference data).
2. Computes per-category Seminario fit quality (bond_length, bond_angle,
   eig_diagonal): R², RMSD, MAE, n_refs.
3. Computes the initial ObjectiveFunction score and the JaxLoss surrogate
   score; reports their ratio.
4. If the ratio is within the configured tolerance (or the tolerance has
   been disabled with ``--ratio-tol none``) it runs the requested workflow
   (default Method E2) using scipy L-BFGS-B with JaxLoss analytical
   gradients and writes the optimized force field as a ``.fld`` file.
5. With ``--n-evals N``, repeats post-hoc ObjectiveFunction evaluations at
   both the initial and optimized parameter vectors to report sample-mean
   scores, t-distribution 95% confidence-interval half-widths, mean
   improvement percentage, and whether the mean change exceeds the summed
   confidence intervals.

Outputs (per system) live under
``<output-dir>/<system-data-dir>/convergence/`` (QFUERZA-start, default) or
``<output-dir>/<system-data-dir>/from-published/`` (``--starting-point published``):

- ``validation_results.json`` — summary numbers (strict JSON,
  no ``Infinity`` or ``NaN``).  Ratio state is encoded across three keys:
  ``ratio`` (numeric or ``null``), ``ratio_status`` (``"ok"``,
  ``"ok_bypassed"``, ``"out_of_band"``, ``"diverged"``, ``"nan"``), and
  ``ratio_passes`` (bool).
- ``paper_metrics.json`` — Seminario + optimized per-category stats.
- ``<system>_optimized.fld`` — optimized force field (only when
  optimization ran and succeeded).

Every output embeds a ``provenance`` block: git SHA + dirty flag for q2mm
and q2mm-data, full command line, all CLI knobs, JAX/OpenMM device
names, ISO-8601 timestamp.

This is the single committed producer for the canonical convergence
artifacts — satisfies AGENTS.md Rule 8 ("every claim grounded in
evidence").
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from q2mm.benchmark_runner import (
    DEFAULT_OUTPUT_DIR,
    run_benchmark_batch,
    write_strict_json,
)

logger = logging.getLogger("scripts.benchmark")


# ---------------------------------------------------------------------------
# Custom argparse types
# ---------------------------------------------------------------------------


def _parse_ratio_tol(value: str) -> float | None:
    """``"none"`` or a negative numeric value → ``None``; otherwise float."""
    if value.lower() in {"none", "off", "disabled", "-1"}:
        return None
    parsed = float(value)
    if parsed < 0:
        return None
    return parsed


def _parse_positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/benchmark.py",
        description=__doc__.splitlines()[0] if __doc__ else None,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sel = parser.add_argument_group("selection")
    sel.add_argument(
        "--system",
        action="append",
        help="System key to process (repeatable). Defaults to all systems in SYSTEMS.",
    )
    sel.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root output directory.",
    )
    sel.add_argument(
        "--combined-output",
        type=Path,
        default=None,
        help="If set, also write a combined JSON aggregating all systems to this path.",
    )

    wf = parser.add_argument_group("workflow")
    wf.add_argument(
        "--workflow",
        choices=("method-e2", "single-stage"),
        default="method-e2",
        help="Workflow to execute per system. Default is the Method E2 protocol "
        "(Limé & Norrby 2015 + Farrugia 2025 Approxn substitution).",
    )
    wf.add_argument(
        "--starting-point",
        choices=("qfuerza", "published"),
        default="qfuerza",
        help="Starting force-field parameters. 'qfuerza' (canonical) uses QFUERZA "
        "Hessian-derived bond/angle values atop the published FF skeleton; "
        "'published' retains the literature OPT values verbatim.",
    )
    wf.add_argument(
        "--qfuerza-replace-with",
        type=float,
        default=1.0,
        help="Replacement value (Hartree/Bohr²) for the negative TS-Hessian eigenvalue during QFUERZA setup.",
    )

    opt = parser.add_argument_group("optimizer")
    opt.add_argument(
        "--ratio-tol",
        type=_parse_ratio_tol,
        default=None,
        help="JaxLoss/ObjFun ratio tolerance (e.g. 0.15). Use 'none' or a "
        "negative value to disable the gate.  Default 'none' because all 5 "
        "publication TS systems have ratio < 0.5.",
    )
    opt.add_argument("--maxiter", type=int, default=500, help="Maximum L-BFGS-B iterations.")
    opt.add_argument(
        "--ftol",
        type=float,
        default=1e-8,
        help="L-BFGS-B function-value tolerance. Tighten (e.g. 1e-12) for "
        "from-poor-start runs where the default exits too soon.",
    )
    opt.add_argument(
        "--fc-fraction",
        type=float,
        default=None,
        help="Fractional bound width for force-constant parameters. Recommended 0.20 for QFUERZA-start runs.",
    )
    opt.add_argument(
        "--eq-fraction",
        type=float,
        default=None,
        help="Fractional bound width for equilibrium parameters. Recommended 0.05 for QFUERZA-start runs.",
    )

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument(
        "--n-evals",
        type=_parse_positive_int,
        default=1,
        help="Post-hoc ObjectiveFunction evaluations at x0/x_final for mean/CI reporting.",
    )
    sampling.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Compute baseline metrics only; do not optimize any system.",
    )

    misc = parser.add_argument_group("misc")
    misc.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from q2mm.systems import SYSTEMS

    systems = args.system or list(SYSTEMS.keys())
    unknown = [s for s in systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"Unknown system(s): {unknown}. Available: {sorted(SYSTEMS)}")

    output_dir = args.output_dir.resolve()
    logger.info("Output directory: %s", output_dir)
    logger.info("Systems: %s", systems)
    logger.info(
        "workflow=%s, starting_point=%s, ratio_tol=%s, maxiter=%d, ftol=%.2e, "
        "fc_fraction=%s, eq_fraction=%s, n_evals=%d",
        args.workflow,
        args.starting_point,
        args.ratio_tol,
        args.maxiter,
        args.ftol,
        args.fc_fraction,
        args.eq_fraction,
        args.n_evals,
    )

    outcome = run_benchmark_batch(
        systems,
        output_dir=output_dir,
        generator="scripts/benchmark.py",
        workflow=args.workflow,
        starting_point=args.starting_point,
        qfuerza_replace_with=args.qfuerza_replace_with,
        ratio_tol=args.ratio_tol,
        maxiter=args.maxiter,
        ftol=args.ftol,
        fc_fraction=args.fc_fraction,
        eq_fraction=args.eq_fraction,
        n_evals=args.n_evals,
        skip_optimization=args.skip_optimization,
    )

    if args.combined_output is not None:
        combined_payload = {
            "results": {k: v.summary for k, v in outcome.results.items()},
            "failed_systems": outcome.failed_systems,
            "no_progress_systems": outcome.no_progress_systems,
        }
        write_strict_json(args.combined_output, combined_payload)
        logger.info("Wrote combined output: %s", args.combined_output)

    if outcome.failed_systems:
        logger.error("Failed systems: %s", outcome.failed_systems)
        return 1
    if not outcome.ok:
        logger.error(
            "BATCH FAILURE: every optimized system exited at n_iterations<=2 with "
            "|improvement_pct|<1%%.  See per-system logs above."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
