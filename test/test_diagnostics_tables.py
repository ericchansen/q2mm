"""Regression tests: every report table must render on a Windows cp1252 console.

``TablePrinter`` documents itself as building ASCII tables, but several
runtime string literals in :mod:`q2mm.diagnostics.tables` used Unicode
characters (``cm⁻¹``, ``→``, ``RMSD₀``) that ``cp1252`` — the default
console codepage on non-UTF-8 Windows terminals — cannot encode. Because
``print(..., file=file)`` uses *file*'s own encoding/errors policy, any
one bad character crashes ``TablePrinter.flush()`` with
``UnicodeEncodeError`` the moment a caller redirects to (or a Windows
console defaults to) a strict cp1252 stream — exactly what
``TestBenchmarkPipeline.test_report_generation`` hit.

Each test below builds one of the report tables with realistic synthetic
data and flushes it to a genuine ``io.TextIOWrapper(encoding="cp1252",
errors="strict")`` wrapping an in-memory buffer — the same failure mode
a real Windows console triggers, without needing OpenMM or any other
optional backend.
"""

from __future__ import annotations

import io

from q2mm.diagnostics.tables import (
    convergence_table,
    frequency_progression_table,
    leaderboard_table,
    parameter_table,
    pes_distortion_table,
    timing_table,
)


def _cp1252_stream() -> io.TextIOWrapper:
    """Build a strict cp1252 text stream backed by an in-memory buffer."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_frequency_progression_table_renders_on_cp1252() -> None:
    qm_freqs = [1040.6, 1191.7, 1191.8, 1495.3, 1512.3]
    stages = [
        ("Default FF", [900.1, 1050.2, 1050.3, 1400.0, 1420.5]),
        ("QFUERZA", [1000.5, 1120.6, 1120.7, 1450.1, 1490.2]),
        ("Optimized", [1038.9, 1189.4, 1189.5, 1493.8, 1510.9]),
    ]
    t = frequency_progression_table(qm_freqs, stages)
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_pes_distortion_table_renders_on_cp1252() -> None:
    distortion_results = [
        {
            "mode_idx": i,
            "freq_cm1": freq,
            "displacements": [
                {"d_ang": 0.05, "e_qm": 0.976, "e_mm": 1.1, "pct_err": 12.7},
                {"d_ang": 0.10, "e_qm": 3.903, "e_mm": 4.2, "pct_err": 7.6},
            ],
        }
        for i, freq in enumerate([1040.6, 1191.7, 1512.3], start=1)
    ]
    t = pes_distortion_table(distortion_results, elapsed_s=0.0959)
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_pes_distortion_table_empty_renders_on_cp1252() -> None:
    """The ``no data`` early-return path is also exercised (Hessian unavailable)."""
    t = pes_distortion_table([])
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_timing_table_renders_on_cp1252() -> None:
    timings = {
        "seminario_s": 0.0016,
        "optimization_s": 5.79,
        "n_eval": 71,
        "per_eval_ms": 0.0816,
        "pes_distortion_s": 0.0959,
    }
    t = timing_table(timings)
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_parameter_table_renders_on_cp1252() -> None:
    names = ["kb_C-F", "r0_C-F", "ka_F-C-H", "th0_F-C-H"]
    default_values = [300.0, 1.35, 60.0, 109.5]
    seminario_values = [270.6, 1.399, 36.0, 108.4]
    optimized_values = [270.7, 1.516, 36.2, 124.8]
    t = parameter_table(names, default_values, seminario_values, optimized_values)
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_convergence_table_renders_on_cp1252() -> None:
    """Exercises the exact lines that used to embed ``→`` and ``cm⁻¹``."""
    t = convergence_table(
        initial_score=0.331602,
        final_score=0.019420,
        n_eval=71,
        converged=True,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
        initial_rmsd=191.9,
        final_rmsd=579.9,
        elapsed_s=5.6,
    )
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_convergence_table_without_rmsd_renders_on_cp1252() -> None:
    """The RMSD row (and its arrow) is only emitted when both RMSDs are given."""
    t = convergence_table(
        initial_score=0.331602,
        final_score=0.019420,
        n_eval=71,
        converged=True,
        message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    )
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_leaderboard_table_renders_on_cp1252() -> None:
    """Exercises the exact header that used to embed ``RMSD₀`` (subscript zero)."""
    rows = [
        {
            "backend": "OpenMM",
            "optimizer": "L-BFGS-B",
            "rmsd": 579.9,
            "mae": 325.2,
            "time_s": 5.6,
            "n_eval": 71,
            "final_score": 0.019420,
            "converged": True,
            "message": "",
            "error": None,
            "initial_rmsd": 191.9,
        },
        {
            "backend": "Tinker",
            "optimizer": "Nelder-Mead",
            "rmsd": float("nan"),
            "mae": float("nan"),
            "time_s": 0.0,
            "n_eval": 0,
            "final_score": float("nan"),
            "converged": False,
            "message": "",
            "error": "backend unavailable",
            "initial_rmsd": float("nan"),
        },
    ]
    t = leaderboard_table(rows)
    with _cp1252_stream() as stream:
        t.flush(file=stream)


def test_all_tables_render_on_cp1252_via_to_string() -> None:
    """``to_string()`` (used by non-flush callers) must also survive strict re-encoding.

    ``to_string()`` itself renders through an in-memory ``io.StringIO``
    (always UTF-8-safe), so the regression only shows up once the
    resulting string is re-encoded for a real cp1252 destination — e.g.
    a caller that does ``dest.write(t.to_string())`` on a cp1252 file.
    """
    t = convergence_table(
        initial_score=0.331602,
        final_score=0.019420,
        n_eval=71,
        converged=True,
        initial_rmsd=191.9,
        final_rmsd=579.9,
    )
    rendered = t.to_string()
    # Must not raise UnicodeEncodeError.
    rendered.encode("cp1252")
