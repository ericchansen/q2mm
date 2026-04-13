"""Diagnostics and benchmarking tools for Q2MM.

Provides reusable table formatting, benchmark result serialization,
PES distortion analysis, and cross-backend comparison reporting.
"""

from q2mm.diagnostics.benchmark import (
    BenchmarkResult,
    frequency_mae,
    frequency_rmsd,
    real_frequencies,
    run_combo,
)
from q2mm.diagnostics.history import RunSummary, build_run_summary, load_history
from q2mm.diagnostics.pes_distortion import compute_distortions, load_normal_modes
from q2mm.diagnostics.report import detailed_report, full_report
from q2mm.diagnostics.systems import SYSTEMS, BenchmarkSystem, SystemData
from q2mm.diagnostics.tables import TablePrinter

__all__ = [
    "BenchmarkResult",
    "BenchmarkSystem",
    "RunSummary",
    "SYSTEMS",
    "SystemData",
    "TablePrinter",
    "build_run_summary",
    "compute_distortions",
    "detailed_report",
    "frequency_mae",
    "frequency_rmsd",
    "full_report",
    "load_history",
    "load_normal_modes",
    "real_frequencies",
    "run_combo",
]
