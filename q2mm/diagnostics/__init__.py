"""Diagnostics and benchmarking tools for Q2MM.

Provides reusable table formatting, benchmark result serialization,
PES distortion analysis, and cross-backend comparison reporting.
"""

from q2mm.diagnostics.benchmark import BenchmarkResult, run_benchmark
from q2mm.diagnostics.pes_distortion import compute_distortions, load_normal_modes
from q2mm.diagnostics.report import detailed_report, full_report
from q2mm.diagnostics.tables import TablePrinter

__all__ = [
    "BenchmarkResult",
    "TablePrinter",
    "compute_distortions",
    "detailed_report",
    "full_report",
    "load_normal_modes",
    "run_benchmark",
]
