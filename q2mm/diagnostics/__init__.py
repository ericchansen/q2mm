"""Diagnostics and benchmarking tools for Q2MM.

Provides reusable table formatting, benchmark result serialization,
PES distortion analysis, reference data analysis, and cross-backend
comparison reporting.
"""

from q2mm.diagnostics.benchmark import (
    BenchmarkResult,
    frequency_mae,
    frequency_rmsd,
    real_frequencies,
    run_benchmark,
)
from q2mm.diagnostics.pes_distortion import compute_distortions, load_normal_modes
from q2mm.diagnostics.reference_analysis import (
    EigenvalueAnalysis,
    FrequencyComparison,
    ModeCouplingAnalysis,
    SymmetryCheck,
    analyze_eigenvalues,
    analyze_mode_coupling,
    check_symmetry,
    compare_frequencies,
    format_hessian_report,
)
from q2mm.diagnostics.report import detailed_report, full_report
from q2mm.diagnostics.systems import SYSTEMS, BenchmarkSystem, SystemData
from q2mm.diagnostics.tables import TablePrinter

__all__ = [
    "BenchmarkResult",
    "BenchmarkSystem",
    "EigenvalueAnalysis",
    "FrequencyComparison",
    "ModeCouplingAnalysis",
    "SYSTEMS",
    "SymmetryCheck",
    "SystemData",
    "TablePrinter",
    "analyze_eigenvalues",
    "analyze_mode_coupling",
    "check_symmetry",
    "compare_frequencies",
    "compute_distortions",
    "detailed_report",
    "format_hessian_report",
    "frequency_mae",
    "frequency_rmsd",
    "full_report",
    "load_normal_modes",
    "real_frequencies",
    "run_benchmark",
]
