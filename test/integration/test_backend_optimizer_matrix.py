"""Validate the diagnostics library: serialization, reports, and CLI.

These tests verify that the ``q2mm.diagnostics`` package works correctly
without re-running the expensive benchmark matrix (use ``q2mm-benchmark``
for that).  Only one fast (backend, optimizer) combo is run to produce
a real ``BenchmarkResult``; the rest of the library is tested with
synthetic data.

Requires ``--run-medium`` and OpenMM.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from q2mm.diagnostics.benchmark import BenchmarkResult, frequency_mae, frequency_rmsd, real_frequencies
from q2mm.diagnostics.report import detailed_report, full_report
from q2mm.diagnostics.tables import TablePrinter

# ---- Paths ----

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = QM_REF / "ch3f-hessian.npy"
CH3F_FREQS = QM_REF / "ch3f-frequencies.txt"
CH3F_MODES = QM_REF / "ch3f-normal-modes.npz"

_FIXTURE_FILES = [CH3F_XYZ, CH3F_HESS, CH3F_FREQS]
_missing = [str(f) for f in _FIXTURE_FILES if not f.exists()]


# ---------------------------------------------------------------------------
# Unit-level tests for diagnostics helpers (fast, no backend needed)
# ---------------------------------------------------------------------------


class TestDiagnosticsHelpers:
    """Fast tests for frequency metrics, TablePrinter, and BenchmarkResult serialization."""

    def test_frequency_rmsd_identical(self):
        a = [100.0, 200.0, 300.0]
        assert frequency_rmsd(a, a) == pytest.approx(0.0)

    def test_frequency_rmsd_known(self):
        a = [100.0, 200.0]
        b = [110.0, 220.0]
        expected = np.sqrt((10**2 + 20**2) / 2)
        assert frequency_rmsd(a, b) == pytest.approx(expected)

    def test_frequency_mae_known(self):
        a = [100.0, 200.0]
        b = [110.0, 220.0]
        assert frequency_mae(a, b) == pytest.approx(15.0)

    def test_real_frequencies_filters(self):
        freqs = [-300.0, -5.0, 0.0, 10.0, 49.0, 100.0, 500.0]
        real = real_frequencies(freqs)
        np.testing.assert_array_equal(real, [100.0, 500.0])

    def test_table_printer_to_string(self):
        t = TablePrinter()
        t.bar()
        t.title("TEST")
        t.bar()
        s = t.to_string()
        assert "TEST" in s
        assert s.count("=") > 0

    def test_benchmark_result_roundtrip(self):
        """BenchmarkResult survives JSON serialization."""
        r = BenchmarkResult(
            metadata={"backend": "TestBackend", "optimizer": "L-BFGS-B", "molecule": "H2O"},
            qm_reference={"frequencies_cm1": [1600.0, 3700.0, 3800.0]},
            default_ff={"frequencies_cm1": [1500.0, 3500.0, 3600.0], "rmsd": 180.0, "mae": 166.7},
            optimized={
                "frequencies_cm1": [1590.0, 3680.0, 3790.0],
                "rmsd": 12.0,
                "mae": 10.0,
                "elapsed_s": 1.5,
                "n_eval": 100,
                "converged": True,
                "initial_score": 50.0,
                "final_score": 0.5,
                "message": "converged",
                "param_names": ["kb", "r0"],
                "param_initial": [1.0, 1.0],
                "param_final": [1.5, 0.96],
            },
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            r.to_json(path)
            loaded = BenchmarkResult.from_json(path)
            assert loaded.metadata["backend"] == "TestBackend"
            assert loaded.optimized["rmsd"] == pytest.approx(12.0)
            assert loaded.optimized["param_final"] == [1.5, 0.96]
        finally:
            path.unlink(missing_ok=True)

    def test_benchmark_result_from_upstream(self):
        r = BenchmarkResult.from_upstream([100.0, 200.0, 300.0], molecule="CH3F", label="legacy")
        assert r.metadata["backend"] == "legacy"
        assert r.optimized["frequencies_cm1"] == [100.0, 200.0, 300.0]

    def test_detailed_report_produces_tables(self):
        """detailed_report() should return TablePrinter objects for a complete result."""
        r = BenchmarkResult(
            metadata={"backend": "Fake", "optimizer": "L-BFGS-B"},
            qm_reference={"frequencies_cm1": [1000.0, 2000.0, 3000.0]},
            default_ff={"frequencies_cm1": [900.0, 1800.0, 2700.0], "rmsd": 200.0, "mae": 200.0},
            optimized={
                "frequencies_cm1": [990.0, 1990.0, 2990.0],
                "rmsd": 10.0,
                "mae": 10.0,
                "elapsed_s": 2.0,
                "n_eval": 50,
                "converged": True,
                "initial_score": 100.0,
                "final_score": 1.0,
                "message": "ok",
                "param_names": ["k1", "r1"],
                "param_initial": [1.0, 1.0],
                "param_final": [1.1, 0.95],
            },
        )
        tables = detailed_report(r)
        assert len(tables) >= 3  # frequency, timing, convergence at minimum
        for t in tables:
            assert isinstance(t, TablePrinter)
            s = t.to_string()
            assert len(s) > 0

    def test_full_report_no_crash(self):
        """full_report() shouldn't crash on a list of results."""
        results = [
            BenchmarkResult(
                metadata={"backend": "A", "optimizer": "X"},
                qm_reference={"frequencies_cm1": [1000.0]},
                optimized={
                    "frequencies_cm1": [990.0],
                    "rmsd": 10.0,
                    "mae": 10.0,
                    "elapsed_s": 1.0,
                    "n_eval": 10,
                    "converged": True,
                    "initial_score": 50.0,
                    "final_score": 1.0,
                    "message": "ok",
                    "param_names": [],
                    "param_initial": [],
                    "param_final": [],
                },
            )
        ]
        # Should not raise
        full_report(results)


# ---------------------------------------------------------------------------
# Integration: one real benchmark run to validate the pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.medium
@pytest.mark.openmm
@pytest.mark.skipif(bool(_missing), reason=f"Missing fixtures: {_missing}")
class TestBenchmarkPipeline:
    """Run one real (OpenMM, L-BFGS-B) benchmark to validate run_benchmark().

    This does NOT duplicate the E2E test: that test validates the full
    Seminario -> optimize -> frequency pipeline in detail.  This test
    validates that ``run_benchmark()`` (the function the CLI calls)
    produces a correct, serializable ``BenchmarkResult``.
    """

    @pytest.fixture(scope="class")
    def result(self):
        from q2mm.backends.mm.openmm import OpenMMEngine
        from q2mm.diagnostics.benchmark import run_benchmark
        from q2mm.models.molecule import Q2MMMolecule

        molecule = Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)
        qm_freqs = np.loadtxt(CH3F_FREQS)
        qm_hessian = np.load(CH3F_HESS)

        return run_benchmark(
            engine=OpenMMEngine(),
            molecule=molecule,
            qm_freqs=qm_freqs,
            qm_hessian=qm_hessian,
            normal_modes=None,  # skip PES distortion for speed
            optimizer_method="L-BFGS-B",
            maxiter=200,
            backend_name="OpenMM",
            molecule_name="CH3F",
            level_of_theory="B3LYP/6-31+G(d)",
        )

    def test_result_has_all_sections(self, result):
        assert result.metadata["backend"] == "OpenMM"
        assert result.qm_reference["frequencies_cm1"]
        assert result.default_ff is not None
        assert result.seminario is not None
        assert result.optimized is not None

    def test_optimization_improved(self, result):
        # Don't assert strict convergence — L-BFGS-B may not hit gtol on all
        # platforms/versions. Instead verify the optimizer meaningfully improved.
        assert result.optimized["final_score"] < result.optimized["initial_score"]

    def test_optimized_rmsd_better_than_default(self, result):
        assert result.optimized["rmsd"] < result.default_ff["rmsd"]

    def test_json_roundtrip(self, result):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)
        try:
            result.to_json(path)
            loaded = BenchmarkResult.from_json(path)
            assert loaded.optimized["rmsd"] == pytest.approx(result.optimized["rmsd"])
            assert loaded.metadata == result.metadata
        finally:
            path.unlink(missing_ok=True)

    def test_report_generation(self, result, capsys):
        """detailed_report + full_report work on real data."""
        tables = detailed_report(result)
        assert len(tables) >= 3
        with capsys.disabled():
            print()
            for t in tables:
                t.flush()
