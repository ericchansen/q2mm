"""Validate the unified benchmark runner pipeline end-to-end.

Fast tests cover the frequency-space benchmark-analysis helpers with
synthetic data (no backend needed).  One real ``run_profile`` call on the
smallest system exercises the full resolve -> execute -> acceptance ->
persist -> promote path with OpenMM.

Requires ``--run-integration`` and OpenMM for the pipeline test.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from q2mm.benchmarks.runner import frequency_mae, frequency_rmsd, real_frequencies

if TYPE_CHECKING:
    from q2mm.benchmarks.runner import CandidateResult

# ---- Paths ----

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from q2mm.resources import sn2_reference_dir

QM_REF = sn2_reference_dir()

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = QM_REF / "ch3f-hessian.npy"
CH3F_FREQS = QM_REF / "ch3f-frequencies.txt"

_FIXTURE_FILES = [CH3F_XYZ, CH3F_HESS, CH3F_FREQS]
_missing = [str(f) for f in _FIXTURE_FILES if not f.exists()]


# ---------------------------------------------------------------------------
# Fast tests for the frequency-space benchmark-analysis helpers
# ---------------------------------------------------------------------------


class TestFrequencyHelpers:
    """Frequency RMSD/MAE/real-mode helpers (delegate RMSD/MAE to objectives.metrics)."""

    def test_frequency_rmsd_identical(self) -> None:
        a = [100.0, 200.0, 300.0]
        assert frequency_rmsd(a, a) == pytest.approx(0.0)

    def test_frequency_rmsd_known(self) -> None:
        expected = np.sqrt((10**2 + 20**2) / 2)
        assert frequency_rmsd([100.0, 200.0], [110.0, 220.0]) == pytest.approx(expected)

    def test_frequency_mae_known(self) -> None:
        assert frequency_mae([100.0, 200.0], [110.0, 220.0]) == pytest.approx(15.0)

    def test_frequency_helpers_truncate_to_shorter(self) -> None:
        # The extra element in the longer array is ignored.
        assert frequency_rmsd([100.0, 200.0, 999.0], [100.0, 200.0]) == pytest.approx(0.0)

    def test_real_frequencies_filters(self) -> None:
        freqs = [-300.0, -5.0, 0.0, 10.0, 49.0, 100.0, 500.0]
        np.testing.assert_array_equal(real_frequencies(freqs), [100.0, 500.0])


# ---------------------------------------------------------------------------
# Integration: one real run_profile call to validate the pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.openmm
@pytest.mark.skipif(bool(_missing), reason=f"Missing fixtures: {_missing}")
class TestRunnerPipeline:
    """Run one real (OpenMM, L-BFGS-B) profile to validate run_profile()."""

    @pytest.fixture(scope="class")
    def candidate(self) -> CandidateResult:
        from q2mm.backends.registry import load_backend
        from q2mm.benchmarks.profiles import RunProfile
        from q2mm.benchmarks.runner import run_profile

        backend = load_backend("openmm")
        profile = RunProfile(
            system="ch3f",
            backend="openmm",
            functional_form="mm3",
            optimizer="scipy-lbfgsb",
            workflow="single-stage",
            maxiter=200,
            n_evals=0,
        )
        return run_profile(profile, backend=backend, analyze=True, include_device=False)

    def test_candidate_ran(self, candidate: CandidateResult) -> None:
        from q2mm.benchmarks.acceptance import CandidateStatus

        assert candidate.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED), candidate.reason
        assert candidate.summary["backend_name"]
        assert "seminario" in candidate.summary
        assert "optimized" in candidate.summary

    def test_optimization_made_progress(self, candidate: CandidateResult) -> None:
        assert candidate.summary["improvement_pct"] > 50.0, candidate.summary["improvement_pct"]

    def test_resolved_provenance_complete(self, candidate: CandidateResult) -> None:
        assert candidate.resolved is not None
        prov = candidate.resolved.to_dict()
        for key in ("backend_name", "capabilities", "functional_form", "layout_fingerprint", "optimizer_method"):
            assert prov[key], f"missing provenance field {key!r}"

    def test_persist_and_load_roundtrip(self, candidate: CandidateResult, tmp_path: Path) -> None:
        from q2mm.benchmarks.runner import load_candidates, persist_candidate

        persist_candidate(tmp_path, candidate, provenance={"generator": "test"})
        loaded = load_candidates(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].candidate_id == candidate.candidate_id
        assert loaded[0].status is candidate.status
