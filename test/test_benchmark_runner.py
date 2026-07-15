"""Tests for :mod:`q2mm.benchmark_runner`.

Validation strategy:

1. **Module surface** — ``q2mm.benchmark`` is the public facade.
2. **Workflow resolution** — string identifiers map to the right
   workflow class; instances pass through; bad input raises.
3. **System validation** — unknown system raises a clear error.
4. **Single-system end-to-end** — ``run_benchmark`` returns a
   :class:`BenchmarkRunResult` with the expected schema.  Uses CH3F
   (smallest system).
5. **Batch + persistence** — :func:`run_benchmark_batch` writes the
   canonical artifacts and computes the no-progress watchdog.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from q2mm.benchmark_runner import (
    BatchOutcome,
    BenchmarkRunResult,
    classify_ratio,
    resolve_workflow,
    run_benchmark,
    run_benchmark_batch,
)
from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField


class TestPublicSurface:
    """Public API discoverability."""

    def test_q2mm_top_level_facade(self) -> None:
        """``from q2mm import benchmark`` works."""
        from q2mm import benchmark

        assert benchmark is run_benchmark

    def test_q2mm_top_level_exports(self) -> None:
        """``q2mm.__all__`` documents the new entry points."""
        import q2mm

        for name in ("benchmark", "run_benchmark_batch", "BenchmarkRunResult", "BatchOutcome"):
            assert name in q2mm.__all__, f"missing from q2mm.__all__: {name!r}"


class TestResolveWorkflow:
    """Workflow identifier resolution."""

    def test_default_method_e2(self) -> None:
        assert isinstance(resolve_workflow("method-e2"), MethodE2Workflow)

    def test_aliases_method_e2(self) -> None:
        for alias in ("method_e2", "e2", "Method-E2"):
            assert isinstance(resolve_workflow(alias), MethodE2Workflow)

    def test_single_stage(self) -> None:
        assert isinstance(resolve_workflow("single-stage"), SingleStageWorkflow)

    def test_single_stage_aliases(self) -> None:
        for alias in ("single", "single_stage", "Single-Stage"):
            assert isinstance(resolve_workflow(alias), SingleStageWorkflow)

    def test_instance_passthrough(self) -> None:
        wf = MethodE2Workflow(negative_fc_threshold=0.5)
        assert resolve_workflow(wf) is wf

    def test_unknown_identifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown workflow"):
            resolve_workflow("dual-stage")

    def test_bad_type_raises(self) -> None:
        with pytest.raises(TypeError, match="workflow must be"):
            resolve_workflow(42)  # type: ignore[arg-type]


class TestClassifyRatio:
    """Ratio gate classification."""

    def test_ratio_in_band_passes(self) -> None:
        info = classify_ratio(1.05, tol=0.15)
        assert info["ratio"] == pytest.approx(1.05)
        assert info["ratio_status"] == "ok"
        assert info["ratio_passes"] is True

    def test_ratio_out_of_band_fails(self) -> None:
        info = classify_ratio(0.3, tol=0.15)
        assert info["ratio_status"] == "out_of_band"
        assert info["ratio_passes"] is False

    def test_ratio_nan_diverged(self) -> None:
        info = classify_ratio(float("nan"), tol=0.15)
        assert info["ratio"] is None
        assert info["ratio_status"] == "nan"
        assert info["ratio_passes"] is False

    def test_ratio_inf_diverged(self) -> None:
        info = classify_ratio(float("inf"), tol=0.15)
        assert info["ratio_status"] == "diverged"

    def test_ratio_tol_none_bypasses(self) -> None:
        info = classify_ratio(0.001, tol=None)
        assert info["ratio_status"] == "ok_bypassed"
        assert info["ratio_passes"] is True


class TestRunBenchmarkInputValidation:
    """Argument validation paths that don't require an backend."""

    def test_unknown_system_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown system"):
            run_benchmark("nonexistent-system")


@pytest.mark.jax
class TestRunBenchmarkEndToEnd:
    """Single-system end-to-end with JAX backend."""

    def test_run_benchmark_ch3f_skip_optimization(self) -> None:
        """Skip-only path: Seminario metrics, no workflow execution."""
        result = run_benchmark("ch3f", skip_optimization=True)
        assert isinstance(result, BenchmarkRunResult)
        assert result.system_key == "ch3f"
        assert result.skipped is True
        assert result.skip_reason == "user_requested"
        assert "seminario" in result.summary
        assert "seminario" in result.paper
        # Final FF identical to initial when no optimization runs.
        assert result.final_ff is result.initial_ff

    def test_run_benchmark_ch3f_method_e2(self) -> None:
        """Default Method E2 workflow runs end-to-end on CH3F."""
        result = run_benchmark(
            "ch3f",
            ratio_tol=None,
            maxiter=2,
            n_evals=0,
        )
        assert result.skipped is False
        assert result.workflow_name == "method-e2"
        assert "optimized_categories" in result.summary
        assert "stages" in result.summary
        assert len(result.summary["stages"]) >= 1
        # Summary is strict-JSON-safe (no NaN/Inf).
        import json

        from q2mm.benchmark_runner import sanitize_for_json

        json.dumps(sanitize_for_json(result.summary), allow_nan=False)
        json.dumps(sanitize_for_json(result.paper), allow_nan=False)

    def test_run_benchmark_single_stage_workflow_string(self) -> None:
        """``workflow="single-stage"`` runs the legacy single-stage path."""
        result = run_benchmark(
            "ch3f",
            workflow="single-stage",
            ratio_tol=None,
            maxiter=2,
            n_evals=0,
        )
        assert result.workflow_name == "single-stage"


@pytest.mark.jax
class TestRunBenchmarkBatch:
    """Batch wrapper + artifact persistence."""

    def test_batch_writes_canonical_artifacts(self, tmp_path: Path) -> None:
        outcome = run_benchmark_batch(
            ["ch3f"],
            output_dir=tmp_path,
            ratio_tol=None,
            maxiter=2,
            n_evals=0,
        )
        assert isinstance(outcome, BatchOutcome)
        assert "ch3f" in outcome.results
        assert not outcome.failed_systems

        sys_out = tmp_path / "ch3f" / "convergence"
        assert (sys_out / "validation_results.json").is_file()
        assert (sys_out / "paper_metrics.json").is_file()
        # This runner explicitly selects harmonic for its JAX CH3F path,
        # so the artifact uses the compatible .frcmod serializer.
        assert outcome.results["ch3f"].final_ff.functional_form.value == "harmonic"
        assert (sys_out / "ch3f_optimized.frcmod").is_file()
        assert not (sys_out / "ch3f_optimized.fld").exists()

    def test_batch_skip_optimization_no_fld_written(self, tmp_path: Path) -> None:
        outcome = run_benchmark_batch(
            ["ch3f"],
            output_dir=tmp_path,
            skip_optimization=True,
        )
        sys_out = tmp_path / "ch3f" / "convergence"
        assert (sys_out / "validation_results.json").is_file()
        assert not (sys_out / "ch3f_optimized.fld").exists()
        assert not (sys_out / "ch3f_optimized.frcmod").exists()

    def test_batch_unknown_system_collects_failure(self, tmp_path: Path) -> None:
        outcome = run_benchmark_batch(
            ["nonexistent"],
            output_dir=tmp_path,
        )
        assert "nonexistent" in outcome.failed_systems
        assert outcome.ok is False


class TestSaveOptimizedFf:
    """``_save_optimized_ff`` dispatches to the serializer matching the FF's actual form."""

    @staticmethod
    def _result(final_ff: ForceField) -> BenchmarkRunResult:
        return BenchmarkRunResult(
            system_key="unit-test-system",
            workflow_name="single-stage",
            initial_ff=final_ff,
            final_ff=final_ff,
            skipped=False,
            skip_reason=None,
            summary={},
            paper={},
        )

    def test_mm3_form_writes_fld(self, tmp_path: Path) -> None:
        from q2mm.benchmark_runner import _save_optimized_ff
        from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm

        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 300.0)],
            functional_form=FunctionalForm.MM3,
        )
        _save_optimized_ff(tmp_path, self._result(ff))
        assert (tmp_path / "unit-test-system_optimized.fld").is_file()
        assert not (tmp_path / "unit-test-system_optimized.frcmod").exists()

    def test_harmonic_form_writes_frcmod(self, tmp_path: Path) -> None:
        from q2mm.benchmark_runner import _save_optimized_ff
        from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm

        ff = ForceField(
            bonds=[BondParam(("C", "C"), 1.54, 300.0)],
            functional_form=FunctionalForm.HARMONIC,
        )
        _save_optimized_ff(tmp_path, self._result(ff))
        assert (tmp_path / "unit-test-system_optimized.frcmod").is_file()
        assert not (tmp_path / "unit-test-system_optimized.fld").exists()
