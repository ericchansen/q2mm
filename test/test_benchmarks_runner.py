"""Tests for :mod:`q2mm.benchmarks.runner`.

Synthetic (backend-free) tests cover the immutable candidate/outcome model,
JSON-safe serialization, incremental persistence, atomic promotion (and its
failure-injection guarantee), the preserved canonical result for accepted
*and* rejected runs, backend error-vs-skip classification, and the shared
single/batch/matrix path.  JAX-marked tests exercise the real pipeline,
including a deterministic rejection that still preserves its full result.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from q2mm.benchmarks.acceptance import AcceptancePolicy, CandidateStatus, NoProgressPolicy
from q2mm.benchmarks.profiles import RunProfile, resolve
from q2mm.benchmarks.runner import (
    CandidateResult,
    RunOutcome,
    classify_ratio,
    load_candidates,
    persist_candidate,
    promote_candidate,
    run_profile,
    run_profiles,
    sanitize_for_json,
)

if TYPE_CHECKING:
    from q2mm.benchmarks.profiles import ResolvedProfile
    from q2mm.models.forcefield import ForceField


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _harmonic_ff() -> ForceField:
    from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm

    return ForceField(bonds=[BondParam(("C", "C"), 1.54, 300.0)], functional_form=FunctionalForm.HARMONIC)


def _resolved(profile: RunProfile) -> ResolvedProfile:
    from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

    info = BackendInfo(
        name="fake",
        role=BackendRole.MM,
        capabilities=frozenset({Capability.ENERGY}),
        functional_forms=frozenset({"harmonic"}),
        provenance=BackendProvenance(backend="fake", role=BackendRole.MM),
    )
    return resolve(
        profile,
        descriptor=None,
        backend_info=info,
        functional_form="harmonic",
        evaluator="python",
        gradient_mode="none",
        expected_result_gradient_mode="finite_difference",
        fd_step=None,
        effective_regularization=0.0,
        optimizer_settings={"kind": "scipy"},
        workflow_settings={"name": "single-stage"},
        layout_fingerprint="sha256:abc",
        n_active_params=1,
        n_full_params=1,
        n_molecules=1,
        data_provenance={"metadata": {}},
        resolved_data_roots={},
        include_device=False,
    )


def _opt_result(*, initial: float = 100.0, final: float = 50.0, n_iter: int = 10) -> Any:
    """Minimal canonical OptimizationResult for synthetic accepted/rejected candidates."""
    from q2mm.models.results import OptimizationResult

    return OptimizationResult(
        success=True,
        message="ok",
        initial_score=initial,
        final_score=final,
        n_iterations=n_iter,
        n_evaluations=n_iter + 1,
        n_params=1,
        layout_fingerprint="sha256:abc",
        initial_params=np.array([1.0]),
        final_params=np.array([0.9]),
        history=(initial, final),
        method="L-BFGS-B",
        gradient_mode="finite_difference",
    )


def _candidate(
    status: CandidateStatus,
    *,
    vary: int = 0,
    ff: ForceField | None = None,
    summary: dict[str, Any] | None = None,
) -> CandidateResult:
    """Build a synthetic candidate that satisfies the CandidateResult invariants.

    The candidate ID is always the resolved profile's canonical ID (per the
    invariant); *vary* changes the profile (via ``seed``) to obtain distinct
    IDs/filenames.  Accepted/rejected candidates carry a canonical
    OptimizationResult + final force field; skipped/error candidates carry
    neither.
    """
    profile = RunProfile(system="ch3f", functional_form="harmonic", seed=vary)
    resolved = _resolved(profile)
    has_run = status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED)
    result = _opt_result() if has_run else None
    final_ff = (ff or _harmonic_ff()) if has_run else None
    return CandidateResult(
        candidate_id=resolved.candidate_id(),
        status=status,
        reason=f"synthetic {status.value}",
        profile=profile,
        resolved=resolved,
        summary=summary or {"improvement_pct": 12.5},
        optimization_result=result,
        final_force_field=final_ff,
    )


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_candidate_summary_is_deeply_frozen(self) -> None:
        from types import MappingProxyType

        cand = _candidate(CandidateStatus.ACCEPTED, summary={"nested": {"a": 1}})
        assert isinstance(cand.summary, MappingProxyType)
        assert isinstance(cand.summary["nested"], MappingProxyType)
        with pytest.raises(TypeError):
            cand.summary["nested"]["a"] = 2  # type: ignore[index]

    def test_mutating_source_summary_does_not_leak(self) -> None:
        src = {"k": [1, 2]}
        cand = _candidate(CandidateStatus.ACCEPTED, summary=src)
        src["k"].append(3)
        src["new"] = 9
        assert "new" not in cand.summary
        assert cand.summary["k"] == (1, 2)

    def test_run_outcome_is_frozen(self) -> None:
        outcome = RunOutcome(candidates=(_candidate(CandidateStatus.SKIPPED),))
        assert isinstance(outcome.candidates, tuple)
        with pytest.raises(Exception):
            outcome.candidates = ()  # type: ignore[misc]

    def test_loaded_candidate_record_is_frozen(self, tmp_path: Path) -> None:
        from types import MappingProxyType

        persist_candidate(tmp_path, _candidate(CandidateStatus.ACCEPTED), provenance={})
        loaded = load_candidates(tmp_path)[0]
        assert isinstance(loaded.record, MappingProxyType)


# ---------------------------------------------------------------------------
# CandidateResult invariants
# ---------------------------------------------------------------------------


class TestCandidateResultInvariants:
    def test_candidate_and_outcome_use_identity_equality(self) -> None:
        first = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        second = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        first_outcome = RunOutcome(candidates=(first,), promoted={})
        second_outcome = RunOutcome(candidates=(second,), promoted={})

        assert first != second
        assert first_outcome != second_outcome
        assert len({first, second, first_outcome, second_outcome}) == 4

    def _base(self, status: CandidateStatus, **overrides: Any) -> dict[str, Any]:
        profile = RunProfile(system="ch3f", functional_form="harmonic")
        resolved = _resolved(profile)
        has_run = status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED)
        kwargs: dict[str, Any] = {
            "candidate_id": resolved.candidate_id(),
            "status": status,
            "reason": f"synthetic {status.value}",
            "profile": profile,
            "resolved": resolved,
            "optimization_result": _opt_result() if has_run else None,
            "final_force_field": _harmonic_ff() if has_run else None,
        }
        kwargs.update(overrides)
        return kwargs

    def test_empty_reason_rejected(self) -> None:
        with pytest.raises(ValueError, match="reason must be non-empty"):
            CandidateResult(**self._base(CandidateStatus.SKIPPED, reason=""))

    def test_accepted_without_result_rejected(self) -> None:
        with pytest.raises(ValueError, match="must carry both"):
            CandidateResult(**self._base(CandidateStatus.ACCEPTED, optimization_result=None))

    def test_accepted_without_ff_rejected(self) -> None:
        with pytest.raises(ValueError, match="must carry both"):
            CandidateResult(**self._base(CandidateStatus.ACCEPTED, final_force_field=None))

    def test_rejected_without_result_rejected(self) -> None:
        with pytest.raises(ValueError, match="must carry both"):
            CandidateResult(**self._base(CandidateStatus.REJECTED, optimization_result=None))

    def test_skipped_with_result_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not carry"):
            CandidateResult(**self._base(CandidateStatus.SKIPPED, optimization_result=_opt_result()))

    def test_error_with_ff_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not carry"):
            CandidateResult(**self._base(CandidateStatus.ERROR, final_force_field=_harmonic_ff()))

    def test_resolved_id_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="must equal resolved"):
            CandidateResult(**self._base(CandidateStatus.SKIPPED, candidate_id="mismatched"))

    def test_pre_resolution_id_must_equal_requested_profile(self) -> None:
        profile = RunProfile(system="ch3f", functional_form="harmonic")
        # resolved=None -> id must equal the requested profile candidate_id().
        with pytest.raises(ValueError, match="requested profile"):
            CandidateResult(
                candidate_id="wrong",
                status=CandidateStatus.ERROR,
                reason="pre-resolution failure",
                profile=profile,
                resolved=None,
            )
        # The correct requested-profile ID is accepted.
        ok = CandidateResult(
            candidate_id=profile.candidate_id(),
            status=CandidateStatus.ERROR,
            reason="pre-resolution failure",
            profile=profile,
            resolved=None,
        )
        assert ok.status is CandidateStatus.ERROR


# ---------------------------------------------------------------------------
# JSON-safe serialization + ratio gate
# ---------------------------------------------------------------------------


class TestSanitizeForJson:
    def test_non_finite_floats_become_sentinels(self) -> None:
        out = sanitize_for_json({"a": float("nan"), "b": float("inf"), "c": float("-inf"), "d": 1.5})
        assert out == {"a": "NaN", "b": "Infinity", "c": "-Infinity", "d": 1.5}

    def test_numpy_scalars_and_arrays(self) -> None:
        out = sanitize_for_json({"i": np.int64(3), "f": np.float64(2.5), "arr": np.array([1.0, 2.0])})
        assert out == {"i": 3, "f": 2.5, "arr": [1.0, 2.0]}


class TestClassifyRatio:
    def test_states(self) -> None:
        assert classify_ratio(1.05, 0.15)["executor_ratio_status"] == "ok"
        assert classify_ratio(0.3, 0.15)["executor_ratio_status"] == "out_of_band"
        assert classify_ratio(float("nan"), 0.15)["executor_ratio_status"] == "nan"
        assert classify_ratio(float("inf"), 0.15)["executor_ratio_status"] == "diverged"
        assert classify_ratio(0.001, None)["executor_ratio_status"] == "ok_bypassed"


# ---------------------------------------------------------------------------
# Persistence + atomic promotion (synthetic, no backend)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_every_status_is_loadable(self, tmp_path: Path) -> None:
        for i, status in enumerate(CandidateStatus):
            persist_candidate(tmp_path, _candidate(status, vary=i), provenance={})
        assert {c.status for c in load_candidates(tmp_path)} == set(CandidateStatus)

    def test_error_candidate_is_loadable(self, tmp_path: Path) -> None:
        persist_candidate(tmp_path, _candidate(CandidateStatus.ERROR, summary={"error": "boom"}), provenance={})
        loaded = load_candidates(tmp_path)[0]
        assert loaded.status is CandidateStatus.ERROR
        assert loaded.summary["error"] == "boom"


class TestPromotion:
    def test_refuses_non_accepted(self, tmp_path: Path) -> None:
        for status in (CandidateStatus.REJECTED, CandidateStatus.SKIPPED, CandidateStatus.ERROR):
            with pytest.raises(ValueError, match="refusing to promote"):
                promote_candidate(tmp_path, _candidate(status), provenance={})

    def test_promote_writes_result_and_ff(self, tmp_path: Path) -> None:
        cand = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        cid = cand.candidate_id
        promoted = promote_candidate(tmp_path, cand, provenance={})
        assert promoted["result"] == tmp_path / "accepted" / f"{cid}.json"
        assert promoted["result"].is_file()
        assert promoted["force_field"] == tmp_path / "forcefields" / f"{cid}.frcmod"
        assert promoted["force_field"].is_file()

    def test_rejected_never_overwrites_accepted(self, tmp_path: Path) -> None:
        accepted = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        cid = accepted.candidate_id
        promote_candidate(tmp_path, accepted, provenance={})
        canonical = tmp_path / "accepted" / f"{cid}.json"
        original = canonical.read_bytes()
        rejected = _candidate(CandidateStatus.REJECTED)  # same default profile -> same id
        assert rejected.candidate_id == cid
        persist_candidate(tmp_path, rejected, provenance={})
        with pytest.raises(ValueError):
            promote_candidate(tmp_path, rejected, provenance={})
        assert canonical.read_bytes() == original

    def test_promotion_is_atomic_on_ff_serialization_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # First accepted promotion writes canonical JSON + FF.
        cand = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        cid = cand.candidate_id
        promote_candidate(tmp_path, cand, provenance={})
        json_bytes = (tmp_path / "accepted" / f"{cid}.json").read_bytes()
        ff_bytes = (tmp_path / "forcefields" / f"{cid}.frcmod").read_bytes()

        # A later promotion whose FF serialization raises must leave both
        # canonical artifacts byte-identical (no partial write, no replace).
        import q2mm.io.amber as amber

        def _boom(*_a: object, **_k: object) -> None:
            raise RuntimeError("synthetic serialization failure")

        monkeypatch.setattr(amber, "save_amber_frcmod", _boom)
        with pytest.raises(RuntimeError, match="synthetic serialization failure"):
            promote_candidate(tmp_path, _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff()), provenance={})
        assert (tmp_path / "accepted" / f"{cid}.json").read_bytes() == json_bytes
        assert (tmp_path / "forcefields" / f"{cid}.frcmod").read_bytes() == ff_bytes
        # No leftover temp files.
        assert not list((tmp_path / "forcefields").glob("*.tmp-*"))
        assert not list((tmp_path / "accepted").glob("*.tmp-*"))

    def test_promotion_rolls_back_on_second_replace_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Establish a prior accepted JSON + FF.
        cand = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        cid = cand.candidate_id
        promote_candidate(tmp_path, cand, provenance={})
        json_path = tmp_path / "accepted" / f"{cid}.json"
        ff_path = tmp_path / "forcefields" / f"{cid}.frcmod"
        json_bytes = json_path.read_bytes()
        ff_bytes = ff_path.read_bytes()

        # Fail on the SECOND os.replace (the force field commit) after the
        # first (JSON) has already been committed: rollback must restore the
        # pre-existing JSON bytes exactly and leave the prior FF untouched.
        import q2mm.benchmarks.runner as runner_mod

        real_replace = os.replace
        calls = {"n": 0}

        def _flaky_replace(src: object, dst: object) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("synthetic replace failure")
            real_replace(src, dst)

        monkeypatch.setattr(runner_mod.os, "replace", _flaky_replace)
        with pytest.raises(OSError, match="synthetic replace failure"):
            promote_candidate(tmp_path, _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff()), provenance={})

        assert json_path.read_bytes() == json_bytes
        assert ff_path.read_bytes() == ff_bytes
        assert not list((tmp_path / "accepted").glob("*.tmp-*"))
        assert not list((tmp_path / "accepted").glob("*.bak-*"))
        assert not list((tmp_path / "forcefields").glob("*.tmp-*"))
        assert not list((tmp_path / "forcefields").glob("*.bak-*"))

    def test_promotion_rollback_leaves_nothing_when_no_prior(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No prior canonical artifact: a failed second replace must leave the
        # accepted/ JSON absent (rolled back), not a partial file.
        cand = _candidate(CandidateStatus.ACCEPTED, ff=_harmonic_ff())
        cid = cand.candidate_id
        import q2mm.benchmarks.runner as runner_mod

        real_replace = os.replace
        calls = {"n": 0}

        def _flaky_replace(src: object, dst: object) -> None:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("synthetic replace failure")
            real_replace(src, dst)

        monkeypatch.setattr(runner_mod.os, "replace", _flaky_replace)
        with pytest.raises(OSError, match="synthetic replace failure"):
            promote_candidate(tmp_path, cand, provenance={})
        assert not (tmp_path / "accepted" / f"{cid}.json").exists()
        assert not list((tmp_path / "accepted").glob("*.tmp-*"))
        assert not list((tmp_path / "accepted").glob("*.bak-*"))


# ---------------------------------------------------------------------------
# run_profile: every requested profile is exactly one classified candidate
# ---------------------------------------------------------------------------


class TestRunProfileClassification:
    def test_unknown_backend_is_error(self) -> None:
        cand = run_profile(RunProfile(system="ch3f", backend="does-not-exist", optimizer="scipy-lbfgsb"))
        assert cand.status is CandidateStatus.ERROR
        assert (
            cand.candidate_id
            == RunProfile(system="ch3f", backend="does-not-exist", optimizer="scipy-lbfgsb").candidate_id()
        )

    def test_unavailable_dependency_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force a registered backend's cheap probe to report unavailable.
        import q2mm.backends.registry as registry

        real = registry.get_descriptor("jax")

        class _Desc:
            name = real.name
            api_version = real.api_version
            factory = real.factory
            info = real.info

            def is_available(self) -> tuple[bool, str]:
                return False, "synthetic missing dependency"

        monkeypatch.setattr(registry, "get_descriptor", lambda key: _Desc() if key == "jax" else real)
        cand = run_profile(RunProfile(system="ch3f", backend="jax", optimizer="scipy-lbfgsb-jax"))
        assert cand.status is CandidateStatus.SKIPPED
        assert "missing dependency" in cand.reason

    def test_unknown_system_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

            info = BackendInfo(
                name="fake",
                role=BackendRole.MM,
                capabilities=frozenset({Capability.ENERGY, Capability.FREQUENCIES}),
                functional_forms=frozenset({"harmonic"}),
                provenance=BackendProvenance(backend="fake", role=BackendRole.MM),
            )

        def _boom(*_a: object, **_k: object) -> object:
            raise KeyError("unknown system")

        monkeypatch.setattr("q2mm.benchmarks.systems.load_system", _boom)
        cand = run_profile(
            RunProfile(system="ch3f", backend="fake", functional_form="harmonic", optimizer="scipy-lbfgsb"),
            backend=_FakeBackend(),  # type: ignore[arg-type]
        )
        assert cand.status is CandidateStatus.ERROR

    def test_missing_data_is_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

            info = BackendInfo(
                name="fake",
                role=BackendRole.MM,
                capabilities=frozenset({Capability.ENERGY, Capability.FREQUENCIES}),
                functional_forms=frozenset({"harmonic"}),
                provenance=BackendProvenance(backend="fake", role=BackendRole.MM),
            )

        def _missing(*_a: object, **_k: object) -> object:
            raise FileNotFoundError("no external data")

        monkeypatch.setattr("q2mm.benchmarks.systems.load_system", _missing)
        cand = run_profile(
            RunProfile(system="ch3f", backend="fake", functional_form="harmonic", optimizer="scipy-lbfgsb"),
            backend=_FakeBackend(),  # type: ignore[arg-type]
        )
        assert cand.status is CandidateStatus.SKIPPED

    def test_empty_functional_forms_supports_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeBackend:
            from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability

            info = BackendInfo(
                name="fake",
                role=BackendRole.MM,
                capabilities=frozenset({Capability.ENERGY}),
                functional_forms=frozenset(),
                provenance=BackendProvenance(backend="fake", role=BackendRole.MM),
            )

        cand = run_profile(
            RunProfile(system="ch3f", backend="fake", functional_form="harmonic", optimizer="scipy-lbfgsb"),
            backend=_FakeBackend(),  # type: ignore[arg-type]
        )
        assert cand.status is CandidateStatus.SKIPPED
        assert "does not support" in cand.reason


class TestRunOutcomeOk:
    def test_skips_only_is_ok(self) -> None:
        assert RunOutcome(candidates=(_candidate(CandidateStatus.SKIPPED),)).ok is True

    def test_error_is_not_ok(self) -> None:
        assert RunOutcome(candidates=(_candidate(CandidateStatus.ERROR),)).ok is False

    def test_optimized_none_accepted_is_not_ok(self) -> None:
        assert RunOutcome(candidates=(_candidate(CandidateStatus.REJECTED),)).ok is False

    def test_one_accepted_is_ok(self) -> None:
        outcome = RunOutcome(candidates=(_candidate(CandidateStatus.ACCEPTED), _candidate(CandidateStatus.REJECTED)))
        assert outcome.ok is True


# ---------------------------------------------------------------------------
# Real pipeline (JAX)
# ---------------------------------------------------------------------------


@pytest.mark.jax
class TestRunProfilePipeline:
    def test_accepted_run_persists_full_result_and_promotes(self, tmp_path: Path) -> None:
        profile = RunProfile(
            system="ch3f", backend="jax", functional_form="harmonic", optimizer="scipy-lbfgsb-jax", maxiter=5, n_evals=0
        )
        outcome = run_profiles([profile], output_dir=tmp_path, analyze=False)
        cand = outcome.candidates[0]
        assert cand.status is CandidateStatus.ACCEPTED
        assert cand.optimization_result is not None and cand.final_force_field is not None
        cid = cand.candidate_id
        assert (tmp_path / "candidates" / f"{cid}.json").is_file()
        assert (tmp_path / "accepted" / f"{cid}.json").is_file()
        assert (tmp_path / "forcefields" / f"{cid}.frcmod").is_file()
        # The persisted candidate carries the full canonical result projection.
        loaded = load_candidates(tmp_path)[0]
        opt = loaded.record["optimization_result"]
        assert opt is not None
        assert len(opt["final_params"]) == opt["n_params"]
        assert "category_metrics" in opt and "history" in opt

    def test_rejection_preserves_full_result_and_ff_and_never_promotes(self, tmp_path: Path) -> None:
        # A policy that demands 100% improvement deterministically rejects.
        policy = AcceptancePolicy(no_progress=NoProgressPolicy(max_iterations=10**9, min_improvement_pct=100.0))
        profile = RunProfile(
            system="ch3f", backend="jax", functional_form="harmonic", optimizer="scipy-lbfgsb-jax", maxiter=5, n_evals=0
        )
        cand = run_profile(profile, policy=policy, analyze=False)
        assert cand.status is CandidateStatus.REJECTED
        # Rejected runs still carry the full canonical result + final FF.
        assert cand.optimization_result is not None
        assert cand.final_force_field is not None
        # ... but never promote.
        outcome = run_profiles([profile], output_dir=tmp_path, policy=policy, analyze=False)
        assert not outcome.promoted
        assert not (tmp_path / "accepted").exists()
        assert (tmp_path / "candidates" / f"{cand.candidate_id}.json").is_file()

    def test_data_roots_thread_into_provenance(self, tmp_path: Path) -> None:
        # ch3f accepts a data_dir override; point it at the packaged resource.
        from q2mm.resources import sn2_reference_dir

        profile = RunProfile(
            system="ch3f",
            backend="jax",
            functional_form="harmonic",
            optimizer="scipy-lbfgsb-jax",
            maxiter=2,
            n_evals=0,
            data_roots={"ch3f": str(sn2_reference_dir())},
        )
        cand = run_profile(profile, analyze=False, include_device=False)
        assert cand.resolved is not None
        assert cand.resolved.resolved_data_roots["ch3f"] == str(sn2_reference_dir())

    def test_omitted_data_roots_record_packaged_default(self) -> None:
        # With no explicit data_roots, provenance still records the ACTUAL
        # resolved ch3f root (the packaged resource) rather than an empty map.
        from pathlib import Path as _Path

        from q2mm.resources import sn2_reference_dir

        cand = run_profile(
            RunProfile(
                system="ch3f",
                backend="jax",
                functional_form="harmonic",
                optimizer="scipy-lbfgsb-jax",
                maxiter=2,
                n_evals=0,
            ),
            analyze=False,
            include_device=False,
        )
        assert cand.resolved is not None
        roots = cand.resolved.resolved_data_roots
        assert roots.get("ch3f")
        assert _Path(roots["ch3f"]).resolve() == sn2_reference_dir().expanduser().resolve()

    def test_single_batch_matrix_share_result_model(self) -> None:
        single = run_profiles(
            [RunProfile(system="ch3f", backend="jax", functional_form="harmonic", maxiter=2, n_evals=0)], analyze=False
        )
        matrix = run_profiles(
            [
                RunProfile(system="ch3f", backend="jax", functional_form="harmonic", optimizer=o, maxiter=2, n_evals=0)
                for o in ("scipy-lbfgsb", "scipy-lbfgsb-jax")
            ],
            analyze=False,
        )
        assert len(single.candidates) == 1 and len(matrix.candidates) == 2
        assert all(isinstance(c, CandidateResult) for c in (*single.candidates, *matrix.candidates))

    def test_result_gradient_mode_agrees_with_profile(self) -> None:
        # JAX optimizer -> analytical gradient in the actual result.
        cand = run_profile(
            RunProfile(
                system="ch3f",
                backend="jax",
                functional_form="harmonic",
                optimizer="scipy-lbfgsb-jax",
                maxiter=3,
                n_evals=0,
            ),
            analyze=False,
            include_device=False,
        )
        assert cand.summary["result_gradient_mode"] == "analytical"

    def test_python_executor_result_gradient_is_finite_difference(self) -> None:
        # scipy-lbfgsb uses the Python executor with SciPy internal FD; the
        # actual result gradient mode must be finite_difference and agree with
        # the profile's derived expectation.
        cand = run_profile(
            RunProfile(
                system="ch3f",
                backend="jax",
                functional_form="harmonic",
                optimizer="scipy-lbfgsb",
                maxiter=2,
                n_evals=0,
            ),
            analyze=False,
            include_device=False,
        )
        assert cand.summary["expected_result_gradient_mode"] == "finite_difference"
        assert cand.summary["result_gradient_mode"] == "finite_difference"
        # A successful executed candidate necessarily agrees with the expected
        # gradient mode (a mismatch fails closed as a resolved error), so there
        # is no success-shaped "matches" flag to assert.
        assert cand.status is CandidateStatus.ACCEPTED

    def test_optimizer_and_workflow_built_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Provenance + execution must share one build: resolve_optimizer and
        # _resolve_workflow are each called exactly once per profile run.
        import q2mm.benchmarks.runner as runner_mod

        opt_calls = {"n": 0}
        wf_calls = {"n": 0}
        real_opt = runner_mod.resolve_optimizer
        real_wf = runner_mod._resolve_workflow

        def _spy_opt(profile: RunProfile) -> Any:
            opt_calls["n"] += 1
            return real_opt(profile)

        def _spy_wf(profile: RunProfile) -> Any:
            wf_calls["n"] += 1
            return real_wf(profile)

        monkeypatch.setattr(runner_mod, "resolve_optimizer", _spy_opt)
        monkeypatch.setattr(runner_mod, "_resolve_workflow", _spy_wf)
        run_profile(
            RunProfile(
                system="ch3f",
                backend="jax",
                functional_form="harmonic",
                optimizer="scipy-lbfgsb-jax",
                maxiter=2,
                n_evals=0,
            ),
            analyze=False,
            include_device=False,
        )
        assert opt_calls["n"] == 1
        assert wf_calls["n"] == 1

    def test_post_resolution_execute_failure_is_resolved_error(self, tmp_path: Path) -> None:
        # An exception raised *after* the profile resolves must surface as a
        # resolved ERROR candidate (resolved ID + provenance), never as a
        # requested-profile-ID error and never promoted.
        import json

        import pytest as _pytest

        import q2mm.benchmarks.runner as runner_mod

        profile = RunProfile(
            system="ch3f", backend="jax", functional_form="harmonic", optimizer="scipy-lbfgsb-jax", maxiter=2, n_evals=0
        )
        with _pytest.MonkeyPatch.context() as mp:

            def _boom(**_kw: object) -> CandidateResult:
                raise RuntimeError("synthetic execute failure")

            mp.setattr(runner_mod, "_execute", _boom)
            outcome = run_profiles([profile], output_dir=tmp_path, analyze=False)

        cand = outcome.candidates[0]
        assert cand.status is CandidateStatus.ERROR
        assert cand.resolved is not None
        # Resolved identity, NOT the requested-profile ID.
        assert cand.candidate_id == cand.resolved.candidate_id()
        # The resolved fingerprint is strictly richer than the requested one,
        # so the resolved candidate ID differs from the pre-resolution ID.
        assert cand.candidate_id != profile.candidate_id()
        assert "synthetic execute failure" in cand.reason
        # Error candidates carry neither result nor force field.
        assert cand.optimization_result is None and cand.final_force_field is None
        # The persisted record preserves the resolved fingerprint + provenance.
        rec_path = tmp_path / "candidates" / f"{cand.candidate_id}.json"
        assert rec_path.is_file()
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        assert rec["resolved"] is not None
        assert rec["resolved_fingerprint"] == cand.resolved.fingerprint()
        assert rec["status"] == "error"
        # No canonical promotion for an error candidate.
        assert not outcome.promoted
        assert not (tmp_path / "accepted").exists()

    def test_gradient_provenance_mismatch_fails_closed(self, tmp_path: Path) -> None:
        # Force the expected result gradient to disagree with what the JAX
        # optimizer actually produces ("analytical"): the candidate must fail
        # closed as a resolved ERROR, not be accepted.
        import pytest as _pytest

        import q2mm.benchmarks.runner as runner_mod

        profile = RunProfile(
            system="ch3f", backend="jax", functional_form="harmonic", optimizer="scipy-lbfgsb-jax", maxiter=2, n_evals=0
        )
        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(runner_mod, "_expected_result_gradient", lambda spec: "none")
            outcome = run_profiles([profile], output_dir=tmp_path, analyze=False)

        cand = outcome.candidates[0]
        assert cand.status is CandidateStatus.ERROR
        assert cand.resolved is not None
        assert cand.candidate_id == cand.resolved.candidate_id()
        assert "gradient mode" in cand.reason
        assert cand.optimization_result is None and cand.final_force_field is None
        assert not outcome.promoted
        assert not (tmp_path / "accepted").exists()
