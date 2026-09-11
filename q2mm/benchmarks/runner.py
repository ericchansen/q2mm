"""The one benchmark execution and promotion coordinator.

``run_profile`` resolves a request, executes the canonical application
workflow, evaluates the existing acceptance policy, and produces one terminal
candidate. ``run_profiles`` drives that same path for single, batch, and
matrix requests, persisting every outcome and promoting only accepted ones.

Configuration is owned by ``profiles``, diagnostic calculations by
``analysis``, record/provenance projection by ``records``, and file mechanics
by ``artifacts``. Public imports historically provided here remain direct
re-exports of those canonical owners, not parallel implementations.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from q2mm.benchmarks import analysis, artifacts, profiles, records
from q2mm.benchmarks.acceptance import AcceptanceDecision, AcceptancePolicy, CandidateStatus, improvement_percent
from q2mm.benchmarks.acceptance import classify_ratio as classify_ratio
from q2mm.benchmarks.analysis import (
    compute_distortions as compute_distortions,
    frequency_mae as frequency_mae,
    frequency_rmsd as frequency_rmsd,
    real_frequencies as real_frequencies,
)
from q2mm.benchmarks.artifacts import (
    load_candidates as load_candidates,
    persist_candidate as persist_candidate,
    promote_candidate as promote_candidate,
    read_json as read_json,
    write_json as write_json,
)
from q2mm.benchmarks.profiles import (
    DATA_DIR_FOR_SYSTEM as DATA_DIR_FOR_SYSTEM,
    ConfigurationError as ConfigurationError,
    RunProfile as RunProfile,
    resolve_optimizer as resolve_optimizer,
)
from q2mm.benchmarks.records import (
    REPO_ROOT as REPO_ROOT,
    CandidateResult as CandidateResult,
    LoadedCandidate as LoadedCandidate,
    RunOutcome as RunOutcome,
    build_run_provenance as build_run_provenance,
    result_to_dict as result_to_dict,
    sanitize_for_json as sanitize_for_json,
)
from q2mm.objectives.metrics import category_metrics
from q2mm.optimizers.catalog import expected_result_gradient as _expected_result_gradient

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend, BackendDescriptor
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.benchmarks.profiles import ResolvedProfile
    from q2mm.models.forcefield import ForceField
    from q2mm.models.results import OptimizationResult

logger = logging.getLogger("q2mm.benchmarks.runner")


class ExecutionError(RuntimeError):
    """A candidate failed during execution *after* its profile resolved.

    Raised from within :func:`_execute` (e.g. a gradient-provenance
    mismatch).  The runner converts it into a resolved ``error`` candidate
    that preserves the resolved fingerprint/provenance and never promotes.
    """


def _is_jax_backend(backend: Any) -> bool:
    try:
        from q2mm.backends.mm.jax_engine import JaxBackend
    except Exception:
        return False
    return isinstance(backend, JaxBackend)


def _terminal(
    candidate_id: str,
    decision: AcceptanceDecision,
    profile: RunProfile,
    resolved: ResolvedProfile | None,
    summary: Mapping[str, Any],
    *,
    result: OptimizationResult | None = None,
    final_ff: ForceField | None = None,
) -> CandidateResult:
    return CandidateResult(
        candidate_id=candidate_id,
        status=decision.status,
        reason=decision.reason,
        profile=profile,
        resolved=resolved,
        summary=summary,
        optimization_result=result,
        final_force_field=final_ff,
    )


def run_profile(
    profile: RunProfile,
    *,
    backend: Backend | None = None,
    descriptor: BackendDescriptor | None = None,
    policy: AcceptancePolicy | None = None,
    analyze: bool = True,
    include_device: bool = True,
) -> CandidateResult:
    """Execute one :class:`RunProfile` and return its terminal candidate.

    Never raises: a configuration typo (unknown backend/system/profile) or a
    broken backend factory yields ``error`` (with the requested-profile ID),
    an unavailable dependency or missing data yields ``skipped``, and any
    unexpected execution failure yields ``error`` — so every requested
    profile becomes exactly one candidate.
    """
    policy = policy or AcceptancePolicy()
    requested_id = profile.candidate_id()
    try:
        return _run_profile_inner(
            profile,
            backend=backend,
            descriptor=descriptor,
            policy=policy,
            analyze=analyze,
            include_device=include_device,
        )
    except ConfigurationError as exc:
        return _terminal(requested_id, AcceptancePolicy.errored(str(exc)), profile, None, {"error": str(exc)})
    except Exception as exc:  # never raise before a candidate exists
        logger.exception("[%s] unexpected failure", requested_id)
        return _terminal(
            requested_id, AcceptancePolicy.errored(f"unexpected failure: {exc!r}"), profile, None, {"error": repr(exc)}
        )


def _run_profile_inner(
    profile: RunProfile,
    *,
    backend: Backend | None,
    descriptor: BackendDescriptor | None,
    policy: AcceptancePolicy,
    analyze: bool,
    include_device: bool,
) -> CandidateResult:
    from q2mm.backends.contracts import BackendUnavailableError

    requested_id = profile.candidate_id()
    spec = profile.optimizer_spec

    # ---- backend: unknown/broken -> error; unhealthy dep -> skipped ------
    if backend is None:
        try:
            descriptor, backend = profiles._classify_backend(profile)
        except BackendUnavailableError as exc:
            return _terminal(requested_id, AcceptancePolicy.skipped(str(exc)), profile, None, {})
    elif descriptor is None:
        # Injected backend: recover the static descriptor for provenance when
        # the key is registered.  Only a genuine "not registered" is tolerated;
        # any other registry failure surfaces rather than silently degrading.
        from q2mm.backends.registry import BackendNotRegistered, get_descriptor

        try:
            descriptor = get_descriptor(profile.backend)
        except BackendNotRegistered:
            descriptor = None
    backend_info = backend.info

    # ---- functional form + backend support (empty forms => none) --------
    form = profile.functional_form or profiles._default_form(profile.system)
    if form not in backend_info.functional_forms:
        return _terminal(
            requested_id,
            AcceptancePolicy.skipped(f"backend {backend_info.name!r} does not support functional form {form!r}"),
            profile,
            None,
            {},
        )

    # ---- JAX-only optimizers require a JAX executor ---------------------
    if spec.evaluator == "jax" and not _is_jax_backend(backend):
        return _terminal(
            requested_id,
            AcceptancePolicy.skipped(
                f"optimizer {profile.optimizer!r} requires the JAX executor; backend is {backend_info.name!r}"
            ),
            profile,
            None,
            {},
        )

    # ---- load the system: missing data -> skipped; else -> error --------
    load_kwargs, resolved_roots = profiles._load_kwargs(profile, form)
    from q2mm.benchmarks.systems import load_system

    try:
        if profile.system in ("ch3f", "ch3f-sn2"):
            load_kwargs["backend"] = backend
        case = load_system(profile.system, **load_kwargs)
    except Exception as exc:
        from q2mm.benchmarks.publications import PublicationProfileBlockedError, PublicationProfileError

        if isinstance(exc, PublicationProfileBlockedError):
            publication = exc.record
            return _terminal(
                requested_id,
                AcceptancePolicy.errored(str(exc)),
                profile,
                None,
                {
                    "system": profile.system,
                    "objective_profile": publication.objective_profile.identifier,
                    "reproduction_status": publication.status.value,
                    "publication_metadata": publication.to_dict(),
                    "publication_metadata_fingerprint": publication.fingerprint,
                    "blocked": True,
                },
            )
        if isinstance(exc, PublicationProfileError):
            raise ConfigurationError(str(exc)) from exc
        if not isinstance(exc, FileNotFoundError):
            raise
        return _terminal(
            requested_id,
            AcceptancePolicy.skipped(f"system {profile.system!r} data unavailable: {exc}"),
            profile,
            None,
            {},
        )

    problem = case.problem
    success_spec = None
    if problem.publication_metadata is not None:
        from q2mm.benchmarks.publications import publication_success_spec

        success_spec = publication_success_spec(
            profile.system,
            profile.effective_objective_profile or "",
            profile.starting_point,
        )
    evaluator_kind = spec.evaluator
    gradient_mode = spec.gradient_mode
    fd_step = spec.fd_step if spec.gradient_mode == "finite_difference" else None
    expected_grad = _expected_result_gradient(spec)

    # Build the optimizer + workflow exactly once; the same instances feed both
    # provenance and execution so recorded settings identify what actually ran.
    optimizer_obj, optimizer_settings = profiles.resolve_optimizer(profile)
    workflow_obj, workflow_settings = profiles._resolve_workflow(profile)

    data_provenance = records._data_provenance(case, resolved_roots)
    if success_spec is not None:
        data_provenance["publication_success_spec"] = success_spec.to_dict()
    resolved = profiles.resolve(
        profile,
        descriptor=descriptor,
        backend_info=backend_info,
        functional_form=form,
        evaluator=evaluator_kind,
        gradient_mode=gradient_mode,
        expected_result_gradient_mode=expected_grad,
        fd_step=fd_step,
        effective_regularization=profile.effective_regularization,
        optimizer_settings=optimizer_settings,
        workflow_settings=workflow_settings,
        layout_fingerprint=problem.layout.fingerprint,
        n_active_params=problem.active_space.n_active,
        n_full_params=problem.active_space.n_full,
        n_molecules=len(problem.molecules),
        data_provenance=data_provenance,
        resolved_data_roots=resolved_roots,
        include_device=include_device,
    )
    candidate_id = resolved.candidate_id()

    base_summary: dict[str, Any] = records._resolved_summary(
        profile=profile,
        backend_info=backend_info,
        case=case,
        form=form,
        spec=spec,
        evaluator_kind=evaluator_kind,
        gradient_mode=gradient_mode,
        expected_grad=expected_grad,
        success_spec=success_spec,
    )

    # Execute with the resolved identity in hand: any failure here is a
    # POST-resolution error, so it must surface as a resolved ``error``
    # candidate (resolved ID + provenance), not fall through to the outer
    # pre-resolution handler that only has the requested-profile ID.
    try:
        return _execute(
            profile=profile,
            policy=policy,
            backend=backend,
            case=case,
            resolved=resolved,
            candidate_id=candidate_id,
            spec=spec,
            optimizer=optimizer_obj,
            workflow=workflow_obj,
            expected_grad=expected_grad,
            base_summary=base_summary,
            analyze=analyze,
            is_jax_backend=_is_jax_backend(backend),
            success_spec=success_spec,
        )
    except Exception as exc:
        logger.exception("[%s] execution failed after resolution", candidate_id)
        return _terminal(
            candidate_id,
            AcceptancePolicy.errored(f"execution failed: {exc!r}"),
            profile,
            resolved,
            {**base_summary, "error": repr(exc)},
        )


def _execute(
    *,
    profile: RunProfile,
    policy: AcceptancePolicy,
    backend: Backend,
    case: BenchmarkCase,
    resolved: ResolvedProfile,
    candidate_id: str,
    spec: Any,
    optimizer: Any,
    workflow: Any,
    expected_grad: str,
    base_summary: dict[str, Any],
    analyze: bool,
    is_jax_backend: bool,
    success_spec: Any | None,
) -> CandidateResult:
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor

    problem = case.problem
    initial_ff = problem.starting_force_field
    layout = problem.layout
    baseline = np.asarray(problem.active_space.baseline, dtype=float)
    regularization = profile.effective_regularization

    # ---- objective-of-record baseline (Python executor, regularized) ----
    record_plan = ObjectivePlan.from_problem(problem, regularization=regularization)
    obj_initial = PythonObjectiveExecutor(record_plan, backend, initial_ff)
    initial_evaluation = obj_initial.evaluate(baseline)
    initial_score = float(initial_evaluation.total)
    initial_category_scores = dict(initial_evaluation.category_scores)
    seminario_categories = category_metrics(record_plan, initial_evaluation)

    summary: dict[str, Any] = records._initial_summary(
        base_summary, initial_score, initial_category_scores, seminario_categories
    )

    ratio_info: dict[str, Any] = {}
    if is_jax_backend:
        from q2mm.objectives.jax import JaxObjectiveExecutor

        jax_score = float(JaxObjectiveExecutor(record_plan, backend, initial_ff).value(baseline))
        summary["initial_jax_score"] = jax_score if math.isfinite(jax_score) else float("inf")
        ratio = jax_score / initial_score if initial_score > 0 else float("nan")
        ratio_info = classify_ratio(ratio, profile.executor_ratio_tol)
        summary.update(ratio_info)

    if analyze:
        freq = analysis._frequency_analysis(backend, case, initial_ff, None)
        if freq:
            summary["frequencies"] = freq

    # ---- skip decisions -------------------------------------------------
    if profile.skip_optimization:
        summary["skipped"] = True
        return _terminal(
            candidate_id, AcceptancePolicy.skipped("skip_optimization requested"), profile, resolved, summary
        )
    if profile.executor_ratio_tol is not None and ratio_info and not ratio_info["executor_ratio_passes"]:
        status = ratio_info["executor_ratio_status"]
        reason = "executor-ratio gate closed: " + ("out of band" if status == "out_of_band" else status)
        summary["skipped"] = True
        return _terminal(candidate_id, AcceptancePolicy.skipped(reason), profile, resolved, summary)

    # ---- run the workflow through the generic application boundary --------
    from q2mm.application.optimization import execute_optimization
    from q2mm.objectives.protocols import GradientMode

    executor_kind: Literal["python", "jax"] = "jax" if spec.evaluator == "jax" else "python"
    executor_gradient = (
        GradientMode.ANALYTICAL
        if executor_kind == "jax"
        else GradientMode.FINITE_DIFFERENCE
        if spec.gradient_mode == "finite_difference"
        else GradientMode.NONE
    )
    t0 = time.perf_counter()
    result, final_ff = execute_optimization(
        problem,
        backend,
        optimizer,
        workflow,
        executor=executor_kind,
        gradient_mode=executor_gradient,
        fd_step=spec.fd_step,
        n_evals=profile.n_evals,
        regularization=regularization,
    )
    elapsed = time.perf_counter() - t0

    final_vector = np.asarray(result.final_params, dtype=float)

    obj_final = PythonObjectiveExecutor(record_plan, backend, final_ff)
    final_evaluation = obj_final.evaluate(final_vector)
    final_score = float(final_evaluation.total)
    final_category_scores = dict(final_evaluation.category_scores)
    optimized_categories = category_metrics(record_plan, final_evaluation)
    improvement_pct = improvement_percent(initial_score, final_score)
    final_executor_ratio = float(result.final_score) / final_score if final_score > 0 else float("nan")

    # Fail closed on a gradient-provenance mismatch: a successful executed
    # candidate must report the expected gradient mode.  A disagreement means
    # the optimizer did not run the objective the way provenance claims, so
    # the candidate cannot be trusted or promoted.
    actual_grad = str(result.gradient_mode)
    if actual_grad != expected_grad:
        raise ExecutionError(
            f"result gradient mode {actual_grad!r} != expected {expected_grad!r} "
            f"(method={spec.method}, optimizer={profile.optimizer!r})"
        )

    summary.update(
        records._optimization_summary(
            result=result,
            final_score=final_score,
            final_category_scores=final_category_scores,
            improvement_pct=improvement_pct,
            actual_grad=actual_grad,
            expected_grad=expected_grad,
            elapsed=elapsed,
            optimized_categories=optimized_categories,
            final_executor_ratio=final_executor_ratio,
        )
    )
    summary.update(
        analysis._score_interval_summary(result.initial_samples, result.final_samples, executor=executor_kind)
    )

    if analyze:
        freq = analysis._frequency_analysis(backend, case, initial_ff, final_ff)
        if freq:
            summary["frequencies"] = freq
        if case.normal_modes is not None and len(problem.molecules) == 1:
            pes = analysis._pes_distortion_summary(backend, case, final_ff)
            if pes:
                summary["pes_distortion"] = pes

    decision = policy.evaluate(
        n_iterations=int(result.n_iterations),
        initial_score=initial_score,
        final_score=final_score,
        converged=bool(result.success),
    )
    if success_spec is not None and success_spec.methodology_blocker is not None:
        summary["publication_methodology_blocker"] = success_spec.methodology_blocker
        decision = AcceptanceDecision(
            CandidateStatus.REJECTED,
            f"publication optimization proof blocked: {success_spec.methodology_blocker}",
        )
    elif success_spec is not None and success_spec.canonical_full_run:
        success_audit = success_spec.audit(
            improvement_percent=improvement_pct,
            initial_executor_ratio=ratio_info.get("executor_ratio"),
            final_executor_ratio=final_executor_ratio,
            initial_category_scores=initial_category_scores,
            final_category_scores=final_category_scores,
            optimizer_converged=bool(result.success),
            accepted=decision.is_accepted,
        )
        summary["publication_success_audit"] = success_audit
        if not success_audit["passes"]:
            decision = AcceptanceDecision(
                CandidateStatus.REJECTED,
                "publication success gate failed: " + "; ".join(success_audit["failures"]),
            )
    summary["acceptance"] = {"status": decision.status.value, "reason": decision.reason}

    # Both accepted AND rejected retain the full canonical result + final FF.
    return _terminal(candidate_id, decision, profile, resolved, summary, result=result, final_ff=final_ff)


def run_profiles(
    profiles: Sequence[RunProfile],
    *,
    output_dir: Path | None = None,
    generator: str = "q2mm.benchmarks.runner",
    policy: AcceptancePolicy | None = None,
    analyze: bool = True,
    promote: bool = True,
) -> RunOutcome:
    """Run every profile, persist each candidate, and promote accepted ones.

    The one execution/result/provenance path shared by the CLI's single,
    batch, and matrix operations — they differ only in how many profiles they
    hand it.
    """
    policy = policy or AcceptancePolicy()
    candidates: list[CandidateResult] = []
    promoted: dict[str, Mapping[str, Path]] = {}
    provenance: Mapping[str, Any] = {}
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        provenance = records.build_run_provenance(generator=generator, output_dir=output_dir)

    for profile in profiles:
        logger.info("[%s] running", profile.candidate_id())
        candidate = run_profile(profile, policy=policy, analyze=analyze)
        candidates.append(candidate)
        if output_dir is not None:
            artifacts.persist_candidate(output_dir, candidate, provenance)
            if promote and candidate.accepted:
                promoted[candidate.candidate_id] = artifacts.promote_candidate(output_dir, candidate, provenance)
        logger.info("[%s] %s: %s", candidate.candidate_id, candidate.status.value, candidate.reason)

    outcome = RunOutcome(candidates=tuple(candidates), promoted=promoted)
    if not outcome.ok:
        logger.error(
            "BATCH FAILURE: %d optimization(s) ran but none was accepted (or a candidate errored); the optimizer did "
            "not make acceptable progress. Inspect executor_ratio_tol, ftol, bounds, and the starting force field.",
            len(outcome.optimized_candidates),
        )
    return outcome
