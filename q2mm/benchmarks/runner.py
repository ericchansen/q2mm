"""The one benchmark execution, result, provenance, and persistence path.

:func:`run_profile` is the single execution primitive: it resolves a
:class:`~q2mm.benchmarks.profiles.RunProfile`, composes the canonical
``ObjectivePlan`` / evaluator / optimizer / workflow contracts, evaluates
:class:`~q2mm.benchmarks.acceptance.AcceptancePolicy` **before** anything
is promoted, and returns one immutable :class:`CandidateResult` whose
status is exactly one of accepted / rejected / skipped / error.  It never
raises for a run-time failure — every requested profile becomes exactly one
candidate.

:func:`run_profiles` drives single, batch (convergence), and matrix
(backend x form x optimizer) requests through that same primitive.  Every
candidate is persisted incrementally to a stable ``candidates/`` location
regardless of outcome (with the full canonical ``OptimizationResult``
projection for accepted *and* rejected runs), and only *accepted*
candidates are atomically promoted to the canonical result and force-field
names.  A rejected / skipped / errored candidate can never overwrite an
existing accepted canonical artifact; promotion serialises both the JSON
and the force field to temporary siblings and ``os.replace``s them into
place only after both succeed.

Fit metrics (R2 / RMSD / MAE / per-category) come solely from
:mod:`q2mm.objectives.metrics`.  Frequency-space and PES-distortion
analysis are benchmark analysis that live here (not in a models or
diagnostics layer); PES distortion preserves each molecule's explicit
topology via :meth:`~q2mm.models.molecule.Molecule.with_geometry`.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from q2mm.benchmarks.acceptance import AcceptanceDecision, AcceptancePolicy, CandidateStatus, improvement_percent
from q2mm.benchmarks.profiles import RunProfile
from q2mm._canonical import json_value
from q2mm.constants import REAL_FREQUENCY_THRESHOLD
from q2mm.models.results import deep_freeze
from q2mm.objectives.metrics import category_metrics, category_stats

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend, BackendDescriptor
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.benchmarks.profiles import ResolvedProfile
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Molecule
    from q2mm.models.results import OptimizationResult

logger = logging.getLogger("q2mm.benchmarks.runner")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Registry key -> canonical q2mm-data directory name.
DATA_DIR_FOR_SYSTEM: Mapping[str, str] = MappingProxyType(
    {
        "ch3f": "ch3f",
        "ch3f-sn2": "ch3f-sn2",
        "rh-enamide": "rh-enamide",
        "heck-relay": "heck-relay",
        "pd-allyl": "pd-allyl-amination",
        "pd-conjugate": "pd-1,4-conjugate-addition",
        "rh-conjugate": "rh-1,4-conjugate-addition",
        "ferrocene": "ferrocene",
    }
)


class ConfigurationError(RuntimeError):
    """A profile referenced something that does not exist (typo / bad config)."""


class ExecutionError(RuntimeError):
    """A candidate failed during execution *after* its profile resolved.

    Raised from within :func:`_execute` (e.g. a gradient-provenance
    mismatch).  The runner converts it into a resolved ``error`` candidate
    that preserves the resolved fingerprint/provenance and never promotes.
    """


# ---------------------------------------------------------------------------
# Deterministic JSON-safe serialization
# ---------------------------------------------------------------------------


def sanitize_for_json(value: Any) -> Any:
    """Recursively coerce *value* into strict-JSON-safe primitives.

    Non-finite floats become the sentinel strings ``"NaN"`` /
    ``"Infinity"`` / ``"-Infinity"`` (valid strict JSON, still readable),
    NumPy scalars/arrays become Python scalars/lists, and read-only
    mappings become plain dicts.
    """
    return json_value(value, strict=False, coerce_keys=True)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write *payload* to *path* as strict, sorted-key JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(sanitize_for_json(payload), fh, indent=2, allow_nan=False, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Public atomic strict-JSON writer (see :func:`_write_json_atomic`)."""
    _write_json_atomic(path, payload)


def read_json(path: Path) -> dict[str, Any]:
    """Load a JSON candidate record written by :func:`write_json`."""
    with Path(path).open(encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)
    return data


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_info(repo: Path) -> dict[str, Any]:
    if not (repo / ".git").exists():
        return {}
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, stderr=subprocess.DEVNULL, text=True
        ).strip()
        return {"git_sha": sha, "git_dirty": bool(dirty)}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {}


def build_run_provenance(*, generator: str, output_dir: Path) -> dict[str, Any]:
    """Build the run-level provenance block (timestamp/command are non-identity)."""
    return {
        "generator": generator,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line": shlex.join(sys.argv),
        "q2mm": _git_info(REPO_ROOT),
        "output_dir": str(output_dir),
    }


# ---------------------------------------------------------------------------
# Frequency-space benchmark analysis
# ---------------------------------------------------------------------------


def real_frequencies(freqs: Iterable[float], threshold: float = REAL_FREQUENCY_THRESHOLD) -> np.ndarray:
    """Return the sorted real (non-imaginary/non-rigid) frequencies above *threshold*."""
    arr = np.asarray(list(freqs), dtype=float)
    return np.sort(arr[arr > threshold])


def frequency_rmsd(a: Iterable[float], b: Iterable[float]) -> float:
    """RMSD between two frequency arrays (truncated to the shorter length).

    Delegates the RMSD formula to :func:`q2mm.objectives.metrics.category_stats`
    so there is one RMSD implementation.
    """
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return float("nan")
    return category_stats(arr_a[:n], arr_b[:n])["rmsd"]


def frequency_mae(a: Iterable[float], b: Iterable[float]) -> float:
    """MAE between two frequency arrays (truncated to the shorter length)."""
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return float("nan")
    return category_stats(arr_a[:n], arr_b[:n])["mae"]


def _mm_real_frequencies(backend: Backend, molecules: Sequence[Any], ff: ForceField) -> np.ndarray:
    from q2mm.backends.contracts import FrequencyRequest, PreparationRequest
    from q2mm.models.parameters import ParameterLayout

    params = ParameterLayout.from_force_field(ff).vector(ff)
    all_real: list[float] = []
    for idx, mol in enumerate(molecules):
        prepared = backend.prepare(PreparationRequest(case_id=str(idx), molecule=mol, force_field=ff))
        mm_freqs = prepared.frequencies(FrequencyRequest(parameters=params)).frequencies
        all_real.extend(real_frequencies(mm_freqs).tolist())
    return np.array(sorted(all_real), dtype=float)


def _frequency_analysis(
    backend: Backend,
    case: BenchmarkCase,
    initial_ff: ForceField,
    final_ff: ForceField | None,
) -> dict[str, Any]:
    if not case.qm_freqs_per_mol:
        return {}
    from q2mm.backends.contracts import Capability

    if Capability.FREQUENCIES not in backend.info.capabilities:
        return {}
    qm_real = np.sort(np.concatenate([np.asarray(f, dtype=float) for f in case.qm_freqs_per_mol]))
    molecules = list(case.problem.molecules)
    analysis: dict[str, Any] = {"n_qm_real": int(qm_real.size)}
    init_real = _mm_real_frequencies(backend, molecules, initial_ff)
    analysis["initial_rmsd"] = frequency_rmsd(qm_real, init_real)
    if final_ff is not None:
        final_real = _mm_real_frequencies(backend, molecules, final_ff)
        analysis["final_rmsd"] = frequency_rmsd(qm_real, final_real)
        analysis["final_mae"] = frequency_mae(qm_real, final_real)
    return analysis


# ---------------------------------------------------------------------------
# PES distortion benchmark analysis (topology-preserving)
# ---------------------------------------------------------------------------

_HA_TO_KCAL = 627.5094740631


def compute_distortions(
    mol: Molecule,
    ff: ForceField,
    backend: Backend,
    modes: Mapping[str, np.ndarray],
    target_norms_ang: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Displace a molecule along QM normal modes and compare MM to QM energies.

    Displaced geometries are produced with
    :meth:`~q2mm.models.molecule.Molecule.with_geometry`, so the molecule's
    explicit topology (including an explicitly empty topology), atom types,
    charge, multiplicity, and identity are preserved rather than re-inferred.
    """
    from q2mm.backends.contracts import EnergyRequest, PreparationRequest
    from q2mm.constants import AMU_TO_KG, BOHR_TO_ANG, HARTREE_TO_J, SPEED_OF_LIGHT_MS
    from q2mm.models.parameters import ParameterLayout

    if target_norms_ang is None:
        target_norms_ang = (0.05, 0.10, 0.15)

    params = ParameterLayout.from_force_field(ff).vector(ff)

    def _mm_energy(structure: Molecule, case_id: str) -> float:
        prepared = backend.prepare(PreparationRequest(case_id=case_id, molecule=structure, force_field=ff))
        return float(prepared.energy(EnergyRequest(parameters=params)).energy)

    eigenvalues = np.asarray(modes["eigenvalues"], dtype=float)
    eigenvectors = np.asarray(modes["eigenvectors"], dtype=float)
    masses_amu = np.asarray(modes["masses_amu"], dtype=float)

    bohr_to_m = BOHR_TO_ANG * 1e-10
    sqrt_m = np.sqrt(np.repeat(masses_amu, 3))
    real_mode_indices = [i for i, ev in enumerate(eigenvalues) if ev > 1e-3]

    e_eq = _mm_energy(mol, "eq")
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for mi in real_mode_indices:
        ev = eigenvalues[mi]
        evec_mw = eigenvectors[:, mi]
        ev_si = ev * HARTREE_TO_J / (bohr_to_m**2 * AMU_TO_KG)
        freq_cm1 = float(np.sqrt(ev_si) / (2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0))
        v_cart = evec_mw / sqrt_m
        v_cart_ang = v_cart * BOHR_TO_ANG
        v_norm = float(np.linalg.norm(v_cart_ang))
        displacements: list[dict[str, Any]] = []
        for d_ang in target_norms_ang:
            q = d_ang / v_norm
            e_qm = 0.5 * ev * q**2 * _HA_TO_KCAL
            delta_xyz = (q * v_cart * BOHR_TO_ANG).reshape(-1, 3)
            disp_mol = mol.with_geometry(mol.geometry + delta_xyz)
            e_mm = _mm_energy(disp_mol, f"disp_{mi}_{d_ang}") - e_eq
            pct_err = ((e_mm - e_qm) / e_qm * 100.0) if abs(e_qm) > 1e-8 else 0.0
            displacements.append({"d_ang": d_ang, "e_qm": e_qm, "e_mm": e_mm, "pct_err": pct_err})
        results.append({"mode_idx": mi, "freq_cm1": freq_cm1, "displacements": displacements})
    elapsed = time.perf_counter() - t0
    return results, e_eq, elapsed


def _pes_distortion_summary(backend: Backend, case: BenchmarkCase, final_ff: ForceField) -> dict[str, Any]:
    if case.normal_modes is None:
        return {}
    from q2mm.backends.contracts import Capability

    if Capability.ENERGY not in backend.info.capabilities:
        return {}
    molecules = list(case.problem.molecules)
    modes = {k: np.asarray(v, dtype=float) for k, v in case.normal_modes.items()}
    distortions, _e_eq, elapsed = compute_distortions(molecules[0], final_ff, backend, modes)
    errors = [abs(d["pct_err"]) for m in distortions for d in m["displacements"]]
    return {
        "modes": distortions,
        "median_error_pct": float(np.median(errors)) if errors else 0.0,
        "max_error_pct": float(np.max(errors)) if errors else 0.0,
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Executor-ratio gate + mean/CI
# ---------------------------------------------------------------------------


def classify_ratio(ratio: float, tol: float | None) -> dict[str, Any]:
    """Classify the JAX-executor / objective-score ratio for the safety gate."""
    if not math.isfinite(ratio):
        return {
            "executor_ratio": None,
            "executor_ratio_status": "diverged" if math.isinf(ratio) else "nan",
            "executor_ratio_passes": False,
        }
    if tol is None:
        return {"executor_ratio": ratio, "executor_ratio_status": "ok_bypassed", "executor_ratio_passes": True}
    passes = (1.0 - tol) <= ratio <= (1.0 + tol)
    return {
        "executor_ratio": ratio,
        "executor_ratio_status": "ok" if passes else "out_of_band",
        "executor_ratio_passes": passes,
    }


def _mean_ci95(samples: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    if not math.isfinite(std) or std == 0.0:
        return mean, 0.0
    from scipy.stats import t

    ci95 = float(t.ppf(0.975, arr.size - 1) * std / math.sqrt(arr.size))
    return mean, ci95


def _score_interval_summary(initial: Sequence[float], final: Sequence[float]) -> dict[str, Any]:
    if not initial or not final:
        return {}
    initial_mean, initial_ci95 = _mean_ci95(initial)
    final_mean, final_ci95 = _mean_ci95(final)
    improvement = 100.0 * (1.0 - final_mean / initial_mean) if initial_mean > 0 else 0.0
    return {
        "initial_obj_score_mean": initial_mean,
        "initial_obj_score_ci95": initial_ci95,
        "final_obj_score_mean": final_mean,
        "final_obj_score_ci95": final_ci95,
        "improvement_pct_mean": improvement,
        "improvement_significant": bool(abs(final_mean - initial_mean) > (initial_ci95 + final_ci95)),
    }


# ---------------------------------------------------------------------------
# Canonical result projection
# ---------------------------------------------------------------------------


def _candidate_record_to_dict(rec: Any) -> dict[str, Any]:
    return {
        "index": int(rec.index),
        "status": rec.status,
        "n_params": int(rec.n_params),
        "layout_fingerprint": rec.layout_fingerprint,
        "initial_params": np.asarray(rec.initial_params, dtype=float).tolist(),
        "final_params": np.asarray(rec.final_params, dtype=float).tolist(),
        "initial_score": float(rec.initial_score),
        "final_score": float(rec.final_score),
        "message": str(rec.message),
        "seed": rec.seed,
    }


def _stage_to_dict(stage: Any) -> dict[str, Any]:
    return {
        "name": stage.name,
        "initial_score": float(stage.initial_score),
        "final_score": float(stage.final_score),
        "n_iterations": int(stage.n_iterations),
        "n_evaluations": int(stage.n_evaluations),
        "converged": bool(stage.converged),
        "message": str(stage.message),
        "gradient_mode": str(stage.gradient_mode),
        "fd_step": stage.fd_step,
        "elapsed_s": float(stage.elapsed_s),
        "locked_param_indices": list(stage.locked_param_indices),
    }


def result_to_dict(result: OptimizationResult) -> dict[str, Any]:
    """Full JSON-safe projection of the one canonical :class:`OptimizationResult`.

    Includes layout identity, full initial/final vectors, counts, history,
    gradient mode / FD step, multi-start candidate records, workflow stage
    records, endpoint samples, and per-category metrics — so an accepted or
    rejected candidate persists its complete result, not just scores.
    """
    return {
        "success": bool(result.success),
        "message": str(result.message),
        "initial_score": float(result.initial_score),
        "final_score": float(result.final_score),
        "n_iterations": int(result.n_iterations),
        "n_evaluations": int(result.n_evaluations),
        "n_params": int(result.n_params),
        "layout_fingerprint": result.layout_fingerprint,
        "initial_params": np.asarray(result.initial_params, dtype=float).tolist(),
        "final_params": np.asarray(result.final_params, dtype=float).tolist(),
        "history": [float(x) for x in result.history],
        "method": result.method,
        "gradient_mode": result.gradient_mode,
        "fd_step": result.fd_step,
        "initial_samples": [float(x) for x in result.initial_samples],
        "final_samples": [float(x) for x in result.final_samples],
        "category_metrics": {k: dict(v) for k, v in result.category_metrics.items()},
        "candidates": [_candidate_record_to_dict(c) for c in result.candidates],
        "stages": [_stage_to_dict(s) for s in result.stages],
    }


# ---------------------------------------------------------------------------
# Optimizer / evaluator / workflow resolution (records effective settings)
# ---------------------------------------------------------------------------


def resolve_optimizer(profile: RunProfile) -> tuple[Any, dict[str, Any]]:
    """Compatibility adapter over :mod:`q2mm.optimizers.catalog`."""
    from q2mm.optimizers.catalog import optimizer_option_names, resolve_optimizer as _resolve

    options = {
        "maxiter": profile.maxiter,
        "ftol": profile.ftol,
        "fc_fraction": profile.fc_fraction,
        "eq_fraction": profile.eq_fraction,
        "learning_rate": profile.learning_rate,
        "max_params": profile.max_params,
        "max_cycles": profile.max_cycles,
        "convergence": profile.convergence,
        "seed": profile.seed,
    }
    allowed = optimizer_option_names(profile.optimizer_spec)
    return _resolve(profile.optimizer_spec, {key: value for key, value in options.items() if key in allowed})


def _build_evaluator_factory(backend: Backend, base_ff: ForceField, spec: Any) -> Any:
    from q2mm.objectives.protocols import GradientMode
    from q2mm.workflows import make_evaluator_factory

    if spec.evaluator == "jax":
        return make_evaluator_factory(backend, base_ff, executor="jax")
    gm = GradientMode.FINITE_DIFFERENCE if spec.gradient_mode == "finite_difference" else GradientMode.NONE
    return make_evaluator_factory(backend, base_ff, executor="python", gradient_mode=gm, fd_step=spec.fd_step)


def _resolve_workflow(profile: RunProfile) -> tuple[Any, dict[str, Any]]:
    from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow

    if profile.workflow == "method-e2":
        wf = MethodE2Workflow()
        settings = {
            "name": "method-e2",
            "negative_fc_threshold": wf.negative_fc_threshold,
            "replace_with_round2": wf.replace_with_round2,
            "allow_negative": wf.allow_negative,
            "near_zero_replace_with": dict(wf.near_zero_replace_with),
        }
        return wf, settings
    return SingleStageWorkflow(), {"name": "single-stage"}


# ---------------------------------------------------------------------------
# Candidate result model (immutable)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CandidateResult:
    """One immutable, terminal outcome of :func:`run_profile`.

    Attributes:
        candidate_id: Stable, filesystem-safe identity (readable prefix +
            deterministic fingerprint suffix).
        status: Terminal :class:`~q2mm.benchmarks.acceptance.CandidateStatus`.
        reason: Human-readable explanation of the status.
        profile: The originating :class:`RunProfile`.
        resolved: The provenance-complete
            :class:`~q2mm.benchmarks.profiles.ResolvedProfile`, or ``None``
            when the run failed before resolution.
        summary: Deeply-frozen JSON-safe metrics/scores for the run.
        optimization_result: The one canonical ``OptimizationResult`` for an
            accepted *or* rejected run (``None`` for skipped/errored).
        final_force_field: The materialized force field for an accepted *or*
            rejected run (``None`` for skipped/errored); only accepted
            candidates are ever promoted.

    """

    candidate_id: str
    status: CandidateStatus
    reason: str
    profile: RunProfile
    resolved: ResolvedProfile | None
    summary: Mapping[str, Any] = field(default_factory=dict)
    optimization_result: OptimizationResult | None = None
    final_force_field: ForceField | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "summary", deep_freeze(dict(self.summary)))
        if not self.reason:
            raise ValueError("CandidateResult.reason must be non-empty.")
        has_run = self.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED)
        if has_run and (self.optimization_result is None or self.final_force_field is None):
            raise ValueError(
                f"{self.status.value} candidate must carry both an OptimizationResult and a final force field."
            )
        if not has_run and (self.optimization_result is not None or self.final_force_field is not None):
            raise ValueError(f"{self.status.value} candidate must not carry an OptimizationResult or force field.")
        if self.resolved is not None and self.candidate_id != self.resolved.candidate_id():
            raise ValueError("CandidateResult.candidate_id must equal resolved.candidate_id() when resolved.")
        if self.resolved is None and self.candidate_id != self.profile.candidate_id():
            raise ValueError(
                "CandidateResult.candidate_id must equal the requested profile candidate_id() before resolution."
            )

    @property
    def accepted(self) -> bool:
        """``True`` only for an accepted candidate."""
        return self.status is CandidateStatus.ACCEPTED

    def record(self) -> dict[str, Any]:
        """Return the JSON-safe persisted record (without run provenance)."""
        return {
            "candidate_id": self.candidate_id,
            "status": self.status.value,
            "reason": self.reason,
            "profile": {**self.profile.canonical_dict(), "label": self.profile.label},
            "profile_fingerprint": self.profile.fingerprint(),
            "resolved": self.resolved.to_dict() if self.resolved is not None else None,
            "resolved_fingerprint": self.resolved.fingerprint() if self.resolved is not None else None,
            "summary": sanitize_for_json(dict(self.summary)),
            "optimization_result": (
                result_to_dict(self.optimization_result) if self.optimization_result is not None else None
            ),
        }


@dataclass(frozen=True, eq=False)
class LoadedCandidate:
    """A deeply-frozen candidate record loaded from disk (incl. failures)."""

    candidate_id: str
    status: CandidateStatus
    reason: str
    path: Path
    record: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", deep_freeze(dict(self.record)))

    @property
    def summary(self) -> Mapping[str, Any]:
        """The persisted metrics/scores summary."""
        summary = self.record.get("summary", {})
        return summary if isinstance(summary, Mapping) else MappingProxyType({})


# ---------------------------------------------------------------------------
# Backend / system loading with error-vs-skip classification
# ---------------------------------------------------------------------------


def _classify_backend(profile: RunProfile) -> tuple[Any, Any]:
    """Return ``(descriptor, backend)``.

    Raises :class:`ConfigurationError` for an unknown backend key (typo) or a
    broken/misconfigured factory; a registered backend whose cheap probe is
    unhealthy raises :class:`~q2mm.backends.contracts.BackendUnavailableError`
    (a graceful skip).
    """
    from q2mm.backends.contracts import BackendUnavailableError
    from q2mm.backends.registry import BackendNotRegistered, get_descriptor

    try:
        descriptor = get_descriptor(profile.backend)
    except BackendNotRegistered as exc:
        raise ConfigurationError(f"unknown backend {profile.backend!r}: {exc}") from exc

    healthy, reason = descriptor.is_available()
    if not healthy:
        raise BackendUnavailableError(f"backend {profile.backend!r} unavailable: {reason}")

    load_kwargs: dict[str, Any] = {}
    if profile.backend == "openmm" and profile.platform is not None:
        load_kwargs["platform_name"] = profile.platform
    try:
        backend = descriptor.load(**load_kwargs)
    except BackendUnavailableError:
        raise
    except Exception as exc:  # broken/misconfigured factory -> configuration error
        raise ConfigurationError(f"backend {profile.backend!r} failed to construct: {exc!r}") from exc
    return descriptor, backend


def _norm_path(raw: str) -> Path:
    """Normalize a user-supplied data-root path: expand ~ and resolve."""
    return Path(raw).expanduser().resolve()


def _load_kwargs(profile: RunProfile, form: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Build ``load_system`` kwargs and the *actually resolved* data-root map.

    Explicit roots are normalized with ``expanduser().resolve()``.  When a
    root is omitted, the actual location the loader will use is still
    recorded — the packaged ``sn2_reference_dir()`` for CH3F, or the
    environment-fallback ``ExternalDataRoots`` for publication systems — so
    provenance never claims ``{}`` while an environment root was in force.
    """
    kwargs: dict[str, Any] = {
        "functional_form": form,
        "starting_point": profile.starting_point,
        "qfuerza_replace_with": profile.qfuerza_replace_with,
    }
    roots = dict(profile.data_roots)
    resolved: dict[str, str] = {}

    if profile.system in ("ch3f", "ch3f-sn2"):
        if "ch3f" in roots:
            data_dir = _norm_path(roots["ch3f"])
            kwargs["data_dir"] = data_dir
            resolved["ch3f"] = str(data_dir)
        else:
            # Record the packaged resource directory the loader will use.
            from q2mm.resources import sn2_reference_dir

            resolved["ch3f"] = str(_norm_path(str(sn2_reference_dir())))
        return kwargs, resolved

    objective_profile = profile.effective_objective_profile
    if objective_profile is None:
        raise ConfigurationError(f"Publication system {profile.system!r} has no resolved objective profile.")
    kwargs["objective_profile"] = objective_profile

    from q2mm.benchmarks.systems._paths import ExternalDataRoots, resolve_external_roots

    explicit = ExternalDataRoots(
        rh_enamide=_norm_path(roots["rh_enamide"]) if "rh_enamide" in roots else None,
        supporting_info=_norm_path(roots["supporting_info"]) if "supporting_info" in roots else None,
        mm3_base=_norm_path(roots["mm3_base"]) if "mm3_base" in roots else None,
    )
    # Fold explicit overrides together with the documented environment
    # fallbacks so provenance reflects exactly what the loader resolves.
    effective = resolve_external_roots(explicit)
    kwargs["data_roots"] = effective
    for key in ("rh_enamide", "supporting_info", "mm3_base"):
        value = getattr(effective, key)
        if value is not None:
            resolved[key] = str(Path(value).expanduser().resolve())
    return kwargs, resolved


def _data_provenance(case: BenchmarkCase, resolved_roots: Mapping[str, str]) -> dict[str, Any]:
    problem = case.problem
    cases = [{"case_id": c.case_id, "stationary_point": c.stationary_point.value} for c in problem.cases]
    hessians: list[dict[str, Any]] = []
    for c in problem.cases:
        hp = c.molecule.hessian_provenance
        hessians.append(
            {
                "case_id": c.case_id,
                "units": None if hp is None else hp.units.value,
                "source": None if hp is None else hp.source,
                "path": None if hp is None else hp.path,
            }
        )
    return {
        "metadata": dict(case.metadata),
        "objective_profile": (
            problem.publication_metadata.objective_profile.identifier
            if problem.publication_metadata is not None
            else None
            if problem.preparation_provenance is None
            else problem.preparation_provenance.profile
        ),
        "publication_metadata": (
            None if problem.publication_metadata is None else problem.publication_metadata.to_dict()
        ),
        "publication_metadata_fingerprint": (
            None if problem.publication_metadata is None else problem.publication_metadata.fingerprint
        ),
        "cases": cases,
        "hessians": hessians,
        "default_forms": list(case.default_forms),
        "description": case.description,
        "resolved_data_roots": dict(resolved_roots),
    }


def _default_form(system: str) -> str:
    from q2mm.benchmarks.systems import system_metadata

    forms = system_metadata(system).default_forms
    return forms[0] if forms else "mm3"


def _is_jax_backend(backend: Any) -> bool:
    try:
        from q2mm.backends.mm.jax_engine import JaxBackend
    except Exception:
        return False
    return isinstance(backend, JaxBackend)


# ---------------------------------------------------------------------------
# Core execution primitive
# ---------------------------------------------------------------------------


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
            descriptor, backend = _classify_backend(profile)
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
    form = profile.functional_form or _default_form(profile.system)
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
    load_kwargs, resolved_roots = _load_kwargs(profile, form)
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
    optimizer_obj, optimizer_settings = resolve_optimizer(profile)
    workflow_obj, workflow_settings = _resolve_workflow(profile)

    from q2mm.benchmarks.profiles import resolve as _resolve_profile

    data_provenance = _data_provenance(case, resolved_roots)
    if success_spec is not None:
        data_provenance["publication_success_spec"] = success_spec.to_dict()
    resolved = _resolve_profile(
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

    base_summary: dict[str, Any] = {
        "system": profile.system,
        "backend": profile.backend,
        "backend_name": backend_info.name,
        "functional_form": form,
        "workflow": profile.workflow,
        "optimizer": profile.optimizer,
        "optimizer_method": spec.method,
        "optimizer_label": spec.label,
        "evaluator": evaluator_kind,
        "gradient_mode": gradient_mode,
        "expected_result_gradient_mode": expected_grad,
        "effective_regularization": profile.effective_regularization,
        "starting_point": profile.starting_point,
        "objective_profile": profile.effective_objective_profile,
        "reproduction_status": (
            None if problem.publication_metadata is None else problem.publication_metadata.status.value
        ),
        "publication_metadata_fingerprint": (
            None if problem.publication_metadata is None else problem.publication_metadata.fingerprint
        ),
        "publication_success_spec": None if success_spec is None else success_spec.to_dict(),
        "starting_point_audit": case.metadata.get("starting_point_audit"),
        "n_molecules": len(problem.molecules),
        "n_active_params": problem.active_space.n_active,
    }

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


def _expected_result_gradient(spec: Any) -> str:
    """Derive the gradient mode the resulting ``OptimizationResult`` should report.

    - JAX executor -> ``analytical`` (SciPy-JAX, Optax, JaxOpt, cycling-JAX).
    - Python executor-FD -> ``finite_difference``.
    - Python NONE + a gradient-driven method (L-BFGS-B / cycling / multistart /
      basinhopping) -> ``finite_difference`` (SciPy internal FD).
    - Derivative-free (Nelder-Mead / Powell) -> ``none``.
    """
    from q2mm.optimizers.catalog import expected_result_gradient

    return expected_result_gradient(spec)


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

    summary: dict[str, Any] = {
        **base_summary,
        "initial_obj_score": initial_score,
        "initial_category_scores": initial_category_scores,
        "seminario": seminario_categories,
    }

    ratio_info: dict[str, Any] = {}
    if is_jax_backend:
        from q2mm.objectives.jax import JaxObjectiveExecutor

        jax_score = float(JaxObjectiveExecutor(record_plan, backend, initial_ff).value(baseline))
        summary["initial_jax_score"] = jax_score if math.isfinite(jax_score) else float("inf")
        ratio = jax_score / initial_score if initial_score > 0 else float("nan")
        ratio_info = classify_ratio(ratio, profile.executor_ratio_tol)
        summary.update(ratio_info)

    if analyze:
        freq = _frequency_analysis(backend, case, initial_ff, None)
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
        {
            "final_obj_score": final_score,
            "final_category_scores": final_category_scores,
            "improvement_pct": improvement_pct,
            "initial_optimizer_score": float(result.initial_score),
            "final_optimizer_score": float(result.final_score),
            "n_iterations": int(result.n_iterations),
            "n_evaluations": int(result.n_evaluations),
            "converged": bool(result.success),
            "message": str(result.message),
            "result_gradient_mode": actual_grad,
            "expected_result_gradient_mode": expected_grad,
            "result_fd_step": result.fd_step,
            "opt_time_s": elapsed,
            "optimized": optimized_categories,
            "final_executor_ratio": final_executor_ratio,
            "stages": [_stage_to_dict(s) for s in result.stages],
        }
    )
    summary.update(_score_interval_summary(list(result.initial_samples), list(result.final_samples)))

    if analyze:
        freq = _frequency_analysis(backend, case, initial_ff, final_ff)
        if freq:
            summary["frequencies"] = freq
        if case.normal_modes is not None and len(problem.molecules) == 1:
            pes = _pes_distortion_summary(backend, case, final_ff)
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


# ---------------------------------------------------------------------------
# Persistence + atomic promotion
# ---------------------------------------------------------------------------


def _candidate_path(output_dir: Path, candidate_id: str) -> Path:
    return output_dir / "candidates" / f"{candidate_id}.json"


def persist_candidate(output_dir: Path, candidate: CandidateResult, provenance: Mapping[str, Any]) -> Path:
    """Write *candidate* to the stable ``candidates/`` location (all statuses).

    Persists the complete canonical result projection for accepted and
    rejected candidates alike; the collision-free candidate ID names the file.
    """
    path = _candidate_path(output_dir, candidate.candidate_id)
    _write_json_atomic(path, {"provenance": dict(provenance), **candidate.record()})
    return path


def _ff_extension(ff: ForceField) -> str:
    from q2mm.models.forcefield import FunctionalForm

    if ff.functional_form is FunctionalForm.MM3:
        return ".fld"
    if ff.functional_form is FunctionalForm.HARMONIC:
        return ".frcmod"
    raise ValueError(f"no force-field serializer for functional form {ff.functional_form!r}.")


def _serialize_ff(ff: ForceField, path: Path) -> None:
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.application.persistence import save

    if ff.functional_form is FunctionalForm.MM3:
        save(ff, path, format="mm3_fld")
    elif ff.functional_form is FunctionalForm.HARMONIC:
        save(ff, path, format="amber_frcmod")
    else:
        raise ValueError(f"no force-field serializer for functional form {ff.functional_form!r}.")


def _opposite_ext(ext: str) -> str:
    return ".fld" if ext == ".frcmod" else ".frcmod"


def promote_candidate(output_dir: Path, candidate: CandidateResult, provenance: Mapping[str, Any]) -> dict[str, Path]:
    """Atomically promote an accepted candidate to canonical output names.

    Refuses any non-accepted candidate.  The accepted result JSON and the
    optimized force field are serialised to temporary siblings first (a
    serialisation failure changes nothing).  Pre-existing canonical artifacts
    are snapshotted, then committed with ``os.replace``; a failure at **any**
    commit step (including the second replace) restores every pre-existing
    canonical JSON / force field byte-identically, and leaves nothing behind
    when there was no prior artifact.  The stale opposite-form force field is
    removed only after a fully successful commit.
    """
    if not candidate.accepted:
        raise ValueError(
            f"refusing to promote non-accepted candidate {candidate.candidate_id!r} ({candidate.status.value})."
        )
    accepted_dir = output_dir / "accepted"
    ff_dir = output_dir / "forcefields"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    result_path = accepted_dir / f"{candidate.candidate_id}.json"
    payload = sanitize_for_json({"provenance": dict(provenance), **candidate.record()})

    ff = candidate.final_force_field
    ext = _ff_extension(ff) if ff is not None else None
    ff_path = ff_dir / f"{candidate.candidate_id}{ext}" if ext is not None else None
    tmp_ff = ff_path.with_name(f"{ff_path.name}.tmp-{os.getpid()}") if ff_path is not None else None
    tmp_json = result_path.with_name(f"{result_path.name}.tmp-{os.getpid()}")

    # ---- serialise both temporaries first (failure here changes nothing) --
    with tmp_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, allow_nan=False, sort_keys=True)
        fh.write("\n")
    try:
        if ff is not None and tmp_ff is not None:
            ff_dir.mkdir(parents=True, exist_ok=True)
            _serialize_ff(ff, tmp_ff)
    except BaseException:
        tmp_json.unlink(missing_ok=True)
        raise

    # ---- snapshot pre-existing canonical targets, then commit -----------
    import shutil

    targets: list[tuple[Path, Path]] = [(tmp_json, result_path)]
    if ff is not None and tmp_ff is not None and ff_path is not None:
        targets.append((tmp_ff, ff_path))

    backups: list[tuple[Path, Path | None]] = []
    committed: list[Path] = []
    for _tmp, target in targets:
        if target.exists():
            backup_path = target.with_name(f"{target.name}.bak-{os.getpid()}")
            shutil.copy2(target, backup_path)
            backups.append((target, backup_path))
        else:
            backups.append((target, None))

    try:
        for tmp, target in targets:
            os.replace(tmp, target)
            committed.append(target)
        # Fully committed: drop the opposite-form stale file and the backups.
        if ff_path is not None and ext is not None:
            stale = ff_dir / f"{candidate.candidate_id}{_opposite_ext(ext)}"
            stale.unlink(missing_ok=True)
        for _target, bak in backups:
            if bak is not None:
                bak.unlink(missing_ok=True)
    except BaseException:
        # Roll back every committed replace to its pre-existing bytes (or
        # remove it when there was no prior artifact).
        for target, bak in backups:
            if target in committed:
                if bak is not None:
                    os.replace(bak, target)
                else:
                    target.unlink(missing_ok=True)
            elif bak is not None:
                bak.unlink(missing_ok=True)
        raise
    finally:
        tmp_json.unlink(missing_ok=True)
        if tmp_ff is not None:
            tmp_ff.unlink(missing_ok=True)

    promoted: dict[str, Path] = {"result": result_path}
    if ff_path is not None:
        promoted["force_field"] = ff_path
    return promoted


def load_candidates(directory: Path) -> list[LoadedCandidate]:
    """Load every persisted candidate record under *directory* (all statuses)."""
    directory = Path(directory)
    search = directory / "candidates" if (directory / "candidates").is_dir() else directory
    loaded: list[LoadedCandidate] = []
    for path in sorted(search.glob("*.json")):
        try:
            record = read_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("could not load candidate %s: %s", path.name, exc)
            continue
        status_value = str(record.get("status", "error"))
        try:
            status = CandidateStatus(status_value)
        except ValueError:
            status = CandidateStatus.ERROR
        loaded.append(
            LoadedCandidate(
                candidate_id=str(record.get("candidate_id", path.stem)),
                status=status,
                reason=str(record.get("reason", "")),
                path=path,
                record=record,
            )
        )
    return loaded


# ---------------------------------------------------------------------------
# High-level drivers (single / batch / matrix all share run_profiles)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RunOutcome:
    """Immutable aggregate outcome of a :func:`run_profiles` invocation."""

    candidates: tuple[CandidateResult, ...] = ()
    promoted: Mapping[str, Mapping[str, Path]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(
            self, "promoted", MappingProxyType({k: MappingProxyType(dict(v)) for k, v in self.promoted.items()})
        )

    def by_status(self, status: CandidateStatus) -> tuple[CandidateResult, ...]:
        """Return candidates in a given terminal status."""
        return tuple(c for c in self.candidates if c.status is status)

    @property
    def accepted(self) -> tuple[CandidateResult, ...]:
        """Accepted candidates."""
        return self.by_status(CandidateStatus.ACCEPTED)

    @property
    def optimized_candidates(self) -> tuple[CandidateResult, ...]:
        """Candidates that actually ran an optimization (accepted or rejected)."""
        return tuple(c for c in self.candidates if c.status in (CandidateStatus.ACCEPTED, CandidateStatus.REJECTED))

    @property
    def ok(self) -> bool:
        """False on any error, or when optimizations ran but none was accepted.

        A run of only skips (e.g. registered-but-unavailable backends, or a
        matrix/smoke with no runnable combos) is still ``ok``.
        """
        if self.by_status(CandidateStatus.ERROR):
            return False
        ran = self.optimized_candidates
        return not (ran and not self.accepted)


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
        provenance = build_run_provenance(generator=generator, output_dir=output_dir)

    for profile in profiles:
        logger.info("[%s] running", profile.candidate_id())
        candidate = run_profile(profile, policy=policy, analyze=analyze)
        candidates.append(candidate)
        if output_dir is not None:
            persist_candidate(output_dir, candidate, provenance)
            if promote and candidate.accepted:
                promoted[candidate.candidate_id] = promote_candidate(output_dir, candidate, provenance)
        logger.info("[%s] %s: %s", candidate.candidate_id, candidate.status.value, candidate.reason)

    outcome = RunOutcome(candidates=tuple(candidates), promoted=promoted)
    if not outcome.ok:
        logger.error(
            "BATCH FAILURE: %d optimization(s) ran but none was accepted (or a candidate errored); the optimizer did "
            "not make acceptable progress. Inspect executor_ratio_tol, ftol, bounds, and the starting force field.",
            len(outcome.optimized_candidates),
        )
    return outcome
