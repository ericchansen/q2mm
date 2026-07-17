"""Immutable run profiles and deterministic, provenance-complete resolution.

A :class:`RunProfile` is the immutable *requested* specification of one
benchmark run: system, backend, functional form, starting point, external
data roots, evaluator/optimizer/workflow, and their exact settings.  It
carries no loaded data and performs no I/O — a plain value object that can
be compared, serialised, and fingerprinted deterministically.

:func:`resolve` turns a profile plus the loaded backend descriptor, case,
and effective optimizer/workflow settings into a :class:`ResolvedProfile`
— a provenance-complete record of *exactly* what a run executed.  Two
fingerprints exist and mean different things:

- :meth:`RunProfile.fingerprint` — the **logical requested** identity
  (settings only; no environment).
- :meth:`ResolvedProfile.fingerprint` — the **exact resolved** identity
  over every resolved field: static descriptor (name/api_version/factory/
  static info), runtime backend provenance, capabilities/forms, concrete
  form, evaluator/gradient/FD settings, effective optimizer/workflow
  settings, layout fingerprint/counts, complete data provenance,
  dependency versions, device/platform, seed, and settings.  Differing
  settings, data, or environment therefore never collide.

Candidate identities are a readable prefix plus a deterministic fingerprint
suffix: a pre-resolution error uses :meth:`RunProfile.candidate_id`; a
resolved run uses :meth:`ResolvedProfile.candidate_id` (which embeds the
concrete resolved form and the full resolved fingerprint).

All fingerprints use canonical, cross-process-deterministic JSON — never
Python's salted ``hash`` or dict iteration order.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from q2mm.models.results import deep_freeze

if TYPE_CHECKING:
    from q2mm.backends.contracts import BackendDescriptor, BackendInfo

__all__ = [
    "OptimizerSpec",
    "OPTIMIZER_CATALOG",
    "FUNCTIONAL_FORMS",
    "STARTING_POINTS",
    "WORKFLOWS",
    "EVALUATORS",
    "GRADIENT_MODES",
    "DATA_ROOT_KEYS",
    "RunProfile",
    "ResolvedProfile",
    "canonical_json",
    "canonical_fingerprint",
    "dependency_versions",
    "device_info",
    "resolve",
]


# ---------------------------------------------------------------------------
# Deterministic serialization / fingerprint
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    """Coerce mappings/sequences/scalars to plain JSON-safe values.

    Read-only mapping views become plain dicts; tuples/lists become lists;
    non JSON-native scalars fall back to ``str``.  Used to render frozen
    provenance structures for canonical serialization and on-disk output.
    """
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        # Encode non-finite floats as sentinels so JSON stays strict and the
        # fingerprint is stable regardless of the platform's NaN payload.
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return str(value)


def canonical_json(payload: Any) -> str:
    """Serialise *payload* to canonical, cross-process-deterministic JSON.

    ASCII output, sorted object keys, fixed separators, ``allow_nan=False``
    — identical bytes across processes and ``PYTHONHASHSEED`` values.
    """
    return json.dumps(_jsonify(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def canonical_fingerprint(payload: Any) -> str:
    """Return ``sha256:<hex>`` over :func:`canonical_json` of *payload*."""
    digest = hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()
    return f"sha256:{digest}"


def _short(fingerprint: str) -> str:
    """Return the full hex digest of a ``sha256:<hex>`` fingerprint.

    The complete 64-character digest is used as the candidate-ID suffix so
    the ID is genuinely collision-free; the readable prefix keeps IDs
    human-scannable and (with the current prefix lengths) within Windows
    path limits.
    """
    return fingerprint.split(":", 1)[-1]


def _slug(*parts: str) -> str:
    """Join *parts* into one filesystem-safe, lowercase-preserving slug."""
    return "_".join(parts).replace(" ", "-").replace(":", "-").replace("/", "-")


# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: Objective executor kinds.
EVALUATORS: frozenset[str] = frozenset({"python", "jax"})

#: Evaluator-declared gradient modes (values of GradientMode).
GRADIENT_MODES: frozenset[str] = frozenset({"analytical", "finite_difference", "none"})

#: Supported functional forms benchmarked by default.
FUNCTIONAL_FORMS: tuple[str, ...] = ("harmonic", "mm3")

#: Starting-point strategies for TS systems.
STARTING_POINTS: frozenset[str] = frozenset({"qfuerza", "published"})

#: Workflow identifiers accepted by a profile.
WORKFLOWS: frozenset[str] = frozenset({"single-stage", "method-e2"})

#: External data-root keys a profile may configure.  ``ch3f`` maps to the
#: packaged-resource ``data_dir`` override; the others map to
#: :class:`~q2mm.benchmarks.systems._paths.ExternalDataRoots` fields.
DATA_ROOT_KEYS: frozenset[str] = frozenset({"ch3f", "rh_enamide", "supporting_info", "mm3_base"})


# ---------------------------------------------------------------------------
# Optimizer catalog
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class OptimizerSpec:
    """One entry in the optimizer catalog.

    Attributes:
        key: Stable CLI/profile slug (e.g. ``"scipy-lbfgsb-jax"``).
        label: Human-readable leaderboard label.
        method: Method string consumed by the runner's optimizer resolver
            (e.g. ``"L-BFGS-B"``, ``"optax:adam"``, ``"cycling"``).
        evaluator: ``"python"`` or ``"jax"`` — the objective executor kind.
        gradient_mode: The evaluator's declared gradient mode
            (``"analytical"`` for JAX, ``"none"`` for a Python executor that
            leaves finite differences to the optimizer, or
            ``"finite_difference"`` for a Python executor that computes its
            own finite differences).
        fd_step: Central finite-difference step for a
            ``finite_difference`` Python executor (ignored otherwise).
        extra: Deeply-frozen method-specific keyword arguments (e.g.
            ``schedule``, ``n_starts``, ``regularization``).

    """

    key: str
    label: str
    method: str
    evaluator: str
    gradient_mode: str = "none"
    fd_step: float = 1e-4
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evaluator not in EVALUATORS:
            raise ValueError(f"OptimizerSpec.evaluator must be one of {sorted(EVALUATORS)}, got {self.evaluator!r}.")
        if self.gradient_mode not in GRADIENT_MODES:
            raise ValueError(f"OptimizerSpec.gradient_mode must be one of {sorted(GRADIENT_MODES)}.")
        if self.evaluator == "jax" and self.gradient_mode != "analytical":
            raise ValueError("A JAX-executor optimizer must declare gradient_mode='analytical'.")
        if not (math.isfinite(self.fd_step) and self.fd_step > 0.0):
            raise ValueError(f"OptimizerSpec.fd_step must be positive and finite, got {self.fd_step!r}.")
        object.__setattr__(self, "extra", deep_freeze(dict(self.extra)))

    @property
    def regularization(self) -> float:
        """L2 regularization strength this optimizer requests (0 if none)."""
        return float(self.extra.get("regularization", 0.0))


def _spec(key: str, label: str, method: str, evaluator: str, *, gradient_mode: str, **extra: Any) -> OptimizerSpec:
    return OptimizerSpec(
        key=key, label=label, method=method, evaluator=evaluator, gradient_mode=gradient_mode, extra=extra
    )


#: The one catalog of optimizer configurations shared by runner and CLI.
OPTIMIZER_CATALOG: Mapping[str, OptimizerSpec] = MappingProxyType(
    {
        spec.key: spec
        for spec in (
            _spec("scipy-lbfgsb", "SciPy L-BFGS-B (SciPy FD)", "L-BFGS-B", "python", gradient_mode="none"),
            _spec("scipy-lbfgsb-jax", "SciPy L-BFGS-B (JAX grad)", "L-BFGS-B", "jax", gradient_mode="analytical"),
            _spec(
                "scipy-lbfgsb-fd",
                "SciPy L-BFGS-B (executor FD)",
                "L-BFGS-B",
                "python",
                gradient_mode="finite_difference",
            ),
            _spec("scipy-nm", "Nelder-Mead", "Nelder-Mead", "python", gradient_mode="none"),
            _spec("scipy-powell", "Powell", "Powell", "python", gradient_mode="none"),
            _spec("grad-simp", "Grad-Simp", "cycling", "python", gradient_mode="none"),
            _spec("grad-simp-auto", "Grad-Simp (JAX grad)", "cycling", "jax", gradient_mode="analytical"),
            _spec("optax-adam", "Optax Adam", "optax:adam", "jax", gradient_mode="analytical"),
            _spec(
                "optax-adam-cosine",
                "Optax Adam+cosine",
                "optax:adam",
                "jax",
                gradient_mode="analytical",
                schedule="cosine",
            ),
            _spec("optax-adagrad", "Optax AdaGrad", "optax:adagrad", "jax", gradient_mode="analytical"),
            _spec("optax-sgd", "Optax SGD", "optax:sgd", "jax", gradient_mode="analytical"),
            _spec("basinhopping", "Basin-hopping (T=1.0)", "basinhopping", "python", gradient_mode="none", niter=25),
            _spec(
                "basinhopping-cold",
                "Basin-hopping (T=0.5)",
                "basinhopping",
                "python",
                gradient_mode="none",
                niter=25,
                T=0.5,
            ),
            _spec("multi-lbfgsb-5", "Multi-start n=5", "multi:L-BFGS-B", "python", gradient_mode="none", n_starts=5),
            _spec("multi-lbfgsb-10", "Multi-start n=10", "multi:L-BFGS-B", "python", gradient_mode="none", n_starts=10),
            _spec(
                "scipy-lbfgsb-l2",
                "SciPy L-BFGS-B+L2 (SciPy FD)",
                "L-BFGS-B",
                "python",
                gradient_mode="none",
                regularization=0.01,
            ),
            _spec(
                "optax-adam-l2", "Optax Adam+L2", "optax:adam", "jax", gradient_mode="analytical", regularization=0.01
            ),
            _spec(
                "jaxopt-lbfgs", "JaxOpt L-BFGS", "jaxopt:lbfgs", "jax", gradient_mode="analytical", regularization=0.01
            ),
            _spec("jaxopt-lbfgsb", "JaxOpt L-BFGS-B", "jaxopt:lbfgsb", "jax", gradient_mode="analytical"),
            _spec(
                "grad-simp-multi",
                "Grad-Simp (multi inner)",
                "cycling",
                "python",
                gradient_mode="none",
                full_method="multi:L-BFGS-B",
            ),
        )
    }
)


# ---------------------------------------------------------------------------
# RunProfile
# ---------------------------------------------------------------------------


def _validate_fraction(value: float | None, name: str) -> None:
    if value is None:
        return
    if not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise ValueError(f"RunProfile.{name} must be in (0, 1] or None, got {value!r}.")


@dataclass(frozen=True, eq=False)
class RunProfile:
    """Immutable specification of one requested benchmark run.

    Attributes:
        system: Registry key of the benchmark system (e.g. ``"ch3f"``).
        backend: Registry key of the MM backend (e.g. ``"jax"``).
        functional_form: Explicit form (``"harmonic"``/``"mm3"``), or
            ``None`` to use the backend/system default.
        starting_point: ``"qfuerza"`` (Farrugia 2025) or ``"published"``.
        workflow: ``"single-stage"`` or ``"method-e2"``.
        optimizer: Key into :data:`OPTIMIZER_CATALOG`.
        maxiter: Optimizer iteration cap (non-negative), or ``None`` for the
            optimizer's own default.
        ftol: Positive L-BFGS-B function-value tolerance.
        fc_fraction / eq_fraction: Fractional bound widths in ``(0, 1]`` for
            force-constant / equilibrium params, or ``None`` for sanity bounds.
        regularization: Non-negative L2 penalty strength applied to the
            objective plan (overrides the optimizer catalog's default when
            > 0; otherwise the catalog's value is used).
        n_evals: Non-negative post-hoc real-objective samples per endpoint.
        executor_ratio_tol: Non-negative JAX/Python score-ratio gate
            tolerance; ``None`` disables the gate.
        skip_optimization: Compute the baseline only; never optimize.
        qfuerza_replace_with: Positive replacement (Hartree/Bohr^2) for the
            negative TS-Hessian eigenvalue during QFUERZA construction.
        platform: OpenMM platform override (ignored by other backends).
        data_roots: Immutable mapping of external data-root keys (a subset of
            :data:`DATA_ROOT_KEYS`) to filesystem path strings.
        seed: Deterministic seed for stochastic optimizers.
        label: Optional human-facing label; not part of run identity.

    """

    system: str
    backend: str = "jax"
    functional_form: str | None = None
    starting_point: str = "qfuerza"
    workflow: str = "single-stage"
    optimizer: str = "scipy-lbfgsb-jax"
    maxiter: int | None = None
    ftol: float = 1e-8
    fc_fraction: float | None = None
    eq_fraction: float | None = None
    regularization: float | None = None
    learning_rate: float = 1e-3
    max_params: int = 3
    max_cycles: int = 10
    convergence: float = 0.01
    n_evals: int = 1
    executor_ratio_tol: float | None = None
    skip_optimization: bool = False
    qfuerza_replace_with: float = 1.0
    platform: str | None = None
    data_roots: Mapping[str, str] = field(default_factory=dict)
    seed: int = 0
    label: str = ""

    def __post_init__(self) -> None:
        if not self.system:
            raise ValueError("RunProfile.system must be a non-empty registry key.")
        if not self.backend:
            raise ValueError("RunProfile.backend must be a non-empty registry key.")
        if self.functional_form is not None and self.functional_form not in FUNCTIONAL_FORMS:
            raise ValueError(
                f"RunProfile.functional_form must be one of {FUNCTIONAL_FORMS} or None, got {self.functional_form!r}."
            )
        if self.starting_point not in STARTING_POINTS:
            raise ValueError(f"RunProfile.starting_point must be one of {sorted(STARTING_POINTS)}.")
        if self.workflow not in WORKFLOWS:
            raise ValueError(f"RunProfile.workflow must be one of {sorted(WORKFLOWS)}.")
        if self.optimizer not in OPTIMIZER_CATALOG:
            raise ValueError(
                f"RunProfile.optimizer must be a catalog key {sorted(OPTIMIZER_CATALOG)}, got {self.optimizer!r}."
            )
        if self.maxiter is not None and (not isinstance(self.maxiter, int) or self.maxiter < 0):
            raise ValueError(f"RunProfile.maxiter must be a non-negative int or None, got {self.maxiter!r}.")
        if not (math.isfinite(self.ftol) and self.ftol > 0.0):
            raise ValueError(f"RunProfile.ftol must be positive and finite, got {self.ftol!r}.")
        _validate_fraction(self.fc_fraction, "fc_fraction")
        _validate_fraction(self.eq_fraction, "eq_fraction")
        if self.regularization is not None and not (math.isfinite(self.regularization) and self.regularization >= 0.0):
            raise ValueError(
                f"RunProfile.regularization must be non-negative and finite or None, got {self.regularization!r}."
            )
        if not (math.isfinite(self.learning_rate) and self.learning_rate > 0.0):
            raise ValueError(f"RunProfile.learning_rate must be positive and finite, got {self.learning_rate!r}.")
        if not isinstance(self.max_params, int) or self.max_params < 1:
            raise ValueError(f"RunProfile.max_params must be an int >= 1, got {self.max_params!r}.")
        if not isinstance(self.max_cycles, int) or self.max_cycles < 1:
            raise ValueError(f"RunProfile.max_cycles must be an int >= 1, got {self.max_cycles!r}.")
        if not (math.isfinite(self.convergence) and self.convergence > 0.0):
            raise ValueError(f"RunProfile.convergence must be positive and finite, got {self.convergence!r}.")
        if not isinstance(self.n_evals, int) or self.n_evals < 0:
            raise ValueError(f"RunProfile.n_evals must be a non-negative int, got {self.n_evals!r}.")
        if self.executor_ratio_tol is not None and not (
            math.isfinite(self.executor_ratio_tol) and self.executor_ratio_tol >= 0.0
        ):
            raise ValueError(
                f"RunProfile.executor_ratio_tol must be non-negative and finite or None, got {self.executor_ratio_tol!r}."
            )
        if not (math.isfinite(self.qfuerza_replace_with) and self.qfuerza_replace_with > 0.0):
            raise ValueError(
                f"RunProfile.qfuerza_replace_with must be positive and finite, got {self.qfuerza_replace_with!r}."
            )
        if not isinstance(self.seed, int):
            raise ValueError(f"RunProfile.seed must be an int, got {self.seed!r}.")
        unknown_roots = set(self.data_roots) - DATA_ROOT_KEYS
        if unknown_roots:
            raise ValueError(
                f"RunProfile.data_roots keys must be a subset of {sorted(DATA_ROOT_KEYS)}; got {sorted(unknown_roots)}."
            )
        for k, v in self.data_roots.items():
            if not isinstance(v, str) or not v:
                raise ValueError(f"RunProfile.data_roots[{k!r}] must be a non-empty path string, got {v!r}.")
        object.__setattr__(self, "data_roots", deep_freeze(dict(self.data_roots)))

    @property
    def optimizer_spec(self) -> OptimizerSpec:
        """The catalog entry for :attr:`optimizer`."""
        return OPTIMIZER_CATALOG[self.optimizer]

    @property
    def effective_regularization(self) -> float:
        """L2 strength actually used.

        An explicit :attr:`regularization` (including ``0.0`` to disable an
        optimizer's L2 preset) wins; otherwise the optimizer catalog's
        default applies.
        """
        return self.optimizer_spec.regularization if self.regularization is None else float(self.regularization)

    def canonical_dict(self) -> dict[str, Any]:
        """Return the identity-relevant config as a JSON-safe dict (excludes label)."""
        return {
            "system": self.system,
            "backend": self.backend,
            "functional_form": self.functional_form,
            "starting_point": self.starting_point,
            "workflow": self.workflow,
            "optimizer": self.optimizer,
            "maxiter": self.maxiter,
            "ftol": float(self.ftol),
            "fc_fraction": None if self.fc_fraction is None else float(self.fc_fraction),
            "eq_fraction": None if self.eq_fraction is None else float(self.eq_fraction),
            "regularization": None if self.regularization is None else float(self.regularization),
            "learning_rate": float(self.learning_rate),
            "max_params": int(self.max_params),
            "max_cycles": int(self.max_cycles),
            "convergence": float(self.convergence),
            "n_evals": int(self.n_evals),
            "executor_ratio_tol": None if self.executor_ratio_tol is None else float(self.executor_ratio_tol),
            "skip_optimization": bool(self.skip_optimization),
            "qfuerza_replace_with": float(self.qfuerza_replace_with),
            "platform": self.platform,
            "data_roots": {k: self.data_roots[k] for k in sorted(self.data_roots)},
            "seed": int(self.seed),
        }

    def fingerprint(self) -> str:
        """Deterministic ``sha256:<hex>`` of the requested (logical) config."""
        return canonical_fingerprint(self.canonical_dict())

    def prefix(self) -> str:
        """Readable, filesystem-safe slug of the requested config (form may be ``auto``)."""
        form = self.functional_form or "auto"
        parts = [self.system, self.backend, form, self.optimizer, self.starting_point]
        if self.skip_optimization:
            parts.append("skip")
        return _slug(*parts)

    def candidate_id(self) -> str:
        """Pre-resolution candidate ID: requested prefix + requested fingerprint suffix."""
        return f"{self.prefix()}__{_short(self.fingerprint())}"


# ---------------------------------------------------------------------------
# Environment probes (best effort, never raise)
# ---------------------------------------------------------------------------


def dependency_versions(
    packages: tuple[str, ...] = ("q2mm", "numpy", "scipy", "jax", "jaxlib", "jaxopt", "optax", "openmm"),
) -> dict[str, str]:
    """Return installed versions for *packages* (missing ones omitted).

    Only :class:`~importlib.metadata.PackageNotFoundError` is swallowed (the
    package is simply absent); any other metadata error surfaces rather than
    silently corrupting provenance.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def device_info() -> dict[str, Any]:
    """Probe JAX/OpenMM device names — best effort, never raises.

    Imports optional heavy backends, so only called at run/resolution time
    (never from the side-effect-free ``list`` CLI path).
    """
    info: dict[str, Any] = {}
    try:
        import jax

        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # pragma: no cover - environment dependent
        info["jax_devices_error"] = repr(exc)
    try:
        import openmm

        info["openmm_platforms"] = [
            openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())
        ]
    except Exception as exc:  # pragma: no cover - environment dependent
        info["openmm_platforms_error"] = repr(exc)
    return info


# ---------------------------------------------------------------------------
# ResolvedProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ResolvedProfile:
    """Provenance-complete record of exactly what one run executed.

    Every field is a plain scalar or a deeply-frozen mapping/tuple, so a
    resolved profile is safe to embed in a persisted candidate record.  Its
    :meth:`fingerprint` covers **all** resolved fields — including device,
    dependency versions, and complete data provenance — so different
    settings, data, or environment never collide.
    """

    profile: RunProfile
    # Complete static descriptor identity + runtime provenance key.
    static_descriptor: Mapping[str, Any]
    runtime_backend_key: str
    # Runtime backend info.
    backend_name: str
    backend_role: str
    backend_version: str
    backend_details: Mapping[str, Any]
    capabilities: tuple[str, ...]
    backend_functional_forms: tuple[str, ...]
    # Objective / optimizer / workflow.
    functional_form: str
    evaluator: str
    gradient_mode: str
    expected_result_gradient_mode: str
    fd_step: float | None
    effective_regularization: float
    optimizer_method: str
    optimizer_settings: Mapping[str, Any]
    workflow: str
    workflow_settings: Mapping[str, Any]
    # Parameter model.
    layout_fingerprint: str
    n_active_params: int
    n_full_params: int
    n_molecules: int
    # Data + environment.
    data_provenance: Mapping[str, Any]
    resolved_data_roots: Mapping[str, str]
    dependency_versions: Mapping[str, str]
    device: Mapping[str, Any]
    seed: int
    settings: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "backend_functional_forms", tuple(self.backend_functional_forms))
        for name in (
            "static_descriptor",
            "backend_details",
            "optimizer_settings",
            "workflow_settings",
            "data_provenance",
            "resolved_data_roots",
            "dependency_versions",
            "device",
            "settings",
        ):
            object.__setattr__(self, name, deep_freeze(dict(getattr(self, name))))

    def _identity(self) -> dict[str, Any]:
        return {
            "profile": self.profile.canonical_dict(),
            "static_descriptor": _jsonify(dict(self.static_descriptor)),
            "runtime_backend_key": self.runtime_backend_key,
            "backend_name": self.backend_name,
            "backend_role": self.backend_role,
            "backend_version": self.backend_version,
            "backend_details": _jsonify(dict(self.backend_details)),
            "capabilities": list(self.capabilities),
            "backend_functional_forms": list(self.backend_functional_forms),
            "functional_form": self.functional_form,
            "evaluator": self.evaluator,
            "gradient_mode": self.gradient_mode,
            "expected_result_gradient_mode": self.expected_result_gradient_mode,
            "fd_step": self.fd_step,
            "effective_regularization": self.effective_regularization,
            "optimizer_method": self.optimizer_method,
            "optimizer_settings": _jsonify(dict(self.optimizer_settings)),
            "workflow": self.workflow,
            "workflow_settings": _jsonify(dict(self.workflow_settings)),
            "layout_fingerprint": self.layout_fingerprint,
            "n_active_params": self.n_active_params,
            "n_full_params": self.n_full_params,
            "n_molecules": self.n_molecules,
            "data_provenance": _jsonify(dict(self.data_provenance)),
            "resolved_data_roots": dict(self.resolved_data_roots),
            "dependency_versions": dict(self.dependency_versions),
            "device": _jsonify(dict(self.device)),
            "seed": self.seed,
            "settings": _jsonify(dict(self.settings)),
        }

    def fingerprint(self) -> str:
        """Deterministic identity over the full exact resolved provenance."""
        return canonical_fingerprint(self._identity())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe provenance dict (adds both fingerprints)."""
        payload = self._identity()
        payload["profile_fingerprint"] = self.profile.fingerprint()
        payload["resolved_fingerprint"] = self.fingerprint()
        return payload

    def prefix(self) -> str:
        """Readable slug embedding the concrete resolved functional form."""
        p = self.profile
        parts = [p.system, p.backend, self.functional_form, p.optimizer, p.starting_point]
        if p.skip_optimization:
            parts.append("skip")
        return _slug(*parts)

    def candidate_id(self) -> str:
        """Return the resolved candidate ID: resolved prefix + resolved fingerprint suffix."""
        return f"{self.prefix()}__{_short(self.fingerprint())}"


def _static_descriptor_map(descriptor: BackendDescriptor | None, profile_backend: str) -> dict[str, Any]:
    """Build the complete, JSON-safe static descriptor identity mapping.

    ``None`` (an injected backend) yields an explicit placeholder rather than
    a silent gap, so an injected run still fingerprints deterministically and
    is distinguishable from a registry-loaded one.
    """
    from q2mm.backends.contracts import BACKEND_API_VERSION

    if descriptor is None:
        return {
            "name": profile_backend,
            "backend_api_version": BACKEND_API_VERSION,
            "factory": "<injected>",
            "probe_modules": [],
            "probe_executables": [],
            "role": "",
            "capability_ceiling": [],
            "functional_form_ceiling": [],
        }
    probe = descriptor.probe
    return {
        "name": descriptor.name,
        "backend_api_version": descriptor.backend_api_version,
        "factory": descriptor.factory,
        "probe_modules": sorted(getattr(probe, "modules", ()) or ()),
        "probe_executables": sorted(getattr(probe, "executables", ()) or ()),
        "role": descriptor.role.value,
        "capability_ceiling": sorted(cap.value for cap in descriptor.capability_ceiling),
        "functional_form_ceiling": sorted(descriptor.functional_form_ceiling),
    }


def resolve(
    profile: RunProfile,
    *,
    descriptor: BackendDescriptor | None,
    backend_info: BackendInfo,
    functional_form: str,
    evaluator: str,
    gradient_mode: str,
    expected_result_gradient_mode: str,
    fd_step: float | None,
    effective_regularization: float,
    optimizer_settings: Mapping[str, Any],
    workflow_settings: Mapping[str, Any],
    layout_fingerprint: str,
    n_active_params: int,
    n_full_params: int,
    n_molecules: int,
    data_provenance: Mapping[str, Any],
    resolved_data_roots: Mapping[str, str],
    include_device: bool = True,
) -> ResolvedProfile:
    """Build a :class:`ResolvedProfile` from a profile and loaded run state.

    *descriptor* is the static backend descriptor when the backend was loaded
    from the registry; ``None`` for an injected backend (an explicit
    ``"<injected>"`` placeholder descriptor is recorded).
    """
    prov = backend_info.provenance
    return ResolvedProfile(
        profile=profile,
        static_descriptor=_static_descriptor_map(descriptor, profile.backend),
        runtime_backend_key="" if prov is None else prov.backend,
        backend_name=backend_info.name,
        backend_role=backend_info.role.value,
        backend_version="" if prov is None else prov.version,
        backend_details={} if prov is None else prov.details,
        capabilities=tuple(sorted(cap.value for cap in backend_info.capabilities)),
        backend_functional_forms=tuple(sorted(backend_info.functional_forms)),
        functional_form=functional_form,
        evaluator=evaluator,
        gradient_mode=gradient_mode,
        expected_result_gradient_mode=expected_result_gradient_mode,
        fd_step=fd_step,
        effective_regularization=float(effective_regularization),
        optimizer_method=profile.optimizer_spec.method,
        optimizer_settings=dict(optimizer_settings),
        workflow=profile.workflow,
        workflow_settings=dict(workflow_settings),
        layout_fingerprint=layout_fingerprint,
        n_active_params=int(n_active_params),
        n_full_params=int(n_full_params),
        n_molecules=int(n_molecules),
        data_provenance=dict(data_provenance),
        resolved_data_roots=dict(resolved_data_roots),
        dependency_versions=dependency_versions(),
        device=device_info() if include_device else {},
        seed=int(profile.seed),
        settings=profile.canonical_dict(),
    )
