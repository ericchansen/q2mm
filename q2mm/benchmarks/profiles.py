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

Profile-level runtime adapters construct backends/optimizers/workflows and
resolve loader arguments through the existing catalog and data-root helpers.
They are called by the coordinator, never during module import.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from q2mm._canonical import canonical_fingerprint as _canonical_fingerprint
from q2mm._canonical import canonical_json as _canonical_json
from q2mm._canonical import json_value
from q2mm.benchmarks.publications import (
    KNOWN_OBJECTIVE_PROFILES,
    PUBLICATION_SYSTEM_KEYS,
    REPOSITORY_OBJECTIVE_PROFILE,
)
from q2mm.models.results import deep_freeze
from q2mm.optimizers.catalog import optimizer_method, optimizer_spec
from q2mm.settings import (
    Key,
    ObjectiveSettings,
    OptimizationSettings,
    OptimizerSettings,
    PathOptions,
    Positive,
    ScipySettings,
    Settings,
    TransitionStateOptimizationSettings,
)

if TYPE_CHECKING:
    from q2mm.backends.contracts import BackendDescriptor, BackendInfo

__all__ = [
    "FUNCTIONAL_FORMS",
    "DATA_ROOT_KEYS",
    "DATA_DIR_FOR_SYSTEM",
    "KNOWN_OBJECTIVE_PROFILES",
    "ConfigurationError",
    "RunProfile",
    "QFuerzaPublicationProfile",
    "HeckPublicationProfile",
    "ResolvedProfile",
    "canonical_json",
    "canonical_fingerprint",
    "dependency_versions",
    "device_info",
    "resolve",
    "recommended_publication_profile",
    "catalog_settings",
    "BenchmarkSettings",
    "BenchmarkSelection",
    "PROFILE_VERSION",
]


# ---------------------------------------------------------------------------
# Deterministic serialization / fingerprint
# ---------------------------------------------------------------------------


def _jsonify(value: Any) -> Any:
    """Render the benchmark's legacy permissive canonical JSON value."""
    return json_value(value, strict=False, stringify_unknown=True, coerce_keys=True)


def canonical_json(payload: Any) -> str:
    """Serialise *payload* to canonical, cross-process-deterministic JSON.

    ASCII output, sorted object keys, fixed separators, ``allow_nan=False``
    — identical bytes across processes and ``PYTHONHASHSEED`` values.
    """
    return _canonical_json(payload, strict=False, stringify_unknown=True, coerce_keys=True)


def canonical_fingerprint(payload: Any) -> str:
    """Return ``sha256:<hex>`` over :func:`canonical_json` of *payload*."""
    return _canonical_fingerprint(payload, strict=False, stringify_unknown=True, coerce_keys=True)


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

#: Supported functional forms benchmarked by default.
FUNCTIONAL_FORMS: tuple[str, ...] = ("harmonic", "mm3")

#: External data-root keys a profile may configure.  ``ch3f`` maps to the
#: packaged-resource ``data_dir`` override; the others map to
#: :class:`~q2mm.benchmarks.systems._paths.ExternalDataRoots` fields.
DATA_ROOT_KEYS: frozenset[str] = frozenset({"ch3f", "rh_enamide", "supporting_info", "mm3_base"})

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


# ---------------------------------------------------------------------------
# RunProfile
# ---------------------------------------------------------------------------


PROFILE_VERSION = "benchmark-profile-v2"


class BenchmarkSettings(OptimizationSettings):
    """Benchmark execution policy, using the same component models as the SDK."""

    recipe_id: Key = "benchmark-v2"
    optimizer: OptimizerSettings = optimizer_spec("scipy-lbfgsb-jax").settings
    objective: ObjectiveSettings = ObjectiveSettings(executor="jax", gradient_mode="analytical")


def catalog_settings(key: str) -> BenchmarkSettings:
    """Select actual catalog defaults, without constructing engines or solvers."""
    spec = optimizer_spec(key)
    return BenchmarkSettings(
        recipe_id=key,
        optimizer=spec.settings,
        objective=spec.objective,
    )


class BenchmarkSelection(Settings):
    """Scientific loader selections shared by single, batch, and matrix commands."""

    functional_form: Literal["harmonic", "mm3"] | None = None
    starting_point: Literal["qfuerza", "published"] = "qfuerza"
    objective_profile: str | None = None
    skip_optimization: bool = False
    qfuerza_replace_with: Positive = 1.0
    data_roots: PathOptions = Field(default_factory=dict)
    label: str = ""

    def model_post_init(self, context: Any) -> None:
        """Validate loader choices without reading data or environment sources."""
        if self.objective_profile is not None and self.objective_profile not in KNOWN_OBJECTIVE_PROFILES:
            raise ValueError(f"objective_profile must be one of {sorted(KNOWN_OBJECTIVE_PROFILES)}.")
        unknown_roots = set(self.data_roots) - DATA_ROOT_KEYS
        if unknown_roots:
            raise ValueError(
                f"data_roots keys must be a subset of {sorted(DATA_ROOT_KEYS)}; got {sorted(unknown_roots)}."
            )
        for key, value in self.data_roots.items():
            if not value.strip():
                raise ValueError(f"data_roots[{key!r}] must be a non-empty path string.")


class RunProfile(BenchmarkSelection, BenchmarkSettings):
    """One immutable benchmark request with authoritative typed execution settings.

    ``canonical_dict`` is the explicitly versioned requested identity projection,
    separate from model inspection. Existing persisted v1 candidates are read
    unchanged, not reconstructed as v2 requests or assigned new fingerprints.
    """

    system: Key

    def model_post_init(self, context: Any) -> None:
        """Validate benchmark vocabularies without constructing runtime components."""
        super().model_post_init(context)
        if self.system not in PUBLICATION_SYSTEM_KEYS and self.objective_profile is not None:
            raise ValueError("objective_profile applies only to registered publication systems.")

    @property
    def effective_objective_profile(self) -> str | None:
        """Selected publication objective profile, or ``None`` for non-publication systems."""
        if self.system not in PUBLICATION_SYSTEM_KEYS:
            return None
        return self.objective_profile or REPOSITORY_OBJECTIVE_PROFILE

    @property
    def effective_regularization(self) -> float:
        """The objective's actual L2 strength."""
        return self.objective.regularization

    @property
    def seed(self) -> int | None:
        """Stochastic solver seed, when the selected family exposes one."""
        from q2mm.settings import BasinHoppingSettings, JaxMultiStartSettings, MultiStartSettings

        if isinstance(self.optimizer, (BasinHoppingSettings, MultiStartSettings, JaxMultiStartSettings)):
            return self.optimizer.seed
        return None

    def canonical_dict(self) -> dict[str, Any]:
        """Return the identity-relevant config as a JSON-safe dict (excludes label)."""
        return {
            "version": PROFILE_VERSION,
            "system": self.system,
            "recipe_id": self.recipe_id,
            "backend": {"key": self.backend.key, "options": self.backend.factory_options()},
            "functional_form": self.functional_form,
            "starting_point": self.starting_point,
            "objective_profile": self.effective_objective_profile,
            "workflow": self.workflow.model_dump(mode="json"),
            "optimizer": self.optimizer.model_dump(mode="json"),
            "objective": {
                "executor": self.objective.executor,
                "gradient_mode": self.objective.gradient_mode,
                "fd_step": self.objective.fd_step,
                "regularization": self.objective.regularization,
            },
            "n_evals": int(self.n_evals),
            "ratio_tol": self.ratio_tol,
            "skip_optimization": bool(self.skip_optimization),
            "qfuerza_replace_with": float(self.qfuerza_replace_with),
            "data_roots": {k: self.data_roots[k] for k in sorted(self.data_roots)},
        }

    def fingerprint(self) -> str:
        """Deterministic ``sha256:<hex>`` of the requested (logical) config."""
        return canonical_fingerprint(self.canonical_dict())

    def prefix(self, form: str | None = None) -> str:
        """Readable, filesystem-safe slug of the config (form may be ``auto``)."""
        resolved_form = form or self.functional_form or "auto"
        parts = [self.system, self.backend.key, resolved_form, self.optimizer.kind, self.starting_point]
        if self.skip_optimization:
            parts.append("skip")
        return _slug(*parts)

    def candidate_id(self) -> str:
        """Pre-resolution candidate ID: requested prefix + requested fingerprint suffix."""
        return f"{self.prefix()}__{_short(self.fingerprint())}"


class QFuerzaPublicationProfile(RunProfile):
    """Publication QFUERZA policy with native partial updates to TS basin bounds."""

    optimizer: OptimizerSettings = TransitionStateOptimizationSettings().optimizer


class HeckPublicationProfile(QFuerzaPublicationProfile):
    """Heck-relay QFUERZA policy with the documented tighter force-constant bounds."""

    optimizer: OptimizerSettings = ScipySettings(ftol=1e-12, fc_fraction=0.05, eq_fraction=0.05, verbose=False)


def recommended_publication_profile(
    system: str,
    *,
    starting_point: Literal["qfuerza", "published"] = "qfuerza",
    objective_profile: str = REPOSITORY_OBJECTIVE_PROFILE,
    **overrides: Any,
) -> RunProfile:
    """Build the documented publication-run policy without changing SDK defaults.

    QFUERZA starts use the pre-flight ``ftol`` and fractional basin bounds.
    Heck's tighter force-constant fraction is deliberately visible here rather
    than inferred by the generic optimizer/application service. Partial component
    dictionaries update the selected policy's defaults through BaseSettings;
    complete typed components replace them.
    """
    if system not in PUBLICATION_SYSTEM_KEYS:
        raise ValueError(f"No canonical publication policy is defined for {system!r}.")
    if system == "ferrocene" and starting_point != "published":
        raise ValueError("Ferrocene has no QFUERZA policy; use the explicit published-start partial profile.")
    profile_type: type[RunProfile]
    if starting_point == "qfuerza":
        profile_type = HeckPublicationProfile if system == "heck-relay" else QFuerzaPublicationProfile
    elif starting_point == "published":
        profile_type = RunProfile
    else:
        raise ValueError("starting_point must be 'qfuerza' or 'published'.")
    return profile_type(system=system, starting_point=starting_point, objective_profile=objective_profile, **overrides)


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
    seed: int | None
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
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))

    def _identity(self) -> dict[str, Any]:
        publication = self.data_provenance.get("publication_metadata")
        pub_map = publication if isinstance(publication, Mapping) else {}
        pub_identity = pub_map.get("objective_profile")
        raw_profile = (
            pub_identity.get("identifier")
            if isinstance(pub_identity, Mapping) and pub_identity.get("identifier") is not None
            else self.data_provenance.get("objective_profile")
        )
        objective_profile = str(raw_profile) if raw_profile is not None else self.profile.effective_objective_profile
        reproduction_status = str(pub_map["status"]) if pub_map.get("status") is not None else None
        return {
            "profile": self.profile.canonical_dict(),
            "objective_profile": objective_profile,
            "static_descriptor": _jsonify(self.static_descriptor),
            "runtime_backend_key": self.runtime_backend_key,
            "backend_name": self.backend_name,
            "backend_role": self.backend_role,
            "backend_version": self.backend_version,
            "backend_details": _jsonify(self.backend_details),
            "capabilities": list(self.capabilities),
            "backend_functional_forms": list(self.backend_functional_forms),
            "functional_form": self.functional_form,
            "evaluator": self.evaluator,
            "gradient_mode": self.gradient_mode,
            "expected_result_gradient_mode": self.expected_result_gradient_mode,
            "fd_step": self.fd_step,
            "effective_regularization": self.effective_regularization,
            "optimizer_method": self.optimizer_method,
            "optimizer_settings": _jsonify(self.optimizer_settings),
            "workflow": self.workflow,
            "workflow_settings": _jsonify(self.workflow_settings),
            "layout_fingerprint": self.layout_fingerprint,
            "n_active_params": self.n_active_params,
            "n_full_params": self.n_full_params,
            "n_molecules": self.n_molecules,
            "data_provenance": _jsonify(self.data_provenance),
            "reproduction_status": reproduction_status,
            "publication_metadata": None if publication is None else _jsonify(publication),
            "resolved_data_roots": dict(self.resolved_data_roots),
            "dependency_versions": dict(self.dependency_versions),
            "device": _jsonify(self.device),
            "seed": self.seed,
            "settings": _jsonify(self.settings),
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
        return self.profile.prefix(self.functional_form)

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
        static_descriptor=_static_descriptor_map(descriptor, profile.backend.key),
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
        optimizer_method=optimizer_method(profile.optimizer),
        optimizer_settings=dict(optimizer_settings),
        workflow=profile.workflow.name,
        workflow_settings=dict(workflow_settings),
        layout_fingerprint=layout_fingerprint,
        n_active_params=int(n_active_params),
        n_full_params=int(n_full_params),
        n_molecules=int(n_molecules),
        data_provenance=dict(data_provenance),
        resolved_data_roots=dict(resolved_data_roots),
        dependency_versions=dependency_versions(),
        device=device_info() if include_device else {},
        seed=profile.seed,
        settings=profile.canonical_dict(),
    )


# ---------------------------------------------------------------------------
# Runtime configuration adapters
# ---------------------------------------------------------------------------


def _classify_backend(profile: RunProfile) -> tuple[Any, Any]:
    """Return ``(descriptor, backend)``.

    Explicit construction through the registry is authoritative, including
    configured executable paths when the catalog's generic probe is unhealthy.
    Actual missing dependencies propagate as
    :class:`~q2mm.backends.contracts.BackendUnavailableError` (a graceful skip);
    unknown keys and configuration/runtime-contract failures become
    :class:`ConfigurationError`. Cheap probes are for listing only.
    """
    from q2mm.backends.contracts import BackendConfigurationError
    from q2mm.backends.registry import BackendNotRegistered, get_descriptor, load_backend

    try:
        descriptor = get_descriptor(profile.backend.key)
        backend = load_backend(profile.backend.key, **profile.backend.factory_options())
    except BackendNotRegistered as exc:
        raise ConfigurationError(f"unknown backend {profile.backend.key!r}: {exc}") from exc
    except BackendConfigurationError as exc:
        raise ConfigurationError(f"backend {profile.backend.key!r} failed to construct: {exc}") from exc
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

    ext_keys = ("rh_enamide", "supporting_info", "mm3_base")
    explicit = ExternalDataRoots(**{k: _norm_path(roots[k]) for k in ext_keys if k in roots})
    # Fold explicit overrides together with the documented environment
    # fallbacks so provenance reflects exactly what the loader resolves.
    effective = resolve_external_roots(explicit)
    kwargs["data_roots"] = effective
    resolved = {k: str(_norm_path(str(val))) for k in ext_keys if (val := getattr(effective, k)) is not None}
    return kwargs, resolved


def _default_form(system: str) -> str:
    from q2mm.benchmarks.systems import system_metadata

    forms = system_metadata(system).default_forms
    return forms[0] if forms else "mm3"
