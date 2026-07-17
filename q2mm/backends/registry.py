"""Lazy, descriptor-based backend registry for Q2MM.

Both built-in backends and out-of-tree plugins are described by JSON-safe
*manifests* that pass through a single validator
(:func:`q2mm.backends.discovery.validate_manifest`) to produce validated
:class:`~q2mm.backends.contracts.BackendDescriptor` records.  Each descriptor
carries a **static** :class:`~q2mm.backends.contracts.BackendInfo` plus an
import-string factory and a cheap, side-effect-free dependency probe
(:class:`~q2mm.backends.contracts.DependencyProbe`).

Discovery is **lazy and cached**: importing this module does not enumerate
entry points or import any descriptor/implementation module.  The first call to
:func:`descriptors`, :func:`catalog`, :func:`registered_backends`,
:func:`available_backends`, :func:`get_descriptor`, or :func:`load_backend`
triggers one deterministic discovery snapshot (built-in manifests plus the
``q2mm.backends`` entry-point group), which is then cached.  Newly installed
plugins (or test injection) are picked up with :func:`refresh`.

Listing the catalog never constructs a backend, enumerates a device, or
initializes CUDA/XLA/OpenMM platforms — it reports each descriptor's exact
capabilities/forms from its static info plus the probe's health.  A backend is
imported and constructed only on explicit :func:`load_backend` (or
:meth:`~q2mm.backends.contracts.BackendDescriptor.load`), which does not gate on
the probe and validates the constructed backend's runtime info against the
static descriptor.  A load that fails is recorded as a typed discovery record;
the catalog then reports that descriptor as unhealthy while healthy descriptors
stay visible.

.. warning::

   This is an internal, unstable API and is documented as such until Milestone
   PR 3.  The descriptor/manifest contract, the ``q2mm.backends`` entry-point
   group, and the discovery-record vocabulary are not covered by semantic
   versioning and carry no compatibility promise; they may change without
   notice between Q2MM releases.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

from q2mm.backends.contracts import (
    BACKEND_API_VERSION,
    BackendConfigurationError,
    BackendRole,
    BackendStatus,
    BackendUnavailableError,
)
from q2mm.backends.discovery import (
    DiscoveryIssueKind,
    DiscoveryRecord,
    DiscoveryReport,
    DiscoverySnapshot,
    DiscoverySource,
    DiscoveryState,
    build_snapshot,
)

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend, BackendDescriptor


class BackendNotRegistered(BackendUnavailableError):
    """Raised when a requested backend name is not in the registry."""

    def __init__(self, name: str, *, registered: list[str] | None = None) -> None:
        self.name = name
        self.registered = registered or []
        msg = f"Backend {name!r} is not registered."
        if self.registered:
            msg += f" Registered backends: {', '.join(sorted(self.registered))}"
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Built-in manifests (JSON-safe; validated through discovery.validate_manifest,
# exactly like out-of-tree plugin manifests — no privileged construction path)
# ---------------------------------------------------------------------------
_BUILTIN_MANIFESTS: tuple[dict[str, object], ...] = (
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "openmm",
        "role": "mm",
        "capability_ceiling": [
            "energy",
            "minimize",
            "hessian",
            "frequencies",
            "parameter_gradient",
            "reusable_state",
        ],
        "functional_form_ceiling": ["harmonic", "mm3"],
        "factory": "q2mm.backends.mm.openmm:OpenMMBackend",
        "probe": {"modules": ["openmm"]},
    },
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "tinker",
        "role": "mm",
        "capability_ceiling": ["energy", "minimize", "hessian", "frequencies"],
        "functional_form_ceiling": ["mm3"],
        "factory": "q2mm.backends.mm.tinker:TinkerBackend",
        "probe": {"executables": ["analyze"]},
    },
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "jax",
        "role": "mm",
        "capability_ceiling": [
            "energy",
            "minimize",
            "hessian",
            "frequencies",
            "parameter_gradient",
            "hessian_parameter_jacobian",
            "batched_energy",
            "batched_hessian",
            "reusable_state",
        ],
        "functional_form_ceiling": ["harmonic", "mm3"],
        "factory": "q2mm.backends.mm.jax_engine:JaxBackend",
        "probe": {"modules": ["jax", "jaxlib"]},
    },
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "jax-md",
        "role": "mm",
        "capability_ceiling": [
            "energy",
            "minimize",
            "hessian",
            "frequencies",
            "parameter_gradient",
            "batched_energy",
            "reusable_state",
        ],
        "functional_form_ceiling": ["harmonic"],
        "factory": "q2mm.backends.mm.jax_md_engine:JaxMdBackend",
        "probe": {"modules": ["jax", "jaxlib", "jax_md"]},
    },
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "psi4",
        "role": "reference",
        "capability_ceiling": ["energy", "hessian", "frequencies", "geometry_optimization"],
        "functional_form_ceiling": [],
        "factory": "q2mm.backends.qm.psi4:Psi4Backend",
        "probe": {"modules": ["psi4"]},
    },
    {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "qcengine",
        "role": "reference",
        "capability_ceiling": ["energy", "coordinate_gradient", "hessian"],
        "functional_form_ceiling": [],
        "factory": "q2mm.backends.reference.qcengine:QCEngineBackend",
        "probe": {"modules": ["qcengine", "qcelemental"]},
    },
)


# ---------------------------------------------------------------------------
# Lazy, cached, thread-safe discovery snapshot
# ---------------------------------------------------------------------------
_lock = threading.RLock()
_snapshot: DiscoverySnapshot | None = None
#: Load failures discovered on explicit load, keyed by descriptor name.  This
#: overlays the (immutable) snapshot and is cleared by :func:`refresh`.
_load_records: dict[str, DiscoveryRecord] = {}


def _ensure_snapshot() -> DiscoverySnapshot:
    """Return the cached discovery snapshot, building it once on first use."""
    global _snapshot
    snapshot = _snapshot
    if snapshot is not None:
        return snapshot
    with _lock:
        if _snapshot is None:
            _snapshot = build_snapshot(_BUILTIN_MANIFESTS)
        return _snapshot


def refresh() -> None:
    """Discard the cached snapshot and load-failure overlay (internal).

    Forces the next registry access to run a fresh discovery pass.  Use after
    installing a new plugin distribution, or in tests that inject entry points.
    """
    global _snapshot
    with _lock:
        _snapshot = None
        _load_records.clear()


# ---------------------------------------------------------------------------
# Catalog / listing (side-effect free — cheap probes only)
# ---------------------------------------------------------------------------


def descriptors() -> dict[str, BackendDescriptor]:
    """Return all registered descriptors keyed by name (regardless of health)."""
    return dict(_ensure_snapshot().descriptors)


def catalog(*, role: BackendRole | None = None) -> list[BackendStatus]:
    """Report every descriptor's health via cheap probes only.

    Both healthy and unavailable descriptors are reported explicitly; nothing
    is silently omitted.  No backend is constructed and no device/platform is
    initialized.  A descriptor with a known load failure (from a previous
    explicit :func:`load_backend`) is reported unhealthy with that reason,
    while healthy descriptors remain visible.

    Args:
        role: Optional filter to MM or reference descriptors.

    Returns:
        list[BackendStatus]: One status per descriptor, sorted by name.

    """
    snapshot = _ensure_snapshot()
    with _lock:
        load_records = dict(_load_records)
    statuses: list[BackendStatus] = []
    for name in sorted(snapshot.descriptors):
        desc = snapshot.descriptors[name]
        if role is not None and desc.role is not role:
            continue
        load_failure = load_records.get(name)
        if load_failure is not None:
            statuses.append(
                BackendStatus(descriptor=desc, healthy=False, reason=f"load failed: {load_failure.message}")
            )
            continue
        healthy, reason = desc.is_available()
        statuses.append(BackendStatus(descriptor=desc, healthy=healthy, reason=reason))
    return statuses


def available_backends(*, role: BackendRole | None = None) -> list[str]:
    """Return names of backends whose cheap dependency probe passes."""
    return [status.name for status in catalog(role=role) if status.healthy]


def available_mm_backends() -> list[str]:
    """Return names of available MM backends."""
    return available_backends(role=BackendRole.MM)


def available_reference_backends() -> list[str]:
    """Return names of available reference backends."""
    return available_backends(role=BackendRole.REFERENCE)


def registered_backends(*, role: BackendRole | None = None) -> list[str]:
    """Return all registered backend names (regardless of availability)."""
    snapshot = _ensure_snapshot()
    return sorted(name for name, desc in snapshot.descriptors.items() if role is None or desc.role is role)


# ---------------------------------------------------------------------------
# Discovery records / report (internal, unstable accessors)
# ---------------------------------------------------------------------------


def discovery_records() -> tuple[DiscoveryRecord, ...]:
    """Return all discovery records, including any explicit-load failures.

    The base snapshot records (built-in + entry-point outcomes) are merged with
    the load-failure overlay and returned in a deterministic order.  This is an
    internal, unstable accessor for diagnostics and tests.
    """
    snapshot = _ensure_snapshot()
    with _lock:
        load_records = list(_load_records.values())
    records = list(snapshot.records)
    records.extend(load_records)
    return tuple(sorted(records, key=lambda record: record.sort_key))


def discovery_report() -> DiscoveryReport:
    """Return a :class:`~q2mm.backends.discovery.DiscoveryReport` over all records.

    Internal, unstable accessor: exposes registered names and typed issues
    (rejections, unavailability, load failures) for diagnostics and tests.
    """
    return DiscoveryReport(records=discovery_records())


# ---------------------------------------------------------------------------
# Loading (explicit request only — the sole path that imports a backend)
# ---------------------------------------------------------------------------


def get_descriptor(name: str) -> BackendDescriptor:
    """Return the descriptor for *name*.

    Raises:
        BackendNotRegistered: If *name* is not registered.

    """
    snapshot = _ensure_snapshot()
    try:
        return snapshot.descriptors[name]
    except KeyError:
        raise BackendNotRegistered(name, registered=list(snapshot.descriptors)) from None


def _record_load_failure(name: str, exc: BackendUnavailableError | BackendConfigurationError) -> None:
    """Record a typed load-failure discovery record for *name*."""
    if isinstance(exc, BackendConfigurationError):
        issue = DiscoveryIssueKind.BROKEN_FACTORY
    else:
        issue = DiscoveryIssueKind.MISSING_DEPENDENCY
    with _lock:
        _load_records[name] = DiscoveryRecord(
            source=DiscoverySource.LOAD,
            state=DiscoveryState.LOAD_FAILED,
            name=name,
            issue=issue,
            message=str(exc) or f"backend {name!r} load failed",
        )


def _clear_load_failure(name: str) -> None:
    """Drop any prior load-failure overlay for *name* (after a successful load)."""
    with _lock:
        _load_records.pop(name, None)


def load_backend(name: str, **kwargs: object) -> Backend:
    """Construct a registered backend by name.

    This is the only registry path that imports a backend module and constructs
    it.  It does not gate on the dependency probe (explicit configuration is
    honoured), validates the runtime info against the static descriptor, and
    returns typed :class:`~q2mm.backends.contracts.BackendUnavailableError` /
    :class:`~q2mm.backends.contracts.BackendConfigurationError` on failure.  A
    failure is recorded as a typed discovery record (a ``broken_factory`` or
    ``missing_dependency`` load record) so the catalog can report the descriptor
    unhealthy afterwards; a later **successful** load of the same name clears
    that overlay, so a single transient/invalid-config failure does not poison
    the catalog forever.

    Args:
        name: Registry key (e.g. ``"openmm"``, ``"psi4"``).
        **kwargs: Forwarded to the backend factory.

    Raises:
        BackendNotRegistered: If *name* is not registered.
        BackendUnavailableError: If the backend's dependencies are missing.
        BackendConfigurationError: If the backend is mis-configured or its
            runtime info disagrees with the descriptor.

    """
    descriptor = get_descriptor(name)
    try:
        backend = descriptor.load(**kwargs)
    except (BackendUnavailableError, BackendConfigurationError) as exc:
        _record_load_failure(name, exc)
        raise
    _clear_load_failure(name)
    return backend
