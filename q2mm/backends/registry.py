"""Descriptor-based backend registry for Q2MM.

Built-in backends are described by lightweight, validated
:class:`~q2mm.backends.contracts.BackendDescriptor` records.  Each descriptor
carries a **static** :class:`~q2mm.backends.contracts.BackendInfo` (generic
provenance) plus an import-string factory and a cheap, side-effect-free
dependency probe (:class:`~q2mm.backends.contracts.DependencyProbe`, which uses
only ``importlib.util.find_spec`` and ``shutil.which``).

Listing the catalog never constructs a backend, enumerates a device, or
initializes CUDA/XLA/OpenMM platforms — it reports each descriptor's exact
capabilities/forms from its static info plus the probe's health via
:class:`~q2mm.backends.contracts.BackendStatus`.

A backend is imported and constructed only on explicit :func:`load_backend`,
which does **not** gate on the probe (so explicit user configuration is
honoured even when a generic PATH probe is unhealthy) and validates the
constructed backend's runtime info against the static descriptor info.

.. warning::

   This is an internal, unstable API framing.  Out-of-tree plugin discovery
   (entry points) is intentionally deferred to a later phase; the descriptor
   API (versioned via
   :data:`~q2mm.backends.contracts.DESCRIPTOR_API_VERSION`) is designed to be
   extended for it without a compatibility bridge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from q2mm.backends.contracts import (
    BackendDescriptor,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendStatus,
    BackendUnavailableError,
    Capability,
    DependencyProbe,
)

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend


class BackendNotRegistered(BackendUnavailableError):
    """Raised when a requested backend name is not in the registry."""

    def __init__(self, name: str, *, registered: list[str] | None = None) -> None:
        self.name = name
        self.registered = registered or []
        msg = f"Backend {name!r} is not registered."
        if self.registered:
            msg += f" Registered backends: {', '.join(sorted(self.registered))}"
        super().__init__(msg)


def _info(name: str, role: BackendRole, capabilities: set[Capability], forms: set[str]) -> BackendInfo:
    return BackendInfo(
        name=name,
        role=role,
        capabilities=frozenset(capabilities),
        functional_forms=frozenset(forms),
        provenance=BackendProvenance(backend=name, role=role),
    )


# ---------------------------------------------------------------------------
# Built-in descriptors (static capability declarations)
# ---------------------------------------------------------------------------
_BUILTIN_DESCRIPTORS: tuple[BackendDescriptor, ...] = (
    BackendDescriptor(
        name="openmm",
        info=_info(
            "openmm",
            BackendRole.MM,
            {
                Capability.ENERGY,
                Capability.MINIMIZE,
                Capability.HESSIAN,
                Capability.FREQUENCIES,
                Capability.PARAMETER_GRADIENT,
                Capability.REUSABLE_STATE,
            },
            {"harmonic", "mm3"},
        ),
        factory="q2mm.backends.mm.openmm:OpenMMBackend",
        probe=DependencyProbe(modules=("openmm",)),
    ),
    BackendDescriptor(
        name="tinker",
        info=_info(
            "tinker",
            BackendRole.MM,
            {Capability.ENERGY, Capability.MINIMIZE, Capability.HESSIAN, Capability.FREQUENCIES},
            {"mm3"},
        ),
        factory="q2mm.backends.mm.tinker:TinkerBackend",
        probe=DependencyProbe(executables=("analyze",)),
    ),
    BackendDescriptor(
        name="jax",
        info=_info(
            "jax",
            BackendRole.MM,
            {
                Capability.ENERGY,
                Capability.MINIMIZE,
                Capability.HESSIAN,
                Capability.FREQUENCIES,
                Capability.PARAMETER_GRADIENT,
                Capability.HESSIAN_PARAMETER_JACOBIAN,
                Capability.BATCHED_ENERGY,
                Capability.BATCHED_HESSIAN,
                Capability.REUSABLE_STATE,
            },
            {"harmonic", "mm3"},
        ),
        factory="q2mm.backends.mm.jax_engine:JaxBackend",
        probe=DependencyProbe(modules=("jax", "jaxlib")),
    ),
    BackendDescriptor(
        name="jax-md",
        info=_info(
            "jax-md",
            BackendRole.MM,
            {
                Capability.ENERGY,
                Capability.MINIMIZE,
                Capability.HESSIAN,
                Capability.FREQUENCIES,
                Capability.PARAMETER_GRADIENT,
                Capability.BATCHED_ENERGY,
                Capability.REUSABLE_STATE,
            },
            {"harmonic"},
        ),
        factory="q2mm.backends.mm.jax_md_engine:JaxMdBackend",
        probe=DependencyProbe(modules=("jax", "jaxlib", "jax_md")),
    ),
    BackendDescriptor(
        name="psi4",
        info=_info(
            "psi4",
            BackendRole.QM,
            {
                Capability.ENERGY,
                Capability.HESSIAN,
                Capability.FREQUENCIES,
                Capability.GEOMETRY_OPTIMIZATION,
            },
            set(),
        ),
        factory="q2mm.backends.qm.psi4:Psi4Backend",
        probe=DependencyProbe(modules=("psi4",)),
    ),
)

_DESCRIPTORS: dict[str, BackendDescriptor] = {}
for _desc in _BUILTIN_DESCRIPTORS:
    if _desc.name in _DESCRIPTORS:
        raise ValueError(f"Duplicate built-in backend descriptor {_desc.name!r}.")
    _DESCRIPTORS[_desc.name] = _desc


# ---------------------------------------------------------------------------
# Catalog / listing (side-effect free)
# ---------------------------------------------------------------------------


def descriptors() -> dict[str, BackendDescriptor]:
    """Return all registered descriptors keyed by name (regardless of health)."""
    return dict(_DESCRIPTORS)


def catalog(*, role: BackendRole | None = None) -> list[BackendStatus]:
    """Report every descriptor's health via cheap probes only.

    Both healthy and unavailable descriptors are reported explicitly; nothing
    is silently omitted.  No backend is constructed and no device/platform is
    initialized.  Exact capabilities/forms are available from each status's
    static :attr:`~q2mm.backends.contracts.BackendStatus.info`.

    Args:
        role: Optional filter to MM or QM descriptors.

    Returns:
        list[BackendStatus]: One status per descriptor, sorted by name.

    """
    statuses: list[BackendStatus] = []
    for name in sorted(_DESCRIPTORS):
        desc = _DESCRIPTORS[name]
        if role is not None and desc.role is not role:
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


def available_qm_backends() -> list[str]:
    """Return names of available QM backends."""
    return available_backends(role=BackendRole.QM)


def registered_backends(*, role: BackendRole | None = None) -> list[str]:
    """Return all registered backend names (regardless of availability)."""
    return sorted(name for name, desc in _DESCRIPTORS.items() if role is None or desc.role is role)


# ---------------------------------------------------------------------------
# Loading (explicit request only)
# ---------------------------------------------------------------------------


def get_descriptor(name: str) -> BackendDescriptor:
    """Return the descriptor for *name*.

    Raises:
        BackendNotRegistered: If *name* is not registered.

    """
    try:
        return _DESCRIPTORS[name]
    except KeyError:
        raise BackendNotRegistered(name, registered=list(_DESCRIPTORS)) from None


def load_backend(name: str, **kwargs: object) -> Backend:
    """Construct a registered backend by name.

    This is the only path that imports a backend module and constructs it.  It
    does not gate on the dependency probe (explicit configuration is honoured),
    validates the runtime info against the static descriptor, and returns typed
    :class:`~q2mm.backends.contracts.BackendUnavailableError` /
    :class:`~q2mm.backends.contracts.BackendConfigurationError` on failure.

    Args:
        name: Registry key (e.g. ``"openmm"``, ``"psi4"``).
        **kwargs: Forwarded to the backend factory.

    Raises:
        BackendNotRegistered: If *name* is not registered.
        BackendUnavailableError: If the backend's dependencies are missing.
        BackendConfigurationError: If the backend is mis-configured or its
            runtime info disagrees with the descriptor.

    """
    return get_descriptor(name).load(**kwargs)
