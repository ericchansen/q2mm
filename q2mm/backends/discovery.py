"""Internal, unstable lazy backend discovery.

This module turns *manifests* — JSON-safe mappings describing a backend — into
validated :class:`~q2mm.backends.contracts.BackendDescriptor` records, and
composes a deterministic :class:`DiscoverySnapshot` from the built-in manifests
plus any out-of-tree plugin manifests advertised on the
:data:`ENTRY_POINT_GROUP` entry-point group.

Design contract (kept deliberately narrow):

* **One validator.** Both built-in declarations and external manifests go
  through :func:`validate_manifest`.  There is no privileged, separately
  constructed path for built-ins.  Incompatible API versions and invalid
  capability/form/shape claims are classified into typed
  :class:`DiscoveryIssueKind` values *before* a
  :class:`~q2mm.backends.contracts.BackendDescriptor` is constructed.
* **Lazy, descriptor-only enumeration.**  :func:`iter_backend_entry_points`
  and provider loading import only the plugin's lightweight *descriptor*
  module (the entry-point target).  They never import or construct a backend
  implementation and never enumerate a device.  The backend factory import
  string is resolved only by an explicit
  :meth:`~q2mm.backends.contracts.BackendDescriptor.load`.
* **Failure isolation.**  A missing dependency, descriptor import error,
  incompatible API version, duplicate name, invalid descriptor/capability/form,
  or broken factory is captured as a typed :class:`DiscoveryRecord` and never
  hides a healthy built-in or a healthy external plugin.
* **Deterministic.**  Records are deep-immutable and emitted in a stable order
  regardless of entry-point iteration order or distribution naming.

.. warning::

   This is an internal, unstable API.  It is documented and shipped as
   *internal* until Milestone PR 3; it is not covered by semantic versioning,
   carries no compatibility promise, and may change without notice between
   Q2MM releases.
"""

from __future__ import annotations

import enum
import importlib.metadata as importlib_metadata
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from q2mm.backends.contracts import (
    BACKEND_API_VERSION,
    BackendDescriptor,
    BackendRole,
    Capability,
    DependencyProbe,
)

#: The single internal entry-point group out-of-tree plugins advertise on.
ENTRY_POINT_GROUP = "q2mm.backends"

#: The complete, closed set of manifest keys.  This is an internal API with **no
#: compatibility promise**: an unknown key is rejected as an invalid descriptor
#: (see :func:`validate_manifest`) rather than silently ignored, so a typo or a
#: forward-incompatible extension fails loudly instead of being misinterpreted.
#: Genuinely newer descriptors are gated first by ``backend_api_version``.
MANIFEST_KEYS = frozenset(
    {
        "backend_api_version",
        "name",
        "role",
        "capability_ceiling",
        "functional_form_ceiling",
        "factory",
        "probe",
    }
)

#: Registry-key / entry-point-name grammar: ASCII alphanumeric start, then
#: alphanumerics and ``.`` / ``_`` / ``-``.  Forbids slashes, whitespace, and
#: leading punctuation; ``..`` is additionally rejected in
#: :func:`_valid_registry_key` to block path traversal.
_REGISTRY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_registry_key(name: str) -> bool:
    """Return ``True`` if *name* is a filesystem/CLI-safe registry key."""
    return bool(_REGISTRY_KEY_RE.fullmatch(name)) and ".." not in name


def _is_dotted_module_name(value: str) -> bool:
    """Return ``True`` if *value* is a dotted sequence of Python identifiers."""
    parts = value.split(".")
    return bool(parts) and all(part.isidentifier() for part in parts)


def _valid_executable(value: str) -> bool:
    """Return ``True`` if *value* is a usable probe executable string.

    An executable is a command name or a path (path separators allowed); it must
    be non-empty, carry no surrounding or internal whitespace, and contain no NUL
    byte.  Whitespace-bearing paths are intentionally rejected so a probe string
    is never ambiguous when passed to ``shutil.which``.
    """
    return bool(value) and value == value.strip() and not any(ch.isspace() for ch in value) and "\x00" not in value


def _validate_factory(factory: object) -> str | None:
    """Validate a manifest ``factory`` import string, returning an error or None.

    The string must be resolvable by
    :meth:`~q2mm.backends.contracts.BackendDescriptor.load`, which does exactly
    one ``partition(":")`` and a single ``getattr``.  So it must contain exactly
    one colon, a dotted-identifier module path, and a single-identifier
    attribute, with no whitespace.
    """
    if not isinstance(factory, str) or not factory:
        return f"manifest 'factory' must be a non-empty 'module:attr' string; got {factory!r}."
    if any(ch.isspace() for ch in factory):
        return f"manifest 'factory' must not contain whitespace; got {factory!r}."
    if factory.count(":") != 1:
        return f"manifest 'factory' must contain exactly one ':' (module:attr); got {factory!r}."
    module_path, _, attr = factory.partition(":")
    if not _is_dotted_module_name(module_path):
        return f"manifest 'factory' module path {module_path!r} is not a valid dotted module name."
    if not attr.isidentifier():
        return (
            f"manifest 'factory' attribute {attr!r} must be a single identifier "
            "(BackendDescriptor.load resolves it with one getattr; dotted attributes are unsupported)."
        )
    return None


def _functional_form_values() -> frozenset[str]:
    """Return the set of supported functional-form strings.

    Imported lazily so that importing this module (and therefore
    :mod:`q2mm.backends.registry`) does not eagerly import the force-field
    model.  The values are the canonical
    :class:`~q2mm.models.forcefield.FunctionalForm` members.
    """
    from q2mm.models.forcefield import FunctionalForm

    return frozenset(form.value for form in FunctionalForm)


# ---------------------------------------------------------------------------
# Discovery vocabulary (internal, unstable)
# ---------------------------------------------------------------------------


class DiscoverySource(str, enum.Enum):
    """Where a discovery record originated."""

    BUILTIN = "builtin"
    ENTRY_POINT = "entry-point"
    LOAD = "load"


class DiscoveryState(str, enum.Enum):
    """Terminal state of one discovery attempt."""

    #: Validated, registered, and its cheap dependency probe passed.
    REGISTERED = "registered"
    #: Validated and registered, but its dependency probe is unhealthy.
    UNAVAILABLE = "unavailable"
    #: Recorded but not registered (an isolated failure).
    REJECTED = "rejected"
    #: A registered descriptor whose explicit load has failed at least once.
    LOAD_FAILED = "load-failed"


class DiscoveryIssueKind(str, enum.Enum):
    """Typed classification of a discovery problem.

    A healthy registration carries no issue (``issue is None``); every other
    record carries exactly one of these kinds.
    """

    MISSING_DEPENDENCY = "missing_dependency"
    DESCRIPTOR_IMPORT_ERROR = "descriptor_import_error"
    INCOMPATIBLE_API_VERSION = "incompatible_api_version"
    DUPLICATE_NAME = "duplicate_name"
    INVALID_DESCRIPTOR = "invalid_descriptor"
    INVALID_CAPABILITY_CLAIM = "invalid_capability_claim"
    INVALID_FUNCTIONAL_FORM_CLAIM = "invalid_functional_form_claim"
    BROKEN_FACTORY = "broken_factory"


#: Deterministic sort rank per source.
_SOURCE_RANK = {
    DiscoverySource.BUILTIN: 0,
    DiscoverySource.ENTRY_POINT: 1,
    DiscoverySource.LOAD: 2,
}


@dataclass(frozen=True)
class DiscoveryRecord:
    """Immutable record of one discovery outcome (healthy or failed).

    Args:
        source: Whether the record came from a built-in manifest, an
            entry-point plugin, or an explicit load attempt.
        state: Terminal :class:`DiscoveryState` of this attempt.
        name: Descriptor name if known (else ``""``).
        distribution: Distribution/package identity that advertised the
            entry point (else ``""``).
        entry_point: Entry-point name (else ``""``).
        issue: Typed issue kind, or ``None`` for a healthy registration.
        message: Explicit human-readable message (empty only for a healthy
            ``REGISTERED`` record).

    """

    source: DiscoverySource
    state: DiscoveryState
    name: str = ""
    distribution: str = ""
    entry_point: str = ""
    issue: DiscoveryIssueKind | None = None
    message: str = ""

    def __post_init__(self) -> None:
        """Validate field types and enforce state/issue/identity invariants."""
        if not isinstance(self.source, DiscoverySource):
            raise ValueError("DiscoveryRecord.source must be a DiscoverySource.")
        if not isinstance(self.state, DiscoveryState):
            raise ValueError("DiscoveryRecord.state must be a DiscoveryState.")
        if self.issue is not None and not isinstance(self.issue, DiscoveryIssueKind):
            raise ValueError("DiscoveryRecord.issue must be a DiscoveryIssueKind or None.")
        for field_name in ("name", "distribution", "entry_point", "message"):
            if not isinstance(getattr(self, field_name), str):
                raise ValueError(f"DiscoveryRecord.{field_name} must be a string.")
        # A healthy registration carries no issue and no message; every other
        # terminal state carries a typed issue and an explicit message.
        if self.state is DiscoveryState.REGISTERED:
            if self.issue is not None or self.message:
                raise ValueError("DiscoveryRecord REGISTERED must have no issue and an empty message.")
        else:
            if self.issue is None or not self.message:
                raise ValueError(
                    f"DiscoveryRecord {self.state.value} must carry a typed issue and a non-empty message."
                )
        # Registered/unavailable/load-failed records must identify their
        # descriptor by name; only a REJECTED record may lack one (validation
        # can fail before a name is known, or global enumeration can fail).
        if self.state is not DiscoveryState.REJECTED and not self.name:
            raise ValueError(f"DiscoveryRecord {self.state.value} must name its descriptor.")
        # Built-ins have no entry point or distribution and always have a name.
        if self.source is DiscoverySource.BUILTIN and (self.entry_point or self.distribution or not self.name):
            raise ValueError("DiscoveryRecord BUILTIN must have a name and no entry-point/distribution identity.")

    @property
    def registered(self) -> bool:
        """Return ``True`` if this record's descriptor is in the registry.

        A descriptor stays registered after an explicit load failure, so
        ``LOAD_FAILED`` counts as registered too; only ``REJECTED`` (an isolated
        discovery failure that never entered the registry) does not.
        """
        return self.state in (
            DiscoveryState.REGISTERED,
            DiscoveryState.UNAVAILABLE,
            DiscoveryState.LOAD_FAILED,
        )

    @property
    def sort_key(self) -> tuple[int, str, str, str, str, str]:
        """Deterministic ordering key independent of enumeration order."""
        return (
            _SOURCE_RANK[self.source],
            self.name,
            self.distribution,
            self.entry_point,
            self.issue.value if self.issue is not None else "",
            self.state.value,
        )


@dataclass(frozen=True)
class ManifestFailure:
    """A classified manifest-validation failure (not a descriptor)."""

    kind: DiscoveryIssueKind
    message: str

    def __post_init__(self) -> None:
        """Validate the failure carries a typed kind and a non-empty message."""
        if not isinstance(self.kind, DiscoveryIssueKind):
            raise ValueError("ManifestFailure.kind must be a DiscoveryIssueKind.")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("ManifestFailure.message must be a non-empty string.")


@dataclass(frozen=True, eq=False)
class DiscoverySnapshot:
    """Immutable result of one discovery pass.

    Args:
        descriptors: Registered descriptors keyed by name (read-only proxy).
        records: All discovery records in deterministic order.

    """

    descriptors: Mapping[str, BackendDescriptor]
    records: tuple[DiscoveryRecord, ...]

    def __post_init__(self) -> None:
        """Validate contents and freeze into read-only, deterministically-ordered form."""
        frozen: dict[str, BackendDescriptor] = {}
        for key, descriptor in dict(self.descriptors).items():
            if not isinstance(descriptor, BackendDescriptor):
                raise ValueError(f"DiscoverySnapshot descriptor for {key!r} must be a BackendDescriptor.")
            if key != descriptor.name:
                raise ValueError(f"DiscoverySnapshot key {key!r} must equal descriptor name {descriptor.name!r}.")
            frozen[key] = descriptor
        records = tuple(self.records)
        if not all(isinstance(record, DiscoveryRecord) for record in records):
            raise ValueError("DiscoverySnapshot.records must all be DiscoveryRecord instances.")
        object.__setattr__(self, "descriptors", MappingProxyType(frozen))
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda record: record.sort_key)))


@dataclass(frozen=True)
class DiscoveryReport:
    """Read-only view over a set of discovery records.

    Args:
        records: Discovery records (normalized to a tuple in deterministic
            order on construction).

    """

    records: tuple[DiscoveryRecord, ...] = ()

    def __post_init__(self) -> None:
        """Validate and normalize records to a deterministically-ordered tuple."""
        records = tuple(self.records)
        if not all(isinstance(record, DiscoveryRecord) for record in records):
            raise ValueError("DiscoveryReport.records must all be DiscoveryRecord instances.")
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda record: record.sort_key)))

    @property
    def registered(self) -> tuple[str, ...]:
        """Names of descriptors that are registered (available or not)."""
        seen: list[str] = []
        for record in self.records:
            if record.registered and record.name and record.name not in seen:
                seen.append(record.name)
        return tuple(seen)

    @property
    def issues(self) -> tuple[DiscoveryRecord, ...]:
        """Records that carry a typed issue (rejections, unavailability, load failures)."""
        return tuple(record for record in self.records if record.issue is not None)

    def for_name(self, name: str) -> tuple[DiscoveryRecord, ...]:
        """Return every record referring to descriptor *name*."""
        return tuple(record for record in self.records if record.name == name)


# ---------------------------------------------------------------------------
# Manifest validation (single path for built-ins and plugins)
# ---------------------------------------------------------------------------


def _as_str_sequence(value: object) -> list[str] | None:
    """Coerce *value* to a list of non-empty strings, or ``None`` if invalid."""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return None
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            return None
        items.append(item)
    return items


def _validate_probe(raw: object) -> DependencyProbe | ManifestFailure:
    """Validate a manifest ``probe`` mapping into a :class:`DependencyProbe`."""
    if raw is None:
        return DependencyProbe()
    if not isinstance(raw, Mapping):
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            "manifest 'probe' must be a mapping with 'modules'/'executables' string lists.",
        )
    unknown = sorted(str(key) for key in raw if key not in ("modules", "executables"))
    if unknown:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest 'probe' has unknown key(s): {unknown}; allowed: ['executables', 'modules'].",
        )
    modules = _as_str_sequence(raw.get("modules", ()))
    if modules is None:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            "manifest 'probe.modules' must be a sequence of non-empty strings.",
        )
    for module in modules:
        if not _is_dotted_module_name(module):
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_DESCRIPTOR,
                f"manifest 'probe.modules' entry {module!r} is not a valid dotted import name.",
            )
    executables = _as_str_sequence(raw.get("executables", ()))
    if executables is None:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            "manifest 'probe.executables' must be a sequence of non-empty strings.",
        )
    for executable in executables:
        if not _valid_executable(executable):
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_DESCRIPTOR,
                f"manifest 'probe.executables' entry {executable!r} must be a non-empty command/path "
                "with no whitespace or NUL byte.",
            )
    return DependencyProbe(modules=tuple(modules), executables=tuple(executables))


def validate_manifest(manifest: object, *, entry_point_name: str | None = None) -> BackendDescriptor | ManifestFailure:
    """Validate a backend *manifest* into a :class:`BackendDescriptor`.

    This is the single validation path used for both built-in declarations and
    out-of-tree plugin manifests.  On any problem it returns a classified
    :class:`ManifestFailure` (never raises), so the caller can record a typed
    :class:`DiscoveryRecord` and keep every other descriptor intact.

    Args:
        manifest: A JSON-safe mapping with keys ``backend_api_version``,
            ``name``, ``role``, ``capability_ceiling``,
            ``functional_form_ceiling``, ``factory``, and optional ``probe``.
            Both ceilings default to empty.
        entry_point_name: When the manifest is advertised by an entry point,
            the entry-point name it must agree with (both must be non-empty).
            ``None`` for built-ins (which supply their own name).

    Returns:
        A validated :class:`BackendDescriptor`, or a :class:`ManifestFailure`
        carrying the typed :class:`DiscoveryIssueKind` and an explicit message.

    """
    if not isinstance(manifest, Mapping):
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest must be a mapping; got {type(manifest).__name__}.",
        )

    # 1. API-version gate first, so an incompatible plugin is classified before
    #    any other claim is inspected.
    backend_api_version = manifest.get("backend_api_version")
    if not isinstance(backend_api_version, int) or isinstance(backend_api_version, bool):
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest 'backend_api_version' must be an int; got {backend_api_version!r}.",
        )
    if backend_api_version != BACKEND_API_VERSION:
        return ManifestFailure(
            DiscoveryIssueKind.INCOMPATIBLE_API_VERSION,
            f"manifest targets backend_api_version {backend_api_version}, this runtime is {BACKEND_API_VERSION}.",
        )

    # 1b. Unknown keys are rejected (this is an internal API with no
    #     compatibility promise; a genuinely newer descriptor is caught by the
    #     api_version gate above, so a leftover unknown key here is a mistake).
    unknown_keys = sorted(str(key) for key in manifest if key not in MANIFEST_KEYS)
    if unknown_keys:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest has unknown key(s): {unknown_keys}; allowed keys: {sorted(MANIFEST_KEYS)}.",
        )

    # 2. Name (non-empty, filesystem/CLI-safe, agreeing with the entry-point
    #    name when present).
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest 'name' must be a non-empty string; got {name!r}.",
        )
    if not _valid_registry_key(name):
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest 'name' {name!r} is not a valid registry key "
            "(ASCII alphanumeric start, then alphanumerics and '.'/'_'/'-'; no whitespace, "
            "slashes, or '..').",
        )
    if entry_point_name is not None:
        if not entry_point_name:
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_DESCRIPTOR,
                "entry-point name must be non-empty.",
            )
        if name != entry_point_name:
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_DESCRIPTOR,
                f"manifest 'name' {name!r} disagrees with entry-point name {entry_point_name!r}.",
            )

    # 3. Role.
    role_raw = manifest.get("role")
    try:
        role = BackendRole(role_raw)
    except ValueError:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_DESCRIPTOR,
            f"manifest 'role' must be one of {[r.value for r in BackendRole]}; got {role_raw!r}.",
        )

    # 4. Capabilities (empty by default; every value must be a Capability).
    capability_strings = _as_str_sequence(manifest.get("capability_ceiling", ()))
    if capability_strings is None:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM,
            "manifest 'capability_ceiling' must be a sequence of non-empty strings.",
        )
    capabilities: set[Capability] = set()
    valid_capabilities = {c.value for c in Capability}
    for value in capability_strings:
        if value not in valid_capabilities:
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM,
                f"manifest declares unknown capability {value!r}; valid capabilities: {sorted(valid_capabilities)}.",
            )
        capabilities.add(Capability(value))
    if role is BackendRole.MM and Capability.COORDINATE_GRADIENT in capabilities:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM,
            "manifest declares coordinate_gradient for an MM backend; coordinate gradients are reference-only.",
        )

    # 5. Functional forms (restricted to supported values; reference must be empty).
    form_strings = _as_str_sequence(manifest.get("functional_form_ceiling", ()))
    if form_strings is None:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM,
            "manifest 'functional_form_ceiling' must be a sequence of non-empty strings.",
        )
    valid_forms = _functional_form_values()
    for value in form_strings:
        if value not in valid_forms:
            return ManifestFailure(
                DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM,
                f"manifest declares unsupported functional form {value!r}; valid forms: {sorted(valid_forms)}.",
            )
    if role is BackendRole.REFERENCE and form_strings:
        return ManifestFailure(
            DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM,
            "manifest declares a functional-form ceiling for a reference backend (reference must declare none).",
        )

    # 6. Factory import string (module:attr, resolvable by BackendDescriptor.load).
    factory = manifest.get("factory")
    factory_error = _validate_factory(factory)
    if factory_error is not None:
        return ManifestFailure(DiscoveryIssueKind.INVALID_DESCRIPTOR, factory_error)
    assert isinstance(factory, str)  # narrowed by _validate_factory

    # 7. Probe.
    probe = _validate_probe(manifest.get("probe"))
    if isinstance(probe, ManifestFailure):
        return probe

    # 8. Construct the descriptor.
    try:
        return BackendDescriptor(
            name=name,
            role=role,
            capability_ceiling=frozenset(capabilities),
            functional_form_ceiling=frozenset(form_strings),
            factory=factory,
            probe=probe,
            backend_api_version=backend_api_version,
        )
    except ValueError as exc:  # defensive: any residual invariant violation
        return ManifestFailure(DiscoveryIssueKind.INVALID_DESCRIPTOR, f"invalid manifest: {exc}")


# ---------------------------------------------------------------------------
# Entry-point enumeration (lazy, descriptor-only)
# ---------------------------------------------------------------------------


@runtime_checkable
class EntryPointLike(Protocol):
    """Structural protocol for the entry-point objects discovery consumes."""

    name: str
    value: str

    def load(self) -> object:
        """Import and return the entry-point target (a manifest or provider)."""
        ...


def _distribution_name(entry_point: object) -> str:
    """Best-effort distribution name for *entry_point* across Python versions."""
    dist = getattr(entry_point, "dist", None)
    if dist is None:
        return ""
    name = getattr(dist, "name", None)
    if isinstance(name, str) and name:
        return name
    metadata = getattr(dist, "metadata", None)
    if metadata is not None:
        try:
            got = metadata["Name"]
        except Exception:  # noqa: BLE001 - metadata access differs across versions
            got = None
        if isinstance(got, str) and got:
            return got
    return ""


def iter_backend_entry_points(group: str = ENTRY_POINT_GROUP) -> list[EntryPointLike]:
    """Return backend entry points for *group* in deterministic order.

    Uses :func:`importlib.metadata.entry_points` in a way that is correct on
    Python 3.10 through 3.13 (``select`` when available, ``get`` otherwise) and
    sorts the result by ``(distribution, name, value)`` so downstream discovery
    is independent of installation order.  This does **not** load any target
    module; it only reads distribution metadata.
    """
    entry_points = importlib_metadata.entry_points()
    selected: Iterable[object]
    if hasattr(entry_points, "select"):
        selected = entry_points.select(group=group)
    else:  # pragma: no cover - Python < 3.10 fallback
        selected = entry_points.get(group, [])  # type: ignore[attr-defined]
    ordered = sorted(
        selected,
        key=lambda ep: (
            _distribution_name(ep),
            getattr(ep, "name", ""),
            getattr(ep, "value", ""),
        ),
    )
    return [ep for ep in ordered if isinstance(ep, EntryPointLike)]


def _load_manifest(entry_point: EntryPointLike) -> object | ManifestFailure:
    """Load an entry point's manifest (importing only the descriptor module)."""
    try:
        target = entry_point.load()
    except Exception as exc:  # noqa: BLE001 - any import/attr failure is isolated
        return ManifestFailure(
            DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR,
            f"descriptor entry point could not be imported: {exc}",
        )
    if callable(target) and not isinstance(target, Mapping):
        try:
            target = target()
        except Exception as exc:  # noqa: BLE001 - provider callable failure is isolated
            return ManifestFailure(
                DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR,
                f"descriptor provider raised: {exc}",
            )
    return target


# ---------------------------------------------------------------------------
# Snapshot composition (built-ins + entry points, with failure isolation)
# ---------------------------------------------------------------------------


def _probe_record(
    descriptor: BackendDescriptor, *, source: DiscoverySource, distribution: str, entry_point: str
) -> DiscoveryRecord:
    """Build the REGISTERED/UNAVAILABLE record for a registered descriptor."""
    healthy, reason = descriptor.is_available()
    if healthy:
        return DiscoveryRecord(
            source=source,
            state=DiscoveryState.REGISTERED,
            name=descriptor.name,
            distribution=distribution,
            entry_point=entry_point,
        )
    return DiscoveryRecord(
        source=source,
        state=DiscoveryState.UNAVAILABLE,
        name=descriptor.name,
        distribution=distribution,
        entry_point=entry_point,
        issue=DiscoveryIssueKind.MISSING_DEPENDENCY,
        message=reason or "dependency probe reported the backend unavailable",
    )


def build_snapshot(
    builtin_manifests: Sequence[Mapping[str, object]],
    *,
    entry_points: Iterable[EntryPointLike] | None = None,
) -> DiscoverySnapshot:
    """Compose a deterministic discovery snapshot.

    Built-in manifests are validated through :func:`validate_manifest` and
    always registered first (their names win every conflict).  Entry-point
    plugins are then validated; each failure is isolated into a typed record.
    If two external plugins claim the same name, **every** claimant for that
    name is rejected (deterministic, none registered) rather than letting an
    arbitrary one win.

    Args:
        builtin_manifests: The in-tree built-in backend manifests.  A malformed
            built-in manifest is a programming error and raises.
        entry_points: Injected entry points (for tests).  When ``None``, the
            real :func:`iter_backend_entry_points` enumeration is used.

    Returns:
        An immutable :class:`DiscoverySnapshot`.

    """
    registered: dict[str, BackendDescriptor] = {}
    records: list[DiscoveryRecord] = []

    # --- Built-ins (validated through the same path; must be well-formed) ---
    for manifest in builtin_manifests:
        result = validate_manifest(manifest, entry_point_name=None)
        if isinstance(result, ManifestFailure):
            raise RuntimeError(f"Built-in backend manifest is invalid ({result.kind.value}): {result.message}")
        if result.name in registered:
            raise RuntimeError(f"Duplicate built-in backend manifest name {result.name!r}.")
        registered[result.name] = result
        records.append(_probe_record(result, source=DiscoverySource.BUILTIN, distribution="", entry_point=""))

    builtin_names = set(registered)

    # --- Entry-point plugins (validated; failures isolated) ---
    #     The *global* enumeration itself can fail on corrupt distribution
    #     metadata; isolate that so the built-ins above always survive.
    resolved_entry_points: list[EntryPointLike] = []
    try:
        source_iter = iter_backend_entry_points() if entry_points is None else entry_points
        resolved_entry_points = list(source_iter)
    except Exception as exc:  # noqa: BLE001 - enumeration failure must not hide built-ins
        records.append(
            DiscoveryRecord(
                source=DiscoverySource.ENTRY_POINT,
                state=DiscoveryState.REJECTED,
                issue=DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR,
                message=f"backend entry-point enumeration failed: {exc}",
            )
        )
        resolved_entry_points = []

    valid_external: list[tuple[EntryPointLike, str, BackendDescriptor]] = []
    for entry_point in resolved_entry_points:
        distribution = _distribution_name(entry_point)
        ep_name = getattr(entry_point, "name", "")
        # Preserve the descriptor name when the entry-point name is itself a
        # valid registry key, so discovery_report().for_name(...) can find a
        # rejected plugin.  Unsafe/empty entry-point names stay "".
        known_name = ep_name if isinstance(ep_name, str) and _valid_registry_key(ep_name) else ""
        manifest_or_failure = _load_manifest(entry_point)
        if isinstance(manifest_or_failure, ManifestFailure):
            records.append(
                DiscoveryRecord(
                    source=DiscoverySource.ENTRY_POINT,
                    state=DiscoveryState.REJECTED,
                    name=known_name,
                    distribution=distribution,
                    entry_point=ep_name,
                    issue=manifest_or_failure.kind,
                    message=manifest_or_failure.message,
                )
            )
            continue
        result = validate_manifest(manifest_or_failure, entry_point_name=ep_name)
        if isinstance(result, ManifestFailure):
            records.append(
                DiscoveryRecord(
                    source=DiscoverySource.ENTRY_POINT,
                    state=DiscoveryState.REJECTED,
                    name=known_name,
                    distribution=distribution,
                    entry_point=ep_name,
                    issue=result.kind,
                    message=result.message,
                )
            )
            continue
        valid_external.append((entry_point, distribution, result))

    # Reject every external claimant that collides with a built-in or with
    # another external plugin of the same name (deterministic, none registered).
    external_name_counts = Counter(descriptor.name for _, _, descriptor in valid_external)
    for entry_point, distribution, descriptor in valid_external:
        ep_name = getattr(entry_point, "name", "")
        if descriptor.name in builtin_names:
            records.append(
                DiscoveryRecord(
                    source=DiscoverySource.ENTRY_POINT,
                    state=DiscoveryState.REJECTED,
                    name=descriptor.name,
                    distribution=distribution,
                    entry_point=ep_name,
                    issue=DiscoveryIssueKind.DUPLICATE_NAME,
                    message=f"name {descriptor.name!r} is a built-in backend; built-ins win.",
                )
            )
            continue
        if external_name_counts[descriptor.name] > 1:
            records.append(
                DiscoveryRecord(
                    source=DiscoverySource.ENTRY_POINT,
                    state=DiscoveryState.REJECTED,
                    name=descriptor.name,
                    distribution=distribution,
                    entry_point=ep_name,
                    issue=DiscoveryIssueKind.DUPLICATE_NAME,
                    message=(
                        f"name {descriptor.name!r} is claimed by "
                        f"{external_name_counts[descriptor.name]} external plugins; all rejected."
                    ),
                )
            )
            continue
        registered[descriptor.name] = descriptor
        records.append(
            _probe_record(
                descriptor,
                source=DiscoverySource.ENTRY_POINT,
                distribution=distribution,
                entry_point=ep_name,
            )
        )

    ordered = tuple(sorted(records, key=lambda record: record.sort_key))
    return DiscoverySnapshot(descriptors=registered, records=ordered)
