"""Discovery, isolation, and capability-conformance tests for the backend registry.

These tests never ``pip install`` anything into the developer environment.  The
out-of-tree reference package (``examples/backend-plugin``) is exposed on an
isolated ``sys.path`` and advertised either via realistic fake ``EntryPoint``
objects or a temporary ``.dist-info`` directory.  The registry's real install
smoke lives in ``scripts/check_release_artifacts.py``.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from q2mm.backends import discovery, registry
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendConfigurationError,
    BackendDescriptor,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    DependencyProbe,
    EnergyResult,
    EnergyUnit,
    HessianResult,
    HessianUnit,
    PreparationRequest,
    UnsupportedCapabilityError,
)
from q2mm.backends.conformance import (
    ConformanceError,
    MMConformanceCase,
    run_mm_conformance,
)
from q2mm.backends.discovery import (
    BACKEND_API_VERSION,
    DiscoveryIssueKind,
    DiscoveryRecord,
    DiscoveryReport,
    DiscoverySnapshot,
    DiscoverySource,
    DiscoveryState,
    ManifestFailure,
    build_snapshot,
    validate_manifest,
)
from q2mm.models.parameters import ParameterLayout
from test._shared import REPO_ROOT

FIXTURE_DIR = REPO_ROOT / "examples" / "backend-plugin"
FIXTURE_VALUE = "q2mm_reference_backend.descriptor:MANIFEST"

_UNSET = object()


# ---------------------------------------------------------------------------
# Realistic fake entry points (mirror the stdlib EntryPoint surface discovery uses)
# ---------------------------------------------------------------------------


class FakeDist:
    """Minimal distribution stub exposing ``.name`` (like ``Distribution``)."""

    def __init__(self, name: str) -> None:
        self.name = name


class FakeEntryPoint:
    """A realistic fake ``EntryPoint`` for injecting plugin manifests.

    ``load()`` mirrors the stdlib behaviour (import ``module``, resolve dotted
    ``attr``) unless a pre-built ``load_result`` or ``load_error`` is supplied,
    which lets tests drive malformed-manifest and import-failure paths without a
    real distribution.
    """

    group = discovery.ENTRY_POINT_GROUP

    def __init__(
        self,
        name: str,
        value: str = "",
        *,
        dist_name: str = "",
        load_result: Any = _UNSET,
        load_error: BaseException | None = None,
    ) -> None:
        self.name = name
        self.value = value
        self.dist = FakeDist(dist_name) if dist_name else None
        self._load_result = load_result
        self._load_error = load_error

    def load(self) -> object:
        if self._load_error is not None:
            raise self._load_error
        if self._load_result is not _UNSET:
            return self._load_result
        module_name, _, attr = self.value.partition(":")
        module = importlib.import_module(module_name)
        obj: object = module
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj


def _manifest(**overrides: object) -> dict[str, object]:
    """Return a valid manifest mapping with *overrides* applied."""
    manifest: dict[str, object] = {
        "backend_api_version": BACKEND_API_VERSION,
        "name": "extra",
        "role": "mm",
        "capability_ceiling": ["energy"],
        "functional_form_ceiling": ["harmonic"],
        "factory": "q2mm_reference_backend.backend:HarmonicReferenceBackend",
        "probe": {"modules": ["numpy"]},
    }
    manifest.update(overrides)
    return manifest


def _fixture_entry_point(
    name: str = "harmonic-reference", *, dist_name: str = "q2mm-backend-reference"
) -> FakeEntryPoint:
    """Return a fake entry point pointing at the real fixture descriptor manifest."""
    return FakeEntryPoint(name, FIXTURE_VALUE, dist_name=dist_name)


@pytest.fixture
def fixture_on_path() -> Iterator[None]:
    """Expose the out-of-tree fixture package on ``sys.path`` (no pip install)."""
    path = str(FIXTURE_DIR)
    inserted = path not in sys.path
    if inserted:
        sys.path.insert(0, path)
    purged = {name: mod for name, mod in list(sys.modules.items()) if name.startswith("q2mm_reference_backend")}
    for name in purged:
        del sys.modules[name]
    try:
        yield
    finally:
        if inserted and path in sys.path:
            sys.path.remove(path)
        for name in [n for n in list(sys.modules) if n.startswith("q2mm_reference_backend")]:
            del sys.modules[name]


@pytest.fixture
def inject_entry_points(monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[list[FakeEntryPoint]], None]]:
    """Inject fake entry points into the live registry, then restore it."""

    def _inject(entry_points: list[FakeEntryPoint]) -> None:
        monkeypatch.setattr(
            discovery, "iter_backend_entry_points", lambda group=discovery.ENTRY_POINT_GROUP: list(entry_points)
        )
        registry.refresh()

    try:
        yield _inject
    finally:
        registry.refresh()


def _harmonic_case() -> tuple[Any, Any]:
    """Return a (molecule, harmonic force field) pair for conformance runs."""
    from q2mm.benchmarks.systems.ch3f import load_molecule
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.models.seminario import qfuerza_fresh

    molecule = load_molecule()
    force_field = qfuerza_fresh(molecule, functional_form=FunctionalForm.HARMONIC, invert_ts_curvature=False)
    return molecule, force_field


def _descriptor_for_backend(backend: Any, *, role: BackendRole | None = None) -> BackendDescriptor:
    """Build a static test descriptor whose ceilings cover a fake runtime."""
    info = backend.info
    descriptor_role = role or info.role
    name = info.provenance.backend
    forms = info.functional_forms if descriptor_role is BackendRole.MM else frozenset()
    return BackendDescriptor(
        name=name,
        role=descriptor_role,
        capability_ceiling=info.capabilities,
        functional_form_ceiling=forms,
        factory="test.test_backend_discovery:_ReferenceLikeBackend",
    )


# ---------------------------------------------------------------------------
# Test-double backends for the conformance helper
# ---------------------------------------------------------------------------

_RS_PROV = BackendProvenance(backend="rs-fake", role=BackendRole.MM)
_RS_INFO = BackendInfo(
    name="rs-fake",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY, Capability.REUSABLE_STATE}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_RS_PROV,
)


class _RsPrepared(AbstractPreparedBackend):
    def _energy(self, request: Any) -> EnergyResult:
        return EnergyResult(energy=1.0, unit=EnergyUnit.KCAL_PER_MOL, provenance=_RS_PROV)


class _ReusableStateBackend:
    """Declares ENERGY + REUSABLE_STATE and counts how often it prepares."""

    def __init__(self) -> None:
        self.prepared_sessions = 0

    @property
    def info(self) -> BackendInfo:
        return _RS_INFO

    def prepare(self, request: PreparationRequest) -> _RsPrepared:
        self.prepared_sessions += 1
        return _RsPrepared(
            info=_RS_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )


_SPY_PROV = BackendProvenance(backend="spy", role=BackendRole.MM)
_SPY_INFO = BackendInfo(
    name="spy",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.ENERGY}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_SPY_PROV,
)


class _SpyPrepared(AbstractPreparedBackend):
    """Records which implementation hook bodies actually run.

    Every hook (declared or not) records its name; undeclared ones additionally
    raise so the helper's undeclared-assertion still sees UnsupportedCapability.
    Because the base guard fires before dispatch, an undeclared hook must never
    be recorded — the invariant the claim-gating test relies on.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hooks_invoked: list[str] = []

    def _energy(self, request: Any) -> EnergyResult:
        self.hooks_invoked.append("_energy")
        return EnergyResult(energy=0.0, unit=EnergyUnit.KCAL_PER_MOL, provenance=_SPY_PROV)

    def _minimize(self, request: Any) -> Any:
        self.hooks_invoked.append("_minimize")
        raise UnsupportedCapabilityError(self.info.name, Capability.MINIMIZE)

    def _hessian(self, request: Any) -> Any:
        self.hooks_invoked.append("_hessian")
        raise UnsupportedCapabilityError(self.info.name, Capability.HESSIAN)

    def _frequencies(self, request: Any) -> Any:
        self.hooks_invoked.append("_frequencies")
        raise UnsupportedCapabilityError(self.info.name, Capability.FREQUENCIES)

    def _parameter_gradient(self, request: Any) -> Any:
        self.hooks_invoked.append("_parameter_gradient")
        raise UnsupportedCapabilityError(self.info.name, Capability.PARAMETER_GRADIENT)

    def _hessian_parameter_jacobian(self, request: Any) -> Any:
        self.hooks_invoked.append("_hessian_parameter_jacobian")
        raise UnsupportedCapabilityError(self.info.name, Capability.HESSIAN_PARAMETER_JACOBIAN)

    def _batched_energy(self, request: Any) -> Any:
        self.hooks_invoked.append("_batched_energy")
        raise UnsupportedCapabilityError(self.info.name, Capability.BATCHED_ENERGY)


class _HookSpyBackend:
    """Declares ENERGY only; exposes the prepared spy session for inspection."""

    def __init__(self) -> None:
        self.prepared: _SpyPrepared | None = None

    @property
    def info(self) -> BackendInfo:
        return _SPY_INFO

    def prepare(self, request: PreparationRequest) -> _SpyPrepared:
        self.prepared = _SpyPrepared(
            info=_SPY_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )
        return self.prepared


_REFERENCE_PROV = BackendProvenance(backend="reference-like", role=BackendRole.REFERENCE)
_REFERENCE_INFO = BackendInfo(
    name="reference-like",
    role=BackendRole.REFERENCE,
    capabilities=frozenset({Capability.ENERGY}),
    functional_forms=frozenset(),
    provenance=_REFERENCE_PROV,
)


class _ReferenceLikeBackend:
    """A reference-role backend the MM conformance helper must refuse."""

    @property
    def info(self) -> BackendInfo:
        return _REFERENCE_INFO

    def prepare(self, request: PreparationRequest) -> Any:  # pragma: no cover - never reached
        raise AssertionError("MM conformance helper must not prepare a reference backend")


_HR_PROV = BackendProvenance(backend="hr-fake", role=BackendRole.MM)
_HR_INFO = BackendInfo(
    name="hr-fake",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.HESSIAN, Capability.REUSABLE_STATE}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_HR_PROV,
)


class _HrPrepared(AbstractPreparedBackend):
    def _hessian(self, request: Any) -> HessianResult:
        n3 = 3 * len(self.molecule.symbols)
        return HessianResult(hessian=np.eye(n3), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=_HR_PROV)


class _HessianReuseBackend:
    """Declares HESSIAN + REUSABLE_STATE but NOT ENERGY (reuse via HESSIAN)."""

    @property
    def info(self) -> BackendInfo:
        return _HR_INFO

    def prepare(self, request: PreparationRequest) -> _HrPrepared:
        return _HrPrepared(
            info=_HR_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )


# ---------------------------------------------------------------------------
# Manifest validation (single path for built-ins and plugins)
# ---------------------------------------------------------------------------


class TestManifestValidation:
    def test_valid_manifest_produces_descriptor(self) -> None:
        result = validate_manifest(_manifest(name="ok"), entry_point_name="ok")
        assert not isinstance(result, ManifestFailure)
        assert result.name == "ok"
        assert result.role is BackendRole.MM
        assert result.capability_ceiling == frozenset({Capability.ENERGY})
        assert result.functional_form_ceiling == frozenset({"harmonic"})

    def test_builtins_use_same_validator(self) -> None:
        # Every built-in manifest must validate through the one public validator.
        for manifest in registry._BUILTIN_MANIFESTS:
            result = validate_manifest(manifest, entry_point_name=None)
            assert not isinstance(result, ManifestFailure), manifest

    def test_incompatible_api_version(self) -> None:
        result = validate_manifest(_manifest(backend_api_version=BACKEND_API_VERSION + 1))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INCOMPATIBLE_API_VERSION

    def test_non_int_api_version_is_invalid_descriptor(self) -> None:
        result = validate_manifest(_manifest(backend_api_version="1"))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_name_mismatch_with_entry_point(self) -> None:
        result = validate_manifest(_manifest(name="a"), entry_point_name="b")
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_empty_name_rejected(self) -> None:
        result = validate_manifest(_manifest(name=""))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_invalid_role(self) -> None:
        result = validate_manifest(_manifest(role="banana"))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_invalid_capability_claim(self) -> None:
        result = validate_manifest(_manifest(capability_ceiling=["energy", "fly"]))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM

    @pytest.mark.parametrize(
        ("role", "capability"),
        [
            ("mm", "coordinate_gradient"),
            ("mm", "geometry_optimization"),
            ("reference", "minimize"),
            ("reference", "parameter_gradient"),
            ("reference", "hessian_parameter_jacobian"),
            ("reference", "batched_energy"),
            ("reference", "batched_hessian"),
        ],
    )
    def test_role_incompatible_capability_claim_rejected(self, role: str, capability: str) -> None:
        result = validate_manifest(_manifest(role=role, capability_ceiling=[capability], functional_form_ceiling=[]))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM

    def test_invalid_functional_form_claim(self) -> None:
        result = validate_manifest(_manifest(functional_form_ceiling=["harmonic", "banana"]))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM

    def test_reference_backend_must_declare_no_forms(self) -> None:
        result = validate_manifest(
            _manifest(
                name="reference",
                role="reference",
                functional_form_ceiling=["harmonic"],
                capability_ceiling=["energy"],
            )
        )
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM

    def test_reference_backend_with_empty_forms_valid(self) -> None:
        result = validate_manifest(
            _manifest(
                name="reference",
                role="reference",
                functional_form_ceiling=[],
                capability_ceiling=["energy"],
                factory="numpy:zeros",
            )
        )
        assert not isinstance(result, ManifestFailure)
        assert result.role is BackendRole.REFERENCE

    def test_invalid_factory_shapes(self) -> None:
        for bad in ("nocolon", ":attr", "module:", "", 5):
            result = validate_manifest(_manifest(factory=bad))
            assert isinstance(result, ManifestFailure), bad
            assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_invalid_probe_shape(self) -> None:
        for bad in ({"modules": "openmm"}, {"modules": [""]}, {"executables": [1]}, "notamap"):
            result = validate_manifest(_manifest(probe=bad))
            assert isinstance(result, ManifestFailure), bad
            assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_non_mapping_manifest(self) -> None:
        result = validate_manifest(["not", "a", "mapping"])
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_defaults_capabilities_and_forms_empty(self) -> None:
        manifest = {
            "backend_api_version": BACKEND_API_VERSION,
            "name": "bare",
            "role": "mm",
            "factory": "numpy:zeros",
        }
        result = validate_manifest(manifest)
        assert not isinstance(result, ManifestFailure)
        assert result.capability_ceiling == frozenset()
        assert result.functional_form_ceiling == frozenset()

    # -- Hardening: unknown keys, strict grammar, strict factory/probe --------

    def test_unknown_key_rejected(self) -> None:
        result = validate_manifest(_manifest(surprise=True))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR
        assert "surprise" in result.message

    def test_non_json_safe_manifest_rejected(self) -> None:
        result = validate_manifest(_manifest(probe={"modules": [object()]}))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR
        assert "JSON-safe" in result.message

    @pytest.mark.parametrize(
        ("old_key", "value"),
        [
            ("api_version", BACKEND_API_VERSION),
            ("capabilities", ["energy"]),
            ("forms", ["harmonic"]),
        ],
    )
    def test_pre_v1_manifest_fields_are_rejected(self, old_key: str, value: object) -> None:
        result = validate_manifest(_manifest(**{old_key: value}))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR
        assert old_key in result.message

    def test_unknown_key_with_future_api_is_incompatible(self) -> None:
        # api_version gate wins so a genuinely newer descriptor is classed as
        # incompatible, not invalid-for-its-new-keys.
        result = validate_manifest(_manifest(backend_api_version=BACKEND_API_VERSION + 1, surprise=True))
        assert isinstance(result, ManifestFailure)
        assert result.kind is DiscoveryIssueKind.INCOMPATIBLE_API_VERSION

    @pytest.mark.parametrize(
        "bad_name",
        ["../evil", "a/b", "a\\b", "a b", "..", ".hidden", "-lead", "_lead", "a..b", "name\tx", "évil"],
    )
    def test_malicious_or_invalid_names_rejected(self, bad_name: str) -> None:
        result = validate_manifest(_manifest(name=bad_name))
        assert isinstance(result, ManifestFailure), bad_name
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    @pytest.mark.parametrize("good_name", ["openmm", "jax-md", "harmonic-reference", "a.b_c-1", "A0", "x"])
    def test_valid_registry_keys_accepted(self, good_name: str) -> None:
        result = validate_manifest(_manifest(name=good_name))
        assert not isinstance(result, ManifestFailure), good_name

    @pytest.mark.parametrize(
        "bad_factory",
        ["nocolon", "a:b:c", "a b:C", "mod:a.b", "a::b", ":x", "m:", "mod:1abc", "mod:", " mod:C", "mod :C"],
    )
    def test_strict_factory_rejected(self, bad_factory: str) -> None:
        result = validate_manifest(_manifest(factory=bad_factory))
        assert isinstance(result, ManifestFailure), bad_factory
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_valid_factory_accepted(self) -> None:
        result = validate_manifest(_manifest(factory="pkg.sub.mod:Attr"))
        assert not isinstance(result, ManifestFailure)

    @pytest.mark.parametrize(
        "bad_probe",
        [
            {"modules": ["a b"]},
            {"modules": ["1x"]},
            {"modules": ["a..b"]},
            {"executables": ["a b"]},
            {"executables": ["a\x00b"]},
            {"executables": [" spaced"]},
            {"weird": []},
        ],
    )
    def test_strict_probe_rejected(self, bad_probe: dict[str, object]) -> None:
        result = validate_manifest(_manifest(probe=bad_probe))
        assert isinstance(result, ManifestFailure), bad_probe
        assert result.kind is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_valid_probe_accepted(self) -> None:
        result = validate_manifest(_manifest(probe={"modules": ["a.b", "c"], "executables": ["/usr/bin/tool"]}))
        assert not isinstance(result, ManifestFailure)

    def test_builtin_manifests_use_runtime_backend_api_version(self) -> None:
        for manifest in registry._BUILTIN_MANIFESTS:
            assert manifest["backend_api_version"] == BACKEND_API_VERSION, manifest["name"]

    def test_fixture_manifest_uses_runtime_backend_api_version(self, fixture_on_path: None) -> None:
        import q2mm_reference_backend.descriptor as descriptor_module

        assert descriptor_module.MANIFEST["backend_api_version"] == BACKEND_API_VERSION
        # Importing the descriptor must not import the implementation module.
        assert "q2mm_reference_backend.backend" not in sys.modules


# ---------------------------------------------------------------------------
# Snapshot composition and failure isolation
# ---------------------------------------------------------------------------


class TestSnapshotComposition:
    def test_healthy_external_discovered(self, fixture_on_path: None) -> None:
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[_fixture_entry_point()])
        assert "harmonic-reference" in snapshot.descriptors
        desc = snapshot.descriptors["harmonic-reference"]
        assert desc.role is BackendRole.MM
        assert desc.capability_ceiling == frozenset({Capability.ENERGY})
        assert desc.functional_form_ceiling == frozenset({"harmonic"})
        record = next(r for r in snapshot.records if r.name == "harmonic-reference")
        assert record.source is DiscoverySource.ENTRY_POINT
        assert record.state is DiscoveryState.REGISTERED
        assert record.issue is None
        assert record.distribution == "q2mm-backend-reference"

    def test_callable_provider_entry_point(self) -> None:
        ep = FakeEntryPoint("provider", load_result=lambda: _manifest(name="provider"))
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "provider" in snapshot.descriptors

    def test_no_implementation_import_during_discovery(self, fixture_on_path: None) -> None:
        # Importing the descriptor manifest must not import the implementation.
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[_fixture_entry_point()])
        assert "q2mm_reference_backend.descriptor" in sys.modules
        assert "q2mm_reference_backend.backend" not in sys.modules
        # Only an explicit load imports it.
        backend = snapshot.descriptors["harmonic-reference"].load()
        assert "q2mm_reference_backend.backend" in sys.modules
        assert backend.info.provenance.backend == "harmonic-reference"

    def test_builtins_always_present_and_win_conflicts(self, fixture_on_path: None) -> None:
        # An external plugin claiming a built-in name is rejected; built-in survives.
        clash = FakeEntryPoint("openmm", load_result=_manifest(name="openmm"), dist_name="rogue")
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[clash, _fixture_entry_point()])
        assert "openmm" in snapshot.descriptors
        # The built-in openmm descriptor, not the rogue one.
        assert snapshot.descriptors["openmm"].factory == "q2mm.backends.mm.openmm:OpenMMBackend"
        rogue = next(r for r in snapshot.records if r.name == "openmm" and r.source is DiscoverySource.ENTRY_POINT)
        assert rogue.state is DiscoveryState.REJECTED
        assert rogue.issue is DiscoveryIssueKind.DUPLICATE_NAME
        # The healthy fixture is unaffected.
        assert "harmonic-reference" in snapshot.descriptors

    def test_duplicate_external_names_all_rejected(self) -> None:
        one = FakeEntryPoint("dup", load_result=_manifest(name="dup"), dist_name="dist-a")
        two = FakeEntryPoint("dup", load_result=_manifest(name="dup"), dist_name="dist-b")
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[one, two])
        assert "dup" not in snapshot.descriptors
        dup_records = [r for r in snapshot.records if r.name == "dup"]
        assert len(dup_records) == 2
        assert all(r.state is DiscoveryState.REJECTED for r in dup_records)
        assert all(r.issue is DiscoveryIssueKind.DUPLICATE_NAME for r in dup_records)

    def test_duplicate_builtin_raises(self) -> None:
        doubled = registry._BUILTIN_MANIFESTS + (dict(registry._BUILTIN_MANIFESTS[0]),)
        with pytest.raises(RuntimeError, match="Duplicate built-in"):
            build_snapshot(doubled, entry_points=[])

    def test_invalid_builtin_manifest_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Built-in backend manifest is invalid"):
            build_snapshot([_manifest(name="bad", capability_ceiling=["fly"])], entry_points=[])

    def test_entry_point_name_mismatch(self) -> None:
        ep = FakeEntryPoint("advertised", load_result=_manifest(name="declared"))
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "declared" not in snapshot.descriptors
        assert "advertised" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "advertised")
        assert record.issue is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_missing_dependency_registers_but_unavailable(self) -> None:
        ep = FakeEntryPoint(
            "dep-missing",
            load_result=_manifest(name="dep-missing", probe={"modules": ["definitely_missing_xyz_123"]}),
        )
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "dep-missing" in snapshot.descriptors  # registered ...
        record = next(r for r in snapshot.records if r.name == "dep-missing")
        assert record.state is DiscoveryState.UNAVAILABLE  # ... but unavailable
        assert record.issue is DiscoveryIssueKind.MISSING_DEPENDENCY

    def test_descriptor_import_error(self) -> None:
        ep = FakeEntryPoint(
            "boom",
            value="q2mm_reference_backend._does_not_exist:MANIFEST",
            load_error=ModuleNotFoundError("no module named q2mm_reference_backend._does_not_exist"),
        )
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "boom" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "boom")
        assert record.state is DiscoveryState.REJECTED
        assert record.issue is DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR

    def test_provider_callable_failure(self) -> None:
        def _boom() -> dict[str, object]:
            raise RuntimeError("provider exploded")

        ep = FakeEntryPoint("provider-boom", load_result=_boom)
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "provider-boom" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "provider-boom")
        assert record.issue is DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR

    def test_incompatible_api_version_isolated(self) -> None:
        ep = FakeEntryPoint(
            "old",
            load_result=_manifest(name="old", backend_api_version=BACKEND_API_VERSION + 5),
        )
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "old" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "old")
        assert record.issue is DiscoveryIssueKind.INCOMPATIBLE_API_VERSION

    def test_invalid_capability_and_form_isolated(self) -> None:
        cap_ep = FakeEntryPoint("badcap", load_result=_manifest(name="badcap", capability_ceiling=["fly"]))
        form_ep = FakeEntryPoint(
            "badform",
            load_result=_manifest(name="badform", functional_form_ceiling=["banana"]),
        )
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[cap_ep, form_ep])
        assert "badcap" not in snapshot.descriptors
        assert "badform" not in snapshot.descriptors
        cap_rec = next(r for r in snapshot.records if r.entry_point == "badcap")
        form_rec = next(r for r in snapshot.records if r.entry_point == "badform")
        assert cap_rec.issue is DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM
        assert form_rec.issue is DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM

    def test_invalid_shape_factory_manifest_isolated(self) -> None:
        factory_ep = FakeEntryPoint("badfactory", load_result=_manifest(name="badfactory", factory="nocolon"))
        shape_ep = FakeEntryPoint("badshape", load_result="not-a-mapping")
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[factory_ep, shape_ep])
        assert "badfactory" not in snapshot.descriptors
        assert "badshape" not in snapshot.descriptors
        for name in ("badfactory", "badshape"):
            record = next(r for r in snapshot.records if r.entry_point == name)
            assert record.issue is DiscoveryIssueKind.INVALID_DESCRIPTOR

    def test_deterministic_record_order(self, fixture_on_path: None) -> None:
        eps_forward = [
            _fixture_entry_point(),
            FakeEntryPoint("zzz", load_result=_manifest(name="zzz", probe={"modules": ["missing_zzz"]})),
            FakeEntryPoint("badcap", load_result=_manifest(name="badcap", capability_ceiling=["fly"])),
        ]
        eps_reversed = list(reversed(eps_forward))
        snapshot_a = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=eps_forward)
        snapshot_b = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=eps_reversed)
        assert snapshot_a.records == snapshot_b.records

    def test_healthy_isolation_matrix(self, fixture_on_path: None) -> None:
        # One healthy external survives a barrage of broken plugins.
        eps = [
            _fixture_entry_point(),
            FakeEntryPoint("import-fail", load_error=ImportError("nope")),
            FakeEntryPoint("api-fail", load_result=_manifest(name="api-fail", backend_api_version=99)),
            FakeEntryPoint("cap-fail", load_result=_manifest(name="cap-fail", capability_ceiling=["fly"])),
            FakeEntryPoint(
                "form-fail",
                load_result=_manifest(name="form-fail", functional_form_ceiling=["banana"]),
            ),
            FakeEntryPoint("shape-fail", load_result=_manifest(name="shape-fail", factory="bad")),
            FakeEntryPoint("dep-fail", load_result=_manifest(name="dep-fail", probe={"modules": ["missing_qqq"]})),
            FakeEntryPoint("dupe", load_result=_manifest(name="dupe")),
            FakeEntryPoint("dupe", load_result=_manifest(name="dupe"), dist_name="other"),
            FakeEntryPoint("openmm", load_result=_manifest(name="openmm")),
        ]
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=eps)
        # Healthy external available; all built-ins registered; dep-fail registered-unavailable.
        assert "harmonic-reference" in snapshot.descriptors
        assert {"openmm", "tinker", "jax", "jax-md", "psi4"} <= set(snapshot.descriptors)
        assert snapshot.descriptors["openmm"].factory == "q2mm.backends.mm.openmm:OpenMMBackend"
        assert "dep-fail" in snapshot.descriptors  # registered but unhealthy
        for rejected in ("import-fail", "api-fail", "cap-fail", "form-fail", "shape-fail", "dupe"):
            assert rejected not in snapshot.descriptors, rejected
        issues = {(r.entry_point, r.issue) for r in snapshot.records if r.issue is not None}
        assert ("import-fail", DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR) in issues
        assert ("api-fail", DiscoveryIssueKind.INCOMPATIBLE_API_VERSION) in issues
        assert ("cap-fail", DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM) in issues
        assert ("form-fail", DiscoveryIssueKind.INVALID_FUNCTIONAL_FORM_CLAIM) in issues
        assert ("shape-fail", DiscoveryIssueKind.INVALID_DESCRIPTOR) in issues
        assert ("dep-fail", DiscoveryIssueKind.MISSING_DEPENDENCY) in issues

    # -- Hardening: unknown-key / malicious-name / enumeration / probe crash --

    def test_unknown_key_isolated(self) -> None:
        ep = FakeEntryPoint("weird", load_result=_manifest(name="weird", surprise=True))
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "weird" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "weird")
        assert record.issue is DiscoveryIssueKind.INVALID_DESCRIPTOR
        assert {"openmm", "psi4"} <= set(snapshot.descriptors)

    def test_malicious_name_isolated(self) -> None:
        ep = FakeEntryPoint("../evil", load_result=_manifest(name="../evil"))
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "../evil" not in snapshot.descriptors
        record = next(r for r in snapshot.records if r.entry_point == "../evil")
        assert record.issue is DiscoveryIssueKind.INVALID_DESCRIPTOR
        assert {"openmm", "psi4"} <= set(snapshot.descriptors)

    def test_dotted_missing_parent_probe_isolated(self) -> None:
        # A probe whose module has a missing parent must not crash discovery;
        # the descriptor registers but is unavailable.
        ep = FakeEntryPoint(
            "dotted-dep",
            load_result=_manifest(name="dotted-dep", probe={"modules": ["missing_parent_xyz.child"]}),
        )
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        assert "dotted-dep" in snapshot.descriptors
        record = next(r for r in snapshot.records if r.name == "dotted-dep")
        assert record.state is DiscoveryState.UNAVAILABLE
        assert record.issue is DiscoveryIssueKind.MISSING_DEPENDENCY

    def test_dependency_probe_isolates_spec_errors(self) -> None:
        # find_spec("missing_parent.child") raises ModuleNotFoundError; the probe
        # must catch it and report unhealthy rather than propagating a crash.
        healthy, reason = DependencyProbe(modules=("missing_parent_abc.child",)).check()
        assert healthy is False
        assert "missing_parent_abc.child" in reason
        assert DependencyProbe(modules=("numpy",)).check() == (True, "")

    def test_enumeration_failure_isolated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # When the global entry-point enumeration itself raises, built-ins still
        # register and the failure is captured as a discovery record.
        def _boom(group: str = discovery.ENTRY_POINT_GROUP) -> list[FakeEntryPoint]:
            raise RuntimeError("corrupt distribution metadata")

        monkeypatch.setattr(discovery, "iter_backend_entry_points", _boom)
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS)  # entry_points=None -> enumerate
        assert {"openmm", "tinker", "jax", "jax-md", "psi4"} <= set(snapshot.descriptors)
        enum_records = [
            r
            for r in snapshot.records
            if r.source is DiscoverySource.ENTRY_POINT and r.issue is DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR
        ]
        assert len(enum_records) == 1
        assert "enumeration failed" in enum_records[0].message

    def test_injected_entry_points_iteration_failure_isolated(self) -> None:
        def _bad_iter() -> Any:
            raise RuntimeError("generator blew up")
            yield  # pragma: no cover

        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=_bad_iter())
        assert {"openmm", "psi4"} <= set(snapshot.descriptors)
        assert any(
            r.issue is DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR
            for r in snapshot.records
            if r.source is DiscoverySource.ENTRY_POINT
        )

    def test_rejected_records_preserve_valid_entry_point_name(self) -> None:
        # Import/API/claim failures with a valid entry-point name preserve that
        # name so discovery_report().for_name(...) can find the rejected plugin.
        eps = [
            FakeEntryPoint("importboom", load_error=ImportError("x")),
            FakeEntryPoint(
                "oldapi",
                load_result=_manifest(name="oldapi", backend_api_version=BACKEND_API_VERSION + 9),
            ),
            FakeEntryPoint("badcap", load_result=_manifest(name="badcap", capability_ceiling=["fly"])),
        ]
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=eps)
        report = DiscoveryReport(records=snapshot.records)
        for name, kind in (
            ("importboom", DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR),
            ("oldapi", DiscoveryIssueKind.INCOMPATIBLE_API_VERSION),
            ("badcap", DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM),
        ):
            found = report.for_name(name)
            assert found, name
            assert found[0].name == name
            assert found[0].state is DiscoveryState.REJECTED
            assert found[0].issue is kind
            assert name not in snapshot.descriptors  # rejected, never registered

    def test_rejected_record_drops_unsafe_entry_point_name(self) -> None:
        ep = FakeEntryPoint("../evil", load_error=ImportError("x"))
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[ep])
        record = next(r for r in snapshot.records if r.entry_point == "../evil")
        assert record.name == ""  # unsafe entry-point name is not preserved as a key
        assert DiscoveryReport(records=snapshot.records).for_name("../evil") == ()


# ---------------------------------------------------------------------------
# Discovery record / snapshot / report invariants and deep immutability
# ---------------------------------------------------------------------------


class TestRecordInvariants:
    def test_snapshot_uses_identity_equality_for_mapping_backed_state(self) -> None:
        first = DiscoverySnapshot(descriptors={}, records=())
        second = DiscoverySnapshot(descriptors={}, records=())

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_registered_record_rejects_issue_or_message(self) -> None:
        with pytest.raises(ValueError, match="REGISTERED"):
            DiscoveryRecord(
                source=DiscoverySource.BUILTIN,
                state=DiscoveryState.REGISTERED,
                name="x",
                issue=DiscoveryIssueKind.INVALID_DESCRIPTOR,
                message="bad",
            )

    def test_non_registered_requires_issue_and_message(self) -> None:
        with pytest.raises(ValueError, match="typed issue and a non-empty message"):
            DiscoveryRecord(source=DiscoverySource.ENTRY_POINT, state=DiscoveryState.REJECTED)
        with pytest.raises(ValueError, match="typed issue and a non-empty message"):
            DiscoveryRecord(
                source=DiscoverySource.ENTRY_POINT,
                state=DiscoveryState.UNAVAILABLE,
                name="x",
                issue=DiscoveryIssueKind.MISSING_DEPENDENCY,
                message="",
            )

    def test_registered_requires_name(self) -> None:
        with pytest.raises(ValueError, match="must name its descriptor"):
            DiscoveryRecord(source=DiscoverySource.ENTRY_POINT, state=DiscoveryState.REGISTERED)

    def test_builtin_record_forbids_entry_point_identity(self) -> None:
        with pytest.raises(ValueError, match="BUILTIN"):
            DiscoveryRecord(
                source=DiscoverySource.BUILTIN,
                state=DiscoveryState.REGISTERED,
                name="x",
                entry_point="ep",
            )

    def test_rejected_record_may_omit_name(self) -> None:
        # A global enumeration failure has no descriptor name or entry point.
        record = DiscoveryRecord(
            source=DiscoverySource.ENTRY_POINT,
            state=DiscoveryState.REJECTED,
            issue=DiscoveryIssueKind.DESCRIPTOR_IMPORT_ERROR,
            message="enumeration failed",
        )
        assert record.name == "" and record.entry_point == ""

    def test_load_failed_record_is_registered(self) -> None:
        # A descriptor stays registered after an explicit load failure.
        record = DiscoveryRecord(
            source=DiscoverySource.LOAD,
            state=DiscoveryState.LOAD_FAILED,
            name="openmm",
            issue=DiscoveryIssueKind.BROKEN_FACTORY,
            message="factory blew up",
        )
        assert record.registered is True

    def test_report_counts_load_failed_descriptor_as_registered(self) -> None:
        # A report holding both the base and the load record lists the name once.
        report = DiscoveryReport(
            records=[
                DiscoveryRecord(source=DiscoverySource.BUILTIN, state=DiscoveryState.REGISTERED, name="openmm"),
                DiscoveryRecord(
                    source=DiscoverySource.LOAD,
                    state=DiscoveryState.LOAD_FAILED,
                    name="openmm",
                    issue=DiscoveryIssueKind.BROKEN_FACTORY,
                    message="factory blew up",
                ),
            ]
        )
        assert report.registered == ("openmm",)
        assert any(r.state is DiscoveryState.LOAD_FAILED for r in report.issues)

    def test_manifest_failure_requires_kind_and_message(self) -> None:
        with pytest.raises(ValueError):
            ManifestFailure(DiscoveryIssueKind.INVALID_DESCRIPTOR, "")

    def test_report_normalizes_records_to_sorted_tuple(self) -> None:
        report = DiscoveryReport(
            records=[
                DiscoveryRecord(source=DiscoverySource.BUILTIN, state=DiscoveryState.REGISTERED, name="zzz"),
                DiscoveryRecord(source=DiscoverySource.BUILTIN, state=DiscoveryState.REGISTERED, name="aaa"),
            ]
        )
        assert isinstance(report.records, tuple)
        assert [r.name for r in report.records] == ["aaa", "zzz"]

    def test_report_rejects_non_records(self) -> None:
        with pytest.raises(ValueError, match="DiscoveryRecord"):
            DiscoveryReport(records=["not-a-record"])  # type: ignore[list-item]

    def test_snapshot_validates_key_matches_descriptor_name(self) -> None:
        descriptor = registry._BUILTIN_MANIFESTS  # need a real descriptor; build one
        result = validate_manifest(descriptor[0], entry_point_name=None)
        assert not isinstance(result, ManifestFailure)
        with pytest.raises(ValueError, match="must equal descriptor name"):
            DiscoverySnapshot(descriptors={"wrong-key": result}, records=())

    def test_snapshot_rejects_non_descriptor_values(self) -> None:
        with pytest.raises(ValueError, match="must be a BackendDescriptor"):
            DiscoverySnapshot(descriptors={"x": object()}, records=())  # type: ignore[dict-item]

    def test_snapshot_normalizes_records_to_sorted_tuple(self) -> None:
        records = [
            DiscoveryRecord(source=DiscoverySource.ENTRY_POINT, state=DiscoveryState.REGISTERED, name="zzz"),
            DiscoveryRecord(source=DiscoverySource.BUILTIN, state=DiscoveryState.REGISTERED, name="aaa"),
        ]
        snapshot = DiscoverySnapshot(descriptors={}, records=tuple(records))
        # BUILTIN (rank 0) sorts before ENTRY_POINT (rank 1) regardless of input order.
        assert [r.name for r in snapshot.records] == ["aaa", "zzz"]

    def test_records_are_deeply_immutable(self) -> None:
        record = DiscoveryRecord(source=DiscoverySource.BUILTIN, state=DiscoveryState.REGISTERED, name="x")
        with pytest.raises((AttributeError, TypeError)):
            record.name = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Entry-point enumeration itself (realistic temporary dist-info)
# ---------------------------------------------------------------------------


def _write_dist_info(root: Path, dist_name: str, version: str, entry_points_body: str) -> None:
    """Write a minimal ``.dist-info`` so importlib.metadata discovers it."""
    info = root / f"{dist_name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {dist_name}\nVersion: {version}\n", encoding="utf-8")
    (info / "entry_points.txt").write_text(entry_points_body, encoding="utf-8")


class TestEntryPointEnumeration:
    def test_iter_backend_entry_points_from_dist_info(
        self, tmp_path: Path, fixture_on_path: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_dist_info(
            tmp_path,
            "q2mm-backend-reference",
            "1.0.0",
            f"[{discovery.ENTRY_POINT_GROUP}]\nharmonic-reference = {FIXTURE_VALUE}\n",
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()
        entry_points = discovery.iter_backend_entry_points()
        matches = [ep for ep in entry_points if getattr(ep, "name", "") == "harmonic-reference"]
        assert matches, "temporary dist-info entry point was not discovered"
        ep = matches[0]
        assert ep.value == FIXTURE_VALUE
        assert discovery._distribution_name(ep) == "q2mm-backend-reference"
        # Enumeration must not import the implementation.
        assert "q2mm_reference_backend.backend" not in sys.modules

    def test_iter_backend_entry_points_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        unordered = [
            FakeEntryPoint("b", "m:B", dist_name="z-dist"),
            FakeEntryPoint("a", "m:A", dist_name="a-dist"),
            FakeEntryPoint("a", "m:A2", dist_name="a-dist"),
        ]

        class _Selectable:
            def select(self, *, group: str) -> list[FakeEntryPoint]:
                return list(unordered)

        monkeypatch.setattr(discovery.importlib_metadata, "entry_points", lambda: _Selectable())
        ordered = discovery.iter_backend_entry_points()
        keys = [(discovery._distribution_name(ep), ep.name, ep.value) for ep in ordered]
        assert keys == sorted(keys)


# ---------------------------------------------------------------------------
# Registry-level lazy cache, refresh, and load-failure overlay
# ---------------------------------------------------------------------------


class TestRegistryDiscovery:
    def test_import_does_not_build_snapshot(self) -> None:
        # Importing the registry module must not eagerly enumerate/discover.
        import subprocess

        code = (
            f"import sys;sys.path.insert(0, {str(REPO_ROOT)!r});"
            "import q2mm.backends.registry as reg;"
            "assert reg._snapshot is None, 'snapshot built at import time';"
            "print('lazy-ok')"
        )
        # Use a fresh interpreter but retain its configured site-packages:
        # some CI images provide NumPy through the user site, which ``-I``
        # intentionally removes.  The source root is inserted explicitly.
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO_ROOT))
        assert result.returncode == 0, result.stderr
        assert "lazy-ok" in result.stdout

    def test_lazy_cache_and_refresh(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points([])
        assert "harmonic-reference" not in registry.registered_backends()
        inject_entry_points([_fixture_entry_point()])
        assert "harmonic-reference" in registry.registered_backends()

    def test_registered_external_available(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points([_fixture_entry_point()])
        assert "harmonic-reference" in registry.available_mm_backends()
        desc = registry.get_descriptor("harmonic-reference")
        assert desc.role is BackendRole.MM

    def test_catalog_no_factory_import(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points([_fixture_entry_point()])
        registry.catalog()
        registry.available_backends()
        assert "q2mm_reference_backend.backend" not in sys.modules

    def test_broken_factory_recorded_on_explicit_load(
        self, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        ep = FakeEntryPoint(
            "broken",
            load_result=_manifest(name="broken", factory="numpy:ThisAttrDoesNotExist"),
        )
        inject_entry_points([ep])
        # Registered and healthy by probe (numpy present) before the load attempt.
        assert "broken" in registry.available_backends()
        with pytest.raises(BackendConfigurationError):
            registry.load_backend("broken")
        # Now catalog reports it unhealthy, while healthy descriptors remain.
        status = {s.name: s for s in registry.catalog()}
        assert status["broken"].healthy is False
        assert "load failed" in status["broken"].reason
        assert status["openmm"].healthy in (True, False)  # unaffected either way
        record = registry.discovery_report().for_name("broken")
        assert any(r.issue is DiscoveryIssueKind.BROKEN_FACTORY for r in record)

    def test_runtime_info_mismatch_recorded(self, inject_entry_points: Callable[[list[FakeEntryPoint]], None]) -> None:
        ep = FakeEntryPoint(
            "mismatch",
            load_result=_manifest(name="mismatch", factory="test.test_registry:MismatchedCapsBackend"),
        )
        inject_entry_points([ep])
        with pytest.raises(BackendConfigurationError):
            registry.load_backend("mismatch")
        record = registry.discovery_report().for_name("mismatch")
        assert any(r.issue is DiscoveryIssueKind.BROKEN_FACTORY for r in record)

    def test_discovery_report_lists_registered_and_issues(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points(
            [
                _fixture_entry_point(),
                FakeEntryPoint("cap-fail", load_result=_manifest(name="cap-fail", capability_ceiling=["fly"])),
            ]
        )
        report = registry.discovery_report()
        assert "harmonic-reference" in report.registered
        assert {"openmm", "psi4"} <= set(report.registered)
        assert any(r.issue is DiscoveryIssueKind.INVALID_CAPABILITY_CLAIM for r in report.issues)

    def test_load_failure_overlay_cleared_on_success(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points([_fixture_entry_point()])
        # Simulate a prior transient/invalid-config failure poisoning the catalog.
        registry._record_load_failure("harmonic-reference", BackendConfigurationError("transient config"))
        assert {s.name: s.healthy for s in registry.catalog()}["harmonic-reference"] is False
        # A later successful load must clear the overlay so the catalog recovers.
        backend = registry.load_backend("harmonic-reference")
        assert backend.info.provenance.backend == "harmonic-reference"
        assert {s.name: s.healthy for s in registry.catalog()}["harmonic-reference"] is True
        lingering = [
            r
            for r in registry.discovery_records()
            if r.name == "harmonic-reference" and r.state is DiscoveryState.LOAD_FAILED
        ]
        assert lingering == []


# ---------------------------------------------------------------------------
# Capability-specific conformance
# ---------------------------------------------------------------------------


class TestCapabilityConformance:
    def test_fixture_energy_conformance_via_discovery(
        self, fixture_on_path: None, inject_entry_points: Callable[[list[FakeEntryPoint]], None]
    ) -> None:
        inject_entry_points([_fixture_entry_point()])
        backend = registry.load_backend("harmonic-reference")
        molecule, force_field = _harmonic_case()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=registry.get_descriptor("harmonic-reference"),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
            )
        )
        assert outcome.executed == (Capability.ENERGY,)
        # Every other drivable capability is proven typed-unsupported, including
        # the backend-level BATCHED_HESSIAN surface.
        for capability in (
            Capability.MINIMIZE,
            Capability.HESSIAN,
            Capability.FREQUENCIES,
            Capability.PARAMETER_GRADIENT,
            Capability.COORDINATE_GRADIENT,
            Capability.HESSIAN_PARAMETER_JACOBIAN,
            Capability.BATCHED_ENERGY,
            Capability.BATCHED_HESSIAN,
        ):
            assert capability in outcome.unsupported_verified

    def test_reusable_state_reuses_same_session(self) -> None:
        # A backend declaring ENERGY + REUSABLE_STATE is exercised by reusing the
        # SAME prepared session; REUSABLE_STATE is recorded as executed.
        molecule, force_field = _harmonic_case()
        backend = _ReusableStateBackend()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=_descriptor_for_backend(backend),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
                capabilities=frozenset({Capability.ENERGY, Capability.REUSABLE_STATE}),
            )
        )
        assert Capability.ENERGY in outcome.executed
        assert Capability.REUSABLE_STATE in outcome.executed
        assert backend.prepared_sessions == 1  # prepared once, not twice

    def test_reusable_state_reuse_without_energy(self) -> None:
        # A backend declaring HESSIAN + REUSABLE_STATE (no ENERGY) demonstrates
        # reuse via its declared+executed HESSIAN driver.
        molecule, force_field = _harmonic_case()
        backend = _HessianReuseBackend()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=_descriptor_for_backend(backend),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
                capabilities=frozenset({Capability.HESSIAN, Capability.REUSABLE_STATE}),
            )
        )
        assert Capability.HESSIAN in outcome.executed
        assert Capability.REUSABLE_STATE in outcome.executed
        assert Capability.ENERGY in outcome.unsupported_verified

    def test_reusable_state_selected_without_drivable_capability_raises(self) -> None:
        # REUSABLE_STATE selected but nothing runnable was executed -> failure,
        # not a silent omission.
        molecule, force_field = _harmonic_case()
        backend = _HessianReuseBackend()
        with pytest.raises(ConformanceError, match="no selected prepared-session capability"):
            run_mm_conformance(
                MMConformanceCase(
                    descriptor=_descriptor_for_backend(backend),
                    backend=backend,
                    molecule=molecule,
                    force_field=force_field,
                    capabilities=frozenset({Capability.REUSABLE_STATE}),
                )
            )

    def test_only_declared_implementation_hooks_run(self) -> None:
        # The helper may invoke undeclared public wrappers to confirm the base
        # guard, but no undeclared implementation hook body may execute.
        molecule, force_field = _harmonic_case()
        backend = _HookSpyBackend()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=_descriptor_for_backend(backend),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
            )
        )
        assert outcome.executed == (Capability.ENERGY,)
        # Only the ENERGY hook implementation ran; every undeclared hook was
        # blocked by the base capability guard before dispatch.
        assert backend.prepared.hooks_invoked == ["_energy"]

    def test_fixture_unsupported_capability_not_invoked(self, fixture_on_path: None) -> None:
        # The base class raises before dispatch, so the hook body never runs.
        snapshot = build_snapshot(registry._BUILTIN_MANIFESTS, entry_points=[_fixture_entry_point()])
        backend = snapshot.descriptors["harmonic-reference"].load()
        molecule, force_field = _harmonic_case()
        from q2mm.backends.contracts import HessianRequest

        prepared = backend.prepare(PreparationRequest(case_id="c", molecule=molecule, force_field=force_field))
        vector = ParameterLayout.from_force_field(force_field).vector(force_field)
        with pytest.raises(UnsupportedCapabilityError):
            prepared.hessian(HessianRequest(parameters=vector))

    def test_conformance_rejects_qm_backend(self) -> None:
        molecule, force_field = _harmonic_case()
        backend = _ReferenceLikeBackend()
        with pytest.raises(ConformanceError, match="runtime role"):
            run_mm_conformance(
                MMConformanceCase(
                    descriptor=_descriptor_for_backend(backend, role=BackendRole.MM),
                    backend=backend,
                    molecule=molecule,
                    force_field=force_field,
                )
            )

    @pytest.mark.openmm
    def test_openmm_claim_gated_conformance(self) -> None:
        backend = registry.load_backend("openmm", platform_name="CPU")
        molecule, force_field = _harmonic_case()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=registry.get_descriptor("openmm"),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
                capabilities=frozenset({Capability.ENERGY, Capability.REUSABLE_STATE}),
            )
        )
        assert Capability.ENERGY in outcome.executed
        assert Capability.REUSABLE_STATE in outcome.executed  # session reuse works
        # OpenMM declares neither of these; they must be typed-unsupported.
        assert Capability.HESSIAN_PARAMETER_JACOBIAN in outcome.unsupported_verified
        assert Capability.BATCHED_ENERGY in outcome.unsupported_verified
        assert Capability.BATCHED_HESSIAN in outcome.unsupported_verified

    @pytest.mark.jax
    def test_jax_claim_gated_conformance(self) -> None:
        backend = registry.load_backend("jax")
        molecule, force_field = _harmonic_case()
        outcome = run_mm_conformance(
            MMConformanceCase(
                descriptor=registry.get_descriptor("jax"),
                backend=backend,
                molecule=molecule,
                force_field=force_field,
            )
        )
        assert Capability.ENERGY in outcome.executed
