"""Tests for the descriptor-based backend registry."""

import pytest

from q2mm.backends.contracts import (
    BackendConfigurationError,
    BackendRole,
    BackendUnavailableError,
)
from q2mm.backends.registry import (
    BackendNotRegistered,
    available_backends,
    available_mm_backends,
    available_qm_backends,
    catalog,
    descriptors,
    get_descriptor,
    load_backend,
    registered_backends,
)


class TestRegisteredDescriptors:
    """Verify that all built-in backends are described and validated."""

    def test_all_mm_backends_registered(self) -> None:
        names = set(registered_backends(role=BackendRole.MM))
        assert {"openmm", "tinker", "jax", "jax-md"} <= names

    def test_qm_backend_registered(self) -> None:
        assert "psi4" in registered_backends(role=BackendRole.QM)

    def test_registered_backends_has_all(self) -> None:
        names = set(registered_backends())
        for name in ("openmm", "tinker", "jax", "jax-md", "psi4"):
            assert name in names

    def test_descriptors_keys_match_catalog(self) -> None:
        assert set(descriptors()) == {s.name for s in catalog()}

    def test_batched_hessian_capability_matrix(self) -> None:
        """Only JAX declares BATCHED_HESSIAN in its static descriptor info."""
        from q2mm.backends.contracts import Capability

        descs = descriptors()
        assert Capability.BATCHED_HESSIAN in descs["jax"].info.capabilities
        for name in ("jax-md", "openmm", "tinker", "psi4"):
            assert Capability.BATCHED_HESSIAN not in descs[name].info.capabilities, name


class TestAvailability:
    """Verify cheap, side-effect-free availability reporting."""

    def test_available_backends_returns_list(self) -> None:
        assert isinstance(available_backends(), list)

    def test_available_mm_subset(self) -> None:
        assert set(available_mm_backends()) <= set(available_backends())

    def test_available_qm_subset(self) -> None:
        assert set(available_qm_backends()) <= set(available_backends())

    def test_available_sorted(self) -> None:
        result = available_backends()
        assert result == sorted(result)

    def test_catalog_reports_unavailable_explicitly(self) -> None:
        # Every descriptor appears in the catalog with an explicit status.
        for status in catalog():
            assert isinstance(status.healthy, bool)
            if not status.healthy:
                assert status.reason


class TestLoadBackend:
    """Verify backend construction through the registry."""

    @pytest.mark.openmm
    def test_load_openmm(self) -> None:
        backend = load_backend("openmm")
        assert backend.info.role is BackendRole.MM
        assert backend.info.provenance.backend == "openmm"

    @pytest.mark.openmm
    def test_load_openmm_passes_kwargs(self) -> None:
        backend = load_backend("openmm", platform_name="CPU")
        assert "CPU" in backend.info.name

    def test_load_unknown_raises(self) -> None:
        with pytest.raises(BackendNotRegistered, match="not-a-real-backend"):
            load_backend("not-a-real-backend")

    def test_not_registered_message_lists_registered(self) -> None:
        with pytest.raises(BackendNotRegistered) as exc_info:
            load_backend("nonexistent")
        assert "Registered backends:" in str(exc_info.value)

    def test_load_unavailable_raises_typed(self) -> None:
        # Find an unavailable descriptor and confirm load() raises typed error.
        for status in catalog():
            if not status.healthy:
                with pytest.raises(BackendUnavailableError):
                    load_backend(status.name)
                return
        pytest.skip("all backends available")


class TestDescriptorValidation:
    """Built-ins go through the same descriptor validation path."""

    def test_get_descriptor_openmm(self) -> None:
        desc = get_descriptor("openmm")
        assert desc.role is BackendRole.MM
        assert ":" in desc.factory

    def test_descriptor_rejects_bad_factory(self) -> None:
        from q2mm.backends.contracts import BackendDescriptor, BackendInfo, BackendProvenance

        info = BackendInfo(
            name="x", role=BackendRole.MM, provenance=BackendProvenance(backend="x", role=BackendRole.MM)
        )
        with pytest.raises(ValueError):
            BackendDescriptor(name="x", info=info, factory="missing_colon")

    def test_descriptor_rejects_empty_name(self) -> None:
        from q2mm.backends.contracts import BackendDescriptor, BackendInfo, BackendProvenance

        info = BackendInfo(
            name="x", role=BackendRole.MM, provenance=BackendProvenance(backend="x", role=BackendRole.MM)
        )
        # An empty descriptor name is rejected before the provenance-match check.
        with pytest.raises(ValueError):
            BackendDescriptor(name="", info=info, factory="a:b")

    def test_descriptor_load_bad_attr_raises_config_error(self) -> None:
        from q2mm.backends.contracts import (
            BackendDescriptor,
            BackendInfo,
            BackendProvenance,
            DependencyProbe,
        )

        # Point at a real module but a missing attribute; probe passes (numpy
        # is installed) so load() reaches the attribute lookup.
        info = BackendInfo(
            name="fake", role=BackendRole.MM, provenance=BackendProvenance(backend="fake", role=BackendRole.MM)
        )
        desc = BackendDescriptor(
            name="fake",
            info=info,
            factory="numpy:ThisAttrDoesNotExist",
            probe=DependencyProbe(modules=("numpy",)),
        )
        with pytest.raises(BackendConfigurationError):
            desc.load()


# ---------------------------------------------------------------------------
# Fake backends for descriptor factory-validation tests
# ---------------------------------------------------------------------------

from typing import Any

from q2mm.backends.contracts import BackendInfo, BackendProvenance, Capability


def _mk_info(
    name: str,
    caps: frozenset[Capability] = frozenset(),
    forms: frozenset[str] = frozenset(),
    *,
    prov_backend: str | None = None,
) -> BackendInfo:
    prov = BackendProvenance(backend=prov_backend or name, role=BackendRole.MM)
    return BackendInfo(name=name, role=BackendRole.MM, capabilities=caps, functional_forms=forms, provenance=prov)


class _PreparedNoop:
    pass


class GoodFakeBackend:
    """A well-formed fake backend matching the ``good-fake`` descriptor info."""

    def __init__(self, **kwargs: object) -> None:
        pass

    @property
    def info(self) -> BackendInfo:
        return _mk_info("good-fake", frozenset({Capability.ENERGY}), frozenset({"harmonic"}))

    def prepare(self, request: object) -> _PreparedNoop:
        return _PreparedNoop()


class MismatchedCapsBackend:
    """Runtime info declares different capabilities than its descriptor."""

    def __init__(self, **kwargs: object) -> None:
        pass

    @property
    def info(self) -> BackendInfo:
        return _mk_info("mismatch", frozenset({Capability.ENERGY, Capability.HESSIAN}), frozenset({"harmonic"}))

    def prepare(self, request: object) -> _PreparedNoop:
        return _PreparedNoop()


class WrongProvenanceBackend:
    """Runtime provenance.backend differs from the descriptor name."""

    def __init__(self, **kwargs: object) -> None:
        pass

    @property
    def info(self) -> BackendInfo:
        return _mk_info(
            "provwrong",
            frozenset({Capability.ENERGY}),
            frozenset({"harmonic"}),
            prov_backend="something-else",
        )

    def prepare(self, request: object) -> _PreparedNoop:
        return _PreparedNoop()


class NotABackend:
    """Missing the ``prepare`` method — does not satisfy the Backend protocol."""

    def __init__(self, **kwargs: object) -> None:
        pass

    @property
    def info(self) -> BackendInfo:
        return _mk_info("notbackend", frozenset(), frozenset())


def _descriptor(
    name: str,
    factory: str,
    *,
    caps: frozenset[Capability] | None = None,
    forms: frozenset[str] | None = None,
    probe: Any = None,
) -> Any:
    from q2mm.backends.contracts import BackendDescriptor, DependencyProbe

    caps = caps if caps is not None else frozenset({Capability.ENERGY})
    forms = forms if forms is not None else frozenset({"harmonic"})
    info = _mk_info(name, caps, forms)
    return BackendDescriptor(
        name=name,
        info=info,
        factory=factory,
        probe=probe if probe is not None else DependencyProbe(),
    )


class TestFactoryValidation:
    """Descriptor.load() validates the factory's runtime object structurally."""

    def test_good_factory_loads(self) -> None:
        desc = _descriptor("good-fake", "test.test_registry:GoodFakeBackend")
        backend = desc.load()
        assert backend.info.provenance.backend == "good-fake"

    def test_capabilities_mismatch_raises(self) -> None:
        # Descriptor declares only ENERGY; runtime declares ENERGY+HESSIAN.
        desc = _descriptor("mismatch", "test.test_registry:MismatchedCapsBackend")
        with pytest.raises(BackendConfigurationError):
            desc.load()

    def test_provenance_backend_mismatch_raises(self) -> None:
        desc = _descriptor("provwrong", "test.test_registry:WrongProvenanceBackend")
        with pytest.raises(BackendConfigurationError):
            desc.load()

    def test_missing_protocol_raises(self) -> None:
        desc = _descriptor("notbackend", "test.test_registry:NotABackend", caps=frozenset(), forms=frozenset())
        with pytest.raises(BackendConfigurationError):
            desc.load()

    def test_api_version_mismatch_rejected(self) -> None:
        from q2mm.backends.contracts import BackendDescriptor

        info = _mk_info("good-fake")
        with pytest.raises(ValueError):
            BackendDescriptor(
                name="good-fake", info=info, factory="test.test_registry:GoodFakeBackend", api_version=999
            )


class TestExplicitConfigBypassesProbe:
    """An unhealthy PATH probe must not block an explicit valid load."""

    def test_unhealthy_probe_still_loads_valid_backend(self) -> None:
        from q2mm.backends.contracts import DependencyProbe

        # Probe is unhealthy (module does not exist), but load() must bypass it
        # and construct the valid backend anyway.
        probe = DependencyProbe(modules=("definitely_not_installed_xyz_123",))
        healthy, _ = probe.check()
        assert healthy is False
        desc = _descriptor("good-fake", "test.test_registry:GoodFakeBackend", probe=probe)
        backend = desc.load()
        assert backend.info.provenance.backend == "good-fake"
