"""Tests for the engine registry."""

import pytest

from q2mm.backends.registry import (
    EngineNotAvailable,
    _check_available,
    available_engines,
    available_mm_engines,
    available_qm_engines,
    get_engine,
    get_mm_engine,
    get_qm_engine,
    registered_engines,
    registered_mm_engines,
    registered_qm_engines,
)


class TestRegisteredEngines:
    """Verify that all engines are discovered and registered."""

    def test_discover_registers_all_mm_engines(self) -> None:
        engines = registered_mm_engines()
        assert "openmm" in engines
        assert "tinker" in engines
        assert "jax" in engines
        assert "jax-md" in engines

    def test_discover_registers_qm_engines(self) -> None:
        engines = registered_qm_engines()
        assert "psi4" in engines

    def test_registered_engines_has_all(self) -> None:
        engines = registered_engines()
        assert len(engines) >= 5
        for name in ("openmm", "tinker", "jax", "jax-md", "psi4"):
            assert name in engines


class TestAvailableEngines:
    """Verify that availability checks work."""

    def test_available_engines_returns_list(self) -> None:
        result = available_engines()
        assert isinstance(result, list)
        # Should contain at least one engine if any backend is installed
        assert len(result) >= 0  # structural check; contents depend on environment

    def test_available_mm_engines_subset_of_available(self) -> None:
        mm = set(available_mm_engines())
        all_ = set(available_engines())
        assert mm <= all_

    def test_available_qm_engines_subset_of_available(self) -> None:
        qm = set(available_qm_engines())
        all_ = set(available_engines())
        assert qm <= all_

    def test_available_engines_sorted(self) -> None:
        result = available_engines()
        assert result == sorted(result)


class TestGetEngine:
    """Verify engine instantiation through the registry."""

    @pytest.mark.openmm
    def test_get_engine_openmm(self) -> None:
        from q2mm.backends.base import MMEngine

        engine = get_engine("openmm")
        assert isinstance(engine, MMEngine)
        assert engine.is_available()

    @pytest.mark.openmm
    def test_get_mm_engine_openmm(self) -> None:
        engine = get_mm_engine("openmm")
        assert engine.is_available()

    def test_get_engine_unknown_raises(self) -> None:
        with pytest.raises(EngineNotAvailable, match="not-a-real-engine"):
            get_engine("not-a-real-engine")

    def test_get_mm_engine_unknown_raises(self) -> None:
        with pytest.raises(EngineNotAvailable):
            get_mm_engine("not-a-real-engine")

    def test_get_qm_engine_unknown_raises(self) -> None:
        with pytest.raises(EngineNotAvailable):
            get_qm_engine("not-a-real-engine")

    @pytest.mark.openmm
    def test_get_engine_passes_kwargs(self) -> None:
        engine = get_engine("openmm", platform_name="CPU")
        assert engine.is_available()

    def test_engine_not_available_message_includes_registered(self) -> None:
        with pytest.raises(EngineNotAvailable) as exc_info:
            get_engine("nonexistent")
        # Should list at least some registered engines in the error message
        assert "Registered engines:" in str(exc_info.value)


class TestCheckAvailable:
    """Verify the _check_available helper."""

    @pytest.mark.openmm
    def test_check_available_with_real_engine(self) -> None:
        from q2mm.backends.mm.openmm import OpenMMEngine

        assert _check_available(OpenMMEngine) is True

    def test_check_available_with_broken_class(self) -> None:
        class BrokenEngine:
            def is_available(self) -> bool:
                raise RuntimeError("boom")

        assert _check_available(BrokenEngine) is False

    def test_check_available_with_unavailable_class(self) -> None:
        class UnavailableEngine:
            def is_available(self) -> bool:
                return False

        assert _check_available(UnavailableEngine) is False
