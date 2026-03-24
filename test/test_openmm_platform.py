"""Unit tests for OpenMM platform detection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

openmm = pytest.importorskip("openmm")

from q2mm.backends.mm.openmm import (  # noqa: E402
    OpenMMEngine,
    _PLATFORM_PRIORITY,
    detect_best_platform,
)


def _mock_platforms(names: list[str]) -> tuple:
    """Build mock getNumPlatforms / getPlatform callables.

    Args:
        names: Ordered list of platform names the mock should expose.

    Returns:
        (getNumPlatforms, getPlatform) pair for patching.

    """
    platforms = [SimpleNamespace(getName=lambda n=n: n) for n in names]
    return len(platforms), lambda i: platforms[i]


class TestDetectBestPlatform:
    """Tests for detect_best_platform()."""

    def test_prefers_cuda_over_cpu(self) -> None:
        num, get = _mock_platforms(["Reference", "CPU", "CUDA"])
        with (
            patch.object(openmm.Platform, "getNumPlatforms", return_value=num),
            patch.object(openmm.Platform, "getPlatform", side_effect=get),
        ):
            assert detect_best_platform() == "CUDA"

    def test_prefers_opencl_when_no_cuda(self) -> None:
        num, get = _mock_platforms(["Reference", "CPU", "OpenCL"])
        with (
            patch.object(openmm.Platform, "getNumPlatforms", return_value=num),
            patch.object(openmm.Platform, "getPlatform", side_effect=get),
        ):
            assert detect_best_platform() == "OpenCL"

    def test_falls_back_to_cpu(self) -> None:
        num, get = _mock_platforms(["Reference", "CPU"])
        with (
            patch.object(openmm.Platform, "getNumPlatforms", return_value=num),
            patch.object(openmm.Platform, "getPlatform", side_effect=get),
        ):
            assert detect_best_platform() == "CPU"

    def test_cuda_beats_opencl(self) -> None:
        num, get = _mock_platforms(["OpenCL", "CUDA", "CPU", "Reference"])
        with (
            patch.object(openmm.Platform, "getNumPlatforms", return_value=num),
            patch.object(openmm.Platform, "getPlatform", side_effect=get),
        ):
            assert detect_best_platform() == "CUDA"

    def test_priority_constant_has_expected_order(self) -> None:
        assert _PLATFORM_PRIORITY == ("CUDA", "OpenCL", "CPU", "Reference")


class TestPrecisionValidation:
    """Tests for precision parameter validation in OpenMMEngine."""

    def test_valid_precision_accepted(self) -> None:
        for p in ("single", "mixed", "double"):
            engine = OpenMMEngine(precision=p)
            assert engine._precision == p

    def test_precision_normalized_to_lowercase(self) -> None:
        engine = OpenMMEngine(precision="Mixed")
        assert engine._precision == "mixed"

    def test_precision_strips_whitespace(self) -> None:
        engine = OpenMMEngine(precision="  double  ")
        assert engine._precision == "double"

    def test_invalid_precision_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid precision"):
            OpenMMEngine(precision="half")

    def test_none_precision_allowed(self) -> None:
        engine = OpenMMEngine(precision=None)
        assert engine._precision is None
