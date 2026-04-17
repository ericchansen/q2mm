"""Tests for the q2mm.optimizers package-level registry.

Covers ``available_optimizers``, ``get_optimizer``, and the module
``__getattr__`` descriptive-error fallback.  These ensure that a
missing optional dependency surfaces with an actionable install hint
instead of being silently swallowed (as it was prior to PR-C).
"""

from __future__ import annotations

import pytest

from q2mm import optimizers


def test_available_optimizers_returns_list() -> None:
    """Available optimizers must be a list of strings."""
    available = optimizers.available_optimizers()
    assert isinstance(available, list)
    assert all(isinstance(name, str) for name in available)


def test_available_optimizers_contains_objective_independent_ones() -> None:
    """Optimizers that need no optional deps should always register.

    ``ObjectiveFunction`` is unconditionally re-exported (not part of the
    optional registry) so it is not in ``available_optimizers()``.  We test
    instead that the consumer-visible API is non-empty when scipy is
    installed (which it is in any environment that runs the test suite).
    """
    pytest.importorskip("scipy")
    available = optimizers.available_optimizers()
    assert "ScipyOptimizer" in available
    assert "BasinHoppingOptimizer" in available
    assert "MultiStartOptimizer" in available


def test_get_optimizer_returns_class_when_available() -> None:
    """``get_optimizer`` must return the actual class for installed deps."""
    pytest.importorskip("scipy")
    cls = optimizers.get_optimizer("ScipyOptimizer")
    assert cls.__name__ == "ScipyOptimizer"


def test_get_optimizer_unknown_raises_keyerror() -> None:
    """Unknown name → ``KeyError`` with the list of known names."""
    with pytest.raises(KeyError, match="Unknown optimizer 'NotARealOptimizer'"):
        optimizers.get_optimizer("NotARealOptimizer")


def test_module_getattr_unknown_raises_attributeerror() -> None:
    """Unknown attribute → standard ``AttributeError`` (Python convention)."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = optimizers.NonexistentAttribute  # type: ignore[attr-defined]


def test_get_optimizer_missing_dep_raises_descriptive_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing-dep simulation: ``get_optimizer`` raises ``ImportError`` with hint.

    Inject a fake failure record into ``_FAILED`` and verify the user-visible
    error names the install command, the original exception, and chains it
    via ``__cause__``.
    """
    fake_exc = ImportError("No module named 'fake_dep'")
    monkeypatch.setitem(
        optimizers._FAILED,
        "FakeOptimizer",
        ("pip install 'q2mm[fake]'", fake_exc),
    )
    with pytest.raises(ImportError) as excinfo:
        optimizers.get_optimizer("FakeOptimizer")
    msg = str(excinfo.value)
    assert "FakeOptimizer" in msg
    assert "pip install 'q2mm[fake]'" in msg
    assert "fake_dep" in msg
    assert excinfo.value.__cause__ is fake_exc


def test_module_getattr_missing_dep_raises_descriptive_importerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from q2mm.optimizers import X`` → descriptive ``ImportError``.

    This is the most common consumer pattern and the highest-leverage fix:
    before PR-C this would be ``ImportError: cannot import name 'X'`` with
    no install hint.
    """
    fake_exc = ImportError("No module named 'fake_dep'")
    monkeypatch.setitem(
        optimizers._FAILED,
        "FakeOptimizer",
        ("pip install 'q2mm[fake]'", fake_exc),
    )
    with pytest.raises(ImportError) as excinfo:
        _ = optimizers.FakeOptimizer  # type: ignore[attr-defined]
    msg = str(excinfo.value)
    assert "FakeOptimizer" in msg
    assert "pip install 'q2mm[fake]'" in msg
    assert excinfo.value.__cause__ is fake_exc
