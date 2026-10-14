"""The public optimizer namespace exposes one catalog and lazy concrete classes."""

from __future__ import annotations

import importlib

import pytest

from q2mm import optimizers
from q2mm.optimizers import catalog
from q2mm.optimizers.scipy_opt import ScipyOptimizer


@pytest.mark.parametrize(
    "name",
    [
        "OPTIMIZER_CATALOG",
        "OptimizerSpec",
        "build_optimizer",
        "optimizer_spec",
    ],
)
def test_package_catalog_exports_are_canonical_identities(name: str) -> None:
    assert getattr(optimizers, name) is getattr(catalog, name)


def test_concrete_class_export_is_the_actual_implementation() -> None:
    assert optimizers.ScipyOptimizer is ScipyOptimizer
    assert optimizers.ScipyOptimizer(optimizers.optimizer_spec("scipy-nm").settings).settings.method == "Nelder-Mead"


def test_unknown_catalog_key_is_an_actionable_error() -> None:
    with pytest.raises(ValueError, match="Unknown optimizer 'NotARealOptimizer'.*choose one"):
        optimizers.optimizer_spec("NotARealOptimizer")


def test_module_unknown_attribute_is_standard_attributeerror() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = optimizers.NonexistentAttribute  # type: ignore[attr-defined]


def test_missing_implementation_import_preserves_original_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    failure = ImportError("No module named 'fake_dep'")
    requests = []

    def unavailable(name: str) -> object:
        requests.append(name)
        raise failure

    monkeypatch.delattr(optimizers, "OptaxOptimizer", raising=False)
    monkeypatch.setattr(optimizers, "import_module", unavailable)
    with pytest.raises(ImportError) as caught:
        _ = optimizers.OptaxOptimizer
    assert requests == ["q2mm.optimizers.optax"]
    assert "OptaxOptimizer" in str(caught.value)
    assert "install q2mm[jax]" in str(caught.value)
    assert "fake_dep" in str(caught.value)
    assert caught.value.__cause__ is failure


def test_lazy_class_is_imported_and_cached_once(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []
    original = importlib.import_module
    monkeypatch.delattr(optimizers, "BasinHoppingOptimizer", raising=False)

    def load(name: str) -> object:
        requests.append(name)
        return original(name)

    monkeypatch.setattr(optimizers, "import_module", load)
    first = optimizers.BasinHoppingOptimizer
    second = optimizers.BasinHoppingOptimizer
    assert first is second
    assert requests == ["q2mm.optimizers.basinhopping"]
