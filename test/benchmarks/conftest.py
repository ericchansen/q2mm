"""Shared fixtures for benchmark and validation tests.

Provides molecule, force-field, and backend fixtures used across the
benchmark suite.  Engine fixtures auto-skip when the corresponding
backend is not installed.
"""

from __future__ import annotations
from q2mm.backends.registry import load_backend

from typing import TYPE_CHECKING

import pytest

# ---------------------------------------------------------------------------
# pytest-benchmark stub — CI backend containers install with --no-deps
# and lack pytest-benchmark. Provide a passthrough so tests still run as
# correctness checks (without timing collection).
# ---------------------------------------------------------------------------

try:
    import pytest_benchmark  # noqa: F401
except ImportError:
    # CI backend containers install with --no-deps and lack pytest-benchmark.
    # Provide a passthrough so tests still run as correctness checks.
    # Note: if pytest-benchmark is installed but disabled via -p no:benchmark,
    # the import succeeds and this fallback is not triggered; use
    # --benchmark-disable instead of -p no:benchmark in that case.
    from collections.abc import Callable
    from typing import Any

    @pytest.fixture
    def benchmark() -> Callable[..., Any]:  # type: ignore[override]
        """Passthrough stub: calls the function without timing."""

        def _passthrough(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return _passthrough


from test._shared import CH3F_XYZ

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Molecule


# ---------------------------------------------------------------------------
# Molecule / reference-data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ch3f_mol() -> Molecule:
    """Load the CH3F molecule from QM reference data."""
    from q2mm.io.xyz import load_xyz

    return load_xyz(CH3F_XYZ, bond_tolerance=1.5)


@pytest.fixture(scope="session")
def ch3f_ff(ch3f_mol: Molecule) -> ForceField:
    """Create a default MM3-tagged force field for CH3F.

    Used by OpenMM/Tinker (both MM3-only-or-MM3-capable) and by the
    MM3-only cross-backend parity tests in ``test_cross_engine.py``. JAX
    and JAX-MD only evaluate the harmonic functional form — see
    :func:`ch3f_ff_harmonic` for the same auto-generated bond/angle
    values re-tagged for those two backends.
    """
    from q2mm.models.forcefield import ForceField, FunctionalForm

    return ForceField.create_for_molecule(ch3f_mol, functional_form=FunctionalForm.MM3)


@pytest.fixture(scope="session")
def ch3f_ff_harmonic(ch3f_ff: ForceField) -> ForceField:
    """Re-tag the same CH3F auto-generated bond/angle values as harmonic.

    For JAX/JAX-MD, which only evaluate the harmonic functional form
    (JAX-MD is harmonic-only; JAX supports both but the same values are
    equally valid interpreted as harmonic springs). Every
    :class:`~q2mm.models.forcefield.ForceField` must carry an explicit,
    correct functional form for the backend that will evaluate it — the
    same numeric bond_k/bond_eq/angle_k/angle_eq values cannot be
    silently shared as "unset" across backends that disagree on
    functional form.
    """
    import dataclasses

    from q2mm.models.forcefield import FunctionalForm

    return dataclasses.replace(ch3f_ff, functional_form=FunctionalForm.HARMONIC)


# ---------------------------------------------------------------------------
# Engine fixtures — auto-skip when backend is missing
# ---------------------------------------------------------------------------

_BACKEND_AVAILABILITY: dict[str, bool] = {}


def _backend_available(name: str) -> bool:
    """Check whether a backend is available (cached)."""
    if name not in _BACKEND_AVAILABILITY:
        from test.backend_fixtures import backend_is_usable

        for key in ("openmm", "tinker", "jax", "jax-md"):
            _BACKEND_AVAILABILITY[key] = backend_is_usable(key)
    return _BACKEND_AVAILABILITY.get(name, False)


@pytest.fixture(scope="session")
def openmm_backend() -> object:
    """Create an OpenMM backend, skipping if unavailable."""
    if not _backend_available("openmm"):
        pytest.skip("OpenMM not available")

    return load_backend("openmm")


@pytest.fixture(scope="session")
def tinker_backend() -> object:
    """Create a Tinker backend, skipping if unavailable."""
    if not _backend_available("tinker"):
        pytest.skip("Tinker not available")

    from test.backend_fixtures import load_test_backend

    return load_test_backend("tinker")


@pytest.fixture(scope="session")
def jax_backend() -> object:
    """Create a JAX (harmonic) backend, skipping if unavailable."""
    if not _backend_available("jax"):
        pytest.skip("JAX not available")

    return load_backend("jax")


@pytest.fixture(scope="session")
def jax_md_backend() -> object:
    """Create a JAX-MD (OPLSAA) backend, skipping if unavailable."""
    if not _backend_available("jax-md"):
        pytest.skip("JAX-MD not available")

    return load_backend("jax-md")
