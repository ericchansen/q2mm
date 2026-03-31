"""Shared fixtures for benchmark and validation tests.

Provides molecule, force-field, and engine fixtures used across the
benchmark suite.  Engine fixtures auto-skip when the corresponding
backend is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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


from test._shared import (
    CH3F_DATA_AVAILABLE,
    CH3F_ENERGY,
    CH3F_FREQS,
    CH3F_HESS,
    CH3F_MODES,
    CH3F_XYZ,
    REPO_ROOT,
)

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule


# ---------------------------------------------------------------------------
# Molecule / reference-data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ch3f_mol() -> Q2MMMolecule:
    """Load the CH3F molecule from QM reference data."""
    if not CH3F_DATA_AVAILABLE:
        pytest.skip("CH3F reference data not found")
    from q2mm.models.molecule import Q2MMMolecule

    return Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)


@pytest.fixture(scope="session")
def ch3f_ff(ch3f_mol: Q2MMMolecule) -> ForceField:
    """Create a default force field for CH3F."""
    from q2mm.models.forcefield import ForceField

    return ForceField.create_for_molecule(ch3f_mol)


@pytest.fixture(scope="session")
def ch3f_qm_freqs() -> np.ndarray:
    """Load QM reference frequencies for CH3F (cm⁻¹)."""
    if not CH3F_FREQS.exists():
        pytest.skip("CH3F frequencies file not found")
    return np.loadtxt(CH3F_FREQS)


@pytest.fixture(scope="session")
def ch3f_qm_hessian() -> np.ndarray:
    """Load QM Hessian matrix for CH3F."""
    if not CH3F_HESS.exists():
        pytest.skip("CH3F Hessian file not found")
    return np.load(CH3F_HESS)


@pytest.fixture(scope="session")
def ch3f_qm_energy() -> float:
    """Load the QM reference energy for CH3F (hartrees)."""
    if not CH3F_ENERGY.exists():
        pytest.skip("CH3F energy file not found")
    return float(np.loadtxt(CH3F_ENERGY))


@pytest.fixture(scope="session")
def ch3f_normal_modes() -> dict[str, np.ndarray] | None:
    """Load QM normal modes for CH3F, or ``None`` if unavailable."""
    if not CH3F_MODES.exists():
        return None
    data = np.load(CH3F_MODES)
    return {
        "eigenvalues": data["eigenvalues"],
        "eigenvectors": data["eigenvectors"],
        "masses_amu": data["masses_amu"],
    }


# ---------------------------------------------------------------------------
# Engine fixtures — auto-skip when backend is missing
# ---------------------------------------------------------------------------

_BACKEND_AVAILABILITY: dict[str, bool] = {}


def _backend_available(name: str) -> bool:
    """Check whether a backend is available (cached)."""
    if name not in _BACKEND_AVAILABILITY:
        from q2mm.backends.registry import available_engines

        _avail = set(available_engines())
        for key in ("openmm", "tinker", "jax", "jax-md"):
            _BACKEND_AVAILABILITY[key] = key in _avail
    return _BACKEND_AVAILABILITY.get(name, False)


@pytest.fixture(scope="session")
def openmm_engine() -> object:
    """Create an OpenMM engine, skipping if unavailable."""
    if not _backend_available("openmm"):
        pytest.skip("OpenMM not available")
    from q2mm.backends.mm.openmm import OpenMMEngine

    return OpenMMEngine()


@pytest.fixture(scope="session")
def tinker_engine() -> object:
    """Create a Tinker engine, skipping if unavailable."""
    if not _backend_available("tinker"):
        pytest.skip("Tinker not available")
    from q2mm.backends.mm.tinker import TinkerEngine

    return TinkerEngine()


@pytest.fixture(scope="session")
def jax_engine() -> object:
    """Create a JAX (harmonic) engine, skipping if unavailable."""
    if not _backend_available("jax"):
        pytest.skip("JAX not available")
    from q2mm.backends.mm.jax_engine import JaxEngine

    return JaxEngine()


@pytest.fixture(scope="session")
def jax_md_engine() -> object:
    """Create a JAX-MD (OPLSAA) engine, skipping if unavailable."""
    if not _backend_available("jax-md"):
        pytest.skip("JAX-MD not available")
    from q2mm.backends.mm.jax_md_engine import JaxMDEngine

    return JaxMDEngine()


# ---------------------------------------------------------------------------
# Archived benchmark results (golden values)
# ---------------------------------------------------------------------------

_BENCHMARK_DIR = REPO_ROOT / "benchmarks" / "ch3f" / "results"


@pytest.fixture(scope="session")
def golden_results() -> dict[str, dict[str, object]]:
    """Load archived benchmark JSON results keyed by filename stem.

    Only loads BenchmarkResult-shaped JSONs (must contain ``metadata``
    and ``default_ff`` keys).  Cycling logs and error-only results are
    skipped.

    Returns a dict like ``{"ch3f_openmm_mm3_cpu_lbfgsb": {...}, ...}``.
    """
    import json

    results: dict[str, dict[str, object]] = {}
    if not _BENCHMARK_DIR.exists():
        pytest.skip("Archived benchmark results not found")
    for path in sorted(_BENCHMARK_DIR.glob("*.json")):
        with open(path) as fh:
            data = json.load(fh)
        if "metadata" not in data or not data.get("default_ff"):
            continue
        results[path.stem] = data
    return results
