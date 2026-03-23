"""Shared pytest configuration and fixtures.

Speed tier markers
------------------
Tests are categorized into three speed tiers:

- **fast** (unmarked): < 1 second. Unit tests, parsers, models. Default run.
- **medium** (``@pytest.mark.medium``): 1–10 seconds. Integration tests with
  a single optimizer run or backend call.
- **slow** (``@pytest.mark.slow``): > 10 seconds. Cross-backend parity,
  multi-iteration optimizer convergence.

By default, ``pytest`` runs only fast tests (~13s). Use CLI flags to include
slower tiers::

    pytest                     # fast only (~13s)
    pytest --run-medium        # fast + medium (~49s)
    pytest --run-slow          # everything (~330s)

Backend markers
---------------
Tests can be tagged with ``@pytest.mark.openmm``, ``@pytest.mark.tinker``,
``@pytest.mark.jax``, or ``@pytest.mark.psi4`` to indicate which backend they
require.  Tests are **auto-skipped** when the corresponding dependency is not
installed.

Use ``-m`` to filter::

    pytest -m openmm           # only OpenMM tests
    pytest -m "not tinker"     # skip Tinker tests
    pytest -m "openmm and slow"  # slow OpenMM tests only
"""

import pytest

# Re-export shared constants and factories so conftest fixtures can use them.
# Test files should import directly from ``test._shared``.
from test._shared import (  # noqa: F401
    CH3F_DATA_AVAILABLE,
    CH3F_ENERGY,
    CH3F_FREQS,
    CH3F_HESS,
    CH3F_MODES,
    CH3F_XYZ,
    COMPLEX_XYZ,
    ETHANE_DIR,
    EXAMPLES_DIR,
    GS_FCHK,
    REPO_ROOT,
    SN2_DATA_AVAILABLE,
    SN2_ENERGY,
    SN2_FREQS,
    SN2_HESSIAN,
    SN2_QM_REF,
    SN2_XYZ,
    TS_FCHK,
    make_diatomic,
    make_noble_gas_pair,
    make_water,
)

# ---------------------------------------------------------------------------
# Backend availability detection (runs once at collection time)
# ---------------------------------------------------------------------------

from q2mm.backends.registry import available_engines as _available_engines

_AVAILABLE_BACKENDS = set(_available_engines())

# Mapping from pytest marker names to registry keys.
# Marker names use underscores (Python identifiers); registry keys use hyphens.
_MARKER_TO_REGISTRY = {
    "openmm": "openmm",
    "tinker": "tinker",
    "jax": "jax",
    "jax_md": "jax-md",
    "psi4": "psi4",
}


def pytest_addoption(parser):
    parser.addoption(
        "--run-medium",
        action="store_true",
        default=False,
        help="Include medium-speed tests (1-10s each)",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Include slow tests (>10s each); implies --run-medium",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (>10s each)")
    config.addinivalue_line("markers", "medium: marks tests as medium speed (1-10s each)")
    config.addinivalue_line("markers", "openmm: requires OpenMM backend")
    config.addinivalue_line("markers", "tinker: requires Tinker backend")
    config.addinivalue_line("markers", "jax: requires JAX backend")
    config.addinivalue_line("markers", "jax_md: requires JAX-MD backend")
    config.addinivalue_line("markers", "psi4: requires Psi4 QM backend")


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--run-slow")
    run_medium = config.getoption("--run-medium") or run_slow

    if not run_slow:
        skip_slow = pytest.mark.skip(reason="need --run-slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not run_medium:
        skip_medium = pytest.mark.skip(reason="need --run-medium to run")
        for item in items:
            if "medium" in item.keywords:
                item.add_marker(skip_medium)

    # Auto-skip tests that require a missing backend
    for marker_name, registry_key in _MARKER_TO_REGISTRY.items():
        if registry_key not in _AVAILABLE_BACKENDS:
            skip_marker = pytest.mark.skip(reason=f"{registry_key} not available")
            for item in items:
                if marker_name in item.keywords:
                    item.add_marker(skip_marker)
