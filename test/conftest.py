"""Shared pytest configuration and fixtures.

Test tiers
----------
Tests are categorized by purpose and time budget:

- **core** (unmarked): < 1 second. Unit tests, parsers, models. Always run.
- **integration** (``@pytest.mark.integration``): 1–10 seconds. Backend
  integration: backend contracts, optimizer convergence, parity checks.
- **validation** (``@pytest.mark.validation``): 1–30 seconds. Correctness
  validation with *no* optimizer loops — QFUERZA estimation, published
  FF evaluation, ethane TS.
- **nightly** (``@pytest.mark.nightly``): 1–10 minutes. Heavy tests with
  optimizer loops (L-BFGS-B 200 iterations), Rh-enamide full loops, gradient
  speedup.

By default, ``pytest`` runs only core tests::

    pytest                        # core only (~13s)
    pytest --run-integration      # core + integration (~49s)
    pytest --run-validation       # core + integration + validation (~80s)
    pytest --run-nightly          # everything (~330s+)

GPU enforcement
---------------
By default, all backends are forced to CPU to prevent GPU memory
allocation from locking up the local machine.  This applies only when
``JAX_PLATFORMS`` and ``OPENMM_DEFAULT_PLATFORM`` are not already set
in the environment.  Use ``--gpu`` or set ``Q2MM_USE_GPU=1`` to opt
into GPU execution::

    pytest --gpu               # allow GPU backends
    Q2MM_USE_GPU=1 pytest      # same via env var

Backend markers
---------------
Tests can be tagged with ``@pytest.mark.openmm``, ``@pytest.mark.tinker``,
``@pytest.mark.jax``, or ``@pytest.mark.psi4`` to indicate which backend they
require.  Tests are **auto-skipped** when the corresponding dependency is not
installed.

Use ``-m`` to filter::

    pytest -m openmm                     # only OpenMM tests
    pytest -m "not tinker"               # skip Tinker tests
    pytest -m "openmm and nightly"       # nightly OpenMM tests only
"""

from __future__ import annotations

import pytest

# Re-export shared constants and factories so conftest fixtures can use them.
# Test files should import directly from ``test._shared``.
from test._shared import (  # noqa: F401
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
    SN2_ENERGY,
    SN2_FREQS,
    SN2_HESSIAN,
    SN2_QM_REF,
    SN2_XYZ,
    SUPPORTING_INFO_DIR,
    TS_FCHK,
    make_diatomic,
    make_noble_gas_pair,
    make_water,
)

# ---------------------------------------------------------------------------
# Backend availability detection
# ---------------------------------------------------------------------------

# Mapping from pytest marker names to registry keys.
# Marker names use underscores (Python identifiers); registry keys use hyphens.
_MARKER_TO_REGISTRY = {
    "openmm": "openmm",
    "tinker": "tinker",
    "jax": "jax",
    "jax_md": "jax-md",
    "psi4": "psi4",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Include integration tests (backend contracts, parity, ~1-10s each)",
    )
    parser.addoption(
        "--run-validation",
        action="store_true",
        default=False,
        help="Include validation tests (no optimizer loops, ~1-30s each); implies --run-integration",
    )
    parser.addoption(
        "--run-nightly",
        action="store_true",
        default=False,
        help="Include nightly tests (optimizer loops, heavy computation, ~1-10min each); implies --run-validation",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Allow GPU backends (JAX CUDA, OpenMM CUDA). Without this flag, "
        "tests default to CPU unless JAX_PLATFORMS or OPENMM_DEFAULT_PLATFORM "
        "is already set in the environment.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: backend integration tests (~1-10s each)")
    config.addinivalue_line("markers", "validation: correctness validation, no optimizer loops (~1-30s each)")
    config.addinivalue_line("markers", "nightly: heavy tests with optimizer loops (~1-10min each)")
    config.addinivalue_line("markers", "cross_backend: parity tests executing two or more backends")
    config.addinivalue_line(
        "markers",
        "benchmark: benchmark timing disabled by default; use --benchmark-enable to collect timing",
    )
    config.addinivalue_line("markers", "openmm: requires OpenMM backend")
    config.addinivalue_line("markers", "tinker: requires Tinker backend")
    config.addinivalue_line("markers", "jax: requires JAX backend")
    config.addinivalue_line("markers", "jax_md: requires JAX-MD backend")
    config.addinivalue_line("markers", "psi4: requires Psi4 QM backend")
    config.addinivalue_line(
        "markers",
        "external_data: requires external validation data in validation/supporting-info/",
    )

    # Force CPU-only unless the user explicitly opts into GPU execution.
    # JAX initializes CUDA at import time, consuming ~25 GiB of VRAM.
    # OpenMM defaults to the "best" platform (CUDA > OpenCL > CPU).
    # Both are dangerous when automated tools invoke pytest without
    # knowing they need CPU-only configuration.
    import os

    gpu_opt_in = getattr(config.option, "gpu", False) or os.environ.get("Q2MM_USE_GPU", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if not gpu_opt_in:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("OPENMM_DEFAULT_PLATFORM", "CPU")

    rh_enamide = REPO_ROOT / "examples" / "publication" / "rh-enamide"
    if rh_enamide.is_dir():
        os.environ.setdefault("Q2MM_RH_ENAMIDE", str(rh_enamide))
    if SUPPORTING_INFO_DIR is not None:
        os.environ.setdefault("Q2MM_SUPPORTING_INFO", str(SUPPORTING_INFO_DIR))
    mm3_base = REPO_ROOT / "validation" / "published_ffs" / "mm3_base.fld"
    if mm3_base.is_file():
        os.environ.setdefault("Q2MM_MM3_BASE", str(mm3_base))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_nightly = config.getoption("--run-nightly")
    run_validation = config.getoption("--run-validation") or run_nightly
    run_integration = config.getoption("--run-integration") or run_validation

    if not run_nightly:
        skip_nightly = pytest.mark.skip(reason="need --run-nightly to run")
        for item in items:
            if "nightly" in item.keywords:
                item.add_marker(skip_nightly)

    if not run_validation:
        skip_validation = pytest.mark.skip(reason="need --run-validation to run")
        for item in items:
            if "validation" in item.keywords:
                item.add_marker(skip_validation)

    if not run_integration:
        skip_integration = pytest.mark.skip(reason="need --run-integration to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)

    # Auto-skip tests that require a backend unusable under the test
    # configuration. The production catalog is intentionally only a cheap
    # dependency probe; Tinker also needs a parameter file to be testable.
    # Import lazily so that the JAX_PLATFORMS guard above takes effect first.
    from test.backend_fixtures import backend_is_usable

    for marker_name, registry_key in _MARKER_TO_REGISTRY.items():
        if not backend_is_usable(registry_key):
            skip_marker = pytest.mark.skip(reason=f"{registry_key} not usable with the test configuration")
            for item in items:
                if marker_name in item.keywords:
                    item.add_marker(skip_marker)

    # Auto-skip external_data tests when supporting-info directory is absent.
    if SUPPORTING_INFO_DIR is None:
        skip_data = pytest.mark.skip(
            reason="external validation data not found "
            "(set Q2MM_SUPPORTING_INFO or extract to validation/supporting-info/)"
        )
        for item in items:
            if "external_data" in item.keywords:
                item.add_marker(skip_data)
