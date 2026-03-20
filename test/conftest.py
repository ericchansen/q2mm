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
"""

import pytest


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
