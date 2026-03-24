"""Energy calculation benchmarks across engines.

Each test measures the wall-clock time of a single-point energy evaluation
using ``pytest-benchmark`` and asserts the result is a finite float.
These are ``medium``-tier tests, skipped by default.  Pass
``--run-medium`` to enable them.  ``--benchmark-enable`` only controls
whether ``pytest-benchmark`` collects timing data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule

pytestmark = [pytest.mark.benchmark, pytest.mark.medium]


# ---------------------------------------------------------------------------
# Per-engine energy benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.openmm
def test_energy_openmm(
    benchmark: object,
    openmm_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark single-point energy with OpenMM."""
    result: float = benchmark(openmm_engine.energy, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.tinker
def test_energy_tinker(
    benchmark: object,
    tinker_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark single-point energy with Tinker."""
    result: float = benchmark(tinker_engine.energy, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.jax
def test_energy_jax(
    benchmark: object,
    jax_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark single-point energy with JAX (harmonic)."""
    result: float = benchmark(jax_engine.energy, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    assert isinstance(result, float)
    assert np.isfinite(result)


@pytest.mark.jax_md
def test_energy_jax_md(
    benchmark: object,
    jax_md_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark single-point energy with JAX-MD (OPLSAA)."""
    result: float = benchmark(jax_md_engine.energy, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    assert isinstance(result, float)
    assert np.isfinite(result)
