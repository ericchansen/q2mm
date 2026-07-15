"""Full force-field optimization benchmark.

Runs ``diagnostics.benchmark.run_combo`` end-to-end for each available
engine and validates that the pipeline produces valid results.

These tests are ``slow`` because each optimization takes 10–60 s depending
on the backend.
"""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.benchmark, pytest.mark.nightly]


def _run_and_validate(engine: object) -> None:
    """Run the benchmark pipeline and validate structure."""
    from q2mm.benchmarks.systems import load_system
    from q2mm.diagnostics.benchmark import run_combo
    from test._shared import engine_functional_form

    case = load_system("ch3f", engine=engine, functional_form=engine_functional_form(engine).value)
    result = run_combo(
        engine=engine,
        case=case,
        optimizer_method="L-BFGS-B",
    )

    assert result.seminario is not None
    assert result.optimized is not None

    seminario_rmsd: float = result.seminario["rmsd"]
    optimized_rmsd: float = result.optimized["rmsd"]

    assert math.isfinite(seminario_rmsd), f"Seminario RMSD is {seminario_rmsd}"
    assert math.isfinite(optimized_rmsd), f"Optimized RMSD is {optimized_rmsd}"
    assert result.optimized["n_eval"] > 0


@pytest.mark.openmm
def test_optimization_openmm(openmm_engine: object) -> None:
    """Full optimization benchmark with OpenMM."""
    _run_and_validate(openmm_engine)


@pytest.mark.jax
def test_optimization_jax(jax_engine: object) -> None:
    """Full optimization benchmark with JAX (harmonic)."""
    _run_and_validate(jax_engine)


@pytest.mark.jax_md
def test_optimization_jax_md(jax_md_engine: object) -> None:
    """Full optimization benchmark with JAX-MD (OPLSAA)."""
    _run_and_validate(jax_md_engine)
