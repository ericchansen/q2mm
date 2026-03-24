"""Full force-field optimization benchmark.

Runs ``diagnostics.benchmark.run_benchmark`` end-to-end for each
available engine and validates that the optimized force field improves
the frequency RMSD relative to the default starting point.

These tests are ``slow`` because each optimization takes 10–60 s depending
on the backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from q2mm.models.molecule import Q2MMMolecule

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


def _run_and_validate(
    engine: object,
    mol: Q2MMMolecule,
    qm_freqs: np.ndarray,
    qm_hessian: np.ndarray,
    normal_modes: dict[str, np.ndarray] | None,
) -> None:
    """Run the benchmark pipeline and assert improvement."""
    from q2mm.diagnostics.benchmark import run_benchmark

    result = run_benchmark(
        engine=engine,
        molecule=mol,
        qm_freqs=qm_freqs,
        qm_hessian=qm_hessian,
        normal_modes=normal_modes,
        optimizer_method="L-BFGS-B",
    )

    # Basic structural checks
    assert result.default_ff is not None
    assert result.optimized is not None

    default_rmsd: float = result.default_ff["rmsd"]
    optimized_rmsd: float = result.optimized["rmsd"]

    # Optimization must reduce RMSD
    assert optimized_rmsd < default_rmsd, (
        f"Optimization did not improve RMSD: {default_rmsd:.1f} → {optimized_rmsd:.1f}"
    )


# ---------------------------------------------------------------------------
# Per-engine full-optimization benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.openmm
def test_optimization_openmm(
    openmm_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_qm_freqs: np.ndarray,
    ch3f_qm_hessian: np.ndarray,
    ch3f_normal_modes: dict[str, np.ndarray] | None,
) -> None:
    """Full optimization benchmark with OpenMM."""
    _run_and_validate(openmm_engine, ch3f_mol, ch3f_qm_freqs, ch3f_qm_hessian, ch3f_normal_modes)


@pytest.mark.jax
def test_optimization_jax(
    jax_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_qm_freqs: np.ndarray,
    ch3f_qm_hessian: np.ndarray,
    ch3f_normal_modes: dict[str, np.ndarray] | None,
) -> None:
    """Full optimization benchmark with JAX (harmonic)."""
    _run_and_validate(jax_engine, ch3f_mol, ch3f_qm_freqs, ch3f_qm_hessian, ch3f_normal_modes)


@pytest.mark.jax_md
def test_optimization_jax_md(
    jax_md_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_qm_freqs: np.ndarray,
    ch3f_qm_hessian: np.ndarray,
    ch3f_normal_modes: dict[str, np.ndarray] | None,
) -> None:
    """Full optimization benchmark with JAX-MD (OPLSAA)."""
    _run_and_validate(jax_md_engine, ch3f_mol, ch3f_qm_freqs, ch3f_qm_hessian, ch3f_normal_modes)
