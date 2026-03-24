"""Frequency calculation benchmarks and validation.

Benchmarks the ``engine.frequencies()`` call for each backend and
validates that the returned frequencies are physically reasonable
(positive, sorted, correct count).

Note: ``engine.frequencies()`` returns all 3N modes (including 6 near-zero
translational/rotational modes for non-linear molecules).  Validation
focuses on the 3N-6 vibrational modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule

pytestmark = [pytest.mark.benchmark, pytest.mark.medium]

# CH3F has 5 atoms → 3N = 15 total modes, 3N-6 = 9 vibrational modes
_N_ATOMS = 5
_EXPECTED_TOTAL_MODES = 3 * _N_ATOMS
_EXPECTED_VIB_MODES = _EXPECTED_TOTAL_MODES - 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_frequencies(freqs: list[float]) -> None:
    """Assert the frequency vector looks physically reasonable.

    Accepts either 3N modes (all modes including translational/rotational)
    or 3N-6 vibrational-only modes, since some engines strip the near-zero
    modes before returning.
    """
    n_modes = len(freqs)
    assert n_modes in (_EXPECTED_TOTAL_MODES, _EXPECTED_VIB_MODES), (
        f"Expected {_EXPECTED_TOTAL_MODES} (3N) or {_EXPECTED_VIB_MODES} (3N-6) modes, got {n_modes}"
    )
    arr = np.asarray(freqs)
    assert np.all(np.isfinite(arr)), "Frequencies contain non-finite values"

    # Frequencies should be returned in ascending order by the engine
    assert np.all(np.diff(arr) >= -1e-6), "Frequencies not returned in ascending order"

    arr = np.sort(arr)
    # Vibrational modes: those above the 50 cm⁻¹ threshold
    vib = arr[arr > 50.0]
    assert len(vib) == _EXPECTED_VIB_MODES, (
        f"Expected {_EXPECTED_VIB_MODES} vibrational modes (>50 cm⁻¹), got {len(vib)}"
    )
    assert np.all(vib > 0), f"Vibrational modes should be positive, min={vib.min():.2f}"


# ---------------------------------------------------------------------------
# Per-engine frequency benchmarks
# ---------------------------------------------------------------------------


@pytest.mark.openmm
def test_frequencies_openmm(
    benchmark: object,
    openmm_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark frequencies with OpenMM."""
    freqs: list[float] = benchmark(openmm_engine.frequencies, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    _validate_frequencies(freqs)


@pytest.mark.tinker
def test_frequencies_tinker(
    benchmark: object,
    tinker_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark frequencies with Tinker."""
    freqs: list[float] = benchmark(tinker_engine.frequencies, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    _validate_frequencies(freqs)


@pytest.mark.jax
def test_frequencies_jax(
    benchmark: object,
    jax_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark frequencies with JAX (harmonic)."""
    freqs: list[float] = benchmark(jax_engine.frequencies, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    _validate_frequencies(freqs)


@pytest.mark.jax_md
def test_frequencies_jax_md(
    benchmark: object,
    jax_md_engine: object,
    ch3f_mol: Q2MMMolecule,
    ch3f_ff: ForceField,
) -> None:
    """Benchmark frequencies with JAX-MD (OPLSAA)."""
    freqs: list[float] = benchmark(jax_md_engine.frequencies, ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
    _validate_frequencies(freqs)
