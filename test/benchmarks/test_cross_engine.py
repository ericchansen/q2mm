"""Cross-engine parity tests.

Validates that different MM backends produce consistent results for the
same molecule and force field.  These tests catch regressions in engine
implementations by asserting that energies and frequencies agree within
a tolerance.

Because force-field functional forms differ across backends (e.g. MM3 in
OpenMM/Tinker vs. pure harmonic in JAX), only engines that share a
functional form are compared directly.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pytest

from q2mm.backends.registry import available_engines as _available_engines

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule

pytestmark = [pytest.mark.benchmark, pytest.mark.medium]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Engines that use the same functional form can be compared directly.
_MM3_BACKENDS = {"openmm", "tinker"}


def _get_engine(name: str) -> object:
    """Instantiate a backend engine by registry name."""
    from q2mm.backends.registry import get_engine

    return get_engine(name)


def _available_from(pool: set[str]) -> list[str]:
    """Return the subset of *pool* that is actually installed."""
    avail = set(_available_engines())
    return sorted(pool & avail)


def _skip_unless_pair(pool: set[str]) -> None:
    """Skip if fewer than two backends in *pool* are available."""
    available = _available_from(pool)
    if len(available) < 2:
        pytest.skip(f"Need ≥2 backends from {pool}; have {available}")


# ---------------------------------------------------------------------------
# Energy parity
# ---------------------------------------------------------------------------


class TestEnergyParity:
    """Assert that engines sharing a functional form agree on energy."""

    def test_mm3_energy_parity(
        self,
        ch3f_mol: Q2MMMolecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and Tinker should agree on MM3 energy."""
        _skip_unless_pair(_MM3_BACKENDS)
        engines = {name: _get_engine(name) for name in _available_from(_MM3_BACKENDS)}

        energies: dict[str, float] = {}
        for name, eng in engines.items():
            energies[name] = eng.energy(ch3f_mol, ch3f_ff)

        names = list(energies.keys())
        for a, b in combinations(names, 2):
            assert energies[a] == pytest.approx(energies[b], abs=1e-3), (
                f"Energy mismatch: {a}={energies[a]:.6f} vs {b}={energies[b]:.6f}"
            )


# ---------------------------------------------------------------------------
# Frequency parity
# ---------------------------------------------------------------------------


class TestFrequencyParity:
    """Assert that engines sharing a functional form agree on frequencies."""

    def test_mm3_frequency_parity(
        self,
        ch3f_mol: Q2MMMolecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and Tinker should agree on MM3 vibrational frequencies."""
        _skip_unless_pair(_MM3_BACKENDS)
        engines = {name: _get_engine(name) for name in _available_from(_MM3_BACKENDS)}

        # engines.frequencies() returns all 3N modes; compare only the
        # 3N-6 vibrational modes (skip near-zero translation/rotation).
        n_vib = 3 * len(ch3f_mol.symbols) - 6

        freq_sets: dict[str, np.ndarray] = {}
        for name, eng in engines.items():
            all_freqs = np.sort(np.asarray(eng.frequencies(ch3f_mol, ch3f_ff)))
            freq_sets[name] = all_freqs[-n_vib:]

        names = list(freq_sets.keys())
        for a, b in combinations(names, 2):
            assert len(freq_sets[a]) == len(freq_sets[b]), (
                f"Mode count mismatch: {a} has {len(freq_sets[a])}, {b} has {len(freq_sets[b])}"
            )
            np.testing.assert_allclose(
                freq_sets[a],
                freq_sets[b],
                atol=1.0,
                rtol=1e-3,
                err_msg=f"Vibrational frequency mismatch between {a} and {b}",
            )


# ---------------------------------------------------------------------------
# Golden-value validation (archived benchmark results)
# ---------------------------------------------------------------------------


class TestGoldenValues:
    """Validate current engine output against archived benchmark results."""

    @pytest.mark.openmm
    def test_openmm_frequencies_match_golden(
        self,
        openmm_engine: object,
        ch3f_mol: Q2MMMolecule,
        ch3f_ff: ForceField,
        golden_results: dict[str, dict[str, object]],
    ) -> None:
        """Current OpenMM frequencies should match archived values."""
        key = "ch3f_openmm_mm3_cpu_lbfgsb"
        if key not in golden_results:
            pytest.skip(f"No golden result for {key}")

        golden_default_freqs = golden_results[key]["default_ff"]["frequencies_cm1"]
        n_golden = len(golden_default_freqs)

        # Engine returns all 3N modes; archived results store only 3N-6.
        current_freqs = openmm_engine.frequencies(ch3f_mol, ch3f_ff)  # type: ignore[attr-defined]
        current_vib = current_freqs[-n_golden:]

        np.testing.assert_allclose(
            current_vib,
            golden_default_freqs,
            atol=0.1,
            err_msg="OpenMM default-FF frequencies drifted from archived values",
        )

    @pytest.mark.openmm
    def test_qm_reference_consistency(
        self,
        ch3f_qm_freqs: np.ndarray,
        golden_results: dict[str, dict[str, object]],
    ) -> None:
        """QM reference frequencies should match what the benchmarks used."""
        key = "ch3f_openmm_mm3_cpu_lbfgsb"
        if key not in golden_results:
            pytest.skip(f"No golden result for {key}")

        archived_qm = np.asarray(golden_results[key]["qm_reference"]["frequencies_cm1"])
        np.testing.assert_allclose(
            ch3f_qm_freqs,
            archived_qm,
            atol=1e-6,
            err_msg="QM reference frequencies differ from archived benchmark",
        )

    def test_golden_rmsd_sanity(
        self,
        golden_results: dict[str, dict[str, object]],
    ) -> None:
        """All archived benchmarks should have optimized RMSD < default RMSD."""
        for name, result in golden_results.items():
            default_rmsd = result["default_ff"]["rmsd"]
            optimized_rmsd = result["optimized"]["rmsd"]
            assert optimized_rmsd < default_rmsd, (
                f"{name}: optimized RMSD ({optimized_rmsd:.1f}) >= default ({default_rmsd:.1f})"
            )
