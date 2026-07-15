"""Cross-backend parity tests.

Validates that different MM backends produce consistent results for the
same molecule and force field.  These tests catch regressions in backend
implementations by asserting that energies and frequencies agree within
a tolerance.

Because force-field functional forms differ across backends, only backends
that share a functional form are compared directly.  MM3 is supported by
OpenMM, Tinker, and JAX, so the OpenMM/JAX pair is compared explicitly and
does not depend on Tinker being installed.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
import pytest

from q2mm.backends.contracts import EnergyRequest, FrequencyRequest
from q2mm.backends.registry import available_backends as _available_backends
from q2mm.backends.registry import load_backend
from test.backend_fixtures import backend_is_usable, load_test_backend, param_vector, prepare_case

if TYPE_CHECKING:
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Molecule

pytestmark = [pytest.mark.benchmark, pytest.mark.cross_backend, pytest.mark.integration]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Backends that use the same functional form can be compared directly.
_MM3_BACKENDS = {"openmm", "tinker"}


def _available_from(pool: set[str]) -> list[str]:
    """Return the subset of *pool* that is actually installed."""
    return sorted(name for name in pool if backend_is_usable(name))


def _skip_unless_pair(pool: set[str]) -> None:
    """Skip if fewer than two backends in *pool* are available."""
    available = _available_from(pool)
    if len(available) < 2:
        pytest.skip(f"Need >=2 backends from {pool}; have {available}")


# ---------------------------------------------------------------------------
# Energy parity
# ---------------------------------------------------------------------------


class TestEnergyParity:
    """Assert that backends sharing a functional form agree on energy."""

    @pytest.mark.openmm
    @pytest.mark.tinker
    def test_mm3_energy_parity(
        self,
        ch3f_mol: Molecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and Tinker should agree on MM3 energy."""
        _skip_unless_pair(_MM3_BACKENDS)
        backends = {name: load_test_backend(name) for name in _available_from(_MM3_BACKENDS)}

        energies: dict[str, float] = {}
        for name, bk in backends.items():
            energies[name] = (
                prepare_case(bk, ch3f_mol, ch3f_ff).energy(EnergyRequest(parameters=param_vector(ch3f_ff))).energy
            )

        names = list(energies.keys())
        for a, b in combinations(names, 2):
            assert energies[a] == pytest.approx(energies[b], abs=1e-3), (
                f"Energy mismatch: {a}={energies[a]:.6f} vs {b}={energies[b]:.6f}"
            )

    @pytest.mark.openmm
    @pytest.mark.jax
    def test_openmm_jax_mm3_energy_parity(
        self,
        ch3f_mol: Molecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and JAX must agree on MM3 energy (independent of Tinker)."""
        avail = set(_available_backends())
        if not {"openmm", "jax"} <= avail:
            pytest.skip("Need both OpenMM and JAX for this MM3 parity check")
        omm = prepare_case(load_backend("openmm"), ch3f_mol, ch3f_ff)
        jbk = prepare_case(load_backend("jax"), ch3f_mol, ch3f_ff)
        e_omm = omm.energy(EnergyRequest(parameters=param_vector(ch3f_ff))).energy
        e_jax = jbk.energy(EnergyRequest(parameters=param_vector(ch3f_ff))).energy
        assert e_omm == pytest.approx(e_jax, abs=1e-3), f"OpenMM {e_omm:.6f} vs JAX {e_jax:.6f}"


# ---------------------------------------------------------------------------
# Frequency parity
# ---------------------------------------------------------------------------


class TestFrequencyParity:
    """Assert that backends sharing a functional form agree on frequencies."""

    @pytest.mark.openmm
    @pytest.mark.tinker
    def test_mm3_frequency_parity(
        self,
        ch3f_mol: Molecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and Tinker should agree on MM3 vibrational frequencies."""
        _skip_unless_pair(_MM3_BACKENDS)
        backends = {name: load_test_backend(name) for name in _available_from(_MM3_BACKENDS)}

        # frequencies() returns all 3N modes; compare only the
        # 3N-6 vibrational modes (skip near-zero translation/rotation).
        n_vib = 3 * len(ch3f_mol.symbols) - 6

        freq_sets: dict[str, np.ndarray] = {}
        for name, bk in backends.items():
            result = prepare_case(bk, ch3f_mol, ch3f_ff).frequencies(FrequencyRequest(parameters=param_vector(ch3f_ff)))
            all_freqs = np.sort(np.asarray([float(f) for f in result.frequencies]))
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

    @pytest.mark.openmm
    @pytest.mark.jax
    def test_openmm_jax_mm3_frequency_parity(
        self,
        ch3f_mol: Molecule,
        ch3f_ff: ForceField,
    ) -> None:
        """OpenMM and JAX must agree on MM3 vibrational frequencies."""
        avail = set(_available_backends())
        if not {"openmm", "jax"} <= avail:
            pytest.skip("Need both OpenMM and JAX for this MM3 parity check")
        n_vib = 3 * len(ch3f_mol.symbols) - 6
        omm = prepare_case(load_backend("openmm"), ch3f_mol, ch3f_ff).frequencies(
            FrequencyRequest(parameters=param_vector(ch3f_ff))
        )
        jbk = prepare_case(load_backend("jax"), ch3f_mol, ch3f_ff).frequencies(
            FrequencyRequest(parameters=param_vector(ch3f_ff))
        )
        f_omm = np.sort(np.asarray([float(f) for f in omm.frequencies]))[-n_vib:]
        f_jax = np.sort(np.asarray([float(f) for f in jbk.frequencies]))[-n_vib:]
        np.testing.assert_allclose(f_omm, f_jax, atol=1.0, rtol=1e-3, err_msg="OpenMM vs JAX MM3 frequency mismatch")


# ---------------------------------------------------------------------------
# Golden-value validation (archived benchmark results)
# ---------------------------------------------------------------------------


class TestGoldenValues:
    """Validate current backend output against archived benchmark results."""

    @pytest.mark.openmm
    def test_openmm_frequencies_match_golden(
        self,
        openmm_backend: object,
        ch3f_mol: Molecule,
        ch3f_ff: ForceField,
        golden_results: dict[str, dict[str, object]],
    ) -> None:
        """Current OpenMM frequencies should match archived values."""
        key = "ch3f_openmm_mm3_cpu_lbfgsb"
        if key not in golden_results:
            pytest.skip(f"No golden result for {key}")

        golden_default_freqs = golden_results[key]["default_ff"]["frequencies_cm1"]
        n_golden = len(golden_default_freqs)

        # Backend returns all 3N modes; archived results store only 3N-6.
        current_freqs = [
            float(f)
            for f in prepare_case(openmm_backend, ch3f_mol, ch3f_ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ch3f_ff)))
            .frequencies
        ]
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
