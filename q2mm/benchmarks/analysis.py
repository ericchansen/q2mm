"""Benchmark frequency, PES-distortion, and optimizer-sample analysis.

These diagnostics describe a run; they do not execute its optimization or
decide whether to promote it. Objective metric formulas remain owned by
``q2mm.objectives.metrics``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from q2mm.constants import REAL_FREQUENCY_THRESHOLD
from q2mm.objectives.metrics import category_stats

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend
    from q2mm.benchmarks.cases import BenchmarkCase
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Molecule


def real_frequencies(freqs: Iterable[float], threshold: float = REAL_FREQUENCY_THRESHOLD) -> np.ndarray:
    """Return the sorted real (non-imaginary/non-rigid) frequencies above *threshold*."""
    arr = np.asarray(list(freqs), dtype=float)
    return np.sort(arr[arr > threshold])


def frequency_rmsd(a: Iterable[float], b: Iterable[float]) -> float:
    """RMSD between two frequency arrays (truncated to the shorter length).

    Delegates the RMSD formula to :func:`q2mm.objectives.metrics.category_stats`
    so there is one RMSD implementation.
    """
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return float("nan")
    return category_stats(arr_a[:n], arr_b[:n])["rmsd"]


def frequency_mae(a: Iterable[float], b: Iterable[float]) -> float:
    """MAE between two frequency arrays (truncated to the shorter length)."""
    arr_a = np.asarray(list(a), dtype=float)
    arr_b = np.asarray(list(b), dtype=float)
    n = min(len(arr_a), len(arr_b))
    if n == 0:
        return float("nan")
    return category_stats(arr_a[:n], arr_b[:n])["mae"]


def _mm_real_frequencies(backend: Backend, molecules: Sequence[Any], ff: ForceField) -> np.ndarray:
    from q2mm.backends.contracts import FrequencyRequest, PreparationRequest
    from q2mm.models.parameters import ParameterLayout

    params = ParameterLayout.from_force_field(ff).vector(ff)
    all_real: list[float] = []
    for idx, mol in enumerate(molecules):
        prepared = backend.prepare(PreparationRequest(case_id=str(idx), molecule=mol, force_field=ff))
        mm_freqs = prepared.frequencies(FrequencyRequest(parameters=params)).frequencies
        all_real.extend(real_frequencies(mm_freqs).tolist())
    return np.array(sorted(all_real), dtype=float)


def _frequency_analysis(
    backend: Backend,
    case: BenchmarkCase,
    initial_ff: ForceField,
    final_ff: ForceField | None,
) -> dict[str, Any]:
    if not case.qm_freqs_per_mol:
        return {}
    from q2mm.backends.contracts import Capability

    if Capability.FREQUENCIES not in backend.info.capabilities:
        return {}
    qm_real = np.sort(np.concatenate([np.asarray(f, dtype=float) for f in case.qm_freqs_per_mol]))
    molecules = list(case.problem.molecules)
    analysis: dict[str, Any] = {"n_qm_real": int(qm_real.size)}
    init_real = _mm_real_frequencies(backend, molecules, initial_ff)
    analysis["initial_rmsd"] = frequency_rmsd(qm_real, init_real)
    if final_ff is not None:
        final_real = _mm_real_frequencies(backend, molecules, final_ff)
        analysis["final_rmsd"] = frequency_rmsd(qm_real, final_real)
        analysis["final_mae"] = frequency_mae(qm_real, final_real)
    return analysis


_HA_TO_KCAL = 627.5094740631


def compute_distortions(
    mol: Molecule,
    ff: ForceField,
    backend: Backend,
    modes: Mapping[str, np.ndarray],
    target_norms_ang: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], float, float]:
    """Displace a molecule along QM normal modes and compare MM to QM energies.

    Displaced geometries are produced with
    :meth:`~q2mm.models.molecule.Molecule.with_geometry`, so the molecule's
    explicit topology (including an explicitly empty topology), atom types,
    charge, multiplicity, and identity are preserved rather than re-inferred.
    """
    from q2mm.backends.contracts import EnergyRequest, PreparationRequest
    from q2mm.constants import AMU_TO_KG, BOHR_TO_ANG, HARTREE_TO_J, SPEED_OF_LIGHT_MS
    from q2mm.models.parameters import ParameterLayout

    if target_norms_ang is None:
        target_norms_ang = (0.05, 0.10, 0.15)

    params = ParameterLayout.from_force_field(ff).vector(ff)

    def _mm_energy(structure: Molecule, case_id: str) -> float:
        prepared = backend.prepare(PreparationRequest(case_id=case_id, molecule=structure, force_field=ff))
        return float(prepared.energy(EnergyRequest(parameters=params)).energy)

    eigenvalues = np.asarray(modes["eigenvalues"], dtype=float)
    eigenvectors = np.asarray(modes["eigenvectors"], dtype=float)
    masses_amu = np.asarray(modes["masses_amu"], dtype=float)

    bohr_to_m = BOHR_TO_ANG * 1e-10
    sqrt_m = np.sqrt(np.repeat(masses_amu, 3))
    real_mode_indices = [i for i, ev in enumerate(eigenvalues) if ev > 1e-3]

    e_eq = _mm_energy(mol, "eq")
    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    for mi in real_mode_indices:
        ev = eigenvalues[mi]
        evec_mw = eigenvectors[:, mi]
        ev_si = ev * HARTREE_TO_J / (bohr_to_m**2 * AMU_TO_KG)
        freq_cm1 = float(np.sqrt(ev_si) / (2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0))
        v_cart = evec_mw / sqrt_m
        v_cart_ang = v_cart * BOHR_TO_ANG
        v_norm = float(np.linalg.norm(v_cart_ang))
        displacements: list[dict[str, Any]] = []
        for d_ang in target_norms_ang:
            q = d_ang / v_norm
            e_qm = 0.5 * ev * q**2 * _HA_TO_KCAL
            delta_xyz = (q * v_cart * BOHR_TO_ANG).reshape(-1, 3)
            disp_mol = mol.with_geometry(mol.geometry + delta_xyz)
            e_mm = _mm_energy(disp_mol, f"disp_{mi}_{d_ang}") - e_eq
            pct_err = ((e_mm - e_qm) / e_qm * 100.0) if abs(e_qm) > 1e-8 else 0.0
            displacements.append({"d_ang": d_ang, "e_qm": e_qm, "e_mm": e_mm, "pct_err": pct_err})
        results.append({"mode_idx": mi, "freq_cm1": freq_cm1, "displacements": displacements})
    elapsed = time.perf_counter() - t0
    return results, e_eq, elapsed


def _pes_distortion_summary(backend: Backend, case: BenchmarkCase, final_ff: ForceField) -> dict[str, Any]:
    if case.normal_modes is None:
        return {}
    from q2mm.backends.contracts import Capability

    if Capability.ENERGY not in backend.info.capabilities:
        return {}
    molecules = list(case.problem.molecules)
    modes = {k: np.asarray(v, dtype=float) for k, v in case.normal_modes.items()}
    distortions, _e_eq, elapsed = compute_distortions(molecules[0], final_ff, backend, modes)
    errors = [abs(d["pct_err"]) for m in distortions for d in m["displacements"]]
    return {
        "modes": distortions,
        "median_error_pct": float(np.median(errors)) if errors else 0.0,
        "max_error_pct": float(np.max(errors)) if errors else 0.0,
        "elapsed_s": elapsed,
    }


def _mean_ci95(samples: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float("nan"), 0.0
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, 0.0
    std = float(np.std(arr, ddof=1))
    if not math.isfinite(std) or std == 0.0:
        return mean, 0.0
    from scipy.stats import t

    ci95 = float(t.ppf(0.975, arr.size - 1) * std / math.sqrt(arr.size))
    return mean, ci95


def _score_interval_summary(
    initial: Sequence[float],
    final: Sequence[float],
    *,
    executor: Literal["python", "jax"],
    compare_endpoints: bool = True,
) -> dict[str, Any]:
    """Describe workflow samples without attributing them to the score of record."""
    if not initial or not final:
        return {}
    initial_mean, initial_ci95 = _mean_ci95(initial)
    final_mean, final_ci95 = _mean_ci95(final)
    statistics: dict[str, Any] = {
        "initial_optimizer_score_mean": initial_mean,
        "initial_optimizer_score_ci95": initial_ci95,
        "final_optimizer_score_mean": final_mean,
        "final_optimizer_score_ci95": final_ci95,
        "optimizer_samples_executor": executor,
        "optimizer_initial_sample_count": len(initial),
        "optimizer_final_sample_count": len(final),
        "optimizer_sample_statistics_version": 1,
    }
    if not compare_endpoints:
        statistics["optimizer_sample_comparison_omitted"] = "multiple_workflow_stages"
        return statistics
    statistics["optimizer_improvement_pct_mean"] = (
        100.0 * (1.0 - final_mean / initial_mean) if initial_mean > 0 else 0.0
    )
    statistics["optimizer_improvement_significant"] = bool(abs(final_mean - initial_mean) > (initial_ci95 + final_ci95))
    return statistics
