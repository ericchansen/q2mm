"""Benchmark runner and result serialization for Q2MM.

Run a single (backend, optimizer, molecule) combination and produce a
:class:`BenchmarkResult` that can be saved to / loaded from JSON.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkResult:
    """Serializable result from a single benchmark run.

    Attributes
    ----------
    metadata : dict
        backend, optimizer, molecule, timestamp, source, level_of_theory.
    qm_reference : dict
        frequencies_cm1, level_of_theory.
    default_ff : dict or None
        frequencies_cm1, rmsd, mae (if evaluated).
    seminario : dict or None
        frequencies_cm1, rmsd, mae, elapsed_s.
    optimized : dict or None
        frequencies_cm1, rmsd, mae, elapsed_s, n_eval, converged,
        initial_score, final_score, message, param_names, param_initial,
        param_final.
    pes_distortion : dict or None
        modes (list), median_error_pct, max_error_pct, elapsed_s.
    """

    metadata: dict = field(default_factory=dict)
    qm_reference: dict = field(default_factory=dict)
    default_ff: dict | None = None
    seminario: dict | None = None
    optimized: dict | None = None
    pes_distortion: dict | None = None

    def to_json(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            raise TypeError(f"Cannot serialize {type(obj)}")

        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=_convert)

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkResult:
        """Load a result from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_upstream(
        cls,
        frequencies_cm1: list[float],
        *,
        molecule: str = "unknown",
        label: str = "upstream",
        level_of_theory: str = "unknown",
    ) -> BenchmarkResult:
        """Create a result from externally-computed frequencies.

        Use this to import results from the upstream/legacy q2mm code
        or any other source for comparison.
        """
        return cls(
            metadata={
                "backend": label,
                "optimizer": "external",
                "molecule": molecule,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": label,
            },
            qm_reference={"level_of_theory": level_of_theory},
            optimized={
                "frequencies_cm1": list(frequencies_cm1),
                "rmsd": None,
                "mae": None,
                "elapsed_s": None,
                "n_eval": None,
                "converged": None,
                "initial_score": None,
                "final_score": None,
                "message": "imported from external source",
                "param_names": [],
                "param_initial": [],
                "param_final": [],
            },
        )


def frequency_rmsd(a, b) -> float:
    """RMSD between two frequency arrays (truncates to shorter)."""
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    return float(np.sqrt(np.mean((arr_a[:n] - arr_b[:n]) ** 2)))


def frequency_mae(a, b) -> float:
    """Mean absolute error between two frequency arrays."""
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    return float(np.mean(np.abs(arr_a[:n] - arr_b[:n])))


def real_frequencies(freqs, threshold: float = 50.0) -> np.ndarray:
    """Extract and sort real (non-imaginary, non-translational) frequencies."""
    arr = np.asarray(freqs)
    return np.sort(arr[arr > threshold])


def _param_names(ff) -> list[str]:
    """Build human-readable names for each parameter in get_param_vector() order."""
    names = []
    for b in ff.bonds:
        label = "-".join(b.key) + (f"[{b.env_id}]" if b.env_id else "")
        names.append(f"kb_{label}")
        names.append(f"r0_{label}")
    for a in ff.angles:
        label = "-".join(a.key) + (f"[{a.env_id}]" if a.env_id else "")
        names.append(f"ka_{label}")
        names.append(f"th0_{label}")
    for t in ff.torsions:
        label = "-".join(t.elements) + f"_n{t.periodicity}"
        names.append(f"kt_{label}")
    for v in ff.vdws:
        label = v.atom_type or v.element
        names.append(f"rvdw_{label}")
        names.append(f"evdw_{label}")
    return names


def run_benchmark(
    engine,
    molecule,
    qm_freqs: np.ndarray,
    qm_hessian: np.ndarray | None = None,
    normal_modes: dict | None = None,
    *,
    optimizer_method: str = "L-BFGS-B",
    optimizer_kwargs: dict[str, Any] | None = None,
    maxiter: int = 200,
    backend_name: str = "unknown",
    molecule_name: str = "unknown",
    level_of_theory: str = "unknown",
) -> BenchmarkResult:
    """Run a complete benchmark for one (backend, optimizer) combination.

    Parameters
    ----------
    engine : MMEngine
        The MM backend engine to use.
    molecule : Q2MMMolecule
        The molecule (at QM equilibrium geometry).
    qm_freqs : ndarray
        QM reference frequencies (all real modes, cm-1).
    qm_hessian : ndarray, optional
        QM Hessian matrix for Seminario estimation. If None, Seminario
        step is skipped.
    normal_modes : dict, optional
        Pre-computed normal modes from ``load_normal_modes()``. If None,
        PES distortion is skipped.
    optimizer_method : str
        Scipy optimizer method (e.g. "L-BFGS-B", "Nelder-Mead").
    optimizer_kwargs : dict, optional
        Extra kwargs for ScipyOptimizer (e.g. {"jac": "analytical"}).
    maxiter : int
        Maximum optimizer iterations.
    backend_name : str
        Human-readable backend name for the result metadata.
    molecule_name : str
        Human-readable molecule name.
    level_of_theory : str
        QM level of theory string.

    Returns
    -------
    BenchmarkResult
        Complete result with all metrics.
    """
    from q2mm.models.forcefield import ForceField
    from q2mm.models.seminario import estimate_force_constants
    from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    qm_real = np.sort(np.asarray(qm_freqs)[np.asarray(qm_freqs) > 50.0])

    result = BenchmarkResult(
        metadata={
            "backend": backend_name,
            "optimizer": optimizer_method,
            "molecule": molecule_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "q2mm",
            "optimizer_kwargs": {k: str(v) for k, v in optimizer_kwargs.items()},
        },
        qm_reference={
            "frequencies_cm1": qm_real.tolist(),
            "level_of_theory": level_of_theory,
        },
    )

    # --- Default FF baseline ---
    default_ff = ForceField.create_for_molecule(molecule, name=f"{molecule_name} default")
    default_freqs_all = engine.frequencies(molecule, default_ff)
    default_real = real_frequencies(default_freqs_all)
    result.default_ff = {
        "frequencies_cm1": default_real.tolist(),
        "rmsd": frequency_rmsd(qm_real, default_real),
        "mae": frequency_mae(qm_real, default_real),
    }

    # --- Seminario estimation ---
    seminario_ff = None
    if qm_hessian is not None:
        mol_h = molecule.with_hessian(qm_hessian)
        t0 = time.perf_counter()
        seminario_ff = estimate_force_constants(mol_h)
        sem_elapsed = time.perf_counter() - t0

        sem_freqs_all = engine.frequencies(molecule, seminario_ff)
        sem_real = real_frequencies(sem_freqs_all)
        result.seminario = {
            "frequencies_cm1": sem_real.tolist(),
            "rmsd": frequency_rmsd(qm_real, sem_real),
            "mae": frequency_mae(qm_real, sem_real),
            "elapsed_s": sem_elapsed,
        }
    else:
        # No Hessian — create a default-based starting point
        seminario_ff = default_ff.copy()

    # --- Optimization ---
    ff_to_optimize = seminario_ff.copy()

    # Build reference data with correct data_idx mapping
    mm_all = engine.frequencies(molecule, ff_to_optimize)
    mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])

    ref = ReferenceData()
    n = min(len(qm_real), len(mm_real_indices))
    for k in range(n):
        ref.add_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, molecule_idx=0)

    obj = ObjectiveFunction(ff_to_optimize, engine, [molecule], ref)

    opt_kwargs = {"method": optimizer_method, "maxiter": maxiter, "verbose": False}
    opt_kwargs.update(optimizer_kwargs)
    opt = ScipyOptimizer(**opt_kwargs)

    t0 = time.perf_counter()
    opt_result = opt.optimize(obj)
    opt_elapsed = time.perf_counter() - t0

    opt_freqs_all = engine.frequencies(molecule, ff_to_optimize)
    opt_real = real_frequencies(opt_freqs_all)

    # Collect parameter info
    param_names = _param_names(ff_to_optimize)
    param_final = ff_to_optimize.get_param_vector().tolist()
    param_initial = list(opt_result.initial_params) if opt_result.initial_params is not None else []

    result.optimized = {
        "frequencies_cm1": opt_real.tolist(),
        "rmsd": frequency_rmsd(qm_real, opt_real),
        "mae": frequency_mae(qm_real, opt_real),
        "elapsed_s": opt_elapsed,
        "n_eval": opt_result.n_evaluations,
        "converged": opt_result.success,
        "initial_score": opt_result.initial_score,
        "final_score": opt_result.final_score,
        "message": opt_result.message,
        "param_names": param_names,
        "param_initial": list(param_initial),
        "param_final": param_final,
    }

    # --- PES distortion (if normal modes available and engine supports it) ---
    if normal_modes is not None:
        from q2mm.diagnostics.pes_distortion import compute_distortions

        distortion_results, _, dist_elapsed = compute_distortions(molecule, ff_to_optimize, engine, normal_modes)

        all_errs = []
        for m in distortion_results:
            for d in m["displacements"]:
                all_errs.append(abs(d["pct_err"]))

        result.pes_distortion = {
            "modes": distortion_results,
            "median_error_pct": float(np.median(all_errs)) if all_errs else 0.0,
            "max_error_pct": float(np.max(all_errs)) if all_errs else 0.0,
            "elapsed_s": dist_elapsed,
        }

    return result
