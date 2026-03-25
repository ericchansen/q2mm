"""Benchmark runner and result serialization for Q2MM.

Run a single (backend, optimizer, molecule) combination and produce a
:class:`BenchmarkResult` that can be saved to / loaded from JSON.
"""

from __future__ import annotations


import json
import time
from dataclasses import dataclass, field, fields as field_list, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from q2mm.constants import REAL_FREQUENCY_THRESHOLD

if TYPE_CHECKING:
    from q2mm.backends.base import MMEngine
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule


@dataclass
class BenchmarkResult:
    """Serializable result from a single benchmark run.

    Attributes:
        metadata (dict): Backend, optimizer, molecule, timestamp, source,
            and level_of_theory metadata.
        qm_reference (dict): QM reference data including frequencies_cm1
            and level_of_theory.
        default_ff (dict | None): Default force field results including
            frequencies_cm1, rmsd, and mae (if evaluated).
        seminario (dict | None): Seminario estimation results including
            frequencies_cm1, rmsd, mae, and elapsed_s.
        optimized (dict | None): Optimization results including
            frequencies_cm1, rmsd, mae, elapsed_s, n_eval, converged,
            initial_score, final_score, message, param_names, param_initial,
            and param_final.
        pes_distortion (dict | None): PES distortion results including
            modes (list), median_error_pct, max_error_pct, and elapsed_s.
        optimized_ff (ForceField | None): The optimized force field object.
            Not serialized to JSON — use :meth:`save_forcefields` to persist.

    """

    metadata: dict = field(default_factory=dict)
    qm_reference: dict = field(default_factory=dict)
    default_ff: dict | None = None
    seminario: dict | None = None
    optimized: dict | None = None
    pes_distortion: dict | None = None
    optimized_ff: ForceField | None = field(default=None, repr=False)

    def to_json(self, path: str | Path) -> None:
        """Save result to a JSON file.

        The :attr:`optimized_ff` field is excluded from serialization.
        Use :meth:`save_forcefields` to persist the force field object.

        Args:
            path (str | Path): Destination file path. Parent directories
                are created if they do not exist.

        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = asdict(self)
        data.pop("optimized_ff", None)

        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            raise TypeError(f"Cannot serialize {type(obj)}")

        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_convert)

    @classmethod
    def from_json(cls, path: str | Path) -> BenchmarkResult:
        """Load a result from a JSON file.

        Args:
            path (str | Path): Path to the JSON file.

        Returns:
            BenchmarkResult: Deserialized benchmark result.

        Raises:
            ValueError: If the JSON is missing required fields (``metadata``).

        """
        with open(path) as f:
            data = json.load(f)
        if "metadata" not in data or not isinstance(data.get("metadata"), dict):
            raise ValueError(f"Invalid BenchmarkResult JSON: 'metadata' key missing or not a dict in {path}")
        # Drop keys not in the dataclass (e.g. optimized_ff is never serialized)
        valid_fields = {f.name for f in field_list(cls)}
        data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**data)

    def save_forcefields(
        self,
        directory: str | Path,
        stem: str | None = None,
        molecule: Q2MMMolecule | None = None,
    ) -> list[Path]:
        """Save the optimized force field in all compatible native formats.

        Writes one file per format that is compatible with the force
        field's functional form (e.g. MM3 → ``.fld``, ``.prm``, ``.xml``
        but not ``.frcmod``; harmonic → ``.frcmod`` only).

        Args:
            directory (str | Path): Output directory (created if needed).
            stem (str | None): Base filename without extension. Defaults to
                ``'{backend}_{optimizer}'`` from metadata.
            molecule (Q2MMMolecule | None): Molecule for OpenMM XML residue
                generation. If ``None``, a minimal XML is written.

        Returns:
            list[Path]: Paths to the files that were successfully written.

        """
        if self.optimized_ff is None:
            return []

        from q2mm.models.ff_io import (
            save_amber_frcmod,
            save_mm3_fld,
            save_openmm_xml,
            save_tinker_prm,
        )

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if stem is None:
            backend = self.metadata.get("backend", "unk")
            optimizer = self.metadata.get("optimizer", "unk")
            stem = f"{backend}_{optimizer}".replace(" ", "_").replace("+", "_")

        ff = self.optimized_ff
        form = getattr(ff, "functional_form", None)
        if form is None:
            import logging

            logging.getLogger(__name__).warning(
                "Cannot save force field: functional_form is not set. "
                "Set it explicitly on the ForceField before running the benchmark."
            )
            return []
        form_value = form.value if hasattr(form, "value") else str(form)

        # Map format name → (saver function, extension, extra kwargs)
        savers: list[tuple[Any, str, dict[str, Any]]] = []
        if form_value == "mm3":
            savers.append((save_mm3_fld, ".fld", {}))
            savers.append((save_tinker_prm, ".prm", {}))
            savers.append((save_openmm_xml, ".xml", {"molecule": molecule}))
        if form_value == "harmonic":
            savers.append((save_amber_frcmod, ".frcmod", {}))

        saved: list[Path] = []
        for save_fn, ext, kwargs in savers:
            out_path = directory / (stem + ext)
            try:
                save_fn(ff, out_path, **kwargs)
                saved.append(out_path)
            except Exception as exc:
                import logging

                logging.getLogger(__name__).warning("Failed to save %s format for %s: %s", ext, stem, exc)

        return saved

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

        Args:
            frequencies_cm1 (list[float]): Vibrational frequencies in cm⁻¹.
            molecule (str): Human-readable molecule name.
            label (str): Source label (e.g., ``'upstream'``).
            level_of_theory (str): QM level of theory string.

        Returns:
            BenchmarkResult: Result populated with the given frequencies and
                metadata suitable for leaderboard comparison.

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


def frequency_rmsd(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Compute RMSD between two frequency arrays (truncates to shorter).

    Args:
        a (array-like): First array of frequencies.
        b (array-like): Second array of frequencies.

    Returns:
        float: Root-mean-square deviation between the two arrays.

    """
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    return float(np.sqrt(np.mean((arr_a[:n] - arr_b[:n]) ** 2)))


def frequency_mae(a: np.ndarray | list, b: np.ndarray | list) -> float:
    """Compute mean absolute error between two frequency arrays.

    Args:
        a (array-like): First array of frequencies.
        b (array-like): Second array of frequencies.

    Returns:
        float: Mean absolute error between the two arrays.

    """
    arr_a, arr_b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(arr_a), len(arr_b))
    return float(np.mean(np.abs(arr_a[:n] - arr_b[:n])))


def real_frequencies(freqs: np.ndarray | list, threshold: float = REAL_FREQUENCY_THRESHOLD) -> np.ndarray:
    """Extract and sort real (non-imaginary, non-translational) frequencies.

    Args:
        freqs (array-like): Input frequencies.
        threshold (float): Minimum frequency in cm⁻¹ to include. Modes
            below this are treated as translational/rotational.

    Returns:
        np.ndarray: Sorted array of frequencies above the threshold.

    """
    arr = np.asarray(freqs)
    return np.sort(arr[arr > threshold])


def _param_names(ff: ForceField) -> list[str]:
    """Build human-readable names for each parameter in get_param_vector() order.

    Args:
        ff (ForceField): Force field object with ``bonds``, ``angles``,
            ``torsions``, and ``vdws`` attributes.

    Returns:
        list[str]: Parameter name strings (e.g., ``'kb_C-H'``, ``'ka_H-C-H'``).

    """
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
    engine: MMEngine,
    molecule: Q2MMMolecule,
    qm_freqs: np.ndarray,
    qm_hessian: np.ndarray | None = None,
    normal_modes: dict | None = None,
    *,
    optimizer_method: str = "L-BFGS-B",
    optimizer_kwargs: dict[str, Any] | None = None,
    maxiter: int = 10_000,
    backend_name: str = "unknown",
    molecule_name: str = "unknown",
    level_of_theory: str = "unknown",
) -> BenchmarkResult:
    """Run a complete benchmark for one (backend, optimizer) combination.

    Args:
        engine (MMEngine): The MM backend engine to use.
        molecule (Q2MMMolecule): The molecule (at QM equilibrium geometry).
        qm_freqs (np.ndarray): QM reference frequencies (all real modes, cm⁻¹).
        qm_hessian (np.ndarray | None): QM Hessian matrix for Seminario
            estimation. If ``None``, Seminario step is skipped.
        normal_modes (dict | None): Pre-computed normal modes from
            ``load_normal_modes()``. If ``None``, PES distortion is skipped.
        optimizer_method (str): Scipy optimizer method (e.g.,
            ``'L-BFGS-B'``, ``'Nelder-Mead'``).
        optimizer_kwargs (dict[str, Any] | None): Extra keyword arguments
            for ``ScipyOptimizer`` (e.g., ``{'jac': 'analytical'}``).
        maxiter (int): Maximum optimizer iterations.
        backend_name (str): Human-readable backend name for result metadata.
        molecule_name (str): Human-readable molecule name.
        level_of_theory (str): QM level of theory string.

    Returns:
        BenchmarkResult: Complete result with all metrics.

    """
    from q2mm.models.forcefield import ForceField
    from q2mm.models.seminario import estimate_force_constants
    from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    qm_real = np.sort(np.asarray(qm_freqs)[np.asarray(qm_freqs) > REAL_FREQUENCY_THRESHOLD])

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
    default_params = default_ff.get_param_vector().tolist()
    result.default_ff = {
        "frequencies_cm1": default_real.tolist(),
        "rmsd": frequency_rmsd(qm_real, default_real),
        "mae": frequency_mae(qm_real, default_real),
        "param_values": default_params,
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
            "param_values": seminario_ff.get_param_vector().tolist(),
        }
    else:
        # No Hessian — create a default-based starting point
        seminario_ff = default_ff.copy()

    # --- Optimization ---
    ff_to_optimize = seminario_ff.copy()

    # Build reference data with correct data_idx mapping
    mm_all = engine.frequencies(molecule, ff_to_optimize)
    mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > REAL_FREQUENCY_THRESHOLD])

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

    result.optimized_ff = ff_to_optimize.copy()

    return result
