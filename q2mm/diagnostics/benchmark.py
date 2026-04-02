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
    from q2mm.diagnostics.systems import SystemData
    from q2mm.models.forcefield import ForceField
    from q2mm.models.molecule import Q2MMMolecule


# ── Engine display name → (engine_key, ff_label) mapping ────────────
# Engine.name returns human-readable strings like "JAX (gpu)",
# "JAX-MD (OPLSAA, cpu)", "OpenMM (CUDA)", "Tinker".  We need to
# decompose those into shell-safe filename segments.
_ENGINE_MAP: dict[str, tuple[str, str]] = {
    "jax": ("jax", "harmonic"),
    "jax-md": ("jax-md", "oplsaa"),
    "openmm": ("openmm", "mm3"),
    "tinker": ("tinker", "mm3"),
}


def _parse_engine_name(display_name: str) -> tuple[str, str, str]:
    """Parse an engine display name into (engine, ff, device).

    Examples::

        >>> _parse_engine_name("JAX (gpu)")
        ('jax', 'harmonic', 'gpu')
        >>> _parse_engine_name("JAX-MD (OPLSAA, cpu)")
        ('jax-md', 'oplsaa', 'cpu')
        >>> _parse_engine_name("OpenMM (CUDA)")
        ('openmm', 'mm3', 'gpu')
        >>> _parse_engine_name("Tinker")
        ('tinker', 'mm3', 'cpu')

    """
    import re

    name = display_name.strip()
    # Extract parenthesised suffix if present
    m = re.match(r"^([^(]+?)(?:\s*\(([^)]*)\))?$", name)
    if not m:
        return name.lower(), "unk", "cpu"

    base = m.group(1).strip().lower()
    paren = m.group(2) or ""
    parts = [p.strip().lower() for p in paren.split(",") if p.strip()]

    # Look up engine/ff from the base key
    engine_key, ff_label = _ENGINE_MAP.get(base, (base, "unk"))

    # Determine device from parenthesised parts
    device = "cpu"
    for p in parts:
        if p in ("gpu", "cpu"):
            device = p
        elif p == "cuda":
            device = "gpu"
        # Skip FF labels already handled by _ENGINE_MAP (e.g. "oplsaa")

    return engine_key, ff_label, device


def benchmark_stem(metadata: dict) -> str:
    """Build a shell-safe, self-describing filename stem from metadata.

    Pattern: ``{system}_{engine}_{form}_{device}_{optimizer}``

    The ``form`` segment is taken from the explicit ``functional_form``
    metadata key when present, falling back to the form inferred from
    the backend name string.

    All segments are lowercase.  Underscores separate segments; hyphens
    only appear within naturally hyphenated names (e.g. ``jax-md``,
    ``rh-enamide``, ``nelder-mead``).

    Examples::

        >>> benchmark_stem({"molecule": "CH3F", "backend": "JAX (gpu)", "optimizer": "L-BFGS-B", "functional_form": "harmonic"})
        'ch3f_jax_harmonic_gpu_lbfgsb'
        >>> benchmark_stem({"molecule": "CH3F", "backend": "JAX (gpu)", "optimizer": "L-BFGS-B", "functional_form": "mm3"})
        'ch3f_jax_mm3_gpu_lbfgsb'
        >>> benchmark_stem({"molecule": "Rh-enamide", "backend": "JAX-MD (OPLSAA, cpu)", "optimizer": "Powell"})
        'rh-enamide_jax-md_oplsaa_cpu_powell'

    """
    system = metadata.get("molecule", "unk").lower().replace(" ", "-")
    backend = metadata.get("backend", "unk")
    engine, ff_from_backend, device = _parse_engine_name(backend)
    # Allow explicit device override from metadata
    device = metadata.get("device", device).lower()

    # Prefer explicit functional_form over the one parsed from backend name.
    # Guard against None values (e.g. from JSON null) before lowercasing.
    form = (metadata.get("functional_form") or ff_from_backend).lower()

    optimizer = metadata.get("optimizer", "unk").lower()
    if optimizer == "l-bfgs-b":
        optimizer = "lbfgsb"
    optimizer = optimizer.replace(" ", "-")

    return f"{system}_{engine}_{form}_{device}_{optimizer}"


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
            stem = benchmark_stem(self.metadata)

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


def run_combo(
    engine: MMEngine,
    sys_data: SystemData,
    *,
    optimizer_method: str = "L-BFGS-B",
    optimizer_kwargs: dict[str, Any] | None = None,
    maxiter: int = 10_000,
    backend_name: str = "unknown",
) -> BenchmarkResult:
    """Run one benchmark combination on *any* system (N ≥ 1 molecules).

    This is the single execution path for all benchmark runs — there is no
    separate single-molecule vs multi-molecule code.

    Args:
        engine: The MM backend engine to use.
        sys_data: Fully-loaded system data (molecules, forcefield, freq_ref).
        optimizer_method: Scipy optimizer method (e.g. ``'L-BFGS-B'``).
        optimizer_kwargs: Extra keyword arguments for ``ScipyOptimizer``.
        maxiter: Maximum optimizer iterations.
        backend_name: Human-readable backend name for result metadata.

    Returns:
        BenchmarkResult with all available metrics.

    """
    from q2mm.optimizers.objective import ObjectiveFunction

    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    ff = sys_data.forcefield.copy()
    molecule_name = sys_data.metadata.get("molecule_name", "unknown")
    level_of_theory = sys_data.metadata.get("level_of_theory", "unknown")

    # Aggregate QM real frequencies (sorted) across all molecules
    all_qm_real = np.sort(np.concatenate(sys_data.qm_freqs_per_mol))

    # Aggregate MM frequencies across all molecules for initial RMSD
    all_mm_real_init: list[float] = []
    for mol in sys_data.molecules:
        mm_freqs = engine.frequencies(mol, ff)
        all_mm_real_init.extend(real_frequencies(mm_freqs).tolist())
    all_mm_real_init_arr = np.array(sorted(all_mm_real_init))

    n_init = min(len(all_qm_real), len(all_mm_real_init_arr))
    initial_rmsd = frequency_rmsd(all_qm_real[:n_init], all_mm_real_init_arr[:n_init])

    seminario_params = ff.get_param_vector().copy()
    param_names = _param_names(ff)

    result = BenchmarkResult(
        metadata={
            "backend": backend_name,
            "optimizer": optimizer_method,
            "molecule": molecule_name,
            "functional_form": ff.functional_form.value if ff.functional_form else "unknown",
            "n_molecules": len(sys_data.molecules),
            "source": "q2mm",
            "level_of_theory": level_of_theory,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        qm_reference={
            "frequencies_cm1": all_qm_real.tolist(),
            "level_of_theory": level_of_theory,
        },
    )

    # Initial (Seminario) state
    obj = ObjectiveFunction(ff, engine, sys_data.molecules, sys_data.freq_ref)
    initial_score = obj(seminario_params)

    result.seminario = {
        "rmsd": initial_rmsd,
        "frequencies_cm1": all_mm_real_init_arr.tolist(),
        "param_values": seminario_params.tolist(),
        "param_names": param_names,
        "score": initial_score,
    }

    # Optimize — dispatch to cycling or scipy
    if optimizer_method == "cycling":
        from q2mm.optimizers.cycling import OptimizationLoop

        loop_kwargs: dict[str, Any] = {
            "objective": obj,
            "max_params": optimizer_kwargs.get("max_params", 3),
            "convergence": optimizer_kwargs.get("convergence", 0.01),
            "max_cycles": optimizer_kwargs.get("max_cycles", 10),
            "verbose": False,
        }
        # Only override per-pass maxiter if caller explicitly provides them;
        # otherwise let OptimizationLoop use its own defaults (200).
        if "full_maxiter" in optimizer_kwargs:
            loop_kwargs["full_maxiter"] = optimizer_kwargs["full_maxiter"]
        if "simp_maxiter" in optimizer_kwargs:
            loop_kwargs["simp_maxiter"] = optimizer_kwargs["simp_maxiter"]

        loop = OptimizationLoop(**loop_kwargs)
        t0 = time.perf_counter()
        loop_result = loop.run()
        opt_elapsed = time.perf_counter() - t0

        n_eval = loop_result.n_eval
        converged = loop_result.success
        opt_initial_score = loop_result.initial_score
        opt_final_score = loop_result.final_score
        opt_message = loop_result.message
        extra_opt_data = {
            "n_cycles": loop_result.n_cycles,
            "cycle_scores": loop_result.cycle_scores,
        }
    else:
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        opt_kwargs = {"method": optimizer_method, "maxiter": maxiter, "verbose": False, "jac": "auto"}
        opt_kwargs.update(optimizer_kwargs)
        opt = ScipyOptimizer(**opt_kwargs)

        t0 = time.perf_counter()
        opt_result = opt.optimize(obj)
        opt_elapsed = time.perf_counter() - t0

        n_eval = opt_result.n_evaluations
        converged = opt_result.success
        opt_initial_score = opt_result.initial_score
        opt_final_score = opt_result.final_score
        opt_message = opt_result.message
        extra_opt_data = {}

    # Final aggregate frequencies and RMSD
    all_mm_real_final: list[float] = []
    for mol in sys_data.molecules:
        mm_freqs = engine.frequencies(mol, ff)
        all_mm_real_final.extend(real_frequencies(mm_freqs).tolist())
    all_mm_real_final_arr = np.array(sorted(all_mm_real_final))

    n_final = min(len(all_qm_real), len(all_mm_real_final_arr))
    final_rmsd = frequency_rmsd(all_qm_real[:n_final], all_mm_real_final_arr[:n_final])

    param_final = ff.get_param_vector().tolist()

    result.optimized = {
        "frequencies_cm1": all_mm_real_final_arr.tolist(),
        "rmsd": final_rmsd,
        "mae": frequency_mae(all_qm_real[:n_final], all_mm_real_final_arr[:n_final]),
        "elapsed_s": opt_elapsed,
        "n_eval": n_eval,
        "converged": converged,
        "initial_score": opt_initial_score,
        "final_score": opt_final_score,
        "message": opt_message,
        "param_names": param_names,
        "param_initial": seminario_params.tolist(),
        "param_final": param_final,
        **extra_opt_data,
    }

    # PES distortion (if the system provides normal modes)
    if sys_data.normal_modes is not None:
        from q2mm.diagnostics.pes_distortion import compute_distortions

        distortion_results, _, dist_elapsed = compute_distortions(
            sys_data.molecules[0],
            ff,
            engine,
            sys_data.normal_modes,
        )
        all_errs: list[float] = []
        for m in distortion_results:
            for d in m["displacements"]:
                all_errs.append(abs(d["pct_err"]))
        result.pes_distortion = {
            "modes": distortion_results,
            "median_error_pct": float(np.median(all_errs)) if all_errs else 0.0,
            "max_error_pct": float(np.max(all_errs)) if all_errs else 0.0,
            "elapsed_s": dist_elapsed,
        }

    result.optimized_ff = ff.copy()
    return result
