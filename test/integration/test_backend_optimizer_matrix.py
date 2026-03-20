"""Cross-product benchmark: every (backend x optimizer) on CH3F.

Runs the full Q2MM pipeline for each available backend and optimizer
method, collects timing and accuracy metrics, and produces:

1. A summary leaderboard (Table 1)
2. Detailed SI tables per combination (frequency progression, PES
   distortion, timing, parameters, convergence)
3. Saved JSON result files for later comparison

This is a slow test (~minutes) and requires ``--run-slow``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from q2mm.diagnostics.benchmark import BenchmarkResult, run_benchmark
from q2mm.diagnostics.pes_distortion import load_normal_modes
from q2mm.diagnostics.report import full_report

# ---- Paths ----

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = QM_REF / "ch3f-hessian.npy"
CH3F_FREQS = QM_REF / "ch3f-frequencies.txt"
CH3F_MODES = QM_REF / "ch3f-normal-modes.npz"

# ---- Backend discovery ----

_BACKENDS: list[tuple[str, type, str]] = []  # (name, engine_class, marker)

try:
    import openmm  # noqa: F401

    from q2mm.backends.mm.openmm import OpenMMEngine

    _BACKENDS.append(("OpenMM", OpenMMEngine, "openmm"))
except ImportError:
    pass

try:
    from q2mm.backends.mm.tinker import TinkerEngine

    if TinkerEngine().is_available():
        _BACKENDS.append(("Tinker", TinkerEngine, "tinker"))
except (ImportError, FileNotFoundError, OSError):
    pass

try:
    import jax  # noqa: F401

    from q2mm.backends.mm.jax_engine import JaxEngine

    _BACKENDS.append(("JAX", JaxEngine, "jax"))
except ImportError:
    pass

# ---- Optimizer configs ----

_OPTIMIZERS: list[tuple[str, dict]] = [
    ("L-BFGS-B", {"method": "L-BFGS-B"}),
    ("Nelder-Mead", {"method": "Nelder-Mead"}),
    ("Powell", {"method": "Powell"}),
]

# Add JAX analytical gradient variant if JAX is available
try:
    import jax  # noqa: F811

    _OPTIMIZERS.append(("L-BFGS-B+analytical", {"method": "L-BFGS-B", "jac": "analytical"}))
except ImportError:
    pass

# ---- Fixtures ----

_FIXTURE_FILES = [CH3F_XYZ, CH3F_HESS, CH3F_FREQS]
_missing = [str(f) for f in _FIXTURE_FILES if not f.exists()]


@pytest.mark.slow
@pytest.mark.skipif(bool(_missing), reason=f"Missing fixtures: {_missing}")
@pytest.mark.skipif(not _BACKENDS, reason="No MM backends available")
class TestBackendOptimizerMatrix:
    """Run all (backend x optimizer) combos on CH3F ground state."""

    @pytest.fixture(scope="class")
    def molecule(self):
        from q2mm.models.molecule import Q2MMMolecule

        return Q2MMMolecule.from_xyz(CH3F_XYZ, bond_tolerance=1.5)

    @pytest.fixture(scope="class")
    def qm_freqs(self):
        return np.loadtxt(CH3F_FREQS)

    @pytest.fixture(scope="class")
    def qm_hessian(self):
        return np.load(CH3F_HESS)

    @pytest.fixture(scope="class")
    def normal_modes(self):
        if CH3F_MODES.exists():
            return load_normal_modes(CH3F_MODES)
        return None

    @pytest.fixture(scope="class")
    def all_results(self, molecule, qm_freqs, qm_hessian, normal_modes, capsys):
        """Run every (backend x optimizer) combo, collect results."""
        results: list[BenchmarkResult] = []

        for backend_name, engine_cls, marker in _BACKENDS:
            engine = engine_cls()

            for opt_label, opt_config in _OPTIMIZERS:
                # Skip analytical gradients for non-JAX backends
                if opt_config.get("jac") == "analytical" and backend_name != "JAX":
                    continue

                combo = f"{backend_name} + {opt_label}"

                try:
                    method = opt_config["method"]
                    extra_kwargs = {k: v for k, v in opt_config.items() if k != "method"}

                    # Tinker has no Hessian -> skip PES distortion
                    modes = normal_modes
                    hess = qm_hessian
                    if backend_name == "Tinker":
                        modes = None

                    r = run_benchmark(
                        engine=engine,
                        molecule=molecule,
                        qm_freqs=qm_freqs,
                        qm_hessian=hess,
                        normal_modes=modes,
                        optimizer_method=method,
                        optimizer_kwargs=extra_kwargs,
                        maxiter=200,
                        backend_name=backend_name,
                        molecule_name="CH3F",
                        level_of_theory="B3LYP/6-31+G(d)",
                    )
                    results.append(r)
                    with capsys.disabled():
                        print(f"  {combo}: RMSD={r.optimized['rmsd']:.1f}  {r.optimized['elapsed_s']:.1f}s")

                except Exception as e:
                    with capsys.disabled():
                        print(f"  {combo}: FAILED ({e})", file=sys.stderr)
                    results.append(
                        BenchmarkResult(
                            metadata={
                                "backend": backend_name,
                                "optimizer": opt_label,
                                "molecule": "CH3F",
                                "source": "q2mm",
                            },
                            qm_reference={"frequencies_cm1": qm_freqs.tolist()},
                            optimized={
                                "frequencies_cm1": [],
                                "rmsd": float("nan"),
                                "mae": float("nan"),
                                "elapsed_s": 0.0,
                                "n_eval": 0,
                                "converged": False,
                                "initial_score": None,
                                "final_score": None,
                                "message": str(e),
                                "param_names": [],
                                "param_initial": [],
                                "param_final": [],
                            },
                        )
                    )

        return results

    @pytest.fixture(scope="class")
    def saved_results_dir(self, all_results):
        """Save all results to a temp directory."""
        tmpdir = Path(tempfile.mkdtemp(prefix="q2mm_benchmark_"))
        for i, r in enumerate(all_results):
            meta = r.metadata
            filename = f"{meta.get('backend', 'unk')}_{meta.get('optimizer', 'unk')}_{i:02d}.json"
            filename = filename.replace("+", "_").replace(" ", "_")
            r.to_json(tmpdir / filename)
        return tmpdir

    # ---- Tests ----

    def test_full_report(self, all_results, capsys):
        """Print the complete leaderboard + SI tables."""
        with capsys.disabled():
            print()
            full_report(all_results)

    def test_results_saved(self, saved_results_dir, capsys):
        """Verify all result JSONs were saved and can be reloaded."""
        json_files = list(saved_results_dir.glob("*.json"))
        assert len(json_files) > 0, "No result files saved"

        with capsys.disabled():
            print(f"\n  Results saved to: {saved_results_dir}")
            for jf in json_files:
                loaded = BenchmarkResult.from_json(jf)
                assert loaded.metadata.get("backend"), f"Missing backend in {jf.name}"
                print(f"    {jf.name}")

    def test_all_optimizations_improved(self, all_results):
        """Every combo that genuinely converged should improve over default.

        Known issue: Powell and Nelder-Mead don't support parameter bounds
        in scipy.  Without bounds, these methods can wander into non-physical
        parameter space, which invalidates the fixed-index frequency mapping
        used by the objective function.  The optimizer reports "converged"
        (the mapped objective decreased) but the actual sorted-frequency
        RMSD is worse because the mode ordering changed.

        Combos where this happens are collected and reported; the test only
        hard-fails if a *bounded* method (L-BFGS-B, trust-constr) degrades.
        """
        unbounded_methods = {"Powell", "Nelder-Mead"}
        hard_failures = []
        expected_failures = []

        for r in all_results:
            opt = r.optimized
            if not opt or not opt.get("converged"):
                continue
            combo = f"{r.metadata['backend']} + {r.metadata['optimizer']}"
            default_rmsd = r.default_ff["rmsd"] if r.default_ff else float("inf")
            opt_rmsd = opt["rmsd"]

            if opt_rmsd >= default_rmsd:
                msg = f"{combo}: RMSD {default_rmsd:.1f} -> {opt_rmsd:.1f} (worse)"
                if r.metadata["optimizer"] in unbounded_methods:
                    expected_failures.append(msg)
                else:
                    hard_failures.append(msg)

        if expected_failures:
            import warnings

            warnings.warn(
                "Unbounded optimizer(s) degraded RMSD (no bounds support):\n  " + "\n  ".join(expected_failures),
                stacklevel=1,
            )

        assert not hard_failures, "Bounded optimizer(s) degraded RMSD despite convergence:\n  " + "\n  ".join(
            hard_failures
        )

    def test_at_least_one_backend_ran(self, all_results):
        """At least one backend must have produced results."""
        assert len(all_results) > 0, "No benchmark results collected"
        converged = [r for r in all_results if r.optimized and r.optimized.get("converged")]
        assert len(converged) > 0, "No optimizations converged"
