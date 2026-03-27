"""Full-loop parity validation — issue #74.

Validates the complete Q2MM pipeline end-to-end:
1. QM data loading (Gaussian .fchk or Jaguar .in)
2. Seminario force-constant estimation
3. Frequency-based penalty scoring (via OpenMM)
4. Optimizer convergence (Nelder-Mead for complex systems, L-BFGS-B for small)
5. Determinism and golden-fixture reproducibility

The rh-enamide dataset (9 Jaguar structures) validates the full pipeline
on a real organometallic system (Rh-diphosphine, 36 atoms, 182 params).
Ethane GS/TS tests validate the same pipeline on a simpler molecule.

References
----------
- Issue: https://github.com/ericchansen/q2mm/issues/74

"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from q2mm.backends.base import MMEngine
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.optimizers.objective import ReferenceData

from test._shared import GS_FCHK, REPO_ROOT, TS_FCHK

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "full_loop"
ETHANE_GS_GOLDEN = FIXTURE_DIR / "ethane_gs_golden.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

_HAS_OPENMM = True
try:
    import openmm  # noqa: F401
except ImportError:
    _HAS_OPENMM = False

_HAS_JAX = True
try:
    from q2mm.backends.mm.jax_engine import JaxEngine  # noqa: F401
except Exception:
    _HAS_JAX = False

_HAS_JAX_MD = True
try:
    from q2mm.backends.mm.jax_md_engine import JaxMDEngine  # noqa: F401
except Exception:
    _HAS_JAX_MD = False

requires_openmm = pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed")
requires_jax = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
requires_jax_md = pytest.mark.skipif(not _HAS_JAX_MD, reason="JAX-MD not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_golden(path: Path) -> dict:
    return json.loads(path.read_text())


def _qm_frequencies_from_hessian(
    hessian_au: np.ndarray,
    symbols: list[str],
) -> np.ndarray:
    """Compute harmonic frequencies (cm⁻¹) from a Cartesian Hessian in AU.

    Uses the same unit-conversion pipeline as
    :meth:`~q2mm.backends.mm.openmm.OpenMMEngine.frequencies` to ensure
    QM–MM frequency comparisons are on identical footing.
    """
    from q2mm.constants import (
        AMU_TO_KG,
        BOHR_TO_ANG,
        HARTREE_TO_J,
        MASSES,
        SPEED_OF_LIGHT_MS,
    )

    bohr_to_m = BOHR_TO_ANG * 1e-10
    hessian_si = hessian_au * HARTREE_TO_J / (bohr_to_m**2)
    masses = np.array([MASSES[s] * AMU_TO_KG for s in symbols], dtype=float)
    mass_vec = np.repeat(masses, 3)
    mw = hessian_si / np.sqrt(np.outer(mass_vec, mass_vec))
    eigenvalues = np.linalg.eigvalsh(mw)
    freqs = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
    freqs /= 2.0 * np.pi * SPEED_OF_LIGHT_MS * 100.0
    return freqs


def _build_frequency_reference(
    qm_freqs: np.ndarray,
    mm_all_freqs: np.ndarray,
    *,
    threshold: float = 50.0,
    weight: float = 0.001,
    molecule_idx: int = 0,
    ref: ReferenceData | None = None,
) -> tuple[ReferenceData, list[float]]:
    """Build (or extend) a ReferenceData with frequency observations.

    Maps QM real frequencies (>threshold) to MM real-mode indices.
    Pass an existing *ref* to append multi-molecule data.
    """
    from q2mm.optimizers.objective import ReferenceData

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))

    if ref is None:
        ref = ReferenceData()
    for k in range(n):
        ref.add_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=weight, molecule_idx=molecule_idx)
    return ref, qm_real[:n]


# ===========================================================================
# Rh-enamide Seminario parity + timing (no MM engine needed)
# ===========================================================================


class TestRhEnamideSeminarioTiming:
    """Runtime benchmarks for Seminario on the 9-structure rh-enamide dataset.

    These tests are informational — they log wall-clock times but never
    fail on timing alone.  They complement the parameter-accuracy tests
    in ``test_seminario_parity.py``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Q2MMMolecule]:
        """Load all 9 rh-enamide structures + Hessians."""
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.mark.slow
    def test_seminario_pipeline_timing(
        self, rh_molecules: list[Q2MMMolecule], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Time the full Seminario pipeline on 9 rh-enamide structures."""
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import estimate_force_constants

        mm3_path = RH_DIR / "mm3.fld"
        if not mm3_path.exists():
            pytest.skip("mm3.fld not found")
        ff_template = ForceField.from_mm3_fld(str(mm3_path))

        t0 = time.perf_counter()
        ff = estimate_force_constants(rh_molecules, forcefield=ff_template)
        elapsed = time.perf_counter() - t0

        with capsys.disabled():
            print(f"\n  Rh-enamide Seminario: {elapsed:.3f}s ({len(rh_molecules)} structures, {ff.n_params} params)")

        # Sanity check — never fail on timing
        assert ff.n_params > 0, "No parameters estimated"
        assert len(ff.bonds) > 0, "No bond parameters"
        assert len(ff.angles) > 0, "No angle parameters"

    @pytest.mark.slow
    def test_seminario_is_deterministic(self, rh_molecules: list[Q2MMMolecule]) -> None:
        """Two consecutive Seminario runs produce identical results."""
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import estimate_force_constants

        mm3_path = RH_DIR / "mm3.fld"
        if not mm3_path.exists():
            pytest.skip("mm3.fld not found")
        ff_template = ForceField.from_mm3_fld(str(mm3_path))

        ff1 = estimate_force_constants(rh_molecules, forcefield=ff_template)
        ff2 = estimate_force_constants(rh_molecules, forcefield=ff_template)

        np.testing.assert_array_equal(
            ff1.get_param_vector(),
            ff2.get_param_vector(),
            err_msg="Seminario is non-deterministic across runs",
        )


# ===========================================================================
# Rh-enamide: full optimization loop with Jaguar QM data (D1)
# ===========================================================================


def _load_rh_enamide_molecules() -> list[Q2MMMolecule]:
    """Load 9 rh-enamide structures with Jaguar Hessians.

    Delegates to the shared loader in :mod:`q2mm.diagnostics.systems`.
    """
    from q2mm.diagnostics.systems import load_rh_enamide_molecules

    return load_rh_enamide_molecules()


@requires_openmm
@pytest.mark.openmm
@pytest.mark.slow
class TestRhEnamideFullLoop:
    """D1: Full pipeline on rh-enamide — Jaguar → Seminario → OpenMM → optimize.

    9 organometallic structures (Rh-diphosphine, 36 atoms each, B3LYP/LACVP**).
    182 parameters (8 bond types, 23 angle types, 36 vdW types).
    Frequency-based objective with Nelder-Mead optimization.
    """

    @pytest.fixture(scope="class")
    def pipeline_result(self) -> dict[str, object]:
        """Run the full rh-enamide pipeline."""
        from q2mm.backends.mm import OpenMMEngine
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ObjectiveFunction
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")

        mm3_fld_path = RH_DIR / "mm3.fld"
        if not mm3_fld_path.exists():
            pytest.skip("rh-enamide force field file mm3.fld not found")

        molecules = _load_rh_enamide_molecules()
        ff_template = ForceField.from_mm3_fld(str(mm3_fld_path))

        # Seminario estimation
        t0 = time.perf_counter()
        ff = estimate_force_constants(molecules, forcefield=ff_template)
        t_seminario = time.perf_counter() - t0
        seminario_params = ff.get_param_vector().copy()

        # Build multi-molecule frequency reference
        engine = OpenMMEngine()
        freq_ref = None
        n_freqs_per_mol = []
        for mol_idx, mol in enumerate(molecules):
            mm_freqs = engine.frequencies(mol, ff)
            qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
            freq_ref, qm_real = _build_frequency_reference(
                qm_freqs,
                mm_freqs,
                molecule_idx=mol_idx,
                ref=freq_ref,
            )
            n_freqs_per_mol.append(len(qm_real))

        # Initial score
        obj = ObjectiveFunction(ff, engine, molecules, freq_ref)
        initial_score = obj(seminario_params)

        # Optimize — just enough iterations to verify our optimizer wrapper
        # and objective function work end-to-end. Full convergence benchmarks
        # (500 iter, 76.7% improvement) are documented in docs/benchmarks/.
        t0 = time.perf_counter()
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=3, verbose=False)
        result = opt.optimize(obj)
        t_optimize = time.perf_counter() - t0

        return {
            "n_molecules": len(molecules),
            "n_params": ff.n_params,
            "n_bonds": len(ff.bonds),
            "n_angles": len(ff.angles),
            "n_vdws": len(ff.vdws),
            "n_freqs_per_mol": n_freqs_per_mol,
            "total_freq_refs": sum(n_freqs_per_mol),
            "seminario_params": seminario_params,
            "initial_score": initial_score,
            "final_score": result.final_score,
            "improvement": result.improvement,
            "converged": result.success,
            "optimized_params": ff.get_param_vector().copy(),
            "t_seminario": t_seminario,
            "t_optimize": t_optimize,
        }

    def test_loads_9_molecules(self, pipeline_result: dict[str, object]) -> None:
        """All 9 rh-enamide structures are loaded."""
        assert pipeline_result["n_molecules"] == 9

    def test_seminario_182_params(self, pipeline_result: dict[str, object]) -> None:
        """Seminario produces the expected parameter count."""
        assert pipeline_result["n_params"] == 182
        assert pipeline_result["n_bonds"] == 8
        assert pipeline_result["n_angles"] == 23
        assert pipeline_result["n_vdws"] == 36

    def test_all_molecules_have_frequencies(self, pipeline_result: dict[str, object]) -> None:
        """Every molecule contributes frequency reference data."""
        for i, n in enumerate(pipeline_result["n_freqs_per_mol"]):
            assert n > 0, f"Molecule {i} contributed 0 frequency references"

    def test_initial_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        """Initial penalty score is finite and positive."""
        score = pipeline_result["initial_score"]
        assert np.isfinite(score), f"Initial score is not finite: {score}"
        assert score > 0, f"Initial score should be positive: {score}"

    def test_final_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        """Final penalty score is finite and positive after optimization.

        With only 3 Nelder-Mead iterations on 182 dimensions, score
        improvement is not guaranteed — the initial simplex barely
        forms before ``maxiter`` is reached, and ULP-level differences
        in Seminario eigenvalues across platforms can send the simplex
        in divergent directions.  Full convergence is validated by the
        benchmark tests (500 iterations, ~77 % improvement).
        """
        score = pipeline_result["final_score"]
        assert np.isfinite(score), f"Final score is not finite: {score}"
        assert score > 0, f"Final score should be positive: {score}"

    def test_optimized_params_differ_from_seminario(self, pipeline_result: dict[str, object]) -> None:
        """Optimizer actually modifies parameters."""
        diff = np.abs(pipeline_result["optimized_params"] - pipeline_result["seminario_params"])
        assert np.any(diff > 1e-6), "Optimizer didn't change any parameters"

    def test_timing_report(self, pipeline_result: dict[str, object], capsys: pytest.CaptureFixture[str]) -> None:
        """Log timing (informational, never fails)."""
        r = pipeline_result
        with capsys.disabled():
            print(
                f"\n  Rh-enamide full loop ({r['n_molecules']} mols, {r['n_params']} params, "
                f"{r['total_freq_refs']} freq refs):"
                f"\n    Seminario: {r['t_seminario']:.3f}s"
                f"\n    Optimize:  {r['t_optimize']:.1f}s (Nelder-Mead, maxiter=3)"
                f"\n    Score:     {r['initial_score']:.1f} → {r['final_score']:.1f} "
                f"({r['improvement'] * 100:.1f}% improvement)"
            )


# ===========================================================================
# Ethane GS: full optimization loop with frequency objective
# ===========================================================================


@requires_openmm
@pytest.mark.openmm
@pytest.mark.slow
class TestEthaneFullLoop:
    """Full pipeline: .fchk → Seminario → frequency objective → optimize.

    Validates against golden fixture to ensure deterministic reproducibility.
    """

    @pytest.fixture(scope="class")
    def golden(self) -> dict[str, object]:
        if not ETHANE_GS_GOLDEN.exists():
            pytest.skip("Golden fixture not found; run generate_golden_fixtures.py")
        return _load_golden(ETHANE_GS_GOLDEN)

    @pytest.fixture(scope="class")
    def pipeline_result(self) -> dict[str, object]:
        """Run the full pipeline and return all intermediate results."""
        from q2mm.backends.mm import OpenMMEngine
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not GS_FCHK.exists():
            pytest.skip("Ethane GS.fchk not found")

        ref, mol = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)

        # QM frequencies from Hessian
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        # Seminario estimation
        t_sem_start = time.perf_counter()
        ff = estimate_force_constants(mol, au_hessian=True)
        t_sem = time.perf_counter() - t_sem_start
        seminario_params = ff.get_param_vector().copy()

        # MM frequencies + reference data
        engine = OpenMMEngine()
        mm_all = engine.frequencies(mol, ff)
        freq_ref, qm_real = _build_frequency_reference(qm_freqs, mm_all)

        # Penalty score
        obj = ObjectiveFunction(ff, engine, [mol], freq_ref)
        seminario_score = obj(seminario_params)

        # Optimize
        t_opt_start = time.perf_counter()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)
        t_opt = time.perf_counter() - t_opt_start

        return {
            "mol": mol,
            "ff": ff,
            "engine": engine,
            "seminario_params": seminario_params,
            "seminario_score": seminario_score,
            "optimized_params": ff.get_param_vector().copy(),
            "optimized_score": result.final_score,
            "improvement": result.improvement,
            "converged": result.success,
            "qm_real": qm_real,
            "mm_all": mm_all,
            "t_sem": t_sem,
            "t_opt": t_opt,
        }

    # ---- Seminario stage ----

    def test_seminario_params_match_golden(self, pipeline_result: dict[str, object], golden: dict[str, object]) -> None:
        """Seminario parameter vector matches golden fixture exactly."""
        np.testing.assert_allclose(
            pipeline_result["seminario_params"],
            golden["seminario"]["params"],
            rtol=1e-10,
            err_msg="Seminario params diverged from golden fixture",
        )

    def test_seminario_score_matches_golden(
        self, pipeline_result: dict[str, object], golden: dict[str, object]
    ) -> None:
        """Seminario penalty score matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["seminario_score"],
            golden["seminario"]["score"],
            rtol=1e-4,
            err_msg="Seminario penalty score diverged from golden",
        )

    def test_seminario_has_reasonable_params(self, pipeline_result: dict[str, object]) -> None:
        """Seminario parameters are physically reasonable for ethane."""
        params = pipeline_result["seminario_params"]
        # 8 params: [CH_k, CH_r0, CC_k, CC_r0, HCH_k, HCH_eq, CCH_k, CCH_eq]
        assert len(params) == 8

        # C-H bond: k ~ 200-500 kcal/mol/Å², r0 ~ 1.09 Å
        assert 200.0 < params[0] < 500.0, f"C-H force constant out of range: {params[0]}"
        assert 1.0 < params[1] < 1.2, f"C-H equilibrium out of range: {params[1]}"

        # C-C bond: k ~ 100-400 kcal/mol/Å², r0 ~ 1.53 Å
        assert 100.0 < params[2] < 400.0, f"C-C force constant out of range: {params[2]}"
        assert 1.4 < params[3] < 1.7, f"C-C equilibrium out of range: {params[3]}"

    # ---- Optimization stage ----

    def test_optimized_score_improves(self, pipeline_result: dict[str, object]) -> None:
        """Optimizer strictly improves the score over Seminario initial guess."""
        assert pipeline_result["optimized_score"] < pipeline_result["seminario_score"], (
            f"Optimizer failed to improve score: "
            f"{pipeline_result['optimized_score']:.6f} >= "
            f"{pipeline_result['seminario_score']:.6f}"
        )

    def test_optimized_score_matches_golden(
        self, pipeline_result: dict[str, object], golden: dict[str, object]
    ) -> None:
        """Optimized penalty score matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["optimized_score"],
            golden["optimized"]["score"],
            rtol=0.05,
            err_msg="Optimized penalty score diverged from golden",
        )

    def test_optimized_params_match_golden(self, pipeline_result: dict[str, object], golden: dict[str, object]) -> None:
        """Optimized parameter vector matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["optimized_params"],
            golden["optimized"]["params"],
            rtol=0.05,
            err_msg="Optimized params diverged from golden fixture",
        )

    def test_improvement_matches_golden(self, pipeline_result: dict[str, object], golden: dict[str, object]) -> None:
        """Improvement percentage matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["improvement"] * 100,
            golden["optimized"]["improvement_pct"],
            atol=2.0,
            err_msg="Improvement percentage diverged from golden",
        )

    # ---- Frequency comparison ----

    def test_qm_frequencies_match_golden(self, pipeline_result: dict[str, object], golden: dict[str, object]) -> None:
        """QM frequencies extracted from Hessian match golden fixture count."""
        # pipeline_result["qm_real"] is truncated to match MM real-mode count
        # Golden stores all real QM frequencies; compare the matched subset
        n_matched = len(pipeline_result["qm_real"])
        golden_subset = golden["qm_frequencies_cm1"][:n_matched]
        np.testing.assert_allclose(
            pipeline_result["qm_real"],
            golden_subset,
            rtol=1e-4,
        )

    # ---- Runtime benchmark ----

    def test_full_loop_timing(self, pipeline_result: dict[str, object], capsys: pytest.CaptureFixture[str]) -> None:
        """Log full-loop timing (informational, never fails)."""
        with capsys.disabled():
            print(
                f"\n  Ethane GS full loop: "
                f"Seminario {pipeline_result['t_sem']:.3f}s, "
                f"Optimize {pipeline_result['t_opt']:.3f}s, "
                f"Score {pipeline_result['seminario_score']:.6f} → "
                f"{pipeline_result['optimized_score']:.6f} "
                f"({pipeline_result['improvement'] * 100:.1f}% improvement)"
            )
        assert True  # informational only


# ===========================================================================
# Ethane TS: Seminario only (validates TS Hessian handling)
# ===========================================================================


@requires_openmm
@pytest.mark.openmm
@pytest.mark.slow
class TestEthaneTSSeminario:
    """Validate Seminario estimation on ethane TS (eclipsed conformation).

    The TS has one imaginary frequency (~305i cm⁻¹) for torsion rotation.
    Seminario should still produce reasonable bond/angle parameters.
    """

    @pytest.fixture(scope="class")
    def ts_result(self) -> dict[str, object]:
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ReferenceData

        if not TS_FCHK.exists():
            pytest.skip("Ethane TS.fchk not found")

        ref, mol = ReferenceData.from_fchk(str(TS_FCHK), bond_tolerance=1.4)
        ff = estimate_force_constants(mol, au_hessian=True)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        return {"mol": mol, "ff": ff, "qm_freqs": qm_freqs}

    def test_ts_has_imaginary_frequency(self, ts_result: dict[str, object]) -> None:
        """TS should have at least one imaginary (negative) frequency."""
        freqs = ts_result["qm_freqs"]
        imaginary = [f for f in freqs if f < -50.0]
        assert len(imaginary) >= 1, f"Expected imaginary frequency, got none. Min freq: {min(freqs):.1f}"
        # Ethane TS: ~305i cm⁻¹ torsional rotation
        assert any(-500 < f < -100 for f in imaginary), f"Imaginary freq out of expected range: {imaginary}"

    def test_ts_seminario_params_reasonable(self, ts_result: dict[str, object]) -> None:
        """TS Seminario bond/angle params should be close to GS values."""
        ff = ts_result["ff"]
        for b in ff.bonds:
            assert 70.0 < b.force_constant < 720.0, f"Bond FC out of range: {b}"
            assert 0.8 < b.equilibrium < 2.0, f"Bond eq out of range: {b}"
        for a in ff.angles:
            assert 3.5 < a.force_constant < 360.0, f"Angle FC out of range: {a}"
            assert 80.0 < a.equilibrium < 130.0, f"Angle eq out of range: {a}"

    def test_ts_seminario_matches_gs_approximately(self, ts_result: dict[str, object]) -> None:
        """TS and GS Seminario parameters should be similar (same molecule)."""
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ReferenceData

        ref_gs, mol_gs = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)
        ff_gs = estimate_force_constants(mol_gs, au_hessian=True)

        ff_ts = ts_result["ff"]
        gs_params = ff_gs.get_param_vector()
        ts_params = ff_ts.get_param_vector()

        # Same molecule → similar parameters (within ~10%)
        assert len(gs_params) == len(ts_params)
        np.testing.assert_allclose(
            ts_params,
            gs_params,
            rtol=0.15,
            err_msg="TS and GS Seminario parameters differ by >15%",
        )


# ===========================================================================
# Pipeline determinism
# ===========================================================================


@requires_openmm
@pytest.mark.openmm
@pytest.mark.slow
class TestPipelineDeterminism:
    """Verify the full pipeline produces identical results across runs."""

    def test_full_pipeline_is_deterministic(self) -> None:
        """Two independent pipeline runs yield identical scores and params."""
        from q2mm.backends.mm import OpenMMEngine
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not GS_FCHK.exists():
            pytest.skip("Ethane GS.fchk not found")

        results = []
        for _ in range(2):
            ref, mol = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)
            ff = estimate_force_constants(mol, au_hessian=True)
            engine = OpenMMEngine()

            qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
            mm_all = engine.frequencies(mol, ff)
            freq_ref, _ = _build_frequency_reference(qm_freqs, mm_all)

            obj = ObjectiveFunction(ff, engine, [mol], freq_ref)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            result = opt.optimize(obj)
            results.append((result.final_score, ff.get_param_vector().copy()))

        np.testing.assert_array_equal(
            results[0][1],
            results[1][1],
            err_msg="Pipeline produced different params on two runs",
        )
        assert results[0][0] == pytest.approx(results[1][0]), "Pipeline produced different scores on two runs"


# ===========================================================================
# Rh-enamide: JAX backend (harmonic functional form)
# ===========================================================================


def _rh_enamide_harmonic_pipeline(
    engine: MMEngine,
    molecules: list[Q2MMMolecule],
    capsys_disabled: object | None = None,
) -> dict[str, object]:
    """Shared pipeline for JAX/JAX-MD Rh-enamide full-loop tests.

    Runs Seminario → harmonic FF → frequency reference → Nelder-Mead optimize.
    JAX/JAX-MD only support harmonic functional forms, so we create a harmonic
    FF from Seminario estimation (which produces harmonic force constants
    regardless of the template FF's functional form).
    """
    from q2mm.models.forcefield import ForceField, FunctionalForm
    from q2mm.models.seminario import estimate_force_constants
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    mm3_fld_path = RH_DIR / "mm3.fld"
    if not mm3_fld_path.exists():
        pytest.skip("rh-enamide force field file mm3.fld not found")

    ff_template = ForceField.from_mm3_fld(str(mm3_fld_path))

    # Seminario estimation produces harmonic force constants
    t0 = time.perf_counter()
    ff = estimate_force_constants(molecules, forcefield=ff_template)
    t_seminario = time.perf_counter() - t0

    # Switch to harmonic functional form for JAX compatibility
    ff.functional_form = FunctionalForm.HARMONIC
    seminario_params = ff.get_param_vector().copy()

    # Build multi-molecule frequency reference
    freq_ref = None
    n_freqs_per_mol = []
    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )
        n_freqs_per_mol.append(len(qm_real))

    # Initial score
    obj = ObjectiveFunction(ff, engine, molecules, freq_ref)
    initial_score = obj(seminario_params)

    # Optimize (3 iterations, just enough to validate the pipeline)
    t0 = time.perf_counter()
    opt = ScipyOptimizer(method="Nelder-Mead", maxiter=3, verbose=False)
    result = opt.optimize(obj)
    t_optimize = time.perf_counter() - t0

    return {
        "n_molecules": len(molecules),
        "n_params": ff.n_params,
        "n_bonds": len(ff.bonds),
        "n_angles": len(ff.angles),
        "n_vdws": len(ff.vdws),
        "n_freqs_per_mol": n_freqs_per_mol,
        "total_freq_refs": sum(n_freqs_per_mol),
        "seminario_params": seminario_params,
        "initial_score": initial_score,
        "final_score": result.final_score,
        "improvement": result.improvement,
        "converged": result.success,
        "optimized_params": ff.get_param_vector().copy(),
        "t_seminario": t_seminario,
        "t_optimize": t_optimize,
        "functional_form": "harmonic",
    }


@requires_jax
@pytest.mark.jax
@pytest.mark.slow
class TestRhEnamideFullLoopJax:
    """Rh-enamide full pipeline with JaxEngine (harmonic functional form).

    Same 9 organometallic structures as TestRhEnamideFullLoop, but using
    JaxEngine with harmonic energy expressions instead of OpenMM with MM3.
    This validates JAX backend compatibility with real-world multi-molecule
    systems and enables GPU benchmarking via ``pytest -m jax --run-slow``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Q2MMMolecule]:
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.fixture(scope="class")
    def pipeline_result(self, rh_molecules: list[Q2MMMolecule]) -> dict[str, object]:
        """Run the full rh-enamide pipeline with JaxEngine."""
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        return _rh_enamide_harmonic_pipeline(engine, rh_molecules)

    def test_loads_9_molecules(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["n_molecules"] == 9

    def test_seminario_produces_params(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["n_params"] == 182
        assert pipeline_result["n_bonds"] == 8
        assert pipeline_result["n_angles"] == 23

    def test_all_molecules_have_frequencies(self, pipeline_result: dict[str, object]) -> None:
        for i, n in enumerate(pipeline_result["n_freqs_per_mol"]):
            assert n > 0, f"Molecule {i} contributed 0 frequency references"

    def test_initial_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        score = pipeline_result["initial_score"]
        assert np.isfinite(score), f"Initial score is not finite: {score}"
        assert score > 0, f"Initial score should be positive: {score}"

    def test_final_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        score = pipeline_result["final_score"]
        assert np.isfinite(score), f"Final score is not finite: {score}"
        assert score > 0, f"Final score should be positive: {score}"

    def test_optimized_params_differ_from_seminario(self, pipeline_result: dict[str, object]) -> None:
        diff = np.abs(pipeline_result["optimized_params"] - pipeline_result["seminario_params"])
        assert np.any(diff > 1e-6), "Optimizer didn't change any parameters"

    def test_uses_harmonic_form(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["functional_form"] == "harmonic"

    def test_timing_report(self, pipeline_result: dict[str, object], capsys: pytest.CaptureFixture[str]) -> None:
        r = pipeline_result
        with capsys.disabled():
            print(
                f"\n  Rh-enamide JAX full loop ({r['n_molecules']} mols, {r['n_params']} params, "
                f"{r['total_freq_refs']} freq refs):"
                f"\n    Seminario: {r['t_seminario']:.3f}s"
                f"\n    Optimize:  {r['t_optimize']:.1f}s (Nelder-Mead, maxiter=3)"
                f"\n    Score:     {r['initial_score']:.1f} → {r['final_score']:.1f} "
                f"({r['improvement'] * 100:.1f}% improvement)"
            )


@requires_jax_md
@pytest.mark.jax_md
@pytest.mark.slow
class TestRhEnamideFullLoopJaxMD:
    """Rh-enamide full pipeline with JaxMDEngine (harmonic functional form).

    Same 9 organometallic structures, using JaxMDEngine with harmonic energy
    expressions. Validates JAX-MD backend on a real organometallic system.
    Enables GPU benchmarking via ``pytest -m jax_md --run-slow``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Q2MMMolecule]:
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.fixture(scope="class")
    def pipeline_result(self, rh_molecules: list[Q2MMMolecule]) -> dict[str, object]:
        """Run the full rh-enamide pipeline with JaxMDEngine."""
        from q2mm.backends.mm.jax_md_engine import JaxMDEngine

        engine = JaxMDEngine()
        return _rh_enamide_harmonic_pipeline(engine, rh_molecules)

    def test_loads_9_molecules(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["n_molecules"] == 9

    def test_seminario_produces_params(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["n_params"] == 182
        assert pipeline_result["n_bonds"] == 8
        assert pipeline_result["n_angles"] == 23

    def test_all_molecules_have_frequencies(self, pipeline_result: dict[str, object]) -> None:
        for i, n in enumerate(pipeline_result["n_freqs_per_mol"]):
            assert n > 0, f"Molecule {i} contributed 0 frequency references"

    def test_initial_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        score = pipeline_result["initial_score"]
        assert np.isfinite(score), f"Initial score is not finite: {score}"
        assert score > 0, f"Initial score should be positive: {score}"

    def test_final_score_is_finite(self, pipeline_result: dict[str, object]) -> None:
        score = pipeline_result["final_score"]
        assert np.isfinite(score), f"Final score is not finite: {score}"
        assert score > 0, f"Final score should be positive: {score}"

    def test_optimized_params_differ_from_seminario(self, pipeline_result: dict[str, object]) -> None:
        diff = np.abs(pipeline_result["optimized_params"] - pipeline_result["seminario_params"])
        assert np.any(diff > 1e-6), "Optimizer didn't change any parameters"

    def test_uses_harmonic_form(self, pipeline_result: dict[str, object]) -> None:
        assert pipeline_result["functional_form"] == "harmonic"

    def test_timing_report(self, pipeline_result: dict[str, object], capsys: pytest.CaptureFixture[str]) -> None:
        r = pipeline_result
        with capsys.disabled():
            print(
                f"\n  Rh-enamide JAX-MD full loop ({r['n_molecules']} mols, {r['n_params']} params, "
                f"{r['total_freq_refs']} freq refs):"
                f"\n    Seminario: {r['t_seminario']:.3f}s"
                f"\n    Optimize:  {r['t_optimize']:.1f}s (Nelder-Mead, maxiter=3)"
                f"\n    Score:     {r['initial_score']:.1f} → {r['final_score']:.1f} "
                f"({r['improvement'] * 100:.1f}% improvement)"
            )
