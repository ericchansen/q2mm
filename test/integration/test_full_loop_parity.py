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
from q2mm.backends.contracts import (
    FrequencyRequest,
)
from test.backend_fixtures import param_vector, prepare_case
from q2mm.backends.registry import load_backend

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from q2mm.backends.contracts import Backend
    from q2mm.models.molecule import Molecule
    from q2mm.models.observations import ObservationSet

from test._shared import GS_FCHK, REPO_ROOT, TS_FCHK


def _make_freq_evaluator(ff, backend, molecules, reference, layout):  # noqa: ANN001, ANN202
    """Build a Python objective executor for a frequency objective."""
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.models.problem import StationaryPointKind
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor

    mols = list(molecules)
    space = ActiveParameterSpace.all_active(layout, ff)
    plan = ObjectivePlan(
        case_ids=tuple(str(i) for i in range(len(mols))),
        molecules=tuple(mols),
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in mols),
        observations=reference,
        layout=layout,
        active_space=space,
    )
    return PythonObjectiveExecutor(plan, backend, ff)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "full_loop"
ETHANE_GS_GOLDEN = FIXTURE_DIR / "ethane_gs_golden.json"

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

import importlib.util

_HAS_OPENMM = True
try:
    import openmm  # noqa: F401
except ImportError:
    _HAS_OPENMM = False

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_JAX_MD = importlib.util.find_spec("jax_md") is not None

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
    :meth:`~q2mm.backends.mm.openmm.OpenMMBackend.frequencies` to ensure
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
    case_id: str = "0",
    ref: ObservationSet | None = None,
) -> tuple[ObservationSet, list[float]]:
    """Build (or extend) an ObservationSet with frequency observations.

    Maps QM real frequencies (>threshold) to MM real-mode indices.
    Pass an existing *ref* to append multi-molecule data.
    """
    from q2mm.models.observations import ObservationSet

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))

    if ref is None:
        ref = ObservationSet()
    for k in range(n):
        ref = ref.with_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=weight, case_id=case_id)
    return ref, qm_real[:n]


# ===========================================================================
# Rh-enamide Seminario parity + timing (no MM backend needed)
# ===========================================================================


class TestRhEnamideSeminarioTiming:
    """Runtime benchmarks for Seminario on the 9-structure rh-enamide dataset.

    These tests are informational — they log wall-clock times but never
    fail on timing alone.  They complement the parameter-accuracy tests
    in ``test_seminario_parity.py``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Molecule]:
        """Load all 9 rh-enamide structures + Hessians."""
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.mark.validation
    def test_seminario_pipeline_timing(self, rh_molecules: list[Molecule], capsys: pytest.CaptureFixture[str]) -> None:
        """Time the full Seminario pipeline on 9 rh-enamide structures."""
        from q2mm.benchmarks.systems._forcefield import load_published_opt
        from q2mm.models.parameters import ParameterLayout
        from q2mm.models.seminario import qfuerza_into

        mm3_path = RH_DIR / "mm3.fld"
        if not mm3_path.exists():
            pytest.skip("mm3.fld not found")
        _, ff_template = load_published_opt(mm3_path)

        t0 = time.perf_counter()
        ff = qfuerza_into(ff_template, rh_molecules, invert_ts_curvature=True)
        elapsed = time.perf_counter() - t0
        layout = ParameterLayout.from_force_field(ff)

        with capsys.disabled():
            print(f"\n  Rh-enamide Seminario: {elapsed:.3f}s ({len(rh_molecules)} structures, {len(layout)} params)")

        # Sanity check — never fail on timing
        assert len(layout) > 0, "No parameters estimated"
        assert len(ff.bonds) > 0, "No bond parameters"
        assert len(ff.angles) > 0, "No angle parameters"

    @pytest.mark.validation
    def test_seminario_is_deterministic(self, rh_molecules: list[Molecule]) -> None:
        """Two consecutive Seminario runs produce identical results."""
        from q2mm.benchmarks.systems._forcefield import load_published_opt
        from q2mm.models.parameters import ParameterLayout
        from q2mm.models.seminario import qfuerza_into

        mm3_path = RH_DIR / "mm3.fld"
        if not mm3_path.exists():
            pytest.skip("mm3.fld not found")
        _, ff_template = load_published_opt(mm3_path)

        ff1 = qfuerza_into(ff_template, rh_molecules, invert_ts_curvature=True)
        ff2 = qfuerza_into(ff_template, rh_molecules, invert_ts_curvature=True)
        layout = ParameterLayout.from_force_field(ff1)

        np.testing.assert_array_equal(
            layout.vector(ff1),
            layout.vector(ff2),
            err_msg="Seminario is non-deterministic across runs",
        )


# ===========================================================================
# Rh-enamide: full optimization loop with Jaguar QM data (D1)
# ===========================================================================


def _load_rh_enamide_molecules() -> list[Molecule]:
    """Load 9 rh-enamide structures with Jaguar Hessians.

    Delegates to the benchmark-system loader.
    """
    from q2mm.benchmarks.systems.rh_enamide import load_molecules

    return load_molecules()


@requires_openmm
@pytest.mark.openmm
@pytest.mark.nightly
class TestRhEnamideFullLoop:
    """D1: Full pipeline on rh-enamide — Jaguar → Seminario → OpenMM → optimize.

    9 organometallic structures (Rh-diphosphine, 36 atoms each, B3LYP/LACVP**).
    182 parameters (8 bond types, 23 angle types, 36 vdW types).
    Frequency-based objective with Nelder-Mead optimization.
    """

    @pytest.fixture(scope="class")
    def pipeline_result(self) -> dict[str, object]:
        """Run the full rh-enamide pipeline."""
        from q2mm.benchmarks.systems._forcefield import load_published_opt
        from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
        from q2mm.models.seminario import qfuerza_into
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")

        mm3_fld_path = RH_DIR / "mm3.fld"
        if not mm3_fld_path.exists():
            pytest.skip("rh-enamide force field file mm3.fld not found")

        molecules = _load_rh_enamide_molecules()
        _, ff_template = load_published_opt(mm3_fld_path)

        # Seminario estimation
        t0 = time.perf_counter()
        ff = qfuerza_into(ff_template, molecules, invert_ts_curvature=True)
        t_seminario = time.perf_counter() - t0
        layout = ParameterLayout.from_force_field(ff)
        seminario_params = layout.vector(ff).copy()

        # Build multi-molecule frequency reference
        backend = load_backend("openmm")
        freq_ref = None
        n_freqs_per_mol = []
        for mol_idx, mol in enumerate(molecules):
            mm_freqs = [
                float(_f)
                for _f in prepare_case(backend, mol, ff)
                .frequencies(FrequencyRequest(parameters=param_vector(ff)))
                .frequencies
            ]
            qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
            freq_ref, qm_real = _build_frequency_reference(
                qm_freqs,
                mm_freqs,
                case_id=str(mol_idx),
                ref=freq_ref,
            )
            n_freqs_per_mol.append(len(qm_real))

        # Initial score
        obj = _make_freq_evaluator(ff, backend, molecules, freq_ref, layout)
        initial_score = obj.value(seminario_params)

        # Optimize — just enough iterations to verify our optimizer wrapper
        # and objective function work end-to-end. Full convergence benchmarks
        # (500 iter, 76.7% improvement) are documented in docs/benchmarks/.
        t0 = time.perf_counter()
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=3, verbose=False)
        result = opt.optimize(obj, ActiveParameterSpace.all_active(layout, ff))
        t_optimize = time.perf_counter() - t0

        return {
            "n_molecules": len(molecules),
            "n_params": len(layout),
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
            "optimized_params": result.final_params.copy(),
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
@pytest.mark.nightly
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
        from q2mm.io.fchk import load_fchk
        from q2mm.models.forcefield import FunctionalForm
        from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
        from q2mm.models.seminario import qfuerza_fresh
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not GS_FCHK.exists():
            pytest.skip("Ethane GS.fchk not found")

        mol = load_fchk(GS_FCHK, bond_tolerance=1.4)

        # QM frequencies from Hessian
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        # Seminario estimation
        t_sem_start = time.perf_counter()
        ff = qfuerza_fresh(mol, functional_form=FunctionalForm.MM3, au_hessian=True)
        t_sem = time.perf_counter() - t_sem_start
        layout = ParameterLayout.from_force_field(ff)
        seminario_params = layout.vector(ff).copy()

        # MM frequencies + reference data
        backend = load_backend("openmm")
        mm_all = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        freq_ref, qm_real = _build_frequency_reference(qm_freqs, mm_all)

        # Penalty score
        obj = _make_freq_evaluator(ff, backend, [mol], freq_ref, layout)
        seminario_score = obj.value(seminario_params)

        # Optimize
        t_opt_start = time.perf_counter()
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, ActiveParameterSpace.all_active(layout, ff))
        t_opt = time.perf_counter() - t_opt_start

        return {
            "mol": mol,
            "ff": ff,
            "backend": backend,
            "seminario_params": seminario_params,
            "seminario_score": seminario_score,
            "optimized_params": result.final_params.copy(),
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
        assert "t_sem" in pipeline_result
        assert "t_opt" in pipeline_result
        assert pipeline_result["improvement"] >= 0  # optimization should not make things worse


# ===========================================================================
# Ethane TS: Seminario only (validates TS Hessian handling)
# ===========================================================================


@requires_openmm
@pytest.mark.openmm
@pytest.mark.validation
class TestEthaneTSSeminario:
    """Validate Seminario estimation on ethane TS (eclipsed conformation).

    The TS has one imaginary frequency (~305i cm⁻¹) for torsion rotation.
    Seminario should still produce reasonable bond/angle parameters.
    """

    @pytest.fixture(scope="class")
    def ts_result(self) -> dict[str, object]:
        from q2mm.io.fchk import load_fchk
        from q2mm.models.forcefield import FunctionalForm
        from q2mm.models.seminario import qfuerza_fresh

        if not TS_FCHK.exists():
            pytest.skip("Ethane TS.fchk not found")

        mol = load_fchk(TS_FCHK, bond_tolerance=1.4)
        ff = qfuerza_fresh(mol, functional_form=FunctionalForm.MM3, au_hessian=True, invert_ts_curvature=True)
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
        from q2mm.io.fchk import load_fchk
        from q2mm.models.forcefield import FunctionalForm
        from q2mm.models.parameters import ParameterLayout
        from q2mm.models.seminario import qfuerza_fresh

        mol_gs = load_fchk(GS_FCHK, bond_tolerance=1.4)
        ff_gs = qfuerza_fresh(mol_gs, functional_form=FunctionalForm.MM3, au_hessian=True)

        ff_ts = ts_result["ff"]
        layout = ParameterLayout.from_force_field(ff_gs)
        gs_params = layout.vector(ff_gs)
        ts_params = layout.vector(ff_ts)

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
@pytest.mark.nightly
class TestPipelineDeterminism:
    """Verify the full pipeline produces identical results across runs."""

    def test_full_pipeline_is_deterministic(self) -> None:
        """Two independent pipeline runs yield identical scores and params."""
        from q2mm.io.fchk import load_fchk
        from q2mm.models.forcefield import FunctionalForm
        from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
        from q2mm.models.seminario import qfuerza_fresh
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        if not GS_FCHK.exists():
            pytest.skip("Ethane GS.fchk not found")

        results = []
        for _ in range(2):
            mol = load_fchk(GS_FCHK, bond_tolerance=1.4)
            ff = qfuerza_fresh(mol, functional_form=FunctionalForm.MM3, au_hessian=True)
            backend = load_backend("openmm")
            layout = ParameterLayout.from_force_field(ff)

            qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
            mm_all = [
                float(_f)
                for _f in prepare_case(backend, mol, ff)
                .frequencies(FrequencyRequest(parameters=param_vector(ff)))
                .frequencies
            ]
            freq_ref, _ = _build_frequency_reference(qm_freqs, mm_all)

            obj = _make_freq_evaluator(ff, backend, [mol], freq_ref, layout)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            result = opt.optimize(obj, ActiveParameterSpace.all_active(layout, ff))
            results.append((result.final_score, result.final_params.copy()))

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
    backend: Backend,
    molecules: list[Molecule],
) -> dict[str, object]:
    """Shared pipeline for JAX/JAX-MD Rh-enamide full-loop tests.

    Runs Seminario → harmonic FF → frequency reference → Nelder-Mead optimize.
    JAX/JAX-MD only support harmonic functional forms, so we create a harmonic
    FF from Seminario estimation (which produces harmonic force constants
    regardless of the template FF's functional form).
    """
    from dataclasses import replace

    from q2mm.benchmarks.systems._forcefield import load_published_opt
    from q2mm.models.forcefield import FunctionalForm
    from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
    from q2mm.models.seminario import qfuerza_into
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    mm3_fld_path = RH_DIR / "mm3.fld"
    if not mm3_fld_path.exists():
        pytest.skip("rh-enamide force field file mm3.fld not found")

    _, ff_template = load_published_opt(mm3_fld_path)

    # Seminario estimation produces harmonic force constants
    t0 = time.perf_counter()
    ff = qfuerza_into(ff_template, molecules, invert_ts_curvature=True)
    t_seminario = time.perf_counter() - t0

    # Switch to harmonic functional form for JAX compatibility
    ff = replace(ff, functional_form=FunctionalForm.HARMONIC)
    layout = ParameterLayout.from_force_field(ff)
    seminario_params = layout.vector(ff).copy()

    # Build multi-molecule frequency reference
    freq_ref = None
    n_freqs_per_mol = []
    for mol_idx, mol in enumerate(molecules):
        mm_freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, ff)
            .frequencies(FrequencyRequest(parameters=param_vector(ff)))
            .frequencies
        ]
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            case_id=str(mol_idx),
            ref=freq_ref,
        )
        n_freqs_per_mol.append(len(qm_real))

    # Initial score
    obj = _make_freq_evaluator(ff, backend, molecules, freq_ref, layout)
    initial_score = obj.value(seminario_params)

    # Optimize (3 iterations, just enough to validate the pipeline)
    t0 = time.perf_counter()
    opt = ScipyOptimizer(method="Nelder-Mead", maxiter=3, verbose=False)
    result = opt.optimize(obj, ActiveParameterSpace.all_active(layout, ff))
    t_optimize = time.perf_counter() - t0

    return {
        "n_molecules": len(molecules),
        "n_params": len(layout),
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
        "optimized_params": result.final_params.copy(),
        "t_seminario": t_seminario,
        "t_optimize": t_optimize,
        "functional_form": "harmonic",
    }


@requires_jax
@pytest.mark.jax
@pytest.mark.nightly
class TestRhEnamideFullLoopJax:
    """Rh-enamide full pipeline with JaxBackend (harmonic functional form).

    Same 9 organometallic structures as TestRhEnamideFullLoop, but using
    JaxBackend with harmonic energy expressions instead of OpenMM with MM3.
    This validates JAX backend compatibility with real-world multi-molecule
    systems and enables GPU benchmarking via ``pytest -m jax --run-nightly``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Molecule]:
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.fixture(scope="class")
    def pipeline_result(self, rh_molecules: list[Molecule]) -> dict[str, object]:
        """Run the full rh-enamide pipeline with JaxBackend."""
        backend = load_backend("jax")
        return _rh_enamide_harmonic_pipeline(backend, rh_molecules)

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
@pytest.mark.nightly
class TestRhEnamideFullLoopJaxMD:
    """Rh-enamide full pipeline with JaxMdBackend (harmonic functional form).

    Same 9 organometallic structures, using JaxMdBackend with harmonic energy
    expressions. Validates JAX-MD backend on a real organometallic system.
    Enables GPU benchmarking via ``pytest -m jax_md --run-nightly``.
    """

    @pytest.fixture(scope="class")
    def rh_molecules(self) -> list[Molecule]:
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.fixture(scope="class")
    def pipeline_result(self, rh_molecules: list[Molecule]) -> dict[str, object]:
        """Run the full rh-enamide pipeline with JaxMdBackend."""
        backend = load_backend("jax-md")
        return _rh_enamide_harmonic_pipeline(backend, rh_molecules)

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
