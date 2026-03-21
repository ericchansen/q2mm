"""Full-loop parity validation — issue #74.

Validates the complete Q2MM pipeline end-to-end:
1. QM data loading (Gaussian .fchk, Jaguar .in, or Psi4)
2. Seminario force-constant estimation
3. Frequency-based penalty scoring (via OpenMM)
4. L-BFGS-B optimization
5. Determinism and golden-fixture reproducibility

The rh-enamide dataset (9 Jaguar structures) validates Seminario
parameter extraction for a large organometallic system.  The ethane
GS/TS systems validate the full optimization loop with frequency-based
objective functions.

Psi4 cross-validation tests compare QM Hessians from Psi4 against
Gaussian .fchk data to ensure backend-independent results.

Golden fixtures in ``test/fixtures/full_loop/`` store the expected
pipeline outputs.  These are deterministic — the same code + data must
produce bit-identical scores and parameter vectors.

References
----------
- Issue: https://github.com/ericchansen/q2mm/issues/74
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

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

requires_openmm = pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed")


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


def _build_frequency_reference(qm_freqs, mm_all_freqs, *, threshold: float = 50.0, weight: float = 0.001):
    """Build a ReferenceData of frequency observations.

    Maps QM real frequencies (>threshold) to MM real-mode indices,
    following the same pattern as ``test_e2e_sn2_validation.py``.
    """
    from q2mm.optimizers.objective import ReferenceData

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))

    ref = ReferenceData()
    for k in range(n):
        ref.add_frequency(float(qm_real[k]), data_idx=mm_real_idx[k], weight=weight, molecule_idx=0)
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
    def rh_molecules(self):
        """Load all 9 rh-enamide structures + Hessians."""
        import re

        from q2mm.models.molecule import Q2MMMolecule
        from q2mm.parsers import JaguarIn, MacroModel

        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")

        mm = MacroModel(str(MMO_PATH))
        jag_files = sorted(
            JAG_DIR.glob("*.in"), key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)]
        )

        molecules = []
        for struct, jag_path in zip(mm.structures, jag_files):
            jag = JaguarIn(str(jag_path))
            hess = jag.get_hessian(len(struct.atoms))
            mol = Q2MMMolecule.from_structure(struct, hessian=hess)
            molecules.append(mol)
        return molecules

    @pytest.mark.slow
    def test_seminario_pipeline_timing(self, rh_molecules, capsys):
        """Time the full Seminario pipeline on 9 rh-enamide structures."""
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.parsers import MacroModel

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
    def test_seminario_is_deterministic(self, rh_molecules):
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
    def golden(self):
        if not ETHANE_GS_GOLDEN.exists():
            pytest.skip("Golden fixture not found; run generate_golden_fixtures.py")
        return _load_golden(ETHANE_GS_GOLDEN)

    @pytest.fixture(scope="class")
    def pipeline_result(self):
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

    def test_seminario_params_match_golden(self, pipeline_result, golden):
        """Seminario parameter vector matches golden fixture exactly."""
        np.testing.assert_allclose(
            pipeline_result["seminario_params"],
            golden["seminario"]["params"],
            rtol=1e-10,
            err_msg="Seminario params diverged from golden fixture",
        )

    def test_seminario_score_matches_golden(self, pipeline_result, golden):
        """Seminario penalty score matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["seminario_score"],
            golden["seminario"]["score"],
            rtol=1e-6,
            err_msg="Seminario penalty score diverged from golden",
        )

    def test_seminario_has_reasonable_params(self, pipeline_result):
        """Seminario parameters are physically reasonable for ethane."""
        params = pipeline_result["seminario_params"]
        # 8 params: [CH_k, CH_r0, CC_k, CC_r0, HCH_k, HCH_eq, CCH_k, CCH_eq]
        assert len(params) == 8

        # C-H bond: k ~ 4-6 mdyn/Å, r0 ~ 1.09 Å
        assert 3.0 < params[0] < 7.0, f"C-H force constant out of range: {params[0]}"
        assert 1.0 < params[1] < 1.2, f"C-H equilibrium out of range: {params[1]}"

        # C-C bond: k ~ 2-5 mdyn/Å, r0 ~ 1.53 Å
        assert 1.5 < params[2] < 5.0, f"C-C force constant out of range: {params[2]}"
        assert 1.4 < params[3] < 1.7, f"C-C equilibrium out of range: {params[3]}"

    # ---- Optimization stage ----

    def test_optimization_converges(self, pipeline_result):
        """Optimizer reports convergence."""
        assert pipeline_result["converged"], "L-BFGS-B did not converge"

    def test_optimized_score_improves(self, pipeline_result):
        """Optimized score ≤ Seminario score (optimizer must not worsen)."""
        assert pipeline_result["optimized_score"] <= pipeline_result["seminario_score"] + 1e-12

    def test_optimized_score_matches_golden(self, pipeline_result, golden):
        """Optimized penalty score matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["optimized_score"],
            golden["optimized"]["score"],
            rtol=1e-4,
            err_msg="Optimized penalty score diverged from golden",
        )

    def test_optimized_params_match_golden(self, pipeline_result, golden):
        """Optimized parameter vector matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["optimized_params"],
            golden["optimized"]["params"],
            rtol=1e-4,
            err_msg="Optimized params diverged from golden fixture",
        )

    def test_improvement_matches_golden(self, pipeline_result, golden):
        """Improvement percentage matches golden fixture."""
        np.testing.assert_allclose(
            pipeline_result["improvement"] * 100,
            golden["optimized"]["improvement_pct"],
            atol=0.5,
            err_msg="Improvement percentage diverged from golden",
        )

    # ---- Frequency comparison ----

    def test_qm_frequencies_match_golden(self, pipeline_result, golden):
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

    def test_full_loop_timing(self, pipeline_result, capsys):
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
    def ts_result(self):
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ReferenceData

        if not TS_FCHK.exists():
            pytest.skip("Ethane TS.fchk not found")

        ref, mol = ReferenceData.from_fchk(str(TS_FCHK), bond_tolerance=1.4)
        ff = estimate_force_constants(mol, au_hessian=True)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)
        return {"mol": mol, "ff": ff, "qm_freqs": qm_freqs}

    def test_ts_has_imaginary_frequency(self, ts_result):
        """TS should have at least one imaginary (negative) frequency."""
        freqs = ts_result["qm_freqs"]
        imaginary = [f for f in freqs if f < -50.0]
        assert len(imaginary) >= 1, f"Expected imaginary frequency, got none. Min freq: {min(freqs):.1f}"
        # Ethane TS: ~305i cm⁻¹ torsional rotation
        assert any(-500 < f < -100 for f in imaginary), f"Imaginary freq out of expected range: {imaginary}"

    def test_ts_seminario_params_reasonable(self, ts_result):
        """TS Seminario bond/angle params should be close to GS values."""
        ff = ts_result["ff"]
        for b in ff.bonds:
            assert 1.0 < b.force_constant < 10.0, f"Bond FC out of range: {b}"
            assert 0.8 < b.equilibrium < 2.0, f"Bond eq out of range: {b}"
        for a in ff.angles:
            assert 0.05 < a.force_constant < 5.0, f"Angle FC out of range: {a}"
            assert 80.0 < a.equilibrium < 130.0, f"Angle eq out of range: {a}"

    def test_ts_seminario_matches_gs_approximately(self, ts_result):
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

    def test_full_pipeline_is_deterministic(self):
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
            opt.optimize(obj)
            results.append((obj.history[-1], ff.get_param_vector().copy()))

        np.testing.assert_array_equal(
            results[0][1],
            results[1][1],
            err_msg="Pipeline produced different params on two runs",
        )
        assert results[0][0] == results[1][0], "Pipeline produced different scores on two runs"


# ===========================================================================
# Psi4 cross-validation
# ===========================================================================


@pytest.mark.psi4
@pytest.mark.slow
class TestPsi4CrossValidation:
    """Cross-validate Psi4 QM results against Gaussian .fchk for ethane.

    These tests run actual Psi4 computations and compare the resulting
    Hessians and Seminario parameters to the Gaussian reference.
    Only runs when Psi4 is installed (CI psi4 container).
    """

    @pytest.fixture(scope="class")
    def psi4_ethane(self):
        """Compute ethane GS Hessian via Psi4."""
        psi4 = pytest.importorskip("psi4")
        from q2mm.backends.qm.psi4 import Psi4Engine
        from q2mm.models.molecule import Q2MMMolecule
        from q2mm.optimizers.objective import ReferenceData

        # Load ethane geometry from .fchk
        ref, mol_fchk = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)

        # Compute Hessian with Psi4 at B3LYP/6-31G*
        with Psi4Engine(method="b3lyp", basis="6-31g*") as engine:
            hessian = engine.hessian(mol_fchk)

        mol_psi4 = Q2MMMolecule(
            symbols=mol_fchk.symbols,
            geometry=mol_fchk.geometry,
            name="ethane-psi4",
            bond_tolerance=1.4,
            hessian=hessian,
        )
        return mol_psi4, mol_fchk

    def test_psi4_hessian_close_to_gaussian(self, psi4_ethane):
        """Psi4 and Gaussian Hessians agree within 1% for same basis set."""
        mol_psi4, mol_fchk = psi4_ethane
        # Both Hessians in Hartree/Bohr²
        np.testing.assert_allclose(
            mol_psi4.hessian,
            mol_fchk.hessian,
            rtol=0.01,
            atol=1e-5,
            err_msg="Psi4 vs Gaussian Hessian mismatch",
        )

    def test_psi4_seminario_close_to_gaussian(self, psi4_ethane):
        """Seminario params from Psi4 match Gaussian-derived params."""
        from q2mm.models.seminario import estimate_force_constants

        mol_psi4, mol_fchk = psi4_ethane
        ff_psi4 = estimate_force_constants(mol_psi4, au_hessian=True)
        ff_gauss = estimate_force_constants(mol_fchk, au_hessian=True)

        np.testing.assert_allclose(
            ff_psi4.get_param_vector(),
            ff_gauss.get_param_vector(),
            rtol=0.02,
            err_msg="Psi4 vs Gaussian Seminario params differ >2%",
        )

    def test_psi4_frequencies_close_to_gaussian(self, psi4_ethane):
        """QM frequencies from Psi4 Hessian match Gaussian frequencies."""
        mol_psi4, mol_fchk = psi4_ethane
        freqs_psi4 = _qm_frequencies_from_hessian(mol_psi4.hessian, mol_psi4.symbols)
        freqs_gauss = _qm_frequencies_from_hessian(mol_fchk.hessian, mol_fchk.symbols)

        real_psi4 = sorted(f for f in freqs_psi4 if f > 50.0)
        real_gauss = sorted(f for f in freqs_gauss if f > 50.0)

        assert len(real_psi4) == len(real_gauss), "Different number of real frequencies"
        np.testing.assert_allclose(
            real_psi4,
            real_gauss,
            rtol=0.01,
            err_msg="Psi4 vs Gaussian frequency mismatch",
        )


@pytest.mark.psi4
@pytest.mark.slow
class TestPsi4FullLoop:
    """Full optimization loop using Psi4-generated QM data.

    Runs the complete pipeline: Psi4 Hessian → Seminario → OpenMM
    frequency objective → L-BFGS-B optimization.  Validates that the
    pipeline converges and produces physically reasonable results
    independent of which QM backend generated the Hessian.
    """

    def test_psi4_ethane_full_optimization(self):
        """Psi4 → Seminario → OpenMM optimize → verify convergence."""
        psi4 = pytest.importorskip("psi4")
        from q2mm.backends.mm import OpenMMEngine
        from q2mm.backends.qm.psi4 import Psi4Engine
        from q2mm.models.molecule import Q2MMMolecule
        from q2mm.models.seminario import estimate_force_constants
        from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
        from q2mm.optimizers.scipy_opt import ScipyOptimizer

        # Load geometry
        ref, mol_fchk = ReferenceData.from_fchk(str(GS_FCHK), bond_tolerance=1.4)

        # Psi4 Hessian
        with Psi4Engine(method="b3lyp", basis="6-31g*") as engine:
            hessian = engine.hessian(mol_fchk)

        mol = Q2MMMolecule(
            symbols=mol_fchk.symbols,
            geometry=mol_fchk.geometry,
            name="ethane-psi4",
            bond_tolerance=1.4,
            hessian=hessian,
        )

        # Seminario
        ff = estimate_force_constants(mol, au_hessian=True)
        assert ff.n_params > 0

        # Frequency objective
        mm_engine = OpenMMEngine()
        qm_freqs = _qm_frequencies_from_hessian(hessian, mol.symbols)
        mm_all = mm_engine.frequencies(mol, ff)
        freq_ref, _ = _build_frequency_reference(qm_freqs, mm_all)

        obj = ObjectiveFunction(ff, mm_engine, [mol], freq_ref)
        sem_score = obj(ff.get_param_vector())

        # Optimize
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        assert result.success, "Psi4-based optimization did not converge"
        assert result.final_score <= sem_score + 1e-12, "Optimization worsened the score"
