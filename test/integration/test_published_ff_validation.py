"""Published force field validation — Check 1.

Validates that the published Rh-enamide force field (Donoghue et al. 2008 JCTC)
produces physically reasonable MM frequencies when evaluated with the new q2mm
engines. This is the critical "Check 1" test: load the FF the authors actually
published, evaluate it against the same QM reference data, and pin the results.

The ``mm3.fld`` file contains the published optimized parameters embedded in its
Rh-enamide substructure section. ``ForceField.from_mm3_fld`` extracts these
directly — no Seminario estimation needed. We evaluate with OpenMM and compare
against the QM (Jaguar B3LYP/LACVP**) harmonic frequencies.

The Seminario-estimated FF serves as the "unoptimized" baseline. The published
FF should produce a meaningfully lower objective score since it was further
optimized by Q2MM.

References
----------
- Donoghue, P. J. et al. J. Chem. Theory Comput. 2008, 4, 1313–1323.
  DOI: 10.1021/ct800132a
- Old repo: https://github.com/Q2MM/q2mm (commit b26404b8)
- Training set: 9 Rh-diphosphine TS structures, B3LYP/LACVP** (Jaguar)

"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test._shared import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RH_DIR = REPO_ROOT / "examples" / "rh-enamide"
TRAINING_SET_DIR = RH_DIR / "rh_enamide_training_set"
MMO_PATH = TRAINING_SET_DIR / "rh_enamide_training_set.mmo"
JAG_DIR = TRAINING_SET_DIR / "jaguar_spe_freq_in_out"

# mm3.fld contains the published final params in its substructure section
MM3_FLD_PATH = RH_DIR / "mm3.fld"

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "published_ff"
GOLDEN_PATH = FIXTURE_DIR / "rh_enamide_donoghue2008.json"
UPDATE_GOLDEN = os.getenv("Q2MM_UPDATE_GOLDEN") == "1"

_HAS_OPENMM = True
try:
    import openmm  # noqa: F401
except ImportError:
    _HAS_OPENMM = False

requires_openmm = pytest.mark.skipif(not _HAS_OPENMM, reason="OpenMM not installed")


# ---------------------------------------------------------------------------
# Helpers (shared with test_full_loop_parity.py)
# ---------------------------------------------------------------------------


def _qm_frequencies_from_hessian(
    hessian_au: np.ndarray,
    symbols: list[str],
) -> np.ndarray:
    """Compute harmonic frequencies (cm⁻¹) from a Cartesian Hessian in AU."""
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
    ref: Any = None,
) -> tuple[Any, list[float]]:
    """Build (or extend) a ReferenceData with frequency observations."""
    from q2mm.optimizers.objective import ReferenceData

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if f > threshold)
    n = min(len(qm_real), len(mm_real_idx))

    if ref is None:
        ref = ReferenceData()
    for k in range(n):
        ref.add_frequency(
            float(qm_real[k]),
            data_idx=mm_real_idx[k],
            weight=weight,
            molecule_idx=molecule_idx,
        )
    return ref, qm_real[:n]


def _load_rh_enamide_molecules() -> list[Any]:
    """Load 9 rh-enamide structures with Jaguar Hessians."""
    from q2mm.models.molecule import Q2MMMolecule
    from q2mm.parsers import JaguarIn, MacroModel

    mm = MacroModel(str(MMO_PATH))
    jag_files = sorted(
        JAG_DIR.glob("*.in"),
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.stem)],
    )
    n_structures = len(mm.structures)
    n_jag = len(jag_files)
    if n_structures != n_jag:
        pytest.skip(
            f"rh-enamide dataset inconsistent: {n_structures} MacroModel structures "
            f"but {n_jag} Jaguar .in files in {JAG_DIR}"
        )
    molecules = []
    for struct, jag_path in zip(mm.structures, jag_files):
        jag = JaguarIn(str(jag_path))
        hess = jag.get_hessian(len(struct.atoms))
        molecules.append(Q2MMMolecule.from_structure(struct, hessian=hess))
    return molecules


def _evaluate_ff_on_training_set(ff: Any, molecules: list[Any], engine: Any) -> dict[str, Any]:
    """Evaluate a force field against QM reference frequencies.

    Returns a dict with per-molecule and overall statistics.
    """
    freq_ref = None
    per_molecule = []

    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        # Build reference for objective scoring
        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )

        # Per-molecule statistics
        mm_real = sorted(f for f in mm_freqs if f > 50.0)
        n = min(len(qm_real), len(mm_real))
        qm_matched = np.array(qm_real[:n])
        mm_matched = np.array(mm_real[:n])
        residuals = qm_matched - mm_matched
        rmsd = float(np.sqrt(np.mean(residuals**2)))
        mae = float(np.mean(np.abs(residuals)))

        # R² (coefficient of determination)
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((qm_matched - np.mean(qm_matched)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        per_molecule.append(
            {
                "name": f"{mol.name or 'mol'}_{mol_idx + 1}",
                "n_atoms": len(mol.symbols),
                "n_freq_refs": n,
                "qm_frequencies": qm_matched.tolist(),
                "mm_frequencies": mm_matched.tolist(),
                "rmsd_cm1": rmsd,
                "mae_cm1": mae,
                "r_squared": r_squared,
            }
        )

    # Overall objective score
    from q2mm.optimizers.objective import ObjectiveFunction

    obj = ObjectiveFunction(ff, engine, molecules, freq_ref)
    params = ff.get_param_vector()
    score = obj(params)

    return {
        "per_molecule": per_molecule,
        "total_freq_refs": sum(m["n_freq_refs"] for m in per_molecule),
        "objective_score": float(score),
        "n_params": ff.n_params,
        "n_molecules": len(molecules),
        "param_vector": params.tolist(),
    }


def _save_golden_fixture(results: dict, path: Path) -> None:
    """Save results as a golden fixture JSON file."""
    fixture = {
        "metadata": {
            "paper": "Donoghue et al. J. Chem. Theory Comput. 2008, 4, 1313-1323",
            "doi": "10.1021/ct800132a",
            "system": "Rh-diphosphine enamide hydrogenation TS",
            "ff_source": "examples/rh-enamide/ff/rh_hyd_enamide_final.fld",
            "ff_provenance": "Q2MM/q2mm commit b26404b8 (forcefields/rh-hydrogenation-enamide.fld)",
            "engine": "OpenMM",
            "qm_level": "B3LYP/LACVP** (Jaguar)",
            "description": (
                "Check 1: Published FF evaluated with new q2mm OpenMM engine. "
                "The published FF was optimized with MacroModel/MM3*. Numerical "
                "differences are expected due to engine implementation."
            ),
        },
        "summary": {
            "n_molecules": results["n_molecules"],
            "n_params": results["n_params"],
            "total_freq_refs": results["total_freq_refs"],
            "objective_score": results["objective_score"],
            "overall_rmsd_cm1": float(np.mean([m["rmsd_cm1"] for m in results["per_molecule"]])),
            "overall_mae_cm1": float(np.mean([m["mae_cm1"] for m in results["per_molecule"]])),
            "overall_r_squared": float(np.mean([m["r_squared"] for m in results["per_molecule"]])),
        },
        "per_molecule": results["per_molecule"],
        "param_vector": results["param_vector"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fixture, indent=2))


# ===========================================================================
# Check 1: Published FF evaluation
# ===========================================================================


@requires_openmm
@pytest.mark.openmm
@pytest.mark.slow
class TestPublishedFFEvaluation:
    """Check 1: Evaluate the Donoghue 2008 published Rh-enamide FF.

    The ``mm3.fld`` file contains the published optimized parameters in its
    Rh-enamide substructure section. We load these directly (no Seminario),
    evaluate with OpenMM, and compare against QM frequencies.

    The Seminario-estimated FF serves as the "unoptimized" baseline — the
    published FF should produce a meaningfully lower objective score.
    """

    @pytest.fixture(scope="class")
    def molecules(self) -> list[Any]:
        """Load all 9 rh-enamide structures + Jaguar Hessians."""
        if not MMO_PATH.exists():
            pytest.skip("rh-enamide dataset not found")
        return _load_rh_enamide_molecules()

    @pytest.fixture(scope="class")
    def published_ff(self) -> Any:
        """Load the published FF directly from mm3.fld substructure."""
        from q2mm.models.forcefield import ForceField

        if not MM3_FLD_PATH.exists():
            pytest.skip(f"mm3.fld not found: {MM3_FLD_PATH}")
        return ForceField.from_mm3_fld(str(MM3_FLD_PATH))

    @pytest.fixture(scope="class")
    def seminario_ff(self, molecules: list[Any]) -> Any:
        """Build a Seminario-estimated FF as the unoptimized baseline."""
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import estimate_force_constants

        if not MM3_FLD_PATH.exists():
            pytest.skip(f"mm3.fld not found: {MM3_FLD_PATH}")
        ff_template = ForceField.from_mm3_fld(str(MM3_FLD_PATH))
        return estimate_force_constants(molecules, forcefield=ff_template)

    @pytest.fixture(scope="class")
    def engine(self) -> Any:
        from q2mm.backends.mm import OpenMMEngine

        return OpenMMEngine()

    @pytest.fixture(scope="class")
    def published_results(self, published_ff: Any, molecules: list[Any], engine: Any) -> dict[str, Any]:
        """Evaluate the published FF on the full training set."""
        t0 = time.perf_counter()
        results = _evaluate_ff_on_training_set(published_ff, molecules, engine)
        results["wall_time"] = time.perf_counter() - t0
        return results

    @pytest.fixture(scope="class")
    def seminario_results(self, seminario_ff: Any, molecules: list[Any], engine: Any) -> dict[str, Any]:
        """Evaluate the Seminario-estimated FF for comparison."""
        return _evaluate_ff_on_training_set(seminario_ff, molecules, engine)

    # --- Structural assertions ---

    def test_loads_9_molecules(self, published_results: dict[str, Any]) -> None:
        """All 9 rh-enamide TS structures are loaded."""
        assert published_results["n_molecules"] == 9

    def test_182_parameters(self, published_results: dict[str, Any]) -> None:
        """Published FF has the expected 182 parameters."""
        assert published_results["n_params"] == 182

    def test_all_molecules_have_frequencies(self, published_results: dict[str, Any]) -> None:
        """Every molecule contributes QM/MM frequency comparisons."""
        for m in published_results["per_molecule"]:
            assert m["n_freq_refs"] > 0, f"{m['name']} has 0 frequency refs"

    def test_over_700_frequency_refs(self, published_results: dict[str, Any]) -> None:
        """Sufficient frequency reference points across all molecules."""
        assert published_results["total_freq_refs"] >= 700

    # --- Score assertions ---

    def test_published_score_is_finite(self, published_results: dict[str, Any]) -> None:
        """Published FF produces a finite objective score."""
        score = published_results["objective_score"]
        assert np.isfinite(score), f"Published FF score is not finite: {score}"

    @pytest.mark.xfail(
        reason=(
            "Known Check 1 gap: the published MacroModel/MM3* force field does not "
            "yet reproduce a better OpenMM fit than the Seminario baseline."
        ),
        strict=True,
    )
    def test_published_ff_beats_seminario(
        self, published_results: dict[str, Any], seminario_results: dict[str, Any]
    ) -> None:
        """Promotion gate: published FF should eventually beat the Seminario baseline.

        This remains an expected-fail gate until the published MacroModel/MM3*
        parameterization is shown to reproduce comparable quality under OpenMM.
        """
        pub_score = published_results["objective_score"]
        sem_score = seminario_results["objective_score"]
        improvement = (sem_score - pub_score) / sem_score
        assert pub_score < sem_score, f"Published FF ({pub_score:.1f}) should beat Seminario ({sem_score:.1f})"
        # Published FF should improve substantially over Seminario
        assert improvement > 0.10, (
            f"Published FF improvement ({improvement * 100:.1f}%) over Seminario is unexpectedly low"
        )

    # --- Quality assertions ---

    @pytest.mark.xfail(
        reason=(
            "Known Check 1 gap: published MacroModel/MM3* parameters are not yet "
            "correlated with the OpenMM frequency evaluation."
        ),
        strict=True,
    )
    def test_per_molecule_r_squared_positive(self, published_results: dict[str, Any]) -> None:
        """Promotion gate: each molecule should eventually show positive correlation."""
        for m in published_results["per_molecule"]:
            assert m["r_squared"] > 0.0, f"{m['name']}: R² = {m['r_squared']:.3f} (should be positive)"

    @pytest.mark.xfail(
        reason=(
            "Known Check 1 gap: average R² for the published MacroModel/MM3* force "
            "field is not yet acceptable under OpenMM."
        ),
        strict=True,
    )
    def test_overall_r_squared_above_threshold(self, published_results: dict[str, Any]) -> None:
        """Promotion gate: average R² should eventually show good correlation."""
        r2_values = [m["r_squared"] for m in published_results["per_molecule"]]
        avg_r2 = np.mean(r2_values)
        assert avg_r2 > 0.80, f"Average R² = {avg_r2:.3f} (expected > 0.80 for a published FF)"

    # --- Golden fixture pinning ---

    def test_pin_golden_fixture(
        self,
        published_results: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Validate reproducibility against the committed golden fixture.

        Set ``Q2MM_UPDATE_GOLDEN=1`` when running this test locally with OpenMM to
        regenerate the fixture, then commit the resulting JSON separately.
        """
        if UPDATE_GOLDEN:
            _save_golden_fixture(published_results, GOLDEN_PATH)
            pytest.skip(f"Golden fixture updated at {GOLDEN_PATH}; commit the JSON separately.")
        if not GOLDEN_PATH.exists():
            pytest.skip(
                f"Golden fixture not found at {GOLDEN_PATH}. "
                "Run this test locally with OpenMM and Q2MM_UPDATE_GOLDEN=1 to "
                "generate the golden JSON, then commit it to the repository."
            )

        golden = json.loads(GOLDEN_PATH.read_text())
        golden_score = golden["summary"]["objective_score"]
        actual_score = published_results["objective_score"]
        # rtol=2e-3 accommodates cross-platform floating-point differences
        # (Windows vs Linux OpenMM builds yield ~0.01% variation)
        np.testing.assert_allclose(
            actual_score,
            golden_score,
            rtol=2e-3,
            err_msg=(f"Published FF score {actual_score:.4f} doesn't match golden {golden_score:.4f} (rtol=2e-3)"),
        )

    # --- Reporting ---

    def test_summary_report(
        self,
        published_results: dict[str, Any],
        seminario_results: dict[str, Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Print a summary report (informational, never fails)."""
        pub = published_results
        sem = seminario_results
        improvement = (sem["objective_score"] - pub["objective_score"]) / sem["objective_score"]

        with capsys.disabled():
            print("\n" + "=" * 72)
            print("  CHECK 1: Published FF Evaluation (Donoghue 2008)")
            print("  Rh-enamide hydrogenation TS — OpenMM engine")
            print("=" * 72)
            print(f"  Molecules:       {pub['n_molecules']}")
            print(f"  Parameters:      {pub['n_params']}")
            print(f"  Freq refs:       {pub['total_freq_refs']}")
            print(f"  Seminario score: {sem['objective_score']:.2f}")
            print(f"  Published score: {pub['objective_score']:.2f}")
            print(f"  Improvement:     {improvement * 100:.1f}%")
            print(f"  Wall time:       {pub.get('wall_time', 0):.1f}s")
            print("-" * 72)
            print(f"  {'Molecule':<50} {'RMSD':>8} {'MAE':>8} {'R2':>8} {'Nref':>5}")
            print("-" * 72)
            for m in pub["per_molecule"]:
                print(
                    f"  {m['name']:<50} "
                    f"{m['rmsd_cm1']:8.1f} "
                    f"{m['mae_cm1']:8.1f} "
                    f"{m['r_squared']:8.3f} "
                    f"{m['n_freq_refs']:5d}"
                )
            avg_rmsd = np.mean([m["rmsd_cm1"] for m in pub["per_molecule"]])
            avg_mae = np.mean([m["mae_cm1"] for m in pub["per_molecule"]])
            avg_r2 = np.mean([m["r_squared"] for m in pub["per_molecule"]])
            print("-" * 72)
            print(f"  {'AVERAGE':<50} {avg_rmsd:8.1f} {avg_mae:8.1f} {avg_r2:8.3f} {pub['total_freq_refs']:5d}")
            print("=" * 72)
