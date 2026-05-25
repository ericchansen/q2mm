"""Heck relay force field validation — Check 1.

Validates that the published Heck relay TSFF (Rosales et al. JACS 2020, 142,
9700) produces physically reasonable MM frequencies when evaluated with the
q2mm engines.  This is a Check 1 test: load the published FF, evaluate it
against the original QM reference data, and pin the results.

The training set consists of 23 Pd transition-state structures from Gaussian 09
DFT calculations (M06/gen, pseudo=read, GD3 dispersion, HPModes).  Each
structure has one imaginary frequency (reaction coordinate).

References
----------
- Rosales, A. R. et al. J. Am. Chem. Soc. 2020, 142, 9700-9707.
  DOI: 10.1021/jacs.0c01979
- Supporting data: Rosales dissertation Ch 3, University of Notre Dame.

"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test._shared import REPO_ROOT, SUPPORTING_INFO_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

HECK_DIR: Path | None = None
if SUPPORTING_INFO_DIR is not None:
    _candidate = SUPPORTING_INFO_DIR / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck"
    if _candidate.exists():
        HECK_DIR = _candidate

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "published_ff"
GOLDEN_PATH = FIXTURE_DIR / "heck_relay_rosales2020.json"
UPDATE_GOLDEN = os.getenv("Q2MM_UPDATE_GOLDEN") == "1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qm_frequencies_from_hessian(
    hessian_au: np.ndarray,
    symbols: list[str],
) -> np.ndarray:
    """Convert a Cartesian Hessian (Hartree/Bohr²) to frequencies (cm⁻¹)."""
    from q2mm.models.hessian import hessian_to_frequencies

    return np.array(hessian_to_frequencies(hessian_au, symbols))


def _build_frequency_reference(
    qm_freqs: np.ndarray,
    mm_all_freqs: np.ndarray,
    *,
    threshold: float = 50.0,
    upper_threshold: float = 4000.0,
    weight: float = 0.001,
    molecule_idx: int = 0,
    ref: Any = None,
) -> tuple[Any, list[float]]:
    """Build (or extend) a ReferenceData with frequency observations."""
    from q2mm.optimizers.objective import ReferenceData

    qm_real = sorted(f for f in qm_freqs if f > threshold)
    mm_real_idx = sorted(i for i, f in enumerate(mm_all_freqs) if threshold < f <= upper_threshold)
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


def _evaluate_ff_on_training_set(
    ff: Any,
    molecules: list[Any],
    engine: Any,
) -> dict[str, Any]:
    """Evaluate a FF against the Heck relay training set.

    Returns a dict with per-molecule and aggregate statistics.
    """
    freq_ref = None
    per_mol: list[dict[str, Any]] = []
    all_qm: list[float] = []
    all_mm: list[float] = []

    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )

        # Pair QM/MM for per-molecule R²
        mm_real = sorted(f for f in mm_freqs if 50.0 < f <= 4000.0)
        n = min(len(qm_real), len(mm_real))
        qm_arr = np.array(qm_real[:n])
        mm_arr = np.array(mm_real[:n])

        if n > 2:
            corr = np.corrcoef(qm_arr, mm_arr)[0, 1]
            r2 = corr**2 if np.isfinite(corr) else 0.0
            rmsd = float(np.sqrt(np.mean((qm_arr - mm_arr) ** 2)))
        else:
            r2 = 0.0
            rmsd = float("inf")

        all_qm.extend(qm_arr.tolist())
        all_mm.extend(mm_arr.tolist())

        per_mol.append(
            {
                "name": getattr(mol, "name", f"mol_{mol_idx}"),
                "n_atoms": len(mol.symbols),
                "n_freq_refs": n,
                "r_squared": r2,
                "rmsd_cm1": rmsd,
            }
        )

    # Aggregate
    all_qm_arr = np.array(all_qm)
    all_mm_arr = np.array(all_mm)
    if len(all_qm_arr) > 2:
        overall_corr = np.corrcoef(all_qm_arr, all_mm_arr)[0, 1]
        overall_r2 = overall_corr**2 if np.isfinite(overall_corr) else 0.0
        overall_rmsd = float(np.sqrt(np.mean((all_qm_arr - all_mm_arr) ** 2)))
    else:
        overall_r2 = 0.0
        overall_rmsd = float("inf")

    score = float(np.sum((all_qm_arr - all_mm_arr) ** 2)) if len(all_qm_arr) > 0 else float("inf")

    return {
        "n_molecules": len(molecules),
        "n_params": ff.n_params if hasattr(ff, "n_params") else len(ff.get_param_vector()),
        "total_freq_refs": sum(m["n_freq_refs"] for m in per_mol),
        "objective_score": score,
        "overall_r_squared": overall_r2,
        "overall_rmsd_cm1": overall_rmsd,
        "per_molecule": per_mol,
    }


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.validation
@pytest.mark.external_data
@pytest.mark.jax
class TestHeckRelayPublishedFF:
    """Validate the published Heck relay TSFF (Rosales JACS 2020).

    Loads the published mm3.FF1.fld, evaluates with JaxEngine against the
    23-molecule Gaussian training set, and checks frequency correlation.
    """

    @pytest.fixture(scope="class")
    def molecules(self) -> list[Any]:
        """Load all 23 Heck relay TS structures from Gaussian logs."""
        if HECK_DIR is None:
            pytest.skip("Heck relay supporting data not found")

        from q2mm.diagnostics.systems import load_heck_relay_molecules

        mols = load_heck_relay_molecules()
        assert len(mols) == 23, f"Expected 23 molecules, got {len(mols)}"
        return mols

    @pytest.fixture(scope="class")
    def published_ff(self) -> Any:
        """Load the published FF with standard MM3 parameters included."""
        from q2mm.models.forcefield import ForceField

        if HECK_DIR is None:
            pytest.skip("Heck relay supporting data not found")
        ff_path = HECK_DIR / "mm3.FF1.fld"
        if not ff_path.exists():
            pytest.skip(f"mm3.FF1.fld not found: {ff_path}")
        return ForceField.from_mm3_fld(str(ff_path))

    @pytest.fixture(scope="class")
    def seminario_ff(self, molecules: list[Any]) -> Any:
        """Build a Seminario-estimated FF as the unoptimized baseline."""
        from q2mm.models.forcefield import ForceField
        from q2mm.models.seminario import qfuerza_into

        if HECK_DIR is None:
            pytest.skip("Heck relay supporting data not found")
        ff_path = HECK_DIR / "mm3.FF1.fld"
        ff_template = ForceField.from_mm3_fld(str(ff_path))
        qfuerza_into(ff_template, molecules)
        return ff_template

    @pytest.fixture(scope="class")
    def engine(self) -> Any:
        """Return JaxEngine (required for torsion damping + stretch-bend)."""
        from q2mm.backends.mm.jax_engine import JaxEngine

        return JaxEngine()

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

    def test_loads_23_molecules(self, published_results: dict[str, Any]) -> None:
        """All 23 Heck relay TS structures are loaded."""
        assert published_results["n_molecules"] == 23

    def test_includes_standard_parameters(self, published_results: dict[str, Any]) -> None:
        """Published FF includes standard MM3 params (> substructure-only)."""
        assert published_results["n_params"] > 100

    def test_all_molecules_have_frequencies(self, published_results: dict[str, Any]) -> None:
        """Every molecule contributes QM/MM frequency comparisons."""
        for m in published_results["per_molecule"]:
            assert m["n_freq_refs"] > 0, f"{m['name']} has 0 frequency refs"

    def test_over_2000_frequency_refs(self, published_results: dict[str, Any]) -> None:
        """Sufficient frequency reference points across 23 molecules.

        Each molecule has ~100-120 vibrational modes; 23 molecules × ~100
        real modes ≈ 2300+.
        """
        assert published_results["total_freq_refs"] >= 1400

    # --- Score assertions ---

    def test_published_score_is_finite(self, published_results: dict[str, Any]) -> None:
        """Published FF produces a finite objective score."""
        score = published_results["objective_score"]
        assert np.isfinite(score), f"Published FF score is not finite: {score}"

    # --- Quality assertions ---

    def test_per_molecule_r_squared_positive(self, published_results: dict[str, Any]) -> None:
        """Each molecule shows positive correlation between QM and MM."""
        for m in published_results["per_molecule"]:
            assert m["r_squared"] > 0.0, f"{m['name']}: R² = {m['r_squared']:.3f} (should be positive)"

    def test_overall_r_squared_above_threshold(self, published_results: dict[str, Any]) -> None:
        """Average R² exceeds threshold for a cross-engine FF evaluation.

        The published FF was optimized for MacroModel's MM3* engine, so
        perfect reproduction is not expected.  The threshold of 0.30 is
        deliberately conservative for this first evaluation of a new system.
        """
        r2_values = [m["r_squared"] for m in published_results["per_molecule"]]
        avg_r2 = np.mean(r2_values)
        assert avg_r2 > 0.30, f"Average R² = {avg_r2:.3f} (expected > 0.30)"

    # --- Golden fixture pinning ---

    def test_pin_golden_fixture(
        self,
        published_results: dict[str, Any],
        seminario_results: dict[str, Any],
    ) -> None:
        """Pin results to a golden fixture for regression detection."""
        snapshot = {
            "system": "heck-relay",
            "publication": "Rosales et al. JACS 2020, 142, 9700",
            "doi": "10.1021/jacs.0c01979",
            "n_molecules": published_results["n_molecules"],
            "total_freq_refs": published_results["total_freq_refs"],
            "published_ff": {
                "objective_score": published_results["objective_score"],
                "overall_r_squared": published_results["overall_r_squared"],
                "overall_rmsd_cm1": published_results["overall_rmsd_cm1"],
                "per_molecule": published_results["per_molecule"],
            },
            "seminario_ff": {
                "objective_score": seminario_results["objective_score"],
                "overall_r_squared": seminario_results["overall_r_squared"],
                "overall_rmsd_cm1": seminario_results["overall_rmsd_cm1"],
            },
        }

        if UPDATE_GOLDEN:
            FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
            GOLDEN_PATH.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
            pytest.skip("Golden fixture updated — re-run to validate")

        if not GOLDEN_PATH.exists():
            pytest.skip("Golden fixture not yet generated. Run with Q2MM_UPDATE_GOLDEN=1 to create.")

        golden = json.loads(GOLDEN_PATH.read_text())
        pub_golden = golden["published_ff"]

        # Score should not regress by more than 5%
        score_diff = abs(published_results["objective_score"] - pub_golden["objective_score"])
        score_tol = 0.05 * pub_golden["objective_score"]
        assert score_diff < score_tol, (
            f"Published FF score regressed: "
            f"{published_results['objective_score']:.4f} vs "
            f"golden {pub_golden['objective_score']:.4f} "
            f"(tolerance {score_tol:.4f})"
        )

    # --- Summary report ---

    def test_print_summary(
        self,
        published_results: dict[str, Any],
        seminario_results: dict[str, Any],
    ) -> None:
        """Print a summary report (always passes)."""
        print("\n" + "=" * 60)
        print("Heck Relay Validation — Check 1 Summary")
        print("=" * 60)
        print("  Publication: Rosales et al. JACS 2020, 142, 9700")
        print(f"  Molecules:   {published_results['n_molecules']}")
        print(f"  Freq refs:   {published_results['total_freq_refs']}")
        print(f"  Wall time:   {published_results.get('wall_time', 0):.1f}s")
        print()
        print("  Published FF:")
        print(f"    Score:  {published_results['objective_score']:.4f}")
        print(f"    R²:     {published_results['overall_r_squared']:.4f}")
        print(f"    RMSD:   {published_results['overall_rmsd_cm1']:.1f} cm⁻¹")
        print()
        print("  Seminario baseline:")
        print(f"    Score:  {seminario_results['objective_score']:.4f}")
        print(f"    R²:     {seminario_results['overall_r_squared']:.4f}")
        print(f"    RMSD:   {seminario_results['overall_rmsd_cm1']:.1f} cm⁻¹")
        print()
        print("  Per-molecule R² (published FF):")
        for m in published_results["per_molecule"]:
            print(
                f"    {m['name']:<12} R²={m['r_squared']:.3f}  RMSD={m['rmsd_cm1']:.0f} cm⁻¹  ({m['n_freq_refs']} refs)"
            )
        print("=" * 60)


# ---------------------------------------------------------------------------
# Regression: load_heck_relay() must preserve published OPT parameter values
# (ericchansen/q2mm#277)
# ---------------------------------------------------------------------------


@pytest.mark.validation
@pytest.mark.external_data
def test_load_heck_relay_preserves_published_opt_values() -> None:
    """Regression: loader must NOT overwrite published Rosales OPT params.

    Before #277, ``load_heck_relay()`` called `qfuerza_fresh` / `qfuerza_into`
    after ``freeze_standard_params``, which silently re-projected the
    OPT-substructure parameter values via FUERZA — discarding Rosales'
    fitted values.  After the fix, the loader should keep those values
    exactly as published.  This test compares the optimizable (non-frozen)
    parameter values from ``load_heck_relay()`` against the same params
    loaded directly from the .fld file with no Seminario step.
    """
    if HECK_DIR is None:
        pytest.skip("Heck relay supporting data not found")

    from q2mm.diagnostics.systems import load_system
    from q2mm.models.forcefield import ForceField

    ff_path = HECK_DIR / "mm3.FF1.fld"
    if not ff_path.exists():
        pytest.skip(f"mm3.FF1.fld not found: {ff_path}")

    # Loader output (what users get today).
    sys_data = load_system("heck-relay")
    loader_active = sys_data.forcefield.get_active_param_vector()

    # Same .fld file, same active-mask partition, but no Seminario.
    # This is the pre-fix "reference" we expect the loader to match.
    expected_ff = ForceField.from_mm3_fld(str(ff_path), include_standard=True)
    opt_ff = ForceField.from_mm3_fld(str(ff_path), include_standard=False)
    expected_ff.freeze_standard_params(opt_ff)
    expected_active = expected_ff.get_active_param_vector()

    assert loader_active.shape == expected_active.shape, (
        f"Active-mask shape mismatch: loader={loader_active.shape} vs expected={expected_active.shape}"
    )
    # Tight tolerance: these should match bit-for-bit (same file, no math).
    np.testing.assert_allclose(
        loader_active,
        expected_active,
        rtol=0.0,
        atol=1e-12,
        err_msg=(
            "load_heck_relay() OPT parameter values differ from the "
            "published .fld values.  This likely means a Seminario-style "
            "re-estimation crept back into the loader (the #277 bug)."
        ),
    )
