"""Published force field validation — Check 1 (all systems).

"Check 1" is the foundational Q2MM validation: take the force field the
authors actually *published*, evaluate it against the same QM reference data
with our engines, and pin the result as a golden fixture so future engine
changes can't silently move it.

This module runs Check 1 for every registered publication TS system through a
single, uniform pipeline:

* molecules + published force field come from
  ``load_system(key, starting_point="published")`` (one loader path for all
  systems — see :mod:`q2mm.diagnostics.systems`);
* frequencies come from :func:`q2mm.models.hessian.hessian_to_frequencies`
  (QM) and ``JaxEngine.frequencies`` (MM);
* the objective score is the real :class:`~q2mm.optimizers.objective.
  ObjectiveFunction` frequency penalty;
* R² is the coefficient of determination (``1 - SSres/SStot``).

The per-system parameters (expected molecule count, citation metadata, golden
filename) live in :data:`CHECK1_SPECS`.  Everything else is derived, so adding
a sixth system is a single dict entry plus a regenerated golden.

Regression, not quality gate
----------------------------
The published force fields were optimized for MacroModel's MM3* engine, which
has physics ours does not yet reproduce (stretch-bend cross terms, metal-center
torsion rules — ericchansen/q2mm#255, tracked for the backend-parity audit).
Under ``JaxEngine`` the cross-engine agreement ranges from good
(rh-enamide R² ≈ 0.80) to poor (pd-allyl R² < 0).  This module therefore does
**not** assert an absolute R² floor — it pins each system's score and R² to a
committed golden and fails only on *regression* from that pinned value.  The
absolute quality of the cross-engine reproduction is the subject of #255, not
of this test.

Regenerating goldens
--------------------
Run locally on GPU with::

    Q2MM_UPDATE_GOLDEN=1 python -m pytest \
        test/integration/test_published_ff_validation.py --run-validation

then commit the updated JSON files under ``test/fixtures/published_ff/``.

References
----------
- Donoghue, P. J. et al. J. Chem. Theory Comput. 2008, 4, 1313. (rh-enamide)
- Rosales, A. R. et al. J. Am. Chem. Soc. 2020, 142, 9700. (heck-relay)
- Wahlers, J. et al. Nat. Commun. 2021, 12, 6508. (pd-allyl)
- Wahlers, J. et al. J. Org. Chem. 2021, 86, 5660. (pd-conjugate)
- Wahlers, J. Ph.D. Dissertation, Univ. of Notre Dame, 2022, Ch. 6. (rh-conjugate)

"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from test._shared import REPO_ROOT

# ---------------------------------------------------------------------------
# Paths / environment
# ---------------------------------------------------------------------------

FIXTURE_DIR = REPO_ROOT / "test" / "fixtures" / "published_ff"
UPDATE_GOLDEN = os.getenv("Q2MM_UPDATE_GOLDEN") == "1"

_HAS_JAX = importlib.util.find_spec("jax") is not None

# Score regression tolerance.  Windows vs Linux OpenMM/JAX yields ~0.01%
# relative difference; 0.05% gives ~4× headroom while still catching real
# regressions.
_SCORE_RTOL = 5e-4
# R² is a derived, bounded quantity; an absolute tolerance is more meaningful
# than a relative one (relative is meaningless near R² = 0).
_R2_ATOL = 1e-3


# ---------------------------------------------------------------------------
# Per-system specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check1Spec:
    """Everything Check 1 needs to know about one publication system."""

    key: str
    """``load_system`` registry key."""
    golden_name: str
    """Golden fixture filename under :data:`FIXTURE_DIR`."""
    n_molecules: int
    """Expected training-set size (independent sanity check)."""
    metadata: dict[str, Any] = field(default_factory=dict)
    """Citation / provenance block copied verbatim into the golden."""

    @property
    def golden_path(self) -> Path:
        return FIXTURE_DIR / self.golden_name


# Citation metadata is sourced from the previously committed golden fixtures
# (traceable provenance — AGENTS.md §2).  ``qm_level`` documents the QM method
# behind each training set.
CHECK1_SPECS: dict[str, Check1Spec] = {
    "rh-enamide": Check1Spec(
        key="rh-enamide",
        golden_name="rh_enamide_donoghue2008.json",
        n_molecules=9,
        metadata={
            "paper": "Donoghue et al. J. Chem. Theory Comput. 2008, 4, 1313-1323",
            "doi": "10.1021/ct800132a",
            "system": "Rh-diphosphine enamide hydrogenation TS",
            "qm_level": "B3LYP/LACVP** (Jaguar)",
        },
    ),
    "heck-relay": Check1Spec(
        key="heck-relay",
        golden_name="heck_relay_rosales2020.json",
        n_molecules=23,
        metadata={
            "paper": "Rosales et al. J. Am. Chem. Soc. 2020, 142, 9700-9707",
            "doi": "10.1021/jacs.0c01979",
            "system": "Heck-relay Pd migratory-insertion TS",
            "qm_level": "M06/gen GD3, pseudo=read (Gaussian 09)",
        },
    ),
    "pd-allyl": Check1Spec(
        key="pd-allyl",
        golden_name="pd_allyl_wahlers2021.json",
        n_molecules=21,
        metadata={
            "paper": "Wahlers, J. et al. Nat. Commun. 2021, 12, 6508",
            "doi": "10.1038/s41467-021-27065-2",
            "system": "Pd-catalyzed enantioselective allylic amination",
            "qm_level": "Wahlers dissertation Ch. 3 (Gaussian)",
            "paper_internal_r2_hessian": 0.998,
            "paper_internal_r2_geometry": 0.988,
            "paper_internal_r2_charges": 0.822,
            "paper_external_n_predictions": 77,
            "paper_external_mue_kjmol": 4.4,
            "paper_external_r2_selectivity": 0.41,
        },
    ),
    "pd-conjugate": Check1Spec(
        key="pd-conjugate",
        golden_name="pd_conjugate_wahlers2021.json",
        n_molecules=10,
        metadata={
            "paper": "Wahlers, J. et al. J. Org. Chem. 2021, 86, 5660",
            "doi": "10.1021/acs.joc.1c00136",
            "system": "Pd-catalyzed 1,4-conjugate addition",
            "qm_level": "Wahlers dissertation Ch. 5 (Gaussian)",
        },
    ),
    "rh-conjugate": Check1Spec(
        key="rh-conjugate",
        golden_name="rh_conjugate_wahlers2022.json",
        n_molecules=10,
        metadata={
            "paper": "Wahlers, J. Ph.D. Dissertation, University of Notre Dame, 2022, Ch. 6",
            "doi": None,
            "system": "Rh-catalyzed 1,4-conjugate addition",
            "qm_level": "Wahlers dissertation Ch. 6 (Gaussian)",
        },
    ),
}

_SYSTEM_IDS = list(CHECK1_SPECS)
_SPEC_PARAMS = list(CHECK1_SPECS.values())


# ---------------------------------------------------------------------------
# Canonical evaluation helpers
# ---------------------------------------------------------------------------


def _qm_frequencies_from_hessian(hessian_au: np.ndarray, symbols: list[str]) -> np.ndarray:
    """Harmonic frequencies (cm⁻¹) from a Cartesian Hessian in Hartree/Bohr²."""
    from q2mm.models.hessian import hessian_to_frequencies

    return np.asarray(hessian_to_frequencies(hessian_au, symbols), dtype=float)


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
    """Build (or extend) a ReferenceData with frequency observations.

    Frequencies below *threshold* are excluded (near-zero rigid-body modes and
    QM imaginary modes).  MM frequencies above *upper_threshold* are excluded —
    these correspond to the reaction-coordinate mode whose QM counterpart is
    imaginary and thus already excluded, matching the Q2MM eig_i = 0.00 weight
    convention.
    """
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
    *,
    upper_threshold: float = 4000.0,
) -> dict[str, Any]:
    """Evaluate a force field against QM reference frequencies.

    Returns a dict with per-molecule and overall statistics.  R² is the
    coefficient of determination; the objective score is the real
    :class:`ObjectiveFunction` frequency penalty.
    """
    freq_ref = None
    per_molecule: list[dict[str, Any]] = []

    for mol_idx, mol in enumerate(molecules):
        mm_freqs = engine.frequencies(mol, ff)
        qm_freqs = _qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        freq_ref, qm_real = _build_frequency_reference(
            qm_freqs,
            mm_freqs,
            upper_threshold=upper_threshold,
            molecule_idx=mol_idx,
            ref=freq_ref,
        )

        mm_real = sorted(f for f in mm_freqs if 50.0 < f <= upper_threshold)
        n = min(len(qm_real), len(mm_real))
        qm_matched = np.array(qm_real[:n])
        mm_matched = np.array(mm_real[:n])
        residuals = qm_matched - mm_matched
        rmsd = float(np.sqrt(np.mean(residuals**2))) if n else float("inf")
        mae = float(np.mean(np.abs(residuals))) if n else float("inf")

        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((qm_matched - np.mean(qm_matched)) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        per_molecule.append(
            {
                "name": f"{getattr(mol, 'name', None) or 'mol'}_{mol_idx + 1}",
                "n_atoms": len(mol.symbols),
                "n_freq_refs": n,
                "qm_frequencies": qm_matched.tolist(),
                "mm_frequencies": mm_matched.tolist(),
                "rmsd_cm1": rmsd,
                "mae_cm1": mae,
                "r_squared": r_squared,
            }
        )

    from q2mm.optimizers.objective import ObjectiveFunction

    obj = ObjectiveFunction(ff, engine, molecules, freq_ref)
    params = ff.get_param_vector()
    score = float(obj(params))

    return {
        "per_molecule": per_molecule,
        "total_freq_refs": sum(m["n_freq_refs"] for m in per_molecule),
        "objective_score": score,
        "overall_rmsd_cm1": float(np.mean([m["rmsd_cm1"] for m in per_molecule])),
        "overall_mae_cm1": float(np.mean([m["mae_cm1"] for m in per_molecule])),
        "overall_r_squared": float(np.mean([m["r_squared"] for m in per_molecule])),
        "n_params": ff.n_params if hasattr(ff, "n_params") else len(params),
        "n_molecules": len(molecules),
        "param_vector": params.tolist(),
    }


def _golden_payload(spec: Check1Spec, results: dict[str, Any]) -> dict[str, Any]:
    """Assemble the on-disk golden fixture from an evaluation result."""
    metadata = dict(spec.metadata)
    metadata.setdefault("engine", "JaxEngine")
    metadata["description"] = (
        "Check 1: published FF evaluated with q2mm JaxEngine. "
        "Reaction-coordinate frequencies (>4000 cm⁻¹) excluded from comparison, "
        "matching the eig_i = 0.00 weight convention. The published FF was "
        "optimized for MacroModel/MM3*; cross-engine differences are expected "
        "and tracked by ericchansen/q2mm#255."
    )
    return {
        "metadata": metadata,
        "summary": {
            "n_molecules": results["n_molecules"],
            "n_params": results["n_params"],
            "total_freq_refs": results["total_freq_refs"],
            "objective_score": results["objective_score"],
            "overall_rmsd_cm1": results["overall_rmsd_cm1"],
            "overall_mae_cm1": results["overall_mae_cm1"],
            "overall_r_squared": results["overall_r_squared"],
        },
        "per_molecule": results["per_molecule"],
        "param_vector": results["param_vector"],
    }


# ---------------------------------------------------------------------------
# Memoized per-system evaluation (one GPU pass per system)
# ---------------------------------------------------------------------------

_ENGINE: Any = None
_RESULTS_CACHE: dict[str, dict[str, Any]] = {}


def _get_engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        from q2mm.backends.mm.jax_engine import JaxEngine

        _ENGINE = JaxEngine()
    return _ENGINE


def _results_for(spec: Check1Spec) -> dict[str, Any]:
    """Load + evaluate a system once, caching across the module's tests.

    Skips gracefully when the (gitignored) training data is absent.
    """
    if spec.key in _RESULTS_CACHE:
        return _RESULTS_CACHE[spec.key]

    from q2mm.diagnostics.systems import load_system

    try:
        sd = load_system(spec.key, starting_point="published")
    except FileNotFoundError as exc:
        pytest.skip(f"{spec.key}: training data not found ({exc})")

    t0 = time.perf_counter()
    results = _evaluate_ff_on_training_set(sd.forcefield, sd.molecules, _get_engine())
    results["wall_time"] = time.perf_counter() - t0

    if UPDATE_GOLDEN:
        spec.golden_path.parent.mkdir(parents=True, exist_ok=True)
        spec.golden_path.write_text(json.dumps(_golden_payload(spec, results), indent=2) + "\n")

    _RESULTS_CACHE[spec.key] = results
    return results


# ===========================================================================
# Check 1: parametrized over every publication system
# ===========================================================================


@pytest.mark.skipif(not _HAS_JAX, reason="JAX required for torsion damping + stretch-bend")
@pytest.mark.validation
@pytest.mark.external_data
@pytest.mark.jax
@pytest.mark.parametrize("spec", _SPEC_PARAMS, ids=_SYSTEM_IDS)
class TestPublishedFFCheck1:
    """Evaluate each published TSFF and pin its behavior to a golden fixture."""

    # --- Structural assertions ---

    def test_loads_expected_molecules(self, spec: Check1Spec) -> None:
        """The training set has the expected number of structures."""
        results = _results_for(spec)
        assert results["n_molecules"] == spec.n_molecules

    def test_includes_standard_parameters(self, spec: Check1Spec) -> None:
        """The composed FF carries the full MM3 backbone, not just substructure."""
        results = _results_for(spec)
        assert results["n_params"] > 182

    def test_all_molecules_have_frequencies(self, spec: Check1Spec) -> None:
        """Every molecule contributes at least one QM/MM frequency pair."""
        results = _results_for(spec)
        for m in results["per_molecule"]:
            assert m["n_freq_refs"] > 0, f"{m['name']} has 0 frequency refs"

    def test_score_is_finite(self, spec: Check1Spec) -> None:
        """The objective score is a finite number."""
        results = _results_for(spec)
        score = results["objective_score"]
        assert np.isfinite(score), f"{spec.key}: score is not finite ({score})"

    # --- Golden regression gate ---

    def test_matches_golden(self, spec: Check1Spec) -> None:
        """Score, R², and structural counts match the committed golden.

        This is the real gate: it detects *regression* from the pinned
        cross-engine behavior.  It does not judge whether that behavior is
        good (see the module docstring and #255).
        """
        results = _results_for(spec)
        if UPDATE_GOLDEN:
            pytest.skip(f"Golden updated at {spec.golden_path}; commit the JSON separately.")
        if not spec.golden_path.exists():
            pytest.skip(
                f"Golden fixture not found at {spec.golden_path}. "
                "Run locally on GPU with Q2MM_UPDATE_GOLDEN=1 to generate it, then commit."
            )

        golden = json.loads(spec.golden_path.read_text())
        gsum = golden["summary"]

        # Deterministic integer structural fields must match exactly.
        assert results["n_molecules"] == gsum["n_molecules"]
        assert results["n_params"] == gsum["n_params"]
        assert results["total_freq_refs"] == gsum["total_freq_refs"]

        np.testing.assert_allclose(
            results["objective_score"],
            gsum["objective_score"],
            rtol=_SCORE_RTOL,
            err_msg=(
                f"{spec.key}: score {results['objective_score']:.6g} regressed from "
                f"golden {gsum['objective_score']:.6g} (rtol={_SCORE_RTOL})"
            ),
        )
        np.testing.assert_allclose(
            results["overall_r_squared"],
            gsum["overall_r_squared"],
            atol=_R2_ATOL,
            err_msg=(
                f"{spec.key}: R² {results['overall_r_squared']:.4f} regressed from "
                f"golden {gsum['overall_r_squared']:.4f} (atol={_R2_ATOL})"
            ),
        )

    # --- Reporting ---

    def test_summary_report(self, spec: Check1Spec, capsys: pytest.CaptureFixture[str]) -> None:
        """Print a per-system summary (informational, never fails)."""
        results = _results_for(spec)
        with capsys.disabled():
            print("\n" + "=" * 72)
            print(f"  CHECK 1: {spec.key} — {spec.metadata.get('paper', '')}")
            print("=" * 72)
            print(f"  Molecules:   {results['n_molecules']}")
            print(f"  Parameters:  {results['n_params']}")
            print(f"  Freq refs:   {results['total_freq_refs']}")
            print(f"  Score:       {results['objective_score']:.4f}")
            print(f"  Overall R²:  {results['overall_r_squared']:.4f}")
            print(f"  Overall RMSD:{results['overall_rmsd_cm1']:8.1f} cm⁻¹")
            print(f"  Wall time:   {results.get('wall_time', 0):.1f}s")
        assert results["per_molecule"], "summary should contain at least one molecule"


# ---------------------------------------------------------------------------
# Regression: load_heck_relay() must preserve published OPT parameter values
# (ericchansen/q2mm#277) — system-specific, not part of the parametrized sweep.
# ---------------------------------------------------------------------------


@pytest.mark.validation
@pytest.mark.external_data
def test_load_heck_relay_preserves_published_opt_values() -> None:
    """Regression: loader must NOT overwrite published Rosales OPT params.

    Before #277, ``load_heck_relay()`` re-projected the OPT-substructure
    parameters via FUERZA after ``freeze_standard_params``, discarding Rosales'
    fitted values.  After the fix, the loader keeps them exactly.  This test
    compares the optimizable (non-frozen) parameter values from
    ``load_system("heck-relay", starting_point="published")`` against the same
    params loaded directly from the .fld file with no Seminario step.
    """
    from q2mm.diagnostics.systems import _heck_relay_ff_path, load_system
    from q2mm.models.forcefield import ForceField

    ff_path = _heck_relay_ff_path()
    if not ff_path.exists():
        pytest.skip(f"Heck relay FF not found: {ff_path}")

    # Loader output, pinned to the published starting point (the default
    # "qfuerza" start intentionally overwrites OPT bond/angle scalars).
    sys_data = load_system("heck-relay", starting_point="published")
    loader_active = sys_data.forcefield.get_active_param_vector()

    # Same .fld, same active-mask partition, but no Seminario re-projection.
    expected_ff = ForceField.from_mm3_fld(str(ff_path), include_standard=True)
    opt_ff = ForceField.from_mm3_fld(str(ff_path), include_standard=False)
    expected_ff.freeze_standard_params(opt_ff)
    expected_active = expected_ff.get_active_param_vector()

    assert loader_active.shape == expected_active.shape, (
        f"Active-mask shape mismatch: loader={loader_active.shape} vs expected={expected_active.shape}"
    )
    np.testing.assert_allclose(
        loader_active,
        expected_active,
        rtol=0.0,
        atol=1e-12,
        err_msg=(
            "load_heck_relay() OPT parameter values differ from the published "
            ".fld values.  This likely means a Seminario-style re-estimation "
            "crept back into the loader (the #277 bug)."
        ),
    )
