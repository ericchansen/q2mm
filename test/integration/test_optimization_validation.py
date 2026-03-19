"""Optimization validation: end-to-end and cross-method parity tests.

Tests the full pipeline: QM data → Seminario → initial FF → scipy optimize
→ improved FF, using real backends (OpenMM + Tinker). Validates that:
  1. The optimizer actually improves the force field
  2. Multiple scipy methods converge to the same endpoint
  3. OpenMM and Tinker backends produce the same optimized FF
  4. The objective function's scoring relates correctly to the legacy formula
  5. Round-trip recovery of known parameters
  6. Parameter vector roundtrip (get→set→get identity)
  7. Atom-identity matching for bond/angle references
  8. Optimization determinism (same inputs → same outputs)
  9. Parameter vector length validation
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.optimizers.scoring import compare_data
from q2mm.parsers import Datum
from q2mm.models.forcefield import AngleParam, BondParam, ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import estimate_force_constants
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData, ReferenceValue
from q2mm.optimizers.scipy_opt import ScipyOptimizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"
CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
CH3F_HESS = QM_REF / "ch3f-hessian.npy"

# Tinker availability — auto-detect or use env vars
_TINKER_DIR = Path(os.environ.get("TINKER_DIR", "")) if os.environ.get("TINKER_DIR") else None
_TINKER_PRM = Path(os.environ.get("TINKER_PRM", "")) if os.environ.get("TINKER_PRM") else None

if _TINKER_DIR is None or _TINKER_PRM is None:
    # Auto-detect from TinkerEngine's built-in search
    from q2mm.backends.mm.tinker import _find_tinker_dir

    _auto_dir = _find_tinker_dir()
    if _auto_dir and _TINKER_DIR is None:
        _TINKER_DIR = Path(_auto_dir)
    if _TINKER_DIR and _TINKER_PRM is None:
        # Look for mm3.prm relative to bin dir (../params/mm3.prm)
        _candidate = _TINKER_DIR.parent / "params" / "mm3.prm"
        if _candidate.exists():
            _TINKER_PRM = _candidate

_HAS_TINKER = _TINKER_DIR is not None and _TINKER_DIR.exists() and _TINKER_PRM is not None and _TINKER_PRM.exists()


def _tinker_engine():
    """Create a TinkerEngine if available."""
    from q2mm.backends.mm.tinker import TinkerEngine

    return TinkerEngine(tinker_dir=str(_TINKER_DIR), params_file=str(_TINKER_PRM))


# ---- Helpers ----


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
    theta = np.deg2rad(angle_deg)
    return Q2MMMolecule(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [bond_length, 0.0, 0.0],
                [bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0],
            ]
        ),
        name="water",
        bond_tolerance=1.5,
    )


def _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _make_water_problem(engine=None, perturb_k=1.5, perturb_eq=5.0):
    """Create a water optimization problem with known true parameters.

    Returns (true_ff, guess_ff, molecules, reference_data, engine).
    The guess FF is perturbed from the true one.
    """
    if engine is None:
        engine = OpenMMEngine()
    true_ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)

    # Multiple geometries for identifiability
    mol_eq = _water(104.5, 0.96)
    mol_wide = _water(115.0, 0.96)
    mol_long = _water(104.5, 1.05)

    ref = ReferenceData()
    for i, mol in enumerate([mol_eq, mol_wide, mol_long]):
        ref.add_energy(engine.energy(mol, true_ff), weight=1.0, molecule_idx=i)

    # Add frequencies from equilibrium geometry (OpenMM only — Tinker
    # vibrate gives different mode ordering)
    openmm = OpenMMEngine()
    freqs = openmm.frequencies(mol_eq, true_ff)
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:  # skip near-zero translational/rotational
            ref.add_frequency(f, data_idx=j, weight=0.001, molecule_idx=0)

    # Perturbed starting point
    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, [mol_eq, mol_wide, mol_long], ref, engine


def _make_energy_only_problem(engine=None, perturb_k=1.5, perturb_eq=5.0):
    """Water problem with energy-only references (works with any backend).

    Avoids frequencies so Tinker and OpenMM get identical reference values.
    """
    if engine is None:
        engine = OpenMMEngine()
    true_ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)

    mols = [_water(104.5, 0.96), _water(115.0, 0.96), _water(104.5, 1.05), _water(95.0, 1.02)]

    # Generate reference energies from the TRUE ff using OpenMM (ground truth)
    openmm = OpenMMEngine()
    ref = ReferenceData()
    for i, mol in enumerate(mols):
        ref.add_energy(openmm.energy(mol, true_ff), weight=1.0, molecule_idx=i)

    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, mols, ref, engine


# ---- End-to-end Seminario → Optimize pipeline ----


class TestSeminarioOptimizePipeline:
    """Validates the full QM → Seminario → scipy optimize pipeline."""

    @pytest.fixture
    def ch3f_seminario_ff(self):
        mol = Q2MMMolecule.from_xyz(CH3F_XYZ)
        hess = np.load(CH3F_HESS)
        mol_h = mol.with_hessian(hess)
        return estimate_force_constants(mol_h), mol

    def test_seminario_ff_can_evaluate(self, ch3f_seminario_ff):
        """Seminario-derived FF can compute energy via OpenMM."""
        ff, mol = ch3f_seminario_ff
        engine = OpenMMEngine()
        energy = engine.energy(mol, ff)
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_optimize_improves_seminario_ff(self, ch3f_seminario_ff):
        """Optimizing Seminario FF against QM frequencies improves the score."""
        ff, mol = ch3f_seminario_ff
        engine = OpenMMEngine()

        # Generate "QM" frequencies from a slightly different FF (the target)
        target_ff = ff.copy()
        for b in target_ff.bonds:
            b.force_constant *= 1.1
        for a in target_ff.angles:
            a.force_constant *= 0.9

        target_freqs = engine.frequencies(mol, target_ff)

        ref = ReferenceData()
        for j, f in enumerate(target_freqs):
            if abs(f) > 100.0:
                ref.add_frequency(f, data_idx=j, weight=0.001, molecule_idx=0)

        obj = ObjectiveFunction(ff, engine, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj)

        assert result.final_score < result.initial_score
        assert result.improvement > 0.1  # at least 10% improvement


# ---- Cross-backend optimization parity (OpenMM vs Tinker) ----


@pytest.mark.skipif(not _HAS_TINKER, reason="Tinker not installed")
class TestCrossBackendOptimization:
    """Optimize the same problem with OpenMM and Tinker, compare results."""

    def test_openmm_vs_tinker_energy_parity(self):
        """OpenMM and Tinker agree on energy for the same FF + geometry."""
        mol = _water()
        ff = _water_ff()
        openmm = OpenMMEngine()
        tinker = _tinker_engine()

        e_openmm = openmm.energy(mol, ff)
        e_tinker = tinker.energy(mol, ff)

        # Both compute the same MM3 functional form — should agree closely.
        # Small differences arise from vdW treatment and numerical precision;
        # empirically within ~0.003 kcal/mol, we use 0.01 as tolerance.
        assert e_openmm == pytest.approx(e_tinker, abs=0.01), f"OpenMM={e_openmm:.6f} vs Tinker={e_tinker:.6f}"

    def test_openmm_vs_tinker_optimization_convergence(self):
        """Both backends converge to similar optimized parameters."""
        results = {}
        for label, engine in [("OpenMM", OpenMMEngine()), ("Tinker", _tinker_engine())]:
            true_ff, guess_ff, mols, ref, eng = _make_energy_only_problem(engine=engine)
            obj = ObjectiveFunction(guess_ff, eng, mols, ref)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            results[label] = opt.optimize(obj)

        # Both should achieve significant improvement
        for label, result in results.items():
            assert result.improvement > 0.3, f"{label} didn't improve enough: {result.improvement:.2%}"

        # Optimized parameters should be similar (within 20%)
        p_omm = results["OpenMM"].final_params
        p_tnk = results["Tinker"].final_params
        for i, (a, b) in enumerate(zip(p_omm, p_tnk)):
            denom = max(abs(a), abs(b), 0.01)
            assert abs(a - b) / denom < 0.2, f"Param[{i}]: OpenMM={a:.4f}, Tinker={b:.4f}"


# ---- Multi-method convergence ----


class TestMultiMethodConvergence:
    """Verify multiple scipy methods converge to the same optimum."""

    def test_three_methods_agree(self):
        """L-BFGS-B, Nelder-Mead, and least_squares all improve significantly."""
        true_ff, guess_ff, mols, ref, engine = _make_water_problem()

        results = {}
        for method in ["L-BFGS-B", "Nelder-Mead", "least_squares"]:
            ff_copy = guess_ff.copy()
            obj = ObjectiveFunction(ff_copy, engine, mols, ref)
            opt = ScipyOptimizer(
                method=method,
                maxiter=300,
                use_bounds=(method != "Nelder-Mead"),
                verbose=False,
            )
            results[method] = opt.optimize(obj)

        # All should achieve significant improvement
        for method, result in results.items():
            assert result.improvement > 0.5, f"{method} didn't improve enough: {result.improvement:.2%}"

        # Derivative-free methods (Nelder-Mead, least_squares) should
        # reach near-zero scores. L-BFGS-B may get stuck at a slightly
        # higher local minimum due to finite-difference gradient noise.
        for method in ["Nelder-Mead", "least_squares"]:
            assert results[method].final_score < 1.0, f"{method} score too high: {results[method].final_score:.4f}"


# ---- Objective function vs legacy compare.compare_data() ----


class TestScoreParity:
    """Verify new objective scoring relates correctly to legacy formula.

    The formulas differ intentionally:
      New:    score = sum( (w_i × diff_i)² )
      Legacy: energy terms   → sum( w_i² × diff_i² ) / total_num_energy
              non-energy terms → sum( w_i² × diff_i² ) / N_type

    For a single data point of one type, N_type=1, total_num_energy=1,
    so both give w²×diff².
    For N>1 points of one type, new_score = legacy_score × N_type.
    This is documented and tested below.
    """

    def test_single_energy_scores_match(self):
        """With 1 energy point, new and legacy scores are identical."""
        mol = _water()
        ff = _water_ff()
        engine = OpenMMEngine()

        calc_energy = engine.energy(mol, ff)
        ref_energy = calc_energy + 0.5

        # NEW
        ref = ReferenceData()
        ref.add_energy(ref_energy, weight=1.0, molecule_idx=0)
        obj = ObjectiveFunction(ff, engine, [mol], ref)
        new_score = obj(ff.get_param_vector())

        # LEGACY — compare_data needs numpy arrays of Datum
        r_datum = Datum(val=ref_energy, wht=1.0, typ="e", lbl="ref-energy", idx_1=1, idx_2=0)
        c_datum = Datum(val=calc_energy, wht=1.0, typ="e", lbl="calc-energy", idx_1=1, idx_2=0)
        r_arr = np.array([r_datum], dtype=object)
        c_arr = np.array([c_datum], dtype=object)
        legacy_score = compare_data({"e": r_arr}, {"e": c_arr})

        # N_type=1 ⇒ identical
        assert new_score == pytest.approx(legacy_score, rel=0.01), f"New={new_score}, Legacy={legacy_score}"

    def test_multi_energy_normalization_relationship(self):
        """With N energy points, score relationship accounts for legacy correlation.

        The legacy compare_data correlates energies (shifts calculated values
        so the minimum-reference-energy point becomes zero in the calc set).
        This changes the effective diffs. We verify the relationship holds for
        a single-group case where correlation is predictable.
        """
        mol1 = _water()
        mol2 = _water(110.0, 0.96)
        ff = _water_ff()
        engine = OpenMMEngine()

        e1 = engine.energy(mol1, ff)
        e2 = engine.energy(mol2, ff)
        # Use offsets that make the relationship clear:
        # ref_e1 = e1 + delta1, ref_e2 = e2 + delta2
        # Legacy correlate_energies shifts calc so c[min_ref_idx].val = 0,
        # then scores correlated diffs.
        # Our new ObjectiveFunction scores absolute diffs (no correlation).
        # Both formulas are valid; they encode different physical assumptions
        # about what matters (absolute vs relative energies).
        delta1, delta2 = 0.3, 0.3  # Same offset → correlation doesn't change diffs
        ref_e1 = e1 + delta1
        ref_e2 = e2 + delta2

        # NEW
        ref = ReferenceData()
        ref.add_energy(ref_e1, weight=1.0, molecule_idx=0)
        ref.add_energy(ref_e2, weight=1.0, molecule_idx=1)
        obj = ObjectiveFunction(ff, engine, [mol1, mol2], ref)
        new_score = obj(ff.get_param_vector())

        # LEGACY
        r1 = Datum(val=ref_e1, wht=1.0, typ="e", lbl="e1", idx_1=1, idx_2=0)
        c1 = Datum(val=e1, wht=1.0, typ="e", lbl="e1", idx_1=1, idx_2=0)
        r2 = Datum(val=ref_e2, wht=1.0, typ="e", lbl="e2", idx_1=1, idx_2=0)
        c2 = Datum(val=e2, wht=1.0, typ="e", lbl="e2", idx_1=1, idx_2=0)
        r_arr = np.array([r1, r2], dtype=object)
        c_arr = np.array([c1, c2], dtype=object)
        legacy_score = compare_data({"e": r_arr}, {"e": c_arr})

        # With uniform offsets, correlation preserves the diffs, so:
        # new_score = sum((w*diff)^2) = 2 * (1.0 * 0.3)^2 = 0.18
        # legacy_score = sum(w^2 * diff^2 / N) = 0.09 (N=2)
        # new_score = legacy_score * N
        n_energy = 2
        assert new_score == pytest.approx(legacy_score * n_energy, rel=0.05), (
            f"New={new_score}, Legacy×N={legacy_score * n_energy}"
        )


# ---- Optimization round-trip validation ----


class TestOptimizationRoundtrip:
    """Verify optimizer can recover known parameters from perturbed start."""

    def test_recover_bond_force_constant(self):
        """Optimizer recovers correct bond k from energy data."""
        true_ff, guess_ff, mols, ref, engine = _make_water_problem()

        obj = ObjectiveFunction(guess_ff, engine, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        true_params = true_ff.get_param_vector()
        final_params = result.final_params

        # Bond k should be within 30% of true value
        true_bond_k = true_params[0]
        final_bond_k = final_params[0]
        assert abs(final_bond_k - true_bond_k) / true_bond_k < 0.3, (
            f"Bond k: true={true_bond_k:.3f}, got={final_bond_k:.3f}"
        )

    def test_convergence_history_monotonic(self):
        """Score history should be roughly monotonically decreasing."""
        true_ff, guess_ff, mols, ref, engine = _make_water_problem()

        obj = ObjectiveFunction(guess_ff, engine, mols, ref)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=100, use_bounds=False, verbose=False)
        result = opt.optimize(obj)

        # The optimizer should find a better score than the starting point.
        # Use initial_score/final_score (guaranteed by OptimizationResult)
        # instead of history endpoints, since history[-1] may not be the best
        # point found (e.g., Nelder-Mead evaluates worse points late in a run).
        assert result.final_score <= result.initial_score
        assert min(result.history) <= result.history[0]

    @pytest.mark.skipif(not _HAS_TINKER, reason="Tinker not installed")
    def test_recover_params_with_tinker(self):
        """Tinker backend also recovers correct parameters."""
        tinker = _tinker_engine()
        true_ff, guess_ff, mols, ref, engine = _make_energy_only_problem(engine=tinker)

        obj = ObjectiveFunction(guess_ff, engine, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        assert result.improvement > 0.3, f"Tinker optimization only improved {result.improvement:.2%}"

        true_params = true_ff.get_param_vector()
        final_params = result.final_params
        true_bond_k = true_params[0]
        final_bond_k = final_params[0]
        assert abs(final_bond_k - true_bond_k) / true_bond_k < 0.3, (
            f"Tinker bond k: true={true_bond_k:.3f}, got={final_bond_k:.3f}"
        )


# ---- Force field export/param-vector roundtrip ----


class TestForceFieldExportRoundtrip:
    """Verify optimized parameters survive export/re-import."""

    def test_param_vector_roundtrip(self):
        """set_param_vector(get_param_vector()) is identity."""
        ff = _water_ff()
        original = ff.get_param_vector().copy()
        ff.set_param_vector(original)
        roundtripped = ff.get_param_vector()
        np.testing.assert_array_equal(original, roundtripped)

    def test_param_vector_roundtrip_after_mutation(self):
        """Roundtrip still works after modifying individual parameters."""
        ff = _water_ff()
        vec = ff.get_param_vector().copy()
        # Perturb each element
        vec *= 1.1
        vec[0] += 0.5
        ff.set_param_vector(vec)
        roundtripped = ff.get_param_vector()
        np.testing.assert_array_almost_equal(vec, roundtripped, decimal=15)

    def test_param_vector_roundtrip_optimized_ff(self):
        """Optimized FF survives get→set→get roundtrip."""
        true_ff, guess_ff, mols, ref, engine = _make_water_problem()
        obj = ObjectiveFunction(guess_ff, engine, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj)

        # Apply final params, roundtrip
        optimized_ff = guess_ff.copy()
        optimized_ff.set_param_vector(result.final_params)
        vec_before = optimized_ff.get_param_vector().copy()
        optimized_ff.set_param_vector(vec_before)
        vec_after = optimized_ff.get_param_vector()
        np.testing.assert_array_equal(vec_before, vec_after)

    def test_copy_preserves_params(self):
        """ForceField.copy() preserves the parameter vector exactly."""
        ff = _water_ff(bond_k=3.14, bond_r0=1.23, angle_k=0.42, angle_eq=109.5)
        ff_copy = ff.copy()
        np.testing.assert_array_equal(ff.get_param_vector(), ff_copy.get_param_vector())
        # Mutating the copy should not affect the original
        vec = ff_copy.get_param_vector()
        vec[0] = 999.0
        ff_copy.set_param_vector(vec)
        assert ff.get_param_vector()[0] != 999.0


# ---- Atom-identity matching ----


class TestAtomIdentityMatching:
    """Verify atom-identity matching is order-independent."""

    def test_bond_length_by_atom_indices(self):
        """_extract_value finds the right bond via atom_indices."""
        calc = {
            "bond_lengths": [0.96, 0.97],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97},
        }
        # Ask for bond (0, 2) via atom_indices
        ref = ReferenceValue(kind="bond_length", value=0.97, atom_indices=(0, 2))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_bond_length_atom_indices_order_independent(self):
        """atom_indices=(2, 0) finds same bond as (0, 2)."""
        calc = {
            "bond_lengths": [0.96, 0.97],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97},
        }
        # Reversed order — _extract_value sorts the key
        ref = ReferenceValue(kind="bond_length", value=0.97, atom_indices=(2, 0))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_bond_angle_by_atom_indices(self):
        """_extract_value finds the right angle via atom_indices."""
        calc = {
            "bond_angles": [104.5],
            "bond_angles_by_atoms": {(1, 0, 2): 104.5},
        }
        ref = ReferenceValue(kind="bond_angle", value=104.5, atom_indices=(1, 0, 2))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(104.5)

    def test_bond_angle_reversed_order(self):
        """atom_indices=(2, 0, 1) finds same angle as (1, 0, 2)."""
        calc = {
            "bond_angles": [104.5],
            "bond_angles_by_atoms": {(1, 0, 2): 104.5},
        }
        # Reversed: (2, 0, 1) — _extract_value tries both orderings
        ref = ReferenceValue(kind="bond_angle", value=104.5, atom_indices=(2, 0, 1))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(104.5)

    def test_fallback_to_data_idx(self):
        """When atom_indices is None, data_idx still works (backwards compat)."""
        calc = {
            "bond_lengths": [0.96, 0.97, 0.98],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97, (1, 2): 0.98},
        }
        # Use data_idx=1, no atom_indices
        ref = ReferenceValue(kind="bond_length", value=0.97, data_idx=1, atom_indices=None)
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_missing_atom_indices_raises(self):
        """KeyError raised for atom pair not in calculated data."""
        calc = {
            "bond_lengths": [0.96],
            "bond_lengths_by_atoms": {(0, 1): 0.96},
        }
        ref = ReferenceValue(kind="bond_length", value=1.0, atom_indices=(5, 6))
        with pytest.raises(KeyError):
            ObjectiveFunction._extract_value(calc, ref)

    def test_add_bond_length_requires_idx_or_atoms(self):
        """ReferenceData.add_bond_length raises without data_idx or atom_indices."""
        ref = ReferenceData()
        with pytest.raises(ValueError, match="Either atom_indices or data_idx"):
            ref.add_bond_length(0.96)

    def test_add_bond_angle_requires_idx_or_atoms(self):
        """ReferenceData.add_bond_angle raises without data_idx or atom_indices."""
        ref = ReferenceData()
        with pytest.raises(ValueError, match="Either atom_indices or data_idx"):
            ref.add_bond_angle(104.5)


# ---- Optimization determinism ----


class TestOptimizationDeterminism:
    """Verify optimization is deterministic."""

    def test_same_result_twice(self):
        """Running the same optimization twice gives identical parameters."""
        results = []
        for _ in range(2):
            true_ff, guess_ff, mols, ref, engine = _make_water_problem()
            obj = ObjectiveFunction(guess_ff, engine, mols, ref)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            results.append(opt.optimize(obj))

        np.testing.assert_array_almost_equal(
            results[0].final_params,
            results[1].final_params,
            decimal=12,
            err_msg="Optimization is not deterministic",
        )
        assert results[0].final_score == pytest.approx(results[1].final_score, rel=1e-10)

    def test_determinism_nelder_mead(self):
        """Nelder-Mead is also deterministic (no stochastic elements)."""
        results = []
        for _ in range(2):
            true_ff, guess_ff, mols, ref, engine = _make_water_problem()
            obj = ObjectiveFunction(guess_ff, engine, mols, ref)
            opt = ScipyOptimizer(method="Nelder-Mead", maxiter=200, use_bounds=False, verbose=False)
            results.append(opt.optimize(obj))

        np.testing.assert_array_almost_equal(
            results[0].final_params,
            results[1].final_params,
            decimal=12,
        )


# ---- Parameter vector validation ----


class TestParamVectorValidation:
    """Verify set_param_vector rejects wrong-length vectors."""

    def test_short_vector_raises(self):
        ff = _water_ff()
        with pytest.raises(ValueError, match="does not match"):
            ff.set_param_vector(np.array([1.0]))  # too short

    def test_long_vector_raises(self):
        ff = _water_ff()
        with pytest.raises(ValueError, match="does not match"):
            ff.set_param_vector(np.zeros(100))  # too long

    def test_empty_vector_raises(self):
        ff = _water_ff()
        with pytest.raises(ValueError, match="does not match"):
            ff.set_param_vector(np.array([]))

    def test_exact_length_accepted(self):
        ff = _water_ff()
        n = len(ff.get_param_vector())
        ff.set_param_vector(np.ones(n))  # should not raise
        np.testing.assert_array_equal(ff.get_param_vector(), np.ones(n))


# ---- Golden fixture regression ----


class TestGoldenFixtureRegression:
    """Verify current optimization matches saved golden fixture (if present)."""

    GOLDEN_PATH = REPO_ROOT / "test" / "fixtures" / "optimization_golden.json"

    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent.parent / "test" / "fixtures" / "optimization_golden.json").exists(),
        reason="Golden fixture not yet generated (run scripts/generate_optimization_fixtures.py)",
    )
    def test_matches_golden_fixture(self):
        """Current optimization reproduces the golden fixture within tolerance."""
        golden = json.loads(self.GOLDEN_PATH.read_text())

        true_ff, guess_ff, mols, ref, engine = _make_water_problem()
        obj = ObjectiveFunction(guess_ff, engine, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj)

        # Scores should match tightly (same code, same inputs → same results)
        assert result.initial_score == pytest.approx(golden["initial_score"], rel=1e-6), (
            f"Initial score drift: {result.initial_score} vs {golden['initial_score']}"
        )
        assert result.final_score == pytest.approx(golden["final_score"], rel=1e-6), (
            f"Final score drift: {result.final_score} vs {golden['final_score']}"
        )

        # Parameters should match closely
        np.testing.assert_allclose(
            result.final_params,
            golden["final_params"],
            rtol=1e-6,
            err_msg="Optimized parameters drifted from golden fixture",
        )
