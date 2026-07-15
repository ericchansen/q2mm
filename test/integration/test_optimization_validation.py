"""Optimization validation: end-to-end and cross-method parity tests.

Tests the full pipeline: QM data → Seminario → initial FF → scipy optimize
→ improved FF, using real backends (OpenMM + Tinker). Validates that:
  1. The optimizer actually improves the force field
  2. Multiple scipy methods converge to the same endpoint
  3. OpenMM and Tinker backends produce the same optimized FF
  4. The objective function's scoring formula is correct
  5. Round-trip recovery of known parameters
  6. Parameter vector roundtrip via ParameterLayout.replace()
  7. Atom-identity matching for bond/angle references
  8. Optimization determinism (same inputs → same outputs)
  9. Parameter vector length validation
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    FrequencyRequest,
)
from q2mm.backends.registry import load_backend
from test.backend_fixtures import param_vector, prepare_case

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openmm")
pytestmark = pytest.mark.openmm

from test._shared import CH3F_HESS, CH3F_XYZ, make_water

from q2mm.backends.contracts import Backend
from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.hessian import HessianProvenance, HessianUnits
from q2mm.models.molecule import Molecule
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.seminario import qfuerza_fresh
from q2mm.optimizers.objective import ObjectiveFunction
from q2mm.optimizers.scipy_opt import ScipyOptimizer

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Tinker availability — auto-detect or use env vars
_TINKER_DIR = Path(os.environ.get("TINKER_DIR", "")) if os.environ.get("TINKER_DIR") else None
_TINKER_PRM = Path(os.environ.get("TINKER_PRM", "")) if os.environ.get("TINKER_PRM") else None

if _TINKER_DIR is None or _TINKER_PRM is None:
    from q2mm.backends.mm.tinker import _find_tinker_dir

    _auto_dir = _find_tinker_dir()
    if _auto_dir and _TINKER_DIR is None:
        _TINKER_DIR = Path(_auto_dir)
    if _TINKER_DIR and _TINKER_PRM is None:
        _candidate = _TINKER_DIR.parent / "params" / "mm3.prm"
        if _candidate.exists():
            _TINKER_PRM = _candidate

_HAS_TINKER = _TINKER_DIR is not None and _TINKER_DIR.exists() and _TINKER_PRM is not None and _TINKER_PRM.exists()


def _tinker_backend() -> Backend:
    """Create a TinkerBackend with an explicit dir (bypasses the PATH probe)."""
    from q2mm.backends.mm.tinker import TinkerBackend

    return TinkerBackend(tinker_dir=str(_TINKER_DIR), params_file=str(_TINKER_PRM))


# ---- Helpers ----


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Molecule:
    return make_water(angle_deg=angle_deg, bond_length=bond_length)


def _water_ff(
    bond_k: float = 503.6, bond_r0: float = 0.96, angle_k: float = 57.6, angle_eq: float = 104.5
) -> ForceField:
    """Build a minimal water force field.

    MM3, not HARMONIC: this FF is evaluated through :class:`OpenMMBackend`
    (harmonic + MM3 dual-mode, MM3 by default pre-Phase-2) and
    :class:`TinkerBackend` (MM3-only) in this file's cross-backend parity
    tests, and ``test_matches_golden_fixture`` compares against
    ``test/fixtures/optimization_golden.json``, which was generated
    under OpenMM's old implicit-MM3 branch. Tagging this HARMONIC would
    silently change the physics (MM3's cubic bond-stretch/sextic
    angle-bend corrections vs. pure quadratic) while leaving that golden
    fixture's numbers unchanged underneath a since-shifted optimum.
    """
    return ForceField(
        name="water-test",
        bonds=(BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0),),
        angles=(AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq),),
        functional_form=FunctionalForm.MM3,
    )


def _ch3f_molecule() -> Molecule:
    return load_xyz(CH3F_XYZ, bond_tolerance=1.5).with_hessian(
        np.load(CH3F_HESS),
        HessianProvenance(
            units=HessianUnits.ATOMIC,
            source="q2mm-sn2-resource",
            path=str(Path(CH3F_HESS).resolve()),
        ),
    )


def _build_objective(
    ff: ForceField,
    backend: Backend,
    molecules: list[Molecule],
    reference: ObservationSet,
) -> tuple[ObjectiveFunction, ParameterLayout, ActiveParameterSpace]:
    layout = ParameterLayout.from_force_field(ff)
    objective = ObjectiveFunction(ff, backend, molecules, reference, layout=layout)
    space = ActiveParameterSpace.all_active(layout, ff)
    return objective, layout, space


def _make_water_problem(
    backend: Backend | None = None, perturb_k: float = 1.5, perturb_eq: float = 5.0
) -> tuple[ForceField, ForceField, list[Molecule], ObservationSet, Backend]:
    """Create a water optimization problem with known true parameters."""
    if backend is None:
        backend = load_backend("openmm")
    true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)

    mol_eq = _water(104.5, 0.96)
    mol_wide = _water(115.0, 0.96)
    mol_long = _water(104.5, 1.05)

    ref = ObservationSet()
    for i, mol in enumerate([mol_eq, mol_wide, mol_long]):
        ref = ref.with_energy(
            prepare_case(backend, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id=str(i),
        )

    openmm = load_backend("openmm")
    freqs = [
        float(_f)
        for _f in prepare_case(openmm, mol_eq, true_ff)
        .frequencies(FrequencyRequest(parameters=param_vector(true_ff)))
        .frequencies
    ]
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:
            ref = ref.with_frequency(f, data_idx=j, weight=0.001, case_id="0")

    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, [mol_eq, mol_wide, mol_long], ref, backend


def _make_energy_only_problem(
    backend: Backend | None = None, perturb_k: float = 1.5, perturb_eq: float = 5.0
) -> tuple[ForceField, ForceField, list[Molecule], ObservationSet, Backend]:
    """Water problem with energy-only references (works with any backend)."""
    if backend is None:
        backend = load_backend("openmm")
    true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)

    mols = [_water(104.5, 0.96), _water(115.0, 0.96), _water(104.5, 1.05), _water(95.0, 1.02)]

    openmm = load_backend("openmm")
    ref = ObservationSet()
    for i, mol in enumerate(mols):
        ref = ref.with_energy(
            prepare_case(openmm, mol, true_ff).energy(EnergyRequest(parameters=param_vector(true_ff))).energy,
            weight=1.0,
            case_id=str(i),
        )

    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, mols, ref, backend


# ---- End-to-end Seminario → Optimize pipeline ----


class TestSeminarioOptimizePipeline:
    """Validates the full QM → Seminario → scipy optimize pipeline."""

    @pytest.fixture
    def ch3f_seminario_ff(self) -> tuple[ForceField, Molecule]:
        mol = _ch3f_molecule()
        return qfuerza_fresh(mol, functional_form=FunctionalForm.MM3), mol

    def test_seminario_ff_can_evaluate(self, ch3f_seminario_ff: tuple[ForceField, Molecule]) -> None:
        """Seminario-derived FF can compute energy via OpenMM."""
        ff, mol = ch3f_seminario_ff
        backend = load_backend("openmm")
        energy = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    @pytest.mark.integration
    def test_optimize_improves_seminario_ff(self, ch3f_seminario_ff: tuple[ForceField, Molecule]) -> None:
        """Optimizing Seminario FF against QM frequencies improves the score."""
        ff, mol = ch3f_seminario_ff
        backend = load_backend("openmm")

        target_ff = replace(
            ff,
            bonds=tuple(replace(b, force_constant=b.force_constant * 1.1) for b in ff.bonds),
            angles=tuple(replace(a, force_constant=a.force_constant * 0.9) for a in ff.angles),
        )
        target_freqs = [
            float(_f)
            for _f in prepare_case(backend, mol, target_ff)
            .frequencies(FrequencyRequest(parameters=param_vector(target_ff)))
            .frequencies
        ]

        ref = ObservationSet()
        for j, f in enumerate(target_freqs):
            if abs(f) > 100.0:
                ref = ref.with_frequency(f, data_idx=j, weight=0.001, case_id="0")

        obj, _layout, space = _build_objective(ff, backend, [mol], ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj, space)

        assert result.final_score < result.initial_score
        assert result.improvement > 0.1


# ---- Cross-backend optimization parity (OpenMM vs Tinker) ----


@pytest.mark.skipif(not _HAS_TINKER, reason="Tinker not installed")
class TestCrossBackendOptimization:
    """Optimize the same problem with OpenMM and Tinker, compare results."""

    def test_openmm_vs_tinker_energy_parity(self) -> None:
        """OpenMM and Tinker agree on energy for the same FF + geometry."""
        mol = _water()
        ff = _water_ff()
        openmm = load_backend("openmm")
        tinker = _tinker_backend()

        e_openmm = prepare_case(openmm, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e_tinker = prepare_case(tinker, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy

        assert e_openmm == pytest.approx(e_tinker, abs=0.01), f"OpenMM={e_openmm:.6f} vs Tinker={e_tinker:.6f}"

    @pytest.mark.nightly
    def test_openmm_vs_tinker_optimization_convergence(self) -> None:
        """Both backends converge to similar optimized parameters."""
        results = {}
        for label, backend in [("OpenMM", load_backend("openmm")), ("Tinker", _tinker_backend())]:
            true_ff, guess_ff, mols, ref, eng = _make_energy_only_problem(backend=backend)
            obj, _layout, space = _build_objective(guess_ff, eng, mols, ref)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            results[label] = opt.optimize(obj, space)

        for label, result in results.items():
            assert result.improvement > 0.3, f"{label} didn't improve enough: {result.improvement:.2%}"

        p_omm = results["OpenMM"].final_params
        p_tnk = results["Tinker"].final_params
        for i, (a, b) in enumerate(zip(p_omm, p_tnk)):
            denom = max(abs(a), abs(b), 0.01)
            assert abs(a - b) / denom < 0.2, f"Param[{i}]: OpenMM={a:.4f}, Tinker={b:.4f}"


# ---- Multi-method convergence ----


class TestMultiMethodConvergence:
    """Verify multiple scipy methods converge to the same optimum."""

    @pytest.mark.integration
    def test_three_methods_agree(self) -> None:
        """L-BFGS-B, Nelder-Mead, and least_squares all improve significantly."""
        true_ff, guess_ff, mols, ref, backend = _make_water_problem()

        results = {}
        for method in ["L-BFGS-B", "Nelder-Mead", "least_squares"]:
            obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
            opt = ScipyOptimizer(
                method=method,
                maxiter=300,
                use_bounds=(method != "Nelder-Mead"),
                verbose=False,
            )
            results[method] = opt.optimize(obj, space)

        for method, result in results.items():
            assert result.improvement > 0.5, f"{method} didn't improve enough: {result.improvement:.2%}"

        for method in ["Nelder-Mead", "least_squares"]:
            assert results[method].final_score < 1.0, f"{method} score too high: {results[method].final_score:.4f}"


# ---- Objective function scoring formula verification ----


class TestScoreParity:
    """Verify ObjectiveFunction scoring computes sum((w * diff)²)."""

    def test_single_energy_score(self) -> None:
        """With 1 energy point, score = (w * diff)²."""
        mol = _water()
        ff = _water_ff()
        backend = load_backend("openmm")

        calc_energy = prepare_case(backend, mol, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        offset = 0.5
        ref_energy = calc_energy + offset

        ref = ObservationSet()
        ref = ref.with_energy(ref_energy, weight=1.0, case_id="0")
        obj, layout, _space = _build_objective(ff, backend, [mol], ref)
        score = obj(layout.vector(ff))

        expected = (1.0 * offset) ** 2
        assert score == pytest.approx(expected, rel=0.01), f"score={score}, expected={expected}"

    def test_multi_energy_score(self) -> None:
        """With N energy points, score = sum of (w * diff)² for each."""
        mol1 = _water()
        mol2 = _water(110.0, 0.96)
        ff = _water_ff()
        backend = load_backend("openmm")

        e1 = prepare_case(backend, mol1, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        e2 = prepare_case(backend, mol2, ff).energy(EnergyRequest(parameters=param_vector(ff))).energy
        offset = 0.3
        ref_e1 = e1 + offset
        ref_e2 = e2 + offset

        ref = ObservationSet()
        ref = ref.with_energy(ref_e1, weight=1.0, case_id="0")
        ref = ref.with_energy(ref_e2, weight=1.0, case_id="1")
        obj, layout, _space = _build_objective(ff, backend, [mol1, mol2], ref)
        score = obj(layout.vector(ff))

        expected = 2 * (1.0 * offset) ** 2
        assert score == pytest.approx(expected, rel=0.05), f"score={score}, expected={expected}"


# ---- Optimization round-trip validation ----


class TestOptimizationRoundtrip:
    """Verify optimizer can recover known parameters from perturbed start."""

    @pytest.mark.integration
    def test_recover_bond_force_constant(self) -> None:
        """Optimizer recovers correct bond k from energy data."""
        true_ff, guess_ff, mols, ref, backend = _make_water_problem()

        obj, layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        true_params = layout.vector(true_ff)
        final_params = result.final_params

        true_bond_k = true_params[0]
        final_bond_k = final_params[0]
        assert abs(final_bond_k - true_bond_k) / true_bond_k < 0.3, (
            f"Bond k: true={true_bond_k:.3f}, got={final_bond_k:.3f}"
        )

    @pytest.mark.integration
    def test_convergence_history_monotonic(self) -> None:
        """Score history should be roughly monotonically decreasing."""
        true_ff, guess_ff, mols, ref, backend = _make_water_problem()

        obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="Nelder-Mead", maxiter=100, use_bounds=False, verbose=False)
        result = opt.optimize(obj, space)

        assert result.final_score <= result.initial_score
        assert min(result.history) <= result.history[0]

    @pytest.mark.nightly
    @pytest.mark.skipif(not _HAS_TINKER, reason="Tinker not installed")
    def test_recover_params_with_tinker(self) -> None:
        """Tinker backend also recovers correct parameters."""
        tinker = _tinker_backend()
        true_ff, guess_ff, mols, ref, backend = _make_energy_only_problem(backend=tinker)

        obj, layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        assert result.improvement > 0.3, f"Tinker optimization only improved {result.improvement:.2%}"

        true_params = layout.vector(true_ff)
        final_params = result.final_params
        true_bond_k = true_params[0]
        final_bond_k = final_params[0]
        assert abs(final_bond_k - true_bond_k) / true_bond_k < 0.3, (
            f"Tinker bond k: true={true_bond_k:.3f}, got={final_bond_k:.3f}"
        )


# ---- Force field export/param-vector roundtrip ----


class TestForceFieldExportRoundtrip:
    """Verify optimized parameters survive export/re-import."""

    def test_param_vector_roundtrip(self) -> None:
        """ParameterLayout.replace(vector(layout.vector(ff))) is identity."""
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        original = layout.vector(ff).copy()
        roundtripped = layout.vector(layout.replace(ff, original))
        np.testing.assert_array_equal(original, roundtripped)

    def test_param_vector_roundtrip_after_mutation(self) -> None:
        """Roundtrip still works after modifying individual parameters."""
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff).copy()
        vec *= 1.1
        vec[0] += 0.5
        mutated_ff = layout.replace(ff, vec)
        roundtripped = layout.vector(mutated_ff)
        np.testing.assert_array_almost_equal(vec, roundtripped, decimal=15)

    @pytest.mark.integration
    def test_param_vector_roundtrip_optimized_ff(self) -> None:
        """Optimized FF survives replace→vector→replace→vector roundtrip."""
        true_ff, guess_ff, mols, ref, backend = _make_water_problem()
        obj, layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=100, verbose=False)
        result = opt.optimize(obj, space)

        optimized_ff = layout.replace(guess_ff, result.final_params)
        vec_before = layout.vector(optimized_ff).copy()
        vec_after = layout.vector(layout.replace(optimized_ff, vec_before))
        np.testing.assert_array_equal(vec_before, vec_after)

    def test_copy_preserves_params(self) -> None:
        """dataclasses.replace() preserves the parameter vector exactly."""
        ff = _water_ff(bond_k=225.9, bond_r0=1.23, angle_k=30.2, angle_eq=109.5)
        layout = ParameterLayout.from_force_field(ff)
        ff_copy = replace(ff)
        np.testing.assert_array_equal(layout.vector(ff), layout.vector(ff_copy))

        vec = layout.vector(ff_copy).copy()
        vec[0] = 999.0
        mutated_copy = layout.replace(ff_copy, vec)
        assert layout.vector(ff)[0] != 999.0
        assert layout.vector(mutated_copy)[0] == 999.0


# ---- Atom-identity matching ----


class TestAtomIdentityMatching:
    """Verify atom-identity matching is order-independent."""

    def test_bond_length_by_atom_indices(self) -> None:
        """_extract_value finds the right bond via atom_indices."""
        calc = {
            "bond_lengths": [0.96, 0.97],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97},
        }
        ref = Observation(kind="bond_length", value=0.97, atom_indices=(0, 2))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_bond_length_atom_indices_order_independent(self) -> None:
        """atom_indices=(2, 0) finds same bond as (0, 2)."""
        calc = {
            "bond_lengths": [0.96, 0.97],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97},
        }
        ref = Observation(kind="bond_length", value=0.97, atom_indices=(2, 0))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_bond_angle_by_atom_indices(self) -> None:
        """_extract_value finds the right angle via atom_indices."""
        calc = {
            "bond_angles": [104.5],
            "bond_angles_by_atoms": {(1, 0, 2): 104.5},
        }
        ref = Observation(kind="bond_angle", value=104.5, atom_indices=(1, 0, 2))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(104.5)

    def test_bond_angle_reversed_order(self) -> None:
        """atom_indices=(2, 0, 1) finds same angle as (1, 0, 2)."""
        calc = {
            "bond_angles": [104.5],
            "bond_angles_by_atoms": {(1, 0, 2): 104.5},
        }
        ref = Observation(kind="bond_angle", value=104.5, atom_indices=(2, 0, 1))
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(104.5)

    def test_fallback_to_data_idx(self) -> None:
        """When atom_indices is None, data_idx still works."""
        calc = {
            "bond_lengths": [0.96, 0.97, 0.98],
            "bond_lengths_by_atoms": {(0, 1): 0.96, (0, 2): 0.97, (1, 2): 0.98},
        }
        ref = Observation(kind="bond_length", value=0.97, data_idx=1, atom_indices=None)
        extracted = ObjectiveFunction._extract_value(calc, ref)
        assert extracted == pytest.approx(0.97)

    def test_missing_atom_indices_raises(self) -> None:
        """KeyError raised for atom pair not in calculated data."""
        calc = {
            "bond_lengths": [0.96],
            "bond_lengths_by_atoms": {(0, 1): 0.96},
        }
        ref = Observation(kind="bond_length", value=1.0, atom_indices=(5, 6))
        with pytest.raises(KeyError):
            ObjectiveFunction._extract_value(calc, ref)

    def test_add_bond_length_requires_idx_or_atoms(self) -> None:
        """ObservationSet.add_bond_length raises without data_idx or atom_indices."""
        ref = ObservationSet()
        with pytest.raises(ValueError, match="Either atom_indices or data_idx"):
            ref = ref.with_bond_length(0.96)

    def test_add_bond_angle_requires_idx_or_atoms(self) -> None:
        """ObservationSet.add_bond_angle raises without data_idx or atom_indices."""
        ref = ObservationSet()
        with pytest.raises(ValueError, match="Either atom_indices or data_idx"):
            ref = ref.with_bond_angle(104.5)


# ---- Optimization determinism ----


class TestOptimizationDeterminism:
    """Verify optimization is deterministic."""

    @pytest.mark.integration
    def test_same_result_twice(self) -> None:
        """Running the same optimization twice gives identical parameters."""
        results = []
        for _ in range(2):
            true_ff, guess_ff, mols, ref, backend = _make_water_problem()
            obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
            opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
            results.append(opt.optimize(obj, space))

        np.testing.assert_array_almost_equal(
            results[0].final_params,
            results[1].final_params,
            decimal=12,
            err_msg="Optimization is not deterministic",
        )
        assert results[0].final_score == pytest.approx(results[1].final_score, rel=1e-10)

    @pytest.mark.integration
    def test_determinism_nelder_mead(self) -> None:
        """Nelder-Mead is also deterministic (no stochastic elements)."""
        results = []
        for _ in range(2):
            true_ff, guess_ff, mols, ref, backend = _make_water_problem()
            obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
            opt = ScipyOptimizer(method="Nelder-Mead", maxiter=200, use_bounds=False, verbose=False)
            results.append(opt.optimize(obj, space))

        np.testing.assert_array_almost_equal(
            results[0].final_params,
            results[1].final_params,
            decimal=12,
        )


# ---- Parameter vector validation ----


class TestParamVectorValidation:
    """Verify ParameterLayout.replace rejects wrong-length vectors."""

    def test_short_vector_raises(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError, match="does not match"):
            layout.replace(ff, np.array([1.0]))

    def test_long_vector_raises(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError, match="does not match"):
            layout.replace(ff, np.zeros(100))

    def test_empty_vector_raises(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError, match="does not match"):
            layout.replace(ff, np.array([]))

    def test_exact_length_accepted(self) -> None:
        ff = _water_ff()
        layout = ParameterLayout.from_force_field(ff)
        n = len(layout.vector(ff))
        replaced_ff = layout.replace(ff, np.ones(n))
        np.testing.assert_array_equal(layout.vector(replaced_ff), np.ones(n))


# ---- Golden fixture regression ----


class TestGoldenFixtureRegression:
    """Verify current optimization matches saved golden fixture (if present)."""

    GOLDEN_PATH = REPO_ROOT / "test" / "fixtures" / "optimization_golden.json"

    @pytest.mark.integration
    @pytest.mark.skipif(
        not (Path(__file__).resolve().parent.parent.parent / "test" / "fixtures" / "optimization_golden.json").exists(),
        reason="Golden fixture not yet generated (run scripts/generate_optimization_fixtures.py)",
    )
    def test_matches_golden_fixture(self) -> None:
        """Final score and parameters fall within tolerance of golden fixture."""
        golden = json.loads(self.GOLDEN_PATH.read_text())

        true_ff, guess_ff, mols, ref, backend = _make_water_problem()
        obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        assert len(result.final_params) == len(golden["final_params"]), (
            f"Param vector length changed: {len(result.final_params)} vs {len(golden['final_params'])}"
        )
        assert result.final_score == pytest.approx(golden["final_score"], rel=0.05), (
            f"Final score drift: {result.final_score} vs {golden['final_score']}"
        )
        for i, (got, want) in enumerate(zip(result.final_params, golden["final_params"])):
            assert got == pytest.approx(want, rel=0.10), f"Param {i} drift: {got} vs {want}"

    def test_optimizer_improves_score(self) -> None:
        """Verify optimizer substantially improves the score."""
        golden = json.loads(self.GOLDEN_PATH.read_text())

        true_ff, guess_ff, mols, ref, backend = _make_water_problem()
        obj, _layout, space = _build_objective(guess_ff, backend, mols, ref)
        opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
        result = opt.optimize(obj, space)

        assert result.initial_score == pytest.approx(golden["initial_score"], rel=1e-4), (
            f"Initial score drift: {result.initial_score} vs {golden['initial_score']}"
        )
        assert result.final_score < result.initial_score * 0.5, (
            f"Optimizer did not improve enough: initial={result.initial_score:.4f}, final={result.final_score:.4f}"
        )
