"""Integration tests for JaxOptOptimizer with cycling loop dispatch.

Verifies that ``OptimizationLoop(full_method="jaxopt:lbfgs")`` and
``"jaxopt:lbfgsb"`` properly dispatch to JaxOptOptimizer, and that
JaxOptOptimizer converges on real systems (CH₃F, water, SN₂ TS).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_HAS_JAX = importlib.util.find_spec("jax") is not None
_HAS_JAXOPT = importlib.util.find_spec("jaxopt") is not None

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.skipif(not _HAS_JAXOPT, reason="jaxopt not installed"),
    pytest.mark.jax,
    pytest.mark.integration,
]

from test._shared import (
    CH3F_FREQS,
    CH3F_HESS,
    CH3F_XYZ,
    SN2_FREQS,
    SN2_HESSIAN,
    SN2_XYZ,
    make_water,
)

from q2mm.io.xyz import load_xyz
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.seminario import qfuerza_fresh
from q2mm.optimizers.objective import ObjectiveFunction

JaxEngine = None


def _load_qm_frequencies(path: Path) -> np.ndarray:
    """Load QM frequencies from text file, skipping comment lines."""
    lines = path.read_text().strip().splitlines()
    return np.array([float(line) for line in lines if not line.startswith("#")])


def _layout(forcefield: ForceField) -> ParameterLayout:
    return ParameterLayout.from_force_field(forcefield)


def _params(forcefield: ForceField) -> np.ndarray:
    return _layout(forcefield).vector(forcefield)


def _materialize(forcefield: ForceField, vector: np.ndarray) -> ForceField:
    return _layout(forcefield).replace(forcefield, vector)


def _make_objective(
    forcefield: ForceField, engine: object, molecules: list, reference: ObservationSet, **kwargs: object
) -> ObjectiveFunction:
    return ObjectiveFunction(
        forcefield=forcefield,
        engine=engine,
        molecules=molecules,
        reference=reference,
        layout=_layout(forcefield),
        **kwargs,
    )


def _all_active_space(objective: ObjectiveFunction) -> ActiveParameterSpace:
    return ActiveParameterSpace.all_active(objective.layout, objective.forcefield)


def _scale_force_constants(
    forcefield: ForceField,
    *,
    bond_scale: float | None = None,
    angle_scale: float | None = None,
) -> ForceField:
    vector = _params(forcefield).copy()
    layout = _layout(forcefield)
    for slot in layout:
        if slot.field != "force_constant":
            continue
        if slot.owner == "bonds" and bond_scale is not None:
            vector[slot.index] *= bond_scale
        elif slot.owner == "angles" and angle_scale is not None:
            vector[slot.index] *= angle_scale
    return layout.replace(forcefield, vector)


def _water_ff(
    bond_k: float = 553.0,
    bond_r0: float = 0.96,
    angle_k: float = 49.9,
    angle_eq: float = 104.5,
) -> ForceField:
    return ForceField(
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
        functional_form=FunctionalForm.HARMONIC,
    )


@pytest.fixture(autouse=True)
def _init_jax() -> None:
    from q2mm.backends.mm._jax_common import ensure_jax, ensure_jaxopt

    ensure_jax()
    ensure_jaxopt()
    global JaxEngine  # noqa: PLW0603
    from q2mm.backends.mm.jax_engine import JaxEngine as _JE

    JaxEngine = _JE


# ---------------------------------------------------------------------------
# Cycling dispatch tests (water — fast, no QM data dependency)
# ---------------------------------------------------------------------------


class TestCyclingJaxoptDispatch:
    """OptimizationLoop jaxopt: dispatch integration tests."""

    def test_jaxopt_lbfgs_dispatch(self) -> None:
        """OptimizationLoop dispatches jaxopt:lbfgs and improves score."""
        from q2mm.optimizers.cycling import OptimizationLoop

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=110.0)
        engine = JaxEngine()

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        loop = OptimizationLoop(
            obj,
            _all_active_space(obj),
            full_method="jaxopt:lbfgs",
            max_cycles=1,
            full_maxiter=100,
            verbose=False,
        )
        result = loop.run()

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )

    def test_jaxopt_lbfgsb_dispatch(self) -> None:
        """OptimizationLoop dispatches jaxopt:lbfgsb and improves score."""
        from q2mm.optimizers.cycling import OptimizationLoop

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        ff = _water_ff(bond_k=400.0, bond_r0=1.05, angle_k=35.0, angle_eq=110.0)
        engine = JaxEngine()

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        loop = OptimizationLoop(
            obj,
            _all_active_space(obj),
            full_method="jaxopt:lbfgsb",
            max_cycles=1,
            full_maxiter=100,
            verbose=False,
        )
        result = loop.run()

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )


# ---------------------------------------------------------------------------
# End-to-end validation on real QM systems
# ---------------------------------------------------------------------------
class TestJaxOptCH3FValidation:
    """Validate JaxOptOptimizer on CH₃F with real QM reference data."""

    @pytest.fixture(scope="class")
    def ch3f_mol(self) -> Molecule:
        return load_xyz(CH3F_XYZ, bond_tolerance=1.5)

    @pytest.fixture(scope="class")
    def qm_freqs(self) -> np.ndarray:
        return _load_qm_frequencies(CH3F_FREQS)

    @pytest.fixture(scope="class")
    def seminario_ff(self, ch3f_mol: Molecule) -> ForceField:
        hess = np.load(CH3F_HESS)
        mol_h = ch3f_mol.with_hessian(hess)
        return qfuerza_fresh(mol_h, functional_form=FunctionalForm.HARMONIC)

    def test_freq_only_convergence(
        self,
        ch3f_mol: Molecule,
        seminario_ff: ForceField,
        qm_freqs: np.ndarray,
    ) -> None:
        """JaxOpt L-BFGS-B converges on CH₃F frequency-only objective.

        Asserts score improves and reaches a comparable minimum to SciPy.
        Note: RMSD may not improve from Seminario starting point — this is
        expected behavior on the CH₃F harmonic landscape (~529 cm⁻¹).
        """
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        engine = JaxEngine()

        # Compute mode indices from Seminario FF (correct mode ordering)
        mm_all = engine.frequencies(ch3f_mol, seminario_ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(qm_freqs[qm_freqs > 50.0])
        n = min(len(qm_real), len(mm_real_indices))

        ref = ObservationSet()
        for k in range(n):
            ref = ref.with_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, case_id="0")

        # Perturb to create non-trivial starting point
        ff = _scale_force_constants(seminario_ff, bond_scale=0.9, angle_scale=0.9)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[ch3f_mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgsb", maxiter=500)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )
        # JaxOpt should reach near-zero loss (comparable to SciPy ~0.01)
        assert result.final_score < 0.1, f"Final score should be small, got {result.final_score:.6f}"

    def test_mixed_objective_convergence(
        self,
        ch3f_mol: Molecule,
        seminario_ff: ForceField,
        qm_freqs: np.ndarray,
    ) -> None:
        """JaxOpt L-BFGS converges on CH₃F energy + frequency objective."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        engine = JaxEngine()

        # Mode indices from Seminario FF
        mm_all = engine.frequencies(ch3f_mol, seminario_ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(qm_freqs[qm_freqs > 50.0])
        n = min(len(qm_real), len(mm_real_indices))

        ref = ObservationSet()
        ref = ref.with_energy(value=0.0, case_id="0", weight=1.0)
        for k in range(n):
            ref = ref.with_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, case_id="0")

        ff = _scale_force_constants(seminario_ff, bond_scale=0.8)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[ch3f_mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=500)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score, (
            f"Mixed objective should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )

    def test_cycling_dispatch_on_ch3f(
        self,
        ch3f_mol: Molecule,
        seminario_ff: ForceField,
        qm_freqs: np.ndarray,
    ) -> None:
        """OptimizationLoop jaxopt:lbfgs on CH₃F for 2 cycles."""
        from q2mm.optimizers.cycling import OptimizationLoop

        engine = JaxEngine()

        # Mode indices from Seminario FF
        mm_all = engine.frequencies(ch3f_mol, seminario_ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(qm_freqs[qm_freqs > 50.0])
        n = min(len(qm_real), len(mm_real_indices))

        ref = ObservationSet()
        for k in range(n):
            ref = ref.with_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, case_id="0")

        ff = _scale_force_constants(seminario_ff, bond_scale=0.8)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[ch3f_mol], reference=ref)

        loop = OptimizationLoop(
            obj,
            _all_active_space(obj),
            full_method="jaxopt:lbfgs",
            simp_method="Nelder-Mead",
            max_cycles=2,
            full_maxiter=200,
            verbose=False,
        )
        result = loop.run()

        assert result.n_cycles >= 1, f"Expected at least 1 cycle, got {result.n_cycles}"
        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )


class TestJaxOptWaterHessianValidation:
    """Validate JaxOptOptimizer on water with hessian-element objective."""

    def test_hessian_only_convergence(self) -> None:
        """JaxOpt L-BFGS converges on water hessian-element objective."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        mol = make_water(bond_length=0.96, angle_deg=104.5)
        eq_ff = _water_ff()
        engine = JaxEngine()

        # Compute hessian at equilibrium to serve as "QM" reference
        qm_hess = engine.hessian(mol, eq_ff)

        # Perturbed starting FF
        ff = _water_ff(bond_k=400.0, bond_r0=1.02, angle_k=35.0, angle_eq=108.0)

        ref = ObservationSet()
        ref = ref.with_hessian_from_matrix(qm_hess, diagonal_only=True, case_id="0", diagonal_weight=1.0)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgs", maxiter=200)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score, (
            f"Hessian loss should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )


class TestJaxOptSN2TSValidation:
    """Validate JaxOptOptimizer on SN₂ transition state with frequencies."""

    @pytest.fixture(scope="class")
    def ts_mol(self) -> Molecule:
        return load_xyz(SN2_XYZ, bond_tolerance=1.5)

    @pytest.fixture(scope="class")
    def ts_qm_freqs(self) -> np.ndarray:
        return _load_qm_frequencies(SN2_FREQS)

    @pytest.fixture(scope="class")
    def ts_seminario_ff(self, ts_mol: Molecule) -> ForceField:
        hess = np.load(SN2_HESSIAN)
        mol_h = ts_mol.with_hessian(hess)
        return qfuerza_fresh(mol_h, functional_form=FunctionalForm.HARMONIC, invert_ts_curvature=True)

    def test_sn2_ts_freq_convergence(
        self,
        ts_mol: Molecule,
        ts_seminario_ff: ForceField,
        ts_qm_freqs: np.ndarray,
    ) -> None:
        """JaxOpt L-BFGS-B converges on SN₂ TS frequency objective."""
        from q2mm.optimizers.jaxopt_opt import JaxOptOptimizer

        engine = JaxEngine()

        # Mode indices from Seminario FF (correct ordering)
        mm_all = engine.frequencies(ts_mol, ts_seminario_ff)
        mm_real_indices = sorted([i for i, f in enumerate(mm_all) if f > 50.0])
        qm_real = sorted(ts_qm_freqs[ts_qm_freqs > 50.0])
        n = min(len(qm_real), len(mm_real_indices))

        ref = ObservationSet()
        for k in range(n):
            ref = ref.with_frequency(float(qm_real[k]), data_idx=mm_real_indices[k], weight=0.001, case_id="0")

        ff = _scale_force_constants(ts_seminario_ff, bond_scale=0.9)

        obj = _make_objective(forcefield=ff, engine=engine, molecules=[ts_mol], reference=ref)

        optimizer = JaxOptOptimizer(method="lbfgsb", maxiter=500)
        result = optimizer.optimize(obj, _all_active_space(obj))

        assert result.final_score < result.initial_score, (
            f"Score should improve: {result.initial_score:.6f} → {result.final_score:.6f}"
        )
