"""Contract tests for the Python and JAX objective executors.

Covers Python/JAX parity (value, calculated values, residuals, gradients),
full-vector gradient identity, frozen-slot preservation, no-mutation,
prepare-once-per-case, per-case JIT split, explicit gradient-mode
declaration, and the no-silent-fallback ObjectiveGradientError.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pytest

from q2mm.models.observations import ObservationSet
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import Evaluation, GradientMode, ObjectiveGradientError

_HAS_JAX = importlib.util.find_spec("jax") is not None
_RH_PUBLICATION_CATEGORY_RTOL = 1e-5
_RH_PUBLICATION_CATEGORY_ATOL = 1e-4

pytestmark = [
    pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed"),
    pytest.mark.jax,
]


@pytest.fixture(scope="module")
def ch3f_problem() -> Any:
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems import load_system

    backend = JaxBackend()
    case = load_system("ch3f", backend=backend, functional_form="harmonic")
    return backend, case.problem


@pytest.fixture(scope="module")
def sn2_geometry_plan() -> Any:
    """Return a geometry + eigenmatrix TS plan (exercises the geometry path)."""
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems import load_system

    backend = JaxBackend()
    case = load_system("ch3f-sn2", backend=backend, functional_form="harmonic")
    problem = case.problem
    obs = ObservationSet.from_molecules(
        list(problem.molecules),
        case_ids=list(problem.case_ids),
        include_geometry=True,
        include_eigenmatrix=True,
        eigenmatrix_diagonal_only=True,
    )
    plan = ObjectivePlan(
        case_ids=problem.case_ids,
        molecules=problem.molecules,
        stationary_points=tuple(c.stationary_point for c in problem.cases),
        observations=obs,
        layout=problem.layout,
        active_space=problem.active_space,
    )
    return backend, problem, plan


def _executors(backend: Any, problem: Any) -> Any:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    plan = ObjectivePlan.from_problem(problem)
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    return plan, py, jx


def test_gradient_modes_declared(ch3f_problem: Any) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    assert PythonObjectiveExecutor(plan, backend, problem.starting_force_field).gradient_mode is GradientMode.NONE
    assert (
        PythonObjectiveExecutor(
            plan, backend, problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL
        ).gradient_mode
        is GradientMode.ANALYTICAL
    )
    assert JaxObjectiveExecutor(plan, backend, problem.starting_force_field).gradient_mode is GradientMode.ANALYTICAL


def test_value_parity(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, jx = _executors(backend, problem)
    x = problem.active_space.baseline
    assert py.value(x) == pytest.approx(jx.value(x), rel=1e-8)


def test_calculated_and_residual_parity(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, jx = _executors(backend, problem)
    x = problem.active_space.baseline
    ev_py = py.evaluate(x)
    ev_jx = jx.evaluate(x)
    assert isinstance(ev_py, Evaluation)
    np.testing.assert_allclose(ev_py.calculated, ev_jx.calculated, atol=1e-6)
    np.testing.assert_allclose(ev_py.weighted_residuals, ev_jx.weighted_residuals, atol=1e-6)
    np.testing.assert_allclose(ev_py.raw_residuals, ev_jx.raw_residuals, atol=1e-6)
    assert ev_py.category_scores.keys() == ev_jx.category_scores.keys()


def test_gradient_parity_full_length(ch3f_problem: Any) -> None:
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    x = problem.active_space.baseline
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL)
    _p2, _pyn, jx = _executors(backend, problem)
    _v_py, g_py = py.value_and_gradient(x)
    _v_jx, g_jx = jx.value_and_gradient(x)
    # both gradients are full length
    assert g_py.shape == (plan.n_params,)
    assert g_jx.shape == (plan.n_params,)
    np.testing.assert_allclose(g_py, g_jx, atol=1e-6)


def test_evaluate_does_not_mutate_or_count(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    x = np.array(problem.active_space.baseline)
    x_copy = x.copy()
    before = py.n_evaluations
    py.evaluate(x)
    assert py.n_evaluations == before  # evaluate does not count
    np.testing.assert_array_equal(x, x_copy)  # input not mutated
    # base force field is not mutated
    assert py.base_force_field is problem.starting_force_field


def test_value_counts_and_history(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    x = problem.active_space.baseline
    py.reset()
    py.value(x)
    py.value(x)
    assert py.n_evaluations == 2
    assert len(py.history) == 2


def test_prepared_once_per_case(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, jx = _executors(backend, problem)
    x = problem.active_space.baseline
    for _ in range(3):
        py.value(x)
    assert set(py._prepared.keys()) == set(problem.case_ids)
    assert len(py._prepared) == len(problem.case_ids)
    for _ in range(3):
        jx.value(x)
    assert set(jx._sessions.keys()) == set(problem.case_ids)


def test_jax_per_case_jit_split(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, _py, jx = _executors(backend, problem)
    # One compiled value_and_grad fragment per case-with-references (no single
    # all-molecule graph). CH3F is single-molecule => exactly one fragment.
    assert len(jx._compiled_vag_fns) == len(jx._compiled_value_fns)
    assert len(jx._compiled_vag_fns) >= len(problem.case_ids)


def test_analytical_geometry_raises_for_python(sn2_geometry_plan: Any) -> None:
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, _problem, plan = sn2_geometry_plan
    assert "geometry" in plan.categories
    with pytest.raises(ObjectiveGradientError):
        PythonObjectiveExecutor(plan, backend, _problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL)


def test_value_and_gradient_none_mode_raises(ch3f_problem: Any) -> None:
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field, gradient_mode=GradientMode.NONE)
    with pytest.raises(ObjectiveGradientError):
        py.value_and_gradient(problem.active_space.baseline)


def test_geometry_value_parity(sn2_geometry_plan: Any) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = sn2_geometry_plan
    x = problem.active_space.baseline
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    ev_py = py.evaluate(x)
    ev_jx = jx.evaluate(x)
    obs = plan.observations.values

    # The two executors relax geometry with different minimizers (native
    # backend minimize vs implicit-diff jaxopt LBFGS) that converge to
    # slightly different points in the flat angle direction of the MM PES.
    # Bond lengths and eigenmatrix elements match tightly; bond angles agree
    # to sub-milli-degree.  Use justified per-kind absolute tolerances, not a
    # relative tolerance on the near-zero total.
    per_kind_atol = {"bond_length": 1e-4, "bond_angle": 5e-3, "eig_diagonal": 1e-7}
    for i, o in enumerate(obs):
        atol = per_kind_atol.get(o.kind, 1e-6)
        assert abs(ev_py.calculated[i] - ev_jx.calculated[i]) < atol, o.kind
    # Total agrees to an absolute tolerance (the objective is ~2e-5).
    assert py.value(x) == pytest.approx(jx.value(x), abs=1e-4)


@pytest.fixture(scope="module")
def all_kinds_plan() -> Any:
    """Return a single-case plan exercising all hessian + geometry kinds."""
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems import load_system

    backend = JaxBackend()
    case = load_system("ch3f-sn2", backend=backend, functional_form="harmonic")
    problem = case.problem
    mol = problem.molecules[0]
    cid = problem.case_ids[0]
    bond = (mol.bonds or ())[0]
    angle = (mol.angles or ())[0]
    obs = (
        ObservationSet()
        .with_energy(0.0, weight=1.0, case_id=cid)
        .with_frequency(0.0, data_idx=6, weight=1.0, case_id=cid)
        .with_bond_length(0.0, atom_indices=(bond.atom_i, bond.atom_j), weight=1.0, case_id=cid)
        .with_bond_angle(0.0, atom_indices=(angle.atom_i, angle.atom_j, angle.atom_k), weight=1.0, case_id=cid)
        .with_hessian_eigenvalue(0.0, mode_idx=6, weight=1.0, case_id=cid)
        .with_hessian_offdiagonal(0.0, row=6, col=7, weight=1.0, case_id=cid)
        .with_hessian_element(0.0, row=0, col=0, weight=1.0, case_id=cid)
    )
    plan = ObjectivePlan(
        case_ids=(cid,),
        molecules=(mol,),
        stationary_points=(problem.cases[0].stationary_point,),
        observations=obs,
        layout=problem.layout,
        active_space=problem.active_space,
    )
    return backend, problem, plan


def test_all_kinds_calc_and_residual_parity(all_kinds_plan: Any) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = all_kinds_plan
    x = problem.active_space.baseline
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    ev_py = py.evaluate(x)
    ev_jx = jx.evaluate(x)
    obs = plan.observations.values
    assert {o.kind for o in obs} == {
        "energy",
        "frequency",
        "bond_length",
        "bond_angle",
        "eig_diagonal",
        "eig_offdiagonal",
        "hessian_element",
    }
    per_kind_atol = {"bond_length": 1e-4, "bond_angle": 5e-3}
    for i, o in enumerate(obs):
        atol = per_kind_atol.get(o.kind, 1e-6)
        assert abs(ev_py.calculated[i] - ev_jx.calculated[i]) < atol, o.kind
        assert abs(ev_py.raw_residuals[i] - ev_jx.raw_residuals[i]) < max(atol, 1e-6)
        assert abs(ev_py.weighted_residuals[i] - ev_jx.weighted_residuals[i]) < max(atol, 1e-6)
    assert ev_py.category_scores.keys() == ev_jx.category_scores.keys()
    for k in ev_py.category_scores:
        # Category scores square the residual, so the geometry-relaxation floor
        # (bond_angle ~2e-5 deg, amplified by the synthetic 180 deg residual)
        # bounds the achievable absolute agreement; a relative gate is correct.
        assert ev_py.category_scores[k] == pytest.approx(ev_jx.category_scores[k], rel=1e-5, abs=1e-6)
    assert ev_py.data_value == pytest.approx(ev_jx.data_value, rel=1e-5, abs=1e-6)


def test_all_kinds_gradient_parity(all_kinds_plan: Any) -> None:
    """Non-geometry analytical gradients match between the executors."""
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = all_kinds_plan
    non_geom = ObservationSet(
        tuple(o for o in plan.observations.values if o.kind not in ("bond_length", "bond_angle", "torsion_angle"))
    )
    plan2 = plan.with_observations(non_geom)
    x = problem.active_space.baseline
    py = PythonObjectiveExecutor(plan2, backend, problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL)
    jx = JaxObjectiveExecutor(plan2, backend, problem.starting_force_field)
    _, g_py = py.value_and_gradient(x)
    _, g_jx = jx.value_and_gradient(x)
    assert g_py.shape == (plan2.n_params,)
    np.testing.assert_allclose(g_py, g_jx, atol=1e-5, rtol=1e-5)


def test_torsion_wrapping_and_kernel_parity() -> None:
    """Torsion-difference wrapping and the JAX dihedral kernel vs NumPy."""
    import numpy as np

    from q2mm.geometry import dihedral_angle
    from q2mm.objectives.jax import _torsion_angles_deg
    from q2mm.objectives.metrics import raw_residual, torsion_wrap
    from test._shared import make_ethane

    # Wrapping of torsion-angle differences into [-180, 180).
    assert torsion_wrap(190.0) == pytest.approx(-170.0)
    assert torsion_wrap(-190.0) == pytest.approx(170.0)
    assert torsion_wrap(179.0) == pytest.approx(179.0)
    # raw_residual applies the wrap for torsion_angle kinds only.
    assert raw_residual("torsion_angle", 179.0, -179.0) == pytest.approx(-2.0)
    assert raw_residual("bond_length", 1.0, 3.0) == pytest.approx(-2.0)

    # The JAX torsion kernel matches the NumPy reference dihedral on fixed
    # coordinates (no force field / minimization needed).
    mol = make_ethane()
    tors = (mol.torsions or ())[0]
    quad = (tors.atom_i, tors.atom_j, tors.atom_k, tors.atom_l)
    coords = np.asarray(mol.geometry, dtype=float)
    atoms = np.array([quad], dtype=int)
    jax_val = float(np.asarray(_torsion_angles_deg(coords, atoms))[0])
    ref_val = float(dihedral_angle(coords[quad[0]], coords[quad[1]], coords[quad[2]], coords[quad[3]]))
    assert jax_val == pytest.approx(ref_val, abs=1e-6)


def test_regularization_and_least_squares_parity(all_kinds_plan: Any) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = all_kinds_plan
    non_geom = ObservationSet(
        tuple(o for o in plan.observations.values if o.kind not in ("bond_length", "bond_angle", "torsion_angle"))
    )
    ref = np.array(plan.active_space.baseline)
    plan_reg = ObjectivePlan(
        case_ids=plan.case_ids,
        molecules=plan.molecules,
        stationary_points=plan.stationary_points,
        observations=non_geom,
        layout=plan.layout,
        active_space=plan.active_space,
        regularization=0.01,
        reference_params=ref,
    )
    x = np.array(ref) + 0.05  # off the anchor so the L2 term is nonzero
    py = PythonObjectiveExecutor(plan_reg, backend, problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL)
    jx = JaxObjectiveExecutor(plan_reg, backend, problem.starting_force_field)
    assert py.evaluate(x).regularization > 0
    assert py.evaluate(x).regularization == pytest.approx(jx.evaluate(x).regularization, rel=1e-10)
    v_py, g_py = py.value_and_gradient(x)
    v_jx, g_jx = jx.value_and_gradient(x)
    assert v_py == pytest.approx(v_jx, rel=1e-6, abs=1e-6)
    np.testing.assert_allclose(g_py, g_jx, atol=1e-5, rtol=1e-5)
    r_py = py.least_squares_residuals(x)
    r_jx = jx.least_squares_residuals(x)
    assert r_py.shape == r_jx.shape
    np.testing.assert_allclose(r_py, r_jx, atol=1e-6)
    assert float(np.sum(r_py**2)) == pytest.approx(py.value(x), rel=1e-6)


def test_category_metrics_parity(all_kinds_plan: Any) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.metrics import category_metrics
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = all_kinds_plan
    x = problem.active_space.baseline
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    m_py = category_metrics(plan, py.evaluate(x))
    m_jx = category_metrics(plan, jx.evaluate(x))
    assert set(m_py) == set(m_jx)
    for kind in m_py:
        for stat in ("n_refs", "r2", "rmsd", "mae"):
            a, b = m_py[kind][stat], m_jx[kind][stat]
            if a != a:  # nan
                assert b != b
            else:
                assert a == pytest.approx(b, rel=1e-3, abs=1e-3)


def test_evaluation_is_deeply_immutable(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    ev = py.evaluate(problem.active_space.baseline)
    assert not ev.calculated.flags.writeable
    with pytest.raises(Exception):
        ev.category_scores["energy"] = 1.0  # type: ignore[index]


def test_reset_retains_prepared_sessions(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    x = problem.active_space.baseline
    py.value(x)
    handles = {cid: id(s) for cid, s in py._prepared.items()}
    assert handles
    py.reset()
    assert py.n_evaluations == 0
    assert py.history == ()
    py.value(x)
    assert {cid: id(s) for cid, s in py._prepared.items()} == handles


def test_sample_does_not_count(ch3f_problem: Any) -> None:
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    x = problem.active_space.baseline
    py.reset()
    py.sample(x)
    py.sample(x)
    assert py.n_evaluations == 0
    assert py.history == ()


def test_frozen_slots_preserved_by_optimizer(ch3f_problem: Any) -> None:
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.objectives.python import PythonObjectiveExecutor
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    layout = plan.layout
    baseline = np.array(problem.active_space.baseline)
    # Freeze all but the first active slot.
    active = np.array(sorted(int(i) for i in problem.active_space.active_indices)[:1], dtype=int)
    space = ActiveParameterSpace(layout=layout, baseline=baseline, active_indices=active)
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    result = ScipyOptimizer(method="L-BFGS-B", maxiter=3, verbose=False).optimize(py, space)
    assert result.final_params.shape == (plan.n_params,)
    frozen = np.array([i for i in range(len(layout)) if i not in set(active.tolist())])
    np.testing.assert_array_equal(result.final_params[frozen], baseline[frozen])


def test_multi_case_per_case_jit_and_python_aggregation(ch3f_problem: Any) -> None:
    """Two same-topology cases compile to two independent JIT fragments.

    Proves the JAX executor keeps one compiled per-case fragment and sums in
    Python (no single all-molecule XLA graph), and that the Python executor
    aggregates the same two cases to the identical total.
    """
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem = ch3f_problem
    mol = problem.molecules[0]
    sp = problem.cases[0].stationary_point
    cids = ("conf-a", "conf-b")
    obs = (
        ObservationSet()
        .with_frequency(0.0, data_idx=6, weight=1.0, case_id="conf-a")
        .with_frequency(0.0, data_idx=6, weight=1.0, case_id="conf-b")
    )
    plan = ObjectivePlan(
        case_ids=cids,
        molecules=(mol, mol),
        stationary_points=(sp, sp),
        observations=obs,
        layout=problem.layout,
        active_space=problem.active_space,
    )
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    # Exactly one compiled fragment per case -> Python-side aggregation.
    assert len(jx._compiled_value_fns) == 2
    assert len(jx._compiled_vag_fns) == 2
    assert set(jx._sessions.keys()) == set(cids)
    x = problem.active_space.baseline
    assert py.value(x) == pytest.approx(jx.value(x), rel=1e-6, abs=1e-9)


@pytest.fixture(scope="module")
def rh_enamide_plan() -> Any:
    """Return a multi-molecule Rh-enamide geometry + eigenmatrix TS plan.

    Skips when the (non-distributed) external Rh-enamide dataset is absent.
    """
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems import load_system

    try:
        case = load_system("rh-enamide", functional_form="harmonic")
    except FileNotFoundError as exc:  # external dataset not present
        pytest.skip(f"Rh-enamide dataset unavailable: {exc}")
    problem = case.problem
    obs = ObservationSet.from_molecules(
        list(problem.molecules),
        case_ids=list(problem.case_ids),
        include_geometry=True,
        include_eigenmatrix=True,
        eigenmatrix_diagonal_only=True,
    )
    plan = ObjectivePlan(
        case_ids=problem.case_ids,
        molecules=problem.molecules,
        stationary_points=tuple(c.stationary_point for c in problem.cases),
        observations=obs,
        layout=problem.layout,
        active_space=problem.active_space,
    )
    return JaxBackend(), problem, plan


def test_rh_enamide_publication_geometry_eigenmatrix_parity(rh_enamide_plan: Any) -> None:
    """Python/JAX total + category parity on the published Rh-enamide TS system."""
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = rh_enamide_plan
    x = problem.active_space.baseline
    # Report the executed scope: multi-molecule cases + geometry/eigenmatrix obs.
    n_obs = len(plan.observations.values)
    assert len(problem.case_ids) >= 2, "Rh-enamide is a multi-molecule system"
    assert n_obs > 0
    print(f"Rh-enamide parity: {len(problem.case_ids)} cases, {n_obs} observations")
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    ev_py = py.evaluate(x)
    ev_jx = jx.evaluate(x)
    # One compiled fragment per case (multi-molecule Python aggregation).
    assert len(jx._compiled_value_fns) >= len(problem.case_ids)
    assert ev_py.category_scores.keys() == ev_jx.category_scores.keys()
    for category in ev_py.category_scores:
        python_value = ev_py.category_scores[category]
        jax_value = ev_jx.category_scores[category]
        assert python_value == pytest.approx(
            jax_value,
            rel=_RH_PUBLICATION_CATEGORY_RTOL,
            abs=_RH_PUBLICATION_CATEGORY_ATOL,
        ), f"Rh-enamide {category} category parity failed: Python={python_value:.15g}, JAX={jax_value:.15g}"
    assert ev_py.data_value == pytest.approx(
        ev_jx.data_value,
        rel=_RH_PUBLICATION_CATEGORY_RTOL,
        abs=_RH_PUBLICATION_CATEGORY_ATOL,
    ), f"Rh-enamide total parity failed: Python={ev_py.data_value:.15g}, JAX={ev_jx.data_value:.15g}"


def test_finite_difference_step_reporting(ch3f_problem: Any) -> None:
    """finite_difference_step is the step only in FD mode; None otherwise."""
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    none_py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    assert none_py.gradient_mode is GradientMode.NONE
    assert none_py.finite_difference_step is None
    fd_py = PythonObjectiveExecutor(
        plan, backend, problem.starting_force_field, gradient_mode=GradientMode.FINITE_DIFFERENCE, fd_step=1e-3
    )
    assert fd_py.finite_difference_step == 1e-3
    an_py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field, gradient_mode=GradientMode.ANALYTICAL)
    assert an_py.finite_difference_step is None
    jx = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)
    assert jx.finite_difference_step is None


def test_record_evaluation_counts_without_backend(ch3f_problem: Any) -> None:
    """record_evaluation increments the counter and history with no backend call."""
    backend, problem = ch3f_problem
    _plan, py, _jx = _executors(backend, problem)
    py.reset()
    n_prepared_before = len(py._prepared)
    py.record_evaluation(1.5)
    py.record_evaluation(2.5)
    assert py.n_evaluations == 2
    assert py.history == (1.5, 2.5)
    # No preparation happened (no backend work).
    assert len(py._prepared) == n_prepared_before


def test_least_squares_nfev_and_history(ch3f_problem: Any) -> None:
    """A least_squares run records real per-call evaluations (nfev, history)."""
    from q2mm.objectives.python import PythonObjectiveExecutor
    from q2mm.optimizers.scipy_opt import ScipyOptimizer

    backend, problem = ch3f_problem
    plan = ObjectivePlan.from_problem(problem)
    py = PythonObjectiveExecutor(plan, backend, problem.starting_force_field)
    result = ScipyOptimizer(method="least_squares", maxiter=5, verbose=False).optimize(py, problem.active_space)
    assert result.gradient_mode == "finite_difference"
    assert result.fd_step is not None
    assert result.n_evaluations > 0
    assert len(result.history) == result.n_evaluations
