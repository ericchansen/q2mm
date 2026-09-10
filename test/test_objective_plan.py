"""Direct validation tests for :class:`q2mm.objectives.plan.ObjectivePlan`.

Covers immutability, case-ID/layout validation, category derivation, and
the derived-plan builders — independent of any backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import Observation, ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.metrics import category_metrics, category_stats, raw_residual, weighted_residual
from q2mm.objectives.plan import KIND_TO_CATEGORY, ObjectivePlan
from q2mm.objectives.protocols import Evaluation
from test._shared import make_diatomic, make_ethane, make_noble_gas_pair, make_water


def _ff() -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=60.0, equilibrium=104.5)],
        functional_form=FunctionalForm.MM3,
    )


def _plan(*, observations: ObservationSet | None = None, molecule: Molecule | None = None) -> ObjectivePlan:
    ff = _ff()
    mol = molecule if molecule is not None else make_water()
    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    obs = observations if observations is not None else ObservationSet().with_energy(1.0, weight=1.0, case_id="0")
    return ObjectivePlan(
        case_ids=("0",),
        molecules=(mol,),
        stationary_points=(StationaryPointKind.GROUND_STATE,),
        observations=obs,
        layout=layout,
        active_space=space,
    )


def test_plan_is_frozen() -> None:
    plan = _plan()
    with pytest.raises(Exception):
        plan.regularization = 1.0  # type: ignore[misc]


def test_array_backed_objective_records_use_identity_equality() -> None:
    first_plan = _plan()
    second_plan = _plan()
    first_evaluation = Evaluation(
        total=1.0,
        data_value=1.0,
        regularization=0.0,
        calculated=np.array([1.0, 2.0, 3.0]),
        raw_residuals=np.array([0.0, 0.0, 0.0]),
        weighted_residuals=np.array([0.0, 0.0, 0.0]),
        category_scores={"energy": 1.0},
    )
    second_evaluation = Evaluation(
        total=1.0,
        data_value=1.0,
        regularization=0.0,
        calculated=np.array([1.0, 2.0, 3.0]),
        raw_residuals=np.array([0.0, 0.0, 0.0]),
        weighted_residuals=np.array([0.0, 0.0, 0.0]),
        category_scores={"energy": 1.0},
    )

    assert first_plan != second_plan
    assert first_evaluation != second_evaluation
    assert len({first_plan, second_plan, first_evaluation, second_evaluation}) == 4


def test_reference_params_default_is_baseline_and_readonly() -> None:
    plan = _plan()
    np.testing.assert_array_equal(plan.reference_params, plan.active_space.baseline)
    assert not plan.reference_params.flags.writeable


def test_n_params_matches_layout() -> None:
    plan = _plan()
    assert plan.n_params == len(plan.layout)


def test_case_index_and_unknown_raises() -> None:
    plan = _plan()
    assert plan.case_index("0") == 0
    with pytest.raises(KeyError):
        plan.case_index("nope")


def test_observation_case_id_must_resolve() -> None:
    bad = ObservationSet().with_energy(1.0, weight=1.0, case_id="does-not-exist")
    with pytest.raises(ValueError, match="does not match|not among"):
        _plan(observations=bad)


def test_duplicate_case_ids_rejected() -> None:
    ff = _ff()
    mol = make_water()
    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    with pytest.raises(ValueError, match="unique"):
        ObjectivePlan(
            case_ids=("0", "0"),
            molecules=(mol, mol),
            stationary_points=(StationaryPointKind.GROUND_STATE, StationaryPointKind.GROUND_STATE),
            observations=ObservationSet(),
            layout=layout,
            active_space=space,
        )


def test_active_space_layout_must_match() -> None:
    ff = _ff()
    other_ff = _ff()
    mol = make_water()
    layout = ParameterLayout.from_force_field(ff)
    # A space built over a *different* layout instance with different slots.
    other_layout = ParameterLayout.from_force_field(
        ForceField(
            name="two-bond",
            bonds=[
                BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),
                BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),
            ],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=60.0, equilibrium=104.5)],
            functional_form=FunctionalForm.MM3,
        )
    )
    mismatched_space = ActiveParameterSpace.all_active(
        other_layout,
        ForceField(
            name="two-bond",
            bonds=[
                BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),
                BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),
            ],
            angles=[AngleParam(elements=("H", "O", "H"), force_constant=60.0, equilibrium=104.5)],
            functional_form=FunctionalForm.MM3,
        ),
    )
    with pytest.raises(ValueError, match="active_space"):
        ObjectivePlan(
            case_ids=("0",),
            molecules=(mol,),
            stationary_points=(StationaryPointKind.GROUND_STATE,),
            observations=ObservationSet(),
            layout=layout,
            active_space=mismatched_space,
        )
    del other_ff


def test_negative_regularization_rejected() -> None:
    ff = _ff()
    mol = make_water()
    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    with pytest.raises(ValueError, match="non-negative"):
        ObjectivePlan(
            case_ids=("0",),
            molecules=(mol,),
            stationary_points=(StationaryPointKind.GROUND_STATE,),
            observations=ObservationSet(),
            layout=layout,
            active_space=space,
            regularization=-1.0,
        )


def test_categories_derived_from_observations() -> None:
    obs = (
        ObservationSet()
        .with_energy(1.0, weight=1.0, case_id="0")
        .with_frequency(100.0, data_idx=0, weight=1.0, case_id="0")
    )
    plan = _plan(observations=obs)
    assert plan.categories == frozenset({"energy", "frequency"})
    for kind in ("energy", "frequency", "bond_length", "eig_diagonal", "hessian_element"):
        assert kind in KIND_TO_CATEGORY


def test_with_observations_and_active_space_are_pure() -> None:
    plan = _plan()
    new_obs = ObservationSet().with_energy(2.0, weight=1.0, case_id="0")
    plan2 = plan.with_observations(new_obs)
    assert plan2.observations is new_obs
    assert plan.observations is not new_obs  # original untouched

    rebased = plan.active_space.with_baseline(plan.active_space.baseline + 0.1)
    plan3 = plan.with_active_space(rebased)
    np.testing.assert_allclose(plan3.reference_params, rebased.baseline)
    # original plan is unchanged
    np.testing.assert_array_equal(plan.active_space.baseline, plan.reference_params)


def test_from_problem_roundtrip() -> None:
    from q2mm.models.problem import OptimizationProblem, TrainingCase

    ff = _ff()
    mol = make_water()
    layout = ParameterLayout.from_force_field(ff)
    space = ActiveParameterSpace.all_active(layout, ff)
    obs = ObservationSet().with_energy(1.0, weight=1.0, case_id="w")
    problem = OptimizationProblem(
        cases=(TrainingCase(case_id="w", molecule=mol, stationary_point=StationaryPointKind.GROUND_STATE),),
        starting_force_field=ff,
        layout=layout,
        active_space=space,
        observations=obs,
    )
    plan = ObjectivePlan.from_problem(problem, regularization=0.01)
    assert plan.case_ids == ("w",)
    assert plan.regularization == 0.01
    np.testing.assert_array_equal(plan.reference_params, space.baseline)


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize(
    ("kind", "data_idx"),
    [("frequency", 9), ("frequency", 99), ("eig_diagonal", 9), ("bond_length", 2), ("bond_angle", 1)],
)
def test_plan_rejects_out_of_range_positional_indices(kind: str, data_idx: int, weight: float) -> None:
    obs = Observation(kind=kind, value=0.0, data_idx=data_idx, weight=weight, label="bad-index")
    with pytest.raises(ValueError, match="bad-index.*out of range"):
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize("kind", ["bond_length", "bond_angle"])
def test_plan_rejects_positional_index_without_topology(kind: str) -> None:
    obs = Observation(kind=kind, value=0.0, data_idx=0)
    with pytest.raises(ValueError, match="out of range"):
        _plan(observations=ObservationSet((obs,)), molecule=make_noble_gas_pair())


@pytest.mark.parametrize(
    ("kind", "arity"),
    [("bond_length", 2), ("bond_angle", 3), ("torsion_angle", 4), ("eig_offdiagonal", 2), ("hessian_element", 2)],
)
@pytest.mark.parametrize("offset", [-1, 1])
def test_plan_requires_exact_index_arity(kind: str, arity: int, offset: int) -> None:
    obs = Observation(kind=kind, value=0.0, atom_indices=(0,) * (arity + offset))
    with pytest.raises(ValueError, match=f"requires exactly {arity}"):
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize("kind", ["torsion_angle", "eig_offdiagonal", "hessian_element"])
def test_plan_requires_explicit_indices(kind: str) -> None:
    obs = Observation(kind=kind, value=0.0)
    with pytest.raises(ValueError, match="requires exactly"):
        _plan(observations=ObservationSet((obs,)), molecule=make_ethane())


@pytest.mark.parametrize("kind", ["eig_offdiagonal", "hessian_element"])
@pytest.mark.parametrize("indices", [(0, 9), (9, 0), (9, 9), (0, 99)])
def test_plan_rejects_matrix_indices_before_flattening(kind: str, indices: tuple[int, int]) -> None:
    obs = Observation(kind=kind, value=0.0, atom_indices=indices)
    with pytest.raises(ValueError, match="out of range"):
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize(
    ("kind", "indices"),
    [("bond_length", (0, 3)), ("bond_angle", (1, 0, 3)), ("torsion_angle", (0, 1, 2, 3))],
)
def test_plan_rejects_atom_indices_outside_case(kind: str, indices: tuple[int, ...]) -> None:
    obs = Observation(kind=kind, value=0.0, atom_indices=indices)
    with pytest.raises(ValueError, match="out of range"):
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize(
    ("kind", "indices"),
    [("bond_length", (1, 2)), ("bond_angle", (0, 1, 2)), ("bond_angle", (1, 2, 0))],
)
def test_plan_requires_bond_and_angle_topology_membership(kind: str, indices: tuple[int, ...]) -> None:
    obs = Observation(kind=kind, value=0.0, atom_indices=indices)
    with pytest.raises(ValueError, match="not in.*topology"):
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize(
    ("kind", "arity"),
    [("bond_length", 2), ("bond_angle", 3), ("torsion_angle", 4), ("eig_offdiagonal", 2), ("hessian_element", 2)],
)
@pytest.mark.parametrize("index", [-1, -0.5, 0.5, 1.0, True, np.bool_(False), "1"])
def test_used_atom_indices_are_not_lossily_coerced(kind: str, arity: int, index: object) -> None:
    with pytest.raises(ValueError, match="atom_indices.*non-negative"):
        obs = Observation(kind=kind, value=0.0, atom_indices=(index,) + (0,) * (arity - 1))
        _plan(observations=ObservationSet((obs,)))


@pytest.mark.parametrize("kind", ["frequency", "eig_diagonal", "bond_length", "bond_angle"])
@pytest.mark.parametrize("index", [-1, 0.5, True, "1"])
def test_used_data_indices_must_be_nonnegative_integers(kind: str, index: object) -> None:
    with pytest.raises(ValueError, match="data_idx.*non-negative integer"):
        obs = Observation(kind=kind, value=0.0, data_idx=index)
        _plan(observations=ObservationSet((obs,)))


def test_plan_keeps_full_cartesian_mode_and_matrix_boundaries() -> None:
    obs = ObservationSet()
    for index in (0, 5):
        obs = obs.with_frequency(0.0, data_idx=index).with_hessian_eigenvalue(0.0, mode_idx=index)
    for row, col in ((0, 0), (0, 5), (5, 0), (5, 5)):
        obs = obs.with_hessian_element(0.0, row=row, col=col).with_hessian_offdiagonal(0.0, row=row, col=col)
    plan = _plan(observations=obs, molecule=make_diatomic())
    assert plan.observations is obs


def test_geometry_indices_preserve_ordered_and_reversed_extraction() -> None:
    from q2mm.objectives._observables import extract_calc_value, geometry_computed

    obs = (
        ObservationSet()
        .with_bond_length(0.96, data_idx=0)
        .with_bond_length(0.96, data_idx=1)
        .with_bond_length(0.96, atom_indices=(np.int64(1), np.int64(0)), data_idx=99)
        .with_bond_angle(104.5, data_idx=0)
        .with_bond_angle(104.5, atom_indices=(1, 0, 2), data_idx=99)
        .with_bond_angle(104.5, atom_indices=(2, 0, 1), data_idx=99)
    )
    plan = _plan(observations=obs)
    mol = plan.molecules[0]
    computed = geometry_computed(mol, mol.geometry, {"bond_length", "bond_angle"})
    assert [extract_calc_value(computed, ref) for ref in obs.values] == pytest.approx(
        [0.96, 0.96, 0.96, 104.5, 104.5, 104.5]
    )
    assert plan.observations is obs


def test_explicit_torsion_does_not_require_topology_membership() -> None:
    from q2mm.objectives._observables import extract_calc_value, geometry_computed

    mol = make_ethane()
    indices = (2, 3, 4, 5)
    assert indices not in {(t.atom_i, t.atom_j, t.atom_k, t.atom_l) for t in mol.torsions}
    obs = ObservationSet().with_torsion_angle(0.0, atom_indices=indices)
    plan = _plan(observations=obs, molecule=mol)
    computed = geometry_computed(mol, mol.geometry, {"torsion_angle"})
    assert np.isfinite(extract_calc_value(computed, obs.values[0]))
    assert plan.observations is obs


def test_derived_plan_revalidates_indices_against_stable_case_id() -> None:
    from dataclasses import replace

    plan = replace(
        _plan(),
        case_ids=("water", "h2"),
        molecules=(make_water(), make_diatomic()),
        stationary_points=(StationaryPointKind.GROUND_STATE,) * 2,
        observations=ObservationSet().with_frequency(0.0, data_idx=8, case_id="water"),
    )
    assert plan.observations.values[0].data_idx == 8
    bad = ObservationSet().with_frequency(0.0, data_idx=6, case_id="h2")
    with pytest.raises(ValueError, match="h2.*out of range"):
        plan.with_observations(bad)
    reordered = replace(plan, case_ids=("h2", "water"), molecules=plan.molecules[::-1])
    with pytest.raises(ValueError, match="h2.*out of range"):
        reordered.with_observations(bad)


@pytest.mark.parametrize("shift", [-720.0, 0.0, 360.0])
def test_torsion_statistics_use_circular_errors_and_decline_r2(shift: float) -> None:
    refs = np.array([-179.0, 179.0, 10.0, 0.0])
    calc = np.array([179.0, -179.0, 370.0, 180.0]) + shift
    stats = category_stats(refs, calc, kind="torsion_angle")
    expected_errors = np.array([2.0, -2.0, 0.0, -180.0])
    assert stats["n_refs"] == 4
    assert stats["rmsd"] == pytest.approx(np.sqrt(np.mean(expected_errors**2)))
    assert stats["mae"] == pytest.approx(np.mean(np.abs(expected_errors)))
    assert np.isnan(stats["r2"])


@pytest.mark.parametrize("shape", [(12,), (3, 4)])
def test_vectorized_torsion_statistics_match_scalar_residuals(shape: tuple[int, ...]) -> None:
    refs = (
        np.array([-180.0, 180.0, -540.0, 540.0, -360.0, 360.0, -179.0, 179.0, 0.0, 12.5, -1e6 - 0.25, 1e6 + 0.25])
        .reshape(shape)
        .T
    )
    calc = np.zeros_like(refs)
    before = refs.copy()
    expected = np.array([raw_residual("torsion_angle", float(ref), 0.0) for ref in refs.flat])
    stats = category_stats(refs, calc, kind="torsion_angle")
    assert stats["n_refs"] == refs.size
    assert stats["rmsd"] == pytest.approx(np.sqrt(np.mean(expected**2)))
    assert stats["mae"] == pytest.approx(np.mean(np.abs(expected)))
    assert np.isnan(stats["r2"])
    np.testing.assert_array_equal(refs, before)
    np.testing.assert_array_equal(calc, np.zeros_like(refs))


def test_torsion_category_metrics_agree_with_loss_and_skip_zero_weight() -> None:
    obs = (
        ObservationSet()
        .with_torsion_angle(-179.0, atom_indices=(2, 0, 1, 5), weight=1.0)
        .with_torsion_angle(179.0, atom_indices=(2, 0, 1, 5), weight=3.0)
        .with_torsion_angle(0.0, atom_indices=(2, 0, 1, 5), weight=0.0)
        .with_energy(1.0)
        .with_energy(3.0)
    )
    calc = np.array([179.0, -179.0, 180.0, 2.0, 2.0])
    raw = np.array([raw_residual(o.kind, o.value, c) for o, c in zip(obs.values, calc, strict=True)])
    weighted = np.array(
        [weighted_residual(o.kind, o.value, c, o.weight) for o, c in zip(obs.values, calc, strict=True)]
    )
    evaluation = Evaluation(
        total=float(weighted @ weighted),
        data_value=float(weighted @ weighted),
        regularization=0.0,
        calculated=calc,
        raw_residuals=raw,
        weighted_residuals=weighted,
        category_scores={"geometry": 40.0, "energy": 2.0},
    )
    metrics = category_metrics(_plan(observations=obs, molecule=make_ethane()), evaluation)
    assert weighted[0] ** 2 == 4.0
    assert evaluation.total == 42.0
    assert metrics["torsion_angle"]["n_refs"] == 2
    assert metrics["torsion_angle"]["rmsd"] == 2.0
    assert metrics["torsion_angle"]["mae"] == 2.0
    assert np.isnan(metrics["torsion_angle"]["r2"])
    assert metrics["energy"] == {"n_refs": 2, "r2": 0.0, "rmsd": 1.0, "mae": 1.0}


def test_category_stats_preserves_linear_defaults_and_empty_results() -> None:
    refs, calc = np.array([1.0, 3.0]), np.array([2.0, 2.0])
    expected = {"n_refs": 2, "r2": 0.0, "rmsd": 1.0, "mae": 1.0}
    assert category_stats(refs, calc) == expected
    assert category_stats(refs, calc, kind="bond_angle") == expected
    for kind in ("torsion_angle", "frequency"):
        empty = category_stats(np.array([]), np.array([]), kind=kind)
        assert empty["n_refs"] == 0
        assert all(np.isnan(empty[key]) for key in ("r2", "rmsd", "mae"))
