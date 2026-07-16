"""Direct validation tests for :class:`q2mm.objectives.plan.ObjectivePlan`.

Covers immutability, case-ID/layout validation, category derivation, and
the derived-plan builders — independent of any backend.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import KIND_TO_CATEGORY, ObjectivePlan
from q2mm.objectives.protocols import Evaluation
from test._shared import make_water


def _ff() -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=60.0, equilibrium=104.5)],
        functional_form=FunctionalForm.MM3,
    )


def _plan(*, observations: ObservationSet | None = None) -> ObjectivePlan:
    ff = _ff()
    mol = make_water()
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
