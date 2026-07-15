"""Tests for immutable training cases and ``OptimizationProblem``.

Covers case-ID uniqueness, observation ``case_id`` resolution validation
(every :class:`~q2mm.models.observations.Observation` must reference an
existing :class:`TrainingCase` by its stable ``case_id`` — never a
positional index), layout/force-field/active-space structural
consistency, and the ``StationaryPointKind`` GS/TS distinction.
"""

from __future__ import annotations

import pytest

from test._shared import make_water

from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase


def _simple_ff() -> ForceField:
    return ForceField(
        bonds=(BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),),
        functional_form=FunctionalForm.HARMONIC,
    )


def _one_case(name: str = "water-000") -> TrainingCase:
    return TrainingCase(case_id=name, molecule=make_water(name=name), stationary_point=StationaryPointKind.GROUND_STATE)


def _problem(
    cases: tuple[TrainingCase, ...] | None = None,
    ff: ForceField | None = None,
    observations: ObservationSet | None = None,
) -> OptimizationProblem:
    ff = ff if ff is not None else _simple_ff()
    layout = ParameterLayout.from_force_field(ff)
    return OptimizationProblem(
        cases=cases if cases is not None else (_one_case(),),
        starting_force_field=ff,
        layout=layout,
        active_space=ActiveParameterSpace.all_active(layout, ff),
        observations=observations if observations is not None else ObservationSet(),
    )


# ---------------------------------------------------------------------------
# StationaryPointKind
# ---------------------------------------------------------------------------


class TestStationaryPointKind:
    def test_has_ground_state_and_transition_state_members(self) -> None:
        assert StationaryPointKind.GROUND_STATE.value == "ground_state"
        assert StationaryPointKind.TRANSITION_STATE.value == "transition_state"

    def test_is_a_str_enum(self) -> None:
        assert StationaryPointKind.GROUND_STATE == "ground_state"


# ---------------------------------------------------------------------------
# TrainingCase
# ---------------------------------------------------------------------------


class TestTrainingCase:
    def test_valid_construction(self) -> None:
        case = TrainingCase(case_id="mol-000", molecule=make_water(), stationary_point=StationaryPointKind.GROUND_STATE)
        assert case.case_id == "mol-000"
        assert case.stationary_point is StationaryPointKind.GROUND_STATE

    def test_empty_case_id_raises(self) -> None:
        with pytest.raises(ValueError, match="case_id"):
            TrainingCase(case_id="", molecule=make_water(), stationary_point=StationaryPointKind.GROUND_STATE)

    def test_non_molecule_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Molecule"):
            TrainingCase(
                case_id="mol-000",
                molecule="not-a-molecule",  # type: ignore[arg-type]
                stationary_point=StationaryPointKind.GROUND_STATE,
            )

    def test_is_frozen(self) -> None:
        case = _one_case()
        with pytest.raises((AttributeError, TypeError)):
            case.case_id = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OptimizationProblem — construction and validation
# ---------------------------------------------------------------------------


class TestOptimizationProblemConstruction:
    def test_uses_identity_equality_for_composite_state(self) -> None:
        first = _problem()
        second = _problem()

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_valid_single_case_problem(self) -> None:
        problem = _problem()
        assert len(problem.cases) == 1
        assert problem.case_ids == ("water-000",)
        assert len(problem.molecules) == 1
        assert problem.molecules[0].name == "water-000"

    def test_valid_multi_case_problem(self) -> None:
        cases = (_one_case("water-000"), _one_case("water-001"), _one_case("water-002"))
        problem = _problem(cases=cases)
        assert problem.case_ids == ("water-000", "water-001", "water-002")
        assert len(problem.molecules) == 3

    def test_requires_at_least_one_case(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            _problem(cases=())

    def test_duplicate_case_ids_raise(self) -> None:
        cases = (_one_case("dup"), _one_case("dup"))
        with pytest.raises(ValueError, match="Duplicate"):
            _problem(cases=cases)

    def test_case_by_id_returns_matching_case(self) -> None:
        cases = (_one_case("a"), _one_case("b"))
        problem = _problem(cases=cases)
        found = problem.case_by_id("b")
        assert found.case_id == "b"

    def test_case_by_id_missing_raises_key_error(self) -> None:
        problem = _problem()
        with pytest.raises(KeyError):
            problem.case_by_id("does-not-exist")

    def test_is_frozen(self) -> None:
        problem = _problem()
        with pytest.raises((AttributeError, TypeError)):
            problem.cases = ()  # type: ignore[misc]


class TestOptimizationProblemObservationValidation:
    def test_observation_matching_case_id_is_accepted(self) -> None:
        ff = _simple_ff()
        ref = ObservationSet().with_energy(-100.0, weight=1.0, case_id="water-000")
        problem = _problem(observations=ref)
        assert len(problem.observations.values) == 1

    def test_observation_unmatched_numeric_case_id_raises(self) -> None:
        # "5" would have been an out-of-range positional index under the
        # old molecule_idx scheme; under case_id resolution it is simply
        # an unmatched identifier (only "water-000" exists).
        ref = ObservationSet().with_energy(-100.0, weight=1.0, case_id="5")
        with pytest.raises(ValueError, match="case_id"):
            _problem(observations=ref)

    def test_observation_negative_looking_case_id_raises(self) -> None:
        # "-1" would have been rejected as a negative molecule_idx before;
        # under case_id resolution it is just another unmatched identifier
        # — asserting the same outcome (rejection) through the new model.
        ref = ObservationSet().with_energy(-100.0, weight=1.0, case_id="-1")
        with pytest.raises(ValueError, match="case_id"):
            _problem(observations=ref)

    def test_multi_case_observation_case_ids_validated_against_case_set(self) -> None:
        cases = (_one_case("a"), _one_case("b"))
        ref = ObservationSet().with_energy(-1.0, weight=1.0, case_id="a").with_energy(-2.0, weight=1.0, case_id="b")
        problem = _problem(cases=cases, observations=ref)
        assert len(problem.observations.values) == 2

        bad_ref = ObservationSet().with_energy(-1.0, weight=1.0, case_id="c")  # only cases a,b exist
        with pytest.raises(ValueError, match="case_id"):
            _problem(cases=cases, observations=bad_ref)


class TestOptimizationProblemStructuralValidation:
    def test_layout_force_field_mismatch_raises(self) -> None:
        """A layout expecting more bonds than the FF has must not validate."""
        bigger_ff = ForceField(
            bonds=(
                BondParam(elements=("H", "O"), force_constant=500.0, equilibrium=0.96),
                BondParam(elements=("O", "O"), force_constant=400.0, equilibrium=1.2),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(bigger_ff)  # expects 2 bonds (4 slots)
        active_space = ActiveParameterSpace.all_active(layout, bigger_ff)
        smaller_ff = _simple_ff()  # only 1 bond — layout.vector() will index out of range
        with pytest.raises(ValueError):
            OptimizationProblem(
                cases=(_one_case(),),
                starting_force_field=smaller_ff,
                layout=layout,
                active_space=active_space,
                observations=ObservationSet(),
            )

    def test_active_space_from_different_layout_raises(self) -> None:
        ff = _simple_ff()
        layout = ParameterLayout.from_force_field(ff)
        other_ff = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        other_layout = ParameterLayout.from_force_field(other_ff)
        mismatched_space = ActiveParameterSpace.all_active(other_layout, other_ff)
        with pytest.raises(ValueError, match="layout"):
            OptimizationProblem(
                cases=(_one_case(),),
                starting_force_field=ff,
                layout=layout,
                active_space=mismatched_space,
                observations=ObservationSet(),
            )

    def test_valid_active_space_over_same_layout_is_accepted(self) -> None:
        ff = _simple_ff()
        layout = ParameterLayout.from_force_field(ff)
        space = ActiveParameterSpace.all_active(layout, ff)
        problem = OptimizationProblem(
            cases=(_one_case(),),
            starting_force_field=ff,
            layout=layout,
            active_space=space,
            observations=ObservationSet(),
        )
        assert problem.active_space is space


class TestOptimizationProblemGsTsMetadata:
    def test_mixed_gs_and_ts_cases_preserved(self) -> None:
        gs_case = TrainingCase(
            case_id="gs-mol", molecule=make_water(name="gs-mol"), stationary_point=StationaryPointKind.GROUND_STATE
        )
        ts_case = TrainingCase(
            case_id="ts-mol",
            molecule=make_water(name="ts-mol"),
            stationary_point=StationaryPointKind.TRANSITION_STATE,
        )
        problem = _problem(cases=(gs_case, ts_case))
        assert problem.case_by_id("gs-mol").stationary_point is StationaryPointKind.GROUND_STATE
        assert problem.case_by_id("ts-mol").stationary_point is StationaryPointKind.TRANSITION_STATE
