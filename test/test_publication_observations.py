from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from q2mm.models.observations import (
    AtomicPartialChargeObservation,
    ChargeUnit,
    DirectElectrostaticPotentialObservation,
    ObservationEnergyUnit,
    ObservationSet,
    ParameterTetherObservation,
    RelativeEnergyObservation,
    ScanCoordinate,
    ScanCoordinateKind,
    ScanEnergyObservation,
    ThermodynamicQuantity,
    observation_payload,
)
from q2mm.models.parameters import ParameterId, ParameterKind, ParameterUnit
from q2mm.objectives.protocols import UnsupportedObservationError


def _parameter_id(field: str = "equilibrium") -> ParameterId:
    return ParameterId(family="bond", identity=("C", "H"), occurrence=0, field=field)


def test_typed_publication_observations_are_frozen_and_path_free() -> None:
    observations = (
        AtomicPartialChargeObservation(value=-0.2, atom_index=1, case_id="case"),
        DirectElectrostaticPotentialObservation(value=0.01, point=(1.0, 2.0, 3.0), case_id="case"),
        ParameterTetherObservation(
            value=1.4,
            parameter_id=_parameter_id(),
            unit=ParameterUnit.ANGSTROM,
            case_id="case",
        ),
    )
    payloads = [observation_payload(observation) for observation in observations]

    assert payloads[0]["unit"] == ChargeUnit.ELEMENTARY_CHARGE.value
    assert payloads[1]["point"] == [1.0, 2.0, 3.0]
    assert payloads[2]["parameter_id"]["field"] == "equilibrium"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        observations[0].value = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: AtomicPartialChargeObservation(value=float("nan"), atom_index=0),
        lambda: AtomicPartialChargeObservation(value=0.0, atom_index=-1),
        lambda: DirectElectrostaticPotentialObservation(value=0.0, point=(0.0, 1.0, float("inf"))),
        lambda: ParameterTetherObservation(
            value=1.0,
            parameter_id=_parameter_id(),
            unit=ParameterUnit.ANGSTROM,
            weight=-1.0,
        ),
    ],
)
def test_typed_observations_reject_nonfinite_or_invalid_values(factory: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()  # type: ignore[operator]


def test_relative_energy_group_requires_explicit_zero_and_consistent_units() -> None:
    observations = ObservationSet().with_relative_energy_group(
        (("zero", 0.0), ("product", 4.184)),
        group_id="reaction",
        reference_case_id="zero",
        unit=ObservationEnergyUnit.KJ_PER_MOL,
        quantity=ThermodynamicQuantity.ENTHALPY,
        weight=2.0,
    )
    assert [observation.case_id for observation in observations.values] == ["zero", "product"]
    assert all(isinstance(observation, RelativeEnergyObservation) for observation in observations.values)

    with pytest.raises(ValueError, match="explicit zero"):
        ObservationSet(
            values=(
                RelativeEnergyObservation(
                    value=1.0,
                    group_id="bad",
                    reference_case_id="zero",
                    unit=ObservationEnergyUnit.KJ_PER_MOL,
                    quantity=ThermodynamicQuantity.ENERGY,
                    case_id="zero",
                ),
            )
        )


def test_scan_group_validates_coordinate_arity_units_and_zero() -> None:
    zero = ScanCoordinate(ScanCoordinateKind.TORSION, (0, 1, 2, 3), 0.0, "degree")
    rotated = ScanCoordinate(ScanCoordinateKind.TORSION, (0, 1, 2, 3), 30.0, "degree")
    observations = ObservationSet().with_scan_energy_group(
        (("scan-0", 0.0, zero), ("scan-30", 2.5, rotated)),
        group_id="cp-rotation",
        reference_case_id="scan-0",
        unit=ObservationEnergyUnit.KCAL_PER_MOL,
    )
    assert all(isinstance(observation, ScanEnergyObservation) for observation in observations.values)
    with pytest.raises(ValueError, match="require 2"):
        ScanCoordinate(ScanCoordinateKind.DISTANCE, (0, 1, 2), 1.0, "angstrom")
    with pytest.raises(ValueError, match="require unit"):
        ScanCoordinate(ScanCoordinateKind.ANGLE, (0, 1, 2), 90.0, "angstrom")


@pytest.fixture(scope="module")
def publication_objective() -> tuple[object, object, object]:
    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.benchmarks.systems import load_system
    from q2mm.models.problem import StationaryPointKind
    from q2mm.objectives.plan import ObjectivePlan

    backend = JaxBackend()
    problem = load_system("ch3f", backend=backend, functional_form="harmonic").problem
    first = problem.molecules[0]
    geometry = np.array(first.geometry)
    geometry[0, 0] += 0.05
    second = first.with_geometry(geometry)
    slot = next(item for item in problem.layout if item.kind is ParameterKind.BOND_EQUILIBRIUM)
    observations = (
        ObservationSet()
        .with_relative_energy_group(
            (("a", 0.0), ("b", 2.0)),
            group_id="pair",
            reference_case_id="a",
            unit=ObservationEnergyUnit.KJ_PER_MOL,
            weight=0.5,
        )
        .with_parameter_tether(
            float(problem.active_space.baseline[slot.index]) + 0.1,
            parameter_id=slot.id,
            unit=slot.unit,
            weight=2.0,
            case_id="a",
        )
    )
    plan = ObjectivePlan(
        case_ids=("a", "b"),
        molecules=(first, second),
        stationary_points=(StationaryPointKind.GROUND_STATE, StationaryPointKind.GROUND_STATE),
        observations=observations,
        layout=problem.layout,
        active_space=problem.active_space,
    )
    return backend, problem, plan


@pytest.mark.jax
def test_relative_energy_and_parameter_tether_python_jax_parity(
    publication_objective: tuple[object, object, object],
) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.protocols import GradientMode
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, plan = publication_objective
    vector = np.asarray(problem.active_space.baseline)  # type: ignore[union-attr]
    python_executor = PythonObjectiveExecutor(  # type: ignore[arg-type,union-attr]
        plan,
        backend,
        problem.starting_force_field,
        gradient_mode=GradientMode.ANALYTICAL,
    )
    jax_executor = JaxObjectiveExecutor(plan, backend, problem.starting_force_field)  # type: ignore[arg-type,union-attr]

    python_evaluation = python_executor.evaluate(vector)
    jax_evaluation = jax_executor.evaluate(vector)
    np.testing.assert_allclose(python_evaluation.calculated, jax_evaluation.calculated, atol=1e-10)
    assert python_evaluation.category_scores.keys() == {"relative_energy", "parameter_tether"}
    python_value, python_gradient = python_executor.value_and_gradient(vector)
    jax_value, jax_gradient = jax_executor.value_and_gradient(vector)
    assert python_value == pytest.approx(jax_value, abs=1e-10)
    np.testing.assert_allclose(python_gradient, jax_gradient, atol=1e-9)


@pytest.mark.jax
def test_relative_enthalpy_is_typed_unsupported(
    publication_objective: tuple[object, object, object],
) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, base_plan = publication_objective
    observations = ObservationSet().with_relative_energy_group(
        (("a", 0.0), ("b", 2.0)),
        group_id="enthalpy-pair",
        reference_case_id="a",
        unit=ObservationEnergyUnit.KJ_PER_MOL,
        quantity=ThermodynamicQuantity.ENTHALPY,
    )
    plan = ObjectivePlan(
        case_ids=base_plan.case_ids,  # type: ignore[union-attr]
        molecules=base_plan.molecules,  # type: ignore[union-attr]
        stationary_points=base_plan.stationary_points,  # type: ignore[union-attr]
        observations=observations,
        layout=base_plan.layout,  # type: ignore[union-attr]
        active_space=base_plan.active_space,  # type: ignore[union-attr]
    )
    with pytest.raises(UnsupportedObservationError, match="relative_enthalpy"):
        PythonObjectiveExecutor(plan, backend, problem.starting_force_field)  # type: ignore[arg-type,union-attr]
    with pytest.raises(UnsupportedObservationError, match="relative_enthalpy"):
        JaxObjectiveExecutor(plan, backend, problem.starting_force_field)  # type: ignore[arg-type,union-attr]


@pytest.mark.jax
@pytest.mark.parametrize("kind", ["atomic_charge", "direct_esp", "scan_energy"])
def test_backend_dependent_publication_observations_fail_typed_without_fallback(
    publication_objective: tuple[object, object, object],
    kind: str,
) -> None:
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor

    backend, problem, base_plan = publication_objective
    if kind == "atomic_charge":
        observations = ObservationSet().with_atomic_partial_charge(0.0, atom_index=0, case_id="a")
    elif kind == "direct_esp":
        observations = ObservationSet().with_direct_electrostatic_potential(
            0.0,
            point=(0.0, 0.0, 0.0),
            case_id="a",
        )
    else:
        coordinate = ScanCoordinate(ScanCoordinateKind.DISTANCE, (0, 1), 1.0, "angstrom")
        observations = ObservationSet().with_scan_energy_group(
            (("a", 0.0, coordinate), ("b", 1.0, coordinate)),
            group_id="scan",
            reference_case_id="a",
            unit=ObservationEnergyUnit.KCAL_PER_MOL,
        )
    plan = ObjectivePlan(
        case_ids=base_plan.case_ids,  # type: ignore[union-attr]
        molecules=base_plan.molecules,  # type: ignore[union-attr]
        stationary_points=base_plan.stationary_points,  # type: ignore[union-attr]
        observations=observations,
        layout=base_plan.layout,  # type: ignore[union-attr]
        active_space=base_plan.active_space,  # type: ignore[union-attr]
    )
    with pytest.raises(UnsupportedObservationError):
        PythonObjectiveExecutor(plan, backend, problem.starting_force_field)  # type: ignore[arg-type,union-attr]
    with pytest.raises(UnsupportedObservationError):
        JaxObjectiveExecutor(plan, backend, problem.starting_force_field)  # type: ignore[arg-type,union-attr]


def test_parameter_tether_rejects_non_equilibrium_slot() -> None:
    from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
    from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
    from q2mm.models.problem import StationaryPointKind
    from q2mm.objectives.plan import ObjectivePlan
    from test._shared import make_water

    force_field = ForceField(
        bonds=(BondParam(("H", "O"), 0.96, 500.0),),
        functional_form=FunctionalForm.HARMONIC,
    )
    layout = ParameterLayout.from_force_field(force_field)
    force_constant = next(slot for slot in layout if slot.kind is ParameterKind.BOND_FORCE_CONSTANT)
    observations = ObservationSet().with_parameter_tether(
        500.0,
        parameter_id=force_constant.id,
        unit=force_constant.unit,
        case_id="water",
    )
    with pytest.raises(ValueError, match="restricted"):
        ObjectivePlan(
            case_ids=("water",),
            molecules=(make_water(),),
            stationary_points=(StationaryPointKind.GROUND_STATE,),
            observations=observations,
            layout=layout,
            active_space=ActiveParameterSpace.all_active(layout, force_field),
        )
