"""Execution contracts that must survive the capability-core rewrite."""

from __future__ import annotations

import numpy as np
import pytest

from test._shared import make_water

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.parameters import ParameterLayout

pytestmark = pytest.mark.jax


def _water_forcefield() -> ForceField:
    return ForceField(
        bonds=[BondParam(("H", "O"), equilibrium=0.96, force_constant=8.0)],
        angles=[AngleParam(("H", "O", "H"), equilibrium=104.5, force_constant=0.7)],
        functional_form=FunctionalForm.MM3,
    )


def _mixed_conformer_objective() -> tuple:
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.models.observations import ObservationSet

    forcefield = _water_forcefield()
    layout = ParameterLayout.from_force_field(forcefield)
    molecules = [
        make_water(angle_deg=100.0, bond_length=0.94, name="water-a"),
        make_water(angle_deg=112.0, bond_length=1.04, name="water-b"),
    ]
    engine = JaxEngine()
    references = ObservationSet()
    per_case: list[tuple[float, float]] = []

    for case_index, molecule in enumerate(molecules):
        handle = engine.create_context(molecule, forcefield)
        energy = engine.energy(handle, forcefield)
        frequencies = engine.frequencies(handle, forcefield)
        frequency = float(frequencies[-1])
        per_case.append((energy, frequency))
        offset = float(case_index + 1)
        references = references.with_energy(energy + offset, case_id=str(case_index))
        references = references.with_frequency(
            frequency + 10.0 * offset,
            data_idx=len(frequencies) - 1,
            case_id=str(case_index),
            weight=0.01,
        )

    objective = ObjectiveFunction(forcefield, engine, molecules, references, layout=layout)
    return objective, forcefield, layout, molecules, per_case


def _independent_case_score(
    forcefield: ForceField,
    molecules: list,
    per_case: list[tuple[float, float]],
) -> float:
    from q2mm.backends.mm.jax_engine import JaxEngine
    from q2mm.optimizers.objective import ObjectiveFunction
    from q2mm.models.observations import ObservationSet

    score = 0.0
    layout = ParameterLayout.from_force_field(forcefield)
    parameters = layout.vector(forcefield)
    for case_index, molecule in enumerate(molecules):
        energy, frequency = per_case[case_index]
        offset = float(case_index + 1)
        reference = ObservationSet()
        reference = reference.with_energy(energy + offset)
        reference = reference.with_frequency(
            frequency + 10.0 * offset,
            data_idx=3 * molecule.n_atoms - 1,
            weight=0.01,
        )
        score += ObjectiveFunction(forcefield, JaxEngine(), [molecule], reference, layout=layout)(parameters)
    return score


def test_same_topology_mixed_objective_preserves_conformer_identity() -> None:
    """Batched Hessians must not make another conformer reuse case state."""
    objective, forcefield, layout, molecules, per_case = _mixed_conformer_objective()
    parameters = layout.vector(forcefield)

    batched_score = objective(parameters)
    independent_score = _independent_case_score(forcefield, molecules, per_case)

    assert independent_score == pytest.approx(5.05, abs=1e-10)
    assert batched_score == pytest.approx(independent_score, rel=1e-10, abs=1e-10)


def test_case_handles_are_distinct_and_reused() -> None:
    """Preparation reuses each case's state without aliasing case identity."""
    objective, forcefield, layout, _molecules, _per_case = _mixed_conformer_objective()
    parameters = layout.vector(forcefield)

    objective(parameters)
    first_handles = {case_index: id(handle) for case_index, handle in objective._handles.items()}
    objective(parameters)
    second_handles = {case_index: id(handle) for case_index, handle in objective._handles.items()}

    assert len(first_handles) == 2
    assert len(set(first_handles.values())) == 2
    assert second_handles == first_handles


def test_objective_evaluation_does_not_mutate_problem_inputs() -> None:
    """Objective execution leaves force-field, parameter, and geometry inputs intact."""
    objective, forcefield, layout, molecules, _per_case = _mixed_conformer_objective()
    parameters = layout.vector(forcefield)
    parameter_snapshot = parameters.copy()
    forcefield_snapshot = layout.vector(forcefield).copy()
    geometry_snapshots = [molecule.geometry.copy() for molecule in molecules]

    objective(parameters)

    np.testing.assert_array_equal(parameters, parameter_snapshot)
    np.testing.assert_array_equal(layout.vector(forcefield), forcefield_snapshot)
    for molecule, geometry in zip(molecules, geometry_snapshots, strict=True):
        np.testing.assert_array_equal(molecule.geometry, geometry)
