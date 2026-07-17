from __future__ import annotations

import numpy as np
import pytest

from q2mm.application import problem_fingerprint
from q2mm.backends.contracts import FrequencyRequest, PreparationRequest
from q2mm.backends.registry import load_backend
from q2mm.benchmarks.systems import load_system
from q2mm.models.forcefield import FunctionalForm
from q2mm.models.hessian import hessian_to_frequencies
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase
from q2mm.models.seminario import qfuerza_fresh

pytestmark = pytest.mark.jax


@pytest.mark.parametrize(
    ("key", "point"),
    [
        ("ch3f", StationaryPointKind.GROUND_STATE),
        ("ch3f-sn2", StationaryPointKind.TRANSITION_STATE),
    ],
)
@pytest.mark.parametrize("form", ["harmonic", "mm3"])
@pytest.mark.parametrize("starting_point", ["published", "qfuerza"])
def test_small_system_prepare_matches_pre_migration_assembly(
    key: str,
    point: StationaryPointKind,
    form: str,
    starting_point: str,
) -> None:
    backend = load_backend("jax")
    migrated = load_system(
        key,
        backend=backend,
        functional_form=form,
        starting_point=starting_point,
    ).problem
    molecule = migrated.molecules[0]
    assert molecule.hessian is not None
    force_field = qfuerza_fresh(
        molecule,
        functional_form=FunctionalForm(form),
        invert_ts_curvature=point is StationaryPointKind.TRANSITION_STATE,
        replace_with=1.0,
    )
    layout = ParameterLayout.from_force_field(force_field)
    space = ActiveParameterSpace.all_active(layout, force_field)
    qm_frequencies = hessian_to_frequencies(molecule.hessian, molecule.symbols, sort=False)
    prepared = backend.prepare(PreparationRequest(case_id=key, molecule=molecule, force_field=force_field))
    mm_frequencies = prepared.frequencies(FrequencyRequest(parameters=layout.vector(force_field))).frequencies
    qm_real = sorted(value for value in qm_frequencies if value > 50.0)
    mm_real_indices = sorted(index for index, value in enumerate(mm_frequencies) if value > 50.0)
    observations = ObservationSet()
    for qm_value, mm_index in zip(qm_real, mm_real_indices, strict=False):
        observations = observations.with_frequency(
            qm_value,
            data_idx=mm_index,
            weight=0.001,
            case_id=key,
        )
    direct = OptimizationProblem(
        cases=(TrainingCase(case_id=key, molecule=molecule, stationary_point=point),),
        starting_force_field=force_field,
        layout=layout,
        active_space=space,
        observations=observations,
    )

    assert problem_fingerprint(migrated) == problem_fingerprint(direct)
    np.testing.assert_array_equal(
        migrated.layout.vector(migrated.starting_force_field),
        layout.vector(force_field),
    )
    assert migrated.observations == observations
