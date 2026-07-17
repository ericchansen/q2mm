from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import q2mm
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    FrequencyRequest,
    FrequencyResult,
    FrequencyUnit,
    PreparationRequest,
)
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterKind, ParameterLayout
from q2mm.models.results import OptimizationResult
from q2mm.objectives.protocols import ObjectiveEvaluator
from q2mm.preparation import (
    MatchedFrequencyObservations,
    PreparationError,
    QFuerzaConfig,
    prepare,
)
from test._shared import make_water


def _water(*, name: str = "water", transition_state: bool = False) -> Molecule:
    hessian = np.eye(9) * 0.1
    if transition_state:
        hessian[0, 0] = -0.2
    return make_water(name=name).with_hessian(hessian)


def _template() -> ForceField:
    return ForceField(
        bonds=(BondParam(("H", "O"), equilibrium=1.2, force_constant=4.0),),
        angles=(AngleParam(("H", "O", "H"), equilibrium=90.0, force_constant=2.0),),
        functional_form=FunctionalForm.HARMONIC,
    )


_PROVENANCE = BackendProvenance(backend="preparation-test", role=BackendRole.MM)
_INFO = BackendInfo(
    name="Preparation test backend",
    role=BackendRole.MM,
    capabilities=frozenset({Capability.FREQUENCIES}),
    functional_forms=frozenset({"harmonic"}),
    provenance=_PROVENANCE,
)


class _FrequencyPrepared(AbstractPreparedBackend):
    def _frequencies(self, request: FrequencyRequest) -> FrequencyResult:
        return FrequencyResult(
            frequencies=np.array([-20.0, 10.0, 60.0, 200.0]),
            unit=FrequencyUnit.INVERSE_CM,
            provenance=_PROVENANCE,
        )


class _FrequencyBackend:
    @property
    def info(self) -> BackendInfo:
        return _INFO

    def prepare(self, request: PreparationRequest) -> _FrequencyPrepared:
        assert request.force_field is not None
        return _FrequencyPrepared(
            info=_INFO,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )


class _NoOpOptimizer:
    def optimize(
        self,
        evaluator: ObjectiveEvaluator,
        space: ActiveParameterSpace,
    ) -> OptimizationResult:
        parameters = np.array(space.baseline, copy=True)
        score = evaluator.value(parameters)
        return OptimizationResult(
            success=True,
            message="no-op",
            initial_score=score,
            final_score=score,
            n_iterations=0,
            n_evaluations=1,
            n_params=space.n_full,
            layout_fingerprint=space.layout.fingerprint,
            initial_params=parameters,
            final_params=parameters,
            gradient_mode="none",
        )


def test_fresh_preparation_builds_all_active_immutable_problem() -> None:
    molecule = _water()
    geometry = molecule.geometry.copy()
    hessian = molecule.hessian.copy()

    problem = prepare(molecule, stationary_point="ground_state", functional_form="harmonic")

    assert problem.case_ids == ("0",)
    assert problem.active_space.n_active == len(problem.layout)
    assert problem.starting_force_field.functional_form is FunctionalForm.HARMONIC
    assert problem.preparation_provenance is not None
    assert problem.preparation_provenance.profile == "repository-geometry-eigenmatrix-v1"
    assert problem.preparation_provenance.initialize_source == "qfuerza"
    assert problem.preparation_provenance.qfuerza_settings["zero_torsions"] is True
    assert problem.preparation_provenance.qfuerza_settings["invalid_policy"] == "keep"
    assert problem.preparation_provenance.qfuerza_settings["replace_with"] == 1.0
    np.testing.assert_array_equal(molecule.geometry, geometry)
    np.testing.assert_array_equal(molecule.hessian, hessian)
    assert not problem.active_space.baseline.flags.writeable
    assert not problem.active_space.active_indices.flags.writeable
    assert not problem.cases[0].molecule.geometry.flags.writeable


@pytest.mark.parametrize(
    ("point", "inverted"),
    [("ground_state", False), ("transition_state", True)],
)
def test_stationary_point_solely_routes_qfuerza_inversion(point: str, inverted: bool) -> None:
    molecule = _water(transition_state=inverted)
    from q2mm import preparation

    with patch.object(preparation, "qfuerza_fresh", wraps=preparation.qfuerza_fresh) as spy:
        prepare(molecule, stationary_point=point, functional_form="harmonic")

    assert spy.call_args.kwargs["invert_ts_curvature"] is inverted


def test_qfuerza_config_routes_invalid_policy_and_torsion_choice() -> None:
    from q2mm import preparation

    config = QFuerzaConfig(strategy="fuerza", zero_torsions=False, invalid_policy="skip")
    with patch.object(preparation, "qfuerza_fresh", wraps=preparation.qfuerza_fresh) as spy:
        problem = prepare(
            _water(),
            stationary_point="ground_state",
            functional_form="harmonic",
            qfuerza=config,
        )

    assert spy.call_args.kwargs["strategy"] == "fuerza"
    assert spy.call_args.kwargs["zero_torsions"] is False
    assert spy.call_args.kwargs["invalid_policy"] == "skip"
    assert problem.preparation_provenance is not None
    assert problem.preparation_provenance.qfuerza_settings["strategy"] == "fuerza"


def test_template_provided_preserves_values_and_explicit_observations() -> None:
    force_field = _template()
    layout = ParameterLayout.from_force_field(force_field)
    observations = ObservationSet().with_energy(2.0, case_id="custom")

    problem = prepare(
        make_water(),
        stationary_point="ground_state",
        force_field=force_field,
        initialize="provided",
        observations=observations,
        case_ids=("custom",),
    )

    assert problem.observations is observations
    np.testing.assert_array_equal(layout.vector(problem.starting_force_field), layout.vector(force_field))
    assert problem.preparation_provenance is not None
    assert problem.preparation_provenance.initialize_source == "provided"
    assert not problem.preparation_provenance.qfuerza_settings
    for counts in problem.preparation_provenance.parameter_counts.values():
        assert counts["overwritten"] == 0


def test_template_qfuerza_subset_and_explicit_scalar_space_are_safe() -> None:
    molecule = _water()
    force_field = _template()
    layout = ParameterLayout.from_force_field(force_field)
    baseline = layout.vector(force_field)
    bond_force_constant = next(slot.index for slot in layout if slot.kind is ParameterKind.BOND_FORCE_CONSTANT)
    explicit = ActiveParameterSpace(
        layout=layout,
        baseline=baseline,
        active_indices=np.array([bond_force_constant]),
    )

    scalar_problem = prepare(
        molecule,
        stationary_point="ground_state",
        force_field=force_field,
        active_parameters=explicit,
        initialize="qfuerza",
    )
    scalar_vector = layout.vector(scalar_problem.starting_force_field)
    inactive = np.setdiff1d(np.arange(len(layout)), np.array([bond_force_constant]))
    np.testing.assert_array_equal(scalar_vector[inactive], baseline[inactive])
    assert scalar_vector[bond_force_constant] != baseline[bond_force_constant]

    subset = ForceField(
        bonds=force_field.bonds,
        functional_form=FunctionalForm.HARMONIC,
    )
    subset_problem = prepare(
        molecule,
        stationary_point="ground_state",
        force_field=force_field,
        active_parameters=subset,
        initialize="qfuerza",
    )
    assert tuple(subset_problem.active_space.kinds) == (
        ParameterKind.BOND_FORCE_CONSTANT,
        ParameterKind.BOND_EQUILIBRIUM,
    )


def test_multi_molecule_template_and_case_id_rules() -> None:
    molecules = (_water(name="a"), _water(name="b"))
    problem = prepare(
        molecules,
        stationary_point="transition_state",
        force_field=_template(),
        initialize="qfuerza",
    )
    assert problem.case_ids == ("0", "1")
    assert problem.preparation_provenance is not None
    assert problem.preparation_provenance.stationary_points == (
        "transition_state",
        "transition_state",
    )

    with pytest.raises(PreparationError, match="shared force_field"):
        prepare(molecules, stationary_point="ground_state", functional_form="harmonic")
    with pytest.raises(PreparationError, match="require explicit case_ids"):
        prepare(
            molecules,
            stationary_point="ground_state",
            force_field=_template(),
            initialize="provided",
            observations=ObservationSet().with_energy(1.0),
        )


def test_automatic_observations_match_domain_factory() -> None:
    molecule = _water()
    problem = prepare(
        molecule,
        stationary_point="ground_state",
        force_field=_template(),
        initialize="provided",
    )
    expected = ObservationSet.from_molecules((molecule,), ("0",), eigenmatrix_diagonal_only=False)
    assert problem.observations == expected


def test_matched_frequency_recipe_reproduces_sorted_real_mode_matching() -> None:
    recipe = MatchedFrequencyObservations(
        qm_frequencies=np.array([300.0, -100.0, 20.0, 100.0]),
        backend=_FrequencyBackend(),
    )
    problem = prepare(
        make_water(),
        stationary_point="ground_state",
        force_field=_template(),
        initialize="provided",
        observations=recipe,
        case_ids=("frequency-case",),
    )

    assert [(value.value, value.data_idx, value.weight, value.case_id) for value in problem.observations.values] == [
        (100.0, 2, 0.001, "frequency-case"),
        (300.0, 3, 0.001, "frequency-case"),
    ]
    provenance = problem.preparation_provenance
    assert provenance is not None
    assert provenance.profile == "matched-frequency-v1"
    assert provenance.observation_recipe["backend"]["key"] == "preparation-test"


def test_preparation_validation_rejects_ambiguous_or_conflicting_requests() -> None:
    molecule = _water()
    with pytest.raises(PreparationError, match="mixed"):
        prepare(molecule, stationary_point=["ground_state"], functional_form="harmonic")  # type: ignore[arg-type]
    with pytest.raises(PreparationError, match="conflicts"):
        prepare(
            molecule,
            stationary_point="ground_state",
            force_field=_template(),
            functional_form="mm3",
            initialize="provided",
        )
    with pytest.raises(PreparationError, match="requires initialize"):
        prepare(molecule, stationary_point="ground_state", force_field=_template())
    with pytest.raises(PreparationError, match="cannot be applied"):
        prepare(
            molecule,
            stationary_point="ground_state",
            force_field=_template(),
            initialize="provided",
            qfuerza=QFuerzaConfig(),
        )
    other_force_field = ForceField(
        bonds=_template().bonds,
        functional_form=FunctionalForm.HARMONIC,
    )
    other_layout = ParameterLayout.from_force_field(other_force_field)
    mismatched_space = ActiveParameterSpace.all_active(other_layout, other_force_field)
    with pytest.raises(PreparationError, match="does not match"):
        prepare(
            molecule,
            stationary_point="ground_state",
            force_field=_template(),
            active_parameters=mismatched_space,
            initialize="qfuerza",
        )
    with pytest.raises(PreparationError, match="positive and finite"):
        QFuerzaConfig(replace_with=0.0)


def test_preparation_audit_fingerprints_are_deterministic() -> None:
    molecule = _water()
    first = prepare(molecule, stationary_point="ground_state", functional_form="harmonic")
    second = prepare(molecule, stationary_point="ground_state", functional_form="harmonic")
    assert first.preparation_provenance == second.preparation_provenance
    assert first.preparation_provenance is not None
    assert first.preparation_provenance.pre_qfuerza_vector_fingerprint.startswith("sha256:")
    assert all(value.startswith("sha256:") for value in first.preparation_provenance.input_fingerprints.values())


def test_optimization_run_records_preparation_audit_fingerprint() -> None:
    problem = prepare(
        make_water(),
        stationary_point="ground_state",
        force_field=_template(),
        initialize="provided",
        observations=MatchedFrequencyObservations(
            qm_frequencies=(100.0,),
            backend=_FrequencyBackend(),
        ),
    )

    run = q2mm.optimize(
        problem,
        backend=_FrequencyBackend(),
        recipe="explicit",
        optimizer=_NoOpOptimizer(),
        workflow="single-stage",
        executor="python",
        n_evals=0,
    )

    assert run.provenance["preparation"]["profile"] == "matched-frequency-v1"
    assert run.provenance["preparation_fingerprint"].startswith("sha256:")
