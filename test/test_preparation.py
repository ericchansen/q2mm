from __future__ import annotations

import dataclasses
import json
from pathlib import Path
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
    MoleculeObservations,
    PreparationError,
    QFuerzaConfig,
    StationaryPointObservations,
    prepare,
)
from test._shared import make_harmonic_diatomic, make_harmonic_water, make_water


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
    assert problem.preparation_provenance.profile == "stationary-point-geometry-eigenmatrix-v1"
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
def test_declared_stationary_point_routes_qfuerza_inversion(point: str, inverted: bool) -> None:
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


def test_explicit_compatibility_observations_match_domain_factory() -> None:
    molecule = _water()
    problem = prepare(
        molecule,
        stationary_point="ground_state",
        force_field=_template(),
        initialize="provided",
        observations=MoleculeObservations(),
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


def _generic_problem(
    molecule: Molecule,
    *,
    point: str = "ground_state",
    recipe: StationaryPointObservations | MoleculeObservations | None = None,
) -> q2mm.OptimizationProblem:
    return prepare(
        molecule,
        stationary_point=point,
        force_field=_template(),
        initialize="provided",
        observations=recipe,
        case_ids=("case",),
    )


def _linear_triatomic() -> Molecule:
    molecule = Molecule(
        symbols=("O", "C", "O"),
        geometry=np.array([[-1.2, 0.0, 0.0], [0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
        bonds=(),
    )
    jacobian = np.array(
        [
            [-1, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, -2, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, -2, 0, 0, 1],
        ],
        dtype=float,
    )
    return molecule.with_hessian(jacobian.T @ np.diag([0.5, 0.5, 0.1, 0.1]) @ jacobian)


@pytest.mark.parametrize(
    ("molecule", "rigid", "retained"),
    [(make_harmonic_diatomic(), 5, 1), (make_harmonic_water(), 6, 3), (_linear_triatomic(), 5, 4)],
    ids=("h2", "water", "linear-triatomic"),
)
def test_ground_state_keeps_all_physical_modes(molecule: Molecule, rigid: int, retained: int) -> None:
    from q2mm.models.hessian import mass_weighted_normal_modes

    problem = _generic_problem(molecule)
    explicit = _generic_problem(molecule, recipe=StationaryPointObservations())
    assert problem.preparation_provenance == explicit.preparation_provenance
    assert problem.observations == explicit.observations
    eigenvalues, _ = mass_weighted_normal_modes(molecule.hessian, molecule.symbols)
    assert np.count_nonzero(eigenvalues > 1e-8) == retained
    diagonals = [o for o in problem.observations.values if o.kind == "eig_diagonal"]
    assert len(diagonals) == 3 * molecule.n_atoms
    assert sum(o.weight == 0 for o in diagonals) == rigid
    assert sum(o.weight == 0.1 for o in diagonals) == retained
    assert sum(o.weight > 0 for o in problem.observations.values if o.kind == "eig_offdiagonal") == (
        retained * (retained - 1) // 2
    )
    details = problem.preparation_provenance.observation_recipe["cases"][0]
    assert details["n_rigid_modes"] == rigid
    assert len(details["retained_mode_indices"]) == retained
    assert details["reaction_mode_index"] is None
    assert details["diagnostics"] == ()
    assert problem.preparation_provenance.observation_recipe["reference_hessian"] == "unmodified"


@pytest.mark.parametrize(
    ("molecule", "old_count", "new_count"),
    [(make_harmonic_diatomic(), 0, 1), (make_harmonic_water(), 2, 3)],
    ids=("h2", "water"),
)
def test_generic_and_compatibility_ground_state_targets_differ_only_in_weights(
    molecule: Molecule, old_count: int, new_count: int
) -> None:
    generic = _generic_problem(molecule)
    compatibility = _generic_problem(molecule, recipe=MoleculeObservations())
    assert compatibility.observations == ObservationSet.from_molecules((molecule,), ("case",))
    for new, old in zip(generic.observations.values, compatibility.observations.values, strict=True):
        assert dataclasses.replace(new, weight=old.weight) == old
    for problem, count in ((compatibility, old_count), (generic, new_count)):
        assert sum(o.weight > 0 for o in problem.observations.values if o.kind == "eig_diagonal") == count
    assert dict(compatibility.preparation_provenance.observation_recipe) == {
        "name": "MoleculeObservations",
        "profile": "repository-geometry-eigenmatrix-v1",
        "geometry": True,
        "eigenmatrix": "full",
    }


@pytest.mark.parametrize("molecule", [make_harmonic_diatomic(), make_harmonic_water(), _linear_triatomic()])
def test_mode_counts_are_rotation_and_translation_independent(molecule: Molecule) -> None:
    direction = np.array([1.0, 2.0, 3.0])
    direction /= np.linalg.norm(direction)
    rotation = 2 * np.outer(direction, direction) - np.eye(3)
    block_rotation = np.kron(np.eye(molecule.n_atoms), rotation)
    transformed = dataclasses.replace(
        molecule,
        geometry=molecule.geometry @ rotation.T + np.array([12.0, -7.0, 3.0]),
        hessian=block_rotation @ molecule.hessian @ block_rotation.T,
    )
    before, after = (_generic_problem(m) for m in (molecule, transformed))
    first = before.preparation_provenance.observation_recipe["cases"][0]
    second = after.preparation_provenance.observation_recipe["cases"][0]
    assert first["resolved_linearity"] == second["resolved_linearity"]
    assert first["n_rigid_modes"] == second["n_rigid_modes"]
    assert len(first["retained_mode_indices"]) == len(second["retained_mode_indices"])
    first_values = [o.value for o in before.observations.values if o.kind == "eig_diagonal" and o.weight > 0]
    second_values = [o.value for o in after.observations.values if o.kind == "eig_diagonal" and o.weight > 0]
    np.testing.assert_allclose(first_values, second_values, atol=1e-12)


@pytest.mark.parametrize("molecule", [make_harmonic_water(), _linear_triatomic()])
def test_ts_masks_reaction_row_and_column_without_modifying_reference(molecule: Molecule) -> None:
    from q2mm.models.hessian import mass_weighted_normal_modes, symbols_to_masses_3n

    eigenvalues, modes = mass_weighted_normal_modes(molecule.hessian, molecule.symbols)
    eigenvalues[-1] = -0.1
    masses = np.sqrt(symbols_to_masses_3n(molecule.symbols))
    hessian = (modes @ np.diag(eigenvalues) @ modes.T) * np.outer(masses, masses)
    molecule = molecule.with_hessian(hessian)
    before = molecule.hessian.copy()
    problem = _generic_problem(molecule, point="transition_state")
    details = problem.preparation_provenance.observation_recipe["cases"][0]
    excluded = set(details["excluded_mode_indices"])
    assert details["reaction_mode_index"] == 0
    assert len(excluded) == details["n_rigid_modes"] + 1
    assert details["diagnostics"] == ()
    for observation in problem.observations.values:
        if observation.kind == "eig_offdiagonal":
            assert (observation.weight == 0) == bool(excluded.intersection(observation.atom_indices))
    np.testing.assert_array_equal(molecule.hessian, before)
    if details["n_rigid_modes"] == 6:
        compatibility = _generic_problem(molecule, point="transition_state", recipe=MoleculeObservations())
        assert compatibility.observations == problem.observations
        from q2mm.application import problem_fingerprint

        assert problem_fingerprint(compatibility) == problem_fingerprint(problem)
        assert compatibility.preparation_provenance.profile != problem.preparation_provenance.profile


def test_near_linearity_threshold_and_override_are_explicit() -> None:
    molecule = _linear_triatomic()
    geometry = molecule.geometry.copy()
    geometry[1, 1] = 1e-7
    molecule = dataclasses.replace(molecule, geometry=geometry)
    automatic = _generic_problem(molecule)
    audit = automatic.preparation_provenance.observation_recipe["cases"][0]
    ratio = audit["linearity_ratio"]
    assert audit["resolved_linearity"] == "nonlinear"
    for tolerance, expected in ((ratio * 0.5, "nonlinear"), (ratio, "linear"), (ratio * 2, "linear")):
        problem = _generic_problem(molecule, recipe=StationaryPointObservations(linearity_tolerance=tolerance))
        assert problem.preparation_provenance.observation_recipe["cases"][0]["resolved_linearity"] == expected
    override = _generic_problem(molecule, recipe=StationaryPointObservations(linearity="linear"))
    overridden = override.preparation_provenance.observation_recipe["cases"][0]
    assert overridden["inferred_linearity"] == "nonlinear"
    assert overridden["resolved_linearity"] == "linear"
    assert overridden["n_rigid_modes"] == 5
    assert override.preparation_provenance != automatic.preparation_provenance


@pytest.mark.parametrize("tolerance", [0.0, -1.0, 1.0, float("nan"), float("inf"), True, "small"])
def test_invalid_linearity_settings_are_rejected(tolerance: object) -> None:
    with pytest.raises(PreparationError, match="linearity_tolerance"):
        StationaryPointObservations(linearity_tolerance=tolerance)
    with pytest.raises(PreparationError, match="linearity must"):
        StationaryPointObservations(linearity="planar")


@pytest.mark.parametrize(
    ("symbols", "geometry", "hessian", "message"),
    [
        ((), np.empty((0, 3)), np.empty((0, 0)), "at least two"),
        (("H",), np.zeros((1, 3)), np.zeros((3, 3)), "at least two"),
        (("H", "H"), np.zeros((2, 3)), np.eye(6), "undefined molecular extent"),
        (("H", "H"), np.array([[0, 0, 0], [float("nan"), 0, 0]]), np.eye(6), "geometry must be finite"),
        (("H", "H"), np.array([[0, 0, 0], [1, 0, 0]]), np.full((6, 6), float("inf")), "finite canonical Hessian"),
        (("H", "H"), np.array([[0, 0, 0], [1, 0, 0]]), np.triu(np.ones((6, 6))), "Hessian must be symmetric"),
        (("X", "H"), np.array([[0, 0, 0], [1, 0, 0]]), np.eye(6), "Unknown element"),
    ],
)
def test_invalid_generic_inputs_fail_before_qfuerza(
    symbols: tuple[str, ...], geometry: np.ndarray, hessian: np.ndarray, message: str
) -> None:
    molecule = Molecule(symbols=symbols, geometry=geometry, hessian=hessian, bonds=())
    with (
        patch("q2mm.preparation.qfuerza_fresh") as initialize,
        pytest.raises(PreparationError, match=message),
    ):
        prepare(molecule, stationary_point="ground_state", functional_form="harmonic", case_ids=("bad",))
    initialize.assert_not_called()


@pytest.mark.parametrize("scale", [1.0, 1e250])
@pytest.mark.parametrize("transpose", [False, True])
def test_materially_asymmetric_hessian_is_rejected(scale: float, transpose: bool) -> None:
    hessian = np.diag(np.arange(1.0, 10.0))
    hessian[7, 8], hessian[8, 7] = 2000.0, 0.01
    hessian = (hessian.T if transpose else hessian) * scale
    molecule = make_harmonic_water().with_hessian(hessian)
    with pytest.raises(PreparationError, match="Hessian must be symmetric"):
        _generic_problem(molecule)


@pytest.mark.parametrize("scale", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize("factor", [0.99, 1.01])
@pytest.mark.parametrize("rotate", [False, True])
def test_hessian_symmetry_tolerance_preserves_reference_and_frame(scale: float, factor: float, rotate: bool) -> None:
    molecule = make_harmonic_water()
    hessian = np.diag(np.arange(1.0, 10.0)) * scale
    limit = 1e-12 + 1e-8 * np.linalg.norm(hessian, ord="fro")
    hessian[7, 8] = factor * limit / np.sqrt(2.0)
    if rotate:
        direction = np.array([1.0, 2.0, 3.0])
        direction /= np.linalg.norm(direction)
        rotation = 2 * np.outer(direction, direction) - np.eye(3)
        block = np.kron(np.eye(molecule.n_atoms), rotation)
        hessian = block @ hessian @ block.T
        molecule = dataclasses.replace(molecule, geometry=molecule.geometry @ rotation.T + 3.0)
    molecule = molecule.with_hessian(hessian)
    if factor > 1:
        with pytest.raises(PreparationError, match="Hessian must be symmetric"):
            _generic_problem(molecule)
    else:
        problem = _generic_problem(molecule)
        np.testing.assert_array_equal(problem.cases[0].molecule.hessian, hessian)
        assert problem.preparation_provenance.observation_recipe["hessian_symmetry"] == {
            "norm": "frobenius",
            "atol": 1e-12,
            "rtol": 1e-8,
        }


def test_explicit_compatibility_does_not_add_hessian_symmetry_validation() -> None:
    molecule = make_harmonic_water().with_hessian(np.triu(np.ones((9, 9))))
    with patch("q2mm.preparation._resolve_linearity", side_effect=AssertionError("generic route")):
        problem = _generic_problem(molecule, recipe=MoleculeObservations())
    assert problem.preparation_provenance.profile == "repository-geometry-eigenmatrix-v1"
    assert "hessian_symmetry" not in problem.preparation_provenance.observation_recipe


def test_nonlinear_diatomic_override_is_rejected() -> None:
    with pytest.raises(PreparationError, match="diatomic cannot"):
        _generic_problem(make_harmonic_diatomic(), recipe=StationaryPointObservations(linearity="nonlinear"))


def test_diatomic_is_linear_even_with_sub_roundoff_tolerance() -> None:
    molecule = make_harmonic_diatomic()
    molecule = dataclasses.replace(molecule, geometry=np.array([[0.1, 0.3, 0.7], [0.9, 1.2, 2.0]]))
    problem = _generic_problem(molecule, recipe=StationaryPointObservations(linearity_tolerance=1e-30))
    assert problem.preparation_provenance.observation_recipe["cases"][0]["n_rigid_modes"] == 5


def test_each_case_resolves_its_own_linearity() -> None:
    problem = prepare(
        (make_harmonic_diatomic(), make_harmonic_water()),
        stationary_point="ground_state",
        force_field=_template(),
        initialize="provided",
        case_ids=("linear", "bent"),
    )
    cases = problem.preparation_provenance.observation_recipe["cases"]
    assert [(c["case_id"], c["n_rigid_modes"]) for c in cases] == [("linear", 5), ("bent", 6)]
    assert [(c["case_id"], len(c["retained_mode_indices"])) for c in cases] == [("linear", 1), ("bent", 3)]


@pytest.mark.parametrize("point", ["ground_state", "transition_state"])
def test_negative_retained_curvature_warns_without_reclassification(
    point: str, caplog: pytest.LogCaptureFixture
) -> None:
    molecule = make_harmonic_water(angle_stiffness=-0.1)
    if point == "transition_state":
        molecule = molecule.with_hessian(-make_harmonic_water().hessian)
    problem = _generic_problem(molecule, point=point)
    details = problem.preparation_provenance.observation_recipe["cases"][0]
    diagnostic = next(d for d in details["diagnostics"] if d["code"] == "negative-retained-curvature")
    assert diagnostic["mode_indices"]
    assert all(value < 0 for value in diagnostic["reference_values"])
    assert problem.cases[0].stationary_point.value == point
    assert details["skip_first"] == (point == "transition_state")
    assert "negative retained reference curvature" in caplog.text
    assert not set(diagnostic["mode_indices"]).intersection(details["excluded_mode_indices"])


def test_nonnegative_reserved_ts_curvature_is_diagnostic_only(caplog: pytest.LogCaptureFixture) -> None:
    problem = _generic_problem(_water(), point="transition_state")
    details = problem.preparation_provenance.observation_recipe["cases"][0]
    assert details["diagnostics"][0]["code"] == "nonnegative-ts-reaction-curvature"
    assert details["reaction_mode_index"] == 0
    assert "reserved reaction curvature is nonnegative" in caplog.text


@pytest.mark.parametrize("key", ["ch3f", "sn2", "ethane-gs", "ethane-ts"])
def test_approximate_qm_references_do_not_require_exact_rigid_zeroes(
    key: str, caplog: pytest.LogCaptureFixture
) -> None:
    from q2mm.io.molecules import load_fchk_molecule
    from q2mm.io.xyz import load_xyz
    from test._shared import CH3F_HESS, CH3F_XYZ, GS_FCHK, SN2_HESSIAN, SN2_XYZ, TS_FCHK

    if key in ("ethane-gs", "ethane-ts"):
        molecule = load_fchk_molecule(GS_FCHK if key == "ethane-gs" else TS_FCHK)
    else:
        xyz, hessian = (CH3F_XYZ, CH3F_HESS) if key == "ch3f" else (SN2_XYZ, SN2_HESSIAN)
        molecule = load_xyz(xyz).with_hessian(np.load(hessian))
    point = "transition_state" if key in ("sn2", "ethane-ts") else "ground_state"
    problem = _generic_problem(molecule, point=point)
    details = problem.preparation_provenance.observation_recipe["cases"][0]
    assert details["n_rigid_modes"] == 6
    assert len(details["retained_mode_indices"]) == 3 * molecule.n_atoms - 6 - int(point == "transition_state")
    assert details["diagnostics"] == ()
    assert "reference curvature" not in caplog.text


def test_soft_ts_reaction_is_reserved_before_rigid_candidates() -> None:
    from q2mm.models.hessian import symbols_to_masses_3n

    molecule = make_water()
    eigenvalues = np.array([-1e-8, 1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 5e-8, 1e-3, 2e-3])
    molecule = molecule.with_hessian(np.diag(eigenvalues * symbols_to_masses_3n(molecule.symbols)))
    problem = _generic_problem(molecule, point="transition_state")
    case = problem.preparation_provenance.observation_recipe["cases"][0]
    assert case["excluded_mode_indices"] == tuple(range(7))
    assert case["retained_mode_indices"] == (7, 8)


def test_explicit_observations_do_not_trigger_generic_classification(caplog: pytest.LogCaptureFixture) -> None:
    molecule = make_harmonic_water(angle_stiffness=-0.1)
    observations = ObservationSet().with_energy(1.0)
    with patch("q2mm.preparation._resolve_linearity", side_effect=AssertionError("generic route")):
        problem = prepare(
            molecule,
            stationary_point="ground_state",
            force_field=_template(),
            initialize="provided",
            observations=observations,
        )
    assert problem.observations is observations
    assert problem.preparation_provenance.profile == "explicit-observation-set-v1"
    assert "reference curvature" not in caplog.text


@pytest.mark.parametrize("angle_stiffness", [0.1, -0.1])
def test_saved_run_retains_generic_profile_masks_and_diagnostics(tmp_path: Path, angle_stiffness: float) -> None:
    from test.test_python_executor import StubBackend, StubPrepared

    molecule = make_harmonic_water(angle_stiffness=angle_stiffness)
    problem = _generic_problem(molecule)
    backend = StubBackend(
        StubPrepared(
            hessian=molecule.hessian,
            minimize_coords=molecule.geometry,
            n_params=len(problem.layout),
        )
    )
    run = q2mm.optimize(
        problem,
        backend=backend,
        recipe="explicit",
        optimizer=_NoOpOptimizer(),
        workflow="single-stage",
        executor="python",
        n_evals=0,
    )
    first = q2mm.save(run, tmp_path / "first.frcmod")
    second = q2mm.save(run, tmp_path / "second.frcmod")
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    manifest = json.loads(first.manifest_path.read_text())
    preparation = manifest["provenance"]["preparation"]
    assert preparation["profile"] == "stationary-point-geometry-eigenmatrix-v1"
    recipe = preparation["observation_recipe"]
    assert recipe["name"] == "StationaryPointObservations"
    assert recipe["hessian_symmetry"] == {"norm": "frobenius", "atol": 1e-12, "rtol": 1e-8}
    assert recipe["linearity_tolerance"] == 1e-8
    assert recipe["cases"][0]["n_rigid_modes"] == 6
    assert len(recipe["cases"][0]["retained_mode_indices"]) == 3
    assert bool(recipe["cases"][0]["diagnostics"]) == (angle_stiffness < 0)
    assert manifest["configuration"]["recipe_id"] == "explicit-v1"
    assert manifest["provenance"]["preparation_fingerprint"] == run.provenance["preparation_fingerprint"]


def test_same_targets_with_different_recipe_settings_change_preparation_fingerprint() -> None:
    from q2mm._canonical import canonical_fingerprint
    from q2mm.application import problem_fingerprint

    molecule = make_harmonic_water()
    first = _generic_problem(molecule)
    second = _generic_problem(molecule, recipe=StationaryPointObservations(linearity_tolerance=1e-6))
    assert first.observations == second.observations
    assert problem_fingerprint(first) == problem_fingerprint(second)
    assert canonical_fingerprint(first.preparation_provenance.observation_recipe) != canonical_fingerprint(
        second.preparation_provenance.observation_recipe
    )


@pytest.mark.jax
@pytest.mark.parametrize("system", ["h2", "water"])
def test_generic_recipe_source_hessian_is_zero_and_changed_stiffness_is_not(system: str) -> None:
    from q2mm.backends.contracts import HessianRequest
    from q2mm.backends.registry import load_backend
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.plan import ObjectivePlan
    from q2mm.objectives.python import PythonObjectiveExecutor
    from test._shared import make_diatomic
    from test.test_jax_executor import _h2_ff, _water_ff

    molecule, force_field = (make_diatomic(), _h2_ff()) if system == "h2" else (make_water(), _water_ff())
    backend = load_backend("jax")
    layout = ParameterLayout.from_force_field(force_field)
    parameters = layout.vector(force_field)
    session = backend.prepare(PreparationRequest(case_id="case", molecule=molecule, force_field=force_field))
    molecule = molecule.with_hessian(session.hessian(HessianRequest(parameters=parameters)).hessian)
    problem = prepare(
        molecule,
        stationary_point="ground_state",
        force_field=force_field,
        initialize="provided",
        case_ids=("case",),
    )
    plan = ObjectivePlan.from_problem(problem)
    python = PythonObjectiveExecutor(plan, backend, force_field)
    jax = JaxObjectiveExecutor(plan, backend, force_field)
    assert python.value(parameters) == pytest.approx(0.0, abs=1e-16)
    assert jax.value(parameters) == pytest.approx(0.0, abs=1e-16)
    slot = next(slot.index for slot in layout if slot.kind is ParameterKind.BOND_FORCE_CONSTANT)
    changed = parameters.copy()
    changed[slot] *= 1.01
    assert python.value(changed) > 1e-16
    assert jax.value(changed) == pytest.approx(python.value(changed), rel=1e-7, abs=1e-16)
