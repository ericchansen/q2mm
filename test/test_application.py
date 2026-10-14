"""Direct tests for the data-independent application-service boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from functools import partial
from typing import Any

import numpy as np
import pytest

from q2mm.application import (
    ApplicationConfigurationError,
    ApplicationEvaluationError,
    ApplicationOptimizationError,
    OptimizationRun,
    OutputExistsError,
    OutputFormatError,
    ResolvedBackendConfiguration,
    ResolvedExecutionConfiguration,
    ResolvedExecutorConfiguration,
    ResolvedOptimizerConfiguration,
    ResolvedWorkflowConfiguration,
    capture_execution_configuration,
    evaluate,
    evaluate_reference,
    optimize,
    problem_fingerprint,
    save,
)
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    Capability,
    EnergyRequest,
    EnergyResult,
    EnergyUnit,
    PreparationRequest,
)
from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import OptimizationProblem, StationaryPointKind, TrainingCase
from q2mm.models.results import OptimizationResult
from q2mm.objectives import create_evaluator
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.catalog import build_optimizer, optimizer_spec
from q2mm.optimizers.scipy_opt import ScipyOptimizer
from q2mm.settings import (
    BackendSettings,
    OptimizationSettings,
    MethodE2Settings,
    ObjectiveSettings,
    ScipySettings,
)
from q2mm.workflows import MethodE2Workflow, SingleStageWorkflow


def _force_field(form: FunctionalForm = FunctionalForm.HARMONIC) -> ForceField:
    return ForceField(
        name="synthetic",
        bonds=(BondParam(("H", "H"), equilibrium=0.75, force_constant=100.0),),
        functional_form=form,
    )


def _problem(*, form: FunctionalForm = FunctionalForm.HARMONIC, ts: bool = False) -> OptimizationProblem:
    molecule = Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]]),
        name="h2",
    )
    force_field = _force_field(form)
    layout = ParameterLayout.from_force_field(force_field)
    return OptimizationProblem(
        cases=(
            TrainingCase(
                case_id="h2",
                molecule=molecule,
                stationary_point=(StationaryPointKind.TRANSITION_STATE if ts else StationaryPointKind.GROUND_STATE),
            ),
        ),
        starting_force_field=force_field,
        layout=layout,
        active_space=ActiveParameterSpace(
            layout=layout,
            baseline=layout.vector(force_field),
            active_indices=np.array([0]),
        ),
        observations=ObservationSet().with_energy(100.75, case_id="h2"),
    )


class _EnergyPrepared(AbstractPreparedBackend):
    def _energy(self, request: EnergyRequest) -> EnergyResult:
        return EnergyResult(
            energy=float(np.sum(request.parameters)),
            unit=EnergyUnit.KCAL_PER_MOL,
            provenance=self.info.provenance,
        )


class _EnergyBackend:
    info = BackendInfo(
        name="synthetic-mm",
        role=BackendRole.MM,
        capabilities=frozenset({Capability.ENERGY}),
        functional_forms=frozenset({"harmonic", "mm3"}),
        provenance=BackendProvenance(backend="synthetic-mm", role=BackendRole.MM),
    )

    def prepare(self, request: PreparationRequest) -> _EnergyPrepared:
        assert request.force_field is not None
        return _EnergyPrepared(
            info=self.info,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )


class _ReferencePrepared(AbstractPreparedBackend):
    def _energy(self, request: object) -> EnergyResult:
        return EnergyResult(
            energy=-1.0,
            unit=EnergyUnit.HARTREE,
            provenance=self.info.provenance,
        )


class _ReferenceBackend:
    info = BackendInfo(
        name="synthetic-reference",
        role=BackendRole.REFERENCE,
        capabilities=frozenset({Capability.ENERGY}),
        provenance=BackendProvenance(backend="synthetic-reference", role=BackendRole.REFERENCE),
    )

    def prepare(self, request: PreparationRequest) -> _ReferencePrepared:
        return _ReferencePrepared(
            info=self.info,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=None,
            layout=None,
        )


def _result(problem: OptimizationProblem, *, gradient_mode: str = "analytical") -> OptimizationResult:
    baseline = problem.active_space.baseline
    return OptimizationResult(
        success=True,
        message="ok",
        initial_score=1.0,
        final_score=1.0,
        n_iterations=1,
        n_evaluations=2,
        n_params=len(problem.layout),
        layout_fingerprint=problem.layout.fingerprint,
        initial_params=baseline,
        final_params=baseline,
        method="synthetic",
        gradient_mode=gradient_mode,
    )


def _configuration() -> ResolvedExecutionConfiguration:
    return ResolvedExecutionConfiguration(
        recipe_id="explicit-v1",
        backend=ResolvedBackendConfiguration(key="x", name="x", role="mm"),
        optimizer=ResolvedOptimizerConfiguration(
            key="x",
            label="x",
            method="x",
            settings={},
            expected_result_gradient_mode="none",
        ),
        workflow=ResolvedWorkflowConfiguration(key="single-stage", settings={}),
        executor=ResolvedExecutorConfiguration(kind="python", gradient_mode="none"),
        n_evals=0,
    )


def _run(problem: OptimizationProblem) -> OptimizationRun:
    return OptimizationRun.from_result(
        problem,
        replace(
            _result(problem, gradient_mode="none"),
            regularization=0.0,
            n_evals=0,
            executor_gradient_mode="none",
        ),
        configuration=_configuration(),
    )


def _factory(problem: OptimizationProblem, backend: Any, settings: ObjectiveSettings) -> Any:
    return partial(create_evaluator, backend=backend, force_field=problem.starting_force_field, settings=settings)


def test_evaluate_matches_direct_executor() -> None:
    problem = _problem()
    backend = _EnergyBackend()
    direct = PythonObjectiveExecutor(
        ObjectivePlan.from_problem(problem),
        backend,
        problem.starting_force_field,
    )
    direct = direct.evaluate(problem.active_space.baseline)

    evaluator = create_evaluator(
        ObjectivePlan.from_problem(problem),
        backend=backend,
        force_field=problem.starting_force_field,
        settings=ObjectiveSettings(),
    )
    result = evaluate(evaluator, problem.active_space.baseline)

    assert result.total == direct.total
    assert np.array_equal(result.calculated, direct.calculated)
    assert np.array_equal(result.raw_residuals, direct.raw_residuals)
    assert result.category_scores == direct.category_scores
    assert evaluator.n_evaluations == 0


def test_evaluator_construction_does_not_load_registered_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()

    def load_backend(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Explicit evaluator construction must not select or load a backend.")

    monkeypatch.setattr("q2mm.backends.registry.load_backend", load_backend)
    evaluator = create_evaluator(
        ObjectivePlan.from_problem(problem),
        backend=_EnergyBackend(),
        force_field=problem.starting_force_field,
        settings=ObjectiveSettings(),
    )
    result = evaluate(evaluator, problem.active_space.baseline)
    assert np.isfinite(result.total)


def test_reference_property_evaluate_and_role_conflicts() -> None:
    molecule = _problem().molecules[0]
    result = evaluate_reference(molecule, backend=_ReferenceBackend(), property="energy")
    assert result.energy == -1.0
    with pytest.raises(ApplicationConfigurationError, match="Unknown property"):
        evaluate_reference(molecule, backend=_ReferenceBackend(), property="unknown")  # type: ignore[arg-type]
    with pytest.raises(ApplicationEvaluationError, match="does not declare capability"):
        evaluate_reference(molecule, backend=_ReferenceBackend(), property="hessian")
    with pytest.raises(ApplicationEvaluationError, match="REFERENCE"):
        evaluate_reference(molecule, backend=_EnergyBackend(), property="energy")


def test_reference_preparation_options_are_forwarded_without_backend_selection() -> None:
    molecule = _problem().molecules[0]
    backend = _ReferenceBackend()
    received = []

    def prepare(request: PreparationRequest) -> _ReferencePrepared:
        received.append(request)
        return _ReferencePrepared(
            info=backend.info, case_id=request.case_id, molecule=request.molecule, force_field=None, layout=None
        )

    backend.prepare = prepare
    result = evaluate_reference(molecule, backend=backend, property="energy", preparation_options={"charge": 1})
    assert result.energy == -1.0
    assert len(received) == 1
    assert received[0].molecule is molecule
    assert received[0].force_field is None
    assert received[0].options == {"charge": 1}


def test_evaluate_preserves_underlying_error_cause() -> None:
    failure = ValueError("invalid scientific vector")

    class FailingEvaluator:
        def evaluate(self, parameters: np.ndarray) -> None:
            raise failure

    with pytest.raises(ApplicationEvaluationError, match="invalid scientific vector") as caught:
        evaluate(FailingEvaluator(), np.array([1.0]))
    assert caught.value.__cause__ is failure


def test_evaluate_rejects_non_evaluation_result() -> None:
    class BadEvaluator:
        def evaluate(self, parameters: np.ndarray) -> dict[str, float]:
            return {"total": 1.0}

    with pytest.raises(ApplicationEvaluationError, match="Evaluation"):
        evaluate(BadEvaluator(), np.array([1.0]))


@pytest.mark.parametrize(
    ("regularization", "n_evals"),
    [(float("nan"), 0), (float("inf"), 0), (-0.01, 0), (True, 0), (0.0, True), (0.0, -1), (0.0, 1.5)],
)
def test_numerical_optimize_rejects_invalid_controls_before_execution(
    regularization: float, n_evals: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid numerical controls entered the workflow.")

    workflow = SingleStageWorkflow()
    monkeypatch.setattr(workflow, "run", forbidden)
    problem = _problem()
    with pytest.raises(ApplicationConfigurationError, match="regularization|n_evals"):
        optimize(
            problem,
            optimizer=ScipyOptimizer(),
            workflow=workflow,
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
            regularization=regularization,
            n_evals=n_evals,
        )


@pytest.mark.parametrize("defect", ["result-type", "layout", "initial-vector", "frozen-vector", "missing-evaluator"])
def test_numerical_optimize_rejects_invalid_full_layout_result(defect: str, monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()
    result: Any = _result(problem)
    expected = "OptimizationResult"
    if defect == "result-type":
        result = {"success": True}
    elif defect == "layout":
        result = replace(result, layout_fingerprint="wrong-layout")
        expected = "parameter layout"
    elif defect == "missing-evaluator":
        expected = "did not create an evaluator"
    else:
        vector = problem.active_space.baseline.copy()
        vector[1] += 1.0
        field = "initial_params" if defect == "initial-vector" else "final_params"
        result = replace(result, **{field: vector})
        expected = "baseline" if defect == "initial-vector" else "frozen"
    workflow = SingleStageWorkflow()
    monkeypatch.setattr(workflow, "run", lambda *args, **kwargs: result)
    with pytest.raises(ApplicationOptimizationError, match=expected):
        optimize(
            problem,
            optimizer=ScipyOptimizer(),
            workflow=workflow,
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
            regularization=0.0,
            n_evals=0,
        )


def test_result_envelope_rejects_disagreement_with_captured_gradient_contract() -> None:
    problem = _problem()
    with pytest.raises(ApplicationOptimizationError, match="gradient"):
        OptimizationRun.from_result(problem, _result(problem), configuration=_configuration())


def test_problem_fingerprint_is_deterministic_and_order_sensitive() -> None:
    problem = _problem()
    assert problem_fingerprint(problem) == problem_fingerprint(problem)
    changed = replace(
        problem,
        observations=ObservationSet().with_energy(100.76, case_id="h2"),
    )
    assert problem_fingerprint(problem) != problem_fingerprint(changed)


@pytest.mark.parametrize(
    ("method", "catalog_key", "as_object"),
    [
        ("L-BFGS-B", "scipy-lbfgsb", False),
        ("Nelder-Mead", "scipy-nm", False),
        ("Powell", "scipy-powell", False),
        ("L-BFGS-B", "scipy-lbfgsb", True),
        ("Nelder-Mead", "scipy-nm", True),
        ("Powell", "scipy-powell", True),
        ("least_squares", "scipy-ls", True),
    ],
)
@pytest.mark.parametrize("mode", ["none", "finite_difference", "analytical"])
def test_scipy_gradient_provenance(
    method: str,
    catalog_key: str,
    mode: str,
    as_object: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    problem = _problem()
    executor = "jax" if mode == "analytical" else "python"
    expected_gradient = "none" if method in ScipyOptimizer.DERIVATIVE_FREE_METHODS else mode
    if method == "least_squares" or (mode == "none" and method == "L-BFGS-B"):
        expected_gradient = "finite_difference"
    expected_step = (
        (1e-3 if mode == "none" or method == "least_squares" else 1e-4)
        if expected_gradient == "finite_difference"
        else None
    )
    backend = _EnergyBackend()
    objective = ObjectiveSettings(executor=executor, gradient_mode=mode)
    make_evaluator = _factory(problem, backend, objective)
    if executor == "jax":

        class _JaxIdentityBackend(_EnergyBackend):
            info = replace(
                _EnergyBackend.info,
                name="jax",
                provenance=BackendProvenance(backend="jax", role=BackendRole.MM),
            )

        backend = _JaxIdentityBackend()

        class _AnalyticalEnergyExecutor(BaseObjectiveExecutor):
            @property
            def gradient_mode(self) -> GradientMode:
                return GradientMode.ANALYTICAL

            def _calculated(self, full_vector: np.ndarray) -> np.ndarray:
                return np.array([np.sum(full_vector)])

            def _data_gradient(self, full_vector: np.ndarray) -> np.ndarray:
                residual = np.sum(full_vector) - problem.observations.values[0].value
                return np.full(full_vector.size, 2.0 * residual)

        make_evaluator = _AnalyticalEnergyExecutor
    optimizer = (
        ScipyOptimizer(ScipySettings(method=method, maxiter=1, verbose=False))
        if as_object
        else build_optimizer(
            ScipySettings.model_validate({**optimizer_spec(catalog_key).settings.model_dump(), "maxiter": 1})
        )
    )
    workflow = SingleStageWorkflow()
    configuration = capture_execution_configuration(
        OptimizationSettings(backend=BackendSettings(key=executor), objective=objective, n_evals=0),
        backend=backend,
        optimizer=optimizer,
        workflow=workflow,
    )
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=make_evaluator,
        regularization=0.0,
        n_evals=0,
    )
    assert isinstance(result, OptimizationResult)
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    assert run.executor_configuration.gradient_mode == mode
    assert run.executor_configuration.fd_step == (1e-4 if mode == "finite_difference" else None)
    assert run.optimizer_configuration.expected_result_gradient_mode == expected_gradient
    assert run.result.gradient_mode == expected_gradient
    assert run.result.fd_step == expected_step
    np.testing.assert_array_equal(run.result.final_params[1:], problem.active_space.baseline[1:])
    saved = save(run, tmp_path / "gradient.frcmod")
    assert saved.manifest_path is not None
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["executor"]["gradient_mode"] == mode
    assert manifest["configuration"]["optimizer"]["expected_result_gradient_mode"] == expected_gradient
    assert manifest["result"]["gradient_mode"] == expected_gradient
    assert manifest["result"]["fd_step"] == expected_step


@pytest.mark.parametrize("method", ["L-BFGS-B", "least_squares"])
@pytest.mark.parametrize("mode", [None, "finite_difference"])
def test_scipy_object_internal_fd_step(method: str, mode: str | None, tmp_path: Path) -> None:
    problem = _problem()
    backend = _EnergyBackend()
    objective = ObjectiveSettings(gradient_mode=mode or "none", fd_step=0.02 if mode is not None else None)
    optimizer = ScipyOptimizer(ScipySettings(method=method, eps=0.03, maxiter=1, verbose=False))
    workflow = SingleStageWorkflow()
    configuration = capture_execution_configuration(
        OptimizationSettings(objective=objective, n_evals=0),
        backend=backend,
        optimizer=optimizer,
        workflow=workflow,
    )
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, objective),
        regularization=0.0,
        n_evals=0,
    )
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    expected_step = 0.03 if method == "least_squares" or mode is None else 0.02
    assert run.executor_configuration.gradient_mode == (mode or "none")
    assert run.executor_configuration.fd_step == (None if mode is None else 0.02)
    assert run.optimizer_configuration.expected_result_gradient_mode == "finite_difference"
    assert run.result.gradient_mode == "finite_difference"
    assert run.result.fd_step == expected_step
    assert run.result.stages[0].fd_step == expected_step
    assert result.executor_gradient_mode == (mode or "none")
    assert result.executor_fd_step == objective.fd_step
    saved = save(run, tmp_path / "internal-fd.frcmod")
    assert saved.manifest_path is not None
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["executor"]["fd_step"] == (None if mode is None else 0.02)
    assert manifest["result"]["fd_step"] == expected_step
    assert manifest["result"]["executor_fd_step"] == objective.fd_step


@pytest.mark.parametrize("step", [1e-4, 0.01, 0.02, 0.04])
def test_explicit_executor_fd_step_is_used_and_persisted(step: float, tmp_path: Path) -> None:
    problem = _problem()
    backend = _EnergyBackend()
    optimizer = ScipyOptimizer(ScipySettings(maxiter=1, eps=0.03, verbose=False))
    workflow = SingleStageWorkflow()
    objective = ObjectiveSettings(gradient_mode="finite_difference", fd_step=step)
    configuration = capture_execution_configuration(
        OptimizationSettings(objective=objective, n_evals=0),
        backend=backend,
        optimizer=optimizer,
        workflow=workflow,
    )
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, objective),
        regularization=0.0,
        n_evals=0,
    )
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    assert run.executor_configuration.fd_step == step
    assert run.result.fd_step == step
    assert run.result.stages[0].fd_step == step
    assert result.executor_gradient_mode == "finite_difference" and result.executor_fd_step == step
    saved = save(run, tmp_path / "executor-fd.frcmod")
    assert saved.manifest_path is not None
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["executor"]["fd_step"] == step
    assert manifest["result"]["fd_step"] == step
    assert manifest["result"]["executor_gradient_mode"] == "finite_difference"
    assert manifest["result"]["executor_fd_step"] == step
    for mismatch in (step * 2, np.nextafter(step, np.inf)):
        captured = replace(configuration, executor=replace(configuration.executor, fd_step=mismatch))
        with pytest.raises(ApplicationOptimizationError, match="Captured executor.*does not match numerical execution"):
            save(OptimizationRun.from_result(problem, result, configuration=captured), tmp_path / "mismatch.frcmod")
        assert not list(tmp_path.glob("mismatch*"))


def test_incomplete_executor_controls_fail_at_numerical_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PythonObjectiveExecutor, "finite_difference_step", property(lambda self: None))
    problem = _problem()
    with pytest.raises(ApplicationOptimizationError, match="executor_fd_step"):
        optimize(
            problem,
            optimizer=ScipyOptimizer(),
            workflow=SingleStageWorkflow(),
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings(gradient_mode="finite_difference")),
            regularization=0.0,
            n_evals=0,
        )


@pytest.mark.parametrize("second_step", [0.02, 0.01])
def test_method_e2_evaluators_preserve_or_reject_actual_executor_controls(
    second_step: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    from test.test_preparation import _generic_problem, _water

    problem = _generic_problem(_water(transition_state=True), point="transition_state")
    problem = replace(problem, observations=ObservationSet().with_energy(100.75, case_id=problem.case_ids[0]))
    monkeypatch.setattr(
        "q2mm.workflows.method_e2._build_round2_observations", lambda *args, **kwargs: problem.observations
    )
    workflow = MethodE2Workflow(MethodE2Settings(negative_fc_threshold=3.0))
    backend = _EnergyBackend()
    optimizer = ScipyOptimizer(ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False))
    objective = ObjectiveSettings(gradient_mode="finite_difference", fd_step=0.02)
    configuration = capture_execution_configuration(
        OptimizationSettings(objective=objective, n_evals=0), backend=backend, optimizer=optimizer, workflow=workflow
    )
    calls = []

    def factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
        calls.append(plan)
        selected = objective if len(calls) == 1 else objective.model_copy(update={"fd_step": second_step})
        return _factory(problem, backend, selected)(plan)

    def run() -> OptimizationResult:
        return optimize(
            problem, optimizer=optimizer, workflow=workflow, make_evaluator=factory, regularization=0.0, n_evals=0
        )

    if second_step != objective.fd_step:
        with pytest.raises(ApplicationOptimizationError, match="Workflow evaluators disagree"):
            run()
        assert len(calls) == 2
    else:
        result = run()
        assert len(calls) == len(result.stages) == 2
        assert result.executor_gradient_mode == "finite_difference" and result.executor_fd_step == 0.02
        assert result.gradient_mode == "none" and result.fd_step is None
        OptimizationRun.from_result(problem, result, configuration=configuration)


@pytest.mark.parametrize("step", [0.0, -0.01, float("nan"), float("inf")])
def test_executor_fd_step_rejects_invalid_values(step: float) -> None:
    with pytest.raises(ValueError, match="fd_step"):
        ObjectiveSettings(gradient_mode="finite_difference", fd_step=step)


@pytest.mark.parametrize("reported_mode", ["none", "finite_difference"])
def test_unknown_optimizer_keeps_explicit_gradient_contract(reported_mode: str) -> None:
    problem = _problem()

    class CustomOptimizer:
        method = "L-BFGS-B"

        def configuration_settings(self) -> dict[str, object]:
            return {"reported_mode": reported_mode}

        def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
            assert evaluator.gradient_mode is GradientMode.NONE
            return _result(problem, gradient_mode=reported_mode)

    def run() -> OptimizationResult:
        return optimize(
            problem,
            optimizer=CustomOptimizer(),
            workflow=SingleStageWorkflow(),
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
            regularization=0.0,
            n_evals=0,
        )

    if reported_mode == "none":
        assert run().gradient_mode == "none"
    else:
        with pytest.raises(ApplicationOptimizationError, match="expected 'none'"):
            run()


@pytest.mark.parametrize("composition", ["subclass", "custom-typed", "nested-multistart", "nested-cycling"])
def test_typed_derivative_free_semantics_preserve_custom_and_subclass_result_contracts(
    composition: str,
    tmp_path: Path,
) -> None:
    from q2mm.optimizers.multistart import MultiStartOptimizer
    from q2mm.optimizers.cycling import CyclingOptimizer
    from q2mm.settings import MultiStartSettings, CyclingSettings

    class ConfiguredSubclass(ScipyOptimizer):
        def configuration_settings(self) -> dict[str, Any]:
            return self.settings.model_dump(mode="json")

    class TypedDelegate:
        def __init__(self, settings: ScipySettings) -> None:
            self.settings = settings
            self.inner = ScipyOptimizer(settings)

        def configuration_settings(self) -> dict[str, Any]:
            return self.settings.model_dump(mode="json")

        def optimize(self, evaluator: ObjectiveEvaluator, space: ActiveParameterSpace) -> OptimizationResult:
            return self.inner.optimize(evaluator, space)

    problem = _problem()
    controls = ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False)
    optimizer: Any = ConfiguredSubclass(controls)
    if composition == "custom-typed":
        optimizer = TypedDelegate(controls)
    elif composition == "nested-multistart":
        optimizer = MultiStartOptimizer(
            MultiStartSettings(optimizer=controls, n_starts=1, verbose=False),
            optimizer=optimizer,
        )
    elif composition == "nested-cycling":
        optimizer = CyclingOptimizer(
            CyclingSettings(
                full_optimizer=MultiStartSettings(optimizer=controls, n_starts=1, verbose=False),
                simplex_optimizer=controls,
                max_cycles=1,
                verbose=False,
            )
        )

    class AnalyticalEnergy(BaseObjectiveExecutor):
        @property
        def gradient_mode(self) -> GradientMode:
            return GradientMode.ANALYTICAL

        def _calculated(self, vector: np.ndarray) -> np.ndarray:
            return np.array([vector.sum()])

        def _data_gradient(self, vector: np.ndarray) -> np.ndarray:
            return np.full(vector.size, 2 * (vector.sum() - 100.75))

    settings = OptimizationSettings(objective=ObjectiveSettings(executor="jax"), n_evals=0)
    workflow = SingleStageWorkflow()
    configuration = capture_execution_configuration(
        settings,
        backend=_EnergyBackend(),
        optimizer=optimizer,
        workflow=workflow,
    )
    assert configuration.optimizer.expected_result_gradient_mode == "none"
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=AnalyticalEnergy,
        regularization=0.0,
        n_evals=0,
    )
    assert result.gradient_mode == "none"
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    saved = save(run, tmp_path / "typed.frcmod")
    assert json.loads(saved.manifest_path.read_text())["result"]["gradient_mode"] == "none"
    with pytest.raises(ApplicationOptimizationError, match="gradient_mode"):
        OptimizationRun.from_result(
            problem,
            replace(result, gradient_mode="analytical"),
            configuration=configuration,
        )


def test_explicit_optimization_materializes_result_and_preserves_frozen_slots() -> None:
    problem = _problem()
    backend = _EnergyBackend()
    optimizer = build_optimizer(
        ScipySettings.model_validate({**optimizer_spec("scipy-nm").settings.model_dump(), "maxiter": 2})
    )
    workflow = SingleStageWorkflow()
    configuration = capture_execution_configuration(
        OptimizationSettings(n_evals=0), backend=backend, optimizer=optimizer, workflow=workflow
    )
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, ObjectiveSettings()),
        regularization=0.0,
        n_evals=0,
    )
    run = OptimizationRun.from_result(problem, result, configuration=configuration)

    assert run.result.n_params == len(problem.layout)
    assert run.result.n_evaluations > 0
    assert run.layout_fingerprint == problem.layout.fingerprint
    assert np.array_equal(run.result.final_params[1:], problem.active_space.baseline[1:])
    assert np.array_equal(problem.layout.vector(run.final_force_field), run.result.final_params)
    assert not run.baseline.flags.writeable


@pytest.mark.parametrize(("regularization", "n_evals"), [(0.0, 0), (0.1, 3)])
def test_numerical_execution_controls_are_retained_and_saved(
    regularization: float,
    n_evals: int,
    tmp_path: Path,
) -> None:
    problem = _problem()
    backend = _EnergyBackend()
    objective = ObjectiveSettings(regularization=regularization)
    optimizer = ScipyOptimizer(ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False))
    workflow = SingleStageWorkflow()
    settings = OptimizationSettings(objective=objective, n_evals=n_evals)
    configuration = capture_execution_configuration(settings, backend=backend, optimizer=optimizer, workflow=workflow)
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, objective),
        regularization=regularization,
        n_evals=n_evals,
    )
    assert result.regularization == regularization and result.n_evals == n_evals
    assert len(result.initial_samples) == len(result.final_samples) == n_evals
    run = OptimizationRun.from_result(problem, result, configuration=configuration)
    saved = save(run, tmp_path / "controlled.frcmod")
    manifest = json.loads(saved.manifest_path.read_text())
    for name in ("regularization", "n_evals"):
        assert manifest["result"][name] == manifest["configuration"][name] == getattr(result, name)


@pytest.mark.parametrize("control,value", [("regularization", 0.1), ("n_evals", 3)])
def test_run_assembly_rejects_false_captures_of_execution_controls(control: str, value: float | int) -> None:
    problem = _problem()
    backend = _EnergyBackend()
    optimizer = ScipyOptimizer(ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False))
    workflow = SingleStageWorkflow()
    configuration = capture_execution_configuration(
        OptimizationSettings(n_evals=0),
        backend=backend,
        optimizer=optimizer,
        workflow=workflow,
    )
    result = optimize(
        problem,
        optimizer=optimizer,
        workflow=workflow,
        make_evaluator=_factory(problem, backend, ObjectiveSettings()),
        regularization=0.0,
        n_evals=0,
    )
    with pytest.raises(ApplicationOptimizationError, match="does not match numerical execution"):
        OptimizationRun.from_result(problem, result, configuration=replace(configuration, **{control: value}))


@pytest.mark.parametrize("endpoint", ["initial_samples", "final_samples"])
@pytest.mark.parametrize("count", [0, 2, 4])
def test_optimization_and_assembly_reject_each_incorrect_endpoint_sample_count(
    endpoint: str,
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    optimizer = ScipyOptimizer(ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False))
    workflow = SingleStageWorkflow()
    result = replace(
        _result(problem, gradient_mode="none"),
        regularization=0.0,
        n_evals=3,
        **{"initial_samples": (1.0,) * 3, "final_samples": (1.0,) * 3, endpoint: (1.0,) * count},
    )
    configuration = replace(_configuration(), n_evals=3)
    with pytest.raises(ApplicationOptimizationError, match="sample counts"):
        OptimizationRun.from_result(problem, result, configuration=configuration)
    monkeypatch.setattr(workflow, "run", lambda *args, **kwargs: result)
    with pytest.raises(ApplicationOptimizationError, match="sample counts"):
        optimize(
            problem,
            optimizer=optimizer,
            workflow=workflow,
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
            regularization=0.0,
            n_evals=3,
        )


def test_run_assembly_rejects_uncaptured_direct_algorithm_controls() -> None:
    problem = _problem()
    with pytest.raises(ApplicationOptimizationError, match="does not match numerical execution"):
        OptimizationRun.from_result(problem, _result(problem, gradient_mode="none"), configuration=_configuration())


@pytest.mark.parametrize("boundary", ["workflow", "factory"])
def test_numerical_boundary_rejects_actual_plan_regularization_disagreement(boundary: str) -> None:
    problem = _problem()
    backend = _EnergyBackend()
    optimizer = ScipyOptimizer(ScipySettings(method="Nelder-Mead", maxiter=0, verbose=False))
    workflow = SingleStageWorkflow()
    factory = _factory(problem, backend, ObjectiveSettings(regularization=0.1))
    if boundary == "workflow":

        class WrongPlanWorkflow:
            def run(self, problem: Any, make_evaluator: Any, optimizer: Any, **kwargs: Any) -> None:
                make_evaluator(ObjectivePlan.from_problem(problem, regularization=0.0))

        workflow = WrongPlanWorkflow()
    else:
        wrong = PythonObjectiveExecutor(ObjectivePlan.from_problem(problem), backend, problem.starting_force_field)

        def factory(plan: ObjectivePlan) -> ObjectiveEvaluator:
            return wrong

    with pytest.raises(ApplicationOptimizationError, match="plan regularization does not match execution"):
        optimize(
            problem,
            optimizer=optimizer,
            workflow=workflow,
            make_evaluator=factory,
            regularization=0.1,
            n_evals=0,
        )


def test_optimization_run_rejects_changed_frozen_slot() -> None:
    problem = _problem()
    result = replace(_result(problem, gradient_mode="none"), regularization=0.0, n_evals=0)
    changed = np.array(result.final_params)
    changed[1] += 1.0
    bad = replace(result, final_params=changed)
    with pytest.raises(ApplicationOptimizationError, match="frozen"):
        OptimizationRun(
            result=bad,
            final_force_field=problem.layout.replace(problem.starting_force_field, changed),
            configuration=_configuration(),
            problem_fingerprint=problem_fingerprint(problem),
            layout_fingerprint=problem.layout.fingerprint,
            input_fingerprints={"problem": problem_fingerprint(problem)},
            active_indices=(0,),
            baseline=problem.active_space.baseline,
        )


@pytest.mark.parametrize(("fraction", "final_value"), [(None, 4000.0), (0.2, 125.0)])
@pytest.mark.parametrize("success", [False, True])
def test_optimize_contains_scipy_single_stage_bound_violations(
    monkeypatch: pytest.MonkeyPatch, fraction: float | None, final_value: float, success: bool
) -> None:
    problem = _problem()
    final = problem.active_space.baseline.copy()
    final[0] = final_value
    result = replace(_result(problem), final_params=final, success=success)
    monkeypatch.setattr(SingleStageWorkflow, "run", lambda *args, **kwargs: result)

    with pytest.raises(ApplicationOptimizationError, match="effective SciPy active bounds"):
        optimize(
            problem,
            optimizer=ScipyOptimizer(ScipySettings(fc_fraction=fraction, verbose=False)),
            workflow=SingleStageWorkflow(),
            make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
            regularization=0.0,
            n_evals=0,
        )


def test_optimize_preserves_explicitly_unbounded_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()
    final = problem.active_space.baseline.copy()
    final[0] = 4000.0
    expected = replace(_result(problem, gradient_mode="finite_difference"), final_params=final)

    def run(self: Any, problem: Any, make_evaluator: Any, *args: Any, **kwargs: Any) -> OptimizationResult:
        make_evaluator(ObjectivePlan.from_problem(problem))
        return expected

    monkeypatch.setattr(SingleStageWorkflow, "run", run)

    result = optimize(
        problem,
        optimizer=ScipyOptimizer(ScipySettings(use_bounds=False, fc_fraction=0.2, verbose=False)),
        workflow=SingleStageWorkflow(),
        make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
        regularization=0.0,
        n_evals=0,
    )
    assert result.regularization == 0.0 and result.n_evals == 0
    np.testing.assert_array_equal(result.final_params, final)


def test_optimize_does_not_apply_initial_fractional_box_to_method_e2(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(ts=True)
    final = problem.active_space.baseline.copy()
    final[0] = 130.0
    expected = replace(_result(problem, gradient_mode="finite_difference"), final_params=final)

    def run(self: Any, problem: Any, make_evaluator: Any, *args: Any, **kwargs: Any) -> OptimizationResult:
        make_evaluator(ObjectivePlan.from_problem(problem))
        return expected

    monkeypatch.setattr(MethodE2Workflow, "run", run)

    result = optimize(
        problem,
        optimizer=ScipyOptimizer(ScipySettings(fc_fraction=0.2, verbose=False)),
        workflow=MethodE2Workflow(MethodE2Settings()),
        make_evaluator=_factory(problem, _EnergyBackend(), ObjectiveSettings()),
        regularization=0.0,
        n_evals=0,
    )
    assert result.regularization == 0.0 and result.n_evals == 0
    np.testing.assert_array_equal(result.final_params, final)


@pytest.mark.parametrize(
    ("form", "extension"),
    [
        (FunctionalForm.MM3, ".fld"),
        (FunctionalForm.MM3, ".prm"),
        (FunctionalForm.HARMONIC, ".frcmod"),
    ],
)
def test_save_semantic_formats(form: FunctionalForm, extension: str, tmp_path: Path) -> None:
    output = tmp_path / f"forcefield{extension}"
    saved = save(_force_field(form), output)
    assert saved.path == output
    assert output.is_file()
    assert saved.manifest_path is None


def test_save_run_manifest_is_deterministic_and_no_overwrite(tmp_path: Path) -> None:
    problem = _problem()
    run = _run(problem)
    first = save(run, tmp_path / "one.frcmod")
    second = save(run, tmp_path / "two.frcmod")
    assert first.manifest_path is not None
    assert second.manifest_path is not None
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    assert b"timestamp" not in first.manifest_path.read_bytes()
    with pytest.raises(OutputExistsError):
        save(run, first.path)


def test_save_preserves_source_template(tmp_path: Path) -> None:
    from q2mm.io.amber import load_amber_frcmod

    template = tmp_path / "template.frcmod"
    save(_force_field(), template)
    template.write_text(f"CUSTOM HEADER\n{template.read_text()}", encoding="utf-8")
    loaded = load_amber_frcmod(template)
    layout = ParameterLayout.from_force_field(loaded)
    vector = layout.vector(loaded)
    vector[0] += 1.0

    output = tmp_path / "updated.frcmod"
    save(layout.replace(loaded, vector), output)

    assert "CUSTOM HEADER" in output.read_text(encoding="utf-8")


def test_resolved_configuration_rejects_secret_fields() -> None:
    with pytest.raises(ApplicationConfigurationError, match="Secret-like"):
        ResolvedBackendConfiguration(
            key="x",
            name="x",
            role="mm",
            details={"api_token": "must-not-serialize"},
        )


def test_save_rejects_incompatible_form_and_cleans_atomic_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(OutputFormatError, match="requires functional form"):
        save(_force_field(FunctionalForm.HARMONIC), tmp_path / "bad.fld")

    problem = _problem()
    run = _run(problem)

    def fail_manifest(path: Path, value: OptimizationRun, format_name: str) -> None:
        path.write_text("partial")
        raise OSError("manifest failed")

    monkeypatch.setattr("q2mm.application.persistence._write_manifest", fail_manifest)
    target = tmp_path / "atomic.frcmod"
    with pytest.raises(Exception, match="manifest failed"):
        save(run, target)
    assert not target.exists()
    assert not Path(f"{target}.manifest.json").exists()
    assert not list(tmp_path.glob(".*q2mm*"))


def test_save_rejects_nonrepresentable_nonbonded_exclusions(tmp_path: Path) -> None:
    force_field = replace(
        _force_field(FunctionalForm.MM3),
        nonbonded_excluded_atom_types=("FE",),
    )
    with pytest.raises(OutputFormatError, match="cannot represent"):
        save(force_field, tmp_path / "excluded.prm")


def test_save_rolls_back_second_atomic_replace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    problem = _problem()
    run = _run(problem)
    target = tmp_path / "replace-failure.frcmod"
    real_replace = os.replace
    calls = 0

    def fail_second(source: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("second replace failed")
        real_replace(source, destination)

    monkeypatch.setattr("q2mm.application.persistence.os.replace", fail_second)
    with pytest.raises(Exception, match="second replace failed"):
        save(run, target)
    assert not target.exists()
    assert not Path(f"{target}.manifest.json").exists()
    assert not list(tmp_path.glob(".*q2mm*"))


def test_save_no_overwrite_resists_concurrent_force_field_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm.application.persistence as persistence

    target = tmp_path / "concurrent.frcmod"
    real_serializer = persistence._serializer("amber_frcmod")

    def serialize_then_compete(force_field: ForceField, temporary: Path) -> Path:
        result = real_serializer(force_field, temporary)
        target.write_text("competing writer\n", encoding="utf-8")
        return result

    monkeypatch.setattr(persistence, "_serializer", lambda _format: serialize_then_compete)
    with pytest.raises(OutputExistsError):
        save(_force_field(), target)
    assert target.read_text(encoding="utf-8") == "competing writer\n"
    assert not list(tmp_path.glob(".*q2mm*"))


def test_save_no_overwrite_resists_concurrent_manifest_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm.application.persistence as persistence

    target = tmp_path / "concurrent-run.frcmod"
    manifest_target = Path(f"{target}.manifest.json")
    real_write_manifest = persistence._write_manifest

    def write_then_compete(path: Path, run: OptimizationRun, format_name: str) -> None:
        real_write_manifest(path, run, format_name)
        manifest_target.write_text("competing writer\n", encoding="utf-8")

    monkeypatch.setattr(persistence, "_write_manifest", write_then_compete)
    with pytest.raises(OutputExistsError):
        save(_run(_problem()), target)
    assert not target.exists()
    assert manifest_target.read_text(encoding="utf-8") == "competing writer\n"
    assert not list(tmp_path.glob(".*q2mm*"))
