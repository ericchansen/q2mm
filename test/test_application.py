"""Direct tests for the data-independent application-service boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
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
    evaluate_problem,
    evaluate_property,
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
from q2mm.objectives._base import BaseObjectiveExecutor
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.protocols import GradientMode, ObjectiveEvaluator
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.catalog import OptimizerSpec
from q2mm.optimizers.scipy_opt import ScipyOptimizer


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
    )


def _run(problem: OptimizationProblem) -> OptimizationRun:
    result = _result(problem, gradient_mode="none")
    return OptimizationRun(
        result=result,
        final_force_field=problem.layout.replace(problem.starting_force_field, result.final_params),
        configuration=_configuration(),
        problem_fingerprint=problem_fingerprint(problem),
        layout_fingerprint=problem.layout.fingerprint,
        input_fingerprints={"problem": problem_fingerprint(problem)},
        active_indices=tuple(int(index) for index in problem.active_space.active_indices),
        baseline=problem.active_space.baseline,
        provenance={"case_ids": list(problem.case_ids)},
    )


def test_problem_evaluate_matches_direct_executor() -> None:
    problem = _problem()
    backend = _EnergyBackend()
    direct = PythonObjectiveExecutor(
        ObjectivePlan.from_problem(problem),
        backend,
        problem.starting_force_field,
    ).evaluate(problem.active_space.baseline)

    result = evaluate_problem(problem, backend)

    assert result.total == direct.total
    assert np.array_equal(result.calculated, direct.calculated)
    assert np.array_equal(result.raw_residuals, direct.raw_residuals)
    assert result.category_scores == direct.category_scores


def test_problem_evaluate_loads_registered_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem()
    received: dict[str, object] = {}

    def load_backend(name: str, **options: object) -> _EnergyBackend:
        received.update({"name": name, **options})
        return _EnergyBackend()

    monkeypatch.setattr("q2mm.backends.registry.load_backend", load_backend)
    result = evaluate_problem(problem, "synthetic", backend_options={"variant": "test"})
    assert np.isfinite(result.total)
    assert received == {"name": "synthetic", "variant": "test"}

    with pytest.raises(ApplicationConfigurationError, match="already-loaded"):
        from q2mm.application import evaluate

        evaluate(problem.molecules[0], "synthetic", property="energy")


def test_property_evaluate_and_typed_conflicts() -> None:
    molecule = _problem().molecules[0]
    result = evaluate_property(molecule, _ReferenceBackend(), property="energy")
    assert result.energy == -1.0
    with pytest.raises(ApplicationConfigurationError, match="not both"):
        evaluate_property(
            molecule,
            _ReferenceBackend(),
            property="energy",
            capability=Capability.HESSIAN,
        )
    with pytest.raises(ApplicationEvaluationError, match="REFERENCE"):
        evaluate_property(molecule, _EnergyBackend(), property="energy")


def test_problem_fingerprint_is_deterministic_and_order_sensitive() -> None:
    problem = _problem()
    assert problem_fingerprint(problem) == problem_fingerprint(problem)
    changed = replace(
        problem,
        observations=ObservationSet().with_energy(100.76, case_id="h2"),
    )
    assert problem_fingerprint(problem) != problem_fingerprint(changed)


def test_recommended_recipe_exact_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(ts=True)

    class _JaxIdentityBackend(_EnergyBackend):
        info = replace(
            _EnergyBackend.info,
            name="jax",
            provenance=BackendProvenance(backend="jax", role=BackendRole.MM),
        )

    def fake_execute(*args: Any, **kwargs: Any) -> tuple[OptimizationResult, ForceField]:
        return _result(problem), problem.starting_force_field

    monkeypatch.setattr("q2mm.application.optimization.execute_optimization", fake_execute)
    run = optimize(problem, _JaxIdentityBackend())

    assert run.configuration.recipe_id == "recommended-jax-ts-v1"
    assert run.optimizer_configuration.settings["ftol"] == 1e-12
    assert run.optimizer_configuration.settings["fc_fraction"] == 0.20
    assert run.optimizer_configuration.settings["eq_fraction"] == 0.05
    assert run.configuration.ratio_tol is None


def test_recommended_rejects_non_jax_mixed_and_unknown_options() -> None:
    problem = _problem()
    with pytest.raises(ApplicationConfigurationError, match="only for the built-in JAX"):
        optimize(problem, _EnergyBackend())
    mixed = replace(
        problem,
        cases=(
            problem.cases[0],
            TrainingCase(
                case_id="h2-ts",
                molecule=problem.molecules[0],
                stationary_point=StationaryPointKind.TRANSITION_STATE,
            ),
        ),
        observations=ObservationSet().with_energy(100.75, case_id="h2").with_energy(100.75, case_id="h2-ts"),
    )

    class _JaxIdentityBackend(_EnergyBackend):
        info = replace(
            _EnergyBackend.info,
            name="jax",
            provenance=BackendProvenance(backend="jax", role=BackendRole.MM),
        )

    with pytest.raises(ApplicationConfigurationError, match="mixed stationary-point"):
        optimize(mixed, _JaxIdentityBackend())
    with pytest.raises(ApplicationConfigurationError, match="Unknown options"):
        optimize(
            problem,
            _EnergyBackend(),
            optimizer="scipy-lbfgsb",
            workflow="single-stage",
            optimizer_options={"bogus": 1},
        )


def test_explicit_optimizer_and_workflow_override_recipe(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem()

    def fake_execute(*args: Any, **kwargs: Any) -> tuple[OptimizationResult, ForceField]:
        return _result(problem, gradient_mode="none"), problem.starting_force_field

    monkeypatch.setattr("q2mm.application.optimization.execute_optimization", fake_execute)
    run = optimize(
        problem,
        _EnergyBackend(),
        optimizer="scipy-nm",
        workflow="single-stage",
    )
    assert run.configuration.recipe_id == "explicit-v1"
    assert run.configuration.overrides == ("optimizer", "workflow")
    assert run.executor_configuration.kind == "python"
    assert run.optimizer_configuration.key == "scipy-nm"


def test_explicit_executor_gradient_conflicts_are_typed() -> None:
    problem = _problem()
    with pytest.raises(ApplicationConfigurationError, match="gradient_mode"):
        optimize(
            problem,
            _EnergyBackend(),
            optimizer="scipy-lbfgsb-fd",
            workflow="single-stage",
            executor="python",
            gradient_mode="none",
        )


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

        def fake_factory(*args: Any, **kwargs: Any) -> type[_AnalyticalEnergyExecutor]:
            assert kwargs["executor"] == "jax"
            return _AnalyticalEnergyExecutor

        monkeypatch.setattr("q2mm.application.optimization.make_evaluator_factory", fake_factory)
    spec = OptimizerSpec(key=catalog_key, label="SciPy", method=method, evaluator=executor, gradient_mode=mode)
    run = optimize(
        problem,
        backend,
        recipe="explicit",
        optimizer=ScipyOptimizer(method=method, maxiter=1, verbose=False) if as_object else spec,
        optimizer_options=None if as_object else {"maxiter": 1},
        workflow="single-stage",
        executor=executor,
        gradient_mode=mode,
        n_evals=0,
    )
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
    run = optimize(
        _problem(),
        _EnergyBackend(),
        recipe="explicit",
        optimizer=ScipyOptimizer(method=method, eps=0.03, maxiter=1, verbose=False),
        workflow="single-stage",
        executor="python",
        gradient_mode=mode,
        **({} if mode is None else {"fd_step": 0.02}),
        n_evals=0,
    )
    expected_step = 0.03 if method == "least_squares" or mode is None else 0.02
    assert run.executor_configuration.gradient_mode == (mode or "none")
    assert run.executor_configuration.fd_step == (None if mode is None else 0.02)
    assert run.optimizer_configuration.expected_result_gradient_mode == "finite_difference"
    assert run.result.gradient_mode == "finite_difference"
    assert run.result.fd_step == expected_step
    assert run.result.stages[0].fd_step == expected_step
    saved = save(run, tmp_path / "internal-fd.frcmod")
    assert saved.manifest_path is not None
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["executor"]["fd_step"] == (None if mode is None else 0.02)
    assert manifest["result"]["fd_step"] == expected_step


@pytest.mark.parametrize("override", [None, 1e-4, 0.03], ids=["omitted", "explicit-default", "explicit-other"])
def test_root_optimize_preserves_fd_step_precedence(override: float | None) -> None:
    import q2mm

    spec = OptimizerSpec(
        key="custom-fd",
        label="Custom FD",
        method="L-BFGS-B",
        evaluator="python",
        gradient_mode="finite_difference",
        fd_step=0.02,
    )
    run = q2mm.optimize(
        _problem(),
        backend=_EnergyBackend(),
        recipe="explicit",
        optimizer=spec,
        optimizer_options={"maxiter": 1},
        workflow="single-stage",
        **({} if override is None else {"fd_step": override}),
        n_evals=0,
    )
    expected = spec.fd_step if override is None else override
    assert run.executor_configuration.fd_step == expected
    assert run.result.fd_step == expected
    assert ("fd_step" in run.configuration.overrides) is (override is not None)


@pytest.mark.parametrize("source", ["spec", "catalog", "object"])
@pytest.mark.parametrize("override", [None, 1e-4, 0.04], ids=["omitted", "explicit-default", "explicit-other"])
def test_executor_fd_step_precedence(source: str, override: float | None, tmp_path: Path) -> None:
    optimizer: str | OptimizerSpec | ScipyOptimizer
    if source == "spec":
        optimizer = OptimizerSpec(
            key="custom-fd",
            label="Custom FD",
            method="L-BFGS-B",
            evaluator="python",
            gradient_mode="finite_difference",
            fd_step=0.02,
        )
    elif source == "catalog":
        optimizer = "scipy-lbfgsb-fd"
    else:
        optimizer = ScipyOptimizer(maxiter=1, verbose=False)
    effective_step = override if override is not None else (0.02 if source == "spec" else 1e-4)
    run = optimize(
        _problem(),
        _EnergyBackend(),
        recipe="explicit",
        optimizer=optimizer,
        optimizer_options=None if source == "object" else {"maxiter": 1},
        workflow="single-stage",
        executor="python",
        gradient_mode="finite_difference",
        **({} if override is None else {"fd_step": override}),
        n_evals=0,
    )
    assert run.executor_configuration.fd_step == effective_step
    assert run.result.fd_step == effective_step
    assert run.result.stages[0].fd_step == effective_step
    assert ("fd_step" in run.configuration.overrides) is (override is not None)
    saved = save(run, tmp_path / "executor-fd.frcmod")
    assert saved.manifest_path is not None
    manifest = json.loads(saved.manifest_path.read_text(encoding="utf-8"))
    assert manifest["configuration"]["executor"]["fd_step"] == effective_step
    assert manifest["result"]["fd_step"] == effective_step


@pytest.mark.parametrize("as_object", [False, True], ids=["catalog", "object"])
@pytest.mark.parametrize("mode", ["none", "analytical"])
@pytest.mark.parametrize("step", [1e-4, 0.02])
def test_explicit_fd_step_rejects_incompatible_modes_before_execution(
    as_object: bool, mode: str, step: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected_execute(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An incompatible FD step must fail before execution.")

    monkeypatch.setattr("q2mm.application.optimization.execute_optimization", unexpected_execute)
    executor = "jax" if mode == "analytical" else "python"
    with pytest.raises(ApplicationConfigurationError, match="fd_step applies only"):
        optimize(
            _problem(),
            _EnergyBackend(),
            recipe="explicit",
            optimizer=ScipyOptimizer(verbose=False)
            if as_object
            else ("scipy-lbfgsb-jax" if mode == "analytical" else "scipy-lbfgsb"),
            workflow="single-stage",
            executor=executor,
            gradient_mode=mode,
            fd_step=step,
            n_evals=0,
        )


@pytest.mark.parametrize("as_object", [False, True], ids=["catalog", "object"])
@pytest.mark.parametrize("step", [0.0, -0.01, float("nan"), float("inf")])
def test_executor_fd_step_rejects_invalid_values(as_object: bool, step: float, monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_execute(*args: Any, **kwargs: Any) -> None:
        pytest.fail("An invalid FD step must fail before execution.")

    monkeypatch.setattr("q2mm.application.optimization.execute_optimization", unexpected_execute)
    with pytest.raises(ApplicationConfigurationError, match="fd_step must be positive and finite"):
        optimize(
            _problem(),
            _EnergyBackend(),
            recipe="explicit",
            optimizer=ScipyOptimizer(verbose=False) if as_object else "scipy-lbfgsb-fd",
            workflow="single-stage",
            executor="python",
            gradient_mode="finite_difference",
            fd_step=step,
            n_evals=0,
        )


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

    def run() -> OptimizationRun:
        return optimize(
            problem,
            _EnergyBackend(),
            recipe="explicit",
            optimizer=CustomOptimizer(),
            workflow="single-stage",
            executor="python",
            n_evals=0,
        )

    if reported_mode == "none":
        assert run().optimizer_configuration.expected_result_gradient_mode == "none"
    else:
        with pytest.raises(ApplicationOptimizationError, match="expected 'none'"):
            run()


def test_explicit_optimization_materializes_result_and_preserves_frozen_slots() -> None:
    problem = _problem()
    run = optimize(
        problem,
        _EnergyBackend(),
        recipe="explicit",
        optimizer="scipy-nm",
        optimizer_options={"maxiter": 2},
        workflow="single-stage",
        n_evals=0,
    )

    assert run.result.n_params == len(problem.layout)
    assert run.result.n_evaluations > 0
    assert run.layout_fingerprint == problem.layout.fingerprint
    assert np.array_equal(run.result.final_params[1:], problem.active_space.baseline[1:])
    assert np.array_equal(problem.layout.vector(run.final_force_field), run.result.final_params)
    assert not run.baseline.flags.writeable


def test_optimization_run_rejects_changed_frozen_slot() -> None:
    problem = _problem()
    result = _result(problem, gradient_mode="none")
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
def test_execute_contains_scipy_single_stage_bound_violations(
    monkeypatch: pytest.MonkeyPatch, fraction: float | None, final_value: float, success: bool
) -> None:
    from q2mm.application.optimization import execute_optimization
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from q2mm.workflows import SingleStageWorkflow

    problem = _problem()
    final = problem.active_space.baseline.copy()
    final[0] = final_value
    result = replace(_result(problem), final_params=final, success=success)
    monkeypatch.setattr(SingleStageWorkflow, "run", lambda *args, **kwargs: result)

    with pytest.raises(ApplicationOptimizationError, match="effective SciPy active bounds"):
        execute_optimization(
            problem,
            _EnergyBackend(),
            ScipyOptimizer(fc_fraction=fraction, verbose=False),
            SingleStageWorkflow(),
            executor="python",
        )


def test_execute_preserves_explicitly_unbounded_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.application.optimization import execute_optimization
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from q2mm.workflows import SingleStageWorkflow

    problem = _problem()
    final = problem.active_space.baseline.copy()
    final[0] = 4000.0
    expected = replace(_result(problem), final_params=final)
    monkeypatch.setattr(SingleStageWorkflow, "run", lambda *args, **kwargs: expected)

    result, force_field = execute_optimization(
        problem,
        _EnergyBackend(),
        ScipyOptimizer(use_bounds=False, fc_fraction=0.2, verbose=False),
        SingleStageWorkflow(),
        executor="python",
    )
    assert result is expected
    np.testing.assert_array_equal(problem.layout.vector(force_field), final)


def test_execute_does_not_apply_initial_fractional_box_to_method_e2(monkeypatch: pytest.MonkeyPatch) -> None:
    from q2mm.application.optimization import execute_optimization
    from q2mm.optimizers.scipy_opt import ScipyOptimizer
    from q2mm.workflows import MethodE2Workflow

    problem = _problem(ts=True)
    final = problem.active_space.baseline.copy()
    final[0] = 130.0
    expected = replace(_result(problem), final_params=final)
    monkeypatch.setattr(MethodE2Workflow, "run", lambda *args, **kwargs: expected)

    result, force_field = execute_optimization(
        problem,
        _EnergyBackend(),
        ScipyOptimizer(fc_fraction=0.2, verbose=False),
        MethodE2Workflow(),
        executor="python",
    )
    assert result is expected
    np.testing.assert_array_equal(problem.layout.vector(force_field), final)


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
