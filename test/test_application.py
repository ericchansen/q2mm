"""Direct tests for the data-independent application-service boundary."""

from __future__ import annotations

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
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor


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
