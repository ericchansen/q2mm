from __future__ import annotations

import json
import subprocess
import sys
from importlib import import_module

import numpy as np

import q2mm
from q2mm.models.forcefield import FunctionalForm
from test._shared import make_harmonic_water
from q2mm.settings import PreparationSettings


def test_root_all_is_exact() -> None:
    assert q2mm.__all__ == [
        "__version__",
        "Molecule",
        "ForceField",
        "OptimizationProblem",
        "OptimizationResult",
        "OptimizationRun",
        "Evaluation",
        "ExternalDataRoots",
        "prepare",
        "match_frequencies",
        "evaluate",
        "evaluate_reference",
        "create_evaluator",
        "make_evaluator_factory",
        "capture_execution_configuration",
        "build_optimizer",
        "build_workflow",
        "optimize",
        "save",
    ]


def test_root_exports_are_canonical_objects_not_forwarders() -> None:
    owners = {
        "prepare": "q2mm.preparation",
        "match_frequencies": "q2mm.preparation",
        "evaluate": "q2mm.application.evaluation",
        "evaluate_reference": "q2mm.application.evaluation",
        "create_evaluator": "q2mm.objectives",
        "make_evaluator_factory": "q2mm.workflows.base",
        "capture_execution_configuration": "q2mm.application.configuration",
        "build_optimizer": "q2mm.optimizers.catalog",
        "build_workflow": "q2mm.workflows.base",
        "optimize": "q2mm.application.optimization",
        "save": "q2mm.application.persistence",
        "ExternalDataRoots": "q2mm.benchmarks.systems._paths",
    }
    for name, owner in owners.items():
        assert getattr(q2mm, name) is getattr(import_module(owner), name)
    application = import_module("q2mm.application")
    for name in application.__all__:
        if name in q2mm.__all__:
            assert getattr(application, name) is getattr(q2mm, name)


def test_retired_execution_helpers_are_not_exported() -> None:
    for name in (
        "execute_optimization",
        "resolve_optimization_settings",
        "OptimizationLoop",
        "get_optimizer",
        "available_optimizers",
    ):
        assert not hasattr(q2mm, name)


def test_root_import_is_optional_runtime_and_registry_lazy() -> None:
    code = """
import json
import sys
import q2mm
blocked = {"scipy", "jax", "ase", "qcengine", "openmm"}
print(json.dumps({
    "optional": sorted(name for name in sys.modules if name.split(".")[0] in blocked),
    "registry": "q2mm.backends.registry" in sys.modules,
    "preparation": "q2mm.preparation" in sys.modules,
    "application_execution": any(
        name in sys.modules
        for name in (
            "q2mm.application.evaluation",
            "q2mm.application.optimization",
            "q2mm.application.persistence",
        )
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(completed.stdout)
    assert state == {
        "optional": [],
        "registry": False,
        "preparation": False,
        "application_execution": False,
    }


def test_preparation_module_is_dependency_light() -> None:
    code = """
import json
import sys
import q2mm.preparation
blocked = {"scipy", "jax", "ase", "qcengine", "openmm"}
print(json.dumps({
    "optional": sorted(name for name in sys.modules if name.split(".")[0] in blocked),
    "registry": "q2mm.backends.registry" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"optional": [], "registry": False}


def test_root_prepare_delegates_without_mutating_input() -> None:
    molecule = make_harmonic_water()
    geometry = molecule.geometry.copy()

    problem = q2mm.prepare(
        molecule,
        settings=PreparationSettings(stationary_point="ground_state", functional_form="harmonic"),
    )

    assert isinstance(problem, q2mm.OptimizationProblem)
    assert problem.starting_force_field.functional_form is FunctionalForm.HARMONIC
    assert problem.preparation_provenance.profile == "stationary-point-geometry-eigenmatrix-v1"
    assert sum(o.kind == "eig_diagonal" and o.weight > 0 for o in problem.observations.values) == 3
    np.testing.assert_array_equal(molecule.geometry, geometry)
