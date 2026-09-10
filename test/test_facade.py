from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

import q2mm
from q2mm.models.forcefield import FunctionalForm
from test._shared import make_water


def test_root_all_is_exact() -> None:
    assert q2mm.__all__ == [
        "__version__",
        "Molecule",
        "ForceField",
        "OptimizationProblem",
        "Evaluation",
        "OptimizationResult",
        "OptimizationRun",
        "prepare",
        "evaluate",
        "optimize",
        "save",
    ]


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
    molecule = make_water().with_hessian(np.eye(9) * 0.1)
    geometry = molecule.geometry.copy()

    problem = q2mm.prepare(
        molecule,
        stationary_point="ground_state",
        functional_form="harmonic",
    )

    assert isinstance(problem, q2mm.OptimizationProblem)
    assert problem.starting_force_field.functional_form is FunctionalForm.HARMONIC
    np.testing.assert_array_equal(molecule.geometry, geometry)
