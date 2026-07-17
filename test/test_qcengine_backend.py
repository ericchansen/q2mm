"""Contract tests for the optional QCEngine reference backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest

qcengine = pytest.importorskip("qcengine")
pytest.importorskip("qcelemental")
from pydantic import ConfigDict
from qcelemental.models.v2 import AtomicResult, ComputeError, FailedOperation
from qcengine.config import TaskConfig
from qcengine.programs import ProgramHarness

from q2mm.backends import registry
from q2mm.backends.contracts import (
    BackendConfigurationError,
    BackendRole,
    BackendUnavailableError,
    Capability,
    PreparationError,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceFrequencyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
    UnsupportedCapabilityError,
)
from q2mm.backends.reference.qcengine import (
    QCEngineBackend,
    QCEngineEvaluationError,
)
from q2mm.constants import BOHR_TO_ANG
from q2mm.models.molecule import Molecule
from test._conformance import assert_reference_capability_conformance

_PROGRAM = "q2mm-deterministic"
_ENERGY = -1.25
_GRADIENT = np.arange(6, dtype=float).reshape(2, 3) / 100.0
_HESSIAN = np.arange(36, dtype=float).reshape(6, 6) / 1000.0


class _DeterministicHarness(ProgramHarness):
    """In-process QCEngine harness returning deterministic atomic-unit data."""

    _defaults = {
        "name": _PROGRAM,
        "scratch": False,
        "thread_safe": True,
        "thread_parallel": False,
        "node_parallel": False,
        "managed_memory": False,
    }
    model_config = ConfigDict(frozen=False)
    calls: ClassVar[list[tuple[Any, dict[str, object]]]] = []
    mode: ClassVar[str] = "success"

    @staticmethod
    def found(raise_error: bool = False) -> bool:
        return True

    def get_version(self) -> str:
        return "1.2.3"

    def compute(self, input_data: Any, config: TaskConfig) -> AtomicResult | FailedOperation:
        type(self).calls.append(
            (
                input_data.model_copy(deep=True),
                {
                    "ncores": config.ncores,
                    "nnodes": config.nnodes,
                    "memory": config.memory,
                    "retries": config.retries,
                    "cores_per_rank": config.cores_per_rank,
                },
            )
        )
        if type(self).mode == "failed":
            return FailedOperation(
                input_data=input_data.model_dump(),
                error=ComputeError(
                    error_type="deterministic_failure",
                    error_message="structured harness failure",
                ),
            )
        driver = input_data.specification.driver.value
        values: float | np.ndarray
        if driver == "energy":
            values = _ENERGY
        elif driver == "gradient":
            values = _GRADIENT.reshape(-1)
        elif driver == "hessian":
            values = _HESSIAN
        else:  # pragma: no cover - adapter capability gate prevents this
            raise AssertionError(f"unexpected driver {driver}")
        return AtomicResult(
            input_data=input_data,
            molecule=input_data.molecule,
            properties={"return_energy": _ENERGY},
            return_result=values,
            provenance={"creator": _PROGRAM, "version": "1.2.3", "routine": "compute"},
            extras={"deterministic": True},
        )


@pytest.fixture
def registered_harness() -> Iterator[_DeterministicHarness]:
    harness = _DeterministicHarness()
    _DeterministicHarness.calls = []
    _DeterministicHarness.mode = "success"
    if _PROGRAM in qcengine.list_all_programs():
        qcengine.unregister_program(_PROGRAM)
    qcengine.register_program(harness)
    try:
        yield harness
    finally:
        if _PROGRAM in qcengine.list_all_programs():
            qcengine.unregister_program(_PROGRAM)
        registry.refresh()


@pytest.fixture
def charged_doublet() -> Molecule:
    return Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.1, -0.2, 0.3], [0.4, 0.5, 1.0]]),
        charge=1,
        multiplicity=2,
        bonds=(),
        angles=(),
        torsions=(),
    )


def _backend() -> QCEngineBackend:
    return QCEngineBackend(
        program=_PROGRAM,
        method="deterministic-method",
        basis="deterministic-basis",
        keywords={"scf_type": "df", "nested": {"levels": [1, 2]}},
        protocols={"stdout": False},
        task_config={"ncores": 2, "nnodes": 1, "memory": 1.5, "retries": 0, "cores_per_rank": 1},
    )


def test_atomic_drivers_inputs_reuse_and_provenance(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
) -> None:
    keywords: dict[str, object] = {"scf_type": "df", "nested": {"levels": [1, 2]}}
    protocols: dict[str, object] = {"stdout": False}
    task_config: dict[str, object] = {"ncores": 2, "memory": 1.5, "retries": 0}
    backend = QCEngineBackend(
        program=_PROGRAM,
        method="deterministic-method",
        basis="deterministic-basis",
        keywords=keywords,
        protocols=protocols,
        task_config=task_config,
    )
    prepared = backend.prepare(PreparationRequest(case_id="charged-doublet", molecule=charged_doublet))
    assert not registered_harness.calls
    original_geometry = charged_doublet.geometry.copy()

    keywords["scf_type"] = "mutated"
    keywords["nested"]["levels"].append(3)  # type: ignore[index,union-attr]
    protocols["stdout"] = True
    task_config["ncores"] = 99

    energy = prepared.energy(ReferenceEnergyRequest())
    gradient = prepared.coordinate_gradient(ReferenceCoordinateGradientRequest())
    hessian = prepared.hessian(ReferenceHessianRequest())
    repeated = prepared.energy(ReferenceEnergyRequest())

    assert energy.energy == repeated.energy == _ENERGY
    np.testing.assert_array_equal(gradient.gradient, _GRADIENT)
    np.testing.assert_array_equal(hessian.hessian, _HESSIAN)
    assert not gradient.gradient.flags.writeable
    assert not hessian.hessian.flags.writeable
    np.testing.assert_array_equal(charged_doublet.geometry, original_geometry)
    assert [call[0].specification.driver.value for call in registered_harness.calls] == [
        "energy",
        "gradient",
        "hessian",
        "energy",
    ]

    expected_geometry = charged_doublet.geometry / BOHR_TO_ANG
    for atomic_input, config in registered_harness.calls:
        assert atomic_input.schema_name == "qcschema_atomic_input"
        assert atomic_input.schema_version == 2
        assert atomic_input.id == "charged-doublet"
        assert tuple(atomic_input.molecule.symbols) == charged_doublet.symbols
        np.testing.assert_allclose(atomic_input.molecule.geometry, expected_geometry, rtol=0.0, atol=5e-9)
        assert atomic_input.molecule.molecular_charge == 1
        assert atomic_input.molecule.molecular_multiplicity == 2
        assert atomic_input.molecule.fix_com is True
        assert atomic_input.molecule.fix_orientation is True
        assert atomic_input.specification.program == _PROGRAM
        assert atomic_input.specification.model.method == "deterministic-method"
        assert atomic_input.specification.model.basis == "deterministic-basis"
        assert atomic_input.specification.keywords == {"scf_type": "df", "nested": {"levels": [1, 2]}}
        assert atomic_input.specification.protocols.stdout is False
        assert atomic_input.specification.extras == {}
        assert config["ncores"] == 2
        assert config["memory"] == 1.5
        assert config["retries"] == 0

    assert energy.provenance.backend == "qcengine"
    assert energy.provenance.role is BackendRole.REFERENCE
    details = energy.provenance.details
    assert details["adapter"]["backend"] == "qcengine"
    assert details["implementation"]["version"] == qcengine.__version__
    assert details["qcelemental"]["version"]
    assert details["program"] == {"name": _PROGRAM, "version": "1.2.3"}
    assert details["driver"] == {"property": "energy", "qcschema_driver": "energy"}
    assert details["schema"]["input_version"] == 2
    assert details["native_provenance"]["creator"] == _PROGRAM
    serialized = json.dumps(details)
    assert "hostname" not in serialized
    assert "username" not in serialized
    with pytest.raises(TypeError):
        details["program"]["name"] = "mutated"  # type: ignore[index]
    assert gradient.provenance.details["driver"] == {
        "property": "coordinate_gradient",
        "qcschema_driver": "gradient",
    }
    assert hessian.provenance.details["driver"]["property"] == "hessian"


def test_unsupported_operations_gate_before_harness(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
) -> None:
    prepared = _backend().prepare(PreparationRequest(case_id="gated", molecule=charged_doublet))
    with pytest.raises(UnsupportedCapabilityError):
        prepared.frequencies(ReferenceFrequencyRequest())
    with pytest.raises(UnsupportedCapabilityError):
        prepared.optimize_geometry(ReferenceGeometryOptimizationRequest())
    assert not registered_harness.calls


def test_force_field_and_options_are_rejected_at_prepare(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
) -> None:
    backend = _backend()
    with pytest.raises(PreparationError, match="force_field"):
        backend.prepare(
            PreparationRequest(
                case_id="force-field",
                molecule=charged_doublet,
                force_field=object(),  # type: ignore[arg-type]
            )
        )
    with pytest.raises(PreparationError, match="per-case options"):
        backend.prepare(PreparationRequest(case_id="options", molecule=charged_doublet, options={"x": 1}))
    assert not registered_harness.calls


def test_structured_failed_operation_mapping(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
) -> None:
    _DeterministicHarness.mode = "failed"
    prepared = _backend().prepare(PreparationRequest(case_id="failed", molecule=charged_doublet))
    with pytest.raises(QCEngineEvaluationError) as caught:
        prepared.energy(ReferenceEnergyRequest())
    assert caught.value.error_type == "deterministic_failure"
    assert "structured harness failure" in str(caught.value)


def test_native_exception_is_chained(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm.backends.reference.qcengine as adapter

    backend = _backend()

    def explode(*args: object, **kwargs: object) -> object:
        raise RuntimeError("native explosion")

    monkeypatch.setattr(adapter._qcengine, "compute", explode)
    prepared = backend.prepare(PreparationRequest(case_id="native", molecule=charged_doublet))
    with pytest.raises(QCEngineEvaluationError, match="native explosion") as caught:
        prepared.energy(ReferenceEnergyRequest())
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert caught.value.error_type == "native_exception"


@pytest.mark.parametrize(
    ("method_name", "evaluation_request"),
    [
        ("energy", ReferenceEnergyRequest()),
        ("coordinate_gradient", ReferenceCoordinateGradientRequest()),
        ("hessian", ReferenceHessianRequest()),
    ],
)
def test_malformed_success_result_is_typed(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    evaluation_request: object,
) -> None:
    backend = _backend()
    monkeypatch.setattr(
        backend,
        "_compute",
        lambda **kwargs: (SimpleNamespace(return_result={"bad": "value"}), backend.info.provenance),
    )
    prepared = backend.prepare(PreparationRequest(case_id="malformed", molecule=charged_doublet))
    with pytest.raises(QCEngineEvaluationError) as caught:
        getattr(prepared, method_name)(evaluation_request)
    assert caught.value.error_type == "invalid_result"
    assert isinstance(caught.value.__cause__, (TypeError, ValueError))


def test_missing_library_and_program_are_typed(
    registered_harness: _DeterministicHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import q2mm.backends.reference.qcengine as adapter

    installed_qcengine = adapter._qcengine
    installed_qcelemental = adapter._qcelemental
    monkeypatch.setattr(adapter, "_qcengine", None)
    with pytest.raises(BackendUnavailableError, match="requires both"):
        QCEngineBackend(program=_PROGRAM, method="test")
    monkeypatch.setattr(adapter, "_qcengine", installed_qcengine)
    monkeypatch.setattr(adapter, "_qcelemental", None)
    with pytest.raises(BackendUnavailableError, match="requires both"):
        QCEngineBackend(program=_PROGRAM, method="test")
    monkeypatch.setattr(adapter, "_qcelemental", installed_qcelemental)
    with pytest.raises(BackendUnavailableError, match="not registered or available"):
        QCEngineBackend(program="q2mm-program-does-not-exist", method="test")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"program": "", "method": "m"},
        {"program": _PROGRAM, "method": ""},
        {"program": _PROGRAM, "method": " m"},
        {"program": _PROGRAM, "method": "m", "basis": ""},
        {"program": _PROGRAM, "method": "m", "basis": "C:\\private\\basis"},
        {"program": _PROGRAM, "method": "m", "keywords": []},
        {"program": _PROGRAM, "method": "m", "keywords": {"api_token": "value"}},
        {"program": _PROGRAM, "method": "m", "keywords": {"file": "C:\\private\\basis"}},
        {"program": _PROGRAM, "method": "m", "task_config": {"scratch_directory": "relative"}},
        {"program": _PROGRAM, "method": "m", "task_config": {"ncores": 0}},
        {"program": _PROGRAM, "method": "m", "task_config": {"memory": float("inf")}},
        {"program": _PROGRAM, "method": "m", "task_config": {"retries": -1}},
        {"program": _PROGRAM, "method": "m", "protocols": {"stdout": "false"}},
        {"program": _PROGRAM, "method": "m", "protocols": {"wavefunction": 1}},
        {
            "program": _PROGRAM,
            "method": "m",
            "protocols": {"error_correction": {"policies": {"known_error": 1}}},
        },
    ],
)
def test_invalid_configuration_is_typed(
    registered_harness: _DeterministicHarness,
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(BackendConfigurationError):
        QCEngineBackend(**kwargs)  # type: ignore[arg-type]


def test_descriptor_runtime_and_reference_conformance(
    registered_harness: _DeterministicHarness,
    charged_doublet: Molecule,
) -> None:
    descriptor = registry.get_descriptor("qcengine")
    expected = frozenset(
        {
            Capability.ENERGY,
            Capability.COORDINATE_GRADIENT,
            Capability.HESSIAN,
        }
    )
    assert descriptor.role is BackendRole.REFERENCE
    assert descriptor.capability_ceiling == expected
    assert descriptor.functional_form_ceiling == frozenset()
    assert descriptor.factory == "q2mm.backends.reference.qcengine:QCEngineBackend"
    assert descriptor.probe.modules == ("qcengine", "qcelemental")

    backend = registry.load_backend(
        "qcengine",
        program=_PROGRAM,
        method="deterministic-method",
        basis="deterministic-basis",
    )
    assert backend.info.capabilities == expected
    outcome = assert_reference_capability_conformance(backend, molecule=charged_doublet)
    assert set(outcome.executed) == expected
    assert set(outcome.unsupported_verified) == {
        Capability.FREQUENCIES,
        Capability.GEOMETRY_OPTIMIZATION,
    }


def test_direct_psi4_retirement_gate_remains_open() -> None:
    """Direct Psi4 remains: QCEngine lacks frequencies and geometry optimization."""
    qcengine_caps = registry.get_descriptor("qcengine").capability_ceiling
    psi4_caps = registry.get_descriptor("psi4").capability_ceiling
    assert {Capability.FREQUENCIES, Capability.GEOMETRY_OPTIMIZATION} <= psi4_caps
    assert not ({Capability.FREQUENCIES, Capability.GEOMETRY_OPTIMIZATION} & qcengine_caps)
