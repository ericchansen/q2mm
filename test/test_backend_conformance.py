"""Public, dependency-light backend API-v1 conformance tests."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from q2mm.backends.conformance import (
    ConformanceError,
    ConformanceOutcome,
    MMConformanceCase,
    ReferenceConformanceCase,
    run_mm_conformance,
    run_reference_conformance,
)
from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BackendDescriptor,
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
from q2mm.models.molecule import Bond, Molecule
from q2mm.models.parameters import ParameterLayout


def _inputs() -> tuple[Molecule, ForceField]:
    molecule = Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        bonds=(Bond(0, 1, ("H", "H"), 0.8),),
        angles=(),
        torsions=(),
    )
    force_field = ForceField(
        bonds=(BondParam(("H", "H"), equilibrium=0.75, force_constant=100.0),),
        functional_form=FunctionalForm.HARMONIC,
    )
    return molecule, force_field


def _descriptor(
    name: str,
    role: BackendRole,
    capabilities: frozenset[Capability],
    forms: frozenset[str] = frozenset(),
) -> BackendDescriptor:
    return BackendDescriptor(
        name=name,
        role=role,
        capability_ceiling=capabilities,
        functional_form_ceiling=forms,
        factory="test.test_backend_conformance:_EnergyBackend",
    )


class _EnergyPrepared(AbstractPreparedBackend):
    def _energy(self, request: EnergyRequest) -> EnergyResult:
        return EnergyResult(
            energy=1.0,
            unit=EnergyUnit.KCAL_PER_MOL,
            provenance=self.info.provenance,
        )


class _EnergyBackend:
    def __init__(
        self,
        *,
        name: str = "unit-mm",
        capabilities: frozenset[Capability] = frozenset({Capability.ENERGY}),
    ) -> None:
        provenance = BackendProvenance(
            backend=name,
            role=BackendRole.MM,
            details={"implementation": {"name": "unit"}},
        )
        self._info = BackendInfo(
            name=name,
            role=BackendRole.MM,
            capabilities=capabilities,
            functional_forms=frozenset({"harmonic"}),
            provenance=provenance,
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def prepare(self, request: PreparationRequest) -> _EnergyPrepared:
        assert request.force_field is not None
        return _EnergyPrepared(
            info=self.info,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=request.force_field,
            layout=ParameterLayout.from_force_field(request.force_field),
        )


class _MutatingPrepared(_EnergyPrepared):
    def _energy(self, request: EnergyRequest) -> EnergyResult:
        request.parameters.setflags(write=True)
        request.parameters[0] += 1.0
        return super()._energy(request)


class _MutatingBackend(_EnergyBackend):
    def prepare(self, request: PreparationRequest) -> _MutatingPrepared:
        assert request.force_field is not None
        return _MutatingPrepared(
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
    def __init__(self) -> None:
        provenance = BackendProvenance(
            backend="unit-reference",
            role=BackendRole.REFERENCE,
            details={"implementation": {"name": "unit"}},
        )
        self._info = BackendInfo(
            name="unit-reference",
            role=BackendRole.REFERENCE,
            capabilities=frozenset({Capability.ENERGY}),
            provenance=provenance,
        )

    @property
    def info(self) -> BackendInfo:
        return self._info

    def prepare(self, request: PreparationRequest) -> _ReferencePrepared:
        return _ReferencePrepared(
            info=self.info,
            case_id=request.case_id,
            molecule=request.molecule,
            force_field=None,
            layout=None,
        )


def test_public_module_exports_focused_api() -> None:
    import q2mm.backends.conformance as conformance

    assert conformance.__all__ == [
        "ConformanceError",
        "ConformanceOutcome",
        "MMConformanceCase",
        "ReferenceConformanceCase",
        "run_mm_conformance",
        "run_reference_conformance",
    ]


def test_mm_case_is_typed_immutable_and_energy_runs() -> None:
    molecule, force_field = _inputs()
    backend = _EnergyBackend()
    case = MMConformanceCase(
        descriptor=_descriptor(
            "unit-mm",
            BackendRole.MM,
            frozenset({Capability.ENERGY}),
            frozenset({"harmonic"}),
        ),
        backend=backend,
        molecule=molecule,
        force_field=force_field,
    )
    with pytest.raises(FrozenInstanceError):
        case.case_id = "changed"  # type: ignore[misc]
    outcome = run_mm_conformance(case)
    assert outcome == ConformanceOutcome(
        backend="unit-mm",
        role=BackendRole.MM,
        executed=(Capability.ENERGY,),
        unsupported_verified=(
            Capability.MINIMIZE,
            Capability.HESSIAN,
            Capability.FREQUENCIES,
            Capability.GEOMETRY_OPTIMIZATION,
            Capability.PARAMETER_GRADIENT,
            Capability.COORDINATE_GRADIENT,
            Capability.HESSIAN_PARAMETER_JACOBIAN,
            Capability.BATCHED_ENERGY,
            Capability.BATCHED_HESSIAN,
        ),
    )


def test_runtime_capability_ceiling_is_enforced() -> None:
    molecule, force_field = _inputs()
    backend = _EnergyBackend(capabilities=frozenset({Capability.ENERGY, Capability.HESSIAN}))
    case = MMConformanceCase(
        descriptor=_descriptor(
            "unit-mm",
            BackendRole.MM,
            frozenset({Capability.ENERGY}),
            frozenset({"harmonic"}),
        ),
        backend=backend,
        molecule=molecule,
        force_field=force_field,
    )
    with pytest.raises(ConformanceError, match="exceed the static ceiling"):
        run_mm_conformance(case)


def test_declared_energy_is_mandatory_in_selection() -> None:
    molecule, force_field = _inputs()
    backend = _EnergyBackend()
    case = MMConformanceCase(
        descriptor=_descriptor(
            "unit-mm",
            BackendRole.MM,
            frozenset({Capability.ENERGY}),
            frozenset({"harmonic"}),
        ),
        backend=backend,
        molecule=molecule,
        force_field=force_field,
        capabilities=frozenset(),
    )
    with pytest.raises(ConformanceError, match="ENERGY is declared"):
        run_mm_conformance(case)


def test_parameter_request_mutation_is_detected() -> None:
    molecule, force_field = _inputs()
    backend = _MutatingBackend()
    case = MMConformanceCase(
        descriptor=_descriptor(
            "unit-mm",
            BackendRole.MM,
            frozenset({Capability.ENERGY}),
            frozenset({"harmonic"}),
        ),
        backend=backend,
        molecule=molecule,
        force_field=force_field,
    )
    with pytest.raises(ConformanceError, match="mutated its typed request"):
        run_mm_conformance(case)


def test_case_rejects_wrong_typed_inputs() -> None:
    molecule, force_field = _inputs()
    backend = _EnergyBackend()
    descriptor = _descriptor(
        "unit-mm",
        BackendRole.MM,
        frozenset({Capability.ENERGY}),
        frozenset({"harmonic"}),
    )
    with pytest.raises(TypeError, match="Molecule"):
        MMConformanceCase(
            descriptor=descriptor,
            backend=backend,
            molecule=object(),  # type: ignore[arg-type]
            force_field=force_field,
        )
    with pytest.raises(TypeError, match="ForceField"):
        MMConformanceCase(
            descriptor=descriptor,
            backend=backend,
            molecule=molecule,
            force_field=object(),  # type: ignore[arg-type]
        )


def test_reference_case_rejects_mm_descriptor() -> None:
    molecule, _ = _inputs()
    backend = _EnergyBackend()
    with pytest.raises(ValueError, match="reference descriptor"):
        ReferenceConformanceCase(
            descriptor=_descriptor(
                "unit-mm",
                BackendRole.MM,
                frozenset({Capability.ENERGY}),
                frozenset({"harmonic"}),
            ),
            backend=backend,
            molecule=molecule,
        )


def test_reference_energy_conformance() -> None:
    molecule, _ = _inputs()
    backend = _ReferenceBackend()
    descriptor = _descriptor(
        "unit-reference",
        BackendRole.REFERENCE,
        frozenset({Capability.ENERGY}),
    )
    outcome = run_reference_conformance(
        ReferenceConformanceCase(
            descriptor=descriptor,
            backend=backend,
            molecule=molecule,
        )
    )
    assert outcome.executed == (Capability.ENERGY,)
    assert Capability.HESSIAN in outcome.unsupported_verified
    assert {
        Capability.MINIMIZE,
        Capability.PARAMETER_GRADIENT,
        Capability.HESSIAN_PARAMETER_JACOBIAN,
        Capability.BATCHED_ENERGY,
        Capability.BATCHED_HESSIAN,
    } <= set(outcome.unsupported_verified)


def test_import_has_no_optional_backend_runtime_side_effects() -> None:
    code = """
import sys
import q2mm.backends.conformance as conformance
for name in ("pytest", "openmm", "jax", "jaxlib", "ase", "qcengine", "qcelemental", "psi4"):
    assert name not in sys.modules, name
assert conformance.MMConformanceCase
print("CONFORMANCE_IMPORT_OK")
"""
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CONFORMANCE_IMPORT_OK"
