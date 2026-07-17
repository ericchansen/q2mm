"""Contract tests for the optional non-periodic ASE reference backend."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from typing import ClassVar

import numpy as np
import pytest

ase = pytest.importorskip("ase")
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lj import LennardJones
from ase.units import Bohr, Hartree

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
    ReferenceHessianRequest,
    UnsupportedCapabilityError,
)
from q2mm.backends.reference.ase import ASEBackend, ASEEvaluationError
from q2mm.models.molecule import Molecule
from test._conformance import assert_reference_capability_conformance


class _EnergyOnly(Calculator):
    implemented_properties = ["energy", "dipole"]
    calls: ClassVar[int] = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        type(self).calls += 1
        self.results["energy"] = 2.5


class _ForcesOnly(Calculator):
    implemented_properties = ["forces", "stress"]
    calls: ClassVar[int] = 0

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        type(self).calls += 1
        assert atoms is not None
        self.results["forces"] = np.arange(3 * len(atoms), dtype=float).reshape((-1, 3)) / 10.0


class _Failing(Calculator):
    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        raise RuntimeError("native ASE explosion")


class _InjectsPeriodicity(Calculator):
    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        atoms.set_cell([8.0, 8.0, 8.0])
        atoms.set_pbc(True)
        self.results["energy"] = 0.0


class _InjectsInitialCharges(Calculator):
    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        atoms.set_initial_charges(np.ones(len(atoms)))
        self.results["energy"] = 0.0


class _InjectsInitialMagmoms(Calculator):
    implemented_properties = ["energy"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)
        assert atoms is not None
        atoms.set_initial_magnetic_moments(np.ones(len(atoms)))
        self.results["energy"] = 0.0


class _NonCopyable(_EnergyOnly):
    def __deepcopy__(self, memo: dict[int, object]) -> _NonCopyable:
        raise TypeError("copy disabled")


class _PrepareNonCopyable(_EnergyOnly):
    copied = False

    def __deepcopy__(self, memo: dict[int, object]) -> _PrepareNonCopyable:
        if self.copied:
            raise TypeError("session copy disabled")
        clone = type(self)()
        clone.copied = True
        return clone


@pytest.fixture
def argon_pair() -> Molecule:
    return Molecule(
        symbols=("Ar", "Ar"),
        geometry=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.7]]),
        charge=1,
        multiplicity=2,
        bonds=(),
        angles=(),
        torsions=(),
    )


def test_lennard_jones_energy_gradient_conversion_and_structure(argon_pair: Molecule) -> None:
    calculator = LennardJones(epsilon=0.2, sigma=3.1, rc=8.0)
    caller_parameters = dict(calculator.parameters)
    original_geometry = argon_pair.geometry.copy()

    direct_atoms = Atoms(list(argon_pair.symbols), positions=argon_pair.geometry, cell=None, pbc=False)
    direct_atoms.calc = copy.deepcopy(calculator)
    expected_energy_ev = direct_atoms.get_potential_energy()
    expected_forces = direct_atoms.get_forces()

    prepared = ASEBackend(calculator=calculator).prepare(PreparationRequest(case_id="argon-pair", molecule=argon_pair))
    energy = prepared.energy(ReferenceEnergyRequest())
    gradient = prepared.coordinate_gradient(ReferenceCoordinateGradientRequest())
    repeated_energy = prepared.energy(ReferenceEnergyRequest())
    repeated_gradient = prepared.coordinate_gradient(ReferenceCoordinateGradientRequest())

    assert energy.energy == expected_energy_ev / Hartree
    np.testing.assert_array_equal(gradient.gradient, -expected_forces * Bohr / Hartree)
    assert repeated_energy.energy == energy.energy
    np.testing.assert_array_equal(repeated_gradient.gradient, gradient.gradient)
    assert energy.unit.value == "hartree"
    assert gradient.unit.value == "hartree/bohr"
    assert not gradient.gradient.flags.writeable

    assert tuple(prepared._atoms.get_chemical_symbols()) == argon_pair.symbols
    np.testing.assert_array_equal(prepared._atoms.positions, original_geometry)
    assert prepared._atoms.pbc.tolist() == [False, False, False]
    np.testing.assert_array_equal(prepared._atoms.cell.array, np.zeros((3, 3)))
    assert prepared._atoms.info == {"q2mm_charge": 1, "q2mm_multiplicity": 2}
    assert not prepared._atoms.has("initial_charges")
    assert not prepared._atoms.has("initial_magmoms")
    np.testing.assert_array_equal(argon_pair.geometry, original_geometry)

    assert calculator.atoms is None
    assert calculator.results == {}
    assert dict(calculator.parameters) == caller_parameters


def test_runtime_capability_subsets_and_undeclared_gate(argon_pair: Molecule) -> None:
    _EnergyOnly.calls = 0
    energy_backend = ASEBackend(calculator=_EnergyOnly())
    assert energy_backend.info.capabilities == frozenset({Capability.ENERGY})
    energy_session = energy_backend.prepare(PreparationRequest(case_id="energy", molecule=argon_pair))
    with pytest.raises(UnsupportedCapabilityError):
        energy_session.coordinate_gradient(ReferenceCoordinateGradientRequest())
    with pytest.raises(UnsupportedCapabilityError):
        energy_session.hessian(ReferenceHessianRequest())
    assert _EnergyOnly.calls == 0
    assert energy_session.energy(ReferenceEnergyRequest()).energy == 2.5 / Hartree
    assert _EnergyOnly.calls == 1

    _ForcesOnly.calls = 0
    forces_backend = ASEBackend(calculator=_ForcesOnly())
    assert forces_backend.info.capabilities == frozenset({Capability.COORDINATE_GRADIENT})
    forces_session = forces_backend.prepare(PreparationRequest(case_id="forces", molecule=argon_pair))
    with pytest.raises(UnsupportedCapabilityError):
        forces_session.energy(ReferenceEnergyRequest())
    assert _ForcesOnly.calls == 0
    expected_forces = np.arange(6, dtype=float).reshape((2, 3)) / 10.0
    result = forces_session.coordinate_gradient(ReferenceCoordinateGradientRequest())
    np.testing.assert_array_equal(result.gradient, -expected_forces * Bohr / Hartree)
    assert _ForcesOnly.calls == 1


def test_prepared_sessions_have_isolated_calculators(argon_pair: Molecule) -> None:
    caller = LennardJones()
    backend = ASEBackend(calculator=caller)
    first = backend.prepare(PreparationRequest(case_id="first", molecule=argon_pair))
    second = backend.prepare(PreparationRequest(case_id="second", molecule=argon_pair))

    assert first._atoms is not second._atoms
    assert first._calculator is not second._calculator
    assert first._calculator is not caller
    assert second._calculator is not caller
    first.energy(ReferenceEnergyRequest())
    assert first._calculator.results
    assert second._calculator.results == {}
    second.energy(ReferenceEnergyRequest())
    assert first.energy(ReferenceEnergyRequest()).energy == second.energy(ReferenceEnergyRequest()).energy
    assert caller.atoms is None
    assert caller.results == {}


def test_periodic_or_cell_calculator_state_is_rejected(argon_pair: Molecule) -> None:
    periodic = LennardJones()
    periodic.atoms = Atoms("Ar", cell=[5.0, 5.0, 5.0], pbc=True)
    with pytest.raises(BackendConfigurationError, match="non-periodic"):
        ASEBackend(calculator=periodic)

    cell_only = LennardJones()
    cell_only.atoms = Atoms("Ar", cell=[5.0, 5.0, 5.0], pbc=False)
    with pytest.raises(BackendConfigurationError, match="nonzero simulation cell"):
        ASEBackend(calculator=cell_only)

    injected = ASEBackend(calculator=_InjectsPeriodicity()).prepare(
        PreparationRequest(case_id="injected", molecule=argon_pair)
    )
    with pytest.raises(ASEEvaluationError, match="non-periodic structure contract") as caught:
        injected.energy(ReferenceEnergyRequest())
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("calculator", [_InjectsInitialCharges(), _InjectsInitialMagmoms()])
def test_calculator_cannot_invent_per_atom_state(argon_pair: Molecule, calculator: Calculator) -> None:
    prepared = ASEBackend(calculator=calculator).prepare(
        PreparationRequest(case_id="atomic-state", molecule=argon_pair)
    )
    with pytest.raises(ASEEvaluationError, match="unsupported per-atom") as caught:
        prepared.energy(ReferenceEnergyRequest())
    assert isinstance(caught.value.__cause__, ValueError)


def test_typed_configuration_preparation_and_native_failures(argon_pair: Molecule) -> None:
    with pytest.raises(BackendConfigurationError, match="implemented_properties"):
        ASEBackend(calculator=object())

    unrelated = Calculator()
    unrelated.implemented_properties = ["stress"]
    with pytest.raises(BackendConfigurationError, match="at least one"):
        ASEBackend(calculator=unrelated)

    with pytest.raises(BackendConfigurationError, match="copy.deepcopy") as copy_error:
        ASEBackend(calculator=_NonCopyable())
    assert isinstance(copy_error.value.__cause__, TypeError)

    prepare_noncopyable = ASEBackend(calculator=_PrepareNonCopyable())
    with pytest.raises(PreparationError, match="copy.deepcopy") as prepare_copy_error:
        prepare_noncopyable.prepare(PreparationRequest(case_id="copy", molecule=argon_pair))
    assert isinstance(prepare_copy_error.value.__cause__, TypeError)

    backend = ASEBackend(calculator=_EnergyOnly())
    with pytest.raises(PreparationError, match="force_field"):
        backend.prepare(
            PreparationRequest(case_id="ff", molecule=argon_pair, force_field=object())  # type: ignore[arg-type]
        )
    with pytest.raises(PreparationError, match="per-case options"):
        backend.prepare(PreparationRequest(case_id="options", molecule=argon_pair, options={"x": 1}))

    failing = ASEBackend(calculator=_Failing()).prepare(PreparationRequest(case_id="failure", molecule=argon_pair))
    with pytest.raises(ASEEvaluationError, match="native ASE explosion") as native_error:
        failing.energy(ReferenceEnergyRequest())
    assert isinstance(native_error.value.__cause__, RuntimeError)


def test_missing_ase_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    import q2mm.backends.reference.ase as adapter

    monkeypatch.setattr(adapter, "_ase", None)
    with pytest.raises(BackendUnavailableError, match=r"q2mm\[ase\]"):
        ASEBackend(calculator=_EnergyOnly())


def test_descriptor_runtime_subset_conformance_and_provenance(argon_pair: Molecule) -> None:
    descriptor = registry.get_descriptor("ase")
    ceiling = frozenset({Capability.ENERGY, Capability.COORDINATE_GRADIENT})
    assert descriptor.role is BackendRole.REFERENCE
    assert descriptor.capability_ceiling == ceiling
    assert descriptor.functional_form_ceiling == frozenset()
    assert descriptor.factory == "q2mm.backends.reference.ase:ASEBackend"
    assert descriptor.probe.modules == ("ase",)

    backend = registry.load_backend("ase", calculator=LennardJones())
    assert backend.info.capabilities == ceiling
    outcome = assert_reference_capability_conformance(backend, molecule=argon_pair)
    assert set(outcome.executed) == ceiling
    assert set(outcome.unsupported_verified) == {
        Capability.HESSIAN,
        Capability.FREQUENCIES,
        Capability.GEOMETRY_OPTIMIZATION,
    }

    energy_only = registry.load_backend("ase", calculator=_EnergyOnly())
    assert energy_only.info.capabilities == frozenset({Capability.ENERGY})
    assert energy_only.info.capabilities < descriptor.capability_ceiling

    result = backend.prepare(PreparationRequest(case_id="provenance", molecule=argon_pair)).energy(
        ReferenceEnergyRequest()
    )
    details = result.provenance.details
    assert result.provenance.backend == "ase"
    assert result.provenance.version == ase.__version__
    assert details["adapter"]["conversion_version"] == 1
    assert details["implementation"] == {"name": "ASE", "version": ase.__version__}
    assert details["calculator"]["class"] == "LennardJones"
    assert details["runtime"]["implemented_properties"] == tuple(sorted(set(LennardJones.implemented_properties)))
    assert details["driver"] == {"property": "energy", "ase_property": "energy"}
    assert details["units"]["formula"] == "energy_hartree = energy_eV / ase.units.Hartree"
    serialized = json.dumps(details)
    assert "object at 0x" not in serialized
    assert "credential" not in serialized.casefold()

    provenance_calculator = _EnergyOnly()
    provenance_calculator.parameters = {
        "safe": 3.5,
        "nested": {"enabled": True},
        "api_token": "must-not-appear",
        "path": "C:\\private\\calculator.dat",
        "unsafe": object(),
    }
    safe_details = ASEBackend(calculator=provenance_calculator).info.provenance.details
    assert safe_details["calculator"]["parameters"] == {
        "safe": 3.5,
        "nested": {"enabled": True},
    }
    safe_serialized = json.dumps(safe_details)
    assert "must-not-appear" not in safe_serialized
    assert "C:\\\\private" not in safe_serialized
    assert "object at 0x" not in safe_serialized


def test_catalog_is_lazy_for_ase_in_fresh_interpreter() -> None:
    code = (
        "import sys\n"
        "import q2mm.backends.registry as registry\n"
        "status = next(item for item in registry.catalog() if item.name == 'ase')\n"
        "assert status.descriptor.probe.modules == ('ase',)\n"
        "assert 'ase' not in sys.modules\n"
        "assert 'q2mm.backends.reference.ase' not in sys.modules\n"
        "print('ASE_LAZY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "ASE_LAZY_OK" in result.stdout
