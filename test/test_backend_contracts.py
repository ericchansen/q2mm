"""Conformance tests for the backend capability contracts.

These tests exercise the vocabulary in :mod:`q2mm.backends.contracts` and the
descriptor-based :mod:`q2mm.backends.registry` independently of any single
backend, plus per-backend capability/units/provenance/reuse/no-mutation
guarantees and OpenMM↔JAX cross-backend parity.

Backends that are not installed are reported explicitly by the catalog and the
unavailable-path assertions still run.
"""

from __future__ import annotations


from test.backend_fixtures import param_vector, prepare_case

import numpy as np
import pytest

from q2mm.backends.contracts import (
    AbstractPreparedBackend,
    BACKEND_API_VERSION,
    BackendDescriptor,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BatchedHessianRequest,
    Capability,
    CoordinateGradientResult,
    CoordinateGradientUnit,
    DependencyProbe,
    EnergyRequest,
    EnergyUnit,
    EvaluationError,
    FrequencyResult,
    FrequencyUnit,
    FrequencyRequest,
    HessianJacobianRequest,
    HessianRequest,
    HessianResult,
    HessianUnit,
    ParameterGradientRequest,
    PreparationRequest,
    ReferenceCoordinateGradientRequest,
    ReferenceEnergyRequest,
    ReferenceGeometryOptimizationRequest,
    ReferenceHessianRequest,
    UnsupportedCapabilityError,
)
from q2mm.backends.registry import (
    available_backends,
    catalog,
    descriptors,
    get_descriptor,
    load_backend,
)
from q2mm.models.forcefield import ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterLayout


# ---------------------------------------------------------------------------
# Vocabulary-level conformance (no backend required)
# ---------------------------------------------------------------------------


def test_default_backend_info_is_empty() -> None:
    """A default BackendInfo declares no capabilities and no forms."""
    info = BackendInfo(name="x", role=BackendRole.MM)
    assert info.capabilities == frozenset()
    assert info.functional_forms == frozenset()
    assert not info.supports(Capability.ENERGY)
    assert not info.supports_form("harmonic")


def test_array_backed_requests_and_results_use_identity_equality() -> None:
    first_request = EnergyRequest(parameters=np.array([1.0, 2.0, 3.0]))
    second_request = EnergyRequest(parameters=np.array([1.0, 2.0, 3.0]))
    provenance = BackendProvenance(backend="test", role=BackendRole.MM)
    first_result = FrequencyResult(
        frequencies=np.array([100.0, 200.0, 300.0]),
        unit=FrequencyUnit.INVERSE_CM,
        provenance=provenance,
    )
    second_result = FrequencyResult(
        frequencies=np.array([100.0, 200.0, 300.0]),
        unit=FrequencyUnit.INVERSE_CM,
        provenance=provenance,
    )

    assert first_request != second_request
    assert first_result != second_result
    assert len({first_request, second_request, first_result, second_result}) == 4


def test_dependency_probe_missing_module() -> None:
    healthy, reason = DependencyProbe(modules=("definitely_not_a_module_zzz",)).check()
    assert healthy is False
    assert "definitely_not_a_module_zzz" in reason


def test_dependency_probe_missing_executable() -> None:
    healthy, reason = DependencyProbe(executables=("q2mm_no_such_exe_zzz",)).check()
    assert healthy is False
    assert "q2mm_no_such_exe_zzz" in reason


def test_abstract_prepared_backend_raises_unsupported() -> None:
    """A prepared session declaring no capabilities raises typed errors."""
    info = BackendInfo(name="empty", role=BackendRole.MM)

    class _Empty(AbstractPreparedBackend):
        pass

    prepared = _Empty(info=info, case_id="0", molecule=None, force_field=None, layout=None)
    with pytest.raises(UnsupportedCapabilityError):
        prepared.energy(EnergyRequest(parameters=np.zeros(1)))
    with pytest.raises(UnsupportedCapabilityError):
        prepared.hessian(HessianRequest(parameters=np.zeros(1)))


# ---------------------------------------------------------------------------
# Registry / catalog conformance (side-effect free)
# ---------------------------------------------------------------------------


def test_catalog_reports_all_descriptors() -> None:
    names = {s.name for s in catalog()}
    assert names == set(descriptors())
    assert {"openmm", "tinker", "jax", "jax-md", "psi4"} <= names


def test_catalog_reports_status_and_reason() -> None:
    for status in catalog():
        assert isinstance(status.healthy, bool)
        if not status.healthy:
            assert status.reason  # unavailable descriptors carry a reason


def test_available_backends_subset_of_catalog() -> None:
    healthy = {s.name for s in catalog() if s.healthy}
    assert set(available_backends()) == healthy


def test_catalog_is_side_effect_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """Listing must never construct a backend or init a device/platform.

    Patch every backend factory to explode if called; the catalog must still
    complete via cheap probes only.
    """
    import q2mm.backends.mm.openmm as omm
    import q2mm.backends.mm.jax_engine as jx

    def _boom(*args: object, **kwargs: object):  # noqa: ANN202
        raise AssertionError("backend constructed during side-effect-free listing")

    monkeypatch.setattr(omm, "OpenMMBackend", _boom, raising=False)
    monkeypatch.setattr(jx, "JaxBackend", _boom, raising=False)
    # Also patch OpenMM platform enumeration to fail if touched.
    if getattr(omm, "mm", None) is not None:
        monkeypatch.setattr(omm.mm.Platform, "getNumPlatforms", _boom, raising=False)

    statuses = catalog()  # must not raise
    assert statuses
    _ = available_backends()


def test_unknown_backend_raises() -> None:
    from q2mm.backends.registry import BackendNotRegistered

    with pytest.raises(BackendNotRegistered):
        load_backend("no-such-backend")


def test_descriptor_validation_rejects_bad_factory() -> None:
    with pytest.raises(ValueError):
        BackendDescriptor(
            name="x",
            role=BackendRole.MM,
            capability_ceiling=frozenset(),
            functional_form_ceiling=frozenset(),
            factory="no_colon_here",
        )
    with pytest.raises(ValueError):
        BackendDescriptor(
            name="",
            role=BackendRole.MM,
            capability_ceiling=frozenset(),
            functional_form_ceiling=frozenset(),
            factory="a:b",
        )


# ---------------------------------------------------------------------------
# Per-backend conformance (parametrized over available MM backends)
# ---------------------------------------------------------------------------


def _methane() -> tuple[Molecule, ForceField, ParameterLayout]:
    mol = Molecule(
        symbols=["C", "H", "H", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
            dtype=float,
        ),
        atom_types=["C", "H", "H", "H", "H"],
    )
    ff = ForceField.create_for_molecule(mol, functional_form=FunctionalForm.HARMONIC)
    layout = ParameterLayout.from_force_field(ff)
    return mol, ff, layout


_MM_HARMONIC_BACKENDS = [k for k in ("openmm", "jax") if k in available_backends()]


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_capability_claims_match_methods(key: str) -> None:
    """Every declared capability works; every undeclared one raises typed errors.

    Exhaustively invokes each prepared-session evaluation method: declared
    capabilities must return the right result type; undeclared capabilities
    must raise :class:`UnsupportedCapabilityError`.
    """
    from q2mm.backends.contracts import (
        BatchedEnergyRequest,
        BatchedEnergyResult,
        EnergyResult,
        FrequencyResult,
        GeometryResult,
        HessianJacobianResult,
        HessianResult,
        MinimizationRequest,
        ParameterGradientResult,
    )

    mol, ff, layout = _methane()
    backend = load_backend(key)
    info = backend.info
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    vec = layout.vector(ff)
    mat = np.stack([vec, vec])

    # Each prepared-session capability -> (invoke, expected result type).
    cases: dict[Capability, tuple] = {
        Capability.ENERGY: (lambda: prepared.energy(EnergyRequest(parameters=vec)), EnergyResult),
        Capability.MINIMIZE: (lambda: prepared.minimize(MinimizationRequest(parameters=vec)), GeometryResult),
        Capability.HESSIAN: (lambda: prepared.hessian(HessianRequest(parameters=vec)), HessianResult),
        Capability.FREQUENCIES: (
            lambda: prepared.frequencies(FrequencyRequest(parameters=vec)),
            FrequencyResult,
        ),
        Capability.PARAMETER_GRADIENT: (
            lambda: prepared.parameter_gradient(ParameterGradientRequest(parameters=vec)),
            ParameterGradientResult,
        ),
        Capability.HESSIAN_PARAMETER_JACOBIAN: (
            lambda: prepared.hessian_parameter_jacobian(HessianJacobianRequest(parameters=vec)),
            HessianJacobianResult,
        ),
        Capability.BATCHED_ENERGY: (
            lambda: prepared.batched_energy(BatchedEnergyRequest(parameter_matrix=mat)),
            BatchedEnergyResult,
        ),
        # GEOMETRY_OPTIMIZATION is QM-only; for an MM backend it must be
        # undeclared and raise UnsupportedCapabilityError.
        Capability.GEOMETRY_OPTIMIZATION: (
            lambda: prepared.optimize_geometry(ReferenceGeometryOptimizationRequest()),
            GeometryResult,
        ),
        Capability.COORDINATE_GRADIENT: (
            lambda: prepared.coordinate_gradient(ReferenceCoordinateGradientRequest()),
            CoordinateGradientResult,
        ),
    }

    for cap, (invoke, result_type) in cases.items():
        if info.supports(cap):
            assert isinstance(invoke(), result_type), f"{key}: {cap.value} returned wrong result type"
        else:
            with pytest.raises(UnsupportedCapabilityError):
                invoke()


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_exact_request_family_validation(key: str) -> None:
    """Wrong request type for the right role raises EvaluationError before dispatch."""
    from q2mm.backends.contracts import EvaluationError, MinimizationRequest

    mol, ff, layout = _methane()
    backend = load_backend(key)
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    vec = layout.vector(ff)

    # Same-role but wrong operation type.
    with pytest.raises(EvaluationError):
        prepared.energy(HessianRequest(parameters=vec))
    with pytest.raises(EvaluationError):
        prepared.hessian(MinimizationRequest(parameters=vec))
    with pytest.raises(EvaluationError):
        prepared.minimize(EnergyRequest(parameters=vec))
    with pytest.raises(EvaluationError):
        prepared.frequencies(EnergyRequest(parameters=vec))
    # An MM session must reject a reference request family.
    with pytest.raises(EvaluationError):
        prepared.energy(ReferenceEnergyRequest())


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_batched_hessian_capability_via_helper(key: str) -> None:
    """BATCHED_HESSIAN goes through the backend-neutral contracts helper.

    A backend that declares the capability returns typed batches/results; one
    that does not (e.g. OpenMM) raises UnsupportedCapabilityError.
    """
    from q2mm.backends.contracts import (
        BatchedHessianRequest,
        BatchedHessianResult,
        Capability,
        HessianUnit,
        PreparedHessianBatch,
        UnsupportedCapabilityError,
        prepare_hessian_batches,
    )

    mol, ff, layout = _methane()
    backend = load_backend(key)
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    vec = layout.vector(ff)

    if backend.info.supports(Capability.BATCHED_HESSIAN):
        batches = prepare_hessian_batches(backend, [prepared])
        assert len(batches) == 1
        assert isinstance(batches[0], PreparedHessianBatch)
        assert batches[0].case_ids == ("0",)
        result = batches[0].hessians(BatchedHessianRequest(parameters=vec))
        assert isinstance(result, BatchedHessianResult)
        assert result.unit is HessianUnit.HARTREE_PER_BOHR2
        assert result.hessians.shape == (1, 3 * len(mol.symbols), 3 * len(mol.symbols))
    else:
        with pytest.raises(UnsupportedCapabilityError):
            prepare_hessian_batches(backend, [prepared])


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_result_units_provenance_dimensions_readonly(key: str) -> None:
    mol, ff, layout = _methane()
    backend = load_backend(key)
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    vec = layout.vector(ff)

    e = prepared.energy(EnergyRequest(parameters=vec))
    assert e.unit == EnergyUnit.KCAL_PER_MOL
    assert e.provenance is not None and e.provenance.backend == key

    h = prepared.hessian(HessianRequest(parameters=vec))
    assert not h.hessian.flags.writeable  # defensive read-only
    assert h.provenance.backend == key

    f = prepared.frequencies(FrequencyRequest(parameters=vec))
    assert not f.frequencies.flags.writeable

    if backend.info.supports(Capability.PARAMETER_GRADIENT):
        g = prepared.parameter_gradient(ParameterGradientRequest(parameters=vec))
        assert len(g.gradient) == len(layout)  # exactly len(ParameterLayout)
        assert not g.gradient.flags.writeable


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_bad_vector_length_raises(key: str) -> None:
    from q2mm.backends.contracts import EvaluationError

    mol, ff, layout = _methane()
    backend = load_backend(key)
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    with pytest.raises(EvaluationError):
        prepared.energy(EnergyRequest(parameters=np.zeros(len(layout) + 3)))


@pytest.mark.parametrize("key", _MM_HARMONIC_BACKENDS)
def test_prepare_once_reuse_and_no_input_mutation(key: str) -> None:
    """One prepared session is reused across evaluations; inputs unmutated."""
    mol, ff, layout = _methane()
    backend = load_backend(key)
    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    assert prepared.case_id == "0"
    assert prepared.molecule is mol

    vec = layout.vector(ff)
    vec_copy = vec.copy()
    e1 = prepared.energy(EnergyRequest(parameters=vec)).energy
    e2 = prepared.energy(EnergyRequest(parameters=vec)).energy
    assert e1 == pytest.approx(e2)
    # The input vector must not have been mutated by evaluation.
    assert np.array_equal(vec, vec_copy)


@pytest.mark.jax
def test_conformer_batch_equals_independent_and_no_first_case_reuse() -> None:
    """Same-topology conformer batch equals per-case; mixed cases stay isolated.

    Uses the backend-neutral :func:`prepare_hessian_batches` helper and compares
    the typed batch result against the **original** ``s1``/``s2`` sessions —
    proving both reuse (same sessions, no re-preparation) and isolation (each
    row equals its own session's Hessian at its own coordinates).
    """
    from q2mm.backends.contracts import prepare_hessian_batches
    from q2mm.backends.mm.jax_engine import JaxBackend

    if "jax" not in available_backends():
        pytest.skip("jax not available")

    mol, ff, layout = _methane()
    # A second conformer: same topology, different coordinates.
    mol2 = Molecule(
        symbols=mol.symbols,
        geometry=mol.geometry * 1.02,
        atom_types=mol.atom_types,
    )
    backend = JaxBackend()

    # Prepare one session per conformer — each owns its own native state.
    s1 = prepare_case(backend, mol, ff, "case-0")
    s2 = prepare_case(backend, mol2, ff, "case-1")
    batches = prepare_hessian_batches(backend, [s1, s2])
    # Same topology → a single batch tracking two distinct case IDs.
    assert len(batches) == 1
    batch = batches[0]
    assert batch.case_ids == ("case-0", "case-1")

    vec = param_vector(ff)
    result = batch.hessians(BatchedHessianRequest(parameters=vec))
    assert result.case_ids == ("case-0", "case-1")
    row_by_case = dict(zip(result.case_ids, result.hessians))
    # Compare against the ORIGINAL sessions (reuse + isolation), not new ones.
    h1 = s1.hessian(HessianRequest(parameters=vec)).hessian
    h2 = s2.hessian(HessianRequest(parameters=vec)).hessian
    assert np.allclose(row_by_case["case-0"], h1, atol=1e-6)
    assert np.allclose(row_by_case["case-1"], h2, atol=1e-6)
    # Different coordinates -> different Hessians: no first-case state reused.
    assert not np.allclose(row_by_case["case-0"], row_by_case["case-1"])


# ---------------------------------------------------------------------------
# Cross-backend parity (OpenMM ↔ JAX), run now — not deferred
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not ({"openmm", "jax"} <= set(available_backends())),
    reason="requires both OpenMM and JAX",
)
@pytest.mark.cross_backend
@pytest.mark.jax
@pytest.mark.openmm
def test_openmm_jax_energy_hessian_parity() -> None:
    mol, ff, layout = _methane()
    vec = layout.vector(ff)

    omm = load_backend("openmm")
    jx = load_backend("jax")
    p_omm = omm.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    p_jx = jx.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))

    e_omm = p_omm.energy(EnergyRequest(parameters=vec)).energy
    e_jx = p_jx.energy(EnergyRequest(parameters=vec)).energy
    assert e_omm == pytest.approx(e_jx, abs=1e-6)

    h_omm = np.asarray(p_omm.hessian(HessianRequest(parameters=vec)).hessian)
    h_jx = np.asarray(p_jx.hessian(HessianRequest(parameters=vec)).hessian)
    assert np.max(np.abs(h_omm - h_jx)) < 1e-4


# ---------------------------------------------------------------------------
# Unavailable-path conformance (always runnable)
# ---------------------------------------------------------------------------


def test_unavailable_backend_load_raises_typed() -> None:
    """A descriptor whose probe fails raises BackendUnavailableError on load."""
    from q2mm.backends.contracts import BackendUnavailableError

    for status in catalog():
        if not status.healthy:
            desc = get_descriptor(status.name)
            with pytest.raises(BackendUnavailableError):
                desc.load()
            return
    pytest.skip("all backends available; no unavailable path to test")


def test_explicit_config_bypasses_unhealthy_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unhealthy PATH probe must not block a valid explicit load.

    Uses a temporary importable fake module + factory registered via
    ``sys.modules``, with an intentionally unhealthy :class:`DependencyProbe`.
    The factory returns a valid Backend and records the kwargs it received; the
    test asserts the explicit kwargs reached the factory and that
    ``descriptor.load()`` succeeds despite the failing probe — proving the probe
    is catalog-only.
    """
    import sys
    import types

    from q2mm.backends.contracts import (
        BackendDescriptor,
        BackendInfo,
        BackendProvenance,
        Capability,
        DependencyProbe,
    )

    received: dict = {}

    class _FakeBackend:
        def __init__(self, **kwargs: object) -> None:
            received.update(kwargs)

        @property
        def info(self) -> BackendInfo:
            return BackendInfo(
                name="fake-explicit",
                role=BackendRole.MM,
                capabilities=frozenset({Capability.ENERGY}),
                functional_forms=frozenset({"mm3"}),
                provenance=BackendProvenance(backend="fake-explicit", role=BackendRole.MM),
            )

        def prepare(self, request: PreparationRequest) -> object:
            raise AssertionError("prepare not needed for this test")

    fake_mod = types.ModuleType("q2mm_fake_backend_mod")
    fake_mod._FakeBackend = _FakeBackend  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "q2mm_fake_backend_mod", fake_mod)

    desc = BackendDescriptor(
        name="fake-explicit",
        role=BackendRole.MM,
        capability_ceiling=frozenset({Capability.ENERGY}),
        functional_form_ceiling=frozenset({"mm3"}),
        factory="q2mm_fake_backend_mod:_FakeBackend",
        # Intentionally unhealthy probe (module does not exist).
        probe=DependencyProbe(modules=("definitely_not_installed_xyz_123",)),
    )
    healthy, _ = desc.is_available()
    assert healthy is False  # probe reports unavailable...

    backend = desc.load(tinker_dir="/opt/tinker/bin", params_file="/opt/tinker/mm3.prm")
    # ...but explicit load succeeds and forwards the explicit kwargs.
    assert backend.info.provenance.backend == "fake-explicit"
    assert received == {"tinker_dir": "/opt/tinker/bin", "params_file": "/opt/tinker/mm3.prm"}


@pytest.mark.openmm
def test_openmm_platform_failure_raises_typed_no_fallback() -> None:
    """An invalid OpenMM platform raises a typed config error with platform context.

    The old silent GPU->CPU fallback is gone: selecting a nonexistent platform
    must surface a :class:`BackendConfigurationError` naming the platform, and
    must not silently succeed on a different platform.
    """
    from q2mm.backends.contracts import BackendConfigurationError

    if "openmm" not in available_backends():
        pytest.skip("OpenMM not available")

    mol, ff, _layout = _methane()
    bad = load_backend("openmm", platform_name="NONEXISTENT_PLATFORM")
    # Backend state records the requested platform; no mutation to a fallback.
    assert "NONEXISTENT_PLATFORM" in bad.info.name
    with pytest.raises(BackendConfigurationError) as excinfo:
        bad.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    assert "NONEXISTENT_PLATFORM" in str(excinfo.value)
    # The backend must not have silently rewritten its platform to a fallback.
    assert "NONEXISTENT_PLATFORM" in bad.info.name


# ---------------------------------------------------------------------------
# Reference request-family validation (fake reference session)
# ---------------------------------------------------------------------------


def _fake_reference_prepared() -> AbstractPreparedBackend:
    from q2mm.backends.contracts import (
        BackendInfo,
        BackendProvenance,
        Capability,
        EnergyUnit,
        HessianUnit,
    )
    from q2mm.backends.contracts import (
        EnergyResult as _ER,
    )
    from q2mm.backends.contracts import (
        HessianResult as _HR,
    )
    from test.backend_fixtures import mock_molecule

    prov = BackendProvenance(
        backend="fake-reference",
        role=BackendRole.REFERENCE,
        details={"implementation": {"name": "fake"}, "model": {"method": "test"}},
    )
    info = BackendInfo(
        name="fake-reference",
        role=BackendRole.REFERENCE,
        capabilities=frozenset({Capability.ENERGY, Capability.HESSIAN, Capability.COORDINATE_GRADIENT}),
        functional_forms=frozenset(),
        provenance=prov,
    )
    mol = mock_molecule(["H", "H"])

    class _Reference(AbstractPreparedBackend):
        def _energy(self, request: object) -> _ER:  # type: ignore[override]
            return _ER(energy=-1.0, unit=EnergyUnit.HARTREE, provenance=prov)

        def _hessian(self, request: object) -> _HR:  # type: ignore[override]
            return _HR(hessian=np.eye(6), unit=HessianUnit.HARTREE_PER_BOHR2, provenance=prov)

        def _coordinate_gradient(self, request: object) -> CoordinateGradientResult:  # type: ignore[override]
            return CoordinateGradientResult(
                gradient=np.zeros((2, 3)),
                unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
                provenance=prov,
            )

    return _Reference(info=info, case_id="0", molecule=mol, force_field=None, layout=None)


def test_reference_session_request_family_and_coordinate_gradient() -> None:
    from q2mm.backends.contracts import (
        EvaluationError,
    )

    prepared = _fake_reference_prepared()
    assert prepared.energy(ReferenceEnergyRequest()).unit.value == "hartree"
    assert prepared.hessian(ReferenceHessianRequest()).hessian.shape == (6, 6)
    gradient = prepared.coordinate_gradient(ReferenceCoordinateGradientRequest())
    assert gradient.gradient.shape == (2, 3)
    assert not gradient.gradient.flags.writeable
    # MM request families are rejected on a reference session.
    with pytest.raises(EvaluationError):
        prepared.energy(EnergyRequest(parameters=np.zeros(1)))
    with pytest.raises(EvaluationError):
        prepared.hessian(HessianRequest(parameters=np.zeros(1)))


# ---------------------------------------------------------------------------
# Deep-frozen preparation options
# ---------------------------------------------------------------------------


def test_preparation_options_deep_frozen() -> None:
    """Nested option structures are deeply frozen; source mutation cannot leak."""
    from test.backend_fixtures import mock_molecule

    nested = {"a": [1, [2, 3]], "b": {"c": {1, 2}}, "arr": np.array([1.0, 2.0])}
    req = PreparationRequest(case_id="x", molecule=mock_molecule(["H"]), options=nested)
    # Mutate the caller's original structure at several depths.
    nested["a"].append(99)
    nested["a"][1].append(77)
    nested["b"]["c"].add(5)
    nested["arr"][0] = -1.0
    assert req.options["a"] == (1, (2, 3))
    assert req.options["b"]["c"] == frozenset({1, 2})
    assert list(req.options["arr"]) == [1.0, 2.0]
    # The stored options mapping is itself immutable.
    with pytest.raises(TypeError):
        req.options["a"] = 0  # type: ignore[index]


# ---------------------------------------------------------------------------
# Direct-construction result validation
# ---------------------------------------------------------------------------


def test_result_direct_construction_validates() -> None:
    from q2mm.backends.contracts import (
        BackendProvenance,
        EnergyResult,
        EnergyUnit,
        EvaluationError,
        HessianResult,
        HessianUnit,
    )

    prov_mm = BackendProvenance(backend="x", role=BackendRole.MM)
    prov_reference = BackendProvenance(backend="x", role=BackendRole.REFERENCE)

    # Wrong energy unit for the provenance role.
    with pytest.raises(EvaluationError):
        EnergyResult(energy=1.0, unit=EnergyUnit.HARTREE, provenance=prov_mm)
    with pytest.raises(EvaluationError):
        EnergyResult(energy=1.0, unit=EnergyUnit.KCAL_PER_MOL, provenance=prov_reference)
    # Non-finite scalar.
    with pytest.raises(EvaluationError):
        EnergyResult(energy=float("nan"), unit=EnergyUnit.KCAL_PER_MOL, provenance=prov_mm)
    # Non-provenance object.
    with pytest.raises(EvaluationError):
        EnergyResult(energy=1.0, unit=EnergyUnit.KCAL_PER_MOL, provenance="not-a-provenance")  # type: ignore[arg-type]
    # Wrong unit enum type.
    with pytest.raises(EvaluationError):
        HessianResult(hessian=np.eye(3), unit=EnergyUnit.KCAL_PER_MOL, provenance=prov_mm)  # type: ignore[arg-type]
    # Non-finite array.
    bad = np.eye(3)
    bad[0, 0] = np.inf
    with pytest.raises(EvaluationError):
        HessianResult(hessian=bad, unit=HessianUnit.HARTREE_PER_BOHR2, provenance=prov_mm)


def test_backend_info_and_provenance_validation() -> None:
    from q2mm.backends.contracts import BackendInfo, BackendProvenance

    # Empty provenance backend is rejected.
    with pytest.raises(ValueError):
        BackendProvenance(backend="", role=BackendRole.MM)
    # Reference info must declare no functional forms.
    with pytest.raises(ValueError):
        BackendInfo(name="x", role=BackendRole.REFERENCE, functional_forms=frozenset({"mm3"}))
    with pytest.raises(ValueError, match="COORDINATE_GRADIENT"):
        BackendInfo(
            name="x",
            role=BackendRole.MM,
            capabilities=frozenset({Capability.COORDINATE_GRADIENT}),
        )
    # info provenance role must agree with info role.
    with pytest.raises(ValueError):
        BackendInfo(
            name="x",
            role=BackendRole.MM,
            provenance=BackendProvenance(backend="x", role=BackendRole.REFERENCE),
        )


def test_backend_api_v1_has_no_pre_v1_contract_names() -> None:
    import q2mm.backends.contracts as contracts

    assert BACKEND_API_VERSION == 1
    for name in (
        "DESCRIPTOR_API_VERSION",
        "QMEnergyRequest",
        "QMHessianRequest",
        "QMFrequencyRequest",
        "QMGeometryOptimizationRequest",
    ):
        assert not hasattr(contracts, name)


def test_structured_provenance_is_immutable_json_safe_and_secret_free() -> None:
    import json

    source = {
        "implementation": {"name": "engine", "version": "1.2"},
        "model": {"method": "method", "options": [1, True, None]},
    }
    provenance = BackendProvenance(
        backend="reference",
        role=BackendRole.REFERENCE,
        version="1.2",
        details=source,
    )
    source["implementation"]["name"] = "mutated"
    assert provenance.details["implementation"]["name"] == "engine"
    assert json.loads(json.dumps(provenance.details))["model"]["options"] == [1, True, None]
    with pytest.raises(TypeError):
        provenance.details["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        provenance.details["implementation"]["name"] = "value"  # type: ignore[index]

    bad_details = (
        {1: "value"},
        {"config": {"api_token": "value"}},
        {"config": {"authToken": "value"}},
        {"config": {"clientSecret": "value"}},
        {"config": {"value": float("nan")}},
        {"config": object()},
        {"native": "Bearer abc123"},
    )
    for details in bad_details:
        with pytest.raises(ValueError):
            BackendProvenance(backend="x", role=BackendRole.MM, details=details)  # type: ignore[arg-type]


def test_coordinate_gradient_and_hessian_provenance_contracts() -> None:
    from q2mm.models.hessian import HessianUnits

    provenance = BackendProvenance(
        backend="reference",
        role=BackendRole.REFERENCE,
        details={"model": {"method": "test"}},
    )
    gradient = CoordinateGradientResult(
        gradient=np.zeros((2, 3)),
        unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
        provenance=provenance,
    )
    assert gradient.gradient.shape == (2, 3)
    assert not gradient.gradient.flags.writeable
    with pytest.raises(EvaluationError):
        CoordinateGradientResult(
            gradient=np.zeros(6),
            unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
            provenance=provenance,
        )
    with pytest.raises(EvaluationError):
        CoordinateGradientResult(
            gradient=np.full((2, 3), np.inf),
            unit=CoordinateGradientUnit.HARTREE_PER_BOHR,
            provenance=provenance,
        )

    result = HessianResult(
        hessian=np.eye(6),
        unit=HessianUnit.HARTREE_PER_BOHR2,
        provenance=provenance,
    )
    molecule_provenance = result.hessian_provenance
    assert molecule_provenance.units is HessianUnits.ATOMIC
    assert molecule_provenance.source == "reference"
    assert molecule_provenance.path is None
    assert molecule_provenance.source_details["details"]["model"]["method"] == "test"
    with pytest.raises(TypeError):
        molecule_provenance.source_details["new"] = "value"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Preparation reuse: JAX objective executor prepares each case exactly once
# ---------------------------------------------------------------------------


@pytest.mark.jax
def test_jax_objective_executor_prepares_each_case_once() -> None:
    if "jax" not in available_backends():
        pytest.skip("jax not available")

    from q2mm.backends.mm.jax_engine import JaxBackend
    from q2mm.models.observations import ObservationSet
    from q2mm.models.parameters import ActiveParameterSpace
    from q2mm.models.problem import StationaryPointKind
    from q2mm.objectives.jax import JaxObjectiveExecutor
    from q2mm.objectives.plan import ObjectivePlan

    mol, ff, layout = _methane()
    mol2 = Molecule(symbols=mol.symbols, geometry=mol.geometry * 1.03, atom_types=mol.atom_types)
    molecules = [mol, mol2]

    class CountingJaxBackend(JaxBackend):
        prepare_calls = 0

        def prepare(self, request: PreparationRequest) -> object:  # type: ignore[override]
            type(self).prepare_calls += 1
            return super().prepare(request)

    backend = CountingJaxBackend()

    ref = ObservationSet()
    # Frequency references for both cases so the JAX loss path is exercised.
    ref = ref.with_frequency(1000.0, data_idx=6, case_id="0")
    ref = ref.with_frequency(1000.0, data_idx=6, case_id="1")

    space = ActiveParameterSpace.all_active(layout, ff)
    plan = ObjectivePlan(
        case_ids=("0", "1"),
        molecules=tuple(molecules),
        stationary_points=(StationaryPointKind.GROUND_STATE, StationaryPointKind.GROUND_STATE),
        observations=ref,
        layout=layout,
        active_space=space,
    )
    executor = JaxObjectiveExecutor(plan, backend, ff)

    # Exactly one prepared session per case — no double preparation while
    # building the per-case JIT objective.
    assert backend.prepare_calls == len(molecules)
    assert set(executor._sessions) == {"0", "1"}


# ---------------------------------------------------------------------------
# Lazy catalog: proven in a fresh interpreter (no backend/jax/openmm imports)
# ---------------------------------------------------------------------------


def test_catalog_is_lazy_in_fresh_interpreter() -> None:
    """Listing the catalog in a fresh interpreter imports no backend modules.

    A subprocess starting from a clean interpreter imports only the registry,
    runs ``catalog()``, and asserts that neither the heavy backend libraries
    (``jax``, ``openmm``) nor the concrete backend modules were imported —
    proving the catalog is side-effect-free and lazy.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import q2mm.backends.registry as r\n"
        "statuses = r.catalog()\n"
        "assert statuses\n"
        "for m in ('jax', 'jaxlib', 'openmm',\n"
        "          'q2mm.backends.mm.jax_engine', 'q2mm.backends.mm.openmm',\n"
        "          'q2mm.backends.mm.jax_md_engine', 'q2mm.backends.mm.tinker',\n"
        "          'q2mm.backends.qm.psi4'):\n"
        "    assert m not in sys.modules, f'{m} was imported during catalog()'\n"
        "print('LAZY_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "LAZY_OK" in result.stdout


def test_dependency_probe_uses_only_find_spec_and_which(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dependency probe consults only importlib.find_spec and shutil.which."""
    import importlib.util
    import shutil

    import q2mm.backends.contracts as contracts

    find_spec_calls: list = []
    which_calls: list = []
    real_find_spec = importlib.util.find_spec
    real_which = shutil.which

    def spy_find_spec(name: str, *a: object, **k: object) -> object:
        find_spec_calls.append(name)
        return real_find_spec(name, *a, **k)  # type: ignore[arg-type]

    def spy_which(name: str, *a: object, **k: object) -> object:
        which_calls.append(name)
        return real_which(name, *a, **k)  # type: ignore[arg-type]

    monkeypatch.setattr(contracts.importlib.util, "find_spec", spy_find_spec)
    monkeypatch.setattr(contracts.shutil, "which", spy_which)

    healthy, _ = contracts.DependencyProbe(modules=("numpy",), executables=("python",)).check()
    assert isinstance(healthy, bool)
    assert "numpy" in find_spec_calls
    assert "python" in which_calls
