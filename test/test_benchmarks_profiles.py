"""Typed requested-profile v2 and unchanged explicit resolved/scientific identities."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import JsonValue, ValidationError

from q2mm.backends.contracts import (
    BackendDescriptor,
    BackendInfo,
    BackendProvenance,
    BackendRole,
    BackendUnavailableError,
    DependencyProbe,
)
from q2mm.benchmarks.profiles import (
    PROFILE_VERSION,
    ConfigurationError,
    RunProfile,
    _classify_backend,
    canonical_fingerprint,
    canonical_json,
    catalog_settings,
    recommended_publication_profile,
    resolve,
)
from q2mm.benchmarks.publications import FERROCENE_SEVEN_STRUCTURE_PROFILE, REPOSITORY_OBJECTIVE_PROFILE
from q2mm.optimizers.catalog import OPTIMIZER_CATALOG
from q2mm.settings import BackendSettings, MethodE2Settings, ObjectiveSettings, OptaxSettings, ScipySettings


@pytest.fixture
def configured_registry(monkeypatch: pytest.MonkeyPatch) -> tuple[BackendDescriptor, SimpleNamespace, Mock]:
    from q2mm.backends import registry
    from q2mm.backends.discovery import DiscoverySnapshot

    key = "configured-plugin"
    info = BackendInfo(
        name=key,
        role=BackendRole.MM,
        provenance=BackendProvenance(backend=key, role=BackendRole.MM),
    )
    backend = SimpleNamespace(info=info, prepare=Mock(side_effect=AssertionError("scientific computation")))

    def factory(*, configured_path: str | None = None, config: dict[str, JsonValue] | None = None) -> SimpleNamespace:
        if config is not None:
            json.dumps(config, allow_nan=False)
            assert type(config) is dict
            assert type(config["modes"]) is list
            assert type(config["modes"][1]) is dict
        if configured_path is None:
            raise BackendUnavailableError("configured dependency missing")
        if configured_path == "invalid":
            raise ValueError("configured factory failed")
        if configured_path == "wrong-role":
            return SimpleNamespace(
                info=dataclasses.replace(
                    info,
                    role=BackendRole.REFERENCE,
                    provenance=BackendProvenance(backend=key, role=BackendRole.REFERENCE),
                ),
                prepare=backend.prepare,
            )
        assert configured_path == "explicit-location"
        return backend

    spy = Mock(wraps=factory)
    module = ModuleType("_q2mm_configured_backend_test")
    module.__dict__["factory"] = spy
    monkeypatch.setitem(sys.modules, module.__name__, module)
    descriptor = BackendDescriptor(
        name=key,
        role=BackendRole.MM,
        capability_ceiling=frozenset(),
        functional_form_ceiling=frozenset({"harmonic"}),
        factory=f"{module.__name__}:factory",
        probe=DependencyProbe(executables=("configured-tool-not-on-path",)),
    )
    monkeypatch.setattr("q2mm.backends.contracts.shutil.which", lambda _name: None)
    monkeypatch.setattr(registry, "_snapshot", DiscoverySnapshot(descriptors={key: descriptor}, records=()))
    monkeypatch.setattr(registry, "_load_records", {})
    return descriptor, backend, spy


def test_configured_registry_construction_overrides_generic_probe(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
) -> None:
    from q2mm.backends import registry

    descriptor, backend, factory = configured_registry
    assert descriptor.is_available()[0] is False
    assert registry.catalog(role=BackendRole.MM)[0].healthy is False
    factory.assert_not_called()
    profile = RunProfile(
        system="ch3f",
        backend=BackendSettings(key=descriptor.name, options={"configured_path": "explicit-location"}),
        objective=ObjectiveSettings(),
    )
    assert _classify_backend(profile) == (descriptor, backend)
    factory.assert_called_once_with(configured_path="explicit-location")
    assert registry.catalog(role=BackendRole.MM)[0].healthy is False
    assert factory.call_count == 1
    backend.prepare.assert_not_called()


def test_configured_factory_receives_recursively_serializable_options(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
) -> None:
    descriptor, backend, factory = configured_registry
    options = {
        "configured_path": "explicit-location",
        "config": {"threads": 2, "modes": ["fast", {"enabled": False, "limit": 0, "fallback": None}]},
    }
    profile = RunProfile(
        system="ch3f",
        backend=BackendSettings(key=descriptor.name, options=options),
        objective=ObjectiveSettings(),
    )
    assert descriptor.is_available()[0] is False
    assert _classify_backend(profile) == (descriptor, backend)
    factory.assert_called_once_with(**options)
    received = factory.call_args.kwargs
    assert json.loads(json.dumps(received)) == options
    received["config"]["modes"][1]["enabled"] = True
    assert profile.backend.options["config"]["modes"][1]["enabled"] is False
    backend.prepare.assert_not_called()


@pytest.mark.parametrize(
    ("options", "error", "message"),
    [
        ({}, BackendUnavailableError, "configured dependency missing"),
        ({"configured_path": "invalid"}, ConfigurationError, "configured factory failed"),
        ({"configured_path": "wrong-role"}, ConfigurationError, "runtime role"),
    ],
)
def test_actual_construction_preserves_missing_failure_and_role_classification(
    configured_registry: tuple[BackendDescriptor, SimpleNamespace, Mock],
    options: dict[str, str],
    error: type[Exception],
    message: str,
) -> None:
    descriptor, _backend, factory = configured_registry
    profile = RunProfile(
        system="ch3f", backend=BackendSettings(key=descriptor.name, options=options), objective=ObjectiveSettings()
    )
    with pytest.raises(error, match=message):
        _classify_backend(profile)
    factory.assert_called_once_with(**options)


def test_defaults_share_actual_execution_models() -> None:
    profile = RunProfile(system="ch3f")
    assert profile.backend.key == "jax"
    assert profile.optimizer == ScipySettings(verbose=False)
    assert profile.objective == ObjectiveSettings(executor="jax", gradient_mode="analytical")
    assert profile.objective_profile is profile.effective_objective_profile is None
    assert RunProfile(system="rh-enamide").effective_objective_profile == REPOSITORY_OBJECTIVE_PROFILE
    with pytest.raises(ValidationError, match="frozen"):
        profile.system = "other"
    with pytest.raises(ValidationError, match="frozen"):
        profile.optimizer.maxiter = 10


def test_profile_composition_retains_shared_backend_executor_validation() -> None:
    with pytest.raises(ValueError, match="The JAX objective requires backend='jax'"):
        RunProfile(system="ch3f", backend=BackendSettings(key="plugin"))
    profile = RunProfile(
        system="ch3f",
        backend=BackendSettings(key="plugin"),
        objective=ObjectiveSettings(executor="python"),
    )
    assert RunProfile.model_validate_json(profile.model_dump_json()).objective == profile.objective


@pytest.mark.parametrize(
    "values",
    [
        {"system": ""},
        {"backend": {"key": ""}},
        {"functional_form": "bogus"},
        {"starting_point": "bogus"},
        {"objective_profile": "bogus"},
        {"workflow": "bogus"},
        {"optimizer": "not-a-model"},
        {"optimizer": {"maxiter": -1}},
        {"optimizer": {"ftol": float("inf")}},
        {"optimizer": {"fc_fraction": 0.0}},
        {"optimizer": {"fc_fraction": 1.5}},
        {"objective": {"regularization": -1.0}},
        {"n_evals": -1},
        {"ratio_tol": -1.0},
        {"qfuerza_replace_with": 0.0},
        {"data_roots": {"bogus": "/x"}},
        {"data_roots": {"ch3f": " "}},
        {"maxiter": 3},
        {"regularization": 0.0},
    ],
)
def test_invalid_or_superseded_scalar_fields_raise(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RunProfile(**{"system": "ch3f", **values})


@pytest.mark.parametrize("key", tuple(OPTIMIZER_CATALOG))
def test_catalog_is_an_actual_typed_policy_not_a_second_resolver(key: str) -> None:
    policy = catalog_settings(key)
    profile = RunProfile(system="ch3f", **policy.model_dump())
    spec = OPTIMIZER_CATALOG[key]
    assert profile.optimizer == spec.settings
    assert profile.objective == spec.objective
    assert profile.effective_regularization == spec.objective.regularization


def test_partial_defaults_and_family_replacement_are_native() -> None:
    profile = RunProfile(system="ch3f", optimizer={"maxiter": 0}, n_evals=0)
    assert profile.optimizer == ScipySettings(maxiter=0, verbose=False)
    replaced = RunProfile(system="ch3f", optimizer={"kind": "optax", "max_steps": 0})
    assert replaced.optimizer == OptaxSettings(max_steps=0)
    assert "maxiter" not in replaced.optimizer.model_dump()


def test_zero_regularization_and_fd_are_authoritative() -> None:
    policy = catalog_settings("scipy-lbfgsb-l2")
    profile = RunProfile(
        system="ch3f",
        optimizer=policy.optimizer,
        objective=ObjectiveSettings(gradient_mode="finite_difference", fd_step=0.02, regularization=0.0),
    )
    assert profile.effective_regularization == 0.0
    assert profile.objective.fd_step == 0.02
    zero = RunProfile(system="ch3f", optimizer={"ftol": 0.0}, ratio_tol=0.0, n_evals=0)
    assert zero.optimizer.ftol == zero.ratio_tol == zero.n_evals == 0


def test_explicit_versioned_projection_excludes_presentation_and_private_origin() -> None:
    profile = RunProfile(system="ch3f")
    explicit = RunProfile(system="ch3f", **catalog_settings("scipy-lbfgsb-jax").model_dump())
    assert profile.canonical_dict()["version"] == PROFILE_VERSION == "benchmark-profile-v2"
    assert profile.canonical_dict().keys() == {
        "version",
        "system",
        "recipe_id",
        "backend",
        "functional_form",
        "starting_point",
        "objective_profile",
        "workflow",
        "optimizer",
        "objective",
        "n_evals",
        "ratio_tol",
        "skip_optimization",
        "qfuerza_replace_with",
        "data_roots",
    }
    assert profile.fingerprint() == RunProfile(system="ch3f", label="different").fingerprint()
    explicit_equal = RunProfile(**profile.model_dump())
    assert explicit_equal.fingerprint() == profile.fingerprint()
    assert RunProfile.model_validate_json(profile.model_dump_json()).fingerprint() == profile.fingerprint()
    assert "requested_fields" not in profile.canonical_dict()
    assert profile.optimizer == explicit.optimizer


@pytest.mark.parametrize(
    "values",
    [
        {"optimizer": ScipySettings(maxiter=7, verbose=False)},
        {"optimizer": OptaxSettings(max_steps=0)},
        {"workflow": MethodE2Settings()},
        {"objective": ObjectiveSettings(executor="jax", gradient_mode="analytical", regularization=0.01)},
        {"backend": BackendSettings(key="plugin", options={"enabled": False}), "objective": ObjectiveSettings()},
        {"n_evals": 3},
        {"ratio_tol": 0.15},
        {"skip_optimization": True},
        {"qfuerza_replace_with": 2.0},
        {"data_roots": {"ch3f": "/other"}},
        {"starting_point": "published"},
        {"recipe_id": "different-policy"},
    ],
)
def test_exact_settings_change_requested_identity(values: dict[str, object]) -> None:
    assert RunProfile(system="ch3f", **values).fingerprint() != RunProfile(system="ch3f").fingerprint()


def test_publication_profile_identity_and_readable_full_hash() -> None:
    compatibility = RunProfile(system="ferrocene", starting_point="published")
    named = RunProfile(
        system="ferrocene",
        starting_point="published",
        objective_profile=FERROCENE_SEVEN_STRUCTURE_PROFILE,
    )
    assert compatibility.fingerprint() != named.fingerprint()
    prefix, _, suffix = RunProfile(system="ch3f", functional_form="harmonic").candidate_id().partition("__")
    assert prefix == "ch3f_jax_harmonic_scipy_qfuerza"
    assert len(suffix) == 64 and all(char in "0123456789abcdef" for char in suffix)


def test_fingerprint_stable_across_process_and_hashseed() -> None:
    code = (
        "from q2mm.benchmarks.profiles import RunProfile;"
        "print(RunProfile(system='ch3f', data_roots={'ch3f':'/a'}).fingerprint())"
    )
    expected = RunProfile(system="ch3f", data_roots={"ch3f": "/a"}).fingerprint()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        assert subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip() == expected
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert canonical_fingerprint({"a": 1}) != canonical_fingerprint({"a": 2})


def _resolve(profile: RunProfile, **overrides: object) -> object:
    from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability
    from q2mm.backends.registry import get_descriptor

    info = BackendInfo(
        name="jax",
        role=BackendRole.MM,
        capabilities=frozenset({Capability.ENERGY, Capability.FREQUENCIES}),
        functional_forms=frozenset({"harmonic", "mm3"}),
        provenance=BackendProvenance(
            backend="jax", role=BackendRole.MM, version="0.9.0", details={"platform": {"backend": "cpu"}}
        ),
    )
    kwargs = {
        "descriptor": get_descriptor("jax"),
        "backend_info": info,
        "functional_form": "harmonic",
        "evaluator": "jax",
        "gradient_mode": "analytical",
        "expected_result_gradient_mode": "analytical",
        "fd_step": None,
        "effective_regularization": 0.0,
        "optimizer_settings": {"kind": "scipy", "maxiter": 500},
        "workflow_settings": {"name": "single-stage"},
        "layout_fingerprint": "sha256:deadbeef",
        "n_active_params": 12,
        "n_full_params": 100,
        "n_molecules": 1,
        "data_provenance": {"metadata": {"doi": "10.0/x"}, "hessians": []},
        "resolved_data_roots": {},
        "include_device": False,
    }
    return resolve(profile, **{**kwargs, **overrides})


def test_resolved_identity_equality_and_complete_provenance() -> None:
    first = _resolve(RunProfile(system="ch3f"))
    second = _resolve(RunProfile(system="ch3f"))
    assert first != second and len({first, second}) == 2
    provenance = first.to_dict()
    assert {
        "profile_fingerprint",
        "resolved_fingerprint",
        "objective_profile",
        "reproduction_status",
        "publication_metadata",
        "static_descriptor",
        "runtime_backend_key",
        "backend_name",
        "backend_version",
        "backend_details",
        "capabilities",
        "backend_functional_forms",
        "functional_form",
        "evaluator",
        "gradient_mode",
        "expected_result_gradient_mode",
        "fd_step",
        "effective_regularization",
        "optimizer_method",
        "optimizer_settings",
        "workflow",
        "workflow_settings",
        "layout_fingerprint",
        "n_active_params",
        "n_full_params",
        "n_molecules",
        "data_provenance",
        "resolved_data_roots",
        "dependency_versions",
        "device",
        "seed",
        "settings",
    } <= provenance.keys()
    descriptor = provenance["static_descriptor"]
    assert descriptor["backend_api_version"] == 1 and descriptor["factory"].endswith("JaxBackend")
    assert provenance["runtime_backend_key"] == "jax" and provenance["backend_version"] == "0.9.0"
    with pytest.raises(TypeError):
        first.optimizer_settings["kind"] = "changed"


@pytest.mark.parametrize(
    "values",
    [
        {"device": {"jax_devices": ["cuda:0"]}},
        {"dependency_versions": {"jax": "9.9"}},
        {"data_provenance": {"metadata": {"doi": "10.0/y"}, "hessians": []}},
    ],
)
def test_resolved_fingerprint_tracks_actual_runtime_and_data(values: dict[str, object]) -> None:
    base = _resolve(RunProfile(system="ch3f"))
    assert base.fingerprint() != dataclasses.replace(base, **values).fingerprint()


def test_resolved_publication_metadata_and_injected_descriptor() -> None:
    publication = {
        "status": "partial_repository_reproduction",
        "objective_profile": {"identifier": REPOSITORY_OBJECTIVE_PROFILE},
    }
    resolved = _resolve(RunProfile(system="ch3f"), data_provenance={"publication_metadata": publication})
    assert resolved.to_dict()["reproduction_status"] == publication["status"]
    assert resolved.to_dict()["publication_metadata"] == publication
    injected = _resolve(
        RunProfile(system="ch3f", backend=BackendSettings(key="fake"), objective=ObjectiveSettings()), descriptor=None
    )
    assert injected.static_descriptor["factory"] == "<injected>"
    assert injected.static_descriptor["name"] == "fake"
    assert _resolve(RunProfile(system="ch3f"), functional_form="mm3").candidate_id().startswith("ch3f_jax_mm3_")


def test_publication_policies_keep_heck_and_published_defaults_explicit() -> None:
    generic = recommended_publication_profile("rh-enamide")
    heck = recommended_publication_profile("heck-relay")
    published = recommended_publication_profile("heck-relay", starting_point="published")
    assert (generic.optimizer.fc_fraction, generic.optimizer.eq_fraction, generic.optimizer.ftol) == (0.20, 0.05, 1e-12)
    assert (heck.optimizer.fc_fraction, heck.optimizer.eq_fraction, heck.optimizer.ftol) == (0.05, 0.05, 1e-12)
    assert published.optimizer == ScipySettings(verbose=False)
    assert heck.fingerprint() != published.fingerprint()
    assert recommended_publication_profile("ferrocene", starting_point="published").starting_point == "published"
    with pytest.raises(ValueError, match="no QFUERZA"):
        recommended_publication_profile("ferrocene", starting_point="qfuerza")


@pytest.mark.parametrize(
    ("system", "starting_point", "expected_ftol", "expected_fc", "expected_eq"),
    [
        ("heck-relay", "qfuerza", 1e-12, 0.05, 0.05),
        ("rh-enamide", "qfuerza", 1e-12, 0.20, 0.05),
        ("heck-relay", "published", 1e-8, None, None),
        ("rh-enamide", "published", 1e-8, None, None),
    ],
)
def test_partial_publication_optimizer_overrides_preserve_selected_policy(
    system: str,
    starting_point: str,
    expected_ftol: float,
    expected_fc: float | None,
    expected_eq: float | None,
) -> None:
    profile = recommended_publication_profile(system, starting_point=starting_point, optimizer={"maxiter": 10})
    assert profile.optimizer.maxiter == 10
    assert profile.optimizer.ftol == expected_ftol
    assert profile.optimizer.fc_fraction == expected_fc
    assert profile.optimizer.eq_fraction == expected_eq
    assert profile.optimizer.verbose is False
    assert RunProfile.model_validate_json(profile.model_dump_json()).fingerprint() == profile.fingerprint()


@pytest.mark.parametrize("system", ["heck-relay", "rh-enamide"])
def test_publication_partial_none_zero_false_and_typed_replacement(system: str) -> None:
    profile = recommended_publication_profile(
        system,
        optimizer={"maxiter": 0, "fc_fraction": None, "use_bounds": False},
    )
    assert profile.optimizer.maxiter == 0
    assert profile.optimizer.ftol == 1e-12
    assert profile.optimizer.fc_fraction is None
    assert profile.optimizer.eq_fraction == 0.05
    assert profile.optimizer.use_bounds is False
    replacement = ScipySettings(maxiter=10, use_bounds=False, verbose=True)
    assert recommended_publication_profile(system, optimizer=replacement).optimizer == replacement
    family = OptaxSettings(max_steps=0, learning_rate=0.0, verbose=False)
    assert recommended_publication_profile(system, optimizer=family).optimizer == family
    assert recommended_publication_profile(
        system, optimizer={"kind": "optax", "max_steps": 0}
    ).optimizer == OptaxSettings(
        max_steps=0,
    )


@pytest.mark.parametrize("system", ["heck-relay", "rh-enamide"])
def test_publication_partial_optimizer_override_is_validated(system: str) -> None:
    with pytest.raises(ValueError, match="maxiter"):
        recommended_publication_profile(system, optimizer={"maxiter": -1})
