"""Tests for :mod:`q2mm.benchmarks.profiles`.

Cover RunProfile validation, deterministic canonical serialization +
fingerprint (including cross-process stability), candidate-ID uniqueness
across *every* omitted knob, and resolved-profile provenance completeness
(the resolved fingerprint tracks device / versions / data, and differs
from the logical requested fingerprint).
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys

import pytest

from q2mm.benchmarks.profiles import (
    OPTIMIZER_CATALOG,
    ResolvedProfile,
    RunProfile,
    canonical_fingerprint,
    canonical_json,
    resolve,
)


class TestRunProfileValidation:
    def test_uses_identity_equality_for_mapping_backed_state(self) -> None:
        first = RunProfile(system="ch3f", data_roots={"ch3f": "/tmp/data"})
        second = RunProfile(system="ch3f", data_roots={"ch3f": "/tmp/data"})

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_defaults(self) -> None:
        p = RunProfile(system="ch3f")
        assert p.backend == "jax"
        assert p.optimizer == "scipy-lbfgsb-jax"

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            RunProfile(system="ch3f").system = "other"  # type: ignore[misc]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"system": ""},
            {"system": "ch3f", "backend": ""},
            {"system": "ch3f", "functional_form": "bogus"},
            {"system": "ch3f", "starting_point": "bogus"},
            {"system": "ch3f", "workflow": "bogus"},
            {"system": "ch3f", "optimizer": "not-a-real-optimizer"},
            {"system": "ch3f", "maxiter": -1},
            {"system": "ch3f", "ftol": 0.0},
            {"system": "ch3f", "ftol": float("inf")},
            {"system": "ch3f", "fc_fraction": 0.0},
            {"system": "ch3f", "fc_fraction": 1.5},
            {"system": "ch3f", "regularization": -1.0},
            {"system": "ch3f", "n_evals": -1},
            {"system": "ch3f", "executor_ratio_tol": -1.0},
            {"system": "ch3f", "qfuerza_replace_with": 0.0},
            {"system": "ch3f", "qfuerza_replace_with": -1.0},
            {"system": "ch3f", "data_roots": {"bogus": "/x"}},
            {"system": "ch3f", "data_roots": {"ch3f": ""}},
        ],
    )
    def test_invalid_fields_raise(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            RunProfile(**kwargs)  # type: ignore[arg-type]

    def test_effective_regularization(self) -> None:
        # None (default) uses the catalog value; explicit value (incl. 0) wins.
        assert RunProfile(system="ch3f", optimizer="scipy-lbfgsb-l2").effective_regularization == pytest.approx(0.01)
        assert (
            RunProfile(system="ch3f", optimizer="scipy-lbfgsb-l2", regularization=0.5).effective_regularization == 0.5
        )
        assert (
            RunProfile(system="ch3f", optimizer="scipy-lbfgsb-l2", regularization=0.0).effective_regularization == 0.0
        )
        assert RunProfile(system="ch3f", optimizer="scipy-lbfgsb").effective_regularization == 0.0
        assert RunProfile(system="ch3f", optimizer="jaxopt-lbfgs").effective_regularization == pytest.approx(0.01)

    def test_l2_catalog_uses_python_scipy_fd(self) -> None:
        # scipy-lbfgsb-l2 uses the Python evaluator with SciPy internal FD.
        spec = OPTIMIZER_CATALOG["scipy-lbfgsb-l2"]
        assert spec.evaluator == "python" and spec.gradient_mode == "none"

    def test_invalid_new_knobs_raise(self) -> None:
        for kwargs in (
            {"learning_rate": 0.0},
            {"learning_rate": -1.0},
            {"max_params": 0},
            {"max_cycles": 0},
            {"convergence": 0.0},
            {"regularization": -1.0},
        ):
            with pytest.raises(ValueError):
                RunProfile(system="ch3f", **kwargs)  # type: ignore[arg-type]

    def test_catalog_gradient_modes(self) -> None:
        assert OPTIMIZER_CATALOG["scipy-lbfgsb"].gradient_mode == "none"
        assert OPTIMIZER_CATALOG["scipy-lbfgsb-fd"].gradient_mode == "finite_difference"
        assert OPTIMIZER_CATALOG["scipy-lbfgsb-jax"].gradient_mode == "analytical"
        # l2 variants restored.
        assert "scipy-lbfgsb-l2" in OPTIMIZER_CATALOG and "optax-adam-l2" in OPTIMIZER_CATALOG

    def test_optimizer_spec_extra_is_frozen(self) -> None:
        spec = OPTIMIZER_CATALOG["optax-adam-cosine"]
        with pytest.raises(TypeError):
            spec.extra["schedule"] = "linear"  # type: ignore[index]


class TestFingerprintDeterminism:
    def test_canonical_json_sorts_keys(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_label_excluded_from_identity(self) -> None:
        assert (
            RunProfile(system="ch3f", label="one").fingerprint() == RunProfile(system="ch3f", label="two").fingerprint()
        )

    def test_fingerprint_stable_across_process_and_hashseed(self) -> None:
        code = (
            "from q2mm.benchmarks.profiles import RunProfile;"
            "print(RunProfile(system='ch3f', optimizer='optax-adam', seed=7, data_roots={'ch3f':'/a'}).fingerprint())"
        )
        expected = RunProfile(system="ch3f", optimizer="optax-adam", seed=7, data_roots={"ch3f": "/a"}).fingerprint()
        outputs = set()
        for seed in ("0", "1", "12345"):
            env = {**os.environ, "PYTHONHASHSEED": seed}
            outputs.add(subprocess.check_output([sys.executable, "-c", code], env=env, text=True).strip())
        assert outputs == {expected}


class TestCandidateIdCollisions:
    """Two profiles differing in *any* knob must not share a candidate ID."""

    _BASE = dict(system="ch3f", backend="jax", functional_form="harmonic", optimizer="scipy-lbfgsb-jax")

    @pytest.mark.parametrize(
        "override",
        [
            {"workflow": "method-e2"},
            {"maxiter": 7},
            {"ftol": 1e-10},
            {"fc_fraction": 0.2},
            {"eq_fraction": 0.05},
            {"regularization": 0.01},
            {"learning_rate": 0.02},
            {"max_params": 5},
            {"max_cycles": 20},
            {"convergence": 0.001},
            {"n_evals": 3},
            {"executor_ratio_tol": 0.15},
            {"skip_optimization": True},
            {"qfuerza_replace_with": 2.0},
            {"platform": "CUDA"},
            {"seed": 99},
            {"data_roots": {"ch3f": "/other"}},
            {"starting_point": "published"},
        ],
    )
    def test_distinct_knobs_distinct_ids(self, override: dict[str, object]) -> None:
        base = RunProfile(**self._BASE)  # type: ignore[arg-type]
        other = RunProfile(**{**self._BASE, **override})  # type: ignore[arg-type]
        assert base.candidate_id() != other.candidate_id(), f"{override} collided"
        assert base.fingerprint() != other.fingerprint()

    def test_regularization_none_vs_zero_distinct(self) -> None:
        # An explicit 0 (disable an L2 preset) differs from the unset default.
        a = RunProfile(system="ch3f", optimizer="scipy-lbfgsb-l2", regularization=None)
        b = RunProfile(system="ch3f", optimizer="scipy-lbfgsb-l2", regularization=0.0)
        assert a.candidate_id() != b.candidate_id()

    def test_identical_profiles_share_identity(self) -> None:
        assert RunProfile(**self._BASE).candidate_id() == RunProfile(**self._BASE).candidate_id()  # type: ignore[arg-type]

    def test_candidate_id_is_readable_prefix_plus_full_hash(self) -> None:
        cid = RunProfile(system="ch3f", backend="jax", functional_form="harmonic").candidate_id()
        prefix, _, suffix = cid.partition("__")
        assert prefix == "ch3f_jax_harmonic_scipy-lbfgsb-jax_qfuerza"
        # The full 64-char SHA-256 hex digest, so the ID is genuinely collision-free.
        assert len(suffix) == 64 and all(c in "0123456789abcdef" for c in suffix)
        assert ":" not in cid and " " not in cid


def _resolve(profile: RunProfile, **overrides: object) -> ResolvedProfile:
    from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole, Capability
    from q2mm.backends.registry import get_descriptor

    info = BackendInfo(
        name="jax",
        role=BackendRole.MM,
        capabilities=frozenset({Capability.ENERGY, Capability.FREQUENCIES}),
        functional_forms=frozenset({"harmonic", "mm3"}),
        provenance=BackendProvenance(backend="jax", role=BackendRole.MM, version="0.9.0", detail="cpu"),
    )
    kwargs: dict[str, object] = {
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
    kwargs.update(overrides)
    return resolve(profile, **kwargs)  # type: ignore[arg-type]


class TestResolvedProfile:
    def test_uses_identity_equality_for_resolved_mappings(self) -> None:
        first = _resolve(RunProfile(system="ch3f"))
        second = _resolve(RunProfile(system="ch3f"))

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_provenance_is_complete(self) -> None:
        prov = _resolve(RunProfile(system="ch3f")).to_dict()
        required = {
            "profile_fingerprint",
            "resolved_fingerprint",
            "static_descriptor",
            "runtime_backend_key",
            "backend_name",
            "backend_version",
            "backend_detail",
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
        }
        assert required <= set(prov)
        # The complete static descriptor identity is nested, not flattened.
        sd = prov["static_descriptor"]
        assert sd["api_version"] == 1
        assert sd["factory"].endswith("JaxBackend")
        assert sd["name"] == "jax"
        assert set(sd) >= {
            "name",
            "api_version",
            "factory",
            "probe_modules",
            "probe_executables",
            "role",
            "capabilities",
            "functional_forms",
            "provenance",
        }
        assert prov["runtime_backend_key"] == "jax"
        assert prov["backend_version"] == "0.9.0"

    def test_resolved_fingerprint_tracks_device(self) -> None:
        base = _resolve(RunProfile(system="ch3f"))
        mutated = dataclasses.replace(base, device={"jax_devices": ["cuda:0"]})
        assert base.fingerprint() != mutated.fingerprint()

    def test_resolved_fingerprint_tracks_versions(self) -> None:
        base = _resolve(RunProfile(system="ch3f"))
        mutated = dataclasses.replace(base, dependency_versions={"jax": "9.9"})
        assert base.fingerprint() != mutated.fingerprint()

    def test_resolved_fingerprint_tracks_data(self) -> None:
        base = _resolve(RunProfile(system="ch3f"))
        other = _resolve(RunProfile(system="ch3f"), data_provenance={"metadata": {"doi": "10.0/y"}, "hessians": []})
        assert base.fingerprint() != other.fingerprint()

    def test_resolved_id_embeds_concrete_form(self) -> None:
        cid = _resolve(RunProfile(system="ch3f", functional_form=None), functional_form="mm3").candidate_id()
        assert cid.startswith("ch3f_jax_mm3_")

    def test_injected_backend_uses_placeholder_descriptor(self) -> None:
        from q2mm.backends.contracts import BackendInfo, BackendProvenance, BackendRole

        info = BackendInfo(
            name="fake",
            role=BackendRole.MM,
            functional_forms=frozenset({"harmonic"}),
            provenance=BackendProvenance(backend="fake", role=BackendRole.MM),
        )
        r = _resolve(RunProfile(system="ch3f", backend="fake"), descriptor=None, backend_info=info)
        assert r.static_descriptor["factory"] == "<injected>"
        assert r.static_descriptor["name"] == "fake"

    def test_optimizer_settings_frozen(self) -> None:
        r = _resolve(RunProfile(system="ch3f"))
        with pytest.raises(TypeError):
            r.optimizer_settings["kind"] = "changed"  # type: ignore[index]

    def test_canonical_fingerprint_helper(self) -> None:
        assert canonical_fingerprint({"a": 1}) == canonical_fingerprint({"a": 1})
        assert canonical_fingerprint({"a": 1}) != canonical_fingerprint({"a": 2})
