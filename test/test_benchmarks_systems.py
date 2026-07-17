"""Tests for :mod:`q2mm.benchmarks.systems` — the benchmark-system registry.

Split out of the old ``test/test_systems.py`` (formerly testing
``q2mm.systems``, deleted in favour of ``q2mm.benchmarks.systems`` / one
module per scientific system). Two areas are covered:

- ``ExternalDataRoots`` and path resolution (``q2mm.benchmarks.systems._paths``).
- The ``starting_point`` kwarg on :func:`q2mm.benchmarks.systems.load_system`
  (``"qfuerza"`` vs ``"published"``), a Farrugia 2025 QFUERZA-workflow
  regression suite.

The previous custom zero-argument-loader extensibility test has no
equivalent here: the registry is a fixed key-to-module mapping (see
``q2mm.benchmarks.systems.SYSTEM_KEYS``), and each new system is added as
its own module rather than through a monkeypatched dictionary entry.
"""

from __future__ import annotations
from q2mm.backends.registry import load_backend

from pathlib import Path

import numpy as np
import pytest

from q2mm.benchmarks.systems import _paths


class TestExternalDataRoots:
    def test_missing_rh_enamide_root_has_configuration_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("Q2MM_RH_ENAMIDE", raising=False)
        with pytest.raises(FileNotFoundError, match=r"ExternalDataRoots\(rh_enamide"):
            _paths.resolve_rh_enamide_dir()

    def test_environment_roots_are_typed_paths(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        rh_dir = tmp_path / "rh"
        supporting_info = tmp_path / "supporting-info"
        mm3_base = tmp_path / "mm3_base.fld"
        rh_dir.mkdir()
        supporting_info.mkdir()
        mm3_base.write_text("licensed fixture", encoding="utf-8")
        monkeypatch.setenv("Q2MM_RH_ENAMIDE", str(rh_dir))
        monkeypatch.setenv("Q2MM_SUPPORTING_INFO", str(supporting_info))
        monkeypatch.setenv("Q2MM_MM3_BASE", str(mm3_base))

        roots = _paths.ExternalDataRoots.from_environment()

        assert roots == _paths.ExternalDataRoots(
            rh_enamide=rh_dir,
            supporting_info=supporting_info,
            mm3_base=mm3_base,
        )

    def test_explicit_supporting_info_root_overrides_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        explicit = tmp_path / "explicit"
        environment = tmp_path / "environment"
        explicit.mkdir()
        environment.mkdir()
        mm3_base = tmp_path / "mm3_base.fld"
        mm3_base.write_text("licensed fixture", encoding="utf-8")
        monkeypatch.setenv("Q2MM_SUPPORTING_INFO", str(environment))
        monkeypatch.setenv("Q2MM_MM3_BASE", str(mm3_base))
        roots = _paths.ExternalDataRoots(supporting_info=explicit)
        assert _paths.resolve_supporting_info_dir(roots) == explicit
        assert _paths.resolve_mm3_base_path(roots) == mm3_base

    def test_mm3_base_never_falls_back_to_repository_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("Q2MM_MM3_BASE", raising=False)
        with pytest.raises(FileNotFoundError, match="not distributed with q2mm"):
            _paths.resolve_mm3_base_path()

    def test_heck_relay_path_uses_environment_without_explicit_roots(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from q2mm.benchmarks.systems import heck_relay

        monkeypatch.setenv("Q2MM_SUPPORTING_INFO", str(tmp_path))
        expected = tmp_path / "rosales" / "Rosales_Anthony_Supporting_Information" / "Chapter3_Heck" / "mm3.FF1.fld"
        assert heck_relay._ff_path(None) == expected


class TestStartingPoint:
    """Regression tests for the ``starting_point`` kwarg on load_system.

    Validates the Farrugia 2025 "QFUERZA Hessian-derived values on
    published FF skeleton" workflow:

    1. Default path is QFUERZA (the canonical Farrugia 2025 workflow).
    2. Published path (opt-in via ``starting_point="published"``)
       retains the literature OPT values verbatim — used to reproduce
       historical publication-baseline runs.
    3. QFUERZA path overwrites OPT bond/angle values but never touches
       the frozen MM3 backbone.
    4. Reference data is identical between the two paths — the
       starting point should only change the starting parameter values,
       not the optimization target.
    5. The audit dict honestly reports which scalars came from QFUERZA
       vs which retained their published values (e.g. vdW, unmatched
       bond/angle rows).

    """

    @pytest.mark.external_data
    def test_default_starting_point_is_qfuerza(self) -> None:
        """Default ``starting_point="qfuerza"`` matches the explicit kwarg."""
        from q2mm.benchmarks.systems import load_system

        case_default = load_system("rh-enamide")
        case_explicit = load_system("rh-enamide", starting_point="qfuerza")
        layout = case_default.problem.layout
        assert np.allclose(
            layout.vector(case_default.problem.starting_force_field),
            layout.vector(case_explicit.problem.starting_force_field),
        )
        assert case_default.metadata["starting_point"] == "qfuerza"

    @pytest.mark.external_data
    def test_published_path_preserves_literature_values(self) -> None:
        """``starting_point="published"`` retains literature OPT values verbatim."""
        from q2mm.benchmarks.systems import load_system

        case_pub = load_system("rh-enamide", starting_point="published")
        # Audit should classify every active scalar as retained_published
        # (no QFUERZA overwrite was performed).
        audit = case_pub.metadata["starting_point_audit"]
        assert audit["starting_point"] == "published"
        for ptype, bucket in audit["by_type"].items():
            assert bucket["qfuerza_overwritten"] == 0, (
                f"published path must not overwrite any scalars, "
                f"but {ptype} has {bucket['qfuerza_overwritten']} overwrites"
            )

    @pytest.mark.external_data
    def test_qfuerza_overwrites_opt_but_not_frozen(self) -> None:
        """QFUERZA path mutates active OPT scalars; frozen backbone untouched."""
        from q2mm.benchmarks.systems import load_system

        case_pub = load_system("rh-enamide", starting_point="published")
        case_q = load_system("rh-enamide", starting_point="qfuerza")

        layout = case_pub.problem.layout
        vec_pub = layout.vector(case_pub.problem.starting_force_field)
        vec_q = layout.vector(case_q.problem.starting_force_field)
        active_pub = set(case_pub.problem.active_space.active_indices.tolist())
        active_q = set(case_q.problem.active_space.active_indices.tolist())
        mask = np.zeros(len(layout), dtype=bool)
        mask[list(active_pub)] = True

        # Frozen partition is identical
        assert active_pub == active_q
        # Frozen scalars are bit-identical (no QFUERZA leakage into backbone)
        np.testing.assert_array_equal(vec_pub[~mask], vec_q[~mask])
        # At least some active scalars changed (QFUERZA did something)
        assert np.any(vec_pub[mask] != vec_q[mask])

    @pytest.mark.external_data
    def test_reference_data_unchanged_by_starting_point(self) -> None:
        """Reference data depends on QM, not on the starting FF."""
        from q2mm.benchmarks.systems import load_system

        case_pub = load_system("rh-enamide", starting_point="published")
        case_q = load_system("rh-enamide", starting_point="qfuerza")
        refs_pub = [(r.kind, float(r.value), float(r.weight)) for r in case_pub.problem.observations.values]
        refs_q = [(r.kind, float(r.value), float(r.weight)) for r in case_q.problem.observations.values]
        assert refs_pub == refs_q

    @pytest.mark.external_data
    def test_audit_classifies_scalars_consistently(self) -> None:
        """Audit counts must sum to the FF's scalar count, per type."""
        from q2mm.benchmarks.systems import load_system

        case = load_system("rh-enamide", starting_point="qfuerza")
        audit = case.metadata["starting_point_audit"]
        assert audit["starting_point"] == "qfuerza"
        n_total = len(case.problem.layout)
        # Sum of all bucket entries equals total parameters
        total_from_buckets = sum(sum(bucket.values()) for bucket in audit["by_type"].values())
        assert total_from_buckets == n_total
        # Active + frozen = total
        assert audit["n_active"] + audit["n_frozen"] == n_total
        # QFUERZA overwrites at least one bond and one angle in rh-enamide
        assert audit["by_type"]["bond_k"]["qfuerza_overwritten"] > 0
        assert audit["by_type"]["angle_k"]["qfuerza_overwritten"] > 0
        # MM3 backbone bonds/angles are frozen — no QFUERZA writes there
        for ptype in ("bond_k", "bond_eq", "angle_k", "angle_eq"):
            bucket = audit["by_type"][ptype]
            assert bucket["frozen"] > bucket["qfuerza_overwritten"], (
                f"{ptype}: frozen={bucket['frozen']} <= overwritten={bucket['qfuerza_overwritten']}"
            )

    @pytest.mark.jax
    def test_qfuerza_is_noop_for_qfuerza_fresh_strategy(self) -> None:
        """For CH3F (qfuerza_fresh), starting_point='qfuerza' is a no-op."""
        from q2mm.benchmarks.systems import load_system

        backend = load_backend("jax")
        case_default = load_system("ch3f", backend=backend, functional_form="harmonic")
        case_explicit = load_system("ch3f", backend=backend, starting_point="qfuerza", functional_form="harmonic")
        layout = case_default.problem.layout
        assert np.allclose(
            layout.vector(case_default.problem.starting_force_field),
            layout.vector(case_explicit.problem.starting_force_field),
        )
        # No QFUERZA *overwrite* happened (the FF was already QFUERZA-derived
        # by the qfuerza_fresh strategy); the audit reflects that everything
        # active is "retained" relative to itself.
        audit = case_explicit.metadata["starting_point_audit"]
        assert audit["starting_point"] == "qfuerza"
        for bucket in audit["by_type"].values():
            assert bucket["qfuerza_overwritten"] == 0

    def test_unknown_starting_point_raises(self) -> None:
        """Typos in ``starting_point`` must raise rather than silently passing through.

        Any placeholder ``backend`` works — validation happens before the
        backend or molecule is ever touched.
        """
        from q2mm.benchmarks.systems import load_system

        with pytest.raises(ValueError, match="Unknown starting_point"):
            load_system("ch3f", backend=object(), starting_point="qferza", functional_form="harmonic")  # type: ignore[arg-type]

    @pytest.mark.external_data
    def test_qfuerza_replace_with_default_preserves_behavior(self) -> None:
        """Default ``qfuerza_replace_with=1.0`` must produce bit-identical FFs."""
        from q2mm.benchmarks.systems import load_system

        case_default = load_system("rh-enamide")
        case_explicit = load_system("rh-enamide", qfuerza_replace_with=1.0)
        layout = case_default.problem.layout
        np.testing.assert_array_equal(
            layout.vector(case_default.problem.starting_force_field),
            layout.vector(case_explicit.problem.starting_force_field),
        )

    @pytest.mark.external_data
    def test_qfuerza_replace_with_smaller_value_changes_starting_ff(self) -> None:
        """``qfuerza_replace_with=0.03`` (Method D 'natural' value) changes the QFUERZA-overwritten params.

        Guards the plumbing rather than specific physics: ensures the kwarg
        actually reaches ``invert_ts_curvature`` by asserting that several
        active force-constant params change measurably between the two
        settings.  Phase 9.1 empirical findings on rh-enamide show ~19
        params shift, with the reaction-coordinate bond (C2-HX, Hp.Ch in
        Farrugia 2025 notation) changing by ~270 internal units (kcal mol⁻¹ Å⁻²) ≈ ~1.9 mdyn/Å.
        """
        from q2mm.benchmarks.systems import load_system
        from q2mm.models.parameters import ParameterKind

        case_default = load_system("rh-enamide", qfuerza_replace_with=1.0)
        case_alt = load_system("rh-enamide", qfuerza_replace_with=0.03)
        layout = case_default.problem.layout
        vec_default = layout.vector(case_default.problem.starting_force_field)
        vec_alt = layout.vector(case_alt.problem.starting_force_field)
        kinds = layout.kinds
        diff = np.abs(vec_default - vec_alt)
        # At least one force-constant param must shift by > 10 internal units
        # (≈0.07 mdyn for bond_k, ≈0.07 mdyn·Å/rad² for angle_k) — a magnitude
        # large enough to rule out roundoff and verify the kwarg propagated.
        fc_kinds = {
            ParameterKind.BOND_FORCE_CONSTANT,
            ParameterKind.ANGLE_FORCE_CONSTANT,
            ParameterKind.UREY_BRADLEY_FORCE_CONSTANT,
        }
        fc_indices = [i for i, k in enumerate(kinds) if k in fc_kinds]
        assert fc_indices, "Expected at least one force-constant param in active set"
        max_fc_diff = float(np.max(diff[fc_indices]))
        assert max_fc_diff > 10.0, (
            f"Expected at least one force constant to shift by >10 internal units between "
            f"replace_with=1.0 and =0.03; got max FC diff = {max_fc_diff:.4f}"
        )


# ---------------------------------------------------------------------------
# StationaryPointKind -> invert_ts_curvature routing (domain-review Finding #5)
# ---------------------------------------------------------------------------


class TestStationaryPointRoutesInversion:
    """``StationaryPointKind`` must actually route ``invert_ts_curvature``.

    Regression guard: both ``assemble_published_case`` and
    ``assemble_qfuerza_fresh_case`` used to hardcode
    ``invert_ts_curvature=True`` regardless of *stationary_point* — silently
    corrupting genuine ground-state Hessians (which routinely carry tiny
    spurious negative eigenvalues from numerical noise) with a physically
    huge (1.0 Hartree/Bohr²) replacement value. Both CH3F (ground state)
    and CH3F-SN2 (transition state) are packaged with the repo, so this
    spies on the real generic ``qfuerza_fresh`` call (via ``wraps=``, so the
    real computation still runs) without needing any external dataset.
    """

    @pytest.mark.jax
    def test_ground_state_does_not_invert(self) -> None:
        """CH3F (StationaryPointKind.GROUND_STATE) must call with invert_ts_curvature=False."""
        from unittest.mock import patch

        from q2mm import preparation
        from q2mm.benchmarks.systems import load_system

        real_qfuerza_fresh = preparation.qfuerza_fresh
        with patch.object(preparation, "qfuerza_fresh", wraps=real_qfuerza_fresh) as spy:
            load_system("ch3f", backend=load_backend("jax"), functional_form="harmonic")

        assert spy.call_count == 1
        assert spy.call_args.kwargs["invert_ts_curvature"] is False

    @pytest.mark.jax
    def test_transition_state_inverts(self) -> None:
        """CH3F-SN2 (StationaryPointKind.TRANSITION_STATE) must call with invert_ts_curvature=True."""
        from unittest.mock import patch

        from q2mm import preparation
        from q2mm.benchmarks.systems import load_system

        real_qfuerza_fresh = preparation.qfuerza_fresh
        with patch.object(preparation, "qfuerza_fresh", wraps=real_qfuerza_fresh) as spy:
            load_system("ch3f-sn2", backend=load_backend("jax"), functional_form="harmonic")

        assert spy.call_count == 1
        assert spy.call_args.kwargs["invert_ts_curvature"] is True

    @pytest.mark.jax
    def test_ch3f_qm_hessian_actually_has_tiny_negative_eigenvalues(self) -> None:
        """Sanity check that CH3F is a *realistic* regression case.

        If CH3F's raw QM Hessian had no negative eigenvalues at all, the
        GS-must-not-invert routing test above would pass trivially (there
        would be nothing to corrupt either way). This confirms the QM
        Hessian genuinely carries the tiny spurious negative eigenvalues
        (numerical noise, not a real imaginary mode) that motivated
        Finding #5 — so ``test_ground_state_does_not_invert`` is a
        meaningful regression guard, not a vacuous one.
        """
        import numpy as np

        from q2mm.benchmarks.systems import load_system

        case = load_system("ch3f", backend=load_backend("jax"), functional_form="harmonic")
        hessian = case.problem.cases[0].molecule.hessian
        assert hessian is not None
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert eigenvalues.min() < 0.0, (
            "Expected CH3F's QM Hessian to carry at least one tiny spurious "
            "negative eigenvalue (the motivating case for Finding #5); if "
            "this no longer holds, the ground-state-routing regression "
            "test above should be revisited."
        )


class TestCh3fLoaderRequiresFunctionalForm:
    """CH3F/CH3F-SN2 genuinely support both harmonic and MM3 — no default.

    Regression guard: ``ch3f.py``/``ch3f_sn2.py``'s ``load()`` used to
    default ``functional_form=None``, and ``assemble_qfuerza_fresh_case``
    then silently fell back to whatever ``qfuerza_fresh`` happened to
    hardcode internally. Since CH3F is evaluated by both harmonic
    (JAX/JAX-MD) and MM3 (OpenMM/Tinker) backends with no single
    scientifically correct default, every caller must decide explicitly.
    """

    @pytest.mark.jax
    def test_ch3f_rejects_omitted_functional_form(self) -> None:
        from q2mm.benchmarks.systems import load_system

        with pytest.raises(TypeError, match="functional_form"):
            load_system("ch3f", backend=load_backend("jax"))  # type: ignore[call-arg]

    @pytest.mark.jax
    def test_ch3f_sn2_rejects_omitted_functional_form(self) -> None:
        from q2mm.benchmarks.systems import load_system

        with pytest.raises(TypeError, match="functional_form"):
            load_system("ch3f-sn2", backend=load_backend("jax"))  # type: ignore[call-arg]

    @pytest.mark.jax
    @pytest.mark.parametrize("form", ["harmonic", "mm3"])
    def test_ch3f_preserves_requested_functional_form(self, form: str) -> None:
        from q2mm.benchmarks.systems import load_system
        from q2mm.models.forcefield import FunctionalForm

        case = load_system("ch3f", backend=load_backend("jax"), functional_form=form)
        assert case.problem.starting_force_field.functional_form is FunctionalForm(form)
        assert case.metadata["functional_form"] == form

    @pytest.mark.jax
    @pytest.mark.parametrize("form", ["harmonic", "mm3"])
    def test_ch3f_sn2_preserves_requested_functional_form(self, form: str) -> None:
        from q2mm.benchmarks.systems import load_system
        from q2mm.models.forcefield import FunctionalForm

        case = load_system("ch3f-sn2", backend=load_backend("jax"), functional_form=form)
        assert case.problem.starting_force_field.functional_form is FunctionalForm(form)
        assert case.metadata["functional_form"] == form
