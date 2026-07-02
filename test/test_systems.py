"""Tests for benchmark system reference-data helpers."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from test._shared import make_water

from q2mm.diagnostics import systems
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ReferenceData


def _make_water_with_hessian(diag_start: float = 0.01, diag_stop: float = 0.09) -> Q2MMMolecule:
    mol = make_water()
    mol.hessian = np.diag(np.linspace(diag_start, diag_stop, 3 * mol.n_atoms))
    return mol


class TestSystemReferenceConstruction:
    def test_from_molecules_adds_eigenmatrix_and_geometry_but_not_frequencies(self) -> None:
        mol = _make_water_with_hessian()
        qm_freqs = systems._qm_frequencies_from_hessian(mol.hessian, mol.symbols)

        ref = ReferenceData.from_molecules([mol])

        counts = Counter(value.kind for value in ref.values)
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36
        assert counts["bond_length"] == len(mol.bonds)
        assert counts["bond_angle"] == len(mol.angles)
        assert counts["frequency"] == 0
        assert sum(freq > 50.0 for freq in qm_freqs) > 0

    def test_from_molecules_accepts_custom_eigenvalue_weights(self) -> None:
        mol = _make_water_with_hessian(diag_stop=0.2)

        ref = ReferenceData.from_molecules(
            [mol],
            weights={
                "eig_i": 0.1,
                "eig_d_low": 0.2,
                "eig_d_high": 0.3,
                "eig_o": 0.4,
            },
        )

        # Custom weights propagate. With rigid-body exclusion, the first mode
        # (skip_first) and the six smallest-|eigenvalue| rigid modes get eig_i
        # (0.1); surviving diagonal modes get eig_d_low/eig_d_high (0.2/0.3).
        diag_weights = [value.weight for value in ref.values if value.kind == "eig_diagonal"]
        assert diag_weights[0] == 0.1
        assert set(diag_weights) <= {0.1, 0.2, 0.3}
        assert 0.1 in diag_weights  # excluded (rigid-body / reaction-coordinate) modes
        assert any(w in (0.2, 0.3) for w in diag_weights)  # at least one real mode kept

        # Off-diagonal couplings use eig_o (0.4), except those touching an
        # excluded mode, which are zero-weighted.
        offdiag_weights = {value.weight for value in ref.values if value.kind == "eig_offdiagonal"}
        assert offdiag_weights <= {0.0, 0.4}
        assert 0.4 in offdiag_weights

    def test_from_molecules_include_geometry_false(self) -> None:
        """include_geometry=False omits bond and angle references."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecules([mol], include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        # Eigenmatrix data should still be present
        assert counts["eig_diagonal"] == 9
        assert counts["eig_offdiagonal"] == 36

    def test_from_molecule_include_geometry_false(self) -> None:
        """Single-molecule factory also respects include_geometry=False."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecule(mol, include_geometry=False)

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] == 0
        assert counts["bond_angle"] == 0
        assert counts["eig_diagonal"] > 0

    def test_include_geometry_true_is_default(self) -> None:
        """Default behavior includes geometry references."""
        mol = _make_water_with_hessian()
        ref = ReferenceData.from_molecules([mol])

        counts = Counter(value.kind for value in ref.values)
        assert counts["bond_length"] > 0
        assert counts["bond_angle"] > 0


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
        sd_default = systems.load_system("rh-enamide")
        sd_explicit = systems.load_system("rh-enamide", starting_point="qfuerza")
        assert np.allclose(
            sd_default.forcefield.get_param_vector(),
            sd_explicit.forcefield.get_param_vector(),
        )
        assert sd_default.metadata["starting_point"] == "qfuerza"

    @pytest.mark.external_data
    def test_published_path_preserves_literature_values(self) -> None:
        """``starting_point="published"`` retains literature OPT values verbatim."""
        sd_pub = systems.load_system("rh-enamide", starting_point="published")
        # Audit should classify every active scalar as retained_published
        # (no QFUERZA overwrite was performed).
        audit = sd_pub.metadata["starting_point_audit"]
        assert audit["starting_point"] == "published"
        for ptype, bucket in audit["by_type"].items():
            assert bucket["qfuerza_overwritten"] == 0, (
                f"published path must not overwrite any scalars, "
                f"but {ptype} has {bucket['qfuerza_overwritten']} overwrites"
            )

    @pytest.mark.external_data
    def test_qfuerza_overwrites_opt_but_not_frozen(self) -> None:
        """QFUERZA path mutates active OPT scalars; frozen backbone untouched."""
        sd_pub = systems.load_system("rh-enamide", starting_point="published")
        sd_q = systems.load_system("rh-enamide", starting_point="qfuerza")

        vec_pub = sd_pub.forcefield.get_param_vector()
        vec_q = sd_q.forcefield.get_param_vector()
        mask = sd_pub.forcefield.active_mask

        # Frozen partition is identical
        assert np.array_equal(mask, sd_q.forcefield.active_mask)
        # Frozen scalars are bit-identical (no QFUERZA leakage into backbone)
        np.testing.assert_array_equal(vec_pub[~mask], vec_q[~mask])
        # At least some active scalars changed (QFUERZA did something)
        assert np.any(vec_pub[mask] != vec_q[mask])

    @pytest.mark.external_data
    def test_reference_data_unchanged_by_starting_point(self) -> None:
        """Reference data depends on QM, not on the starting FF."""
        sd_pub = systems.load_system("rh-enamide", starting_point="published")
        sd_q = systems.load_system("rh-enamide", starting_point="qfuerza")
        refs_pub = [(r.kind, float(r.value), float(r.weight)) for r in sd_pub.reference.values]
        refs_q = [(r.kind, float(r.value), float(r.weight)) for r in sd_q.reference.values]
        assert refs_pub == refs_q

    @pytest.mark.external_data
    def test_audit_classifies_scalars_consistently(self) -> None:
        """Audit counts must sum to the FF's scalar count, per type."""
        sd = systems.load_system("rh-enamide", starting_point="qfuerza")
        audit = sd.metadata["starting_point_audit"]
        assert audit["starting_point"] == "qfuerza"
        # Sum of all bucket entries equals total parameters
        total_from_buckets = sum(sum(bucket.values()) for bucket in audit["by_type"].values())
        assert total_from_buckets == sd.forcefield.n_params
        # Active + frozen = total
        assert audit["n_active"] + audit["n_frozen"] == sd.forcefield.n_params
        # QFUERZA overwrites at least one bond and one angle in rh-enamide
        assert audit["by_type"]["bond_fc"]["qfuerza_overwritten"] > 0
        assert audit["by_type"]["angle_fc"]["qfuerza_overwritten"] > 0
        # MM3 backbone bonds/angles are frozen — no QFUERZA writes there
        for ptype in ("bond_fc", "bond_eq", "angle_fc", "angle_eq"):
            bucket = audit["by_type"][ptype]
            assert bucket["frozen"] > bucket["qfuerza_overwritten"], (
                f"{ptype}: frozen={bucket['frozen']} <= overwritten={bucket['qfuerza_overwritten']}"
            )

    @pytest.mark.jax
    def test_qfuerza_is_noop_for_qfuerza_fresh_strategy(self) -> None:
        """For CH3F (qfuerza_fresh), starting_point='qfuerza' is a no-op."""
        from q2mm.backends.mm.jax_engine import JaxEngine

        engine = JaxEngine()
        sd_default = systems.load_system("ch3f", engine=engine)
        sd_explicit = systems.load_system("ch3f", engine=engine, starting_point="qfuerza")
        assert np.allclose(
            sd_default.forcefield.get_param_vector(),
            sd_explicit.forcefield.get_param_vector(),
        )
        # No QFUERZA *overwrite* happened (the FF was already QFUERZA-derived
        # by the qfuerza_fresh strategy); the audit reflects that everything
        # active is "retained" relative to itself.
        audit = sd_explicit.metadata["starting_point_audit"]
        assert audit["starting_point"] == "qfuerza"
        for bucket in audit["by_type"].values():
            assert bucket["qfuerza_overwritten"] == 0

    def test_unknown_starting_point_raises(self) -> None:
        """Typos in ``starting_point`` must raise rather than silently passing through."""
        with pytest.raises(ValueError, match="Unknown starting_point"):
            systems.load_system("ch3f", starting_point="qferza")  # type: ignore[arg-type]

    @pytest.mark.external_data
    def test_qfuerza_replace_with_default_preserves_behavior(self) -> None:
        """Default ``qfuerza_replace_with=1.0`` must produce bit-identical FFs."""
        sd_default = systems.load_system("rh-enamide")
        sd_explicit = systems.load_system("rh-enamide", qfuerza_replace_with=1.0)
        np.testing.assert_array_equal(
            sd_default.forcefield.get_param_vector(),
            sd_explicit.forcefield.get_param_vector(),
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
        sd_default = systems.load_system("rh-enamide", qfuerza_replace_with=1.0)
        sd_alt = systems.load_system("rh-enamide", qfuerza_replace_with=0.03)
        vec_default = sd_default.forcefield.get_param_vector()
        vec_alt = sd_alt.forcefield.get_param_vector()
        labels = sd_default.forcefield.get_param_type_labels()
        diff = np.abs(vec_default - vec_alt)
        # At least one force-constant param must shift by > 10 internal units
        # (≈0.07 mdyn for bond_k, ≈0.07 mdyn·Å/rad² for angle_k) — a magnitude
        # large enough to rule out roundoff and verify the kwarg propagated.
        fc_indices = [i for i, lbl in enumerate(labels) if lbl in {"bond_k", "angle_k", "ub_k"}]
        assert fc_indices, "Expected at least one force-constant param in active set"
        max_fc_diff = float(np.max(diff[fc_indices]))
        assert max_fc_diff > 10.0, (
            f"Expected at least one force constant to shift by >10 internal units between "
            f"replace_with=1.0 and =0.03; got max FC diff = {max_fc_diff:.4f}"
        )
