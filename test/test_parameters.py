"""Tests for the one parameter-vector layout and active/frozen projection.

Covers all term families (bonds, angles, torsions, stretch-bends, vdW,
Urey-Bradley), the fixed full-vector order, duplicate semantic-ID
disambiguation via occurrence counters, dimension/round-trip
invariants, frozen/active preservation through :class:`ActiveParameterSpace`,
and the deterministic, value-free :attr:`ParameterLayout.fingerprint`
(including a subprocess check across different ``PYTHONHASHSEED`` values).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    ForceField,
    StretchBendParam,
    TorsionParam,
    VdwParam,
    FunctionalForm,
)
from q2mm.models.parameters import (
    ActiveParameterSpace,
    ParameterId,
    ParameterKind,
    ParameterLayout,
    ParameterSlot,
    ParameterUnit,
    fractional_bounds,
    opt_substructure_membership,
)


def _full_force_field() -> ForceField:
    """One of every term family, including a Urey-Bradley angle."""
    return ForceField(
        name="test-ff",
        bonds=(
            BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),
            BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
        ),
        angles=(
            AngleParam(elements=("F", "C", "H"), force_constant=60.0, equilibrium=109.5),
            AngleParam(
                elements=("H", "C", "H"),
                force_constant=55.0,
                equilibrium=109.5,
                ub_force_constant=12.0,
                ub_equilibrium=1.8,
            ),
        ),
        torsions=(TorsionParam(elements=("F", "C", "H", "H"), periodicity=3, force_constant=0.1),),
        stretch_bends=(StretchBendParam(elements=("F", "C", "H"), force_constant=0.2),),
        vdws=(
            VdwParam(atom_type="1", radius=1.9, epsilon=0.03, element="C"),
            VdwParam(atom_type="5", radius=1.5, epsilon=0.02, element="H"),
        ),
        functional_form=FunctionalForm.HARMONIC,
    )


# ---------------------------------------------------------------------------
# Layout order / dimensions across all term families
# ---------------------------------------------------------------------------


class TestLayoutOrderAndDimensions:
    def test_full_vector_length_matches_legacy_slot_count(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        # 2 bonds * 2 + 2 angles * 2 + 1 torsion * 1 + 1 sb * 1 + 2 vdw * 2 + 1 UB angle * 2
        assert len(layout) == 2 * 2 + 2 * 2 + 1 * 1 + 1 * 1 + 2 * 2 + 1 * 2

    def test_kind_order_matches_fixed_legacy_order(self) -> None:
        """bonds(k,eq) -> angles(k,eq) -> torsions(k) -> sb(k) -> vdw(radius,epsilon) -> UB(k,eq) tail."""
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        expected = [
            ParameterKind.BOND_FORCE_CONSTANT,
            ParameterKind.BOND_EQUILIBRIUM,
            ParameterKind.BOND_FORCE_CONSTANT,
            ParameterKind.BOND_EQUILIBRIUM,
            ParameterKind.ANGLE_FORCE_CONSTANT,
            ParameterKind.ANGLE_EQUILIBRIUM,
            ParameterKind.ANGLE_FORCE_CONSTANT,
            ParameterKind.ANGLE_EQUILIBRIUM,
            ParameterKind.TORSION_FORCE_CONSTANT,
            ParameterKind.STRETCH_BEND_FORCE_CONSTANT,
            ParameterKind.VDW_RADIUS,
            ParameterKind.VDW_EPSILON,
            ParameterKind.VDW_RADIUS,
            ParameterKind.VDW_EPSILON,
            ParameterKind.UREY_BRADLEY_FORCE_CONSTANT,
            ParameterKind.UREY_BRADLEY_EQUILIBRIUM,
        ]
        assert list(layout.kinds) == expected

    def test_indices_are_contiguous_and_match_position(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        assert [slot.index for slot in layout] == list(range(len(layout)))

    def test_bonds_only_ff_has_no_ub_tail(self) -> None:
        ff = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        assert len(layout) == 2
        assert layout.kinds == (ParameterKind.BOND_FORCE_CONSTANT, ParameterKind.BOND_EQUILIBRIUM)

    def test_angle_without_ub_fields_excluded_from_ub_tail(self) -> None:
        ff = ForceField(
            angles=(
                AngleParam(elements=("H", "C", "H"), force_constant=50.0, equilibrium=109.5),
                AngleParam(
                    elements=("F", "C", "H"),
                    force_constant=60.0,
                    equilibrium=109.5,
                    ub_force_constant=10.0,
                    ub_equilibrium=1.8,
                ),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        # 2 angles * 2 (k,eq) + only 1 UB tail (the second angle only)
        assert len(layout) == 2 * 2 + 2
        ub_indices = layout.indices_by_kind[ParameterKind.UREY_BRADLEY_FORCE_CONSTANT]
        assert len(ub_indices) == 1
        assert layout[ub_indices[0]].owner_index == 1

    def test_empty_force_field_has_zero_length_layout(self) -> None:
        layout = ParameterLayout.from_force_field(ForceField(functional_form=FunctionalForm.HARMONIC))
        assert len(layout) == 0
        assert layout.vector(ForceField(functional_form=FunctionalForm.HARMONIC)).shape == (0,)

    def test_indices_by_kind_partitions_all_slots(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        by_kind = layout.indices_by_kind
        all_indices = sorted(i for indices in by_kind.values() for i in indices)
        assert all_indices == list(range(len(layout)))

    def test_names_kinds_units_bounds_steps_same_length_as_layout(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        n = len(layout)
        assert len(layout.names) == n
        assert len(layout.kinds) == n
        assert len(layout.units) == n
        assert layout.bounds.shape == (n, 2)
        assert layout.steps.shape == (n,)


# ---------------------------------------------------------------------------
# ParameterId semantics: deterministic, duplicate-occurrence disambiguation
# ---------------------------------------------------------------------------


class TestParameterIdSemantics:
    def test_duplicate_chemical_identity_gets_distinct_occurrence(self) -> None:
        """Two bonds with identical elements/env_id must get occurrence 0 and 1."""
        ff = ForceField(
            bonds=(
                BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
                BondParam(elements=("C", "H"), force_constant=345.0, equilibrium=1.10),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        occurrences = [slot.id.occurrence for slot in layout if slot.field == "force_constant"]
        assert occurrences == [0, 1]
        # IDs must be unique despite identical family/identity/field.
        ids = [slot.id for slot in layout]
        assert len(ids) == len(set(ids))

    def test_k_and_eq_slots_of_same_row_share_occurrence(self) -> None:
        """force_constant and equilibrium slots of ONE bond share the same occurrence number."""
        ff = ForceField(
            bonds=(
                BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
                BondParam(elements=("C", "H"), force_constant=345.0, equilibrium=1.10),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        # slots 0,1 = first bond (k, eq); slots 2,3 = second bond (k, eq)
        assert layout[0].id.occurrence == layout[1].id.occurrence == 0
        assert layout[2].id.occurrence == layout[3].id.occurrence == 1

    def test_parameter_id_independent_of_object_identity(self) -> None:
        """Two structurally-identical but separately-constructed FFs produce identical IDs."""
        ff_a = _full_force_field()
        ff_b = _full_force_field()
        assert ff_a is not ff_b
        ids_a = ParameterLayout.from_force_field(ff_a).ids
        ids_b = ParameterLayout.from_force_field(ff_b).ids
        assert ids_a == ids_b

    def test_different_chemistry_gives_different_identity(self) -> None:
        ff_cf = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        ff_ch = ForceField(
            bonds=(BondParam(elements=("C", "H"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        id_cf = ParameterLayout.from_force_field(ff_cf)[0].id
        id_ch = ParameterLayout.from_force_field(ff_ch)[0].id
        assert id_cf != id_ch

    def test_parameter_id_is_a_plain_semantic_tuple(self) -> None:
        pid = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert pid.family == "bond"
        assert pid.identity == ("C", "F", "")
        assert pid.occurrence == 0
        assert pid.field == "force_constant"

    def test_equal_ids_for_identical_fields(self) -> None:
        """Two separately-constructed IDs with identical fields compare equal.

        Identity/equality semantics for a parsed parameter now live
        entirely on ``ParameterId``, independent of Python object
        identity or the parser-private per-format staging record
        (``q2mm.io.mm3._Mm3ParameterRow`` / ``q2mm.io.tinker._TinkerParameterRow``)
        that produced it.
        """
        a = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        b = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert a == b
        assert a is not b

    def test_unequal_ids_differ_by_family_identity_occurrence_or_field(self) -> None:
        base = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert base != ParameterId(family="angle", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert base != ParameterId(family="bond", identity=("C", "H", ""), occurrence=0, field="force_constant")
        assert base != ParameterId(family="bond", identity=("C", "F", ""), occurrence=1, field="force_constant")
        assert base != ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="equilibrium")

    def test_hash_equal_for_equal_ids(self) -> None:
        a = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        b = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert hash(a) == hash(b)

    def test_usable_in_set(self) -> None:
        a = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        b = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        c = ParameterId(family="bond", identity=("C", "H", ""), occurrence=0, field="force_constant")
        assert len({a, b, c}) == 2

    def test_not_equal_to_non_parameter_id(self) -> None:
        pid = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert pid != "not a parameter id"
        assert pid != 42

    def test_identity_shortcut(self) -> None:
        pid = ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant")
        assert pid == pid  # noqa: PLR0124


# ---------------------------------------------------------------------------
# ParameterLayout validation, vector/replace round-trip
# ---------------------------------------------------------------------------


class TestParameterLayoutValidation:
    def test_rejects_duplicate_index(self) -> None:
        unit = next(iter(ParameterUnit))
        slot = ParameterSlot(
            index=0,
            id=ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant"),
            kind=ParameterKind.BOND_FORCE_CONSTANT,
            unit=unit,
            name="kb_C-F",
            bounds=(0.0, 1000.0),
            step=1.0,
            owner="bonds",
            owner_index=0,
            field="force_constant",
        )
        with pytest.raises(ValueError, match="index"):
            ParameterLayout(slots=(slot, slot))

    def test_rejects_duplicate_id_with_distinct_indices(self) -> None:
        unit = next(iter(ParameterUnit))

        def make(idx: int) -> ParameterSlot:
            return ParameterSlot(
                index=idx,
                id=ParameterId(family="bond", identity=("C", "F", ""), occurrence=0, field="force_constant"),
                kind=ParameterKind.BOND_FORCE_CONSTANT,
                unit=unit,
                name="kb_C-F",
                bounds=(0.0, 1000.0),
                step=1.0,
                owner="bonds",
                owner_index=idx,
                field="force_constant",
            )

        with pytest.raises(ValueError):
            ParameterLayout(slots=(make(0), make(1)))

    def test_vector_matches_forcefield_values_in_order(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        assert vec[0] == pytest.approx(300.0)  # bond0 k
        assert vec[1] == pytest.approx(1.35)  # bond0 eq
        assert vec[-2] == pytest.approx(12.0)  # UB k
        assert vec[-1] == pytest.approx(1.8)  # UB eq

    def test_replace_returns_new_forcefield_without_mutating_original(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff).copy()
        modified = vec.copy()
        modified[0] = 999.0
        new_ff = layout.replace(ff, modified)
        assert new_ff is not ff
        assert ff.bonds[0].force_constant == pytest.approx(300.0)  # original untouched
        assert new_ff.bonds[0].force_constant == pytest.approx(999.0)

    def test_replace_round_trips_vector_exactly(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        rebuilt = layout.replace(ff, vec)
        np.testing.assert_array_equal(layout.vector(rebuilt), vec)

    def test_vector_wrong_length_raises(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError):
            layout.replace(ff, np.zeros(len(layout) + 1))

    def test_index_of_and_index_by_id_are_consistent(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        for slot in layout:
            assert layout.index_of(slot.id) == slot.index
            assert layout.index_by_id[slot.id] == slot.index


# ---------------------------------------------------------------------------
# ActiveParameterSpace: pack/expand, frozen preservation
# ---------------------------------------------------------------------------


class TestActiveParameterSpace:
    def test_uses_identity_equality_for_array_backed_state(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        first = ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([0, 1]))
        second = ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([0, 1]))

        assert first is not second
        assert first != second
        assert len({first, second}) == 2

    def test_all_active_covers_every_slot(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        space = ActiveParameterSpace.all_active(layout, ff)
        assert space.n_active == space.n_full == len(layout)
        np.testing.assert_array_equal(space.active_indices, np.arange(len(layout)))

    def test_pack_expand_round_trip_when_all_active(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        space = ActiveParameterSpace.all_active(layout, ff)
        full = layout.vector(ff)
        packed = space.pack(full)
        assert packed.shape == (space.n_active,)
        np.testing.assert_array_equal(space.expand(packed), full)

    def test_custom_active_indices_preserve_frozen_values_on_expand(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        # Freeze everything except the first bond's two slots (indices 0, 1).
        active_indices = np.array([0, 1])
        space = ActiveParameterSpace(layout=layout, baseline=full, active_indices=active_indices)
        assert space.n_active == 2
        assert space.n_full == len(layout)

        # Perturb only the active slots; frozen slots must come back untouched.
        packed = space.pack(full)
        perturbed = packed + 5.0
        expanded = space.expand(perturbed)
        np.testing.assert_array_equal(expanded[2:], full[2:])  # frozen backbone preserved
        np.testing.assert_array_equal(expanded[:2], full[:2] + 5.0)  # active slots changed

    def test_expand_with_explicit_base_overrides_baseline(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        space = ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([0]))
        alt_base = full.copy()
        alt_base[1] = -123.0
        expanded = space.expand(space.pack(full), base=alt_base)
        assert expanded[1] == pytest.approx(-123.0)

    def test_with_baseline_preserves_active_indices(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        space = ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([0, 2]))
        new_baseline = full * 2.0
        moved = space.with_baseline(new_baseline)
        np.testing.assert_array_equal(moved.active_indices, space.active_indices)
        np.testing.assert_array_equal(moved.baseline, new_baseline)

    def test_with_active_indices_preserves_baseline(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        space = ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([0, 2]))
        moved = space.with_active_indices(np.array([1, 3]))
        np.testing.assert_array_equal(moved.baseline, space.baseline)
        np.testing.assert_array_equal(moved.active_indices, np.array([1, 3]))

    def test_active_indices_out_of_range_raises(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        with pytest.raises(ValueError):
            ActiveParameterSpace(layout=layout, baseline=full, active_indices=np.array([len(layout) + 10]))

    def test_baseline_wrong_length_raises(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError):
            ActiveParameterSpace(layout=layout, baseline=np.zeros(3), active_indices=np.array([0]))

    def test_bounds_and_names_restricted_to_active_subset(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        full = layout.vector(ff)
        active_indices = np.array([0, 3, 5])
        space = ActiveParameterSpace(layout=layout, baseline=full, active_indices=active_indices)
        assert space.bounds.shape == (3, 2)
        assert space.names == tuple(layout.names[i] for i in active_indices)
        assert space.kinds == tuple(layout.kinds[i] for i in active_indices)


# ---------------------------------------------------------------------------
# opt_substructure_membership - legacy freeze_standard_params replacement
# ---------------------------------------------------------------------------


class TestOptSubstructureMembership:
    def test_membership_matches_opt_only_rows_by_identity(self) -> None:
        composed = ForceField(
            bonds=(
                BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),  # standard backbone
                BondParam(elements=("C", "N"), force_constant=250.0, equilibrium=1.40),  # OPT row
            ),
            angles=(AngleParam(elements=("F", "C", "N"), force_constant=60.0, equilibrium=109.5),),
            functional_form=FunctionalForm.HARMONIC,
        )
        opt_only = ForceField(
            bonds=(BondParam(elements=("C", "N"), force_constant=250.0, equilibrium=1.40),),
            functional_form=FunctionalForm.HARMONIC,
        )

        membership = opt_substructure_membership(composed, opt_only)
        assert membership.bonds == frozenset({1})
        assert membership.angles == frozenset()

    def test_from_membership_produces_active_space_matching_opt_rows(self) -> None:
        composed = ForceField(
            bonds=(
                BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),
                BondParam(elements=("C", "N"), force_constant=250.0, equilibrium=1.40),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        opt_only = ForceField(
            bonds=(BondParam(elements=("C", "N"), force_constant=250.0, equilibrium=1.40),),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(composed)
        membership = opt_substructure_membership(composed, opt_only)
        space = ActiveParameterSpace.from_membership(layout, composed, membership)

        # Only bond index 1's two slots (k, eq) should be active.
        assert space.n_active == 2
        active_owner_indices = {layout[i].owner_index for i in space.active_indices}
        assert active_owner_indices == {1}

    def test_from_membership_separates_angle_bending_from_ub_activeness(self) -> None:
        """An angle's bending (k,eq) activeness is independent from its UB activeness."""
        composed = ForceField(
            angles=(
                AngleParam(
                    elements=("H", "C", "H"),
                    force_constant=50.0,
                    equilibrium=109.5,
                    ub_force_constant=10.0,
                    ub_equilibrium=1.8,
                ),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        opt_only = ForceField(functional_form=FunctionalForm.HARMONIC)  # nothing matches -> angle bending frozen
        layout = ParameterLayout.from_force_field(composed)
        membership = opt_substructure_membership(composed, opt_only)
        # Manually mark the UB term active while bending stays frozen, exercising
        # the from_membership per-field branch (urey_bradley vs bond/angle family).
        from q2mm.models.parameters import OptSubstructureMembership

        membership_with_ub = OptSubstructureMembership(
            bonds=membership.bonds,
            angles=membership.angles,
            torsions=membership.torsions,
            stretch_bends=membership.stretch_bends,
            vdws=membership.vdws,
            urey_bradley=frozenset({0}),
        )
        space = ActiveParameterSpace.from_membership(layout, composed, membership_with_ub)
        active_kinds = {layout[i].kind for i in space.active_indices}
        assert active_kinds == {ParameterKind.UREY_BRADLEY_FORCE_CONSTANT, ParameterKind.UREY_BRADLEY_EQUILIBRIUM}


# ---------------------------------------------------------------------------
# Fingerprint: deterministic, value-free, structure-sensitive
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_has_expected_prefix(self) -> None:
        layout = ParameterLayout.from_force_field(_full_force_field())
        assert layout.fingerprint.startswith("sha256:")
        assert len(layout.fingerprint) == len("sha256:") + 64

    def test_fingerprint_stable_for_structurally_identical_layouts(self) -> None:
        fp_a = ParameterLayout.from_force_field(_full_force_field()).fingerprint
        fp_b = ParameterLayout.from_force_field(_full_force_field()).fingerprint
        assert fp_a == fp_b

    def test_fingerprint_unchanged_by_value_only_changes(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        fp_before = layout.fingerprint

        modified_ff = layout.replace(ff, layout.vector(ff) + 1.0)
        fp_after = ParameterLayout.from_force_field(modified_ff).fingerprint
        assert fp_before == fp_after

    def test_fingerprint_changes_when_a_bond_is_added(self) -> None:
        ff = _full_force_field()
        fp_before = ParameterLayout.from_force_field(ff).fingerprint

        ff_extra = ForceField(
            name=ff.name,
            bonds=(*ff.bonds, BondParam(elements=("C", "C"), force_constant=310.0, equilibrium=1.50)),
            angles=ff.angles,
            torsions=ff.torsions,
            stretch_bends=ff.stretch_bends,
            vdws=ff.vdws,
            functional_form=FunctionalForm.HARMONIC,
        )
        fp_after = ParameterLayout.from_force_field(ff_extra).fingerprint
        assert fp_before != fp_after

    def test_fingerprint_changes_when_ub_tail_added(self) -> None:
        no_ub = ForceField(
            angles=(AngleParam(elements=("H", "C", "H"), force_constant=50.0, equilibrium=109.5),),
            functional_form=FunctionalForm.HARMONIC,
        )
        with_ub = ForceField(
            angles=(
                AngleParam(
                    elements=("H", "C", "H"),
                    force_constant=50.0,
                    equilibrium=109.5,
                    ub_force_constant=10.0,
                    ub_equilibrium=1.8,
                ),
            ),
            functional_form=FunctionalForm.HARMONIC,
        )
        fp_no_ub = ParameterLayout.from_force_field(no_ub).fingerprint
        fp_with_ub = ParameterLayout.from_force_field(with_ub).fingerprint
        assert fp_no_ub != fp_with_ub

    def test_fingerprint_deterministic_across_pythonhashseed_subprocess(self) -> None:
        """Fingerprint must not depend on PYTHONHASHSEED.

        Run the SAME layout construction in fresh subprocesses with
        different seeds and confirm identical output — the fingerprint
        must never depend on Python's str/object hash randomization.
        """
        script = textwrap.dedent(
            """
            from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm, StretchBendParam, TorsionParam, VdwParam
            from q2mm.models.parameters import ParameterLayout

            ff = ForceField(
                bonds=(
                    BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),
                    BondParam(elements=("C", "H"), force_constant=340.0, equilibrium=1.09),
                ),
                angles=(
                    AngleParam(elements=("F", "C", "H"), force_constant=60.0, equilibrium=109.5),
                    AngleParam(
                        elements=("H", "C", "H"),
                        force_constant=55.0,
                        equilibrium=109.5,
                        ub_force_constant=12.0,
                        ub_equilibrium=1.8,
                    ),
                ),
                torsions=(TorsionParam(elements=("F", "C", "H", "H"), periodicity=3, force_constant=0.1),),
                stretch_bends=(StretchBendParam(elements=("F", "C", "H"), force_constant=0.2),),
                vdws=(
                    VdwParam(atom_type="1", radius=1.9, epsilon=0.03, element="C"),
                    VdwParam(atom_type="5", radius=1.5, epsilon=0.02, element="H"),
                ),
                functional_form=FunctionalForm.HARMONIC,
            )
            print(ParameterLayout.from_force_field(ff).fingerprint)
            """
        )

        import os

        fingerprints = set()
        for seed in ("0", "1", "42", "1337"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                env={
                    "PYTHONHASHSEED": seed,
                    "PATH": os.environ.get("PATH", ""),
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"seed={seed} failed: {result.stderr}"
            fingerprints.add(result.stdout.strip())

        assert len(fingerprints) == 1, f"Fingerprint varied across PYTHONHASHSEED values: {fingerprints}"


class TestFingerprintSemanticCompleteness:
    """Row-discriminating metadata (bond_order/context/ff_row/...) is part of the fingerprint.

    Guards against the specific gap the coarse ``key + env_id`` identity
    used to have: two context-specific/bond-order/source-row variants
    that happen to share an element key and environment ID were
    indistinguishable except by an arbitrary occurrence counter, so
    reassigning which physical row got which metadata was invisible to
    the fingerprint even though the parameter *vector* would then map
    indices to different physical rows.
    """

    def test_bond_context_changes_fingerprint(self) -> None:
        base = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35, context="O200 0000"),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35, context="O300 0000"),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_bond_order_changes_fingerprint(self) -> None:
        base = ForceField(
            bonds=(BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.50, bond_order="-"),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            bonds=(BondParam(elements=("C", "C"), force_constant=300.0, equilibrium=1.50, bond_order="="),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_bond_ff_row_changes_fingerprint(self) -> None:
        base = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35, ff_row=1),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35, ff_row=2),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_bond_ff_row_none_uses_explicit_sentinel_not_string_none(self) -> None:
        """An unset ``ff_row`` (``None``) must not collide with a hypothetical field value ``"None"``."""
        from q2mm.models.parameters import _NONE_IDENTITY_SENTINEL, _bond_identity

        bond_unset = BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35, ff_row=None)
        identity = _bond_identity(bond_unset)
        assert identity[-1] == _NONE_IDENTITY_SENTINEL
        assert identity[-1] != "None"

    def test_angle_ff_row_changes_fingerprint(self) -> None:
        base = ForceField(
            angles=(AngleParam(elements=("F", "C", "H"), force_constant=60.0, equilibrium=109.5, ff_row=1),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            angles=(AngleParam(elements=("F", "C", "H"), force_constant=60.0, equilibrium=109.5, ff_row=2),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_stretch_bend_ff_row_changes_fingerprint(self) -> None:
        base = ForceField(
            stretch_bends=(StretchBendParam(elements=("F", "C", "H"), force_constant=0.2, ff_row=1),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            stretch_bends=(StretchBendParam(elements=("F", "C", "H"), force_constant=0.2, ff_row=2),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_torsion_ff_row_changes_fingerprint(self) -> None:
        base = ForceField(
            torsions=(TorsionParam(elements=("F", "C", "H", "H"), periodicity=3, force_constant=0.1, ff_row=1),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            torsions=(TorsionParam(elements=("F", "C", "H", "H"), periodicity=3, force_constant=0.1, ff_row=2),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_vdw_ff_row_changes_fingerprint(self) -> None:
        base = ForceField(
            vdws=(VdwParam(atom_type="1", radius=1.9, epsilon=0.03, element="C", ff_row=1),),
            functional_form=FunctionalForm.HARMONIC,
        )
        changed = ForceField(
            vdws=(VdwParam(atom_type="1", radius=1.9, epsilon=0.03, element="C", ff_row=2),),
            functional_form=FunctionalForm.HARMONIC,
        )
        assert (
            ParameterLayout.from_force_field(base).fingerprint != ParameterLayout.from_force_field(changed).fingerprint
        )

    def test_swapping_two_semantically_distinct_rows_changes_fingerprint(self) -> None:
        """Two rows sharing a key+env_id, distinguished only by row metadata, must not be interchangeable.

        Before Blocker 2, both rows collapsed to the same coarse
        ``key + env_id`` identity and were disambiguated only by an
        arbitrary assignment-order ``occurrence`` counter — so
        reassigning which physical row's context/bond_order/ff_row
        landed at which vector index was invisible to the fingerprint,
        even though the parameter values at that index would then refer
        to a different physical row.
        """

        def _make(order_a: str, ctx_a: str, row_a: int, order_b: str, ctx_b: str, row_b: int) -> ForceField:
            return ForceField(
                bonds=(
                    BondParam(
                        elements=("C", "F"),
                        force_constant=300.0,
                        equilibrium=1.35,
                        env_id="X",
                        bond_order=order_a,
                        context=ctx_a,
                        ff_row=row_a,
                    ),
                    BondParam(
                        elements=("C", "F"),
                        force_constant=310.0,
                        equilibrium=1.36,
                        env_id="X",
                        bond_order=order_b,
                        context=ctx_b,
                        ff_row=row_b,
                    ),
                ),
                functional_form=FunctionalForm.HARMONIC,
            )

        original = _make("-", "O200 0000", 10, "=", "O300 0000", 20)
        swapped = _make("=", "O300 0000", 20, "-", "O200 0000", 10)

        fp_original = ParameterLayout.from_force_field(original).fingerprint
        fp_swapped = ParameterLayout.from_force_field(swapped).fingerprint
        assert fp_original != fp_swapped

    def test_row_metadata_fingerprint_deterministic_across_pythonhashseed_subprocess(self) -> None:
        """The new bond_order/context/ff_row identity fields are also hash-seed-independent."""
        script = textwrap.dedent(
            """
            from q2mm.models.forcefield import BondParam, ForceField, FunctionalForm
            from q2mm.models.parameters import ParameterLayout

            ff = ForceField(
                bonds=(
                    BondParam(
                        elements=("C", "F"), force_constant=300.0, equilibrium=1.35,
                        env_id="X", bond_order="-", context="O200 0000", ff_row=10,
                    ),
                    BondParam(
                        elements=("C", "F"), force_constant=310.0, equilibrium=1.36,
                        env_id="X", bond_order="=", context="O300 0000", ff_row=20,
                    ),
                ),
                functional_form=FunctionalForm.HARMONIC,
            )
            print(ParameterLayout.from_force_field(ff).fingerprint)
            """
        )

        import os

        fingerprints = set()
        for seed in ("0", "1", "42", "1337"):
            result = subprocess.run(
                [sys.executable, "-c", script],
                env={
                    "PYTHONHASHSEED": seed,
                    "PATH": os.environ.get("PATH", ""),
                    "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
                },
                capture_output=True,
                text=True,
                timeout=60,
            )
            assert result.returncode == 0, f"seed={seed} failed: {result.stderr}"
            fingerprints.add(result.stdout.strip())

        assert len(fingerprints) == 1, f"Fingerprint varied across PYTHONHASHSEED values: {fingerprints}"


# ---------------------------------------------------------------------------
# Declarative bounds sanity — parameter range validation is expressed as
# ParameterKind metadata, not raised at parser construction time.
# ---------------------------------------------------------------------------


class TestBoundsSanity:
    """Equilibrium/radius kinds never declare non-positive lower bounds.

    Parser-private per-format staging records
    (``q2mm.io.mm3._Mm3ParameterRow`` / ``q2mm.io.tinker._TinkerParameterRow``)
    perform no range validation at construction time; the canonical,
    declarative bounds live on :data:`ParameterKind` and are enforced by
    optimizers via :class:`ActiveParameterSpace`, not raised at parse time.
    """

    def test_equilibrium_and_radius_kinds_have_positive_lower_bound(self) -> None:
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        positive_lower_bound_kinds = {
            ParameterKind.BOND_EQUILIBRIUM,
            ParameterKind.ANGLE_EQUILIBRIUM,
            ParameterKind.VDW_RADIUS,
            ParameterKind.UREY_BRADLEY_EQUILIBRIUM,
        }
        for slot in layout:
            if slot.kind in positive_lower_bound_kinds:
                lo, _hi = slot.bounds
                assert lo > 0.0, f"{slot.name} ({slot.kind}) has non-positive lower bound {lo}"

    def test_force_constant_kinds_allow_negative_or_zero_lower_bound(self) -> None:
        """Torsion/stretch-bend force constants may be negative.

        Bond/angle/UB force constants are non-negative (TS Hessians are
        curvature-inverted first); torsion/stretch-bend force constants may
        legitimately be negative — neither raises at construction time.
        """
        ff = _full_force_field()
        layout = ParameterLayout.from_force_field(ff)
        for slot in layout:
            if slot.kind in (ParameterKind.TORSION_FORCE_CONSTANT, ParameterKind.STRETCH_BEND_FORCE_CONSTANT):
                lo, _hi = slot.bounds
                assert lo < 0.0


# ---------------------------------------------------------------------------
# fractional_bounds helper
# ---------------------------------------------------------------------------


class TestFractionalBounds:
    def test_fc_and_eq_fractions_scale_around_current_value(self) -> None:
        ff = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        x0 = layout.vector(ff)
        bounds = fractional_bounds(layout.kinds, layout.bounds, x0, fc_fraction=0.20, eq_fraction=0.05)
        # force_constant slot (index 0): +/-20% of 300
        assert bounds[0][0] == pytest.approx(240.0)
        assert bounds[0][1] == pytest.approx(360.0)
        # equilibrium slot (index 1): +/-5% of 1.35
        assert bounds[1][0] == pytest.approx(1.35 * 0.95)
        assert bounds[1][1] == pytest.approx(1.35 * 1.05)

    def test_none_fractions_returns_original_bounds(self) -> None:
        ff = ForceField(
            bonds=(BondParam(elements=("C", "F"), force_constant=300.0, equilibrium=1.35),),
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        x0 = layout.vector(ff)
        bounds = fractional_bounds(layout.kinds, layout.bounds, x0, fc_fraction=None, eq_fraction=None)
        np.testing.assert_array_equal(bounds, layout.bounds)
