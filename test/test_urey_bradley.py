"""Tests for Urey-Bradley term support (#116)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.parameters import ParameterKind, ParameterLayout


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _water_ff_with_ub(
    *,
    bond_k: float = 5.0,
    bond_eq: float = 0.96,
    angle_k: float = 0.5,
    angle_eq: float = 104.5,
    ub_k: float = 10.0,
    ub_eq: float = 1.52,
    functional_form: FunctionalForm | None = None,
) -> ForceField:
    """ForceField for water with one bond type, one angle (with UB)."""
    return ForceField(
        name="water-ub",
        bonds=[
            BondParam(elements=("H", "O"), equilibrium=bond_eq, force_constant=bond_k),
        ],
        angles=[
            AngleParam(
                elements=("H", "O", "H"),
                equilibrium=angle_eq,
                force_constant=angle_k,
                ub_force_constant=ub_k,
                ub_equilibrium=ub_eq,
            ),
        ],
        functional_form=functional_form,
    )


def _water_ff_no_ub() -> ForceField:
    """ForceField for water without UB terms (backward compat check)."""
    return ForceField(
        name="water-no-ub",
        bonds=[
            BondParam(elements=("H", "O"), equilibrium=0.96, force_constant=5.0),
        ],
        angles=[
            AngleParam(elements=("H", "O", "H"), equilibrium=104.5, force_constant=0.5),
        ],
        functional_form=FunctionalForm.HARMONIC,
    )


def _water_molecule(angle_deg: float = 104.5, bond_length: float = 0.96) -> Molecule:
    """Create a water molecule."""
    from test._shared import make_water

    return make_water(angle_deg=angle_deg, bond_length=bond_length)


# ---------------------------------------------------------------------------
# ForceField unit tests
# ---------------------------------------------------------------------------


class TestAngleParamUB:
    """AngleParam with UB fields."""

    def test_ub_fields_default_none(self) -> None:
        a = AngleParam(elements=("H", "O", "H"), equilibrium=104.5, force_constant=0.5)
        assert a.ub_force_constant is None
        assert a.ub_equilibrium is None

    def test_ub_fields_set(self) -> None:
        a = AngleParam(
            elements=("H", "O", "H"),
            equilibrium=104.5,
            force_constant=0.5,
            ub_force_constant=10.0,
            ub_equilibrium=1.52,
        )
        assert a.ub_force_constant == 10.0
        assert a.ub_equilibrium == 1.52


class TestForceFieldUBLayout:
    """ParameterLayout integrates Urey-Bradley terms."""

    def test_layout_length_includes_ub(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        assert len(layout) == 6

    def test_layout_length_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        layout = ParameterLayout.from_force_field(ff)
        assert len(layout) == 4

    def test_layout_vector_includes_ub_tail(self) -> None:
        ff = _water_ff_with_ub(bond_k=5.0, bond_eq=0.96, angle_k=0.5, angle_eq=104.5, ub_k=10.0, ub_eq=1.52)
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        assert len(vec) == 6
        np.testing.assert_allclose(vec, [5.0, 0.96, 0.5, 104.5, 10.0, 1.52])

    def test_layout_replace_round_trip(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        new_vec = np.array([6.0, 1.0, 0.6, 110.0, 12.0, 1.6])
        ff2 = layout.replace(ff, new_vec)
        np.testing.assert_allclose(layout.vector(ff2), new_vec)
        assert ff2.angles[0].ub_force_constant == 12.0
        assert ff2.angles[0].ub_equilibrium == 1.6

    def test_layout_replace_wrong_length_raises(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        with pytest.raises(ValueError, match="does not match"):
            layout.replace(ff, np.array([1.0, 2.0]))

    def test_layout_replace_preserves_ub(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        new_vec = np.array([6.0, 1.0, 0.6, 110.0, 12.0, 1.6])
        ff2 = layout.replace(ff, new_vec)
        np.testing.assert_allclose(layout.vector(ff2), new_vec)
        assert ff2.angles[0].ub_force_constant == 12.0
        assert ff2.angles[0].ub_equilibrium == 1.6
        assert ff.angles[0].ub_force_constant == 10.0

    def test_layout_replace_no_ub_backward_compat(self) -> None:
        ff = _water_ff_no_ub()
        layout = ParameterLayout.from_force_field(ff)
        vec = layout.vector(ff)
        ff2 = layout.replace(ff, vec)
        np.testing.assert_allclose(layout.vector(ff2), vec)
        assert ff2.angles[0].ub_force_constant is None

    def test_indices_by_kind_include_ub(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        indices = layout.indices_by_kind
        assert indices[ParameterKind.UREY_BRADLEY_FORCE_CONSTANT] == (4,)
        assert indices[ParameterKind.UREY_BRADLEY_EQUILIBRIUM] == (5,)

    def test_indices_by_kind_omit_ub_without_terms(self) -> None:
        ff = _water_ff_no_ub()
        layout = ParameterLayout.from_force_field(ff)
        indices = layout.indices_by_kind
        assert ParameterKind.UREY_BRADLEY_FORCE_CONSTANT not in indices
        assert ParameterKind.UREY_BRADLEY_EQUILIBRIUM not in indices

    def test_layout_kinds(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        labels = [kind.value for kind in layout.kinds]
        assert labels == ["bond_k", "bond_eq", "angle_k", "angle_eq", "ub_k", "ub_eq"]

    def test_layout_kinds_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        layout = ParameterLayout.from_force_field(ff)
        labels = [kind.value for kind in layout.kinds]
        assert labels == ["bond_k", "bond_eq", "angle_k", "angle_eq"]

    def test_layout_bounds_include_ub(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        bounds = layout.bounds
        assert len(bounds) == 6
        np.testing.assert_allclose(bounds[4], (0.0, 500.0))
        np.testing.assert_allclose(bounds[5], (1.0, 4.0))

    def test_layout_bounds_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        layout = ParameterLayout.from_force_field(ff)
        bounds = layout.bounds
        assert len(bounds) == 4

    def test_has_urey_bradley(self) -> None:
        assert _water_ff_with_ub().has_urey_bradley
        assert not _water_ff_no_ub().has_urey_bradley

    def test_ub_angles_property(self) -> None:
        ff = _water_ff_with_ub()
        assert len(ff._ub_angles) == 1
        assert ff._ub_angles[0] is ff.angles[0]

    def test_layout_steps_include_ub(self) -> None:
        ff = _water_ff_with_ub()
        layout = ParameterLayout.from_force_field(ff)
        steps = layout.steps
        assert len(steps) == 6

    def test_mixed_ub_and_non_ub_angles(self) -> None:
        ff = ForceField(
            name="mixed",
            angles=[
                AngleParam(elements=("H", "O", "H"), equilibrium=104.5, force_constant=0.5),
                AngleParam(
                    elements=("C", "N", "C"),
                    equilibrium=120.0,
                    force_constant=1.0,
                    ub_force_constant=15.0,
                    ub_equilibrium=2.0,
                ),
            ],
            functional_form=FunctionalForm.HARMONIC,
        )
        layout = ParameterLayout.from_force_field(ff)
        assert len(layout) == 6
        vec = layout.vector(ff)
        assert len(vec) == 6
        np.testing.assert_allclose(vec[4:], [15.0, 2.0])


# ---------------------------------------------------------------------------
# OpenMM engine tests
# ---------------------------------------------------------------------------


@pytest.mark.openmm
class TestOpenMMUreyBradley:
    """OpenMM UB energy evaluation."""

    def test_ub_produces_nonzero_energy(self) -> None:
        """Water with UB should produce energy > 0 when geometry mismatches UB eq."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=120.0, bond_length=1.0)
        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=10.0,
            ub_eq=1.0,  # deliberately different from actual 1-3 distance
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        energy = engine.energy(mol, ff)
        # Energy should be nonzero because of UB strain
        assert energy != 0.0

    def test_ub_zero_at_equilibrium(self) -> None:
        """UB energy is zero when geometry matches equilibrium distance."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=104.5, bond_length=0.96)
        # Compute actual H-H distance for this geometry
        h1 = mol.geometry[1]
        h2 = mol.geometry[2]
        actual_hh = float(np.linalg.norm(h1 - h2))

        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=0.96,
            angle_k=0.5,
            angle_eq=104.5,
            ub_k=10.0,
            ub_eq=actual_hh,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        energy_with_ub = engine.energy(mol, ff)

        # Compare with no-UB energy
        ff_no_ub = ForceField(
            name="no-ub",
            bonds=[BondParam(elements=("H", "O"), equilibrium=0.96, force_constant=5.0)],
            angles=[AngleParam(elements=("H", "O", "H"), equilibrium=104.5, force_constant=0.5)],
            functional_form=FunctionalForm.HARMONIC,
        )
        energy_no_ub = engine.energy(mol, ff_no_ub)

        # When UB eq matches actual distance, UB contributes zero
        np.testing.assert_allclose(energy_with_ub, energy_no_ub, atol=1e-6)

    def test_ub_known_energy(self) -> None:
        """Verify UB energy matches E = k * (r13 - r0)^2 for known geometry."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=104.5, bond_length=0.96)
        h1 = mol.geometry[1]
        h2 = mol.geometry[2]
        actual_hh = float(np.linalg.norm(h1 - h2))

        ub_k = 10.0
        ub_eq = 1.0  # different from actual
        expected_ub_energy = ub_k * (actual_hh - ub_eq) ** 2

        # Use UB-only FF (zero bond_k and angle_k to isolate UB)
        ff = _water_ff_with_ub(
            bond_k=0.0,
            bond_eq=0.96,
            angle_k=0.0,
            angle_eq=104.5,
            ub_k=ub_k,
            ub_eq=ub_eq,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        energy = engine.energy(mol, ff)
        np.testing.assert_allclose(energy, expected_ub_energy, atol=1e-6)

    def test_no_ub_handle_fields(self) -> None:
        """Without UB, handle has empty UB fields."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule()
        ff = ForceField(
            name="no-ub",
            bonds=[BondParam(elements=("H", "O"), equilibrium=0.96, force_constant=5.0)],
            angles=[AngleParam(elements=("H", "O", "H"), equilibrium=104.5, force_constant=0.5)],
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        handle = engine.create_context(mol, ff)
        assert handle.ub_force is None
        assert handle.ub_terms == []

    def test_update_forcefield_ub(self) -> None:
        """update_forcefield should update UB parameters."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=120.0, bond_length=1.0)
        ff1 = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=10.0,
            ub_eq=1.5,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        handle = engine.create_context(mol, ff1)
        e1 = engine.energy(handle, ff1)

        ff2 = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=20.0,
            ub_eq=1.5,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine.update_forcefield(handle, ff2)
        e2 = engine.energy(handle, ff2)
        # Doubling k should change the energy
        assert e1 != e2

    def test_energy_and_param_grad_includes_ub(self) -> None:
        """Analytical-gradient handle must include UB energy + UB gradients (F1).

        Regression: ``_create_diff_handle`` previously dropped the
        Urey-Bradley term entirely, so ``energy_and_param_grad`` returned an
        energy missing the UB contribution and a zero gradient for the UB
        parameters (which live at the tail of the param vector).
        """
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=120.0, bond_length=1.0)
        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=10.0,
            ub_eq=1.0,  # mismatched → nonzero UB strain and gradient
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        e_scalar = engine.energy(mol, ff)
        e_grad, grad = engine.energy_and_param_grad(mol, ff)

        # Diff-handle energy must match the scalar-energy path (UB included).
        np.testing.assert_allclose(e_grad, e_scalar, atol=1e-6)
        # UB params are the tail two entries: index 4 = ub_k, 5 = ub_eq.
        assert grad[4] != 0.0
        assert grad[5] != 0.0

    def test_energy_and_param_grad_ub_matches_finite_difference(self) -> None:
        """UB analytical gradients agree with central finite differences (F1)."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=118.0, bond_length=1.02)
        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=12.0,
            ub_eq=1.1,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine()
        _e, grad = engine.energy_and_param_grad(mol, ff)

        layout = ParameterLayout.from_force_field(ff)
        pv = layout.vector(ff)
        step = 1e-5
        for i in (4, 5):  # ub_k, ub_eq
            pv_plus = np.array(pv, copy=True)
            pv_plus[i] += step
            pv_minus = np.array(pv, copy=True)
            pv_minus[i] -= step
            e_plus = engine.energy(mol, layout.replace(ff, pv_plus))
            e_minus = engine.energy(mol, layout.replace(ff, pv_minus))
            fd = (e_plus - e_minus) / (2.0 * step)
            np.testing.assert_allclose(grad[i], fd, rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# JAX engine tests
# ---------------------------------------------------------------------------


@pytest.mark.jax
class TestJaxUreyBradley:
    """JAX UB energy evaluation."""

    def test_ub_produces_nonzero_energy(self) -> None:
        from q2mm.backends.mm.jax_engine import JaxEngine

        mol = _water_molecule(angle_deg=120.0, bond_length=1.0)
        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=10.0,
            ub_eq=1.0,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = JaxEngine()
        energy = engine.energy(mol, ff)
        assert energy != 0.0

    def test_ub_known_energy(self) -> None:
        """Verify UB energy matches E = k * (r13 - r0)^2 for known geometry."""
        from q2mm.backends.mm.jax_engine import JaxEngine

        mol = _water_molecule(angle_deg=104.5, bond_length=0.96)
        h1 = mol.geometry[1]
        h2 = mol.geometry[2]
        actual_hh = float(np.linalg.norm(h1 - h2))

        ub_k = 10.0
        ub_eq = 1.0
        expected_ub_energy = ub_k * (actual_hh - ub_eq) ** 2

        ff = _water_ff_with_ub(
            bond_k=0.0,
            bond_eq=0.96,
            angle_k=0.0,
            angle_eq=104.5,
            ub_k=ub_k,
            ub_eq=ub_eq,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = JaxEngine()
        energy = engine.energy(mol, ff)
        np.testing.assert_allclose(energy, expected_ub_energy, atol=1e-6)

    def test_no_ub_backward_compat(self) -> None:
        """Without UB, JAX should work the same."""
        from q2mm.backends.mm.jax_engine import JaxEngine

        mol = _water_molecule()
        ff = replace(_water_ff_no_ub(), functional_form=FunctionalForm.HARMONIC)
        engine = JaxEngine()
        energy = engine.energy(mol, ff)
        assert isinstance(energy, float)


# ---------------------------------------------------------------------------
# Cross-engine parity
# ---------------------------------------------------------------------------


@pytest.mark.openmm
@pytest.mark.jax
class TestUreyBradleyParity:
    """OpenMM vs JAX UB energy parity."""

    def test_ub_energy_parity(self) -> None:
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=120.0, bond_length=1.0)
        ff = _water_ff_with_ub(
            bond_k=5.0,
            bond_eq=1.0,
            angle_k=0.5,
            angle_eq=120.0,
            ub_k=10.0,
            ub_eq=1.5,
            functional_form=FunctionalForm.HARMONIC,
        )
        omm_energy = OpenMMEngine().energy(mol, ff)
        jax_energy = JaxEngine().energy(mol, ff)
        np.testing.assert_allclose(omm_energy, jax_energy, atol=1e-5)

    def test_ub_only_energy_parity(self) -> None:
        """UB-only energy (zero bond/angle k) should match between engines."""
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.backends.mm.openmm import OpenMMEngine

        mol = _water_molecule(angle_deg=110.0, bond_length=1.0)
        ff = _water_ff_with_ub(
            bond_k=0.0,
            bond_eq=1.0,
            angle_k=0.0,
            angle_eq=110.0,
            ub_k=15.0,
            ub_eq=1.3,
            functional_form=FunctionalForm.HARMONIC,
        )
        omm_energy = OpenMMEngine().energy(mol, ff)
        jax_energy = JaxEngine().energy(mol, ff)
        np.testing.assert_allclose(omm_energy, jax_energy, atol=1e-5)
