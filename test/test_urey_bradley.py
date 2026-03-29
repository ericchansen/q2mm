"""Tests for Urey-Bradley term support (#116).

Covers the ForceField param vector integration, OpenMM energy evaluation,
JAX energy evaluation, and cross-engine parity for Urey-Bradley 1-3
distance interactions.
"""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Q2MMMolecule


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
    )


def _water_molecule(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
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


class TestForceFieldUBParamVector:
    """ForceField param vector with Urey-Bradley terms."""

    def test_n_params_includes_ub(self) -> None:
        ff = _water_ff_with_ub()
        # 1 bond × 2 + 1 angle × 2 + 1 UB × 2 = 6
        assert ff.n_params == 6

    def test_n_params_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        # 1 bond × 2 + 1 angle × 2 = 4
        assert ff.n_params == 4

    def test_get_param_vector_includes_ub_tail(self) -> None:
        ff = _water_ff_with_ub(bond_k=5.0, bond_eq=0.96, angle_k=0.5, angle_eq=104.5, ub_k=10.0, ub_eq=1.52)
        vec = ff.get_param_vector()
        assert len(vec) == 6
        np.testing.assert_allclose(vec, [5.0, 0.96, 0.5, 104.5, 10.0, 1.52])

    def test_set_param_vector_round_trip(self) -> None:
        ff = _water_ff_with_ub()
        new_vec = np.array([6.0, 1.0, 0.6, 110.0, 12.0, 1.6])
        ff.set_param_vector(new_vec)
        np.testing.assert_allclose(ff.get_param_vector(), new_vec)
        assert ff.angles[0].ub_force_constant == 12.0
        assert ff.angles[0].ub_equilibrium == 1.6

    def test_set_param_vector_wrong_length_raises(self) -> None:
        ff = _water_ff_with_ub()
        with pytest.raises(ValueError, match="does not match"):
            ff.set_param_vector(np.array([1.0, 2.0]))

    def test_with_params_preserves_ub(self) -> None:
        ff = _water_ff_with_ub()
        new_vec = np.array([6.0, 1.0, 0.6, 110.0, 12.0, 1.6])
        ff2 = ff.with_params(new_vec)
        np.testing.assert_allclose(ff2.get_param_vector(), new_vec)
        assert ff2.angles[0].ub_force_constant == 12.0
        assert ff2.angles[0].ub_equilibrium == 1.6
        # Original unchanged
        assert ff.angles[0].ub_force_constant == 10.0

    def test_with_params_no_ub_backward_compat(self) -> None:
        ff = _water_ff_no_ub()
        vec = ff.get_param_vector()
        ff2 = ff.with_params(vec)
        np.testing.assert_allclose(ff2.get_param_vector(), vec)
        assert ff2.angles[0].ub_force_constant is None

    def test_get_param_indices_by_type_includes_ub(self) -> None:
        ff = _water_ff_with_ub()
        indices = ff.get_param_indices_by_type()
        assert "ub_k" in indices
        assert "ub_eq" in indices
        assert indices["ub_k"] == [4]
        assert indices["ub_eq"] == [5]

    def test_get_param_indices_by_type_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        indices = ff.get_param_indices_by_type()
        assert indices["ub_k"] == []
        assert indices["ub_eq"] == []

    def test_get_param_type_labels(self) -> None:
        ff = _water_ff_with_ub()
        labels = ff.get_param_type_labels()
        assert labels == ["bond_k", "bond_eq", "angle_k", "angle_eq", "ub_k", "ub_eq"]

    def test_get_param_type_labels_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        labels = ff.get_param_type_labels()
        assert labels == ["bond_k", "bond_eq", "angle_k", "angle_eq"]

    def test_get_bounds_includes_ub(self) -> None:
        ff = _water_ff_with_ub()
        bounds = ff.get_bounds()
        assert len(bounds) == 6
        # UB bounds
        assert bounds[4] == (0.0, 500.0)  # ub_k
        assert bounds[5] == (1.0, 4.0)  # ub_eq

    def test_get_bounds_no_ub(self) -> None:
        ff = _water_ff_no_ub()
        bounds = ff.get_bounds()
        assert len(bounds) == 4

    def test_has_urey_bradley(self) -> None:
        assert _water_ff_with_ub().has_urey_bradley
        assert not _water_ff_no_ub().has_urey_bradley

    def test_ub_angles_property(self) -> None:
        ff = _water_ff_with_ub()
        assert len(ff._ub_angles) == 1
        assert ff._ub_angles[0] is ff.angles[0]

    def test_get_step_sizes_includes_ub(self) -> None:
        ff = _water_ff_with_ub()
        steps = ff.get_step_sizes()
        assert len(steps) == 6

    def test_mixed_ub_and_non_ub_angles(self) -> None:
        """FF with some angles having UB and some not."""
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
        )
        assert ff.n_params == 4 + 2  # 2 angles × 2 + 1 UB × 2
        vec = ff.get_param_vector()
        assert len(vec) == 6
        # UB params at the tail
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
        ff = _water_ff_no_ub()
        ff.functional_form = FunctionalForm.HARMONIC
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
