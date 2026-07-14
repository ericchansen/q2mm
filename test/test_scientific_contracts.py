"""Durable scientific behavior contracts for the capability-core rewrite."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from test._shared import SN2_HESSIAN, SN2_XYZ

from q2mm.models.forcefield import ForceField, TorsionParam
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import qfuerza_fresh, qfuerza_into
from q2mm.optimizers.reference import ReferenceData


def _sn2_transition_state() -> Q2MMMolecule:
    hessian = np.load(SN2_HESSIAN)
    return Q2MMMolecule.from_xyz(
        SN2_XYZ,
        charge=-1,
        name="sn2-ts",
        bond_tolerance=1.5,
    ).with_hessian(hessian)


def test_ts_qfuerza_projection_preserves_qm_inputs() -> None:
    """TS projection produces positive springs without changing QM data."""
    molecule = _sn2_transition_state()
    geometry_before = molecule.geometry.copy()
    hessian_before = molecule.hessian.copy()

    with pytest.warns(UserWarning, match="negative eigenvalues"):
        forcefield = qfuerza_fresh(molecule, invert_ts_curvature=True)

    assert np.linalg.eigvalsh(hessian_before).min() < 0.0
    assert all(parameter.force_constant > 0.0 for parameter in forcefield.bonds)
    assert all(parameter.force_constant > 0.0 for parameter in forcefield.angles)
    np.testing.assert_array_equal(molecule.geometry, geometry_before)
    np.testing.assert_array_equal(molecule.hessian, hessian_before)


def test_ts_reference_builder_is_geometry_and_eigenmatrix_multitarget() -> None:
    """Publication-style TS references are not replaced by frequencies."""
    reference = ReferenceData.from_molecules([_sn2_transition_state()])
    counts = Counter(value.kind for value in reference.values)

    assert counts == {
        "bond_length": 5,
        "bond_angle": 10,
        "eig_diagonal": 18,
        "eig_offdiagonal": 153,
    }
    assert "frequency" not in counts


def test_qfuerza_zeroes_only_active_initial_torsions() -> None:
    """QFUERZA keeps frozen literature torsions and zeros active OPT torsions."""
    molecule = _sn2_transition_state()
    forcefield = ForceField(
        torsions=[
            TorsionParam(("F", "C", "H", "H"), periodicity=1, force_constant=2.0),
            TorsionParam(("F", "C", "H", "H"), periodicity=2, force_constant=3.0, frozen=True),
        ]
    )

    with pytest.warns(UserWarning, match="negative eigenvalues"):
        qfuerza_into(forcefield, [molecule], invert_ts_curvature=True)

    assert forcefield.torsions[0].force_constant == 0.0
    assert forcefield.torsions[1].force_constant == 3.0
