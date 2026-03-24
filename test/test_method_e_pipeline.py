"""Tests for Method E hybrid pipeline and ts_method parameter.

Validates the full Method E pipeline from Limé & Norrby (J. Comput.
Chem. 2015, 36, 1130):
  Method D → detect problematic → cherry-pick from Method C

Also tests the ts_method parameter on estimate_force_constants().

Covers issue #75.
"""

from __future__ import annotations

import numpy as np
import pytest

from test._shared import GS_FCHK, SN2_DATA_AVAILABLE, SN2_HESSIAN, SN2_XYZ

from q2mm.models.forcefield import ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.models.seminario import (
    estimate_force_constants,
    estimate_force_constants_method_e,
)

# ---------------------------------------------------------------------------
# Fixtures: data paths
# ---------------------------------------------------------------------------
_ETHANE_DATA_AVAILABLE = GS_FCHK.exists()


@pytest.fixture(scope="module")
def sn2_mol() -> Q2MMMolecule:
    """SN2 TS molecule with Hessian attached."""
    mol = Q2MMMolecule.from_xyz(str(SN2_XYZ), bond_tolerance=1.4, name="SN2-TS")
    hessian = np.load(str(SN2_HESSIAN))
    return mol.with_hessian(hessian)


@pytest.fixture(scope="module")
def ethane_mol() -> Q2MMMolecule:
    """Ethane ground-state molecule with Hessian."""
    from q2mm.optimizers.objective import ReferenceData

    _, mol = ReferenceData.from_fchk(GS_FCHK)
    return mol


# ---------------------------------------------------------------------------
# ts_method parameter tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
class TestTSMethodParam:
    """Test ts_method parameter on estimate_force_constants."""

    def test_ts_method_none_is_default(self, sn2_mol: Q2MMMolecule) -> None:
        """ts_method=None should give same result as no ts_method."""
        ff_default = estimate_force_constants(sn2_mol)
        ff_none = estimate_force_constants(sn2_mol, ts_method=None)

        for b_def, b_none in zip(ff_default.bonds, ff_none.bonds):
            assert b_def.force_constant == b_none.force_constant
            assert b_def.equilibrium == b_none.equilibrium
        for a_def, a_none in zip(ff_default.angles, ff_none.angles):
            assert a_def.force_constant == a_none.force_constant
            assert a_def.equilibrium == a_none.equilibrium

    def test_method_c_vs_d_differ_on_ts(self, sn2_mol: Q2MMMolecule) -> None:
        """Methods C and D should produce different FCs for a TS Hessian."""
        ff_c = estimate_force_constants(sn2_mol, ts_method="C")
        ff_d = estimate_force_constants(sn2_mol, ts_method="D")

        # At least one bond FC should differ
        diffs = [abs(bc.force_constant - bd.force_constant) for bc, bd in zip(ff_c.bonds, ff_d.bonds)]
        assert max(diffs) > 0.01, f"C and D gave identical bonds: max diff {max(diffs)}"

    def test_method_d_preserves_raw_behavior(self, sn2_mol: Q2MMMolecule) -> None:
        """Method D (identity transform) should give same FCs as raw Hessian."""
        ff_raw = estimate_force_constants(sn2_mol)
        ff_d = estimate_force_constants(sn2_mol, ts_method="D")

        for b_raw, b_d in zip(ff_raw.bonds, ff_d.bonds):
            np.testing.assert_allclose(b_raw.force_constant, b_d.force_constant, atol=1e-10)
        for a_raw, a_d in zip(ff_raw.angles, ff_d.angles):
            np.testing.assert_allclose(a_raw.force_constant, a_d.force_constant, atol=1e-10)

    def test_method_c_positive_bonds_on_sn2(self, sn2_mol: Q2MMMolecule) -> None:
        """Method C produces positive bond FCs for the SN2 TS system.

        Note: positivity is not a general guarantee of Method C (the
        Seminario sub-block projection can still yield negative values
        for some systems), but for SN2 the TS curvature inversion is
        sufficient to make all bond FCs positive.
        """
        ff_c = estimate_force_constants(sn2_mol, ts_method="C")
        for bond in ff_c.bonds:
            assert bond.force_constant > 0, f"Bond {bond.key} has negative FC with Method C"

    @pytest.mark.skipif(not _ETHANE_DATA_AVAILABLE, reason="Ethane fchk not found")
    def test_gs_molecule_unaffected_by_ts_method(self, ethane_mol: Q2MMMolecule) -> None:
        """For a GS molecule (no negative eigenvalues), ts_method shouldn't matter much."""
        ff_none = estimate_force_constants(ethane_mol)
        ff_c = estimate_force_constants(ethane_mol, ts_method="C")

        # Equilibria should be identical (geometry unchanged)
        for b1, b2 in zip(ff_none.bonds, ff_c.bonds):
            assert b1.equilibrium == b2.equilibrium

    @pytest.mark.skipif(not _ETHANE_DATA_AVAILABLE, reason="Ethane fchk not found")
    def test_invalid_ts_method_raises(self, ethane_mol: Q2MMMolecule) -> None:
        """Invalid ts_method should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported ts_method"):
            estimate_force_constants(ethane_mol, ts_method="X")


# ---------------------------------------------------------------------------
# Method E pipeline tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SN2_DATA_AVAILABLE, reason="SN2 data not found")
class TestMethodEPipeline:
    """Full Method E hybrid pipeline tests."""

    def test_returns_forcefield_and_diagnostics(self, sn2_mol: Q2MMMolecule) -> None:
        """Method E should return (ForceField, dict) tuple."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol)
        assert isinstance(ff_e, ForceField)
        assert isinstance(diag, dict)
        assert "method_d" in diag
        assert "method_c" in diag
        assert "problematic" in diag

    def test_diagnostics_contain_forcefields(self, sn2_mol: Q2MMMolecule) -> None:
        """Diagnostics should contain Method C and D ForceField objects."""
        _, diag = estimate_force_constants_method_e(sn2_mol)
        assert isinstance(diag["method_d"], ForceField)
        assert isinstance(diag["method_c"], ForceField)

    def test_healthy_params_use_method_d_values(self, sn2_mol: Q2MMMolecule) -> None:
        """For non-problematic params, Method E should use Method D values."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol)
        problematic_bond_keys = diag["problematic"]["bonds"]
        problematic_angle_keys = diag["problematic"]["angles"]

        for be, bd in zip(ff_e.bonds, diag["method_d"].bonds):
            if bd.key not in problematic_bond_keys:
                assert be.force_constant == bd.force_constant, f"Healthy bond {bd.key} should use Method D value"

        for ae, ad in zip(ff_e.angles, diag["method_d"].angles):
            if ad.key not in problematic_angle_keys:
                assert ae.force_constant == ad.force_constant, f"Healthy angle {ad.key} should use Method D value"

    def test_problematic_params_use_method_c_values(self, sn2_mol: Q2MMMolecule) -> None:
        """For problematic params, Method E should use Method C values."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol)
        problematic_bond_keys = diag["problematic"]["bonds"]
        problematic_angle_keys = diag["problematic"]["angles"]

        for be, bc in zip(ff_e.bonds, diag["method_c"].bonds):
            if bc.key in problematic_bond_keys:
                assert be.force_constant == bc.force_constant, f"Problematic bond {bc.key} should use Method C value"

        for ae, ac in zip(ff_e.angles, diag["method_c"].angles):
            if ac.key in problematic_angle_keys:
                assert ae.force_constant == ac.force_constant, f"Problematic angle {ac.key} should use Method C value"

    def test_no_problematic_gives_pure_method_d(self, sn2_mol: Q2MMMolecule) -> None:
        """If no params are problematic, E result equals D result."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol, fc_threshold=-999)
        n_problematic = sum(len(v) for v in diag["problematic"].values())
        assert n_problematic == 0, "Expected no problematic params with threshold=-999"

        for be, bd in zip(ff_e.bonds, diag["method_d"].bonds):
            assert be.force_constant == bd.force_constant
        for ae, ad in zip(ff_e.angles, diag["method_d"].angles):
            assert ae.force_constant == ad.force_constant

    def test_all_problematic_gives_pure_method_c(self, sn2_mol: Q2MMMolecule) -> None:
        """If all params are problematic (high threshold), E = C."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol, fc_threshold=999)
        n_problematic = sum(len(v) for v in diag["problematic"].values())
        assert n_problematic > 0, "Expected some problematic params with threshold=999"

        for be, bc in zip(ff_e.bonds, diag["method_c"].bonds):
            if bc.key in diag["problematic"]["bonds"]:
                assert be.force_constant == bc.force_constant

    def test_method_e_backward_compatible(self, sn2_mol: Q2MMMolecule) -> None:
        """Method E should preserve same bond/angle count as C or D."""
        ff_e, diag = estimate_force_constants_method_e(sn2_mol)
        assert len(ff_e.bonds) == len(diag["method_c"].bonds)
        assert len(ff_e.angles) == len(diag["method_c"].angles)
        assert len(ff_e.torsions) == len(diag["method_c"].torsions)

    def test_torsions_zeroed(self, sn2_mol: Q2MMMolecule) -> None:
        """Torsions should be zeroed by default."""
        ff_e, _ = estimate_force_constants_method_e(sn2_mol)
        for t in ff_e.torsions:
            assert t.force_constant == 0.0


# ---------------------------------------------------------------------------
# Method E on GS molecule (no TS treatment needed)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _ETHANE_DATA_AVAILABLE, reason="Ethane fchk not found")
class TestMethodEOnGroundState:
    """Method E should work on GS molecules (no negative eigenvalues)."""

    def test_gs_method_e_no_problematic(self, ethane_mol: Q2MMMolecule) -> None:
        """GS ethane should have no problematic params."""
        ff_e, diag = estimate_force_constants_method_e(ethane_mol)
        n_problematic = sum(len(v) for v in diag["problematic"].values())
        assert n_problematic == 0, f"GS molecule should have no problematic params, got {n_problematic}"

    def test_gs_method_e_matches_raw(self, ethane_mol: Q2MMMolecule) -> None:
        """For GS, Method E result should equal raw estimation (D = identity)."""
        ff_e, _ = estimate_force_constants_method_e(ethane_mol)
        ff_raw = estimate_force_constants(ethane_mol)

        for be, br in zip(ff_e.bonds, ff_raw.bonds):
            np.testing.assert_allclose(be.force_constant, br.force_constant, atol=1e-10)
