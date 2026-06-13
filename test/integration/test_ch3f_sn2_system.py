"""F⁻ + CH3F SN2 transition state — system registration validation.

The TS that Limé & Norrby 2015 used to document the negative-bend-FC
phenomenon (the FACAF angle force constant going to zero under naive
Method C fitting) and to motivate their Method E2 hybrid protocol.

These tests verify the system registration and qualitative QFUERZA
behavior; the Method E2 workflow itself (forthcoming in Phase 9.D)
will exercise the full Limé & Norrby protocol against this system as
its validation target.

References
----------
- Limé, E.; Norrby, P.-O. *J. Comput. Chem.* **2015**, *36*, 244–250.
  DOI: 10.1002/jcc.23797.

"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.mark.jax
class TestCh3fSn2System:
    """Smoke tests for the ``ch3f-sn2`` system registration."""

    def test_system_is_registered(self) -> None:
        """``ch3f-sn2`` appears in the SYSTEMS registry with TS metadata."""
        from q2mm.diagnostics.systems import SYSTEMS

        assert "ch3f-sn2" in SYSTEMS
        spec = SYSTEMS["ch3f-sn2"]
        assert spec.ff_strategy == "qfuerza_fresh"
        assert spec.metadata["is_transition_state"] is True
        assert "Limé" in spec.metadata["publication"] or "Lime" in spec.metadata["publication"]
        assert spec.metadata["doi"] == "10.1002/jcc.23797"

    def test_loader_returns_one_molecule_with_hessian(self) -> None:
        """Loader produces exactly one molecule, with an 18×18 Hessian.

        Also verifies ``charge=-1`` — the QM Hessian was computed on
        the anionic F⁻ + CH3F complex (see
        ``examples/sn2-test/generate_qm_data.py``), so the loaded
        molecule must match or downstream electrostatic / charge-
        sensitive code paths would silently disagree with the QM.
        """
        from q2mm.diagnostics.systems import _load_ch3f_sn2_molecules

        mols = _load_ch3f_sn2_molecules()
        assert len(mols) == 1
        mol = mols[0]
        assert mol.charge == -1, f"Expected charge=-1 (anionic), got {mol.charge}"
        assert mol.hessian is not None
        assert mol.hessian.shape == (18, 18)
        np.testing.assert_allclose(mol.hessian, mol.hessian.T, atol=1e-10)

    def test_geometry_has_two_c_f_bonds(self) -> None:
        """At the TS, both C-F bonds (~1.85 Å each) are detected.

        The default ``Q2MMMolecule.from_xyz`` ``bond_tolerance`` of
        ``1.3`` (a unitless multiplier on the sum of covalent radii)
        misses the partially-formed TS C-F bonds; the loader uses
        ``1.5`` (the canonical TS value documented on
        ``Q2MMMolecule.from_xyz``).
        """
        from q2mm.diagnostics.systems import _load_ch3f_sn2_molecules

        mol = _load_ch3f_sn2_molecules()[0]
        cf_bonds = [b for b in mol.bonds if set(b.elements) == {"C", "F"}]
        assert len(cf_bonds) == 2, (
            f"Expected 2 C-F bonds at the SN2 TS; found {len(cf_bonds)}. "
            "Check the ``bond_tolerance`` in ``_load_ch3f_sn2_molecules``."
        )
        # Both C-F distances should be close to 1.85 Å (the published TS geometry).
        for bond in cf_bonds:
            assert 1.7 < bond.length < 2.0, f"C-F bond length {bond.length:.3f} Å out of expected range"

    def test_hessian_has_one_imaginary_mode(self) -> None:
        """The TS Hessian has exactly one substantially-negative eigenvalue.

        Translational/rotational modes appear as near-zero eigenvalues
        (positive or negative due to numerical noise) — the threshold
        ``< -1e-3`` distinguishes those from the genuine
        reaction-coordinate mode.
        """
        from q2mm.diagnostics.systems import _load_ch3f_sn2_molecules

        mol = _load_ch3f_sn2_molecules()[0]
        evals = np.linalg.eigvalsh(mol.hessian)
        substantially_negative = evals[evals < -1e-3]
        assert len(substantially_negative) == 1, (
            f"Expected 1 substantially-negative eigenvalue (reaction coordinate); "
            f"found {len(substantially_negative)}: {substantially_negative}"
        )

    def test_load_system_succeeds_with_default_kwargs(self) -> None:
        """``load_system('ch3f-sn2')`` produces a usable SystemData.

        The default ``starting_point="qfuerza"`` is a no-op for
        ``qfuerza_fresh`` strategy (FF is already QFUERZA-derived);
        the default ``qfuerza_replace_with=1.0`` applies during the
        Seminario projection.
        """
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.diagnostics.systems import load_system

        sd = load_system("ch3f-sn2", engine=JaxEngine())
        # 6-atom TS → 18 Cartesian DOFs → 12 vibrational modes → 11 real
        # (excluding the imaginary reaction-coordinate mode).
        # qm_freqs_per_mol contains the QM-derived frequencies that
        # made it past the 50 cm⁻¹ threshold and were not imaginary.
        assert len(sd.qm_freqs_per_mol) == 1
        assert len(sd.qm_freqs_per_mol[0]) >= 8, (
            "Expected ≥8 real QM frequencies above the 50 cm⁻¹ threshold; "
            f"got {len(sd.qm_freqs_per_mol[0])}: {sd.qm_freqs_per_mol[0]}"
        )

    def test_loader_raises_targeted_error_when_sn2_files_missing(self, tmp_path: Path) -> None:
        """Loader emits a targeted FileNotFoundError when SN2 TS files are absent.

        ``_find_ch3f_data_dir`` only checks for the GS
        ``ch3f-optimized.xyz`` — a directory with the GS file but
        without the SN2 TS files (a partial checkout or work-in-
        progress data dir) would otherwise fall through to a less
        actionable error inside ``Q2MMMolecule.from_xyz`` or
        ``np.load``.
        """
        from q2mm.diagnostics.systems import _load_ch3f_sn2_molecules

        # Empty data dir → no SN2 TS files
        with pytest.raises(FileNotFoundError, match="SN2 TS reference data missing"):
            _load_ch3f_sn2_molecules(data_dir=tmp_path)

    def test_qfuerza_with_small_replace_with_produces_positive_facaf(self) -> None:
        """Phase 9.1 sensitivity finding extends to this system.

        Limé & Norrby 2015 ¶97 reports the FACAF bend force constant
        going negative under naive Method C (``replace_with=1.0``)
        fitting.  A smaller ``replace_with`` (e.g. 0.03 Ha/Bohr²,
        their Method D "natural" value) should keep the angle FC
        positive in the QFUERZA-projected starting parameters.  This
        test verifies the same kwarg-controlled behavior we observed
        on rh-conjugate's C-C-O angle is present here.
        """
        from q2mm.backends.mm.jax_engine import JaxEngine
        from q2mm.diagnostics.systems import load_system

        engine = JaxEngine()
        sd_large = load_system("ch3f-sn2", engine=engine, qfuerza_replace_with=1.0)
        sd_small = load_system("ch3f-sn2", engine=engine, qfuerza_replace_with=0.03)

        # FACAF is the F-C-F angle; the only such triple in this molecule.
        facaf_large = [a for a in sd_large.forcefield.angles if set(a.elements) == {"F", "C"} and a.elements[1] == "C"]
        facaf_small = [a for a in sd_small.forcefield.angles if set(a.elements) == {"F", "C"} and a.elements[1] == "C"]
        assert len(facaf_large) == 1, f"Expected 1 FACAF angle, got {len(facaf_large)}"
        assert len(facaf_small) == 1
        # (a) The two should differ — verifies the kwarg plumbing reaches Seminario.
        assert facaf_large[0].force_constant != facaf_small[0].force_constant, (
            "FACAF force constant should differ between replace_with=1.0 and =0.03 — "
            f"got {facaf_large[0].force_constant} vs {facaf_small[0].force_constant}. "
            "Either the kwarg isn't reaching Seminario or this Hessian doesn't exhibit "
            "the Limé & Norrby sensitivity."
        )
        # (b) Small-replace_with FACAF must be positive (physical bend) — guards
        # the invariant the test name and docstring claim.  A regression making
        # the small-replacement FACAF go negative would otherwise still pass.
        assert facaf_small[0].force_constant > 0, (
            f"Small replace_with=0.03 should keep FACAF positive (Method D 'natural' "
            f"regime per Limé & Norrby 2015 ¶97); got {facaf_small[0].force_constant}"
        )
