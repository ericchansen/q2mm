"""Tests for CMAP torsion correction support (issue #115).

Tests the full CMAP pipeline:
1. CmapGrid dataclass validation
2. CHARMM .prm CMAP section parser
3. OpenMM CMAPTorsionForce integration
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import logging

    from q2mm.models.molecule import Q2MMMolecule

from q2mm.models.forcefield import (
    AngleParam,
    BondParam,
    CmapGrid,
    ForceField,
    FunctionalForm,
    TorsionParam,
    VdwParam,
)
from q2mm.parsers.charmm_cmap import load_cmap_from_prm, parse_cmap_section

# Path to the real CHARMM36 CMAP excerpt fixture
_FIXTURES = Path(__file__).parent / "fixtures"
_CHARMM36_CMAP = _FIXTURES / "cmap_charmm36_excerpt.prm"

# ---------------------------------------------------------------------------
# CmapGrid dataclass tests
# ---------------------------------------------------------------------------


class TestCmapGrid:
    """Tests for CmapGrid validation and construction."""

    def test_basic_construction(self) -> None:
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=4,
            energy=[0.0] * 16,
        )
        assert grid.resolution == 4
        assert len(grid.energy) == 16

    def test_resolution_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="expected 16"):
            CmapGrid(
                atom_types_phi=("C", "N", "CA", "C"),
                atom_types_psi=("N", "CA", "C", "N"),
                resolution=4,
                energy=[0.0] * 15,
            )

    def test_resolution_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="must be ≥ 2"):
            CmapGrid(
                atom_types_phi=("C", "N", "CA", "C"),
                atom_types_psi=("N", "CA", "C", "N"),
                resolution=1,
                energy=[0.0],
            )

    def test_energy_values_preserved(self) -> None:
        energy = [float(i) for i in range(9)]
        grid = CmapGrid(
            atom_types_phi=("A", "B", "C", "D"),
            atom_types_psi=("B", "C", "D", "E"),
            resolution=3,
            energy=energy,
        )
        assert grid.energy == energy

    def test_label_default(self) -> None:
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=2,
            energy=[0.0] * 4,
        )
        assert grid.label == ""

    def test_label_custom(self) -> None:
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=2,
            energy=[0.0] * 4,
            label="backbone_phi_psi",
        )
        assert grid.label == "backbone_phi_psi"


# ---------------------------------------------------------------------------
# ForceField CMAP integration tests
# ---------------------------------------------------------------------------


class TestForceFieldCmap:
    """Tests for CMAP fields on ForceField."""

    def test_empty_ff_has_no_cmap(self) -> None:
        ff = ForceField()
        assert not ff.has_cmap
        assert ff.cmaps == []

    def test_ff_with_cmap(self) -> None:
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=2,
            energy=[0.0, 1.0, 2.0, 3.0],
        )
        ff = ForceField(cmaps=[grid])
        assert ff.has_cmap
        assert len(ff.cmaps) == 1

    def test_cmap_not_in_param_vector(self) -> None:
        """CMAP grids must NOT be included in the optimizable parameter vector."""
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=2,
            energy=[0.0, 1.0, 2.0, 3.0],
        )
        ff = ForceField(
            bonds=[BondParam(("C", "N"), equilibrium=1.47, force_constant=100.0)],
            cmaps=[grid],
        )
        # n_params should only count bonds (2 params: k + eq)
        assert ff.n_params == 2

    def test_source_format_accepts_charmm_prm(self) -> None:
        ff = ForceField(source_format="charmm_prm")
        assert ff.source_format == "charmm_prm"


# ---------------------------------------------------------------------------
# CHARMM CMAP parser tests
# ---------------------------------------------------------------------------


class TestCmapParser:
    """Tests for parsing CMAP sections from CHARMM parameter files."""

    SIMPLE_CMAP = textwrap.dedent("""\
        CMAP
        CT1  N   CA   C   N   CA   C   CT1   4
         0.0  1.0  2.0  3.0
         4.0  5.0  6.0  7.0
         8.0  9.0 10.0 11.0
        12.0 13.0 14.0 15.0
    """)

    def test_parse_simple_grid(self) -> None:
        grids = parse_cmap_section(self.SIMPLE_CMAP)
        assert len(grids) == 1
        g = grids[0]
        assert g.atom_types_phi == ("CT1", "N", "CA", "C")
        assert g.atom_types_psi == ("N", "CA", "C", "CT1")
        assert g.resolution == 4
        assert g.energy == [float(i) for i in range(16)]

    def test_parse_with_comments(self) -> None:
        text = textwrap.dedent("""\
            CMAP
            ! This is a comment
            CT1  N   CA   C   N   CA   C   CT1   2
            ! phi = -180
             0.5  1.5   ! inline comment
            ! phi = 0
             2.5  3.5
        """)
        grids = parse_cmap_section(text)
        assert len(grids) == 1
        assert grids[0].energy == [0.5, 1.5, 2.5, 3.5]

    def test_parse_multiple_grids(self) -> None:
        text = textwrap.dedent("""\
            CMAP
            CT1  N   CA   C   N   CA   C   CT1   2
             1.0  2.0
             3.0  4.0
            CT2  N   CA   C   N   CA   C   CT2   2
             5.0  6.0
             7.0  8.0
        """)
        grids = parse_cmap_section(text)
        assert len(grids) == 2
        assert grids[0].atom_types_phi[0] == "CT1"
        assert grids[1].atom_types_phi[0] == "CT2"
        assert grids[0].energy == [1.0, 2.0, 3.0, 4.0]
        assert grids[1].energy == [5.0, 6.0, 7.0, 8.0]

    def test_parse_embedded_in_full_prm(self) -> None:
        text = textwrap.dedent("""\
            BONDS
            CT1  N   320.0   1.4300

            ANGLES
            N   CT1  C   50.0   107.0

            CMAP
            CT1  N   CA   C   N   CA   C   CT1   2
             0.0  0.0
             0.0  0.0

            NONBONDED
            CT1  0.0  -0.020  2.275
        """)
        grids = parse_cmap_section(text)
        assert len(grids) == 1
        assert grids[0].resolution == 2

    def test_parse_no_cmap_returns_empty(self) -> None:
        text = "BONDS\nCT1  N   320.0   1.4300\n"
        grids = parse_cmap_section(text)
        assert grids == []

    def test_parse_negative_values(self) -> None:
        text = textwrap.dedent("""\
            CMAP
            C  N  CA  C  N  CA  C  C  2
            -1.5  -2.3
             0.8  -0.4
        """)
        grids = parse_cmap_section(text)
        assert len(grids) == 1
        assert grids[0].energy == [-1.5, -2.3, 0.8, -0.4]


# ---------------------------------------------------------------------------
# OpenMM CMAP integration tests
# ---------------------------------------------------------------------------


@pytest.mark.openmm
class TestCmapOpenMM:
    """Tests for CMAP force creation in the OpenMM engine."""

    @pytest.fixture
    def _alanine_dipeptide_ff(self) -> ForceField:
        """Minimal alanine-dipeptide-like force field with CMAP.

        Simplified 5-atom linear chain: A-B-C-D-E
        with atom types matching phi (A-B-C-D) and psi (B-C-D-E).
        """
        # 4x4 CMAP grid with uniform non-zero values so we always detect energy
        resolution = 4
        energy = [5.0] * (resolution * resolution)  # 5 kcal/mol everywhere

        cmap_grid = CmapGrid(
            atom_types_phi=("A", "B", "C", "D"),
            atom_types_psi=("B", "C", "D", "E"),
            resolution=resolution,
            energy=energy,
        )

        ff = ForceField(
            bonds=[
                BondParam(("C", "C"), equilibrium=1.53, force_constant=100.0),
            ],
            angles=[
                AngleParam(("C", "C", "C"), equilibrium=109.5, force_constant=50.0),
            ],
            torsions=[
                TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=0.5),
            ],
            vdws=[
                VdwParam("C", radius=1.7, epsilon=0.05),
            ],
            cmaps=[cmap_grid],
            functional_form=FunctionalForm.HARMONIC,
        )
        return ff

    @pytest.fixture
    def _linear_chain_molecule(self) -> Q2MMMolecule:
        """5-atom linear chain molecule with atom types A-B-C-D-E."""
        from q2mm.models.molecule import Q2MMMolecule

        # Linear chain: 5 carbon atoms in a row with explicit atom types
        coords = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.53, 0.0, 0.0],
                [2.30, 1.22, 0.0],
                [3.83, 1.22, 0.0],
                [4.60, 2.44, 0.0],
            ]
        )
        mol = Q2MMMolecule(
            symbols=["C", "C", "C", "C", "C"],
            geometry=coords,
            atom_types=["A", "B", "C", "D", "E"],
        )
        return mol

    def test_cmap_force_created(self, _alanine_dipeptide_ff: ForceField, _linear_chain_molecule: Q2MMMolecule) -> None:
        """CMAP force should be created when FF has cmap grids."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        engine = OpenMMEngine(platform_name="CPU")
        handle = engine.create_context(_linear_chain_molecule, _alanine_dipeptide_ff)
        assert handle.cmap_force is not None
        assert len(handle.cmap_terms) > 0

    def test_no_cmap_when_ff_has_none(self, _linear_chain_molecule: Q2MMMolecule) -> None:
        """No CMAP force when FF has no cmap grids."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        ff = ForceField(
            bonds=[BondParam(("C", "C"), equilibrium=1.53, force_constant=100.0)],
            vdws=[VdwParam("C", radius=1.7, epsilon=0.05)],
            functional_form=FunctionalForm.HARMONIC,
        )
        engine = OpenMMEngine(platform_name="CPU")
        handle = engine.create_context(_linear_chain_molecule, ff)
        assert handle.cmap_force is None
        assert handle.cmap_terms == []

    def test_cmap_energy_contribution(
        self, _alanine_dipeptide_ff: ForceField, _linear_chain_molecule: Q2MMMolecule
    ) -> None:
        """CMAP should contribute to the total energy."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        # Energy with CMAP
        engine = OpenMMEngine(platform_name="CPU")
        handle_cmap = engine.create_context(_linear_chain_molecule, _alanine_dipeptide_ff)
        energy_with_cmap = engine.energy(handle_cmap)

        # Energy without CMAP
        ff_no_cmap = ForceField(
            bonds=_alanine_dipeptide_ff.bonds,
            angles=_alanine_dipeptide_ff.angles,
            torsions=_alanine_dipeptide_ff.torsions,
            vdws=_alanine_dipeptide_ff.vdws,
            functional_form=FunctionalForm.HARMONIC,
        )
        handle_no_cmap = engine.create_context(_linear_chain_molecule, ff_no_cmap)
        energy_without_cmap = engine.energy(handle_no_cmap)

        # The CMAP should change the energy
        assert energy_with_cmap != energy_without_cmap

    def test_cmap_term_records(self, _alanine_dipeptide_ff: ForceField, _linear_chain_molecule: Q2MMMolecule) -> None:
        """CMAP term records should contain correct atom indices and types."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        engine = OpenMMEngine(platform_name="CPU")
        handle = engine.create_context(_linear_chain_molecule, _alanine_dipeptide_ff)

        for term in handle.cmap_terms:
            assert len(term.phi_atoms) == 4
            assert len(term.psi_atoms) == 4
            assert term.phi_types == ("A", "B", "C", "D")
            assert term.psi_types == ("B", "C", "D", "E")
            # Phi and psi should overlap: psi starts where phi ends
            assert term.phi_atoms[1:] == term.psi_atoms[:3]

    def test_cmap_immutable_during_update(
        self, _alanine_dipeptide_ff: ForceField, _linear_chain_molecule: Q2MMMolecule
    ) -> None:
        """CMAP should not change when update_forcefield is called."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        engine = OpenMMEngine(platform_name="CPU")
        handle = engine.create_context(_linear_chain_molecule, _alanine_dipeptide_ff)

        # Get energy before update
        e1 = engine.energy(handle)

        # Update force field (change a bond parameter)
        new_ff = ForceField(
            bonds=[BondParam(("C", "C"), equilibrium=1.60, force_constant=110.0)],
            angles=_alanine_dipeptide_ff.angles,
            torsions=_alanine_dipeptide_ff.torsions,
            vdws=_alanine_dipeptide_ff.vdws,
            cmaps=_alanine_dipeptide_ff.cmaps,
            functional_form=FunctionalForm.HARMONIC,
        )
        engine.update_forcefield(handle, new_ff)
        e2 = engine.energy(handle)

        # Energy should change (bond params changed) but CMAP part stays fixed
        assert e1 != e2
        # CMAP force should still be present
        assert handle.cmap_force is not None

    def test_zero_cmap_grid_no_energy_change(self, _linear_chain_molecule: Q2MMMolecule) -> None:
        """A CMAP grid of all zeros should not change the energy."""
        from q2mm.backends.mm.openmm import OpenMMEngine

        zero_grid = CmapGrid(
            atom_types_phi=("A", "B", "C", "D"),
            atom_types_psi=("B", "C", "D", "E"),
            resolution=4,
            energy=[0.0] * 16,
        )
        ff_cmap = ForceField(
            bonds=[BondParam(("C", "C"), equilibrium=1.53, force_constant=100.0)],
            angles=[AngleParam(("C", "C", "C"), equilibrium=109.5, force_constant=50.0)],
            torsions=[TorsionParam(("C", "C", "C", "C"), periodicity=2, force_constant=0.5)],
            vdws=[VdwParam("C", radius=1.7, epsilon=0.05)],
            cmaps=[zero_grid],
            functional_form=FunctionalForm.HARMONIC,
        )
        ff_no_cmap = ForceField(
            bonds=ff_cmap.bonds,
            angles=ff_cmap.angles,
            torsions=ff_cmap.torsions,
            vdws=ff_cmap.vdws,
            functional_form=FunctionalForm.HARMONIC,
        )

        engine = OpenMMEngine(platform_name="CPU")
        e_cmap = engine.energy(engine.create_context(_linear_chain_molecule, ff_cmap))
        e_none = engine.energy(engine.create_context(_linear_chain_molecule, ff_no_cmap))

        assert abs(e_cmap - e_none) < 1e-10


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestCmapEdgeCases:
    """Edge case and error handling tests."""

    def test_cmap_grid_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            CmapGrid(
                atom_types_phi=("C", "N", "CA", "C"),
                atom_types_psi=("N", "CA", "C", "N"),
                resolution=3,
                energy=[0.0] * 10,  # should be 9
            )

    def test_cmap_grid_resolution_2(self) -> None:
        """Minimum valid CMAP grid (2x2)."""
        grid = CmapGrid(
            atom_types_phi=("C", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "N"),
            resolution=2,
            energy=[1.0, 2.0, 3.0, 4.0],
        )
        assert grid.resolution == 2

    def test_cmap_grid_large_resolution(self) -> None:
        """Standard CHARMM resolution (24x24 = 576 values)."""
        grid = CmapGrid(
            atom_types_phi=("CT1", "N", "CA", "C"),
            atom_types_psi=("N", "CA", "C", "CT1"),
            resolution=24,
            energy=[0.0] * 576,
        )
        assert len(grid.energy) == 576

    def test_parser_handles_empty_string(self) -> None:
        grids = parse_cmap_section("")
        assert grids == []

    def test_parser_incomplete_grid_warns(self, caplog: logging.LogCaptureFixture) -> None:
        """Incomplete grid data should warn and skip."""
        text = textwrap.dedent("""\
            CMAP
            CT1  N   CA   C   N   CA   C   CT1   3
             1.0  2.0  3.0
             4.0  5.0
        """)
        import logging

        with caplog.at_level(logging.WARNING):
            grids = parse_cmap_section(text)
        assert grids == []
        assert "Incomplete" in caplog.text

    def test_parser_scientific_notation(self) -> None:
        text = textwrap.dedent("""\
            CMAP
            C  N  CA  C  N  CA  C  C  2
            1.5e-3  -2.0e1
            0.0     1.234e2
        """)
        grids = parse_cmap_section(text)
        assert len(grids) == 1
        assert grids[0].energy[0] == pytest.approx(0.0015)
        assert grids[0].energy[1] == pytest.approx(-20.0)
        assert grids[0].energy[3] == pytest.approx(123.4)


# ---------------------------------------------------------------------------
# CHARMM36 reference file validation tests
# ---------------------------------------------------------------------------


class TestCmapCharmm36Reference:
    """Validate the parser against real CHARMM36 CMAP data.

    Reference: ``par_all36_prot.prm`` (CHARMM36 protein parameters).
    Source: https://github.com/ParmEd/ParmEd/blob/master/examples/charmm/toppar/par_all36_prot.prm

    The fixture ``test/fixtures/cmap_charmm36_excerpt.prm`` contains the
    first two 24×24 CMAP grids (alanine and alanine-before-proline maps)
    extracted from that file.
    """

    @pytest.fixture
    def grids(self) -> list[CmapGrid]:
        """Parse grids from the real CHARMM36 fixture file."""
        assert _CHARMM36_CMAP.exists(), f"Fixture not found: {_CHARMM36_CMAP}"
        return load_cmap_from_prm(_CHARMM36_CMAP)

    def test_grid_count(self, grids: list[CmapGrid]) -> None:
        """Fixture contains two CMAP grids."""
        assert len(grids) == 2

    def test_grid_resolution(self, grids: list[CmapGrid]) -> None:
        """Both grids must be 24×24 (standard CHARMM protein CMAP)."""
        for g in grids:
            assert g.resolution == 24
            assert len(g.energy) == 576

    def test_alanine_atom_types(self, grids: list[CmapGrid]) -> None:
        """First grid: alanine backbone (C-NH1-CT1-C / NH1-CT1-C-NH1)."""
        ala = grids[0]
        assert ala.atom_types_phi == ("C", "NH1", "CT1", "C")
        assert ala.atom_types_psi == ("NH1", "CT1", "C", "NH1")

    def test_alanine_before_proline_atom_types(self, grids: list[CmapGrid]) -> None:
        """Second grid: alanine before proline (C-NH1-CT1-C / NH1-CT1-C-N)."""
        ala_pro = grids[1]
        assert ala_pro.atom_types_phi == ("C", "NH1", "CT1", "C")
        assert ala_pro.atom_types_psi == ("NH1", "CT1", "C", "N")

    def test_alanine_known_values(self, grids: list[CmapGrid]) -> None:
        """Spot-check alanine grid values against the .prm file.

        First row (phi = -180°) starts: 0.126790, 0.768700, 0.971260, ...
        """
        ala = grids[0]
        assert ala.energy[0] == pytest.approx(0.126790)
        assert ala.energy[1] == pytest.approx(0.768700)
        assert ala.energy[2] == pytest.approx(0.971260)
        assert ala.energy[3] == pytest.approx(1.250970)
        # Last value in the grid (phi = 165°, psi = 165°)
        assert ala.energy[-1] == pytest.approx(-1.814368)

    def test_values_in_physical_range(self, grids: list[CmapGrid]) -> None:
        """CMAP energy corrections should be in a physically reasonable range.

        CHARMM36 backbone corrections are typically -8 to +8 kcal/mol.
        """
        for g in grids:
            assert all(-15.0 < v < 15.0 for v in g.energy), (
                f"CMAP values outside expected range: min={min(g.energy):.2f}, max={max(g.energy):.2f}"
            )

    def test_load_cmap_from_prm_api(self) -> None:
        """The file-based API should produce the same results."""
        grids_text = parse_cmap_section(_CHARMM36_CMAP.read_text())
        grids_file = load_cmap_from_prm(_CHARMM36_CMAP)
        assert len(grids_text) == len(grids_file)
        for gt, gf in zip(grids_text, grids_file):
            assert gt.atom_types_phi == gf.atom_types_phi
            assert gt.atom_types_psi == gf.atom_types_psi
            assert gt.energy == gf.energy
