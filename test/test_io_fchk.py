"""Tests for q2mm.io.fchk — Gaussian formatted checkpoint file parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q2mm.io.fchk import load_fchk
from q2mm.models.hessian import HessianUnits
from q2mm.models.molecule import Molecule
from test._shared import GS_FCHK, TS_FCHK


@pytest.fixture()
def ethane_gs() -> Molecule:
    """Parse the ethane ground-state .fchk fixture."""
    return load_fchk(GS_FCHK)


@pytest.fixture()
def ethane_ts() -> Molecule:
    """Parse the ethane transition-state .fchk fixture."""
    return load_fchk(TS_FCHK)


class TestParseFchkEthaneGS:
    def test_symbols(self, ethane_gs: Molecule) -> None:
        assert len(ethane_gs.symbols) == 8
        assert ethane_gs.symbols.count("C") == 2
        assert ethane_gs.symbols.count("H") == 6

    def test_coords_shape(self, ethane_gs: Molecule) -> None:
        assert ethane_gs.geometry.shape == (8, 3)

    def test_coords_in_angstrom(self, ethane_gs: Molecule) -> None:
        # C-H bond should be ~1.09 A; verify we're in Angstrom, not Bohr (~2 Bohr)
        c_idx = ethane_gs.symbols.index("C")
        h_idx = ethane_gs.symbols.index("H")
        ch_dist = np.linalg.norm(ethane_gs.geometry[c_idx] - ethane_gs.geometry[h_idx])
        assert 0.9 < ch_dist < 1.3, f"C-H distance {ch_dist} not in Angstrom range"

    def test_hessian_present_and_symmetric(self, ethane_gs: Molecule) -> None:
        assert ethane_gs.hessian is not None
        n = 8 * 3
        assert ethane_gs.hessian.shape == (n, n)
        np.testing.assert_allclose(ethane_gs.hessian, ethane_gs.hessian.T, atol=1e-15)
        assert ethane_gs.hessian_provenance.units is HessianUnits.ATOMIC
        assert ethane_gs.hessian_provenance.source == "fchk"
        assert ethane_gs.hessian_provenance.path == str(GS_FCHK.resolve())

    def test_charge_and_multiplicity(self, ethane_gs: Molecule) -> None:
        assert ethane_gs.charge == 0
        assert ethane_gs.multiplicity == 1


class TestParseFchkEthaneTS:
    def test_symbols(self, ethane_ts: Molecule) -> None:
        assert len(ethane_ts.symbols) == 8
        assert ethane_ts.symbols.count("C") == 2

    def test_has_hessian(self, ethane_ts: Molecule) -> None:
        assert ethane_ts.hessian is not None

    def test_ts_hessian_has_negative_eigenvalue(self, ethane_ts: Molecule) -> None:
        """A TS .fchk should have at least one negative Hessian eigenvalue."""
        eigenvalues = np.linalg.eigvalsh(ethane_ts.hessian)
        assert np.any(eigenvalues < -1e-6), "TS Hessian should have a negative eigenvalue"


class TestParseFchkErrors:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            load_fchk(tmp_path / "nonexistent.fchk")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.fchk"
        empty.write_text("")
        with pytest.raises(ValueError, match="Could not parse"):
            load_fchk(empty)
