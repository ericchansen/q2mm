"""Tests for q2mm.io.fchk — Gaussian formatted checkpoint file parser."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q2mm.io.fchk import parse_fchk
from test._shared import GS_FCHK, TS_FCHK

# Match the return type of parse_fchk exactly.
FchkResult = tuple[list[str], np.ndarray, np.ndarray | None, int | None, int | None]


@pytest.fixture()
def ethane_gs() -> FchkResult:
    """Parse the ethane ground-state .fchk fixture."""
    return parse_fchk(GS_FCHK)


@pytest.fixture()
def ethane_ts() -> FchkResult:
    """Parse the ethane transition-state .fchk fixture."""
    return parse_fchk(TS_FCHK)


class TestParseFchkEthaneGS:
    def test_symbols(self, ethane_gs: FchkResult) -> None:
        symbols, _, _, _, _ = ethane_gs
        assert len(symbols) == 8
        assert symbols.count("C") == 2
        assert symbols.count("H") == 6

    def test_coords_shape(self, ethane_gs: FchkResult) -> None:
        _, coords, _, _, _ = ethane_gs
        assert coords.shape == (8, 3)

    def test_coords_in_angstrom(self, ethane_gs: FchkResult) -> None:
        symbols, coords, _, _, _ = ethane_gs
        # C-H bond should be ~1.09 A; verify we're in Angstrom, not Bohr (~2 Bohr)
        c_idx = symbols.index("C")
        h_idx = symbols.index("H")
        ch_dist = np.linalg.norm(coords[c_idx] - coords[h_idx])
        assert 0.9 < ch_dist < 1.3, f"C-H distance {ch_dist} not in Angstrom range"

    def test_hessian_present_and_symmetric(self, ethane_gs: FchkResult) -> None:
        _, _, hessian, _, _ = ethane_gs
        assert hessian is not None
        n = 8 * 3
        assert hessian.shape == (n, n)
        np.testing.assert_allclose(hessian, hessian.T, atol=1e-15)

    def test_charge_and_multiplicity(self, ethane_gs: FchkResult) -> None:
        _, _, _, charge, mult = ethane_gs
        assert charge == 0
        assert mult == 1


class TestParseFchkEthaneTS:
    def test_symbols(self, ethane_ts: FchkResult) -> None:
        symbols, _, _, _, _ = ethane_ts
        assert len(symbols) == 8
        assert symbols.count("C") == 2

    def test_has_hessian(self, ethane_ts: FchkResult) -> None:
        _, _, hessian, _, _ = ethane_ts
        assert hessian is not None

    def test_ts_hessian_has_negative_eigenvalue(self, ethane_ts: FchkResult) -> None:
        """A TS .fchk should have at least one negative Hessian eigenvalue."""
        _, _, hessian, _, _ = ethane_ts
        eigenvalues = np.linalg.eigvalsh(hessian)
        assert np.any(eigenvalues < -1e-6), "TS Hessian should have a negative eigenvalue"


class TestParseFchkErrors:
    def test_nonexistent_file(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            parse_fchk(tmp_path / "nonexistent.fchk")

    def test_empty_file(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.fchk"
        empty.write_text("")
        with pytest.raises(ValueError, match="Could not parse"):
            parse_fchk(empty)
