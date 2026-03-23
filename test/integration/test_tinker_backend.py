"""Tinker-engine-specific tests.

Contract tests (energy, frequencies, minimize) are in
test_engine_contract.py and run for every registered engine.  This file
covers only behaviour unique to the Tinker backend:

* File-path-based API (passing XYZ paths instead of Q2MMMolecule objects)
* SN2 TS imaginary-frequency check on the MM3 surface
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
QM_REF = REPO_ROOT / "examples" / "sn2-test" / "qm-reference"

try:
    from q2mm.backends.mm.tinker import TinkerEngine

    _engine = TinkerEngine()
    HAS_TINKER = _engine.is_available()
except (ImportError, FileNotFoundError):
    HAS_TINKER = False

pytestmark = [
    pytest.mark.tinker,
    pytest.mark.skipif(not HAS_TINKER, reason="Tinker not installed"),
]

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
SN2_XYZ = QM_REF / "sn2-ts-optimized.xyz"


class TestTinkerFilePathAPI:
    """Tinker accepts raw file paths in addition to Q2MMMolecule objects."""

    def setup_method(self) -> None:
        self.engine = TinkerEngine()

    @pytest.mark.skipif(not CH3F_XYZ.exists(), reason="CH3F fixture not found")
    def test_energy_from_file_path(self) -> None:
        energy = self.engine.energy(str(CH3F_XYZ))
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    @pytest.mark.skipif(not CH3F_XYZ.exists(), reason="CH3F fixture not found")
    def test_minimize_from_file_path(self) -> None:
        energy, atoms, coords = self.engine.minimize(str(CH3F_XYZ), rms_grad=0.1)
        assert isinstance(energy, float)
        assert len(atoms) == 5  # CH3F
        assert coords.shape == (5, 3)

    @pytest.mark.skipif(not SN2_XYZ.exists(), reason="SN2 TS fixture not found")
    def test_sn2_ts_has_imaginary_frequencies(self) -> None:
        """SN2 TS is not an MM minimum — expect imaginary frequencies."""
        freqs = self.engine.frequencies(str(SN2_XYZ))
        assert len(freqs) == 18  # 6 atoms × 3
        n_imaginary = sum(1 for f in freqs if f < -1.0)
        assert n_imaginary > 0, "TS should have imaginary frequencies on MM surface"
