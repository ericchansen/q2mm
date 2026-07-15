"""Tinker-backend-specific tests.

Contract tests (energy, frequencies, minimize) are in
test_engine_contract.py and run for every registered backend.  This file
covers only behaviour unique to the Tinker backend:

* Prepared-session energy/minimize/frequencies on a Molecule + MM3 force field
* SN2 TS imaginary-frequency check on the MM3 surface
"""

from __future__ import annotations
from q2mm.backends.contracts import (
    EnergyRequest,
    FrequencyRequest,
    MinimizationRequest,
)
from test.backend_fixtures import backend_is_usable, load_test_backend, param_vector, prepare_case

from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
from q2mm.resources import sn2_reference_dir

QM_REF = sn2_reference_dir()

HAS_TINKER = backend_is_usable("tinker")

pytestmark = [
    pytest.mark.tinker,
    pytest.mark.skipif(not HAS_TINKER, reason="Tinker not installed"),
]

CH3F_XYZ = QM_REF / "ch3f-optimized.xyz"
SN2_XYZ = QM_REF / "sn2-ts-optimized.xyz"


class TestTinkerFilePathAPI:
    """Tinker accepts raw file paths in addition to Molecule objects."""

    def setup_method(self) -> None:
        self.backend = load_test_backend("tinker")

    @staticmethod
    def _mm3_ff(mol: object) -> object:
        from q2mm.models.forcefield import ForceField, FunctionalForm

        return ForceField.create_for_molecule(mol, functional_form=FunctionalForm.MM3)

    def test_energy_from_molecule(self) -> None:
        from q2mm.io.xyz import load_xyz

        mol = load_xyz(str(CH3F_XYZ), bond_tolerance=1.5)
        energy = (
            prepare_case(self.backend, mol, self._mm3_ff(mol))
            .energy(EnergyRequest(parameters=param_vector(self._mm3_ff(mol))))
            .energy
        )
        assert isinstance(energy, float)
        assert np.isfinite(energy)

    def test_minimize_from_molecule(self) -> None:
        from q2mm.io.xyz import load_xyz

        mol = load_xyz(str(CH3F_XYZ), bond_tolerance=1.5)
        _ff = self._mm3_ff(mol)
        _min = prepare_case(self.backend, mol, _ff).minimize(MinimizationRequest(parameters=param_vector(_ff)))
        energy, atoms, coords = _min.energy, list(_min.symbols), np.asarray(_min.coordinates)
        assert isinstance(energy, float)
        assert len(atoms) == 5  # CH3F
        assert coords.shape == (5, 3)

    def test_sn2_ts_has_imaginary_frequencies(self) -> None:
        """SN2 TS is not an MM minimum — expect imaginary frequencies."""
        from q2mm.io.xyz import load_xyz

        mol = load_xyz(str(SN2_XYZ), charge=-1, bond_tolerance=1.5)
        freqs = [
            float(_f)
            for _f in prepare_case(self.backend, mol, self._mm3_ff(mol))
            .frequencies(FrequencyRequest(parameters=param_vector(self._mm3_ff(mol))))
            .frequencies
        ]
        assert len(freqs) == 18  # 6 atoms × 3
        n_imaginary = sum(1 for f in freqs if f < -1.0)
        assert n_imaginary > 0, "TS should have imaginary frequencies on MM surface"
