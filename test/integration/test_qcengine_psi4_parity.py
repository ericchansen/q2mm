"""Optional QCEngine-to-direct-Psi4 atomic-result parity."""

from __future__ import annotations

import numpy as np
import pytest

from q2mm.backends.contracts import (
    PreparationRequest,
    ReferenceEnergyRequest,
    ReferenceHessianRequest,
)
from q2mm.backends.qm.psi4 import Psi4Backend
from q2mm.backends.reference.qcengine import QCEngineBackend
from q2mm.models.molecule import Molecule

pytestmark = [pytest.mark.psi4, pytest.mark.integration, pytest.mark.cross_backend]


def test_qcengine_psi4_energy_and_hessian_match_direct_psi4() -> None:
    molecule = Molecule(
        symbols=("H", "H"),
        geometry=np.array([[0.0, 0.0, -0.35], [0.0, 0.0, 0.35]]),
        bonds=(),
        angles=(),
        torsions=(),
    )
    direct = Psi4Backend(method="hf", basis="sto-3g", n_threads=1)
    through_engine = QCEngineBackend(
        program="psi4",
        method="hf",
        basis="sto-3g",
        task_config={"ncores": 1, "memory": 1.0},
    )
    direct_session = direct.prepare(PreparationRequest(case_id="h2-direct", molecule=molecule))
    engine_session = through_engine.prepare(PreparationRequest(case_id="h2-qcengine", molecule=molecule))

    direct_energy = direct_session.energy(ReferenceEnergyRequest())
    engine_energy = engine_session.energy(ReferenceEnergyRequest())
    assert engine_energy.energy == pytest.approx(direct_energy.energy, abs=1e-9)

    direct_hessian = direct_session.hessian(ReferenceHessianRequest())
    engine_hessian = engine_session.hessian(ReferenceHessianRequest())
    np.testing.assert_allclose(
        np.linalg.eigvalsh(engine_hessian.hessian),
        np.linalg.eigvalsh(direct_hessian.hessian),
        rtol=1e-7,
        atol=1e-8,
    )
