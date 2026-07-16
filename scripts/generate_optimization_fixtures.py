#!/usr/bin/env python3
"""Generate golden fixtures for optimization E2E tests.

Runs the full pipeline on the water test system and saves results as JSON.
Used to detect regressions when the pipeline changes intentionally.

Usage:
    python scripts/generate_optimization_fixtures.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from q2mm.backends.registry import load_backend
from q2mm.models.forcefield import AngleParam, BondParam, ForceField, FunctionalForm
from q2mm.models.molecule import Molecule
from q2mm.models.observations import ObservationSet
from q2mm.models.parameters import ActiveParameterSpace, ParameterLayout
from q2mm.models.problem import StationaryPointKind
from q2mm.objectives.plan import ObjectivePlan
from q2mm.objectives.python import PythonObjectiveExecutor
from q2mm.optimizers.scipy_opt import ScipyOptimizer

OUTPUT_PATH = REPO_ROOT / "test" / "fixtures" / "optimization_golden.json"


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Molecule:
    theta = np.deg2rad(angle_deg)
    return Molecule(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [bond_length, 0.0, 0.0],
                [bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0],
            ]
        ),
        name="water",
        bond_tolerance=1.5,
    )


def _water_ff(
    bond_k: float = 503.6, bond_r0: float = 0.96, angle_k: float = 57.6, angle_eq: float = 104.5
) -> ForceField:
    """Build a minimal water force field.

    MM3, not HARMONIC: this is the exact generator for
    ``test/fixtures/optimization_golden.json``, produced under OpenMM's
    old implicit-MM3 branch. The fixture is frozen (never regenerated
    to "fix" a physics change), but this script must keep matching what
    it actually encodes, or a future re-run of it would silently drift
    from the committed golden.
    """
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
        functional_form=FunctionalForm.MM3,
    )


def _mm_energy(backend: object, mol: object, ff: object) -> float:
    """Single-point MM energy via a prepared session."""
    from q2mm.backends.contracts import EnergyRequest, PreparationRequest

    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    return float(prepared.energy(EnergyRequest(parameters=ParameterLayout.from_force_field(ff).vector(ff))).energy)


def _mm_frequencies(backend: object, mol: object, ff: object) -> list[float]:
    """MM vibrational frequencies via a prepared session."""
    from q2mm.backends.contracts import FrequencyRequest, PreparationRequest

    prepared = backend.prepare(PreparationRequest(case_id="0", molecule=mol, force_field=ff))
    params = ParameterLayout.from_force_field(ff).vector(ff)
    return [float(f) for f in prepared.frequencies(FrequencyRequest(parameters=params)).frequencies]


def _make_water_problem(
    backend: object | None = None, perturb_k: float = 1.5, perturb_eq: float = 5.0
) -> tuple[ForceField, ForceField, list[Molecule], ObservationSet, object]:
    """Create a water optimization problem with known true parameters."""
    if backend is None:
        backend = load_backend("openmm")
    true_ff = _water_ff(bond_k=503.6, bond_r0=0.96, angle_k=57.6, angle_eq=104.5)

    mol_eq = _water(104.5, 0.96)
    mol_wide = _water(115.0, 0.96)
    mol_long = _water(104.5, 1.05)

    ref = ObservationSet()
    for i, mol in enumerate([mol_eq, mol_wide, mol_long]):
        ref = ref.with_energy(_mm_energy(backend, mol, true_ff), weight=1.0, case_id=str(i))

    freqs = _mm_frequencies(backend, mol_eq, true_ff)
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:
            ref = ref.with_frequency(f, data_idx=j, weight=0.001, case_id="0")

    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, [mol_eq, mol_wide, mol_long], ref, backend


def main() -> int:
    """Generate and save optimization golden fixture data."""
    print("Generating optimization golden fixture ...")

    true_ff, guess_ff, mols, ref, backend = _make_water_problem()
    layout = ParameterLayout.from_force_field(guess_ff)
    space = ActiveParameterSpace.all_active(layout, guess_ff)
    plan = ObjectivePlan(
        case_ids=tuple(str(i) for i in range(len(mols))),
        molecules=tuple(mols),
        stationary_points=tuple(StationaryPointKind.GROUND_STATE for _ in mols),
        observations=ref,
        layout=layout,
        active_space=space,
    )
    evaluator = PythonObjectiveExecutor(plan, backend, guess_ff)
    opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
    result = opt.optimize(evaluator, space)

    fixture = {
        "metadata": {
            "description": "Golden fixture for water FF optimization regression test.",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "method": "L-BFGS-B",
            "maxiter": 200,
            "system": "water (O, H, H)",
        },
        "initial_params": result.initial_params.tolist(),
        "final_params": result.final_params.tolist(),
        "initial_score": result.initial_score,
        "final_score": result.final_score,
        "n_evaluations": result.n_evaluations,
        "improvement": result.improvement,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(fixture, indent=2) + "\n")
    print(f"Written to {OUTPUT_PATH}")
    print(f"  initial_score: {result.initial_score:.6f}")
    print(f"  final_score:   {result.final_score:.6f}")
    print(f"  improvement:   {result.improvement:.2%}")
    print(f"  n_evaluations: {result.n_evaluations}")
    print(f"  initial_params: {result.initial_params.tolist()}")
    print(f"  final_params:   {result.final_params.tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
