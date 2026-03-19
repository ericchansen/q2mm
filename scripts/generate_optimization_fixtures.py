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

from q2mm.backends.mm.openmm import OpenMMEngine
from q2mm.models.forcefield import AngleParam, BondParam, ForceField
from q2mm.models.molecule import Q2MMMolecule
from q2mm.optimizers.objective import ObjectiveFunction, ReferenceData
from q2mm.optimizers.scipy_opt import ScipyOptimizer

OUTPUT_PATH = REPO_ROOT / "test" / "fixtures" / "optimization_golden.json"


def _water(angle_deg: float = 104.5, bond_length: float = 0.96) -> Q2MMMolecule:
    theta = np.deg2rad(angle_deg)
    return Q2MMMolecule(
        symbols=["O", "H", "H"],
        geometry=np.array([
            [0.0, 0.0, 0.0],
            [bond_length, 0.0, 0.0],
            [bond_length * np.cos(theta), bond_length * np.sin(theta), 0.0],
        ]),
        name="water",
        bond_tolerance=1.5,
    )


def _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5) -> ForceField:
    return ForceField(
        name="water-test",
        bonds=[BondParam(elements=("H", "O"), force_constant=bond_k, equilibrium=bond_r0)],
        angles=[AngleParam(elements=("H", "O", "H"), force_constant=angle_k, equilibrium=angle_eq)],
    )


def _make_water_problem(engine=None, perturb_k=1.5, perturb_eq=5.0):
    """Create a water optimization problem with known true parameters."""
    if engine is None:
        engine = OpenMMEngine()
    true_ff = _water_ff(bond_k=7.0, bond_r0=0.96, angle_k=0.8, angle_eq=104.5)

    mol_eq = _water(104.5, 0.96)
    mol_wide = _water(115.0, 0.96)
    mol_long = _water(104.5, 1.05)

    ref = ReferenceData()
    for i, mol in enumerate([mol_eq, mol_wide, mol_long]):
        ref.add_energy(engine.energy(mol, true_ff), weight=1.0, molecule_idx=i)

    openmm = OpenMMEngine()
    freqs = openmm.frequencies(mol_eq, true_ff)
    for j, f in enumerate(freqs):
        if abs(f) > 50.0:
            ref.add_frequency(f, data_idx=j, weight=0.001, molecule_idx=0)

    guess_ff = _water_ff(
        bond_k=true_ff.bonds[0].force_constant + perturb_k,
        bond_r0=true_ff.bonds[0].equilibrium + 0.05,
        angle_k=true_ff.angles[0].force_constant + 0.3,
        angle_eq=true_ff.angles[0].equilibrium + perturb_eq,
    )

    return true_ff, guess_ff, [mol_eq, mol_wide, mol_long], ref, engine


def main() -> int:
    print("Generating optimization golden fixture ...")

    true_ff, guess_ff, mols, ref, engine = _make_water_problem()
    obj = ObjectiveFunction(guess_ff, engine, mols, ref)
    opt = ScipyOptimizer(method="L-BFGS-B", maxiter=200, verbose=False)
    result = opt.optimize(obj)

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
