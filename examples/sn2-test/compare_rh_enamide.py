"""Compare Rh-enamide direct bond projections against the pinned upstream fixture."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from q2mm.models.seminario import seminario_bond_fc
from q2mm.schrod_indep_filetypes import JaguarIn, Mol2

FIXTURE_PATH = REPO_ROOT / "test" / "fixtures" / "seminario_parity" / "rh_enamide_reference.json"


def main() -> int:
    fixture = json.loads(FIXTURE_PATH.read_text())
    training_dir = REPO_ROOT / "examples" / "rh-enamide" / "rh_enamide_training_set"
    mol2_path = REPO_ROOT / fixture["metadata"]["mol2_path"]
    hessian_path = REPO_ROOT / fixture["metadata"]["direct_hessian_path"]
    scaling = float(fixture["metadata"]["dft_scaling"])

    structure = Mol2(str(mol2_path)).structures[0]
    hessian = JaguarIn(str(hessian_path)).get_hessian(len(structure.atoms))
    coordinates = np.array([[atom.x, atom.y, atom.z] for atom in structure.atoms], dtype=float)

    print("=" * 70)
    print("Rh-Enamide Direct Bond Parity Comparison")
    print("=" * 70)
    print(f"Pinned upstream commit: {fixture['metadata']['upstream_commit']}")
    print(f"Structure source: {mol2_path.relative_to(REPO_ROOT)}")
    print(f"Hessian source:   {hessian_path.relative_to(REPO_ROOT)}")
    print(f"Training set dir: {training_dir.relative_to(REPO_ROOT)}")
    print()
    print(f"{'Bond':<20} {'Fixture':>15} {'New':>15} {'Diff':>12} {'Match':>8}")
    print("-" * 75)

    max_diff = 0.0
    all_match = True
    for bond in fixture["direct_bonds"]:
        actual = seminario_bond_fc(
            bond["atom_i"],
            bond["atom_j"],
            coordinates,
            hessian,
            au_units=True,
            dft_scaling=scaling,
        )
        expected = float(bond["legacy_force_constant_mdyn_a"])
        diff = abs(actual - expected)
        max_diff = max(max_diff, diff)
        match = "OK" if diff < 1e-8 else "DIFF"
        if diff >= 1e-8:
            all_match = False
        print(f"{bond['label']:<20} {expected:>15.8f} {actual:>15.8f} {diff:>12.3e} {match:>8}")

    print()
    if all_match:
        print(f"OK: all direct bond projections match the pinned upstream fixture (max diff {max_diff:.3e}).")
    else:
        print(f"DIFF: at least one direct bond projection differs (max diff {max_diff:.3e}).")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
